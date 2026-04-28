# mw2_merge.py
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

try:
    import faiss  # faiss-gpu or faiss-cpu
except ImportError as e:
    raise RuntimeError("faiss is required. Install faiss-gpu (CUDA) or faiss-cpu.") from e

try:
    from plyfile import PlyData, PlyElement
except ImportError as e:
    raise RuntimeError("plyfile is required to save .ply. pip install plyfile") from e

# tqdm with graceful fallback
try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(kwargs.get("total", 0))


# ----------------- tqdm / env helpers -----------------

def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name, None)
    if v is None:
        return default
    return str(v).lower() not in ("0", "false", "no", "off", "")


def _tqdm_wrap(it, *, enabled: bool, **kwargs):
    return tqdm(it, **kwargs) if enabled else it


def _col1(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return x
    if x.ndim == 2 and x.shape[1] == 1:
        return x
    return x.reshape(-1, 1)


def _cat_op(base_op: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.cat([_col1(base_op), _col1(x)], dim=0)


# ----------------- FAISS GPU resource reuse (speedup) -----------------
_FAISS_GPU_RES = None


def _get_faiss_gpu_res():
    global _FAISS_GPU_RES
    if _FAISS_GPU_RES is None and torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources"):
        _FAISS_GPU_RES = faiss.StandardGpuResources()
    return _FAISS_GPU_RES


def _get_faiss_gpu_id() -> int:
    try:
        return int(os.environ.get("MW2_FAISS_GPU", "0"))
    except Exception:
        return 0


# Local imports
try:
    from .spd import (normalize_quat, quat_to_rotmat, rotmat_to_quat,
                      scales_rots_to_cov, cov_to_scales_rots, sqrtm_spd_3x3)
    from .colmap_io import load_colmap_images_txt, camera_centers_from_poses, umeyama_sim3_from_centers
    from .io_ply import load_ply_gaussians
except Exception:
    from spd import (normalize_quat, quat_to_rotmat, rotmat_to_quat,
                     scales_rots_to_cov, cov_to_scales_rots, sqrtm_spd_3x3)
    from colmap_io import load_colmap_images_txt, camera_centers_from_poses, umeyama_sim3_from_centers
    from io_ply import load_ply_gaussians


def _log(msg: str) -> None:
    print(f"[mw2] {msg}", flush=True)


@dataclass
class MW2Params:
    merge_radius: float = 0.02
    topk: int = 32
    eps: float = 0.02
    sinkhorn_iters: int = 30
    sim3_iters: int = 40
    sim3_lr: float = 3e-3
    insert_ratio_th: float = 0.3
    opacity_th: float = 0.01
    min_scale: float = 1e-4
    tau: float = 0.3  # base-dominant distillation step

    # 主图优先 + 自适应阈值（0 代表自动；也可用环境变量覆盖）
    insert_dist_th: float = 0.0     # auto: 0.5*merge_radius (or env MW2_INSERT_DIST_TH)
    sub_to_main_th: float = 0.0     # auto: 0.5*merge_radius (or env MW2_SUB_TO_MAIN_TH)
    dedup_main_voxel: float = 0.0   # auto: 0.25*merge_radius (or env MW2_DEDUP_MAIN_VOXEL)
    dedup_sub_voxel: float = 0.0    # auto: 0.15*merge_radius (or env MW2_DEDUP_SUB_VOXEL)


def _load_gaussians_any(path: str, device: torch.device):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ply":
        xyz, scales, rots, dc, rest, op, shd = load_ply_gaussians(path, device)
        return xyz, scales, rots, dc, rest, _col1(op), shd

    d = torch.load(path, map_location=device)
    xyz = d["xyz"].float().contiguous()
    scales = d.get("scales", d.get("scaling")).float().contiguous()
    rots = d.get("rotations", d.get("rotation")).float().contiguous()
    dc = d.get("features_dc", d.get("f_dc")).float().contiguous()
    rest = d.get("features_rest", d.get("f_rest"))
    if rest is None:
        rest = torch.empty((xyz.shape[0], 0), device=device, dtype=dc.dtype)
    else:
        rest = rest.float().contiguous()
    op = d.get("opacities", d.get("opacity")).float().contiguous()
    shd = int(d.get("active_sh_degree", d.get("sh_degree", 0)))
    return xyz, scales, rots, dc, rest, _col1(op), shd


def _detect_scale_dim(path: str) -> Optional[int]:
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".ply":
            ply = PlyData.read(path)
            v = ply["vertex"].data
            names = list(v.dtype.names)
            count = len([n for n in names if n.startswith("scale_")])
            return count if count > 0 else None
        else:
            d = torch.load(path, map_location="cpu")
            sc = d.get("scales", d.get("scaling"))
            if sc is not None and sc.ndim == 2:
                return int(sc.shape[1])
            return None
    except Exception:
        return None


# ----------------- KNN backends (flat / IVF, auto) + batched search -----------------

def _build_faiss_index(base_xyz: torch.Tensor):
    """
    Build FAISS index pack for repeated searches.
    Env:
      MW2_KNN: 'auto' | 'flat' | 'ivf' (default 'auto')
      MW2_IVF_MINN: min N to use IVF (default 200000)
      MW2_IVF_NLIST: clusters (default ~ 4*sqrt(N))
      MW2_IVF_NPROBE: probes (default min(16, nlist))
      MW2_IVF_GPU: '1' to allow GPU IVF, '0' to force CPU IVF (default '1')
      MW2_FAISS_GPU: gpu id (default 0)
    """
    base_np = np.ascontiguousarray(base_xyz.detach().float().cpu().numpy().astype(np.float32, copy=False))
    N = base_np.shape[0]
    d = 3
    mode = os.environ.get("MW2_KNN", "auto").lower()
    ivf_minn = int(os.environ.get("MW2_IVF_MINN", "200000"))
    allow_gpu_ivf = os.environ.get("MW2_IVF_GPU", "1").lower() not in ("0", "false", "no")
    use_gpu = torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources")
    gpu_id = _get_faiss_gpu_id()
    res = _get_faiss_gpu_res() if use_gpu else None

    use_ivf = (mode == "ivf") or (mode == "auto" and N >= ivf_minn)
    if not use_ivf or N == 0:
        if use_gpu and res is not None:
            index = faiss.index_cpu_to_gpu(res, gpu_id, faiss.IndexFlatL2(d))
            index.add(base_np)
            return ("gpu-flat", index, res, gpu_id)
        else:
            index = faiss.IndexFlatL2(d)
            index.add(base_np)
            return ("cpu-flat", index, None, None)

    nlist_env = os.environ.get("MW2_IVF_NLIST", None)
    if nlist_env is not None:
        nlist = max(1, int(nlist_env))
    else:
        nlist = max(1, int(4.0 * math.sqrt(max(1.0, float(N)))))
    nlist = min(nlist, max(1, N // 2))
    nprobe_env = os.environ.get("MW2_IVF_NPROBE", None)
    nprobe = max(1, int(nprobe_env)) if nprobe_env is not None else min(16, nlist)

    quantizer = faiss.IndexFlatL2(d)
    ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    if not ivf.is_trained:
        ivf.train(base_np)
    ivf.add(base_np)
    ivf.nprobe = nprobe

    # GPU IVF often unstable with d=3; keep CPU IVF unless explicitly allowed and d>=8
    use_gpu_ivf = use_gpu and allow_gpu_ivf and (d >= 8) and (res is not None)
    if use_gpu_ivf:
        index = faiss.index_cpu_to_gpu(res, gpu_id, ivf)
        try:
            index.nprobe = nprobe
        except Exception:
            pass
        return ("gpu-ivf", index, res, gpu_id)
    else:
        return ("cpu-ivf", ivf, None, None)


def _search_faiss_index(idx_pack, query_xyz: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched search to avoid GPU kernel issues and memory spikes.
    Env: MW2_KNN_BATCH: per-call query batch size (default gpu 200k, cpu 50k)
    """
    mode, index, _, _ = idx_pack
    Q = int(query_xyz.shape[0])
    if Q == 0:
        return (torch.empty((0, k), device=query_xyz.device),
                torch.empty((0, k), dtype=torch.long, device=query_xyz.device))
    query_np = np.ascontiguousarray(query_xyz.detach().float().cpu().numpy().astype(np.float32, copy=False))

    bs_env = os.environ.get("MW2_KNN_BATCH", None)
    if bs_env is not None:
        B = max(1, int(bs_env))
    else:
        B = 200_000 if str(mode).startswith("gpu") else 50_000
    B = max(1, min(B, Q))

    dist2_all = np.empty((Q, k), dtype=np.float32)
    idx_all = np.empty((Q, k), dtype=np.int64)
    for s in range(0, Q, B):
        e = min(Q, s + B)
        sub = query_np[s:e]
        d2, ids = index.search(sub, k)
        dist2_all[s:e] = d2
        idx_all[s:e] = ids

    dist = torch.from_numpy(dist2_all).to(query_xyz.device).sqrt()
    idx = torch.from_numpy(idx_all).to(query_xyz.device).long()
    return dist, idx


# ----------------- OT pieces -----------------

def _w2_bures_edges(
    base_cov: torch.Tensor, base_cov_sqrt: torch.Tensor, base_tr: torch.Tensor,
    sub_cov: torch.Tensor, sub_tr: torch.Tensor,
    row_idx: torch.Tensor, col_idx: torch.Tensor
) -> torch.Tensor:
    A_sqrt = base_cov_sqrt[row_idx]
    B = sub_cov[col_idx]
    A = base_cov[row_idx]
    S = A_sqrt @ B @ A_sqrt.transpose(1, 2)
    S_sqrt = sqrtm_spd_3x3(S)
    tr = torch.sum(A.diagonal(dim1=1, dim2=2), dim=1) + sub_tr[col_idx] - 2.0 * torch.sum(
        S_sqrt.diagonal(dim1=1, dim2=2), dim=1
    )
    return torch.clamp(tr, min=0.0)


def _sinkhorn_sparse_balanced(row_idx, col_idx, C_e, a, b, eps, iters, *,
                              progress: bool = False,
                              desc: str = "Sinkhorn"):
    """
    Sinkhorn internal progress bar:
      - default OFF (too noisy)
      - enable by env MW2_TQDM_SINKHORN=1
    Note: progress must also be True.
    """
    device = C_e.device
    M = int(a.shape[0])
    N = int(b.shape[0])
    K_e = torch.exp(-C_e / eps)
    u = torch.ones(M, device=device, dtype=C_e.dtype)
    v = torch.ones(N, device=device, dtype=C_e.dtype)
    tiny = 1e-12

    show_sinkhorn = progress and _env_flag("MW2_TQDM_SINKHORN", False) and _env_flag("MW2_TQDM_INNER", True)
    it_range = _tqdm_wrap(range(iters), enabled=show_sinkhorn, total=iters, desc=desc, leave=False, dynamic_ncols=True)

    for _ in it_range:
        s_row = torch.zeros(M, device=device, dtype=C_e.dtype)
        s_row.index_add_(0, row_idx, K_e * v[col_idx])
        u = a / torch.clamp(s_row, min=tiny)

        s_col = torch.zeros(N, device=device, dtype=C_e.dtype)
        s_col.index_add_(0, col_idx, K_e * u[row_idx])
        v = b / torch.clamp(s_col, min=tiny)

    pi_e = u[row_idx] * K_e * v[col_idx]
    return pi_e, u, v


def _inverse_sigmoid(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log(1.0 - p)


def _save_ply_gaussians(out_ply: str,
                        xyz: torch.Tensor,
                        scales: torch.Tensor,     # actual sigma (positive)
                        rots: torch.Tensor,       # normalized quat
                        dc: torch.Tensor,
                        rest: Optional[torch.Tensor],
                        op: torch.Tensor) -> None:
    xyz_np = xyz.detach().float().cpu().numpy()
    sc_log_np = torch.log(torch.clamp(scales, min=1e-12)).detach().float().cpu().numpy()
    rt_np = rots.detach().float().cpu().numpy()
    dc_np = dc.detach().float().cpu().numpy()
    if rest is not None and rest.numel() > 0:
        rest_np = rest.detach().float().cpu().numpy()
        rest_dim = rest_np.shape[1]
    else:
        rest_np = None
        rest_dim = 0
    op_logit_np = _inverse_sigmoid(_col1(op).reshape(-1)).detach().float().cpu().numpy()

    N = xyz_np.shape[0]
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
    ]
    if rest_dim > 0:
        dtype += [(f"f_rest_{i}", "f4") for i in range(rest_dim)]
    dtype += [("opacity", "f4")]
    dtype += [(f"scale_{i}", "f4") for i in range(sc_log_np.shape[1])]
    dtype += [(f"rot_{i}", "f4") for i in range(4)]

    data = np.empty(N, dtype=dtype)
    data["x"] = xyz_np[:, 0]
    data["y"] = xyz_np[:, 1]
    data["z"] = xyz_np[:, 2]
    data["f_dc_0"] = dc_np[:, 0]
    data["f_dc_1"] = dc_np[:, 1]
    data["f_dc_2"] = dc_np[:, 2]
    if rest_dim > 0:
        for i in range(rest_dim):
            data[f"f_rest_{i}"] = rest_np[:, i]
    data["opacity"] = op_logit_np
    for i in range(sc_log_np.shape[1]):
        data[f"scale_{i}"] = sc_log_np[:, i]
    data["rot_0"] = rt_np[:, 0]
    data["rot_1"] = rt_np[:, 1]
    data["rot_2"] = rt_np[:, 2]
    data["rot_3"] = rt_np[:, 3]

    out_dir = os.path.dirname(out_ply)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    PlyData([PlyElement.describe(data, "vertex")], text=False).write(out_ply)


# ----------------- Helpers & gates -----------------

def _apply_color_gate(row_idx_list: torch.Tensor,
                      col_idx_list: torch.Tensor,
                      base_dc: torch.Tensor,
                      sub_dc: torch.Tensor,
                      color_th: Optional[float]) -> Tuple[torch.Tensor, torch.Tensor]:
    if color_th is None or float(color_th) <= 0.0 or row_idx_list.numel() == 0:
        return row_idx_list, col_idx_list
    diff = base_dc[row_idx_list] - sub_dc[col_idx_list]
    cd = torch.linalg.norm(diff, dim=1)
    keep = cd <= float(color_th)
    if not keep.any():
        return row_idx_list[:0], col_idx_list[:0]
    return row_idx_list[keep], col_idx_list[keep]


def _partition_voxels(points: torch.Tensor, voxel_size: float):
    mins = points.min(dim=0).values
    vids = torch.floor((points - mins) / voxel_size).to(torch.int64)
    buckets: Dict[Tuple[int, int, int], List[int]] = {}
    for i in range(points.shape[0]):
        key = (int(vids[i, 0].item()), int(vids[i, 1].item()), int(vids[i, 2].item()))
        if key not in buckets:
            buckets[key] = []
        buckets[key].append(i)
    return mins, buckets


def _auto_color_th(base_dc: torch.Tensor) -> Optional[float]:
    try:
        n = base_dc.shape[0]
        k = min(20000, n)
        if k == 0:
            return None
        idx = torch.randperm(n, device=base_dc.device)[:k]
        mags = torch.linalg.norm(base_dc[idx], dim=1)
        med = torch.median(mags).item()
        th = max(0.05, min(0.3, 0.25 * med))
        return th
    except Exception:
        return 0.2


# -------- 场景尺度 / 统计量 / 自适应参数 --------

def _scene_diag(xyz: Optional[torch.Tensor]) -> float:
    """估计场景对角线长度，用于把参数和场景尺度挂钩。"""
    if xyz is None or xyz.numel() == 0:
        return 1.0
    bb_min = xyz.min(dim=0).values
    bb_max = xyz.max(dim=0).values
    return float((bb_max - bb_min).norm().item())


def _gaussian_stats(xyz: torch.Tensor,
                    scales: torch.Tensor,
                    op: torch.Tensor) -> Dict[str, float]:
    """简单统计：中位 scale、opacity、点数等。"""
    if xyz is None or xyz.numel() == 0:
        return dict(N=0, med_scale=1e-3, med_op=0.5, diag=1.0)

    diag = _scene_diag(xyz)
    vol = (scales[:, 0] * scales[:, 1] * scales[:, 2]).clamp_min(1e-12)
    med_scale = float(torch.median(vol ** (1.0 / 3.0)).item())
    med_op = float(torch.median(_col1(op).reshape(-1)).item())
    return dict(
        N=int(xyz.shape[0]),
        med_scale=med_scale,
        med_op=med_op,
        diag=diag,
    )


def _auto_topk(Nbase: int, diag: float) -> int:
    """
    根据 base 点数规模和场景大小自动选择 topk。
    点数越多 / 场景越大 -> topk 越小，保证速度和稳定性。
    """
    if Nbase < 200_000:
        return 32
    elif Nbase < 1_000_000:
        return 24
    elif Nbase < 5_000_000:
        return 16
    else:
        return 8


def _auto_tau(base_stats: Dict[str, float],
              sub_stats: Dict[str, float]) -> float:
    """
    根据 base / sub 的 scale、opacity 关系，自动给一个 tau。
    - sub 更锐利（scale 小），且不太透明：加大 tau
    - sub 模糊或 opacity 低：减小 tau
    """
    base_ms, sub_ms = base_stats["med_scale"], sub_stats["med_scale"]
    base_op, sub_op = base_stats["med_op"], sub_stats["med_op"]

    sharp_ratio = base_ms / max(sub_ms, 1e-9)     # >1 表示 sub 更小、更锐
    op_ratio = sub_op / max(base_op, 1e-6)        # <1 表示 sub 更透明

    tau = 0.3
    if sharp_ratio > 1.2 and op_ratio > 0.7:
        tau = 0.35          # sub 又锐又不透明
    elif sharp_ratio < 0.8 or op_ratio < 0.5:
        tau = 0.2           # sub 模糊或明显更透明
    elif op_ratio < 0.7:
        tau = 0.25

    tau = float(max(0.15, min(0.45, tau)))
    return tau


# -------- 只更新 active rows 的协方差（大提速） --------
def _update_cov_only_active_rows(
    base_scales, base_rots,
    wA, denom,
    row_idx_list, col_idx_list, pi_e,
    base_xyz, xyz_Bp, cov_Bp,
):
    device = base_xyz.device
    if row_idx_list.numel() == 0:
        return base_scales, base_rots

    rows = torch.unique(row_idx_list)
    row_map = -torch.ones(base_xyz.shape[0], device=device, dtype=torch.long)
    row_map[rows] = torch.arange(rows.shape[0], device=device, dtype=torch.long)
    row_loc = row_map[row_idx_list]

    cov_A_act = scales_rots_to_cov(base_scales[rows], base_rots[rows])
    term_base = cov_A_act * wA[rows][:, None, None]
    term_sub = torch.zeros_like(term_base)

    d = xyz_Bp[col_idx_list] - base_xyz[row_idx_list]
    outer = torch.einsum("ni,nj->nij", d, d)
    contrib = (pi_e[:, None, None] * (cov_Bp[col_idx_list] + outer))

    term_sub_flat = term_sub.reshape(term_sub.shape[0], 9)
    contrib_flat = contrib.reshape(contrib.shape[0], 9)
    term_sub_flat.index_add_(0, row_loc, contrib_flat)
    term_sub = term_sub_flat.reshape_as(term_sub)

    new_cov = (term_base + term_sub) / denom[rows][:, None, None].clamp_min(1e-8)
    new_cov = 0.5 * (new_cov + new_cov.transpose(1, 2)) + 1e-8 * torch.eye(3, device=device).unsqueeze(0)

    new_sc, new_rt = cov_to_scales_rots(new_cov)
    base_scales[rows] = new_sc
    base_rots[rows] = new_rt
    return base_scales, base_rots


# -------- voxel 去重（用于最终保存，属性加权平均） --------
def _dedup_voxel_weighted(
    xyz: torch.Tensor,
    scales: torch.Tensor,
    rots: torch.Tensor,
    dc: torch.Tensor,
    rest: torch.Tensor,
    op: torch.Tensor,
    voxel: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if voxel is None or voxel <= 0.0 or xyz.numel() == 0:
        return xyz, scales, rots, dc, rest, op

    device = xyz.device
    voxel = float(voxel)

    mins = xyz.min(dim=0).values
    vids = torch.floor((xyz - mins) / voxel).to(torch.int64)

    vx, vy, vz = vids[:, 0], vids[:, 1], vids[:, 2]
    key = vx * 73856093 + vy * 19349663 + vz * 83492791

    uniq, inv = torch.unique(key, return_inverse=True)
    C = int(uniq.shape[0])

    vol = (4.0 / 3.0) * math.pi * (scales[:, 0] * scales[:, 1] * scales[:, 2])
    w = torch.clamp(_col1(op).reshape(-1), 0.0, 1.0) * vol
    w = w.clamp_min(1e-12)

    wsum = torch.zeros((C,), device=device, dtype=xyz.dtype)
    wsum.index_add_(0, inv, w)

    xyz_sum = torch.zeros((C, 3), device=device, dtype=xyz.dtype)
    xyz_sum.index_add_(0, inv, xyz * w[:, None])
    xyz_new = xyz_sum / wsum[:, None]

    dc_sum = torch.zeros((C, dc.shape[1]), device=device, dtype=dc.dtype)
    dc_sum.index_add_(0, inv, dc * w[:, None])
    dc_new = dc_sum / wsum[:, None]

    if rest is None or rest.numel() == 0:
        rest_new = rest
    else:
        rest_sum = torch.zeros((C, rest.shape[1]), device=device, dtype=rest.dtype)
        rest_sum.index_add_(0, inv, rest * w[:, None])
        rest_new = rest_sum / wsum[:, None]

    op_sum = torch.zeros((C,), device=device, dtype=op.dtype)
    op_sum.index_add_(0, inv, _col1(op).reshape(-1) * w)
    op_new = (op_sum / wsum).clamp(0.0, 1.0).reshape(-1, 1)

    sc_log = torch.log(torch.clamp(scales, min=1e-12))
    sc_sum = torch.zeros((C, scales.shape[1]), device=device, dtype=scales.dtype)
    sc_sum.index_add_(0, inv, sc_log * w[:, None])
    scales_new = torch.exp(sc_sum / wsum[:, None]).clamp_min(1e-12)

    q = rots
    sign = torch.where(q[:, 0:1] >= 0, torch.ones_like(q[:, 0:1]), -torch.ones_like(q[:, 0:1]))
    q = q * sign
    q_sum = torch.zeros((C, 4), device=device, dtype=q.dtype)
    q_sum.index_add_(0, inv, q * w[:, None])
    rots_new = normalize_quat(q_sum)

    return xyz_new, scales_new, rots_new, dc_new, rest_new, op_new


def _filter_sub_by_main_nn(main_xyz, sub_xyz, th, idx_pack_main=None):
    if sub_xyz.numel() == 0:
        return torch.empty((0,), device=sub_xyz.device, dtype=torch.bool)
    if idx_pack_main is None:
        idx_pack_main = _build_faiss_index(main_xyz)
    d1, _ = _search_faiss_index(idx_pack_main, sub_xyz, 1)
    keep = (d1.reshape(-1) >= th)
    return keep


# -------- 细节补洞式子图选择（主图不减点，只在空缺体素插） --------
def _select_sub_detail_points(
    main_xyz: torch.Tensor,
    sub_xyz: torch.Tensor,
    *,
    merge_radius: float,
    dist_th: float,
    voxel: float,
    main_min_count: int,
    sub_max_per_voxel: int,
    idx_pack_main=None,
) -> torch.Tensor:
    device = sub_xyz.device
    N = int(sub_xyz.shape[0])
    if N == 0:
        return torch.zeros((0,), device=device, dtype=torch.bool)

    if idx_pack_main is None:
        idx_pack_main = _build_faiss_index(main_xyz)

    d1, _ = _search_faiss_index(idx_pack_main, sub_xyz, 1)
    far_mask = (d1.reshape(-1) >= float(dist_th))

    if voxel <= 0.0:
        return far_mask

    mins = torch.min(torch.min(main_xyz, dim=0).values, torch.min(sub_xyz, dim=0).values)
    main_vid = torch.floor((main_xyz - mins) / float(voxel)).to(torch.int64)
    sub_vid = torch.floor((sub_xyz - mins) / float(voxel)).to(torch.int64)

    def _hash(v):
        return v[:, 0] * 73856093 + v[:, 1] * 19349663 + v[:, 2] * 83492791

    main_key = _hash(main_vid)
    sub_key = _hash(sub_vid)

    uniq_m, inv_m = torch.unique(main_key, return_inverse=True)
    mcount = torch.zeros((uniq_m.shape[0],), device=device, dtype=torch.int32)
    ones = torch.ones_like(inv_m, dtype=torch.int32)
    mcount.index_add_(0, inv_m, ones)

    sort_m, perm = torch.sort(uniq_m)
    mcount_sorted = mcount[perm]
    pos = torch.searchsorted(sort_m, sub_key)
    hit = (pos < sort_m.shape[0]) & (sort_m[pos] == sub_key)
    sub_main_count = torch.zeros((N,), device=device, dtype=torch.int32)
    sub_main_count[hit] = mcount_sorted[pos[hit]]

    sparse_mask = (sub_main_count < int(main_min_count))

    cand = far_mask & sparse_mask
    if not cand.any():
        return cand

    cand_idx = torch.nonzero(cand, as_tuple=False).reshape(-1)
    cand_key = sub_key[cand_idx]
    cand_d = d1.reshape(-1)[cand_idx]

    order = torch.argsort(cand_d, descending=True, stable=True)
    cand_key2 = cand_key[order]
    cand_idx2 = cand_idx[order]

    order2 = torch.argsort(cand_key2, stable=True)
    cand_key3 = cand_key2[order2]
    cand_idx3 = cand_idx2[order2]

    uniq_s, inv_s = torch.unique(cand_key3, return_inverse=True)
    ctr = torch.zeros((uniq_s.shape[0],), device=device, dtype=torch.int32)
    keep = torch.zeros((cand_idx3.shape[0],), device=device, dtype=torch.bool)

    for i in range(cand_idx3.shape[0]):
        g = int(inv_s[i].item())
        if ctr[g] < int(sub_max_per_voxel):
            keep[i] = True
            ctr[g] += 1

    mask = torch.zeros((N,), device=device, dtype=torch.bool)
    mask[cand_idx3[keep]] = True
    return mask


def _all_true_mask(n: int, device) -> torch.Tensor:
    return torch.ones((n,), device=device, dtype=torch.bool)


def _choose_insert_mask(
    *,
    insert_mode: str,
    base_xyz: torch.Tensor,
    xyz_Bp: torch.Tensor,
    params: MW2Params,
    insert_dist_th: float,
    detail_vox: float,
    main_min: int,
    sub_cap: int,
    idx_pack_main
) -> torch.Tensor:
    if str(insert_mode).lower() == "all":
        return _all_true_mask(int(xyz_Bp.shape[0]), xyz_Bp.device)
    return _select_sub_detail_points(
        base_xyz, xyz_Bp,
        merge_radius=params.merge_radius,
        dist_th=insert_dist_th,
        voxel=detail_vox,
        main_min_count=main_min,
        sub_max_per_voxel=sub_cap,
        idx_pack_main=idx_pack_main,
    )


# -------- 自动去雾 --------
def _defog_pass(
    xyz: torch.Tensor,
    scales: torch.Tensor,
    rots: torch.Tensor,
    dc: torch.Tensor,
    rest: Optional[torch.Tensor],
    op: torch.Tensor,
    *,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    简单“去雾”：
      1. 检测小 scale + 低 opacity 的点的比例
      2. 比例过高则删除这些点
      3. 再做一轮比较粗的 voxel 聚合，进一步压掉毛刺
    """
    if xyz is None or xyz.numel() == 0:
        return xyz, scales, rots, dc, rest, op

    device = xyz.device
    stats = _gaussian_stats(xyz, scales, op)
    med_scale = stats["med_scale"]
    med_op = stats["med_op"]

    vol = (scales[:, 0] * scales[:, 1] * scales[:, 2]).clamp_min(1e-12)
    scale_len = vol ** (1.0 / 3.0)

    small_mask = scale_len < (0.4 * med_scale)
    low_op_mask = _col1(op).reshape(-1) < max(0.03, 0.4 * med_op)
    fog_mask = small_mask & low_op_mask

    fog_ratio = float(fog_mask.float().mean().item())
    if verbose:
        _log(f"Defog: med_scale={med_scale:.4e}, med_op={med_op:.3f}, fog_ratio={fog_ratio*100:.1f}%")

    if fog_ratio < 0.15:
        if verbose:
            _log("Defog: fog_ratio < 15%, skip defog.")
        return xyz, scales, rots, dc, rest, op

    if verbose:
        _log(f"Defog: removing {int(fog_mask.sum().item())}/{xyz.shape[0]} fog points...")

    keep = ~fog_mask
    xyz = xyz[keep]
    scales = scales[keep]
    rots = rots[keep]
    dc = dc[keep]
    op = op[keep]
    if rest is not None and rest.numel() > 0:
        rest = rest[keep]
    else:
        rest = rest

    diag = _scene_diag(xyz)
    voxel = max(0.01 * diag, 4.0 * med_scale)

    if verbose:
        _log(f"Defog: additional voxel merge with voxel={voxel:.4f} ...")

    xyz, scales, rots, dc, rest, op = _dedup_voxel_weighted(
        xyz, scales, rots, dc, rest, op, voxel=voxel
    )

    if verbose:
        _log(f"Defog: final N={xyz.shape[0]} after removing fog + voxel merge.")

    return xyz, scales, rots, dc, rest, op


class MW2Merger:
    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _get_faiss_gpu_res()

    def _maybe_load_colmap_centers(self, colmap_path: Optional[str]) -> Optional[Dict[str, torch.Tensor]]:
        if not colmap_path:
            return None
        try:
            poses = load_colmap_images_txt(colmap_path)
            centers = camera_centers_from_poses(poses)
            return centers
        except Exception:
            return None

    def merge(self,
              base_ckpt: Optional[str],
              sub_ckpts: List[str],
              out_ply: str,
              params: MW2Params = MW2Params(),
              base_colmap: Optional[str] = None,
              sub_colmap_paths: Optional[List[Optional[str]]] = None,
              progress: bool = False,
              insert_mode: str = "detail",
              no_quality_gate: bool = False) -> None:
        t0 = time.perf_counter()
        device = self.device
        sub_ckpts = sorted(list({str(p) for p in sub_ckpts}))
        S = len(sub_ckpts)

        tq_major = bool(progress)
        tq_inner = bool(progress) and _env_flag("MW2_TQDM_INNER", True)

        # --- 自适应/可覆盖参数 ---
        env_insert_dist = os.environ.get("MW2_INSERT_DIST_TH", None)
        insert_dist_th = float(env_insert_dist) if env_insert_dist is not None else (
            params.insert_dist_th if params.insert_dist_th > 0 else 0.5 * params.merge_radius
        )

        env_sub2main = os.environ.get("MW2_SUB_TO_MAIN_TH", None)
        sub_to_main_th = float(env_sub2main) if env_sub2main is not None else (
            params.sub_to_main_th if params.sub_to_main_th > 0 else 0.5 * params.merge_radius
        )

        env_main_vox = os.environ.get("MW2_DEDUP_MAIN_VOXEL", None)
        dedup_main_vox = float(env_main_vox) if env_main_vox is not None else (
            params.dedup_main_voxel if params.dedup_main_voxel > 0 else 0.25 * params.merge_radius
        )

        env_sub_vox = os.environ.get("MW2_DEDUP_SUB_VOXEL", None)
        dedup_sub_vox = float(env_sub_vox) if env_sub_vox is not None else (
            params.dedup_sub_voxel if params.dedup_sub_voxel > 0 else 0.15 * params.merge_radius
        )

        # detail-aware insert params
        detail_vox_env = os.environ.get("MW2_DETAIL_VOXEL", None)
        detail_vox = float(detail_vox_env) if detail_vox_env is not None else max(2.0 * params.merge_radius, 1e-6)
        main_min = int(os.environ.get("MW2_MAIN_MIN_PER_VOXEL", "2"))
        sub_cap = int(os.environ.get("MW2_SUB_MAX_PER_VOXEL", "8"))

        _log(f"Start MW2 merge | device={device} | subs={S} | merge_radius={params.merge_radius} topk={params.topk} eps={params.eps}")
        _log(f"Main-priority: insert_dist_th={insert_dist_th:.6f} sub_to_main_th={sub_to_main_th:.6f} "
             f"dedup_main_vox={dedup_main_vox:.6f} dedup_sub_vox={dedup_sub_vox:.6f}")
        _log(f"Detail insert: detail_vox={detail_vox:.6f} main_min={main_min} sub_cap={sub_cap}")
        _log(f"Insert mode: {insert_mode} | no_quality_gate={no_quality_gate}")
        if base_ckpt:
            _log(f"Base: {base_ckpt}")
        if base_colmap:
            _log(f"Base COLMAP: {base_colmap}")

        base_xyz = None
        base_scales = None
        base_rots = None
        base_dc = None
        base_rest = None
        base_op = None
        origin = None  # 0=main/base, 1=sub-inserted

        base_N0 = None  # 记录初始 base 点数（保证最终不少于它）

        save_scale_dim: Optional[int] = None
        if base_ckpt is not None and os.path.exists(base_ckpt):
            _log("Loading base gaussians...")
            base_xyz, base_scales, base_rots, base_dc, base_rest, base_op, _ = _load_gaussians_any(base_ckpt, device)
            base_op = _col1(base_op)
            origin = torch.zeros((base_xyz.shape[0],), device=device, dtype=torch.uint8)  # base is main
            save_scale_dim = _detect_scale_dim(base_ckpt)
            base_N0 = int(base_xyz.shape[0])
            _log(f"Base loaded: N={base_N0}")

        base_centers = self._maybe_load_colmap_centers(base_colmap)

        # Adaptive color gate
        color_th_env = os.environ.get("MW2_COLOR_TH", None)
        color_th = float(color_th_env) if color_th_env is not None else _auto_color_th(base_dc) if base_dc is not None else None

        sub_iter = _tqdm_wrap(sub_ckpts, enabled=tq_major, total=S, desc="Submaps", dynamic_ncols=True, leave=True)
        for sidx, spath in enumerate(sub_iter):
            sub_start = time.perf_counter()
            last_L_mw2 = None
            last_L_prior = None
            last_edges = 0

            step_bar = tqdm(total=8, desc=f"  Steps {sidx+1}/{S}", leave=False, dynamic_ncols=True) if tq_inner else None

            _log(f"[{sidx+1}/{S}] Submap: {spath}")
            xyz_s, sc_s, rt_s, dc_s, rest_s, op_s, _ = _load_gaussians_any(spath, device)
            op_s = _col1(op_s)
            if save_scale_dim is None:
                save_scale_dim = _detect_scale_dim(spath)
            _log(f"  sub points: {xyz_s.shape[0]}")
            if step_bar: step_bar.update(1)  # loaded

            # sub 统计
            sub_stats = _gaussian_stats(xyz_s, sc_s, op_s)

            if base_xyz is None:
                base_xyz = xyz_s.clone()
                base_scales = sc_s.clone()
                base_rots = rt_s.clone()
                base_dc = dc_s.clone()
                base_rest = rest_s.clone() if rest_s is not None else torch.empty((xyz_s.shape[0], 0), device=device, dtype=dc_s.dtype)
                base_op = _col1(op_s.clone())
                origin = torch.zeros((base_xyz.shape[0],), device=device, dtype=torch.uint8)  # first base is main
                base_N0 = int(base_xyz.shape[0])
                _log(f"  initialized base from first submap (as main). N={base_xyz.shape[0]}")
                if step_bar:
                    step_bar.update(step_bar.total - step_bar.n)
                    step_bar.close()
                if tq_major:
                    sub_iter.set_postfix_str(f"N={base_xyz.shape[0]}")
                continue

            base_op = _col1(base_op)

            # base 统计（随 merge 更新）
            base_stats = _gaussian_stats(base_xyz, base_scales, base_op)

            # Adaptive topk & iters based on base size
            Nbase = int(base_xyz.shape[0])

            # 自动 topk：params.topk <= 0 时启用
            if params.topk <= 0:
                eff_topk = _auto_topk(Nbase, base_stats["diag"])
            else:
                eff_topk = params.topk

            # 大规模场景再额外兜底
            if Nbase >= 10_000_000:
                eff_topk = min(eff_topk, 8)
            elif Nbase >= 2_000_000:
                eff_topk = min(eff_topk, 16)

            # 自动 tau：params.tau <= 0 时启用
            if params.tau <= 0:
                tau_eff = _auto_tau(base_stats, sub_stats)
            else:
                tau_eff = params.tau

            sim3_iters_eff = params.sim3_iters if Nbase < 5_000_000 else max(4, (params.sim3_iters * 3) // 5)
            sinkhorn_iters_eff = params.sinkhorn_iters if Nbase < 5_000_000 else max(4, (params.sinkhorn_iters * 3) // 5)
            if step_bar: step_bar.update(1)  # adaptive

            # align rest dims
            if base_rest is None or base_rest.numel() == 0:
                base_rest = torch.empty((base_xyz.shape[0], 0), device=device, dtype=dc_s.dtype)
            if rest_s is None:
                rest_s = torch.empty((xyz_s.shape[0], base_rest.shape[1]), device=device, dtype=dc_s.dtype)
            if base_rest.shape[1] != rest_s.shape[1]:
                if base_rest.shape[1] < rest_s.shape[1]:
                    pad = torch.zeros((base_rest.shape[0], rest_s.shape[1] - base_rest.shape[1]), device=device, dtype=base_rest.dtype)
                    base_rest = torch.cat([base_rest, pad], dim=1)
                else:
                    pad = torch.zeros((rest_s.shape[0], base_rest.shape[1] - rest_s.shape[1]), device=device, dtype=rest_s.dtype)
                    rest_s = torch.cat([rest_s, pad], dim=1)
            if step_bar: step_bar.update(1)  # rest aligned

            _log("  precomputing covariances (base/sub)...")
            cov_A = scales_rots_to_cov(base_scales, base_rots)
            cov_A_sqrt = sqrtm_spd_3x3(cov_A)
            cov_B0 = scales_rots_to_cov(sc_s, rt_s)
            if step_bar: step_bar.update(1)  # cov done

            wA = torch.clamp(base_op.reshape(-1), 0.0, 1.0) * ((4.0 / 3.0) * math.pi * (base_scales[:, 0] * base_scales[:, 1] * base_scales[:, 2]))
            wB = torch.clamp(op_s.reshape(-1), 0.0, 1.0) * ((4.0 / 3.0) * math.pi * (sc_s[:, 0] * sc_s[:, 1] * sc_s[:, 2]))
            wA = wA / torch.sum(wA).clamp_min(1e-8)
            wB = wB / torch.sum(wB).clamp_min(1e-8)
            if step_bar: step_bar.update(1)  # weights

            # Init Sim(3)
            q = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device))
            t = nn.Parameter(torch.zeros((1, 3), device=device))
            logs = nn.Parameter(torch.zeros((1, 1), device=device))

            if base_centers is not None and sub_colmap_paths is not None and sidx < len(sub_colmap_paths) and sub_colmap_paths[sidx]:
                try:
                    sub_poses = load_colmap_images_txt(sub_colmap_paths[sidx])
                    sub_centers = camera_centers_from_poses(sub_poses)
                    common = len(set(base_centers.keys()) & set(sub_centers.keys()))
                    sim3 = umeyama_sim3_from_centers(base_centers, sub_centers, device, min_common=3)
                    if sim3 is not None:
                        s0, R0, t0_ = sim3
                        q.data = rotmat_to_quat(R0.unsqueeze(0))
                        t.data = t0_.unsqueeze(0)
                        logs.data = torch.log(s0.clamp_min(1e-8)).reshape(1, 1)
                        _log(f"  COLMAP Sim(3) init: common={common}, scale={float(s0):.4f}")
                except Exception as e:
                    _log(f"  COLMAP init failed: {e}")

            optT = torch.optim.Adam([q, t, logs], lr=params.sim3_lr)
            I3 = torch.eye(3, device=device)

            # Build reusable FAISS index (auto flat/ivf, GPU if available)
            idx_pack = _build_faiss_index(base_xyz)
            if step_bar: step_bar.update(1)  # index built

            it_iter = _tqdm_wrap(
                range(sim3_iters_eff),
                enabled=tq_inner,
                total=sim3_iters_eff,
                desc=f"  Sim3 {sidx+1}/{S}",
                leave=False,
                dynamic_ncols=True
            )

            for _ in it_iter:
                optT.zero_grad()
                R_now = quat_to_rotmat(normalize_quat(q))[0]
                s_now = torch.exp(logs)[0, 0]
                xyz_Bp = s_now * (xyz_s @ R_now.transpose(0, 1)) + t
                cov_Bp = (s_now * s_now) * (R_now @ cov_B0 @ R_now.transpose(0, 1))

                dist, idx = _search_faiss_index(idx_pack, xyz_Bp, eff_topk)
                valid = dist <= params.merge_radius
                has_col = torch.any(valid, dim=1)

                if not torch.any(has_col):
                    L_prior = 1e-3 * (logs ** 2).sum() + 5e-4 * ((R_now - I3) ** 2).sum() + 5e-4 * (t ** 2).sum()
                    L_prior.backward()
                    optT.step()
                    with torch.no_grad():
                        q.data = normalize_quat(q.data)
                    last_L_mw2 = 0.0
                    last_L_prior = float(L_prior.detach().item())
                    last_edges = 0
                    if tq_inner and hasattr(it_iter, "set_postfix_str"):
                        it_iter.set_postfix_str(f"no-overlap Lmw2=0 Lp={last_L_prior:.2e} |t|={float(t.norm()):.3f} s={float(s_now):.3f}")
                    continue

                sel_rows = idx[has_col]
                sel_mask = valid[has_col]
                row_idx_list = sel_rows[sel_mask]
                col_idx_rep = torch.arange(xyz_Bp.shape[0], device=device)[has_col]
                col_idx_list = col_idx_rep.repeat_interleave(sel_mask.sum(dim=1))

                row_idx_list, col_idx_list = _apply_color_gate(row_idx_list, col_idx_list, base_dc, dc_s, color_th)
                if row_idx_list.numel() == 0:
                    L_prior = 1e-3 * (logs ** 2).sum() + 5e-4 * ((R_now - I3) ** 2).sum() + 5e-4 * (t ** 2).sum()
                    L_prior.backward()
                    optT.step()
                    with torch.no_grad():
                        q.data = normalize_quat(q.data)
                    last_L_mw2 = 0.0
                    last_L_prior = float(L_prior.detach().item())
                    last_edges = 0
                    if tq_inner and hasattr(it_iter, "set_postfix_str"):
                        it_iter.set_postfix_str(f"gated-all Lmw2=0 Lp={last_L_prior:.2e} |t|={float(t.norm()):.3f} s={float(s_now):.3f}")
                    continue

                row_act = torch.unique(row_idx_list)
                col_act = torch.unique(col_idx_list)

                row_map = -torch.ones(base_xyz.shape[0], device=device, dtype=torch.long)
                col_map = -torch.ones(xyz_Bp.shape[0], device=device, dtype=torch.long)
                row_map[row_act] = torch.arange(row_act.shape[0], device=device, dtype=torch.long)
                col_map[col_act] = torch.arange(col_act.shape[0], device=device, dtype=torch.long)
                row_loc = row_map[row_idx_list]
                col_loc = col_map[col_idx_list]

                a = (wA[row_act] / torch.sum(wA[row_act]).clamp_min(1e-8))
                b = (wB[col_act] / torch.sum(wB[col_act]).clamp_min(1e-8))

                pos_i = base_xyz[row_idx_list]
                pos_j = xyz_Bp[col_idx_list]
                C_pos = torch.sum((pos_i - pos_j) ** 2, dim=1)

                cov_A_act = cov_A[row_act]
                cov_A_sqrt_act = sqrtm_spd_3x3(cov_A_act)
                tr_A_act = torch.sum(cov_A_act.diagonal(dim1=1, dim2=2), dim=1)

                cov_B_act = cov_Bp[col_act]
                tr_B_act = torch.sum(cov_B_act.diagonal(dim1=1, dim2=2), dim=1)

                C_bures = _w2_bures_edges(cov_A_act, cov_A_sqrt_act, tr_A_act, cov_B_act, tr_B_act, row_loc, col_loc)
                C_e = C_pos + C_bures

                pi_e, _, _ = _sinkhorn_sparse_balanced(
                    row_loc, col_loc, C_e, a, b, params.eps, sinkhorn_iters_eff,
                    progress=tq_inner, desc=f"    Sinkhorn {sidx+1}/{S}"
                )
                L_mw2 = torch.sum(pi_e * C_e)
                L_prior = 1e-3 * (logs ** 2).sum() + 5e-4 * ((R_now - I3) ** 2).sum() + 5e-4 * (t ** 2).sum()
                (L_mw2 + L_prior).backward()
                optT.step()
                with torch.no_grad():
                    q.data = normalize_quat(q.data)

                last_L_mw2 = float(L_mw2.detach().item())
                last_L_prior = float(L_prior.detach().item())
                last_edges = int(row_idx_list.shape[0])

                if tq_inner and hasattr(it_iter, "set_postfix_str"):
                    it_iter.set_postfix_str(
                        f"Lmw2={last_L_mw2:.2e} Lp={last_L_prior:.2e} "
                        f"|t|={float(t.norm()):.3f} s={float(torch.exp(logs)[0,0]):.3f} edges={last_edges}"
                    )

            if step_bar: step_bar.update(1)  # sim3 done

            with torch.no_grad():
                R_now = quat_to_rotmat(normalize_quat(q))[0]
                s_now = torch.exp(logs)[0, 0]
                xyz_Bp = s_now * (xyz_s @ R_now.transpose(0, 1)) + t
                cov_Bp = (s_now * s_now) * (R_now @ cov_B0 @ R_now.transpose(0, 1))

            # Adaptive voxel size (if not set)
            vox_env = os.environ.get("MW2_VOXEL_SIZE", None)
            voxel_size = float(vox_env) if vox_env is not None else 0.0
            voxel_overlap = float(os.environ.get("MW2_VOXEL_OVERLAP", "0.2"))
            voxel_overlap = max(0.0, min(0.9, voxel_overlap))

            if voxel_size <= 0.0:
                try:
                    bb_min = xyz_Bp.min(dim=0).values
                    bb_max = xyz_Bp.max(dim=0).values
                    vol = float(torch.clamp((bb_max - bb_min), min=1e-6).prod().item())
                    h = (vol / max(1, int(xyz_Bp.shape[0]))) ** (1.0 / 3.0)
                    voxel_size = max(4.0 * params.merge_radius, 4.0 * h)
                    if Nbase + int(xyz_Bp.shape[0]) < 2_000_000:
                        voxel_size = 0.0
                except Exception:
                    voxel_size = 0.0
            if step_bar: step_bar.update(1)  # voxel decided

            idx_pack = _build_faiss_index(base_xyz)

            if voxel_size > 0.0:
                margin = voxel_overlap * voxel_size + params.merge_radius
                mins, buckets = _partition_voxels(xyz_Bp, voxel_size)

                edge_rows: List[torch.Tensor] = []
                edge_cols: List[torch.Tensor] = []
                edge_pi: List[torch.Tensor] = []
                mass_row = torch.zeros(base_xyz.shape[0], device=device)
                mass_col = torch.zeros(xyz_Bp.shape[0], device=device)

                bucket_iter = buckets.items()
                bucket_iter = _tqdm_wrap(
                    bucket_iter,
                    enabled=tq_inner,
                    total=len(buckets),
                    desc=f"  Voxel buckets {sidx+1}/{S}",
                    leave=False,
                    dynamic_ncols=True
                )
                for key, idx_list in bucket_iter:
                    sub_idx_tile = torch.tensor(idx_list, device=device, dtype=torch.long)
                    if sub_idx_tile.numel() == 0:
                        continue
                    tile_min = mins + torch.tensor(key, device=device, dtype=torch.float32) * voxel_size
                    tile_max = tile_min + voxel_size
                    lo = tile_min - margin
                    hi = tile_max + margin

                    inb = (base_xyz[:, 0] >= lo[0]) & (base_xyz[:, 0] <= hi[0]) & \
                          (base_xyz[:, 1] >= lo[1]) & (base_xyz[:, 1] <= hi[1]) & \
                          (base_xyz[:, 2] >= lo[2]) & (base_xyz[:, 2] <= hi[2])
                    base_idx_tile = torch.nonzero(inb, as_tuple=False).reshape(-1)
                    if base_idx_tile.numel() == 0:
                        continue

                    base_tile = base_xyz[base_idx_tile]
                    query_tile = xyz_Bp[sub_idx_tile]
                    base_np = np.ascontiguousarray(base_tile.detach().float().cpu().numpy().astype(np.float32, copy=False))
                    query_np = np.ascontiguousarray(query_tile.detach().float().cpu().numpy().astype(np.float32, copy=False))

                    use_gpu_tile = torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources") and base_tile.shape[0] > 1024
                    if use_gpu_tile and _get_faiss_gpu_res() is not None:
                        index_t = faiss.index_cpu_to_gpu(_get_faiss_gpu_res(), _get_faiss_gpu_id(), faiss.IndexFlatL2(3))
                        index_t.add(base_np)
                    else:
                        index_t = faiss.IndexFlatL2(3)
                        index_t.add(base_np)

                    dist2, idx_local = index_t.search(query_np, eff_topk)
                    dist_tile = torch.from_numpy(dist2).to(device).sqrt()
                    idx_local_t = torch.from_numpy(idx_local).to(device).long()
                    valid = dist_tile <= params.merge_radius
                    has_any = torch.any(valid, dim=1)
                    if not has_any.any():
                        continue

                    sel_rows = idx_local_t[has_any]
                    sel_mask = valid[has_any]
                    row_idx_list = base_idx_tile[sel_rows[sel_mask]]
                    col_idx_rep = sub_idx_tile[has_any]
                    col_idx_list = col_idx_rep.repeat_interleave(sel_mask.sum(dim=1))

                    row_idx_list, col_idx_list = _apply_color_gate(row_idx_list, col_idx_list, base_dc, dc_s, color_th)
                    if row_idx_list.numel() == 0:
                        continue

                    row_act = torch.unique(row_idx_list)
                    col_act = torch.unique(col_idx_list)
                    row_map = -torch.ones(base_xyz.shape[0], device=device, dtype=torch.long)
                    col_map = -torch.ones(xyz_Bp.shape[0], device=device, dtype=torch.long)
                    row_map[row_act] = torch.arange(row_act.shape[0], device=device, dtype=torch.long)
                    col_map[col_act] = torch.arange(col_act.shape[0], device=device, dtype=torch.long)
                    row_loc = row_map[row_idx_list]
                    col_loc = col_map[col_idx_list]

                    a = (wA[row_act] / torch.sum(wA[row_act]).clamp_min(1e-8))
                    b = (wB[col_act] / torch.sum(wB[col_act]).clamp_min(1e-8))

                    pos_i = base_xyz[row_idx_list]
                    pos_j = xyz_Bp[col_idx_list]
                    C_pos = torch.sum((pos_i - pos_j) ** 2, dim=1)
                    cov_A_act = cov_A[row_act]
                    cov_A_sqrt_act = sqrtm_spd_3x3(cov_A_act)
                    tr_A_act = torch.sum(cov_A_act.diagonal(dim1=1, dim2=2), dim=1)
                    cov_B_act = cov_Bp[col_act]
                    tr_B_act = torch.sum(cov_B_act.diagonal(dim1=1, dim2=2), dim=1)
                    C_bures = _w2_bures_edges(cov_A_act, cov_A_sqrt_act, tr_A_act, cov_B_act, tr_B_act, row_loc, col_loc)
                    C_e = C_pos + C_bures

                    pi_e, _, _ = _sinkhorn_sparse_balanced(
                        row_loc, col_loc, C_e, a, b, params.eps, sinkhorn_iters_eff,
                        progress=tq_inner, desc=f"    Sinkhorn {sidx+1}/{S}"
                    )
                    mass_row.index_add_(0, row_idx_list, pi_e)
                    mass_col.index_add_(0, col_idx_list, pi_e)

                    edge_rows.append(row_idx_list)
                    edge_cols.append(col_idx_list)
                    edge_pi.append(pi_e)

                if len(edge_pi) > 0:
                    r_col = torch.clamp(mass_col / wB.clamp_min(1e-8), 0.0, 1.0)
                    denom = (wA + mass_row).clamp_min(1e-8)

                    row_all = torch.cat(edge_rows, dim=0)
                    col_all = torch.cat(edge_cols, dim=0)
                    pi_all = torch.cat(edge_pi, dim=0)
                    pi_scaled_all = pi_all * (tau_eff * r_col[col_all])

                    num_xyz = base_xyz * wA[:, None]
                    add_xyz = torch.zeros_like(base_xyz)
                    add_xyz.index_add_(0, row_all, pi_scaled_all[:, None] * xyz_Bp[col_all])
                    base_xyz = (num_xyz + add_xyz) / denom[:, None]

                    num_dc = base_dc * wA[:, None]
                    add_dc = torch.zeros_like(base_dc)
                    add_dc.index_add_(0, row_all, pi_scaled_all[:, None] * dc_s[col_all])
                    base_dc = (num_dc + add_dc) / denom[:, None]

                    if base_rest is not None and base_rest.numel() > 0:
                        num_rest = base_rest * wA[:, None]
                        add_rest = torch.zeros_like(base_rest)
                        add_rest.index_add_(0, row_all, pi_scaled_all[:, None] * rest_s[col_all])
                        base_rest = (num_rest + add_rest) / denom[:, None]

                    num_op = base_op.reshape(-1) * wA
                    add_op = torch.zeros_like(base_op.reshape(-1))
                    add_op.index_add_(0, row_all, pi_scaled_all * op_s.reshape(-1)[col_all])
                    base_op = torch.clamp((num_op + add_op) / denom, 0.0, 1.0).reshape(-1, 1)

                    base_scales, base_rots = _update_cov_only_active_rows(
                        base_scales, base_rots, wA, denom,
                        row_all, col_all, pi_all,
                        base_xyz, xyz_Bp, cov_Bp
                    )

                # --- insertion (detail or all) ---
                insert_mask = _choose_insert_mask(
                    insert_mode=insert_mode,
                    base_xyz=base_xyz,
                    xyz_Bp=xyz_Bp,
                    params=params,
                    insert_dist_th=insert_dist_th,
                    detail_vox=detail_vox,
                    main_min=main_min,
                    sub_cap=sub_cap,
                    idx_pack_main=idx_pack
                )

                if insert_mask.any():
                    ins_cov = cov_Bp[insert_mask]
                    ins_sc, ins_rt = cov_to_scales_rots(ins_cov)
                    ins_xyz = xyz_Bp[insert_mask]
                    ins_dc = dc_s[insert_mask]
                    ins_rest = rest_s[insert_mask] if rest_s is not None else torch.empty(
                        (ins_xyz.shape[0], base_rest.shape[1]), device=device, dtype=base_dc.dtype
                    )
                    ins_op = op_s[insert_mask]

                    if no_quality_gate:
                        keep_ins = torch.ones((ins_xyz.shape[0],), device=device, dtype=torch.bool)
                    else:
                        keep_ins = (ins_op.reshape(-1) >= params.opacity_th) & torch.all(ins_sc >= params.min_scale, dim=1)

                    if keep_ins.any():
                        ins_xyz = ins_xyz[keep_ins]
                        ins_sc = ins_sc[keep_ins]
                        ins_rt = ins_rt[keep_ins]
                        ins_dc = ins_dc[keep_ins]
                        ins_rest = ins_rest[keep_ins]
                        ins_op = ins_op[keep_ins]
                        add_n = int(ins_xyz.shape[0])
                        base_xyz = torch.cat([base_xyz, ins_xyz], dim=0)
                        base_scales = torch.cat([base_scales, ins_sc], dim=0)
                        base_rots = torch.cat([base_rots, ins_rt], dim=0)
                        base_dc = torch.cat([base_dc, ins_dc], dim=0)
                        base_op = _cat_op(base_op, ins_op)
                        base_rest = torch.cat([base_rest, ins_rest], dim=0)
                        origin = torch.cat([origin, torch.ones((add_n,), device=device, dtype=torch.uint8)], dim=0)

            else:
                # ----------------- non-voxel version -----------------
                dist, idx = _search_faiss_index(idx_pack, xyz_Bp, eff_topk)
                valid = dist <= params.merge_radius
                has_col = torch.any(valid, dim=1)
                row_idx_list = torch.zeros((0,), dtype=torch.long, device=device)
                col_idx_list = torch.zeros((0,), dtype=torch.long, device=device)
                pi_e = torch.zeros((0,), dtype=torch.float32, device=device)

                if has_col.any():
                    sel_rows = idx[has_col]
                    sel_mask = valid[has_col]
                    row_idx_list = sel_rows[sel_mask]
                    col_idx_rep = torch.arange(xyz_Bp.shape[0], device=device)[has_col]
                    col_idx_list = col_idx_rep.repeat_interleave(sel_mask.sum(dim=1))

                    row_idx_list, col_idx_list = _apply_color_gate(row_idx_list, col_idx_list, base_dc, dc_s, color_th)
                    if row_idx_list.numel() > 0:
                        row_act = torch.unique(row_idx_list)
                        col_act = torch.unique(col_idx_list)

                        row_map = -torch.ones(base_xyz.shape[0], device=device, dtype=torch.long)
                        col_map = -torch.ones(xyz_Bp.shape[0], device=device, dtype=torch.long)
                        row_map[row_act] = torch.arange(row_act.shape[0], device=device, dtype=torch.long)
                        col_map[col_act] = torch.arange(col_act.shape[0], device=device, dtype=torch.long)
                        row_loc = row_map[row_idx_list]
                        col_loc = col_map[col_idx_list]

                        a = (wA[row_act] / torch.sum(wA[row_act]).clamp_min(1e-8))
                        b = (wB[col_act] / torch.sum(wB[col_act]).clamp_min(1e-8))

                        pos_i = base_xyz[row_idx_list]
                        pos_j = xyz_Bp[col_idx_list]
                        C_pos = torch.sum((pos_i - pos_j) ** 2, dim=1)
                        cov_A_act = cov_A[row_act]
                        cov_A_sqrt_act = sqrtm_spd_3x3(cov_A_act)
                        tr_A_act = torch.sum(cov_A_act.diagonal(dim1=1, dim2=2), dim=1)
                        cov_B_act = cov_Bp[col_act]
                        tr_B_act = torch.sum(cov_B_act.diagonal(dim1=1, dim2=2), dim=1)
                        C_bures = _w2_bures_edges(cov_A_act, cov_A_sqrt_act, tr_A_act, cov_B_act, tr_B_act, row_loc, col_loc)
                        C_e = C_pos + C_bures
                        pi_e, _, _ = _sinkhorn_sparse_balanced(
                            row_loc, col_loc, C_e, a, b, params.eps, sinkhorn_iters_eff,
                            progress=tq_inner, desc=f"    Sinkhorn {sidx+1}/{S}"
                        )

                if pi_e.numel() > 0:
                    mass_row = torch.zeros(base_xyz.shape[0], device=device)
                    mass_row.index_add_(0, row_idx_list, pi_e)
                    mass_col = torch.zeros(xyz_Bp.shape[0], device=device)
                    mass_col.index_add_(0, col_idx_list, pi_e)

                    r_col = torch.clamp(mass_col / wB.clamp_min(1e-8), 0.0, 1.0)
                    pi_scaled = pi_e * (tau_eff * r_col[col_idx_list])

                    denom = (wA + mass_row).clamp_min(1e-8)

                    num_xyz = base_xyz * wA[:, None]
                    add_xyz = torch.zeros_like(base_xyz)
                    add_xyz.index_add_(0, row_idx_list, pi_scaled[:, None] * xyz_Bp[col_idx_list])
                    base_xyz = (num_xyz + add_xyz) / denom[:, None]

                    num_dc = base_dc * wA[:, None]
                    add_dc = torch.zeros_like(base_dc)
                    add_dc.index_add_(0, row_idx_list, pi_scaled[:, None] * dc_s[col_idx_list])
                    base_dc = (num_dc + add_dc) / denom[:, None]

                    if base_rest is not None and base_rest.numel() > 0:
                        num_rest = base_rest * wA[:, None]
                        add_rest = torch.zeros_like(base_rest)
                        add_rest.index_add_(0, row_idx_list, pi_scaled[:, None] * rest_s[col_idx_list])
                        base_rest = (num_rest + add_rest) / denom[:, None]

                    num_op = base_op.reshape(-1) * wA
                    add_op = torch.zeros_like(base_op.reshape(-1))
                    add_op.index_add_(0, row_idx_list, pi_scaled * op_s.reshape(-1)[col_idx_list])
                    base_op = torch.clamp((num_op + add_op) / denom, 0.0, 1.0).reshape(-1, 1)

                    base_scales, base_rots = _update_cov_only_active_rows(
                        base_scales, base_rots, wA, denom,
                        row_idx_list, col_idx_list, pi_e,
                        base_xyz, xyz_Bp, cov_Bp
                    )

                # insertion (detail or all)
                insert_mask = _choose_insert_mask(
                    insert_mode=insert_mode,
                    base_xyz=base_xyz,
                    xyz_Bp=xyz_Bp,
                    params=params,
                    insert_dist_th=insert_dist_th,
                    detail_vox=detail_vox,
                    main_min=main_min,
                    sub_cap=sub_cap,
                    idx_pack_main=idx_pack
                )

                if insert_mask.any():
                    ins_cov = cov_Bp[insert_mask]
                    ins_sc, ins_rt = cov_to_scales_rots(ins_cov)
                    ins_xyz = xyz_Bp[insert_mask]
                    ins_dc = dc_s[insert_mask]
                    ins_rest = rest_s[insert_mask] if rest_s is not None else torch.empty(
                        (ins_xyz.shape[0], base_rest.shape[1]), device=device, dtype=base_dc.dtype
                    )
                    ins_op = op_s[insert_mask]

                    if no_quality_gate:
                        keep_ins = torch.ones((ins_xyz.shape[0],), device=device, dtype=torch.bool)
                    else:
                        keep_ins = (ins_op.reshape(-1) >= params.opacity_th) & torch.all(ins_sc >= params.min_scale, dim=1)

                    if keep_ins.any():
                        ins_xyz = ins_xyz[keep_ins]
                        ins_sc = ins_sc[keep_ins]
                        ins_rt = ins_rt[keep_ins]
                        ins_dc = ins_dc[keep_ins]
                        ins_rest = ins_rest[keep_ins]
                        ins_op = ins_op[keep_ins]
                        add_n = int(ins_xyz.shape[0])
                        base_xyz = torch.cat([base_xyz, ins_xyz], dim=0)
                        base_scales = torch.cat([base_scales, ins_sc], dim=0)
                        base_rots = torch.cat([base_rots, ins_rt], dim=0)
                        base_dc = torch.cat([base_dc, ins_dc], dim=0)
                        base_op = _cat_op(base_op, ins_op)
                        base_rest = torch.cat([base_rest, ins_rest], dim=0)
                        origin = torch.cat([origin, torch.ones((add_n,), device=device, dtype=torch.uint8)], dim=0)

            # 不再对 base 做 prune，保证主图点数不减少

            if step_bar:
                step_bar.update(step_bar.total - step_bar.n)
                step_bar.close()

            sub_time = time.perf_counter() - sub_start
            if last_L_mw2 is None:
                last_L_mw2 = 0.0
            if last_L_prior is None:
                last_L_prior = 0.0
            _log(f"  merged base_N={int(base_xyz.shape[0])} time={sub_time:.2f}s | last(Lmw2={last_L_mw2:.2e}, Lp={last_L_prior:.2e}, edges={last_edges})")
            if tq_major:
                sub_iter.set_postfix_str(f"base_N={int(base_xyz.shape[0])} Lmw2={last_L_mw2:.1e}")

        if base_xyz is None:
            _log("No points to save; writing empty ply.")
            _save_ply_gaussians(out_ply, torch.zeros((0, 3)), torch.zeros((0, 3)),
                                torch.zeros((0, 4)), torch.zeros((0, 3)),
                                torch.zeros((0, 0)), torch.zeros((0, 1)))
            return

        # ----------------- FINAL: 主图优先去重/冲突剔除（只影响保存） -----------------
        # 默认关闭，避免任何形式的“下采样”；若需要可手动开启 MW2_DEDUP=1
        if os.environ.get("MW2_DEDUP", "0").lower() not in ("0", "false", "no"):
            _log("Final main-priority dedup before save...")
            main_mask = (origin == 0)
            sub_mask = (origin == 1)

            main_xyz, main_sc, main_rt, main_dc = base_xyz[main_mask], base_scales[main_mask], base_rots[main_mask], base_dc[main_mask]
            main_op = base_op[main_mask]
            if base_rest is not None and base_rest.numel() > 0:
                main_rest = base_rest[main_mask]
            else:
                main_rest = base_rest

            sub_xyz, sub_sc, sub_rt, sub_dc = base_xyz[sub_mask], base_scales[sub_mask], base_rots[sub_mask], base_dc[sub_mask]
            sub_op = base_op[sub_mask]
            if base_rest is not None and base_rest.numel() > 0:
                sub_rest = base_rest[sub_mask]
            else:
                sub_rest = base_rest

            if dedup_main_vox > 0:
                _log(f"  [warn] main dedup enabled (dedup_main_vox={dedup_main_vox}), this may reduce base points.")

            if dedup_main_vox > 0:
                before_m = int(main_xyz.shape[0])
                main_xyz, main_sc, main_rt, main_dc, main_rest, main_op = _dedup_voxel_weighted(
                    main_xyz, main_sc, main_rt, main_dc, main_rest, main_op, voxel=dedup_main_vox
                )
                _log(f"  main dedup: {before_m} -> {int(main_xyz.shape[0])}")

            if sub_xyz.numel() > 0 and sub_to_main_th > 0:
                idx_pack_main = _build_faiss_index(main_xyz)
                keep_sub = _filter_sub_by_main_nn(main_xyz, sub_xyz, sub_to_main_th, idx_pack_main)
                before_s = int(sub_xyz.shape[0])
                sub_xyz, sub_sc, sub_rt, sub_dc, sub_op = sub_xyz[keep_sub], sub_sc[keep_sub], sub_rt[keep_sub], sub_dc[keep_sub], sub_op[keep_sub]
                if sub_rest is not None and sub_rest.numel() > 0:
                    sub_rest = sub_rest[keep_sub]
                _log(f"  sub->main filter(th={sub_to_main_th:.6f}): {before_s} -> {int(sub_xyz.shape[0])}")

            if sub_xyz.numel() > 0 and dedup_sub_vox > 0:
                before_s2 = int(sub_xyz.shape[0])
                sub_xyz, sub_sc, sub_rt, sub_dc, sub_rest, sub_op = _dedup_voxel_weighted(
                    sub_xyz, sub_sc, sub_rt, sub_dc, sub_rest, sub_op, voxel=dedup_sub_vox
                )
                _log(f"  sub dedup: {before_s2} -> {int(sub_xyz.shape[0])}")

            base_xyz = torch.cat([main_xyz, sub_xyz], dim=0)
            base_scales = torch.cat([main_sc, sub_sc], dim=0)
            base_rots = torch.cat([main_rt, sub_rt], dim=0)
            base_dc = torch.cat([main_dc, sub_dc], dim=0)
            base_op = torch.cat([main_op, sub_op], dim=0)
            if base_rest is not None and base_rest.numel() > 0:
                base_rest = torch.cat([main_rest, sub_rest], dim=0)

            _log(f"  final main/sub: main={int(main_xyz.shape[0])} sub={int(sub_xyz.shape[0])} total={int(base_xyz.shape[0])}")

        # -------- 自动去雾（可选，MW2_DEFOG=1 开启）--------
        if os.environ.get("MW2_DEFOG", "0").lower() not in ("0", "false", "no"):
            _log("Auto defog enabled (MW2_DEFOG=1), running defog pass...")
            base_xyz, base_scales, base_rots, base_dc, base_rest, base_op = _defog_pass(
                base_xyz, base_scales, base_rots, base_dc, base_rest, base_op, verbose=True
            )

        if base_N0 is not None and int(base_xyz.shape[0]) < base_N0:
            _log(f"[warn] final N({int(base_xyz.shape[0])}) < initial base_N({base_N0}), this may happen after dedup/defog.")

        # 输出 scale 维度对齐
        sc_out = base_scales
        if save_scale_dim is not None and sc_out.shape[1] != save_scale_dim:
            if save_scale_dim < sc_out.shape[1]:
                sc_out = sc_out[:, :save_scale_dim]
            else:
                pad = torch.zeros((sc_out.shape[0], save_scale_dim - sc_out.shape[1]), device=sc_out.device, dtype=sc_out.dtype)
                sc_out = torch.cat([sc_out, pad], dim=1)

        _log(f"Saving PLY -> {out_ply} (N={base_xyz.shape[0]})")
        _save_ply_gaussians(out_ply, base_xyz, sc_out, base_rots, base_dc, base_rest, base_op)
        _log(f"Done. Total time {(time.perf_counter()-t0):.2f}s")
