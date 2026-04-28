import numpy as np
import torch
from typing import Tuple

try:
    from plyfile import PlyData
except ImportError:
    PlyData = None


def _stack_props(props, names, out_dim):
    arrs = []
    for n in names:
        if n not in props:
            raise KeyError(f"PLY missing property: {n}")
        arrs.append(props[n].astype(np.float32)[..., None])
    return np.concatenate(arrs, axis=1).reshape((-1, out_dim)).astype(np.float32)


def _collect_prefix(props, prefix):
    cols = [(int(k[len(prefix):]), k) for k in props.keys() if k.startswith(prefix)]
    if not cols:
        n = len(next(iter(props.values())))
        return np.zeros((n, 0), dtype=np.float32), []
    cols.sort(key=lambda x: x[0])
    mats = [props[k].astype(np.float32)[..., None] for _, k in cols]
    out = np.concatenate(mats, axis=1)
    return out, [k for _, k in cols]


def _infer_sh_degree(rest_dim: int) -> int:
    if rest_dim <= 0:
        return 0
    # rest_dim = 3 * ((deg+1)^2 - 1)
    deg = int(np.sqrt(rest_dim / 3.0 + 1.0) - 1.0 + 1e-6)
    return max(0, deg)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _normalize_quat_np(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.maximum(n, 1e-8)
    q = q / n
    mask = q[:, 0:1] < 0
    q[mask[:, 0]] *= -1.0
    return q


def load_ply_gaussians(path: str, device: torch.device) -> Tuple[torch.Tensor, ...]:
    if PlyData is None:
        raise RuntimeError("plyfile not installed. pip install plyfile")
    ply = PlyData.read(path)
    v = ply["vertex"].data
    props = {name: v[name] for name in v.dtype.names}

    xyz = _stack_props(props, ["x", "y", "z"], 3)

    # Scales: Graphdeco stores log-scales; count any number of scale_*
    scales_raw, _ = _collect_prefix(props, "scale_")
    if scales_raw.shape[1] == 0:
        raise KeyError("PLY missing properties: scale_*")
    sigma = np.exp(scales_raw.astype(np.float32))  # to actual sigma

    # Pad to 3D for SPD/Bures if needed
    if sigma.shape[1] == 2:
        sxy_mean = np.mean(sigma, axis=1, keepdims=True)
        sz = 1e-3 * sxy_mean  # thin thickness
        sigma = np.concatenate([sigma, sz], axis=1)
    elif sigma.shape[1] == 3:
        pass
    else:
        if sigma.shape[1] > 3:
            sigma = sigma[:, :3]
        else:
            pad = np.zeros((sigma.shape[0], 3 - sigma.shape[1]), dtype=np.float32)
            sigma = np.concatenate([sigma, pad], axis=1)

    rots = _stack_props(props, [f"rot_{i}" for i in range(4)], 4)
    rots = _normalize_quat_np(rots)

    # Opacity: Graphdeco saves logits; map to [0,1] for merging
    if "opacity" in props:
        op_raw = props["opacity"].astype(np.float32).reshape(-1, 1)
        if (op_raw.min() < 0.0) or (op_raw.max() > 1.0):
            op = _sigmoid(op_raw)
        else:
            op = np.clip(op_raw, 0.0, 1.0)
    elif "alpha" in props:
        op = props["alpha"].astype(np.float32).reshape(-1, 1)
        op = np.clip(op, 0.0, 1.0)
    else:
        raise KeyError("No opacity/alpha in PLY.")

    # SH DC or fallback to RGB
    try:
        dc = _stack_props(props, [f"f_dc_{i}" for i in range(3)], 3)
    except KeyError:
        if all(k in props for k in ["r", "g", "b"]):
            dc = _stack_props(props, ["r", "g", "b"], 3) / 255.0
        else:
            raise

    rest, _ = _collect_prefix(props, "f_rest_")

    xyz_t = torch.from_numpy(xyz).to(device)
    scales_t = torch.from_numpy(sigma).to(device)  # actual sigma, 3D
    rots_t = torch.from_numpy(rots).to(device)
    dc_t = torch.from_numpy(dc).to(device)
    rest_t = torch.from_numpy(rest).to(device) if rest.shape[1] > 0 else torch.empty((xyz.shape[0], 0), device=device, dtype=xyz_t.dtype)
    op_t = torch.from_numpy(op).to(device)

    sh_deg = _infer_sh_degree(rest_t.shape[1])
    return xyz_t, scales_t, rots_t, dc_t, rest_t, op_t, sh_deg
