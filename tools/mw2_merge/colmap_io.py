import torch
from pathlib import Path
from typing import Dict, Tuple, Optional


def _quat_to_rotmat_colmap(qw: float, qx: float, qy: float, qz: float) -> torch.Tensor:
    w, x, y, z = qw, qx, qy, qz
    R = torch.tensor([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=torch.float32)
    return R


def load_colmap_images_txt(model_dir: str) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    p = Path(model_dir)
    cand = []
    if p.is_file() and p.name == "images.txt":
        cand = [p]
    else:
        for rel in ["images.txt", "sparse/0/images.txt", "sparse/images.txt", "0/images.txt"]:
            f = p / rel
            if f.exists():
                cand = [f]
                break
    if not cand:
        raise FileNotFoundError(f"images.txt not found under: {model_dir}")
    fpath = cand[0]

    poses: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks = s.split()
            if len(toks) < 9:
                continue
            try:
                qw, qx, qy, qz = map(float, toks[1:5])
                tx, ty, tz = map(float, toks[5:8])
                name = " ".join(toks[9:]) if len(toks) > 9 else toks[0]
            except Exception:
                continue
            R = _quat_to_rotmat_colmap(qw, qx, qy, qz)
            t = torch.tensor([tx, ty, tz], dtype=torch.float32)
            poses[name] = (R, t)
    return poses


def camera_centers_from_poses(poses: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    centers: Dict[str, torch.Tensor] = {}
    for k, (R, t) in poses.items():
        C = - R.transpose(0, 1) @ t
        centers[k] = C
    return centers


def umeyama_sim3_from_centers(base_centers: Dict[str, torch.Tensor],
                              sub_centers: Dict[str, torch.Tensor],
                              device: torch.device,
                              min_common: int = 3) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    keys = sorted(list(set(base_centers.keys()) & set(sub_centers.keys())))
    if len(keys) < min_common:
        return None
    P = torch.stack([base_centers[k] for k in keys], dim=0).to(device)  # (N,3)
    Q = torch.stack([sub_centers[k] for k in keys], dim=0).to(device)   # (N,3)
    muP = P.mean(dim=0)
    muQ = Q.mean(dim=0)
    X = P - muP
    Y = Q - muQ
    C = (Y.transpose(0, 1) @ X) / P.shape[0]
    U, S, Vh = torch.linalg.svd(C)
    D = torch.eye(3, device=device, dtype=P.dtype)
    if torch.det(U @ Vh) < 0:
        D[2, 2] = -1.0
    R = U @ D @ Vh
    var = (Y * Y).sum() / P.shape[0]
    s = (S @ torch.diag(D).to(S)).sum() / var.clamp_min(1e-8)
    t = muP - s * (R @ muQ)
    return s.reshape(()), R, t
