import os
import torch
from typing import Tuple


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    n = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
    q = q / n
    sign = torch.sign(q[:, :1])
    sign[sign == 0] = 1.0
    return q * sign


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quat(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.empty((q.shape[0], 3, 3), dtype=q.dtype, device=q.device)
    R[:, 0, 0] = 1 - 2 * (yy + zz)
    R[:, 0, 1] = 2 * (xy - wz)
    R[:, 0, 2] = 2 * (xz + wy)
    R[:, 1, 0] = 2 * (xy + wz)
    R[:, 1, 1] = 1 - 2 * (xx + zz)
    R[:, 1, 2] = 2 * (yz - wx)
    R[:, 2, 0] = 2 * (xz - wy)
    R[:, 2, 1] = 2 * (yz + wx)
    R[:, 2, 2] = 1 - 2 * (xx + yy)
    return R


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    N = R.shape[0]
    q = torch.empty((N, 4), dtype=R.dtype, device=R.device)
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    mask = tr > 0
    if mask.any():
        S = torch.sqrt(tr[mask] + 1.0) * 2
        q[mask, 0] = 0.25 * S
        q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / S
        q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / S
        q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / S
    m = ~mask
    if m.any():
        Rm = R[m]
        idx0 = (Rm[:, 0, 0] > Rm[:, 1, 1]) & (Rm[:, 0, 0] > Rm[:, 2, 2])
        idx1 = ~idx0 & (Rm[:, 1, 1] > Rm[:, 2, 2])
        idx2 = ~idx0 & ~idx1
        rows_all = m.nonzero(as_tuple=False).squeeze(-1)
        if idx0.any():
            Rc = Rm[idx0]
            S = torch.sqrt(1.0 + Rc[:, 0, 0] - Rc[:, 1, 1] - Rc[:, 2, 2]) * 2
            rows = rows_all[idx0]
            q[rows, 0] = (Rc[:, 2, 1] - Rc[:, 1, 2]) / S
            q[rows, 1] = 0.25 * S
            q[rows, 2] = (Rc[:, 0, 1] + Rc[:, 1, 0]) / S
            q[rows, 3] = (Rc[:, 0, 2] + Rc[:, 2, 0]) / S
        if idx1.any():
            Rc = Rm[idx1]
            S = torch.sqrt(1.0 + Rc[:, 1, 1] - Rc[:, 0, 0] - Rc[:, 2, 2]) * 2
            rows = rows_all[idx1]
            q[rows, 0] = (Rc[:, 0, 2] - Rc[:, 2, 0]) / S
            q[rows, 1] = (Rc[:, 0, 1] + Rc[:, 1, 0]) / S
            q[rows, 2] = 0.25 * S
            q[rows, 3] = (Rc[:, 1, 2] + Rc[:, 2, 1]) / S
        if idx2.any():
            Rc = Rm[idx2]
            S = torch.sqrt(1.0 + Rc[:, 2, 2] - Rc[:, 0, 0] - Rc[:, 1, 1]) * 2
            rows = rows_all[idx2]
            q[rows, 0] = (Rc[:, 1, 0] - Rc[:, 0, 1]) / S
            q[rows, 1] = (Rc[:, 0, 2] + Rc[:, 2, 0]) / S
            q[rows, 2] = (Rc[:, 1, 2] + Rc[:, 2, 1]) / S
            q[rows, 3] = 0.25 * S
    return normalize_quat(q)


def scales_rots_to_cov(scales: torch.Tensor, rots: torch.Tensor) -> torch.Tensor:
    # C = R diag(s^2) R^T
    R = quat_to_rotmat(rots)
    S2 = scales * scales
    C = torch.einsum("nij,nj,nkj->nik", R, S2, R.transpose(1, 2))
    eye = torch.eye(3, dtype=C.dtype, device=C.device).unsqueeze(0)
    # Enforce symmetry+SPD jitter
    return 0.5 * (C + C.transpose(1, 2)) + 1e-10 * eye


def _eigh_stable(B: torch.Tensor):
    # Sanitize values, optionally force CPU if env asks
    B = torch.nan_to_num(B, nan=0.0, posinf=1e6, neginf=-1e6)
    force_cpu = os.environ.get("MW2_EIGH_DEVICE", "auto").lower() == "cpu"
    if force_cpu:
        vals, vecs = torch.linalg.eigh(B.double().cpu())
        return vals.float().to(B.device), vecs.float().to(B.device)
    try:
        return torch.linalg.eigh(B)
    except RuntimeError:
        # Fallback to CPU for pathological batches
        vals, vecs = torch.linalg.eigh(B.double().cpu())
        return vals.float().to(B.device), vecs.float().to(B.device)


def cov_to_scales_rots(C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if C.numel() == 0:
        return torch.empty((0, 3), dtype=C.dtype, device=C.device), torch.empty((0, 4), dtype=C.dtype, device=C.device)
    C = 0.5 * (C + C.transpose(1, 2))
    vals, vecs = _eigh_stable(C)
    vals = torch.clamp(vals, min=1e-12)
    s = torch.sqrt(vals)
    order = torch.argsort(s, dim=-1, descending=True)
    batch = torch.arange(C.shape[0], device=C.device).unsqueeze(-1).expand_as(order)
    s_sorted = s[batch, order]
    vecs_sorted = vecs[batch, :, order]
    q = rotmat_to_quat(vecs_sorted)
    return s_sorted, q


def sqrtm_spd_3x3(B: torch.Tensor) -> torch.Tensor:
    if B.numel() == 0:
        return B.clone()
    B = 0.5 * (B + B.transpose(1, 2))
    # Numerical hygiene
    B = torch.nan_to_num(B, nan=0.0, posinf=1e6, neginf=-1e6)
    vals, vecs = _eigh_stable(B)
    vals = torch.clamp(vals, min=1e-12)
    s = torch.sqrt(vals)
    return vecs @ torch.diag_embed(s) @ vecs.transpose(1, 2)
