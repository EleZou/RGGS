#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge multiple COLMAP models into base coordinate frame using Sim(3) alignment
estimated from shared image names (camera centers).

Usage example:

  python scripts/merge_colmap_to_base.py \
      --base ./output/scan1/base/mast3r_sfm/all-sparse/0 \
      --subs ./output/scan1/5views_gen/mast3r_sfm/all-sparse/0 \
      --out  ./output/scan1/reinforce/mast3r_sfm/merged-sparse

要求：
- base_dir 和每个 sub_dir 都包含 cameras.txt / images.txt / points3D.txt
- sub 的 images.txt 中，至少有若干图片与 base 的图片名字相同
  （用来做 Umeyama Sim3 对齐）。如果公共图片 < 3，则该 sub 使用
  单位变换（s=1, R=I, t=0），并给出 warning。
"""

import os
import sys
import argparse
from typing import Dict, List, Tuple

import numpy as np


# ============================================================
#  基本数学工具：四元数 <-> 旋转矩阵 / Umeyama Sim3
# ============================================================

def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    """
    COLMAP convention: q = [qw, qx, qy, qz], world->cam.

    return: 3x3 rotation matrix
    """
    assert q.shape == (4,)
    qw, qx, qy, qz = q
    # normalize just in case
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = q / n

    R = np.array([
        [1 - 2*(qy*qy + qz*qz),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
    ], dtype=np.float64)
    return R


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """
    Inverse of qvec_to_rotmat, returning COLMAP qvec = [qw, qx, qy, qz].
    """
    assert R.shape == (3, 3)
    m = R
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (m[2, 1] - m[1, 2]) * s
        qy = (m[0, 2] - m[2, 0]) * s
        qz = (m[1, 0] - m[0, 1]) * s
    else:
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    q = np.array([qw, qx, qy, qz], dtype=np.float64)
    # enforce qw >= 0 for consistency
    if q[0] < 0:
        q = -q
    return q


def umeyama_sim3(
    src: np.ndarray,  # [N,3]
    dst: np.ndarray,  # [N,3]
    with_scale: bool = True
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Umeyama similarity transform: dst ~= s * R * src + t
    returns: s (scalar), R (3x3), t (3,)
    """
    assert src.shape == dst.shape
    n, dim = src.shape
    assert dim == 3 and n >= 3

    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_centered = src - mean_src
    dst_centered = dst - mean_dst

    cov = dst_centered.T @ src_centered / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    if with_scale:
        var_src = (src_centered ** 2).sum() / n
        s = np.trace(np.diag(D) @ S) / max(var_src, 1e-12)
    else:
        s = 1.0

    t = mean_dst - s * (R @ mean_src)
    return float(s), R, t


# ============================================================
#  COLMAP 文本 IO
# ============================================================

class Camera:
    def __init__(self, camera_id: int, model: str, width: int, height: int, params: List[float]):
        self.id = camera_id
        self.model = model
        self.width = width
        self.height = height
        self.params = list(params)

    def to_line(self) -> str:
        params_str = " ".join(map(str, self.params))
        return f"{self.id} {self.model} {self.width} {self.height} {params_str}"


class Image:
    def __init__(
        self,
        image_id: int,
        qvec: np.ndarray,  # [4] (qw,qx,qy,qz)
        tvec: np.ndarray,  # [3]
        camera_id: int,
        name: str,
        xys: np.ndarray,       # [P,2]
        point3d_ids: np.ndarray  # [P]
    ):
        self.id = image_id
        self.qvec = qvec.astype(np.float64)
        self.tvec = tvec.astype(np.float64)
        self.camera_id = camera_id
        self.name = name
        self.xys = xys.astype(np.float64)
        self.point3d_ids = point3d_ids.astype(np.int64)

    def to_lines(self) -> List[str]:
        q = " ".join(map(str, self.qvec.tolist()))
        t = " ".join(map(str, self.tvec.tolist()))
        header = f"{self.id} {q} {t} {self.camera_id} {self.name}"
        if self.xys.size == 0:
            return [header, ""]
        parts = []
        for (xy, pid) in zip(self.xys, self.point3d_ids):
            parts.append(f"{xy[0]} {xy[1]} {int(pid)}")
        return [header, " ".join(parts)]


class Point3D:
    def __init__(
        self,
        point3d_id: int,
        xyz: np.ndarray,  # [3]
        rgb: np.ndarray,  # [3]
        error: float,
        track: List[Tuple[int, int]]  # list of (image_id, point2d_idx)
    ):
        self.id = point3d_id
        self.xyz = xyz.astype(np.float64)
        self.rgb = rgb.astype(np.float64)
        self.error = float(error)
        self.track = list(track)

    def to_line(self) -> str:
        head = f"{self.id} {self.xyz[0]} {self.xyz[1]} {self.xyz[2]} " \
               f"{int(self.rgb[0])} {int(self.rgb[1])} {int(self.rgb[2])} {self.error}"
        tail = ""
        if len(self.track) > 0:
            parts = []
            for (img_id, idx) in self.track:
                parts.append(f"{img_id} {idx}")
            tail = " " + " ".join(parts)
        return head + tail


def read_cameras_txt(path: str) -> Dict[int, Camera]:
    cams: Dict[int, Camera] = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            toks = line.split()
            camera_id = int(toks[0])
            model = toks[1]
            width = int(toks[2])
            height = int(toks[3])
            params = list(map(float, toks[4:]))
            cams[camera_id] = Camera(camera_id, model, width, height, params)
    return cams


def read_images_txt(path: str) -> Dict[int, Image]:
    images: Dict[int, Image] = {}
    with open(path, 'r') as f:
        lines = [l.rstrip("\n") for l in f]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line[0] == '#':
            i += 1
            continue
        # header
        toks = line.split()
        image_id = int(toks[0])
        qvec = np.array(list(map(float, toks[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, toks[5:8])), dtype=np.float64)
        camera_id = int(toks[8])
        name = " ".join(toks[9:])
        # 2D-3D
        i += 1
        if i >= len(lines):
            xys = np.zeros((0, 2), dtype=np.float64)
            pids = np.zeros((0,), dtype=np.int64)
        else:
            line2 = lines[i].strip()
            if not line2 or line2[0] == '#':
                xys = np.zeros((0, 2), dtype=np.float64)
                pids = np.zeros((0,), dtype=np.int64)
            else:
                toks2 = line2.split()
                assert len(toks2) % 3 == 0
                n = len(toks2) // 3
                xs = []
                ys = []
                pids = []
                for j in range(n):
                    x = float(toks2[3*j + 0])
                    y = float(toks2[3*j + 1])
                    pid = int(toks2[3*j + 2])
                    xs.append(x)
                    ys.append(y)
                    pids.append(pid)
                xys = np.stack([xs, ys], axis=1)
                pids = np.array(pids, dtype=np.int64)
        images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, pids)
        i += 1
    return images


def read_points3D_txt(path: str) -> Dict[int, Point3D]:
    pts: Dict[int, Point3D] = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line[0] == '#':
                continue
            toks = line.split()
            point3d_id = int(toks[0])
            x, y, z = map(float, toks[1:4])
            r, g, b = map(int, toks[4:7])
            error = float(toks[7])
            track_raw = toks[8:]
            assert len(track_raw) % 2 == 0
            track: List[Tuple[int, int]] = []
            for i in range(0, len(track_raw), 2):
                img_id = int(track_raw[i])
                idx = int(track_raw[i + 1])
                track.append((img_id, idx))
            pts[point3d_id] = Point3D(
                point3d_id,
                np.array([x, y, z], dtype=np.float64),
                np.array([r, g, b], dtype=np.float64),
                error,
                track
            )
    return pts


def write_cameras_txt(path: str, cams: Dict[int, Camera]):
    with open(path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: {}\n".format(len(cams)))
        for cam_id in sorted(cams.keys()):
            f.write(cams[cam_id].to_line() + "\n")


def write_images_txt(path: str, images: Dict[int, Image]):
    with open(path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: {}\n".format(len(images)))
        for img_id in sorted(images.keys()):
            lines = images[img_id].to_lines()
            f.write(lines[0] + "\n")
            f.write(lines[1] + "\n")


def write_points3D_txt(path: str, pts: Dict[int, Point3D]):
    with open(path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
                "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: {}\n".format(len(pts)))
        for pid in sorted(pts.keys()):
            f.write(pts[pid].to_line() + "\n")


# ============================================================
#  工具函数
# ============================================================

def image_centers_from_images(images: Dict[int, Image]) -> Dict[str, np.ndarray]:
    """
    根据 image 的 qvec/tvec 计算相机中心（世界坐标），返回 dict: name -> center(3,)
    world->cam: x_cam = R X + t
    camera center: C = - R^T t
    """
    name2center: Dict[str, np.ndarray] = {}
    for img in images.values():
        R = qvec_to_rotmat(img.qvec)
        t = img.tvec
        C = - R.T @ t
        name2center[img.name] = C
    return name2center


def apply_sim3_to_image(
    img: Image,
    s: float,
    R_sim: np.ndarray,
    t_sim: np.ndarray
) -> Image:
    """
    把 sub 模型中的 image extrinsic 转到 base 世界坐标：
    - camera 旋转 R_cam 不变（相机局部坐标不变）
    - camera center C' = s * R_sim * C + t_sim
    - t' = - R_cam * C'

    这样世界点和相机中心都统一到 base frame。
    """
    R_cam = qvec_to_rotmat(img.qvec)
    t_cam = img.tvec
    C = - R_cam.T @ t_cam  # 原 world
    C_new = s * (R_sim @ C) + t_sim
    t_new = - R_cam @ C_new
    q_new = rotmat_to_qvec(R_cam)

    return Image(
        image_id=img.id,  # 仅仅占位，外面会重写 id
        qvec=q_new,
        tvec=t_new,
        camera_id=img.camera_id,
        name=img.name,
        xys=img.xys,
        point3d_ids=img.point3d_ids
    )


def apply_sim3_to_point(
    pt: Point3D,
    s: float,
    R_sim: np.ndarray,
    t_sim: np.ndarray
) -> Point3D:
    xyz_new = s * (R_sim @ pt.xyz) + t_sim
    return Point3D(
        point3d_id=pt.id,  # 外面会重写 id
        xyz=xyz_new,
        rgb=pt.rgb,
        error=pt.error,
        track=list(pt.track)
    )


# ============================================================
#  主逻辑
# ============================================================

def merge_colmap(
    base_dir: str,
    sub_dirs: List[str],
    out_dir: str,
    min_common_views: int = 3
):
    os.makedirs(out_dir, exist_ok=True)

    base_cams = read_cameras_txt(os.path.join(base_dir, "cameras.txt"))
    base_imgs = read_images_txt(os.path.join(base_dir, "images.txt"))
    base_pts = read_points3D_txt(os.path.join(base_dir, "points3D.txt"))

    print(f"[merge-colmap] Base: cams={len(base_cams)} images={len(base_imgs)} pts={len(base_pts)}")

    # 新的全局容器
    merged_cams: Dict[int, Camera] = {}
    merged_imgs: Dict[int, Image] = {}
    merged_pts: Dict[int, Point3D] = {}

    # 先把 base 全部拷进去
    for cam in base_cams.values():
        merged_cams[cam.id] = Camera(cam.id, cam.model, cam.width, cam.height, cam.params)

    for img in base_imgs.values():
        merged_imgs[img.id] = Image(
            img.id, img.qvec.copy(), img.tvec.copy(),
            img.camera_id, img.name,
            img.xys.copy(), img.point3d_ids.copy()
        )

    for pt in base_pts.values():
        merged_pts[pt.id] = Point3D(
            pt.id, pt.xyz.copy(), pt.rgb.copy(), pt.error, list(pt.track)
        )

    # id offset
    next_cam_id = (max(merged_cams.keys()) if merged_cams else 0) + 1
    next_img_id = (max(merged_imgs.keys()) if merged_imgs else 0) + 1
    next_pt_id = (max(merged_pts.keys()) if merged_pts else 0) + 1

    base_centers = image_centers_from_images(base_imgs)

    for si, sdir in enumerate(sub_dirs):
        print(f"[merge-colmap] ==== Sub {si+1}/{len(sub_dirs)}: {sdir} ====")
        cams_s = read_cameras_txt(os.path.join(sdir, "cameras.txt"))
        imgs_s = read_images_txt(os.path.join(sdir, "images.txt"))
        pts_s = read_points3D_txt(os.path.join(sdir, "points3D.txt"))

        print(f"[merge-colmap]   sub cams={len(cams_s)} images={len(imgs_s)} pts={len(pts_s)}")

        # 根据图片名字求共同相机中心，估计 Sim3
        sub_centers = image_centers_from_images(imgs_s)
        common_names = sorted(set(base_centers.keys()) & set(sub_centers.keys()))
        print(f"[merge-colmap]   common views (by name) = {len(common_names)}")

        if len(common_names) >= min_common_views:
            base_pts_arr = np.stack([base_centers[n] for n in common_names], axis=0)
            sub_pts_arr = np.stack([sub_centers[n] for n in common_names], axis=0)
            s, R_sim, t_sim = umeyama_sim3(sub_pts_arr, base_pts_arr, with_scale=True)
            print(f"[merge-colmap]   estimated Sim3: scale={s:.6f}")
        else:
            # 不足以估计 Sim3，退化为单位变换
            print(f"[merge-colmap][WARN] common views < {min_common_views}, "
                  f"use identity Sim3 for this sub.")
            s = 1.0
            R_sim = np.eye(3, dtype=np.float64)
            t_sim = np.zeros(3, dtype=np.float64)

        # ----------------- 合并 cameras -----------------
        cam_id_map: Dict[int, int] = {}
        for cam in cams_s.values():
            new_id = next_cam_id
            next_cam_id += 1
            cam_id_map[cam.id] = new_id
            merged_cams[new_id] = Camera(new_id, cam.model, cam.width, cam.height, cam.params)
        print(f"[merge-colmap]   added {len(cams_s)} cameras")

        # ----------------- 合并 images（并应用 Sim3 到 extrinsics） -----------------
        img_id_map: Dict[int, int] = {}
        for img in imgs_s.values():
            img_aligned = apply_sim3_to_image(img, s, R_sim, t_sim)
            new_img_id = next_img_id
            next_img_id += 1
            img_id_map[img.id] = new_img_id

            new_cam_id = cam_id_map[img.camera_id]
            merged_imgs[new_img_id] = Image(
                image_id=new_img_id,
                qvec=img_aligned.qvec,
                tvec=img_aligned.tvec,
                camera_id=new_cam_id,
                name=img_aligned.name,
                xys=img_aligned.xys,
                point3d_ids=img_aligned.point3d_ids
            )
        print(f"[merge-colmap]   added {len(imgs_s)} images")

        # ----------------- 合并 points3D（也应用 Sim3） -----------------
        added_pts = 0
        for pt in pts_s.values():
            pt_aligned = apply_sim3_to_point(pt, s, R_sim, t_sim)

            # 重映射 track 的 image_id
            new_track: List[Tuple[int, int]] = []
            for (img_id, idx) in pt_aligned.track:
                if img_id in img_id_map:
                    new_track.append((img_id_map[img_id], idx))
            if len(new_track) == 0:
                # 没有任何有效 track 的点，可以直接丢弃
                continue

            new_pt_id = next_pt_id
            next_pt_id += 1

            merged_pts[new_pt_id] = Point3D(
                point3d_id=new_pt_id,
                xyz=pt_aligned.xyz,
                rgb=pt_aligned.rgb,
                error=pt_aligned.error,
                track=new_track
            )
            added_pts += 1

        print(f"[merge-colmap]   added {added_pts} points3D (with non-empty tracks)")

    # ============================================================
    #  写结果
    # ============================================================
    out_cams = os.path.join(out_dir, "cameras.txt")
    out_imgs = os.path.join(out_dir, "images.txt")
    out_pts  = os.path.join(out_dir, "points3D.txt")

    print(f"[merge-colmap] Writing merged model to: {out_dir}")
    print(f"[merge-colmap]   cameras.txt  (N={len(merged_cams)})")
    write_cameras_txt(out_cams, merged_cams)
    print(f"[merge-colmap]   images.txt   (N={len(merged_imgs)})")
    write_images_txt(out_imgs, merged_imgs)
    print(f"[merge-colmap]   points3D.txt (N={len(merged_pts)})")
    write_points3D_txt(out_pts, merged_pts)

    print("[merge-colmap] Done.")


def main():
    ap = argparse.ArgumentParser("Merge COLMAP models into base coordinate frame via Sim3 from shared views.")
    ap.add_argument("--base", type=str, required=True,
                    help="Base COLMAP text model dir (contains cameras.txt, images.txt, points3D.txt)")
    ap.add_argument("--subs", type=str, nargs="+", required=True,
                    help="One or more sub COLMAP model dirs")
    ap.add_argument("--out", type=str, required=True,
                    help="Output directory for merged COLMAP model")
    ap.add_argument("--min-common-views", type=int, default=3,
                    help="Minimum shared image names to estimate Sim3. "
                         "If less, fall back to identity transform.")
    args = ap.parse_args()

    base_dir = args.base
    sub_dirs = args.subs
    out_dir = args.out

    for d in [base_dir] + sub_dirs:
        for fn in ["cameras.txt", "images.txt", "points3D.txt"]:
            p = os.path.join(d, fn)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing {fn} in {d}")

    merge_colmap(base_dir, sub_dirs, out_dir, min_common_views=args.min_common_views)


if __name__ == "__main__":
    main()
