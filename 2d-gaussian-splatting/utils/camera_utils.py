#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED_RESOLUTION = False
WARNED_DEVICE = False


def loadCam(args, id, cam_info, resolution_scale):
    """
    从 cam_info + args 构造一个 Camera。
    这里对 args.resolution / args.data_device 做了兼容处理：
      - resolution in [1,2,4,8]：按原 3DGS 的下采样因子处理
      - resolution 为 None / -1 / 0：不做 global downscale，直接使用原图尺寸（只乘 resolution_scale）
      - 其他数值：按目标宽度缩放
      - data_device 为 None：回退到 "cuda"
    """
    orig_w, orig_h = cam_info.image.size

    # -------------------------
    # 1. 解析 resolution
    # -------------------------
    res = getattr(args, "resolution", None)

    if res in [1, 2, 4, 8]:
        # 与原始 3DGS 一致：整数倍率缩放
        resolution = (
            round(orig_w / (resolution_scale * res)),
            round(orig_h / (resolution_scale * res)),
        )
    else:
        global WARNED_RESOLUTION

        # None / -1 / 0 -> 不缩放（只考虑 resolution_scale）
        if res is None or res in (-1, 0):
            if not WARNED_RESOLUTION:
                print(
                    "[ INFO ] 'resolution' is None / -1 / 0 in loadCam, "
                    "fallback to no global rescaling (global_down = 1.0)"
                )
                WARNED_RESOLUTION = True
            global_down = 1.0
        else:
            # 其它情况：解释为“目标宽度”，按比例缩放
            try:
                global_down = float(orig_w) / float(res)
            except Exception:
                # 万一 cfg_args 里写了奇怪的东西，直接兜底
                if not WARNED_RESOLUTION:
                    print(
                        f"[ WARNING ] Unexpected args.resolution={res}, "
                        "fallback to no global rescaling (global_down = 1.0)"
                    )
                    WARNED_RESOLUTION = True
                global_down = 1.0

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # -------------------------
    # 2. 加载图像 & mask
    # -------------------------
    if len(cam_info.image.split()) > 3:
        # RGBA
        resized_image_rgb = torch.cat(
            [PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]],
            dim=0,
        )
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        # RGB
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    # -------------------------
    # 3. 解析 data_device
    # -------------------------
    global WARNED_DEVICE
    data_device = getattr(args, "data_device", "cuda")

    # 有些 cfg_args 里把 data_device 存成 None，需要兜底
    if data_device is None:
        if not WARNED_DEVICE:
            print(
                "[Warning] args.data_device is None in loadCam, "
                "fallback to default 'cuda'"
            )
            WARNED_DEVICE = True
        data_device = "cuda"

    # 确保 data_device 可以被 torch.device 识别
    try:
        _ = torch.device(data_device)
    except Exception:
        if not WARNED_DEVICE:
            print(
                f"[Warning] Custom device {data_device} failed, "
                "fallback to default 'cuda'"
            )
            WARNED_DEVICE = True
        data_device = "cuda"

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device=data_device,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
    }
    return camera_entry
