import os
import sys
import argparse
import time
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console

CONSOLE = Console(width=120)


def run_command_safe(command: str):
    print(f"Running command: {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        print("Command failed!")
        sys.exit(1)
    else:
        print("Command succeeded!")


def assert_exists(path: str, msg: str = ""):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Path not found: {path}. {msg}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Single refinement using MW2 merged Gaussians on base data"
    )

    # ===== Scene / IO =====
    parser.add_argument(
        "-s", "--source_path", type=str, required=True,
        help="原始数据路径（例如 data/scan1/base，用于 eval / 对齐等）"
    )
    parser.add_argument(
        "--mast3r_scene", type=str, required=True,
        help="已有的 mast3r_sfm 目录，例如 ./output/scan1/base/mast3r_sfm"
    )
    parser.add_argument(
        "--init_gaussians_ply", type=str, required=True,
        help="合并后的 MW2 高斯 PLY，例如 ./output/scan1/mw2_merged.ply"
    )
    parser.add_argument(
        "-o", "--output_path", type=str, required=True,
        help="强化训练输出根目录，例如 ./output/scan1/reinforce"
    )

    # ===== Free Gaussians / Tetra 配置 =====
    parser.add_argument(
        "--free_gaussians_config", type=str, default="default",
        help="configs/free_gaussians_refinement/<name>.yaml"
    )
    parser.add_argument(
        "--dense_regul", type=str, default="default",
        help='Dense 正则强度："default"/"strong"/"weak"/"none"'
    )
    parser.add_argument(
        "--tetra_config", type=str, default="default",
        help="configs/tetra/<name>.yaml"
    )
    parser.add_argument(
        "--tetra_downsample_ratio", type=float, default=0.5,
        help="Tetra downsample ratio（默认 0.5）"
    )

    # ===== 训练 / eval 配置 =====
    parser.add_argument(
        "--iteration", type=int, default=7000,
        help="free_gaussians refine 输出用的 iteration（默认 7000）"
    )
    parser.add_argument(
        "--run_tetra", action="store_true",
        help="强化后顺便跑一下 tetra mesh 提取"
    )
    parser.add_argument(
        "--run_eval", action="store_true",
        help="在最终模型上运行 2d-gaussian-splatting/eval/eval.py"
    )
    parser.add_argument(
        "--sparse_view_num", type=int, default=10,
        help="eval 用的 sparse_view_num（对应 split-xxviews / 评估协议），默认 10"
    )

    args = parser.parse_args()

    # ===== 路径整理 =====
    source_path = args.source_path
    mast3r_scene_path = args.mast3r_scene      # 已有的 mast3r_sfm 目录（base 的 mast3r_sfm）
    init_gaussians_ply = args.init_gaussians_ply
    output_root = args.output_path
    iteration = args.iteration

    free_gaussians_path = os.path.join(output_root, "free_gaussians_reinforce")
    tetra_meshes_path = os.path.join(output_root, "tetra_meshes_reinforce")

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(free_gaussians_path, exist_ok=True)
    os.makedirs(tetra_meshes_path, exist_ok=True)

    # base 的 plane-refine-depths（直接复用 base 数据，不再重跑 See3D/plane refine）
    plane_root_path = os.path.join(mast3r_scene_path, "plane-refine-depths")

    CONSOLE.print("\n[reinforce] === Single-refine config ===")
    CONSOLE.print(f"[reinforce] source_path        : {source_path}")
    CONSOLE.print(f"[reinforce] mast3r_scene       : {mast3r_scene_path}")
    CONSOLE.print(f"[reinforce] init_gaussians_ply : {init_gaussians_ply}")
    CONSOLE.print(f"[reinforce] output_root        : {output_root}")
    CONSOLE.print(f"[reinforce] free_gaussians_cfg : {args.free_gaussians_config}")
    CONSOLE.print(f"[reinforce] dense_regul        : {args.dense_regul}")
    CONSOLE.print(f"[reinforce] plane_refine_depth : {plane_root_path}")
    CONSOLE.print(f"[reinforce] iteration          : {iteration}")
    CONSOLE.print(f"[reinforce] run_tetra          : {args.run_tetra}")
    CONSOLE.print(f"[reinforce] run_eval           : {args.run_eval}")
    CONSOLE.print(f"[reinforce] sparse_view_num    : {args.sparse_view_num}")
    CONSOLE.print("====================================\n")

    # ===== 命令封装 =====
    def get_refine_free_gaussians_cmd(init_gaussians_ply_path: str) -> str:
        return " ".join([
            "python", "scripts/refine_free_gaussians.py",
            "--mast3r_scene", mast3r_scene_path,
            "--output_path", free_gaussians_path,
            "--config", args.free_gaussians_config,
            "--dense_regul", args.dense_regul,
            "--refine_depth_path", plane_root_path,
            "--init_gaussians_ply", init_gaussians_ply_path,
            # 如需 downsample，可加： "--use_downsample_gaussians"
        ])

    render_all_img_command = " ".join([
        "python", "2d-gaussian-splatting/render_multires.py",
        "--source_path", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--skip_test",
        "--skip_mesh",
        "--render_all_img",
        "--use_default_output_dir",
    ])

    tetra_cmd = " ".join([
        "python", "scripts/extract_tetra_mesh.py",
        "--mast3r_scene", mast3r_scene_path,
        "--model_path", free_gaussians_path,
        "--output_path", tetra_meshes_path,
        "--config", args.tetra_config,
        "--downsample_ratio", str(args.tetra_downsample_ratio),
        "--interpolate_views",
    ])

    # eval（默认用 free_gaussians_path 做 model_path）
    eval_model_path = output_root
    eval_command = " ".join([
        "python", "2d-gaussian-splatting/eval/eval.py",
        "--source_path", source_path,
        "--model_path", eval_model_path,
        "--iteration", str(iteration),
        "--sparse_view_num", str(args.sparse_view_num)
    ])

    t1 = time.time()

    # ===================================================================
    # 单次 Stage：使用 merged PLY + base plane-refine-depths 做一次 refine
    # ===================================================================
    CONSOLE.print("[reinforce] Single Stage: refine from merged PLY + base plane-refine-depths")
    assert_exists(init_gaussians_ply, "init_gaussians_ply is required.")
    assert_exists(plane_root_path, "base 的 plane-refine-depths 不存在，请先跑完 base pipeline。")

    cmd_refine = get_refine_free_gaussians_cmd(init_gaussians_ply)
    CONSOLE.print(f"[reinforce] Command:\n{cmd_refine}\n")
    run_command_safe(cmd_refine)

    # 简单检查一下输出的 point_cloud 是否存在（不强依赖路径结构，可以按需删掉）
    pc_ply = os.path.join(
        free_gaussians_path,
        "point_cloud",
        f"iteration_{iteration}",
        "point_cloud.ply",
    )
    try:
        assert_exists(pc_ply, "Refine 没有在预期位置生成 point_cloud.ply，请检查 refine_free_gaussians.py 的保存路径。")
        CONSOLE.print(f"[reinforce] point_cloud.ply written at: {pc_ply}")
    except FileNotFoundError as e:
        CONSOLE.print(str(e))

    # ===================================================================
    # 最后：渲染所有图像 + 可选 Tetra mesh + 可选 Eval
    # ===================================================================
    CONSOLE.print("\n[reinforce] Final: render all images")
    run_command_safe(render_all_img_command)

    if args.run_tetra:
        CONSOLE.print("\n[reinforce] Final: extract tetra mesh")
        CONSOLE.print(f"[reinforce] Command:\n{tetra_cmd}\n")
        run_command_safe(tetra_cmd)

    if args.run_eval:
        CONSOLE.print("\n[reinforce] Final: eval")
        CONSOLE.print(f"[reinforce] Command:\n{eval_command}\n")
        run_command_safe(eval_command)

    t2 = time.time()
    CONSOLE.print(f"\n[reinforce] Single refinement pipeline finished. Total time: {t2 - t1:.2f} seconds\n")
