import os
import sys
import argparse
import time
import shutil
from rich.console import Console

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONSOLE = Console(width=120)


def run_command_safe(command: str):
    CONSOLE.print(f"[cmd] {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        CONSOLE.print("[red]Command failed![/red]")
        sys.exit(1)
    CONSOLE.print("[green]Command succeeded![/green]")


def assert_exists(path: str, msg: str = ""):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Path not found: {path}. {msg}")


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def backup_if_exists(path: str):
    """If path exists (dir/file/symlink), move to *.bak-<timestamp>."""
    if os.path.islink(path) or os.path.exists(path):
        bk = f"{path}.bak-{_timestamp()}"
        shutil.move(path, bk)
        CONSOLE.print(f"[backup] {path} -> {bk}")


def ensure_empty_dir_with_backup(dir_path: str):
    if os.path.exists(dir_path) or os.path.islink(dir_path):
        backup_if_exists(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def ensure_symlink_with_backup(link_path: str, target_path: str):
    if os.path.exists(link_path) or os.path.islink(link_path):
        backup_if_exists(link_path)
    os.makedirs(target_path, exist_ok=True)
    os.symlink(os.path.abspath(target_path), link_path)
    CONSOLE.print(f"[symlink] {link_path} -> {target_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Phase-1: optimize with base See3D once; Phase-2: fresh re-run See3D based on refined model (See3D stages EXACTLY like official)."
    )

    # ===== IO =====
    parser.add_argument("-s", "--source_path", type=str, required=True,
                        help="原始数据路径（用于 eval 等）")
    parser.add_argument("--mast3r_scene", type=str, required=True,
                        help="已有 mast3r_sfm 目录（base），例如 ./output/scan1/base/mast3r_sfm")
    parser.add_argument("--init_gaussians_ply", type=str, required=True,
                        help="MW2 merged ply（仅 Phase-1 用来启动）")
    parser.add_argument("-o", "--output_path", type=str, required=True,
                        help="reinforce 输出根目录")

    # ===== refine / tetra / eval =====
    parser.add_argument("--free_gaussians_config", type=str, default="default")
    parser.add_argument("--dense_regul", type=str, default="default")
    parser.add_argument("--tetra_config", type=str, default="default")
    parser.add_argument("--tetra_downsample_ratio", type=float, default=0.5)

    parser.add_argument("--iteration", type=int, default=7000)
    parser.add_argument("--run_tetra", action="store_true")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--sparse_view_num", type=int, default=10)

    # ===== Phase-2 fresh See3D =====
    # “fresh 模式默认开启” -> 默认 True，用 --skip_fresh_see3d 关闭
    parser.add_argument("--skip_fresh_see3d", action="store_true",
                        help="关闭 Phase-2 fresh See3D（默认开启）")
    parser.add_argument("--select_inpaint_num", type=int, default=20)
    parser.add_argument("--see3d_max_stage", type=int, default=3)

    args = parser.parse_args()

    run_fresh_see3d = (not args.skip_fresh_see3d)

    source_path = args.source_path
    mast3r_scene_path = args.mast3r_scene
    init_gaussians_ply = args.init_gaussians_ply
    output_root = args.output_path
    iteration = args.iteration

    free_gaussians_path = os.path.join(output_root, "free_gaussians")
    tetra_meshes_path = os.path.join(output_root, "tetra_meshes")

    os.makedirs(output_root, exist_ok=True)
    os.makedirs(free_gaussians_path, exist_ok=True)
    os.makedirs(tetra_meshes_path, exist_ok=True)

    # ===== base artifacts (Phase-1 uses them) =====
    base_plane_root = os.path.join(mast3r_scene_path, "plane-refine-depths")
    base_see3d_root = os.path.join(mast3r_scene_path, "see3d_render")
    pnts_path = os.path.join(mast3r_scene_path, "chart_pcd.ply")

    assert_exists(mast3r_scene_path, "mast3r_scene required")
    assert_exists(init_gaussians_ply, "init_gaussians_ply required")
    assert_exists(base_plane_root, "base plane-refine-depths required")
    assert_exists(base_see3d_root, "base see3d_render required")
    assert_exists(pnts_path, "chart_pcd.ply required")

    # ===== Phase-2 fresh real dirs (do not pollute base) =====
    fresh_plane_real = os.path.join(output_root, "plane-refine-depths-resee3d")
    fresh_see3d_real = os.path.join(output_root, "see3d_render-resee3d")

    # IMPORTANT: many scripts hardcode these two:
    plane_link_in_scene = os.path.join(mast3r_scene_path, "plane-refine-depths")
    see3d_link_in_scene = os.path.join(mast3r_scene_path, "see3d_render")

    # ===== command builders =====
    def get_refine_free_gaussians_cmd(refine_depth_path: str, init_ply: str = None) -> str:
        """
        IMPORTANT:
        - Phase-1: pass init_ply (merged ply) to start
        - Phase-2: init_ply MUST be None (EXACTLY like official script)
        """
        parts = [
            "python", "scripts/refine_free_gaussians.py",
            "--mast3r_scene", mast3r_scene_path,
            "--output_path", free_gaussians_path,
            "--config", args.free_gaussians_config,
            "--dense_regul", args.dense_regul,
            "--refine_depth_path", refine_depth_path,
        ]
        if init_ply is not None:
            parts += ["--init_gaussians_ply", init_ply]
        return " ".join(parts)

    def get_plane_refine_depth_cmd(plane_root: str, see3d_root: str = None, anchor_json: str = None) -> str:
        parts = [
            "python", "scripts/plane_refine_depth.py",
            "--source_path", mast3r_scene_path,
            "--plane_root_path", plane_root,
            "--pnts_path", pnts_path,
        ]
        if see3d_root is not None:
            parts += ["--see3d_root_path", see3d_root]
        if anchor_json is not None:
            parts += ["--anchor_view_id_json_path", anchor_json]
        return " ".join(parts)

    def get_see3d_inpaint_cmd(stage: int, plane_root_dir: str) -> str:
        return " ".join([
            "python", "scripts/see3d_inpaint.py",
            "--source_path", mast3r_scene_path,
            "--model_path", free_gaussians_path,
            "--plane_root_dir", plane_root_dir,
            "--iteration", str(iteration),
            "--see3d_stage", str(stage),
            "--select_inpaint_num", str(args.select_inpaint_num),
        ])

    render_charts_cmd = lambda plane_root: " ".join([
        "python", "2d-gaussian-splatting/render_chart_views.py",
        "--source_path", mast3r_scene_path,
        "--save_root_path", plane_root,
    ])

    plane_excavator_cmd = lambda plane_root: " ".join([
        "python", "2d-gaussian-splatting/planes/plane_excavator.py",
        "--plane_root_path", plane_root,
    ])

    render_all_img_cmd = " ".join([
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

    eval_cmd = " ".join([
        "python", "2d-gaussian-splatting/eval/eval.py",
        "--source_path", source_path,
        "--model_path", output_root,
        "--iteration", str(iteration),
        "--sparse_view_num", str(args.sparse_view_num),
    ])

    CONSOLE.print("\n[reinforce] === Config ===")
    CONSOLE.print(f"[reinforce] mast3r_scene       : {mast3r_scene_path}")
    CONSOLE.print(f"[reinforce] output_root        : {output_root}")
    CONSOLE.print(f"[reinforce] iteration          : {iteration}")
    CONSOLE.print(f"[reinforce] Phase-1 base_plane  : {base_plane_root}")
    CONSOLE.print(f"[reinforce] Phase-1 base_see3d  : {base_see3d_root}")
    CONSOLE.print(f"[reinforce] Phase-2 fresh_see3d : {run_fresh_see3d}")
    CONSOLE.print("========================\n")

    t1 = time.time()

    # ===================================================================
    # Phase-1: 用 base 的 see3d_render + 原始视角，先做一次 plane_refine_depth + refine
    # ===================================================================
    CONSOLE.print("[Phase-1] plane_refine_depth with BASE See3D prior")
    run_command_safe(get_plane_refine_depth_cmd(
        plane_root=base_plane_root,
        see3d_root=base_see3d_root,
        anchor_json=None
    ))

    CONSOLE.print("[Phase-1] refine_free_gaussians (init = merged ply, ONLY HERE)")
    run_command_safe(get_refine_free_gaussians_cmd(
        refine_depth_path=base_plane_root,
        init_ply=init_gaussians_ply
    ))

    # ===================================================================
    # Phase-2: fresh re-run See3D（关键：这段要和参考代码完全一致）
    # ===================================================================
    if run_fresh_see3d:
        CONSOLE.print("\n[Phase-2] takeover hardcoded dirs (plane-refine-depths + see3d_render) -> fresh outputs")

        # 1) fresh dirs
        ensure_empty_dir_with_backup(fresh_plane_real)
        ensure_empty_dir_with_backup(fresh_see3d_real)

        # 2) takeover hardcoded paths by symlink (IMPORTANT: BOTH)
        ensure_symlink_with_backup(plane_link_in_scene, fresh_plane_real)
        ensure_symlink_with_backup(see3d_link_in_scene, fresh_see3d_real)

        # 3) fresh prepare planes (must happen under mast3r_scene/plane-refine-depths i.e. symlinked)
        CONSOLE.print("[Phase-2] fresh prepare: render_chart_views + plane_excavator + init plane_refine_depth(no see3d)")
        run_command_safe(render_charts_cmd(fresh_plane_real))
        run_command_safe(plane_excavator_cmd(fresh_plane_real))
        run_command_safe(get_plane_refine_depth_cmd(
            plane_root=fresh_plane_real,
            see3d_root=None,
            anchor_json=None
        ))

        # 4) See3D stages (EXACTLY LIKE OFFICIAL)
        max_stage = min(max(args.see3d_max_stage, 1), 3)
        for stage in range(1, max_stage + 1):
            CONSOLE.print(f"\n[Phase-2] Stage{stage}: inpaint -> plane_refine_depth(with fresh see3d) -> mv point_cloud -> refine(continue)")

            # (A) inpaint
            run_command_safe(get_see3d_inpaint_cmd(stage, plane_root_dir=fresh_plane_real))

            # (B) plane_refine_depth with fresh see3d prior
            if stage == 3:
                anchor_json = os.path.join(see3d_link_in_scene, "stage3", "anchor_view_id.json")
                run_command_safe(get_plane_refine_depth_cmd(
                    plane_root=fresh_plane_real,
                    see3d_root=see3d_link_in_scene,
                    anchor_json=anchor_json if os.path.exists(anchor_json) else None
                ))
            else:
                run_command_safe(get_plane_refine_depth_cmd(
                    plane_root=fresh_plane_real,
                    see3d_root=see3d_link_in_scene,
                    anchor_json=None
                ))

            # (C) mv point_cloud (same naming as official)
            backup_name = {1: "point_cloud-ori", 2: "point_cloud-s1", 3: "point_cloud-s2"}[stage]
            mv_cmd = f"mv {free_gaussians_path}/point_cloud {free_gaussians_path}/{backup_name}"
            run_command_safe(mv_cmd)

            # (D) refine continue (IMPORTANT: DO NOT pass --init_gaussians_ply)
            run_command_safe(get_refine_free_gaussians_cmd(
                refine_depth_path=fresh_plane_real,
                init_ply=None
            ))

    # ===================================================================
    # Final
    # ===================================================================
    CONSOLE.print("\n[Final] render all images")
    run_command_safe(render_all_img_cmd)

    if args.run_tetra:
        CONSOLE.print("\n[Final] extract tetra mesh")
        run_command_safe(tetra_cmd)

    if args.run_eval:
        CONSOLE.print("\n[Final] eval")
        run_command_safe(eval_cmd)

    t2 = time.time()
    CONSOLE.print(f"\n[Done] Total time: {t2 - t1:.2f} seconds\n")
