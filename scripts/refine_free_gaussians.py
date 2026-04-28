import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml

from rich.console import Console


def run_command_safe(command):
    print(f"Running command: {command}")
    exit_code = os.system(command)
    if exit_code != 0:
        print("Command failed!")
        sys.exit(1)
    else:
        print("Command succeeded!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Refine free Gaussians with plane-refine-depths (MAtCha/G4Splat)"
    )
    
    # Scene / IO
    parser.add_argument('-s', '--mast3r_scene', type=str, required=True,
                        help='Path to the mast3r_sfm directory')
    parser.add_argument('-o', '--output_path', type=str, default=None,
                        help='Path to the output directory for refined Gaussians')
    parser.add_argument('--white_background', type=bool, default=False)

    # Dense supervision（可选，不用就留 None）
    parser.add_argument("--dense_data_path", type=str, default=None)
    parser.add_argument('--depthanythingv2_checkpoint_dir', type=str,
                        default='./Depth-Anything-V2/checkpoints/')
    parser.add_argument('--depthanything_encoder', type=str, default='vitl')
    parser.add_argument('--dense_regul', type=str, default='default',
                        help='Strength of dense regularization. Can be "default", "strong", "weak", or "none".')

    # Free Gaussians config
    parser.add_argument('-c', '--config', type=str, default='default',
                        help='free_gaussians_refinement config name')

    # Plane-aware depth
    parser.add_argument('--refine_depth_path', type=str, default=None,
                        help='Path to the refine depth directory (plane-refine-depths)')

    parser.add_argument('--use_downsample_gaussians', action='store_true',
                        help='Use voxel downsampling for Gaussians')

    # ⭐ 新增：reinforce 时传入的 merged 高斯 PLY
    parser.add_argument('--init_gaussians_ply', type=str, default=None,
                        help='Use this PLY to initialize Gaussians (e.g., MW2 merged result)')

    args = parser.parse_args()
    
    # Console
    CONSOLE = Console(width=120)
    
    # Output path
    if args.output_path is None:
        # 默认用 mast3r_scene 目录名
        base_name = os.path.basename(os.path.normpath(args.mast3r_scene))
        args.output_path = os.path.join('output', base_name, 'refined_free_gaussians')
    os.makedirs(args.output_path, exist_ok=True)
    CONSOLE.print(f"[INFO] Refined free gaussians will be saved to: {args.output_path}")
    
    # Load config
    config_path = os.path.join('configs/free_gaussians_refinement', args.config + '.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Dense supervision
    if args.dense_data_path is not None:
        dense_arg = " ".join([
            "--dense_data_path", args.dense_data_path,
        ])
    else:
        dense_arg = ""

    if args.refine_depth_path is None:
        raise ValueError('refine depth path is required (e.g., mast3r_sfm/plane-refine-depths)')
    else:
        CONSOLE.print(f"[INFO] refine depth path: {args.refine_depth_path}, train gs use refine depth")

    # 组装命令
    cmd_parts = [
        "python", "2d-gaussian-splatting/train_with_refine_depth.py",
        "-s", args.mast3r_scene,
        "-m", args.output_path,
        "--iterations", str(config['iterations']),
        "--densify_until_iter", str(config['densify_until_iter']),
        "--opacity_reset_interval", str(config['opacity_reset_interval']),
        "--depth_ratio", str(config['depth_ratio']),
        "--use_mip_filter" if config.get('use_mip_filter', False) else "",
        dense_arg,
        "--normal_consistency_from", str(config['normal_consistency_from']),
        "--distortion_from", str(config['distortion_from']),
        "--depthanythingv2_checkpoint_dir", args.depthanythingv2_checkpoint_dir,
        "--depthanything_encoder", args.depthanything_encoder,
        "--dense_regul", args.dense_regul,
        "--refine_depth_path", args.refine_depth_path,
        "--use_downsample_gaussians" if args.use_downsample_gaussians else "",
    ]

    # ⭐ reinforce 模式才会用到 merged.ply，这里透传
    if args.init_gaussians_ply is not None:
        cmd_parts.extend([
            "--init_gaussians_ply", args.init_gaussians_ply
        ])

    # 清理空字符串
    cmd_parts = [p for p in cmd_parts if p != ""]

    command = " ".join(cmd_parts)
    
    CONSOLE.print(f"[INFO] Running command:\n{command}")
    run_command_safe(command)
