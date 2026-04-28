import os
import sys
import argparse

def run_command_safe(cmd):
    print(f"[refine_free_gaussians] Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print("[refine_free_gaussians] FAILED!")
        sys.exit(1)
    print("[refine_free_gaussians] OK.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Wrapper for free Gaussian refinement")

    parser.add_argument("-s", "--mast3r_scene", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, required=True)

    parser.add_argument("--init_gaussians_ply", type=str, required=True,
                        help="MW2 merged PLY used as initial Gaussian model")

    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--dense_regul", type=str, default="default")
    parser.add_argument("--use_downsample_gaussians", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    cmd = " ".join([
        "python", "2d-gaussian-splatting/train_with_refine_depth.py",
        "-s", args.mast3r_scene,
        "-m", args.output_path,
        "--iterations", str(args.iterations),
        "--dense_regul", args.dense_regul,
        "--init_gaussians_ply", args.init_gaussians_ply,
        "--refine_depth_path", args.mast3r_scene,    # dummy for API compatibility
        "--use_downsample_gaussians" if args.use_downsample_gaussians else "",
    ])

    run_command_safe(cmd)
