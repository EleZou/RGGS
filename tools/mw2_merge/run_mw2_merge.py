#!/usr/bin/env python
import argparse
from glob import glob
from typing import List
import torch

def _log(msg: str) -> None:
    print(f"[mw2-run] {msg}", flush=True)

# Import that works both as module and script
try:
    from .mw2_merge import MW2Merger, MW2Params
except Exception:
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from mw2_merge import MW2Merger, MW2Params


def _expand(pats: List[str]) -> List[str]:
    out: List[str] = []
    for p in pats:
        if any(ch in p for ch in "*?[]"):
            out += glob(p)
        else:
            out.append(p)
    # unique, stable
    return list(dict.fromkeys(sorted(out)))


def main():
    ap = argparse.ArgumentParser("MW2-OT Merge (RegGS) with COLMAP alignment")
    ap.add_argument("--base", type=str, default=None, help="Base gaussian (.ply/.pt/.ckpt). Optional.")
    ap.add_argument("--subs", type=str, nargs="+", required=True, help="Submap gaussian paths or globs")
    ap.add_argument("--out", type=str, required=True, help="Output .ply path")
    ap.add_argument("--merge-radius", type=float, default=0.02)
    ap.add_argument("--topk", type=int, default=32, help="KNN topk; set 0 to enable auto-topk")
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--sinkhorn-iters", type=int, default=30)
    ap.add_argument("--sim3-iters", type=int, default=40)
    ap.add_argument("--sim3-lr", type=float, default=3e-3)
    ap.add_argument("--insert-ratio-th", type=float, default=0.3)
    ap.add_argument("--opacity-th", type=float, default=0.01)
    ap.add_argument("--min-scale", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.3, help="Distillation weight; set 0 to enable auto-tau")
    ap.add_argument("--base-colmap", type=str, default=None, help="Path to base COLMAP dir or images.txt")
    ap.add_argument("--sub-colmap", type=str, nargs="*", default=None, help="List or glob(s) for sub COLMAP images.txt")
    ap.add_argument("--progress", action="store_true", help="Show tqdm progress bars")

    # 新增：插入模式 & 是否跳过质量门控
    ap.add_argument("--insert-mode", type=str, default="all", choices=["detail", "all"],
                    help="detail: hole-filling insert; all: append all sub points after alignment")
    ap.add_argument("--no-quality-gate", action="store_true",
                    help="If set, do not filter inserted points by opacity/min_scale")

    args = ap.parse_args()

    params = MW2Params(
        merge_radius=args.merge_radius, topk=args.topk, eps=args.eps,
        sinkhorn_iters=args.sinkhorn_iters, sim3_iters=args.sim3_iters, sim3_lr=args.sim3_lr,
        insert_ratio_th=args.insert_ratio_th, opacity_th=args.opacity_th, min_scale=args.min_scale, tau=args.tau
    )

    subs = _expand(args.subs)
    _log(f"expanded subs={len(subs)}")
    if args.base:
        _log(f"base={args.base}")
    if args.base_colmap:
        _log(f"base_colmap={args.base_colmap}")
    _log(f"insert_mode={args.insert_mode} no_quality_gate={args.no_quality_gate}")

    sub_colmap = []
    if args.sub_colmap is not None and len(args.sub_colmap) > 0:
        expanded = []
        for p in args.sub_colmap:
            expanded += _expand([p])
        if len(expanded) == 1 and len(subs) > 1:
            sub_colmap = expanded * len(subs)
        else:
            sub_colmap = (expanded + [None] * len(subs))[:len(subs)]
        _log(f"expanded sub_colmap={len([x for x in sub_colmap if x])}")
    else:
        sub_colmap = [None] * len(subs)

    merger = MW2Merger(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    merger.merge(
        args.base, subs, args.out, params,
        base_colmap=args.base_colmap, sub_colmap_paths=sub_colmap,
        progress=args.progress,
        insert_mode=args.insert_mode,
        no_quality_gate=args.no_quality_gate
    )


if __name__ == "__main__":
    main()
