#!/usr/bin/env bash
set -euo pipefail

# MW2-OT merge for 2dgs (.ply or .ckpt), with optional COLMAP alignment
# 推荐额外环境变量（可选）：
#   export MW2_DEDUP=1
#   export MW2_SUB_TO_MAIN_TH=0.01
#   export MW2_DEDUP_SUB_VOXEL=0.01
#   export MW2_DEFOG=1          # 开启自动去雾

BASE_CKPT=""
SUBS_GLOB=""
OUT_PLY=""
BASE_COLMAP=""
SUB_COLMAP_GLOB=""

# ---------- 推荐默认参数 ----------
# 假设场景单位大致是“米”，如果你的场景单位更小/更大，可以整体乘一个系数
MERGE_RADIUS="0.001"     # 主超参数：允许匹配的最大距离

# 设为 0 -> 使用自动选择逻辑（根据点数和场景大小）
TOPK="0"                # 每个 sub 点最多找 K 个 base 邻居；0 表示自动
TAU="0"                 # 从 sub 向 base 蒸馏的权重；0 表示自动

EPS="0.02"              # Sinkhorn 平滑项
SINKHORN_ITERS="30"
SIM3_ITERS="40"
SIM3_LR="0.003"

INSERT_RATIO_TH="0.3"   # 当前代码中基本已弃用，留作兼容

OPACITY_TH="0.0"        # 先不过早砍，交给 defog 处理；需要的话可改成 0.05
MIN_SCALE="0.0"         # 同上，如需更激进过滤可设为 3e-4
# -------------------------------

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-ckpt)        BASE_CKPT="$2"; shift 2 ;;
    --subs-glob)        SUBS_GLOB="$2"; shift 2 ;;
    --out)              OUT_PLY="$2"; shift 2 ;;
    --base-colmap)      BASE_COLMAP="$2"; shift 2 ;;
    --sub-colmap-glob)  SUB_COLMAP_GLOB="$2"; shift 2 ;;

    --merge-radius)     MERGE_RADIUS="$2"; shift 2 ;;
    --topk)             TOPK="$2"; shift 2 ;;
    --eps)              EPS="$2"; shift 2 ;;
    --sinkhorn-iters)   SINKHORN_ITERS="$2"; shift 2 ;;
    --sim3-iters)       SIM3_ITERS="$2"; shift 2 ;;
    --sim3-lr)          SIM3_LR="$2"; shift 2 ;;
    --tau)              TAU="$2"; shift 2 ;;
    --insert-ratio-th)  INSERT_RATIO_TH="$2"; shift 2 ;;
    --opacity-th)       OPACITY_TH="$2"; shift 2 ;;
    --min-scale)        MIN_SCALE="$2"; shift 2 ;;

    -h|--help)
      cat <<USAGE
Usage: $0 --subs-glob '.../*.ply' --out out.ply
          [--base-ckpt base.ply]
          [--base-colmap images.txt]
          [--sub-colmap-glob '.../images.txt']
          [--merge-radius 0.02]
          [--topk 0]          # 0 = auto
          [--tau 0]           # 0 = auto
          [--opacity-th 0.0]
          [--min-scale 0.0]
USAGE
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${SUBS_GLOB}" || -z "${OUT_PLY}" ]]; then
  echo "Usage: $0 --subs-glob '.../*.ply' --out out.ply [--base-ckpt base.ply] [--base-colmap images.txt] [--sub-colmap-glob '.../images.txt']" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_PLY}")"
PY=${PYTHON:-python}

CMD=(
  "${PY}" "./tools/mw2_merge/run_mw2_merge.py"
  --subs "${SUBS_GLOB}"
  --out "${OUT_PLY}"
  --merge-radius "${MERGE_RADIUS}"
  --topk "${TOPK}"
  --eps "${EPS}"
  --sinkhorn-iters "${SINKHORN_ITERS}"
  --sim3-iters "${SIM3_ITERS}"
  --sim3-lr "${SIM3_LR}"
  --insert-ratio-th "${INSERT_RATIO_TH}"
  --opacity-th "${OPACITY_TH}"
  --min-scale "${MIN_SCALE}"
  --tau "${TAU}"
  --progress
)

[[ -n "${BASE_CKPT}" ]]        && CMD+=( --base "${BASE_CKPT}" )
[[ -n "${BASE_COLMAP}" ]]      && CMD+=( --base-colmap "${BASE_COLMAP}" )
[[ -n "${SUB_COLMAP_GLOB}" ]]  && CMD+=( --sub-colmap "${SUB_COLMAP_GLOB}" )

echo "[mw2-merge] Running:"
printf ' %q' "${CMD[@]}"; echo
"${CMD[@]}"
echo "[mw2-merge] Done -> ${OUT_PLY}"
