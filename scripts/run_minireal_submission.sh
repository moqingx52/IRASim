#!/usr/bin/env bash
# Chain: rdt_action_txt_to_npy -> rollout_minireal -> rerank_and_export
#
# Run from anywhere; defaults assume /workspace layout. Override via env:
#
#   IRASIM_ROOT          IRASim repo (default: parent of this script)
#   SAMPLE_RESULT_RDT_SRC   Input: <ep>/action.txt (default: /workspace/sample_result_rdt)
#   SAMPLE_RESULT_RDT_NPY_OUT Output flat *.npy (default: /workspace/sample_result_rdt_npy)
#   TEST_DATA            Test set root (default: /workspace/test)
#   CHECKPOINT           Fine-tuned .pt path (required)
#   IRASIM_ROLLOUTS      Rollout cache dir (default: /workspace/irasim_rollouts)
#   DATA_RESULT          Final submission tree (default: /workspace/data_result)
#   FPS                  Video fps for export (default: 30)
#   ADD_BASELINE         Set to 1 to pass --add-baseline-repeat to rerank
#   ROLLOUT_CUDA_DEVICE  Index passed to rollout_minireal.py --cuda-device (default: 0)
#   ROLLOUT_VIDEO_SIZE   H,W for rollout_minireal.py --video-size (default: 288,512)
#
# Multi-GPU (e.g. 8×4090): this script does NOT spawn multiple GPUs. Rollout is
# single-process, one model on one device. To use 8 cards in parallel, launch
# 8 separate rollout jobs, each with a disjoint --episodes list and a dedicated
# GPU.
#
# Episode counts that are not divisible by 8 (e.g. 100): do NOT use naive
# "100/8=12" floor-only splits — that drops 4 episodes. Use contiguous shards
# where the first (N %% W) workers get one extra episode each (100→13,13,13,13,12,12,12,12).
# scripts/shard_episodes_for_rollout.py prints exact --episodes strings and can
# verify disjoint full coverage:
#
#   python scripts/shard_episodes_for_rollout.py --test-data "$TEST_DATA" --workers 8 --summary
#   for i in $(seq 0 7); do
#     EPS=$(python scripts/shard_episodes_for_rollout.py --test-data "$TEST_DATA" --workers 8 --print-shard "$i")
#     CUDA_VISIBLE_DEVICES=$i python scripts/rollout_minireal.py \
#       --config configs/evaluation/minireal/frame_ada.yaml \
#       --checkpoint "$CHECKPOINT" --test-data "$TEST_DATA" \
#       --rdt-actions "$SAMPLE_RESULT_RDT_NPY_OUT" --out "$IRASIM_ROLLOUTS" \
#       --video-size "$ROLLOUT_VIDEO_SIZE" --cuda-device 0 --episodes "$EPS" &
#   done
#   wait
#   # then run rerank_and_export once (Step 3 above) on the merged IRASIM_ROLLOUTS
#
# With CUDA_VISIBLE_DEVICES set to a single card, each process should keep
# --cuda-device 0 (logical cuda:0 is that physical card).
#
# Example:
#   export CHECKPOINT=/workspace/IRASim/results/.../0350000.pt
#   bash scripts/run_minireal_submission.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IRASIM_ROOT="${IRASIM_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

SAMPLE_RESULT_RDT_SRC="${SAMPLE_RESULT_RDT_SRC:-/workspace/sample_result_rdt}"
SAMPLE_RESULT_RDT_NPY_OUT="${SAMPLE_RESULT_RDT_NPY_OUT:-/workspace/sample_result_rdt_npy}"
TEST_DATA="${TEST_DATA:-/workspace/test}"
IRASIM_ROLLOUTS="${IRASIM_ROLLOUTS:-/workspace/irasim_rollouts}"
DATA_RESULT="${DATA_RESULT:-/workspace/data_result}"
FPS="${FPS:-30}"
ADD_BASELINE="${ADD_BASELINE:-0}"
ROLLOUT_CUDA_DEVICE="${ROLLOUT_CUDA_DEVICE:-0}"
ROLLOUT_VIDEO_SIZE="${ROLLOUT_VIDEO_SIZE:-288,512}"

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "ERROR: set CHECKPOINT to your finetuned .pt (e.g. export CHECKPOINT=.../0350000.pt)" >&2
  exit 1
fi

cd "$IRASIM_ROOT"

echo "== Step 1: RDT action.txt -> npy -> ${SAMPLE_RESULT_RDT_NPY_OUT}"
python scripts/rdt_action_txt_to_npy.py \
  --src "$SAMPLE_RESULT_RDT_SRC" \
  --out "$SAMPLE_RESULT_RDT_NPY_OUT" \
  --only-test "$TEST_DATA"

echo "== Step 2: IRASim rollout -> ${IRASIM_ROLLOUTS}"
python scripts/rollout_minireal.py \
  --config configs/evaluation/minireal/frame_ada.yaml \
  --checkpoint "$CHECKPOINT" \
  --test-data "$TEST_DATA" \
  --rdt-actions "$SAMPLE_RESULT_RDT_NPY_OUT" \
  --out "$IRASIM_ROLLOUTS" \
  --video-size "$ROLLOUT_VIDEO_SIZE" \
  --cuda-device "$ROLLOUT_CUDA_DEVICE"

BASELINE_FLAG=()
if [[ "$ADD_BASELINE" == "1" ]]; then
  BASELINE_FLAG=(--add-baseline-repeat)
fi

echo "== Step 3: rerank + export -> ${DATA_RESULT}"
python scripts/rerank_and_export.py \
  --rollouts "$IRASIM_ROLLOUTS" \
  --test-data "$TEST_DATA" \
  --out "$DATA_RESULT" \
  --fps "$FPS" \
  "${BASELINE_FLAG[@]}"

echo "Done. Submission root: ${DATA_RESULT}"
