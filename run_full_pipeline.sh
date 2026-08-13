#!/usr/bin/env bash
# Full CIFAR10 Deep-Arguing pipeline, start to finish, seed 0 throughout:
#   1. Pretrain the ResNet-32 backbone
#   2. Fit GradualAACBR on top of it + export the misclassified QBAF
#   3. Contest irrelevance edges (new-case-to-casebase corrections)
#   4. Build the irrelevance fine-tune dataset
#   5. Contest casebase-internal edges (batch_contest -- edits shared model.A)
#   6. Build the casebase fine-tune dataset
#   7. Fine-tune feature_weights_1 against both correction sources
#   8. Evaluate the result (baseline vs. finetuned accuracy)
#
# Uses tuning/cifar10/resnet/relu/ (ReluSemantics) -- the config everything
# this session has been built and tuned against. Known risk (see updates.md):
# stage 5's batch_contest edits a SHARED model.A under ReluSemantics, which
# has previously caused catastrophic global-accuracy collapse from a single
# bad edit (both for brainwear and, historically, for CIFAR10 itself) --
# there was no conservative override applied here, so stage 8's baseline-vs-
# finetuned accuracy comparison is your real safety check. A large negative
# accuracy delta there means this is what happened; rerun with tighter
# contest_all.py hyperparameters (e.g. --alpha-init 0.001 --divergence-bound
# 2.0) or fall back to --casebase-lambda 0 in stage 7 if so.
#
# Expect this to take hours (stage 1 alone is a 200-epoch CIFAR10 training
# run) -- run this under tmux/screen/nohup, not a session that might drop.
#
# Usage: ./run_full_pipeline.sh 2>&1 | tee pipeline_run.log

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SEED=0
CONFIG_DIR="tuning/cifar10/resnet/relu"

# Avoids an interactive wandb login prompt blocking an unattended run.
# Unset this (or export WANDB_MODE=online) first if you want real tracking.
export WANDB_MODE="${WANDB_MODE:-offline}"

# Without this, Python block-buffers stdout once it's not attached to a real
# terminal (i.e. as soon as it's piped into `tee`) -- plain print() status
# lines would then sit in a buffer and only show up in delayed bursts
# instead of as they happen. tqdm progress bars mostly self-flush already,
# but this makes everything (including the stage-by-stage prints in the
# Python scripts themselves) show up live, not just the bar.
export PYTHONUNBUFFERED=1

echo "################################################################"
echo "# Stage 1/8: Pretrain ResNet-32 backbone (seed=$SEED)"
echo "################################################################"
python src/scripts/pretrain_resnet.py --seed "$SEED"

echo "################################################################"
echo "# Stage 2/8: Fit GradualAACBR + export misclassified QBAF (seed=$SEED)"
echo "################################################################"
python -m deeparguing.cli.run \
  --config "$CONFIG_DIR/model_cifar10_image.yaml" \
           "$CONFIG_DIR/hyperparameters_cifar10_image.yaml" \
           "$CONFIG_DIR/data_cifar10_image.yaml" \
  --seed "$SEED" --run_test --misclassified_log

echo "################################################################"
echo "# Stage 3/8: Contest irrelevance edges (new-case corrections)"
echo "################################################################"
python -m deeparguing.contest.scripts.contest_all_irrelevance \
  --checkpoint model_checkpoint.pt --qbaf misclassified_qbaf.json

echo "################################################################"
echo "# Stage 4/8: Build irrelevance fine-tune dataset"
echo "################################################################"
python -m deeparguing.contest.scripts.build_irrelevance_finetune_dataset \
  --checkpoint model_checkpoint.pt --qbaf misclassified_qbaf.json --seed "$SEED"

echo "################################################################"
echo "# Stage 5/8: Contest casebase-internal edges (batch_contest)"
echo "# NOTE: this edits a SHARED model.A -- see the risk note at the top"
echo "# of this script. Watch the 'Cleared X/Y samples, N edges changed'"
echo "# line below."
echo "################################################################"
CONTEST_ALL_OUTPUT="$(mktemp)"
python -m deeparguing.contest.scripts.contest_all \
  --checkpoint model_checkpoint.pt --qbaf misclassified_qbaf.json \
  --save-checkpoint "" 2>&1 | tee "$CONTEST_ALL_OUTPUT"
CONTEST_ALL_LOG="$(grep "^Saved run log to " "$CONTEST_ALL_OUTPUT" | sed 's/^Saved run log to //')"
rm -f "$CONTEST_ALL_OUTPUT"
if [ -z "$CONTEST_ALL_LOG" ]; then
  echo "ERROR: couldn't find contest_all.py's saved log path in its output -- aborting." >&2
  exit 1
fi
echo "contest_all.py log: $CONTEST_ALL_LOG"

echo "################################################################"
echo "# Stage 6/8: Build casebase fine-tune dataset"
echo "################################################################"
python -m deeparguing.contest.scripts.build_casebase_finetune_dataset \
  --contest-log "$CONTEST_ALL_LOG" --checkpoint model_checkpoint.pt --seed "$SEED"

echo "################################################################"
echo "# Stage 7/8: Fine-tune feature_weights_1 (irrelevance + casebase correction)"
echo "################################################################"
python -m deeparguing.contest.scripts.run_finetune \
  --checkpoint model_checkpoint.pt --dataset irrelevance_finetune_dataset.pt \
  --casebase-dataset casebase_finetune_dataset.pt \
  --seed "$SEED"

echo "################################################################"
echo "# Stage 8/8: Evaluate"
echo "# THIS IS THE REAL SAFETY CHECK for stage 5's risk -- if 'finetuned'"
echo "# accuracy has collapsed relative to baseline, that's it happening."
echo "################################################################"
python -m deeparguing.contest.scripts.evaluate_irrelevance_finetune \
  --baseline-checkpoint model_checkpoint.pt --finetuned-checkpoint finetuned_checkpoint.pt

echo "################################################################"
echo "# Done. All outputs are in today's outputs/<date>/ folder."
echo "################################################################"
