#!/bin/bash
# =============================================================================
# FlexMDM fine-tuning launcher — 4 nodes x 4 GPUs (16 GPUs), FSDP HYBRID_SHARD.
# Reproduces the submission training run (config/fsdp_train.yaml).
#
# SLURM account/partition below are PLACEHOLDERS — set them for your cluster
# (or override on the command line). See docs/CLUSTER.md.
# =============================================================================
#SBATCH --job-name=flexmdm_train
#SBATCH --account=CHANGE_ME_ACCOUNT
#SBATCH --partition=CHANGE_ME_PARTITION
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512GB
#SBATCH --time=3-00:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

# Under sbatch, SLURM copies this script to a spool dir, so BASH_SOURCE does
# not point into the repo; prefer REPO_ROOT / the sbatch submission dir.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# FSDP wants OMP_NUM_THREADS=1; set before sourcing env.sh.
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/env.sh"

CONFIG_NAME="${CONFIG_NAME:-fsdp_train}"          # the released training recipe
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/checkpoints}"
WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/wandb}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
# Appended to the master hostname (e.g. ".rc.fas.harvard.edu" on FASRC); empty
# by default. Set MASTER_ADDR_SUFFIX if your cluster needs an FQDN.
MASTER_ADDR_SUFFIX="${MASTER_ADDR_SUFFIX:-}"

set -u
mkdir -p "$SAVE_ROOT" "$WANDB_DIR"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_FAMILY=AF_INET
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

DEFAULT_IF=$(ip -o -4 route show to default | awk '{print $5}' | head -n 1)
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$DEFAULT_IF}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$DEFAULT_IF}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${SLURM_NNODES:-4}"
MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)${MASTER_ADDR_SUFFIX}"
export MASTER_ADDR MASTER_PORT

cd "$REPO_ROOT"
echo "=== FlexMDM FSDP ${NNODES}-node x ${NPROC_PER_NODE}-GPU run ==="
echo "config=$CONFIG_NAME  save_root=$SAVE_ROOT  master=$MASTER_ADDR:$MASTER_PORT"

# EXTRA_OVERRIDES: space-separated Hydra overrides, e.g.
#   EXTRA_OVERRIDES="trainer.total_training_steps=10 trainer.save_checkpoint_steps=2"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

# AUTO_RESUME=1: resume from the highest COMPLETE checkpoint under $SAVE_ROOT
# (has both training_state.pt and flexmdm_extras.pt). Prepended so a user
# resume_from in EXTRA_OVERRIDES still wins (hydra: last override wins).
AUTO_RESUME="${AUTO_RESUME:-0}"
if [ "$AUTO_RESUME" = "1" ]; then
  for d in $(ls -d "$SAVE_ROOT"/global_step_* 2>/dev/null | \
             awk -F'global_step_' '{print $NF, $0}' | sort -k1 -nr | awk '{print $2}'); do
    if [ -s "$d/training_state.pt" ] && [ -s "$d/flexmdm_extras.pt" ]; then
      echo "auto_resume_from=$d"
      EXTRA_OVERRIDES="trainer.resume_from=$d $EXTRA_OVERRIDES"
      break
    fi
  done
fi
[ -n "$EXTRA_OVERRIDES" ] && echo "extra_overrides=$EXTRA_OVERRIDES"

srun --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c '
python -m torch.distributed.run \
    --nnodes '"$NNODES"' \
    --nproc_per_node '"$NPROC_PER_NODE"' \
    --node_rank "$SLURM_NODEID" \
    --master_addr '"$MASTER_ADDR"' \
    --master_port '"$MASTER_PORT"' \
    -m flexmdm.train_fsdp \
    --config-name '"$CONFIG_NAME"' \
    "trainer.default_local_dir='"$SAVE_ROOT"'" \
    "trainer.wandb.dir='"$WANDB_DIR"'" \
    "model.attn_implementation='"$ATTN_IMPL"'" \
    "fsdp.sharding_strategy=hybrid_shard" \
    '"$EXTRA_OVERRIDES"'
'
