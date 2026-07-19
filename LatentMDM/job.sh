#!/usr/bin/env bash
#SBATCH --job-name tinygsm_rep

### Logging
#SBATCH --output job_logs/train_%j.out
#SBATCH --error  job_logs/train_%j.err
#SBATCH --mail-type=all
#SBATCH --mail-user=sgkim@utexas.edu

### Node info
#SBATCH --account ASC25024
#SBATCH --nodes 1
#SBATCH --partition gh
#SBATCH --ntasks-per-node=1
#SBATCH --time 48:00:00

source ~/.bashrc
micromamba activate genuine-any-order

MASTER_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_ADDR=$(srun -N1 -n1 -w "$MASTER_HOST" hostname -I | awk '{print $1}')
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))

export HF_HOME="/scratch/10816/sk58348/vista/hf_cache"
export XDG_CACHE_HOME="/scratch/10816/sk58348/vista/xdg_cache"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# CHECKPOINT_PATH="ckpts/<run>/step=<N>.pt"   # set to resume; see --resume below
# Let NCCL/GLOO auto-detect the correct network interface
# export NCCL_SOCKET_IFNAME=ib0
# export GLOO_SOCKET_IFNAME=ib0

# Hardcoded OmegaConf dotlist overrides — uncomment / edit as needed.
# CLI args passed after CFG_PATH override these (later entries win).
OVERRIDES=(
  wandb.wandb=true
  wandb.name="lpmdm_eos_mean_e2_p10_d2_no_fnorm_tinygsm_split_v2_bs256_1_1_lr3e-4-vista"
  training.eval_steps=5000
  model.planner.apply_final_norm=false
  model.encoder.apply_final_norm=false
  model.planner.num_layers=10
  model.encoder.num_layers=2
  model.decoder.num_layers=2
  # model.decoder.apply_final_norm=true
  # model.encoder.segment_pooling="cls"
  # model.encoder.max_position=33
)

srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
  torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=1 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    train_lpmdm.py --cfg yaml_files/tinygsm_lpmdm.yaml "${OVERRIDES[@]}" "$@" # --resume "$CHECKPOINT_PATH"
    # --resume "$CHECKPOINT_PATH"
    # train_lpmdm.py --cfg yaml_files/tinygsm_lpmdm.yaml --resume "$CHECKPOINT_PATH"
