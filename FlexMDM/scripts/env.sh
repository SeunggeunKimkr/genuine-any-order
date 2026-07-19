# Shared conda + CUDA + HF-cache setup. `source` this from the job scripts.
# Not directly executable; intentionally has no shebang.
#
# Reads (all optional, with defaults):
#   CONDA_ENV             - conda env name (default: flexmdm)
#   CONDA_PREFIX_OVERRIDE - absolute env path, used instead of the name if set
#   CONDA_BASE            - conda install root (auto-detected if unset)
#   REPO_ROOT             - repo checkout dir (auto-detected from this file)
#   HF_HOME               - HuggingFace cache root (default: ~/.cache/huggingface)
#   CUDA_MODULE           - `module load` name, if your cluster uses Lmod
#
# Cluster-specific tweaks (broken libffi symlinks, site Miniforge paths, etc.)
# are intentionally NOT hardcoded here — see docs/CLUSTER.md.

CONDA_ENV="${CONDA_ENV:-flexmdm}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# Load a paths.env if present (see paths.env.template).
if [ -f "$REPO_ROOT/paths.env" ]; then
  # shellcheck disable=SC1091
  set -a; source "$REPO_ROOT/paths.env"; set +a
fi

HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Do NOT export legacy TRANSFORMERS_CACHE/HF_DATASETS_CACHE here: transformers
# treats TRANSFORMERS_CACHE as an override that redirects its model cache away
# from $HF_HOME/hub, which breaks offline runs against a prewarmed cache.
# Everything derives from HF_HOME.

# Resolve the conda base if not provided.
if [ -z "${CONDA_BASE:-}" ]; then
  if [ -n "${CONDA_EXE:-}" ]; then
    CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
  elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
  else
    echo "ERROR: could not determine CONDA_BASE; set it or put conda on PATH" >&2
    return 2 2>/dev/null || exit 2
  fi
fi

# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
if [ -n "${CONDA_PREFIX_OVERRIDE:-}" ]; then
  conda activate "$CONDA_PREFIX_OVERRIDE"
elif conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda activate "$CONDA_ENV"
else
  echo "ERROR: conda env '$CONDA_ENV' not found (set CONDA_ENV or CONDA_PREFIX_OVERRIDE)" >&2
  return 2 2>/dev/null || exit 2
fi

# Optional Lmod CUDA module (clusters that provide one).
if [ -n "${CUDA_MODULE:-}" ]; then
  if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
  command -v module >/dev/null 2>&1 && module load "$CUDA_MODULE" || true
fi

mkdir -p "$HF_HOME"
export HF_HOME
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
