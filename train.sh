set -euo pipefail

PROOF_R1_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PROOF_R1_ROOT
: "${MODEL_PATH:?Set MODEL_PATH to the backbone model directory or Hugging Face model ID}"
: "${VERL_ROOT:?Set VERL_ROOT to the installed verl checkout}"

model="${1:-qwen25_7b}"
if [[ $# -gt 0 ]]; then
    shift
fi
case "$model" in
    qwen25_7b|qwen3_8b) ;;
    *) printf 'Unknown configuration: %s\n' "$model" >&2; exit 2 ;;
esac

cd -- "$VERL_ROOT"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
config="recipe/formally_verifiable/config/${model}.yaml"
if [[ ! -f "$config" ]]; then
    printf 'Install the recipe with Training/install_into_verl.py first.\n' >&2
    exit 1
fi

python -m recipe.formally_verifiable.data_preprocess --recipe-config "$config"
python -m recipe.formally_verifiable.main_ppo --recipe-config "$config" "$@"
