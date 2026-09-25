# Proof-R1

Training code for Proof-R1. The model generates a structured proof, receives
step rewards from Z3 and a rule checker, and assigns process credit to the
actions in the dependency closure supporting its answer.

The release includes LoRA format warmup, reinforcement learning, and the
ProverQA data used by these stages. Evaluation programs, validation and test
sets, model weights, and experiment outputs are not included.

## Contents

```text
data/proverqa/train.jsonl       2,935 problems for reinforcement learning
data/proverqa/warmup.jsonl         56 supervised proof trajectories
Training/Warmup/                LoRA format warmup
Training/recipe/                Proof-R1 prompts, verification, rewards, and trainer
Training/verl_overlay/          Required verl integration files
Training/install_into_verl.py   Installer for the pinned verl revision
train.sh                       Data preparation and RL launch
```

## Environment

Use Linux, Python 3.12, and CUDA-capable NVIDIA GPUs. The supplied RL configs
use four GPUs; the recorded training environment had about 48 GiB per GPU.
The runtime versions are taken from that environment: PyTorch 2.8.0 with
CUDA 12.8, vLLM 0.11.0, Transformers 4.57.1, PEFT 0.17.1, and
FlashAttention 2.8.1.

Run the following from this repository in a fresh environment. Keep the verl
checkout outside this repository; the installer updates its integration files.

```bash
export PROOF_R1_ROOT="$PWD"
export VERL_ROOT="$(dirname "$PWD")/verl"

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/verl-project/verl.git "$VERL_ROOT"
git -C "$VERL_ROOT" checkout 91666d9964282b890c75dd0b2d330edaee201c2f
python -m pip install -r requirements.txt -e "$VERL_ROOT"
python -m pip install flash-attn==2.8.1 --no-build-isolation
python Training/install_into_verl.py --verl-root "$VERL_ROOT"
```

Building FlashAttention requires a CUDA development toolkit compatible with
the installed PyTorch. A matching prebuilt wheel can also be used.

## Train

Set `MODEL_PATH` to a local model directory or the corresponding Hugging Face
model ID. Start with format warmup, then launch RL using the resulting adapter.

### Qwen2.5-7B-Instruct

```bash
export MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
python Training/Warmup/train_lora.py \
  --config Training/Warmup/configs/qwen25_7b_explicit.yaml
bash train.sh qwen25_7b
```

### Qwen3-8B

```bash
export MODEL_PATH=Qwen/Qwen3-8B
python Training/Warmup/train_lora.py \
  --config Training/Warmup/configs/qwen3_8b_native.yaml
bash train.sh qwen3_8b
```

Qwen2.5 uses explicit `<think>` tags. Qwen3 uses its native thinking template;
warmup, preprocessing, and rollout use the same mode. The Qwen2.5 warmup
length limits are increased from the earlier experiment script to retain all
56 complete demonstrations. Overlong warmup examples raise an error.

The RL defaults use LoRA rank 8 and alpha 16, 8 prompts per batch, 8 sampled
responses per prompt, a learning rate of `5e-6`, and 1,098 updates over three
epochs. Prompt and response limits are each 4,096 tokens. Numerical settings
are in `Training/recipe/formally_verifiable/config/`.

Append Hydra overrides to change an RL setting, for example:

```bash
bash train.sh qwen25_7b trainer.total_training_steps=1
```

After editing a recipe or its configuration, rerun the installer before
launching. RL runs from the installed copy in the verl checkout. Training
does not load validation or test data.

## Data

The problems are derived from [ProverQA](https://huggingface.co/datasets/opendatalab/ProverQA).
The RL file contains 2,935 formally verified A/B examples: 1,488 true and
1,447 false. Each row includes `id`, `question`, `nl2fol`, `options`, `answer`,
and `conclusion_fol`. The warmup file contains 56 problem/response pairs
with 272 checked proof steps. Its `problem` field uses the same raw schema;
`response` contains the supervised reasoning and structured proof.

`train.sh` converts the raw training file into verl records under
`outputs/data/<model>/`. Keep `data/proverqa/train.jsonl` as the raw input;
do not preprocess the generated file a second time. Each generated record
stores the raw problem as a JSON string so Arrow preserves premise order.

## Checkpoints

Warmup adapters are saved in `outputs/warmup/<model>/`. RL checkpoints are
saved in `outputs/<model>/global_step_<step>/`, and exported PEFT adapters in
`outputs/<model>/adapter/global_step_<step>/`. Checkpoints include the rule
EMA state so resumed training continues with the same reward baseline.
By default, verl resumes the latest checkpoint in the output directory.
