from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


PROOF_R1_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROOF_R1_ROOT / "Training"
for path in (PROOF_R1_ROOT, TRAINING_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Training.Warmup import expand_config_paths
from recipe.formally_verifiable.rule_grounded_process_rl.structured_prompt import (
    build_chat_prompt,
)


THINKING_MODES = {"explicit", "native"}
_LEADING_THINK = re.compile(r"^\s*<think>\s*", re.IGNORECASE)


def prepare_response_target(response: str, thinking_mode: str) -> str:
    if thinking_mode not in THINKING_MODES:
        raise ValueError(f"Unsupported thinking_mode: {thinking_mode!r}")
    if thinking_mode == "explicit":
        if _LEADING_THINK.match(response) is None:
            raise ValueError("Explicit warmup responses must start with <think>.")
        return response

    match = _LEADING_THINK.match(response)
    target = response[match.end() :] if match else response
    if "</think>" not in target or "<summary>" not in target:
        raise ValueError(
            "Native warmup target must contain </think> and <summary>; the chat "
            "template supplies the opening <think>."
        )
    return target


class WarmupDataset(Dataset):
    def __init__(self, path: str, max_records: int | None = None):
        if max_records is not None and max_records < 1:
            raise ValueError("max_records must be positive.")
        self.records: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                self.records.append(json.loads(line))
                if max_records is not None and len(self.records) >= max_records:
                    break

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.records[index])
        item["_dataset_index"] = index
        return item


class WarmupCollator:
    def __init__(
        self,
        tokenizer,
        *,
        max_length: int,
        max_prompt_length: int,
        max_response_length: int,
        thinking_mode: str,
    ):
        if thinking_mode not in THINKING_MODES:
            raise ValueError(f"Unsupported thinking_mode: {thinking_mode!r}")
        if max_length < 3 or min(max_prompt_length, max_response_length) < 1:
            raise ValueError("Token limits must allow a prompt, a response, and EOS.")
        if tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer must define EOS and padding tokens.")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.thinking_mode = thinking_mode

    def _encode(self, prompt: str, response: str) -> dict[str, Any]:
        response = prepare_response_target(response, self.thinking_mode)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        if not prompt_ids or not response_ids:
            raise ValueError("Each training record must have a nonempty prompt and response.")
        total_length = len(prompt_ids) + len(response_ids) + 1
        if (
            len(prompt_ids) > self.max_prompt_length
            or len(response_ids) > self.max_response_length
            or total_length > self.max_length
        ):
            raise ValueError(
                f"Training record exceeds the token limits: prompt={len(prompt_ids)}, "
                f"response={len(response_ids)}, total_with_eos={total_length}. "
                "Increase max_prompt_length, max_response_length, or max_length "
                "in the warmup config to keep the complete proof."
            )
        input_ids = prompt_ids + response_ids + [self.tokenizer.eos_token_id]
        prompt_length = len(prompt_ids)
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
            "response_tokens": sum(label != -100 for label in labels),
        }

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = [
            self._encode(
                build_chat_prompt(
                    self.tokenizer,
                    item["problem"],
                    thinking_mode=self.thinking_mode,
                ),
                item["response"],
            )
            for item in batch
        ]
        max_length = max(len(item["input_ids"]) for item in encoded)
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []
        for item in encoded:
            pad_length = max_length - len(item["input_ids"])
            input_ids.append(
                item["input_ids"] + [self.tokenizer.pad_token_id] * pad_length
            )
            attention_mask.append(item["attention_mask"] + [0] * pad_length)
            labels.append(item["labels"] + [-100] * pad_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "metadata": [
                {
                    "dataset_index": item.get("_dataset_index"),
                    "problem_id": item.get("problem_id"),
                    "difficulty": item.get("difficulty"),
                    "answer": item.get("answer"),
                    "response_tokens": encoded[index]["response_tokens"],
                }
                for index, item in enumerate(batch)
            ],
        }


def load_model_and_tokenizer(config: dict[str, Any]):
    model_path = str(config["model_name_or_path"])
    if model_path.startswith("${"):
        raise ValueError(
            f"Unresolved model path {model_path!r}; set the model environment variable."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype_name = str(config.get("dtype", "float16")).lower()
    dtype_by_name = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    dtype = dtype_by_name[dtype_name] if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=config.get("device_map", "auto"),
        trust_remote_code=True,
    )
    model.config.use_cache = False

    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Warmup training requires peft.") from exc

    lora = config.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora.get("r", 8)),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora.get(
            "target_modules",
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model = get_peft_model(model, peft_config)
    if config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, tokenizer


def token_loss(model, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :].contiguous()
    labels = batch["labels"][:, 1:].contiguous()
    token_count = labels.ne(-100).sum()
    if token_count.item() == 0:
        raise ValueError("The batch has no supervised response tokens.")
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )
    return loss, token_count


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structured LoRA SFT warmup.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = expand_config_paths(
        yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    )
    thinking_mode = str(config.get("thinking_mode", "explicit"))
    if thinking_mode not in THINKING_MODES:
        raise ValueError(f"Unsupported thinking_mode: {thinking_mode!r}")

    seed = int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "training_logs"
    logs_dir.mkdir(exist_ok=True)
    (output_dir / "train_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model, tokenizer = load_model_and_tokenizer(config)
    dataset = WarmupDataset(
        config["train_file"], max_records=config.get("max_train_records")
    )
    if not dataset:
        raise ValueError(f"Warmup dataset is empty: {config['train_file']}")
    collator = WarmupCollator(
        tokenizer,
        max_length=int(config.get("max_length", 2048)),
        max_prompt_length=int(config.get("max_prompt_length", 1536)),
        max_response_length=int(config.get("max_response_length", 512)),
        thinking_mode=thinking_mode,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(config.get("micro_batch_size", 1)),
        shuffle=True,
        collate_fn=collator,
    )

    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 8))
    epochs = int(config.get("num_epochs", 1))
    if gradient_accumulation_steps < 1 or epochs < 1:
        raise ValueError("gradient_accumulation_steps and num_epochs must be positive.")
    updates_per_epoch = (
        len(dataloader) + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    total_updates = epochs * updates_per_epoch
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config.get("learning_rate", 2e-5)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(
            config.get("warmup_steps", max(1, total_updates // 10))
        ),
        num_training_steps=total_updates,
    )

    device = next(model.parameters()).device
    update = 0
    accumulated_micro_batches = 0
    running_loss = 0.0
    running_tokens = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=total_updates, desc=f"Warmup SFT ({thinking_mode})")
    for epoch in range(epochs):
        for micro_index, batch in enumerate(dataloader, start=1):
            tensors = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            loss, token_count = token_loss(model, tensors)
            loss.backward()
            accumulated_micro_batches += 1
            running_loss += float(loss.detach().cpu())
            running_tokens += int(token_count.detach().cpu())

            should_step = (
                micro_index % gradient_accumulation_steps == 0
                or micro_index == len(dataloader)
            )
            if not should_step:
                continue
            for parameter in trainable_parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(accumulated_micro_batches)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                max_norm=float(config.get("max_grad_norm", 1.0)),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            average_loss = running_loss / max(1, accumulated_micro_batches)
            append_jsonl(
                logs_dir / "updates.jsonl",
                {
                    "update": update,
                    "epoch": epoch + 1,
                    "thinking_mode": thinking_mode,
                    "loss": average_loss,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "tokens": running_tokens,
                },
            )
            progress.set_postfix(loss=f"{average_loss:.3f}")
            progress.update(1)
            accumulated_micro_batches = 0
            running_loss = 0.0
            running_tokens = 0

    progress.close()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metrics = {
        "num_records": len(dataset),
        "num_epochs": epochs,
        "total_updates": update,
        "thinking_mode": thinking_mode,
        "output_dir": str(output_dir),
    }
    (output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
