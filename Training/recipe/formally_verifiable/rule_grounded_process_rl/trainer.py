from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import torch
import transfer_queue as tq
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import apply_kl_penalty
from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.v1.trainer_base import register_trainer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils.dataset.rl_dataset import collate_fn
from verl.workers.utils.padding import response_to_nested

from recipe.formally_verifiable.rule_grounded_process_rl.advantages import assign_action_advantages
from recipe.formally_verifiable.rule_grounded_process_rl.reward import ProcessRewardConfig, RuleEmaBaseline


def _list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _rows(uids, extras) -> list[dict[str, Any]]:
    rows = []
    for uid, extra in zip(_list(uids), _list(extras), strict=False):
        extra = extra or {}
        rows.append(
            {
                "uid": str(uid),
                "problem_id": extra.get("problem_id"),
                "trajectory_reward": float(extra.get("trajectory_reward", 0.0)),
                "scored_response": extra.get("scored_response") or {},
                "response_text": extra.get("response_text", ""),
                "actions": [dict(action) for action in (extra.get("actions") or [])],
            }
        )
    return rows


def _flatten_scalars(prefix: str, value: Any, output: dict[str, float]) -> None:
    if isinstance(value, bool):
        output[prefix] = float(value)
    elif isinstance(value, (int, float)):
        output[prefix] = float(value)
    elif isinstance(value, dict):
        for key, nested in value.items():
            _flatten_scalars(f"{prefix}/{key}", nested, output)


@register_trainer("rule_grounded_process_rl_sync")
class RuleGroundedProcessRLTrainer(PPOTrainerSync):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.config.algorithm.get("rule_grounded_process_rl", {})
        reward_cfg = cfg.get("reward", {})
        process_cfg = ProcessRewardConfig(
            **{key: reward_cfg[key] for key in ProcessRewardConfig.__dataclass_fields__ if key in reward_cfg}
        )
        self.rule_baseline = RuleEmaBaseline(process_cfg)
        snapshot = cfg.get("initial_baseline_snapshot") or {}
        if snapshot:
            self.rule_baseline.load_snapshot(snapshot)
        self.retry_buffer = deque(maxlen=int(cfg.get("retry_buffer_max_size", 4096)))

    def _init_dataloader(self):
        trainer = self.config.trainer
        if trainer.get("val_before_train", False) or trainer.get("val_only", False):
            raise ValueError("Proof-R1 requires val_before_train=False and val_only=False.")
        if int(trainer.get("test_freq", -1)) > 0:
            raise ValueError("Proof-R1 requires test_freq=-1 for training without evaluation.")

        self.train_dataset = create_rl_dataset(
            self.config.data.train_files,
            self.config.data,
            self.tokenizer,
            self.processor,
            is_train=True,
            max_samples=self.config.data.get("train_max_samples", -1),
        )
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=self.config.data["dataloader_num_workers"],
            drop_last=True,
            collate_fn=collate_fn,
            sampler=create_rl_sampler(self.config.data, self.train_dataset),
        )
        if not len(self.train_dataloader):
            raise ValueError("The training dataset must contain at least one complete batch.")
        self.train_dataloader_it = None
        self.val_dataset = None
        self.val_dataloader = None

        total_steps = trainer.get("total_training_steps")
        if total_steps is None:
            total_steps = len(self.train_dataloader) * trainer.total_epochs
        self.total_training_steps = int(total_steps)
        if self.total_training_steps <= 0:
            raise ValueError("total_training_steps must be positive.")
        with open_dict(self.config):
            for path in ("actor_rollout_ref.actor.optim", "critic.optim"):
                optimizer = OmegaConf.select(self.config, path)
                if optimizer is not None:
                    optimizer.total_training_steps = self.total_training_steps

    def _save_checkpoint(self):
        folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        folder.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 1,
            "global_steps": int(self.global_steps),
            "rule_baseline": self.rule_baseline.snapshot(),
            "retry_buffer": list(self.retry_buffer),
        }
        temporary_path = None
        try:
            with NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, delete=False) as handle:
                temporary_path = Path(handle.name)
                json.dump(state, handle, ensure_ascii=False, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(folder / "proof_r1_state.json")
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        super()._save_checkpoint()

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if self.global_steps == 0:
            return
        trainer = self.config.trainer
        if trainer.resume_mode == "resume_path":
            folder = Path(trainer.resume_from_path)
        else:
            folder = Path(trainer.default_local_dir) / f"global_step_{self.global_steps}"
        path = folder / "proof_r1_state.json"
        if not path.is_file():
            raise FileNotFoundError(f"Cannot resume the rule baseline: missing {path}")
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("version") != 1 or state.get("global_steps") != self.global_steps:
            raise ValueError(f"Invalid Proof-R1 checkpoint state: {path}")
        retry_buffer = state.get("retry_buffer")
        if not isinstance(retry_buffer, list) or any(not isinstance(item, str) for item in retry_buffer):
            raise ValueError(f"Invalid retry buffer in {path}")
        baseline = RuleEmaBaseline(self.rule_baseline.config)
        baseline.load_snapshot(state["rule_baseline"])
        self.rule_baseline = baseline
        self.retry_buffer.clear()
        self.retry_buffer.extend(retry_buffer)


    def _write_sampled_responses(self, rows: list[dict[str, Any]], diagnostics: dict[str, Any]) -> None:
        cfg = self.config.algorithm.get("rule_grounded_process_rl", {})
        frequency = int(cfg.get("sampled_response_log_steps", 10))
        if frequency <= 0 or self.global_steps % frequency:
            return
        output_dir = Path(str(self.config.trainer.default_local_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "global_step": int(self.global_steps),
            "outcome_mode": diagnostics.get("outcome_mode"),
            "outcome_rewards": diagnostics.get("outcome_rewards"),
            "outcome_advantages": diagnostics.get("outcome_advantages"),
            "baseline_snapshot": diagnostics.get("baseline_snapshot"),
            "samples": [
                {
                    "uid": row["uid"],
                    "problem_id": row.get("problem_id"),
                    "response": row.get("response_text"),
                    "trajectory_reward": row.get("trajectory_reward"),
                    "scored_response": row.get("scored_response"),
                    "actions": row.get("actions"),
                }
                for row in rows
            ],
        }
        with (output_dir / "sampled_responses.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


    def _compute_advantage(self, batch, metrics: dict):
        fields = [
            "uid", "response_mask", "rm_scores", "rollout_log_probs", "old_log_probs",
            "ref_log_prob", "values", "extra_fields",
        ]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        response_mask_nested = data["response_mask"]
        padded = data.to_padded_tensor()
        response_mask = padded["response_mask"]
        rows = _rows(data["uid"], data["extra_fields"])

        proto = DataProto(batch=padded)
        proto.non_tensor_batch["uid"] = np.array([row["uid"] for row in rows], dtype=object)
        if self.config.algorithm.use_kl_in_reward:
            proto.batch["token_level_scores"] = proto.batch["rm_scores"]
            proto, kl_metrics = apply_kl_penalty(
                proto, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            metrics.update(kl_metrics)

        cfg = self.config.algorithm.get("rule_grounded_process_rl", {})
        negative_mass_ratio = cfg.get("negative_mass_ratio")
        advantages, diagnostics = assign_action_advantages(
            rows,
            response_mask,
            self.rule_baseline,
            lambda_process=float(cfg.get("lambda_process", 1.0)),
            lambda_outcome=float(cfg.get("lambda_outcome", 1.0)),
            failed_group_process_weight=float(cfg.get("failed_group_process_weight", 0.25)),
            outcome_requires_valid_response=bool(cfg.get("outcome_requires_valid_response", True)),
            eps=float(cfg.get("advantage_eps", 1e-6)),
            goal_binding_process_scale=float(cfg.get("goal_binding_process_scale", 1.0)),
            negative_mass_ratio=(
                float(negative_mass_ratio) if negative_mass_ratio is not None else None
            ),
            no_positive_process_weight=float(cfg.get("no_positive_process_weight", 1.0)),
        )
        self.retry_buffer.extend(diagnostics.pop("retry_problem_ids", []))
        diagnostics["retry_buffer_size"] = len(self.retry_buffer)
        self._write_sampled_responses(rows, diagnostics)
        scalar_metrics: dict[str, float] = {}
        _flatten_scalars("rule_grounded_process_rl", diagnostics, scalar_metrics)
        metrics.update(scalar_metrics)

        proto.batch["advantages"] = advantages
        proto.batch["returns"] = advantages.clone()
        output = {
            "advantages": response_to_nested(proto.batch["advantages"], response_mask_nested),
            "returns": response_to_nested(proto.batch["returns"], response_mask_nested),
        }
        if self.config.algorithm.use_kl_in_reward:
            output["token_level_rewards"] = response_to_nested(
                proto.batch["token_level_rewards"], response_mask_nested
            )
        return tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=TensorDict(output, batch_size=len(batch)),
        )
