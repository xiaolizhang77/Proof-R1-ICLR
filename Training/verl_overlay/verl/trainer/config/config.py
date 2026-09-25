from dataclasses import dataclass, field
from typing import Any, Optional

from verl.base_config import BaseConfig

__all__ = ["CheckpointConfig", "ProfileConfig", "BaseModelConfig"]


@dataclass
class CheckpointConfig(BaseConfig):


    save_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    load_contents: list[str] = field(default_factory=lambda: ["model", "optimizer", "extra"])
    async_save: bool = False
    strict: bool = True
    export_lora_adapter_dir: Optional[str] = None


@dataclass
class ProfileConfig(BaseConfig):


    profile_ranks: Optional[list[int]] = None
    step_start: int = -1
    step_end: int = -1
    save_path: Optional[str] = None


@dataclass
class BaseModelConfig(BaseConfig):


    path: str = "~/models/deepseek-llm-7b-chat"
    tokenizer_path: Optional[str] = None
    override_config: dict[str, Any] = field(default_factory=dict)
    external_lib: Optional[str] = None
    trust_remote_code: bool = False
    lora: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleConfig(BaseConfig):


    path: Optional[str] = None
    name: Optional[str] = None
