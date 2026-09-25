from __future__ import annotations

import os
from pathlib import Path
from typing import Any


PROOF_R1_ROOT = Path(os.environ.get("PROOF_R1_ROOT", Path(__file__).resolve().parents[2]))


def expand_config_paths(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    for key in ("model_name_or_path", "train_file", "output_dir"):
        value = result.get(key)
        if not isinstance(value, str):
            continue
        value = value.replace("${PROOF_R1_ROOT}", str(PROOF_R1_ROOT))
        result[key] = os.path.expandvars(value)
    return result
