from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
import sys
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from recipe.formally_verifiable.config_utils import load_recipe_config
from recipe.formally_verifiable.rule_grounded_process_rl.structured_prompt import (
    build_messages as build_rule_grounded_messages,
)

METHOD = "rule_grounded_process_rl"
DATA_SOURCE = "formally_verifiable/rule_grounded_process_rl"


def validate_raw_problem(problem: Any) -> None:
    if not isinstance(problem, dict):
        raise ValueError("Expected a raw problem object, not a JSON scalar or array.")
    if any(key in problem for key in ("prompt", "extra_info", "reward_model")):
        raise ValueError(
            "Expected a raw problem, but received an already converted VERL record. "
            "Use data/proverqa/train.jsonl as raw_train_files."
        )
    missing = [
        key for key in ("id", "question", "nl2fol", "options", "answer", "conclusion_fol")
        if key not in problem or problem[key] is None
    ]
    if missing:
        raise ValueError(f"Raw problem is missing required fields: {missing}")
    if not str(problem["id"]).strip():
        raise ValueError("Raw problem id must not be empty.")
    for key in ("question", "conclusion_fol"):
        if not isinstance(problem[key], str) or not problem[key].strip():
            raise ValueError(f"Raw problem {key} must be a nonempty string.")
    premises = problem["nl2fol"]
    if not isinstance(premises, dict) or not premises or any(
        not isinstance(value, str) or not value.strip()
        for pair in premises.items() for value in pair
    ):
        raise ValueError("Raw problem nl2fol must map nonempty text to nonempty formulas.")
    options = problem["options"]
    if not isinstance(options, list) or not options or any(
        not isinstance(option, str) or not option.strip() for option in options
    ):
        raise ValueError("Raw problem options must be a nonempty list of strings.")
    if problem["answer"] not in ("A", "B", "C"):
        raise ValueError("Raw problem answer must be A, B, or C.")


def convert_problem_record(
    problem: dict[str, Any], *, method: str, thinking_mode: str = "explicit"
) -> dict[str, Any]:
    if method != METHOD:
        raise ValueError(f"Unsupported method: {method}")
    validate_raw_problem(problem)
    prompt = build_rule_grounded_messages(problem, thinking_mode=thinking_mode)
    serialized_problem = json.dumps(problem, ensure_ascii=False)
    return {
        "data_source": DATA_SOURCE,
        "prompt": prompt,
        "ability": "formal_logic",
        "reward_model": {"style": "rule", "ground_truth": serialized_problem},
        "extra_info": {
            "problem": serialized_problem,
            "problem_id": problem.get("id"),
            "method": method,
        },
    }


def iter_converted_records(
    input_path: str | Path,
    *,
    method: str,
    exclude_uncertain: bool = True,
    thinking_mode: str = "explicit",
) -> Iterator[dict[str, Any]]:
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                problem = json.loads(line)
                converted = convert_problem_record(problem, method=method, thinking_mode=thinking_mode)
            except ValueError as exc:
                raise ValueError(f"{input_path}:{line_number}: {exc}") from exc
            if exclude_uncertain and str(problem.get("answer", "")).strip().upper() == "C":
                continue
            yield converted


def write_jsonl(records: Iterator[dict[str, Any]], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _as_file_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _preprocess_file_pairs(
    *,
    raw_files: Any,
    output_files: Any,
    method: str,
    exclude_uncertain: bool,
    thinking_mode: str,
) -> None:
    raw_paths = _as_file_list(raw_files)
    output_paths = _as_file_list(output_files)
    if len(raw_paths) != len(output_paths):
        raise ValueError(
            "raw and processed file lists must have the same length: "
            f"{len(raw_paths)} raw vs {len(output_paths)} processed"
        )
    for raw_path, output_path in zip(raw_paths, output_paths, strict=True):
        if Path(raw_path).resolve() == Path(output_path).resolve():
            raise ValueError("The processed output must not overwrite the raw training data")
        write_jsonl(
            iter_converted_records(
                raw_path, method=method, exclude_uncertain=exclude_uncertain, thinking_mode=thinking_mode
            ),
            output_path,
        )


def preprocess_from_config(config_path: str | Path) -> None:
    config = load_recipe_config(config_path)

    method = config["method"]
    data = config["data"]
    unresolved = [
        str(data[key])
        for key in ("raw_train_files", "train_files")
        if "${oc.env:" in str(data[key])
    ]
    if unresolved:
        raise ValueError("Set PROOF_R1_ROOT before preprocessing; unresolved paths: " + ", ".join(unresolved))
    default_exclude_uncertain = bool(data.get("exclude_uncertain", True))
    exclude_uncertain_train = bool(
        data.get("exclude_uncertain_train", default_exclude_uncertain)
    )
    _preprocess_file_pairs(
        raw_files=data["raw_train_files"],
        output_files=data["train_files"],
        method=method,
        exclude_uncertain=exclude_uncertain_train,
        thinking_mode=config["rule_grounded_process_rl"].get("thinking_mode", "explicit"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe-config")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--method", choices=[METHOD], default=METHOD)
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument("--thinking-mode", choices=["explicit", "native"], default="explicit")
    args = parser.parse_args()
    if args.recipe_config:
        preprocess_from_config(args.recipe_config)
        return
    if not args.input or not args.output or not args.method:
        parser.error("--input, --output, and --method are required unless --recipe-config is set")
    _preprocess_file_pairs(
        raw_files=args.input, output_files=args.output, method=args.method,
        exclude_uncertain=not args.include_uncertain, thinking_mode=args.thinking_mode,
    )


if __name__ == "__main__":
    main()
