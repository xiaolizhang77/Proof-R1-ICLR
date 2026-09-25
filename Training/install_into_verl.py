from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


TESTED_VERL_COMMIT = "91666d9964282b890c75dd0b2d330edaee201c2f"
TRAINING_ROOT = Path(__file__).resolve().parent


def git_revision(verl_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(verl_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def copy_tree(source: Path, destination: Path) -> None:
    for source_file in source.rglob("*"):
        if not source_file.is_file() or source_file.suffix not in {".py", ".yaml"}:
            continue
        target = destination / source_file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", required=True, type=Path)
    args = parser.parse_args()

    verl_root = args.verl_root.resolve()
    if not (verl_root / "verl" / "__init__.py").exists():
        raise FileNotFoundError(f"Not a VERL checkout: {verl_root}")
    revision = git_revision(verl_root)
    if revision != TESTED_VERL_COMMIT:
        raise RuntimeError(
            f"Expected VERL {TESTED_VERL_COMMIT}, found {revision[:8]}. "
            "Check out the required revision before installing."
        )

    copy_tree(TRAINING_ROOT / "recipe", verl_root / "recipe")
    copy_tree(TRAINING_ROOT / "verl_overlay", verl_root)
    print(f"Installed Proof-R1 into {verl_root}")
    print(f"VERL revision: {revision[:8]}")


if __name__ == "__main__":
    main()
