"""Summarize Edge Transformer experiment log completion status.

This is intentionally log-based: it does not assume that the Vera results
directory has already been synced back to the local machine.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


COMPLETE_MARKERS = (
    "Saved aligned summary to",
    '"test_metrics"',
    "[OldFlowMeanBaseline]",
    "All datasets saved to:",
)
FAILURE_MARKERS = (
    "Traceback (most recent call last):",
    "ValueError:",
    "EOFError:",
    "Exited with exit code",
)


@dataclass(frozen=True)
class LogStatus:
    stem: str
    group: str
    status: str
    reason: str
    out_file: str
    err_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check completion/failure state of Edge Transformer logs."
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        help="Directory containing *.out/*.err files.",
    )
    parser.add_argument(
        "--only-problems",
        action="store_true",
        help="Print only incomplete and failed runs.",
    )
    parser.add_argument(
        "--include-data",
        action="store_true",
        help="Also include data-generation logs such as ana_fast/ana_resume.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def group_for_name(name: str) -> str:
    if name.startswith("et_lcon"):
        return "lambda_con_sweep"
    if name.startswith("et_depth"):
        return "depth_sweep"
    if name.startswith("et_steps"):
        return "diffusion_steps_sweep"
    if name.startswith("abl_"):
        return "core_ablation"
    if name.startswith("edge_") or name.startswith("edge_full"):
        return "our_model"
    if name.startswith(("base_", "gate_", "old_mean")):
        return "baseline"
    if name.startswith("sue_runtime"):
        return "sue_runtime"
    if name.startswith("ana_"):
        return "data_generation"
    return "other"


def classify(out_path: Path) -> LogStatus:
    stem = out_path.stem
    err_path = out_path.with_suffix(".err")
    out_text = read_text(out_path)
    err_text = read_text(err_path)
    combined = f"{out_text}\n{err_text}"

    if any(marker in combined for marker in FAILURE_MARKERS):
        first_error = next(
            (line.strip() for line in combined.splitlines() if "Error:" in line or "Traceback" in line),
            "error marker found",
        )
        return LogStatus(stem, group_for_name(stem), "failed", first_error, out_path.name, err_path.name)

    has_end = "END_TIME" in out_text
    has_summary = any(marker in out_text for marker in COMPLETE_MARKERS)
    if has_end and has_summary:
        return LogStatus(stem, group_for_name(stem), "complete", "final summary present", out_path.name, err_path.name)

    if re.search(r"\[\s*\d+/\d+\]", out_text) or re.search(r"> Epoch \d+:", out_text):
        return LogStatus(stem, group_for_name(stem), "incomplete", "progress log without final summary", out_path.name, err_path.name)

    return LogStatus(stem, group_for_name(stem), "incomplete", "missing END_TIME or final summary", out_path.name, err_path.name)


def main() -> None:
    args = parse_args()
    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_absolute():
        logs_dir = project_root() / logs_dir
    if not logs_dir.exists():
        raise FileNotFoundError(f"Missing logs directory: {logs_dir}")

    rows = [classify(path) for path in sorted(logs_dir.glob("*.out"))]
    if not args.include_data:
        rows = [row for row in rows if row.group != "data_generation"]
    if args.only_problems:
        rows = [row for row in rows if row.status != "complete"]

    print("| group | status | run | reason |")
    print("|---|---|---|---|")
    for row in rows:
        print(f"| {row.group} | {row.status} | {row.stem} | {row.reason} |")


if __name__ == "__main__":
    main()
