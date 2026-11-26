"""Summarize the validity of sampled excerpts per run.

Each run folder holds a ``generations.csv`` from ``generation.sampling``. The report gives, per
run, the share of valid, invalid, incomplete and empty excerpts, the grammar errors behind the
invalid ones, why incomplete excerpts stopped, and the same-pitch collisions that the
tokenizer's normalization rule resolved in decodable excerpts.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import re
import statistics

from .artifacts import output_path, read_json, write_json
from .excerpts import STATUSES
from .tokenizer import TOKEN_IDS


EOS = TOKEN_IDS["EOS"]
COLUMN_ORDER = ("valid", "invalid", "incomplete", "empty")  # must match the table header
assert set(COLUMN_ORDER) == set(STATUSES)


def error_kind(message: str) -> str:
    """Group decoder messages that differ only in token indices or pitches."""
    return re.sub(r"\d+", "N", message).removesuffix(" at body token N")


def summarize_run(folder: Path) -> dict:
    with (folder / "generations.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{folder} has no generations")
    sequences = read_json(folder / "tokens.json")["sequences"]
    first = rows[0]
    statuses = Counter(row["status"] for row in rows)
    decodable = [row for row in rows if row["status"] in ("valid", "incomplete")]
    collisions = [int(row["overlaps_resolved"]) + int(row["same_onset_merged"]) for row in decodable]
    notes = [int(row["n_notes"]) for row in decodable]
    incomplete = [row for row in rows if row["status"] == "incomplete"]
    return {
        "run": folder.name,
        "system": first["system"],
        "bars": int(first["bars"]),
        "sampling": "syntax mask" if first["constrained"] == "True" else "free",
        "temperature": float(first["temperature"]),
        "top_k": int(first["top_k"]),
        "top_p": float(first["top_p"]),
        "seeds": len(rows),
        **{status: statuses.get(status, 0) for status in STATUSES},
        **{f"{status}_rate": statuses.get(status, 0) / len(rows) for status in STATUSES},
        "invalid_errors": dict(Counter(error_kind(row["error"]) for row in rows if row["status"] == "invalid")),
        "incomplete_by_eos": sum(sequences[row["gen_id"]][-1] == EOS for row in incomplete),
        "incomplete_by_token_limit": sum(sequences[row["gen_id"]][-1] != EOS for row in incomplete),
        "collisions_per_excerpt_median": statistics.median(collisions) if collisions else None,
        "collisions_per_note": sum(collisions) / sum(notes) if notes else None,
    }


def table(runs: list[dict]) -> str:
    lines = [
        "| System | Bars | Sampling | Temperature | Seeds | Valid | Invalid | Incomplete | Empty | Collisions per excerpt (median) | Collisions per note |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        median = run["collisions_per_excerpt_median"]
        per_note = run["collisions_per_note"]
        lines.append(
            f"| {run['system']} | {run['bars']} | {run['sampling']} | {run['temperature']:g} | {run['seeds']} "
            + " ".join(f"| {run[s]} ({run[s + '_rate']:.0%})" for s in COLUMN_ORDER)
            + f" | {'–' if median is None else f'{median:g}'} | {'–' if per_note is None else f'{per_note:.3f}'} |"
        )
    lines.append("")
    lines.append("Grammar errors behind invalid excerpts:")
    lines.append("")
    for run in runs:
        if run["invalid_errors"]:
            errors = ", ".join(f"{kind} ({count})" for kind, count in sorted(run["invalid_errors"].items(),
                                                                          key=lambda item: -item[1]))
            lines.append(f"- {run['system']}, {run['bars']} bars, {run['sampling']}: {errors}.")
    lines.append("")
    lines.append("Incomplete excerpts stopped by EOS / by the token limit:")
    lines.append("")
    for run in runs:
        if run["incomplete"]:
            lines.append(f"- {run['system']}, {run['bars']} bars, {run['sampling']}: "
                         f"{run['incomplete_by_eos']} / {run['incomplete_by_token_limit']}.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", type=Path, nargs="+", help="run folders from generation.sampling")
    parser.add_argument("--output", type=Path, required=True, help="folder for validity.json and validity.md")
    parser.add_argument("--name", default="validity", help="file stem of the report")
    args = parser.parse_args()
    runs = sorted((summarize_run(folder) for folder in args.runs),
                  key=lambda run: (run["bars"], run["sampling"], run["temperature"], run["system"]))
    write_json(runs, args.output / f"{args.name}.json")
    with output_path(args.output / f"{args.name}.md") as temporary:
        temporary.write_text(table(runs), encoding="utf-8")
    print(json.dumps([{key: run[key] for key in ("run", "seeds", "valid", "invalid", "incomplete")} for run in runs],
                     indent=1))


if __name__ == "__main__":
    main()
