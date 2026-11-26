"""Build the result tables from result JSONs, as Markdown and LaTeX (booktabs).

Every result table should come from this script, never from hand edits.
Each `--metrics` file becomes one block of rows in the metrics table, labelled with `--labels`
(defaulting to the file name). Each `--memorization` file becomes one row per view.
"""

import argparse
from pathlib import Path

from .artifacts import output_path, read_json
from .memorization import NGRAM_SIZES, VIEWS
from .metrics import METRICS


METRIC_NAMES = {
    "pitch_range": "Pitch range",
    "n_pitch_classes_used": "Pitch classes used",
    "pitch_class_entropy": "Pitch-class entropy",
    "scale_consistency": "Scale consistency",
    "polyphony": "Polyphony",
    "empty_beat_rate": "Empty-beat rate",
    "groove_consistency": "Groove consistency",
}
VIEW_NAMES = {"exact": "Exact", "transposition_invariant": "Transposition-invariant"}


def metrics_rows(report: dict, label: str) -> list[list[str]]:
    rows = []
    for name in METRICS:
        values = report["metrics"][name]
        rows.append([
            label, METRIC_NAMES[name],
            f"{values['evaluated_mean']:.3f} ± {values['evaluated_std']:.3f}",
            f"{values['reference_mean']:.3f} ± {values['reference_std']:.3f}",
            f"{values['overlap_area']:.3f}", f"{values['kl_divergence']:.3f}",
        ])
    return rows


def memorization_rows(report: dict, label: str) -> list[list[str]]:
    rows = []
    for view in VIEWS:
        values = report[view]
        rows.append([
            label, VIEW_NAMES[view], str(values["pieces"]), str(values["exact_copies"]),
            f"{values['longest_shared_notes_mean']:.1f} ({values['longest_shared_notes_median']:.0f}, "
            f"{values['longest_shared_notes_max']})",
            *(f"{100 * values[f'shared_{size}gram_fraction_mean']:.3f}%" for size in NGRAM_SIZES),
        ])
    return rows


def markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * 2 + ["---:"] * (len(header) - 2)) + "|"]
    return "\n".join([*lines, *("| " + " | ".join(row) + " |" for row in rows)]) + "\n"


def latex(header: list[str], rows: list[list[str]]) -> str:
    def escape(cell: str) -> str:
        return cell.replace("%", r"\%").replace("±", r"$\pm$").replace("_", r"\_")
    body = [" & ".join(escape(cell) for cell in row) + r" \\" for row in rows]
    return "\n".join([
        r"\begin{tabular}{ll" + "r" * (len(header) - 2) + "}", r"\toprule",
        " & ".join(escape(cell) for cell in header) + r" \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}",
    ]) + "\n"


def write_table(header: list[str], rows: list[list[str]], stem: Path) -> None:
    for suffix, render in ((".md", markdown), (".tex", latex)):
        with output_path(stem.with_suffix(suffix)) as temporary:
            temporary.write_text(render(header, rows), encoding="utf-8")


def labelled(paths: list[Path], labels: list[str] | None) -> list[tuple[Path, str]]:
    if labels is not None and len(labels) != len(paths):
        raise ValueError("Give one label per result file")
    return list(zip(paths, labels or [path.stem for path in paths]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", type=Path, nargs="*", default=[])
    parser.add_argument("--metrics-labels", nargs="*")
    parser.add_argument("--memorization", type=Path, nargs="*", default=[])
    parser.add_argument("--memorization-labels", nargs="*")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.metrics:
            rows = [row for path, label in labelled(args.metrics, args.metrics_labels)
                    for row in metrics_rows(read_json(path), label)]
            write_table(["Set", "Metric", "Evaluated", "Reference", "OA", "KL"], rows, args.out_dir / "metrics")
        if args.memorization:
            rows = [row for path, label in labelled(args.memorization, args.memorization_labels)
                    for row in memorization_rows(read_json(path), label)]
            header = ["Set", "View", "Pieces", "Exact copies", "Longest run, mean (median, max)",
                      *(f"Shared {size}-grams" for size in NGRAM_SIZES)]
            write_table(header, rows, args.out_dir / "memorization")
        print(f"Wrote tables to {args.out_dir}")
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
