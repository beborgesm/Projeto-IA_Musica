"""Write the report's compact tables and figures from the result files (Portuguese labels).

The report includes these files with \\input and \\includegraphics, so no number in it is typed by
hand. Rerun after any result changes:

    python -m generation.paper --out-dir generation/results/report
"""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from .artifacts import output_path, read_json
from .memorization import VIEWS
from .metrics import METRICS, density, metric_table
from .runs import load_excerpts, run_rows


RESULTS = Path("generation/results")
OUTPUTS = Path("generation/outputs")
SYSTEMS = (("train", "Humano"), ("markov", "Markov"), ("transformer", "Transformer"))
METRIC_NAMES = {
    "pitch_range": "Pitch range",
    "n_pitch_classes_used": "Pitch classes used",
    "pitch_class_entropy": "Pitch-class entropy",
    "scale_consistency": "Scale consistency",
    "polyphony": "Polyphony",
    "empty_beat_rate": "Empty-beat rate",
    "groove_consistency": "Groove consistency",
}
FIGURE_METRICS = ("pitch_class_entropy", "scale_consistency", "polyphony", "groove_consistency")
TEMPERATURES = (0.5, 0.8, 1.0, 1.2, 1.5)
MIN_VALID = 20  # fewer valid excerpts give no usable distribution, so only validity is reported


def number(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def write(path: Path, text: str) -> None:
    with output_path(path) as temporary:
        temporary.write_text(text, encoding="utf-8")


def tabular(columns: str, header: list[str], rows: list[list[str]], rules: tuple[int, ...] = ()) -> str:
    lines = [rf"\begin{{tabular}}{{{columns}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for index, row in enumerate(rows):
        if index in rules:
            lines.append(r"\midrule")
        lines.append(" & ".join(row) + r" \\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}"]) + "\n"


def dataset_table(corpora: Path) -> str:
    stats = {split: read_json(corpora / f"{split}.json")["stats"] for split in ("train", "val", "test")}
    def row(label, key, digits=0, scale=1):
        return [label, *(number(stats[split][key] * scale, digits) for split in stats)]
    rows = [
        row("Músicas", "songs"), row("Horas", "total_duration_hours", 1), row("Duração média (s)", "mean_duration_seconds", 1),
        row("Notas por música", "mean_notes_per_song", 0), row("Tokens por música", "mean_tokens_per_song", 0),
        row("Compassos por música", "mean_bars_per_song", 1),
        ["Maior / menor", *(f"{stats[split]['major_key_songs']} / {stats[split]['minor_key_songs']}" for split in stats)],
    ]
    return tabular("lrrr", ["", "Treino", "Val.", "Teste"], rows)


def perplexity_validity_table(bars: int) -> str:
    perplexity = {(row["system"].split()[0].lower(), row["split"]): row for row in read_json(RESULTS / "perplexity.json")}
    validity = {row["run"]: row for row in read_json(RESULTS / "validity.json")}
    rows = []
    for system, label in SYSTEMS[1:]:
        free, masked = validity[f"free-{system}-b{bars}"], validity[f"eval-{system}-b{bars}"]
        rows.append([
            label, number(perplexity[system, "val"]["perplexity"]), number(perplexity[system, "test"]["perplexity"]),
            *(f"{round(100 * free[f'{status}_rate'])}" for status in ("valid", "invalid", "incomplete")),
            f"{round(100 * masked['valid_rate'])}", f"{round(100 * masked['incomplete_rate'])}",
        ])
    header = ["Sistema", "PPL val.", "PPL teste", r"Vál.", r"Inv.", r"Inc.", r"Vál.", r"Inc."]
    body = tabular("lrrrrrrr", header, rows)
    group = r" & \multicolumn{2}{c}{Perplexidade $\downarrow$} & \multicolumn{3}{c}{Livre (\%)} & \multicolumn{2}{c}{Máscara (\%)} \\"
    rule = r"\cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-8}"
    return body.replace(r"\toprule" + "\n", r"\toprule" + "\n" + group + "\n" + rule + "\n", 1)


def metrics_table(bars: int) -> str:
    reports = {system: read_json(RESULTS / f"metrics-{system}-b{bars}.json")["metrics"] for system, _ in SYSTEMS}
    rows = []
    for name in METRICS:
        rows.append([METRIC_NAMES[name], *(cell for system, _ in SYSTEMS for cell in (
            number(reports[system][name]["overlap_area"]), number(reports[system][name]["kl_divergence"])))])
    rows.append([r"\textbf{Média}", *(rf"\textbf{{{number(statistics.mean(v[key] for v in reports[system].values()))}}}"
                                      for system, _ in SYSTEMS for key in ("overlap_area", "kl_divergence"))])
    header = ["Métrica", *(cell for _ in SYSTEMS for cell in (r"OA $\uparrow$", r"KL $\downarrow$"))]
    body = tabular("lrrrrrr", header, rows, rules=(len(METRICS),))
    group = " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{label}}}" for _, label in SYSTEMS) + r" \\"
    rule = r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}"
    return body.replace(r"\toprule" + "\n", r"\toprule" + "\n" + group + "\n" + rule + "\n", 1)


def memorization_table(bars: int) -> str:
    runs = (("reference", "Humano"), ("eval-markov", "Markov"), ("eval-transformer", "Transformer"))
    reports = {run: read_json(RESULTS / f"memorization-{run}-b{bars}.json") for run, _ in runs}
    rows = []
    for view, view_label in zip(VIEWS, ("exata", "transp.")):
        rows += [
            [f"Cópias integrais, {view_label}", *(str(reports[run][view]["exact_copies"]) for run, _ in runs)],
            [f"4-gramas no treino (\\%), {view_label}",
             *(number(100 * reports[run][view]["shared_4gram_fraction_mean"]) for run, _ in runs)],
            [f"8-gramas no treino (\\%), {view_label}",
             *(number(100 * reports[run][view]["shared_8gram_fraction_mean"], 3) for run, _ in runs)],
            [f"Maior trecho (notas), {view_label}",
             *(f"{number(reports[run][view]['longest_shared_notes_mean'], 1)} ({reports[run][view]['longest_shared_notes_max']})"
               for run, _ in runs)],
        ]
    return tabular("lrrr", ["Indicador", *(label for _, label in runs)], rows, rules=(4,))


def run_name(system: str, bars: int, temperature: float) -> str:
    return f"eval-{system}-b{bars}" if temperature == 1.0 else f"temp-{system}-b{bars}-t{temperature:g}"


def temperature_rows(bars: int) -> list[dict]:
    """One row per system and temperature: validity from the run, metrics and memorization if computed."""
    rows = []
    for system, label in SYSTEMS[1:]:
        for temperature in TEMPERATURES:
            run = run_name(system, bars, temperature)
            statuses = [row["status"] for row in run_rows(OUTPUTS / run)]
            row = {"system": label, "temperature": temperature, "seeds": len(statuses),
                   "valid": statuses.count("valid"), "valid_rate": statuses.count("valid") / len(statuses)}
            metrics = RESULTS / (f"metrics-{system}-b{bars}.json" if temperature == 1.0 else f"metrics-{run}.json")
            memorization = RESULTS / f"memorization-{run}.json"
            if row["valid"] >= MIN_VALID:
                values = read_json(metrics)["metrics"].values()
                exact = read_json(memorization)["exact"]
                row.update(mean_oa=statistics.mean(v["overlap_area"] for v in values),
                           shared_4grams=exact["shared_4gram_fraction_mean"],
                           longest=exact["longest_shared_notes_mean"])
            rows.append(row)
    return rows


def human_lines(bars: int) -> dict:
    metrics = read_json(RESULTS / f"metrics-train-b{bars}.json")["metrics"].values()
    exact = read_json(RESULTS / f"memorization-reference-b{bars}.json")["exact"]
    return {"mean_oa": statistics.mean(v["overlap_area"] for v in metrics),
            "shared_4grams": exact["shared_4gram_fraction_mean"], "longest": exact["longest_shared_notes_mean"]}


def temperature_table(rows: list[dict], human: dict) -> str:
    body = []
    for temperature in TEMPERATURES:
        cells = [number(temperature, 1)]
        for label in ("Markov", "Transformer"):
            row = next(r for r in rows if r["system"] == label and r["temperature"] == temperature)
            cells.append(f"{round(100 * row['valid_rate'])}")
            if "mean_oa" in row:
                cells += [number(row["mean_oa"]), number(100 * row["shared_4grams"]), number(row["longest"], 1)]
            else:
                cells.append(rf"\multicolumn{{3}}{{c}}{{{row['valid']} válidos em {row['seeds']}}}")
        body.append(cells)
    reference = [number(human["mean_oa"]), number(100 * human["shared_4grams"]), number(human["longest"], 1)]
    body.append(["Humano", "--", *reference, "--", *reference])
    header = [r"$\tau$", *([r"Vál. (\%)", r"OA $\uparrow$", r"4-gr. (\%)", "Maior"] * 2)]
    table = tabular("lrrrrrrrr", header, body, rules=(len(TEMPERATURES),))
    group = r" & \multicolumn{4}{c}{Markov} & \multicolumn{4}{c}{Transformer} \\"
    rule = r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}"
    return table.replace(r"\toprule" + "\n", r"\toprule" + "\n" + group + "\n" + rule + "\n", 1)


def temperature_figure(rows: list[dict], human: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    panels = (("valid_rate", "Sementes válidas (%)", 100), ("mean_oa", "OA média", 1),
              ("shared_4grams", "4-gramas no treino (%)", 100))
    figure, axes = pyplot.subplots(1, len(panels), figsize=(7.0, 1.45))
    for axis, (key, title, scale) in zip(axes, panels):
        # Same colors as the metric figure: Markov orange, Transformer green.
        for label, style, color in (("Markov", "--o", "tab:orange"), ("Transformer", "-.s", "tab:green")):
            points = [(r["temperature"], scale * r[key]) for r in rows if r["system"] == label and key in r]
            axis.plot(*zip(*points), style, color=color, label=label, markersize=3, linewidth=1.2)
        if key in human:
            axis.axhline(scale * human[key], color="gray", linestyle=":", linewidth=1, label="Humano")
        axis.set_title(title, fontsize=8)
        axis.set_xticks(TEMPERATURES)
        axis.tick_params(labelsize=6)
        axis.set_xlabel(r"$\tau$", fontsize=7)
    axes[1].legend(fontsize=6, frameon=False)
    figure.tight_layout(pad=0.3)
    with output_path(path) as temporary:
        figure.savefig(temporary, format="pdf")
    pyplot.close(figure)


def metric_figure(bars: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    sets = {
        "Humano (teste)": load_excerpts(OUTPUTS / f"reference-b{bars}")[0],
        "Markov": load_excerpts(OUTPUTS / f"eval-markov-b{bars}")[0],
        "Transformer": load_excerpts(OUTPUTS / f"eval-transformer-b{bars}")[0],
    }
    values = {label: metric_table(list(pieces.values())) for label, pieces in sets.items()}
    figure, axes = pyplot.subplots(1, len(FIGURE_METRICS), figsize=(7.0, 1.45))
    for axis, name in zip(axes, FIGURE_METRICS):
        samples = [values[label][name][np.isfinite(values[label][name])] for label in sets]
        low, high = min(s.min() for s in samples), max(s.max() for s in samples)
        pad = 0.05 * (high - low or 1)
        grid = np.linspace(low - pad, high + pad, 500)
        curves = [density(sample, grid) for sample in samples]
        for curve, label, style in zip(curves, sets, ("-", "--", "-.")):
            axis.plot(grid, curve, style, label=label, linewidth=1.2)
        # A narrow Markov peak would flatten the others; scale to the human and Transformer curves.
        axis.set_ylim(0, 1.25 * max(curves[0].max(), curves[2].max()))
        axis.set_title(METRIC_NAMES[name], fontsize=8)
        axis.tick_params(labelsize=6)
        axis.set_yticks([])
    axes[1].legend(fontsize=6, frameon=False, loc="upper left")
    figure.tight_layout(pad=0.3)
    with output_path(path) as temporary:
        figure.savefig(temporary, format="pdf")
    pyplot.close(figure)


def training_figure(checkpoint: Path, path: Path) -> None:
    import matplotlib
    import matplotlib.ticker
    import torch

    matplotlib.use("Agg")
    from matplotlib import pyplot

    rows = torch.load(checkpoint, map_location="cpu", weights_only=True)["history"]
    epochs = [int(row["epoch"]) for row in rows]
    best = min(rows, key=lambda row: float(row["val_mean_nll_nats"]))
    figure, axis = pyplot.subplots(figsize=(3.3, 1.45))
    axis.plot(epochs, [float(row["train_mean_nll_nats"]) for row in rows], label="Treino")
    axis.plot(epochs, [float(row["val_mean_nll_nats"]) for row in rows], "--", label="Validação")
    axis.axvline(int(best["epoch"]), color="gray", linewidth=0.8, linestyle=":")
    axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axis.set_xlabel("Época", fontsize=8)
    axis.set_ylabel("Perda (nats)", fontsize=8)
    axis.tick_params(labelsize=7)
    axis.legend(fontsize=7, frameon=False)
    figure.tight_layout(pad=0.3)
    with output_path(path) as temporary:
        figure.savefig(temporary, format="pdf")
    pyplot.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bars", type=int, default=32)
    parser.add_argument("--corpora", type=Path, default=Path("generation/tokens"))
    parser.add_argument("--checkpoint", type=Path, default=Path("generation/checkpoints/transformer-remi.pt"))
    args = parser.parse_args()
    write(args.out_dir / "tab_dataset.tex", dataset_table(args.corpora))
    write(args.out_dir / "tab_perplexity_validity.tex", perplexity_validity_table(args.bars))
    write(args.out_dir / "tab_metrics.tex", metrics_table(args.bars))
    write(args.out_dir / "tab_memorization.tex", memorization_table(args.bars))
    metric_figure(args.bars, args.out_dir / "fig_metrics.pdf")
    training_figure(args.checkpoint, args.out_dir / "fig_training.pdf")
    rows, human = temperature_rows(args.bars), human_lines(args.bars)
    write(args.out_dir / "tab_temperature.tex", temperature_table(rows, human))
    temperature_figure(rows, human, args.out_dir / "fig_temperature.pdf")
    print(json.dumps(sorted(path.name for path in args.out_dir.iterdir())))


if __name__ == "__main__":
    main()
