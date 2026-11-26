"""Compare symbolic music metrics between two sets of excerpts (RQ2).

Each metric is computed per excerpt with muspy. The two sets are compared as in Yang & Lerch
(2020): the intra-set distances of the reference set are compared with the inter-set distances
between the sets, through the overlap area (OA) and the KL divergence of their kernel density
estimates. When the evaluated set resembles the reference, both distance distributions match:
OA approaches 1 and KL approaches 0.
"""

import argparse
import json
from pathlib import Path

import muspy
import numpy as np

from .artifacts import output_path, read_json, write_json
from .runs import DEFAULT_CORPORA, load_excerpts, run_bars, run_rows
from .tokenizer import BAR_STEPS, STEPS_PER_QUARTER, Note


METRICS = {
    "pitch_range": muspy.pitch_range,
    "n_pitch_classes_used": muspy.n_pitch_classes_used,
    "pitch_class_entropy": muspy.pitch_class_entropy,
    "scale_consistency": muspy.scale_consistency,
    "polyphony": muspy.polyphony,
    "empty_beat_rate": muspy.empty_beat_rate,
    "groove_consistency": lambda music: muspy.groove_consistency(music, measure_resolution=BAR_STEPS),
}
GRID_POINTS = 1000
DENSITY_FLOOR = 1e-10  # added to both densities before KL so empty regions stay finite


def to_music(notes: list[Note]) -> muspy.Music:
    track = muspy.Track(program=0, notes=[
        muspy.Note(time=note.start, pitch=note.pitch, duration=note.end - note.start, velocity=note.velocity)
        for note in notes
    ])
    return muspy.Music(resolution=STEPS_PER_QUARTER, tracks=[track])


def metric_table(excerpts: list[list[Note]]) -> dict[str, np.ndarray]:
    """Return one array of per-excerpt values for each metric; undefined values become NaN."""
    musics = [to_music(notes) for notes in excerpts]
    return {name: np.array([function(music) for music in musics], dtype=float)
            for name, function in METRICS.items()}


def intra_distances(values: np.ndarray) -> np.ndarray:
    rows, columns = np.triu_indices(len(values), k=1)
    return np.abs(values[rows] - values[columns])


def inter_distances(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs(first[:, None] - second[None, :]).ravel()


def density(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Gaussian KDE with Scott's bandwidth, normalized to integrate to one on the grid."""
    spread = samples.std()
    if spread == 0:  # identical samples: fall back to one grid cell
        spread = (grid[-1] - grid[0]) / GRID_POINTS * len(samples) ** 0.2
    bandwidth = spread * len(samples) ** -0.2
    values = np.exp(-0.5 * ((grid[:, None] - samples[None, :]) / bandwidth) ** 2).sum(axis=1)
    return values / np.trapezoid(values, grid)


def compare_distributions(intra: np.ndarray, inter: np.ndarray) -> dict[str, float]:
    upper = max(intra.max(), inter.max())
    grid = np.linspace(0, upper if upper > 0 else 1, GRID_POINTS)
    p, q = density(intra, grid), density(inter, grid)
    p_floor, q_floor = p + DENSITY_FLOOR, q + DENSITY_FLOOR
    return {
        "overlap_area": float(np.trapezoid(np.minimum(p, q), grid)),
        "kl_divergence": float(np.trapezoid(p_floor * np.log(p_floor / q_floor), grid)),
    }


def compare_sets(evaluated: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> dict:
    """Summarize each metric per set and compare the evaluated set against the reference."""
    report = {}
    for name in METRICS:
        first, second = evaluated[name], reference[name]
        first, second = first[np.isfinite(first)], second[np.isfinite(second)]
        if len(first) < 2 or len(second) < 2:
            raise ValueError(f"Metric {name} has fewer than two defined values in a set")
        report[name] = {
            "evaluated_mean": float(first.mean()), "evaluated_std": float(first.std()),
            "reference_mean": float(second.mean()), "reference_std": float(second.std()),
            "evaluated_undefined": int(len(evaluated[name]) - len(first)),
            "reference_undefined": int(len(reference[name]) - len(second)),
            **compare_distributions(intra_distances(second), inter_distances(first, second)),
        }
    return report


def plot(evaluated: dict, reference: dict, labels: tuple[str, str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    figure, axes = pyplot.subplots(2, 4, figsize=(14, 6))
    for axis, name in zip(axes.flat, METRICS):
        values = [values[np.isfinite(values)] for values in (evaluated[name], reference[name])]
        low, high = min(v.min() for v in values), max(v.max() for v in values)
        grid = np.linspace(low - 0.1 * (high - low or 1), high + 0.1 * (high - low or 1), GRID_POINTS)
        for samples, label in zip(values, labels):
            axis.plot(grid, density(samples, grid), label=label)
        axis.set_title(name.replace("_", " "))
        axis.set_yticks([])
    axes.flat[-1].axis("off")
    axes.flat[0].legend()
    figure.tight_layout()
    with output_path(path) as temporary:
        figure.savefig(temporary, format=path.suffix.lstrip(".") or "png", dpi=150)
    pyplot.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluated", type=Path, required=True, help="run folder or token corpus JSON to evaluate")
    parser.add_argument("--reference", type=Path, required=True, help="human run folder or token corpus JSON")
    parser.add_argument("--bars", type=int, help="bars per excerpt; run folders supply their own")
    parser.add_argument("--sample", type=int, help="evaluate a fixed random subset of a token corpus")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpora", type=Path, default=DEFAULT_CORPORA, help="token corpora for human run rows")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    try:
        bars = args.bars
        if bars is None and args.reference.is_dir():  # a corpus takes its bar count from the run
            bars = run_bars(run_rows(args.reference))
        evaluated, evaluated_rejected, bars = load_excerpts(args.evaluated, bars, args.sample, args.seed, args.corpora)
        reference, reference_rejected, _ = load_excerpts(args.reference, bars, corpora=args.corpora)
        evaluated_values = metric_table(list(evaluated.values()))
        reference_values = metric_table(list(reference.values()))
        report = {
            "evaluated": str(args.evaluated), "reference": str(args.reference), "bars": bars,
            "sample": args.sample, "seed": args.seed,
            "evaluated_pieces": len(evaluated), "evaluated_rejected": evaluated_rejected,
            "reference_pieces": len(reference), "reference_rejected": reference_rejected,
            "metrics": compare_sets(evaluated_values, reference_values),
        }
        write_json(report, args.output)
        if args.figure is not None:
            plot(evaluated_values, reference_values, (args.evaluated.stem, args.reference.stem), args.figure)
        print(json.dumps(report["metrics"], indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")

if __name__ == "__main__":
    main()
