"""Plot training and validation perplexity from a Transformer run's CSV."""

import argparse
import csv
from pathlib import Path

from .artifacts import output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    with args.history.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        parser.error("the history CSV has no epochs")
    epochs = [int(row["epoch"]) for row in rows]
    figure, axis = plt.subplots(figsize=(7, 4), layout="constrained")
    axis.plot(epochs, [float(row["train_perplexity"]) for row in rows], "o-", label="Training")
    axis.plot(epochs, [float(row["val_perplexity"]) for row in rows], "s-", label="Validation")
    axis.set(xlabel="Epoch", ylabel="Perplexity")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(alpha=0.25)
    axis.legend()
    with output_path(args.output) as temporary:
        figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
