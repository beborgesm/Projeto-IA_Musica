"""Score both systems on the same tokens and write the perplexity table (RQ1).

Both systems predict every token after BOS, EOS included, with no temperature, top-k or top-p.
The Markov model resets its context at each song; the Transformer resets at each song and each
block of its context length. The Transformer runs in Float32 so the table does not depend on
reduced precision.
"""

import argparse
import json
from pathlib import Path

from .artifacts import output_path, read_json, write_json
from .markov import evaluate_markov


def transformer_report(model: Path, corpus: dict, device_name: str) -> dict:
    from .neural import evaluate_checkpoint, select_device

    return evaluate_checkpoint(model, corpus, batch_size=16, device=select_device(device_name), precision="float32")


def table(rows: list[dict]) -> str:
    lines = [
        "| System | Split | Songs | Predicted tokens | Mean NLL (nats) | Perplexity |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['split']} | {row['songs']} | {row['predicted_tokens']:,} "
            f"| {row['mean_nll_nats']:.5f} | {row['perplexity']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--markov", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, nargs="+", required=True, help="token JSONs, e.g. val and test")
    parser.add_argument("--output", type=Path, required=True, help="folder for perplexity.json and perplexity.md")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    markov = read_json(args.markov)
    rows = []
    for path in args.corpus:
        corpus = read_json(path)
        rows.append({"system": "Markov (order 1)", "model": args.markov.name, **evaluate_markov(markov, corpus)})
        rows.append(
            {"system": "Transformer", "model": args.transformer.name,
             **transformer_report(args.transformer, corpus, args.device)}
        )
    for split in {row["split"] for row in rows}:
        counts = {row["predicted_tokens"] for row in rows if row["split"] == split}
        if len(counts) != 1:
            raise SystemExit(f"The systems scored different token counts on {split}: {sorted(counts)}")
    write_json(rows, args.output / "perplexity.json")
    with output_path(args.output / "perplexity.md") as temporary:
        temporary.write_text(table(rows), encoding="utf-8")
    print(json.dumps([{key: row[key] for key in ("system", "split", "perplexity")} for row in rows], indent=2))


if __name__ == "__main__":
    main()
