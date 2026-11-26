"""Sample bar-limited excerpts from the Transformer or the Markov baseline.

Both systems share one sampler. A system only supplies next-token scores over the 353 output
classes (class i is token ID i + 1, so BOS is never sampled). Sampling is free by default:
malformed excerpts are kept and counted by ``excerpts.classify``. With ``constrained=True`` a
syntax mask, identical for both systems, allows only tokens the decoder accepts at that point.
Same-pitch collisions are not masked; ``excerpts.classify`` resolves them in both modes.

Each piece stops at the first of: EOS, the Bar token that would open bar ``bars + 1``, or
``max_tokens`` tokens. Every piece draws from its own seeded generator, so its tokens depend
only on its seed, the model and the sampling settings, not on the other pieces in the batch.
"""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

from .artifacts import read_json
from .excerpts import checkpoint_hash, record_generation, training_tempo, write_table
from .markov import next_token_log_probabilities
from .tokenizer import TOKEN_IDS, VOCABULARY


BOS = TOKEN_IDS["BOS"]
EOS = TOKEN_IDS["EOS"]
BAR = TOKEN_IDS["Bar"]
OUTPUT_CLASSES = len(VOCABULARY) - 1
TOKENS_PER_BAR_LIMIT = 200  # about three times the human density of ~71 tokens per bar


def _classes(prefix: str) -> torch.Tensor:
    mask = torch.zeros(OUTPUT_CLASSES, dtype=torch.bool)
    for token, event in enumerate(VOCABULARY):
        if token and (event == prefix or event.startswith(prefix + "_")):
            mask[token - 1] = True
    return mask


BARS, POSITIONS, PITCHES = _classes("Bar"), _classes("Position"), _classes("Pitch")
DURATIONS, VELOCITIES, ENDS = _classes("Duration"), _classes("Velocity"), _classes("EOS")
FIRST_POSITION = TOKEN_IDS["Position_0"] - 1


def syntax_mask(history: list[int]) -> torch.Tensor:
    """Output classes the decoder accepts after ``history``, which starts with BOS.

    After BOS: Bar. After Bar: Bar or a Position. After a Position: a Pitch. After a Pitch: a
    Duration. After a Duration: a Duration or a Velocity. After a Velocity: Bar, a later
    Position in the same bar, another Pitch at the same position, or EOS.
    """
    kind = VOCABULARY[history[-1]].split("_", 1)[0]
    if kind == "BOS":
        return BARS.clone()
    if kind == "Bar":
        return BARS | POSITIONS
    if kind == "Position":
        return PITCHES.clone()
    if kind == "Pitch":
        return DURATIONS.clone()
    if kind == "Duration":
        return DURATIONS | VELOCITIES
    if kind == "Velocity":
        position = -1
        for token in reversed(history):
            event = VOCABULARY[token]
            if event.startswith("Position_"):
                position = int(event.split("_")[1])
                break
            if event == "Bar":
                break
        later = POSITIONS.clone()
        later[FIRST_POSITION : FIRST_POSITION + position + 1] = False
        return BARS | later | PITCHES | ENDS
    raise ValueError(f"No continuation after {VOCABULARY[history[-1]]}")


def filter_probabilities(scores: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Turn next-token scores (log-probabilities or logits) into sampling probabilities.

    Order: temperature, then top-k (0 disables it), then top-p (1 disables it). Top-p keeps the
    smallest set of most likely tokens whose probability reaches ``top_p``; the most likely
    token is always kept.
    """
    if not temperature > 0 or top_k < 0 or not 0 < top_p <= 1:
        raise ValueError("Use temperature > 0, top_k >= 0 and 0 < top_p <= 1")
    scores = scores.float() / temperature
    if 0 < top_k < scores.shape[-1]:
        kth = scores.topk(top_k, dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < kth, -math.inf)
    if top_p < 1:
        ordered, order = scores.sort(dim=-1, descending=True)
        probabilities = ordered.softmax(dim=-1)
        before = probabilities.cumsum(dim=-1) - probabilities
        ordered = ordered.masked_fill(before >= top_p, -math.inf)
        scores = torch.full_like(scores, -math.inf).scatter(-1, order, ordered)
    return scores.softmax(dim=-1)


def sample_sequences(
    next_scores,
    seeds: list[int],
    *,
    bars: int,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    constrained: bool = False,
) -> list[list[int]]:
    """Sample one sequence per seed, growing all unfinished pieces by one token per step.

    ``next_scores(histories)`` returns a (len(histories), 353) tensor of scores. With
    ``constrained``, the syntax mask is applied before temperature, top-k and top-p.
    """
    if bars < 1 or max_tokens < 2:
        raise ValueError("Use at least one bar and a token limit of at least two")
    sequences = [[BOS] for _ in seeds]
    generators = [torch.Generator().manual_seed(seed) for seed in seeds]
    bar_counts = [0 for _ in seeds]
    active = list(range(len(seeds)))
    while active:
        scores = next_scores([sequences[index] for index in active]).float().cpu()
        if constrained:
            allowed = torch.stack([syntax_mask(sequences[index]) for index in active])
            scores = scores.masked_fill(~allowed, -math.inf)
        probabilities = filter_probabilities(scores, temperature, top_k, top_p)
        still_active = []
        for row, index in enumerate(active):
            token = int(torch.multinomial(probabilities[row], 1, generator=generators[index])) + 1
            sequences[index].append(token)
            bar_counts[index] += token == BAR
            stop = token == EOS or len(sequences[index]) >= max_tokens or (token == BAR and bar_counts[index] > bars)
            if not stop:
                still_active.append(index)
        active = still_active
    return sequences


class TransformerScores:
    """Next-token logits from a trained checkpoint, over a sliding window of the last tokens."""

    def __init__(self, checkpoint: dict, device: torch.device):
        from .transformer import PianoTransformer, TransformerConfig

        config = TransformerConfig(**checkpoint["model_config"])
        self.context = config.context
        self.device = device
        self.model = PianoTransformer(config).to(device)
        self.model.load_state_dict(checkpoint["best_model"])
        self.model.eval()

    @torch.no_grad()
    def __call__(self, histories: list[list[int]]) -> torch.Tensor:
        # Active pieces always share one length: each gains exactly one token per step.
        window = torch.tensor([history[-self.context :] for history in histories], device=self.device)
        return self.model(window)[:, -1]


class MarkovScores:
    """Next-token log-probabilities of the first-order Markov baseline."""

    def __init__(self, model: dict):
        self.table = torch.tensor(next_token_log_probabilities(model))

    def __call__(self, histories: list[list[int]]) -> torch.Tensor:
        return self.table[[history[-1] for history in histories]]


def generate(
    system: str,
    scores,
    checkpoint: Path,
    train_corpus: dict,
    output: Path,
    *,
    count: int,
    seed: int,
    bars: int,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    constrained: bool = False,
    until_valid: bool = False,
    max_seeds: int | None = None,
    overwrite: bool = False,
) -> list[dict]:
    """Sample pieces with consecutive seeds from ``seed`` and write them with their metadata.

    Without ``until_valid``, exactly ``count`` seeds are sampled. With it, seeds are drawn in
    order, in batches, until ``count`` excerpts are valid; the run then keeps every seed up to
    the one that completed the count, so the table also gives the validity rate. The evaluation
    set is the valid rows, which is a fixed rule rather than a hand-picked selection.
    ``max_seeds`` (default 20 times ``count``) bounds the search. When it runs out first, the
    table is still written with every drawn seed, so the low validity is recorded; the caller
    sees fewer than ``count`` valid rows.
    """
    if (output / "generations.csv").exists() and not overwrite:
        raise FileExistsError(f"{output / 'generations.csv'} exists; pass --overwrite to replace it")
    max_seeds = max_seeds or 20 * count
    digest = checkpoint_hash(checkpoint)
    tag = f"b{bars}-t{temperature:g}-k{top_k}-p{top_p:g}" + ("-syntax" if constrained else "")
    rows, raw = [], {}
    next_seed = seed
    valid = 0
    while True:
        needed = count - valid if until_valid else count - len(rows)
        batch_size = min(max(needed, 1) * 2 if until_valid and rows else needed, seed + max_seeds - next_seed)
        if needed <= 0 or batch_size <= 0:
            break
        seeds = list(range(next_seed, next_seed + batch_size))
        next_seed += batch_size
        sequences = sample_sequences(
            scores, seeds, bars=bars, max_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p,
            constrained=constrained,
        )
        for piece_seed, tokens in zip(seeds, sequences):
            if until_valid and valid == count:
                break
            gen_id = f"{system}-{tag}-s{piece_seed}"
            _, tempo = training_tempo(train_corpus, piece_seed)
            row = record_generation(
                output, gen_id, system, tokens, bars, seed=piece_seed, temperature=temperature, top_k=top_k,
                top_p=top_p, checkpoint=checkpoint, checkpoint_sha256=digest, tempo_us=tempo,
                constrained=constrained,
            )
            rows.append(row)
            raw[gen_id] = tokens
            valid += row["status"] == "valid"
        if not until_valid:
            break
    if until_valid and valid < count:
        print(f"Warning: only {valid} of {count} valid excerpts within {max_seeds} seeds", file=sys.stderr)
    write_table(rows, output, raw)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    systems = parser.add_subparsers(dest="system", required=True)
    for name in ("transformer", "markov"):
        command = systems.add_parser(name)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--train", type=Path, required=True, help="train token JSON, the tempo source")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--count", type=int, default=100)
        command.add_argument(
            "--until-valid", action="store_true",
            help="draw seeds in order until --count excerpts are valid (all drawn seeds are recorded)",
        )
        command.add_argument("--max-seeds", type=int, help="limit for --until-valid (default: 20 x --count)")
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--bars", type=int, default=16)
        command.add_argument("--max-tokens", type=int, help=f"default: {TOKENS_PER_BAR_LIMIT} per bar")
        command.add_argument("--temperature", type=float, default=1.0)
        command.add_argument("--top-k", type=int, default=0, help="0 disables top-k")
        command.add_argument("--top-p", type=float, default=1.0, help="1 disables top-p")
        command.add_argument("--syntax", action="store_true", help="mask tokens the decoder would reject")
        command.add_argument("--overwrite", action="store_true")
        if name == "transformer":
            command.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    max_tokens = args.max_tokens or TOKENS_PER_BAR_LIMIT * args.bars
    try:
        if args.system == "transformer":
            from .neural import load_checkpoint, select_device

            device = select_device(args.device)
            scores = TransformerScores(load_checkpoint(args.model, device), device)
        else:
            scores = MarkovScores(read_json(args.model))
        rows = generate(
            args.system, scores, args.model, read_json(args.train), args.output, count=args.count, seed=args.seed,
            bars=args.bars, max_tokens=max_tokens, temperature=args.temperature, top_k=args.top_k,
            top_p=args.top_p, constrained=args.syntax, until_valid=args.until_valid, max_seeds=args.max_seeds,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    statuses = {}
    for row in rows:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    seeds = [row["seed"] for row in rows]
    print(json.dumps({"pieces": len(rows), "seeds": [min(seeds), max(seeds)], "max_tokens": max_tokens,
                      "statuses": statuses}))


if __name__ == "__main__":
    main()
