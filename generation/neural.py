"""Train and evaluate the causal piano Transformer on prepared token corpora."""

import argparse
from contextlib import nullcontext
import csv
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .artifacts import output_path, read_json
from .tokenizer import FORMAT_VERSION, TOKEN_IDS, VOCABULARY, token_sequences
from .transformer import PianoTransformer, TransformerConfig


CHECKPOINT_VERSION = 1
IGNORE_TARGET = -100
BOS = TOKEN_IDS["BOS"]


def prediction_blocks(corpus: dict, context: int) -> TensorDataset:
    """Use every next-token target once without carrying context across songs."""
    windows = []
    for _, tokens in token_sequences(corpus):
        for start in range(0, len(tokens) - 1, context):
            window = tokens[start : start + context + 1]
            windows.append(window + [BOS] * (context + 1 - len(window)))
    blocks = torch.tensor(windows, dtype=torch.long)
    targets = blocks[:, 1:]
    # IDs 1 through 352 map to output classes 0 through 351.
    targets = torch.where(targets == BOS, IGNORE_TARGET, targets - 1)
    return TensorDataset(blocks[:, :-1].contiguous(), targets)


def measure(
    model: PianoTransformer,
    batches: DataLoader,
    device: torch.device,
    precision: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    total_nll = 0.0
    total_targets = 0
    non_blocking = device.type == "cuda"
    with torch.set_grad_enabled(training):
        for inputs, targets in batches:
            token_count = int(targets.ne(IGNORE_TARGET).sum())
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()
            with autocast:
                logits = model(inputs.to(device, non_blocking=non_blocking))
                loss = F.cross_entropy(
                    logits.flatten(0, 1),
                    targets.flatten().to(device, non_blocking=non_blocking),
                    ignore_index=IGNORE_TARGET,
                    reduction="sum",
                )
            if optimizer is not None:
                (loss / token_count).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_nll += loss.detach().item()
            total_targets += token_count
    nll = total_nll / total_targets
    if not math.isfinite(nll):
        raise ValueError("Non-finite next-token loss")
    return {"predicted_tokens": total_targets, "mean_nll_nats": nll, "perplexity": math.exp(nll)}


def load_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if (
        checkpoint["checkpoint_version"] != CHECKPOINT_VERSION
        or checkpoint["token_format_version"] != FORMAT_VERSION
        or tuple(checkpoint["vocabulary"]) != VOCABULARY
    ):
        raise ValueError("Incompatible Transformer checkpoint")
    return checkpoint


def evaluate_checkpoint(
    path: Path, corpus: dict, batch_size: int, device: torch.device, precision: str
) -> dict:
    checkpoint = load_checkpoint(path, device)
    config = TransformerConfig(**checkpoint["model_config"])
    model = PianoTransformer(config).to(device)
    model.load_state_dict(checkpoint["best_model"])
    batches = DataLoader(
        prediction_blocks(corpus, config.context), batch_size=batch_size, pin_memory=device.type == "cuda"
    )
    return {
        "split": corpus["split"],
        "songs": len(corpus["songs"]),
        "epoch": checkpoint["best_epoch"],
        "precision": precision,
        **measure(model, batches, device, precision),
    }


def write_history(path: Path, rows: list[dict]) -> None:
    with output_path(path) as temporary, temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train(
    train_file: Path,
    validation_file: Path,
    output: Path,
    config: TransformerConfig,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
    precision: str,
    resume: bool = False,
) -> dict:
    train_bytes = train_file.read_bytes()
    validation_bytes = validation_file.read_bytes()
    train_corpus, validation_corpus = json.loads(train_bytes), json.loads(validation_bytes)
    for split, corpus in (("train", train_corpus), ("val", validation_corpus)):
        if corpus["split"] != split:
            raise ValueError(f"Expected the {split} corpus, got {corpus['split']}")
    train_ids = {song["song_id"] for song in train_corpus["songs"]}
    val_ids = {song["song_id"] for song in validation_corpus["songs"]}
    if train_ids & val_ids:
        raise ValueError("Training and validation contain shared song IDs")
    training_blocks = prediction_blocks(train_corpus, config.context)
    pin_memory = device.type == "cuda"
    validation_batches = DataLoader(
        prediction_blocks(validation_corpus, config.context), batch_size=batch_size, pin_memory=pin_memory
    )

    torch.manual_seed(seed)
    model = PianoTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    run = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "token_format_version": FORMAT_VERSION,
        "vocabulary": VOCABULARY,
        "model_config": asdict(config),
        "training_config": {
            "seed": seed, "batch_size": batch_size, "learning_rate": learning_rate, "precision": precision
        },
        "train_sha256": sha256(train_bytes).hexdigest(),
        "val_sha256": sha256(validation_bytes).hexdigest(),
    }
    history = []
    best_loss = math.inf
    best_epoch = 0
    best_model = {}
    if resume:
        prior = load_checkpoint(output, device)
        for name, value in run.items():
            if prior[name] != value:
                raise ValueError(f"Cannot resume with a different {name}")
        model.load_state_dict(prior["model"])
        optimizer.load_state_dict(prior["optimizer"])
        history = prior["history"]
        best_epoch = prior["best_epoch"]
        best_loss = history[best_epoch - 1]["val_mean_nll_nats"]
        best_model = prior["best_model"]
        if epochs < len(history):
            raise ValueError(f"The checkpoint already contains {len(history)} epochs")
        if epochs == len(history):
            write_history(output.with_suffix(".csv"), history)
            return history[best_epoch - 1]

    print(
        f"Training on {device} ({precision}) with {len(train_ids)} train and {len(val_ids)} validation songs",
        flush=True,
    )
    for epoch in range(len(history) + 1, epochs + 1):
        epoch_seed = seed + epoch
        torch.manual_seed(epoch_seed)
        generator = torch.Generator().manual_seed(epoch_seed)
        training_batches = DataLoader(
            training_blocks, batch_size=batch_size, shuffle=True, generator=generator, pin_memory=pin_memory
        )
        training = measure(model, training_batches, device, precision, optimizer)
        validation = measure(model, validation_batches, device, precision)
        result = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in training.items()},
            **{f"val_{key}": value for key, value in validation.items()},
        }
        history.append(result)
        if validation["mean_nll_nats"] < best_loss:
            best_loss = validation["mean_nll_nats"]
            best_epoch = epoch
            best_model = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        checkpoint = {
            **run,
            "history": history,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_model": best_model,
        }
        with output_path(output) as temporary:
            torch.save(checkpoint, temporary)
        write_history(output.with_suffix(".csv"), history)
        print(json.dumps(result), flush=True)
    return history[best_epoch - 1]


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch installation")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this PyTorch installation")
    return torch.device(name)


def select_precision(name: str, device: torch.device) -> str:
    if name == "float32":
        return name
    supports_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    if name == "bfloat16" and not supports_bfloat16:
        raise ValueError("BFloat16 requires a CUDA GPU with native BFloat16 support")
    return "bfloat16" if supports_bfloat16 else "float32"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--train", type=Path, required=True)
    training.add_argument("--val", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    defaults = TransformerConfig()
    training.add_argument("--epochs", type=positive_integer, default=20)
    training.add_argument("--context", type=positive_integer, default=defaults.context)
    training.add_argument("--width", type=positive_integer, default=defaults.width)
    training.add_argument("--heads", type=positive_integer, default=defaults.heads)
    training.add_argument("--layers", type=positive_integer, default=defaults.layers)
    training.add_argument("--dropout", type=float, default=defaults.dropout)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--learning-rate", type=float, default=0.0003)
    training.add_argument("--resume", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--corpus", type=Path, required=True)
    for command in (training, evaluation):
        command.add_argument("--batch-size", type=positive_integer, default=16)
        command.add_argument("--threads", type=positive_integer, default=4)
        command.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
        command.add_argument("--precision", choices=("auto", "float32", "bfloat16"), default="auto")
    training.set_defaults(device="cuda")
    evaluation.set_defaults(device="auto")
    args = parser.parse_args()

    try:
        device = select_device(args.device)
        precision = select_precision(args.precision, device)
        if device.type == "cpu":
            torch.set_num_threads(args.threads)
        if args.command == "train":
            if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
                raise ValueError("Expected a finite positive learning rate")
            if args.output.suffix != ".pt":
                raise ValueError("Use a .pt output path for the Transformer checkpoint")
            config = TransformerConfig(args.context, args.width, args.heads, args.layers, args.dropout)
            best = train(
                args.train, args.val, args.output, config,
                epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                seed=args.seed, device=device, precision=precision, resume=args.resume,
            )
            print(json.dumps({"best_epoch": best["epoch"], "val_perplexity": best["val_perplexity"]}))
        else:
            print(json.dumps(
                evaluate_checkpoint(args.model, read_json(args.corpus), args.batch_size, device, precision)
            ))
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
