"""Prepare POP909 token sequences and run generation baselines."""

import argparse
from collections import Counter
import csv
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path

import mido

from .artifacts import output_path, read_json, write_json
from .dataset import build_split, first_downbeat_seconds, read_key, write_csv
from .markov import evaluate_markov, fit_markov
from .excerpts import human_reference, write_table
from .ngram import NGramModel, evaluate_ngram, fit_ngram, select_order
from .tokenizer import (
    FORMAT_VERSION,
    TOKEN_IDS,
    VOCABULARY,
    decode_midi,
    encode_midi,
    grid_step_ticks,
    read_piano_notes,
    read_tempo_events,
    ticks_to_seconds,
)


SPLITS = ("train", "val", "test")
CONTEXT_LENGTHS = (256, 512, 1024)


def split_rows(manifest: Path, split: str) -> list[dict[str, str]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {"song_id", "split", "midi_path"}
        if not fields.issubset(reader.fieldnames or ()):
            raise ValueError("The manifest must include song_id, split, and midi_path")
        rows = []
        song_ids = set()
        for row in reader:
            song_id = row["song_id"]
            if not song_id or song_id in song_ids:
                raise ValueError(f"Missing or duplicate song ID on manifest line {reader.line_num}")
            if row["split"] not in SPLITS or not row["midi_path"]:
                raise ValueError(f"Invalid split or path for song {song_id}")
            song_ids.add(song_id)
            if row["split"] == split:
                rows.append(row)
    if not rows:
        raise ValueError(f"No songs in split {split}")
    return rows


def prepare_split(dataset_root: Path, manifest: Path, split: str) -> dict:
    """Tokenize every song in a split and verify its MIDI serialization."""
    dataset_root = dataset_root.resolve()
    songs = []
    observed_tokens = set()
    counts = Counter()
    keys = Counter()
    duration_seconds = 0.0
    for row in split_rows(manifest, split):
        song_id = row["song_id"]
        path = (dataset_root / row["midi_path"]).resolve()
        if not path.is_relative_to(dataset_root):
            raise ValueError(f"MIDI path escapes the dataset directory for song {song_id}")
        try:
            key = read_key(path.parent / "key_audio.txt")
            sequence, normalized, song_counts = encode_midi(
                mido.MidiFile(path), first_downbeat_seconds(path.parent / "beat_midi.txt")
            )
            decoded = decode_midi(sequence)
            with BytesIO() as buffer:
                decoded.save(file=buffer)
                buffer.seek(0)
                recovered = mido.MidiFile(file=buffer)
            step_ticks = grid_step_ticks(sequence["ticks_per_beat"])
            origin = sequence["origin_tick"]
            expected = [
                replace(note, start=origin + note.start * step_ticks, end=origin + note.end * step_ticks)
                for note in normalized
            ]
            programs = [
                message.program
                for track in recovered.tracks
                if track.name == "PIANO"
                for message in track
                if message.type == "program_change"
            ]
            if (
                read_piano_notes(recovered) != expected
                or read_tempo_events(recovered) != sequence["tempo_events"]
                or recovered.ticks_per_beat != sequence["ticks_per_beat"]
                or len(recovered.tracks) != 2
                or programs != [0]
            ):
                raise ValueError("MIDI round-trip mismatch")
        except (ValueError, OSError, EOFError) as error:
            raise ValueError(f"Song {song_id} ({row['midi_path']}) failed: {error}") from error

        counts.update(song_counts)
        counts["output_notes"] += len(normalized)
        counts["tokens"] += len(sequence["tokens"])
        counts["bars"] += sequence["tokens"].count(TOKEN_IDS["Bar"])
        observed_tokens.update(sequence["tokens"])
        keys[key] += 1
        seconds = ticks_to_seconds(sequence["tempo_events"], sequence["ticks_per_beat"], max(note.end for note in expected))
        duration_seconds += seconds
        songs.append({"song_id": song_id, "midi_path": row["midi_path"], "key": key,
                      "duration_seconds": seconds, **sequence})
    stats = {
        "songs": len(songs),
        "total_duration_hours": duration_seconds / 3600,
        "mean_duration_seconds": duration_seconds / len(songs),
        "mean_notes_per_song": counts["output_notes"] / len(songs),
        "major_key_songs": sum(count for key, count in keys.items() if key.endswith(":maj")),
        "minor_key_songs": sum(count for key, count in keys.items() if key.endswith(":min")),
        "distinct_keys": len(keys),
        "keys": dict(keys.most_common()),
        "vocabulary_size": len(VOCABULARY),
        "observed_vocabulary_size": len(observed_tokens),
        "mean_tokens_per_song": counts["tokens"] / len(songs),
        "mean_bars_per_song": counts["bars"] / len(songs),
        "mean_tokens_per_bar": counts["tokens"] / counts["bars"],
        **{f"bars_per_{length}_tokens": length * counts["bars"] / counts["tokens"] for length in CONTEXT_LENGTHS},
        "input_notes": counts["input_notes"],
        "output_notes": counts["output_notes"],
        "short_notes_extended": counts["short_notes_extended"],
        "same_onset_notes_merged": counts["same_onset_notes_merged"],
        "overlapping_notes_shortened": counts["overlapping_notes_shortened"],
        "mean_onset_error_steps": counts["onset_error_steps_sum"] / counts["input_notes"],
        "mean_end_error_steps": counts["end_error_steps_sum"] / counts["input_notes"],
    }
    return {
        "format_version": FORMAT_VERSION,
        "split": split,
        "vocabulary": VOCABULARY,
        "stats": stats,
        "songs": songs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("encode", "encode the merged POP909 tracks as a JSON token sequence"),
        ("decode", "decode a JSON token sequence back to MIDI"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("input", type=Path)
        command.add_argument("output", type=Path)
    split = commands.add_parser("split", help="filter POP909 to duple meter and write the song-level split")
    split.add_argument("--dataset-root", type=Path, required=True)
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--excluded", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="tokenize and verify a manifest split")
    prepare.add_argument("--dataset-root", type=Path, required=True)
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--split", choices=SPLITS, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train-markov", help="fit the Markov baseline on training tokens")
    train.add_argument("--corpus", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("evaluate-markov", help="measure next-token perplexity")
    evaluate.add_argument("--model", type=Path, required=True)
    evaluate.add_argument("--corpus", type=Path, required=True)
    evaluate.add_argument("--output", type=Path)
    select = commands.add_parser("select-ngram", help="score n-gram orders on validation and pick one")
    select.add_argument("--train", type=Path, required=True)
    select.add_argument("--val", type=Path, required=True)
    select.add_argument("--max-order", type=int, default=8)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument(
        "--discounts", type=float, nargs="*", default=[0.5, 0.75, 0.9],
        help="also score absolute discounting with these discounts (robustness check)",
    )
    train_ngram = commands.add_parser("train-ngram", help="fit the Witten-Bell n-gram baseline")
    train_ngram.add_argument("--corpus", type=Path, required=True)
    train_ngram.add_argument("--order", type=int, required=True)
    train_ngram.add_argument("--output", type=Path, required=True)
    train_ngram.add_argument("--smoothing", choices=("witten-bell", "absolute"), default="witten-bell")
    train_ngram.add_argument("--discount", type=float, default=0.75)
    evaluate_ngram_command = commands.add_parser("evaluate-ngram", help="measure n-gram perplexity")
    evaluate_ngram_command.add_argument("--model", type=Path, required=True)
    evaluate_ngram_command.add_argument("--corpus", type=Path, required=True)
    evaluate_ngram_command.add_argument("--output", type=Path)
    reference = commands.add_parser("reference", help="write the opening bars of a split's songs as MIDI")
    reference.add_argument("--corpus", type=Path, required=True)
    reference.add_argument("--bars", type=int, required=True)
    reference.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "encode":
            beats = args.input.parent / "beat_midi.txt"
            downbeat = first_downbeat_seconds(beats) if beats.exists() else None
            sequence, _, _ = encode_midi(mido.MidiFile(args.input), downbeat)
            write_json(sequence, args.output)
        elif args.command == "decode":
            midi = decode_midi(read_json(args.input))
            with output_path(args.output) as temporary:
                midi.save(temporary)
        elif args.command == "split":
            kept, excluded = build_split(args.dataset_root)
            write_csv(kept, ["song_id", "split", "midi_path"], args.manifest)
            write_csv(excluded, ["song_id", "split", "reason"], args.excluded)
            print(json.dumps({name: sum(row["split"] == name for row in kept) for name in SPLITS}))
        elif args.command == "prepare":
            corpus = prepare_split(args.dataset_root, args.manifest, args.split)
            write_json(corpus, args.output)
            print(json.dumps(corpus["stats"], indent=2))
        elif args.command == "train-markov":
            model = fit_markov(read_json(args.corpus))
            write_json(model, args.output)
            print(f"Fitted {model['training_transitions']} transitions from {model['training_songs']} songs")
        elif args.command == "select-ngram":
            if args.max_order < 1:
                raise ValueError("--max-order must be at least 1")
            smoothings = [("witten-bell", 0.75), *(("absolute", discount) for discount in args.discounts)]
            report = select_order(read_json(args.train), read_json(args.val), args.max_order, smoothings)
            write_json(report, args.output)
            for row in report["orders"]:
                keys = ("smoothing", "discount", "order", "perplexity", "seen_context_rate", "contexts")
                print(json.dumps({key: row[key] for key in keys}))
            print(f"Selected order {report['selected_order']}")
        elif args.command == "train-ngram":
            model = fit_ngram(read_json(args.corpus), args.order, args.smoothing, args.discount)
            write_json(model.to_json(), args.output)
            print(f"Fitted an order-{model.order} model on {model.training_songs} songs")
        elif args.command == "reference":
            rows = human_reference(read_json(args.corpus), args.bars, args.output)
            write_table(rows, args.output)
            notes = [row["n_notes"] for row in rows]
            print(json.dumps({"excerpts": len(rows), "bars": args.bars, "mean_notes": sum(notes) / len(notes)}))
        elif args.command == "evaluate-ngram":
            report = evaluate_ngram(NGramModel.from_json(read_json(args.model)), read_json(args.corpus))
            if args.output is not None:
                write_json(report, args.output)
            print(json.dumps(report, indent=2))
        else:
            report = evaluate_markov(read_json(args.model), read_json(args.corpus))
            if args.output is not None:
                write_json(report, args.output)
            print(json.dumps(report, indent=2))
    except (ValueError, TypeError, KeyError, OSError, EOFError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
