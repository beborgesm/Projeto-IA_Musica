"""Measure how much of each excerpt also appears in the training songs (RQ3).

Pieces are compared as note sequences ordered by onset and pitch. A run of notes matches when every
note has the same pitch and duration and every note after the first has the same onset gap from
its predecessor, all in grid steps. The transposition-invariant view replaces each pitch after the
first with the interval from the previous note and ignores the first pitch, so a copy shifted to
another key still matches. Velocity is ignored in both views.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .artifacts import read_json, write_json
from .runs import DEFAULT_CORPORA, load_excerpts
from .tokenizer import Note, decode_notes, token_sequences


NGRAM_SIZES = (4, 8, 16)
VIEWS = ("exact", "transposition_invariant")
_BASE = np.uint64(0x9E3779B97F4A7C15)  # odd, so it has an inverse modulo 2**64
_BASE_INVERSE = np.uint64(pow(int(_BASE), -1, 2**64))


def note_symbols(notes: list[Note], view: str) -> list[tuple[int, int, int]]:
    """Describe each note relative to its predecessor: (onset gap, pitch or interval, duration).

    The first note's gap and interval are zero; windows never compare them (see WindowHashes).
    """
    ordered = sorted(notes, key=lambda note: (note.start, note.pitch))
    symbols = []
    previous = None
    for note in ordered:
        onset_gap = note.start - previous.start if previous else 0
        pitch = note.pitch if view == "exact" else note.pitch - previous.pitch if previous else 0
        symbols.append((onset_gap, pitch, note.end - note.start))
        previous = note
    return symbols


def head_symbol(note: tuple[int, int, int], previous_pitch: int | None, view: str) -> tuple:
    """What a window compares for its first note: pitch and duration, or the duration alone."""
    return ("head", previous_pitch, note[2]) if view == "exact" else ("head", note[2])


class SymbolTable:
    def __init__(self):
        self.ids = {}

    def encode(self, notes: list[Note], view: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the IDs of each note's full symbol and of its symbol as a window's first note."""
        symbols = note_symbols(notes, view)
        pitches = [note.pitch for note in sorted(notes, key=lambda note: (note.start, note.pitch))]
        heads = [head_symbol(symbol, pitch, view) for symbol, pitch in zip(symbols, pitches)]
        def ids(values):
            return np.array([self.ids.setdefault(value, len(self.ids) + 1) for value in values], dtype=np.uint64)
        return ids(heads), ids(symbols)


class WindowHashes:
    """Position-independent polynomial hashes of every contiguous window, modulo 2**64.

    A window hashes its first note's head symbol with the full symbols of the notes after it, so
    the gap and interval to a note outside the window never take part.
    """

    def __init__(self, sequences: list[tuple[np.ndarray, np.ndarray]]):
        empty = np.zeros(0, dtype=np.uint64)
        self.heads = np.concatenate([heads for heads, _ in sequences]) if sequences else empty
        values = np.concatenate([full for _, full in sequences]) if sequences else empty
        count = len(values) + 1
        with np.errstate(over="ignore"):
            inverse_powers = np.cumprod(np.full(count, _BASE_INVERSE, dtype=np.uint64)) * _BASE
            self.powers = np.cumprod(np.full(count, _BASE, dtype=np.uint64)) * _BASE_INVERSE
            self.prefix = np.concatenate([[np.uint64(0)], np.cumsum(values * inverse_powers[:-1], dtype=np.uint64)])
        sizes = [len(full) for _, full in sequences]
        self.room = np.repeat(np.cumsum(sizes), sizes) - np.arange(len(values))  # notes left in the song

    def windows(self, length: int) -> np.ndarray:
        starts = np.nonzero(self.room >= length)[0]
        with np.errstate(over="ignore"):
            tail = (self.prefix[starts + length] - self.prefix[starts + 1]) * self.powers[starts + 1]
            return tail * _BASE + self.heads[starts]


def shared_fraction(piece: WindowHashes, reference: np.ndarray, length: int) -> float:
    windows = piece.windows(length)
    return float(np.isin(windows, reference).mean()) if len(windows) else float("nan")


def longest_shared(pieces: list[WindowHashes], train: WindowHashes) -> list[int]:
    """Longest contiguous run of each piece found anywhere in the training songs.

    Lengths grow one note at a time for all pieces together. A shared run of length L + 1 contains
    one of length L, so a piece drops out at the first length it no longer shares.
    """
    longest = [0] * len(pieces)
    active = list(range(len(pieces)))
    length = 0
    while active:
        length += 1
        reference = np.unique(train.windows(length))
        active = [index for index in active if np.isin(pieces[index].windows(length), reference).any()]
        for index in active:
            longest[index] = length
    return longest


def _finite_mean(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def memorization_report(pieces: dict[str, list[Note]], train_songs: list[list[Note]]) -> dict:
    report = {}
    for view in VIEWS:
        table = SymbolTable()
        train = WindowHashes([table.encode(notes, view) for notes in train_songs])
        piece_ids = list(pieces)
        encoded = [table.encode(pieces[piece_id], view) for piece_id in piece_ids]
        hashed = [WindowHashes([sequence]) for sequence in encoded]
        lengths = [len(full) for _, full in encoded]
        longest = longest_shared(hashed, train)
        shared = {}
        for size in NGRAM_SIZES:
            reference = np.unique(train.windows(size))
            shared[size] = [shared_fraction(piece, reference, size) for piece in hashed]
        per_piece = {
            piece_id: {
                "notes": lengths[index],
                "longest_shared_notes": longest[index],
                "exact_copy": longest[index] == lengths[index],
                **{f"shared_{size}gram_fraction": shared[size][index] for size in NGRAM_SIZES},
            }
            for index, piece_id in enumerate(piece_ids)
        }
        longest_ratio = np.array(longest) / np.maximum(lengths, 1)
        report[view] = {
            "pieces": len(piece_ids),
            "exact_copies": int(sum(row["exact_copy"] for row in per_piece.values())),
            "longest_shared_notes_mean": float(np.mean(longest)),
            "longest_shared_notes_median": float(np.median(longest)),
            "longest_shared_notes_max": int(np.max(longest)),
            "longest_shared_fraction_mean": float(longest_ratio.mean()),
            **{f"shared_{size}gram_fraction_mean": _finite_mean(shared[size]) for size in NGRAM_SIZES},
            "per_piece": per_piece,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True, help="training token corpus JSON")
    parser.add_argument("--pieces", type=Path, required=True, help="run folder or token corpus JSON to check")
    parser.add_argument("--bars", type=int, help="bars per excerpt; run folders supply their own")
    parser.add_argument("--sample", type=int, help="check a fixed random subset of a token corpus")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--corpora", type=Path, default=DEFAULT_CORPORA, help="token corpora for human run rows")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        train_corpus = read_json(args.train)
        if train_corpus["split"] != "train":
            raise ValueError(f"Expected the train corpus, got {train_corpus['split']}")
        if not args.pieces.is_dir() and read_json(args.pieces).get("split") == "train":
            raise ValueError("Checking training songs against themselves would only find exact copies")
        pieces, rejected, bars = load_excerpts(args.pieces, args.bars, args.sample, args.seed, args.corpora)
        train_songs = [decode_notes(tokens) for _, tokens in token_sequences(train_corpus)]
        report = {
            "train": str(args.train), "pieces": str(args.pieces), "bars": bars,
            "sample": args.sample, "seed": args.seed, "rejected": rejected,
            **memorization_report(pieces, train_songs),
        }
        write_json(report, args.output)
        print(json.dumps({view: {key: value for key, value in report[view].items() if key != "per_piece"}
                          for view in VIEWS}, indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")

if __name__ == "__main__":
    main()
