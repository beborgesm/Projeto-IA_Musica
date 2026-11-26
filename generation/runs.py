"""Load excerpts as grid-step notes from a sampling run folder or from a token corpus.

A run folder is what ``generation.sampling`` and ``generation reference`` write: a
``generations.csv`` and, for generated runs, the raw sequences in ``tokens.json``. Only rows whose
status is ``valid`` are used. Human rows carry no sequences; they are read back from the token
corpus of their split, which is what ``generation reference`` cut them from. Every excerpt goes
through ``excerpts.classify``, the same rule the validity report uses, and notes that ring past
the last bar are clipped at its end so all excerpts span exactly the same length.
"""

import csv
from pathlib import Path
import random

from .artifacts import read_json
from .excerpts import classify, parse_notes
from .tokenizer import BAR_STEPS, Note, token_sequences


DEFAULT_CORPORA = Path("generation/tokens")


def clip(notes: list[Note], bars: int) -> list[Note]:
    end = bars * BAR_STEPS
    return [Note(note.pitch, note.start, min(note.end, end), note.velocity) for note in notes]


def excerpt(tokens: list[int], bars: int) -> list[Note] | None:
    """Return the clipped notes of a valid excerpt, or None for any other status."""
    result = classify(tokens, bars)
    return clip(parse_notes(result["excerpt"]), bars) if result["status"] == "valid" else None


def run_rows(folder: Path) -> list[dict]:
    with (folder / "generations.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{folder} has no generations")
    return rows


def run_bars(rows: list[dict]) -> int:
    bars = {int(row["bars"]) for row in rows}
    if len(bars) != 1:
        raise ValueError(f"A run must use one bar count, found {sorted(bars)}")
    return bars.pop()


def run_excerpts(folder: Path, corpora: Path = DEFAULT_CORPORA) -> tuple[dict[str, list[Note]], int]:
    rows = run_rows(folder)
    bars = run_bars(rows)
    sequences = read_json(folder / "tokens.json")["sequences"] if (folder / "tokens.json").exists() else {}
    songs = {}
    pieces = {}
    for row in rows:
        if row["status"] != "valid":
            continue
        if row["system"] == "human":
            if row["split"] not in songs:
                corpus = read_json(corpora / f"{row['split']}.json")
                songs[row["split"]] = dict(token_sequences(corpus))
            tokens = songs[row["split"]][row["source_song"]]
        else:
            tokens = sequences[row["gen_id"]]
        notes = excerpt(tokens, bars)
        if notes is None:
            raise ValueError(f"{row['gen_id']} is marked valid but does not give a valid {bars}-bar excerpt")
        pieces[row["gen_id"]] = notes
    if not pieces:
        raise ValueError(f"{folder} has no valid excerpts")
    return pieces, bars


def corpus_excerpts(corpus: dict, bars: int, sample: int | None = None, seed: int = 0) -> tuple[dict, list[str]]:
    """Return the valid excerpts of a token corpus and the IDs that gave none.

    With ``sample``, a fixed-seed random subset of the songs is drawn before the check.
    """
    songs = list(token_sequences(corpus))
    if sample is not None and sample < len(songs):
        songs = random.Random(seed).sample(songs, sample)
    pieces, rejected = {}, []
    for song_id, tokens in songs:
        notes = excerpt(tokens, bars)
        if notes is None:
            rejected.append(song_id)
        else:
            pieces[song_id] = notes
    return pieces, rejected


def load_excerpts(
    path: Path, bars: int | None = None, sample: int | None = None, seed: int = 0, corpora: Path = DEFAULT_CORPORA
) -> tuple[dict[str, list[Note]], list[str], int]:
    """Load a run folder or a token corpus JSON; return the excerpts, rejected IDs and bar count."""
    if path.is_dir():
        if sample is not None:
            raise ValueError("--sample applies to token corpora, not to run folders")
        pieces, run_bar_count = run_excerpts(path, corpora)
        if bars is not None and bars != run_bar_count:
            raise ValueError(f"{path} holds {run_bar_count}-bar excerpts, not {bars}")
        return pieces, [], run_bar_count
    if bars is None:
        raise ValueError(f"Give --bars to cut excerpts from the corpus {path}")
    pieces, rejected = corpus_excerpts(read_json(path), bars, sample, seed)
    return pieces, rejected, bars
