"""Read POP909 annotations and build the song-level split of duple-meter songs."""

import csv
from pathlib import Path
import random

import mido
import openpyxl

from .artifacts import output_path
from .tokenizer import STEPS_PER_QUARTER, read_tempo_events, seconds_to_ticks


SPLIT_SEED = 42
SPLIT_SIZES = (("train", 727), ("val", 91), ("test", 91))
DUPLE_METER = ("2", "2")  # index.xlsx: beats per measure, quavers per beat


def read_beats(path: Path) -> list[tuple[float, bool]]:
    """Return each annotated beat's time and whether it starts a four-beat bar."""
    beats = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            seconds, _, downbeat = line.split()
            beats.append((float(seconds), float(downbeat) == 1))
    if not beats:
        raise ValueError(f"No beats in {path}")
    return beats


def first_downbeat_seconds(path: Path) -> float:
    for seconds, downbeat in read_beats(path):
        if downbeat:
            return seconds
    raise ValueError(f"No annotated downbeat in {path}")


def read_key(path: Path) -> str:
    """Return the first key in key_audio.txt, such as Gb:maj."""
    lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"No key in {path}")
    return lines[0][-1]


def beat_grid_deviation(midi: mido.MidiFile, beats: list[tuple[float, bool]]) -> float:
    """Largest gap between consecutive annotated beats and one quarter note, in quarters."""
    tempo_events = read_tempo_events(midi)
    quarters = [seconds_to_ticks(tempo_events, midi.ticks_per_beat, seconds) / midi.ticks_per_beat
                for seconds, _ in beats]
    return max((abs(b - a - 1) for a, b in zip(quarters, quarters[1:])), default=0.0)


def exclusion_reason(meter: tuple[str, str], midi: mido.MidiFile, beats: list[tuple[float, bool]]) -> str:
    """Explain why a song leaves the corpus, or return an empty string to keep it."""
    if meter != DUPLE_METER:
        return f"meter {meter[0]}x{meter[1]} in index.xlsx"
    if not any(downbeat for _, downbeat in beats):
        return "no annotated downbeat"
    deviation = beat_grid_deviation(midi, beats)
    if deviation > 1 / STEPS_PER_QUARTER:
        return f"annotated beats drift from the quarter-note grid by {deviation:.3f} quarters"
    return ""


def read_meters(dataset_root: Path) -> dict[str, tuple[str, str]]:
    sheet = openpyxl.load_workbook(dataset_root / "index.xlsx", read_only=True).active
    rows = sheet.iter_rows(values_only=True)
    header = next(rows)
    song, beats, quavers = (header.index(name) for name in
                            ("song_id", "num_beats_per_measure", "num_quavers_per_beat"))
    return {str(row[song]): (str(row[beats]), str(row[quavers])) for row in rows if row[song]}


def assign_splits(song_ids: list[str]) -> dict[str, str]:
    """Shuffle the sorted IDs with a fixed seed and cut them into train, val, and test."""
    if len(song_ids) != sum(size for _, size in SPLIT_SIZES):
        raise ValueError(f"Expected {sum(size for _, size in SPLIT_SIZES)} songs, got {len(song_ids)}")
    order = sorted(song_ids)
    random.Random(SPLIT_SEED).shuffle(order)
    splits = {}
    start = 0
    for split, size in SPLIT_SIZES:
        splits.update({song_id: split for song_id in order[start : start + size]})
        start += size
    return splits


def build_split(dataset_root: Path) -> tuple[list[dict], list[dict]]:
    """Assign all songs to splits first, then drop the excluded ones so the filter cannot leak."""
    meters = read_meters(dataset_root)
    splits = assign_splits(list(meters))
    kept, excluded = [], []
    for song_id in sorted(meters):
        midi_path = f"{song_id}/{song_id}.mid"
        beats = read_beats(dataset_root / song_id / "beat_midi.txt")
        reason = exclusion_reason(meters[song_id], mido.MidiFile(dataset_root / midi_path), beats)
        row = {"song_id": song_id, "split": splits[song_id]}
        if reason:
            excluded.append({**row, "reason": reason})
        else:
            kept.append({**row, "midi_path": midi_path})
    return kept, excluded


def write_csv(rows: list[dict], fields: list[str], path: Path) -> None:
    with output_path(path) as temporary, temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
