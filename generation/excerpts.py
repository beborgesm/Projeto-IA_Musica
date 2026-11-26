"""Cut token sequences to a fixed number of bars and write excerpts with their metadata.

The same rules apply to human and generated sequences:
- an excerpt holds the first ``bars`` bars of a sequence, starting at its first ``Bar`` token;
- trailing empty bars are dropped before decoding, since a decodable body ends with a note;
- the token grammar is checked strictly: a malformed excerpt is ``invalid``;
- notes of one pitch that collide are resolved with the tokenizer's normalization rule, the one
  already applied to the human data: notes on the same onset merge, and a note still sounding
  when the same pitch starts again ends at that onset. Both repairs are counted per excerpt.

Human references are the opening bars of the test songs, matching unconditional generation,
which also starts at the beginning of a piece.
"""

import csv
from hashlib import sha256
from pathlib import Path
import random

from .artifacts import output_path, write_json
from .tokenizer import (
    BAR_STEPS,
    FORMAT_VERSION,
    TOKEN_IDS,
    VOCABULARY,
    Note,
    decode_midi,
    encode_notes,
    quantize_notes,
    token_sequences,
)


BOS = TOKEN_IDS["BOS"]
EOS = TOKEN_IDS["EOS"]
BAR = TOKEN_IDS["Bar"]
TICKS_PER_BEAT = 480
STATUSES = ("valid", "incomplete", "empty", "invalid")
COLUMNS = (
    "gen_id",
    "system",
    "status",
    "midi_path",
    "bars",
    "n_tokens",
    "excerpt_tokens",
    "n_notes",
    "overlaps_resolved",
    "same_onset_merged",
    "constrained",
    "temperature",
    "top_k",
    "top_p",
    "seed",
    "checkpoint",
    "checkpoint_sha256",
    "tempo_us",
    "source_song",
    "split",
    "error",
)


def cut_bars(tokens: list[int], bars: int) -> tuple[list[int], int]:
    """Return the first ``bars`` bars as a BOS…EOS sequence and how many bars were found.

    Reading stops at EOS, at the end of the list, or at the Bar token that would open bar
    ``bars + 1``. Trailing empty bars are dropped from the excerpt but still counted. A list
    that ends without either stop was cut by the token limit, so a trailing partial note (any
    tokens after the last Velocity or Bar) is dropped too.
    """
    if bars < 1:
        raise ValueError("Use at least one bar")
    if not tokens or tokens[0] != BOS:
        raise ValueError("A sequence starts with BOS")
    body = []
    found = 0
    closed = False
    for token in tokens[1:]:
        if token == EOS:
            closed = True
            break
        if token == BAR:
            if found == bars:
                closed = True
                break
            found += 1
        body.append(token)
    if not closed:
        while body and VOCABULARY[body[-1]] != "Bar" and not VOCABULARY[body[-1]].startswith("Velocity_"):
            body.pop()
    while body and body[-1] == BAR:
        body.pop()
    return [BOS, *body, EOS], found


def parse_notes(tokens: list[int]) -> list[Note]:
    """Read grid-step notes from a BOS…EOS sequence, checking the grammar but not collisions.

    The grammar is the decoder's: a Position needs an open bar and grows inside it, a Pitch needs
    a Position, and each Pitch takes one or more Durations and then a Velocity.
    """
    events = [VOCABULARY[token].split("_", 1) for token in tokens[1:-1]]
    if any(kind in ("BOS", "EOS") for kind, *_ in events):
        raise ValueError("BOS or EOS inside the sequence")
    notes = []
    bar, position, onset, index = -1, -1, None, 0
    while index < len(events):
        kind = events[index][0]
        if kind == "Bar":
            bar, position, onset = bar + 1, -1, None
            index += 1
        elif kind == "Position":
            value = int(events[index][1])
            if bar < 0 or value <= position:
                raise ValueError(f"Expected increasing positions inside a bar at body token {index}")
            position, onset = value, bar * BAR_STEPS + value
            index += 1
        elif kind == "Pitch" and onset is not None:
            pitch = int(events[index][1])
            index += 1
            duration = 0
            while index < len(events) and events[index][0] == "Duration":
                duration += int(events[index][1])
                index += 1
            if not duration or index == len(events) or events[index][0] != "Velocity":
                raise ValueError("Expected a positive duration and a velocity after each pitch")
            notes.append(Note(pitch, onset, onset + duration, int(events[index][1])))
            index += 1
        else:
            raise ValueError(f"Expected a bar, a position, or a pitch after a position at body token {index}")
    return notes


def classify(tokens: list[int], bars: int) -> dict:
    """Cut ``tokens`` to ``bars`` bars, check it and return its status and normalized excerpt.

    - ``invalid``: the excerpt breaks the token grammar;
    - ``empty``: the excerpt holds no note;
    - ``incomplete``: the sequence ends (EOS or token limit) before ``bars`` bars;
    - ``valid``: otherwise.
    Only the excerpt is checked, so tokens after the last kept bar do not matter. The returned
    ``excerpt`` is the normalized sequence for valid and incomplete excerpts.
    """
    excerpt, found = cut_bars(tokens, bars)
    result = {"status": "", "excerpt": excerpt, "error": "", "notes": 0, "overlaps": 0, "merged": 0}
    try:
        notes = parse_notes(excerpt)
    except ValueError as error:
        return {**result, "status": "invalid", "error": str(error)}
    if not notes:
        return {**result, "status": "empty"}
    normalized, stats = quantize_notes(notes, step_ticks=1)
    return {
        **result,
        "status": "incomplete" if found < bars else "valid",
        "excerpt": encode_notes(normalized),
        "notes": len(normalized),
        "overlaps": stats["overlapping_notes_shortened"],
        "merged": stats["same_onset_notes_merged"],
    }


def write_midi(path: Path, tokens: list[int], ticks_per_beat: int, tempo_events: list, origin_tick: int) -> int:
    """Decode ``tokens`` to a MIDI file and return its note count."""
    sequence = {
        "format_version": FORMAT_VERSION,
        "ticks_per_beat": ticks_per_beat,
        "tempo_events": tempo_events,
        "origin_tick": origin_tick,
        "tokens": tokens,
    }
    midi = decode_midi(sequence)
    with output_path(path) as temporary:
        midi.save(temporary)
    return len(parse_notes(tokens))


def training_tempo(train_corpus: dict, seed: int) -> tuple[str, int]:
    """Pick a training song with ``seed`` and return its ID and initial tempo (µs per quarter)."""
    if train_corpus["split"] != "train":
        raise ValueError(f"Expected the train corpus, got {train_corpus['split']}")
    songs = sorted(train_corpus["songs"], key=lambda song: song["song_id"])
    song = random.Random(seed).choice(songs)
    return song["song_id"], song["tempo_events"][0][1]


def checkpoint_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def record_generation(
    output: Path,
    gen_id: str,
    system: str,
    tokens: list[int],
    bars: int,
    *,
    seed: int,
    temperature: float,
    top_k: int,
    top_p: float,
    checkpoint: Path,
    checkpoint_sha256: str,
    tempo_us: int,
    constrained: bool = False,
) -> dict:
    """Classify one sampled sequence, write its excerpt as MIDI when it decodes, and return its row.

    ``tokens`` is the raw sampled sequence starting with BOS; it may be invalid or truncated.
    """
    result = classify(tokens, bars)
    midi_path = ""
    if result["status"] in ("valid", "incomplete"):
        midi_path = f"midi/{gen_id}.mid"
        write_midi(output / midi_path, result["excerpt"], TICKS_PER_BEAT, [[0, tempo_us]], 0)
    return {
        "gen_id": gen_id,
        "system": system,
        "status": result["status"],
        "midi_path": midi_path,
        "bars": bars,
        "n_tokens": len(tokens),
        "excerpt_tokens": len(result["excerpt"]),
        "n_notes": result["notes"],
        "overlaps_resolved": result["overlaps"],
        "same_onset_merged": result["merged"],
        "constrained": constrained,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seed": seed,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha256,
        "tempo_us": tempo_us,
        "source_song": "",
        "split": "",
        "error": result["error"],
    }


def human_reference(corpus: dict, bars: int, output: Path) -> list[dict]:
    """Write the first ``bars`` bars of every song in ``corpus`` and return their rows."""
    rows = []
    songs = {song["song_id"]: song for song in corpus["songs"]}
    for song_id, tokens in sorted(token_sequences(corpus)):
        song = songs[song_id]
        result = classify(tokens, bars)
        excerpt = result["excerpt"]
        if result["status"] != "valid":
            raise ValueError(f"Song {song_id} gives an {result['status']} {bars}-bar excerpt {result['error']}".strip())
        if excerpt != cut_bars(tokens, bars)[0]:
            raise ValueError(f"Song {song_id} changed under normalization; human tokens must already be normalized")
        gen_id = f"human-{corpus['split']}-{song_id}-b{bars}"
        midi_path = f"midi/{gen_id}.mid"
        notes = write_midi(
            output / midi_path, excerpt, song["ticks_per_beat"], song["tempo_events"], song["origin_tick"]
        )
        rows.append(
            {
                **{column: "" for column in COLUMNS},
                "gen_id": gen_id,
                "system": "human",
                "status": "valid",
                "midi_path": midi_path,
                "bars": bars,
                "n_tokens": len(tokens),
                "excerpt_tokens": len(excerpt),
                "n_notes": notes,
                "overlaps_resolved": 0,
                "same_onset_merged": 0,
                "tempo_us": song["tempo_events"][0][1],
                "source_song": song_id,
                "split": corpus["split"],
            }
        )
    return rows


def write_table(rows: list[dict], output: Path, sequences: dict[str, list[int]] | None = None) -> None:
    """Write ``generations.csv`` and, when given, the raw sequences to ``tokens.json``."""
    for row in rows:
        if row["status"] not in STATUSES:
            raise ValueError(f"Unknown status {row['status']!r} for {row['gen_id']}")
    if len({row["gen_id"] for row in rows}) != len(rows):
        raise ValueError("gen_id values must be unique")
    table = output / "generations.csv"
    with output_path(table) as temporary, temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    if sequences is not None:
        write_json({"format_version": FORMAT_VERSION, "vocabulary": VOCABULARY, "sequences": sequences},
                   output / "tokens.json")
