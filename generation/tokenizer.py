"""Merge POP909 tracks into one piano, quantize it, and encode it as REMI tokens."""

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from typing import TypedDict

import mido


FORMAT_VERSION = 2
STEPS_PER_QUARTER = 12
BAR_STEPS = 4 * STEPS_PER_QUARTER
MAX_TOKEN_STEPS = 48
SOURCE_TRACKS = ("MELODY", "BRIDGE", "PIANO")
VOCABULARY = (
    "BOS",
    "EOS",
    "Bar",
    *(f"Position_{step}" for step in range(BAR_STEPS)),
    *(f"Pitch_{pitch}" for pitch in range(128)),
    *(f"Duration_{step}" for step in range(1, MAX_TOKEN_STEPS + 1)),
    *(f"Velocity_{velocity}" for velocity in range(1, 128)),
)
TOKEN_IDS = {event: index for index, event in enumerate(VOCABULARY)}


class Sequence(TypedDict):
    format_version: int
    ticks_per_beat: int
    origin_tick: int
    tempo_events: list[tuple[int, int]]
    tokens: list[int]


@dataclass(frozen=True)
class Note:
    """A note with start and end in ticks or in grid steps."""

    pitch: int
    start: int
    end: int
    velocity: int


def token_sequences(corpus: dict):
    """Yield the song IDs and sequences from a compatible token corpus."""
    if corpus["format_version"] != FORMAT_VERSION or tuple(corpus["vocabulary"]) != VOCABULARY:
        raise ValueError("The corpus uses an incompatible token format or vocabulary")
    if not corpus["songs"]:
        raise ValueError(f"The {corpus['split']} corpus contains no songs")
    bos, eos = TOKEN_IDS["BOS"], TOKEN_IDS["EOS"]
    for song in corpus["songs"]:
        tokens = song["tokens"]
        if (
            len(tokens) < 2
            or tokens[0] != bos
            or tokens[-1] != eos
            or any(type(token) is not int or not 0 <= token < len(VOCABULARY) for token in tokens)
            or any(token in (bos, eos) for token in tokens[1:-1])
        ):
            raise ValueError(f"Invalid token sequence for song {song['song_id']}")
        yield song["song_id"], tokens


def _absolute_messages(track):
    tick = 0
    for message in track:
        tick += message.time
        yield tick, message


def _track_notes(track) -> list[Note]:
    """Pair note events within one track, using FIFO for overlaps."""
    active = defaultdict(deque)
    notes = []
    for tick, message in _absolute_messages(track):
        if message.type not in ("note_on", "note_off"):
            continue
        key = (message.channel, message.note)
        if message.type == "note_on" and message.velocity > 0:
            active[key].append((tick, message.velocity))
            continue
        if not active[key]:
            raise ValueError(f"Unmatched note-off for pitch {message.note} at tick {tick}")
        start, velocity = active[key].popleft()
        notes.append(Note(message.note, start, tick, velocity))
    if any(active.values()):
        raise ValueError(f"Track {track.name} contains unmatched note-ons")
    return notes


def read_track_notes(midi: mido.MidiFile, names: tuple[str, ...]) -> list[Note]:
    """Concatenate the notes of the named tracks, each of which must appear exactly once."""
    notes = []
    for name in names:
        tracks = [track for track in midi.tracks if track.name == name]
        if len(tracks) != 1:
            raise ValueError(f"Expected exactly one track named {name}")
        notes.extend(_track_notes(tracks[0]))
    return sorted(notes, key=lambda note: (note.start, note.pitch))


def read_piano_notes(midi: mido.MidiFile) -> list[Note]:
    return read_track_notes(midi, ("PIANO",))


def read_tempo_events(midi: mido.MidiFile) -> list[tuple[int, int]]:
    events = [
        (tick, message.tempo)
        for track in midi.tracks
        for tick, message in _absolute_messages(track)
        if message.type == "set_tempo"
    ]
    return sorted(events, key=lambda event: event[0])


def seconds_to_ticks(tempo_events: list[tuple[int, int]], ticks_per_beat: int, seconds: float) -> float:
    """Map a time in seconds to a fractional tick under a tempo map (120 BPM before any event)."""
    tick, tempo, elapsed = 0, 500000, 0.0
    for event_tick, event_tempo in tempo_events:
        event_seconds = elapsed + (event_tick - tick) * tempo / 1e6 / ticks_per_beat
        if event_seconds > seconds:
            break
        tick, tempo, elapsed = event_tick, event_tempo, event_seconds
    return tick + (seconds - elapsed) * 1e6 / tempo * ticks_per_beat


def ticks_to_seconds(tempo_events: list[tuple[int, int]], ticks_per_beat: int, target: int) -> float:
    tick, tempo, elapsed = 0, 500000, 0.0
    for event_tick, event_tempo in tempo_events:
        if event_tick > target:
            break
        elapsed += (event_tick - tick) * tempo / 1e6 / ticks_per_beat
        tick, tempo = event_tick, event_tempo
    return elapsed + (target - tick) * tempo / 1e6 / ticks_per_beat


def grid_step_ticks(ticks_per_beat: int) -> int:
    if (
        type(ticks_per_beat) is not int
        or not 0 < ticks_per_beat <= 32767
        or ticks_per_beat % STEPS_PER_QUARTER
    ):
        raise ValueError("Expected a positive ticks-per-quarter value divisible by 12")
    return ticks_per_beat // STEPS_PER_QUARTER


def quantize_notes(notes: list[Note], step_ticks: int, origin: int = 0) -> tuple[list[Note], Counter]:
    """Snap ticks to the grid that starts at origin and resolve repeated notes of the same pitch."""
    stats = Counter(input_notes=len(notes))
    onsets = {}
    for note in notes:
        start = (note.start - origin + step_ticks // 2) // step_ticks
        end = (note.end - origin + step_ticks // 2) // step_ticks
        if start < 0:
            raise ValueError(f"A note at tick {note.start} falls before the grid origin")
        stats["onset_error_steps_sum"] += abs(origin + start * step_ticks - note.start) / step_ticks
        stats["end_error_steps_sum"] += abs(origin + end * step_ticks - note.end) / step_ticks
        if end <= start:
            end = start + 1
            stats["short_notes_extended"] += 1

        key = (note.pitch, start)
        previous = onsets.get(key)
        velocity = note.velocity
        if previous is not None:
            end = max(end, previous.end)
            velocity = max(velocity, previous.velocity)
            stats["same_onset_notes_merged"] += 1
        onsets[key] = Note(note.pitch, start, end, velocity)

    next_onset = {}
    normalized = []
    for note in sorted(onsets.values(), key=lambda note: note.start, reverse=True):
        end = min(note.end, next_onset.get(note.pitch, note.end))
        if end < note.end:
            stats["overlapping_notes_shortened"] += 1
        normalized.append(replace(note, end=end))
        next_onset[note.pitch] = note.start
    return sorted(normalized, key=lambda note: (note.start, note.pitch)), stats


def _append_steps(tokens: list[int], event: str, steps: int) -> None:
    full_chunks, remainder = divmod(steps, MAX_TOKEN_STEPS)
    tokens.extend([TOKEN_IDS[f"{event}_{MAX_TOKEN_STEPS}"]] * full_chunks)
    if remainder:
        tokens.append(TOKEN_IDS[f"{event}_{remainder}"])


def encode_notes(notes: list[Note]) -> list[int]:
    """Encode normalized grid-step notes, sorted by onset then pitch, as a BOS…EOS REMI sequence."""
    tokens = [TOKEN_IDS["BOS"]]
    bar = -1
    position = None
    for note in notes:
        note_bar, note_position = divmod(note.start, BAR_STEPS)
        while bar < note_bar:
            tokens.append(TOKEN_IDS["Bar"])
            bar += 1
            position = None
        if note_position != position:
            tokens.append(TOKEN_IDS[f"Position_{note_position}"])
            position = note_position
        tokens.append(TOKEN_IDS[f"Pitch_{note.pitch}"])
        _append_steps(tokens, "Duration", note.end - note.start)
        tokens.append(TOKEN_IDS[f"Velocity_{note.velocity}"])
    tokens.append(TOKEN_IDS["EOS"])
    return tokens


def encode_midi(
    midi: mido.MidiFile, downbeat_seconds: float | None = None
) -> tuple[Sequence, list[Note], Counter]:
    """Return the sequence, its normalized reference notes, and quantization counts.

    Bars are a fixed four-quarter grid anchored at the annotated downbeat (tick zero when absent)
    and moved back by whole bars so the first bar holds the first note.
    """
    if midi.type not in (0, 1):
        raise ValueError("Expected a synchronous type 0 or type 1 MIDI file")
    step_ticks = grid_step_ticks(midi.ticks_per_beat)
    notes = read_track_notes(midi, SOURCE_TRACKS)
    if not notes:
        raise ValueError("The source tracks contain no notes")
    tempo_events = read_tempo_events(midi)
    anchor = 0
    if downbeat_seconds is not None:
        anchor = round(seconds_to_ticks(tempo_events, midi.ticks_per_beat, downbeat_seconds))
    bar_ticks = BAR_STEPS * step_ticks
    origin = anchor + (notes[0].start - anchor) // bar_ticks * bar_ticks
    notes, stats = quantize_notes(notes, step_ticks, origin)
    tokens = encode_notes(notes)
    return {
        "format_version": FORMAT_VERSION,
        "ticks_per_beat": midi.ticks_per_beat,
        "origin_tick": origin,
        "tempo_events": tempo_events,
        "tokens": tokens,
    }, notes, stats


def decode_notes(tokens: list[int]) -> list[Note]:
    if any(type(token) is not int or not 0 <= token < len(VOCABULARY) for token in tokens):
        raise ValueError("The sequence contains an invalid token ID")
    if len(tokens) < 5 or tokens[0] != TOKEN_IDS["BOS"] or tokens[-1] != TOKEN_IDS["EOS"]:
        raise ValueError("Expected a nonempty note sequence delimited by BOS and EOS")
    events = [VOCABULARY[token].split("_", 1) for token in tokens[1:-1]]
    notes = []
    ends = {}
    bar = -1
    position = -1
    onset = None
    index = 0
    while index < len(events):
        if events[index][0] == "Bar":
            bar += 1
            position = -1
            onset = None
            index += 1
            continue
        if events[index][0] == "Position":
            if bar < 0 or int(events[index][1]) <= position:
                raise ValueError(f"Expected increasing positions inside a bar at body token {index}")
            position = int(events[index][1])
            onset = bar * BAR_STEPS + position
            index += 1
            continue
        if events[index][0] != "Pitch" or onset is None:
            raise ValueError(f"Expected a bar, a position, or a pitch after a position at body token {index}")
        pitch = int(events[index][1])
        index += 1
        duration = 0
        while index < len(events) and events[index][0] == "Duration":
            duration += int(events[index][1])
            index += 1
        if not duration or index == len(events) or events[index][0] != "Velocity":
            raise ValueError("Expected a positive duration and a velocity after each pitch")
        if onset < ends.get(pitch, 0):
            raise ValueError(f"Overlapping notes for pitch {pitch} at step {onset}")
        end = onset + duration
        notes.append(Note(pitch, onset, end, int(events[index][1])))
        ends[pitch] = end
        index += 1
    if events[-1][0] != "Velocity":
        raise ValueError("Expected a note after the final bar or position")
    return notes


def decode_midi(sequence: Sequence) -> mido.MidiFile:
    """Build a MIDI with a conductor track and one program-0 piano track."""
    if sequence["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported token format version")
    ticks_per_beat = sequence["ticks_per_beat"]
    step_ticks = grid_step_ticks(ticks_per_beat)
    origin = sequence["origin_tick"]
    if type(origin) is not int:
        raise ValueError("Expected an integer origin tick")
    notes = decode_notes(sequence["tokens"])
    midi = mido.MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    conductor = midi.add_track()
    last_tick = 0
    for tick, tempo in sequence["tempo_events"]:
        if type(tick) is not int or tick < last_tick or type(tempo) is not int or tempo <= 0:
            raise ValueError("Expected ordered tempo events with integer ticks and positive tempos")
        conductor.append(mido.MetaMessage("set_tempo", tempo=tempo, time=tick - last_tick))
        last_tick = tick

    piano = midi.add_track("PIANO")
    piano.append(mido.Message("program_change", program=0))
    events = []
    for note in notes:
        events.append((note.start, 1, note.pitch, note.velocity))
        events.append((note.end, 0, note.pitch, 0))
    last_tick = 0
    for step, is_onset, pitch, velocity in sorted(events):
        tick = origin + step * step_ticks
        if tick < 0:
            raise ValueError("A decoded note falls before tick zero")
        kind = "note_on" if is_onset else "note_off"
        piano.append(mido.Message(kind, note=pitch, velocity=velocity, time=tick - last_tick))
        last_tick = tick
    return midi
