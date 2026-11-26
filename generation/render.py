"""Render excerpts to audio with one soundfont and one fixed process for every piece.

FluidSynth synthesizes each MIDI at 44.1 kHz. FFmpeg then normalizes the loudness to -16 LUFS
(EBU R128, single pass) and encodes a VBR MP3. Human and generated pieces go through exactly the
same commands, so differences in the audio come from the notes alone.
"""

import argparse
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from .artifacts import output_path
from .runs import run_rows


# MuseScore_General.sf3 v0.2 (MIT), from ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/
SOUNDFONT_SHA256 = "5b85b6c2c61d10b2b91cddd41efcce7b25cd31c8271d511c73afafbef20b6fa3"
SAMPLE_RATE = 44100
LOUDNESS = "loudnorm=I=-16:TP=-1.5:LRA=11"


def check_soundfont(path: Path) -> None:
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != SOUNDFONT_SHA256:
        raise ValueError(f"{path} is not the pinned soundfont (SHA-256 {digest})")


def render(midi_path: Path, soundfont: Path, output: Path) -> None:
    for tool in ("fluidsynth", "ffmpeg"):
        if shutil.which(tool) is None:
            raise OSError(f"{tool} is not installed")
    with TemporaryDirectory() as directory:
        wav = Path(directory) / "render.wav"
        subprocess.run(["fluidsynth", "-ni", "-q", "-r", str(SAMPLE_RATE), "-F", str(wav), str(soundfont),
                        str(midi_path)], check=True, capture_output=True)
        with output_path(output) as temporary:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-af", LOUDNESS, "-ar", str(SAMPLE_RATE),
                            "-codec:a", "libmp3lame", "-q:a", "2", "-f", "mp3", str(temporary)],
                           check=True, capture_output=True)


def select_rows(rows: list[dict], count: int) -> list[dict]:
    """Pick pieces by a fixed rule: the valid rows with the lowest seeds (human rows by ID)."""
    valid = [row for row in rows if row["status"] == "valid"]
    if len(valid) < count:
        raise ValueError(f"Only {len(valid)} valid pieces, fewer than {count}")
    return sorted(valid, key=lambda row: (int(row["seed"]) if row["seed"] else -1, row["gen_id"]))[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run folder with generations.csv and midi/")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--soundfont", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        check_soundfont(args.soundfont)
        for row in select_rows(run_rows(args.run), args.count):
            midi = args.out_dir / f"{row['gen_id']}.mid"
            with output_path(midi) as temporary:
                shutil.copyfile(args.run / row["midi_path"], temporary)
            render(midi, args.soundfont, midi.with_suffix(".mp3"))
            print(midi.with_suffix(".mp3"), flush=True)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")

if __name__ == "__main__":
    main()
