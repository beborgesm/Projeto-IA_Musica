"""Read and atomically write the generation pipeline's local artifacts."""

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


@contextmanager
def output_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(value, path: Path) -> None:
    with output_path(path) as temporary, temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, separators=(",", ":"))
        handle.write("\n")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
