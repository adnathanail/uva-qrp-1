import json
import os
from pathlib import Path


def serialize_key(x: tuple[int, ...]) -> str:
    return json.dumps(list(x))


def atomic_write(path: Path, content: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(content)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
