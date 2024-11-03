from typing import Any, Dict, Iterable
import json


def read_jsonl(filename: str) -> Iterable[Dict[str, Any]]:
    """Read a JSONL file."""
    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            yield json.loads(line)
