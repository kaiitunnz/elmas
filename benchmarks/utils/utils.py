import asyncio
import json
from collections.abc import Coroutine
from typing import Any, Callable


def read_jsonl(filename: str):
    """Read a JSONL file."""
    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            yield json.loads(line)


def run_coroutine(coroutine: Callable[..., Coroutine], *args) -> Any:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coroutine(*args))
    finally:
        loop.close()
