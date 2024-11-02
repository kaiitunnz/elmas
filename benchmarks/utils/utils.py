import asyncio
import json
import random
from collections.abc import Coroutine
from typing import Any, Callable, Optional

import numpy as np


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


def set_seed(seed: Optional[int] = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
