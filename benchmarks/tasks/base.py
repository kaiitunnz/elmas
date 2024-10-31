import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BenchmarkConfigBase:
    result_file: Optional[Path] = None

    def __post_init__(self):
        self.result_file = self.result_file or Path(os.devnull)

    @property
    def result_file_str(self) -> str:
        assert self.result_file is not None
        return str(self.result_file.absolute())


@dataclass
class SGLangConfigBase(BenchmarkConfigBase):
    benchmark_root: Path = Path.home() / "Workspace/Projects/sglang/benchmark"
