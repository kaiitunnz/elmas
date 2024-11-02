import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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
class VLLMConfigBase(BenchmarkConfigBase):
    benchmark_root: Path = Path.cwd() / "benchmarks/tasks"

    disabled_pbar: bool = False
    selected_percentile_metrics: List[str] = field(
        default_factory=lambda: ["ttft", "tpot", "itl"]
    )
    selected_percentiles: List[float] = field(default_factory=lambda: [99])
