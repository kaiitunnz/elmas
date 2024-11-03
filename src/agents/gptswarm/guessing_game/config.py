from ...config import BaseClientConfig

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ClientConfig(BaseClientConfig):
    temperature: float = 1.0
    num_comps: Optional[int] = None
    max_tokens: Optional[int] = None

    def load_json(self, path: str) -> "ClientConfig":
        with open(path, "r") as f:
            config: Dict[str, Any] = json.load(f)[self.model]
        self.temperature = config.get("temperature", 1.0)
        self.num_comps = config.get("num_comps", None)
        self.max_tokens = config.get("max_tokens", None)
        return self
    
    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "temperature": self.temperature,
            "num_comps": self.num_comps,
            "max_tokens": self.max_tokens,
        }
