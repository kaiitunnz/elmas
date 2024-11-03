from ..config import BaseProfilingConfig

from dataclasses import dataclass
from typing import Any, Dict

from openai import OpenAI
from openai.types import Completion

from ..utils.vllm.profiling import VLLMProfiling


@dataclass
class ProfilingConfig(BaseProfilingConfig):
    max_tokens: int = 2
    n: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.8
    echo: bool = True

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "echo": self.echo,
        }


class OpenAIClient:
    def __init__(self, client_config: ProfilingConfig):
        self.config = client_config
        self.openai = OpenAI(
            api_key=client_config.api_key, base_url=client_config.base_url
        )

    def generate(self, prompt: str) -> str:
        response: Completion = self.openai.completions.create(
            prompt=prompt, **self.config.to_kwargs()
        )
        return response.choices[0].text


def main():
    prompts = [
        " Hello Hello Hello Hello Hello Hello Hello Hello",
        " Hello Hello Hello Hello Hello Hello Hello Hello",
        " Hi Hi Hi Hi Hi Hi Hi Hi",
        " Hi Hi Hi Hi Hi Hi Hi Hi",
        " Dude Dude Dude Dude Dude Dude Dude Dude",
        " Dude Dude Dude Dude Dude Dude Dude Dude",
        " What What What What What What What What",
        " Hello Hello Hello Hello Hello Hello Hello Hello",
        " Hi Hi Hi Hi Hi Hi Hi Hi",
    ]

    config = ProfilingConfig()
    client = OpenAIClient(config)
    profiling = VLLMProfiling(config)

    profiling.start()
    for prompt in prompts:
        print(client.generate(prompt))
    profiling.stop()


if __name__ == "__main__":
    main()
