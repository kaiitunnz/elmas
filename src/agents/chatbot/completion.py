from dataclasses import dataclass
from typing import Any, Dict

from openai import OpenAI
from openai.types import Completion

from config import BaseClientConfig


@dataclass
class ClientConfig(BaseClientConfig):
    max_tokens: int = 300
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
            "echo": self.echo
        }


class OpenAIClient:
    def __init__(self, client_config: ClientConfig):
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
    client = OpenAIClient(ClientConfig())

    while True:
        prompt = input("Prompt: ")
        print(client.generate(prompt))


if __name__ == "__main__":
    main()