from dataclasses import dataclass
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import BaseClientConfig


@dataclass
class ClientConfig(BaseClientConfig):
    temperature: float = 0.7
    n: int = 1
    top_p: float = 1.0
    max_tokens: int = 300

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "n": self.n,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }


def main():
    conversation = [
        SystemMessage("You are a helpful assistant."),
    ]

    client_config = ClientConfig()
    client = ChatOpenAI(**client_config.to_kwargs())

    conversation = []

    while True:
        user_input = input("User: ")
        conversation.append(HumanMessage(user_input))
        response = client.invoke(conversation)
        conversation.append(response)
        print("\nBot:", response.content, "\n")


if __name__ == "__main__":
    main()
