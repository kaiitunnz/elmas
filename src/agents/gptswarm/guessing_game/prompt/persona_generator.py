# -*- coding: utf-8 -*-
"""
Generate Persona with LLM

Adapted from https://github.dev/modelscope/agentscope.
"""
from ....config import BaseClientConfig

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List

import backoff
import numpy as np
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from tqdm import tqdm  # type: ignore

BEGIN_TAG = "[PERSONA BEGIN]"
END_TAG = "[PERSONA END]"

SYS_PROMPT = """
You are a role personality description assistant, you need to generate a complete role personality description based on the provided JSON. The generated description should follow the following format:

```
    [PERSONA BEGIN]
    - Name: Required (You must come up with a suitable name.)
    - Gender: Male/Female/I don't want to disclose
    - Age: xx years old/I don't want to disclose
    - Personality Description: A brief description of the role's personality
    [PERSONA END]
```
"""

USER_PROMPT = "Please generate a role persona based on the following JSON:\n"


@dataclass
class ClientConfig(BaseClientConfig):
    temperature: float = 1.0
    n: int = 1

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "n": self.n,
        }


class PersonaGenerator:
    def __init__(self, config: ClientConfig):
        self.config = config
        self.openai = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self.sys_prompt = {"role": "system", "content": SYS_PROMPT}

    def _extract_persona(self, content: str) -> str:
        if BEGIN_TAG in content and END_TAG in content:
            return content[
                content.find(BEGIN_TAG) + len(BEGIN_TAG) : content.find(END_TAG)
            ]
        else:
            raise ValueError("Invalid persona format")

    def _get_user_prompt(self, desc: Dict[str, Any]) -> str:
        return USER_PROMPT + json.dumps(desc, indent=2, ensure_ascii=False)

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def generate(self, desc: Dict[str, Any]) -> str:
        user_prompt = {"role": "user", "content": self._get_user_prompt(desc)}
        response: ChatCompletion = await self.openai.chat.completions.create(
            messages=[self.sys_prompt, user_prompt], **self.config.to_kwargs()  # type: ignore
        )
        content = response.choices[0].message.content
        assert content is not None
        persona = self._extract_persona(content)
        return persona


def generate_samples(config_path: str) -> List[Dict[str, str]]:
    """Generate samples based on the given config"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    total_num = config["total_num"]
    samples: List[Dict[str, str]] = [{} for _ in range(total_num)]
    for distribution in config["distributions"]:
        distribution_name = distribution["distribution_name"]
        categories = distribution["categories"]

        # Extract category names and percentages
        category_names = [category["category_name"] for category in categories]
        percentages = [category["percentage"] for category in categories]
        attributes = {
            category["category_name"]: category.get(
                "attributes",
                {distribution_name: category["category_name"]},
            )
            for category in categories
        }

        # Convert percentages to actual numbers of samples
        num_samples_per_category = (np.array(percentages) * total_num).astype(
            int,
        )

        # Adjust any rounding errors to ensure total_num samples
        while num_samples_per_category.sum() < total_num:
            diff = total_num - num_samples_per_category.sum()
            for i in range(diff):
                # Add one to the first category that needs more samples
                num_samples_per_category[i % len(num_samples_per_category)] += 1
        while num_samples_per_category.sum() > total_num:
            diff = num_samples_per_category.sum() - total_num
            for i in range(diff):
                # Subtract one from the first category that has more samples
                num_samples_per_category[i % len(num_samples_per_category)] -= 1

        # Create samples for current distribution
        category_samples = []
        for category, count in zip(category_names, num_samples_per_category):
            category_samples.extend([category] * count)

        # Shuffle to avoid ordering biases
        np.random.shuffle(category_samples)

        # Assign the generated samples to the overall sample list
        for i in range(total_num):
            samples[i].update(attributes[category_samples[i]])

    return samples


async def main(config_path: str, save_path: str) -> None:
    """The main function to generate persona"""
    samples = generate_samples(config_path)
    print(samples)
    generator = PersonaGenerator(ClientConfig())
    tasks: List[asyncio.Task] = [
        asyncio.create_task(generator.generate(sample)) for sample in samples
    ]
    results = await asyncio.gather(*tasks)
    with open(save_path, "w", encoding="utf-8") as f:
        for result in tqdm(results):
            f.write(
                json.dumps({"prompt": result}, ensure_ascii=False) + "\n",
            )


def parse_args() -> Any:
    """Parse args"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        "-c",
        type=str,
        help="path of the config file",
    )
    parser.add_argument(
        "--save-path",
        "-o",
        type=str,
        help="path of the output file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.config_path, args.save_path))
