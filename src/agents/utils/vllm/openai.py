from typing import List

import requests
from prometheus_client import Metric, parser

from ...config import BaseClientConfig


def request_openai_chat_completions(
    config: BaseClientConfig,
    endpoint: str,
    prompt: str,
    max_tokens: int = 10,
    ignore_eos: bool = True,
) -> str:
    # Adapted from https://github.com/vllm-project/vllm/blob/8c746226c956f7c8a4672689fee91c7d22befed6/benchmarks/backend_request_func.py
    api_url = f"{config.base_url.rstrip('/v1')}/{endpoint}"
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    payload = {
        "model": config.model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "ignore_eos": ignore_eos,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    response = requests.post(api_url, data=payload, headers=headers)
    response.raise_for_status()
    return response.content.decode("utf-8")


def request_openai_health(config: BaseClientConfig) -> requests.Response:
    api_url = f"{config.base_url.rstrip('/v1')}/health"
    return requests.get(api_url)


def request_openai_metrics(config: BaseClientConfig) -> List[Metric]:
    api_url = f"{config.base_url.rstrip('/v1')}/metrics"
    response = requests.get(api_url)
    response.raise_for_status()
    metrics = list(parser.text_string_to_metric_families(response.text))
    return metrics
