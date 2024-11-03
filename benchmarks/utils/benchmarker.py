"""
Adapted from https://github.com/vllm-project/vllm.
"""

from agents.config import BaseClientConfig

import json
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import aiohttp
import numpy as np
from agents.utils.vllm.openai import request_openai_metrics
from tqdm.asyncio import tqdm  # type: ignore
from vllm.transformers_utils.tokenizer import AnyTokenizer, get_tokenizer


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    stop: Optional[str] = None
    temperature: float = 0.0
    best_of: int = 1
    use_beam_search: bool = False
    logprobs: Optional[int] = None
    multi_modal_content: Optional[dict] = None
    seed: Optional[int] = None


@dataclass
class RequestFuncChatInput:
    messages: List[Dict[str, str]]
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    stop: Optional[str] = None
    temperature: float = 0.0
    best_of: int = 1
    use_beam_search: bool = False
    logprobs: Optional[int] = None
    multi_modal_content: Optional[dict] = None
    seed: Optional[int] = None


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""


@dataclass
class InputRequest:
    prompt: str
    prompt_len: int
    output_len: int
    temperature: float = 0.0
    best_of: int = 1
    stop: Optional[str] = None
    seed: Optional[int] = None

    @classmethod
    def create(
        cls,
        prompt: str,
        max_tokens: int,
        tokenizer: AnyTokenizer,
        temperature: float = 0.0,
        best_of: int = 1,
        stop: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> "InputRequest":
        prompt_len = len(tokenizer(prompt).input_ids)
        return cls(
            prompt=prompt,
            prompt_len=prompt_len,
            output_len=max_tokens,
            temperature=temperature,
            best_of=best_of,
            stop=stop,
            seed=seed,
        )


@dataclass
class ChatInputRequest:
    messages: List[Dict[str, str]]
    prompt_len: int
    output_len: int
    temperature: float = 0.0
    best_of: int = 1
    stop: Optional[str] = None
    seed: Optional[int] = None

    @classmethod
    def create(
        cls,
        messages: List[Dict[str, str]],
        max_tokens: int,
        tokenizer: AnyTokenizer,
        temperature: float = 0.0,
        best_of: int = 1,
        stop: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> "ChatInputRequest":
        prompt_len = len(tokenizer.apply_chat_template(messages, tokenize=True))  # type: ignore
        return cls(
            messages=messages,
            prompt_len=prompt_len,
            output_len=max_tokens,
            temperature=temperature,
            best_of=best_of,
            stop=stop,
            seed=seed,
        )


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: List[Tuple[float, float]]
    mean_tpot_ms: float
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: List[Tuple[float, float]]
    mean_itl_ms: float
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: List[Tuple[float, float]]
    # E2EL stands for end-to-end latency per request.
    # It is the time taken on the client side from sending
    # a request to receiving a complete response.
    mean_e2el_ms: float
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: List[Tuple[float, float]]


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


async def async_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        ("completions", "profile")
    ), "OpenAI Completions API URL must end with 'completions' or 'profile'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": request_func_input.temperature,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "stop": request_func_input.stop,
            "seed": request_func_input.seed,
        }
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar is not None:
        pbar.update(1)
    return output


async def async_openai_chat_completions(
    request_func_chat_input: RequestFuncChatInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_chat_input.api_url
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Completions API URL must end with 'chat/completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_chat_input.use_beam_search
        payload = {
            "model": request_func_chat_input.model,
            "messages": request_func_chat_input.messages,
            "temperature": request_func_chat_input.temperature,
            "best_of": request_func_chat_input.best_of,
            "max_tokens": request_func_chat_input.output_len,
            "logprobs": request_func_chat_input.logprobs,
            "stream": True,
            "stop": request_func_chat_input.stop,
            "seed": request_func_chat_input.seed,
        }
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

        output = RequestFuncOutput()
        output.prompt_len = request_func_chat_input.prompt_len

        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)

                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # want to check a token was generated
                            delta = data["choices"][0]["delta"]
                            if delta and delta["content"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0.0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += delta["content"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar is not None:
        pbar.update(1)
    return output


class Benchmarker:
    CACHE_METRICS = {
        "cache_block_size": "Cache block size",
        "num_gpu_blocks": "Number of GPU blocks",
        "num_cpu_blocks": "Number of CPU blocks",
        "gpu_prefix_cache_hit_rate": "GPU prefix cache hit rate (%)",
        "cpu_prefix_cache_hit_rate": "CPU prefix cache hit rate (%)",
    }

    def __init__(
        self,
        server_config: BaseClientConfig,
        disabled_pbar: bool = False,
        seed: Optional[int] = None,
    ):
        self.server_config = server_config
        self.disabled_pbar = disabled_pbar
        self._tokenizer = get_tokenizer(server_config.model)
        self._duration: float = -1
        self._seed = seed
        self._reset()

    def _reset(self) -> None:
        self._start_time: float = -1
        self._data_len: int = -1
        self._pbar: Optional[tqdm] = None

    def create_input_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        best_of: int = 1,
        stop: Optional[str] = None,
    ) -> InputRequest:
        return InputRequest.create(
            prompt, max_tokens, self._tokenizer, temperature, best_of, stop, self._seed
        )

    def create_chat_input_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        best_of: int = 1,
        stop: Optional[str] = None,
    ) -> ChatInputRequest:
        return ChatInputRequest.create(
            messages,
            max_tokens,
            self._tokenizer,
            temperature,
            best_of,
            stop,
            self._seed,
        )

    def create_request_func_input(
        self, input_request: InputRequest, endpoint: str = "/completions"
    ) -> RequestFuncInput:
        return RequestFuncInput(
            prompt=input_request.prompt,
            api_url=self.server_config.base_url + endpoint,
            prompt_len=input_request.prompt_len,
            output_len=input_request.output_len,
            model=self.server_config.model,
            stop=input_request.stop,
            temperature=input_request.temperature,
            best_of=input_request.best_of,
            use_beam_search=False,
            logprobs=None,
            multi_modal_content=None,
            seed=input_request.seed,
        )

    def create_request_func_chat_input(
        self, chat_input_request: ChatInputRequest, endpoint: str = "/chat/completions"
    ) -> RequestFuncChatInput:
        return RequestFuncChatInput(
            messages=chat_input_request.messages,
            api_url=self.server_config.base_url + endpoint,
            prompt_len=chat_input_request.prompt_len,
            output_len=chat_input_request.output_len,
            model=self.server_config.model,
            stop=chat_input_request.stop,
            temperature=chat_input_request.temperature,
            best_of=chat_input_request.best_of,
            use_beam_search=False,
            logprobs=None,
            multi_modal_content=None,
            seed=chat_input_request.seed,
        )

    def set_num_requests(self, num_requests: int) -> None:
        self._data_len = num_requests

    async def async_request(
        self, request_func_input: RequestFuncInput
    ) -> RequestFuncOutput:
        return await async_openai_completions(request_func_input, pbar=self._pbar)

    async def async_chat_request(
        self, request_func_chat_input: RequestFuncChatInput
    ) -> RequestFuncOutput:
        return await async_openai_chat_completions(
            request_func_chat_input, pbar=self._pbar
        )

    def calculate_metrics(
        self,
        input_requests: List[Union[InputRequest, ChatInputRequest]],
        outputs: List[RequestFuncOutput],
        selected_percentiles: List[float],
    ):
        tokenizer = self._tokenizer
        dur_s = self.duration

        actual_output_lens: List[int] = []
        total_input = 0
        completed = 0
        itls: List[float] = []
        tpots: List[float] = []
        ttfts: List[float] = []
        e2els: List[float] = []
        for i in range(len(outputs)):
            if outputs[i].success:
                # We use the tokenizer to count the number of output tokens for all
                # serving backends instead of looking at len(outputs[i].itl) since
                # multiple output tokens may be bundled together
                # Note : this may inflate the output token count slightly
                output_len = len(
                    tokenizer(
                        outputs[i].generated_text, add_special_tokens=False
                    ).input_ids
                )
                actual_output_lens.append(output_len)
                total_input += input_requests[i].prompt_len
                if output_len > 1:
                    tpots.append(
                        (outputs[i].latency - outputs[i].ttft) / (output_len - 1)
                    )
                itls += outputs[i].itl
                ttfts.append(outputs[i].ttft)
                e2els.append(outputs[i].latency)
                completed += 1
            else:
                actual_output_lens.append(0)

        if completed == 0:
            warnings.warn(
                "All requests failed. This is likely due to a misconfiguration "
                "on the benchmark arguments.",
                stacklevel=2,
            )
        metrics = BenchmarkMetrics(
            completed=completed,
            total_input=total_input,
            total_output=sum(actual_output_lens),
            request_throughput=completed / dur_s,
            output_throughput=sum(actual_output_lens) / dur_s,
            total_token_throughput=(total_input + sum(actual_output_lens)) / dur_s,
            mean_ttft_ms=np.mean(ttfts or 0).item()
            * 1000,  # ttfts is empty if streaming is not supported by backend
            std_ttft_ms=np.std(ttfts or 0).item() * 1000,
            median_ttft_ms=np.median(ttfts or 0).item() * 1000,
            percentiles_ttft_ms=[
                (p, np.percentile(ttfts or 0, p).item() * 1000)
                for p in selected_percentiles
            ],
            mean_tpot_ms=np.mean(tpots or 0).item() * 1000,
            std_tpot_ms=np.std(tpots or 0).item() * 1000,
            median_tpot_ms=np.median(tpots or 0).item() * 1000,
            percentiles_tpot_ms=[
                (p, np.percentile(tpots or 0, p).item() * 1000)
                for p in selected_percentiles
            ],
            mean_itl_ms=np.mean(itls or 0).item() * 1000,
            std_itl_ms=np.std(itls or 0).item() * 1000,
            median_itl_ms=np.median(itls or 0).item() * 1000,
            percentiles_itl_ms=[
                (p, np.percentile(itls or 0, p).item() * 1000)
                for p in selected_percentiles
            ],
            mean_e2el_ms=np.median(e2els or 0).item() * 1000,
            std_e2el_ms=np.std(e2els or 0).item() * 1000,
            median_e2el_ms=np.mean(e2els or 0).item() * 1000,
            percentiles_e2el_ms=[
                (p, np.percentile(e2els or 0, p).item() * 1000)
                for p in selected_percentiles
            ],
        )

        return metrics, actual_output_lens

    def calculate_results(
        self,
        input_requests: List[Union[InputRequest, ChatInputRequest]],
        outputs: List[RequestFuncOutput],
        selected_percentile_metrics: List[str],
        selected_percentiles: List[float],
    ) -> Dict[str, float]:
        benchmark_duration = self.duration
        metrics, actual_output_lens = self.calculate_metrics(
            input_requests, outputs, selected_percentiles
        )
        result = {
            "duration": benchmark_duration,
            "completed": metrics.completed,
            "total_input_tokens": metrics.total_input,
            "total_output_tokens": metrics.total_output,
            "request_throughput": metrics.request_throughput,
            "output_throughput": metrics.output_throughput,
            "total_token_throughput": metrics.total_token_throughput,
            "input_lens": [output.prompt_len for output in outputs],
            "output_lens": actual_output_lens,
            "ttfts": [output.ttft for output in outputs],
            "itls": [output.itl for output in outputs],
            "generated_texts": [output.generated_text for output in outputs],
            "errors": [output.error for output in outputs],
        }

        def process_one_metric(metric_attribute_name: str):
            if metric_attribute_name not in selected_percentile_metrics:
                return
            result[f"mean_{metric_attribute_name}_ms"] = getattr(
                metrics, f"mean_{metric_attribute_name}_ms"
            )
            result[f"median_{metric_attribute_name}_ms"] = getattr(
                metrics, f"median_{metric_attribute_name}_ms"
            )
            result[f"std_{metric_attribute_name}_ms"] = getattr(
                metrics, f"std_{metric_attribute_name}_ms"
            )
            for p, value in getattr(metrics, f"percentiles_{metric_attribute_name}_ms"):
                p_word = str(int(p)) if int(p) == p else str(p)
                result[f"p{p_word}_{metric_attribute_name}_ms"] = value

        process_one_metric("ttft")
        process_one_metric("tpot")
        process_one_metric("itl")
        process_one_metric("e2el")

        # Add server metrics
        cache_metrics = request_openai_metrics(self.server_config)
        for metric in cache_metrics:
            metric_name = metric.name[len("vllm:") :]
            if metric_name == "cache_config_info":
                labels = metric.samples[0].labels
                result["cache_block_size"] = int(labels["block_size"])
                result["num_gpu_blocks"] = int(labels["num_gpu_blocks"])
                result["num_cpu_blocks"] = int(labels["num_cpu_blocks"])
                continue
            if metric_name not in self.CACHE_METRICS:
                continue
            sample = next(
                sample
                for sample in metric.samples
                if sample.labels["model_name"] == self.server_config.model
            )
            result[metric_name] = sample.value * 100

        return result

    def report_results(
        self,
        benchmark_results: Dict[str, float],
        selected_percentile_metrics: List[str],
    ) -> None:
        def print_header(header: str, c: str = "-"):
            print("{s:{c}^{n}}".format(s=header, n=50, c=c))

        def print_int_metric(name: str, value: float):
            print("{:<40} {:<10}".format(name, value))

        def print_float_metric(name: str, value: float):
            print("{:<40} {:<10.2f}".format(name, value))

        print_header(" Serving Benchmark Result ", "=")
        print_int_metric("Successful requests:", benchmark_results["completed"])
        print_float_metric("Benchmark duration (s):", benchmark_results["duration"])
        print_int_metric("Total input tokens:", benchmark_results["total_input_tokens"])
        print_int_metric(
            "Total generated tokens:", benchmark_results["total_output_tokens"]
        )
        print_float_metric(
            "Request throughput (req/s):", benchmark_results["request_throughput"]
        )
        print_float_metric(
            "Output token throughput (tok/s):",
            benchmark_results["output_throughput"],
        )
        print_float_metric(
            "Total token throughput (tok/s):",
            benchmark_results["total_token_throughput"],
        )

        def report_one_metric(
            # E.g., "ttft"
            metric_attribute_name: str,
            # E.g., "TTFT"
            metric_name: str,
            # E.g., "Time to First Token"
            metric_header: str,
        ):
            # This function print and add statistics of the specified
            # metric.
            if metric_attribute_name not in selected_percentile_metrics:
                return
            print_header(metric_header)
            print_float_metric(
                f"Mean {metric_name} (ms):",
                benchmark_results[f"mean_{metric_attribute_name}_ms"],
            )
            print_float_metric(
                f"Median {metric_name} (ms):",
                benchmark_results[f"median_{metric_attribute_name}_ms"],
            )
            for metric_name, value in benchmark_results.items():
                splitted = metric_name.split("_")
                if splitted != 3:
                    continue
                p, attribute_name, _ = splitted
                if attribute_name == metric_attribute_name and p.startswith("p"):
                    p_word = p[1:]
                    print_float_metric(f"P{p_word} {metric_name} (ms):", value)

        report_one_metric("ttft", "TTFT", "Time to First Token")
        report_one_metric("tpot", "TPOT", "Time per Output Token (excl. 1st token)")
        report_one_metric("itl", "ITL", "Inter-token Latency")
        report_one_metric("e2el", "E2EL", "End-to-end Latency")

        print_header("Cache Metrics")
        int_metrics = ["cache_block_size", "num_gpu_blocks", "num_cpu_blocks"]
        for metric_name, metric_desc in self.CACHE_METRICS.items():
            if metric_name in benchmark_results:
                if metric_name in int_metrics:
                    print_int_metric(metric_desc, benchmark_results[metric_name])
                else:
                    print_float_metric(metric_desc, benchmark_results[metric_name])

        print("=" * 50)

    def save_results(
        self, benchmark_results: Dict[str, float], result_file: str
    ) -> None:
        with open(result_file, "w") as f:
            json.dump(benchmark_results, f)

    def __enter__(self):
        self._start_time = time.perf_counter()
        if not self.disabled_pbar:
            self._pbar = tqdm(total=self._data_len if self._data_len >= 0 else None)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._pbar is not None:
            self._pbar.close()
        self._duration = time.perf_counter() - self._start_time
        self._reset()

    @property
    def tokenizer(self) -> AnyTokenizer:
        return self._tokenizer

    @property
    def duration(self) -> float:
        if self._duration < 0:
            raise ValueError("Benchmark not yet completed.")
        return self._duration
