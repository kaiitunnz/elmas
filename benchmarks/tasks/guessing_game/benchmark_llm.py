import async_timeout
import asyncio
from dataclasses import asdict
from typing import List, Optional, Tuple, Union

from swarm.llm.format import Message
from swarm.llm.llm import LLM
from swarm.llm.llm_registry import LLMRegistry
from tenacity import retry, wait_random_exponential, stop_after_attempt

from utils.benchmarker import Benchmarker, ChatInputRequest, RequestFuncOutput  # type: ignore


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(10))
async def benchmark_achat(
    benchmarker: Benchmarker,
    record: List[Tuple[ChatInputRequest, RequestFuncOutput]],
    messages: List[Message],
    max_tokens: int = 8192,
    temperature: float = 0.0,
    num_comps=1,
) -> str:
    if messages[0].content == "$skip$":
        return ""

    formated_messages = [asdict(message) for message in messages]
    input_request = benchmarker.create_chat_input_request(
        formated_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        best_of=num_comps,
    )
    request_func_chat_input = benchmarker.create_request_func_chat_input(input_request)

    try:
        async with async_timeout.timeout(1000):
            output = await benchmarker.async_chat_request(request_func_chat_input)
            record.append((input_request, output))
    except asyncio.TimeoutError:
        print("Timeout")
        raise TimeoutError("GPT Timeout")

    return output.generated_text


@LLMRegistry.register("BenchmarkLLM")
class BenchmarkLLM(LLM):
    benchmarker: Optional[Benchmarker] = None
    record: List[Tuple[ChatInputRequest, RequestFuncOutput]] = []

    def __init__(self, model_name: str):
        self.model_name = model_name

    @classmethod
    def configure(cls, benchmarker: Benchmarker):
        cls.benchmarker = benchmarker
        cls.record = []

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        return await benchmark_achat(
            self.benchmarker, self.record, messages, max_tokens, temperature, num_comps
        )

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        raise NotImplementedError()
