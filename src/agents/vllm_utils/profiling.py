from openai import OpenAI

from ..vllm_utils.openai import request_openai_chat_completions, request_openai_health
from ..config import BaseProfilingConfig


class VLLMProfiling:
    def __init__(self, config: BaseProfilingConfig):
        self.config = config
        self.openai = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.running = False

    def start(self):
        if self.running:
            raise ValueError("Profiling is already running.")
        request_openai_chat_completions(self.config, "start_profile", "Hello")
        self.running = True
        print("Profiling started.")

    def stop(self):
        if not self.running:
            return
        request_openai_chat_completions(self.config, "stop_profile", "Hello")
        self.running = False
        print("Profiling stopped.")

    def __del__(self):
        if self.running:
            self.stop()
            print("Profiling stopped automatically.")
