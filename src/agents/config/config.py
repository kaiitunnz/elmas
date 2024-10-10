import os
from dataclasses import dataclass

from dotenv import load_dotenv


def init_config(config_path: str = os.getenv("SERVER_CONFIG_PATH", ".env")) -> None:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    load_dotenv(config_path)


init_config()


@dataclass
class BaseClientConfig:
    api_key: str = os.environ["OPENAI_API_KEY"]
    model: str = os.environ["DEFAULT_MODEL"]
    host: str = os.environ["DEFAULT_HOST"]
    port: int = int(os.environ["DEFAULT_PORT"])

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


@dataclass
class BaseProfilingConfig(BaseClientConfig):
    pass