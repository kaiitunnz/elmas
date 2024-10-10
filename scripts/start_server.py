import os
import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path

from agents.config import BaseClientConfig


class ServerConfig(BaseClientConfig):
    # root_dir: Path = Path("..")
    root_dir: Path = Path("/home/noppanat/Workspace/Projects/new-project")
    device: str = os.getenv("DEFAULT_DEVICE", "cpu")
    server_path: str = "vllm.entrypoints.openai.api_server"
    log_level: str = "info"

    @property
    def trace_dir(self) -> Path:
        return self.root_dir / "traces"


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--profiling",
        action="store_true",
        default=False,
        help="Start the server in profiling mode. # Traces can be viewed in `https://ui.perfetto.dev/",
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action="store_true",
        default=False,
        help="Enable prefix caching for the server.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    config = ServerConfig()

    env = os.environ
    if args.profiling:
        env = env | {"VLLM_TORCH_PROFILER_DIR": config.trace_dir}

    subprocess.run(
        (
            [
                f"python",
                f"-m",
                f"{config.server_path}",
                f"--model={config.model}",
                f"--host={config.host}",
                f"--port={config.port}",
                f"--device={config.device}",
                f"--uvicorn-log-level={config.log_level}",
            ]
            + (["--enable-prefix-caching"] if args.enable_prefix_caching else [])
        ),
        env=env,
    )


if __name__ == "__main__":
    main()
