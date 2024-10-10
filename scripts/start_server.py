import os
import signal
import socket
from argparse import Namespace
from pathlib import Path
from typing import Optional

from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.version import __version__ as VLLM_VERSION

import uvloop
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


def parse_args(namespace: Optional[Namespace] = None) -> Namespace:
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)

    parser.add_argument(
        "--profiling",
        action="store_true",
        default=False,
        help="Start the server in profiling mode. # Traces can be viewed in `https://ui.perfetto.dev/",
    )
    parser.add_argument(
        "--reuse-addr",
        action="store_true",
        default=True,
        help="Enable SO_REUSEADDR for the server.",
    )

    return parser.parse_args(namespace=namespace)


async def run_server(args, **uvicorn_kwargs) -> None:
    from vllm.entrypoints.openai.api_server import (
        TIMEOUT_KEEP_ALIVE,
        build_async_engine_client,
        build_app,
        init_app_state,
        logger,
    )

    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    if args.tool_parser_plugin and len(args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(args.tool_parser_plugin)

    valide_tool_parses = ToolParserManager.tool_parsers.keys()
    if args.enable_auto_tool_choice and args.tool_call_parser not in valide_tool_parses:
        raise KeyError(
            f"invalid tool call parser: {args.tool_call_parser} "
            f"(chose from {{ {','.join(valide_tool_parses)} }})"
        )

    temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if args.reuse_addr:
        temp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    temp_socket.bind(("", args.port))

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    async with build_async_engine_client(args) as engine_client:
        app = build_app(args)

        model_config = await engine_client.get_model_config()
        init_app_state(engine_client, model_config, app.state, args)

        temp_socket.close()

        shutdown_task = await serve_http(
            app,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            ssl_ca_certs=args.ssl_ca_certs,
            ssl_cert_reqs=args.ssl_cert_reqs,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task


def main(args: Namespace, config: ServerConfig):
    if args.profiling:
        os.environ["VLLM_TORCH_PROFILER_DIR"] = str(config.trace_dir)

    uvloop.run(run_server(args))


if __name__ == "__main__":
    config = ServerConfig()
    args = parse_args(
        Namespace(
            model=config.model,
            host=config.host,
            port=config.port,
            device=config.device,
            uvicorn_log_level=config.log_level,
        )
    )
    main(args, config)
