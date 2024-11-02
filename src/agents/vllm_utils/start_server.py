from agents.config import BaseClientConfig

import asyncio
import os
import signal
import socket
from argparse import Namespace
from dataclasses import dataclass
from multiprocessing.synchronize import Semaphore
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
import uvloop
from fastapi import FastAPI
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.launcher import _add_shutdown_handlers
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.entrypoints.openai.api_server import (
    TIMEOUT_KEEP_ALIVE,
    build_async_engine_client,
    build_app,
    init_app_state,
    logger,
)
from vllm.utils import find_process_using_port
from vllm.version import __version__ as VLLM_VERSION


@dataclass
class BaseServerConfig(BaseClientConfig):
    root_dir: Path = Path.cwd()
    device: str = os.getenv("DEFAULT_DEVICE", "cpu")
    server_path: str = "vllm.entrypoints.openai.api_server"
    uvicorn_log_level: str = "warning"
    log_level: str = "WARN"
    reuse_addr: bool = True

    profiling: bool = True

    preemption_mode: Optional[str] = "recompute"
    num_gpu_blocks_override: Optional[int] = 512
    num_cpu_blocks_override: Optional[int] = 512
    max_model_len: Optional[int] = 8192
    block_size: int = 16

    enable_prefix_caching: bool = True

    enable_multi_tier_prefix_caching: bool = True
    enable_async_swapping: bool = True
    enable_prefix_aware_scheduling: bool = True
    enable_async_prefetching: bool = True

    @property
    def trace_dir(self) -> Path:
        return self.root_dir / "traces"

    def default_args(self) -> Namespace:
        return Namespace(
            model=self.model,
            host=self.host,
            port=self.port,
            device=self.device,
            uvicorn_log_level=self.uvicorn_log_level,
            preemption_mode=self.preemption_mode,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            num_cpu_blocks_override=self.num_cpu_blocks_override,
            max_model_len=self.max_model_len,
            block_size=self.block_size,
            enable_prefix_caching=self.enable_prefix_caching,
            enable_multi_tier_prefix_caching=self.enable_multi_tier_prefix_caching,
            enable_async_swapping=self.enable_async_swapping,
            enable_prefix_aware_scheduling=self.enable_prefix_aware_scheduling,
            enable_async_prefetching=self.enable_async_prefetching,
        )

    def engine_args(self) -> Dict[str, Any]:
        server_args = self.parse_args([]).__dict__
        for k in ["log_level", "profiling", "reuse_addr"]:
            server_args.pop(k, None)
        return server_args

    def parse_args(self, args: Optional[Any] = None) -> Namespace:
        parser = FlexibleArgumentParser(
            description="vLLM OpenAI-Compatible RESTful API server."
        )
        parser = make_arg_parser(parser)

        parser.add_argument(
            "--log-level", default=self.log_level, help="Logging level."
        )
        parser.add_argument(
            "--profiling",
            action="store_true",
            default=self.profiling,
            help="Start the server in profiling mode. # Traces can be viewed in `https://ui.perfetto.dev/`",
        )
        parser.add_argument(
            "--reuse-addr",
            action="store_true",
            default=self.reuse_addr,
            help="Enable SO_REUSEADDR for the server.",
        )

        return parser.parse_args(args, self.default_args())


async def serve_http(
    app: FastAPI, sema: Optional[Semaphore] = None, **uvicorn_kwargs: Any
):
    logger.info("Available routes are:")
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)

        if methods is None or path is None:
            continue

        logger.info("Route: %s, Methods: %s", path, ", ".join(methods))

    config = uvicorn.Config(app, **uvicorn_kwargs)
    server = uvicorn.Server(config)
    _add_shutdown_handlers(app, server)

    loop = asyncio.get_running_loop()

    server_task = loop.create_task(server.serve())

    def signal_handler() -> None:
        # prevents the uvicorn signal handler to exit early
        server_task.cancel()

    async def dummy_shutdown() -> None:
        pass

    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    if sema is not None:
        sema.release()

    try:
        await server_task
        return dummy_shutdown()
    except asyncio.CancelledError:
        port = uvicorn_kwargs["port"]
        process = find_process_using_port(port)
        if process is not None:
            logger.debug(
                "port %s is used by process %s launched with command:\n%s",
                port,
                process,
                " ".join(process.cmdline()),
            )
        logger.info("Shutting down FastAPI HTTP server.")
        return server.shutdown()


async def run_server(args, sema: Optional[Semaphore] = None, **uvicorn_kwargs) -> None:
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
            sema,
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


def main(
    args: Namespace, config: BaseServerConfig, sema: Optional[Semaphore] = None
) -> None:
    log_level = (
        config.log_level if getattr(args, "log_level", None) is None else args.log_level
    )
    os.environ["VLLM_LOGGING_LEVEL"] = log_level

    if args.profiling:
        os.environ["VLLM_TORCH_PROFILER_DIR"] = str(config.trace_dir)

    uvloop.run(run_server(args, sema))


if __name__ == "__main__":
    config = BaseServerConfig()
    args = config.parse_args()
    main(args, config)
