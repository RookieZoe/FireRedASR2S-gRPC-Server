# Copyright 2026 FireRedTeam

"""Entry point for starting gRPC server."""

import logging
from typing import Optional
from .config import ApiConfig
from .grpc_server import serve as serve_grpc

logger = logging.getLogger(__name__)


async def main(
    config: Optional[ApiConfig] = None,
    host: Optional[str] = None,
    grpc_port: Optional[int] = None,
) -> None:
    """Start gRPC server."""
    if config is None:
        config = ApiConfig()

    logger.info("Starting FireRedASR2S API server...")

    try:
        await serve_grpc(config, host=host, port=grpc_port)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    from .cli import main as cli_main
    cli_main()
