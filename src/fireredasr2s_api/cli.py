# Copyright 2026 FireRedTeam

"""Command-line interface for FireRedASR2S API."""

import argparse
import asyncio
import logging
import sys

from .config import ApiConfig, AsrBackendConfig
from .serve import main as serve_main

logger = logging.getLogger(__name__)


def main() -> None:
    """
    CLI entry point for FireRedASR2S API.

    Supports running servers with custom host and port configuration.
    """
    parser = argparse.ArgumentParser(
        description="FireRedASR2S API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=50051,
        help="gRPC server port (default: 50051)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Base directory for pretrained models (overrides MODEL_DIR env var)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--enable-lid",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable Language Identification (0=disable, 1=enable, default: 1)",
    )
    parser.add_argument(
        "--enable-punc",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable Punctuation prediction (0=disable, 1=enable, default: 1)",
    )
    parser.add_argument(
        "--asr-type",
        type=str,
        default="aed",
        choices=["aed", "llm"],
        help="ASR backend type (default: aed)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = ApiConfig(
        model_base_dir=args.model_dir,
        enable_lid=bool(args.enable_lid),
        enable_punc=bool(args.enable_punc),
        asr=AsrBackendConfig(asr_type=args.asr_type),
    )

    try:
        asyncio.run(
            serve_main(
                config=config,
                host=args.host,
                grpc_port=args.grpc_port,
            )
        )
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
