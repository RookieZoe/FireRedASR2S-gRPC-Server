# FireRedASR2S gRPC Server

The FireRedASR2S gRPC Server provides a high-performance, real-time Automatic Speech Recognition (ASR) interface for the FireRedASR2S models. It supports bidirectional gRPC streaming, enabling low-latency transcription for service-to-service communication.

- **gRPC Streaming**: Bidirectional streaming RPC for real-time audio processing.
- **Stateful VAD**: Streaming Voice Activity Detection with slice-based segmentation.
- **Flexible Backends**: Supports both AED (Advanced Endpoint Detection) and LLM-based ASR backends.
- **Post-Processing**: Integrated Language Identification (LID) and Punctuation (Punc) modules.
- **Rich Metadata**: Provides word-level timestamps, confidence scores, and audio event detection.

## Caution

- Only tested on Linux with Python 3.10+, RTX4090, CUDA 12.4, and PyTorch 2.5.x.
- Only tested with long audio files (up to 2 hour) and batch size of 1, chunk size of 30 second.
- Performance may vary with different hardware, audio lengths, and batch sizes.
- Real-time scenario has not been tested

> with client [FireRedASR2S-gRPC-Client](https://github.com/RookieZoe/FireRedASR2S-gRPC-Client)

## Quick Start

See [HOW-TO-USE.md](HOW-TO-USE.md) for detailed installation and setup instructions.

```bash
# Start server with default settings
uv run python -m asr2s_grpc.serve
```

## Documentation

- [HOW-TO-USE.md](HOW-TO-USE.md): Installation, model setup, and server configuration.
- [docs/API.md](docs/API.md): Detailed gRPC protocol specification and message definitions.

## Development

Compact commands for local development:

- **Run Tests**: `uv run pytest tests/ -v`
- **Type Checking**: `uv run mypy src/asr2s_grpc`
- **Code Formatting**: `uv run black src/ tests/ && uv run isort src/ tests/`

## Thanks (Acknowledgment)

This project is a gRPC service wrapper for the [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S) project. We credit the FireRedTeam for their excellent ASR models and core implementation.

- **Upstream Project**: [FireRedTeam/FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S)
- **Upstream License**: Apache 2.0

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
