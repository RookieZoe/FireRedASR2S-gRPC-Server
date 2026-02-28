# How to Use FireRedASR2S gRPC Server

This guide provides practical instructions for setting up and running the FireRedASR2S gRPC server.

## Prerequisites

- **Python**: 3.10, 3.11, or 3.12.
- **Environment Manager**: [uv](https://github.com/astral-sh/uv) (recommended).
- **Hardware**: GPU with CUDA support for optimal performance.
- **FireRedASR2S Vendor Code**: A clone of the [FireRedASR2S](https://github.com/FireRedTeam/FireRedASR2S.git) repository.

## Installation

1. **Sync Dependencies**:
   Install all required Python packages using `uv`.
   ```bash
   cd FireRedASR2S-gRPC-Server
   uv sync
   ```

2. **Clone Vendor Code**:
   The server depends on the original FireRedASR2S implementation. Clone it into the `vendor/` directory.
   ```bash
   git clone https://github.com/FireRedTeam/FireRedASR2S.git vendor/FireRedASR2S
   ```

3. **Download Model Files**:
   Download the pretrained model files and place them in the `models/` directory (see [Model Directory Structure](#model-directory-structure) below).

## Starting the Server

### Recommended Method
The `./run_server.sh` script is the recommended way to start the server. It automatically configures `LD_LIBRARY_PATH` (to ensure correct cuBLAS version) and `PYTHONPATH` (to include the vendor code).

#### Running in LLM Mode (Prominent)
To run the server using the **Large Language Model (LLM)** backend:
```bash
./run_server.sh --asr-type llm
```

#### Running in AED Mode (Default)
To run the server using the **Attention-based Encoder-Decoder (AED)** backend:
```bash
./run_server.sh --asr-type aed
```

### Alternative Method
If you prefer running directly with `uv run`, ensure you set the `PYTHONPATH`:
```bash
PYTHONPATH=vendor/FireRedASR2S uv run python -m asr2s_grpc.serve [flags]
```

## Server CLI Flags Reference

| Flag | Default | Choices | Description |
| :--- | :--- | :--- | :--- |
| `--host` | `0.0.0.0` | - | Server host address to bind to. |
| `--grpc-port` | `50051` | - | gRPC server port. |
| `--model-dir` | `None` | - | Base directory for pretrained models. Overrides `MODEL_DIR` env var. If `None`, it auto-detects based on repo root or CWD. |
| `--log-level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | Logging level. |
| `--enable-lid` | `1` | `0`, `1` | Enable (1) or disable (0) Language Identification. |
| `--enable-punc` | `1` | `0`, `1` | Enable (1) or disable (0) Punctuation prediction. |
| `--asr-type` | `aed` | `aed`, `llm` | ASR backend type. |
| `--vad-type` | `all` | `vad`, `stream-vad`, `aed`, `all` | VAD mode. `all` loads all VAD models for versatility. |

## Model Directory Structure

The `--model-dir` flag (or `MODEL_DIR` environment variable) should point to a directory containing the following subdirectories:

- `models/FireRedASR2-AED`: Required for AED mode.
- `models/FireRedASR2-LLM`: Required for LLM mode.
- `models/FireRedVAD`: Contains VAD, Stream-VAD, and AED-VAD models.
- `models/FireRedLID`: Required for Language Identification.
- `models/FireRedPunc`: Required for Punctuation prediction.

## Environment Variables

The server supports the following environment variable:

- `MODEL_DIR`: The base directory where models are located. This is overridden by the `--model-dir` CLI flag.

## LLM Backend Notes

When using `--asr-type llm`, please be aware of the following:

- **Max Audio Length**: Supported up to 40 seconds per request.
- **No Timestamps**: Word-level timestamps are not available in LLM mode.
- **No Confidence Score**: The confidence field is not returned.
- **Per-Request Parameters**: LLM mode supports specific tuning parameters in gRPC requests:
  - `decode_min_len`
  - `repetition_penalty`
  - `llm_length_penalty`
  - `temperature`

## VAD Mode Notes

The `--vad-type` flag controls which Voice Activity Detection models are loaded:
- `vad`: Standard non-streaming VAD.
- `stream-vad`: Optimized for real-time streaming.
- `aed`: AED-based VAD.
- `all`: (Default) Loads all of the above to handle various request types.

## Client Usage

To interact with this server, please refer to the [FireRedASR2S-gRPC-Client](https://github.com/RookieZoe/FireRedASR2S-gRPC-Client) repository for client-side implementation and examples.
