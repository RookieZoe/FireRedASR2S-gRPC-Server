# FireRedASR2S API

Real-time ASR API for FireRedASR2S with gRPC streaming support.

## Features

- **gRPC API**: Bidirectional streaming for service-to-service communication
- **Streaming VAD**: Voice Activity Detection with real-time processing
- **Partial Results**: Revisable intermediate transcription results
- **Final Processing**: LID and Punctuation on completed segments

## Quick Start

### Installation

```bash
cd api
pip install -e ".[dev]"
```

### Start Server

Standard usage:
```bash
python -m fireredasr2s_api.serve
```

With custom model directory and features:
```bash
# Disable LID and Punc
python -m fireredasr2s_api.serve --enable-lid 0 --enable-punc 0

# Using custom model directory
python -m fireredasr2s_api.serve --model-dir /path/to/models

```

Using environment variables:
```bash
MODEL_DIR=/path/to/models python -m fireredasr2s_api.serve
```

Running from the `api/` directory:
```bash
# The API automatically detects the repository root and finds pretrained_models/
cd api
python -m fireredasr2s_api.serve
```

## Path Resolution

The API uses a robust path resolution strategy to find model files:

1.  **CLI Flag**: `--model-dir` takes highest precedence.
2.  **Environment Variable**: `MODEL_DIR` is used if the CLI flag is not provided.
3.  **Automatic Detection**: If neither is provided, the API looks for a `pretrained_models/` directory relative to the repository root.

**Note on "Base Directory" semantics**: `MODEL_DIR` should point to the directory *containing* the individual model folders (e.g., `FireRedASR2-AED`, `FireRedVAD`, etc.). If `MODEL_DIR=/path/to/models`, the API will look for `/path/to/models/FireRedASR2-AED`.

## API Endpoint

### gRPC `StreamingRecognize`

Bidirectional streaming RPC for service-to-service communication.

**Client → Server Messages (StreamingRecognizeRequest oneof):**
- `config`: RecognitionConfig with sample_rate, format, beam_size, and enable_timestamps.
- `audio_slice`: AudioSlice message with **index**, **n_ms**, and **data** fields. `index` must be contiguous starting from 0. `n_ms` must be ≤ 30000.
- `end_stream`: Signal end of stream.

**Server → Client Messages (StreamingRecognizeResponse oneof):**
- `slice_vad`: SliceVad message with `slice_index`, `slice_m_ms`, `slice_n_ms`, `ended_speaking`, and `entirely_speech`. Emitted before `partial` or `final` for the same slice.
- `partial`: PartialResult with text, confidence, timestamps, and **slice_index**.
- `final`: FinalResult with punctuated text, sentences, words, **language**, **language_confidence**, and **slice_index**.
- `error`: ErrorResult with code and message.

### Slice-Based VAD Segmentation

The API supports slice-based streaming where each audio slice `[0, n]` is processed independently with stateful VAD:

1. Client sends `audio_slice` metadata with `slice_index` and `n_ms` (where `n_ms` ≤ 30s).
2. Client sends binary audio data for that slice.
3. Server runs streaming VAD on the slice and emits a `slice_vad` event.
4. If speech is detected, the server emits `partial` and `final` results for the slice.
5. The `slice_vad` event is **always** emitted before any `partial` or `final` results for the same slice.
6. Client maintains cumulative position `q`. If `slice_m_ms` is the boundary, the next slice covers absolute range `[q+m, q+m+n]`.

### Final Post-Processing (LID and Punc)

The API includes optional Language Identification (LID) and Punctuation (Punc) modules:

- **LID**: Runs Language Identification on the **entire audio stream** once the client signals the end. The result is returned in the `final` message as `language` and `lang_confidence`.
- **Punctuation**: Automatically predicts and adds punctuation to the final transcription text. Punctuation is **not** applied to `partial` results.
- **Enabled by Default**: Both features are enabled by default but can be toggled via CLI flags (`--enable-lid 0`, `--enable-punc 0`).

**Final Output Example (gRPC):**
```json
{
  "text": "Hello, world!",
  "language": "en",
  "language_confidence": 0.998,
  "slice_index": 5
}
```

**Protocol Constraints:**
- **Strict Ordering**: Messages must follow the sequence: `config` → (`audio_slice` → `slice_vad` → `partial`* → `final`?) x N.
- **Contiguous Indices**: `slice_index` must start at 0 and increase by exactly 1 for each subsequent slice.
- **Slice Duration**: Each slice (`n_ms`) must be ≤ 30,000ms (30 seconds).

**SliceVad event fields:**
| Field | Type | Description |
|-------|------|-------------|
| `slice_index` | int32 | Client-assigned slice identifier (non-negative) |
| `slice_m_ms` | int32 | Server-computed speech boundary in milliseconds |
| `slice_n_ms` | int32 | Total duration of the audio slice processed |
| `ended_speaking`| bool | Whether a speech segment ended within this slice |
| `entirely_speech`| bool | Whether the entire slice was detected as speech |

## Configuration

Environment variables:

- `API_HOST`: Server host (default: `0.0.0.0`)
- `GRPC_PORT`: gRPC port (default: `50051`)
- `MODEL_DIR`: Base directory for pretrained models (e.g., `/path/to/models`)
- `LOG_LEVEL`: Logging level (default: `INFO`)

### ASR Backend Selection (`--asr-type`)

The server selects the ASR backend at startup via the `--asr-type` CLI flag. Clients cannot change the backend via the protocol.

| `--asr-type` | Model directory | Max audio length | Timestamps | Confidence |
|------------|-----------------|-----------------|-----------|------------|
| `aed` (default) | `pretrained_models/FireRedASR2-AED` | 60 seconds | Yes | Yes |
| `llm` | `pretrained_models/FireRedASR2-LLM` | **40 seconds** | No | No |

**Usage examples:**

```bash
# Start with default AED backend
python -m fireredasr2s_api.serve --asr-type aed

# Start with LLM backend
python -m fireredasr2s_api.serve --asr-type llm
```

**LLM-specific decoding parameters** (per-request, passed in `llm_params`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decode_min_len` | `0` | Minimum output token length |
| `repetition_penalty` | `1.0` | Penalty factor for repeated tokens (>1 discourages repetition) |
| `llm_length_penalty` | `0.0` | Length penalty applied during beam search |
| `temperature` | `1.0` | Sampling temperature (lower = more deterministic) |

**LLM backend limitations:**

- Maximum audio input is **40 seconds**. Longer audio will be rejected with an error.
- Word-level timestamps are **not supported** — `words` is always empty (`[]`).
- The `confidence` field is **not returned** in LLM transcription results.
- Proto3 zero-values for LLM params (e.g., `temperature=0.0`) are treated as unset and fall back to defaults.

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Type Checking

```bash
mypy src/fireredasr2s_api
```

### Code Formatting

```bash
black src/ tests/
isort src/ tests/
```

## License

MIT License - see LICENSE file for details.
