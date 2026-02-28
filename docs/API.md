# FireRedASR2S gRPC API Reference

This document describes the gRPC protocol for FireRedASR2S, focusing on the bidirectional streaming interface for speech recognition and audio event detection.

## Overview

The `ASRService` provides a single bidirectional streaming RPC method, `StreamingRecognize`. This method allows clients to stream audio data in real-time and receive corresponding transcription results, VAD (Voice Activity Detection) events, and AED (Audio Event Detection) results.

The protocol follows a slice-based approach, where the client sends a configuration followed by one or more audio slices. The server processes each slice and returns intermediate results and metadata.

## Service Definition

```proto
service ASRService {
  rpc StreamingRecognize(stream StreamingRecognizeRequest) returns (stream StreamingRecognizeResponse);
}
```

---

## Client → Server Messages (`StreamingRecognizeRequest`)

The `StreamingRecognizeRequest` is a `oneof` message. The client must send a `config` message first, followed by audio data or stream control signals.

| Field | Type | Description |
| :--- | :--- | :--- |
| `config` | `RecognitionConfig` | Initial session configuration (must be sent first) |
| `audio_chunk` | `bytes` | Raw PCM audio data (deprecated, use `audio_slice`) |
| `end_stream` | `bool` | Signals the end of the stream |
| `audio_slice` | `AudioSlice` | Slice-based audio payload with index and data |

### RecognitionConfig

| Field | Type | Number | Description |
| :--- | :--- | :--- | :--- |
| `sample_rate` | `int32` | 1 | Audio sample rate (e.g., 16000) |
| `format` | `string` | 2 | Audio format (e.g., "pcm") |
| `enable_timestamps` | `bool` | 3 | Whether to return word/sentence timestamps |
| `asr_type` | - | 4 | **RESERVED/DEPRECATED** |
| `beam_size` | `int32` | 5 | Beam size for ASR decoding |
| `slice_index` | `int32` | 6 | Initial slice index (usually 0) |
| `decode_min_len` | `int32` | 7 | LLM param: Minimum decoding length |
| `repetition_penalty`| `float` | 8 | LLM param: Repetition penalty factor |
| `llm_length_penalty`| `float` | 9 | LLM param: Length penalty factor |
| `temperature` | `float` | 10 | LLM param: Sampling temperature |
| `vad_type` | `string` | 11 | Client-settable: `vad`, `stream-vad`, or `aed` |

### AudioSlice

Used for slice-based streaming. Each slice represents a contiguous segment of audio.

| Field | Type | Number | Description |
| :--- | :--- | :--- | :--- |
| `index` | `int32` | 1 | Sequential index of the slice (starting at 0) |
| `data` | `bytes` | 2 | Raw PCM audio bytes (s16le) |

---

## Server → Client Messages (`StreamingRecognizeResponse`)

The server responds with a `oneof` stream containing results or errors.

| Field | Type | Description |
| :--- | :--- | :--- |
| `partial` | `PartialResult` | Intermediate transcription for the current segment |
| `final` | `FinalResult` | Final transcription for the segment or stream |
| `error` | `ErrorResult` | Protocol or processing error |
| `slice_vad` | `SliceVad` | VAD metadata for the processed slice |
| `vad_detect` | `VadDetectResult` | Non-streaming VAD detection results |
| `aed_detect` | `AedDetectResult` | Audio Event Detection (AED) results |

### PartialResult

| Field | Type | Number | Description |
| :--- | :--- | :--- | :--- |
| `segment_id` | `string` | 1 | Unique ID for the current audio segment |
| `revision` | `int32` | 2 | Revision number (increments as text is updated) |
| `text` | `string` | 3 | Partial transcription text |
| `confidence` | `float` | 4 | Confidence score (0.0 to 1.0) |
| `start_ms` | `int32` | 5 | Start time relative to segment (ms) |
| `end_ms` | `int32` | 6 | End time relative to segment (ms) |
| `is_final` | `bool` | 7 | Whether this is the last partial for the segment |
| `slice_index` | `int32` | 8 | Index of the slice that triggered this result |
| `slice_m_ms` | - | 9 | **RESERVED** |

### FinalResult

| Field | Type | Number | Description |
| :--- | :--- | :--- | :--- |
| `segment_id` | `string` | 1 | Unique ID for the segment |
| `text` | `string` | 2 | Full transcription text |
| `sentences` | `repeated Sentence` | 3 | List of detected sentences |
| `words` | `repeated Word` | 4 | List of detected words |
| `duration_ms` | `int32` | 5 | Total duration of the audio processed (ms) |
| `language` | `string` | 6 | Detected language (if LID enabled) |
| `language_confidence`| `float` | 7 | Confidence of the language detection |
| `slice_index` | `int32` | 8 | Index of the slice that triggered this result |
| `slice_m_ms` | - | 9 | **RESERVED** |

### ErrorResult

| Field | Type | Description |
| :--- | :--- | :--- |
| `code` | `string` | Machine-readable error code |
| `message` | `string` | Human-readable error description |

### SliceVad

| Field | Type | Number | Description |
| :--- | :--- | :--- | :--- |
| `slice_index` | `int32` | 1 | Index of the corresponding slice |
| `slice_m_ms` | `int32` | 2 | Server-computed speech boundary (ms) within the slice |
| `slice_n_ms` | `int32` | 3 | Total duration (ms) of the processed audio slice |
| `ended_speaking`| `bool` | 4 | True if speech ended within this slice |
| `entirely_speech`| `bool` | 5 | True if the entire slice was detected as speech |

### VadDetectResult

| Field | Type | Number | Description |
| :--- | :--- | :--- | :--- |
| `slice_index` | `int32` | 1 | Index of the corresponding slice |
| `duration_s` | `float` | 2 | Duration of the audio in seconds |
| `timestamps` | `repeated VadTimestamp` | 3 | Detected speech intervals |

### AedDetectResult

| Field | Type | Number | Description |
| :--- | :--- | :--- | :--- |
| `slice_index` | `int32` | 1 | Index of the corresponding slice |
| `duration_s` | `float` | 2 | Duration of the audio in seconds |
| `events` | `repeated AudioEvent` | 3 | Detected audio events (e.g., laughter, applause) |

---

## Sub-messages

### Sentence

| Field | Type | Number | Description |
| :--- | :--- | :--- | :--- |
| `text` | `string` | 1 | Sentence text |
| `start_ms` | `int32` | 2 | Start time relative to segment (ms) |
| `end_ms` | `int32` | 3 | End time relative to segment (ms) |
| `confidence` | `float` | 4 | Confidence score |

### Word

| Field | Type | Number | Description |
| :--- | :--- | :--- | :--- |
| `text` | `string` | 1 | Word text |
| `start_ms` | `int32` | 2 | Start time relative to segment (ms) |
| `end_ms` | `int32` | 3 | End time relative to segment (ms) |
| `confidence` | `float` | 4 | Confidence score |

### VadTimestamp

| Field | Type | Number | Description |
| :--- | :--- | :--- | :--- |
| `start_s` | `float` | 1 | Start time in seconds |
| `end_s` | `float` | 2 | End time in seconds |

### AudioEvent

| Field | Type | Number | Description |
| :--- | :--- | :--- | :--- |
| `event_type` | `string` | 1 | Type of event (e.g., "speech", "noise") |
| `timestamps` | `repeated VadTimestamp` | 2 | Intervals where the event occurred |
| `ratio` | `float` | 3 | Ratio of event duration to slice duration |

---

## Protocol Flow

The client must follow a strict message sequence: `Config` → `AudioSlice*` → `end_stream`.

```text
Client                                Server
  |                                     |
  |  Config: {slice_index: 0, ...}      |
  | ─────────────────────────────────→  |
  |                                     |
  |  Audio slice [index: 0, data: ...]  |
  | ─────────────────────────────────→  |
  |                                     |
  |  slice_vad: {slice_index: 0, ...}   |
  | ←─────────────────────────────────  |
  |                                     |
  |  Partial: {text: "Hello", ...}      |
  | ←─────────────────────────────────  |
  |                                     |
  |  Audio slice [index: 1, data: ...]  |
  | ─────────────────────────────────→  |
  |                                     |
  |  End stream: true                   |
  | ─────────────────────────────────→  |
  |                                     |
  |  Final: {text: "Hello world.", ...} |
  | ←─────────────────────────────────  |
  |                                     |
```

### Slice-based VAD Flow
The server computes a speech boundary `slice_m_ms` for each slice. The client uses this value to determine where the next slice should start in its absolute audio timeline, effectively "advancing" the window based on server-side VAD.

---

## Protocol Constraints

*   **Strict Ordering**: `config` must be sent before any `audio_slice`. Sending audio before config results in a `NO_CONFIG` error.
*   **Contiguous Indices**: `audio_slice.index` must start at 0 (or the index specified in `RecognitionConfig.slice_index`) and increment sequentially by 1.
*   **Slice Duration**:
    *   **Minimum**: 200ms. Slices shorter than 200ms result in `SLICE_TOO_SHORT`.
    *   **Maximum**: 30,000ms. Slices longer than 30s result in `SLICE_TOO_LONG`.
*   **Sample Rate**: Defaults to 16,000 Hz if not specified.

---

## Error Codes

| Code | Description |
| :--- | :--- |
| `MISSING_SLICE_INDEX` | `slice_index` or `audio_slice.index` is missing or negative. |
| `SLICE_TOO_SHORT` | Audio slice duration is below the 200ms minimum. |
| `INVALID_VAD_TYPE` | `vad_type` is not one of: `vad`, `stream-vad`, `aed`. |
| `VAD_MODEL_UNAVAILABLE` | The requested VAD model is not loaded on the server. |
| `NON_CONTIGUOUS_SLICE` | Slice index skipped or repeated (out of sequence). |
| `NO_CONFIG` | Audio slice sent before configuration. |
| `BACKEND_UNAVAILABLE` | The requested ASR backend is not initialized. |
| `INTERNAL_ERROR` | An unexpected server-side processing error occurred. |

---

## Reserved Fields

The following fields are reserved in the protobuf definition and should not be used:
*   `RecognitionConfig.asr_type` (field 4)
*   `PartialResult.slice_m_ms` (field 9)
*   `FinalResult.slice_m_ms` (field 9)
