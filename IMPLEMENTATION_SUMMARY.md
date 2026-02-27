# FireRedASR2S API Implementation Summary

## Overview
Complete implementation of real-time ASR API for FireRedASR2S with gRPC streaming support.

## Completed Tasks (7/7)

### Task 1: Package Scaffold ✅
- Created `api/` directory structure
- Added `pyproject.toml` with dependencies
- Set up pytest configuration
- Created package structure under `src/fireredasr2s_api/`
- Added test structure

### Task 2: API Contracts ✅
- Defined gRPC proto (`api/protos/asr.proto`)
- Implemented message parser with type safety
- Support for streaming recognize with partial/final results
- **Protocol Update**: Migrated `slice_m_ms` from `partial`/`final` results to a dedicated `SliceVad` event

### Task 8: Slice-Based VAD Protocol ✅
- Introduced `SliceVad` event message (contains `slice_m_ms`, `slice_n_ms`, `ended_speaking`, `entirely_speech`)
- Implemented `AudioSlice` message for client-side slice metadata
- Enforced strict protocol: contiguous `slice_index`, $n \le 30$s per slice
- Removed legacy `slice_m_ms` from `partial` and `final` result messages

### Task 3: Session Manager ✅
- Implemented `StreamingSession` class
- Built state machine (CONNECTING → STREAMING → FINALIZING → DONE)
- 100ms PCM chunk processing
- VAD integration with `FireRedStreamVad.detect_chunk()`
- 500ms silence timeout support
- Max segment length enforcement
- Revisable partial results

### Task 4: AED Backend ✅
- Created `ASRBackend` abstract base class
- Implemented `AEDBackend` for FireRedASR2-AED
- Implemented `LLMBackend` placeholder for future
- Factory function `create_backend()`
- Backend configuration management
- Transcription with timestamp support

### Task 5: gRPC Service ✅
- Implemented `ASRServiceServicer`
- Bidirectional streaming support
- Config message handling
- Audio chunk processing
- End stream handling
- Partial and final result generation
- Error handling

### Task 7: Tests & Docs ✅
- Test configuration (`conftest.py`)
- Session tests (`test_session.py`)
- README with API documentation
- Usage examples
- Configuration guide

### Task 9: LID & Punctuation Support ✅

### Task 10: LLM Backend Support ✅
- Implemented full `LLMBackend` class in `backend.py` (lazy model loading, 40s max audio, no timestamps)
- Added per-request LLM decoding parameters: `decode_min_len`, `repetition_penalty`, `llm_length_penalty`, `temperature`
- Backend Routing: Server-side ASR backend selection via CLI (LLM/AED); client-side selection removed from protocol
- LLM Support: Per-request decoding parameters (`llm_params`) forwarded to backend when LLM is active on server
- Regression tests: `test_llm_backend.py` (21 tests), `test_llm_params.py` (29 tests)
- Documented LLM limitations: 40s max audio, no word-level timestamps, no `confidence` field
- Evidence: `.sisyphus/evidence/task-6-llm-backend.txt`, `.sisyphus/evidence/task-7-readme-llm.txt`
- Implemented `Postprocessor` class in `postprocessing.py`
- Integrated LID/Punc into gRPC streaming path
- Documented CLI arguments `--enable-lid` and `--enable-punc`
- Support for language and lang_confidence in final results
- Punctuation prediction on the final result

## File Structure

```
api/
├── README.md
├── pyproject.toml
├── protos/
│   └── asr.proto
├── src/
│   └── fireredasr2s_api/
│       ├── __init__.py
│       ├── config.py
│       ├── session.py
│       ├── backend.py
│       ├── grpc_server.py
│       ├── postprocessing.py
│       ├── validation.py
│       └── cli.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_session.py
```

## Key Features

1. **gRPC Streaming**: Bidirectional streaming for service-to-service communication
2. **Streaming VAD**: Real-time voice activity detection
3. **Partial Results**: Revisable intermediate transcriptions
4. **LID & Punc**: Automatic Language Identification and Punctuation for final results
5. **AED Backend**: Production-ready AED ASR support
6. **LLM Backend**: Full FireRedASR2-LLM integration with per-request params
7. **TDD**: Test-driven development with pytest

## 模型使用审计

| Model | Used | Evidence | Notes/Impact |
| :--- | :--- | :--- | :--- |
| FireRedASR2-LLM | **YES** | `api/src/fireredasr2s_api/backend.py:149-225` | Runtime loaded via `FireRedAsr2.from_pretrained("llm", ...)` |
| FireRedASR2-AED | **YES** | `api/src/fireredasr2s_api/backend.py:74-146` | Runtime loaded in `backend.py` via `FireRedAsr2.from_pretrained("aed", ...)` |
| FireRedVAD | **YES** | `api/src/fireredasr2s_api/vad_utils.py:46-69` | Runtime loaded in `vad_utils.py` via `FireRedStreamVad.from_pretrained(...)` |
| FireRedLID | **YES** | `api/src/fireredasr2s_api/postprocessing.py:45-60` | Runtime loaded in `postprocessing.py` via `FireRedLid.from_pretrained(...)` |
| FireRedPunc | **YES** | `api/src/fireredasr2s_api/postprocessing.py:62-75` | Runtime loaded in `postprocessing.py` via `FireRedPunc.from_pretrained(...)` |

## Next Steps

1. Run tests: `pytest api/tests/ -v`
2. Install package: `pip install -e api/`
3. Start gRPC server: `python -m fireredasr2s_api.serve`

## Dependencies

- grpcio >= 1.50.0
- grpcio-tools >= 1.50.0
- pydantic >= 2.0.0
- numpy >= 1.26.0
- pytest-asyncio >= 0.21.0
