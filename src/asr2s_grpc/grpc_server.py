# Copyright 2026 FireRedTeam

"""gRPC server implementation for FireRedASR2S API."""

import asyncio
import threading
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import grpc
import numpy as np
from numpy.typing import NDArray
from google.protobuf import json_format

from . import asr_pb2, asr_pb2_grpc
from .backend import create_backend
from .config import ApiConfig, AsrBackendConfig, resolve_asr_model_dir, resolve_vad_model_dirs
from .postprocessing import Postprocessor
from .session import SessionState, StreamingSession
from .validation import validate_llm_params

logger = logging.getLogger(__name__)

from .vad_utils import _SessionVadState, SliceVadResult, compute_slice_m_ms, preload_vad_models


class ASRServiceServicer(asr_pb2_grpc.ASRServiceServicer):
    """gRPC servicer for ASR streaming."""

    def __init__(self, config: ApiConfig):
        self.config = config
        self.sessions: Dict[str, StreamingSession] = {}
        self._session_vad_states: Dict[str, _SessionVadState] = {}
        self.backend: Optional[Any] = None
        self.backends: Dict[str, Any] = {}
        self._backend_lock = threading.Lock()
        self._vad_models: Dict[str, Any] = {}
        self._vad_model_dirs: Dict[str, str] = {}
        self._setup_backend()
        self._setup_vad()
        self._setup_postprocessor()

    def _setup_backend(self) -> None:
        """Setup ASR backend."""
        try:
            backend_config = AsrBackendConfig(
                asr_type=self.config.asr.asr_type,
                model_dir=self.config.asr.model_dir,
                use_gpu=self.config.asr.use_gpu,
                beam_size=self.config.asr.beam_size,
                return_timestamp=self.config.asr.return_timestamp,
            )

            if backend_config.asr_type == "llm":
                primary = create_backend(
                    backend_config.asr_type,
                    backend_config.model_dir,
                    use_gpu=backend_config.use_gpu,
                    config={
                        "decode_min_len": self.config.asr.decode_min_len,
                        "repetition_penalty": self.config.asr.repetition_penalty,
                        "llm_length_penalty": self.config.asr.llm_length_penalty,
                        "temperature": self.config.asr.temperature,
                    },
                )
            else:
                primary = create_backend(
                    backend_config.asr_type,
                    backend_config.model_dir,
                    use_gpu=backend_config.use_gpu,
                    beam_size=backend_config.beam_size,
                    return_timestamps=backend_config.return_timestamp,
                )

            self.backend = primary
            self.backends[backend_config.asr_type] = primary

            logger.info(f"Initialized {backend_config.asr_type} backend")

        except Exception as e:
            logger.error(f"Failed to setup backend: {e}")
            raise


    def _setup_vad(self) -> None:
        """Preload VAD models at server startup."""
        try:
            self._vad_model_dirs = resolve_vad_model_dirs(
                self.config.vad,
                base_dir=self.config.model_base_dir,
            )
            self._vad_models = preload_vad_models(
                self._vad_model_dirs,
                use_gpu=self.config.vad.use_gpu,
            )
        except Exception as e:
            logger.error("Failed to preload VAD models: %s", e)
            # VAD failure is non-fatal; sessions will get empty models
            self._vad_models = {}
            self._vad_model_dirs = {}

    def _setup_postprocessor(self) -> None:
        """Setup LID/Punc postprocessor."""
        try:
            self.postprocessor = Postprocessor(self.config)
            logger.info(
                f"Postprocessor initialized (lid={self.config.enable_lid}, punc={self.config.enable_punc})"
            )
        except Exception as e:
            logger.error(f"Failed to setup postprocessor: {e}")
            self.postprocessor = None

    def _get_backend(self, asr_type: str) -> Optional[Any]:
        """Get the backend for the given asr_type, lazy-loading if needed."""
        backend = self.backends.get(asr_type)
        if backend is not None:
            return backend
        return self._lazy_load_backend(asr_type)

    def _lazy_load_backend(self, asr_type: str) -> Optional[Any]:
        """Lazily load a backend with double-check locking."""
        with self._backend_lock:
            # Double-check after acquiring lock
            backend = self.backends.get(asr_type)
            if backend is not None:
                return backend

            try:
                model_dir = resolve_asr_model_dir(
                    asr_type,
                    self.config.model_base_dir or "",
                    self.config.asr.model_dir,
                )
                if asr_type == "llm":
                    backend = create_backend(
                        asr_type,
                        model_dir,
                        use_gpu=self.config.asr.use_gpu,
                        config={
                            "decode_min_len": self.config.asr.decode_min_len,
                            "repetition_penalty": self.config.asr.repetition_penalty,
                            "llm_length_penalty": self.config.asr.llm_length_penalty,
                            "temperature": self.config.asr.temperature,
                        },
                    )
                else:
                    backend = create_backend(
                        asr_type,
                        model_dir,
                        use_gpu=self.config.asr.use_gpu,
                        beam_size=self.config.asr.beam_size,
                        return_timestamps=self.config.asr.return_timestamp,
                    )
                self.backends[asr_type] = backend
                logger.info(f"Lazy-loaded {asr_type} backend")
                return backend
            except Exception as e:
                logger.error(f"Failed to lazy-load {asr_type} backend: {e}")
                return None

    async def StreamingRecognize(
        self,
        request_iterator: AsyncIterator[asr_pb2.StreamingRecognizeRequest],
        context: grpc.ServicerContext,
    ) -> AsyncIterator[asr_pb2.StreamingRecognizeResponse]:
        session_id = None
        session = None
        session_backend: Optional[Any] = None
        vad_state: Optional[_SessionVadState] = None
        expected_slice_index: Optional[int] = None

        try:
            async for request in request_iterator:
                if request.HasField("config"):
                    config = request.config

                    if config.slice_index < 0:
                        yield self._create_error_response(
                            "MISSING_SLICE_INDEX",
                            "slice_index is required and must be non-negative",
                        )
                        continue

                    asr_type = self.config.asr.asr_type
                    session_backend = self._get_backend(asr_type)
                    if session_backend is None:
                        yield self._create_error_response(
                            "BACKEND_UNAVAILABLE",
                            f"Backend '{asr_type}' is not available on this server",
                        )
                        continue

                    session_id = f"session_{id(context)}"
                    session = self._create_session(session_id, config, session_backend)
                    session.slice_index = config.slice_index
                    session.asr_type = asr_type

                    # Parse LLM params from proto (proto3 zero-value = unset)
                    session.llm_params = validate_llm_params(
                        decode_min_len=config.decode_min_len if config.decode_min_len != 0 else None,
                        repetition_penalty=config.repetition_penalty if config.repetition_penalty != 0.0 else None,
                        llm_length_penalty=config.llm_length_penalty if config.llm_length_penalty != 0.0 else None,
                        temperature=config.temperature if config.temperature != 0.0 else None,
                    )

                    self.sessions[session_id] = session

                    vad_state = _SessionVadState(
                        vad_type=self.config.vad.vad_type,
                        preloaded_models=self._vad_models,
                        model_dirs=self._vad_model_dirs,
                    )
                    self._session_vad_states[session_id] = vad_state

                    logger.info(
                        f"Created session {session_id} slice_index={config.slice_index} asr_type={asr_type}"
                    )
                    expected_slice_index = config.slice_index

                elif request.HasField("audio_chunk"):
                    yield self._create_error_response(
                        "LEGACY_UNSUPPORTED",
                        "audio_chunk is deprecated, use audio_slice",
                    )
                    continue

                elif request.HasField("audio_slice"):
                    if session is None:
                        yield self._create_error_response(
                            "NO_CONFIG",
                            "Config must be sent before audio slices",
                        )
                        continue

                    audio_slice = request.audio_slice

                    if audio_slice.index < 0:
                        yield self._create_error_response(
                            "MISSING_SLICE_INDEX",
                            "audio_slice.index must be non-negative",
                        )
                        continue

                    if (
                        expected_slice_index is not None
                        and audio_slice.index != expected_slice_index
                    ):
                        yield self._create_error_response(
                            "NON_CONTIGUOUS_SLICE",
                            f"expected slice_index {expected_slice_index}, got {audio_slice.index}",
                        )
                        continue

                    session.slice_index = audio_slice.index
                    expected_slice_index = audio_slice.index + 1

                    audio_data = np.frombuffer(audio_slice.data, dtype=np.int16)

                    slice_n_ms = len(audio_data) * 1000 // session.sample_rate
                    if slice_n_ms < 200:
                        yield self._create_error_response(
                            "SLICE_TOO_SHORT",
                            f"audio_slice duration {slice_n_ms}ms is below 200ms minimum",
                        )
                        continue
                    if slice_n_ms > 30000:
                        yield self._create_error_response(
                            "SLICE_TOO_LONG",
                            f"audio_slice duration {slice_n_ms}ms exceeds 30000ms limit",
                        )
                        continue

                    if vad_state is not None:
                        vad_result = vad_state.process_slice_audio(
                            audio_data, session.sample_rate
                        )
                        session.slice_m_ms = compute_slice_m_ms(vad_result)
                        session.advance_global_frames(vad_result.n_frames)

                        slice_vad = asr_pb2.SliceVad(
                            slice_index=session.slice_index,
                            slice_m_ms=session.slice_m_ms,
                            slice_n_ms=slice_n_ms,
                            ended_speaking=vad_result.ended_speaking,
                            entirely_speech=vad_result.entirely_speech,
                        )
                        response = asr_pb2.StreamingRecognizeResponse()
                        response.slice_vad.CopyFrom(slice_vad)
                        yield response

                    partial_results = await self._process_audio_chunk(
                        session, audio_data, session_backend
                    )

                    for result in partial_results:
                        if session.asr_type == "llm" or result.get("confidence", 0.0) > 0.3:
                            yield self._create_partial_response(result, session)
                elif request.HasField("end_stream"):
                    if session is None:
                        yield self._create_error_response(
                            "NO_SESSION",
                            "No active session",
                        )
                        continue

                    final_results = await self._finalize_session(
                        session, session_backend
                    )

                    for result in final_results:
                        yield self._create_final_response(result, session)

                    if session_id is not None:
                        self.sessions.pop(session_id, None)
                        self._session_vad_states.pop(session_id, None)
                    logger.info(f"Closed session {session_id}")
                    break

        except Exception as e:
            logger.exception(f"Streaming error: {e}")
            yield self._create_error_response("INTERNAL_ERROR", str(e))

            if session_id:
                self.sessions.pop(session_id, None)
                self._session_vad_states.pop(session_id, None)

    def _create_session(
        self,
        session_id: str,
        config: asr_pb2.RecognitionConfig,
        backend: Any,
    ) -> StreamingSession:
        from .config import SessionConfig

        session_config = SessionConfig(
            sample_rate=config.sample_rate or 16000,
            chunk_duration_ms=100,
            silence_timeout_ms=500,
            max_segment_duration_ms=int(backend.get_max_audio_length() * 1000),
        )

        return StreamingSession(
            session_id=session_id,
            sample_rate=session_config.sample_rate,
            chunk_duration_ms=session_config.chunk_duration_ms,
            silence_timeout_ms=session_config.silence_timeout_ms,
            max_segment_duration_ms=session_config.max_segment_duration_ms,
            enable_lid=self.config.enable_lid,
            enable_punc=self.config.enable_punc,
        )

    async def _process_audio_chunk(
        self,
        session: StreamingSession,
        audio_data: NDArray[np.int16],
        backend: Any,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        try:
            session.add_audio(audio_data)

            if session.has_complete_segment():
                segment_audio = session.get_segment_audio()

                result = backend.transcribe(
                    segment_audio,
                    sample_rate=session.sample_rate,
                    return_timestamps=True,
                )

                results.append(
                    {
                        "segment_id": session.current_segment_id,
                        "revision": session.get_next_revision(),
                        "text": result.get("text", ""),
                        "confidence": result.get("confidence", 0.0),
                        "words": result.get("words", []),
                    }
                )

                session.mark_segment_processed()

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            session.error(str(e))

        return results

    async def _finalize_session(
        self,
        session: StreamingSession,
        backend: Any,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        try:
            full_audio = session.get_full_audio()
            if full_audio.size > 0:
                result = backend.transcribe(
                    full_audio,
                    sample_rate=session.sample_rate,
                    return_timestamps=True,
                )

                text = result.get("text", "")
                language = ""
                language_confidence = 0.0

                if self.postprocessor is not None:
                    try:
                        language, language_confidence = self.postprocessor.process_lid(
                            full_audio, session.sample_rate
                        )
                    except Exception as e:
                        logger.error(f"LID postprocessing failed: {e}")

                    try:
                        text = self.postprocessor.process_punc(text)
                    except Exception as e:
                        logger.error(f"Punc postprocessing failed: {e}")

                words = result.get("words", [])
                sentences: List[Dict[str, Any]] = []
                if words:
                    sentences.append(
                        {
                            "text": text,
                            "start_ms": words[0].get("start_ms", 0),
                            "end_ms": words[-1].get("end_ms", 0),
                            "confidence": result.get("confidence", 0.0),
                        }
                    )

                duration_ms = len(full_audio) * 1000 // session.sample_rate
                results.append(
                    {
                        "segment_id": session.current_segment_id,
                        "text": text,
                        "confidence": result.get("confidence", 0.0),
                        "words": words,
                        "sentences": sentences,
                        "duration_ms": duration_ms,
                        "is_final": True,
                        "language": language,
                        "language_confidence": language_confidence,
                    }
                )

            session.end_stream()

        except Exception as e:
            logger.error(f"Error finalizing session: {e}")
            session.error(str(e))

        return results

    def _create_partial_response(
        self,
        result: Dict[str, Any],
        session: StreamingSession,
    ) -> asr_pb2.StreamingRecognizeResponse:
        partial = asr_pb2.PartialResult(
            segment_id=result.get("segment_id", ""),
            revision=result.get("revision", 0),
            text=result.get("text", ""),
            confidence=result.get("confidence", 0.0),
            start_ms=result.get("start_ms", 0),
            end_ms=result.get("end_ms", 0),
            slice_index=session.slice_index,
        )

        response = asr_pb2.StreamingRecognizeResponse()
        response.partial.CopyFrom(partial)
        return response

    def _create_final_response(
        self,
        result: Dict[str, Any],
        session: StreamingSession,
    ) -> asr_pb2.StreamingRecognizeResponse:
        sentences = [
            asr_pb2.Sentence(
                text=s.get("text", ""),
                start_ms=s.get("start_ms", 0),
                end_ms=s.get("end_ms", 0),
                confidence=s.get("confidence", 0.0),
            )
            for s in result.get("sentences", [])
        ]

        words = [
            asr_pb2.Word(
                text=w.get("text", ""),
                start_ms=w.get("start_ms", 0),
                end_ms=w.get("end_ms", 0),
                confidence=w.get("confidence", 0.0),
            )
            for w in result.get("words", [])
        ]

        final = asr_pb2.FinalResult(
            segment_id=result.get("segment_id", ""),
            text=result.get("text", ""),
            sentences=sentences,
            words=words,
            duration_ms=result.get("duration_ms", 0),
            language=result.get("language", ""),
            language_confidence=result.get("language_confidence", 0.0),
            slice_index=session.slice_index,
        )

        response = asr_pb2.StreamingRecognizeResponse()
        response.final.CopyFrom(final)
        return response

    def _create_error_response(
        self,
        code: str,
        message: str,
    ) -> asr_pb2.StreamingRecognizeResponse:
        """Create error response."""
        error = asr_pb2.ErrorResult(
            code=code,
            message=message,
        )

        response = asr_pb2.StreamingRecognizeResponse()
        response.error.CopyFrom(error)
        return response


async def serve(
    config: ApiConfig,
    host: Optional[str] = None,
    port: Optional[int] = None,
) -> None:
    """
    Start gRPC server.

    Args:
        config: API configuration
        host: Host address (default from config)
        port: Port number (default from config)
    """
    host = host or config.host
    port = port or config.grpc_port

    server = grpc.aio.server(
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )

    asr_pb2_grpc.add_ASRServiceServicer_to_server(
        ASRServiceServicer(config),
        server,
    )

    server.add_insecure_port(f"{host}:{port}")
    await server.start()

    logger.info(f"gRPC server started on {host}:{port}")

    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        await server.stop(5)
