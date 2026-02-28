from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StreamingRecognizeRequest(_message.Message):
    __slots__ = ("config", "audio_chunk", "end_stream", "audio_slice")
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    AUDIO_CHUNK_FIELD_NUMBER: _ClassVar[int]
    END_STREAM_FIELD_NUMBER: _ClassVar[int]
    AUDIO_SLICE_FIELD_NUMBER: _ClassVar[int]
    config: RecognitionConfig
    audio_chunk: bytes
    end_stream: bool
    audio_slice: AudioSlice
    def __init__(self, config: _Optional[_Union[RecognitionConfig, _Mapping]] = ..., audio_chunk: _Optional[bytes] = ..., end_stream: bool = ..., audio_slice: _Optional[_Union[AudioSlice, _Mapping]] = ...) -> None: ...

class AudioSlice(_message.Message):
    __slots__ = ("index", "data")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    index: int
    data: bytes
    def __init__(self, index: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class RecognitionConfig(_message.Message):
    __slots__ = ("sample_rate", "format", "enable_timestamps", "beam_size", "slice_index", "decode_min_len", "repetition_penalty", "llm_length_penalty", "temperature")
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    ENABLE_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    BEAM_SIZE_FIELD_NUMBER: _ClassVar[int]
    SLICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    DECODE_MIN_LEN_FIELD_NUMBER: _ClassVar[int]
    REPETITION_PENALTY_FIELD_NUMBER: _ClassVar[int]
    LLM_LENGTH_PENALTY_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    sample_rate: int
    format: str
    enable_timestamps: bool
    beam_size: int
    slice_index: int
    decode_min_len: int
    repetition_penalty: float
    llm_length_penalty: float
    temperature: float
    def __init__(self, sample_rate: _Optional[int] = ..., format: _Optional[str] = ..., enable_timestamps: bool = ..., beam_size: _Optional[int] = ..., slice_index: _Optional[int] = ..., decode_min_len: _Optional[int] = ..., repetition_penalty: _Optional[float] = ..., llm_length_penalty: _Optional[float] = ..., temperature: _Optional[float] = ...) -> None: ...

class StreamingRecognizeResponse(_message.Message):
    __slots__ = ("partial", "final", "error", "slice_vad")
    PARTIAL_FIELD_NUMBER: _ClassVar[int]
    FINAL_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    SLICE_VAD_FIELD_NUMBER: _ClassVar[int]
    partial: PartialResult
    final: FinalResult
    error: ErrorResult
    slice_vad: SliceVad
    def __init__(self, partial: _Optional[_Union[PartialResult, _Mapping]] = ..., final: _Optional[_Union[FinalResult, _Mapping]] = ..., error: _Optional[_Union[ErrorResult, _Mapping]] = ..., slice_vad: _Optional[_Union[SliceVad, _Mapping]] = ...) -> None: ...

class PartialResult(_message.Message):
    __slots__ = ("segment_id", "revision", "text", "confidence", "start_ms", "end_ms", "is_final", "slice_index")
    SEGMENT_ID_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    START_MS_FIELD_NUMBER: _ClassVar[int]
    END_MS_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    SLICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    segment_id: str
    revision: int
    text: str
    confidence: float
    start_ms: int
    end_ms: int
    is_final: bool
    slice_index: int
    def __init__(self, segment_id: _Optional[str] = ..., revision: _Optional[int] = ..., text: _Optional[str] = ..., confidence: _Optional[float] = ..., start_ms: _Optional[int] = ..., end_ms: _Optional[int] = ..., is_final: bool = ..., slice_index: _Optional[int] = ...) -> None: ...

class FinalResult(_message.Message):
    __slots__ = ("segment_id", "text", "sentences", "words", "duration_ms", "language", "language_confidence", "slice_index")
    SEGMENT_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    SENTENCES_FIELD_NUMBER: _ClassVar[int]
    WORDS_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    SLICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    segment_id: str
    text: str
    sentences: _containers.RepeatedCompositeFieldContainer[Sentence]
    words: _containers.RepeatedCompositeFieldContainer[Word]
    duration_ms: int
    language: str
    language_confidence: float
    slice_index: int
    def __init__(self, segment_id: _Optional[str] = ..., text: _Optional[str] = ..., sentences: _Optional[_Iterable[_Union[Sentence, _Mapping]]] = ..., words: _Optional[_Iterable[_Union[Word, _Mapping]]] = ..., duration_ms: _Optional[int] = ..., language: _Optional[str] = ..., language_confidence: _Optional[float] = ..., slice_index: _Optional[int] = ...) -> None: ...

class Sentence(_message.Message):
    __slots__ = ("text", "start_ms", "end_ms", "confidence")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    START_MS_FIELD_NUMBER: _ClassVar[int]
    END_MS_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    text: str
    start_ms: int
    end_ms: int
    confidence: float
    def __init__(self, text: _Optional[str] = ..., start_ms: _Optional[int] = ..., end_ms: _Optional[int] = ..., confidence: _Optional[float] = ...) -> None: ...

class Word(_message.Message):
    __slots__ = ("text", "start_ms", "end_ms", "confidence")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    START_MS_FIELD_NUMBER: _ClassVar[int]
    END_MS_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    text: str
    start_ms: int
    end_ms: int
    confidence: float
    def __init__(self, text: _Optional[str] = ..., start_ms: _Optional[int] = ..., end_ms: _Optional[int] = ..., confidence: _Optional[float] = ...) -> None: ...

class ErrorResult(_message.Message):
    __slots__ = ("code", "message")
    CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    code: str
    message: str
    def __init__(self, code: _Optional[str] = ..., message: _Optional[str] = ...) -> None: ...

class SliceVad(_message.Message):
    __slots__ = ("slice_index", "slice_m_ms", "slice_n_ms", "ended_speaking", "entirely_speech")
    SLICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    SLICE_M_MS_FIELD_NUMBER: _ClassVar[int]
    SLICE_N_MS_FIELD_NUMBER: _ClassVar[int]
    ENDED_SPEAKING_FIELD_NUMBER: _ClassVar[int]
    ENTIRELY_SPEECH_FIELD_NUMBER: _ClassVar[int]
    slice_index: int
    slice_m_ms: int
    slice_n_ms: int
    ended_speaking: bool
    entirely_speech: bool
    def __init__(self, slice_index: _Optional[int] = ..., slice_m_ms: _Optional[int] = ..., slice_n_ms: _Optional[int] = ..., ended_speaking: bool = ..., entirely_speech: bool = ...) -> None: ...


class VadTimestamp(_message.Message):
    __slots__ = ("start_s", "end_s")
    START_S_FIELD_NUMBER: _ClassVar[int]
    END_S_FIELD_NUMBER: _ClassVar[int]
    start_s: float
    end_s: float
    def __init__(self, start_s: _Optional[float] = ..., end_s: _Optional[float] = ...) -> None: ...

class AudioEvent(_message.Message):
    __slots__ = ("event_type", "timestamps", "ratio")
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    RATIO_FIELD_NUMBER: _ClassVar[int]
    event_type: str
    timestamps: _containers.RepeatedCompositeFieldContainer[VadTimestamp]
    ratio: float
    def __init__(self, event_type: _Optional[str] = ..., timestamps: _Optional[_Iterable[_Union[VadTimestamp, _Mapping]]] = ..., ratio: _Optional[float] = ...) -> None: ...

class VadDetectResult(_message.Message):
    __slots__ = ("slice_index", "duration_s", "timestamps")
    SLICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    DURATION_S_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    slice_index: int
    duration_s: float
    timestamps: _containers.RepeatedCompositeFieldContainer[VadTimestamp]
    def __init__(self, slice_index: _Optional[int] = ..., duration_s: _Optional[float] = ..., timestamps: _Optional[_Iterable[_Union[VadTimestamp, _Mapping]]] = ...) -> None: ...

class AedDetectResult(_message.Message):
    __slots__ = ("slice_index", "duration_s", "events")
    SLICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    DURATION_S_FIELD_NUMBER: _ClassVar[int]
    EVENTS_FIELD_NUMBER: _ClassVar[int]
    slice_index: int
    duration_s: float
    events: _containers.RepeatedCompositeFieldContainer[AudioEvent]
    def __init__(self, slice_index: _Optional[int] = ..., duration_s: _Optional[float] = ..., events: _Optional[_Iterable[_Union[AudioEvent, _Mapping]]] = ...) -> None: ...