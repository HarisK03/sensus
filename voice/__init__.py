"""Voice pipeline: mic capture, STT, TTS.

Watson helpers are loaded lazily so ``python -m sensus.voice.watson_stt`` does not
pre-import ``watson_stt`` via this package (avoids runpy double-import warnings).
"""

from __future__ import annotations

from typing import Any

from sensus.voice.stt import MicCapture

__all__ = [
    "MicCapture",
    "StreamingWatsonSTT",
    "create_mic_segment_transcriber",
    "load_stt_env",
    "transcribe_segment_pcm",
    "speak",
    "speak_http",
    "load_tts_env",
]

_LAZY = {
    "StreamingWatsonSTT",
    "create_mic_segment_transcriber",
    "load_stt_env",
    "transcribe_segment_pcm",
    "speak",
    "speak_http",
    "load_tts_env",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        if name in ("speak", "speak_http", "load_tts_env"):
            from sensus.voice import tts as _t

            return getattr(_t, name)
        from sensus.voice import watson_stt as _w

        return getattr(_w, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
