"""
Phase 1.2 — IBM Watson Speech to Text (WebSocket streaming + optional REST segments).

Streaming: 16 kHz mono PCM over WebSocket with interim_results for live transcripts.
Uses ibm_watson.websocket.AudioSource(queue, is_buffer=True) per IBM SDK.

Environment:
  IBM_STT_API_KEY — IAM API key
  IBM_STT_URL     — e.g. https://api.us-south.speech-to-text.watson.cloud.ibm.com
Optional:
  IBM_STT_MODEL   — default en-US_BroadbandModel

Optional CLI ``segments`` — pipes Phase 1.1 VAD chunks to HTTP ``recognize`` (no word-by-word
interim). The shipped overlay (Phase 3) should use WebSocket streaming only; ``segments`` is
for debugging / comparing VAD cuts vs live STT.
"""

from __future__ import annotations

import io
import logging
import os
import queue
import threading
import time
import wave
from typing import Any, Callable, List, Optional

import pyaudio
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import AudioSource, RecognizeCallback

from sensus.voice.stt import CHANNELS, SAMPLE_RATE, MicCapture, SpeechSegment

logger = logging.getLogger(__name__)

# IBM expects l16; official examples vary spacing — this form is widely accepted.
CONTENT_TYPE_L16_16K = "audio/l16;rate=16000"

DEFAULT_MODEL = "en-US_BroadbandModel"


def load_stt_env() -> tuple[str, str, str]:
    """Return (api_key, service_url, model)."""
    key = os.environ.get("IBM_STT_API_KEY", "").strip()
    url = os.environ.get("IBM_STT_URL", "").strip()
    model = os.environ.get("IBM_STT_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if not key or not url:
        raise RuntimeError(
            "Set IBM_STT_API_KEY and IBM_STT_URL (IBM Cloud → Speech to Text → credentials)."
        )
    return key, url, model


def _flatten_transcripts(transcripts: List[dict]) -> str:
    parts: List[str] = []
    for t in transcripts:
        if isinstance(t, dict) and t.get("transcript"):
            parts.append(t["transcript"])
    return "".join(parts).strip()


def _speech_service(api_key: str, service_url: str) -> SpeechToTextV1:
    auth = IAMAuthenticator(api_key)
    service = SpeechToTextV1(authenticator=auth)
    service.set_service_url(service_url)
    return service


def transcribe_segment_pcm(
    service: SpeechToTextV1,
    segment: SpeechSegment,
    *,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Send one VAD-flushed utterance (raw PCM) via HTTP recognize — good when you already
    use MicCapture segments instead of live WebSocket streaming.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(segment.pcm_bytes)
    buf.seek(0)
    resp = service.recognize(
        audio=buf,
        content_type="audio/wav",
        model=model,
    ).get_result()
    texts: List[str] = []
    for r in resp.get("results", []) or []:
        alts = r.get("alternatives") or []
        if alts and alts[0].get("transcript"):
            texts.append(alts[0]["transcript"])
    return "".join(texts).strip()


class _StreamingRecognizeCallback(RecognizeCallback):
    def __init__(
        self,
        on_interim: Optional[Callable[[str], None]],
        on_final: Optional[Callable[[str], None]],
    ) -> None:
        super().__init__()
        self._on_interim = on_interim
        self._on_final = on_final

    def on_transcription(self, transcripts: List[dict]) -> None:
        text = _flatten_transcripts(transcripts)
        if text and self._on_final:
            self._on_final(text)

    def on_hypothesis(self, hypothesis: str) -> None:
        if hypothesis and self._on_interim:
            self._on_interim(hypothesis)

    def on_connected(self) -> None:
        logger.info("Watson STT WebSocket connected")

    def on_error(self, error: object) -> None:
        logger.error("Watson STT error: %s", error)

    def on_inactivity_timeout(self, error: object) -> None:
        logger.warning("Watson STT inactivity timeout: %s", error)

    def on_listening(self) -> None:
        logger.info("Watson STT listening for audio")

    def on_data(self, data: object) -> None:
        pass

    def on_close(self) -> None:
        logger.info("Watson STT WebSocket closed")


class StreamingWatsonSTT:
    """
    Stream microphone PCM to Watson STT WebSocket with interim_results.

    Uses a bounded queue + AudioSource(..., is_buffer=True, is_recording=True).
    Call ``stop()`` to end the stream (signals IBM STOP and closes mic).
    """

    def __init__(
        self,
        *,
        api_key: str,
        service_url: str,
        model: str = DEFAULT_MODEL,
        input_device_index: Optional[int] = None,
        interim_results: bool = True,
        inactivity_timeout: int = -1,
        on_interim: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._api_key = api_key
        self._service_url = service_url
        self._model = model
        self._input_device_index = input_device_index
        self._interim_results = interim_results
        self._inactivity_timeout = inactivity_timeout
        self._on_interim = on_interim
        self._on_final = on_final

        self._audio_q: queue.Queue[bytes] = queue.Queue()
        self._audio_source = AudioSource(self._audio_q, is_recording=True, is_buffer=True)
        self._stop = threading.Event()
        self._mic_thread: Optional[threading.Thread] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

    def start(self) -> None:
        if self._mic_thread and self._mic_thread.is_alive():
            return

        service = _speech_service(self._api_key, self._service_url)
        cb = _StreamingRecognizeCallback(self._on_interim, self._on_final)

        def run_ws() -> None:
            try:
                service.recognize_using_websocket(
                    self._audio_source,
                    CONTENT_TYPE_L16_16K,
                    cb,
                    model=self._model,
                    interim_results=self._interim_results,
                    inactivity_timeout=self._inactivity_timeout,
                )
            except Exception:
                logger.exception("Watson STT WebSocket thread failed")

        self._stop.clear()
        self._ws_thread = threading.Thread(target=run_ws, name="sensus-watson-ws", daemon=True)
        self._ws_thread.start()

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self._input_device_index,
            frames_per_buffer=1024,
        )
        self._mic_thread = threading.Thread(target=self._mic_loop, name="sensus-watson-mic", daemon=True)
        self._mic_thread.start()

    def _mic_loop(self) -> None:
        assert self._stream is not None
        while not self._stop.is_set():
            try:
                chunk = self._stream.read(1024, exception_on_overflow=False)
            except Exception as e:
                logger.error("Mic read failed: %s", e)
                break
            self._audio_q.put(chunk)
        self._audio_source.completed_recording()

    def stop(self) -> None:
        """Join mic + WebSocket before PyAudio.terminate() — terminating PA while the mic
        thread is still in read() can segfault (especially on Ctrl+C).
        """
        self._stop.set()
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.debug("Mic stream close: %s", e)
        if self._mic_thread:
            self._mic_thread.join(timeout=5.0)
            self._mic_thread = None
        self._stream = None
        if self._ws_thread:
            self._ws_thread.join(timeout=15.0)
            self._ws_thread = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception as e:
                logger.debug("PyAudio terminate: %s", e)
            self._pa = None


def create_mic_segment_transcriber(
    on_final_text: Callable[[str], None],
    *,
    model: Optional[str] = None,
    **mic_kw: Any,
) -> MicCapture:
    """
    Phase 1.1 + 1.2 (REST): each VAD-flushed segment from :class:`MicCapture` is sent to
    Watson ``recognize`` (HTTP). No WebSocket interim line within a segment — use
    :class:`StreamingWatsonSTT` for live partials.

    ``mic_kw`` is passed to :class:`~sensus.voice.stt.MicCapture` (e.g. ``vad_aggressiveness``).
    """
    key, url, env_model = load_stt_env()
    use_model = model or env_model
    svc = _speech_service(key, url)

    def on_seg(seg: SpeechSegment) -> None:
        try:
            text = transcribe_segment_pcm(svc, seg, model=use_model)
            if text:
                on_final_text(text)
        except Exception:
            logger.exception("Segment transcription failed")

    return MicCapture(on_segment=on_seg, **mic_kw)


def main_segments() -> None:
    """VAD segments → Watson REST (matches your Phase 1.1 segment logs). Ctrl+C to stop."""
    logging.basicConfig(level=logging.INFO)

    def on_final(t: str) -> None:
        print(f"[final] {t}")

    cap = create_mic_segment_transcriber(on_final)
    cap.start()
    print("Listening… VAD segments go to Watson HTTP recognize. Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()


def main() -> None:
    """Live WebSocket transcription (interim + final). Ctrl+C to stop."""
    logging.basicConfig(level=logging.INFO)

    key, url, model = load_stt_env()

    def interim(t: str) -> None:
        print(f"\r[partial] {t}", end="", flush=True)

    def final(t: str) -> None:
        print(f"\n[final]   {t}")

    session = StreamingWatsonSTT(
        api_key=key,
        service_url=url,
        model=model,
        on_interim=interim,
        on_final=final,
    )
    session.start()
    print("Streaming to Watson STT… speak freely. Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        session.stop()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "segments":
        main_segments()
    else:
        main()
