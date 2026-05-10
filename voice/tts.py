"""
Phase 1.6 — IBM Watson Text to Speech playback.

Uses WebSocket synthesis so binary audio arrives incrementally; we push chunks to a queue and
write to PyAudio **as soon as the WAV header is parsed and PCM begins** (no waiting for the full
HTTP body — HTTP ``synthesize`` buffers the entire response first).

Environment (``sensus/.env``):
  IBM_TTS_API_KEY — IAM API key (Speech to Text key does not work — use TTS instance credentials)
  IBM_TTS_URL     — e.g. https://api.us-south.text-to-speech.watson.cloud.ibm.com
Optional:
  IBM_TTS_VOICE   — default en-US_AllisonV3Voice
"""

from __future__ import annotations

import io
import logging
import os
import queue
import struct
import threading
import wave
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pyaudio
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson import TextToSpeechV1
from ibm_watson.websocket import SynthesizeCallback

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "en-US_AllisonV3Voice"
DEFAULT_ACCEPT_WS = "audio/wav"
_SENTINEL = object()
_tts_lock = threading.Lock()
_tts_stop_event = threading.Event()
_current_tts_thread: threading.Thread | None = None
_current_audio_q: queue.Queue | None = None


def _is_normal_ws_close_error(error: object) -> bool:
    """Return True when Watson reports a normal WebSocket close frame/code 1000."""
    text = str(error).lower()
    return (
        "opcode=8" in text
        or "close frame" in text
        or "\\x03\\xe8" in text
        or "1000" in text
        or "normal closure" in text
    )


def _try_load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def is_speaking() -> bool:
    return _tts_lock.locked()


def wait_until_idle() -> None:
    _tts_lock.acquire()
    _tts_lock.release()


def stop() -> None:
    _tts_stop_event.set()
    q = _current_audio_q
    if q is not None:
        q.put(_SENTINEL)


@lru_cache(maxsize=1)
def load_tts_env() -> tuple[str, str, str]:
    """Return (api_key, service_url, voice)."""
    _try_load_dotenv()
    key = os.environ.get("IBM_TTS_API_KEY", "").strip()
    url = os.environ.get("IBM_TTS_URL", "").strip()
    voice = os.environ.get(
        "IBM_TTS_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE
    if not key or not url:
        raise RuntimeError(
            "Set IBM_TTS_API_KEY and IBM_TTS_URL (IBM Cloud → Text to Speech → credentials)."
        )
    return key, url, voice


def _pa_format(sample_width: int) -> int:
    if sample_width == 1:
        return pyaudio.paUInt8
    if sample_width == 2:
        return pyaudio.paInt16
    if sample_width == 4:
        return pyaudio.paInt32
    raise ValueError(f"Unsupported WAV sample width: {sample_width}")


def try_parse_wav_pcm_start(buf: bytes) -> Optional[tuple[int, int, int, int]]:
    """
    If ``buf`` contains a standard PCM WAV header through the start of the ``data`` chunk,
    return (sample_rate, channels, sample_width_bytes, pcm_byte_offset).
    Returns None if headers are incomplete or invalid.
    """
    if len(buf) < 12:
        return None
    if buf[:4] != b"RIFF" or buf[8:12] != b"WAVE":
        return None
    i = 12
    fmt_info: Optional[tuple[int, int, int]] = None
    while i + 8 <= len(buf):
        cid = buf[i: i + 4]
        sz = struct.unpack_from("<I", buf, i + 4)[0]
        chunk_total = 8 + sz
        chunk_end = i + chunk_total

        if chunk_end > len(buf):
            if cid == b"data" and fmt_info is not None and len(buf) >= i + 8:
                rate, n_ch, width = fmt_info
                return (rate, n_ch, width, i + 8)
            return None

        chunk_data = buf[i + 8: chunk_end]
        if cid == b"fmt ":
            if len(chunk_data) < 16:
                return None
            audio_fmt, n_ch, rate, _, _, bits = struct.unpack(
                "<HHIIHH", chunk_data[:16])
            if audio_fmt != 1:
                raise ValueError(
                    "Only linear PCM WAV from Watson is supported")
            fmt_info = (rate, n_ch, bits // 8)
        elif cid == b"data":
            if fmt_info is None:
                return None
            rate, n_ch, width = fmt_info
            return (rate, n_ch, width, i + 8)
        i = chunk_end
    return None


class _QueueSynthesizeCallback(SynthesizeCallback):
    def __init__(self, audio_q: queue.Queue[bytes], errors: list[BaseException]) -> None:
        super().__init__()
        self._q = audio_q
        self._errors = errors

    def on_connected(self) -> None:
        logger.info("Watson TTS WebSocket connected")

    def on_audio_stream(self, audio_stream: bytes) -> None:
        self._q.put(audio_stream)

    def on_error(self, error: object) -> None:
        if _is_normal_ws_close_error(error):
            logger.debug("Watson TTS WebSocket closed normally: %s", error)
            return
        logger.error("Watson TTS WebSocket error: %s", error)
        self._errors.append(RuntimeError(str(error)))

    def on_close(self) -> None:
        logger.debug("Watson TTS WebSocket closed")
        self._q.put(_SENTINEL)

    def on_content_type(self, content_type: str) -> None:
        logger.debug("Watson TTS content-type: %s", content_type)

    def on_timing_information(self, timing_information: object) -> None:
        pass

    def on_data(self, data: object) -> None:
        pass


def _playback_worker(
    audio_q: queue.Queue,
    errors: list[BaseException],
    stop_event: threading.Event,
) -> None:
    """Drain queued WAV/PCM chunks and play via PyAudio as data arrives."""
    header_buf = bytearray()
    rate: Optional[int] = None
    channels: Optional[int] = None
    width: Optional[int] = None
    pa: Optional[pyaudio.PyAudio] = None
    stream: Optional[pyaudio.Stream] = None
    # Coalesce small WebSocket chunks before stream.write(); tiny writes plus
    # network jitter empty the PortAudio buffer and sound like static / cutouts.
    pcm_write_acc = bytearray()
    min_write_bytes = 0
    # Pre-buffer so the output device does not underrun while the WebSocket
    # catches up (avoids static / choppy first syllables on some systems).
    prebuffer_ms = int(os.environ.get("SENSUS_TTS_PREBUFFER_MS", "400"))
    min_write_ms = int(os.environ.get("SENSUS_TTS_MIN_WRITE_MS", "80"))
    frames_per_buffer = int(os.environ.get(
        "SENSUS_TTS_FRAMES_PER_BUFFER", "4096"))

    def _flush_pcm_acc() -> None:
        if stream and pcm_write_acc:
            stream.write(bytes(pcm_write_acc))
            pcm_write_acc.clear()

    def _write_coalesced(data: bytes) -> None:
        pcm_write_acc.extend(data)
        while len(pcm_write_acc) >= min_write_bytes:
            stream.write(bytes(pcm_write_acc[:min_write_bytes]))
            del pcm_write_acc[:min_write_bytes]

    try:
        while True:
            chunk = audio_q.get()
            if stop_event.is_set():
                _flush_pcm_acc()
                break
            if chunk is _SENTINEL:
                _flush_pcm_acc()
                break

            if stream is None:
                header_buf.extend(chunk)
                parsed = try_parse_wav_pcm_start(bytes(header_buf))
                if parsed is None:
                    continue
                rate, channels, width, pcm_off = parsed
                pa = pyaudio.PyAudio()
                stream = pa.open(
                    format=_pa_format(width),
                    channels=channels,
                    rate=rate,
                    output=True,
                    frames_per_buffer=frames_per_buffer,
                )
                min_write_bytes = max(
                    (rate * channels * width * min_write_ms) // 1000,
                    frames_per_buffer * channels * width,
                    width * channels * 512,
                )
                # Leading digital silence lets the DAC/Bluetooth settle so the first
                # word is not scratchy (SENSUS_TTS_LEADING_SILENCE_MS=0 to disable).
                lead_ms = int(os.environ.get("SENSUS_TTS_LEADING_SILENCE_MS", "90"))
                if lead_ms > 0:
                    lead_bytes = (rate * channels * width * lead_ms) // 1000
                    if lead_bytes > 0:
                        stream.write(b"\x00" * lead_bytes)
                pcm_acc = bytearray(header_buf[pcm_off:])
                header_buf.clear()
                pre_target = max(
                    (rate * channels * width * prebuffer_ms) // 1000,
                    width * channels * 256,
                    min_write_bytes,
                )
                while len(pcm_acc) < pre_target:
                    c2 = audio_q.get()
                    if stop_event.is_set() or c2 is _SENTINEL:
                        if pcm_acc:
                            stream.write(bytes(pcm_acc))
                        _flush_pcm_acc()
                        return
                    pcm_acc.extend(c2)
                stream.write(bytes(pcm_acc))
                continue

            if stream is not None:
                if stop_event.is_set():
                    _flush_pcm_acc()
                    break
                _write_coalesced(chunk)

    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                logger.debug("PyAudio stream close: %s", e)
        if pa:
            try:
                pa.terminate()
            except Exception as e:
                logger.debug("PyAudio terminate: %s", e)


def speak(
    text: str,
    *,
    voice: Optional[str] = None,
    api_key: Optional[str] = None,
    service_url: Optional[str] = None,
) -> None:
    """
    Speak ``text`` through the default output device.

    Streams Watson WebSocket audio into PyAudio so playback starts as soon as PCM blocks arrive
    after the WAV header (low perceived latency vs buffering the full HTTP ``synthesize`` body).
    """
    t = (text or "").strip()
    if not t:
        return

    _tts_lock.acquire()
    global _current_tts_thread, _current_audio_q
    k, url, v_default = load_tts_env()
    k = api_key or k
    url = (service_url or url).strip().rstrip("/")
    voice_name = (voice or v_default).strip() or DEFAULT_VOICE

    auth = IAMAuthenticator(k)
    service = TextToSpeechV1(authenticator=auth)
    service.set_service_url(url)

    audio_q: queue.Queue = queue.Queue()
    errors: list[BaseException] = []
    _tts_stop_event.clear()
    _current_tts_thread = threading.current_thread()
    _current_audio_q = audio_q

    cb = _QueueSynthesizeCallback(audio_q, errors)

    player = threading.Thread(
        target=_playback_worker,
        args=(audio_q, errors, _tts_stop_event),
        name="sensus-tts-playback",
        daemon=True,
    )
    player.start()

    def run_ws() -> None:
        try:
            service.synthesize_using_websocket(
                t,
                cb,
                accept=DEFAULT_ACCEPT_WS,
                voice=voice_name,
            )
        except Exception as e:
            if _is_normal_ws_close_error(e):
                logger.debug(
                    "Watson TTS websocket ended with normal close: %s", e)
                audio_q.put(_SENTINEL)
                return
            logger.exception("Watson TTS WebSocket failed")
            errors.append(e)
            audio_q.put(_SENTINEL)

    ws_thread = threading.Thread(
        target=run_ws, name="sensus-tts-ws", daemon=True)
    ws_thread.start()
    try:
        while ws_thread.is_alive():
            ws_thread.join(timeout=0.1)
            if _tts_stop_event.is_set():
                break

        while player.is_alive():
            player.join(timeout=0.1)
            if _tts_stop_event.is_set():
                break

        if errors and not _tts_stop_event.is_set():
            raise errors[0]
    finally:
        _current_audio_q = None
        _current_tts_thread = None
        _tts_stop_event.clear()
        _tts_lock.release()


def speak_http(
    text: str,
    *,
    voice: Optional[str] = None,
    api_key: Optional[str] = None,
    service_url: Optional[str] = None,
) -> None:
    """
    Fallback: HTTP ``synthesize`` (full WAV buffered, then play). Higher latency; same voice/format.
    """
    t = (text or "").strip()
    if not t:
        return

    k, url, v_default = load_tts_env()
    k = api_key or k
    url = (service_url or url).strip().rstrip("/")
    voice_name = (voice or v_default).strip() or DEFAULT_VOICE

    auth = IAMAuthenticator(k)
    service = TextToSpeechV1(authenticator=auth)
    service.set_service_url(url)

    resp = service.synthesize(text=t, voice=voice_name,
                              accept="audio/wav").get_result()
    raw = resp.content
    bio = io.BytesIO(raw)
    with wave.open(bio, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    pa = pyaudio.PyAudio()
    try:
        stream = pa.open(
            format=_pa_format(sampwidth),
            channels=n_channels,
            rate=framerate,
            output=True,
        )
        stream.write(pcm)
        stream.stop_stream()
        stream.close()
    finally:
        pa.terminate()


def main() -> None:
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Watson TTS smoke test (WebSocket streaming).")
    parser.add_argument("text", nargs="*",
                        default=["Hello from Sensus."], help="Text to speak")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Use HTTP synthesize instead of WebSocket (full buffer before play).",
    )
    args = parser.parse_args()
    phrase = " ".join(args.text).strip()
    try:
        if args.http:
            speak_http(phrase)
        else:
            speak(phrase)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
