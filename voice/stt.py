"""
Phase 1.1 — Mic capture for Ubuntu (PipeWire/PulseAudio via PyAudio).

16 kHz mono PCM, webrtcvad aggressiveness 2, speech segments queued on
silence-after-speech. Flush handlers receive raw PCM bytes ready for Watson STT (§1.2).

Ubuntu: ensure mic works (`pw-cli ls` / Settings → Sound) and install PortAudio dev
libs before pip: sudo apt install portaudio19-dev python3-pyaudio
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
import time
from typing import Callable, Optional

import pyaudio
import webrtcvad

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # int16
FRAME_DURATION_MS = 30  # webrtcvad: 10, 20, or 30 ms only

SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
BYTES_PER_FRAME = SAMPLES_PER_FRAME * SAMPLE_WIDTH_BYTES

# End-of-utterance: this many consecutive silent VAD frames after speech triggers a flush.
DEFAULT_END_SILENCE_FRAMES = 12  # ~360 ms at 30 ms/frame

# Optional: ignore very short noise bursts before committing to "speech".
DEFAULT_START_SPEECH_FRAMES = 2


@dataclass(frozen=True)
class SpeechSegment:
    """One flushed utterance: raw 16-bit mono PCM at SAMPLE_RATE Hz."""

    pcm_bytes: bytes


class MicCapture:
    """
    Capture microphone audio, run WebRTC VAD (aggressiveness 2), enqueue completed
    speech segments when silence follows speech.

    Use ``segment_queue`` to consume ``SpeechSegment`` in another thread, and/or
    install ``on_segment`` callbacks for the same events.
    """

    def __init__(
        self,
        *,
        vad_aggressiveness: int = 2,
        end_silence_frames: int = DEFAULT_END_SILENCE_FRAMES,
        start_speech_frames: int = DEFAULT_START_SPEECH_FRAMES,
        input_device_index: Optional[int] = None,
        on_segment: Optional[Callable[[SpeechSegment], None]] = None,
    ) -> None:
        if vad_aggressiveness not in (0, 1, 2, 3):
            raise ValueError("webrtcvad aggressiveness must be 0–3")
        self._vad = webrtcvad.Vad(vad_aggressiveness)
        self._end_silence_frames = end_silence_frames
        self._start_speech_frames = start_speech_frames
        self._input_device_index = input_device_index
        self._on_segment = on_segment

        self.segment_queue: queue.Queue[SpeechSegment] = queue.Queue()
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self._input_device_index,
            frames_per_buffer=SAMPLES_PER_FRAME,
        )
        self._thread = threading.Thread(target=self._capture_loop, name="sensus-mic", daemon=True)
        self._thread.start()
        logger.info("Mic capture started (16 kHz mono, VAD agg=%s)", self._vad)

    def stop(self) -> None:
        self._stop_event.set()
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                logger.debug("Error closing stream: %s", e)
            self._stream = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception as e:
                logger.debug("Error terminating PyAudio: %s", e)
            self._pa = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        logger.info("Mic capture stopped")

    def _emit_segment(self, buf: bytearray) -> None:
        if not buf:
            return
        raw = bytes(buf)
        seg = SpeechSegment(pcm_bytes=raw)
        self.segment_queue.put(seg)
        if self._on_segment:
            try:
                self._on_segment(seg)
            except Exception:
                logger.exception("on_segment callback failed")

    def _capture_loop(self) -> None:
        assert self._stream is not None
        buf = bytearray()
        in_speech = False
        speech_run = 0
        silence_after_speech = 0
        pending = bytearray()

        while not self._stop_event.is_set():
            try:
                data = self._stream.read(SAMPLES_PER_FRAME, exception_on_overflow=False)
            except Exception as e:
                logger.error("Mic read failed: %s", e)
                break

            pending.extend(data)
            while len(pending) >= BYTES_PER_FRAME:
                frame = bytes(pending[:BYTES_PER_FRAME])
                del pending[:BYTES_PER_FRAME]

                try:
                    is_speech = self._vad.is_speech(frame, SAMPLE_RATE)
                except Exception as e:
                    logger.warning("VAD frame rejected: %s", e)
                    continue

                if not in_speech:
                    if is_speech:
                        buf.extend(frame)
                        speech_run += 1
                        if speech_run >= self._start_speech_frames:
                            in_speech = True
                            silence_after_speech = 0
                            speech_run = 0
                    else:
                        speech_run = 0
                        buf.clear()
                else:
                    if is_speech:
                        silence_after_speech = 0
                        buf.extend(frame)
                    else:
                        silence_after_speech += 1
                        if silence_after_speech >= self._end_silence_frames:
                            self._emit_segment(buf)
                            buf.clear()
                            in_speech = False
                            silence_after_speech = 0
                            speech_run = 0

        if buf and in_speech:
            self._emit_segment(buf)


def main() -> None:
    """Quick Ubuntu sanity check: prints segment sizes until Ctrl+C."""
    logging.basicConfig(level=logging.INFO)

    def on_seg(seg: SpeechSegment) -> None:
        n = len(seg.pcm_bytes) // 2
        duration_s = n / SAMPLE_RATE
        print(f"Segment: {n} samples (~{duration_s:.2f}s), {len(seg.pcm_bytes)} bytes PCM")

    cap = MicCapture(on_segment=on_seg)
    cap.start()
    print("Listening… speak, then pause; segments print when you stop. Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        cap.stop()


if __name__ == "__main__":
    main()
