from __future__ import annotations

import threading
import time
from typing import Any, List

import numpy as np
import sounddevice as sd

from stt.devices import DevicePref, device_name, refresh_devices, resolve_input_device


class Recorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: DevicePref = None,
        blocksize: int = 1024,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.blocksize = blocksize
        self._frames: List[np.ndarray] = []
        self._stream_reports: List[str] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._recording = False
        self.device_name: str = "?"
        self.device_warning: str | None = None
        self.level: float = 0.0  # RMS of the latest chunk, for live meters

    def _callback(self, indata: np.ndarray, frames: int, t: Any, status: Any) -> None:
        del frames, t
        chunk = indata.copy()
        self.level = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
        with self._lock:
            self._frames.append(chunk)
            if status:
                self._stream_reports.append(str(status))

    def start(self) -> None:
        if self._recording:
            raise RuntimeError("already recording")
        self._frames = []
        self._stream_reports = []
        # Re-scan hardware each time so a mic switched in System Settings (or plugged in)
        # since the last recording is picked up without restarting.
        refresh_devices()
        index, self.device_warning = resolve_input_device(self.device)
        self.device_name = device_name(index)
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                device=index,
                blocksize=self.blocksize,
                latency="high",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            self._stream = None
            raise RuntimeError(f"could not open input device: {e}") from e
        self._recording = True

    def stop(self) -> np.ndarray:
        if not self._recording or self._stream is None:
            raise RuntimeError("no recording in progress")
        # Let the high-latency input buffer drain so the last words aren't clipped.
        time.sleep(0.3)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._recording = False
        self.level = 0.0

        with self._lock:
            if not self._frames:
                return np.zeros((0,), dtype=np.float32)
            audio = np.concatenate(self._frames, axis=0)

        if audio.ndim == 2:
            audio = audio[:, 0]
        return audio.astype(np.float32, copy=False)

    def take_stream_reports(self) -> list[str]:
        with self._lock:
            out = self._stream_reports.copy()
            self._stream_reports.clear()
            return out

    @property
    def is_recording(self) -> bool:
        return self._recording
