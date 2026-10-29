"""Pre- and post-processing that keeps Whisper away from silence and phantom text."""

from __future__ import annotations

import re

import numpy as np

_FRAME_S = 0.03
_LEAD_PAD_S = 0.25
_TAIL_PAD_S = 0.5  # generous: Whisper clips words that end right at the buffer edge

# Phrases Whisper emits on silence/noise (training-data artifacts). Only dropped when
# they are the *entire* transcript, so real sentences containing them are untouched.
_PHANTOMS = frozenset(
    {
        "you",
        "thank you",
        "thank you.",
        "thanks",
        "thanks for watching",
        "thank you for watching",
        "thanks for watching!",
        "bye",
        "bye bye",
        "subtitles by the amara.org community",
        "please subscribe",
        "like and subscribe",
        "untertitel der amara.org-community",
        "untertitelung des zdf, 2020",
    }
)


_PHANTOM_KEYS = frozenset(p.strip(".!? ") for p in _PHANTOMS)


def trim_silence(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Drop leading/trailing silence. Returns an empty array if nothing voiced is found."""
    audio = np.asarray(audio, dtype=np.float32)
    frame = max(1, int(sample_rate * _FRAME_S))
    n = len(audio) // frame
    if n == 0:
        return audio
    frames = audio[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    loud = float(np.percentile(rms, 95))
    if loud < 1e-4:
        return audio[:0]
    # Relative to the speech level, with an absolute floor above typical mic hiss.
    voiced = np.flatnonzero(rms > max(1.5e-3, 0.08 * loud))
    if voiced.size == 0:
        return audio[:0]
    start = max(0, int(voiced[0] * frame - _LEAD_PAD_S * sample_rate))
    end = min(len(audio), int((voiced[-1] + 1) * frame + _TAIL_PAD_S * sample_rate))
    return audio[start:end]


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def collapse_repeats(text: str, max_repeats: int = 2) -> str:
    """Safety net for decoding loops: keep at most `max_repeats` identical sentences in a row."""
    parts = _SENTENCE_SPLIT.split(text.strip())
    out: list[str] = []
    run = 0
    prev = None
    for part in parts:
        key = part.strip().lower()
        if key and key == prev:
            run += 1
            if run >= max_repeats:
                continue
        else:
            run = 0
        prev = key
        out.append(part)
    return " ".join(out)


def clean_transcript(text: str) -> str:
    text = collapse_repeats(text.strip())
    if text.strip().lower().strip(".!? ") in _PHANTOM_KEYS:
        return ""
    return text
