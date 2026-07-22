"""Read WAV segments without loading entire long files into memory."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def read_wav_segment(wav_path: str, start_second: int, end_second: int) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    duration = end_second - start_second
    if duration <= 0:
        return np.zeros((0,), dtype=np.float32), 16000

    try:
        with sf.SoundFile(wav_path, "r") as f:
            sr = int(f.samplerate)
            start_frame = int(start_second * sr)
            frames = int(duration * sr)
            f.seek(start_frame)
            if f.channels != 1:
                audio = f.read(frames, dtype="float32", always_2d=True)
                audio = np.mean(audio, axis=1)
            else:
                audio = f.read(frames, dtype="float32")
        return audio.astype(np.float32), sr
    except (sf.LibsndfileError, OSError, RuntimeError):
        import librosa

        audio, sr = librosa.load(
            wav_path,
            sr=None,
            mono=True,
            offset=float(start_second),
            duration=float(duration),
        )
        return audio.astype(np.float32), int(sr)
