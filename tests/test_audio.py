"""Tests for audio decoding + chunking."""

import subprocess

import numpy as np

from extraction.audio import chunk, load_audio


def test_load_audio_decodes_and_resamples(tmp_path) -> None:
    # synthesize a 1-second 44.1kHz tone, then decode it down to 24kHz
    wav = tmp_path / "sine.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-ar",
            "44100",
            str(wav),
        ],
        check=True,
    )
    audio = load_audio(wav, target_sr=24000)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(len(audio) - 24000) < 2400  # ~1s at 24kHz, allow codec slack


def test_load_audio_missing_file(tmp_path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        load_audio(tmp_path / "nope.m4a", target_sr=24000)


def test_chunk_pads_final_window() -> None:
    chunks = chunk(np.ones(2500, dtype=np.float32), 1000)
    assert len(chunks) == 3
    assert all(len(c) == 1000 for c in chunks)
    assert chunks[-1][:500].sum() == 500  # 500 real samples
    assert chunks[-1][500:].sum() == 0  # zero-padded tail


def test_chunk_short_waveform_yields_one_padded_chunk() -> None:
    chunks = chunk(np.ones(10, dtype=np.float32), 1000)
    assert len(chunks) == 1
    assert len(chunks[0]) == 1000
