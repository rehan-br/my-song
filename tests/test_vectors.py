"""Tests for the Parquet embedding store."""

import numpy as np

from storage.vectors import EmbeddingRow, read_embeddings, write_embeddings


def test_write_then_read_roundtrip(tmp_path) -> None:
    path = tmp_path / "mert_song.parquet"
    rows = [
        EmbeddingRow("t1", np.array([1, 2, 3, 4], dtype=np.float32), "mert", "h1"),
        EmbeddingRow("t2", np.array([5, 6, 7, 8], dtype=np.float32), "mert", "h1"),
    ]
    assert write_embeddings(path, rows) == 2

    loaded = read_embeddings(path)
    assert set(loaded) == {"t1", "t2"}
    assert loaded["t1"].config_hash == "h1"
    assert loaded["t1"].model == "mert"
    assert loaded["t1"].embedding.dtype == np.float32
    np.testing.assert_array_equal(loaded["t2"].embedding, [5, 6, 7, 8])


def test_read_missing_shard_returns_empty(tmp_path) -> None:
    assert read_embeddings(tmp_path / "absent.parquet") == {}


def test_write_overwrites_whole_shard(tmp_path) -> None:
    # write_embeddings is a full overwrite — merging is the caller's job
    path = tmp_path / "s.parquet"
    write_embeddings(path, [EmbeddingRow("t1", np.zeros(2, np.float32), "m", "h1")])
    write_embeddings(path, [EmbeddingRow("t2", np.zeros(2, np.float32), "m", "h2")])
    assert set(read_embeddings(path)) == {"t2"}
