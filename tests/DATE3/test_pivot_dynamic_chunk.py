"""Structural checks for DATE3 runtime Chunk granularity."""

from pathlib import Path

from scalesim.memory.memdomain_runner import load_runner_config
from scalesim.memory.pivot_ca_runner import (
    _atomic_pivot_config,
    _coalesce_epoch,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/MoE/DATE3/joint_prefetch/w4_c4.json"


def test_pivot_recovers_seed_independent_atomic_tiles():
    configured = load_runner_config(CONFIG)
    atomic = _atomic_pivot_config(configured)
    expected = (
        int(configured.payload["topology_provenance"]["tile_size"]) ** 2
        * int(configured.payload["topology_provenance"]["weight_bytes_per_element"])
    )
    assert sum(item.size_bytes for item in atomic.chunks) == sum(
        item.size_bytes for item in configured.chunks
    )
    assert max(item.size_bytes for item in atomic.chunks) <= expected
    assert len(atomic.chunks) > len(configured.chunks)


def test_runtime_chunk_value_is_tiles_per_request():
    atomic = _atomic_pivot_config(load_runner_config(CONFIG))
    stage = [
        item for item in atomic.chunks
        if (item.expert_id, item.ffn_part) == (0, 1)
    ][:8]
    total = sum(item.size_bytes for item in stage)
    for chunk_tiles in (1, 2, 4, 8):
        grouped = _coalesce_epoch(stage, chunk_tiles, 1, "test")
        assert len(grouped) == 8 // chunk_tiles
        assert sum(item.size_bytes for item in grouped) == total
        assert all(item.size_bytes <= chunk_tiles * 256 for item in grouped)
