import struct
import time

from pathlib import Path

import pytest
from utils import *

server: ServerProcess

# fixed .dkv header prefix, must match server-task.cpp (DKV2):
#   off 0: u32 magic | off 4: u64 compat_hash | off 12: u32 hits | off 16: i64 last_hit_unix
DKV2_HEADER = struct.Struct("<IQIq")
DKV2_MAGIC = 0x32564B44  # "DKV2"


def read_dkv_headers(cache_dir: str) -> list[tuple[int, int, int, int]]:
    headers = []
    for f in sorted(Path(cache_dir).glob("*.dkv")):
        data = f.read_bytes()
        assert len(data) >= DKV2_HEADER.size, f"truncated cache entry {f}"
        headers.append(DKV2_HEADER.unpack_from(data))
    return headers


@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    server = ServerPreset.tinyllama2()
    server.temperature = 0.0
    server.cache_ram = 256                          # large enough that the entry stays in the RAM tier until shutdown
    server.cache_disk = str(tmp_path / "dkv")
    server.cache_disk_min_tokens = 8                # default (2048) would never persist these tiny prompts


def test_cache_disk_persists_across_restart(tmp_path):
    global server
    cache_dir = server.cache_disk
    log1 = tmp_path / "run1.log"
    log2 = tmp_path / "run2.log"

    # --- run 1: populate the cache, then shut down gracefully ---
    server.log_path = str(log1)
    server.start()

    # n_predict is kept small: the cached entry's token vector includes the generated tokens,
    # and a disk hit requires f_keep = lcp/entry_len >= 0.25 when the prompt is replayed
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 0,
        "cache_prompt": True,
        "n_predict": 4,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 21  # fully processed (nothing cached yet)

    # a different prompt on the same slot displaces the first one into the RAM prompt cache
    res = server.make_request("POST", "/completion", data={
        "prompt": "Tell me a story about a brave little cat named Whiskers",
        "id_slot": 0,
        "cache_prompt": True,
        "n_predict": 4,
    })
    assert res.status_code == 200

    # graceful stop (SIGTERM): the prompt cache dtor must spill the RAM tier to disk
    server.stop()
    assert "prompt cache: shutdown spill queued" in log1.read_text()

    headers = read_dkv_headers(cache_dir)
    assert len(headers) >= 1
    now = time.time()
    for magic, _compat, hits, last_hit in headers:
        assert magic == DKV2_MAGIC
        assert hits == 0                     # freshly written entries start with no hits
        assert 0 < last_hit <= now + 60      # last_hit_unix initialized to the write time

    # --- run 2: restart, replay the first prompt -> must restore from disk and persist the hit ---
    server.log_path = str(log2)
    server.start()

    # NO id_slot pin here: selecting a slot by id bypasses the prompt-cache
    # save/load path entirely (only LRU/similarity selection sets update_cache) —
    # the restore can only fire on an unpinned request
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "cache_prompt": True,
        "n_predict": 4,
    })
    assert res.status_code == 200
    # the prefix was restored from the disk tier, so only the tail is re-processed
    assert res.body["timings"]["prompt_n"] < 21

    server.stop()
    assert "disk prompt cache: restored" in log2.read_text()

    # the hit was rewritten in place in the .dkv header (survives any further restart)
    headers2 = read_dkv_headers(cache_dir)
    assert len(headers2) >= 1
    hit_entries = [h for h in headers2 if h[2] >= 1]
    assert len(hit_entries) == 1, f"expected exactly one entry with a persisted hit, got {headers2}"
    _, _, hits, last_hit = hit_entries[0]
    assert hits == 1
    assert last_hit >= min(h[3] for h in headers)  # touched at hit time
