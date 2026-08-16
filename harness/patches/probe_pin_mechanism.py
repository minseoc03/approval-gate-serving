#!/usr/bin/env python3
"""
Approach-A probe (CLAUDE.md 13.1.2), disk-free variant: validates the
victim-exclusion mechanism by monkeypatching BlockPool IN-PROCESS only.
Nothing in site-packages is modified.

Reproduces the Resident KV Claims allocator scenario (2605.24259):
an 80-block usable pool, a 60-block resident claim, then a 70-block active
allocation. Stock LRU silently evicts the residents; with victim exclusion
the allocator must refuse instead.

Checks:
  P1  pinned blocks leave the free queue      (free count drops by 60)
  P2  oversized allocation is REFUSED         (ValueError, not silent evict)
  P3  cached hashes survive the refusal       (blocks still hit-able)
  P4  touch() on a pinned block unpins safely (resume path, no queue corruption)
  P5  unpin-all restores the pool             (70-block alloc now succeeds)
CPU-only; run with .venv-vllm python on the login node.
"""

import sys

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock  # noqa: F401


def ss_pin_blocks(pool, req_id, blocks):
    kept, passthrough = [], []
    for b in blocks:
        b.ref_cnt -= 1
        if b.ref_cnt == 0 and not b.is_null:
            if b.block_hash is not None:
                pool.pinned_block_ids.add(b.block_id)
                kept.append(b)
            else:
                passthrough.append(b)
    pool.free_block_queue.prepend_n(passthrough)
    if kept:
        pool.pinned_by_request[req_id] = kept
    return kept


def ss_unpin_all(pool):
    n = 0
    for _req, blocks in list(pool.pinned_by_request.items()):
        live = [b for b in blocks
                if b.block_id in pool.pinned_block_ids and b.ref_cnt == 0]
        pool.free_block_queue.append_n(live)
        for b in live:
            pool.pinned_block_ids.discard(b.block_id)
        n += len(live)
    pool.pinned_by_request.clear()
    pool.pinned_block_ids.clear()
    return n


def patched_touch(pool, blocks):
    for block in blocks:
        if block.ref_cnt == 0 and not block.is_null:
            if block.block_id in pool.pinned_block_ids:
                pool.pinned_block_ids.discard(block.block_id)
            else:
                pool.free_block_queue.remove(block)
        block.ref_cnt += 1


def main():
    # 81 = 80 usable + the null block the pool reserves for itself
    pool = BlockPool(num_gpu_blocks=81, enable_caching=True,
                     hash_block_size=16)
    pool.pinned_block_ids = set()
    pool.pinned_by_request = {}
    assert pool.get_num_free_blocks() == 80, pool.get_num_free_blocks()

    # resident claim: allocate 60, give them fake hashes (cached content),
    # then "finish the request" via the pin path instead of free_blocks
    resident = pool.get_new_blocks(60)
    for i, b in enumerate(resident):
        b._block_hash = ("ss-probe", i)     # non-None marks reusable content
    kept = ss_pin_blocks(pool, "suspended-0", resident)
    free_after_pin = pool.get_num_free_blocks()
    p1 = (len(kept) == 60 and free_after_pin == 20)
    print(f"P1 pinned withheld from queue : {'PASS' if p1 else 'FAIL'} "
          f"(pinned={len(kept)}, free={free_after_pin})")

    # active 70-block allocation against 20 free -> must REFUSE
    try:
        pool.get_new_blocks(70)
        p2, note = False, "silently allocated (residents evicted?)"
    except ValueError as e:
        p2, note = True, f"refused: {e}"
    print(f"P2 oversubscription refused   : {'PASS' if p2 else 'FAIL'} ({note})")

    # cached hashes intact after the refusal
    p3 = all(b.block_hash is not None for b in kept) and \
        len(pool.pinned_block_ids) == 60
    print(f"P3 hashes survive refusal     : {'PASS' if p3 else 'FAIL'}")

    # resume path: touch 10 pinned blocks (prefix hit) — must not corrupt queue
    patched_touch(pool, kept[:10])
    p4 = (len(pool.pinned_block_ids) == 50
          and all(b.ref_cnt == 1 for b in kept[:10])
          and pool.get_num_free_blocks() == 20)
    print(f"P4 touch-on-pinned unpins     : {'PASS' if p4 else 'FAIL'} "
          f"(pinned={len(pool.pinned_block_ids)}, free={pool.get_num_free_blocks()})")
    for b in kept[:10]:                    # release the resumed blocks normally
        b.ref_cnt -= 1
    pool.free_block_queue.append_n(kept[:10])

    # unpin-all restores capacity
    ss_unpin_all(pool)
    free_final = pool.get_num_free_blocks()
    try:
        got = pool.get_new_blocks(70)
        p5 = len(got) == 70
    except ValueError:
        p5 = False
    print(f"P5 unpin restores pool        : {'PASS' if p5 else 'FAIL'} "
          f"(free before alloc={free_final})")

    ok = all([p1, p2, p3, p4, p5])
    print("\nMECHANISM " + ("CONFIRMED — Approach B is worth building"
                            if ok else "BROKEN — do not proceed to B"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
