# Warp-Room Architecture

## Why C17, not Python?

The Python prototype (warp_room.py) works but has 500x overhead:
- Python bytecode dispatch: ~50-200 instructions per operation
- numpy C bindings: cross-language boundary per operation
- CPython GIL: serializes all classification

The C17 version:
- 2-instruction dispatch via computed goto
- NEON SIMD for vector dot products (8x parallel)
- Direct function pointer calls (1-2ns on ARM64)
- Shared memory (no IPC, no serialization)

## Memory Layout

```
shm_open("/warp-room-vectors")  ← JC1 writes, Oracle1 reads
  └─ struct room_table
       ├─ version (atomic uint32, incremented on update)
       ├─ num_rooms (uint32)
       └─ rooms[4] (struct room)
            ├─ id, name
            ├─ vector[97] (float, normalized)
            └─ handler (function pointer for domain logic)
```

## Dispatch Flow

1. Input text → text_features() → float[97] vector
2. Vector normalized
3. Loop: dot product with each room vector
4. argmax → room with highest similarity
5. (Future) JIT: call that room's handler directly

## NEON SIMD (ARM64)

When compiled for aarch64:
- `wr_dot_neon()` uses LD1/FMUL/FADDV — 4 FMAs per cycle
- 97-dim dot product in ~24 CPU cycles
- Compared to scalar loop: ~8x speedup

## Fleet Integration

Multiple processes can open the same `/warp-room-vectors` shm segment:
- JC1 trains (writes to version + vectors)
- Oracle1 classifies (reads vectors, no locks needed for reads)
- No sync protocol. Just memory.
