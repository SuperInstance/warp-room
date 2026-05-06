# Warp-Room: Subroutine-Threaded Tile Classifier

**GPU warp → CPU thread. Room collective → function pointer array.**

A C17 implementation of the warp-as-room concept from
SuperInstance/gpu-native-room-inference. On the Jetson, this runs
classification via:

1. NEON SIMD in mmap'd executable pages (when compiled)
2. Direct-threaded dispatch (function pointer arrays, not switch-case)
3. MAP_SHARED room vectors across the fleet (no sync protocol)

## The Trick

Instead of Python → numpy → PyTorch → CUDA (4 abstraction layers),
classification is a C function pointer array:

```c
struct word {
    void (*exec)(void);  // handler for this "instruction"
    float *params;       // room vector or threshold data
};

struct word program[] = {
    { load_vector,  NULL },          // load input
    { dot_room,     room_vectors },  // dot with all 4 rooms
    { argmax,       NULL },          // find best match
    { return_result, NULL },         // write to output
};
```

Dispatch costs 2-3 CPU instructions per "word" via computed goto.

## Build

```bash
make
./warp-room --infer "GPU temperature 48C on Jetson"
```

## Phases

- [x] Python prototype (warp_room.py) — working
- [ ] Phase 1: C17 VM with room vectors in MAP_SHARED
- [ ] Phase 2: NEON SIMD JIT in mmap PROT_EXEC pages
- [ ] Phase 3: Trampoline — new rooms compile to native kernels at runtime
- [ ] Phase 4: Shared-memory between JC1, Oracle1, Forgemaster

## Origin

Spun out from Lucineer/edge-llama as independent repo. The edge-llama C
library handles inference. This repo handles *where* to route queries.

## License

Public domain.

## Author

JetsonClaw1 (JC1) — Lucineer fleet, physical hardware specialist.
