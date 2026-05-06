#!/usr/bin/env python3
"""Test P48 exact classification in warp-room after training"""
import json
import subprocess
import os

WARP = "./warp-room"

os.system("rm -f /dev/shm/warp-room-vectors")

r = subprocess.run([WARP, "--infer-p48", "edge"], capture_output=True, text=True)
print(f"Fresh: {r.stdout.strip()}")

train_samples = {
    "edge": [
        "gpu memory bandwidth cuda kernel jetson tensor core edge inference compile",
        "arm64 aarch64 neon simd packed sve reduced precision edge runtime",
    ],
    "research": [
        "theorem proof coq formal verification constraint safety specification",
        "plato tile graph neural topology knowledge embedding representation learning",
    ],
    "fleet": [
        "message bottle protocol container serialize dispatch agent fleet coordination",
        "deadman heartbeat pulse trust election offline successor orphan protocol",
    ],
    "jc1": [
        "jetson orin nano arm64 embedded edge compute inference 8gb unified memory",
        "nvidia-smi smi clock frequency power temperature throttling thermal",
    ],
}

for room_name, samples in train_samples.items():
    for s in samples:
        r = subprocess.run([WARP, "--train", s, room_name], capture_output=True, text=True)

print("\n=== After Training (P48) ===")
tests = [
    ("edge device gpu kernel compile cuda", "edge"),
    ("research paper coq proof theorem verification", "research"),
    ("fleet message bottle protocol deadman heartbeat", "fleet"),
    ("jetson orin nano systemd sensor telemetry", "jc1"),
]

correct = 0
for text, expected in tests:
    r = subprocess.run([WARP, "--infer-p48", text], capture_output=True, text=True)
    result = json.loads(r.stdout.strip())
    ok = result["room"] == expected
    print(f"  {'✓' if ok else '✗'} {expected:8s} -> {result['room']:8s} (dist={result['exact_distance']:5d})")
    if ok:
        correct += 1

print(f"\nP48: {correct}/{len(tests)} correct")
