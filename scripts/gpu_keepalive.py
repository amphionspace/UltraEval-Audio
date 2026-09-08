"""Small, interruptible CUDA workload; logs its PID and uses all visible GPUs."""
import argparse
import os
import signal
import time

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-seconds", type=float, default=0.15)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    running = True

    def stop(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    tensors = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            a = torch.randn(1024, 1024, device=f"cuda:{index}", dtype=torch.float16)
            tensors.append((a, torch.empty_like(a)))
    if not tensors:
        raise RuntimeError("No CUDA devices available")
    print(f"pid={os.getpid()} GPUs={len(tensors)} active={args.active_seconds}s interval={args.interval}s", flush=True)
    with torch.inference_mode():
        while running:
            cycle_start = time.monotonic()
            while running and time.monotonic() - cycle_start < args.active_seconds:
                for a, out in tensors:
                    torch.mm(a, a, out=out)
                for index in range(len(tensors)):
                    torch.cuda.synchronize(index)
            time.sleep(max(0, args.interval - (time.monotonic() - cycle_start)))
    print("Stopped", flush=True)


if __name__ == "__main__":
    main()
