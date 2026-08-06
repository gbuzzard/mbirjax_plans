"""Phase 4 spike 2: the substrate discriminator on 2 real GPUs.

Three measurements (port_plan Phase 4: single-process threads vs one process
per GPU with NCCL):
  a. THREADED dual-GPU scaling: the same warm compiled projector-ish chain on
     one GPU vs two GPUs driven by two python threads.  Scaling well below
     ~1.8x means GIL/dispatch contention rules out single-process threads.
  b. P2P band copy: cuda:0 -> cuda:1 tensor.to() bandwidth at band sizes
     (the forward broadcast's unit transfer), with peer access as configured.
  c. NCCL collectives (only under torchrun --nproc_per_node=2): broadcast and
     reduce at the same band sizes.

Run directly for (a)+(b):   python p4s2_dual_gpu.py
Run under torchrun for (c): torchrun --nproc_per_node=2 p4s2_dual_gpu.py nccl
"""

import os
import sys
import threading
import time

import torch

BAND_MB_SIZES = [(256, 2**20 // 4), (1024, 2**20), (4096, 2**20)]  # label, floats


def chain(dev, n_iter=30, size=1408):
    a = torch.rand((size, size), device=dev)
    b = torch.rand((size, size), device=dev)
    idx = torch.randint(0, size, (size * 4,), device=dev)
    acc = torch.zeros((size, size), device=dev)

    def body():
        # matmul + gather + index_add: the projector op mix in miniature.
        c = a @ b
        g = c.reshape(-1)[idx].reshape(-1, size // 4 * 4 // 4)
        acc.index_add_(0, idx[:size], c[:size])
        return c.sum() + g.sum()

    for _ in range(3):
        body()
    torch.cuda.synchronize(dev)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        body()
    torch.cuda.synchronize(dev)
    return time.perf_counter() - t0


def main_single():
    assert torch.cuda.device_count() >= 2, "need 2 GPUs"
    t1 = chain("cuda:0")
    print(f"one GPU:            {t1:.3f} s")

    results = {}

    def run(dev):
        results[dev] = chain(dev)

    th = [threading.Thread(target=run, args=(f"cuda:{i}",)) for i in (0, 1)]
    t0 = time.perf_counter()
    [t.start() for t in th]
    [t.join() for t in th]
    wall = time.perf_counter() - t0
    print(f"two GPUs, threads:  wall {wall:.3f} s  per-dev "
          f"{results['cuda:0']:.3f}/{results['cuda:1']:.3f} s  "
          f"scaling {2 * t1 / (2 * wall):.2f}x of ideal")

    for label, n in BAND_MB_SIZES:
        x = torch.rand(n, device="cuda:0")
        x.to("cuda:1"); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            y = x.to("cuda:1")
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
        print(f"p2p .to() {label:>5}: {n * 4 / dt / 2**30:7.1f} GiB/s "
              f"({dt * 1e3:.2f} ms)")


def main_nccl():
    import torch.distributed as dist
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    for label, n in BAND_MB_SIZES:
        x = torch.rand(n, device=f"cuda:{rank}")
        for _ in range(3):
            dist.broadcast(x, src=0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            dist.broadcast(x, src=0)
        torch.cuda.synchronize()
        bt = (time.perf_counter() - t0) / 10
        for _ in range(3):
            dist.reduce(x, dst=0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            dist.reduce(x, dst=0)
        torch.cuda.synchronize()
        rt = (time.perf_counter() - t0) / 10
        if rank == 0:
            print(f"nccl {label:>5}: bcast {n * 4 / bt / 2**30:7.1f} GiB/s "
                  f"({bt * 1e3:.2f} ms)  reduce {n * 4 / rt / 2**30:7.1f} GiB/s "
                  f"({rt * 1e3:.2f} ms)")
    dist.destroy_process_group()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "nccl":
        main_nccl()
    else:
        main_single()
