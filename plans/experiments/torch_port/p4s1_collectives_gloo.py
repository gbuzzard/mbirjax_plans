"""Phase 4 spike 1: the two banded collectives as torch.distributed code,
runnable ANYWHERE via the gloo backend with CPU ranks (the local
multi-device test path, standing in for mbirjax's XLA host-device split).

Validates the SPMD structure of mbirjax's sharding on torch primitives:
  - band BROADCAST (forward): each slice-owner broadcasts its recon band to
    all view-owners, which accumulate per-band forward projections locally.
  - band REDUCE-SCATTER (back): every view-owner holds per-band partial back
    projections; each band is summed across view-owners and lands on its
    slice-owner.
Here the "projection" is a placeholder matmul; the point is the collective
pattern, the shapes, and a value check against a single-process reference.

Run:  <torch python> p4s1_collectives_gloo.py [nranks]   (default 2)
"""

import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

N_SLICES, N_PIXELS, N_VIEWS, N_CH = 32, 300, 24, 40


def band_bounds(n_slices, n_ranks, rank):
    per = (n_slices + n_ranks - 1) // n_ranks
    lo = min(rank * per, n_slices)
    return lo, min(lo + per, n_slices)


def worker(rank, world, results):
    dist.init_process_group("gloo", rank=rank, world_size=world,
                            init_method="tcp://127.0.0.1:29517")
    g = torch.Generator().manual_seed(7)
    recon = torch.rand((N_PIXELS, N_SLICES), generator=g)      # replicated ref
    op = torch.rand((N_VIEWS * N_CH, N_PIXELS), generator=g)   # placeholder projector

    # Ownership: recon slices banded over ranks; views banded over ranks.
    s_lo, s_hi = band_bounds(N_SLICES, world, rank)
    v_lo, v_hi = band_bounds(N_VIEWS, world, rank)
    my_band = recon[:, s_lo:s_hi].contiguous()                 # this rank's slices
    my_op = op.reshape(N_VIEWS, N_CH, N_PIXELS)[v_lo:v_hi].reshape(-1, N_PIXELS)

    # FORWARD: broadcast each slice-owner's band; every view-owner projects it
    # into its own view shard, accumulating over bands.
    sino_shard = torch.zeros((my_op.shape[0], N_SLICES))
    for owner in range(world):
        o_lo, o_hi = band_bounds(N_SLICES, world, owner)
        band = my_band.clone() if owner == rank else \
            torch.empty((N_PIXELS, o_hi - o_lo))
        dist.broadcast(band, src=owner)
        sino_shard[:, o_lo:o_hi] = my_op @ band

    # BACK: each view-owner computes per-band partials; reduce each band onto
    # its slice-owner (gloo has no reduce_scatter_tensor; per-band reduce).
    back_partials = [my_op.T @ sino_shard[:, band_bounds(N_SLICES, world, o)[0]:
                                          band_bounds(N_SLICES, world, o)[1]]
                     for o in range(world)]
    for owner in range(world):
        dist.reduce(back_partials[owner], dst=owner)
    my_back_band = back_partials[rank]

    # Reference on rank 0: single-process equivalents.
    if rank == 0:
        full_sino = op @ recon                                  # (V*C, S)
        ref_back = op.T @ full_sino
        # Gather the sino shards and back bands for comparison.
    gathered_sino = [torch.empty(( (band_bounds(N_VIEWS, world, r)[1]
                                    - band_bounds(N_VIEWS, world, r)[0]) * N_CH,
                                  N_SLICES)) for r in range(world)]
    dist.all_gather(gathered_sino, sino_shard.reshape(-1, N_SLICES))
    gathered_back = [torch.empty((N_PIXELS,
                                  band_bounds(N_SLICES, world, r)[1]
                                  - band_bounds(N_SLICES, world, r)[0]))
                     for r in range(world)]
    dist.all_gather(gathered_back, my_back_band)
    if rank == 0:
        sino_all = torch.cat(gathered_sino, dim=0)
        back_all = torch.cat(gathered_back, dim=1)
        fwd_err = float(torch.max(torch.abs(sino_all - full_sino)))
        back_err = float(torch.max(torch.abs(back_all - ref_back)))
        print(f"nranks={world}  fwd rel_max={fwd_err / float(full_sino.abs().max()):.2e}  "
              f"back rel_max={back_err / float(ref_back.abs().max()):.2e}")
    dist.destroy_process_group()


def main():
    world = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    mp.spawn(worker, args=(world, None), nprocs=world, join=True)


if __name__ == "__main__":
    main()
