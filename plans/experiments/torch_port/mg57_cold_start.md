# mg57 and mg58 run record

The finding and its reading are in
`plans/torch_port/active/multigpu_findings.md` §1.48; this file
holds the run detail for both.

## mg57, the cold-start phase split (job 15449106)

* 2026-08-23, node h000, four H100 80GB, 6 minutes 12 seconds by
  sacct, exit 0, instrument health healthy, no findings outside
  the tables.
* The tree: the committed tip 9825a43, rsynced to mbirtorch_src
  with md5 verification and editable-installed by the job.  torch
  2.13.0+cu130, triton 3.7.1.
* The cell is parallel beam (1024, 1008, 992).  The sinogram was
  staged by a SEPARATE process into an npz with an md5 sidecar,
  because a measured arm that forward-projected to build its own
  input would have compiled the kernels it exists to time.  Every
  arm loaded and md5-verified that file.
* Four arms, fresh subprocesses, each owning both compile
  directories (TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR).
  This is load-bearing: the sbatch files in this series export the
  first and leave the second, so without it Triton would have
  reused compiles from earlier jobs and no arm would have been
  cold.  Arm cache states verified on disk: the one-device pair
  went 0 to 57 inductor files and 0 to 353 Triton files on the
  first run and did not move in the cached arm.
* Walls: first_run_n1 49.48 s, cached_n1 26.02 s then 21.17 s
  in-process; first_run_n4 41.11 s, cached_n4 19.51 s then
  9.60 s.  The un-instrumented passes read 21.17 s and 9.49 s, so
  the phase instrument costs nothing measurable at one device and
  about 0.1 s at four.
* Phase table at one device, warm (the second reconstruction of
  cached_n1), seconds: `_vcd_recon` 17.62 of which the iterations
  are 13.40, `initialize_recon` 2.61 of which the partitions are
  0.39, `_initial_error_state` 2.56, outside both phases 0.95,
  `recon_direct` 0.89, `compute_hessian_diagonal` 0.78,
  `_apply_device_policy` 0.00.
* Compile attribution, from the tools' own accounting: dynamo's
  outermost phase 16.07 s (first run, one device), 4.42 s
  (cached), 25.20 s and 8.00 s at four devices; unique graphs 9
  and 36; Triton compiling launches 10 costing 1.19 s and 57
  costing 4.82 s.  Dynamo's phases nest, so the outermost is
  reported and the sum is not.
* Once per process, unchanged across arms: import torch 1.39 s,
  import mbirtorch 0.03 s, CUDA context 0.15 s at one device and
  0.54 s at four, host weights build 2.24 s, staged load with md5
  10.07 s (a harness cost, not a library one).
* Output rows:
  `rows/mg57_cold_start_h000_20260823_055631.jsonl` (md5
  9411d5a61cf8bcc57d46b861b36dcdaa, verified after copy).

## mg58, the setup attribution probe (job 15449170)

* 2026-08-23, one H100, under a minute.  It exists because
  mg57's `initialize_recon` phase did not change with device
  count, which is the signature of host-side work, and naming
  that work needed a direct measurement rather than a reading of
  the source.
* At the same cell, on 3.81 GiB arrays: `np.isfinite` over the
  sinogram 0.502 s and over the weights 0.504 s, both host-side;
  the same check on the device 0.0076 s; the host-to-device
  transfer of the sinogram, which the reconstruction pays anyway,
  0.514 s; `auto_set_regularization_params` 0.320 s.
* The pixel partitions: all eleven granularity levels 0.414 s
  against 0.125 s for the three levels a three-iteration run
  visits (4, 16, 64 from the default granularity list).
* Output: `rows/mg58_setup_probe_h000_20260823.json`, beside
  `mg58_setup_probe.py` and its sbatch.
