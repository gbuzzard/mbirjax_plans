# mg50 run record

One run.  The finding is in
`plans/torch_port/active/multigpu_findings.md` section 1.39, together
with mg49's, because the two runs answer one question.  This file holds
the run detail.

## mg50, the denoiser's output gather (job 15403256)

* Four H100s, 7 minutes 24 seconds of wall, exit 0.  Twelve arms, each
  in a fresh subprocess.
* Why it ran.  mg49 attributed a 1024-class denoise and could not
  account for sixty percent of the call: the sweep read 175 ms and
  setup 1,165 ms of a 3,401 ms call.  The seam list had no entry for
  the output gather, which `denoise` performs on the way out unless the
  caller asks for the device form.  This run turns that residual into a
  measurement.
* The design is a single-variable ablation.  Every arm runs the floors
  protocol -- same model construction, same seed, same staged noisy
  image, three iterations, cold pass discarded, warm median of three --
  and varies only `output_sharded`.  Every arm synchronizes every one
  of its devices before the clock stops, so a mode cannot post a faster
  wall by leaving work in flight.
* The walls, warm medians in seconds:

  | cell | devices | host form | device form | gather |
  | --- | --- | --- | --- | --- |
  | (1024, 1008, 992) | 1 | 2.164 | 1.223 | 0.941 |
  | (1024, 1008, 992) | 2 | 3.307 | 1.299 | 2.009 |
  | (1024, 1008, 992) | 4 | 3.633 | 1.344 | 2.289 |
  | (1664, 1648, 1632) | 1 | 8.513 | 4.283 | 4.230 |
  | (1664, 1648, 1632) | 2 | 12.997 | 4.648 | 8.348 |
  | (1664, 1648, 1632) | 4 | 13.863 | 4.247 | 9.617 |

* The speed verdict inside each mode, against one device: host form
  0.654x and 0.596x at the 1024-class, 0.655x and 0.614x at the
  1664-class; device form 0.942x and 0.910x at the 1024-class, 0.921x
  and 1.009x at the 1664-class.
* The host-form walls reproduce mg49's within 2 percent at every cell
  and count, which is the check that this run measured the same thing
  the ladder did.
* The checksums span 3.8e-8 across the six arms of the 1024-class cell
  and 1.0e-7 across the 1664-class, so every arm reconstructed the same
  volume.  They are computed on device for the sharded arms, so reading
  a result never performs the transfer under test.
* Arm records: `results/mg50_gather/` on scratch (torch_p3,
  purge-eligible); the copy under `rows/` is durable.
