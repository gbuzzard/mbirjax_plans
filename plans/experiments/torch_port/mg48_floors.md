# mg48 run record

One run.  The finding is in
`plans/torch_port/active/multigpu_findings.md` §1.38; the remedy it
gates is findings §1.37; the run detail and the proposed paste are in
this file.  Nothing is pasted into the library by this run: the
sentinel rulings are Greg's.

## mg48, the full floors refresh on the recompile-remedied tree (job 15399595)

* Node h007, four H100s, 1 hour 34 minutes of wall (mg26, the same
  full refresh before the remedy, ran 2 hours 57 minutes).  Exit 0,
  chained behind mg47 on the shared environment.
* The tree is the remedied 8959e32 tip plus the projectors.py remedy,
  verified by the job's own preflight (padding witness, sorted-forward
  witness, the recompile floor, and the compiling-thread form).
* ZERO torch recompile-limit warnings anywhere in the log.  The
  pre-remedy full refresh's log carried 33.  The remedy holds at every
  family and every device count, four devices included.
* The crossover verdicts, per family:
  - multiaxis n=2: 1.525x / 1.461x / 1.515x at the 512-, 768-, and
    1024-class cells; SENTINEL CLEARED, proposed floor at the
    512-class (88,080,384 elements).
  - multiaxis n=4 (now against n=1): 2.027x / 2.185x / 2.167x;
    SENTINEL CLEARED, proposed floor at the 512-class.
  - translation n=2: 0.662x / 1.192x / 1.264x across the
    production-anchored cells; SENTINEL CLEARED, proposed floor at
    (256, 950, 1500) (364,800,000 elements).
  - translation n=4: 0.379x / 0.936x / 1.433x; SENTINEL CLEARED,
    proposed floor at the production cell (1,459,200,000 elements).
  - cone n=2: 0.872x / 1.305x / 1.606x; floor unchanged at the
    512-class.
  - cone n=4: 1.121x / 1.601x; the 768-class stays under the 1.15x
    margin, so the floor stays at the 1024-class.
  - parallel n=2: 1.022x / 1.363x / 1.479x; floor unchanged at the
    768-class.  parallel n=4: 0.828x / 1.336x; floor unchanged at the
    1024-class.
  - denoiser n=2 and n=4: 0.46x to 0.67x; sentinels stay, as expected
    for the one family with no projectors.
* The multiaxis warm walls across counts, in seconds (n1 / n2 / n4):
  11.39 / 7.47 / 5.62 at the 512-class; 56.28 / 38.52 / 25.75 at the
  768-class; 308.37 / 203.51 / 142.28 at the 1024-class.  The family
  is monotone in device count at every measured size.
* Cross-harness agreement: the refresh's ratios match the component
  harness's independent readings within noise -- 1.525 against 1.53
  (multiaxis 512 n=2, mg45), 1.515 against 1.52 (multiaxis 1024 n=2,
  mg47), 1.264 against 1.25 (translation production n=2, mg45).
* One coverage note for the ruling: the sentinel probes are a
  family's top three ladder cells, so the 384-class was not probed.
  The pre-anomaly record (mg22) read a 1.25x multiaxis n=2 win there,
  and the proposed 512-class floor forgoes it.  A ladder extension
  can revisit; the forgone win is bounded by that cell's 3.9 s
  one-device wall.
* Arm records: `results/mg48_floors/` on scratch (torch_p3;
  purge-eligible).  The paste block below is the durable copy.

## The proposed paste, verbatim from the job log

The tool leaves each row's `note` as `'...'` for a person to write,
and its closing instructions (check MEASURED_GPU and MEASURED_CONFIG,
rewrite the notes, then `--bless`) are included at the end.

```
PASTE INTO mbirtorch/_widening_floors.py (FLOORS, then the three
bound constants printed by --bless).  All of it, or none of it.
==============================================================================
FLOORS = {
    ('cone', 2): Floor(
        family='cone', count=2, elements=88_080_384, cell=(512, 448, 384),
        against=1,
        bracket=Bracket(losing_cell=(384, 336, 288), losing_speedup=0.87,
                        winning_cell=(512, 448, 384), winning_speedup=1.30),
        spread=0.007311, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-20', commit='unknown',
        largest_tested=297_271_296,
        note='...'),
    ('cone', 4): Floor(
        family='cone', count=4, elements=1_023_934_464, cell=(1024, 1008, 992),
        against=2,
        bracket=Bracket(losing_cell=(768, 672, 576), losing_speedup=1.12,
                        winning_cell=(1024, 1008, 992), winning_speedup=1.60),
        spread=0.01469, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-20', commit='unknown',
        largest_tested=1_023_934_464,
        note='...'),
    ('denoiser', 2): Floor(
        family='denoiser', count=2, elements=None, cell=None,
        against=1,
        bracket=Bracket(losing_cell=(1024, 1008, 992), losing_speedup=0.67,
                        winning_cell=None, winning_speedup=None),
        spread=0.02768, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-20', commit='unknown',
        largest_tested=1_023_934_464,
        note='...'),
    ('denoiser', 4): Floor(
        family='denoiser', count=4, elements=None, cell=None,
        against=1,
        bracket=Bracket(losing_cell=(1024, 1008, 992), losing_speedup=0.61,
                        winning_cell=None, winning_speedup=None),
        spread=0.05412, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-20', commit='unknown',
        largest_tested=1_023_934_464,
        note='...'),
    ('multiaxis', 2): Floor(
        family='multiaxis', count=2, elements=88_080_384, cell=(512, 448, 384),
        against=1,
        bracket=Bracket(losing_cell=None, losing_speedup=None,
                        winning_cell=(512, 448, 384), winning_speedup=1.52),
        spread=0.006272, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-20', commit='unknown',
        largest_tested=1_023_934_464,
        note='...'),
    ('multiaxis', 4): Floor(
        family='multiaxis', count=4, elements=88_080_384, cell=(512, 448, 384),
        against=1,
        bracket=Bracket(losing_cell=None, losing_speedup=None,
                        winning_cell=(512, 448, 384), winning_speedup=2.03),
        spread=0.006272, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-20', commit='unknown',
        largest_tested=1_023_934_464,
        note='...'),
    ('parallel', 2): Floor(
        family='parallel', count=2, elements=297_271_296, cell=(768, 672, 576),
        against=1,
        bracket=Bracket(losing_cell=(512, 448, 384), losing_speedup=1.02,
                        winning_cell=(768, 672, 576), winning_speedup=1.36),
        spread=0.01263, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-20', commit='unknown',
        largest_tested=1_023_934_464,
        note='...'),
    ('parallel', 4): Floor(
        family='parallel', count=4, elements=1_023_934_464, cell=(1024, 1008, 992),
        against=2,
        bracket=Bracket(losing_cell=(768, 672, 576), losing_speedup=0.83,
                        winning_cell=(1024, 1008, 992), winning_speedup=1.34),
        spread=0.01263, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-20', commit='unknown',
        largest_tested=1_023_934_464,
        note='...'),
    ('translation', 2): Floor(
        family='translation', count=2, elements=364_800_000, cell=(256, 950, 1500),
        against=1,
        bracket=Bracket(losing_cell=(256, 475, 750), losing_speedup=0.66,
                        winning_cell=(256, 950, 1500), winning_speedup=1.19),
        spread=0.01036, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-20', commit='unknown',
        largest_tested=1_459_200_000,
        note='...'),
    ('translation', 4): Floor(
        family='translation', count=4, elements=1_459_200_000, cell=(256, 1900, 3000),
        against=1,
        bracket=Bracket(losing_cell=(256, 950, 1500), losing_speedup=0.94,
                        winning_cell=(256, 1900, 3000), winning_speedup=1.43),
        spread=0.009759, gpu=MEASURED_GPU, config=MEASURED_CONFIG,
        measured='2026-08-20', commit='unknown',
        largest_tested=1_459_200_000,
        note='...'),
}
Check MEASURED_GPU and MEASURED_CONFIG still describe this run, and rewrite each note by hand: a note is the one field a machine cannot fill in.
denoiser: QGGMRFDenoiser sets its sinogram shape equal to its image shape, so a denoiser floor is read in IMAGE VOXELS where every other family reads sinogram elements.  Say so in the row's note.
translation: a translation floor is read in sinogram elements like the other projection families, but its probe cells are not the shared ladder.  They are production-anchored translation scans: a fixed 16x16 translation grid, with the detector and the spacing scaled together.  Say so in the row's note and name the cell, so nobody reads a translation floor as a size on the shared ladder.
then: python dev_scripts/refresh_widening_floors.py --bless
```

## The landing (2026-08-20)

Greg accepted the full proposed table.  The paste is in
`mbirtorch/_widening_floors.py` with hand-written notes, the blessed
hashes and checksum, and the module docstring updated; the two
behavior tests that pinned the old sentinel rulings are re-pinned to
the new floors.  The floors, device-policy, and full suites pass
(599).  The finding is multigpu_findings.md §1.38.
