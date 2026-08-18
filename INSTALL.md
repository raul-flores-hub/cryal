# Installing CrYAL

CrYAL itself is five pure-Python dependencies and installs in a minute. The
work is in the energy backend: a LAMMPS build that can run a MACE-OFF model
through the ML-IAP interface. Everything below assumes Linux with an NVIDIA GPU,
which is what the Kokkos path needs.

## 1. The Python side

```bash
python -m venv cryal-env
source cryal-env/bin/activate
pip install -r requirements.txt
```

That covers `import cryal`. Check it:

```bash
python -m unittest discover -s tests -t .
```

80 tests, no LAMMPS, no GPU, under a second. One is an expected failure and is
supposed to be — it records a known limitation of `unwrap_molecule`.

## 2. LAMMPS with ML-IAP and Kokkos

You need LAMMPS built with the `ML-IAP` and `KOKKOS` packages, plus the `lammps`
Python module importable by whichever interpreter runs it.

**The Kokkos build is not optional.** The MACE ML-IAP interface calls
`forward_exchange`, which exists only on the Kokkos code path. A plain build
fails at the first force evaluation with:

```
AttributeError: 'MLIAPDataPy' object has no attribute 'forward_exchange'
```

And the run command must carry the Kokkos flags, in `INPUT.txt`:

```
lammpsCommand = lmp -k on g 1 -sf kk -pk kokkos newton on neigh half
```

The embedded interpreter inside `lmp` does not find the `lammps` module on its
own. CrYAL handles this for you — `cryal/backends/lammps_mace.py::_lammps_env()` puts the
site-packages path on `PYTHONPATH` for the subprocess — but if you run `lmp` by
hand you need both the binary on `PATH` and:

```bash
export PYTHONPATH=/path/to/your/venv/lib/python3.12/site-packages
```

Without it you get `ModuleNotFoundError: No module named 'lammps'`, reported as
`Loading mliappy unified module failure`.

## 3. The MACE-OFF model

Export a MACE-OFF model for ML-IAP (`*-mliap_lammps.pt`) and point the LAMMPS
input at it. `mace-torch` and `torch` are needed by the interpreter that LAMMPS
embeds, not by CrYAL.

### If an existing .pt file stops loading

An exported model is a *pickled object*: its attributes come from the file, its
class code from the installed `mace` package. When mace-torch adds a field to
that class, older files break — every relaxation fails in seconds and CrYAL
reports `FAILED (LAMMPS or integrity)`, which sounds like bad geometry and is
not.

**Read `AL_results/steps/<structure>/lammps.output`, not `log.lammps`.** The
real error only appears in the former:

```
AttributeError: 'MACEEdgeForcesWrapper' object has no attribute 'total_charge'
```

mace-torch 0.3.16 added `total_charge` and `total_spin`. Reinstalling does not
help — the package is fine, the pickle is old. Patch the file instead:

```bash
python tools/patch_mliap_model.py MACE-OFF23_small.model-mliap_lammps.pt
```

This writes `..._v2.pt` beside it, leaves the original untouched, and changes no
weights. Defaults are neutral and closed-shell; use `--charge/--spin` otherwise.

## 4. Validate before trusting a rebuilt stack

This matters more than it looks. Between June and August 2026 an unrelated
`pip install -U` on the reference machine changed the stack under a project that
nobody had touched, and broke the potential. Environments drift on their own.

Any recalculation that has to be comparable with earlier numbers should be
checked against a deterministic result first: relax a structure whose energy you
know with `box/relax` plus `minimize` and **no MD**, then compare. When the
patched model above was validated this way it agreed to 0.88 meV per unit cell —
about 0.021 kJ/mol per molecule — and 0.008 % in volume, which is minimizer
noise rather than a change of model.

Do not use an MD-annealed protocol for this check: it is not deterministic.

## Version drift

The pins in `requirements.txt` are the versions verified today, not the ones
that produced the published run. The published results were computed with an
earlier `mace-torch`; the two stacks were compared with the deterministic check
above and agree.

One consequence worth knowing: **a fixed random seed does not reproduce
identical geometries across PyXtal versions.** PyXtal 1.1.3 and NumPy 2.4.2
consume the random stream differently from the 2026-06 releases. The sequence of
space-group draws is reproducible; the geometries placed inside them are not. So
two runs configured identically on different library versions are independent
trajectories, not paired replicas — which is how the surrogate ablation reported
in the article had to be described.

## Reference machine

For calibration, the configuration these instructions were verified on:

| | |
|---|---|
| OS / kernel | Ubuntu 24.04, 6.17 |
| GPU | NVIDIA RTX 4060 Ti, driver 580 |
| Python | 3.12.3 |
| LAMMPS | built with ML-IAP + KOKKOS (CUDA) |
| torch / mace-torch | 2.12.1 / 0.3.16 |

One local hazard, if you are on Ubuntu with this driver series: driver 580 does
not build against 7.x kernels, so an `apt` upgrade that pulls in a new HWE
kernel silently costs you the GPU. Check what `apt` plans to install before
running it.
