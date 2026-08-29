# Installing CrYAL

CrYAL itself is five pure-Python dependencies and installs in a minute. The
work is in the energy backend: a LAMMPS build that can run a MACE-OFF model
through the ML-IAP interface. Everything below assumes Linux with an NVIDIA GPU,
which is what the Kokkos path needs.

## 1. The Python side

```bash
python -m venv cryal-env
source cryal-env/bin/activate
pip install -r requirements.txt      # the verified pins
# or, to install CrYAL itself with dependency bounds instead of pins:
pip install .
```

That covers a default run. Five of those packages are what `import cryal`
needs; the sixth, RDKit, is what the CH-pi optimizer uses, and
`useChPiOptimizer` is on unless you turn it off -- so it is a dependency of the
shipped configuration, not of an option. Check it:

```bash
python -m unittest discover -s tests -t .
```

162 tests, no LAMMPS, no GPU, no network, under a second. One is an expected
failure and is supposed to be — it records a known limitation of
`unwrap_molecule`, and will report an *unexpected success* if that is ever
fixed.

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

# or, if CrYAL is installed rather than cloned:
cryal-patch-mliap-model MACE-OFF23_small.model-mliap_lammps.pt
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

## 5. UMA (optional, for `energyBackend = uma`)

CrYAL needs **only the Python package**:

```bash
pip install fairchem-core
```

into the same interpreter that runs CrYAL. It pulls `torch` and `ase`. Then a
Hugging Face token once, because the UMA checkpoints are a gated repository:

```bash
huggingface-cli login          # or: export HF_TOKEN=...
```

The first run downloads `uma-s-1p1` (1.17 GB) into `~/.cache/fairchem`.

### What CrYAL does *not* need — read this if UMA is already on the machine

UMA is commonly installed for a different purpose: to serve gradients to **ORCA**
through `orca-external-tools`, which runs a Flask server (`server.py`) that ORCA
reaches with `client.py` and an `extopt` block. If a machine here already has
UMA, it is probably that installation.

**That setup is a superset, not a prerequisite.** CrYAL drives UMA in this
process through ASE and uses neither the server nor the client. Two practical
consequences:

- An existing ORCA-oriented environment **works for CrYAL as it is** — point
  the run at its interpreter. Nothing has to be started, and no port is used.
- A fresh install for CrYAL alone should be the one-line `pip install` above.
  Do not build the ORCA route for this.

And one that is not optional to know: **the ORCA route cannot do periodic
systems at all**, because ORCA has no periodic boundary conditions. For a
crystal, ASE is the only route.

### The trap: which interpreter

An ORCA-oriented install often lives on a mounted volume, beside a *second*
directory of the same name that holds only the Flask server. On the reference
network, one machine has both:

```
.../orca-external-tools/orca-uma/bin/python   <- 7.5 GB, has torch + fairchem
.../orca-uma/bin/python                       <- 22 MB, only Flask
```

The second imports fine and then fails at the first evaluation with
`ModuleNotFoundError: No module named 'torch'`. Check before configuring:

```bash
<interpreter> -c "import fairchem.core, torch; print(torch.cuda.is_available())"
```

Note also that a search of `$HOME` alone finds neither.

### Configuring it

```
energyBackend = uma

% BACKEND_UMA
umaModel  = uma-s-1p1
umaTask   = omc
umaDevice = cuda
```

`umaTask` is **mandatory and has no default.** Each UMA head was trained on a
different domain and each carries its own zero of energy: for one periodic
C50Au4 cell, `omat` returns −466.55 eV and `oc20` −447.06 eV. Energies from two
heads are not comparable, and nothing downstream would reveal a run that mixed
them. `omc` is the organic-molecular-crystal head, which is the one for CSP.

The relaxation is configured by the shared `aseOptimizer` / `aseFmax` /
`aseMaxSteps` / `aseRelaxCell` keys. Note that `aseMaxSteps` matters more here
than with MACE: on a set of 19 crystal candidates MACE-OFF converged in about 25
steps and UMA needed 39–582 from the same geometries, so a budget tuned for one
silently fails structures with the other.

### Versions across machines

`fairchem` moves quickly and the machines on one network drift apart. Checked
here: 2.13.0 and 2.21.0 returned −466.551008 and −466.551014 eV for the same
periodic cell with `uma-s-1p1` and `omat` — 6 µeV, GPU noise rather than a
version effect. That agreement is evidence for one system and one head, not a
general licence; a newer install may also offer checkpoints (`uma-s-1p2`) and
heads (`oc25`) that an older one does not. Re-check with a single point
whenever the model, the head or the fairchem version changes.

Done once more for a second head, because a parallel run over these two
machines writes both their energies into one database: the same 2.13.0 and
2.21.0 returned **−76.159631 eV each**, equal to every digit reported, for one
benzene cell with `uma-s-1p1` and `omc`. Two heads agreeing is still not a
general licence -- it is two data points -- but it is the check to repeat, and
it costs one single point per machine.

For a distributed run, prefer **copying** the cached checkpoint between machines
over downloading it on each: the Hugging Face blob is named by its sha256, so a
copy is provably the same weights, while a fresh download can pick up a
different revision without saying so.

## 6. The workers of a parallel run

`useParallel` hands whole candidates to other machines over ssh. The preflight
checks every worker before the first cycle and disables the ones that fail, so
the run tells you about a broken machine in its first seconds instead of
counting its candidates as lost structures. What it demands of each worker:
passwordless ssh, an interpreter that can `import cryal` and `ase`, and
whatever the chosen backend needs -- `lmp` for `lammps_mace`, `fairchem` for
`uma`.

The interpreter is the part that catches people out. It is the third field of
the worker entry, and on a machine set up for a potential it is almost never
the `python3` on `PATH`:

```
parallelWorkers = raul@10.0.0.11:2:/home/raul/orca-uma/bin/python
```

Install CrYAL into *that* interpreter. Build a wheel on the server and
push it, so every machine runs the same code:

```bash
# on the server, from a checkout (pypa/build is not required)
python - <<'EOF'
from setuptools import build_meta as b
print(b.build_wheel("dist"))
EOF

W=cryal-1.1.0.dev0-py3-none-any.whl
scp dist/$W raul@10.0.0.11:/tmp/
ssh raul@10.0.0.11 "/home/raul/orca-uma/bin/python -m pip install \
    --no-deps --force-reinstall /tmp/$W"
```

**`--no-deps` is the whole point of doing it this way.** A machine that already
runs the potential necessarily has `numpy`, `ase` and the engine; resolving
CrYAL's dependency bounds against that environment can move `torch` underneath
the potential, which is exactly how a working MACE install was lost here once.
The worker only ever relaxes, so `pyxtal` and `scikit-learn` are not needed
there -- structure generation and the surrogate never leave the server.
`--force-reinstall` matters because a development version keeps its number
across edits: without it a redeploy silently does nothing.

Then verify precisely what the preflight will ask, which is cheaper than
discovering it mid-run:

```bash
ssh raul@10.0.0.11 '/home/raul/orca-uma/bin/python -c \
    "import cryal, ase; print(cryal.__version__, ase.__version__)"'
```

A CrYAL version mismatch between server and worker is reported as a
warning rather than a refusal, because it is only the energies that are at
stake and only you know whether the two versions score identically. Treat it as
a reason to redeploy, not as noise to run through.

The **server** does not need the backend at all when it relaxes nothing
itself. Set `parallelLocalSlots = 0`, name only remote machines, and the
backend's load-time check is skipped here -- which is what lets a MACE
environment steer a network of UMA machines without installing the second
stack beside the first, the exact risk this section is otherwise about. The
check is not lost, only moved: the preflight runs that same `validate_config()`
on every worker, so a wrong `umaTask` or a missing checkpoint still surfaces in
the first seconds, reported by the machines it concerns.

The one thing to know is what happens when the workers all go away. Evaluation
would normally fall back to running serially here, and on a server that
cannot run the backend that would mean relaxing candidates with something
nothing has validated. So it refuses instead, naming the check it deferred at
startup. A server that *can* run the backend keeps the old behaviour and
simply carries on alone.

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
