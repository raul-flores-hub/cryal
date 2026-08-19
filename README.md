# CrYAL — Crystal Structure Prediction via Active Learning

Reference implementation of **BACH** (*Bayesian Active Crystal Hopping*), a blind
crystal-structure-prediction workflow for molecular organic crystals.

BACH couples three ingredients in a closed active-learning loop:

1. **Symmetry-aware random generation** of candidate crystals (PyXtal), with
   space groups drawn from their Cambridge Structural Database frequencies.
2. **Local relaxation and energy evaluation** on a machine-learned interatomic
   potential (MACE-OFF23) driven through LAMMPS.
3. A **Gaussian-process surrogate with Expected-Improvement acquisition** that
   learns online which regions of the crystallographic search space are
   energetically favorable and steers the next batch of candidates.

As in basin hopping, the search proceeds by jumps between locally relaxed
minima — but the jumps are *directed* by a Bayesian surrogate rather than driven
by random perturbations.

The search is **blind by design**: no information about the target structure —
not its space group, not its cell, not its `beta` angle — is supplied. The only
physical prior is a plausible density window for organic crystals
(0.8–1.6 g cm⁻³).

## Requirements

- Python 3.10+ with `numpy`, `scipy`, `scikit-learn`, `ase`, `pyxtal`
  (`pip install -r requirements.txt` for the verified pins)
- **LAMMPS** built with the ML-IAP package and **Kokkos** (GPU), plus the
  `lammps` Python module importable by the interpreter that runs CrYAL
- A **MACE-OFF** model exported for ML-IAP (e.g. `MACE-OFF23_small.model-mliap_lammps.pt`)

The Kokkos build is not optional: the MACE ML-IAP interface reaches
`forward_exchange`, which exists only on the Kokkos code path. A plain `lmp`
fails with `AttributeError: 'MLIAPDataPy' object has no attribute 'forward_exchange'`.

**[INSTALL.md](INSTALL.md)** has the full procedure, including what to do when an
exported `.pt` model stops loading after a `mace-torch` upgrade, and how to check
a rebuilt stack against a deterministic result before trusting its numbers.

## Installing

CrYAL runs from a checkout with no installation — that is how the run in the
article was produced, and `python run_cryal.py` keeps working. To install it as
a package instead:

```bash
pip install .                                          # from a checkout
pip install git+https://github.com/raul-flores-hub/cryal
```

That provides two commands, `cryal` (the search) and `cryal-patch-mliap-model`
(the model patcher described in [INSTALL.md](INSTALL.md)), and pulls the five
Python dependencies as *lower bounds*. `requirements.txt` is the other half of
the story: it pins the exact versions this workflow is verified on, which is
what you want for reproducing a run rather than for installing a library.

Neither installs LAMMPS, torch or mace-torch. The relaxation engine is external
by design, and pulling torch into this environment is what broke the exported
MACE model once already.

## Tests

```bash
python -m unittest discover -s tests -t .
```

101 tests over the pure-Python core — no LAMMPS, no GPU, under a second, and no
dependency beyond the standard library. One expected failure is deliberate: it
records that `unwrap_molecule` cannot rebuild a molecule longer than half the
cell, which is safe only because its single caller is off by default.

## Usage

```bash
python run_cryal.py [INPUT.txt]      # from a checkout; INPUT.txt is the default
python run_cryal.py --resume         # continue an interrupted run

cryal [INPUT.txt]                    # the same thing, once installed
cryal --resume
```

The repository ships a runnable example — `INPUT.txt`, `examples/benzene.xyz`
and `in_v3.lammps` — chosen so that a clone can be exercised end to end without
supplying anything. **The molecule you study is your own input and does not
live here**: point `moleculeFile` at your own XYZ, and scale `minCellAxis` and
the density window to it. `specOrder` in `INPUT.txt` and `pair_coeff`/`mass` in
the LAMMPS script must list the same elements in the same order, which a test
checks for the shipped pair.

An interrupted run (power cut, walltime) resumes from `outputDir` without
recomputing: the database is checkpointed after every evaluation, and the
early-stopping state is reconstructed so a resumed run converges identically to
one that was never interrupted.

## Configuration

`INPUT.txt` uses a USPEX-style format — `% SECTION` headers and `KEY = VALUE`
assignments. The sections are `GENERAL`, `ACTIVE_LEARNING`, `SPACE_GROUPS`,
`VOLUME`, `CELL`, `BETA_MONOCLINIC`, `GP`, `CHPI_OPTIMIZER`, `CONTACTS`,
`SEEDS` and `INTEGRITY`; every key is documented inline in the shipped file.

Two settings deserve attention:

- `lammpsCommand` must carry the Kokkos flags, e.g.
  `lmp -k on g 1 -sf kk -pk kokkos newton on neigh half`
- `betaDistribution` must stay `uniform` for a genuinely blind search. The
  `triangular` option exists for informed runs and encodes prior knowledge.

Optional CIF seeds can be placed in `Seeds/`; if the folder is empty the run
starts from PyXtal alone.

## Layout

| path | role |
|---|---|
| `run_cryal.py` | entry point for a checkout (wrapper over `cryal/cli.py`) |
| `cryal/cli.py` | the command line, installed as `cryal` |
| `cryal/config.py` | `INPUT.txt` parser → configuration dataclass |
| `cryal/structure_gen.py` | symmetry-aware generation at a target volume |
| `cryal/backends/` | selectable energy backends (`base.py` holds the contract) |
| `cryal/lammps_runner.py` | compatibility shim for the pre-backends API |
| `cryal/gp_model.py` | GP surrogate and Expected-Improvement acquisition |
| `cryal/active_learning.py` | the active-learning loop |
| `cryal/chpi_optimizer.py` | optional geometric CH–π contact optimizer |
| `cryal/utils.py` | structure I/O, integrity and contact checks |
| `cryal/tools/` | maintenance utilities (the ML-IAP model patcher) |
| `examples/` | the example molecule (benzene) for the shipped `INPUT.txt` |
| `pyproject.toml` | packaging metadata: dependency bounds, console scripts |

## Notes on the method

- **Structure generation is top-down.** Cells are built *at* a target volume and
  expanded in small steps if PyXtal cannot place the molecules, rather than
  generated sparse and compressed. Compressing large planar molecules from low
  density is an NP-hard packing problem and fails in practice.
- **The surrogate acquires in seven dimensions but injects through two.** The GP
  is trained on `(a, b, c, alpha, beta, gamma, rho)` and EI is maximized over all
  seven, but suggestions reach the generator through their `(beta, V)`
  projection: the remaining cell parameters are re-drawn subject to the space
  group and target volume, because they are not independent once symmetry and
  volume are fixed, and constraining them all over-determines the packing.
- **The CH–π optimizer is off by design in production runs.** An ablation showed
  it does not improve the search. It remains available via `useChpiOptimizer`.
- **The Bayesian guidance can be ablated.** Setting `useGP = false` runs the same
  workflow with the surrogate switched off: candidates are generated without the
  Expected-Improvement bias, while the generator, the potential, the relaxation
  protocol and the stopping rule stay as they are. This is the control that
  separates what the GP contributes from what the structure generator does. Note
  that it is not a pure random search — the pool of perturbed incumbents is
  independent of the GP and remains active. Omitting the key leaves the surrogate
  on, so inputs written before this option behave exactly as before.

## Energy backends

`energyBackend` in `INPUT.txt` chooses the engine that relaxes each candidate
and returns its energy:

| value | engine |
|---|---|
| `lammps_mace` (default) | an external LAMMPS process driving MACE-OFF23 through `mliap-unified` — the backend used for the published run |
| `ase` | in-process relaxation with any ASE calculator, named by dotted path |

Omitting the key means `lammps_mace`, so every input written before backends
existed — including the one archived with `v1.0.0` — keeps its exact meaning.

The `ase` backend takes no dependency of its own: the calculator is imported
from the dotted path you give it, built once, and reused for the whole run.

```
% ENERGY_BACKEND
energyBackend       = ase

% BACKEND_ASE
aseCalculator       = mace.calculators.mace_off
aseCalculatorKwargs = model=medium device=cuda
aseOptimizer        = FIRE
aseFmax             = 0.05
aseRelaxCell        = true
```

Two caveats. The `ase` backend performs a *local* relaxation (optimizer plus
cell filter), not the MD-NPT annealing protocol written in the LAMMPS input
script used in the article, so it explores a smaller basin around each
candidate. And energies are comparable only within one backend and one
potential: never merge databases built with different engines, and keep the
backend you started with when resuming a run.

## Extending CrYAL

A new backend is a subclass of `EnergyBackend` (`cryal/backends/base.py`) with
one method:

```python
@register
class MyBackend(EnergyBackend):
    name = "my_engine"

    def relax(self, atoms, step_dir, logger=None):
        ...
        return relaxed_atoms, energy_eV   # or (None, None) on failure
```

The contract is narrow: take an ASE `Atoms`, return a relaxed `Atoms` and a
total energy in eV, or `(None, None)` if the relaxation fails. Failures are
reported, not raised — one bad candidate must not end a search. Everything
around `relax()` is inherited from the base class and applies to every engine:

1. the **pre-relaxation integrity gate**, which rejects sub-Ångström atomic
   overlaps before the engine starts. This one matters: such overlaps make a
   machine-learned potential produce forces of order 10⁵ eV/Å, and the run then
   dies minutes later with an error pointing at the relaxation instead of at
   the geometry. The gate costs seconds instead of minutes per bad candidate;
2. a **sanity bound** on the returned energy (`energySanityMax`);
3. the **molecular-integrity check**, which rejects relaxations that broke or
   formed covalent bonds.

Nothing else in the loop is aware of the calculator: the surrogate consumes
only cell parameters, density and energy. `CP2K`, `Quantum ESPRESSO` and
`DFTB+` all ship ASE calculator interfaces and can be driven today through the
`ase` backend without writing any code.

`cryal/lammps_runner.py` remains as a shim: `evaluate_structure()` and
`get_reference_bonds()` still import from there, and `evaluate_structure()` is
the LAMMPS backend regardless of `energyBackend`.

The acquisition strategy is self-contained in the same way. `cryal/gp_model.py`
owns both the Gaussian process and the Expected-Improvement criterion, so other
surrogates or acquisition functions can replace them without touching the loop.

## Citing

If you use CrYAL, please cite both the software and the accompanying article.
See [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

CrYAL drives LAMMPS as a separate process and does not incorporate or link any
LAMMPS code. LAMMPS and the MACE-OFF models must be obtained separately, under
their own licenses.
