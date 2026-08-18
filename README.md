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
- **LAMMPS** built with the ML-IAP package and **Kokkos** (GPU), plus the
  `lammps` Python module importable by the interpreter that runs CrYAL
- A **MACE-OFF** model exported for ML-IAP (e.g. `MACE-OFF23_small.model-mliap_lammps.pt`)

The Kokkos build is not optional: the MACE ML-IAP interface reaches
`forward_exchange`, which exists only on the Kokkos code path. A plain `lmp`
fails with `AttributeError: 'MLIAPDataPy' object has no attribute 'forward_exchange'`.

## Usage

```bash
python run_cryal.py [INPUT.txt]      # INPUT.txt is the default
python run_cryal.py --resume         # continue an interrupted run
```

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
| `run_cryal.py` | entry point |
| `cryal/config.py` | `INPUT.txt` parser → configuration dataclass |
| `cryal/structure_gen.py` | symmetry-aware generation at a target volume |
| `cryal/lammps_runner.py` | LAMMPS/MACE-OFF relaxation and energy evaluation |
| `cryal/gp_model.py` | GP surrogate and Expected-Improvement acquisition |
| `cryal/active_learning.py` | the active-learning loop |
| `cryal/chpi_optimizer.py` | optional geometric CH–π contact optimizer |
| `cryal/utils.py` | structure I/O, integrity and contact checks |

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

## Extending CrYAL

The relaxation backend is deliberately isolated. The active-learning loop
reaches the energy engine through a single call, in `cryal/active_learning.py`:

```python
relaxed, energy = evaluate_structure(
    atoms, cfg, step_dir, ref_bonds, mol_size, logger=logger)
```

The contract is narrow: take an ASE `Atoms` object, return a relaxed `Atoms`
and a total energy in eV, or `(None, None)` if the relaxation fails. Any engine
that honors it can stand in for LAMMPS — CP2K, Quantum ESPRESSO and DFTB+ all
ship ASE calculator interfaces that fit. Nothing else in the loop is aware of
the calculator: the surrogate consumes only cell parameters, density and energy.

The acquisition strategy is self-contained in the same way. `cryal/gp_model.py`
owns both the Gaussian process and the Expected-Improvement criterion, so other
surrogates or acquisition functions can replace them without touching the loop.

These alternative backends and algorithms are **not implemented**; this section
documents where they would attach.

## Citing

If you use CrYAL, please cite both the software and the accompanying article.
See [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

CrYAL drives LAMMPS as a separate process and does not incorporate or link any
LAMMPS code. LAMMPS and the MACE-OFF models must be obtained separately, under
their own licenses.
