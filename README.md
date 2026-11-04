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

- Python 3.11+ with `numpy`, `scipy`, `scikit-learn`, `ase`, `pyxtal`
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
python run_cryal.py [cryal.toml]     # from a checkout; cryal.toml is the default
python run_cryal.py --resume         # continue an interrupted run

cryal [cryal.toml]                   # the same thing, once installed
cryal --resume
cryal convert INPUT.txt cryal.toml   # an input in the old format -> cryal.toml
```

The repository ships a runnable example — `cryal.toml`, `examples/benzene.xyz`
and `in_v3.lammps` — chosen so that a clone can be exercised end to end without
supplying anything. **The molecule you study is your own input and does not
live here**: point `general.molecule` at your own XYZ, and scale `cell.min_axis`
and the density window to it. `backend.lammps.species` and `pair_coeff`/`mass` in
the LAMMPS script must list the same elements in the same order, which a test
checks for the shipped pair.

An interrupted run (power cut, walltime) resumes from `general.output_dir` without
recomputing: the database is checkpointed after every evaluation, and the
early-stopping state is reconstructed so a resumed run converges identically to
one that was never interrupted.

## Configuration

**[docs/MANUAL.md](docs/MANUAL.md)** is the input manual: how to set up each
search algorithm, and every option, its type, its default and what it does.

`cryal.toml` is [TOML](https://toml.io), read with Python's own `tomllib`. The
tables are `[general]`, `[backend]` (with `[backend.lammps]`, `[backend.ase]`
and `[backend.uma]`), `[relaxation]`, `[active_learning]`, `[space_groups]`,
`[volume]`, `[cell]`, `[monoclinic_beta]`, `[gp]`, `[chpi]`, `[contacts]`,
`[seeds]`, `[integrity]` and `[parallel]`; every key is documented inline in the
shipped file, and any key left out takes its default.

```toml
[general]
molecule = "examples/benzene.xyz"
Z = 4

[space_groups]                # group = weight
weights = { 14 = 36, 2 = 24, 19 = 11, 15 = 7, 4 = 6, 9 = 4, 33 = 2, 61 = 2 }

[volume]
density = [0.8, 1.6]          # g/cm³; the volume window is estimated from it
```

Every key is checked when the file is read. A misspelt key, a key in the wrong
table or a value of the wrong type stops the run with the path of the entry and,
where there is one, the key that was probably meant
(`active_learning.cycle: unknown key in [active_learning]; did you mean 'cycles'?`),
instead of leaving a default silently in its place.

Inputs in the format CrYAL used before — `% SECTION` headers and
`keyName = value`, as `INPUT.txt` or `cryal.in` — are no longer read, but
`cryal convert OLD_FILE cryal.toml` writes the cryal.toml that means the same
run. Each value is converted as the old loader actually read it: an old
`useGP = ture` was false and is written as `enabled = false`, and keys the old
loader ignored are listed at the top of the output rather than dropped.

Two settings deserve attention:

- `backend.lammps.command` must carry the Kokkos flags, e.g.
  `lmp -k on g 1 -sf kk -pk kokkos newton on neigh half`
- `monoclinic_beta.distribution` must stay `"uniform"` for a genuinely blind
  search. The `"triangular"` option exists for informed runs and encodes prior
  knowledge.

Optional CIF seeds can be placed in `Seeds/`; if the folder is empty the run
starts from PyXtal alone.

## The shipped example and the article

`cryal.toml` and `in_v3.lammps` are set up for benzene (`examples/benzene.xyz`,
ideal D6h geometry), a small example that anyone can run. To search your own
molecule, point `general.molecule` at it, list its elements in
`backend.lammps.species`, give the LAMMPS script one `mass` line per element and
the same elements in `pair_coeff`, and scale `cell.min_axis` to the molecule
(5 Å suits benzene; a large planar molecule needs much more).

`in_v3.lammps` is the relaxation protocol of the TMPA-An search reported in the
accompanying article: a staged *NPT* anneal on a triclinic cell (20 K at
3000 bar, then 20 → 10 K and 10 K at 1 bar; 0.4 fs, 12 000 steps), then
`box/relax` and a conjugate-gradient minimization. Its parameters were chosen
for that system; they relax benzene, but they are not tuned for it. The other
settings of `cryal.toml` are those of that search too — BACH, 20 cycles, the
CH–π optimizer off — except the ones that depend on the molecule. The exact
inputs of the TMPA-An search (molecule, input file and LAMMPS script) are in its
data deposit, DOI 10.5281/zenodo.22014958.

**Versions.** 1.0.0 (Zenodo, DOI 10.5281/zenodo.21896733) shipped a different
`in_v3.lammps` — `box/relax` minimizations around a short *NPT* run — which is
not the protocol of the article, and did not have the GP ablation. 1.1.0, on
the `release-1.1` branch and `main`, is the version of the corrected article:
the article's protocol, benzene as the example, and the old `INPUT.txt` input
format. This branch is the development that follows it: `cryal.toml`, energy
backends, distributed relaxations and the alternative searches; a 1.1.0
`INPUT.txt` converts with `cryal convert INPUT.txt cryal.toml`.

## Layout

| path | role |
|---|---|
| `docs/MANUAL.md` | the input manual: every algorithm and every option |
| `run_cryal.py` | entry point for a checkout (wrapper over `cryal/cli.py`) |
| `cryal/cli.py` | the command line, installed as `cryal` |
| `cryal/config.py` | `cryal.toml` loader and schema → configuration dataclass |
| `cryal/legacy.py` | the pre-TOML input format, kept to convert (`cryal convert`) |
| `cryal/structure_gen.py` | symmetry-aware generation at a target volume |
| `cryal/backends/` | selectable energy backends (`base.py` holds the contract) |
| `cryal/lammps_runner.py` | compatibility shim for the pre-backends API |
| `cryal/gp_model.py` | GP surrogate and Expected-Improvement acquisition |
| `cryal/sg_kernel.py` | hierarchical space-group kernel (group, Bravais lattice, crystal family; optional point group and centrosymmetry); not yet used by the GP |
| `tools/distinct_minima.py` | how many distinct minima a finished run found, and the fingerprint threshold |
| `tools/sg_kernel_cv.py` | cross-validation on finished runs that chooses the space-group kernel, by a protocol fixed in its docstring |
| `cryal/active_learning.py` | the active-learning loop |
| `cryal/search.py` | the loop shared by the searches other than BACH; pure random search |
| `cryal/basin_hopping.py` | basin hopping (`general.algorithm = "basin_hopping"`) |
| `cryal/pso.py` | particle swarm (`general.algorithm = "pso"`) |
| `cryal/cmaes.py` | CMA-ES (`general.algorithm = "cmaes"`) |
| `cryal/fingerprint.py` | structural fingerprint that tells two minima apart (`[duplicates]`) |
| `cryal/genetic.py` | the genetic algorithm (`general.algorithm = "ga"`), also used by BACH + GA |
| `cryal/backends/uma.py` | UMA as a named backend, with its mandatory task |
| `cryal/parallel.py` | distributing a cycle's candidates over several machines |
| `cryal/_remote_worker.py` | what runs on a worker: one candidate, one verdict |
| `cryal/chpi_optimizer.py` | optional geometric CH–π contact optimizer |
| `cryal/utils.py` | structure I/O, integrity and contact checks |
| `cryal/tools/` | maintenance utilities (the ML-IAP model patcher) |
| `examples/` | the example molecule (benzene) for the shipped `cryal.toml` |
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
  it does not improve the search. It remains available via `chpi.enabled`, and
  the shipped `cryal.toml` sets it to `false` — but a key left out takes its
  default, which is `true`, so an input of your own should say `false` too.
- **The Bayesian guidance can be ablated.** Setting `gp.enabled = false` runs the same
  workflow with the surrogate switched off: candidates are generated without the
  Expected-Improvement bias, while the generator, the potential, the relaxation
  protocol and the stopping rule stay as they are. This is the control that
  separates what the GP contributes from what the structure generator does. Note
  that it is not a pure random search — the pool of perturbed incumbents is
  independent of the GP and remains active. Omitting the key leaves the surrogate
  on, so inputs written before this option behave exactly as before.

## Variants of the Bayesian optimization

BACH's acquisition can be changed in `[gp]`, the rest of the loop staying as it
is. Left out, the keys are the method of the article — Expected Improvement over
the candidates, and a batch of its best ones kept 0.5 apart in scaled feature
space — and a test holds the default path to that function, suggestion for
suggestion and random draw for random draw.

| key | value | what it does |
|---|---|---|
| `acquisition` | `"ei"` (default) | Expected Improvement |
| | `"ucb"` | lower confidence bound μ − κσ, κ = `ucb_kappa` (0 exploits, large explores) |
| | `"thompson"` | Thompson sampling: each suggestion the minimum of one draw from the GP posterior, over `pool` candidates |
| `batch` | `"diverse"` (default) | the best candidates, at least 0.5 apart |
| | `"believer"` | kriging believer: picked one at a time, each believed at its prediction before the next is scored (with `"ei"` or `"ucb"`) |

```toml
[gp]
acquisition = "ucb"
ucb_kappa   = 2.0
batch       = "believer"
```

The believer and Thompson sampling both make a batch spread out for a reason
the model can state — a believed point has no uncertainty left, a posterior draw
is one plausible landscape — where the published batch keeps its points apart by
a fixed distance. Suggestions are saved per cycle with the acquisition that
made them.

## Alternative searches

`general.algorithm` chooses how candidates are proposed. `"bach"`, the default,
is the method of the article and keeps its own loop (`cryal/active_learning.py`).
The alternatives share another (`cryal/search.py`) that is BACH's in everything
but proposing: the same cycle 0, backend, relaxation, integrity check, parallel
pool, database, stopping rule and resume, with `[active_learning]` setting the
budget of all of them. So searches run from one input file with one key changed
differ only in the way they propose, and each cycle's log — and
`final_report.json` — shows what every source of candidates contributed.

| `algorithm` | proposes |
|---|---|
| `"bach"` | half biased by the GP's Expected Improvement, half random, plus perturbed incumbents |
| `"random"` | fresh random structures every cycle, nothing learnt: the baseline |
| `"ga"` | children of the best distinct structures so far (below) |
| `"basin_hopping"` | Metropolis walks from minimum to minimum (below) |
| `"pso"` | a particle swarm, CALYPSO-like (below) |
| `"cmaes"` | the covariance matrix adaptation evolution strategy (below) |

`"random"` is not `gp.enabled = false`: that still adds BACH's perturbed
incumbents of the pool to every cycle. Pure random search learns nothing from
what it has evaluated, which is what makes it the baseline that any guidance —
the GP, the GA — has to beat to mean anything.

### Genetic algorithm

`general.algorithm = "ga"`. Generation 0 is generated as BACH's cycle 0 is —
though not structure for structure: PyXtal places molecules with its own
generator, seeded from the operating system rather than from `random_seed`,
so no two runs, two BACH runs included, are guaranteed the same structures.

```toml
[general]
algorithm = "ga"

[ga]
population_size = 20
operators = { crossover = 0.5, strain = 0.1, rotation = 0.1, translation = 0.1, immigrant = 0.2 }
```

A crystal is a cell and Z rigid molecules, kept in the conformation they
relaxed to. Each generation breeds `structures_per_cycle` children by
tournament selection from the `population_size` lowest-energy *distinct*
structures in the database (equal energy per molecule and density count once),
in fixed proportions: cut-and-splice crossover, cell strain, molecular
rotation, molecular translation, and fresh random immigrants. Children are
written in P1, overlapping molecules are pushed apart as rigid bodies, and a
child that still clashes is expanded by `volume.expansion_factors`, as the
generator does. Every record a child produces says which operator made it
(`source = "ga_crossover"`, …) and which records it was bred from (`parents`),
and `final_report.json` tallies what each operator contributed.

Since the population is read from the database, a resumed GA run continues
exactly where an uninterrupted one would have been.

### BACH + GA

Half of every BACH cycle is random, to keep exploring. `active_learning.ga_fraction`
hands a share of that half to the genetic operators instead: those structures
are bred from the best distinct structures found so far, with the crossover,
strain, rotation and translation settings of `[ga]`.

```toml
[active_learning]
structures_per_cycle = 50
ga_fraction          = 0.5     # 25 GP-biased, 12 bred by GA, 13 random
```

The GP-biased half, the pool of perturbed incumbents and cycle 0 are untouched,
immigrants get no share (the random structures left are the immigrants), and a
child that cannot be made is replaced by a random structure, so a cycle asks
for the same number of evaluations whatever the setting. With `gp.enabled =
false` the whole batch is random and the share applies to all of it. The
default, 0, is BACH as published. Each cycle's log lists what every source —
biased, random, perturbed, each GA operator — contributed and which one found
a new best.

### Basin hopping

`general.algorithm = "basin_hopping"` is the undirected version of what BACH
directs: walk from one relaxed minimum to the next, perturbing the current one
and moving by the Metropolis rule. `walkers` chains run side by side, so that a
cycle's trials can still be relaxed together, starting from cycle 0's lowest
distinct structures. Each cycle every walker makes its share of the
`structures_per_cycle` trial moves — the GA's strain, rotation and translation —
and moves to its best trial if that is lower, or else with probability
exp(−ΔE/kT), ΔE per molecule.

```toml
[general]
algorithm = "basin_hopping"

[basin_hopping]
walkers     = 10
temperature = 0.01        # kT in eV per molecule, about 1 kJ/mol; 0 = downhill only
```

Each trial's record names its `walker`, the minimum it was made from
(`parents`) and whether the walker moved there (`accepted`), so a walk can be
followed in the database and a resumed run picks every walker up where it was.

### Particle swarm

`general.algorithm = "pso"` is the particle-swarm optimization CALYPSO made
standard for crystals: a swarm of structures, each moving every cycle by a
velocity that keeps some of where it was going (the inertia, falling from 0.9
to 0.4 over the run) and is pulled toward the best structure that particle has
visited and the best one found so far. Each new position is relaxed and the
particle continues from the relaxed structure. A share of every cycle,
`random_fraction` (40 %, as CALYPSO), is fresh random structures.

```toml
[general]
algorithm = "pso"

[pso]
random_fraction = 0.4
inertia         = [0.9, 0.4]
c1              = 2.0        # toward the particle's own best
c2              = 2.0        # toward the swarm's best
```

The arithmetic is done on the crystals of the GA (a cell and rigid molecules):
a velocity is a strain of the cell, a shift of every molecular centre and a turn
of every molecule. To pull one crystal toward another, molecules are matched
across the two (over origin shifts too), and orientations are compared up to the
molecule's own symmetry — benzene turned 60° about its axis is benzene not
turned at all. Each record of a step names its `particle` and the `velocity`
that moved it, so a resumed run carries the swarm on where it was.

### CMA-ES

`general.algorithm = "cmaes"` is the standard (μ/μ_w, λ)-CMA-ES of Hansen's
tutorial, with its default parameters: each cycle one generation of
λ = `structures_per_cycle` samples from a multivariate normal, whose mean,
step size and covariance adapt toward what ranked best — it learns which
combinations of cell, positions and orientations the energy is sensitive to.
A sample is a crystal around a starting structure, as a vector (the cell's
six coordinates, kept always valid; each molecular centre; each molecule's
turn), scaled so one unit is 5 % strain, 0.5 Å or 10°; it is repaired as GA
children are and relaxed, and ranked by its relaxed energy. An instance
starts from cycle 0's best and restarts from the next distinct structure when
it has converged or stalled.

```toml
[general]
algorithm = "cmaes"

[cmaes]
sigma = 1.0
stall = 5
```

CMA-ES needs generations to learn its covariance — with Z = 4 the space has 30
dimensions — so more cycles of fewer structures suit it better than the
default budget, at the same cost. Every sample's record keeps the vector drawn,
and the strategy's whole state is rebuilt from the database by replaying its
updates, exactly, on resume.

### Telling structures apart

Two relaxed structures with the same energy per molecule and density are
usually the same minimum, but not always. Wherever a search needs distinct
structures — the GA's population, BACH + GA's, basin hopping's starting
points — `[duplicates]` decides: energies per molecule within `energy` and
packings within `fingerprint`, the Oganov–Valle cosine distance over
*intermolecular* distances (`cryal/fingerprint.py`; distances within a
molecule are nearly the same in every polymorph and would drown the packing
out). It does not see the cell's setting, origin or orientation, a supercell,
or molecules cut by the cell's faces.

```toml
[duplicates]
energy      = 0.002     # eV per molecule
fingerprint = 0.02      # 0 compares densities instead
```

On benzene, one crystal reached twice (0.01 Å of noise, 0.2 % on the cell)
gave D ≤ 0.007, and crystals differing only in how the molecules are turned
D ≥ 0.032; the default sits between. `tools/distinct_minima.py` counts the
distinct minima a finished run found within a window of its best — the
measure of how much of the landscape a search saw, next to how deep it went —
and with `--calibrate` shows the distances on your own run to place the
threshold:

```bash
python tools/distinct_minima.py runBACH/ runGA/ runRandom/ --window 5   # kJ/mol per molecule
python tools/distinct_minima.py runBACH/ --calibrate
```

## Energy backends

`engine` in `[backend]` chooses the engine that relaxes each candidate
and returns its energy:

| value | engine |
|---|---|
| `lammps_mace` (default) | an external LAMMPS process driving MACE-OFF23 through `mliap-unified` — the backend used for the published run |
| `ase` | in-process relaxation with any ASE calculator, named by dotted path |
| `uma` | in-process relaxation with Meta's UMA (`fairchem-core`) |

Omitting the key means `lammps_mace`, so every input written before backends
existed — including the one archived with `v1.0.0` — keeps its exact meaning.

The `ase` backend takes no dependency of its own: the calculator is imported
from the dotted path you give it, built once, and reused for the whole run.

```toml
[backend]
engine = "ase"

[backend.ase]
calculator        = "mace.calculators.mace_off"
calculator_kwargs = { model = "medium", device = "cuda" }

[relaxation]
optimizer  = "FIRE"
fmax       = 0.05
relax_cell = true
```

Two caveats. The `ase` backend performs a *local* relaxation (optimizer plus
cell filter), not the MD-NPT annealing protocol written in the LAMMPS input
script used in the article, so it explores a smaller basin around each
candidate. And energies are comparable only within one backend and one
potential: never merge databases built with different engines, and keep the
backend you started with when resuming a run.

### UMA

UMA is an ASE calculator, but it has a backend of its own because it cannot be
named by a dotted path — its one-call constructor is a classmethod — and
because the factory every user would otherwise write would have to exist, under
the same importable name, on every machine of a distributed run.

```toml
[backend]
engine = "uma"

[backend.uma]
model  = "uma-s-1p1"
task   = "omc"
device = "cuda"
```

`backend.uma.task` is mandatory and has no default. Each UMA head was trained on a
different domain and carries its own zero of energy: for one periodic C50Au4
cell, `omat` returns −466.55 eV and `oc20` −447.06 eV. Energies from two heads
are not comparable and nothing downstream would reveal the mix, so CrYAL refuses
to guess. `omc` is the organic-molecular-crystal head.

Install is `pip install fairchem-core` plus a Hugging Face token; it is
**not** the ORCA external-tools installation, which serves gradients to ORCA
over a Flask port and cannot do periodic systems at all. See INSTALL.md §5,
including which interpreter to point at when a machine already has UMA
installed for ORCA.

## Running on several machines

Relaxing one candidate says nothing about relaxing another, and the
relaxations are the whole cost of a run. `[parallel]` spreads a cycle's
candidates over the Linux machines on your network that you can already reach
by passwordless `ssh`:

```toml
[parallel]
enabled     = true
workers     = ["raul@192.168.1.11:4", "raul@192.168.1.12:2"]
local_slots = 2
```

Each entry is `user@host:slots`, where *slots* is how many candidates that
machine relaxes at once — four on the workstation, two on the older box, two
here. Machines take candidates from a shared queue rather than being handed a
fixed share, so a slow one simply takes fewer and no machine waits for the
slowest to finish the cycle.

If the interpreter that can `import cryal` on a worker is not the `python3` on
its `PATH` — on a machine set up for MACE it usually is not — name it as a
third field:

```toml
workers = ["raul@192.168.1.11:4:/home/raul/mace-env/bin/python"]
```

Every worker needs CrYAL and its dependencies installed, and the engine
(`lmp`) reachable from a non-interactive `ssh` session; note that `~/.bashrc`
is not read by one, so `backend.lammps.command` may need an absolute path.

Each machine is checked before the run starts — reachable, right interpreter,
engine on the PATH, and the selected backend's own `validate_config()` run over
there against the configuration it will be sent. That last check is not
belt-and-braces: a worker that cannot import the calculator answers *every*
candidate with a rejection, and a rejection is a legitimate verdict, so nothing
would ever retire that machine. It would keep taking a share of every cycle and
returning nothing, and the run would look merely unlucky. A machine that fails
any check is reported with the reason and dropped.

A machine that dies mid-run costs the candidates it was holding at most: those
go back on the queue, and after a few consecutive failures the machine is
dropped and the run continues on what is left.

What this does **not** do is split a single relaxation across machines, and it
is deliberately not a backend: a backend turns one structure into one energy,
and there is nothing inside that to distribute.

The results are the point, so three things are fixed. Every candidate is
evaluated exactly once — nothing is broadcast to several machines and reduced.
Every candidate goes through the same `evaluate()` with the same
configuration, which is sent from here rather than reconstructed there, so the
pre-relaxation gate, the energy bound and the integrity check are the ones you
configured. And records enter the database in candidate order, not completion
order, so a distributed run and a serial one produce the same database for the
same structures.

One limit worth knowing: `parallel.local_slots > 1` only helps a backend that
can run twice at once in this process. `lammps_mace` can, because each
relaxation is a separate `lmp` process. The `ase` backend cannot — one
calculator instance is shared by every relaxation — so the local slots are
clamped to one and you are told why. Remote workers are unaffected either way:
those are separate processes on separate machines.

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
2. a **sanity bound** on the returned energy (`backend.energy_sanity_max`);
3. the **molecular-integrity check**, which rejects relaxations that broke or
   formed covalent bonds.

Nothing else in the loop is aware of the calculator: the surrogate consumes
only cell parameters, density and energy. `CP2K`, `Quantum ESPRESSO` and
`DFTB+` all ship ASE calculator interfaces and can be driven today through the
`ase` backend without writing any code.

`cryal/lammps_runner.py` remains as a shim: `evaluate_structure()` and
`get_reference_bonds()` still import from there, and `evaluate_structure()` is
the LAMMPS backend regardless of `backend.engine`.

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
