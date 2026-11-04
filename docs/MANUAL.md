# CrYAL input manual

How to write `cryal.toml`: choosing and configuring each search algorithm, and
every option the input accepts.

- [1. The input file](#1-the-input-file)
- [2. A minimal input](#2-a-minimal-input)
- [3. Choosing a search algorithm](#3-choosing-a-search-algorithm)
- [4. The algorithms, one by one](#4-the-algorithms-one-by-one)
  - [4.1 BACH](#41-bach-algorithm--bach) · [4.2 BACH + GA](#42-bach--ga) ·
    [4.3 Pure random search](#43-pure-random-search-algorithm--random) ·
    [4.4 Genetic algorithm](#44-genetic-algorithm-algorithm--ga) ·
    [4.5 Basin hopping](#45-basin-hopping-algorithm--basin_hopping) ·
    [4.6 Particle swarm](#46-particle-swarm-algorithm--pso) ·
    [4.7 CMA-ES](#47-cma-es-algorithm--cmaes)
- [5. Comparing algorithms](#5-comparing-algorithms)
- [6. Option reference](#6-option-reference) — every table, every key
- [7. What a run writes](#7-what-a-run-writes)
- [8. Resuming, reproducibility, and common errors](#8-resuming-reproducibility-and-common-errors)

---

## 1. The input file

The input is a [TOML](https://toml.io) file, `cryal.toml` by default:

```bash
python run_cryal.py                 # reads ./cryal.toml
python run_cryal.py my_run.toml     # another file
python run_cryal.py --resume        # continue an interrupted run (see §8)
cryal my_run.toml                   # the same, once installed with pip
```

TOML in four rules: settings are grouped under `[tables]`; strings are quoted
(`"FIRE"`); `true`/`false` are booleans; lists go in brackets (`[0.8, 1.6]`) and
inline tables in braces (`{ 14 = 36, 2 = 24 }`). `#` starts a comment.

**Every key is checked when the file is read.** A misspelt key, a key in the
wrong table, or a value of the wrong type stops the run before anything is
computed, naming the entry and, where it can, the key that was meant:

```
active_learning.cycle: unknown key in [active_learning]; did you mean 'cycles'?
gp.cycles: unknown key in [gp]; 'cycles' belongs in [active_learning]
```

**A key left out takes its default** (listed in §6). The defaults are the
values in the code, and in a few places they differ from the values written in
the `cryal.toml` shipped with the repository, which is tuned for its benzene
example — §6 marks those keys with **(example: …)**. So deleting a line from
the shipped file can change the run.

**Old inputs.** Files in the format CrYAL used before TOML (`% SECTION`
headers, `INPUT.txt`, `cryal.in`) are no longer read. Convert one with:

```bash
cryal convert INPUT.txt cryal.toml
```

---

## 2. A minimal input

Only the molecule is truly required; everything else has a default.

```toml
[general]
molecule = "my_molecule.xyz"   # one molecule, XYZ
Z        = 4                   # molecules per cell

[backend.lammps]
command = "lmp -k on g 1 -sf kk -pk kokkos newton on neigh half"
species = ["C", "H", "N", "O"] # same elements, same order as pair_coeff in the LAMMPS script

[cell]
min_axis = 6.0                 # Å; scale with your molecule (§6, [cell])

[chpi]
enabled = false                # off, as in the article; left out, it is on
```

That runs BACH, the published method, with the LAMMPS/MACE-OFF23 backend and
the relaxation protocol of `in_v3.lammps`. Two things deserve thought for any
new molecule: `cell.min_axis` (the default,
9 Å, suits a large molecule and rejects every cell of a small one), and the
density window `volume.density` (0.8–1.6 g/cm³ suits most organic crystals).

---

## 3. Choosing a search algorithm

`[general] algorithm` chooses how candidate structures are **proposed**.
Everything that turns a candidate into an energy is shared by all of them:

| shared by every algorithm | where it is set |
|---|---|
| cycle 0: `initial_structures` random structures from the generator, plus the seeds | `[active_learning]`, `[space_groups]`, `[volume]`, `[cell]`, `[monoclinic_beta]`, `[seeds]` |
| the budget: number of cycles and of candidates per cycle | `[active_learning]` |
| the stopping rule: stop after `max_no_improve_cycles` cycles without a new best | `[active_learning]` |
| relaxation and energy, the pre-relaxation overlap gate, the integrity check | `[backend]`, `[relaxation]`, `[integrity]` |
| the optional CH–π step, the contact distances | `[chpi]`, `[contacts]` |
| distribution of the relaxations over machines | `[parallel]` |
| the database, the reports, `--resume` | — |

So two runs from the same file that differ only in `algorithm` spend the same
number of relaxations and differ only in the way they propose.

| `algorithm` | proposes each cycle | its own table |
|---|---|---|
| `"bach"` (default) | half the structures biased toward the GP's Expected Improvement, half random, plus perturbed copies of the best | `[gp]` |
| `"bach"` + `ga_fraction` | as BACH, with part of the random half bred by the GA | `[gp]`, `[ga]` |
| `"random"` | fresh random structures, learning nothing | — |
| `"ga"` | children of the best distinct structures: crossover, mutations, immigrants | `[ga]` |
| `"basin_hopping"` | Metropolis walks from minimum to minimum | `[basin_hopping]` |
| `"pso"` | a particle swarm (CALYPSO-like) | `[pso]` |
| `"cmaes"` | samples of an adapting multivariate normal (CMA-ES) | `[cmaes]` |

A table that belongs to another algorithm is read and checked but has no
effect, so one file can carry the settings of every algorithm and switch with
one key. The exception is `active_learning.ga_fraction`, which is refused with
any algorithm but `"bach"`.

All the algorithms except BACH write their children in space group P1: they
move molecules and cells without enforcing symmetry, and the relaxation decides
what the structure becomes. Their records have `space_group = null`.

---

## 4. The algorithms, one by one

### 4.1 BACH (`algorithm = "bach"`)

The method of the article. Each cycle a Gaussian process is trained on every
structure evaluated so far — features `(a, b, c, α, β, γ, ρ)` → energy — and
Expected Improvement is maximized over `gp.ei_candidates` random points. The
best `gp.ei_suggestions` points bias half of the next batch (their β and volume
are what the generator uses); the other half is random; and up to a quarter of
a batch more is perturbed copies of the `active_learning.pool_size` best
structures (the "pool").

```toml
[general]
algorithm = "bach"            # or leave it out

[active_learning]
cycles               = 10
initial_structures   = 100
structures_per_cycle = 50
pool_size            = 20
max_no_improve_cycles = 3

[gp]
enabled        = true
kernel         = "matern25"
ei_xi          = 0.01
ei_candidates  = 50000
ei_suggestions = 20
```

**The ablation control.** `gp.enabled = false` keeps everything but the GP:
candidates are generated without the Expected-Improvement bias. It is *not* a
pure random search — the pool of perturbed incumbents stays — which is what
`algorithm = "random"` is (§4.3).

**The anisotropy feature.** `gp.anisotropy = true` adds the longest/shortest
axis ratio as an eighth GP feature, and the biased half reproduces the
suggested ratio. Off by default.

**Variants of the acquisition.** Left out, these keys are the published method,
draw for draw (a test holds the default path to the published function).

| key | value | effect |
|---|---|---|
| `gp.acquisition` | `"ei"` (default) | Expected Improvement |
| | `"ucb"` | lower confidence bound μ − κσ, κ = `gp.ucb_kappa`: 0 picks the lowest predicted energy, larger values explore |
| | `"thompson"` | Thompson sampling: each suggestion is the minimum of one draw from the GP posterior, over `gp.pool` random candidates |
| `gp.batch` | `"diverse"` (default) | the best candidates, at least 0.5 apart in scaled feature space |
| | `"believer"` | kriging believer: picked one at a time, each believed at its predicted energy before the next is scored; works with `"ei"` and `"ucb"`, among the `gp.pool` best |

```toml
[gp]
acquisition = "ucb"
ucb_kappa   = 2.0
batch       = "believer"
pool        = 2000
```

`batch = "believer"` with `acquisition = "thompson"` is refused: Thompson
sampling makes its batch from separate posterior draws already.

**Blind search.** Keep `monoclinic_beta.distribution = "uniform"` and do not
set the volume window or the space groups from the known structure; the
`"triangular"` β distribution exists for informed runs and encodes prior
knowledge.

### 4.2 BACH + GA

BACH, with a share of each cycle's random half bred by the genetic algorithm's
operators instead, from the best distinct structures found so far.

```toml
[general]
algorithm = "bach"

[active_learning]
structures_per_cycle = 50
ga_fraction          = 0.5     # 25 GP-biased, 12 bred by the GA, 13 random

[ga]                           # population, tournament and operator settings
population_size = 20
operators = { crossover = 0.5, strain = 0.1, rotation = 0.1, translation = 0.1 }
```

- `ga_fraction` is a share of the **random part**: with the GP making
  suggestions that is `structures_per_cycle − structures_per_cycle // 2`; with
  `gp.enabled = false` it is the whole batch. The count is rounded to the
  nearest integer (Python rounds 12.5 to 12).
- The GP-biased half, the pool perturbations and cycle 0 are unchanged.
- Immigrants get no share — the random structures left are the immigrants — so
  `ga.operators` needs a positive weight on at least one of crossover, strain,
  rotation or translation; otherwise the input is refused.
- A child that cannot be built is replaced by a random structure, so the
  number of relaxations per cycle does not change.
- `ga_fraction = 0`, the default, is BACH as published. `ga_fraction = 1`
  removes the random exploration altogether.
- What `[ga]` keys mean: §4.4. What "distinct" means: `[duplicates]`, §6.

### 4.3 Pure random search (`algorithm = "random"`)

Every cycle is `structures_per_cycle` fresh structures from the generator, as
in cycle 0; nothing evaluated is used to choose what comes next. This is the
baseline: a guided search has to beat it for its guidance to mean anything.

```toml
[general]
algorithm = "random"
```

It has no table of its own; the generator's settings (`[space_groups]`,
`[volume]`, `[cell]`, `[monoclinic_beta]`) shape what it draws.

### 4.4 Genetic algorithm (`algorithm = "ga"`)

Each generation (one cycle) breeds `structures_per_cycle` children from the
population: the `ga.population_size` lowest-energy distinct structures in the
database, over everything evaluated so far. Parents are drawn by tournament
(best of `ga.tournament_size` drawn at random). The operators, in fixed
proportions per generation:

| operator | what it does | its settings |
|---|---|---|
| `crossover` | cut-and-splice: both parents shifted at random, a plane cuts both along one axis, molecules on one side from each; the cell is the two cells blended in proportion, at their blended volume | — |
| `strain` | a random symmetric strain of the cell, at constant volume; molecular centres keep their fractional coordinates | `strain_sigma` |
| `rotation` | some molecules turn about their centres | `max_rotation`, `fraction_mutated` |
| `translation` | some molecules move | `max_translation`, `fraction_mutated` |
| `immigrant` | a fresh random structure from the generator, for diversity | — |

```toml
[general]
algorithm = "ga"

[ga]
population_size  = 20
tournament_size  = 3
operators        = { crossover = 0.5, strain = 0.1, rotation = 0.1, translation = 0.1, immigrant = 0.2 }
strain_sigma     = 0.10
max_rotation     = 45.0     # degrees
max_translation  = 1.0      # Å
fraction_mutated = 0.5
```

`operators` are weights, normalized; an operator left out of the table gets 0.
The counts per generation are fixed, not drawn (largest remainder), so every
generation has the same mix.

**Every child is repaired before it is relaxed**, as in all the P1 searches:
overlapping molecules are pushed apart as rigid bodies; a cell in which a
molecule meets its own periodic image is lengthened along that direction; and
the child is expanded by `volume.expansion_factors` until no contact is below
`[contacts]`. A child that fails is drawn again (up to 50 attempts per child).
Molecules stay rigid, in the conformation they relaxed to in their parent.

With a population of one structure, crossover slots go to the mutations.

### 4.5 Basin hopping (`algorithm = "basin_hopping"`)

`walkers` chains walk from relaxed minimum to relaxed minimum. They start from
the `walkers` lowest distinct structures of cycle 0. Each cycle, every walker
makes its share of the `structures_per_cycle` trial moves from its current
minimum — strain, rotation or translation, drawn by `moves` weight — and, once
the trials are relaxed, moves to its best trial if

```
ΔE = (E_trial − E_current) / Z  ≤ 0,   or else with probability exp(−ΔE / kT)
```

```toml
[general]
algorithm = "basin_hopping"

[basin_hopping]
walkers          = 10
temperature      = 0.01     # kT, eV per molecule (0.01 ≈ 1 kJ/mol); 0 = downhill only
moves            = { strain = 1, rotation = 1, translation = 1 }
strain_sigma     = 0.10
max_rotation     = 45.0
max_translation  = 1.0
fraction_mutated = 0.5
```

- **Temperature.** The scale of uphill moves the walk accepts. Of the order of
  the energy gaps between the polymorphs you expect (1–5 kJ/mol ≈ 0.01–0.05
  eV/molecule) lets walkers cross between them; 0 makes the walk greedy.
- **Walkers vs trials.** With `structures_per_cycle = 50` and 10 walkers each
  walker makes 5 trials per cycle and moves to the best of them; with 50
  walkers each makes one, which is the textbook walk. Fewer walkers explore
  each basin more thoroughly; more walkers explore more basins.
- Trials that cannot be built are replaced by random structures.

### 4.6 Particle swarm (`algorithm = "pso"`)

The form CALYPSO uses: a swarm of structures, each moving every cycle by a
velocity that keeps part of where it was going and is pulled toward the best
structure that particle has visited (pbest) and the best one found so far
(gbest):

```
v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x),      x ← relaxed(x + v)
```

`r1`, `r2` are uniform in [0, 1); the inertia `w` falls linearly from
`inertia[0]` in the first cycle to `inertia[1]` in the last. Each cycle,
`random_fraction` of the structures are fresh random ones (they belong to no
particle, but can become gbest); the rest are the particles, one step each.
With 50 structures per cycle and `random_fraction = 0.4`, the swarm has 30
particles.

```toml
[general]
algorithm = "pso"

[pso]
random_fraction = 0.4
inertia         = [0.9, 0.4]
c1              = 2.0       # pull toward the particle's own best
c2              = 2.0       # pull toward the swarm's best
max_strain      = 0.10      # per step: Frobenius norm of the cell strain
max_shift       = 1.0       # per step, per molecular centre, Å
max_turn        = 30.0      # per step, per molecule, degrees
```

A velocity has three parts — a strain of the cell, a shift of each molecular
centre, a turn of each molecule — and is clipped each step to `max_strain`,
`max_shift`, `max_turn`. To pull one crystal toward another, molecules are
matched between the two (also over origin shifts), and orientations are
compared up to the molecule's own symmetry (for benzene, a 60° turn about its
axis is no turn). The particles start from cycle 0's lowest distinct
structures, with a small random velocity.

- Larger `c2` relative to `c1` concentrates the swarm on the best structure
  sooner; larger `c1` keeps particles exploring around their own finds.
- Lower `max_*` caps make the swarm move in smaller, safer steps.

### 4.7 CMA-ES (`algorithm = "cmaes"`)

The standard (μ/μ_w, λ) covariance matrix adaptation evolution strategy, with
its default internal parameters. Each cycle is one generation of
λ = `structures_per_cycle` samples from a multivariate normal; the mean, the
step size and the covariance adapt toward the samples that ranked best, so
the strategy learns which combinations of variables the energy depends on.

A sample is a crystal around a starting structure, written as a vector of
`6 + 6·Z` numbers: six for the cell (always a valid cell), three per molecular
centre, three per molecular orientation. Each block is scaled so that one unit
means a comparable change:

```toml
[general]
algorithm = "cmaes"

[cmaes]
sigma         = 1.0     # initial step size, in the units below
strain_scale  = 0.05    # one unit of cell coordinate = 5 % strain
shift_scale   = 0.5     # one unit of a molecular centre = 0.5 Å
turn_scale    = 10.0    # one unit of an orientation = 10 degrees
restart_sigma = 0.05    # restart once the distribution has shrunk below this
stall         = 5       # ... or the best has not improved for this many generations
```

Samples are repaired like GA children, relaxed, and ranked by relaxed energy;
the sample itself stays where it was drawn. The first instance starts from the
best structure of cycle 0; it restarts from the next lowest distinct structure
when it has converged (`restart_sigma`) or stalled (`stall`).

**Budget.** CMA-ES learns its covariance over generations, and with Z = 4 it
works in 30 dimensions. The default budget (10 cycles × 50) gives it 10
generations, too few to learn much. At the same cost, more cycles of fewer
structures suit it better, e.g.:

```toml
[active_learning]
cycles               = 50
structures_per_cycle = 10
max_no_improve_cycles = 15
```

---

## 5. Comparing algorithms

Run the same input once per algorithm, changing only `algorithm` and
`output_dir`, then count what each found:

```bash
python tools/distinct_minima.py run_bach/ run_ga/ run_random/ run_bh/ --window 5
```

For each run, the tool groups the structures within `--window` kJ/mol per
molecule of that run's best into distinct minima (by the `[duplicates]` rule)
and lists them, with how often and how early each was found and which source of
candidates found it first. The lowest energy says how deep a search went; the
number of distinct low-energy minima says how much of the landscape it saw.

Each run's log also lists, per cycle, what every source of candidates
contributed (`biased`, `random`, `perturb`, `seed`, each GA operator, `hop`,
`pso`, `cmaes`) and which one found a new best; `final_report.json` keeps the
totals under `source_stats`.

Two cautions. Change only the algorithm between runs, or the comparison
measures something else. And a fixed `random_seed` does not make two runs
replicas (§8): compare several runs per algorithm, not one.

---

## 6. Option reference

Every key the input accepts, by table. **Default** is the value used when the
key is left out. Where the shipped `cryal.toml` sets something else, it says
**(example: …)**.

### `[general]`

| key | type | default | meaning |
|---|---|---|---|
| `molecule` | string | `"molecule.xyz"` | XYZ file of **one** molecule. Its atom order is the order every crystal is written in. Must exist. (example: `"examples/benzene.xyz"`) |
| `Z` | integer | `4` | molecules per unit cell |
| `random_seed` | integer | `42` | seed of Python's and NumPy's generators (see §8 on what it does and does not fix). (example: `33936`) |
| `output_dir` | string | `"AL_results"` | where everything is written (§7) |
| `log_level` | string | `"INFO"` | `"DEBUG"`, `"INFO"`, `"WARNING"` or `"ERROR"` |
| `resume` | boolean | `false` | continue an interrupted run in `output_dir`; the same as `--resume` |
| `algorithm` | string | `"bach"` | `"bach"`, `"ga"`, `"random"`, `"basin_hopping"`, `"pso"` or `"cmaes"` (§3) |

### `[backend]`

| key | type | default | meaning |
|---|---|---|---|
| `engine` | string | `"lammps_mace"` | `"lammps_mace"` (external LAMMPS with MACE-OFF23, the published run), `"ase"` (any ASE calculator, in process), `"uma"` (Meta's UMA, in process) |
| `energy_sanity_max` | number or `false` | `-100.0` | total energies (eV per cell) above this count as a failed relaxation; `false` switches the check off. Calibrated for MACE-OFF: another potential's zero of energy may need another value |

Energies from different engines or potentials are not comparable: never merge
databases built with different backends, and keep the backend when resuming.

### `[backend.lammps]` — read when `engine = "lammps_mace"`

| key | type | default | meaning |
|---|---|---|---|
| `input` | string | `"in_v3.lammps"` | LAMMPS input script for the relaxation. The shipped one is the article's protocol: a staged NPT anneal on a triclinic cell (20 K at 3000 bar, then 20 → 10 K and 10 K at 1 bar; 0.4 fs, 12 000 steps), then `box/relax` and conjugate gradient |
| `command` | string | `"lmp"` | how to start LAMMPS. MACE needs the Kokkos GPU build: e.g. `"lmp -k on g 1 -sf kk -pk kokkos newton on neigh half"` (that is the example's value) |
| `species` | list of strings | `["C", "H", "O", "N"]` | element order of the LAMMPS atom types; must match `pair_coeff` and `mass` in the script. (example: `["C", "H"]`) |

### `[backend.ase]` — read when `engine = "ase"`

| key | type | default | meaning |
|---|---|---|---|
| `calculator` | string | `""` | dotted path to an ASE calculator class or factory, e.g. `"mace.calculators.mace_off"`; required |
| `calculator_kwargs` | table | `{}` | keyword arguments for it, e.g. `{ model = "medium", device = "cuda" }` |

The calculator is built once and reused. The ASE backend relaxes locally
(optimizer + cell filter, `[relaxation]`), not with the MD-NPT protocol of the
LAMMPS script. With `[parallel]`, its local slots are limited to one.

### `[backend.uma]` — read when `engine = "uma"`

| key | type | default | meaning |
|---|---|---|---|
| `model` | string | `"uma-s-1p1"` | UMA checkpoint |
| `task` | string | `""` | **mandatory, no default**: `"omc"` (organic molecular crystals — the one for CSP), `"omat"`, `"oc20"`, `"omol"`, `"odac"`. Each head has its own zero of energy |
| `device` | string | `"cuda"` | `"cuda"` or `"cpu"` |
| `inference_settings` | string | `"default"` | `"default"` or a preset name |

### `[relaxation]` — read by the `ase` and `uma` engines

| key | type | default | meaning |
|---|---|---|---|
| `optimizer` | string | `"FIRE"` | `"FIRE"`, `"BFGS"`, `"LBFGS"`, `"BFGSLineSearch"` or `"GPMin"` |
| `fmax` | number | `0.05` | convergence: largest force, eV/Å |
| `max_steps` | integer | `500` | a relaxation not converged by then is discarded; the end of each cycle reports how much of this budget the relaxations needed |
| `relax_cell` | boolean | `true` | relax the cell as well as the positions |

### `[active_learning]` — the budget, for every algorithm

| key | type | default | meaning |
|---|---|---|---|
| `cycles` | integer | `10` | cycles (generations) after cycle 0. **(example: `20`, the article's search)** |
| `initial_structures` | integer | `100` | random structures in cycle 0 |
| `structures_per_cycle` | integer | `50` | candidates per cycle; λ for CMA-ES |
| `pool_size` | integer | `20` | BACH only: best structures whose perturbed copies join each cycle |
| `max_no_improve_cycles` | integer | `3` | stop when the best energy has not changed (by 1e-4 eV) for this many cycles |
| `ga_fraction` | number in [0, 1] | `0.0` | BACH only: share of the random part bred by the GA (§4.2) |

### `[space_groups]`

| key | type | default | meaning |
|---|---|---|---|
| `weights` | table `{ group = weight }` | `{ 14 = 36, 2 = 24, 19 = 11, 15 = 7, 4 = 6, 9 = 4, 33 = 2, 61 = 2 }` | space groups the generator draws from (1–230) and their weights, the CSD frequencies for organic crystals by default |

The cell is drawn for each group's crystal system (trigonal and rhombohedral in
the hexagonal setting). Above group 74 a group usually needs `Z` to match its
Wyckoff multiplicities; a group that never yields a structure is named in the
log with PyXtal's reason.

### `[volume]`

| key | type | default | meaning |
|---|---|---|---|
| `min`, `max` | number | — | volume window in Å³. Left out, it is computed from `density` and the molecular weight |
| `density` | `[low, high]` | `[0.8, 1.6]` | g/cm³: the only physical prior of a blind search |
| `expansion_factors` | list of numbers | `[1.0, 1.05, 1.10, 1.15, 1.20, 1.35, 1.50]` | when molecules cannot be placed (generator) or still clash (repair of P1 children), the cell is expanded by these factors in turn |

### `[cell]`

| key | type | default | meaning |
|---|---|---|---|
| `min_axis` | number | `9.0` | Å; cells with an axis shorter than this are rejected. Scale it with the molecule: benzene needs about 5. **(example: `5.0`)** |

### `[monoclinic_beta]`

| key | type | default | meaning |
|---|---|---|---|
| `range` | `[low, high]` | `[90.0, 150.0]` | degrees, the β window for monoclinic groups (also the β range of BACH's candidates) |
| `distribution` | string | `"uniform"` | `"uniform"` (blind) or `"triangular"` (informed) |
| `mode` | number | — | the peak of the triangular distribution; without it `"triangular"` falls back to uniform, with a warning |

### `[gp]` — BACH only

| key | type | default | meaning |
|---|---|---|---|
| `enabled` | boolean | `true` | `false` is the ablation control: no GP bias (the pool stays) |
| `anisotropy` | boolean | `false` | add the longest/shortest axis ratio as a feature |
| `kernel` | string | `"matern25"` | `"matern25"` or `"rbf"` |
| `n_restarts` | integer | `5` | restarts of the hyperparameter optimizer |
| `ei_xi` | number | `0.01` | EI's exploration margin (eV); larger explores more |
| `ei_candidates` | integer | `50000` | random points the acquisition is evaluated on |
| `ei_suggestions` | integer | `20` | suggestions that bias the next batch |
| `acquisition` | string | `"ei"` | `"ei"`, `"ucb"` or `"thompson"` (§4.1) |
| `ucb_kappa` | number ≥ 0 | `2.0` | `"ucb"` only: weight on σ |
| `batch` | string | `"diverse"` | `"diverse"` or `"believer"` (§4.1) |
| `pool` | integer ≥ 1 | `2000` | candidates `"thompson"` and `"believer"` work on |

### `[ga]` — the GA, and BACH + GA

| key | type | default | meaning |
|---|---|---|---|
| `population_size` | integer ≥ 1 | `20` | distinct structures eligible as parents |
| `tournament_size` | integer ≥ 1 | `3` | a parent is the best of this many drawn at random |
| `operators` | table | `{ crossover = 0.5, strain = 0.1, rotation = 0.1, translation = 0.1, immigrant = 0.2 }` | weights; one left out is 0; at least one positive. BACH + GA ignores `immigrant` |
| `strain_sigma` | number ≥ 0 | `0.10` | std of each component of the strain tensor |
| `max_rotation` | number ≥ 0 | `45.0` | degrees, largest turn of a molecule |
| `max_translation` | number ≥ 0 | `1.0` | Å, largest shift of a molecule |
| `fraction_mutated` | number in (0, 1] | `0.5` | share of the molecules a rotation or translation moves (at least one) |

### `[duplicates]` — the GA, BACH + GA, basin hopping, PSO, CMA-ES and `tools/distinct_minima.py`

When two relaxed structures count as the same minimum, wherever a search needs
distinct structures (the GA's population, basin hopping's and PSO's starting
points, CMA-ES's restarts).

| key | type | default | meaning |
|---|---|---|---|
| `energy` | number ≥ 0 | `0.002` | eV per molecule: energies must agree within this |
| `fingerprint` | number in [0, 1] | `0.02` | and the packings within this Oganov–Valle cosine distance, over intermolecular distances. `0` compares densities instead |
| `density` | number ≥ 0 | `0.005` | g/cm³, used only when `fingerprint = 0` |

On benzene, one crystal reached twice (0.01 Å of noise, 0.2 % on the cell) gave
fingerprint distances ≤ 0.007, and crystals differing only in how the molecules
are turned ≥ 0.032; the default sits between. Calibrate it on your own run:
`python tools/distinct_minima.py run/ --calibrate` prints the distances between
every two structures of equal energy, and the threshold goes in the gap.

### `[basin_hopping]` — `algorithm = "basin_hopping"` only

| key | type | default | meaning |
|---|---|---|---|
| `walkers` | integer ≥ 1 | `10` | independent chains |
| `temperature` | number ≥ 0 | `0.01` | kT in eV per molecule; 0 accepts only downhill moves |
| `moves` | table | `{ strain = 1, rotation = 1, translation = 1 }` | weights of the moves; one left out is 0 |
| `strain_sigma` | number ≥ 0 | `0.10` | as in `[ga]` |
| `max_rotation` | number ≥ 0 | `45.0` | degrees |
| `max_translation` | number ≥ 0 | `1.0` | Å |
| `fraction_mutated` | number in (0, 1] | `0.5` | share of the molecules a rotation or translation moves |

### `[pso]` — `algorithm = "pso"` only

| key | type | default | meaning |
|---|---|---|---|
| `random_fraction` | number in [0, 1] | `0.4` | share of each cycle that is fresh random structures; the rest are the particles |
| `inertia` | `[start, end]` | `[0.9, 0.4]` | inertia weight, falling linearly over the run |
| `c1` | number ≥ 0 | `2.0` | pull toward the particle's own best |
| `c2` | number ≥ 0 | `2.0` | pull toward the swarm's best |
| `max_strain` | number ≥ 0 | `0.10` | per step, Frobenius norm of the cell strain |
| `max_shift` | number ≥ 0 | `1.0` | Å per step, per molecular centre |
| `max_turn` | number ≥ 0 | `30.0` | degrees per step, per molecule |

### `[cmaes]` — `algorithm = "cmaes"` only

| key | type | default | meaning |
|---|---|---|---|
| `sigma` | number ≥ 0 | `1.0` | initial step size, in the scaled units |
| `strain_scale` | number ≥ 0 | `0.05` | one unit of a cell coordinate |
| `shift_scale` | number ≥ 0 | `0.5` | Å, one unit of a molecular centre |
| `turn_scale` | number ≥ 0 | `10.0` | degrees, one unit of an orientation |
| `restart_sigma` | number ≥ 0 | `0.05` | restart when σ × the longest axis of the distribution falls below this |
| `stall` | integer ≥ 1 | `5` | restart when the instance's best has not improved for this many generations |

### `[chpi]` — the CH–π contact optimizer

A greedy geometric step, applied to every candidate before relaxation, that
moves molecules to bring H atoms to `d_target` from aromatic ring centroids.
Needs RDKit. An ablation found that it does not improve the search, and the
article's search ran with it off, as the shipped example does. **But the
default, when the key is left out, is `true`**: write `enabled = false` in your
own input.

| key | type | default | meaning |
|---|---|---|---|
| `enabled` | boolean | `true` | run the optimizer. **(example: `false`, as in the article's search)** |
| `d_range` | `[low, high]` | `[2.6, 3.5]` | Å, H···centroid distances counted as a contact |
| `d_target` | number | `3.0` | Å, the distance a move aims for |
| `d_search` | number | `5.0` | Å, search radius when listing contacts. **(example: `6.0`)** |
| `angle_ch_centroid_min` | number | `110.0` | degrees, smallest C–H···centroid angle of a contact |
| `angle_from_normal_max` | number | `55.0` | degrees, largest angle from the ring normal |
| `max_iter` | integer | `1000` | iterations per structure. **(example: `100`)** |
| `max_no_improve` | integer | `50` | stop after this many iterations without more contacts. **(example: `10`)** |

### `[contacts]`

Smallest intermolecular distances a structure may have before relaxation: used
to filter seed perturbations and, for the P1 searches, by the repair of every
child.

| key | type | default | meaning |
|---|---|---|---|
| `min_hh` | number | `1.8` | Å, H···H |
| `min_heavy` | number | `2.5` | Å, between two non-H atoms |
| `min_hx` | number | `1.8` | Å, H···non-H |

### `[seeds]`

| key | type | default | meaning |
|---|---|---|---|
| `enabled` | boolean | `true` | load structures from `folder` into cycle 0 |
| `folder` | string | `"Seeds"` | CIF or POSCAR files; empty or missing means none |
| `perturbations_per_seed` | integer | `30` | perturbed copies added per seed (BACH's pool uses a third of this, at least 2) |
| `max_rotation` | number | `30.0` | degrees, largest turn of a perturbed molecule |
| `max_translation` | number | `1.5` | Å, largest shift of a perturbed molecule. **(example: `0.1`)** |
| `fraction_perturbed` | number | `0.5` | share of the molecules perturbed per copy |

These settings also shape BACH's pool perturbations from cycle 1 on. Seeding a
blind search with the known structure makes it an informed one.

### `[integrity]`

| key | type | default | meaning |
|---|---|---|---|
| `enabled` | boolean | `true` | reject relaxations that broke or formed covalent bonds |
| `max_bond_ratio` | number | `2.0` | a bond is broken when longer than this × the sum of covalent radii |

### `[parallel]`

Relaxes a cycle's candidates on several machines at once (whole candidates;
one relaxation is never split). Results enter the database in candidate order,
so a parallel run records exactly what a serial one would.

| key | type | default | meaning |
|---|---|---|---|
| `enabled` | boolean | `false` | distribute the relaxations |
| `workers` | list of strings | `[]` | one per machine: `"user@host:slots"`, or `"user@host:slots:/path/to/python"` when the interpreter that can import cryal is not `python3`. `"local:N"` names this machine with N slots |
| `local_slots` | integer | `1` | slots on this machine; `0` dispatches everything. Ignored when `workers` names `local` |
| `remote_dir` | string | `"~/.cryal_work"` | per-run work directory on each worker |
| `python` | string | `"python3"` | default interpreter on the workers |
| `ssh_key` | string | `""` | private key, if not the default one |
| `ssh_timeout` | integer | `30` | seconds to give up connecting |
| `task_timeout` | integer | `0` | seconds for one remote candidate; 0 means the backend's own timeout + 600 s |
| `max_failures` | integer | `3` | drop a machine after this many consecutive transport failures |
| `keep_remote` | boolean | `false` | leave the remote work directory for debugging |
| `fetch_on_failure` | boolean | `true` | copy a failed candidate's files back |
| `fetch_step` | boolean | `false` | copy every candidate's files back |

Every worker needs passwordless `ssh`, CrYAL and its dependencies, and the
engine reachable from a non-interactive session (`~/.bashrc` is not read, so
`backend.lammps.command` may need an absolute path). Each worker is checked
before the run; one that fails is dropped with the reason. The `ase` backend
can use only one local slot.

---

## 7. What a run writes

Everything goes to `general.output_dir`:

| path | contents |
|---|---|
| `cryal.log` | the run's log: configuration summary, every candidate, per-cycle summaries |
| `database.json` | one record per relaxed structure (below); saved after every evaluation |
| `structures/` | the relaxed structure of every record, as CIF |
| `steps/` | each candidate's working directory (backend input and output) |
| `cycleNNN_report.json` | per cycle: best structure, energy, density and β statistics |
| `cycleNNN_suggestions.json` | BACH: the GP's suggestions for that cycle |
| `pool_cycleNNN/` | BACH: the pool structures of that cycle |
| `final_report.json` | totals, the 10 best, the configuration, `source_stats` per source of candidates |

Every record has `id`, `cycle`, `source`, `space_group`, `energy` (eV per
cell), `density`, `volume`, `a`, `b`, `c`, `alpha`, `beta`, `gamma`, `cif_path`,
and `n_chpi`. Records made by the other searches add:

| key | written by | meaning |
|---|---|---|
| `parents` | GA, BACH + GA, basin hopping, PSO, CMA-ES | ids of the records the candidate was made from |
| `walker`, `accepted` | basin hopping | the walker, and whether it moved there |
| `particle`, `velocity` | PSO | the particle, and the velocity that moved it there |
| `cma` | CMA-ES | the instance, and the sample vector drawn |

`source` says what made a candidate: `random_sgN`, `biased_sgN…`,
`perturb_…`, `seed_…`, `ga_crossover`, `ga_strain`, `ga_rotation`,
`ga_translation`, `ga_immigrant_sgN`, `hop_<move>`, `pso`, `cmaes`.

---

## 8. Resuming, reproducibility, and common errors

**Resuming.** `python run_cryal.py --resume` (or `resume = true`) continues an
interrupted run from `output_dir` without recomputing anything: the database
is saved after every evaluation, the stopping rule's state is rebuilt, and
every search rebuilds its own state from the database — the GA's population,
the basin-hopping walkers, the swarm, CMA-ES's distribution. Keep the input
file, the backend and the potential the same when resuming.

**Reproducibility.** `random_seed` fixes CrYAL's own random draws (space groups,
volumes, cells, the searches' choices), but PyXtal places the molecules with a
generator seeded from the operating system. So two runs with the same seed are
**not** replicas, even on the same machine and versions: they are independent
trajectories. Do not report a run as reproducible from its seed; compare
algorithms over several runs each.

**Common errors.**

| message (abridged) | cause | fix |
|---|---|---|
| `unknown key in [x]; did you mean 'y'?` | a misspelt key | use the suggested name |
| `'k' belongs in [t]` | a key in the wrong table | move it |
| `expected an integer, got 7.5` | wrong type | integers have no decimal point; strings need quotes |
| `general.molecule not found` | the XYZ path | relative paths are relative to where the run is started |
| `nothing sampled from SG N with Z = 4` (warning) | the group's Wyckoff positions do not fit Z | drop the group or change Z |
| `ga_fraction: ... algorithm = "bach"` | `ga_fraction` with another algorithm | set it to 0 |
| `ga.operators gives no weight to crossover, ...` | BACH + GA with only immigrants | give a breeding operator a weight |
| `batch = "believer" applies to ... "ei" or "ucb"` | believer with Thompson sampling | leave `gp.batch` out |
| `backend.uma.task ...` | UMA without a task | set `task = "omc"` for molecular crystals |
| `AttributeError: 'MLIAPDataPy' object has no attribute 'forward_exchange'` | LAMMPS without Kokkos | use the Kokkos command (see INSTALL.md) |
| every candidate `FAILED (unrealistic energy ...)` | `energy_sanity_max` set for another potential's zero | adjust it, or `false` |
| many `FAILED (... max_steps ...)` | relaxation budget too small for this engine | raise `relaxation.max_steps` |
