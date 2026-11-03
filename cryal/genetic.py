# Copyright 2026 Raúl Rodolfo Flores Mena
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
genetic.py — a genetic algorithm over molecular crystals, as an alternative
search to BACH (`general.algorithm = "ga"`).

What stays the same as BACH
===========================
Everything that turns a candidate into a number: the energy backend, the
pre-relaxation gate, the integrity check, the CH–π step, the parallel pool,
the database, the stopping rule and resume — the loop is cryal/search.py,
shared by every search but BACH. Generation 0 is BACH's cycle 0 call for
call, and `[active_learning]` sets the budget of both. What differs is only
how the next candidates are proposed, which is what a comparison between the
two should isolate.

Call for call is not structure for structure. PyXtal places the molecules
with a generator of its own, seeded from the operating system rather than
from `random_seed` (structure_gen._try_pyxtal passes it no random_state), so
two runs with one seed draw the same cells only until a placement succeeds
in one and fails in the other; after that the streams part. That holds for
two BACH runs as much as for a BACH run and a GA run.

The genome
==========
A crystal is a cell and Z rigid molecules, each one a centre (fractional
coordinates) and the Cartesian positions of its atoms around that centre.
Offspring are written in P1: no operator below keeps a space group, and none
needs to, since every candidate is relaxed before it is judged. Molecules
keep the conformation they relaxed to in their parent.

Atom order is the one contract the rest of CrYAL imposes: molecule m is atoms
m*n .. m*n+n-1, in the order of the molecule file. The integrity check
indexes the relaxed crystal that way, so every structure in the database is
already laid out like that and every child is built like that.

Every genome is put in a standard setting before an operator touches it:
Niggli-reduced, right-handed, and rotated so that a lies along x and b in the
xy plane. Two parents are then described in comparable cells, which a
cut-and-splice needs. The handedness step negates the lattice vectors rather
than mirroring anything: a reflection would turn a chiral molecule into its
enantiomer, which no later check would notice.

Operators
=========
  crossover    cut-and-splice (Deaven–Ho, as in USPEX): each parent is shifted
               at random, a plane cuts both along one axis, the molecules on
               one side come from each; the cell is the two cells blended in
               the proportion each parent contributed, at their blended volume
  strain       the cell is deformed by a random symmetric strain at constant
               volume; molecular centres keep their fractional coordinates
  rotation     some molecules turn about their centres
  translation  some molecules move by up to max_translation Å
  immigrant    a fresh structure from the generator, so the population does
               not close in on itself

A child that does not clear the contact distances in [contacts] first has
its molecules pushed apart as rigid bodies, then is expanded by [volume]
expansion_factors until it clears them — the same top-down rule the
generator follows — and is dropped only if none of them works. The contacts
are checked through every periodic image, a molecule against its own images
included, so a dense, thin cell cannot hide a clash.

Selection
=========
The population is not carried in memory: at each generation it is the
`population_size` lowest-energy distinct structures in the database, over
everything evaluated so far (elitist, and steady-state across generations).
Two structures count as the same when both their energies per molecule and
their densities agree within `duplicate_energy` and `duplicate_density`; a
coarse test, but it keeps one deep minimum, found twenty times, from
becoming twenty parents. Parents are drawn by tournament. Because the
population is a function of the database alone, a resumed run continues
from exactly where an uninterrupted one would have been.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.io import read

from .backends import get_reference_bonds
from .config import GA_BREEDING, GA_OPERATORS, Config
from .search import Strategy, run_search
from .structure_gen import generate_structures
from .utils import check_bond_integrity

#: Attempts at one child before its slot is given up.
MAX_ATTEMPTS = 50


# ---------------------------------------------------------------------------
# The molecule
# ---------------------------------------------------------------------------

@dataclass
class Molecule:
    """What every crystal of this run is made of: symbols, masses, bonds."""
    symbols: List[str]
    masses: np.ndarray
    neighbours: List[List[int]]

    @property
    def size(self) -> int:
        return len(self.symbols)

    @classmethod
    def from_file(cls, path: str) -> "Molecule":
        mol = read(path)
        neighbours = [[] for _ in range(len(mol))]
        for i, j, _ in get_reference_bonds(path):
            neighbours[i].append(j)
            neighbours[j].append(i)
        return cls(mol.get_chemical_symbols(), mol.get_masses(), neighbours)


def _mic(d: np.ndarray, cell: np.ndarray, inv: np.ndarray) -> np.ndarray:
    frac = d @ inv
    return (frac - np.round(frac)) @ cell


def _whole(block: np.ndarray, cell: np.ndarray, mol: Molecule) -> np.ndarray:
    """
    One molecule's atoms made contiguous, by walking its bonds.

    Each bond is taken at its minimum image and the walk adds them up, so the
    molecule comes out whole whatever its size relative to the cell — which
    is what `utils.unwrap_molecule`, imaging every atom against the first
    one, cannot promise for a molecule longer than half the cell. A molecule
    of several fragments has each fragment placed nearest the first atom.
    """
    inv = np.linalg.inv(cell)
    out = block.copy()
    seen = np.zeros(len(block), dtype=bool)
    for root in range(len(block)):
        if seen[root]:
            continue
        if root:
            out[root] = out[0] + _mic(block[root] - block[0], cell, inv)
        seen[root] = True
        stack = [root]
        while stack:
            i = stack.pop()
            for j in mol.neighbours[i]:
                if not seen[j]:
                    out[j] = out[i] + _mic(block[j] - block[i], cell, inv)
                    seen[j] = True
                    stack.append(j)
    return out


# ---------------------------------------------------------------------------
# The genome
# ---------------------------------------------------------------------------

@dataclass
class Genome:
    """A cell and Z rigid molecules: centres (fractional) and shapes (Å)."""
    cell: np.ndarray        # (3, 3), rows are a, b, c
    frac: np.ndarray        # (Z, 3) molecular centres of mass, fractional
    shapes: np.ndarray      # (Z, n, 3) atoms relative to their centre, Å

    @property
    def Z(self) -> int:
        return len(self.frac)

    @property
    def volume(self) -> float:
        return abs(float(np.linalg.det(self.cell)))

    def copy(self) -> "Genome":
        return Genome(self.cell.copy(), self.frac.copy(), self.shapes.copy())

    def with_volume(self, volume: float) -> "Genome":
        """The same genome in a cell scaled isotropically to `volume`."""
        g = self.copy()
        g.cell = g.cell * (volume / self.volume) ** (1.0 / 3.0)
        return g

    def to_atoms(self, mol: Molecule) -> Atoms:
        centres = (self.frac % 1.0) @ self.cell
        positions = (centres[:, None, :] + self.shapes).reshape(-1, 3)
        return Atoms(mol.symbols * self.Z, positions=positions,
                     cell=self.cell, pbc=True)


def genome_from_atoms(atoms: Atoms, mol: Molecule) -> Optional[Genome]:
    """
    Read a crystal into a genome, or None if it is not laid out molecule by
    molecule in the order of the molecule file (see the module docstring).
    """
    n = mol.size
    if len(atoms) == 0 or len(atoms) % n:
        return None
    symbols = atoms.get_chemical_symbols()
    Z = len(atoms) // n
    if symbols != mol.symbols * Z:
        return None
    cell = atoms.cell.array.copy()
    frac, shapes = [], []
    for m in range(Z):
        pos = _whole(atoms.positions[m * n:(m + 1) * n], cell, mol)
        com = mol.masses @ pos / mol.masses.sum()
        shapes.append(pos - com)
        frac.append(com @ np.linalg.inv(cell))
    return Genome(cell, np.array(frac) % 1.0, np.array(shapes))


def standardize(g: Genome) -> Genome:
    """
    Niggli-reduced, right-handed, a along x and b in the xy plane.

    Only the description changes: the crystal is the same one, rotated as a
    rigid body. Reduction also makes the cell's shortest axes its axes, which
    is the form `cell.min_axis` is a meaningful bound on.
    """
    # ASE's convention (ase.build.tools.update_cell_and_positions): the
    # reduced lattice vectors are op.T @ cell, and fractional coordinates go
    # with inv(op).T. The reduced cell ASE returns is already rotated to the
    # standard setting; it is rebuilt here so the molecules turn with it.
    _, op = Cell(g.cell).niggli_reduce()
    cell = op.T @ g.cell
    frac = g.frac @ np.linalg.inv(op).T
    if np.linalg.det(cell) < 0:
        # -cell spans the same lattice and leaves every atom where it is; a
        # reflection would also work on the cell but would mirror the molecules.
        cell, frac = -cell, -frac
    std = Cell.fromcellpar(Cell(cell).cellpar()).array
    rot = np.linalg.solve(cell, std)            # cell @ rot = std, a rotation
    return Genome(std, frac % 1.0, g.shapes @ rot)


# ---------------------------------------------------------------------------
# Operators: each takes standardized parents and returns an unchecked child
# ---------------------------------------------------------------------------

def _rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Proper rotation matrix (det +1), for row vectors: v @ R.T."""
    axis = axis / np.linalg.norm(axis)
    t = np.radians(angle_deg)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def _n_mutated(Z: int, fraction: float) -> int:
    return min(Z, max(1, int(round(fraction * Z))))


def crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Optional[Genome]:
    """
    Cut-and-splice. None when the cut leaves one parent with no molecule in
    the child, which would make it a copy of the other parent.
    """
    Z = a.Z
    fa = (a.frac + rng.random(3)) % 1.0         # any shift is a symmetry of the
    fb = (b.frac + rng.random(3)) % 1.0         # parent: it only moves the cut
    axis = int(rng.integers(3))
    cut = rng.uniform(0.25, 0.75)

    picks = [("a", i) for i in range(Z) if fa[i, axis] < cut] + \
            [("b", i) for i in range(Z) if fb[i, axis] >= cut]
    spare = [("a", i) for i in range(Z) if fa[i, axis] >= cut] + \
            [("b", i) for i in range(Z) if fb[i, axis] < cut]

    def coord(p):
        return (fa if p[0] == "a" else fb)[p[1], axis]

    def gap(p):
        """Distance to the nearest cut plane (at `cut` and at 0 = 1)."""
        x = coord(p)
        return min(abs(x - cut), x, 1.0 - x)

    while len(picks) > Z:                      # the molecules nearest a plane
        picks.remove(min(picks, key=gap))      # are the ones that will clash
    while len(picks) < Z:
        picks.append(spare.pop(int(rng.integers(len(spare)))))

    n_a = sum(p[0] == "a" for p in picks)
    if n_a in (0, Z):
        return None
    w = n_a / Z
    frac = np.array([(fa if p[0] == "a" else fb)[p[1]] for p in picks])
    shapes = np.array([(a if p[0] == "a" else b).shapes[p[1]] for p in picks])
    child = Genome(w * a.cell + (1 - w) * b.cell, frac, shapes)
    if child.volume < 1e-6:
        return None
    return child.with_volume(w * a.volume + (1 - w) * b.volume)


def strain(g: Genome, sigma: float, rng: np.random.Generator) -> Genome:
    """A random symmetric strain, at the parent's volume."""
    eps = rng.normal(0.0, sigma, (3, 3))
    eps = 0.5 * (eps + eps.T)
    child = g.copy()
    child.cell = g.cell @ (np.eye(3) + eps)
    if np.linalg.det(child.cell) <= 1e-6 * g.volume:
        return g.copy()                       # a strain that collapses the cell
    return child.with_volume(g.volume)


def rotate(g: Genome, fraction: float, max_angle: float,
           rng: np.random.Generator) -> Genome:
    """Turn some molecules about their centres by up to max_angle degrees."""
    child = g.copy()
    for m in rng.choice(g.Z, _n_mutated(g.Z, fraction), replace=False):
        R = _rotation(rng.normal(size=3), rng.uniform(-max_angle, max_angle))
        child.shapes[m] = child.shapes[m] @ R.T
    return child


def translate(g: Genome, fraction: float, max_shift: float,
              rng: np.random.Generator) -> Genome:
    """Move some molecular centres by up to max_shift Å."""
    child = g.copy()
    inv = np.linalg.inv(g.cell)
    for m in rng.choice(g.Z, _n_mutated(g.Z, fraction), replace=False):
        d = rng.normal(size=3)
        d *= rng.uniform(0.0, max_shift) / np.linalg.norm(d)
        child.frac[m] = (child.frac[m] + d @ inv) % 1.0
    return child


# ---------------------------------------------------------------------------
# Contacts: the [contacts] distances, over every periodic image
# ---------------------------------------------------------------------------

def image_pairs(positions: np.ndarray, cell: np.ndarray, cutoff: float):
    """
    Every pair of atoms closer than `cutoff` through any periodic image:
    (i, j, d, D, S) with D = positions[j] + S @ cell - positions[i], each pair
    both ways, an atom's own images included (i == j, S != 0).

    The same list as ase.neighborlist.neighbor_list('ijdDS'), which is exact
    but costs milliseconds per call in Python even for fifty atoms. Here
    each pair is first rounded to its nearest image, and then every lattice
    shift that can still reach the cutoff is added: along axis k that is
    floor(cutoff / h_k + 1/2) shifts each way, h_k being the cell's height
    across that axis. In the thin cells a dense search reaches, that is more
    than the single image rounding alone would give.
    """
    inv = np.linalg.inv(cell)
    f = positions @ inv
    df = f[None, :, :] - f[:, None, :]
    base = np.round(df)
    df -= base
    volume = abs(np.linalg.det(cell))
    heights = volume / np.linalg.norm(np.cross(cell[[1, 2, 0]], cell[[2, 0, 1]]), axis=1)
    reach = np.floor(cutoff / heights + 0.5).astype(int)
    out = []
    n = len(positions)
    for S in np.array(np.meshgrid(*[np.arange(-r, r + 1) for r in reach],
                                  indexing="ij")).reshape(3, -1).T:
        D = (df + S) @ cell
        d = np.linalg.norm(D, axis=2)
        hit = d < cutoff
        if not S.any():
            hit[np.arange(n), np.arange(n)] = False
        i, j = np.nonzero(hit)
        if len(i):
            out.append((i, j, d[i, j], D[i, j], S - base[i, j]))
    if not out:
        z = np.zeros(0, dtype=int)
        return z, z, np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    i, j, d, D, S = (np.concatenate(x) for x in zip(*out))
    return i, j, d, D, S.astype(int)


class Contacts:
    """
    The [contacts] minimum distances between atoms of different molecules,
    over every periodic image within reach (image_pairs). Rounding
    fractional coordinates alone finds the right image only for vectors
    shorter than half the cell's height — about 2.5 Å in the dense, thin
    cells a search does reach, which is exactly the length of the contacts
    in question. And where utils.check_close_contacts loops in Python over
    every pair, this is a handful of array operations.
    """

    def __init__(self, mol: Molecule, Z: int, cfg: Config):
        self.h = np.array([s == "H" for s in mol.symbols] * Z)
        self.molid = np.repeat(np.arange(Z), mol.size)
        self.hh, self.heavy, self.hx = (cfg.min_hh_distance, cfg.min_heavy_distance,
                                        cfg.min_hx_distance)
        self.cutoff = max(self.hh, self.heavy, self.hx)

    def pairs(self, atoms: Atoms, margin: float = 0.0, self_images: bool = True,
              shifts: bool = False):
        """
        (i, j, d, D, need) for every listed pair closer than the cutoff +
        margin: atoms of different molecules, and with `self_images` also a
        molecule against its own periodic images. Each pair appears both ways.
        With `shifts`, the lattice shift S of each pair is appended.
        """
        i, j, d, D, S = image_pairs(atoms.positions, atoms.cell.array,
                                    self.cutoff + margin)
        same = self.molid[i] == self.molid[j]
        keep = ~same | (self_images & np.any(S != 0, axis=1))
        i, j, d, D, S = i[keep], j[keep], d[keep], D[keep], S[keep]
        hi, hj = self.h[i], self.h[j]
        need = np.where(hi & hj, self.hh, np.where(~hi & ~hj, self.heavy, self.hx))
        return (i, j, d, D, need, S) if shifts else (i, j, d, D, need)

    def clear(self, atoms: Atoms, self_images: bool = True) -> bool:
        """
        No contact below [contacts]. With self_images=False this is exactly
        utils.check_close_contacts, which ignores every pair within one
        molecule; by default a molecule touching its own periodic image fails
        too, since no amount of pushing molecules apart would fix that.
        """
        _, _, d, _, need = self.pairs(atoms, self_images=self_images)
        return not np.any(d < need)


def relieve(g: Genome, mol: Molecule, contacts: Contacts, iters: int = 100,
            margin: float = 0.2, max_step: float = 0.5,
            patience: int = 10) -> Genome:
    """
    Push overlapping rigid molecules apart, as utils.push_apart_overlaps does
    but vectorized. Every pair of atoms in different molecules closer than
    its threshold + margin asks for the two to move apart along the line
    between them; each molecule moves by the sum of what its n pairs ask,
    divided by sqrt(n), at most max_step Å per iteration; nothing rotates or
    changes shape.

    Why sqrt(n): two benzenes face to face have dozens of short pairs, and
    the plain sum always hits the step limit and bounces between clashes
    instead of settling, while the mean settles but crawls. On crossovers of
    benzene crystals sqrt(n) gave the most usable children in the least time
    of the three. The least-clashing arrangement seen is returned — in a cell
    too thin to fix that is not the last one — and the push stops once the
    total overlap has not fallen by 1 % in `patience` iterations, so a child
    that cannot be fixed at this volume costs little before realize tries
    the next one.

    A splice puts molecules from two crystals side by side at the cut;
    without this, almost no crossover child is usable.
    """
    best, best_excess, stale = g, np.inf, 0
    g = g.copy()
    for _ in range(iters):
        i, _, d, D, need = contacts.pairs(g.to_atoms(mol), margin, self_images=False)
        excess = np.clip(need - d, 0.0, None).sum()
        if excess < 0.99 * best_excess:
            best, best_excess, stale = g.copy(), excess, 0
        else:
            stale += 1
            if stale >= patience:
                break
        short = d < need + margin
        if not short.any():
            return g
        i, d, D, need = i[short], d[short], D[short], need[short]
        push = -((need + margin - d) / (2.0 * np.maximum(d, 1e-8)))[:, None] * D
        step = np.zeros((g.Z, 3))
        np.add.at(step, contacts.molid[i], push)
        count = np.bincount(contacts.molid[i], minlength=g.Z)[:, None]
        step /= np.sqrt(np.maximum(count, 1))
        norm = np.linalg.norm(step, axis=1, keepdims=True)
        step *= np.minimum(1.0, max_step / np.maximum(norm, 1e-12))
        g.frac = (g.frac + step @ np.linalg.inv(g.cell)) % 1.0
    return best


def stretch(g: Genome, mol: Molecule, contacts: Contacts, max_volume: float,
            rounds: int = 5, margin: float = 0.05) -> Genome:
    """
    Lengthen the lattice vectors along which a molecule meets its own
    periodic image too closely.

    A molecule carried from its parent's cell into another one — a crossover
    child's blended cell, a strained one — can be longer than the new cell is
    across, in its own orientation. That clash is between a molecule and
    itself, so pushing molecules apart cannot touch it, and an isotropic
    expansion pays for it in every direction at once. Here only the vectors
    the image shift runs along are lengthened, each by what the worst such
    contact lacks (need - d, plus a margin) — added, not multiplied: a
    0.9 Å contact through a 5.7 Å vector wants 1.6 Å more, not a vector 2.7
    times as long. Repeated until no such contact is left, `rounds` are spent,
    or the volume would exceed max_volume (the largest of
    volume.expansion_factors).
    """
    g = g.copy()
    for _ in range(rounds):
        i, j, d, _, need, S = contacts.pairs(g.to_atoms(mol), shifts=True)
        own = (contacts.molid[i] == contacts.molid[j]) & (d < need)
        if not own.any():
            break
        grow = np.zeros(3)
        for s, lack in zip(S[own], need[own] - d[own] + margin):
            grow[s != 0] = np.maximum(grow[s != 0], lack)
        lengths = np.linalg.norm(g.cell, axis=1)
        cell = g.cell * ((lengths + grow) / lengths)[:, None]
        if abs(np.linalg.det(cell)) > max_volume:
            break
        g.cell = cell
    return g


def realize(child: Genome, mol: Molecule, cfg: Config) -> Tuple[Optional[Atoms], str]:
    """
    Turn a child into a structure the backend can take, or (None, why).

    The child is standardized and held to cell.min_axis; then, at each of
    volume.expansion_factors in turn — the top-down rule the generator uses
    for PyXtal — its molecules are pushed apart (`relieve`) and it is kept at
    the first volume where no contact is below [contacts] and no atoms
    overlap across the periodic boundary.
    """
    try:
        child = standardize(child)
    except Exception as exc:                   # a degenerate cell
        return None, f"cell ({type(exc).__name__})"
    contacts = Contacts(mol, child.Z, cfg)
    why = "axis_short"
    for factor in cfg.volume_expansion_factors:
        g = child.with_volume(child.volume * factor)
        if min(np.linalg.norm(g.cell, axis=1)) < cfg.min_cell_axis:
            continue
        why = "contacts"
        g = stretch(g, mol, contacts, child.volume * max(cfg.volume_expansion_factors))
        atoms = relieve(g, mol, contacts).to_atoms(mol)
        if contacts.clear(atoms) and check_bond_integrity(atoms):
            return atoms, ""
    return None, why


# ---------------------------------------------------------------------------
# Population and selection
# ---------------------------------------------------------------------------

def distinct_lowest(records: List[Dict], size: int, Z: int,
                    e_tol: float, rho_tol: float) -> List[Dict]:
    """
    The `size` lowest-energy records, no two of them the same structure by
    energy per molecule (e_tol, eV) and density (rho_tol, g/cm³).
    """
    kept: List[Dict] = []
    for r in sorted(records, key=lambda r: r['energy']):
        if len(kept) >= size:
            break
        if any(abs(r['energy'] - k['energy']) <= e_tol * Z and
               abs(r['density'] - k['density']) <= rho_tol for k in kept):
            continue
        kept.append(r)
    return kept


def tournament(population: List[Dict], k: int, rng: np.random.Generator,
               exclude: Optional[Dict] = None) -> Dict:
    """The lowest-energy of k members drawn without replacement."""
    pool = [r for r in population if r is not exclude] or population
    picks = rng.choice(len(pool), min(k, len(pool)), replace=False)
    return min((pool[i] for i in picks), key=lambda r: r['energy'])


def allocate(n: int, weights: Dict[str, float]) -> Dict[str, int]:
    """
    n children split over the operators in proportion to their weights,
    by largest remainder: the counts are fixed, not drawn, so every
    generation of a run is built in the same proportions.
    """
    total = sum(weights.values())
    exact = {k: n * w / total for k, w in weights.items()}
    counts = {k: int(np.floor(v)) for k, v in exact.items()}
    by_rest = sorted(exact, key=lambda k: (counts[k] - exact[k], GA_OPERATORS.index(k)))
    for k in by_rest[:n - sum(counts.values())]:
        counts[k] += 1
    return counts


def load_parents(population: List[Dict], mol: Molecule, logger) -> Dict[int, Genome]:
    """Genomes of the population, by record id. One that cannot be read is skipped."""
    genomes = {}
    for r in population:
        try:
            g = genome_from_atoms(read(r['cif_path']), mol)
            genomes[r['id']] = standardize(g) if g is not None else None
        except Exception as exc:
            logger.warning(f"  GA: cannot read parent {r['id']} ({r['cif_path']}): {exc}")
            continue
        if genomes[r['id']] is None:
            logger.warning(f"  GA: parent {r['id']} is not laid out molecule by "
                           f"molecule in the order of the molecule file — skipped")
            del genomes[r['id']]
    return genomes


# ---------------------------------------------------------------------------
# One generation's candidates
# ---------------------------------------------------------------------------

def breed(population: List[Dict], genomes: Dict[int, Genome], n: int,
          mol: Molecule, cfg: Config, rng: np.random.Generator,
          logger, weights: Optional[Dict[str, float]] = None) -> List[Dict]:
    """
    n candidates for the next generation, as dicts the evaluation pipeline
    takes: {'atoms', 'space_group', 'source', 'parents'}. `weights` replaces
    ga.operators (the BACH + GA hybrid gives immigrants none).
    """
    weights = dict(cfg.ga_operators if weights is None else weights)
    population = [r for r in population if r['id'] in genomes]
    counts = allocate(n, weights)
    if not population:
        # Nothing to breed from (every candidate so far was rejected): this
        # generation is all immigrants.
        counts = dict.fromkeys(GA_OPERATORS, 0)
        counts['immigrant'] = n
    elif len(population) < 2 and counts['crossover']:
        # One parent cannot be crossed with anything. Its slots go to the
        # mutations, in their own proportions, or to immigrants without any.
        mutations = {k: weights[k] for k in ("strain", "rotation", "translation")}
        moved = counts.pop('crossover')
        extra = allocate(moved, mutations) if sum(mutations.values()) > 0 \
            else {'immigrant': moved}
        counts = {k: counts.get(k, 0) + extra.get(k, 0) for k in GA_OPERATORS}
        logger.info("  GA: one distinct parent — crossover slots go to mutation")

    k = cfg.ga_tournament_size
    out: List[Dict] = []
    failures: Dict[str, Dict[str, int]] = {}
    for op in ("crossover", "strain", "rotation", "translation"):
        made = 0
        for _ in range(counts[op] * MAX_ATTEMPTS):
            if made >= counts[op]:
                break
            pa = tournament(population, k, rng)
            parents = [pa['id']]
            if op == "crossover":
                pb = tournament(population, k, rng, exclude=pa)
                parents.append(pb['id'])
                child = crossover(genomes[pa['id']], genomes[pb['id']], rng)
            elif op == "strain":
                child = strain(genomes[pa['id']], cfg.ga_strain_sigma, rng)
            elif op == "rotation":
                child = rotate(genomes[pa['id']], cfg.ga_fraction_mutated,
                               cfg.ga_max_rotation, rng)
            else:
                child = translate(genomes[pa['id']], cfg.ga_fraction_mutated,
                                  cfg.ga_max_translation, rng)
            if child is None:
                why = "one-sided cut"
                atoms = None
            else:
                atoms, why = realize(child, mol, cfg)
            if atoms is None:
                f = failures.setdefault(op, {})
                f[why] = f.get(why, 0) + 1
                continue
            out.append({'atoms': atoms, 'space_group': None,
                        'source': f'ga_{op}', 'parents': parents})
            made += 1
        if made < counts[op]:
            why = ", ".join(f"{r} x{c}" for r, c in failures.get(op, {}).items())
            logger.warning(f"  GA: {op} made {made}/{counts[op]} children "
                           f"in {counts[op] * MAX_ATTEMPTS} attempts ({why})")

    if counts['immigrant']:
        immigrants = generate_structures(cfg.molecule_file, cfg, counts['immigrant'],
                                         suggestions=None, logger=logger)
        for s in immigrants:
            s['source'] = 'ga_immigrant_sg' + str(s['space_group'])
            s['parents'] = []
        out.extend(immigrants)

    made = {op: sum(s['source'].startswith(f'ga_{op}') for s in out) for op in GA_OPERATORS}
    logger.info("  GA offspring: " + "  ".join(f"{op} {made[op]}/{counts[op]}"
                                              for op in GA_OPERATORS if counts[op]))
    return out


def bach_children(records: List[Dict], n: int, mol: Molecule, cfg: Config,
                  rng: np.random.Generator, logger) -> List[Dict]:
    """
    BACH + GA (active_learning.ga_fraction): n of a BACH cycle's random
    structures bred instead from the best distinct structures found so far,
    with the breeding operators of ga.operators in their proportions.
    Immigrants get no share: in this mode the random structures that remain
    are the immigrants. Fewer than n come back when children cannot be made
    or there is nothing to breed from; the caller fills the gap.
    """
    weights = {k: (cfg.ga_operators[k] if k in GA_BREEDING else 0.0) for k in GA_OPERATORS}
    population = distinct_lowest(records, cfg.ga_population_size, cfg.Z,
                                 cfg.ga_duplicate_energy, cfg.ga_duplicate_density)
    genomes = load_parents(population, mol, logger)
    if not genomes:
        return []
    children = breed(population, genomes, n, mol, cfg, rng, logger, weights=weights)
    return [c for c in children if not c['source'].startswith('ga_immigrant')]


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

class GeneticSearch(Strategy):
    """general.algorithm = "ga": every generation bred from the population."""

    name = "ga"
    title = "genetic algorithm"
    unit = "GENERATION"

    def propose(self, ctx, cycle):
        cfg = ctx.cfg
        population = distinct_lowest(ctx.db.records, cfg.ga_population_size, cfg.Z,
                                     cfg.ga_duplicate_energy, cfg.ga_duplicate_density)
        genomes = load_parents(population, ctx.mol, ctx.logger)
        if population:
            ctx.logger.info(f"Population: {len(genomes)} distinct parents, "
                            f"E in [{population[0]['energy']:.4f}, "
                            f"{population[-1]['energy']:.4f}] eV")
        return breed(population, genomes, cfg.structures_per_cycle,
                     ctx.mol, cfg, ctx.rng, ctx.logger)

    def settings(self, cfg):
        return {
            'population_size':   cfg.ga_population_size,
            'tournament_size':   cfg.ga_tournament_size,
            'operators':         cfg.ga_operators,
            'strain_sigma':      cfg.ga_strain_sigma,
            'max_rotation':      cfg.ga_max_rotation,
            'max_translation':   cfg.ga_max_translation,
            'fraction_mutated':  cfg.ga_fraction_mutated,
            'duplicate_energy':  cfg.ga_duplicate_energy,
            'duplicate_density': cfg.ga_duplicate_density,
        }


def run(cfg: Config):
    """Execute the genetic-algorithm search (cryal/search.py runs the loop)."""
    run_search(cfg, GeneticSearch())
