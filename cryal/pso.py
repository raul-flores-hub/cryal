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
pso.py — particle-swarm optimization over molecular crystals
(`general.algorithm = "pso"`).

Particle swarm (Kennedy and Eberhart, 1995), as CALYPSO (Wang, Lv, Zhu and
Ma, 2010) uses it for crystals: a swarm of structures, each moving every
cycle by a velocity that remembers where it was going and is pulled toward
the best structure it has visited and the best the swarm has found,

    v  <-  w v  +  c1 r1 (pbest - x)  +  c2 r2 (gbest - x),     x  <-  x + v,

with r1, r2 uniform in [0, 1) and the inertia w falling linearly from
inertia[0] to inertia[1] over the run. Every new position is relaxed, and
the relaxed structure is where the particle is next (the Lamarckian form
CALYPSO uses). A share of every cycle, random_fraction, is fresh random
structures, which belong to no particle and can become the swarm's best.

The particles start from cycle 0's lowest distinct structures ([duplicates]),
one candidate per particle per cycle, with a small random velocity.

What x - y means for two crystals
=================================
A crystal here is the genome of cryal/genetic.py: a cell and Z rigid
molecules. A velocity has three parts, in the particle's own standard frame
(a along x, b in the xy plane; it turns with the crystal whenever the cell is
put back in that frame): a symmetric strain of the cell, a displacement of
each molecular centre (Å), and a turn of each molecule (a rotation vector).
For a particle x and a structure y it is pulled toward:

  - strain: y's cell as a deformation of x's, L_y = L_x F, symmetrized,
    minus the identity;
  - centres: y's molecules matched to x's (Hungarian, on periodic distance,
    over every origin shift that puts one of y's molecules on x's first),
    and each displacement taken to its nearest image;
  - turns: each molecule's orientation found against the molecule file by
    Kabsch, and y's turned into x's by the smallest rotation among those the
    molecule's own symmetry makes equivalent — benzene, D6h, has twelve
    proper rotations that leave it as it was, and a pull of 60 degrees about
    its axis is a pull of none.

The parts are clipped every cycle to max_strain, max_shift and max_turn.
The structure the velocity produces goes through the GA's repair — push
rigid molecules apart, stretch a cell a molecule meets itself in, expand —
before it is relaxed.

State
=====
In the database, like every other search: each candidate's record names its
`particle` and the `velocity` that moved it there, in the frame it was built
in. A particle is at its last record, its personal best is its lowest one
(or where it started), the swarm's best is the lowest structure in the
database. A resumed run carries on from there; only particles that have not
moved yet draw a new starting velocity.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from ase.io import read
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from .config import Config
from .genetic import (Genome, distinct_lowest, duplicate_settings, genome_from_atoms,
                      realize, standardize_with_rotation)
from .search import Strategy, run_search
from .structure_gen import generate_structures


# ---------------------------------------------------------------------------
# Orientations
# ---------------------------------------------------------------------------

def kabsch(ref: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """The proper rotation R with shape ≈ ref @ R.T, both centred."""
    H = ref.T @ shape
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def symmetry_rotations(ref: np.ndarray, symbols: List[str]) -> List[np.ndarray]:
    """
    The proper rotations that map the molecule onto itself, about its centre
    of mass (pymatgen's point-group analysis, which PyXtal brings along).
    Only the identity if that is unavailable: the pulls on orientation are
    then measured without the molecule's symmetry, which is wrong for a
    symmetric molecule but never fails.
    """
    try:
        from pymatgen.core import Molecule as PMolecule
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer
        ops = PointGroupAnalyzer(PMolecule(symbols, ref)).get_symmetry_operations()
        rots = [op.rotation_matrix for op in ops if np.linalg.det(op.rotation_matrix) > 0]
        return rots or [np.eye(3)]
    except Exception:
        return [np.eye(3)]


class Shape:
    """The molecule file's geometry, centred, and its symmetry rotations."""

    def __init__(self, molecule_file: str):
        atoms = read(molecule_file)
        pos, masses = atoms.positions, atoms.get_masses()
        self.ref = pos - masses @ pos / masses.sum()
        self.symmetry = symmetry_rotations(self.ref, atoms.get_chemical_symbols())

    def orientation(self, shape: np.ndarray) -> np.ndarray:
        return kabsch(self.ref, shape)

    def turn(self, R_from: np.ndarray, R_to: np.ndarray) -> np.ndarray:
        """
        The smallest rotation vector taking orientation R_from to R_to, over
        the orientations the molecule's symmetry makes the same as R_to.
        """
        best = None
        for S in self.symmetry:
            rv = Rotation.from_matrix(R_to @ S @ R_from.T).as_rotvec()
            if best is None or np.linalg.norm(rv) < np.linalg.norm(best):
                best = rv
        return best


# ---------------------------------------------------------------------------
# Velocities
# ---------------------------------------------------------------------------

def _zero(Z: int) -> Dict[str, np.ndarray]:
    return {'strain': np.zeros((3, 3)), 'centres': np.zeros((Z, 3)),
            'turns': np.zeros((Z, 3))}


def rotate_velocity(v: Dict[str, np.ndarray], rot: np.ndarray) -> Dict[str, np.ndarray]:
    """The same velocity in a frame turned by rot (row vectors: v @ rot)."""
    return {'strain': rot.T @ v['strain'] @ rot,
            'centres': v['centres'] @ rot,
            'turns': v['turns'] @ rot}


def clip(v: Dict[str, np.ndarray], max_strain: float, max_shift: float,
         max_turn_deg: float) -> Dict[str, np.ndarray]:
    out = {k: a.copy() for k, a in v.items()}
    n = np.linalg.norm(out['strain'])
    if n > max_strain:
        out['strain'] *= max_strain / n
    for key, cap in (('centres', max_shift), ('turns', np.radians(max_turn_deg))):
        n = np.linalg.norm(out[key], axis=1, keepdims=True)
        out[key] *= np.minimum(1.0, cap / np.maximum(n, 1e-12))
    return out


def to_json(v: Dict[str, np.ndarray]) -> Dict[str, list]:
    return {k: np.round(a, 6).tolist() for k, a in v.items()}


def from_json(v: Dict[str, list]) -> Dict[str, np.ndarray]:
    return {k: np.array(a, dtype=float) for k, a in v.items()}


def match(x: Genome, y: Genome) -> Tuple[np.ndarray, np.ndarray]:
    """
    y's molecules matched to x's: (perm, shift) with y's molecule perm[m]
    taken as x's molecule m once y is moved by the fractional shift, chosen
    over the shifts that put one of y's molecules on x's first.
    """
    best = (np.inf, None, None)
    for j in range(y.Z):
        shift = y.frac[j] - x.frac[0]
        d = (y.frac[None, :, :] - shift) - x.frac[:, None, :]
        d -= np.round(d)
        cost = np.linalg.norm(d @ x.cell, axis=2)
        rows, cols = linear_sum_assignment(cost)
        total = cost[rows, cols].sum()
        if total < best[0]:
            best = (total, cols, shift)
    return best[1], best[2]


def pull(x: Genome, y: Genome, shape: Shape,
         orient_x: Optional[List[np.ndarray]] = None) -> Dict[str, np.ndarray]:
    """y - x: the velocity that would take x to y in one step (see the module docstring)."""
    F = np.linalg.solve(x.cell, y.cell)               # y.cell = x.cell @ F
    strain = 0.5 * (F + F.T) - np.eye(3)
    perm, shift = match(x, y)
    d = (y.frac[perm] - shift) - x.frac
    d -= np.round(d)
    centres = d @ x.cell
    orient_x = orient_x or [shape.orientation(s) for s in x.shapes]
    turns = np.array([shape.turn(orient_x[m], shape.orientation(y.shapes[perm[m]]))
                      for m in range(x.Z)])
    return {'strain': strain, 'centres': centres, 'turns': turns}


def step(x: Genome, v: Dict[str, np.ndarray]) -> Genome:
    """Where velocity v takes x: strained cell, moved centres, turned molecules."""
    child = x.copy()
    child.frac = (x.frac + v['centres'] @ np.linalg.inv(x.cell)) % 1.0
    child.cell = x.cell @ (np.eye(3) + v['strain'])
    for m in range(x.Z):
        R = Rotation.from_rotvec(v['turns'][m]).as_matrix()
        child.shapes[m] = x.shapes[m] @ R.T
    return child


def update(v, p, g, w, c1, c2, rng) -> Dict[str, np.ndarray]:
    """w v + c1 r1 p + c2 r2 g, with fresh r1, r2 per molecule and for the cell."""
    out = {}
    for key in v:
        shape = (1,) if key == 'strain' else (len(v[key]), 1)
        r1, r2 = rng.random(shape), rng.random(shape)
        if key == 'strain':
            r1, r2 = r1[0], r2[0]
        out[key] = w * v[key] + c1 * r1 * p[key] + c2 * r2 * g[key]
    return out


# ---------------------------------------------------------------------------
# The swarm
# ---------------------------------------------------------------------------

def swarm_size(cfg: Config) -> int:
    return cfg.structures_per_cycle - int(round(cfg.pso_random_fraction
                                                * cfg.structures_per_cycle))


def starting_points(records: List[Dict], cfg: Config, mol=None) -> List[Dict]:
    """The particles' starting structures: cycle 0's lowest distinct ones."""
    return distinct_lowest([r for r in records if r['cycle'] == 0], swarm_size(cfg),
                           cfg, mol)


def inertia(cfg: Config, cycle: int) -> float:
    w0, w1 = cfg.pso_inertia
    span = max(cfg.num_cycles - 1, 1)
    return w0 + (w1 - w0) * min(max(cycle - 1, 0), span) / span


class ParticleSwarm(Strategy):

    name = "pso"
    title = "particle swarm"

    def start(self, ctx):
        cfg = ctx.cfg
        self.mol = ctx.mol
        self.shape = Shape(cfg.molecule_file)
        self.caps = (cfg.pso_max_strain, cfg.pso_max_shift, cfg.pso_max_turn)
        ctx.logger.info(f"PSO: the molecule has {len(self.shape.symmetry)} proper "
                        "symmetry rotations, used when comparing orientations")

    def state(self, records, cfg, mol):
        """Each particle's (current, best, velocity-or-None)."""
        start = starting_points(records, cfg, mol)
        state = {p: [r, r, None] for p, r in enumerate(start)}
        for r in sorted(records, key=lambda r: r['id']):
            p = r.get('particle')
            if p in state:
                state[p][0] = r
                state[p][2] = r['velocity']
                if r['energy'] < state[p][1]['energy']:
                    state[p][1] = r
        return state

    def propose(self, ctx, cycle):
        cfg, logger, rng = ctx.cfg, ctx.logger, ctx.rng
        n = cfg.structures_per_cycle
        state = self.state(ctx.db.records, cfg, ctx.mol)
        gbest = ctx.db.best_record()
        w = inertia(cfg, cycle)
        out: List[Dict] = []
        failed: Dict[str, int] = {}
        if state:
            g_genome = self.genome(gbest)
            logger.info(f"Swarm: {len(state)} particles, inertia {w:.2f}; "
                        f"best E={gbest['energy']:.4f} eV (record {gbest['id']})")
        for p, (current, best, vel) in sorted(state.items()):
            x_own = self.genome(current, standard=False)
            pb = self.genome(best)
            if x_own is None or pb is None or g_genome is None:
                failed['unreadable'] = failed.get('unreadable', 0) + 1
                continue
            x, rot = standardize_with_rotation(x_own)
            if vel is None:
                v = self.initial_velocity(x.Z, rng)
            else:
                v = rotate_velocity(from_json(vel), rot)
            orient = [self.shape.orientation(s) for s in x.shapes]
            v = update(v, pull(x, pb, self.shape, orient), pull(x, g_genome, self.shape, orient),
                       w, cfg.pso_c1, cfg.pso_c2, rng)
            v = clip(v, cfg.pso_max_strain, cfg.pso_max_shift, cfg.pso_max_turn)
            try:
                child, rot_c = standardize_with_rotation(step(x, v))
            except Exception:
                failed['cell'] = failed.get('cell', 0) + 1
                continue
            atoms, why = realize(child, ctx.mol, cfg, standardized=True)
            if atoms is None:
                failed[why] = failed.get(why, 0) + 1
                continue
            out.append({'atoms': atoms, 'space_group': None, 'source': 'pso',
                        'parents': [current['id']], 'particle': p,
                        'velocity': to_json(rotate_velocity(v, rot_c))})
        if failed:
            logger.warning("  PSO: " + ", ".join(f"{k} x{c}" for k, c in failed.items())
                           + " — those particles stay where they are this cycle")
        if len(out) < n:
            out += generate_structures(cfg.molecule_file, cfg, n - len(out),
                                       suggestions=None, logger=logger)
        return out

    def initial_velocity(self, Z, rng):
        """
        A small random start, each part at most 30 % of its cap. Without it
        the particle at the swarm's best would be pulled by nothing and never
        move.
        """
        small = [0.3 * c for c in self.caps]
        s = rng.normal(size=(3, 3))
        v = {'strain': 0.5 * (s + s.T) * small[0],
             'centres': rng.normal(size=(Z, 3)) * small[1],
             'turns': rng.normal(size=(Z, 3)) * np.radians(small[2])}
        return clip(v, *small)

    def genome(self, record, standard=True):
        try:
            g = genome_from_atoms(read(record['cif_path']), self.mol)
        except Exception:
            return None
        if g is None:
            return None
        return standardize_with_rotation(g)[0] if standard else g

    def settings(self, cfg):
        return {
            'particles':       swarm_size(cfg),
            'random_fraction': cfg.pso_random_fraction,
            'inertia':         cfg.pso_inertia,
            'c1':              cfg.pso_c1,
            'c2':              cfg.pso_c2,
            'max_strain':      cfg.pso_max_strain,
            'max_shift':       cfg.pso_max_shift,
            'max_turn':        cfg.pso_max_turn,
            'duplicates':      duplicate_settings(cfg),
        }


def run(cfg: Config):
    """Execute the particle swarm (cryal/search.py runs the loop)."""
    run_search(cfg, ParticleSwarm())
