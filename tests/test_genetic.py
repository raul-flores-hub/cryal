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

"""The genetic algorithm (cryal/genetic.py).

What these guard is what would quietly make a GA run wrong rather than
merely poor: a child whose molecules are not the parent's molecules (a broken
unwrap, a mirrored chiral molecule), a child laid out in an order the
integrity check would misread, a population that counts one minimum twenty
times, and a run that cannot be resumed.
"""

import json
import logging
import os
import shutil
import tempfile
import unittest

import numpy as np
from ase import Atoms
from ase.io import write

from cryal import genetic as G
from cryal import search as S
from cryal.active_learning import CrystalDatabase, _record_result
from cryal.backends.base import check_molecular_integrity, get_reference_bonds
from cryal.config import GA_OPERATORS, Config, load_config
from cryal.utils import check_close_contacts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENZENE = os.path.join(ROOT, "examples", "benzene.xyz")

QUIET = logging.getLogger("cryal.test.genetic")
QUIET.addHandler(logging.NullHandler())
QUIET.propagate = False


def write_xyz(path, symbols, positions):
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n\n")
        for s, p in zip(symbols, positions):
            f.write(f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def crystal(mol_file, cell, centres_frac, rng, wrap=True):
    """Rigid copies of the molecule, randomly oriented, at fractional centres."""
    mol = G.Molecule.from_file(mol_file)
    from ase.io import read
    ref = read(mol_file)
    com = ref.get_center_of_mass()
    cell = np.asarray(cell, float)
    pos = []
    for f in centres_frac:
        R = random_rotation(rng)
        pos.append((ref.positions - com) @ R.T + np.asarray(f) @ cell)
    atoms = Atoms(mol.symbols * len(centres_frac), positions=np.vstack(pos),
                  cell=cell, pbc=True)
    if wrap:
        atoms.wrap()                     # molecules split across the boundary
    return atoms


def distance_spectrum(atoms):
    return np.sort(atoms.get_all_distances(mic=True).ravel())


def internal_distances(shape):
    return np.linalg.norm(shape[:, None] - shape[None, :], axis=2)


def signed_volume(shape):
    """Chirality of the first four atoms: the sign flips under a mirror."""
    return float(np.linalg.det(shape[1:4] - shape[0]))


SKEWED = [[9.0, 0.0, 0.0], [3.5, 8.0, 0.0], [-2.0, 1.5, 10.0]]
CENTRES = [[0.1, 0.1, 0.1], [0.6, 0.15, 0.55], [0.15, 0.6, 0.6], [0.6, 0.6, 0.05]]


class Fixture(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_ga_")
        self.rng = np.random.default_rng(7)
        self.mol = G.Molecule.from_file(BENZENE)
        self.cfg = Config(molecule_file=BENZENE, Z=4, min_cell_axis=4.0,
                          mol_weight=78.11)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def genome(self, cell=SKEWED, seed=None):
        rng = self.rng if seed is None else np.random.default_rng(seed)
        atoms = crystal(BENZENE, cell, CENTRES, rng)
        return G.standardize(G.genome_from_atoms(atoms, self.mol))


class TestGenome(Fixture):

    def test_round_trip_is_the_same_crystal(self):
        atoms = crystal(BENZENE, SKEWED, CENTRES, self.rng)
        back = G.standardize(G.genome_from_atoms(atoms, self.mol)).to_atoms(self.mol)
        self.assertEqual(back.get_chemical_symbols(), atoms.get_chemical_symbols())
        self.assertAlmostEqual(back.get_volume(), atoms.get_volume(), places=6)
        np.testing.assert_allclose(distance_spectrum(back), distance_spectrum(atoms),
                                   atol=1e-8)

    def test_standard_setting_is_niggli_right_handed_and_a_along_x(self):
        atoms = crystal(BENZENE, SKEWED, CENTRES, self.rng)
        g = G.standardize(G.genome_from_atoms(atoms, self.mol))
        reduced, _ = atoms.cell.niggli_reduce()
        np.testing.assert_allclose(np.linalg.norm(g.cell, axis=1),
                                   reduced.lengths(), atol=1e-8)
        self.assertGreater(np.linalg.det(g.cell), 0)
        self.assertAlmostEqual(g.cell[0, 1], 0.0)
        self.assertAlmostEqual(g.cell[0, 2], 0.0)
        self.assertAlmostEqual(g.cell[1, 2], 0.0)

    def test_a_molecule_longer_than_half_the_cell_comes_out_whole(self):
        # The case utils.unwrap_molecule records as an expected failure: an
        # 8-atom chain 9.1 Å long in a 10 Å cell, cut by the boundary.
        path = os.path.join(self.tmp, "chain.xyz")
        chain = np.array([[1.3 * i, 0.0, 0.0] for i in range(8)])
        write_xyz(path, ["C"] * 8, chain)
        mol = G.Molecule.from_file(path)
        atoms = Atoms("C8", positions=chain + [6.0, 2.0, 2.0],
                      cell=[10.0, 4.0, 4.0], pbc=True)
        atoms.wrap()
        g = G.genome_from_atoms(atoms, mol)
        np.testing.assert_allclose(internal_distances(g.shapes[0]),
                                   internal_distances(chain), atol=1e-9)

    def test_a_skewed_cell_does_not_misplace_an_atom(self):
        # A P-1 cell PyXtal drew, 4.4 Å high across a: there, rounding the
        # fractional coordinates takes benzene's 2.41 Å meta C...C pair (which
        # the integrity check's bond list counts as a bond) through the wrong
        # image. Every molecule must come out with the reference's geometry.
        from ase.cell import Cell
        cell = Cell.fromcellpar([4.72, 8.71, 6.99, 101.72, 73.76, 82.14]).array
        from ase.io import read
        ref = read(BENZENE).positions
        for seed in range(20):
            atoms = crystal(BENZENE, cell, [[0.2, 0.3, 0.1], [0.8, 0.7, 0.9]],
                            np.random.default_rng(seed))
            g = G.genome_from_atoms(atoms, self.mol)
            for shape in g.shapes:
                np.testing.assert_allclose(internal_distances(shape),
                                           internal_distances(ref), atol=1e-6)

    def test_the_walk_takes_only_the_covalent_bonds(self):
        lengths = []
        from ase.io import read
        ref = read(BENZENE).positions
        for i, j in self.mol.walk:
            lengths.append(np.linalg.norm(ref[i] - ref[j]))
        self.assertEqual(len(self.mol.walk), 11)            # a tree on 12 atoms
        self.assertLess(max(lengths), 1.5)

    def test_a_crystal_out_of_molecule_order_is_refused(self):
        atoms = crystal(BENZENE, SKEWED, CENTRES, self.rng)
        shuffled = atoms[np.r_[6:12, 0:6, 12:len(atoms)]]
        self.assertIsNone(G.genome_from_atoms(shuffled, self.mol))

    def test_a_left_handed_cell_is_not_mirrored(self):
        # CHFClBr is chiral. A left-handed cell must be fixed by negating the
        # lattice vectors, never by a reflection that would swap enantiomers.
        path = os.path.join(self.tmp, "chfclbr.xyz")
        geom = np.array([[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-1.0, -1.0, 1.0],
                         [-1.0, 1.0, -1.0], [1.1, -1.1, -1.1]])
        write_xyz(path, ["C", "H", "F", "Cl", "Br"], geom)
        mol = G.Molecule.from_file(path)
        left = np.diag([12.0, 12.0, -12.0])
        atoms = crystal(path, left, [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
                        self.rng)
        g0 = G.genome_from_atoms(atoms, mol)
        g = G.standardize(g0)
        self.assertGreater(np.linalg.det(g.cell), 0)
        for before, after in zip(g0.shapes, g.shapes):
            self.assertGreater(signed_volume(before) * signed_volume(after), 0)
        np.testing.assert_allclose(distance_spectrum(g.to_atoms(mol)),
                                   distance_spectrum(atoms), atol=1e-8)


class TestOperators(Fixture):

    def assertRigidCopies(self, child, *parents):
        """Every child molecule has the internal geometry of a parent molecule."""
        pool = [internal_distances(s) for p in parents for s in p.shapes]
        for s in child.shapes:
            d = internal_distances(s)
            self.assertTrue(any(np.allclose(d, p, atol=1e-8) for p in pool))

    def test_crossover_takes_molecules_from_both_parents(self):
        a, b = self.genome(seed=1), self.genome(seed=2)
        made = 0
        for _ in range(20):
            child = G.crossover(a, b, self.rng)
            if child is None:
                continue
            made += 1
            self.assertEqual(child.Z, 4)
            self.assertRigidCopies(child, a, b)
            from_a = sum(any(np.allclose(s, t) for t in a.shapes) for s in child.shapes)
            self.assertTrue(0 < from_a < 4)
            lo, hi = sorted([a.volume, b.volume])
            self.assertTrue(lo - 1e-6 <= child.volume <= hi + 1e-6)
        self.assertGreater(made, 10)

    def test_strain_keeps_volume_and_molecules(self):
        g = self.genome()
        child = G.strain(g, 0.1, self.rng)
        self.assertAlmostEqual(child.volume, g.volume, places=6)
        self.assertFalse(np.allclose(child.cell, g.cell))
        np.testing.assert_allclose(child.shapes, g.shapes)
        np.testing.assert_allclose(child.frac, g.frac)

    def test_rotation_turns_molecules_rigidly(self):
        g = self.genome()
        child = G.rotate(g, 0.5, 45.0, self.rng)
        moved = [not np.allclose(s, t) for s, t in zip(child.shapes, g.shapes)]
        self.assertEqual(sum(moved), 2)
        self.assertRigidCopies(child, g)
        np.testing.assert_allclose(child.frac, g.frac)

    def test_translation_moves_centres_within_the_bound(self):
        g = self.genome()
        child = G.translate(g, 1.0, 0.8, self.rng)
        np.testing.assert_allclose(child.shapes, g.shapes)
        d = (child.frac - g.frac + 0.5) % 1.0 - 0.5
        self.assertTrue(np.all(np.linalg.norm(d @ g.cell, axis=1) <= 0.8 + 1e-9))

    def test_a_realized_child_is_clear_and_in_molecule_order(self):
        a, b = self.genome(seed=1), self.genome(seed=2)
        ref_bonds = get_reference_bonds(BENZENE)
        blocks = [np.arange(m * 12, (m + 1) * 12) for m in range(4)]
        n = 0
        for _ in range(10):
            child = G.crossover(a, b, self.rng)
            if child is None:
                continue
            atoms, why = G.realize(child, self.mol, self.cfg)
            self.assertIsNotNone(atoms, why)
            n += 1
            self.assertEqual(atoms.get_chemical_symbols(), self.mol.symbols * 4)
            self.assertTrue(check_close_contacts(atoms, blocks, self.cfg))
            self.assertTrue(check_molecular_integrity(atoms, 12, ref_bonds))
            self.assertGreaterEqual(min(atoms.cell.lengths()), self.cfg.min_cell_axis)
        self.assertGreater(n, 5)

    def test_contacts_agree_with_check_close_contacts(self):
        contacts = G.Contacts(self.mol, 4, self.cfg)
        blocks = [np.arange(m * 12, (m + 1) * 12) for m in range(4)]
        verdicts = set()
        for seed in range(12):
            g = self.genome(seed=seed)
            for scale in (0.6, 1.0, 1.6):
                atoms = g.with_volume(g.volume * scale).to_atoms(self.mol)
                ours = contacts.clear(atoms, self_images=False)
                self.assertEqual(ours, check_close_contacts(atoms, blocks, self.cfg))
                verdicts.add(ours)
        self.assertEqual(verdicts, {True, False})     # both branches exercised

    def test_image_pairs_is_ase_neighbor_list(self):
        from ase.neighborlist import neighbor_list
        for seed, scale in [(0, 1.0), (1, 0.5), (2, 0.25), (3, 2.0)]:
            g = self.genome(seed=seed)
            atoms = g.with_volume(g.volume * scale).to_atoms(self.mol)
            ours = G.image_pairs(atoms.positions, atoms.cell.array, 3.0)
            ase = neighbor_list("ijdDS", atoms, 3.0)

            def key(i, j, d, D, S):
                return sorted(zip(i.tolist(), j.tolist(), map(tuple, S.tolist())))
            with self.subTest(seed=seed, scale=scale):
                self.assertEqual(key(*ours), key(*ase))
                order = np.lexsort(np.c_[ours[0], ours[1], ours[4]].T[::-1])
                order_ase = np.lexsort(np.c_[ase[0], ase[1], ase[4]].T[::-1])
                np.testing.assert_allclose(ours[3][order], ase[3][order_ase], atol=1e-9)

    def test_contacts_see_every_image_in_a_thin_cell(self):
        # A 4.6 Å axis with two C2 "molecules" 2.3 Å apart along it: each is
        # 2.3 Å from the other directly AND through the image, the case where
        # rounding the fractional coordinates sees only one of the two.
        path = os.path.join(self.tmp, "c2.xyz")
        write_xyz(path, ["C", "C"], [[0, 0, 0], [1.3, 0, 0]])
        mol = G.Molecule.from_file(path)
        cell = [[8.0, 0, 0], [0, 4.6, 0], [0, 0, 8.0]]
        atoms = Atoms("C4", positions=[[1, 1, 1], [2.3, 1, 1], [1, 3.3, 1], [2.3, 3.3, 1]],
                      cell=cell, pbc=True)
        contacts = G.Contacts(mol, 2, self.cfg)
        blocks = [np.arange(0, 2), np.arange(2, 4)]
        self.assertFalse(contacts.clear(atoms))
        self.assertEqual(contacts.clear(atoms, self_images=False),
                         check_close_contacts(atoms, blocks, self.cfg))
        # Pushing one away from the other pushes it into the other's image:
        # no translation fixes a cell this thin, and relieve must not pretend
        # otherwise -- realize then has to expand it.
        g = G.genome_from_atoms(atoms, mol)
        self.assertFalse(contacts.clear(G.relieve(g, mol, contacts).to_atoms(mol)))
        out, _ = G.realize(g, mol, Config(min_cell_axis=4.0))
        self.assertIsNotNone(out)
        self.assertTrue(contacts.clear(out))

    def test_a_molecule_touching_its_own_image_is_not_clear(self):
        path = os.path.join(self.tmp, "c2.xyz")
        write_xyz(path, ["C", "C"], [[0, 0, 0], [1.3, 0, 0]])
        mol = G.Molecule.from_file(path)
        atoms = Atoms("C2", positions=[[1, 1, 1], [2.3, 1, 1]],
                      cell=[3.4, 8.0, 8.0], pbc=True)      # 2.1 Å to the next image
        contacts = G.Contacts(mol, 1, self.cfg)
        self.assertTrue(check_close_contacts(atoms, [np.arange(2)], self.cfg))
        self.assertTrue(contacts.clear(atoms, self_images=False))
        self.assertFalse(contacts.clear(atoms))

    def test_stretch_lengthens_only_the_axis_a_molecule_meets_itself_on(self):
        path = os.path.join(self.tmp, "c2.xyz")
        write_xyz(path, ["C", "C"], [[0, 0, 0], [1.3, 0, 0]])
        mol = G.Molecule.from_file(path)
        atoms = Atoms("C2", positions=[[1, 1, 1], [2.3, 1, 1]],
                      cell=[3.4, 8.0, 8.0], pbc=True)      # 2.1 Å to its image along a
        contacts = G.Contacts(mol, 1, self.cfg)
        g = G.genome_from_atoms(atoms, mol)
        out = G.stretch(g, mol, contacts, max_volume=2 * g.volume)
        self.assertTrue(contacts.clear(out.to_atoms(mol)))
        lengths = np.linalg.norm(out.cell, axis=1)
        self.assertGreaterEqual(lengths[0], 1.3 + 2.5)     # bond + heavy contact
        self.assertLess(lengths[0], 1.3 + 2.5 + 0.2)       # by what it lacked, no more
        np.testing.assert_allclose(lengths[1:], [8.0, 8.0])
        # And not at all past the volume it is allowed.
        self.assertEqual(G.stretch(g, mol, contacts, max_volume=g.volume).volume, g.volume)


class TestSelection(unittest.TestCase):

    @staticmethod
    def rec(i, e, rho):
        return {'id': i, 'energy': e, 'density': rho}

    def test_duplicates_by_energy_and_density_count_once(self):
        recs = [self.rec(1, -10.000, 1.00), self.rec(2, -10.001, 1.001),   # same
                self.rec(3, -10.001, 1.10),                                # denser
                self.rec(4, -9.0, 1.0)]
        pop = G.distinct_lowest(recs, 10, Z=4, e_tol=0.002, rho_tol=0.005)
        self.assertEqual([r['id'] for r in pop], [2, 3, 4])

    def test_population_is_the_lowest_and_no_larger_than_asked(self):
        recs = [self.rec(i, -float(i), 1.0 + 0.1 * i) for i in range(10)]
        pop = G.distinct_lowest(recs, 3, Z=4, e_tol=0.002, rho_tol=0.005)
        self.assertEqual([r['id'] for r in pop], [9, 8, 7])

    def test_full_tournament_picks_the_best_and_exclude_is_honoured(self):
        rng = np.random.default_rng(0)
        pop = [self.rec(i, -float(i), 1.0) for i in range(5)]
        self.assertEqual(G.tournament(pop, 5, rng)['id'], 4)
        self.assertEqual(G.tournament(pop, 5, rng, exclude=pop[4])['id'], 3)

    def test_allocation_is_exact_and_proportional(self):
        w = dict(crossover=0.5, strain=0.1, rotation=0.1, translation=0.1, immigrant=0.2)
        for n in (1, 7, 10, 50, 53):
            c = G.allocate(n, w)
            self.assertEqual(sum(c.values()), n)
            for k in GA_OPERATORS:
                self.assertLessEqual(abs(c[k] - n * w[k]), 1)
        self.assertEqual(G.allocate(10, dict(w, strain=0.0, rotation=0.0,
                                             translation=0.0))['strain'], 0)


class TestConfig(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_ga_cfg_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def load(self, body):
        path = os.path.join(self.tmp, "cryal.toml")
        with open(path, "w") as f:
            f.write(f'[general]\nmolecule = "{BENZENE}"\n'
                    f'[backend]\nengine = "ase"\n'
                    f'[backend.ase]\ncalculator = "tests.stub_calculator.StubCalculator"\n'
                    + body)
        return load_config(path)

    def test_bach_is_the_default(self):
        self.assertEqual(self.load("").algorithm, "bach")

    def test_ga_table(self):
        cfg = self.load('[ga]\npopulation_size = 12\n'
                        'operators = { crossover = 1, immigrant = 1 }\n')
        self.assertEqual(cfg.ga_population_size, 12)
        self.assertEqual(cfg.ga_operators, dict(crossover=1.0, strain=0.0, rotation=0.0,
                                                translation=0.0, immigrant=1.0))

    def test_algorithm_ga_is_read(self):
        path = os.path.join(self.tmp, "c.toml")
        with open(path, "w") as f:
            f.write(f'[general]\nmolecule = "{BENZENE}"\nalgorithm = "ga"\n'
                    f'[backend]\nengine = "ase"\n'
                    f'[backend.ase]\ncalculator = "tests.stub_calculator.StubCalculator"\n')
        self.assertEqual(load_config(path).algorithm, "ga")

    def test_refusals(self):
        for body, says in [
            ('[ga]\noperators = { crossovr = 1 }\n', "did you mean 'crossover'"),
            ('[ga]\noperators = { crossover = 0 }\n', "positive weight"),
            ('[ga]\noperators = { crossover = -1 }\n', "negative"),
            ('[ga]\nfraction_mutated = 0\n', "(0, 1]"),
            ('[ga]\npopulation_size = 0\n', "at least 1"),
            ('[ga]\npopulaton_size = 5\n', "did you mean 'population_size'"),
        ]:
            with self.subTest(body=body):
                with self.assertRaises(ValueError) as ctx:
                    self.load(body)
                self.assertIn(says, str(ctx.exception))


class TestRecords(unittest.TestCase):

    def test_parents_are_recorded_for_ga_children_only(self):
        tmp = tempfile.mkdtemp(prefix="cryal_ga_rec_")
        try:
            db = CrystalDatabase(os.path.join(tmp, "db.json"), os.path.join(tmp, "s"))
            cfg = Config(Z=1, mol_weight=24.0)
            atoms = Atoms("C2", positions=[[0, 0, 0], [1.3, 0, 0]], cell=[8, 8, 8], pbc=True)
            _record_result({'atoms': atoms, 'source': 'random_sg2'}, atoms.copy(),
                           -10.0, cfg, 0, "a", db, QUIET)
            _record_result({'atoms': atoms, 'source': 'ga_crossover', 'parents': [1, 1]},
                           atoms.copy(), -11.0, cfg, 1, "b", db, QUIET)
            self.assertNotIn('parents', db.records[0])
            self.assertEqual(db.records[1]['parents'], [1, 1])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def run_config(output_dir, **kw):
    """A run small enough for the test suite: benzene, P-1, Z = 2, stub energy."""
    # The volume window needs 1.05 g/cm³ inside it: 0.8 .. 1.6 does.
    cfg = Config(molecule_file=BENZENE, Z=2, random_seed=11, output_dir=output_dir,
                 energy_backend="ase",
                 ase_calculator="tests.stub_calculator.VolumeStub",
                 # Lowest near 1.05 g/cm³, a density benzene actually has.
                 ase_calculator_kwargs={"v0": 2 * 78.11 / 0.602214076 / 1.05},
                 ase_max_steps=5, energy_sanity_max=None,
                 num_cycles=2, initial_structures=6, structures_per_cycle=6,
                 max_no_improve_cycles=5,
                 space_groups=[2], sg_weights=[1.0], min_cell_axis=4.0,
                 use_chpi_optimizer=False, use_seeds_folder=False,
                 ga_population_size=4, mol_weight=78.11,
                 _volume_min_computed=2 * 78.11 / 0.602214076 / 1.6,
                 _volume_max_computed=2 * 78.11 / 0.602214076 / 0.8)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


class TestRun(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_ga_run_")
        # The run logger writes to its first output_dir; start each test clean.
        lg = logging.getLogger("cryal")
        for h in list(lg.handlers):
            lg.removeHandler(h)
            h.close()

    def tearDown(self):
        lg = logging.getLogger("cryal")
        for h in list(lg.handlers):
            lg.removeHandler(h)
            h.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def records(self, d):
        with open(os.path.join(d, "database.json")) as f:
            return json.load(f)

    def test_two_generations_then_resume(self):
        out = os.path.join(self.tmp, "ga")
        G.run(run_config(out, log_level="WARNING"))
        recs = self.records(out)
        ids = {r['id'] for r in recs}
        self.assertEqual({r['cycle'] for r in recs}, {0, 1, 2})
        children = [r for r in recs if r['cycle'] > 0]
        self.assertTrue(children)
        for r in children:
            self.assertTrue(r['source'].startswith('ga_'), r['source'])
            self.assertTrue(set(r['parents']) <= ids)
            if r['source'] == 'ga_crossover':
                self.assertEqual(len(r['parents']), 2)
        bred = [r for r in children if not r['source'].startswith('ga_immigrant')]
        self.assertTrue(bred)
        with open(os.path.join(out, "final_report.json")) as f:
            report = json.load(f)
        self.assertEqual(report['algorithm'], 'ga')
        self.assertEqual(sum(s['kept'] for s in report['source_stats'].values()),
                         len(recs))

        G.run(run_config(out, log_level="WARNING", num_cycles=3, resume=True))
        again = self.records(out)
        self.assertEqual(again[:len(recs)], recs)
        self.assertEqual({r['cycle'] for r in again[len(recs):]}, {3})



class TestHybrid(unittest.TestCase):
    """BACH + GA: active_learning.ga_fraction."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_hybrid_")
        for h in list(logging.getLogger("cryal").handlers):
            logging.getLogger("cryal").removeHandler(h)
            h.close()

    def tearDown(self):
        for h in list(logging.getLogger("cryal").handlers):
            logging.getLogger("cryal").removeHandler(h)
            h.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def load(self, body):
        path = os.path.join(self.tmp, "cryal.toml")
        with open(path, "w") as f:
            f.write(f'[general]\nmolecule = "{BENZENE}"\n'
                    f'[backend]\nengine = "ase"\n'
                    f'[backend.ase]\ncalculator = "tests.stub_calculator.StubCalculator"\n'
                    + body)
        return load_config(path)

    def test_slots_are_a_share_of_the_random_part(self):
        from cryal.active_learning import ga_slots
        self.assertEqual(ga_slots(50, True, 0.0), 0)       # BACH as published
        self.assertEqual(ga_slots(50, True, 0.5), 12)      # of 25 random
        self.assertEqual(ga_slots(50, False, 0.5), 25)     # GP off: all 50 random
        self.assertEqual(ga_slots(50, True, 1.0), 25)
        self.assertEqual(ga_slots(7, True, 1.0), 4)        # 7 - 7 // 2

    def test_config(self):
        self.assertEqual(self.load("").ga_fraction, 0.0)
        self.assertEqual(self.load("[active_learning]\nga_fraction = 0.25\n").ga_fraction,
                         0.25)
        for body, says in [
            ("[active_learning]\nga_fraction = 1.5\n", "[0, 1]"),
            ("[active_learning]\nga_fraction = 0.5\n[ga]\noperators = { immigrant = 1 }\n",
             "immigrants alone"),
        ]:
            with self.subTest(body=body):
                with self.assertRaises(ValueError) as ctx:
                    self.load(body)
                self.assertIn(says, str(ctx.exception))
        path = os.path.join(self.tmp, "ga.toml")
        with open(path, "w") as f:
            f.write(f'[general]\nmolecule = "{BENZENE}"\nalgorithm = "ga"\n'
                    f'[backend]\nengine = "ase"\n'
                    f'[backend.ase]\ncalculator = "tests.stub_calculator.StubCalculator"\n'
                    f'[active_learning]\nga_fraction = 0.5\n')
        with self.assertRaises(ValueError) as ctx:
            load_config(path)
        self.assertIn('algorithm = "bach"', str(ctx.exception))

    def test_generate_structures_keeps_the_biased_half(self):
        from cryal.structure_gen import generate_structures
        cfg = run_config(self.tmp)
        sugg = [{'beta': 100.0, 'volume': 180.0}] * 3
        out = generate_structures(BENZENE, cfg, 8, suggestions=sugg, n_random=1)
        families = [S.source_family(s['source']) for s in out]
        self.assertEqual(families.count('random'), 1)
        self.assertEqual(families.count('biased'), 4)

    def test_bach_cycles_carry_ga_children(self):
        from cryal.active_learning import run as bach
        out = os.path.join(self.tmp, "hybrid")
        bach(run_config(out, log_level="WARNING", ga_fraction=1.0,
                        structures_per_cycle=8))
        with open(os.path.join(out, "database.json")) as f:
            recs = json.load(f)
        ids = {r['id'] for r in recs}
        self.assertFalse(any(r['source'].startswith('ga_') for r in recs if r['cycle'] == 0))
        for cycle in (1, 2):
            fams = [S.source_family(r['source']) for r in recs if r['cycle'] == cycle]
            bred = [f for f in fams if f in G.GA_OPERATORS]
            self.assertTrue(bred, fams)
            self.assertNotIn('immigrant', bred)
            # ga_fraction = 1: the random part is all GA, save children that
            # could not be made and were replaced by random structures.
            self.assertGreater(len(bred), fams.count('random'))
        for r in recs:
            if S.source_family(r['source']) in G.GA_OPERATORS:
                self.assertTrue(set(r['parents']) <= ids)
            else:
                self.assertNotIn('parents', r)


if __name__ == "__main__":
    unittest.main()
