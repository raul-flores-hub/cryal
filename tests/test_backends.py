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

"""Selectable energy backends, and the checks that wrap every one of them.

Two contracts are guarded here.

The first is backward compatibility: an input file without `backend.engine`
-- including the one archived with v1.0.0, which produced the run reported in
the article, once converted to cryal.toml -- must still mean LAMMPS/MACE-OFF, and `cryal.lammps_runner`'s
module-level API must keep working, because that is the extension point the
released README documents.

The second is that the guards live in the base class, not in one backend. The
pre-relaxation integrity gate is the expensive lesson of this project: a
structure with sub-Angstrom overlaps kills the engine minutes later with an
error that points somewhere else entirely. Every backend, present and future,
gets that gate for free -- and these tests fail if it is ever skipped.
"""

import logging
import os
import shutil
import tempfile
import threading
import unittest

import numpy as np
from ase import Atoms

from cryal.backends import (EnergyBackend, available_backends, backend_class,
                            get_backend)
from cryal.backends.ase_calculator import (AseCalculatorBackend,
                                            RelaxationBudget, _import_dotted)
from cryal.backends.lammps_mace import LammpsMaceBackend
from cryal.config import Config, load_config


MINIMAL = """
[general]
molecule = "examples/benzene.xyz"
Z = 4
"""

ASE_STUB = """
[backend]
engine = "ase"

[backend.ase]
calculator = "tests.stub_calculator.StubCalculator"
"""


def sane_structure():
    """Two C2 'molecules', all distances comfortably above the gate."""
    return Atoms("C4",
                 positions=[[2.0, 2.0, 2.0], [3.5, 2.0, 2.0],
                            [2.0, 8.0, 2.0], [3.5, 8.0, 2.0]],
                 cell=np.eye(3) * 15.0, pbc=True)


def overlapping_structure():
    """Two atoms 0.3 A apart through the periodic boundary."""
    atoms = sane_structure()
    atoms.positions[2] = [14.9, 2.0, 2.0]
    atoms.positions[3] = [0.2, 2.0, 2.0]
    return atoms


#: sentinel meaning "hand back the structure you were given"
ECHO = object()


class RecordingBackend(EnergyBackend):
    """A backend that records its calls and returns what it is told to."""

    name = "recording"

    def __init__(self, cfg, logger=None, energy=-1000.0, relaxed=ECHO):
        super().__init__(cfg, logger)
        self.calls = []
        self.energy = energy
        self.relaxed = relaxed

    def relax(self, atoms, step_dir, logger=None):
        self.calls.append(step_dir)
        return (atoms.copy() if self.relaxed is ECHO else self.relaxed,
                self.energy)


class TestRegistry(unittest.TestCase):

    def test_default_config_selects_the_lammps_backend(self):
        self.assertEqual(Config().energy_backend, "lammps_mace")
        self.assertIs(type(get_backend(Config())), LammpsMaceBackend)

    def test_both_backends_are_registered(self):
        for name in ("lammps_mace", "ase"):
            self.assertIn(name, available_backends())

    def test_aliases_resolve(self):
        self.assertIs(backend_class("lammps"), LammpsMaceBackend)
        self.assertIs(backend_class("ase_calculator"), AseCalculatorBackend)

    def test_lookup_is_case_and_space_insensitive(self):
        self.assertIs(backend_class("  LAMMPS_MACE "), LammpsMaceBackend)

    def test_unknown_backend_names_the_alternatives(self):
        with self.assertRaises(ValueError) as ctx:
            backend_class("vasp")
        self.assertIn("vasp", str(ctx.exception))
        self.assertIn("lammps_mace", str(ctx.exception))


class TestEvaluationContract(unittest.TestCase):
    """base.evaluate(): gate, then relax, then the post-relaxation checks."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.step = os.path.join(self.tmp, "step")
        self.cfg = Config()
        self.cfg.check_molecular_integrity = False

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _evaluate(self, backend, atoms, ref_bonds=None, mol_size=2):
        return backend.evaluate(atoms, self.step, ref_bonds or [], mol_size)

    def test_a_sane_structure_reaches_the_engine(self):
        backend = RecordingBackend(self.cfg)
        relaxed, energy = self._evaluate(backend, sane_structure())
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(energy, -1000.0)
        self.assertEqual(len(relaxed), 4)

    def test_the_gate_runs_before_the_engine(self):
        # The whole point: an overlapping structure must never be handed to
        # the relaxation, whatever the engine is.
        backend = RecordingBackend(self.cfg)
        relaxed, energy = self._evaluate(backend, overlapping_structure())
        self.assertEqual(backend.calls, [])
        self.assertIsNone(relaxed)
        self.assertIsNone(energy)

    def test_the_step_directory_is_created_for_the_engine(self):
        backend = RecordingBackend(self.cfg)
        self._evaluate(backend, sane_structure())
        self.assertTrue(os.path.isdir(self.step))

    def test_a_failed_relaxation_is_passed_through(self):
        backend = RecordingBackend(self.cfg, relaxed=None, energy=None)
        self.assertEqual(self._evaluate(backend, sane_structure()), (None, None))

    def test_a_relaxation_without_an_energy_is_a_failure(self):
        backend = RecordingBackend(self.cfg, energy=None)
        self.assertEqual(self._evaluate(backend, sane_structure()), (None, None))

    def test_an_implausible_energy_is_rejected(self):
        backend = RecordingBackend(self.cfg, energy=+5.0)
        self.assertEqual(self._evaluate(backend, sane_structure()), (None, None))

    def test_the_energy_bound_can_be_switched_off(self):
        self.cfg.energy_sanity_max = None
        backend = RecordingBackend(self.cfg, energy=+5.0)
        _, energy = self._evaluate(backend, sane_structure())
        self.assertEqual(energy, +5.0)

    def test_a_broken_molecule_is_rejected_after_relaxation(self):
        self.cfg.check_molecular_integrity = True
        atoms = sane_structure()
        atoms.positions[3] = [7.0, 8.0, 2.0]     # second 'molecule' pulled apart
        backend = RecordingBackend(self.cfg)
        result = self._evaluate(backend, atoms, ref_bonds=[(0, 1, 1.7)])
        self.assertEqual(len(backend.calls), 1)  # it did reach the engine
        self.assertEqual(result, (None, None))   # and was rejected afterwards

    def test_intact_molecules_survive_the_integrity_check(self):
        self.cfg.check_molecular_integrity = True
        backend = RecordingBackend(self.cfg)
        _, energy = self._evaluate(backend, sane_structure(),
                                   ref_bonds=[(0, 1, 1.7)])
        self.assertEqual(energy, -1000.0)


class TestConfigSelection(unittest.TestCase):

    def setUp(self):
        self._paths = []

    def tearDown(self):
        for p in self._paths:
            os.unlink(p)

    def _write(self, body):
        fd, path = tempfile.mkstemp(suffix="_cryal.toml")
        with os.fdopen(fd, "w") as f:
            f.write(body)
        self._paths.append(path)
        return path

    def test_an_input_without_the_key_still_means_lammps(self):
        # Backward-compatibility contract: this is the v1.0.0 input.
        self.assertEqual(load_config(self._write(MINIMAL)).energy_backend,
                         "lammps_mace")

    def test_an_unknown_backend_fails_at_load_time(self):
        with self.assertRaises(ValueError):
            load_config(self._write(MINIMAL + '[backend]\nengine = "quantum_espresso"\n'))

    def test_the_lammps_backend_requires_its_input_script(self):
        cfg = Config()
        cfg.lammps_input = "/nonexistent/in.lammps"
        with self.assertRaises(FileNotFoundError):
            LammpsMaceBackend.validate_config(cfg)

    def test_the_ase_backend_does_not_require_a_lammps_script(self):
        cfg = load_config(self._write(
            MINIMAL + ASE_STUB +
            '[backend.lammps]\ninput = "/nonexistent/in.lammps"\n'))
        self.assertEqual(cfg.energy_backend, "ase")

    def test_a_missing_calculator_fails_at_load_time(self):
        with self.assertRaises(ValueError):
            load_config(self._write(MINIMAL + '[backend]\nengine = "ase"\n'))

    def test_an_unimportable_calculator_fails_at_load_time(self):
        with self.assertRaises(ValueError):
            load_config(self._write(
                MINIMAL + '[backend]\nengine = "ase"\n'
                '[backend.ase]\ncalculator = "nosuchpackage.Calculator"\n'))

    def test_an_unknown_optimizer_fails_at_load_time(self):
        with self.assertRaises(ValueError):
            load_config(self._write(
                MINIMAL + ASE_STUB +
                '[relaxation]\noptimizer = "SteepestDescent"\n'))

    def test_calculator_keyword_arguments_are_typed(self):
        cfg = load_config(self._write(
            MINIMAL + ASE_STUB +
            'calculator_kwargs = { energy = -2500.0, force = 0, device = "cuda" }\n'))
        self.assertEqual(cfg.ase_calculator_kwargs,
                         {"energy": -2500.0, "force": 0, "device": "cuda"})

    def test_the_energy_bound_can_be_disabled_from_the_input(self):
        cfg = load_config(self._write(MINIMAL + "[backend]\nenergy_sanity_max = false\n"))
        self.assertIsNone(cfg.energy_sanity_max)


class TestAseBackend(unittest.TestCase):
    """The second backend — the one that proves the abstraction holds."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.step = os.path.join(self.tmp, "step")
        self.cfg = Config()
        self.cfg.energy_backend = "ase"
        self.cfg.ase_calculator = "tests.stub_calculator.StubCalculator"
        self.cfg.ase_calculator_kwargs = {"energy": -2500.0}
        self.cfg.ase_max_steps = 5
        self.cfg.check_molecular_integrity = False

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_relaxation_returns_structure_and_energy(self):
        backend = get_backend(self.cfg)
        relaxed, energy = backend.evaluate(sane_structure(), self.step, [], 2)
        self.assertIsInstance(backend, AseCalculatorBackend)
        self.assertEqual(energy, -2500.0)
        self.assertEqual(len(relaxed), 4)
        self.assertIsNone(relaxed.calc)   # nothing holds on to the calculator

    def test_it_writes_its_step_output(self):
        get_backend(self.cfg).evaluate(sane_structure(), self.step, [], 2)
        for name in ("minimized.cif", "energy.dat", "relax.log"):
            self.assertTrue(os.path.exists(os.path.join(self.step, name)), name)

    def test_the_calculator_is_built_once_and_reused(self):
        backend = get_backend(self.cfg)
        first = backend.calculator
        self.assertIs(backend.calculator, first)

    def test_a_factory_function_works_as_well_as_a_class(self):
        self.cfg.ase_calculator = "tests.stub_calculator.make_stub"
        _, energy = get_backend(self.cfg).evaluate(sane_structure(), self.step, [], 2)
        self.assertEqual(energy, -2500.0)

    def test_a_relaxation_that_does_not_converge_is_a_failure(self):
        # A structure still far from a minimum has not been evaluated at one,
        # and its energy must not enter the database as if it had.
        self.cfg.ase_calculator_kwargs = {"energy": -2500.0, "force": 1.0}
        result = get_backend(self.cfg).evaluate(sane_structure(), self.step, [], 2)
        self.assertEqual(result, (None, None))

    def test_the_gate_applies_to_this_backend_too(self):
        result = get_backend(self.cfg).evaluate(overlapping_structure(),
                                                self.step, [], 2)
        self.assertEqual(result, (None, None))

    def test_positions_only_relaxation_is_available(self):
        self.cfg.ase_relax_cell = False
        cell_before = sane_structure().cell.array.copy()
        relaxed, _ = get_backend(self.cfg).evaluate(sane_structure(), self.step, [], 2)
        np.testing.assert_allclose(relaxed.cell.array, cell_before)

    def test_describe_names_the_calculator(self):
        self.assertIn("stub_calculator", get_backend(self.cfg).describe())

    def test_dotted_path_errors_are_actionable(self):
        for path in ("", "EMT", "nosuchpackage.Thing",
                     "tests.stub_calculator.NoSuchClass"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    _import_dotted(path)


class TestRejectionReason(unittest.TestCase):
    """Every rejection says why, where the caller can print it.

    "FAILED" is the same word for a structure that was never going to work and
    for a setting that was too tight, and only one of those is the user's to
    fix. The reason is what tells them apart, in the run log and in what a
    remote worker sends home.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.step = os.path.join(self.tmp, "step")
        self.cfg = Config()
        self.cfg.energy_backend = "ase"
        self.cfg.ase_calculator = "tests.stub_calculator.StubCalculator"
        self.cfg.ase_calculator_kwargs = {"energy": -2500.0}
        self.cfg.check_molecular_integrity = False

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_the_gate_says_what_it_rejected(self):
        backend = get_backend(self.cfg)
        backend.evaluate(overlapping_structure(), self.step, [], 2)
        self.assertIn("overlap", backend.last_rejection)

    def test_the_energy_bound_names_the_setting_that_applied_it(self):
        self.cfg.energy_sanity_max = -3000.0      # -2500 eV is above it
        backend = get_backend(self.cfg)
        backend.evaluate(sane_structure(), self.step, [], 2)
        self.assertIn("backend.energy_sanity_max", backend.last_rejection)

    def test_a_reason_does_not_survive_into_the_next_candidate(self):
        # The pool prints this next to a candidate's name; a stale reason
        # would be filed against a structure that never had that problem.
        backend = get_backend(self.cfg)
        backend.evaluate(overlapping_structure(), self.step, [], 2)
        self.assertIsNotNone(backend.last_rejection)
        backend.evaluate(sane_structure(), self.step, [], 2)
        self.assertIsNone(backend.last_rejection)

    def test_a_backend_that_fails_without_a_word_still_gets_one(self):
        backend = RecordingBackend(self.cfg, energy=None)
        backend.evaluate(sane_structure(), self.step, [], 2)
        self.assertTrue(backend.last_rejection)

    def test_reasons_do_not_cross_threads(self):
        # A thread_safe backend serves several of the pool's local slots from
        # one instance. Two candidates in flight must not read each other's.
        backend = get_backend(self.cfg)
        done = threading.Event()

        def reject_elsewhere():
            backend.evaluate(overlapping_structure(),
                             os.path.join(self.tmp, "other"), [], 2)
            done.set()

        t = threading.Thread(target=reject_elsewhere)
        t.start()
        done.wait(timeout=10)
        t.join()
        self.assertIsNone(backend.last_rejection)


class TestStepBudget(unittest.TestCase):
    """relaxation.max_steps, and the tally that keeps it from eating a cycle in silence.

    A relaxation that runs out of steps is discarded, which is correct: it was
    not evaluated at a minimum. What is not correct is that this looks exactly
    like a candidate losing on its merits -- and the budget does not carry
    between engines, so a value calibrated for MACE (~25 steps here) quietly
    throws structures away under UMA (39 to 582 on the same geometries).
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.step = os.path.join(self.tmp, "step")
        self.cfg = Config()
        self.cfg.energy_backend = "ase"
        self.cfg.ase_calculator = "tests.stub_calculator.StubCalculator"
        self.cfg.ase_calculator_kwargs = {"energy": -2500.0, "force": 1.0}
        self.cfg.ase_max_steps = 5
        self.cfg.check_molecular_integrity = False

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _moved(self):
        """A structure the stub calculator has not already answered for."""
        atoms = sane_structure()
        atoms.positions[0][0] += 0.1
        return atoms

    def test_running_out_of_steps_names_the_setting_to_raise(self):
        backend = get_backend(self.cfg)
        backend.evaluate(sane_structure(), self.step, [], 2)
        self.assertIn("relaxation.max_steps", backend.last_rejection)
        self.assertIn("5", backend.last_rejection)

    def test_the_reason_says_how_far_the_forces_still_were(self):
        # The one number that separates "the budget was short" from "this
        # candidate was hopeless": the stub leaves 1 eV/A against a 0.05 target.
        backend = get_backend(self.cfg)
        backend.evaluate(sane_structure(), self.step, [], 2)
        self.assertIn("1.000", backend.last_rejection)

    def test_a_cycle_that_lost_structures_to_the_budget_says_so(self):
        self.cfg.ase_calculator_kwargs = {"energy": -2500.0, "force": 0.0}
        backend = get_backend(self.cfg)
        backend.evaluate(sane_structure(), self.step, [], 2)   # converges
        backend.calculator.constant_force = 1.0                # nothing does now
        backend.evaluate(self._moved(), self.step, [], 2)

        level, message = backend.relaxation_budget().report()
        self.assertEqual(level, logging.WARNING)
        self.assertIn("1 of 2", message)
        self.assertIn("relaxation.max_steps", message)

    def test_a_comfortable_budget_stays_quiet(self):
        # A line printed every cycle is a line nobody reads.
        self.cfg.ase_calculator_kwargs = {"energy": -2500.0, "force": 0.0}
        self.cfg.ase_max_steps = 500
        backend = get_backend(self.cfg)
        backend.evaluate(sane_structure(), self.step, [], 2)
        self.assertIsNone(backend.relaxation_budget().report())

    def test_a_budget_nearly_spent_is_flagged_before_it_bites(self):
        # Nothing has been lost yet, which is the point: the cycle after this
        # one is where a slower structure disappears.
        budget = RelaxationBudget(limit=10)
        budget.record(steps=9, converged=True)
        level, message = budget.report()
        self.assertEqual(level, logging.INFO)
        self.assertIn("9 of 10", message)

    def test_each_cycle_answers_for_itself(self):
        budget = RelaxationBudget(limit=10)
        budget.record(steps=10, converged=False)
        budget.reset()
        self.assertIsNone(budget.report())

    def test_a_workers_numbers_fold_into_the_servers(self):
        # The relaxation ran on another machine; it still spent this run's
        # budget, so the cycle report must count it.
        budget = RelaxationBudget(limit=10)
        budget.record(steps=4, converged=True)
        budget.merge({"converged": 2, "exhausted": 1, "max_steps_used": 9})
        self.assertEqual(budget.as_dict(),
                         {"converged": 3, "exhausted": 1, "max_steps_used": 9})


class TestCompatibilityShim(unittest.TestCase):
    """cryal.lammps_runner is the API the released README documents."""

    def test_the_old_entry_points_still_import(self):
        from cryal.lammps_runner import (evaluate_structure,  # noqa: F401
                                         get_reference_bonds,
                                         check_molecular_integrity,
                                         _lammps_env)

    def test_the_old_call_still_gates_before_running_lammps(self):
        # backend.lammps.command is nonsense on purpose: if the gate did not fire
        # first, this would try to run it.
        from cryal.lammps_runner import evaluate_structure
        cfg = Config()
        cfg.lammps_command = "definitely-not-a-real-binary"
        tmp = tempfile.mkdtemp()
        try:
            result = evaluate_structure(overlapping_structure(), cfg,
                                        os.path.join(tmp, "step"), [], 2)
            self.assertEqual(result, (None, None))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_lammps_environment_exposes_site_packages(self):
        from cryal.lammps_runner import _lammps_env
        import sysconfig
        env = _lammps_env()
        self.assertIn(sysconfig.get_paths()["purelib"], env["PYTHONPATH"])


if __name__ == "__main__":
    unittest.main()
