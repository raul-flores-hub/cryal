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

"""Distributing evaluation must not change what a run finds.

The contract guarded here is not "it goes faster". It is that a cycle
evaluated on five machines produces the database a cycle evaluated on one
would have produced:

  * every candidate is evaluated exactly once -- not broadcast to several
    machines and reduced, which is not parallelism and is not the same
    experiment;
  * results are recorded in candidate order, so ids and the CIF a record
    points at do not depend on which machine happened to finish first;
  * a machine that goes away costs one candidate at most, never the run;
  * a backend that cannot run twice at once in one process is not asked to.

The transport is faked throughout. These tests must pass on a laptop with no
network, so what they exercise is the scheduler, the bookkeeping and the
serialisation -- everything except ssh itself.
"""

import contextlib
import io
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import unittest

import numpy as np
from ase import Atoms

from cryal.config import Config, load_config, parse_input
from cryal.parallel import (Worker, WorkerError, WorkerPool, atoms_from_dict,
                            atoms_to_dict, build_pool, parse_workers)


def quiet_logger(name="cryal.test.parallel"):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def structure(seed: int = 0) -> Atoms:
    return Atoms("C4",
                 positions=[[2.0, 2.0, 2.0], [3.5, 2.0, 2.0],
                            [2.0, 8.0, 2.0], [3.5, 8.0, 2.0]],
                 cell=np.eye(3) * (15.0 + seed), pbc=True)


class RecordingBackend:
    """Stands in for an energy backend, and remembers what it was asked."""

    name = "recording"
    thread_safe = True
    timeout = 10

    def __init__(self, energy_of=None, fail_on=(), thread_safe=True):
        self.energy_of = energy_of or (lambda atoms: -100.0)
        self.fail_on = set(fail_on)
        self.thread_safe = thread_safe
        self.seen = []
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()

    def job_files(self, cfg):
        return {}

    def required_commands(self, cfg):
        return []

    def describe(self):
        return self.name

    def evaluate(self, atoms, step_dir, ref_bonds, mol_size, logger=None):
        with self._lock:
            self.seen.append(os.path.basename(step_dir))
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            time.sleep(0.01)
            if os.path.basename(step_dir) in self.fail_on:
                return None, None
            return atoms.copy(), self.energy_of(atoms)
        finally:
            with self._lock:
                self.concurrent -= 1


class TestWorkerSpec(unittest.TestCase):
    """`parallelWorkers` is the whole user interface to this feature."""

    def test_user_host_slots(self):
        w = Worker.parse("raul@192.168.1.11:4")
        self.assertEqual((w.user, w.host, w.slots), ("raul", "192.168.1.11", 4))
        self.assertEqual(w.target, "raul@192.168.1.11")
        self.assertFalse(w.is_local)

    def test_slots_default_to_one(self):
        self.assertEqual(Worker.parse("raul@node1").slots, 1)

    def test_host_without_user(self):
        w = Worker.parse("node1:2")
        self.assertIsNone(w.user)
        self.assertEqual(w.target, "node1")

    def test_interpreter_override(self):
        w = Worker.parse("raul@node1:4:/home/raul/mace-env/bin/python")
        self.assertEqual(w.python, "/home/raul/mace-env/bin/python")
        self.assertEqual(w.slots, 4)

    def test_local_is_recognised(self):
        for spec in ("local", "local:3"):
            self.assertTrue(Worker.parse(spec).is_local, spec)

    def test_bad_slot_count_names_the_entry(self):
        with self.assertRaises(ValueError) as cm:
            Worker.parse("raul@node1:many")
        self.assertIn("raul@node1:many", str(cm.exception))

    def test_zero_slots_rejected(self):
        with self.assertRaises(ValueError):
            Worker.parse("raul@node1:0")

    def test_duplicate_machines_are_merged(self):
        workers = parse_workers(["raul@node1:2", "raul@node1:3", "raul@node2:1"])
        self.assertEqual(len(workers), 2)
        self.assertEqual(workers[0].slots, 5)


class TestPoolScheduling(unittest.TestCase):
    """The properties a distributed cycle must keep."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_par_")
        self.logger = quiet_logger()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _pool(self, backend, slots=4):
        cfg = Config(use_parallel=True, parallel_local_slots=slots)
        pool = WorkerPool(cfg, backend, self.logger).start()
        return pool

    def _tasks(self, n):
        return [(i, os.path.join(self.tmp, f"cycle000_struct{i:05d}"), structure(i))
                for i in range(n)]

    def test_every_candidate_is_evaluated_exactly_once(self):
        # The failure this guards against: sending one structure to every
        # machine and keeping the best answer. That is n times the work, no
        # speedup, and a different experiment.
        backend = RecordingBackend()
        pool = self._pool(backend, slots=4)
        tasks = self._tasks(12)
        pool.evaluate_all(tasks, ref_bonds=[], mol_size=4)
        self.assertEqual(len(backend.seen), 12)
        self.assertEqual(len(set(backend.seen)), 12)

    def test_results_come_back_in_candidate_order(self):
        # Ids and CIF names must not depend on which slot finished first.
        backend = RecordingBackend(energy_of=lambda a: -float(len(a)) * a.cell[0, 0])
        pool = self._pool(backend, slots=4)
        tasks = self._tasks(10)
        results = pool.evaluate_all(tasks, ref_bonds=[], mol_size=4)
        self.assertEqual([r[0] for r in results], [t[0] for t in tasks])

    def test_slots_actually_run_at_the_same_time(self):
        backend = RecordingBackend()
        pool = self._pool(backend, slots=4)
        pool.evaluate_all(self._tasks(16), ref_bonds=[], mol_size=4)
        self.assertGreater(backend.max_concurrent, 1)
        self.assertLessEqual(backend.max_concurrent, 4)

    def test_a_rejected_candidate_is_a_result_not_a_failure(self):
        backend = RecordingBackend(fail_on={"cycle000_struct00003"})
        pool = self._pool(backend, slots=3)
        results = pool.evaluate_all(self._tasks(6), ref_bonds=[], mol_size=4)
        by_index = {i: (a, e) for i, a, e in results}
        self.assertIsNone(by_index[3][0])
        self.assertEqual(sum(1 for a, _ in by_index.values() if a is not None), 5)
        # Evaluated once: a rejection must not be retried somewhere else.
        self.assertEqual(backend.seen.count("cycle000_struct00003"), 1)

    def test_local_slots_clamped_on_an_unsafe_backend(self):
        # The corruption this prevents is silent: one shared ASE calculator
        # handing two threads each other's energies.
        backend = RecordingBackend(thread_safe=False)
        pool = self._pool(backend, slots=8)
        self.assertEqual(pool.n_slots, 1)

    def test_an_exception_in_the_backend_costs_one_candidate(self):
        class Exploding(RecordingBackend):
            def evaluate(self, atoms, step_dir, ref_bonds, mol_size, logger=None):
                if os.path.basename(step_dir).endswith("00002"):
                    raise RuntimeError("engine blew up")
                return super().evaluate(atoms, step_dir, ref_bonds, mol_size, logger)

        pool = self._pool(Exploding(), slots=2)
        results = pool.evaluate_all(self._tasks(5), ref_bonds=[], mol_size=4)
        by_index = {i: a for i, a, _ in results}
        self.assertIsNone(by_index[2])
        self.assertEqual(sum(1 for a in by_index.values() if a is not None), 4)


class TestWorkerFailure(unittest.TestCase):
    """A machine that goes away costs candidates, not the run."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_par_")
        self.logger = quiet_logger()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _remote_pool(self, n_remote_slots, evaluate_remote):
        cfg = Config(use_parallel=True, parallel_local_slots=1,
                     parallel_max_failures=2)
        pool = WorkerPool(cfg, RecordingBackend(), self.logger)
        pool.workers = [Worker(label="local", slots=1),
                        Worker(label="raul@node1", slots=n_remote_slots,
                               host="node1", user="raul",
                               job_dir="/tmp/job", config_dict={})]
        for w in pool.workers:
            for k in range(1, w.slots + 1):
                from cryal.parallel import _Slot
                pool._slots.put(_Slot(worker=w, number=k))
                pool._live_slots += 1
        pool._started = True
        pool._evaluate_remote = evaluate_remote
        return pool

    def test_a_transport_failure_retries_elsewhere(self):
        attempts = []

        def flaky(worker, atoms, step_dir, ref_bonds, mol_size):
            attempts.append(os.path.basename(step_dir))
            raise WorkerError("network went away")

        pool = self._remote_pool(1, flaky)
        tasks = [(0, os.path.join(self.tmp, "cycle000_struct00000"), structure())]
        results = pool.evaluate_all(tasks, ref_bonds=[], mol_size=4)
        # It came back with a verdict for the candidate, and the local slot
        # did the work the remote one could not.
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 0)
        self.assertIsNotNone(results[0][1])

    def test_a_repeatedly_failing_worker_is_retired(self):
        def always_fails(worker, atoms, step_dir, ref_bonds, mol_size):
            raise WorkerError("host is down")

        pool = self._remote_pool(2, always_fails)
        before = pool.n_slots
        tasks = [(i, os.path.join(self.tmp, f"cycle000_struct{i:05d}"), structure(i))
                 for i in range(8)]
        results = pool.evaluate_all(tasks, ref_bonds=[], mol_size=4)
        self.assertEqual(len(results), 8)
        # Every candidate still got an answer, from the local slot.
        self.assertTrue(all(r[1] is not None for r in results))
        # ...and the dead machine is no longer being offered work.
        self.assertLess(pool.n_slots, before)
        self.assertFalse(
            next(w for w in pool.workers if w.label == "raul@node1").enabled)


class TestSerialisation(unittest.TestCase):
    """What crosses the network must come back unchanged."""

    def test_cell_survives_the_round_trip(self):
        # An .xyz would silently drop it, and every quantity the surrogate
        # learns from -- density, beta, volume -- is the cell.
        atoms = Atoms("C2", positions=[[0, 0, 0], [1.4, 0, 0]],
                      cell=[[10.0, 0, 0], [1.5, 9.0, 0], [0.3, 0.7, 11.0]],
                      pbc=True)
        back = atoms_from_dict(json.loads(json.dumps(atoms_to_dict(atoms))))
        np.testing.assert_allclose(back.cell.array, atoms.cell.array)
        np.testing.assert_allclose(back.get_positions(), atoms.get_positions())
        self.assertEqual(back.get_chemical_symbols(), atoms.get_chemical_symbols())
        self.assertTrue(all(back.pbc))

    def test_the_worker_config_is_the_servers(self):
        # A worker that invents its own energy bound or species order returns
        # numbers that are not comparable, and nothing in the database shows it.
        cfg = Config(use_parallel=True, energy_sanity_max=-42.0,
                     spec_order=["N", "C"], check_molecular_integrity=True)
        pool = WorkerPool(cfg, RecordingBackend(), quiet_logger())
        worker = Worker(label="raul@node1", slots=1, host="node1", user="raul",
                        job_dir="/remote/job")
        d = pool._remote_config(worker)
        self.assertEqual(d["energy_sanity_max"], -42.0)
        self.assertEqual(d["spec_order"], ["N", "C"])
        self.assertTrue(d["check_molecular_integrity"])
        json.dumps(d)          # must survive the trip as JSON

    def test_job_files_are_rewritten_to_the_remote_copy(self):
        class NeedsAScript(RecordingBackend):
            def job_files(self, cfg):
                return {"lammps_input": cfg.lammps_input}

        cfg = Config(use_parallel=True, lammps_input="/home/me/in_v3.lammps")
        pool = WorkerPool(cfg, NeedsAScript(), quiet_logger())
        worker = Worker(label="n1", slots=1, host="n1", job_dir="/remote/job")
        d = pool._remote_config(worker)
        self.assertEqual(d["lammps_input"], "/remote/job/files/in_v3.lammps")


class TestRemoteWorker(unittest.TestCase):
    """The far end of the wire, exercised without a wire.

    `python -m cryal._remote_worker <task_dir>` is what runs on every other
    machine. It is tested here in process, on a real backend with a stub
    calculator, because the thing worth proving is not that ssh works: it is
    that the worker rebuilds the server's configuration and puts the candidate
    through the same evaluate() -- gate, energy bound, integrity check -- that
    the serial path uses.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_worker_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _task(self, atoms, **cfg_kwargs):
        cfg = Config(energy_backend="ase",
                     ase_calculator="tests.stub_calculator.StubCalculator",
                     ase_calculator_kwargs={"energy": -1000.0, "force": 0.0},
                     ase_optimizer="BFGS", ase_relax_cell=False,
                     **cfg_kwargs)
        pool = WorkerPool(cfg, RecordingBackend(), quiet_logger())
        worker = Worker(label="n1", slots=1, host="n1", job_dir="/remote/job")
        payload = {
            "cryal_version": "test",
            "task_id": "cycle000_struct00000",
            "config": pool._remote_config(worker),
            "atoms": atoms_to_dict(atoms),
            "ref_bonds": [],
            "mol_size": len(atoms),
        }
        with open(os.path.join(self.tmp, "task.json"), "w") as f:
            json.dump(payload, f)
        return os.path.join(self.tmp, "result.json")

    def _run(self):
        from cryal import _remote_worker
        # The worker prints its traceback to stderr for the ssh transcript;
        # the test reads the verdict from result.json, not from the console.
        with contextlib.redirect_stderr(io.StringIO()):
            return _remote_worker.main([self.tmp])

    def test_a_good_candidate_comes_back_with_an_energy(self):
        result_path = self._task(structure())
        rc = self._run()
        self.assertEqual(rc, 0)
        with open(result_path) as f:
            result = json.load(f)
        self.assertTrue(result["ok"])
        self.assertAlmostEqual(result["energy"], -1000.0)
        back = atoms_from_dict(result["atoms"])
        self.assertEqual(len(back), 4)
        np.testing.assert_allclose(back.cell.array, structure().cell.array)

    def test_the_pre_relaxation_gate_still_applies_remotely(self):
        # Two atoms on top of each other. The gate that saves five minutes of
        # a doomed relaxation must not be lost by moving the work elsewhere.
        overlapping = Atoms("C2", positions=[[2.0, 2.0, 2.0], [2.3, 2.0, 2.0]],
                            cell=np.eye(3) * 15.0, pbc=True)
        result_path = self._task(overlapping)
        rc = self._run()
        # A verdict, not a crash: 3 means "evaluated and rejected".
        self.assertEqual(rc, 3)
        with open(result_path) as f:
            result = json.load(f)
        self.assertFalse(result["ok"])
        self.assertIn("reason", result)

    def test_the_servers_energy_bound_is_the_one_applied(self):
        # -1000 eV is a fine energy; under a bound of -2000 it is not. The
        # worker must use the server's number, not a plausible local default.
        result_path = self._task(structure(), energy_sanity_max=-2000.0)
        self.assertEqual(self._run(), 3)
        with open(result_path) as f:
            self.assertFalse(json.load(f)["ok"])

    def test_a_broken_task_still_writes_a_verdict(self):
        # Silence is what makes the server retry elsewhere, so a worker that
        # failed for its own reasons must say so rather than say nothing.
        with open(os.path.join(self.tmp, "task.json"), "w") as f:
            f.write("{ this is not json")
        rc = self._run()
        self.assertEqual(rc, 1)
        with open(os.path.join(self.tmp, "result.json")) as f:
            result = json.load(f)
        self.assertFalse(result["ok"])
        self.assertIn("error", result)


class TestRemoteJobCheck(unittest.TestCase):
    """A machine that cannot do the work must be found before it is given any.

    The hole this closes: a worker missing the potential returns `ok: false`
    for every candidate, which is indistinguishable from a structure that was
    evaluated and rejected. The server therefore never retires it, and it eats
    its share of every cycle in silence. The check is the backend's own
    validate_config(), run over there.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cryal_check_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, cfg: Config):
        pool = WorkerPool(cfg, RecordingBackend(), quiet_logger())
        worker = Worker(label="n1", slots=1, host="n1", job_dir=self.tmp)
        with open(os.path.join(self.tmp, "config.json"), "w") as f:
            json.dump(pool._remote_config(worker), f)

    def _check(self):
        from cryal import _remote_worker
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = _remote_worker.main(["--check", self.tmp])
        return rc, json.loads(out.getvalue().strip().splitlines()[-1])

    def test_a_workable_configuration_passes(self):
        self._write(Config(energy_backend="ase",
                           ase_calculator="tests.stub_calculator.StubCalculator",
                           ase_optimizer="BFGS"))
        rc, result = self._check()
        self.assertEqual(rc, 0)
        self.assertTrue(result["ok"])
        self.assertEqual(result["backend"], "ase")

    def test_a_missing_calculator_is_caught_here(self):
        # The real case: fairchem or mace-torch is not installed on that box.
        self._write(Config(energy_backend="ase",
                           ase_calculator="not_installed_anywhere.Calculator"))
        rc, result = self._check()
        self.assertEqual(rc, 1)
        self.assertFalse(result["ok"])
        self.assertIn("not_installed_anywhere", result["error"])

    def test_a_missing_lammps_script_is_caught_here(self):
        self._write(Config(energy_backend="lammps_mace",
                           lammps_input="/nowhere/in_v3.lammps"))
        rc, result = self._check()
        self.assertEqual(rc, 1)
        self.assertIn("in_v3.lammps", result["error"])

    def test_an_unknown_backend_is_caught_here(self):
        self._write(Config(energy_backend="does_not_exist"))
        rc, result = self._check()
        self.assertEqual(rc, 1)
        self.assertFalse(result["ok"])

    def test_the_check_uses_the_servers_paths_not_the_workers(self):
        # job_files rewrites lammps_input to the copy shipped to the worker,
        # and the check must validate that path, not the server's.
        class NeedsAScript(RecordingBackend):
            name = "lammps_mace"

            def job_files(self, cfg):
                return {"lammps_input": cfg.lammps_input}

        script = os.path.join(self.tmp, "files", "in_v3.lammps")
        os.makedirs(os.path.dirname(script), exist_ok=True)
        with open(script, "w") as f:
            f.write("# a lammps script\n")

        cfg = Config(energy_backend="lammps_mace",
                     lammps_input="/on/the/server/in_v3.lammps")
        pool = WorkerPool(cfg, NeedsAScript(), quiet_logger())
        worker = Worker(label="n1", slots=1, host="n1", job_dir=self.tmp)
        with open(os.path.join(self.tmp, "config.json"), "w") as f:
            json.dump(pool._remote_config(worker), f)

        rc, result = self._check()
        self.assertEqual(rc, 0, result)
        self.assertTrue(result["ok"])


class TestBackendHooks(unittest.TestCase):
    """What a backend must tell the pool before its work can be sent away."""

    def test_lammps_ships_its_input_script(self):
        from cryal.backends.lammps_mace import LammpsMaceBackend
        cfg = Config(lammps_input="in_v3.lammps")
        self.assertEqual(LammpsMaceBackend.job_files(cfg),
                         {"lammps_input": "in_v3.lammps"})

    def test_lammps_names_the_executable_not_its_flags(self):
        # lammpsCommand carries the Kokkos flags MACE needs; `command -v` must
        # be given the executable alone.
        from cryal.backends.lammps_mace import LammpsMaceBackend
        cfg = Config(lammps_command="lmp -k on g 1 -sf kk -pk kokkos newton on")
        self.assertEqual(LammpsMaceBackend.required_commands(cfg), ["lmp"])

    def test_lammps_can_run_concurrently_and_ase_cannot(self):
        from cryal.backends.ase_calculator import AseCalculatorBackend
        from cryal.backends.lammps_mace import LammpsMaceBackend
        self.assertTrue(LammpsMaceBackend.thread_safe)
        self.assertFalse(AseCalculatorBackend.thread_safe)

    def test_a_backend_ships_nothing_by_default(self):
        from cryal.backends.base import EnergyBackend
        self.assertEqual(EnergyBackend.job_files(Config()), {})
        self.assertEqual(EnergyBackend.required_commands(Config()), [])


class TestConfiguration(unittest.TestCase):

    MINIMAL = """
% GENERAL
moleculeFile = examples/benzene.xyz
Z = 4
"""

    def _load(self, extra=""):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(self.MINIMAL + extra)
            path = f.name
        try:
            return load_config(path)
        finally:
            os.unlink(path)

    def test_off_by_default(self):
        cfg = self._load()
        self.assertFalse(cfg.use_parallel)
        self.assertEqual(cfg.parallel_workers, [])

    def test_workers_are_read_as_a_list(self):
        cfg = self._load("useParallel = true\n"
                         "parallelWorkers = raul@node1:4 raul@node2:2\n")
        self.assertTrue(cfg.use_parallel)
        self.assertEqual(cfg.parallel_workers, ["raul@node1:4", "raul@node2:2"])

    def test_a_malformed_worker_fails_at_load_time(self):
        # Not at candidate 40, three hours in.
        with self.assertRaises(ValueError):
            self._load("useParallel = true\nparallelWorkers = raul@node1:four\n")

    def test_build_pool_returns_none_when_off(self):
        cfg = self._load()
        self.assertIsNone(build_pool(cfg, RecordingBackend(), quiet_logger()))

    def test_build_pool_declines_a_single_slot(self):
        # One slot through the pool is the serial path with machinery in the way.
        cfg = self._load("useParallel = true\nparallelLocalSlots = 1\n")
        self.assertIsNone(build_pool(cfg, RecordingBackend(), quiet_logger()))


class TestInputKeysAreRead(unittest.TestCase):
    """Every key in INPUT.txt must reach the Config.

    A key the parser looks up under a slightly different name is read as its
    default, silently, for the life of the project: the value in the file is
    simply ignored. That failure has no symptom, so it gets a test.
    """

    def test_no_key_in_the_shipped_input_is_ignored(self):
        import cryal.config as config_module

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        input_path = os.path.join(here, "INPUT.txt")
        if not os.path.exists(input_path):
            self.skipTest("INPUT.txt is not part of this checkout")

        declared = set(parse_input(input_path))

        looked_up = set()
        original = config_module._get

        def spy(raw, key, default=None):
            looked_up.add(key.lower())
            return original(raw, key, default)

        config_module._get = spy
        try:
            load_config(input_path)
        finally:
            config_module._get = original

        ignored = sorted(declared - looked_up)
        self.assertEqual(ignored, [],
                         f"INPUT.txt sets keys that load_config never reads: {ignored}")


if __name__ == "__main__":
    unittest.main()
