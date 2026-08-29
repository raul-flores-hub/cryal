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
parallel.py — evaluating a cycle's candidates on more than one machine.

What is parallel here
=====================
A cycle proposes tens to hundreds of candidate structures, and relaxing one
says nothing about relaxing another: they are independent, and they are the
whole cost of a run. That is the only thing this module distributes. It does
not split a single relaxation across machines, and it is not a backend — a
backend turns *one* structure into an energy, and inside one structure there
is nothing here to parallelise.

The unit of distribution is therefore one candidate, and the unit of capacity
is a *slot*: one concurrent evaluation. A worker declares how many slots it
offers, which is how the user says "this PC runs four at a time and that
laptop runs one".

Scheduling
==========
Slots pull from a shared queue of candidates rather than each being handed a
fixed share up front. A static split finishes at the speed of the slowest
machine — one cycle's worth of candidates on a laptop while a workstation
sits idle. Pulling keeps every machine busy until the cycle is done, and the
number of slots still sets each machine's share of the work.

What must not change
====================
Distributing evaluation must be invisible in the results. Three things
guarantee that:

  * Every candidate is evaluated exactly once. Slots take candidates off a
    queue; nothing is broadcast, and no result is chosen from among several.
  * Every candidate goes through the same `EnergyBackend.evaluate()` — the
    pre-relaxation gate, the energy bound, the integrity check — with the same
    configuration. The remote worker receives a copy of the live Config and
    the reference bonds computed here, so it cannot drift from the server.
  * Records enter the database in candidate order, not completion order, so a
    parallel run and a serial one produce the same database for the same
    structures.

Failure model
=============
A machine that goes away must cost one candidate, not the run. A transport
failure retries the candidate on a different slot; a worker that fails
repeatedly is retired and the run continues with fewer slots. Only the
structure-level verdict — relaxation failed, integrity rejected — comes back
as a normal `(None, None)`, exactly as in a serial run.

Configuration (INPUT.txt)
=========================
    % PARALLEL
    useParallel        = true
    parallelWorkers    = raul@192.168.1.11:4  raul@192.168.1.12:2
    parallelLocalSlots = 2
    parallelRemoteDir  = ~/.cryal_work
"""

import os
import json
import queue
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple

from ase import Atoms

from . import __version__


class WorkerError(RuntimeError):
    """The worker could not be reached or could not report a verdict.

    Distinct from a structure that was evaluated and rejected: that is a
    result, and comes back as (None, None). This means we learned nothing
    about the candidate, so it is worth trying somewhere else.
    """


# ---------------------------------------------------------------------------
# Structure serialisation
# ---------------------------------------------------------------------------

def atoms_to_dict(atoms: Atoms) -> dict:
    """A JSON-safe description of a periodic structure.

    Explicit rather than a file format: the cell must survive the round trip
    intact, and an .xyz silently drops it.
    """
    return {
        "symbols":   list(atoms.get_chemical_symbols()),
        "positions": atoms.get_positions().tolist(),
        "cell":      atoms.cell.array.tolist(),
        "pbc":       [bool(x) for x in atoms.pbc],
    }


def atoms_from_dict(d: dict) -> Atoms:
    """Rebuild what atoms_to_dict() wrote."""
    return Atoms(symbols=d["symbols"], positions=d["positions"],
                 cell=d["cell"], pbc=d.get("pbc", True))


# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------

@dataclass
class Worker:
    """One machine offering `slots` concurrent evaluations."""

    label:  str
    slots:  int
    host:   Optional[str] = None     # None means the server itself
    user:   Optional[str] = None
    python: str = "python3"

    # Filled in by the pool's preflight, for remote workers only.
    home:        Optional[str] = None
    job_dir:     Optional[str] = None
    config_dict: Optional[dict] = None

    enabled: bool = True
    consecutive_failures: int = 0

    @property
    def is_local(self) -> bool:
        return self.host is None

    @property
    def target(self) -> str:
        """What ssh and scp want on the left of the colon."""
        return f"{self.user}@{self.host}" if self.user else str(self.host)

    @classmethod
    def parse(cls, spec: str, default_python: str = "python3") -> "Worker":
        """
        Read one entry of `parallelWorkers`.

            user@host              one slot on that machine
            user@host:4            four slots
            user@host:4:/opt/venv/bin/python   ...and which interpreter to use
            host:4                 same, as the current user
            local:2                two slots on the server itself

        The interpreter field exists because the machine that has LAMMPS and
        MACE installed usually has them in a virtual environment, and the
        `python3` on its PATH is not the one that can import cryal.
        """
        raw = spec.strip()
        if not raw:
            raise ValueError("empty worker specification")

        user = None
        rest = raw
        if "@" in rest:
            user, rest = rest.split("@", 1)
            if not user or not rest:
                raise ValueError(f"malformed worker specification: '{spec}'")

        parts = rest.split(":")
        host = parts[0].strip()
        if not host:
            raise ValueError(f"malformed worker specification: '{spec}'")

        slots = 1
        if len(parts) > 1 and parts[1].strip():
            try:
                slots = int(parts[1])
            except ValueError:
                raise ValueError(
                    f"worker '{spec}': '{parts[1]}' is not a number of slots — "
                    "the format is user@host:slots") from None
        if slots < 1:
            raise ValueError(f"worker '{spec}': slots must be at least 1")

        python = default_python
        if len(parts) > 2 and parts[2].strip():
            python = ":".join(parts[2:]).strip()

        if host.lower() == "local" or (user is None and host.lower() == "localhost"):
            return cls(label="local", slots=slots)

        label = f"{user}@{host}" if user else host
        return cls(label=label, slots=slots, host=host, user=user, python=python)


def parse_workers(specs, default_python: str = "python3") -> List[Worker]:
    """Parse `parallelWorkers`, merging any duplicate entry for one machine."""
    workers: List[Worker] = []
    for spec in specs:
        w = Worker.parse(spec, default_python)
        for existing in workers:
            if existing.label == w.label:
                existing.slots += w.slots
                break
        else:
            workers.append(w)
    return workers



# ---------------------------------------------------------------------------
# SSH transport
# ---------------------------------------------------------------------------

class _Ssh:
    """ssh/scp invocations, with the options a batch run needs.

    BatchMode makes a missing key fail immediately instead of hanging on a
    password prompt no one is there to answer — the failure mode that turns
    an overnight run into an overnight nothing.
    """

    def __init__(self, key_file: str = "", connect_timeout: int = 30):
        self.key_file = os.path.expanduser(key_file) if key_file else ""
        self.connect_timeout = int(connect_timeout)

    def _opts(self) -> List[str]:
        opts = ["-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"ConnectTimeout={self.connect_timeout}"]
        if self.key_file:
            opts += ["-i", self.key_file]
        return opts

    def run(self, worker: Worker, command: str, timeout: Optional[int] = None):
        """Run a shell command on the worker. Returns (rc, stdout, stderr)."""
        cmd = ["ssh"] + self._opts() + [worker.target, command]
        return _capture(cmd, timeout)

    def push(self, worker: Worker, local_path: str, remote_path: str,
             timeout: Optional[int] = None):
        """Copy a local file to the worker."""
        cmd = (["scp"] + self._opts()
               + [local_path, f"{worker.target}:{remote_path}"])
        return _capture(cmd, timeout)

    def pull(self, worker: Worker, remote_path: str, local_path: str,
             recursive: bool = False, timeout: Optional[int] = None):
        """Copy a file from the worker to this machine."""
        cmd = ["scp"] + self._opts()
        if recursive:
            cmd.append("-r")
        cmd += [f"{worker.target}:{remote_path}", local_path]
        return _capture(cmd, timeout)


def _last_line(text: str, default: str) -> str:
    """The part of an ssh failure worth putting in a one-line log message."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else default


def _capture(cmd: List[str], timeout: Optional[int]) -> Tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s: {' '.join(cmd[:2])}"
    except Exception as e:                       # ssh/scp missing, bad path...
        return 127, "", f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Slots
# ---------------------------------------------------------------------------

@dataclass
class _Slot:
    """One unit of concurrency, bound to the machine that provides it."""

    worker: Worker
    number: int
    retired: bool = False

    @property
    def label(self) -> str:
        return f"{self.worker.label}#{self.number}"


# ---------------------------------------------------------------------------
# The pool
# ---------------------------------------------------------------------------

class WorkerPool:
    """
    The slots a cycle's candidates are evaluated on.

    Built once per run. `start()` parses the worker list and checks that every
    remote machine can actually do the work, so an unreachable host is
    reported in the first seconds rather than discovered at candidate 40.
    """

    def __init__(self, cfg, backend, logger):
        self.cfg = cfg
        self.backend = backend
        self.logger = logger
        self.workers: List[Worker] = []
        self.ssh = _Ssh(getattr(cfg, "parallel_ssh_key", ""),
                        getattr(cfg, "parallel_ssh_timeout", 30))
        self.job_id = f"cryal-{int(time.time())}-{os.getpid()}"

        self._slots: "queue.Queue[_Slot]" = queue.Queue()
        self._lock = threading.Lock()
        self._live_slots = 0
        self._started = False

    # -- capacity ----------------------------------------------------------

    @property
    def n_slots(self) -> int:
        with self._lock:
            return self._live_slots

    def describe(self) -> str:
        live = [w for w in self.workers if w.enabled]
        if not live:
            return "no workers"
        return ", ".join(f"{w.label}×{w.slots}" for w in live)

    # -- setup -------------------------------------------------------------

    def start(self) -> "WorkerPool":
        """Parse the worker list, check the remote machines, open the slots."""
        specs = list(getattr(self.cfg, "parallel_workers", []) or [])
        workers = parse_workers(specs, getattr(self.cfg, "parallel_python", "python3"))

        # The server is a worker too. It can be named in the
        # list as `local:N`, which is the more specific statement and wins;
        # parallelLocalSlots is the shorthand for when it is not named.
        local_slots = int(getattr(self.cfg, "parallel_local_slots", 1) or 0)
        if not any(w.is_local for w in workers) and local_slots > 0:
            workers.insert(0, Worker(label="local", slots=local_slots))

        # Refuse to run two in-process relaxations at once on a backend that
        # says it cannot. Silently corrupted energies are worse than a slow run.
        for w in workers:
            if w.is_local and w.slots > 1 and not getattr(self.backend, "thread_safe", False):
                self.logger.warning(
                    f"parallelLocalSlots={w.slots} but the '{self.backend.name}' "
                    "backend is not safe to run concurrently in one process "
                    "(it keeps state between relaxations) — using 1 local slot. "
                    "Remote workers are unaffected: those are separate processes.")
                w.slots = 1

        self.workers = workers
        for w in list(self.workers):
            if not w.is_local:
                self._preflight(w)

        for w in self.workers:
            if not w.enabled:
                continue
            for k in range(1, w.slots + 1):
                self._slots.put(_Slot(worker=w, number=k))
                self._live_slots += 1

        self._started = True
        return self

    def _preflight(self, worker: Worker):
        """Prove a remote machine can run a candidate, before the run needs it."""
        def fail(reason: str):
            worker.enabled = False
            self.logger.warning(f"  worker {worker.label}: DISABLED — {reason}")

        rc, out, err = self.ssh.run(worker, "echo $HOME",
                                    timeout=self.ssh.connect_timeout + 10)
        if rc != 0:
            return fail(f"cannot ssh in — {_last_line(err or out, f'exit {rc}')}")
        worker.home = out.strip().splitlines()[-1] if out.strip() else ""
        if not worker.home:
            return fail("could not determine the remote home directory")

        remote_dir = getattr(self.cfg, "parallel_remote_dir", "~/.cryal_work")
        base = remote_dir.replace("~", worker.home, 1) if remote_dir.startswith("~") else remote_dir
        worker.job_dir = f"{base.rstrip('/')}/{self.job_id}"

        rc, out, err = self.ssh.run(
            worker, f"mkdir -p {shlex.quote(worker.job_dir + '/files')}", timeout=60)
        if rc != 0:
            return fail(f"cannot create {worker.job_dir} ({err.strip() or rc})")

        # The interpreter must be the one that can import cryal — on a machine
        # set up for MACE that is almost never the `python3` on PATH.
        probe = ("import cryal, ase, json; "
                 "print(json.dumps({'cryal': cryal.__version__, 'ase': ase.__version__}))")
        rc, out, err = self.ssh.run(
            worker, f"{shlex.quote(worker.python)} -c {shlex.quote(probe)}", timeout=120)
        if rc != 0:
            return fail(
                f"'{worker.python}' cannot import cryal — "
                f"{_last_line(err, 'exit ' + str(rc))}. "
                "Install CrYAL there, or name the right interpreter as the "
                "third field of the worker (user@host:slots:/path/to/python).")
        try:
            versions = json.loads(out.strip().splitlines()[-1])
        except Exception:
            versions = {}
        if versions.get("cryal") and versions["cryal"] != __version__:
            self.logger.warning(
                f"  worker {worker.label}: CrYAL {versions['cryal']} there vs "
                f"{__version__} here — energies from the two are not "
                "guaranteed comparable")

        # Ship whatever the backend cannot work without.
        for attr, local_path in (self.backend.job_files(self.cfg) or {}).items():
            if not local_path or not os.path.exists(local_path):
                return fail(f"{attr}: local file not found ({local_path})")
            remote_path = f"{worker.job_dir}/files/{os.path.basename(local_path)}"
            rc, out, err = self.ssh.push(worker, local_path, remote_path, timeout=300)
            if rc != 0:
                return fail(f"could not copy {local_path} ({err.strip() or rc})")

        worker.config_dict = self._remote_config(worker)

        # Ask the backend itself whether this machine can run the job. Without
        # this, a worker missing the potential or the calculator stays in the
        # pool and answers every candidate with a rejection — which is a
        # legitimate verdict as far as the server is concerned, so it never
        # gets retired and quietly eats its share of every cycle.
        if not self._validate_remote(worker):
            return

        # A non-interactive ssh session does not read ~/.bashrc, so an engine
        # that works when you log in by hand can still be missing here. Worth
        # a warning now rather than a hundred identical failures later.
        for command in (self.backend.required_commands(self.cfg) or []):
            rc, _, _ = self.ssh.run(worker, f"command -v {shlex.quote(command)}",
                                    timeout=60)
            if rc != 0:
                self.logger.warning(
                    f"  worker {worker.label}: '{command}' is not on the PATH of "
                    "a non-interactive ssh session — every candidate sent there "
                    "will fail. Give it an absolute path in the configuration.")

        self.logger.info(f"  worker {worker.label}: ready — {worker.slots} slot(s), "
                         f"{worker.python}, {worker.job_dir}")

    def _remote_config(self, worker: Worker) -> dict:
        """The live configuration as the worker will see it.

        Sending the real Config rather than rebuilding a plausible one on the
        far side is the point: a worker that invents its own energy bound or
        its own species order returns numbers that are not comparable with
        the server's, and nothing in the results would show it.
        """
        d = {}
        for f in fields(self.cfg):
            value = getattr(self.cfg, f.name)
            try:
                json.dumps(value)
            except TypeError:
                # Dropping it would leave the worker running on this field's
                # default while the server runs on the real value — a
                # divergence nothing downstream could detect. Say so.
                self.logger.warning(
                    f"  {f.name} cannot be sent to a worker "
                    f"({type(value).__name__}); {worker.label} will use its "
                    "default for it")
                continue
            d[f.name] = value
        for attr, local_path in (self.backend.job_files(self.cfg) or {}).items():
            d[attr] = f"{worker.job_dir}/files/{os.path.basename(local_path)}"
        return d

    def _validate_remote(self, worker: Worker) -> bool:
        """Run the backend's own validate_config() on the worker."""
        config_path = os.path.join(tempfile.mkdtemp(prefix="cryal_cfg_"), "config.json")
        try:
            with open(config_path, "w") as f:
                json.dump(worker.config_dict, f)
            rc, out, err = self.ssh.push(
                worker, config_path, f"{worker.job_dir}/config.json", timeout=120)
        finally:
            shutil.rmtree(os.path.dirname(config_path), ignore_errors=True)
        if rc != 0:
            worker.enabled = False
            self.logger.warning(f"  worker {worker.label}: DISABLED — could not send "
                                f"the configuration ({_last_line(err, str(rc))})")
            return False

        command = (f"{shlex.quote(worker.python)} -m cryal._remote_worker "
                   f"--check {shlex.quote(worker.job_dir)}")
        rc, out, err = self.ssh.run(worker, command, timeout=300)
        reason = ""
        try:
            reason = json.loads(_last_line(out, "{}")).get("error", "")
        except Exception:
            pass
        if rc != 0:
            worker.enabled = False
            self.logger.warning(
                f"  worker {worker.label}: DISABLED — cannot run the "
                f"'{self.backend.name}' backend: "
                f"{reason or _last_line(err, 'exit ' + str(rc))}")
            return False
        return True

    # -- slot bookkeeping --------------------------------------------------

    def _acquire(self) -> Optional[_Slot]:
        """Wait for a free slot. None means there are no working slots left."""
        while True:
            try:
                slot = self._slots.get(timeout=0.5)
            except queue.Empty:
                with self._lock:
                    if self._live_slots <= 0:
                        return None
                continue
            if not slot.worker.enabled:
                # The machine was retired while this slot sat in the queue.
                # Drop it here rather than spend a candidate proving it again.
                self._retire(slot)
                continue
            return slot

    def _release(self, slot: _Slot):
        if slot.retired or not slot.worker.enabled:
            self._retire(slot)
            return
        self._slots.put(slot)

    def _retire(self, slot: _Slot):
        with self._lock:
            if not slot.retired:
                slot.retired = True
                self._live_slots -= 1

    def _note_success(self, slot: _Slot):
        with self._lock:
            slot.worker.consecutive_failures = 0

    def _note_failure(self, slot: _Slot, error: Exception):
        w = slot.worker
        limit = int(getattr(self.cfg, "parallel_max_failures", 3))
        # Slots of one machine run on different threads and fail together,
        # so the count that decides its fate is taken under the lock.
        with self._lock:
            w.consecutive_failures += 1
            failures = w.consecutive_failures
            retiring = failures >= limit and w.enabled
            if retiring:
                w.enabled = False
        self.logger.warning(f"  worker {slot.label}: {error}")
        if retiring:
            self.logger.warning(
                f"  worker {w.label}: DISABLED after {failures} consecutive "
                "failures — the run continues on the remaining slots")

    # -- running a cycle ---------------------------------------------------

    def evaluate_all(self, tasks, ref_bonds, mol_size):
        """
        Evaluate every task, returning results in the order they were given.

        `tasks` is a sequence of (index, step_dir, atoms). The return value is
        a list of (index, relaxed_or_None, energy_or_None) in task order, so
        the caller writes the database exactly as a serial run would.
        """
        tasks = list(tasks)
        if not tasks:
            return []

        results: Dict[int, Tuple[Optional[Atoms], Optional[float]]] = {}
        total = len(tasks)
        counter = {"n": 0}

        def work(task):
            index, step_dir, atoms = task
            outcome = self._run_one(index, step_dir, atoms, ref_bonds, mol_size)
            with self._lock:
                counter["n"] += 1
                position = counter["n"]
            self.logger.info(f"  [{position}/{total}] {os.path.basename(step_dir)} "
                             f"→ {outcome[2]}")
            return index, outcome[0], outcome[1]

        with ThreadPoolExecutor(max_workers=max(1, self.n_slots),
                                thread_name_prefix="cryal-eval") as pool:
            futures = [pool.submit(work, t) for t in tasks]
            for fut in as_completed(futures):
                index, relaxed, energy = fut.result()
                results[index] = (relaxed, energy)

        return [(index, *results[index]) for index, _, _ in tasks]

    def _max_attempts(self) -> int:
        """How many times one candidate may be re-offered to the pool.

        A candidate must not be thrown away because a machine died while it
        held it — not while a healthy machine is still standing. So the bound
        is not "try twice": it is large enough that every worker can fail its
        way out of the run first, which is what retires them and what makes
        this terminate. Once the last one is retired, _acquire returns None
        and the candidate is finally recorded as failed.
        """
        limit = max(1, int(getattr(self.cfg, "parallel_max_failures", 3)))
        return 1 + max(1, len(self.workers)) * limit

    def _run_one(self, index, step_dir, atoms, ref_bonds, mol_size):
        """One candidate, re-offered to the pool if the machine holding it dies."""
        attempts = 0
        max_attempts = self._max_attempts()
        while True:
            slot = self._acquire()
            if slot is None:
                return None, None, "FAILED (no workers left)"
            try:
                relaxed, energy = self._evaluate_on(slot, atoms, step_dir,
                                                    ref_bonds, mol_size)
                self._note_success(slot)
                if relaxed is None:
                    return None, None, f"rejected on {slot.label}"
                return relaxed, energy, f"E={energy:.4f} eV on {slot.label}"
            except WorkerError as e:
                self._note_failure(slot, e)
                attempts += 1
                if attempts >= max_attempts:
                    return None, None, f"FAILED ({e})"
                self.logger.info(f"  re-queueing {os.path.basename(step_dir)} "
                                 f"(attempt {attempts + 1}/{max_attempts})")
            finally:
                self._release(slot)

    def _evaluate_on(self, slot: _Slot, atoms, step_dir, ref_bonds, mol_size):
        if slot.worker.is_local:
            return self._evaluate_local(atoms, step_dir, ref_bonds, mol_size)
        return self._evaluate_remote(slot.worker, atoms, step_dir,
                                     ref_bonds, mol_size)

    # -- local -------------------------------------------------------------

    def _evaluate_local(self, atoms, step_dir, ref_bonds, mol_size):
        try:
            return self.backend.evaluate(atoms, step_dir, ref_bonds, mol_size,
                                         logger=self.logger)
        except Exception as e:
            # A single bad candidate must not end the search; the backends
            # already work this way and the pool must not change that.
            self.logger.debug(f"  local evaluation raised {type(e).__name__}: {e}")
            return None, None

    # -- remote ------------------------------------------------------------

    def _task_timeout(self) -> int:
        configured = int(getattr(self.cfg, "parallel_task_timeout", 0) or 0)
        if configured > 0:
            return configured
        # The worker enforces the backend's own per-structure timeout; this is
        # the outer bound that catches a machine that stopped answering.
        return int(getattr(self.backend, "timeout", 1800)) + 600

    def _evaluate_remote(self, worker: Worker, atoms, step_dir, ref_bonds, mol_size):
        task_id = os.path.basename(step_dir.rstrip("/")) or f"task{int(time.time()*1000)}"
        remote_task = f"{worker.job_dir}/{task_id}"
        os.makedirs(step_dir, exist_ok=True)

        payload = {
            "cryal_version": __version__,
            "task_id":  task_id,
            "config":   worker.config_dict,
            "atoms":    atoms_to_dict(atoms),
            "ref_bonds": [[int(i), int(j), float(d)] for i, j, d in (ref_bonds or [])],
            "mol_size": int(mol_size) if mol_size else None,
        }
        local_task = os.path.join(step_dir, "remote_task.json")
        with open(local_task, "w") as f:
            json.dump(payload, f)

        rc, out, err = self.ssh.run(worker, f"mkdir -p {shlex.quote(remote_task)}",
                                    timeout=120)
        if rc != 0:
            raise WorkerError(f"mkdir failed on {worker.label}: {err.strip() or rc}")

        rc, out, err = self.ssh.push(worker, local_task,
                                     f"{remote_task}/task.json", timeout=300)
        if rc != 0:
            raise WorkerError(f"could not send the candidate to {worker.label}: "
                              f"{err.strip() or rc}")

        command = (f"{shlex.quote(worker.python)} -m cryal._remote_worker "
                   f"{shlex.quote(remote_task)}")
        rc, out, err = self.ssh.run(worker, command, timeout=self._task_timeout())
        if out or err:
            with open(os.path.join(step_dir, "remote_worker.log"), "w") as f:
                f.write(out)
                if err:
                    f.write("\n--- stderr ---\n" + err)

        local_result = os.path.join(step_dir, "remote_result.json")
        prc, _, perr = self.ssh.pull(worker, f"{remote_task}/result.json",
                                     local_result, timeout=300)
        if prc != 0:
            # No verdict came back. Whether ssh died, the interpreter died or
            # the machine did, we know nothing about this candidate — so it
            # gets another slot rather than being recorded as a failure.
            self._fetch_debris(worker, remote_task, step_dir)
            raise WorkerError(
                f"no result from {worker.label} (exit {rc}): "
                f"{_last_line(err or perr, 'no output')}")

        try:
            with open(local_result) as f:
                result = json.load(f)
        except Exception as e:
            raise WorkerError(f"unreadable result from {worker.label}: {e}") from None

        if getattr(self.cfg, "parallel_fetch_step", False):
            self.ssh.pull(worker, f"{remote_task}/.", step_dir,
                          recursive=True, timeout=600)
        self._cleanup_task(worker, remote_task)

        if not result.get("ok"):
            # A verdict, not a breakdown: the machine ran the candidate and
            # the candidate lost. Same outcome as the serial path, and the
            # worker's own log says which gate rejected it.
            reason = result.get("error") or result.get("reason") or "rejected"
            self.logger.debug(f"  {worker.label}/{task_id}: {reason}")
            tail = result.get("log") or result.get("traceback")
            if tail:
                self.logger.debug(f"  {worker.label}/{task_id} log:\n{tail}")
            return None, None

        relaxed = atoms_from_dict(result["atoms"])
        return relaxed, float(result["energy"])

    def _fetch_debris(self, worker: Worker, remote_task: str, step_dir: str):
        """Bring back a failed task's directory — that is when it is needed."""
        if not getattr(self.cfg, "parallel_fetch_on_failure", True):
            return
        dest = os.path.join(step_dir, "remote_failed")
        os.makedirs(dest, exist_ok=True)
        self.ssh.pull(worker, f"{remote_task}/.", dest, recursive=True, timeout=300)
        self._cleanup_task(worker, remote_task)

    def _cleanup_task(self, worker: Worker, remote_task: str):
        if getattr(self.cfg, "parallel_keep_remote", False):
            return
        self.ssh.run(worker, f"rm -rf {shlex.quote(remote_task)}", timeout=120)

    # -- teardown ----------------------------------------------------------

    def close(self):
        """Remove this run's working directory from every remote machine."""
        if getattr(self.cfg, "parallel_keep_remote", False):
            for w in self.workers:
                if not w.is_local and w.job_dir:
                    self.logger.info(f"  worker {w.label}: kept {w.job_dir}")
            return
        for w in self.workers:
            if w.is_local or not w.job_dir:
                continue
            # job_dir always ends in this run's unique job id, so this cannot
            # reach anything the run did not create.
            self.ssh.run(w, f"rm -rf {shlex.quote(w.job_dir)}", timeout=120)


# ---------------------------------------------------------------------------
# Entry point used by the active-learning loop
# ---------------------------------------------------------------------------

def build_pool(cfg, backend, logger) -> Optional[WorkerPool]:
    """
    The pool for this run, or None when evaluation stays serial.

    Returns None when `useParallel` is off, and also when the configuration
    resolves to a single slot: one slot through the pool is the serial path
    with extra machinery in the way, and the caller's loop is clearer.
    """
    if not getattr(cfg, "use_parallel", False):
        return None

    logger.info("=" * 60)
    logger.info("PARALLEL EVALUATION")
    logger.info("=" * 60)

    pool = WorkerPool(cfg, backend, logger).start()

    if pool.n_slots == 0:
        logger.error("No usable workers — every machine failed its check. "
                     "Falling back to serial evaluation on this machine.")
        pool.close()
        return None
    if pool.n_slots == 1:
        logger.info(f"One slot ({pool.describe()}) — evaluating serially.")
        pool.close()
        return None

    logger.info(f"{pool.n_slots} slots across {len([w for w in pool.workers if w.enabled])} "
                f"machine(s): {pool.describe()}")
    return pool
