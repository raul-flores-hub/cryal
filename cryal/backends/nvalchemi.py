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
nvalchemi.py — batched relaxation with NVIDIA ALCHEMI Toolkit (nvalchemi).

nvalchemi relaxes many structures together on one GPU: every step evaluates
the whole batch in one model call. That is where its speed comes from, so this
backend is `batched`: the evaluation pipeline hands it a cycle's candidates in
groups of backend.nvalchemi.batch_size, through evaluate_batch().

nvalchemi brings its own torch, CUDA libraries and mace-torch, which must not
meet the ones CrYAL and LAMMPS run with. It therefore lives in an environment
of its own, and this backend runs it the way the LAMMPS backend runs `lmp`: as
a separate process, one per batch,

    <backend.nvalchemi.python> -I _nvalchemi_worker.py task.json result.json

in a directory of its own under steps/. The worker imports nothing from
CrYAL. The protocol — NPT stages, then L-BFGS with the cell — and what it does
about float32 energies, exploding structures and tilted cells are described in
_nvalchemi_worker.py.

It is a different engine, not a faster LAMMPS: on the same 30 candidates,
LAMMPS run twice ended in the same structure 10 times out of 10, LAMMPS and
nvalchemi once in 27. Keep one engine for a whole search.

Configuration (cryal.toml):
    [backend]
    engine = "nvalchemi"

    [backend.nvalchemi]
    python = "/home/me/nvalchemi/bin/python"
    model  = "/home/me/.cache/mace/MACE-OFF23_small.model"
"""

import json
import os
import subprocess

from . import register
from .base import EnergyBackend

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_nvalchemi_worker.py")
PROTOCOL_VERSION = 1


def _atoms_to_dict(a):
    return {"symbols": list(a.get_chemical_symbols()),
            "positions": a.get_positions().tolist(),
            "cell": a.cell.array.tolist(),
            "pbc": [bool(x) for x in a.pbc]}


def _atoms_from_dict(d):
    from ase import Atoms
    return Atoms(symbols=d["symbols"], positions=d["positions"], cell=d["cell"],
                 pbc=d.get("pbc", True))


def worker_env() -> dict:
    """
    The worker's environment: this one, with PyTorch's expandable segments.

    nvalchemi's kernels (Warp) allocate GPU memory beside PyTorch. With the
    default caching allocator, PyTorch's reserved memory fragments over an NPT
    stage — 11.3 GB reserved for 3.9 GB in use, on a 12 GB card, after 3000
    steps of 32 TMPA-An cells — until Warp cannot get 1152 bytes and the whole
    batch fails. Expandable segments kept it at 4.1 GB through the stage. A
    PYTORCH_CUDA_ALLOC_CONF set by the user is left as it is.
    """
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


@register
class NvalchemiBackend(EnergyBackend):
    """A batch of structures relaxed by an nvalchemi worker process."""

    name = "nvalchemi"
    aliases = ("alchemi",)
    batched = True

    @classmethod
    def validate_config(cls, cfg):
        if not cfg.nva_python:
            raise ValueError("backend.nvalchemi.python: required — the interpreter of "
                             "the environment where nvalchemi-toolkit is installed")
        if not (os.path.isfile(cfg.nva_python) and os.access(cfg.nva_python, os.X_OK)):
            raise FileNotFoundError(f"backend.nvalchemi.python is not an executable "
                                    f"file: {cfg.nva_python}")
        if not cfg.nva_model:
            raise ValueError("backend.nvalchemi.model: required — a MACE .model file")
        if not os.path.isfile(cfg.nva_model):
            raise FileNotFoundError(f"backend.nvalchemi.model not found: {cfg.nva_model}")

    def describe(self) -> str:
        c = self.cfg
        stages = ", ".join(f"{s['temperature']:g} K/{s['pressure']:g} bar×{s['steps']}"
                           for s in c.nva_stages if s["steps"] > 0) or "none"
        return (f"{self.name} — {os.path.basename(c.nva_model)}, {c.nva_dtype}, "
                f"batches of {c.nva_batch_size}; NPT {c.nva_pressure_coupling} [{stages}], "
                f"then L-BFGS with cell to fmax {c.nva_fmax:g}")

    def batch_size(self) -> int:
        return self.cfg.nva_batch_size

    def relax(self, atoms, step_dir: str, logger=None):
        """One structure: a batch of one."""
        relaxed, energy, reason = self.relax_batch([atoms], [step_dir], logger=logger)[0]
        if relaxed is None:
            return self._reject(logger or self.logger, reason)
        return relaxed, energy

    def task(self, atoms_list, names) -> dict:
        """What the worker is asked to do (task.json)."""
        c = self.cfg
        return {"version": PROTOCOL_VERSION, "model": os.path.abspath(c.nva_model),
                "device": c.nva_device, "dtype": c.nva_dtype, "cueq": c.nva_cueq,
                "timestep": c.nva_timestep, "thermostat_time": c.nva_thermostat_time,
                "barostat_time": c.nva_barostat_time,
                "pressure_coupling": c.nva_pressure_coupling,
                "stages": c.nva_stages, "fmax": c.nva_fmax, "max_steps": c.nva_max_steps,
                "structures": [{"name": n, "atoms": _atoms_to_dict(a)}
                               for n, a in zip(names, atoms_list)]}

    def relax_batch(self, atoms_list, step_dirs, logger=None):
        logger = logger or self.logger
        names = [os.path.basename(os.path.normpath(d)) for d in step_dirs]
        # The batch's own directory, beside its candidates' step directories.
        # Absolute: the worker runs inside it, and output_dir may be relative.
        batch_dir = os.path.abspath(os.path.join(
            os.path.dirname(os.path.normpath(step_dirs[0])), f"batch_{names[0]}"))
        os.makedirs(batch_dir, exist_ok=True)
        task_path = os.path.join(batch_dir, "task.json")
        result_path = os.path.join(batch_dir, "result.json")
        log_path = os.path.join(batch_dir, "worker.log")
        if os.path.exists(result_path):
            os.remove(result_path)
        with open(task_path, "w") as f:
            json.dump(self.task(atoms_list, names), f)
        for d in step_dirs:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "batch.txt"), "w") as f:
                f.write(f"relaxed together in {batch_dir}\n")

        fail = lambda why: [(None, None, why)] * len(atoms_list)
        cmd = [self.cfg.nva_python, "-I", WORKER, task_path, result_path]
        try:
            with open(log_path, "w") as out:
                proc = subprocess.run(cmd, cwd=batch_dir, stdout=out,
                                      stderr=subprocess.STDOUT, env=worker_env(),
                                      timeout=self.cfg.nva_timeout or None)
        except subprocess.TimeoutExpired:
            return fail(f"the nvalchemi batch timed out after {self.cfg.nva_timeout} s "
                        f"(see {log_path})")
        except Exception as e:
            return fail(f"could not run the nvalchemi worker — {e}")
        if proc.returncode != 0 or not os.path.exists(result_path):
            return fail(f"the nvalchemi worker exited {proc.returncode} (see {log_path})")
        try:
            with open(result_path) as f:
                payload = json.load(f)
            results = payload["results"]
            if len(results) != len(atoms_list):
                raise ValueError(f"{len(results)} results for {len(atoms_list)} structures")
        except Exception as e:
            return fail(f"could not read the nvalchemi result — {e}")

        out, unconverged = [], 0
        for r in results:
            if r["status"] != "ok":
                out.append((None, None, r.get("reason") or "the relaxation failed"))
                continue
            out.append((_atoms_from_dict(r["atoms"]), float(r["energy"]), None))
            unconverged += not r.get("converged", True)
        if logger:
            took = payload.get("total_s")
            logger.info(f"  nvalchemi: {sum(o[0] is not None for o in out)}/{len(out)} "
                        f"relaxed" + (f" in {took / 60:.1f} min" if took else "")
                        + (f"; {unconverged} stopped at max_steps above fmax "
                           f"{self.cfg.nva_fmax:g}" if unconverged else ""))
        return out
