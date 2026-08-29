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
_remote_worker.py — the far end of a distributed evaluation.

The server runs, over ssh:

    <python> -m cryal._remote_worker <task_dir>          # evaluate one candidate
    <python> -m cryal._remote_worker --check <job_dir>   # can this machine do it at all?

`task_dir` holds one `task.json`: a candidate structure, the server's own
configuration, and the reference bonds the server computed.

The `--check` form runs once per machine before the run starts. It rebuilds the
server's configuration and puts it through the selected backend's own
`validate_config()` — the same call that guards a local run — so a worker
missing the potential, the calculator or the input script is reported in the
first seconds. Without it such a machine stays in the pool and returns a
rejection for every candidate it is given, which the server cannot tell from a
structure that genuinely lost: it would quietly eat its share of every cycle. This module
relaxes that one structure and writes `result.json` beside it. It is a
single-shot program, not a daemon: nothing is left running on the worker
between candidates, and a machine that is rebooted mid-run costs one
candidate.

Two rules give the result its meaning.

The configuration is the server's, reconstructed field by field — never
reinvented here. A worker that made up its own energy bound, species order or
integrity setting would return numbers that are not comparable with the
server's, and no inspection of the database would reveal it.

The relaxation goes through `EnergyBackend.evaluate()`, the same call the
serial path makes, so the pre-relaxation gate, the energy bound and the
molecular-integrity check apply here exactly as they do at home.

A rejected structure is a result, not an error: `ok: false` with a reason.
An error is this program failing to reach a verdict at all, and the server
treats the two differently — the first is recorded, the second is retried
somewhere else.
"""

import json
import logging
import os
import sys
import traceback
from dataclasses import fields

#: How much of the worker's own log to hand back with a negative verdict.
#: Enough to say why a candidate was rejected, small enough to sit in JSON.
LOG_TAIL_CHARS = 4000


def _config_from_dict(d: dict):
    """Rebuild the server's Config, ignoring keys this version does not know.

    Tolerating unknown keys keeps a worker one version behind from failing on
    every candidate; the version mismatch is already reported by the server's
    preflight, which is the right place to raise it.
    """
    from .config import Config
    known = {f.name for f in fields(Config)}
    return Config(**{k: v for k, v in d.items() if k in known})


def _write_result(path: str, result: dict):
    """Write the verdict. The server reads this file and nothing else."""
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, path)          # the server never sees a half-written result


def _log_tail(log_path: str) -> str:
    try:
        with open(log_path) as f:
            return f.read()[-LOG_TAIL_CHARS:]
    except Exception:
        return ""


def run_task(task_dir: str) -> dict:
    """Evaluate the single candidate described in task_dir/task.json."""
    from .backends import get_backend
    from .parallel import atoms_from_dict, atoms_to_dict

    with open(os.path.join(task_dir, "task.json")) as f:
        task = json.load(f)

    cfg = _config_from_dict(task["config"])
    atoms = atoms_from_dict(task["atoms"])
    ref_bonds = [tuple(b) for b in task.get("ref_bonds") or []]
    mol_size = task.get("mol_size")

    # Everything the backend logs about this candidate — including why the
    # gate rejected it — goes to a file the server can read back.
    log_path = os.path.join(task_dir, "worker.log")
    logger = logging.getLogger(f"cryal.worker.{task.get('task_id', 'task')}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s",
                                           datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    backend = get_backend(cfg, logger)
    logger.info(f"backend: {backend.describe()}")
    logger.info(f"{len(atoms)} atoms, {len(ref_bonds)} reference bonds, "
                f"mol_size={mol_size}")

    relaxed, energy = backend.evaluate(atoms, task_dir, ref_bonds, mol_size,
                                       logger=logger)
    handler.flush()

    if relaxed is None or energy is None:
        return {"ok": False,
                "reason": "rejected by the backend (gate, energy bound, "
                          "integrity check, or a failed relaxation)",
                "log": _log_tail(log_path)}

    return {"ok": True,
            "energy": float(energy),
            "atoms": atoms_to_dict(relaxed)}


def check_job(job_dir: str) -> dict:
    """Can this machine run the server's configuration?

    Deliberately the backend's own validate_config() rather than a checklist
    written here: whatever a backend requires to relax a structure, it already
    states there, and a second copy of that knowledge would drift.
    """
    from .backends import backend_class

    with open(os.path.join(job_dir, "config.json")) as f:
        cfg = _config_from_dict(json.load(f))

    backend = backend_class(cfg.energy_backend)
    backend.validate_config(cfg)
    return {"ok": True, "backend": backend.name}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if argv and argv[0] == "--check":
        if len(argv) != 2:
            print("usage: python -m cryal._remote_worker --check <job_dir>",
                  file=sys.stderr)
            return 2
        try:
            print(json.dumps(check_job(argv[1])))
            return 0
        except Exception as e:
            # One line the server can put straight in its log, and the full
            # story on stderr for the ssh transcript.
            print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
            print(traceback.format_exc()[-LOG_TAIL_CHARS:], file=sys.stderr)
            return 1

    if len(argv) != 1:
        print("usage: python -m cryal._remote_worker <task_dir>", file=sys.stderr)
        return 2

    task_dir = argv[0]
    result_path = os.path.join(task_dir, "result.json")

    try:
        result = run_task(task_dir)
    except Exception as e:
        # Still write a verdict: a result file that says what went wrong tells
        # the server this machine is alive and the candidate is the problem.
        # No file at all is what makes it retry the candidate elsewhere.
        result = {"ok": False,
                  "error": f"{type(e).__name__}: {e}",
                  "traceback": traceback.format_exc()[-LOG_TAIL_CHARS:]}
        try:
            _write_result(result_path, result)
        except Exception:
            traceback.print_exc()
        print(result["traceback"], file=sys.stderr)
        return 1

    _write_result(result_path, result)
    return 0 if result.get("ok") else 3


if __name__ == "__main__":
    sys.exit(main())
