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
_nvalchemi_worker.py — relax one batch of crystals with NVIDIA ALCHEMI Toolkit.

Run by cryal/backends/nvalchemi.py with the interpreter of the environment
that has nvalchemi-toolkit installed, never imported by CrYAL:

    <backend.nvalchemi.python> -I _nvalchemi_worker.py task.json result.json

It imports nothing from CrYAL, so that environment needs only nvalchemi, torch
and ASE. task.json holds the protocol and the structures; result.json gets, for
each structure in order, its status, relaxed geometry, energy and residual force.

The protocol: the NPT stages of `stages` (triclinic by default), each a fresh
integrator started from where the previous one ended, then L-BFGS with the cell
at zero pressure until fmax or max_steps. A final single point gives the energy
and the residual force.

What the validation of this backend against LAMMPS taught, and what is done
about it here:

* Total energies in float32 cannot rank crystals: the sum of the atomic
  reference energies (~2e5 eV for an organic cell of a few hundred atoms) is
  beyond float32 resolution, and the error changed from one structure to
  another by up to 0.17 eV per cell. With dtype float32 the model is loaded
  with its atomic energies set to zero, so float32 carries only the
  interaction, and their sum is added back in float64 from the model file.
  That reproduced float64 rankings to 2 meV per cell.
* One structure that explodes (non-finite coordinates) makes the neighbour
  list of the whole batch fail. The state at the start of every stage is
  kept; when a stage fails, the structures with non-finite coordinates are
  marked as failed and the stage is repeated, from its start, with the rest.
* The variable-cell L-BFGS needs aligned cells (a along x, b in the xy
  plane): every cell is rotated so, rigidly, before it.
* Strongly tilted cells can leave the neighbour list without a method: the
  final single point is computed on the Minkowski-reduced cell, the same
  lattice in a shorter basis.
* nvalchemi sizes the neighbour matrix from a density of 0.2 atoms/Å³ (80
  neighbours at 4.5 Å), and a cell compressed at 3000 bar exceeded it
  ("NeighborOverflowError: 88 > 80", a batch of 240 TMPA-An cells lost). The
  worker asks for room for 0.4 atoms/Å³ (160) and repeats an overflowing
  stage with twice as much.
* PyTorch's caching allocator fragments over an NPT stage until nvalchemi's
  Warp kernels cannot get GPU memory; the backend starts this process with
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (nvalchemi.worker_env).
* MACEWrapper.from_checkpoint passes every path to MACE's model downloader,
  which in mace-torch 0.3.15 does not accept a local file: a local file is
  returned as it is.
"""

import json
import os
import sys
import time

import numpy as np

BAR = 6.241509074e-7           # eV/Å³ per bar
PROTOCOL_VERSION = 1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def atoms_from_dict(d):
    from ase import Atoms
    return Atoms(symbols=d["symbols"], positions=d["positions"], cell=d["cell"],
                 pbc=d.get("pbc", True))


def atoms_to_dict(a):
    return {"symbols": list(a.get_chemical_symbols()),
            "positions": a.get_positions().tolist(),
            "cell": a.cell.array.tolist(),
            "pbc": [bool(x) for x in a.pbc]}


def finite(a):
    return bool(np.isfinite(a.positions).all() and np.isfinite(a.cell.array).all())


def main(task_path, result_path):
    t_start = time.time()
    with open(task_path) as f:
        task = json.load(f)
    if task.get("version") != PROTOCOL_VERSION:
        raise SystemExit(f"task version {task.get('version')} != {PROTOCOL_VERSION}")

    import torch
    import mace.calculators.foundations_models as fm
    import nvalchemi
    from ase.cell import Cell
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import NPT, LBFGSVariableCell
    from nvalchemi.dynamics import ConvergenceHook
    from nvalchemi.dynamics.hooks import AlignCellHook
    from nvalchemi.models.mace import MACEWrapper
    from nvalchemi.neighbors import compute_neighbors
    from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors

    download = fm.download_mace_mp_checkpoint
    fm.download_mace_mp_checkpoint = lambda p=None, *a, **k: (
        str(p) if p is not None and os.path.isfile(str(p)) else download(p, *a, **k))

    device = torch.device(task["device"])
    dtype = {"float32": torch.float32, "float64": torch.float64}[task["dtype"]]
    raw = torch.load(task["model"], weights_only=False, map_location="cpu")
    e0 = dict(zip(raw.atomic_numbers.tolist(),
                  raw.atomic_energies_fn.atomic_energies.tolist()))
    del raw
    split_e0 = dtype == torch.float32
    model = MACEWrapper.from_checkpoint(
        task["model"], device=device, dtype=dtype, enable_cueq=task["cueq"],
        atomic_energies={z: 0.0 for z in e0} if split_e0 else None)
    model.model_config.active_outputs = {"energy", "forces", "stress"}
    log(f"nvalchemi {nvalchemi.__version__}, torch {torch.__version__}, {task['dtype']}"
        f"{' (interaction only; E0 added in float64)' if split_e0 else ''}, "
        f"cueq {task['cueq']}, {len(task['structures'])} structures")

    names = [s["name"] for s in task["structures"]]
    geoms = [atoms_from_dict(s["atoms"]) for s in task["structures"]]
    failed = {}                                   # index -> reason
    timings = {}

    def to_batch(idx):
        # Every float field in the model's dtype: from_atoms defaults to float32,
        # and Warp's float64 kernels refuse float32 masses.
        return Batch.from_data_list([AtomicData.from_atoms(geoms[i], device="cpu", dtype=dtype)
                                     for i in idx], device=device)

    def from_batch(b, idx):
        pos = b.positions.detach().double().cpu().numpy()
        cell = b.cell.detach().double().cpu().numpy().reshape(-1, 3, 3)
        ptr = np.cumsum([0] + [len(geoms[i]) for i in idx])
        out = []
        for k, i in enumerate(idx):
            a = geoms[i].copy()
            a.set_cell(cell[k], scale_atoms=False)
            a.positions = pos[ptr[k]:ptr[k + 1]]
            out.append(a)
        return out

    # Neighbour-matrix width. nvalchemi's hook sizes it once, from an estimate
    # at 0.2 atoms/Å³: 80 for MACE-OFF23's 4.5 Å cutoff. Relaxed TMPA-An cells
    # have at most 54 neighbours, but under 3000 bar one reached 88 mid-stage
    # ("NeighborOverflowError: 88 > 80") and failed a batch of 240. The floor
    # here is the estimate at 0.4 atoms/Å³ (160 at 4.5 Å; ~16 bytes per atom
    # and neighbour). If a stage overflows anyway, the floor is doubled and the
    # stage repeated from its start, at most three times.
    nc = model.model_config.neighbor_config
    k_floor = [estimate_max_neighbors(cutoff=nc.cutoff + (nc.skin or 0.0), atomic_density=0.4)]
    log(f"neighbour matrix: at least {k_floor[0]} neighbours per atom")

    def run_stage(label, make_engine, n_steps, prepare=None):
        """One stage over the live structures; repeated without the ones that explode."""
        overflows = 0
        while True:
            idx = [i for i in range(len(geoms)) if i not in failed]
            if not idx:
                return
            if prepare:
                for i in idx:
                    prepare(geoms[i])
            batch = to_batch(idx)
            t0 = time.time()
            error = None
            try:
                batch = make_engine().run(batch, n_steps)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            except Exception as exc:              # the batch may hold the culprits
                error = f"{type(exc).__name__}: {exc}"
            new = from_batch(batch, idx)
            bad = [i for i, a in zip(idx, new) if not finite(a)]
            if error is None and not bad:
                for i, a in zip(idx, new):
                    geoms[i] = a
                timings[label] = timings.get(label, 0.0) + time.time() - t0
                log(f"{label}: {len(idx)} structures, {time.time() - t0:.0f} s")
                return
            if not bad and "NeighborOverflowError" in error and overflows < 3:
                overflows += 1
                k_floor[0] *= 2
                log(f"{label}: neighbour matrix overflowed ({error}); repeating the stage "
                    f"from its start with at least {k_floor[0]} neighbours per atom")
                continue
            if not bad:
                # Failed with every structure finite: no culprit to remove.
                for i in idx:
                    failed[i] = f"{label} failed: {error}"
                log(f"{label}: failed with no non-finite structure — {error}")
                return
            for i in bad:
                failed[i] = f"exploded during {label} (non-finite coordinates)"
            log(f"{label}: {', '.join(names[i] for i in bad)} exploded; repeating the "
                f"stage from its start with the other {len(idx) - len(bad)}")
            # geoms[] still holds every structure's state at the stage's start.

    hooks = lambda: model.make_neighbor_hooks(max_neighbors=k_floor[0])
    for k, st in enumerate(task["stages"], 1):
        if st["steps"] <= 0:
            continue
        def engine(st=st):
            return NPT(model=model, dt=task["timestep"], temperature=st["temperature"],
                       pressure=st["pressure"] * BAR,
                       barostat_time=task["barostat_time"],
                       thermostat_time=task["thermostat_time"],
                       pressure_coupling=task["pressure_coupling"], chain_length=3,
                       hooks=hooks())
        run_stage(f"NPT stage {k} ({st['temperature']:g} K, {st['pressure']:g} bar, "
                  f"{st['steps']} steps)", engine, st["steps"])

    def align(a):
        a.set_cell(Cell.fromcellpar(a.cell.cellpar()), scale_atoms=True)

    def minimizer():
        return LBFGSVariableCell(model=model,
                                 hooks=hooks() + [AlignCellHook(frequency=1)],
                                 convergence_hook=ConvergenceHook.from_fmax(task["fmax"]))
    run_stage(f"L-BFGS with cell (fmax {task['fmax']:g}, at most {task['max_steps']} steps)",
              minimizer, task["max_steps"], prepare=align)

    # Final single point, on the Minkowski-reduced cell.
    results = [None] * len(geoms)
    live = [i for i in range(len(geoms)) if i not in failed]
    if live:
        reduced = []
        for i in live:
            a = geoms[i].copy()
            rcell, _ = a.cell.minkowski_reduce()
            a.set_cell(rcell, scale_atoms=False)
            a.wrap()
            reduced.append(a)
        batch = Batch.from_data_list([AtomicData.from_atoms(a, device="cpu", dtype=dtype)
                                      for a in reduced], device=device)
        compute_neighbors(batch, config=model.model_config.neighbor_config,
                          max_neighbors=k_floor[0])
        out = model(batch)
        energy = out["energy"].detach().double().cpu().numpy().reshape(-1)
        forces = out["forces"].detach().double().cpu()
        ptr = np.cumsum([0] + [len(a) for a in reduced])
        for k, (i, a) in enumerate(zip(live, reduced)):
            e = float(energy[k]) + (sum(e0[int(z)] for z in a.numbers) if split_e0 else 0.0)
            fmax = float(forces[ptr[k]:ptr[k + 1]].norm(dim=1).max())
            if not np.isfinite(e):
                failed[i] = "non-finite energy at the final single point"
                continue
            results[i] = {"name": names[i], "status": "ok", "energy": e, "fmax": fmax,
                          "converged": fmax <= task["fmax"], "atoms": atoms_to_dict(a)}
    for i, reason in failed.items():
        results[i] = {"name": names[i], "status": "failed", "reason": reason}

    payload = {"version": PROTOCOL_VERSION, "results": results, "timings_s": timings,
               "total_s": time.time() - t_start,
               "nvalchemi_version": nvalchemi.__version__, "torch_version": torch.__version__,
               "gpu_peak_GB": (torch.cuda.max_memory_allocated() / 1e9
                               if device.type == "cuda" else None)}
    tmp = result_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, result_path)
    n_ok = sum(r["status"] == "ok" for r in results)
    log(f"done: {n_ok}/{len(results)} relaxed in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: _nvalchemi_worker.py task.json result.json")
    main(sys.argv[1], sys.argv[2])
