#!/usr/bin/env python3
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
distinct_minima.py — how many different minima a finished run found.

    python tools/distinct_minima.py AL_results/                 # one run
    python tools/distinct_minima.py runA/ runB/ --window 5      # compare runs
    python tools/distinct_minima.py AL_results/ --calibrate     # pick the threshold

The lowest energy a search reaches says how deep it went; how many distinct
low-energy structures it found says how much of the landscape it saw. A
crystal-structure search is judged on both — the experimental form is often
not the global minimum of the potential, so a search that finds the lowest
structure and nothing near it can miss the one that matters.

Structures within `--window` kJ/mol per molecule of each run's best are
grouped, lowest first, by the rule of [duplicates] (cryal.genetic.same_structure):
the same minimum when energies per molecule agree within --energy and
packings within --fingerprint (cryal/fingerprint.py). Each group is one
minimum; the table lists them with how often and how early the run found
each, and which source of candidates found it first.

--calibrate prints, instead, the fingerprint distance between every two
structures of the window whose energies already agree: one minimum reached
several times gives a cluster of small distances, distinct structures a
spread of larger ones, and the threshold goes in the gap. Nothing about a
potential or a molecule fixes it in advance; the default (0.02) comes from
benzene.

The molecule and Z are read from the run's final_report.json, or given with
--molecule and --Z for a run that did not finish.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cryal.config import Config                         # noqa: E402
from cryal.genetic import (Molecule, record_fingerprint,  # noqa: E402
                           same_structure)
from cryal.search import source_family                  # noqa: E402

KJ_PER_EV = 96.485


def load_run(path, molecule=None, Z=None):
    """(records, molecule file, Z) of a run directory or database.json."""
    run_dir = path if os.path.isdir(path) else os.path.dirname(path)
    db_path = os.path.join(run_dir, "database.json") if os.path.isdir(path) else path
    with open(db_path) as f:
        records = json.load(f)
    report = os.path.join(run_dir, "final_report.json")
    if os.path.exists(report) and (molecule is None or Z is None):
        with open(report) as f:
            conf = json.load(f).get("config", {})
        molecule = molecule or conf.get("molecule")
        Z = Z or conf.get("Z")
    if molecule is None or Z is None:
        raise SystemExit(f"{run_dir}: no final_report.json to read the molecule and "
                         "Z from; give --molecule and --Z")
    for r in records:
        # CIF paths are written relative to where the run was started.
        if not os.path.exists(r['cif_path']):
            alt = os.path.join(run_dir, "structures", os.path.basename(r['cif_path']))
            if os.path.exists(alt):
                r['cif_path'] = alt
    return records, molecule, int(Z)


def window(records, Z, kj):
    best = min(r['energy'] for r in records)
    return sorted((r for r in records if (r['energy'] - best) / Z * KJ_PER_EV <= kj),
                  key=lambda r: r['energy']), best


def group(records, cfg, mol):
    """Minima, lowest first: lists of records, each list one structure."""
    minima = []
    for r in records:
        for m in minima:
            if same_structure(r, m[0], cfg, mol):
                m.append(r)
                break
        else:
            minima.append([r])
    return minima


def calibrate(records, cfg, mol):
    from cryal.fingerprint import distance
    ds = []
    for a in range(len(records)):
        for b in range(a + 1, len(records)):
            ra, rb = records[a], records[b]
            if abs(ra['energy'] - rb['energy']) <= cfg.dup_energy * cfg.Z:
                fa, fb = record_fingerprint(ra, mol), record_fingerprint(rb, mol)
                if fa is not None and fb is not None:
                    ds.append(distance(fa, fb))
    return np.array(ds)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+", help="run directories (or their database.json)")
    p.add_argument("--window", type=float, default=5.0,
                   help="kJ/mol per molecule above each run's best (default 5)")
    p.add_argument("--energy", type=float, default=Config().dup_energy,
                   help="same minimum: energies within this, eV per molecule")
    p.add_argument("--fingerprint", type=float, default=Config().dup_fingerprint,
                   help="same minimum: fingerprint distance within this (0: density)")
    p.add_argument("--density", type=float, default=Config().dup_density,
                   help="g/cm3, used only with --fingerprint 0")
    p.add_argument("--molecule", help="the molecule's XYZ, if not in final_report.json")
    p.add_argument("--Z", type=int, help="molecules per cell, if not in final_report.json")
    p.add_argument("--top", type=int, default=20, help="minima listed per run")
    p.add_argument("--calibrate", action="store_true",
                   help="print fingerprint distances between energy-equal pairs instead")
    p.add_argument("--json", help="also write the result to this file")
    a = p.parse_args(argv)

    out = {}
    for path in a.runs:
        records, molecule, Z = load_run(path, a.molecule, a.Z)
        mol = Molecule.from_file(molecule)
        cfg = Config(Z=Z, dup_energy=a.energy, dup_fingerprint=a.fingerprint,
                     dup_density=a.density)
        near, best = window(records, Z, a.window)
        print(f"\n{path}: {len(records)} structures, best {best:.4f} eV, "
              f"{len(near)} within {a.window:g} kJ/mol per molecule")

        if a.calibrate:
            ds = calibrate(near, cfg, mol)
            if not len(ds):
                print("  no two structures agree in energy: nothing to calibrate on")
                continue
            print(f"  {len(ds)} pairs with energies within {a.energy:g} eV/molecule; "
                  "fingerprint distance D:")
            edges = [0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 1.0]
            hist, _ = np.histogram(ds, bins=edges)
            for lo, hi, n in zip(edges[:-1], edges[1:], hist):
                print(f"    {lo:5.3f} - {hi:5.3f}  {n:6d}  " + "#" * int(60 * n / max(hist)))
            out[path] = {'pairs': len(ds), 'D': sorted(ds.tolist())}
            continue

        minima = group(near, cfg, mol)
        rule = (f"fingerprint <= {a.fingerprint:g}" if a.fingerprint > 0
                else f"density within {a.density:g} g/cm3")
        print(f"  {len(minima)} distinct minima (energy within {a.energy:g} eV/molecule, "
              f"{rule})")
        print(f"  {'#':>3} {'dE kJ/mol':>10} {'rho':>7} {'found':>6} {'first cycle':>12}  first found by")
        rows = []
        for k, m in enumerate(minima):
            first = min(m, key=lambda r: (r['cycle'], r['id']))
            dE = (m[0]['energy'] - best) / Z * KJ_PER_EV
            rows.append({'rank': k + 1, 'dE_kJ_mol': round(dE, 3),
                         'density': m[0]['density'], 'found': len(m),
                         'first_cycle': first['cycle'],
                         'first_source': source_family(first['source']),
                         'ids': [r['id'] for r in m]})
            if k < a.top:
                print(f"  {k + 1:>3} {dE:>10.3f} {m[0]['density']:>7.3f} {len(m):>6} "
                      f"{first['cycle']:>12}  {source_family(first['source'])}")
        out[path] = {'best_energy': best, 'window_kJ_mol': a.window,
                     'n_minima': len(minima), 'minima': rows}

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
