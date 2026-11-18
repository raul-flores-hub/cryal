"""A stand-in for cryal/backends/_nvalchemi_worker.py in the tests.

Same command line and the same task.json / result.json, no nvalchemi: every
structure comes back unchanged, with an energy that depends on its volume.
FAKE_NVA_FAIL names structures to report as exploded (comma-separated step
names), FAKE_NVA_EXIT makes the worker exit with that code, FAKE_NVA_SLEEP
makes it sleep that many seconds first, and FAKE_NVA_LOG appends one line per
batch (its structure names) to that file.
"""
import json
import os
import sys
import time

task_path, result_path = sys.argv[1], sys.argv[2]
task = json.load(open(task_path))
if os.environ.get("FAKE_NVA_SLEEP"):
    time.sleep(float(os.environ["FAKE_NVA_SLEEP"]))
if os.environ.get("FAKE_NVA_EXIT"):
    sys.exit(int(os.environ["FAKE_NVA_EXIT"]))
fail = set(filter(None, os.environ.get("FAKE_NVA_FAIL", "").split(",")))
if os.environ.get("FAKE_NVA_LOG"):
    with open(os.environ["FAKE_NVA_LOG"], "a") as f:
        f.write(",".join(s["name"] for s in task["structures"]) + "\n")


def volume(cell):
    (a, b, c) = cell
    return abs(a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
               + a[2] * (b[0] * c[1] - b[1] * c[0]))


results = []
for s in task["structures"]:
    if s["name"] in fail:
        results.append({"name": s["name"], "status": "failed",
                        "reason": "exploded during NPT stage 1 (non-finite coordinates)"})
        continue
    v = volume(s["atoms"]["cell"])
    results.append({"name": s["name"], "status": "ok", "energy": -1000.0 + 1e-3 * v,
                    "fmax": 5e-4, "converged": True, "atoms": s["atoms"]})
json.dump({"version": task["version"], "results": results, "total_s": 0.0},
          open(result_path, "w"))
