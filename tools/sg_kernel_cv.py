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
sg_kernel_cv.py — choose the space-group kernel from data already computed.

Three open decisions about cryal/sg_kernel.py are settled here by
cross-validation on the databases of finished CrYAL runs:

  1. structure       k_cell x k_SG                 ("prod")
                     k_cell x k_SG + k_SG'          ("sum": adds a per-group
                                                    offset independent of cell)
  2. hierarchy       exchangeable (one correlation for any two groups)  "X"
                     lattice + family                                   "LF"
                     LF + point group                                   "LFP"
                     LF + centrosymmetry                                "LFC"
                     LF + point group + centrosymmetry                  "LFPC"
  3. regularisation  maximum likelihood within wide bounds              "mle"
                     MAP, log-normal prior log w ~ N(0, 3^2) on every
                     space-group weight                                 "map"

20 variants, plus the reference: the GP CrYAL runs today, a Matern over the
7 cell features with no space group ("nosg").

The protocol was fixed before any result was seen:

  Data. A run enters when it is finished, has >= 100 structures with a
  space group, >= 2 groups with >= 10 structures each, and >= 3 cycles.
  Structures without a space group are left out (seeds, pool
  perturbations). Of runs that continue one another (one database holds
  >= 50 % of another's structures) only the largest is kept. The
  statistical unit is the molecule + potential: runs of one molecule are
  averaged before testing.

  Splits.  "cycle" (primary): train on cycles < c, predict cycle c, for
           every c with >= 50 training and >= 5 test structures; this is
           how the GP is used in a run.
           "logo": leave one space group out (groups with >= 10 structures).
           "kfold": 5-fold at random, as a control.
  Training sets are capped at 400 structures, subsampled with a seed fixed
  by (run, split, fold), so every variant sees the same structures. Each fit
  starts the optimiser once, from the same initial hyperparameters.

  Metric. Primary: mean negative log predictive density (NLPD) of the test
  structures, in the cycle split. Secondary: RMSE, Spearman rank
  correlation, and where the true best test structure falls in the
  predicted ranking.

  Decision. Among the 20 space-group variants, take the one with the lowest
  median per-unit NLPD. Compare every other variant with it by a one-sided
  Wilcoxon signed-rank test over units, Holm-corrected over the 19
  comparisons. Of the variants not significantly worse (adjusted p >= 0.05)
  choose the least complex -- structure prod < sum; hierarchy X < LF <
  LFP = LFC < LFPC; mle < map -- breaking ties by median NLPD. A more complex
  option is therefore adopted only where the data show it is better. The
  choice is then compared with "nosg", two-sided.

Usage:
    python tools/sg_kernel_cv.py run    DATA_DIR results.csv [--jobs N]
    python tools/sg_kernel_cv.py report DATA_DIR results.csv [report.md]

DATA_DIR holds MANIFEST.json and one folder per run with its database.json
(the layout of cryal_gp_data_*.tar.gz). `run` appends one row per fit to
results.csv and skips fits already there, so it can be stopped and resumed.
It uses the CPU only. With --jobs N it runs N fits at once, each with one
BLAS thread; leave a core free for every LAMMPS process on the machine.
"""

import os

# One BLAS thread per fit: many small fits run better side by side than one
# at a time across all cores. Set before numpy is imported; an explicit
# setting in the environment wins.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import csv
import hashlib
import json
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CELL = ["a", "b", "c", "alpha", "beta", "gamma", "density"]
MAX_TRAIN = 400
PRIOR_SIGMA = 3.0
STRUCTURES = ("prod", "sum")
HIERARCHIES = ("X", "LF", "LFP", "LFC", "LFPC")
REGULARISATIONS = ("mle", "map")
VARIANTS = [f"{s}-{h}-{r}" for s in STRUCTURES for h in HIERARCHIES
            for r in REGULARISATIONS]
REFERENCE = "nosg"
COMPLEXITY = {"prod": 0, "sum": 1, "X": 0, "LF": 1, "LFP": 2, "LFC": 2,
              "LFPC": 3, "mle": 0, "map": 1}
FIELDS = ["run", "unit", "split", "fold", "variant", "n_train", "n_test",
          "nlpd", "rmse", "spearman", "best_rank", "seconds", "shares", "error"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _key(rec):
    return (round(rec["energy"], 4), round(rec["a"], 3), round(rec["b"], 3),
            round(rec["c"], 3))


def load_runs(data_dir, log=print):
    """Eligible runs, one per lineage: {run: {"unit", "records"}}."""
    manifest = json.load(open(os.path.join(data_dir, "MANIFEST.json")))
    runs = {}
    for entry in manifest["runs"]:
        name = entry["run"]
        path = os.path.join(data_dir, name, "database.json")
        if entry.get("status") != "terminada" or not os.path.exists(path):
            continue
        recs = [r for r in json.load(open(path)) if r.get("space_group") is not None]
        counts = defaultdict(int)
        for r in recs:
            counts[r["space_group"]] += 1
        if (len(recs) < 100 or sum(n >= 10 for n in counts.values()) < 2
                or len({r["cycle"] for r in recs}) < 3):
            continue
        unit = f"{entry.get('molecule') or entry.get('formula')} | {entry.get('potential')}"
        runs[name] = {"unit": unit, "records": recs}

    # Lineages: a run that holds >= 50 % of another's structures continues it.
    keys = {n: {_key(r) for r in v["records"]} for n, v in runs.items()}
    parent = {n: n for n in runs}

    def find(n):
        while parent[n] != n:
            n = parent[n]
        return n

    names = sorted(runs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = len(keys[a] & keys[b])
            if shared >= 0.5 * min(len(keys[a]), len(keys[b])):
                parent[find(a)] = find(b)
    lineages = defaultdict(list)
    for n in names:
        lineages[find(n)].append(n)
    kept = {}
    for members in lineages.values():
        best = max(members, key=lambda n: len(runs[n]["records"]))
        kept[best] = runs[best]
        if len(members) > 1:
            log(f"lineage {sorted(members)} -> kept {best}")
    return kept


def folds(run, records):
    """[(split, fold, train_idx, test_idx)] for one run."""
    cycle = np.array([r["cycle"] for r in records])
    group = np.array([r["space_group"] for r in records])
    out = []
    for c in sorted(set(cycle.tolist())):
        train, test = np.where(cycle < c)[0], np.where(cycle == c)[0]
        if len(train) >= 50 and len(test) >= 5:
            out.append(("cycle", str(c), train, test))
    for g in sorted(set(group.tolist())):
        test = np.where(group == g)[0]
        train = np.where(group != g)[0]
        if len(test) >= 10 and len(train) >= 50:
            out.append(("logo", str(g), train, test))
    order = _rng(run, "kfold", "").permutation(len(records))
    for k in range(5):
        test = np.sort(order[k::5])
        train = np.setdiff1d(np.arange(len(records)), test)
        out.append(("kfold", str(k), train, test))
    return [(s, f, _cap(run, s, f, tr), te) for s, f, tr, te in out]


def _rng(*parts):
    seed = int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def _cap(run, split, fold, train):
    if len(train) <= MAX_TRAIN:
        return train
    return np.sort(_rng(run, split, fold).choice(train, MAX_TRAIN, replace=False))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def build_kernel(variant):
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    from cryal.sg_kernel import OnColumns, SpaceGroupKernel

    white = WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    matern = Matern(length_scale=np.ones(len(CELL)), length_scale_bounds=(1e-2, 1e2), nu=2.5)
    amplitude = ConstantKernel(1.0, (1e-3, 1e3))
    if variant == REFERENCE:
        return amplitude * matern + white

    structure, hierarchy, _ = variant.split("-")

    def sg():
        opts = dict(column=len(CELL), use_point_group="P" in hierarchy[2:],
                    use_centro="C" in hierarchy[2:])
        if hierarchy == "X":
            opts.update(w_lattice=0.0, w_lattice_bounds="fixed",
                        w_family=0.0, w_family_bounds="fixed")
        return SpaceGroupKernel(**opts)

    kernel = amplitude * OnColumns(matern, list(range(len(CELL)))) * sg()
    if structure == "sum":
        kernel = kernel + ConstantKernel(1.0, (1e-3, 1e3)) * sg()
    return kernel + white


def map_optimizer(kernel):
    """L-BFGS-B on -log likelihood + a N(0, PRIOR_SIGMA^2) prior on the log
    of every free space-group weight."""
    from scipy.optimize import minimize

    free = [h for h in kernel.hyperparameters if not h.fixed]
    idx, i = [], 0
    for h in free:
        n = h.n_elements
        if h.name.rsplit("__", 1)[-1].startswith("w_"):
            idx.extend(range(i, i + n))
        i += n
    idx = np.array(idx, dtype=int)

    def optimizer(obj_func, initial_theta, bounds):
        def f(theta):
            value, grad = obj_func(theta, eval_gradient=True)
            value = value + 0.5 * np.sum(theta[idx] ** 2) / PRIOR_SIGMA ** 2
            grad = grad.copy()
            grad[idx] += theta[idx] / PRIOR_SIGMA ** 2
            return value, grad
        res = minimize(f, initial_theta, jac=True, method="L-BFGS-B", bounds=bounds)
        return res.x, res.fun
    return optimizer


def fit_and_score(task):
    """One fit; returns a result row."""
    from scipy.stats import spearmanr
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.preprocessing import StandardScaler

    run, unit, split, fold, variant, X, y, train, test, restarts = task
    row = dict(run=run, unit=unit, split=split, fold=fold, variant=variant,
               n_train=len(train), n_test=len(test), error="")
    t0 = time.time()
    try:
        scaler = StandardScaler().fit(X[train, :len(CELL)])
        Xs = X.copy()
        Xs[:, :len(CELL)] = scaler.transform(X[:, :len(CELL)])
        if variant == REFERENCE:
            Xs = Xs[:, :len(CELL)]
        kernel = build_kernel(variant)
        optimizer = map_optimizer(kernel) if variant.endswith("-map") else "fmin_l_bfgs_b"
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True,
                                      optimizer=optimizer,
                                      n_restarts_optimizer=restarts, random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(Xs[train], y[train])
            mu, sd = gp.predict(Xs[test], return_std=True)
        sd = np.maximum(sd, 1e-9)
        err = y[test] - mu
        row["nlpd"] = float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + err ** 2 / (2 * sd ** 2)))
        row["rmse"] = float(np.sqrt(np.mean(err ** 2)))
        row["spearman"] = float(spearmanr(mu, y[test]).statistic) if len(test) > 2 else ""
        best = int(np.argmin(y[test]))
        row["best_rank"] = float(np.mean(mu < mu[best]))   # 0 = predicted best too
        row["shares"] = json.dumps(_shares(gp.kernel_)) if variant != REFERENCE else ""
    except Exception as e:                                  # recorded, not fatal
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    row["seconds"] = round(time.time() - t0, 2)
    return row


def _shares(kernel):
    from cryal.sg_kernel import SpaceGroupKernel
    out = []

    def walk(k):
        if isinstance(k, SpaceGroupKernel):
            out.append({n: round(v, 6) for n, v in k.variance_shares().items()})
        for attr in ("k1", "k2"):
            if hasattr(k, attr):
                walk(getattr(k, attr))
    walk(kernel)
    return out


def tasks(runs, done, restarts):
    """Every fit not yet in results, primary split first."""
    out = []
    for run, info in sorted(runs.items()):
        recs = info["records"]
        X = np.array([[r[k] for k in CELL] + [r["space_group"]] for r in recs], dtype=float)
        y = np.array([r["energy"] for r in recs], dtype=float)
        for split, fold, train, test in folds(run, recs):
            for variant in [REFERENCE] + VARIANTS:
                if (run, split, fold, variant) not in done:
                    out.append((run, info["unit"], split, fold, variant, X, y,
                                train, test, restarts))
    rank = {"cycle": 0, "logo": 1, "kfold": 2}
    return sorted(out, key=lambda t: rank[t[2]])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def unit_table(rows, split, metric="nlpd"):
    """{unit: {variant: value}}: test-size weighted mean per run, then the
    mean over the unit's runs. Units missing any variant are dropped."""
    per_run = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    unit_of = {}
    for r in rows:
        if r["split"] != split or r["error"] or r[metric] in ("", None):
            continue
        acc = per_run[r["run"]][r["variant"]]
        acc[0] += float(r[metric]) * int(r["n_test"])
        acc[1] += int(r["n_test"])
        unit_of[r["run"]] = r["unit"]
    by_unit = defaultdict(lambda: defaultdict(list))
    for run, variants in per_run.items():
        for v, (s, n) in variants.items():
            by_unit[unit_of[run]][v].append(s / n)
    every = set([REFERENCE] + VARIANTS)
    return {u: {v: float(np.mean(x)) for v, x in vs.items()}
            for u, vs in by_unit.items() if every <= set(vs)}


def complexity(variant):
    return sum(COMPLEXITY[p] for p in variant.split("-"))


def decide(table, alpha=0.05):
    """The decision rule of the module docstring. Returns a dict."""
    from scipy.stats import wilcoxon

    units = sorted(table)
    med = {v: float(np.median([table[u][v] for u in units])) for v in VARIANTS + [REFERENCE]}
    best = min(VARIANTS, key=lambda v: med[v])
    others = [v for v in VARIANTS if v != best]
    raw = {}
    for v in others:
        d = np.array([table[u][v] - table[u][best] for u in units])
        raw[v] = 1.0 if np.allclose(d, 0) else float(
            wilcoxon(d, alternative="greater").pvalue)
    # Holm step-down
    adj, running = {}, 0.0
    for i, v in enumerate(sorted(others, key=lambda v: raw[v])):
        running = max(running, min(1.0, (len(others) - i) * raw[v]))
        adj[v] = running
    keep = [best] + [v for v in others if adj[v] >= alpha]
    chosen = min(keep, key=lambda v: (complexity(v), med[v]))
    d = np.array([table[u][REFERENCE] - table[u][chosen] for u in units])
    p_ref = 1.0 if np.allclose(d, 0) else float(wilcoxon(d).pvalue)
    return dict(units=len(units), median=med, best=best, p_adj=adj,
                not_worse=keep, chosen=chosen, p_vs_reference=p_ref,
                median_gain_vs_reference=float(np.median(d)))


def report(rows, out):
    lines = ["# Space-group kernel: cross-validation", ""]
    ok = [r for r in rows if not r["error"]]
    lines.append(f"{len(rows)} fits, {len(rows) - len(ok)} failed; "
                 f"{len({r['run'] for r in ok})} runs, "
                 f"{len({r['unit'] for r in ok})} units.")
    for split in ("cycle", "logo", "kfold"):
        table = unit_table(ok, split)
        if len(table) < 6:
            lines += ["", f"## {split}: {len(table)} complete units, too few to test"]
            continue
        res = decide(table)
        lines += ["", f"## Split `{split}`" + (" (primary)" if split == "cycle" else ""),
                  "", f"{res['units']} units with every variant. "
                  f"Lowest median NLPD: `{res['best']}`. "
                  f"Not significantly worse (Holm, one-sided Wilcoxon): "
                  f"{', '.join(f'`{v}`' for v in res['not_worse'])}.",
                  "", f"**Choice: `{res['chosen']}`.** Against the GP without "
                  f"space group (`nosg`): median NLPD gain {res['median_gain_vs_reference']:+.4f} "
                  f"per structure, two-sided Wilcoxon p = {res['p_vs_reference']:.3g}.",
                  "", "| variant | median NLPD | Holm p vs best | complexity |",
                  "|---|---|---|---|"]
        for v in sorted(VARIANTS + [REFERENCE], key=lambda v: res["median"][v]):
            p = res["p_adj"].get(v)
            lines.append(f"| `{v}` | {res['median'][v]:.4f} | "
                         f"{'—' if p is None else f'{p:.3g}'} | "
                         f"{'—' if v == REFERENCE else complexity(v)} |")
        for metric in ("rmse", "spearman", "best_rank"):
            t = unit_table(ok, split, metric)
            if t:
                med = {v: np.median([t[u][v] for u in t]) for v in [REFERENCE, res["chosen"], res["best"]]}
                lines.append(f"\n{metric}: " + ", ".join(f"`{v}` {m:.4f}" for v, m in med.items()))
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("data_dir")
    r.add_argument("results")
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--restarts", type=int, default=0,
                   help="optimizer restarts per fit (default 0: one start, "
                        "the same for every variant, as fixed in the protocol)")
    r.add_argument("--dry-run", action="store_true", help="count the fits and stop")
    q = sub.add_parser("report")
    q.add_argument("data_dir")
    q.add_argument("results")
    q.add_argument("out", nargs="?", default="sg_kernel_cv_report.md")
    args = p.parse_args(argv)

    if args.cmd == "report":
        report(_read(args.results), args.out)
        return 0

    runs = load_runs(args.data_dir)
    units = {v["unit"] for v in runs.values()}
    print(f"{len(runs)} runs, {len(units)} units")
    done = {(r["run"], r["split"], r["fold"], r["variant"]) for r in _read(args.results)}
    todo = tasks(runs, done, args.restarts)
    print(f"{len(done)} fits already in {args.results}, {len(todo)} to do")
    if args.dry_run:
        by = defaultdict(int)
        for t in todo:
            by[t[2]] += 1
        print(dict(by))
        return 0

    new = not os.path.exists(args.results)
    with open(args.results, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        t0 = time.time()
        if args.jobs > 1:
            from multiprocessing import Pool
            pool = Pool(args.jobs)
            results = pool.imap_unordered(fit_and_score, todo, chunksize=1)
        else:
            results = map(fit_and_score, todo)
        for i, row in enumerate(results, 1):
            writer.writerow(row)
            f.flush()
            if i % 50 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                print(f"{i}/{len(todo)}  ~{rate * (len(todo) - i) / 3600:.1f} h left",
                      flush=True)
        if args.jobs > 1:
            pool.close()
            pool.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
