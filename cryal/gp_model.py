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
gp_model.py — Gaussian Process surrogate model and Expected Improvement.

The GP learns the mapping:
  features = [a, b, c, alpha, beta, gamma, density]  →  energy (eV)

Training uses relative energies (E - E_min) to improve numerical conditioning.
The kernel is Matérn(nu=2.5) × ConstantKernel + WhiteKernel.

Expected Improvement (EI) is used as the acquisition function:
  EI(x) = E[max(E_best - y(x), 0)]
  where E_best is the best energy observed so far.

The GP does NOT use any experimental structure knowledge. The EI will
naturally push sampling toward:
  1. Regions with high predicted density (monotonic trend in data)
  2. Unexplored regions (high GP uncertainty → high EI)
  Together these identify the favorable (beta, volume) combinations.
"""

import numpy as np
import warnings
from typing import List, Dict, Optional, Tuple

from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (Matern, RBF,
                                               ConstantKernel, WhiteKernel)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

FEATURE_NAMES = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'density']


def features_from_record(record: dict) -> np.ndarray:
    """
    Extract the 7-dimensional feature vector from a database record.
    record must have keys: a, b, c, alpha, beta, gamma, density.
    """
    return np.array([record['a'], record['b'], record['c'],
                     record['alpha'], record['beta'], record['gamma'],
                     record['density']])


def features_from_atoms(atoms, Z: int, mol_weight: float) -> np.ndarray:
    """Extract features directly from an ASE Atoms object."""
    from .utils import compute_density
    cp = atoms.cell.cellpar()
    rho = compute_density(atoms, Z, mol_weight)
    return np.array([cp[0], cp[1], cp[2], cp[3], cp[4], cp[5], rho])


# ---------------------------------------------------------------------------
# GP model
# ---------------------------------------------------------------------------

class GPSurrogate:
    """
    Gaussian Process surrogate for crystal structure energy prediction.

    Attributes
    ----------
    gp        : trained GaussianProcessRegressor
    scaler_X  : StandardScaler for features
    scaler_y  : StandardScaler for relative energies
    e_min     : minimum energy in training set (absolute reference)
    n_train   : number of training points
    """

    def __init__(self, kernel_type: str = "matern25", n_restarts: int = 5):
        self.kernel_type = kernel_type
        self.n_restarts  = n_restarts
        self.gp          = None
        self.scaler_X    = StandardScaler()
        self.scaler_y    = StandardScaler()
        self.e_min       = 0.0
        self.n_train     = 0
        self._length_scales = None

    def _build_kernel(self):
        if self.kernel_type == "rbf":
            base = RBF(length_scale=np.ones(7),
                       length_scale_bounds=(1e-2, 1e2))
        else:   # matern25 (default, more robust)
            base = Matern(length_scale=np.ones(7),
                          length_scale_bounds=(1e-2, 1e2), nu=2.5)
        return (ConstantKernel(1.0, (1e-3, 1e3)) * base
                + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1)))

    def fit(self, records: List[dict]) -> None:
        """
        Train the GP on a list of database records.
        Each record must have keys: a,b,c,alpha,beta,gamma,density,energy.
        """
        if len(records) < 5:
            raise ValueError(f"Need at least 5 records to train GP; got {len(records)}")

        X_raw = np.array([features_from_record(r) for r in records])
        y_raw = np.array([r['energy'] for r in records])

        self.e_min = y_raw.min()
        y_rel = y_raw - self.e_min   # relative energies (0 = best)

        X_scaled = self.scaler_X.fit_transform(X_raw)
        y_scaled = self.scaler_y.fit_transform(y_rel.reshape(-1, 1)).ravel()

        self.gp = GaussianProcessRegressor(
            kernel=self._build_kernel(),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts,
            random_state=42
        )
        self.gp.fit(X_scaled, y_scaled)
        self.n_train = len(records)

        # Store optimized length scales for feature importance
        try:
            self._length_scales = self.gp.kernel_.k1.k2.length_scale
            if np.isscalar(self._length_scales):
                self._length_scales = np.full(7, self._length_scales)
        except Exception:
            self._length_scales = None

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std of the energy (absolute, in eV) for feature matrix X.
        X shape: (n_samples, 7).
        """
        X_scaled = self.scaler_X.transform(X)
        mu_s, std_s = self.gp.predict(X_scaled, return_std=True)
        # Inverse-transform mean
        mu  = self.scaler_y.inverse_transform(mu_s.reshape(-1,1)).ravel() + self.e_min
        # std in original energy units
        std = std_s * self.scaler_y.scale_[0]
        return mu, std

    def feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Return relative feature importance as 1/length_scale (normalized).
        Higher = more important for energy prediction.
        """
        if self._length_scales is None:
            return None
        imp = 1.0 / self._length_scales
        imp /= imp.sum()
        return {name: float(v) for name, v in zip(FEATURE_NAMES, imp)}


# ---------------------------------------------------------------------------
# Expected Improvement
# ---------------------------------------------------------------------------

def expected_improvement(mu: np.ndarray, sigma: np.ndarray,
                         best_y: float, xi: float = 0.01) -> np.ndarray:
    """
    Expected Improvement acquisition function for minimization.
    EI = E[max(best_y - y, 0)]

    Parameters
    ----------
    mu     : predicted mean energy
    sigma  : predicted std of energy
    best_y : best observed energy (most negative)
    xi     : exploration bonus (higher = more exploration)
    """
    improvement = best_y - mu - xi
    Z = improvement / (sigma + 1e-9)
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    return ei


# ---------------------------------------------------------------------------
# Suggestion generation
# ---------------------------------------------------------------------------

def maximize_ei(gp: GPSurrogate, database_records: List[dict],
                cfg, rng: np.random.Generator) -> List[dict]:
    """
    Sample random candidate configurations and return those with highest EI.

    The candidate space covers:
      a, b, c : [min_cell_axis, 35] Å
      alpha, gamma : [70, 110]°
      beta : [cfg.beta_min, cfg.beta_max]°
      density : [density_min, density_max + margin] g/cm³

    Returns a list of suggestion dicts with keys:
      a, b, c, alpha, beta, gamma, density, volume, E_pred, E_std, EI
    """
    best_y = min(r['energy'] for r in database_records)
    NA     = 6.02214076e23

    # Sample candidates
    n = cfg.num_ei_candidates
    lo_ax, hi_ax = cfg.min_cell_axis, 35.0
    C = np.column_stack([
        rng.uniform(lo_ax, hi_ax, n),             # a
        rng.uniform(lo_ax, hi_ax, n),             # b
        rng.uniform(lo_ax, hi_ax, n),             # c
        rng.uniform(70.0, 110.0, n),              # alpha
        rng.uniform(cfg.beta_min, cfg.beta_max, n),  # beta
        rng.uniform(70.0, 110.0, n),              # gamma
        rng.uniform(cfg.density_min,              # density
                    cfg.density_max * 1.1, n),
    ])

    # Filter: reject volumes inconsistent with density
    V_from_density = (cfg.Z * cfg.mol_weight / NA) / (C[:, 6] * 1e-24)
    V_from_abc     = C[:, 0] * C[:, 1] * C[:, 2]  # orthorhombic approximation
    ratio = V_from_density / V_from_abc
    valid = (ratio > 0.4) & (ratio < 2.5)
    C = C[valid]

    if len(C) == 0:
        return []

    mu, std = gp.predict(C)
    ei_vals = expected_improvement(mu, std, best_y, xi=cfg.ei_xi)

    # Select diverse top suggestions (avoid clustering)
    X_scaled  = gp.scaler_X.transform(C)
    idx_sorted = np.argsort(-ei_vals)
    suggestions = []
    used_scaled = []

    for i in idx_sorted:
        if len(suggestions) >= cfg.num_ei_suggestions:
            break
        c_i = X_scaled[i]
        if any(np.linalg.norm(c_i - u) < 0.5 for u in used_scaled):
            continue
        used_scaled.append(c_i)
        V_i = (cfg.Z * cfg.mol_weight / NA) / (C[i, 6] * 1e-24)
        suggestions.append({
            'a':       float(C[i, 0]),
            'b':       float(C[i, 1]),
            'c':       float(C[i, 2]),
            'alpha':   float(C[i, 3]),
            'beta':    float(C[i, 4]),
            'gamma':   float(C[i, 5]),
            'density': float(C[i, 6]),
            'volume':  float(V_i),
            'E_pred':  float(mu[i]),
            'E_std':   float(std[i]),
            'EI':      float(ei_vals[i]),
        })

    return suggestions


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def gp_report(gp: GPSurrogate, suggestions: List[dict],
              best_energy: float, logger=None) -> str:
    """Generate a human-readable GP analysis report."""
    lines = ["  --- GP Surrogate Report ---",
             f"  Training points: {gp.n_train}",
             f"  Best energy:     {best_energy:.4f} eV"]

    imp = gp.feature_importance()
    if imp:
        lines.append("  Feature importance (1/length_scale, normalized):")
        for name, val in sorted(imp.items(), key=lambda x: -x[1]):
            bar = '█' * int(val * 40)
            lines.append(f"    {name:>10}: {bar} ({val:.3f})")

    if suggestions:
        lines.append(f"  Top {min(5, len(suggestions))} EI suggestions:")
        lines.append(f"    {'beta':>8} {'density':>9} {'volume':>8} "
                     f"{'E_pred':>12} {'±':>8} {'EI':>10}")
        for s in suggestions[:5]:
            lines.append(f"    {s['beta']:>8.1f} {s['density']:>9.3f} "
                         f"{s['volume']:>8.0f} {s['E_pred']:>12.4f} "
                         f"{s['E_std']:>8.4f} {s['EI']:>10.6f}")

    report = "\n".join(lines)
    if logger:
        for l in lines:
            logger.info(l)
    return report
