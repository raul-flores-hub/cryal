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
config.py — cryal.toml loader and configuration container.

The input is TOML: [tables] group the settings, values carry their own
types, and the file is read with the standard library's tomllib. Every
key is checked against the schema below, so a misspelt key or a value of
the wrong type stops the load with the path of the offending entry,
instead of leaving a default silently in its place. Files in the format
CrYAL used before (% SECTION headers) are turned into cryal.toml by
`cryal convert`; see legacy.py.

Provides a Config dataclass with typed fields and sensible defaults.
"""

import difflib
import os
import tomllib
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # --- GENERAL ---
    molecule_file:       str   = "molecule.xyz"
    Z:                   int   = 4
    random_seed:         int   = 42
    output_dir:          str   = "AL_results"
    log_level:           str   = "INFO"
    resume:              bool  = False   # continue from an existing output_dir

    # --- ENERGY BACKEND ---
    # Which engine relaxes a candidate and returns its energy. The default is
    # the LAMMPS/MACE-OFF path used for the published run, so an input file
    # written before backends existed keeps its exact meaning.
    energy_backend:      str   = "lammps_mace"
    # Total energies above this (eV, whole cell) are treated as a failed
    # relaxation. Only meaningful against a given potential's zero of energy;
    # false in the input switches the check off.
    energy_sanity_max:   Optional[float] = -100.0

    # --- BACKEND: lammps_mace ---
    lammps_input:        str   = "in_v3.lammps"
    lammps_command:      str   = "lmp"
    spec_order:          List[str] = field(default_factory=lambda: ["C", "H", "O", "N"])

    # --- BACKEND: ase ---
    # Dotted path to a calculator class or factory, called once with
    # ase_calculator_kwargs and reused for the whole run.
    ase_calculator:      str   = ""
    ase_calculator_kwargs: Dict[str, Any] = field(default_factory=dict)
    ase_optimizer:       str   = "FIRE"
    ase_fmax:            float = 0.05     # eV/Å
    ase_max_steps:       int   = 500
    ase_relax_cell:      bool  = True

    # --- BACKEND: uma ---
    # UMA carries a task, and each head has its own zero of energy. There is
    # deliberately no default for it: see cryal/backends/uma.py.
    uma_model:           str  = "uma-s-1p1"
    uma_task:            str  = ""
    uma_device:          str  = "cuda"
    uma_inference_settings: str = "default"

    # --- ACTIVE LEARNING ---
    num_cycles:          int   = 10
    initial_structures:  int   = 100
    structures_per_cycle:int   = 50
    pool_size:           int   = 20
    max_no_improve_cycles:int  = 3

    # --- SPACE GROUPS ---
    space_groups:        List[int]   = field(default_factory=lambda: [14,2,19,15,4,9,33,61])
    sg_weights:          List[float] = field(default_factory=lambda: [36,24,11,7,6,4,2,2])

    # --- VOLUME ---
    target_volume_min:   Optional[float] = None   # None → auto from density range
    target_volume_max:   Optional[float] = None
    density_min:         float = 0.8
    density_max:         float = 1.6
    volume_expansion_factors: List[float] = field(
        default_factory=lambda: [1.0, 1.05, 1.10, 1.15, 1.20, 1.35, 1.50])

    # --- CELL ---
    min_cell_axis:       float = 9.0

    # --- BETA MONOCLINIC ---
    beta_min:            float = 90.0
    beta_max:            float = 150.0
    beta_distribution:   str   = "uniform"   # 'uniform' or 'triangular'
    beta_mode:           Optional[float] = None  # only for triangular

    # --- GP ---
    # [gp] enabled = false runs the identical workflow with the surrogate
    # switched off: candidates are generated at random instead of half-biased
    # by Expected Improvement. It is the ablation control for the Bayesian guidance --
    # same generator, same potential, same relaxation protocol, same stopping
    # rule -- and isolates the contribution of the GP from that of the
    # structure generator.
    #
    # This is NOT a pure random search: the pool of perturbed incumbents is
    # independent of the GP and stays active, so the local refinement of the
    # best structures found so far still happens. What is removed is only the
    # Expected-Improvement bias on where new candidates are generated.
    use_gp:              bool  = True
    # [gp] anisotropy: append longest/shortest cell axis to the GP features and
    # make the GP-biased half of each batch reproduce the suggested
    # anisotropy. Off by default, so every input written before the key
    # existed keeps meaning the 7-feature surrogate it was run with.
    gp_anisotropy:       bool  = False
    gp_kernel:           str   = "matern25"
    gp_n_restarts:       int   = 5
    ei_xi:               float = 0.01
    num_ei_candidates:   int   = 50000
    num_ei_suggestions:  int   = 20

    # --- CHPI OPTIMIZER ---
    use_chpi_optimizer:  bool  = True
    d_min:               float = 2.6
    d_max:               float = 3.5
    d_target:            float = 3.0
    d_search:            float = 5.0
    angle_ch_centroid_min: float = 110.0
    angle_from_normal_max: float = 55.0
    max_chpi_iter:       int   = 1000
    max_no_improve:      int   = 50

    # --- CONTACTS ---
    min_hh_distance:     float = 1.8
    min_heavy_distance:  float = 2.5
    min_hx_distance:     float = 1.8

    # --- SEEDS ---
    use_seeds_folder:    bool  = True
    seeds_folder:        str   = "Seeds"
    num_perturbations_per_seed: int = 30
    max_rot_perturb:     float = 30.0
    max_trans_perturb:   float = 1.5
    frac_mols_perturb:   float = 0.5

    # --- INTEGRITY ---
    check_molecular_integrity: bool  = True
    integrity_max_bond_ratio:  float = 2.0

    # --- PARALLEL EVALUATION ---
    # Candidates within a cycle are independent, so they can be relaxed on
    # several machines at once. See cryal/parallel.py for what is and is not
    # distributed. Off by default: a run that says nothing about this keeps
    # the exact serial behaviour it had before the feature existed.
    use_parallel:         bool = False
    # One entry per machine: user@host:slots, optionally :interpreter.
    parallel_workers:     List[str] = field(default_factory=list)
    # Slots on the server itself. Ignored when the list above names
    # `local` explicitly, which is the more specific statement.
    parallel_local_slots: int  = 1
    parallel_remote_dir:  str  = "~/.cryal_work"
    # The interpreter that can `import cryal` on a worker — on a machine set
    # up for MACE this is usually a virtual environment, not the PATH python3.
    parallel_python:      str  = "python3"
    parallel_ssh_key:     str  = ""
    parallel_ssh_timeout: int  = 30    # seconds to give up on connecting
    # Outer bound on one remote candidate. 0 derives it from the backend's own
    # per-structure timeout, which is what actually stops a runaway relaxation.
    parallel_task_timeout: int = 0
    # Consecutive transport failures before a machine is dropped from the run.
    parallel_max_failures: int = 3
    parallel_keep_remote:  bool = False  # leave the remote work dir in place
    parallel_fetch_on_failure: bool = True   # bring a failed task's files back
    parallel_fetch_step:   bool = False      # bring every task's files back

    # --- Derived (computed after load) ---
    # Why the backend refused this machine, when the run is dispatching every
    # candidate elsewhere and so is allowed not to care. Empty otherwise.
    _local_backend_error: str = ""
    mol_weight:          float = 0.0   # computed from molecule.xyz
    _volume_min_computed: float = 0.0
    _volume_max_computed: float = 0.0

    def volume_min(self) -> float:
        return self.target_volume_min if self.target_volume_min else self._volume_min_computed

    def volume_max(self) -> float:
        return self.target_volume_max if self.target_volume_max else self._volume_max_computed



# ---------------------------------------------------------------------------
# Schema: where each value lives in cryal.toml, and what type it must have
# ---------------------------------------------------------------------------

def _fail(path, value, what):
    raise ValueError(f"{path}: expected {what}, got {value!r}")


def _str(path, v):
    return v if isinstance(v, str) else _fail(path, v, "a string")


def _int(path, v):
    ok = isinstance(v, int) and not isinstance(v, bool)
    return v if ok else _fail(path, v, "an integer")


def _float(path, v):
    ok = isinstance(v, (int, float)) and not isinstance(v, bool)
    return float(v) if ok else _fail(path, v, "a number")


def _bool(path, v):
    return v if isinstance(v, bool) else _fail(path, v, "true or false")


def _list_of(item, what):
    def conv(path, v):
        if not isinstance(v, list):
            _fail(path, v, f"a list of {what}")
        return [item(f"{path}[{i}]", x) for i, x in enumerate(v)]
    return conv


def _table(path, v):
    return dict(v) if isinstance(v, dict) else _fail(path, v, "a table { name = value, ... }")


def _choice(*options):
    def conv(path, v):
        if v not in options:
            _fail(path, v, " or ".join(repr(o) for o in options))
        return v
    return conv


_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


def _log_level(path, v):
    if not (isinstance(v, str) and v.upper() in _LOG_LEVELS):
        _fail(path, v, " or ".join(repr(l) for l in _LOG_LEVELS))
    return v.upper()


def _range(path, v):
    """[low, high] -> (low, high), low <= high."""
    if not (isinstance(v, list) and len(v) == 2):
        _fail(path, v, "[low, high]")
    lo, hi = _float(f"{path}[0]", v[0]), _float(f"{path}[1]", v[1])
    if lo > hi:
        raise ValueError(f"{path}: low {lo} is above high {hi}")
    return lo, hi


def _optfloat(path, v):
    """A number, or false to switch the check off (TOML has no null)."""
    if v is False:
        return None
    if v is True:
        _fail(path, v, "a number, or false to switch the check off")
    return _float(path, v)


def _sg_weights(path, v):
    """{ 14 = 36, 2 = 24, ... } -> ([14, 2, ...], [36.0, 24.0, ...]), in file order."""
    if not isinstance(v, dict) or not v:
        _fail(path, v, "a table { group = weight, ... } with at least one group")
    groups, weights = [], []
    for k, w in v.items():
        if not (k.isdigit() and 1 <= int(k) <= 230):
            raise ValueError(f"{path}: {k!r} is not a space group: valid numbers are 1-230")
        w = _float(f"{path}.{k}", w)
        if w <= 0:
            raise ValueError(f"{path}.{k}: weight must be positive, got {w}")
        groups.append(int(k))
        weights.append(w)
    return groups, weights


_LIST_STR   = _list_of(_str, "strings")
_LIST_FLOAT = _list_of(_float, "numbers")

# section -> key -> (Config field, or a tuple of fields the value fills; converter)
_SCHEMA = {
    "general": {
        "molecule":             ("molecule_file", _str),
        "Z":                    ("Z", _int),
        "random_seed":          ("random_seed", _int),
        "output_dir":           ("output_dir", _str),
        "log_level":            ("log_level", _log_level),
        "resume":               ("resume", _bool),
    },
    "backend": {
        "engine":               ("energy_backend", _str),
        "energy_sanity_max":    ("energy_sanity_max", _optfloat),
    },
    "backend.lammps": {
        "input":                ("lammps_input", _str),
        "command":              ("lammps_command", _str),
        "species":              ("spec_order", _LIST_STR),
    },
    "backend.ase": {
        "calculator":           ("ase_calculator", _str),
        "calculator_kwargs":    ("ase_calculator_kwargs", _table),
    },
    "backend.uma": {
        "model":                ("uma_model", _str),
        "task":                 ("uma_task", _str),
        "device":               ("uma_device", _str),
        "inference_settings":   ("uma_inference_settings", _str),
    },
    # In-process relaxation, shared by the ase and uma engines.
    "relaxation": {
        "optimizer":            ("ase_optimizer", _str),
        "fmax":                 ("ase_fmax", _float),
        "max_steps":            ("ase_max_steps", _int),
        "relax_cell":           ("ase_relax_cell", _bool),
    },
    "active_learning": {
        "cycles":               ("num_cycles", _int),
        "initial_structures":   ("initial_structures", _int),
        "structures_per_cycle": ("structures_per_cycle", _int),
        "pool_size":            ("pool_size", _int),
        "max_no_improve_cycles": ("max_no_improve_cycles", _int),
    },
    "space_groups": {
        "weights":              (("space_groups", "sg_weights"), _sg_weights),
    },
    "volume": {
        "min":                  ("target_volume_min", _float),
        "max":                  ("target_volume_max", _float),
        "density":              (("density_min", "density_max"), _range),
        "expansion_factors":    ("volume_expansion_factors", _LIST_FLOAT),
    },
    "cell": {
        "min_axis":             ("min_cell_axis", _float),
    },
    "monoclinic_beta": {
        "range":                (("beta_min", "beta_max"), _range),
        "distribution":         ("beta_distribution", _choice("uniform", "triangular")),
        "mode":                 ("beta_mode", _float),
    },
    "gp": {
        "enabled":              ("use_gp", _bool),
        "anisotropy":           ("gp_anisotropy", _bool),
        "kernel":               ("gp_kernel", _choice("matern25", "rbf")),
        "n_restarts":           ("gp_n_restarts", _int),
        "ei_xi":                ("ei_xi", _float),
        "ei_candidates":        ("num_ei_candidates", _int),
        "ei_suggestions":       ("num_ei_suggestions", _int),
    },
    "chpi": {
        "enabled":              ("use_chpi_optimizer", _bool),
        "d_range":              (("d_min", "d_max"), _range),
        "d_target":             ("d_target", _float),
        "d_search":             ("d_search", _float),
        "angle_ch_centroid_min": ("angle_ch_centroid_min", _float),
        "angle_from_normal_max": ("angle_from_normal_max", _float),
        "max_iter":             ("max_chpi_iter", _int),
        "max_no_improve":       ("max_no_improve", _int),
    },
    "contacts": {
        "min_hh":               ("min_hh_distance", _float),
        "min_heavy":            ("min_heavy_distance", _float),
        "min_hx":               ("min_hx_distance", _float),
    },
    "seeds": {
        "enabled":              ("use_seeds_folder", _bool),
        "folder":               ("seeds_folder", _str),
        "perturbations_per_seed": ("num_perturbations_per_seed", _int),
        "max_rotation":         ("max_rot_perturb", _float),
        "max_translation":      ("max_trans_perturb", _float),
        "fraction_perturbed":   ("frac_mols_perturb", _float),
    },
    "integrity": {
        "enabled":              ("check_molecular_integrity", _bool),
        "max_bond_ratio":       ("integrity_max_bond_ratio", _float),
    },
    "parallel": {
        "enabled":              ("use_parallel", _bool),
        "workers":              ("parallel_workers", _LIST_STR),
        "local_slots":          ("parallel_local_slots", _int),
        "remote_dir":           ("parallel_remote_dir", _str),
        "python":               ("parallel_python", _str),
        "ssh_key":              ("parallel_ssh_key", _str),
        "ssh_timeout":          ("parallel_ssh_timeout", _int),
        "task_timeout":         ("parallel_task_timeout", _int),
        "max_failures":         ("parallel_max_failures", _int),
        "keep_remote":          ("parallel_keep_remote", _bool),
        "fetch_on_failure":     ("parallel_fetch_on_failure", _bool),
        "fetch_step":           ("parallel_fetch_step", _bool),
    },
}


def _unknown(path: str, section: Optional[str], is_table: bool) -> ValueError:
    """An unknown key or table, with the nearest real one when there is one."""
    name = path.rsplit(".", 1)[-1]
    if is_table:
        what = "table"
        homes = [s for s in _SCHEMA if s.rsplit(".", 1)[-1] == name]
        known = [s.rsplit(".", 1)[-1] for s in _SCHEMA]
    else:
        what = f"key in [{section}]" if section else "key outside any table"
        homes = [s for s, keys in _SCHEMA.items() if name in keys and s != section]
        known = list(_SCHEMA.get(section, {}))
    if homes:
        hint = f"; '{name}' belongs in " + " or ".join(f"[{s}]" for s in homes)
    else:
        close = difflib.get_close_matches(name, known, n=1)
        hint = f"; did you mean '{close[0]}'?" if close else ""
    return ValueError(f"{path}: unknown {what}{hint}")


def _apply(doc: dict, cfg: Config, prefix: str = "") -> None:
    """Walk the parsed document, typing every value into cfg."""
    section = prefix.rstrip(".") or None
    keys = _SCHEMA.get(section, {})
    for name, value in doc.items():
        path = prefix + name
        if isinstance(value, dict) and path in _SCHEMA:
            _apply(value, cfg, path + ".")
        elif name in keys:
            fields, conv = keys[name]
            converted = conv(path, value)
            if isinstance(fields, tuple):
                for f, v in zip(fields, converted):
                    setattr(cfg, f, v)
            else:
                setattr(cfg, fields, converted)
        elif isinstance(value, dict) and any(s.startswith(path + ".") for s in _SCHEMA):
            _apply(value, cfg, path + ".")
        else:
            raise _unknown(path, section, isinstance(value, dict))


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def load_config(filepath: str = "cryal.toml") -> Config:
    """
    Load and validate configuration from cryal.toml.
    Without [volume] min/max, the volume range comes from the density range.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"input file not found: {filepath}")
    with open(filepath, "rb") as f:
        try:
            doc = tomllib.load(f)
        except tomllib.TOMLDecodeError as exc:
            from .legacy import looks_legacy
            if looks_legacy(filepath):
                raise ValueError(
                    f"{filepath} is in the old pre-TOML format (% SECTION, "
                    f"key = value); convert it with: cryal convert {filepath} "
                    "cryal.toml") from exc
            raise ValueError(f"{filepath}: {exc}") from exc

    cfg = Config()
    _apply(doc, cfg)

    if cfg.beta_distribution == "triangular" and cfg.beta_mode is None:
        warnings.warn("monoclinic_beta.distribution = triangular requires "
                      "monoclinic_beta.mode; falling back to uniform")
        cfg.beta_distribution = "uniform"

    # Fail on a malformed worker entry now, not when the first cycle tries to
    # reach a machine whose name was a typo.
    if cfg.use_parallel:
        from .parallel import parse_workers
        parse_workers(cfg.parallel_workers, cfg.parallel_python)

    # --- Validate required files ---
    if not os.path.exists(cfg.molecule_file):
        raise FileNotFoundError(f"general.molecule not found: {cfg.molecule_file}")

    # Everything else the run needs depends on which engine will run it: the
    # LAMMPS input script is mandatory for lammps_mace and meaningless to the
    # ASE backend. Ask the selected backend, and fail here rather than after a
    # hundred structures have been generated. Imported locally to keep the
    # config module free of any dependency on the backends package.
    from .backends import backend_class
    from .parallel import runs_entirely_on_workers
    try:
        backend_class(cfg.energy_backend).validate_config(cfg)
    except Exception as exc:
        # A run that relaxes everything on its workers never calls the backend
        # here, so requiring it here would mean installing a second potential
        # beside the one this machine already has, for nothing. Nothing is lost
        # by deferring: cryal.parallel runs this very same validate_config() on
        # every worker during preflight, so a wrong task or a missing model is
        # still caught in the first seconds — just by the machines that will
        # actually feel it. Recorded rather than discarded, because evaluation
        # can still fall back here, and build_pool refuses to let it.
        if not runs_entirely_on_workers(cfg):
            raise
        cfg._local_backend_error = str(exc)

    # --- Compute molecular weight and auto volume range ---
    cfg.mol_weight = _compute_mol_weight(cfg.molecule_file)
    NA = 6.02214076e23
    cfg._volume_min_computed = (cfg.Z * cfg.mol_weight / NA) / (cfg.density_max * 1e-24)
    cfg._volume_max_computed = (cfg.Z * cfg.mol_weight / NA) / (cfg.density_min * 1e-24)

    return cfg



def _compute_mol_weight(xyz_file: str) -> float:
    """Compute molecular weight in g/mol from an XYZ file."""
    masses = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
              'F': 18.998, 'S': 32.06,  'P': 30.974, 'Cl': 35.45,
              'Br': 79.904, 'I': 126.904}
    mw = 0.0
    with open(xyz_file) as f:
        lines = f.readlines()
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            sym = parts[0].capitalize()
            mw += masses.get(sym, 12.0)
    return mw


def _backend_line(cfg: Config) -> str:
    """One-line description of the selected backend, for the run summary."""
    try:
        from .backends import backend_class
        return backend_class(cfg.energy_backend)(cfg).describe()
    except Exception:
        return cfg.energy_backend


def _parallel_line(cfg: Config) -> str:
    """One line on where this run's candidates will be relaxed."""
    if not cfg.use_parallel:
        return "OFF (serial, this machine)"
    machines = list(cfg.parallel_workers) or []
    where = " ".join(machines) if machines else "this machine only"
    return f"ON — local slots {cfg.parallel_local_slots}; workers: {where}"


def print_config_summary(cfg: Config, logger=None):
    """Print a summary of the active configuration."""
    lines = [
        "=" * 60,
        "CrYAL Configuration Summary",
        "=" * 60,
        f"  Molecule:        {cfg.molecule_file}  (MW={cfg.mol_weight:.2f} g/mol)",
        f"  Z:               {cfg.Z}",
        f"  Energy backend:  {_backend_line(cfg)}",
        f"  Output dir:      {cfg.output_dir}",
        f"  Random seed:     {cfg.random_seed}",
        f"  Resume:          {'ON' if cfg.resume else 'OFF'}",
        "",
        f"  AL cycles:       {cfg.num_cycles}",
        f"  Initial structs: {cfg.initial_structures}",
        f"  Per-cycle structs:{cfg.structures_per_cycle}",
        f"  Pool size:       {cfg.pool_size}",
        "",
        f"  Volume range:    [{cfg.volume_min():.0f}, {cfg.volume_max():.0f}] Å³",
        f"  (density prior:  [{cfg.density_min}, {cfg.density_max}] g/cm³)",
        f"  Beta dist:       {cfg.beta_distribution}  [{cfg.beta_min}°, {cfg.beta_max}°]",
        "",
        f"  Space groups:    {cfg.space_groups}",
        f"  SG weights:      {cfg.sg_weights}",
        "",
        f"  GP surrogate:    {'ON' if cfg.use_gp else 'OFF — no EI bias (ablation control)'}"
        + (" + anisotropy feature" if cfg.use_gp and cfg.gp_anisotropy else ""),
        f"  CH-pi optimizer: {'ON' if cfg.use_chpi_optimizer else 'OFF'}",
        f"  Seeds folder:    {'ON — ' + cfg.seeds_folder if cfg.use_seeds_folder else 'OFF'}",
        f"  Integrity check: {'ON' if cfg.check_molecular_integrity else 'OFF'}",
        f"  Parallel eval:   {_parallel_line(cfg)}",
        "=" * 60,
    ]
    if logger:
        for l in lines:
            logger.info(l)
    else:
        for l in lines:
            print(l)
