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
config.py — INPUT.txt parser and configuration container.

Parses the USPEX-like INPUT.txt format:
  % SECTION_NAME
  KEY = VALUE   # optional inline comment

Provides a Config dataclass with typed fields and sensible defaults.
Unknown keys produce a warning; missing required keys raise ValueError.
"""

import os
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_input(filepath: str) -> dict:
    """
    Parse INPUT.txt into a flat dict of {key: raw_string_value}.
    Section names are dropped; keys are case-insensitive (stored lowercase).
    """
    raw = {}
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"INPUT file not found: {filepath}")

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%'):
                continue
            # Remove inline comment
            if '#' in line:
                line = line[:line.index('#')].strip()
            if '=' not in line:
                continue
            key, _, value = line.partition('=')
            raw[key.strip().lower()] = value.strip()
    return raw


def _get(raw, key, default=None):
    return raw.get(key.lower(), default)


def _int(raw, key, default):
    v = _get(raw, key)
    return int(v) if v is not None else default


def _float(raw, key, default):
    v = _get(raw, key)
    return float(v) if v is not None else default


def _bool(raw, key, default):
    v = _get(raw, key)
    if v is None:
        return default
    return v.lower() in ('true', '1', 'yes')


def _str(raw, key, default):
    v = _get(raw, key)
    return v if v is not None else default


def _floatlist(raw, key, default):
    v = _get(raw, key)
    if v is None:
        return default
    return [float(x) for x in v.split()]


def _intlist(raw, key, default):
    v = _get(raw, key)
    if v is None:
        return default
    return [int(x) for x in v.split()]


def _strlist(raw, key, default):
    v = _get(raw, key)
    if v is None:
        return default
    return v.split()


def _optfloat(raw, key, default):
    """A float, or None when the value spells a disabled check."""
    v = _get(raw, key)
    if v is None:
        return default
    if v.strip().lower() in ('none', 'off', 'disabled'):
        return None
    return float(v)


def _kwargs(raw, key, default):
    """
    Parse 'name=value name2=value2' into a dict, with values typed by
    ast.literal_eval when they look like Python literals and left as strings
    otherwise ('cuda' stays 'cuda', '5.0' becomes 5.0, 'True' becomes True).
    """
    v = _get(raw, key)
    if v is None:
        return default
    import ast
    out = {}
    for token in v.split():
        if '=' not in token:
            warnings.warn(f"{key}: ignoring '{token}' (expected name=value)")
            continue
        name, _, value = token.partition('=')
        try:
            out[name.strip()] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[name.strip()] = value
    return out


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
    # 'none' switches the check off.
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
    # useGP = false runs the identical workflow with the surrogate switched off:
    # candidates are generated at random instead of half-biased by Expected
    # Improvement. It is the ablation control for the Bayesian guidance --
    # same generator, same potential, same relaxation protocol, same stopping
    # rule -- and isolates the contribution of the GP from that of the
    # structure generator.
    #
    # This is NOT a pure random search: the pool of perturbed incumbents is
    # independent of the GP and stays active, so the local refinement of the
    # best structures found so far still happens. What is removed is only the
    # Expected-Improvement bias on where new candidates are generated.
    use_gp:              bool  = True
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
    mol_weight:          float = 0.0   # computed from molecule.xyz
    _volume_min_computed: float = 0.0
    _volume_max_computed: float = 0.0

    def volume_min(self) -> float:
        return self.target_volume_min if self.target_volume_min else self._volume_min_computed

    def volume_max(self) -> float:
        return self.target_volume_max if self.target_volume_max else self._volume_max_computed


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def load_config(filepath: str = "INPUT.txt") -> Config:
    """
    Load and validate configuration from INPUT.txt.
    Auto-computes volume range from density range if 'auto' is specified.
    """
    raw = parse_input(filepath)
    cfg = Config()

    # GENERAL
    cfg.molecule_file     = _str(raw, "moleculefile", cfg.molecule_file)
    cfg.Z                 = _int(raw, "z", cfg.Z)
    cfg.random_seed       = _int(raw, "randomseed", cfg.random_seed)
    cfg.output_dir        = _str(raw, "outputdir", cfg.output_dir)
    cfg.log_level         = _str(raw, "loglevel", cfg.log_level)
    cfg.resume            = _bool(raw, "resume", cfg.resume)

    # ENERGY BACKEND
    cfg.energy_backend    = _str(raw, "energybackend", cfg.energy_backend)
    cfg.energy_sanity_max = _optfloat(raw, "energysanitymax", cfg.energy_sanity_max)

    # BACKEND: lammps_mace
    cfg.lammps_input      = _str(raw, "lammpsinput", cfg.lammps_input)
    cfg.lammps_command    = _str(raw, "lammpscommand", cfg.lammps_command)
    cfg.spec_order        = _strlist(raw, "specorder", cfg.spec_order)

    # BACKEND: ase
    cfg.ase_calculator        = _str(raw, "asecalculator", cfg.ase_calculator)
    cfg.ase_calculator_kwargs = _kwargs(raw, "asecalculatorkwargs",
                                        cfg.ase_calculator_kwargs)
    cfg.ase_optimizer         = _str(raw, "aseoptimizer", cfg.ase_optimizer)
    cfg.ase_fmax              = _float(raw, "asefmax", cfg.ase_fmax)
    cfg.ase_max_steps         = _int(raw, "asemaxsteps", cfg.ase_max_steps)
    cfg.ase_relax_cell        = _bool(raw, "aserelaxcell", cfg.ase_relax_cell)

    # BACKEND: uma
    cfg.uma_model   = _str(raw, "umamodel", cfg.uma_model)
    cfg.uma_task    = _str(raw, "umatask", cfg.uma_task)
    cfg.uma_device  = _str(raw, "umadevice", cfg.uma_device)
    cfg.uma_inference_settings = _str(raw, "umainferencesettings",
                                      cfg.uma_inference_settings)

    # ACTIVE LEARNING
    cfg.num_cycles            = _int(raw, "numcycles", cfg.num_cycles)
    cfg.initial_structures    = _int(raw, "initialstructures", cfg.initial_structures)
    cfg.structures_per_cycle  = _int(raw, "structurespercycle", cfg.structures_per_cycle)
    cfg.pool_size             = _int(raw, "poolsize", cfg.pool_size)
    cfg.max_no_improve_cycles = _int(raw, "maxnoimprovecycles", cfg.max_no_improve_cycles)

    # SPACE GROUPS
    cfg.space_groups = _intlist(raw, "spacegroups", cfg.space_groups)
    cfg.sg_weights   = _floatlist(raw, "sgweights", cfg.sg_weights)
    if len(cfg.sg_weights) != len(cfg.space_groups):
        raise ValueError("spaceGroups and sgWeights must have the same length")

    # VOLUME
    vol_min_raw = _str(raw, "targetvolumemin", "auto")
    vol_max_raw = _str(raw, "targetvolumemax", "auto")
    cfg.target_volume_min = None if vol_min_raw.lower() == "auto" else float(vol_min_raw)
    cfg.target_volume_max = None if vol_max_raw.lower() == "auto" else float(vol_max_raw)
    cfg.density_min = _float(raw, "densitymin", cfg.density_min)
    cfg.density_max = _float(raw, "densitymax", cfg.density_max)
    cfg.volume_expansion_factors = _floatlist(
        raw, "volumeexpansionfactors", cfg.volume_expansion_factors)

    # CELL
    cfg.min_cell_axis = _float(raw, "mincellaxis", cfg.min_cell_axis)

    # BETA
    cfg.beta_min          = _float(raw, "betamin", cfg.beta_min)
    cfg.beta_max          = _float(raw, "betamax", cfg.beta_max)
    cfg.beta_distribution = _str(raw, "betadistribution", cfg.beta_distribution)
    beta_mode_raw = _str(raw, "betamode", None)
    cfg.beta_mode = float(beta_mode_raw) if beta_mode_raw else None

    if cfg.beta_distribution == "triangular" and cfg.beta_mode is None:
        warnings.warn("betaDistribution=triangular requires betaMode; falling back to uniform")
        cfg.beta_distribution = "uniform"

    # GP
    cfg.use_gp            = _bool(raw, "usegp", cfg.use_gp)
    cfg.gp_kernel         = _str(raw, "gpkernel", cfg.gp_kernel)
    cfg.gp_n_restarts     = _int(raw, "gpnrestarts", cfg.gp_n_restarts)
    cfg.ei_xi             = _float(raw, "eixi", cfg.ei_xi)
    cfg.num_ei_candidates = _int(raw, "numeicandidates", cfg.num_ei_candidates)
    cfg.num_ei_suggestions= _int(raw, "numeisuggestions", cfg.num_ei_suggestions)

    # CHPI
    cfg.use_chpi_optimizer    = _bool(raw, "usechpioptimizer", cfg.use_chpi_optimizer)
    cfg.d_min                 = _float(raw, "dmin", cfg.d_min)
    cfg.d_max                 = _float(raw, "dmax", cfg.d_max)
    cfg.d_target              = _float(raw, "dtarget", cfg.d_target)
    cfg.d_search              = _float(raw, "dsearch", cfg.d_search)
    cfg.angle_ch_centroid_min = _float(raw, "anglechcentroidmin", cfg.angle_ch_centroid_min)
    cfg.angle_from_normal_max = _float(raw, "anglefromnormalmax", cfg.angle_from_normal_max)
    cfg.max_chpi_iter         = _int(raw, "maxchpiiter", cfg.max_chpi_iter)
    cfg.max_no_improve        = _int(raw, "maxnoimprove", cfg.max_no_improve)

    # CONTACTS
    cfg.min_hh_distance    = _float(raw, "minhhdistance", cfg.min_hh_distance)
    cfg.min_heavy_distance = _float(raw, "minheavydistance", cfg.min_heavy_distance)
    cfg.min_hx_distance    = _float(raw, "minhxdistance", cfg.min_hx_distance)

    # SEEDS
    cfg.use_seeds_folder          = _bool(raw, "useseedsfolder", cfg.use_seeds_folder)
    cfg.seeds_folder              = _str(raw, "seedsfolder", cfg.seeds_folder)
    cfg.num_perturbations_per_seed= _int(raw, "numperturbationsperseed",
                                         cfg.num_perturbations_per_seed)
    cfg.max_rot_perturb    = _float(raw, "maxrotperturb", cfg.max_rot_perturb)
    cfg.max_trans_perturb  = _float(raw, "maxtransperturb", cfg.max_trans_perturb)
    cfg.frac_mols_perturb  = _float(raw, "fracmolsperturb", cfg.frac_mols_perturb)

    # INTEGRITY
    cfg.check_molecular_integrity = _bool(raw, "checkmolecularintegrity",
                                          cfg.check_molecular_integrity)
    cfg.integrity_max_bond_ratio  = _float(raw, "integritymaxbondratio",
                                           cfg.integrity_max_bond_ratio)

    # PARALLEL
    cfg.use_parallel          = _bool(raw, "useparallel", cfg.use_parallel)
    cfg.parallel_workers      = _strlist(raw, "parallelworkers", cfg.parallel_workers)
    cfg.parallel_local_slots  = _int(raw, "parallellocalslots", cfg.parallel_local_slots)
    cfg.parallel_remote_dir   = _str(raw, "parallelremotedir", cfg.parallel_remote_dir)
    cfg.parallel_python       = _str(raw, "parallelpython", cfg.parallel_python)
    cfg.parallel_ssh_key      = _str(raw, "parallelsshkey", cfg.parallel_ssh_key)
    cfg.parallel_ssh_timeout  = _int(raw, "parallelsshtimeout", cfg.parallel_ssh_timeout)
    cfg.parallel_task_timeout = _int(raw, "paralleltasktimeout", cfg.parallel_task_timeout)
    cfg.parallel_max_failures = _int(raw, "parallelmaxfailures", cfg.parallel_max_failures)
    cfg.parallel_keep_remote  = _bool(raw, "parallelkeepremote", cfg.parallel_keep_remote)
    cfg.parallel_fetch_on_failure = _bool(raw, "parallelfetchonfailure",
                                          cfg.parallel_fetch_on_failure)
    cfg.parallel_fetch_step   = _bool(raw, "parallelfetchstep", cfg.parallel_fetch_step)

    # Fail on a malformed worker entry now, not when the first cycle tries to
    # reach a machine whose name was a typo.
    if cfg.use_parallel:
        from .parallel import parse_workers
        parse_workers(cfg.parallel_workers, cfg.parallel_python)

    # --- Validate required files ---
    if not os.path.exists(cfg.molecule_file):
        raise FileNotFoundError(f"moleculeFile not found: {cfg.molecule_file}")

    # Everything else the run needs depends on which engine will run it: the
    # LAMMPS input script is mandatory for lammps_mace and meaningless to the
    # ASE backend. Ask the selected backend, and fail here rather than after a
    # hundred structures have been generated. Imported locally to keep the
    # config module free of any dependency on the backends package.
    from .backends import backend_class
    backend_class(cfg.energy_backend).validate_config(cfg)

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
        f"  GP surrogate:    {'ON' if cfg.use_gp else 'OFF — no EI bias (ablation control)'}",
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
