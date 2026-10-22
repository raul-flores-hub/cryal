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
legacy.py — the input format CrYAL used before cryal.toml, kept to convert.

The old files (INPUT.txt, then cryal.in) looked like

  % SECTION_NAME
  keyName = value   # inline comment

with the section headers ignored, keys case-insensitive, every value a
string typed by whoever read it, and unknown keys skipped without a word.
CrYAL no longer runs from that format, but the runs reported in the article
were made with it, so `cryal convert` turns such a file into the cryal.toml
that means the same run. "The same" is literal: each value is read exactly
as the old loader read it, quirks included -- `useGP = ture` was false, an
unrecognised gpKernel was Matérn -- and written out as the value the run
actually used. Keys the old loader never read are listed at the top of the
output, not dropped.
"""

import ast
import json
import os
import re
import warnings
from typing import Dict, List, Tuple

from .config import Config


def parse_input(filepath: str) -> dict:
    """
    Parse an old-format file into a flat dict of {key: raw_string_value}.
    Section names are dropped; keys are case-insensitive (stored lowercase).
    """
    raw = {}
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"input file not found: {filepath}")

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


def looks_legacy(filepath: str) -> bool:
    """True if the file has an old-format `% SECTION` header line."""
    try:
        with open(filepath) as f:
            return any(line.lstrip().startswith('%') for line in f)
    except OSError:
        return False


# ---------------------------------------------------------------------------
# How the old loader read each value
# ---------------------------------------------------------------------------

def _bool(v):
    return v.lower() in ('true', '1', 'yes')


def _list(cast):
    return lambda v: [cast(x) for x in v.split()]


def _optfloat(v):
    """False (check disabled) for the old 'none'/'off'/'disabled'."""
    return False if v.strip().lower() in ('none', 'off', 'disabled') else float(v)


def _kwargs(v):
    out = {}
    for token in v.split():
        if '=' not in token:
            warnings.warn(f"aseCalculatorKwargs: ignoring '{token}' (expected name=value)")
            continue
        name, _, value = token.partition('=')
        try:
            out[name.strip()] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[name.strip()] = value
    return out


# old lowercase key -> (TOML section, TOML key, how the old loader read it)
_KEYS: Dict[str, Tuple[str, str, object]] = {
    "moleculefile":          ("general", "molecule", str),
    "z":                     ("general", "Z", int),
    "randomseed":            ("general", "random_seed", int),
    "outputdir":             ("general", "output_dir", str),
    "loglevel":              ("general", "log_level", str),
    "resume":                ("general", "resume", _bool),

    "energybackend":         ("backend", "engine", str),
    "energysanitymax":       ("backend", "energy_sanity_max", _optfloat),

    "lammpsinput":           ("backend.lammps", "input", str),
    "lammpscommand":         ("backend.lammps", "command", str),
    "specorder":             ("backend.lammps", "species", _list(str)),

    "asecalculator":         ("backend.ase", "calculator", str),
    "asecalculatorkwargs":   ("backend.ase", "calculator_kwargs", _kwargs),

    "umamodel":              ("backend.uma", "model", str),
    "umatask":               ("backend.uma", "task", str),
    "umadevice":             ("backend.uma", "device", str),
    "umainferencesettings":  ("backend.uma", "inference_settings", str),

    "aseoptimizer":          ("relaxation", "optimizer", str),
    "asefmax":               ("relaxation", "fmax", float),
    "asemaxsteps":           ("relaxation", "max_steps", int),
    "aserelaxcell":          ("relaxation", "relax_cell", _bool),

    "numcycles":             ("active_learning", "cycles", int),
    "initialstructures":     ("active_learning", "initial_structures", int),
    "structurespercycle":    ("active_learning", "structures_per_cycle", int),
    "poolsize":              ("active_learning", "pool_size", int),
    "maxnoimprovecycles":    ("active_learning", "max_no_improve_cycles", int),

    "volumeexpansionfactors": ("volume", "expansion_factors", _list(float)),
    "mincellaxis":           ("cell", "min_axis", float),

    "gpkernel":              ("gp", "kernel", str),
    "usegp":                 ("gp", "enabled", _bool),
    "gpanisotropy":          ("gp", "anisotropy", _bool),
    "gpnrestarts":           ("gp", "n_restarts", int),
    "eixi":                  ("gp", "ei_xi", float),
    "numeicandidates":       ("gp", "ei_candidates", int),
    "numeisuggestions":      ("gp", "ei_suggestions", int),

    "usechpioptimizer":      ("chpi", "enabled", _bool),
    "dtarget":               ("chpi", "d_target", float),
    "dsearch":               ("chpi", "d_search", float),
    "anglechcentroidmin":    ("chpi", "angle_ch_centroid_min", float),
    "anglefromnormalmax":    ("chpi", "angle_from_normal_max", float),
    "maxchpiiter":           ("chpi", "max_iter", int),
    "maxnoimprove":          ("chpi", "max_no_improve", int),

    "minhhdistance":         ("contacts", "min_hh", float),
    "minheavydistance":      ("contacts", "min_heavy", float),
    "minhxdistance":         ("contacts", "min_hx", float),

    "useseedsfolder":        ("seeds", "enabled", _bool),
    "seedsfolder":           ("seeds", "folder", str),
    "numperturbationsperseed": ("seeds", "perturbations_per_seed", int),
    "maxrotperturb":         ("seeds", "max_rotation", float),
    "maxtransperturb":       ("seeds", "max_translation", float),
    "fracmolsperturb":       ("seeds", "fraction_perturbed", float),

    "checkmolecularintegrity": ("integrity", "enabled", _bool),
    "integritymaxbondratio": ("integrity", "max_bond_ratio", float),

    "useparallel":           ("parallel", "enabled", _bool),
    "parallelworkers":       ("parallel", "workers", _list(str)),
    "parallellocalslots":    ("parallel", "local_slots", int),
    "parallelremotedir":     ("parallel", "remote_dir", str),
    "parallelpython":        ("parallel", "python", str),
    "parallelsshkey":        ("parallel", "ssh_key", str),
    "parallelsshtimeout":    ("parallel", "ssh_timeout", int),
    "paralleltasktimeout":   ("parallel", "task_timeout", int),
    "parallelmaxfailures":   ("parallel", "max_failures", int),
    "parallelkeepremote":    ("parallel", "keep_remote", _bool),
    "parallelfetchonfailure": ("parallel", "fetch_on_failure", _bool),
    "parallelfetchstep":     ("parallel", "fetch_step", _bool),
}

# Keys that become one TOML value together, and are handled below.
_COMBINED = {"spacegroups", "sgweights", "targetvolumemin", "targetvolumemax",
             "densitymin", "densitymax", "betamin", "betamax",
             "betadistribution", "betamode", "dmin", "dmax"}

_SECTION_ORDER = ["general", "backend", "backend.lammps", "backend.ase",
                  "backend.uma", "relaxation", "active_learning",
                  "space_groups", "volume", "cell", "monoclinic_beta", "gp",
                  "chpi", "contacts", "seeds", "integrity", "parallel"]


def legacy_to_tables(raw: dict) -> Tuple[Dict[str, Dict[str, object]], List[str], List[str]]:
    """
    Turn an old-format raw dict into {section: {key: value}}, plus the old
    keys that were not read and notes on values the old loader reinterpreted.
    """
    tables: Dict[str, Dict[str, object]] = {}
    notes: List[str] = []
    cfg = Config()

    def put(section, key, value):
        tables.setdefault(section, {})[key] = value

    for old, value in raw.items():
        if old in _KEYS:
            section, key, read = _KEYS[old]
            put(section, key, read(value))

    # logLevel: the logger upper-cased it and fell back to INFO
    level = tables.get("general", {}).get("log_level")
    if level is not None:
        if level.upper() in ("DEBUG", "INFO", "WARNING", "ERROR"):
            put("general", "log_level", level.upper())
        else:
            notes.append(f"logLevel = {level} was read as INFO")
            put("general", "log_level", "INFO")

    # gpKernel: anything but 'rbf' was Matérn
    kernel = tables.get("gp", {}).get("kernel")
    if kernel is not None and kernel not in ("rbf", "matern25"):
        notes.append(f"gpKernel = {kernel} was read as matern25")
        put("gp", "kernel", "matern25")

    # spaceGroups + sgWeights -> one table, group = weight, in file order
    if "spacegroups" in raw or "sgweights" in raw:
        groups = _list(int)(raw["spacegroups"]) if "spacegroups" in raw else cfg.space_groups
        weights = _list(float)(raw["sgweights"]) if "sgweights" in raw else cfg.sg_weights
        if len(groups) != len(weights):
            raise ValueError("spaceGroups and sgWeights must have the same length")
        put("space_groups", "weights", {str(g): w for g, w in zip(groups, weights)})

    # targetVolumeMin/Max: 'auto' (or absent) is no key at all
    for old, key in (("targetvolumemin", "min"), ("targetvolumemax", "max")):
        v = raw.get(old, "auto")
        if v.lower() != "auto":
            put("volume", key, float(v))

    # Min/max pairs -> [min, max]; a missing half keeps the old default
    for lo, hi, section, key, dlo, dhi in (
            ("densitymin", "densitymax", "volume", "density",
             cfg.density_min, cfg.density_max),
            ("betamin", "betamax", "monoclinic_beta", "range",
             cfg.beta_min, cfg.beta_max),
            ("dmin", "dmax", "chpi", "d_range", cfg.d_min, cfg.d_max)):
        if lo in raw or hi in raw:
            put(section, key, [float(raw[lo]) if lo in raw else dlo,
                               float(raw[hi]) if hi in raw else dhi])

    # betaDistribution / betaMode: triangular without a mode was uniform,
    # and anything but 'triangular' sampled uniformly
    dist = raw.get("betadistribution")
    mode = float(raw["betamode"]) if raw.get("betamode") else None
    if mode is not None:
        put("monoclinic_beta", "mode", mode)
    if dist is not None:
        if dist == "triangular" and mode is None:
            notes.append("betaDistribution = triangular without betaMode "
                         "was read as uniform")
            dist = "uniform"
        elif dist not in ("uniform", "triangular"):
            notes.append(f"betaDistribution = {dist} was read as uniform")
            dist = "uniform"
        put("monoclinic_beta", "distribution", dist)

    unread = sorted(k for k in raw if k not in _KEYS and k not in _COMBINED)
    return tables, unread, notes


# ---------------------------------------------------------------------------
# TOML output
# ---------------------------------------------------------------------------

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(k: str) -> str:
    return k if _BARE_KEY.match(k) else json.dumps(k)


def _value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v in (float("inf"), float("-inf")):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{_key(str(k))} = {_value(x)}"
                                for k, x in v.items()) + " }"
    raise ValueError(f"cannot write {v!r} ({type(v).__name__}) as TOML")


def convert(filepath: str) -> str:
    """The cryal.toml text that means the same run as an old-format file."""
    raw = parse_input(filepath)
    tables, unread, notes = legacy_to_tables(raw)

    lines = [f"# Converted by `cryal convert` from {os.path.basename(filepath)}."]
    if notes:
        lines.append("# Values the old loader reinterpreted, written as it read them:")
        lines += [f"#   {n}" for n in notes]
    if unread:
        lines.append("# Keys the old loader never read, so the run never used them:")
        lines += [f"#   {k} = {raw[k]}" for k in unread]

    for section in _SECTION_ORDER:
        if section not in tables:
            continue
        lines += ["", f"[{section}]"]
        for key, value in tables[section].items():
            if (section, key) == ("backend.ase", "calculator_kwargs") and \
                    any(x is None for x in value.values()):
                raise ValueError("aseCalculatorKwargs holds None, which TOML "
                                 "cannot write; leave that argument out")
            lines.append(f"{_key(key)} = {_value(value)}")
    return "\n".join(lines) + "\n"
