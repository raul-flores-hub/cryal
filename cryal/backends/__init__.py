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
backends — the energy engines CrYAL can drive.

`energyBackend` in INPUT.txt selects one by name. Omitting the key selects
`lammps_mace`, which is what every input written before this package existed
means — including the one archived with v1.0.0 and used for the published run.

Adding a backend is: subclass EnergyBackend, implement relax(), decorate the
class with @register, and import the module here.
"""

from .base import EnergyBackend, get_reference_bonds, check_molecular_integrity

_BACKENDS = {}


def register(cls):
    """Class decorator: make a backend selectable by name (and by alias)."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a name")
    for key in (cls.name,) + tuple(cls.aliases):
        _BACKENDS[key.lower()] = cls
    return cls


def available_backends():
    """Selectable names, aliases included, in a stable order."""
    return sorted(_BACKENDS)


def backend_class(name: str):
    """Look up a backend class by name. Raises ValueError if unknown."""
    try:
        return _BACKENDS[str(name).strip().lower()]
    except KeyError:
        raise ValueError(
            f"unknown energyBackend '{name}' — available: "
            f"{', '.join(available_backends())}") from None


def get_backend(cfg, logger=None) -> EnergyBackend:
    """Instantiate the backend named by cfg.energy_backend."""
    return backend_class(getattr(cfg, "energy_backend", "lammps_mace"))(cfg, logger)


# Importing the modules is what registers them.
from . import lammps_mace   # noqa: E402,F401
from . import ase_calculator  # noqa: E402,F401

__all__ = ["EnergyBackend", "get_reference_bonds", "check_molecular_integrity",
           "register", "available_backends", "backend_class", "get_backend"]
