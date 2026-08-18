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

"""Make an older MACE ML-IAP model file load under a newer mace-torch.

The problem
-----------
The exported `*-mliap_lammps.pt` file is a *pickled object*, not a state dict.
When it is loaded, the attributes come from the pickle but the class code comes
from the installed `mace` package. mace-torch 0.3.16 added two buffers to
`MACEEdgeForcesWrapper.__init__`, `total_charge` and `total_spin`, and its
`forward()` passes them to the model. A file exported before that change does
not carry them, so every force evaluation dies with:

    AttributeError: 'MACEEdgeForcesWrapper' object has no attribute 'total_charge'
    ERROR: Running mliappy unified compute_forces failure.

Reinstalling mace-torch does not help: the installed package is intact, the
mismatch is between the old pickle and the new code.

Read the real error in `AL_results/steps/<structure>/lammps.output`. The
`log.lammps` only shows the generic ML-IAP failure and points nowhere.

The fix
-------
Register the two missing buffers with the defaults the current code uses and
write a *new* file. The weights are untouched, so the patched model is
numerically the same one that produced the published results.

    python tools/patch_mliap_model.py MODEL.pt

writes `MODEL_v2.pt` next to it. The original is never modified.

Defaults are `total_charge = 0.0` and `total_spin = 1.0`: a neutral,
closed-shell system. For a charged or open-shell target, pass --charge/--spin.

Validate before trusting it
---------------------------
Do not assume a patched model reproduces earlier numbers -- verify it. Run a
deterministic minimization (`box/relax` plus `minimize`, no MD) on a structure
whose energy you already know and compare. For this project that check agreed
to 0.88 meV per unit cell, about 0.021 kJ/mol per molecule, and 0.008% in
volume: numerical noise of the minimizer path, not a change of model.
"""

import argparse
import os
import sys

BUFFERS = ("total_charge", "total_spin")


def missing_buffers(model, names=BUFFERS):
    """Return the names in `names` that `model` does not carry."""
    return [n for n in names if not hasattr(model, n)]


def apply_patch(model, charge=0.0, spin=1.0, torch=None):
    """Register the missing buffers on `model`. Returns the names added."""
    if torch is None:
        import torch
    values = {"total_charge": charge, "total_spin": spin}
    added = missing_buffers(model)
    for name in added:
        model.register_buffer(name, torch.tensor([values[name]]))
    return added


def default_output(path):
    root, ext = os.path.splitext(path)
    return f"{root}_v2{ext}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", help="the exported *-mliap_lammps.pt file")
    p.add_argument("-o", "--output", help="output path (default: <model>_v2.pt)")
    p.add_argument("--charge", type=float, default=0.0,
                   help="total charge, default 0.0 (neutral)")
    p.add_argument("--spin", type=float, default=1.0,
                   help="total spin multiplicity, default 1.0 (closed shell)")
    args = p.parse_args(argv)

    import torch

    out = args.output or default_output(args.model)
    if os.path.exists(out):
        sys.exit(f"refusing to overwrite {out}")

    obj = torch.load(args.model, map_location="cpu", weights_only=False)
    model = getattr(obj, "model", obj)

    added = apply_patch(model, args.charge, args.spin, torch=torch)
    if not added:
        print("nothing to do: the model already carries", ", ".join(BUFFERS))
        return 0

    torch.save(obj, out)
    print(f"added {', '.join(added)} -> {out}")
    print("the original file is unchanged")
    print("\nNow validate it: run a deterministic minimization on a structure "
          "whose energy you know and compare before trusting the patched model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
