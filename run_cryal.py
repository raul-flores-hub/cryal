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
run_cryal.py — CrYAL entry point.

Usage:
    python run_cryal.py              # reads INPUT.txt in current directory
    python run_cryal.py my_INPUT.txt # reads a custom input file
    python run_cryal.py --resume     # continue an interrupted run (e.g. after
                                     # a power outage) from its output_dir

CrYAL performs blind crystal structure prediction via active learning:
  1. Random structure generation (no prior knowledge of experimental structure)
  2. LAMMPS/MACE-OFF23 energy evaluation
  3. Gaussian Process surrogate model
  4. Expected Improvement guides generation toward promising regions
  5. Iterates until convergence

See INPUT.txt for all configurable parameters.
"""

import sys
import os
import traceback


def main():
    # --resume can be passed anywhere on the command line; it forces
    # continuation of an existing run regardless of the INPUT.txt setting.
    args = [a for a in sys.argv[1:] if a not in ("--resume", "-r")]
    resume_flag = any(a in ("--resume", "-r") for a in sys.argv[1:])
    input_file = args[0] if args else "INPUT.txt"

    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print("Usage: python run_cryal.py [INPUT.txt] [--resume]")
        sys.exit(1)

    print(f"CrYAL — Crystal Structure Prediction via Active Learning")
    print(f"Reading configuration from: {input_file}")
    print()

    # Load configuration
    try:
        from cryal.config import load_config
        cfg = load_config(input_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    if resume_flag:
        cfg.resume = True
        print("Resume requested (--resume): continuing from existing output_dir\n")

    # Set random seeds
    import random, numpy as np
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    # Run active learning
    try:
        from cryal.active_learning import run
        run(cfg)
    except KeyboardInterrupt:
        print("\nRun interrupted by user.")
        sys.exit(0)
    except Exception:
        print("\nFatal error during active learning run:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
