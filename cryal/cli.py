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
cli.py — the command line, as `cryal` and as `python run_cryal.py`.

    cryal                    # reads INPUT.txt in the current directory
    cryal my_INPUT.txt       # reads a custom input file
    cryal --resume           # continue an interrupted run from its output_dir

This module holds what `run_cryal.py` used to hold, so that installing the
package provides a `cryal` command. The script at the top of the repository
stays as a thin wrapper: it is the invocation the article and the README
document, and it must keep working from a plain checkout.
"""

import os
import sys
import traceback

def usage() -> str:
    """The usage text, named after however this was invoked."""
    prog = os.path.basename(sys.argv[0]) or "cryal"
    if prog.endswith(".py"):
        prog = f"python {prog}"
    return f"""CrYAL — Crystal Structure Prediction via Active Learning

usage: {prog} [INPUT.txt] [--resume] [--version]

  INPUT.txt     configuration file (default: INPUT.txt in the current directory)
  --resume, -r  continue an interrupted run from its outputDir
  --version     print the version and exit
  --help, -h    print this message and exit
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    if any(a in ("--help", "-h") for a in argv):
        print(usage())
        return 0

    if "--version" in argv:
        from . import __version__
        print(f"CrYAL {__version__}")
        return 0

    # --resume can be passed anywhere on the command line; it forces
    # continuation of an existing run regardless of the INPUT.txt setting.
    args = [a for a in argv if a not in ("--resume", "-r")]
    resume_flag = any(a in ("--resume", "-r") for a in argv)
    input_file = args[0] if args else "INPUT.txt"

    if not os.path.exists(input_file):
        print(f"ERROR: Input file not found: {input_file}")
        print(usage())
        sys.exit(1)

    print(f"CrYAL — Crystal Structure Prediction via Active Learning")
    print(f"Reading configuration from: {input_file}")
    print()

    # Load configuration
    try:
        from .config import load_config
        cfg = load_config(input_file)
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    if resume_flag:
        cfg.resume = True
        print("Resume requested (--resume): continuing from existing output_dir\n")

    # Set random seeds
    import random
    import numpy as np
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    # Run active learning
    try:
        from .active_learning import run
        run(cfg)
    except KeyboardInterrupt:
        print("\nRun interrupted by user.")
        sys.exit(0)
    except Exception:
        print("\nFatal error during active learning run:")
        traceback.print_exc()
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
