#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Resolve ``PYTHON_CMD`` from ``--os-profile`` (same rules as workflow prerequisites).

Mapping (must match ``test_native_linux_packages_install.yml`` prerequisites):

- ``ubuntu*`` / ``debian*`` -> ``python3.12``
- ``sles*`` -> ``python3.13``
- anything else (e.g. ``rhel10``) -> ``python3.12``

``--output-format`` matches ``get_s3_config.py``: ``env``, ``json``, ``github``.
Default ``github`` prints ``PYTHON_CMD=value`` for appending to ``GITHUB_ENV``.

Sample usage
------------

GitHub Actions (default format; sets ``PYTHON_CMD`` for later steps)::

    python3 build_tools/packaging/linux/set_python_cmd.py \\
        --os-profile ubuntu2404 >> \"$GITHUB_ENV\"

Explicit ``github`` format (same as default)::

    python3 build_tools/packaging/linux/set_python_cmd.py \\
        --os-profile sles16 --output-format github >> \"$GITHUB_ENV\"

Shell ``export`` lines (e.g. ``eval`` or copy-paste)::

    python3 build_tools/packaging/linux/set_python_cmd.py \\
        --os-profile rhel10 --output-format env
    # prints: export PYTHON_CMD=python3.12

JSON (automation / debugging)::

    python3 build_tools/packaging/linux/set_python_cmd.py \\
        --os-profile ubuntu2404 --output-format json
    # prints: {\"python_cmd\": \"python3.12\"}

Show built-in help::

    python3 build_tools/packaging/linux/set_python_cmd.py --help

Style follows ``docs/development/style_guides/python_style_guide.md`` (TheROCK).
"""

import argparse
import json
import sys
from typing import Literal

# Same vocabulary as build_tools/packaging/linux/get_s3_config.py --output-format.
OutputFormat = Literal["env", "json", "github"]


def resolve_python_cmd(os_profile: str) -> str:
    """Return the interpreter executable name used for pip and tests on this OS profile."""
    # Deb images install python3.12 via apt (see workflow prerequisites).
    if os_profile.startswith(("ubuntu", "debian")):
        return "python3.12"
    # SLES BCI installs python313 (python3.13).
    if os_profile.startswith("sles"):
        return "python3.13"
    # RHEL UBI and other rpm profiles use python3.12 from dnf.
    return "python3.12"


def emit_output(cmd: str, output_format: OutputFormat) -> None:
    """Write one line (or JSON object) to stdout; caller redirects to GITHUB_ENV if needed."""
    if output_format == "json":
        print(json.dumps({"python_cmd": cmd}))
    elif output_format == "github":
        # Same style as get_s3_config.py --output-format github (NAME=value).
        print(f"PYTHON_CMD={cmd}")
    else:
        # --output-format env: shell-exportable assignment.
        print(f"export PYTHON_CMD={cmd}")


def main(argv: list[str]) -> int:
    """CLI entry; accepts ``argv`` without ``sys.argv[0]`` for tests and composition."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--os-profile",
        required=True,
        help="e.g. ubuntu2404, rhel10, sles16",
    )
    parser.add_argument(
        "--output-format",
        choices=["env", "json", "github"],
        default="github",
        help="Like get_s3_config.py: env, json, or github (default: github for GITHUB_ENV)",
    )
    args = parser.parse_args(argv)

    cmd = resolve_python_cmd(args.os_profile)
    emit_output(cmd, args.output_format)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
