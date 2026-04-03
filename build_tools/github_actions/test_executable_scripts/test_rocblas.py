# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

# Importing is_asan from github_actions_api.py
sys.path.append(str(THEROCK_DIR / "build_tools" / "github_actions"))
from github_actions_api import get_visible_gpu_count, is_asan

def main():
    parser = argparse.ArgumentParser(description="Run rocblas tests")
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Run multi-GPU tests only (requires 2+ GPUs)",
    )
    args = parser.parse_args()

    # GTest sharding
    SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
    TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
    environ_vars = os.environ.copy()
    # For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
    environ_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
    environ_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

    if is_asan():
        environ_vars["HSA_XNACK"] = "1"

    test_type = os.getenv("TEST_TYPE", "full")

    if args.multi_gpu:
        if test_type == "quick":
            logging.info("Skipping rocblas multi-GPU tests: quick test mode")
            return 0

        # Verify we have multiple GPUs available
        gpu_count = get_visible_gpu_count(
            env=environ_vars, therock_bin_dir=THEROCK_BIN_DIR
        )
        logging.info(f"Visible GPU count: {gpu_count}")

        if gpu_count < 2:
            logging.warning("Skipping rocblas multi-GPU tests: <2 GPUs visible")
            return 0

        test_filter = ["--gtest_filter=*multi_gpu*"]
    else:
        # If quick tests are enabled, we run quick tests only.
        # Otherwise, we run the normal test suite
        if test_type == "quick":
            test_filter = ["--yaml", f"{THEROCK_BIN_DIR}/rocblas_smoke.yaml"]
        else:
            # only running quick tests due to openBLAS issue: https://github.com/ROCm/TheRock/issues/1605
            test_filter = ["--yaml", f"{THEROCK_BIN_DIR}/rocblas_smoke.yaml"]

    cmd = [f"{THEROCK_BIN_DIR}/rocblas-test"] + test_filter
    logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

    result = subprocess.run(
        cmd,
        cwd=THEROCK_DIR,
        check=False,
        env=environ_vars,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
