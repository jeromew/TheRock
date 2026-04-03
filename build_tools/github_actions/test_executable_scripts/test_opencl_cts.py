import csv
import logging
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

THEROCK_BIN_DIR_STR = os.getenv("THEROCK_BIN_DIR")
if THEROCK_BIN_DIR_STR is None:
    logging.error(
        "++ Error: env(THEROCK_BIN_DIR) is not set. Please set it before executing tests."
    )
    sys.exit(1)

THEROCK_BIN_DIR = Path(THEROCK_BIN_DIR_STR).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
ROCM_PATH = THEROCK_BIN_DIR.parent

CTS_BIN_DIR = ROCM_PATH / "share" / "opencl" / "opencl-cts" / "Release"
QUICK_CSV = CTS_BIN_DIR / "opencl_conformance_tests_quick.csv"
OPENCL_ICD_FILENAMES = ROCM_PATH / "lib" / "opencl" / "libamdocl64.so"

# Device types that apply to GPU runs (non-device-specific entries always run)
_GPU_DEVICE_TYPES = {
    "CL_DEVICE_TYPE_GPU",
    "CL_DEVICE_TYPE_DEFAULT",
    "CL_DEVICE_TYPE_ALL",
}
_ALL_DEVICE_TYPES = _GPU_DEVICE_TYPES | {
    "CL_DEVICE_TYPE_CPU",
    "CL_DEVICE_TYPE_ACCELERATOR",
}

# there are no OpenGL libraries in CI
_SKIPPED_TESTS = set("test_gl")

# Sub-tests to skip within a specific binary. Keys are the test executable
# basename; values are sets of sub-test names as printed by `binary --list`.
# GPU memory access fault during 'constant' sub-test.
_SKIPPED_SUBTESTS: dict[str, set[str]] = {
    "test_basic": {"constant"},
    # Incorrect error codes returned for invalid queue properties / device type.
    # Constant buffer size allocation fails with CL_INVALID_GLOBAL_WORK_SIZE.
    "test_api": {
        "negative_create_command_queue",
        "negative_create_command_queue_with_properties",
        "negative_get_command_queue_info",
        "negative_get_device_ids",
        "min_max_constant_buffer_size",
    },
    # Crash (SIGSEGV) in get_program_info_kernel_names.
    "test_compiler": {"get_program_info_kernel_names"},
    # All read_array_* sub-tests report implausible profiling timestamps
    # (CL_PROFILING_COMMAND_START > CL_PROFILING_COMMAND_END).
    "test_profiling": {
        "read_array_char",
        "read_array_float",
        "read_array_int",
        "read_array_long",
        "read_array_short",
        "read_array_struct",
        "read_array_uchar",
        "read_array_uint",
        "read_array_ulong",
        "read_array_ushort",
    },
    # GPU memory access fault during 'userevents' sub-test.
    "test_events": {"userevents"},
    # printf output mismatches: NaN handling, %% escaping, vector sizes, format.
    "test_printf": {
        "double_limits",
        "float_limits",
        "format_string",
        "half_limits",
        "length_specifier",
        "mixed_format_random",
        "string",
        "vector",
    },
    # islessgreater fp64 fails to execute kernel.
    "test_bruteforce": {"islessgreater"},
    "test_svm": {
        # memory a mismatch at word 512
        "svm_migrate",
        # Unsetting previously set SVM pointers using clSetKernelExecInfo failed
        "svm_set_kernel_exec_info_svm_ptrs",
    },
}

logging.info(f"THEROCK_BIN_DIR: {THEROCK_BIN_DIR}")
logging.info(f"ROCM_PATH: {ROCM_PATH}")
logging.info(f"CTS_BIN_DIR: {CTS_BIN_DIR}")


def build_opencl_env() -> dict:
    """Build environment with OCL_ICD_FILENAMES and LD_LIBRARY_PATH set."""
    env = os.environ.copy()
    env["OCL_ICD_FILENAMES"] = str(OPENCL_ICD_FILENAMES)
    lib_dir = ROCM_PATH / "lib"
    if lib_dir.exists():
        ld_library_path = str(lib_dir)
        if "LD_LIBRARY_PATH" in env:
            ld_library_path = f"{ld_library_path}:{env['LD_LIBRARY_PATH']}"
        env["LD_LIBRARY_PATH"] = ld_library_path
        logging.info(f"Set LD_LIBRARY_PATH to include: {lib_dir}")
    return env


def verify_opencl_runtime():
    """Verify OpenCL runtime is available using clinfo"""
    logging.info("++ Verifying OpenCL runtime availability")

    clinfo_path = ROCM_PATH / "bin" / "clinfo"
    if not clinfo_path.exists():
        logging.warning(f"clinfo not found at {clinfo_path}, skipping verification")
        return

    try:
        cmd = [str(clinfo_path)]
        logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
        result = subprocess.run(
            cmd,
            cwd=THEROCK_DIR,
            capture_output=True,
            text=True,
            timeout=30,
            env=build_opencl_env(),
        )

        if result.returncode == 0:
            logging.info("OpenCL runtime verification successful")
            lines = result.stdout.split("\n")[:10]
            for line in lines:
                if line.strip():
                    logging.info(f"  {line}")
        else:
            logging.warning(f"clinfo returned non-zero exit code: {result.returncode}")
            logging.warning(f"stderr: {result.stderr}")
    except subprocess.TimeoutExpired:
        logging.warning("clinfo verification timed out")
    except Exception as e:
        logging.warning(f"Error running clinfo: {e}")


def parse_quick_csv() -> list[tuple[Path, list[str]]]:
    """Parse opencl_conformance_tests_quick.csv and return (exe_path, args) pairs.

    The CSV format (from run_conformance.py) is:
      name,exe/path [args...]
      device_type, name,exe/path [args...]

    After install all executables are flat in CTS_BIN_DIR, so only the basename
    of the exe path is used.  Device-specific entries for non-GPU device types
    are skipped.
    """
    if not QUICK_CSV.exists():
        logging.error(f"Quick CSV not found at {QUICK_CSV}")
        sys.exit(1)

    if not CTS_BIN_DIR.exists():
        logging.error(
            f"OpenCL-CTS bin directory not found at {CTS_BIN_DIR}. "
            "Please ensure opencl-cts was built and artifacts were created."
        )
        sys.exit(1)

    tests: list[tuple[Path, list[str]]] = []
    with open(QUICK_CSV, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Three-field line: device_type, name, command
            m3 = re.match(r"^\s*(.+?)\s*,\s*(.+?)\s*,\s*(.+?)\s*$", line)
            if m3:
                device_type = m3.group(1)
                if device_type not in _GPU_DEVICE_TYPES:
                    logging.info(
                        f"Skipping device-specific test ({device_type}): {m3.group(2)}"
                    )
                    continue
                command = m3.group(3)
            else:
                # Two-field line: name, command
                m2 = re.match(r"^\s*(.+?)\s*,\s*(.+?)\s*$", line)
                if not m2:
                    continue
                command = m2.group(2)

            # command = "subdir/test_exe [arg1 arg2 ...]"
            cmd_parts = command.split()
            exe_name = Path(cmd_parts[0].replace("/", os.sep)).name
            args = cmd_parts[1:]

            exe_path = CTS_BIN_DIR / exe_name
            tests.append((exe_path, args))

    if not tests:
        logging.error(f"No tests found in {QUICK_CSV}")
        sys.exit(1)

    logging.info(f"Loaded {len(tests)} test entries from {QUICK_CSV}")
    return tests


def get_subtests(exe_path: Path) -> list[str]:
    """Return sub-test names by running `exe --list` (no OpenCL needed)."""
    result = subprocess.run(
        [str(exe_path), "--list"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    subtests = []
    for line in result.stdout.splitlines():
        name = line.strip()
        if name:
            subtests.append(name)
    return subtests


def run_test(test_exe: Path, args: list[str], env: dict) -> bool:
    """Run a single test executable and return True if it passes"""
    test_name = test_exe.name

    if test_name in _SKIPPED_TESTS:
        logging.info(f"SKIPPED: {test_name}")
        return True

    skipped = _SKIPPED_SUBTESTS.get(test_name, set())

    if not test_exe.exists():
        if test_name in _SKIPPED_SUBTESTS:
            logging.info(f"Skipping missing binary: {test_name}")
            return True
        logging.error(f"✗ MISSING: {shlex.join([str(test_exe)] + args)}")
        return False
    if skipped:
        flag_args = [a for a in args if a.startswith("-")]
        subtest_args = [a for a in args if not a.startswith("-")]
        available = subtest_args if subtest_args else get_subtests(test_exe)
        filtered = [t for t in available if t not in skipped]
        logging.info(
            f"Skipping {len(skipped)} sub-test(s) for {test_name}: {sorted(skipped)}"
        )
        if not filtered:
            logging.warning(f"All sub-tests skipped for {test_name}, skipping binary")
            return True
        args = flag_args + filtered

    cmd = [str(test_exe)] + args
    logging.info(f"++ Exec [{test_exe.parent}]$ {shlex.join(cmd)}")

    try:
        print("========================")
        print(f"Running command {cmd}")
        print("========================")
        with subprocess.Popen(
            cmd,
            cwd=str(test_exe.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        ) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
            returncode = proc.wait()

        label = shlex.join([test_name] + args)
        if returncode == 0:
            logging.info(f"✓ PASSED: {label}")
            return True
        else:
            logging.error(f"✗ FAILED: {label} (exit code: {returncode})")
            return False

    except subprocess.TimeoutExpired:
        logging.error(f"✗ TIMEOUT: {test_name} (exceeded 300 seconds)")
        return False
    except Exception as e:
        logging.error(f"✗ ERROR: {test_name} - {e}")
        return False


def run_tests():
    """Run OpenCL CTS tests listed in the quick CSV"""
    logging.info("++ Running OpenCL-CTS tests")

    env = build_opencl_env()
    tests = parse_quick_csv()

    passed = 0
    failed = 0
    for test_exe, args in tests:
        if run_test(test_exe, args, env):
            passed += 1
        else:
            failed += 1

    total = passed + failed
    logging.info("=" * 70)
    logging.info("OpenCL-CTS Test Summary:")
    logging.info(f"  Total:  {total}")
    logging.info(f"  Passed: {passed}")
    logging.info(f"  Failed: {failed}")
    logging.info("=" * 70)

    if failed > 0:
        logging.error(f"{failed} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        verify_opencl_runtime()
        run_tests()
        logging.info("++ OpenCL-CTS tests completed successfully")

    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}: {e.cmd}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)
