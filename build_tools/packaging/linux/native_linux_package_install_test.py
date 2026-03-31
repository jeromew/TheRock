#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Full installation and simulate-install test script for ROCm native packages.

Test modes (--test-type):
- sanity: Basic test. Repo-based install plus basic verification only
  (steps 1 and 2).
- full: Full test. Repo-based install plus basic verification plus full
  verification (steps 1, 2, and 3).
  Steps (invoked one by one from main):
  1. Repo setup and install: set up package-manager repository and install
     ROCm packages (amdrocm-{gfx_arch}, amdrocm-core-sdk-{gfx_arch}).
  2. Basic verification: install prefix, key components, installed packages
     list, rocminfo. (Run for both sanity and full.)
  3. Full verification: rdhc.py / RDHC test. (Run only for full.)
- simulate: Dry-run only. Simulated install of local .deb or .rpm files
  (apt install --simulate or rpm -Uvh --test --nodeps). No repo setup or
  actual install. Requires --packages-dir.

Path and repo name are overridable via environment variables: ROCM_REPO_NAME (repo id used for
APT list, Zypper/Yum repo file and section), ROCM_APT_KEYRING_DIR, ROCM_APT_SOURCES_LIST,
ROCM_APT_KEYRING_FILE, ROCM_ZYPP_REPOS_DIR, ROCM_YUM_REPOS_DIR,
ROCM_RDHC_REL_PATH (relative path from install prefix to rdhc binary).
Non-SLES RPM: before ROCm repo setup, download_and_install_ocl_icd_devel_rpm (Rocky EL9 mirror: ocl-icd + opencl-headers + ocl-icd-devel, then dnf install local RPMs; works without RHEL subscription).
install_ocl_icd_devel_via_rhel_partners_repo remains for internal mirrors.

Prerequisites:
- This script does NOT start Docker or a VM. You must run it inside an existing
 container or VM that matches the target OS (e.g., Ubuntu for deb, AlmaLinux/RHEL
 for rpm, SLES container for sles). Start the appropriate Docker image or VM
 first, then invoke this script from inside that environment.
- Root or sudo permissions may be required (repository setup, package install, keyring writes).
- System packages (install with the OS package manager):
  Debian/Ubuntu: apt install -y python3 python3-pip wget curl
  RHEL/Alma/CentOS/AZL: dnf install -y python3 python3-pip wget curl
  SLES: zypper install -y python3 python3-pip wget curl
- Python packages: listed in build_tools/packaging/linux/tests/requirements.txt.
  Install with: pip install -r build_tools/packaging/linux/tests/requirements.txt
  (or from build_tools/packaging/linux/tests: pip install -r requirements.txt).
  Equivalent one-liner: pip install pyelftools requests prettytable PyYAML

Example invocations:

 # Nightly DEB (Ubuntu 24.04) - run inside ubuntu:24.04 container or VM
 python3 native_linux_package_install_test.py \\
 --os-profile ubuntu2404 \\
 --repo-url https://rocm.nightlies.amd.com/deb/20260204-21658678136/ \\
 --gfx-arch gfx94x \\
 --release-type nightly

 # Prerelease DEB with GPG verification
 python3 native_linux_package_install_test.py \\
 --os-profile ubuntu2404 \\
 --repo-url https://rocm.prereleases.amd.com/packages/ubuntu2404 \\
 --release-type prerelease \\
 --gpg-key-url https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg

 # Nightly RPM (RHEL 8) - run inside rhel8/almalinux container or VM
 python3 native_linux_package_install_test.py \\
 --os-profile rhel8 \\
 --repo-url https://rocm.nightlies.amd.com/rpm/20260204-21658678136/x86_64/ \\
 --gfx-arch gfx94x \\
 --release-type nightly

 # Prerelease RPM (SLES 16)
 python3 native_linux_package_install_test.py \\
 --os-profile sles16 \\
 --repo-url https://rocm.prereleases.amd.com/packages/sles16/x86_64/ \\
 --release-type prerelease \\
 --gpg-key-url https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg

 # --test-type sanity (default): repo install + basic verification only (steps 1-2)
 python3 native_linux_package_install_test.py --test-type sanity \\
 --os-profile ubuntu2404 \\
 --repo-url https://rocm.nightlies.amd.com/deb/20260204-21658678136/ \\
 --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

 # --test-type full: same as sanity plus rdhc full verification (steps 1-3)
 python3 native_linux_package_install_test.py --test-type full \\
 --os-profile ubuntu2404 \\
 --repo-url https://rocm.nightlies.amd.com/deb/20260204-21658678136/ \\
 --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

 # Simulate install (dry-run) from local .deb or .rpm directory
 python3 native_linux_package_install_test.py --test-type simulate --packages-dir /path/to/pkgs --os-profile ubuntu2404
 python3 native_linux_package_install_test.py --test-type simulate --packages-dir /path/to/rpms --pkg-type rpm
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.error
import urllib.request
from argparse import ArgumentParser, Namespace
from pathlib import Path


def _env(key: str, default: str) -> str:
    """Return os.environ[key] if set and non-empty, else default."""
    v = os.environ.get(key, "").strip()
    return v if v else default


# --- Config: paths overridable via environment variables ---
# ROCM_REPO_NAME: logical repo name used for APT list, Zypper/Yum repo file and section id
# ROCM_APT_*, ROCM_ZYPP_*, ROCM_YUM_*, ROCM_RDHC_REL_PATH
REPO_NAME = _env("ROCM_REPO_NAME", "rocm-test")
APT_KEYRING_DIR = _env("ROCM_APT_KEYRING_DIR", "/etc/apt/keyrings")
APT_SOURCES_LIST = _env(
    "ROCM_APT_SOURCES_LIST", f"/etc/apt/sources.list.d/{REPO_NAME}.list"
)
APT_KEYRING_FILE = _env("ROCM_APT_KEYRING_FILE", "/etc/apt/keyrings/rocm.gpg")
ZYPP_REPOS_DIR = _env("ROCM_ZYPP_REPOS_DIR", "/etc/zypp/repos.d")
YUM_REPOS_DIR = _env("ROCM_YUM_REPOS_DIR", "/etc/yum.repos.d")
# RHEL partners repo path segment in artifactory URLs (e.g. 9.4 for RHEL 9.4); set for rhel10, etc.
RHEL_PARTNERS_RELEASE = _env("ROCM_RHEL_PARTNERS_RELEASE", "9.4")
RHEL_PARTNERS_REPO_BASENAME = "RHEL-partners.repo"


def _rpm_arch_dir() -> str:
    m = platform.machine().lower()
    return "aarch64" if m in ("aarch64", "arm64") else "x86_64"


# EL9 OpenCL stack from Rocky public mirror (AppStream + CRB). Override per-RPM if your NVR/mirror differs.
# ocl-icd-devel alone is not enough on unregistered RHEL: dnf needs local runtime + headers RPMs too.
def _default_ocl_icd_runtime_rpm_url() -> str:
    a = _rpm_arch_dir()
    return (
        f"https://dl.rockylinux.org/pub/rocky/9/AppStream/{a}/os/Packages/o/"
        f"ocl-icd-2.2.13-4.el9.{a}.rpm"
    )


def _default_ocl_icd_devel_rpm_url() -> str:
    a = _rpm_arch_dir()
    return (
        f"https://dl.rockylinux.org/pub/rocky/9/CRB/{a}/os/Packages/o/"
        f"ocl-icd-devel-2.2.13-4.el9.{a}.rpm"
    )


OCL_ICD_RUNTIME_RPM_URL = _env(
    "ROCM_OCL_ICD_RUNTIME_RPM_URL",
    _default_ocl_icd_runtime_rpm_url(),
)
OCL_ICD_OPENCL_HEADERS_RPM_URL = _env(
    "ROCM_OCL_ICD_OPENCL_HEADERS_RPM_URL",
    "https://dl.rockylinux.org/pub/rocky/9/AppStream/x86_64/os/Packages/o/"
    "opencl-headers-3.0-6.20201007gitd65bcc5.el9.0.1.noarch.rpm",
)
OCL_ICD_DEVEL_RPM_URL = _env(
    "ROCM_OCL_ICD_DEVEL_RPM_URL",
    _default_ocl_icd_devel_rpm_url(),
)
VERIFY_KEY_COMPONENTS = [
    "bin/rocminfo",
    "bin/hipcc",
    "bin/clinfo",
    "include/hip/hip_runtime.h",
    "lib/libamdhip64.so",
]
# Relative path from install prefix to rdhc binary (script); overridable via ROCM_RDHC_REL_PATH
RDHC_REL_PATH = _env("ROCM_RDHC_REL_PATH", "libexec/rocm-core/rdhc.py")

# Timeouts (seconds) and verification threshold
GPG_MKDIR_TIMEOUT_SEC = 10
GPG_KEY_TIMEOUT_SEC = 60
APT_UPDATE_TIMEOUT_SEC = 120
ZYPP_CLEAN_TIMEOUT_SEC = 60
ZYPP_REFRESH_TIMEOUT_SEC = 120
DNF_CLEAN_TIMEOUT_SEC = 60
INSTALL_TIMEOUT_SEC = 1800  # 30 minutes
ROCMINFO_TIMEOUT_SEC = 30
RDHC_TIMEOUT_SEC = 30
VERIFY_MIN_COMPONENTS = 2
OCL_ICD_DEVEL_DOWNLOAD_TIMEOUT_SEC = 300


def _run_streaming(
    cmd: list[str],
    timeout_sec: int,
    cwd: str | Path | None = None,
) -> int:
    """Run a command with streaming stdout/stderr and return its exit code.

    Lines are printed as they are produced. Raises subprocess.TimeoutExpired
    (after killing the process) or OSError on failure.
    """
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        for line in process.stdout:
            print(line.rstrip())
            sys.stdout.flush()
        return process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        process.kill()
        raise


def _try_enable_el9_crb_repos() -> None:
    """Best-effort: enable CRB (or equivalent) on EL9 so dnf can resolve ocl-icd-devel deps."""
    try:
        ver = (
            Path("/etc/os-release").read_text(encoding="utf-8", errors="ignore").lower()
        )
    except OSError:
        return
    if "el9" not in ver and 'version_id="9' not in ver and "version_id=9" not in ver:
        return
    print(
        "[INFO] EL9: enabling CRB / codeready-builder (needed for ocl-icd-devel dependencies)..."
    )
    subprocess.run(
        ["dnf", "install", "-y", "dnf-plugins-core"],
        check=False,
        timeout=300,
    )
    for repo in (
        "crb",
        "powertools",
        "codeready-builder-for-rhel-9-x86_64-rpms",
        "codeready-builder-for-rhel-9-aarch64-rpms",
    ):
        try:
            r = subprocess.run(
                ["dnf", "config-manager", "--set-enabled", repo],
                check=False,
                timeout=120,
                capture_output=True,
                text=True,
            )
            if r.returncode == 0:
                print(f"[INFO] enabled repo id: {repo}")
                break
        except OSError:
            continue
    subprocess.run(["dnf", "makecache"], check=False, timeout=300)


def is_non_sles_rpm_platform() -> bool:
    """True if not SLES (for ocl-icd-devel RPM path); uses /etc/os-release ID."""
    try:
        for line in (
            Path("/etc/os-release")
            .read_text(encoding="utf-8", errors="ignore")
            .splitlines()
        ):
            if line.startswith("ID="):
                ident = line.split("=", 1)[1].strip().strip('"').lower()
                return ident != "sles"
    except OSError:
        return True
    return True


def _download_rpm_from_url(
    rpm_url: str,
    dest_dir: Path,
    *,
    timeout_sec: int = OCL_ICD_DEVEL_DOWNLOAD_TIMEOUT_SEC,
) -> Path:
    """Download one .rpm from URL into dest_dir. Raises RuntimeError on failure."""
    rpm_url = rpm_url.strip()
    if not rpm_url:
        raise RuntimeError("RPM URL is empty")
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = rpm_url.rsplit("/", 1)[-1]
    if not filename.endswith(".rpm"):
        filename = "package.rpm"
    out_path = dest_dir / filename
    try:
        with urllib.request.urlopen(rpm_url, timeout=timeout_sec) as resp:
            out_path.write_bytes(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"download failed: {e}") from e
    if not out_path.is_file() or out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)
        raise RuntimeError("downloaded file is missing or empty")
    return out_path


def download_ocl_icd_el9_bundle_rpms(
    dest_dir: Path | str | None = None,
    *,
    devel_url: str | None = None,
    timeout_sec: int = OCL_ICD_DEVEL_DOWNLOAD_TIMEOUT_SEC,
) -> list[Path]:
    """Download ocl-icd (runtime), opencl-headers, and ocl-icd-devel for EL9 (Rocky mirror defaults).

    Needed for ``dnf install`` on hosts without AppStream/CRB (e.g. unregistered RHEL): dependencies
    are satisfied from local RPMs in one transaction.

    Env overrides: ROCM_OCL_ICD_RUNTIME_RPM_URL, ROCM_OCL_ICD_OPENCL_HEADERS_RPM_URL,
    ROCM_OCL_ICD_DEVEL_RPM_URL (devel_url overrides the last).

    Returns paths in dependency-friendly order (runtime, headers, devel).
    """
    if not is_non_sles_rpm_platform():
        raise RuntimeError(
            "ocl-icd bundle download is for non-SLES rpm hosts; this host looks like SLES."
        )
    out_dir = Path(dest_dir or os.getcwd()).resolve()
    runtime = _download_rpm_from_url(
        OCL_ICD_RUNTIME_RPM_URL, out_dir, timeout_sec=timeout_sec
    )
    headers = _download_rpm_from_url(
        OCL_ICD_OPENCL_HEADERS_RPM_URL, out_dir, timeout_sec=timeout_sec
    )
    devel = _download_rpm_from_url(
        (devel_url or OCL_ICD_DEVEL_RPM_URL).strip(), out_dir, timeout_sec=timeout_sec
    )
    return [runtime, headers, devel]


def download_ocl_icd_devel_rpm(
    dest_dir: Path | str | None = None,
    url: str | None = None,
    *,
    timeout_sec: int = OCL_ICD_DEVEL_DOWNLOAD_TIMEOUT_SEC,
) -> Path:
    """Download ocl-icd-devel RPM only (EL9 CRB by default). Raises RuntimeError on failure.

    For installation without subscription, use download_ocl_icd_el9_bundle_rpms + install.
    See ROCM_OCL_ICD_DEVEL_RPM_URL / OCL_ICD_DEVEL_RPM_URL.
    """
    if not is_non_sles_rpm_platform():
        raise RuntimeError(
            "ocl-icd-devel RPM download is for non-SLES rpm hosts; this host looks like SLES."
        )
    out_dir = Path(dest_dir or os.getcwd()).resolve()
    return _download_rpm_from_url(
        (url or OCL_ICD_DEVEL_RPM_URL).strip(), out_dir, timeout_sec=timeout_sec
    )


def install_ocl_icd_el9_local_rpms(
    rpm_paths: list[Path],
    *,
    timeout_sec: int = INSTALL_TIMEOUT_SEC,
) -> None:
    """Install EL9 OpenCL RPMs from local paths in one dnf transaction (no subscription required).

    Pass runtime + opencl-headers + ocl-icd-devel together so dependencies are satisfied offline.
    """
    paths = [p.resolve() for p in rpm_paths]
    for p in paths:
        if not p.is_file():
            raise RuntimeError(f"RPM not found: {p}")
    try:
        rc = _run_streaming(
            ["dnf", "install", "-y"] + [str(p) for p in paths],
            timeout_sec,
        )
    except FileNotFoundError as e:
        raise RuntimeError("dnf not found") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("dnf install timed out") from e
    if rc != 0:
        raise RuntimeError(f"dnf install failed (exit {rc})")


def install_ocl_icd_devel_rpm(
    rpm_path: Path,
    *,
    timeout_sec: int = INSTALL_TIMEOUT_SEC,
) -> None:
    """Install a single ocl-icd-devel .rpm with dnf (expects repos to supply dependencies)."""
    path = rpm_path.resolve()
    if not path.is_file():
        raise RuntimeError(f"RPM not found: {path}")
    _try_enable_el9_crb_repos()
    try:
        rc = _run_streaming(
            ["dnf", "install", "-y", str(path)],
            timeout_sec,
        )
    except FileNotFoundError as e:
        raise RuntimeError("dnf not found") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("dnf install timed out") from e
    if rc != 0:
        raise RuntimeError(
            f"dnf install failed (exit {rc}). On unregistered RHEL use "
            "download_ocl_icd_el9_bundle_rpms + install_ocl_icd_el9_local_rpms instead."
        )


def download_and_install_ocl_icd_devel_rpm(rpm_url: str | None = None) -> bool:
    """Download EL9 ocl-icd + opencl-headers + ocl-icd-devel, dnf install locally, then cleanup.

    Uses a bundle so unregistered RHEL (no AppStream/CRB) can install without subscription.
    """
    if not is_non_sles_rpm_platform():
        return True
    tmp_dir = Path(tempfile.mkdtemp(prefix="rocm-ocl-icd-devel-"))
    staged: list[Path] = []
    try:
        bundle = download_ocl_icd_el9_bundle_rpms(dest_dir=tmp_dir, devel_url=rpm_url)
        for p in bundle:
            sp = Path(tempfile.gettempdir()) / f"rocm-oclicd-{os.getpid()}-{p.name}"
            shutil.copy2(p, sp)
            staged.append(sp)
        print(
            "Installing ocl-icd stack with dnf (local RPMs): "
            + ", ".join(str(s) for s in staged)
        )
        install_ocl_icd_el9_local_rpms(staged)
        print("[PASS] ocl-icd stack installed (ocl-icd, opencl-headers, ocl-icd-devel)")
        return True
    except RuntimeError as e:
        print(f"[FAIL] ocl-icd stack: {e}", file=sys.stderr)
        return False
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        for sp in staged:
            sp.unlink(missing_ok=True)


def rhel_partners_repo_file_content(release_ver: str) -> str:
    """Build /etc/yum.repos.d/RHEL-partners.repo body for AMD SC V artifactory.

    Used for non-SLES RPM (dnf) hosts (e.g. rhel9, rhel10) to pull ocl-icd-devel.
    release_ver is the path segment after rhel-remote/ (default from ROCM_RHEL_PARTNERS_RELEASE).
    """
    root = "http://scvartifactory.amd.com/artifactory/list/rhel-remote"
    return (
        "[RHEL-partners-baseos]\n"
        f"name=Red Hat Enterprise Linux {release_ver} Partners - $basearch\n"
        f"baseurl={root}/{release_ver}/$basearch/os/BaseOS\n"
        "enabled=1\n"
        "gpgcheck=0\n"
        "\n"
        "[RHEL-partners-appstream]\n"
        f"name=Red Hat Enterprise Linux {release_ver} Partners (AppStream) - $basearch\n"
        f"baseurl={root}/{release_ver}/$basearch/os/AppStream\n"
        "enabled=1\n"
        "gpgcheck=0\n"
        "\n"
        "[RHEL-partners-crb]\n"
        "name=Red Hat Enterprise Linux $releasever Partners (CRB) - $basearch\n"
        f"baseurl={root}/{release_ver}/$basearch/os/CRB\n"
        "enabled=1\n"
        "gpgcheck=0\n"
    )


def install_ocl_icd_devel_via_rhel_partners_repo(
    yum_repos_dir: str | None = None,
    release_ver: str | None = None,
) -> bool:
    """Write RHEL-partners.repo and install ocl-icd-devel via dnf (non-SLES RPM only).

    Intended for non-SLES RPM hosts that use dnf; callers should not invoke this for deb or SLES.

    Returns:
        True if repo write and dnf install succeeded; False otherwise.
    """
    repos_dir = Path(yum_repos_dir or YUM_REPOS_DIR)
    rel = (release_ver or RHEL_PARTNERS_RELEASE).strip() or "9.4"
    repo_path = repos_dir / RHEL_PARTNERS_REPO_BASENAME
    content = rhel_partners_repo_file_content(rel)

    print("\n" + "=" * 80)
    print("RHEL PARTNERS REPO: ocl-icd-devel (non-SLES RPM)")
    print("=" * 80)
    print(f"\nRelease path segment: {rel}")
    print(f"Repo file: {repo_path}")

    try:
        repos_dir.mkdir(parents=True, exist_ok=True)
        repo_path.write_text(content, encoding="utf-8")
        print(f"[PASS] Wrote {repo_path}")
        print("\nRepository configuration:\n")
        print(content)
    except OSError as e:
        print(f"[FAIL] Could not write RHEL partners repo file: {e}", file=sys.stderr)
        return False

    print("\nInstalling ocl-icd-devel via dnf...")
    try:
        rc = _run_streaming(
            ["dnf", "install", "-y", "ocl-icd-devel"], INSTALL_TIMEOUT_SEC
        )
        if rc == 0:
            print("\n[PASS] ocl-icd-devel installed")
            return True
        print(
            f"\n[FAIL] dnf install ocl-icd-devel failed (exit code: {rc})",
            file=sys.stderr,
        )
        return False
    except subprocess.TimeoutExpired:
        print("\n[FAIL] dnf install ocl-icd-devel timed out", file=sys.stderr)
        return False
    except OSError as e:
        print(f"\n[FAIL] dnf install ocl-icd-devel: {e}", file=sys.stderr)
        return False


def run_simulate_install_test(pkg_type: str, packages_dir: str) -> bool:
    """Run simulated package install test (dry-run only, no actual install).

    Equivalent to the GitHub Actions 'Simulated install Test' step:
    - deb: apt install --simulate *.deb
    - rpm: rpm -Uvh --test --nodeps *.rpm

    Returns:
    True if simulate succeeded, False otherwise.
    """
    path = Path(packages_dir).resolve()
    if not path.is_dir():
        print(f"[FAIL] Not a directory: {packages_dir}", file=sys.stderr)
        return False

    if pkg_type == "deb":
        debs = [str(p.resolve()) for p in path.glob("*.deb")]
        if not debs:
            print(f"[FAIL] No .deb files found in {packages_dir}", file=sys.stderr)
            return False
        print("Simulate installing DEB packages on host system for testing")
        # Use absolute paths so apt treats them as local files, not package names
        cmd = ["apt", "install", "--simulate"] + debs
    elif pkg_type == "rpm":
        rpms = [str(p.resolve()) for p in path.glob("*.rpm")]
        if not rpms:
            print(f"[FAIL] No .rpm files found in {packages_dir}", file=sys.stderr)
            return False
        print("Simulate installing RPM packages for testing")
        # Use absolute paths for consistency
        cmd = ["rpm", "-Uvh", "--test", "--nodeps"] + rpms
    else:
        print(
            f"[FAIL] Unsupported pkg_type: {pkg_type}. Use 'deb' or 'rpm'.",
            file=sys.stderr,
        )
        return False

    try:
        subprocess.run(cmd, check=True)
        print("[PASS] Simulated install test completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(
            f"[FAIL] Simulated install failed with exit code {e.returncode}",
            file=sys.stderr,
        )
        return False
    except FileNotFoundError as e:
        print(f"[FAIL] Command not found: {e}", file=sys.stderr)
        return False


class NativeLinuxPackageInstallTest:
    """Runner for the native Linux package install test (repo setup, install, verification)."""

    @staticmethod
    def _derive_package_type(os_profile: str) -> str:
        """Derive package type from OS profile.

        Args:
        os_profile: OS profile (e.g., ubuntu2404, rhel8, debian12, sles16, almalinux9, centos7, azl3)

        Returns:
        Package type ('deb' or 'rpm')
        """
        os_profile_lower = os_profile.lower()
        if os_profile_lower.startswith(("ubuntu", "debian")):
            return "deb"
        elif os_profile_lower.startswith(
            ("rhel", "sles", "almalinux", "centos", "azl")
        ):
            return "rpm"
        else:
            raise ValueError(
                f"Unable to derive package type from OS profile: {os_profile}. "
                "Supported profiles: ubuntu*, debian*, rhel*, sles*, almalinux*, centos*, azl*"
            )

    def _is_sles(self) -> bool:
        """Check if the OS profile is SLES (SUSE Linux Enterprise Server).

        Returns:
        True if SLES, False otherwise
        """
        return self.os_profile.lower().startswith("sles")

    def setup_rhel_partners_repo_and_install_ocl_icd_devel(self) -> bool:
        """For RPM using dnf (not SLES): download_and_install_ocl_icd_devel_rpm (CRB, then dnf).

        For internal networks, install_ocl_icd_devel_via_rhel_partners_repo remains available
        but is not used on this code path. No-op (returns True) for deb and SLES.

        Returns:
        True if no-op (deb/SLES) or install succeeded; False on failure.
        """
        if self.package_type != "rpm" or self._is_sles():
            return True
        return download_and_install_ocl_icd_devel_rpm()

    def __init__(
        self,
        repo_url: str,
        os_profile: str,
        release_type: str = "nightly",
        install_prefix: str | None = None,
        gfx_arch: str | list[str] | None = None,
        gpg_key_url: str | None = None,
    ):
        """Initialize the native Linux package install test runner.

        Args:
        repo_url: Full repository URL (constructed in YAML)
        os_profile: OS profile (e.g., ubuntu2404, rhel8, debian12, sles15, sles16, almalinux9, centos7, azl3)
        release_type: Type of release ('nightly' or 'prerelease')
        install_prefix: Installation prefix (default: /opt/rocm/core)
        gfx_arch: GPU architecture(s) as a single value or list (default: gfx94x).
        Only the first element is used for package name and installation.
        gpg_key_url: GPG key URL
        """
        self.os_profile = os_profile.lower()
        self.package_type = self._derive_package_type(os_profile)
        self.repo_url = repo_url.rstrip("/")
        self.release_type = release_type.lower()
        self.install_prefix = install_prefix
        # Normalize to list; only the first element is used for now
        if gfx_arch is None:
            self.gfx_arch_list: list[str] = ["gfx94x"]
        elif isinstance(gfx_arch, str):
            self.gfx_arch_list = [gfx_arch] if gfx_arch.strip() else ["gfx94x"]
        else:
            self.gfx_arch_list = [a for a in gfx_arch if a and str(a).strip()] or [
                "gfx94x"
            ]
        self.gfx_arch = self.gfx_arch_list[0].lower()
        self.gpg_key_url = gpg_key_url

        # Packages to install, in order
        self.package_names = [
            f"amdrocm-{self.gfx_arch}",
            f"amdrocm-core-sdk-{self.gfx_arch}",
        ]

    def setup_gpg_key(self) -> bool:
        """Setup GPG key for repositories that require GPG verification.

        Returns:
        True if setup successful, False otherwise
        """
        if not self.gpg_key_url:
            return True  # Not needed if no GPG key URL provided

        print("\n" + "=" * 80)
        print("SETTING UP GPG KEY")
        print("=" * 80)

        print(f"\nGPG Key URL: {self.gpg_key_url}")

        if self.package_type == "deb":
            # For DEB, import GPG key using pipeline approach
            keyring_dir = Path(APT_KEYRING_DIR)
            keyring_file = keyring_dir / "rocm.gpg"

            try:
                # Create keyring directory
                print(f"\nCreating keyring directory: {keyring_dir}...")
                subprocess.run(
                    ["mkdir", "--parents", "--mode=0755", str(keyring_dir)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=GPG_MKDIR_TIMEOUT_SEC,
                )
                print(f"[PASS] Created keyring directory: {keyring_dir}")

                # Download, dearmor, and write GPG key using pipeline
                # wget URL -O - | gpg --dearmor | tee keyring_file > /dev/null
                print(f"\nDownloading and importing GPG key from {self.gpg_key_url}...")
                pipeline_cmd = (
                    f"wget -q -O - {self.gpg_key_url} | "
                    f"gpg --dearmor | "
                    f"tee {keyring_file} > /dev/null"
                )

                subprocess.run(
                    pipeline_cmd,
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=GPG_KEY_TIMEOUT_SEC,
                )

                # Set proper permissions on the keyring file
                keyring_file.chmod(0o644)
                print(f"[PASS] GPG key imported to {keyring_file}")
                return True

            except subprocess.CalledProcessError as e:
                print(f"[FAIL] Failed to setup GPG key: {e}")
                if e.stderr:
                    print(f"Error output: {e.stderr.decode()}")
                return False
            except OSError as e:
                print(f"[FAIL] Error setting up GPG key: {e}")
                return False
        else:  # rpm
            # For RPM (including SLES), GPG key URL is specified in repo file
            # zypper will automatically fetch and use the GPG key from the URL
            # No need to download or import separately (following official ROCm documentation)
            return True

    def setup_deb_repository(self) -> bool:
        """Setup DEB repository on the system.

        Returns:
        True if setup successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("SETTING UP DEB REPOSITORY")
        print("=" * 80)

        print(f"\nRepository URL: {self.repo_url}")
        print(f"Release Type: {self.release_type}")

        # Setup GPG key if GPG key URL is provided
        if self.gpg_key_url:
            if not self.setup_gpg_key():
                return False

        # Add repository to sources list
        print("\nAdding ROCm repository...")
        sources_list = Path(APT_SOURCES_LIST)

        if self.gpg_key_url:
            # Use GPG key verification
            apt_keyring = Path(APT_KEYRING_FILE)
            repo_entry = f"deb [arch=amd64 signed-by={apt_keyring}] {self.repo_url} stable main\n"
        else:
            # No GPG check (trusted=yes)
            repo_entry = f"deb [arch=amd64 trusted=yes] {self.repo_url} stable main\n"

        try:
            sources_list.write_text(repo_entry, encoding="utf-8")
            print(f"[PASS] Repository added to {sources_list}")
            print(f" {repo_entry.strip()}")
        except OSError as e:
            print(f"[FAIL] Failed to add repository: {e}")
            return False

        # Update package lists
        print("\nUpdating package lists...")
        print("=" * 80)
        try:
            return_code = _run_streaming(["apt", "update"], APT_UPDATE_TIMEOUT_SEC)
            if return_code == 0:
                print("\n[PASS] Package lists updated")
                return True
            print(f"\n[FAIL] Failed to update package lists (exit code: {return_code})")
            return False
        except subprocess.TimeoutExpired:
            print("\n[FAIL] apt update timed out")
            return False
        except OSError as e:
            print(f"[FAIL] Error updating package lists: {e}")
            return False

    def _setup_sles_repository(self) -> bool:
        """Setup repository for SLES using zypper.

        Returns:
        True if setup successful, False otherwise
        """
        repo_name = REPO_NAME
        repo_file = Path(ZYPP_REPOS_DIR) / f"{repo_name}.repo"

        # Remove existing repository if it exists
        print(f"\nRemoving existing repository '{repo_name}' if it exists...")
        subprocess.run(
            ["zypper", "--non-interactive", "removerepo", repo_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )  # Ignore errors if repo doesn't exist

        # Create repository file following official ROCm documentation format
        # Reference: https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager/package-manager-sles.html
        print(f"\nCreating ROCm repository file at {repo_file}...")
        if self.gpg_key_url:
            # Use GPG key verification (gpgcheck=1)
            repo_content = f"""[{repo_name}]
name=ROCm {self.release_type} repository
baseurl={self.repo_url}
enabled=1
gpgcheck=1
gpgkey={self.gpg_key_url}
"""
        else:
            # No GPG check (gpgcheck=0)
            repo_content = f"""[{repo_name}]
name=ROCm {self.release_type} repository
baseurl={self.repo_url}
enabled=1
gpgcheck=0
"""

        try:
            repo_file.write_text(repo_content, encoding="utf-8")
            print(f"[PASS] Repository file created: {repo_file}")
            print("\nRepository configuration:")
            print(repo_content)
        except OSError as e:
            print(f"[FAIL] Failed to create repository file: {e}")
            return False

        # Clean zypper cache
        print("\nCleaning zypper cache...")
        try:
            result = subprocess.run(
                ["zypper", "--non-interactive", "clean", "--all"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=ZYPP_CLEAN_TIMEOUT_SEC,
            )
            if result.returncode == 0:
                print("[PASS] zypper cache cleaned")
            else:
                print(
                    f"[WARN] zypper clean returned {result.returncode} (may not be critical)"
                )
        except subprocess.TimeoutExpired:
            print("[WARN] zypper clean timed out (may not be critical)")
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"[WARN] zypper clean failed: {e} (may not be critical)")

        # Refresh repository metadata
        print("\nRefreshing repository metadata...")
        try:
            # Use --non-interactive to avoid prompts
            # If GPG key URL is provided, use --gpg-auto-import-keys to automatically import and trust GPG keys
            refresh_cmd = ["zypper", "--non-interactive"]
            if self.gpg_key_url:
                refresh_cmd.append("--gpg-auto-import-keys")
            refresh_cmd.extend(["refresh", repo_name])
            return_code = _run_streaming(refresh_cmd, ZYPP_REFRESH_TIMEOUT_SEC)
            if return_code == 0:
                print("\n[PASS] Repository metadata refreshed")
                return True
            print(
                f"\n[FAIL] Failed to refresh repository metadata (exit code: {return_code})"
            )
            return False
        except subprocess.TimeoutExpired:
            print("\n[FAIL] zypper refresh timed out")
            return False
        except OSError as e:
            print(f"[FAIL] Error refreshing repository metadata: {e}")
            return False

    def _setup_dnf_repository(self) -> bool:
        """Setup repository for RHEL/AlmaLinux/CentOS using dnf/yum.

        Returns:
        True if setup successful, False otherwise
        """
        print("\nUsing dnf/yum for repository setup...")

        # Create repository file
        print("\nCreating ROCm repository file...")
        repo_name = REPO_NAME
        repo_file = Path(YUM_REPOS_DIR) / f"{repo_name}.repo"

        if self.gpg_key_url:
            # Use GPG key verification
            repo_content = f"""[{repo_name}]
name=ROCm Repository
baseurl={self.repo_url}
enabled=1
gpgcheck=1
gpgkey={self.gpg_key_url}
"""
        else:
            # No GPG check
            repo_content = f"""[{repo_name}]
name=Native Linux Package Test Repository
baseurl={self.repo_url}
enabled=1
gpgcheck=0
"""

        try:
            repo_file.write_text(repo_content, encoding="utf-8")
            print(f"[PASS] Repository file created: {repo_file}")
            print("\nRepository configuration:")
            print(repo_content)
        except OSError as e:
            print(f"[FAIL] Failed to create repository file: {e}")
            return False

        # Clean dnf cache
        print("\nCleaning dnf cache...")
        try:
            subprocess.run(
                ["dnf", "clean", "all"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=DNF_CLEAN_TIMEOUT_SEC,
            )
            print("[PASS] dnf cache cleaned")
        except subprocess.CalledProcessError as e:
            print("[WARN] Failed to clean dnf cache (may not be critical)")
            print(f"Error: {e.stdout}")
        except subprocess.TimeoutExpired:
            print("[WARN] dnf clean timed out (may not be critical)")

        print("\n[PASS] DNF repository setup complete")
        return True

    def setup_rpm_repository(self) -> bool:
        """Setup RPM repository on the system.

        Returns:
        True if setup successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("SETTING UP RPM REPOSITORY")
        print("=" * 80)

        print(f"\nRepository URL: {self.repo_url}")
        print(f"Release Type: {self.release_type}")
        print(f"OS Profile: {self.os_profile}")

        # Setup GPG key if GPG key URL is provided (only needed for non-SLES systems)
        # SLES uses --gpg-auto-import-keys flag which handles it automatically
        if self.gpg_key_url and not self._is_sles():
            if not self.setup_gpg_key():
                return False

        # SLES uses zypper, others use dnf/yum
        if self._is_sles():
            return self._setup_sles_repository()
        else:
            return self._setup_dnf_repository()

    def install_deb_packages(self) -> bool:
        """Install ROCm DEB packages from repository.

        Returns:
        True if installation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("INSTALLING DEB PACKAGES FROM REPOSITORY")
        print("=" * 80)

        print(f"\nPackages to install (in order): {self.package_names}")

        # Install using apt (packages in list order)
        cmd = ["apt", "install", "-y"] + self.package_names
        print(f"\nRunning: {' '.join(cmd)}")
        print("=" * 80)
        print("Installation progress (streaming output):\n")

        try:
            return_code = _run_streaming(cmd, INSTALL_TIMEOUT_SEC)
            if return_code == 0:
                print("\n" + "=" * 80)
                print("[PASS] DEB packages installed successfully from repository")
                return True
            print("\n" + "=" * 80)
            print(f"[FAIL] Failed to install DEB packages (exit code: {return_code})")
            return False
        except subprocess.TimeoutExpired:
            print("\n" + "=" * 80)
            print(f"[FAIL] Installation timed out after {INSTALL_TIMEOUT_SEC} minutes")
            return False
        except OSError as e:
            print(f"\n[FAIL] Error during installation: {e}")
            return False

    def install_rpm_packages(self) -> bool:
        """Install ROCm RPM packages from repository.

        Returns:
        True if installation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("INSTALLING RPM PACKAGES FROM REPOSITORY")
        print("=" * 80)

        print(f"\nPackages to install (in order): {self.package_names}")

        # Use zypper for SLES, dnf for others
        if self._is_sles():
            # If no GPG key URL, skip GPG checks during installation
            if not self.gpg_key_url:
                cmd = [
                    "zypper",
                    "--non-interactive",
                    "--no-gpg-checks",
                    "install",
                    "-y",
                ] + self.package_names
            else:
                # If GPG key URL is provided, use --gpg-auto-import-keys to automatically import and trust GPG keys
                cmd = [
                    "zypper",
                    "--non-interactive",
                    "--gpg-auto-import-keys",
                    "install",
                    "-y",
                ] + self.package_names
            print("[INFO] Using zypper for SLES package installation")
        else:
            cmd = ["dnf", "install", "-y"] + self.package_names
        print(f"\nRunning: {' '.join(cmd)}")
        print("=" * 80)
        print("Installation progress (streaming output):\n")

        try:
            return_code = _run_streaming(cmd, INSTALL_TIMEOUT_SEC)
            if return_code == 0:
                print("\n" + "=" * 80)
                print("[PASS] RPM packages installed successfully from repository")
                return True
            print("\n" + "=" * 80)
            print(f"[FAIL] Failed to install RPM packages (exit code: {return_code})")
            return False
        except subprocess.TimeoutExpired:
            print("\n" + "=" * 80)
            print(f"[FAIL] Installation timed out after {INSTALL_TIMEOUT_SEC} minutes")
            return False
        except OSError as e:
            print(f"\n[FAIL] Error during installation: {e}")
            return False

    def run_repo_setup_and_install(self) -> bool:
        """Step 1: Repo setup and install. Run for both sanity (basic) and full test.

        Returns:
        True if repository setup and package installation both succeeded.
        """
        print("\n" + "=" * 80)
        print("STEP 1: REPOSITORY SETUP AND PACKAGE INSTALLATION")
        print("=" * 80)
        print(f"\nOS Profile: {self.os_profile}")
        print(f"Package Type (derived): {self.package_type.upper()}")
        print(f"Repository URL: {self.repo_url}")
        print(f"Packages (in order): {self.package_names}")

        if self.package_type == "deb":
            if not self.setup_deb_repository():
                return False
            return self.install_deb_packages()
        else:
            if not self.setup_rhel_partners_repo_and_install_ocl_icd_devel():
                return False
            if not self.setup_rpm_repository():
                return False
            return self.install_rpm_packages()

    def run_basic_verification(self) -> bool:
        """Step 2: Basic test — install prefix, key components, packages list, rocminfo.

        Used by both --test-type sanity and full. Does not run test_rdhc
        (that is Step 3 / run_full_verification, full test only).

        Returns:
        True if basic verification passed (enough components found).
        """
        print("\n" + "=" * 80)
        print("STEP 2: BASIC INSTALL VERIFICATION")
        print("=" * 80)

        install_path = Path(self.install_prefix)
        if not install_path.exists():
            print(f"\n[FAIL] Installation directory not found: {self.install_prefix}")
            return False

        print(f"\n[PASS] Installation directory exists: {self.install_prefix}")

        key_components = VERIFY_KEY_COMPONENTS
        print("\nChecking for key ROCm components:")
        found_count = 0
        for component in key_components:
            component_path = install_path / component
            if component_path.exists():
                print(f" [PASS] {component}")
                found_count += 1
            else:
                print(f" [WARN] {component} (not found)")

        print(f"\nComponents found: {found_count}/{len(key_components)}")

        # Check installed packages
        print("\nChecking installed packages:")
        try:
            if self.package_type == "deb":
                cmd = ["dpkg", "-l"]
                grep_pattern = "rocm"
            elif self._is_sles():
                cmd = ["zypper", "--non-interactive", "search", "-i", "rocm"]
                grep_pattern = "rocm"
            else:
                cmd = ["rpm", "-qa"]
                grep_pattern = "rocm"

            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            rocm_packages = [
                line
                for line in result.stdout.split("\n")
                if grep_pattern.lower() in line.lower()
            ]
            print(f" Found {len(rocm_packages)} ROCm packages installed")
            if rocm_packages:
                print("\n Sample packages (Show first 5):")
                for pkg in rocm_packages[:5]:
                    print(f" {pkg.strip()}")
                if len(rocm_packages) > 5:
                    print(f" ... and {len(rocm_packages) - 5} more")
        except subprocess.CalledProcessError:
            print(" [WARN] Could not query installed packages")

        # Try to run rocminfo if available
        rocminfo_path = install_path / "bin" / "rocminfo"
        if rocminfo_path.exists():
            print("\nTrying to run rocminfo...")
            try:
                result = subprocess.run(
                    [str(rocminfo_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=ROCMINFO_TIMEOUT_SEC,
                )
                print(" [PASS] rocminfo executed successfully")
                lines = result.stdout.split("\n")[:10]
                print("\n First few lines of rocminfo output:")
                for line in lines:
                    if line.strip():
                        print(f" {line}")
            except subprocess.TimeoutExpired:
                print(" [WARN] rocminfo timed out (may require GPU hardware)")
            except subprocess.CalledProcessError:
                print(" [WARN] rocminfo failed (may require GPU hardware)")
            except OSError as e:
                print(f" [WARN] Could not run rocminfo: {e}")

        if found_count >= VERIFY_MIN_COMPONENTS:
            print("\n[PASS] Basic verification PASSED")
            return True
        print("\n[FAIL] Basic verification FAILED (insufficient components)")
        return False

    def run_full_verification(self) -> bool:
        """Step 3: Full test — runs test_rdhc (rdhc.py) only. Used when --test-type is full."""
        print("\n" + "=" * 80)
        print("STEP 3: FULL VERIFICATION (RDHC)")
        print("=" * 80)
        return self.test_rdhc()

    def test_rdhc(self) -> bool:
        """Test rdhc.py binary in libexec/rocm-core/.

        Returns:
        True if test successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("TESTING RDHC.PY")
        print("=" * 80)

        install_path = Path(self.install_prefix).resolve()
        rdhc_script = (install_path / RDHC_REL_PATH).resolve()
        rocm_install_prefix_arg = str(install_path)

        # Check if script exists
        if not rdhc_script.exists():
            print(f"\n[WARN] rdhc.py not found at: {rdhc_script}")
            print(" This is expected if rocm-core package is not installed")
            return False

        print(f"\n[PASS] rdhc.py found at: {rdhc_script}")

        # Check if script is executable or can be run with python
        if os.access(rdhc_script, os.X_OK):
            cmd = [str(rdhc_script)]
        else:
            cmd = [sys.executable, str(rdhc_script)]

        # Set RDHC arguments for full test
        test_args = ["--rocm-install-prefix", rocm_install_prefix_arg, "--all"]
        print(
            f"\nRun rdhc.py with --rocm-install-prefix {rocm_install_prefix_arg} --all..."
        )
        print(f"Command: {' '.join(cmd + test_args)}")

        try:
            result = subprocess.run(
                cmd + test_args,
                cwd=str(install_path),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=RDHC_TIMEOUT_SEC,
            )
            print(" [PASS] rdhc.py executed successfully")
            if result.stdout:
                # Print first few lines of output
                lines = result.stdout.split("\n")[:5]
                print("\n First few lines of output:")
                for line in lines:
                    if line.strip():
                        print(f" {line}")
            return True
        except subprocess.TimeoutExpired:
            print(" [WARN] rdhc.py --all timed out")
            return False
        except subprocess.CalledProcessError:
            print(" [WARN] rdhc.py --all failed")
            return False
        except OSError as e:
            print(f" [WARN] Could not run rdhc.py: {e}")
            return False


_CLI_EXAMPLES_EPILOG = """
Examples:
 # Nightly DEB (Ubuntu 24.04) - run inside matching container/VM
 python native_linux_package_install_test.py --os-profile ubuntu2404 \\
 --repo-url https://rocm.nightlies.amd.com/deb/20260204-21658678136/ \\
 --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

 # Prerelease DEB with GPG verification
 python native_linux_package_install_test.py --os-profile ubuntu2404 \\
 --repo-url https://rocm.prereleases.amd.com/packages/ubuntu2404 \\
 --gfx-arch gfx94x --release-type prerelease --install-prefix /opt/rocm/core \\
 --gpg-key-url https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg

 # Nightly RPM (RHEL 8)
 python native_linux_package_install_test.py --os-profile rhel8 \\
 --repo-url https://rocm.nightlies.amd.com/rpm/20260204-21658678136/rhel8/x86_64/ \\
 --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

 # Prerelease RPM (RHEL 8)
 python native_linux_package_install_test.py --os-profile rhel8 \\
 --repo-url https://rocm.prereleases.amd.com/packages/rhel8/x86_64/ \\
 --gfx-arch gfx94x --release-type prerelease --install-prefix /opt/rocm/core \\
 --gpg-key-url https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg

 # --test-type sanity (default): repo install + basic verification only
 python native_linux_package_install_test.py --test-type sanity --os-profile ubuntu2404 \\
 --repo-url https://rocm.nightlies.amd.com/deb/20260204-21658678136/ \\
 --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

 # --test-type full: install + basic verification + rdhc
 python native_linux_package_install_test.py --test-type full --os-profile ubuntu2404 \\
 --repo-url https://rocm.nightlies.amd.com/deb/20260204-21658678136/ \\
 --gfx-arch gfx94x --release-type nightly --install-prefix /opt/rocm/core

 # Simulate install (dry-run) from local packages
 python native_linux_package_install_test.py --test-type simulate --packages-dir /path/to/pkgs --os-profile ubuntu2404
 python native_linux_package_install_test.py --test-type simulate --packages-dir /path/to/rpms --pkg-type rpm
"""


def _build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Full installation and simulate-install test for ROCm native packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EXAMPLES_EPILOG,
    )
    parser.add_argument(
        "--os-profile",
        type=str,
        help="OS profile (e.g., ubuntu2404, rhel8, debian12, sles15, sles16, almalinux9, centos7, azl3). Required for sanity/full; for simulate, used only to derive pkg-type if --pkg-type is omitted.",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        help="Full repository URL (constructed in YAML workflow). Required for sanity/full; not used for simulate.",
    )
    parser.add_argument(
        "--gfx-arch",
        type=str,
        nargs="+",
        metavar="ARCH",
        help="GPU architecture(s) as a list. Only the first is used for now. Required for sanity/full; not used for simulate. Examples: gfx94x, gfx110x gfx1151",
    )
    parser.add_argument(
        "--release-type",
        type=str,
        choices=["nightly", "prerelease"],
        help="Type of release: 'nightly' or 'prerelease'",
    )
    parser.add_argument(
        "--install-prefix",
        type=str,
        help="Installation prefix (e.g. /opt/rocm/core)",
    )
    parser.add_argument(
        "--gpg-key-url",
        type=str,
        help="GPG key URL",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["sanity", "full", "simulate"],
        default="sanity",
        help="Test type: 'sanity' = basic test only; 'full' = basic + full test; 'simulate' = simulated install only (requires --packages-dir).",
    )
    parser.add_argument(
        "--packages-dir",
        type=str,
        metavar="DIR",
        help="Directory containing .deb or .rpm files. Required when --test-type is 'simulate'.",
    )
    parser.add_argument(
        "--pkg-type",
        type=str,
        choices=["deb", "rpm"],
        help="Package type (deb or rpm). For --test-type simulate only; if omitted, derived from --os-profile.",
    )
    return parser


def _validate_cli_args(parser: ArgumentParser, args: Namespace) -> None:
    if args.test_type == "simulate":
        if not args.packages_dir:
            parser.error("--packages-dir is required when --test-type is 'simulate'")
        if not args.pkg_type and not args.os_profile:
            parser.error(
                "When --test-type is 'simulate', provide --pkg-type or --os-profile"
            )
        if args.os_profile and not args.pkg_type:
            try:
                NativeLinuxPackageInstallTest._derive_package_type(args.os_profile)
            except ValueError as e:
                parser.error(str(e))
        return
    if not args.os_profile:
        parser.error("--os-profile is required when --test-type is 'sanity' or 'full'")
    if not args.repo_url:
        parser.error("--repo-url is required when --test-type is 'sanity' or 'full'")
    if not args.gfx_arch:
        parser.error("--gfx-arch is required when --test-type is 'sanity' or 'full'")


def parse_cli_arguments(argv: list[str] | None = None) -> Namespace:
    """Build parser, parse argv, validate; may call parser.error (exits process)."""
    parser = _build_argument_parser()
    args = parser.parse_args(argv)
    _validate_cli_args(parser, args)
    return args


def run_tests(args: Namespace) -> int:
    """Run simulate or repo-based install test from parsed CLI args. Returns exit code (0 success)."""
    if args.test_type == "simulate":
        pkg_type = args.pkg_type or NativeLinuxPackageInstallTest._derive_package_type(
            args.os_profile
        )
        print("\n" + "=" * 80)
        print("SIMULATED INSTALL TEST")
        print("=" * 80)
        ok = run_simulate_install_test(pkg_type, args.packages_dir)
        return 0 if ok else 1

    try:
        derived_package_type = NativeLinuxPackageInstallTest._derive_package_type(
            args.os_profile
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"OS Profile: {args.os_profile}")
    print(f"Package Type (derived): {derived_package_type}")
    print(f"Release Type: {args.release_type}")
    print(f"Repository URL: {args.repo_url}")
    print(f"GPU Architecture(s): {args.gfx_arch} (using first: {args.gfx_arch[0]})")
    print(f"Install Prefix: {args.install_prefix}")
    print(f"Test Type: {args.test_type}")
    if args.gpg_key_url:
        print(f"GPG Key URL: {args.gpg_key_url}")
    print("=" * 80)

    test_runner = NativeLinuxPackageInstallTest(
        os_profile=args.os_profile,
        repo_url=args.repo_url,
        release_type=args.release_type,
        install_prefix=args.install_prefix,
        gfx_arch=args.gfx_arch,
        gpg_key_url=args.gpg_key_url,
    )

    print("\n" + "=" * 80)
    print("INSTALLATION TEST - NATIVE LINUX PACKAGES")
    print("=" * 80)
    print(f"Release Type: {test_runner.release_type.upper()}")
    print(f"Install Prefix: {test_runner.install_prefix}")
    print(f"Test Type: {args.test_type}")
    print("=" * 80)

    try:
        if not test_runner.run_repo_setup_and_install():
            print("\n[FAIL] Step 1 (repo setup and install) failed.")
            return 1
        if not test_runner.run_basic_verification():
            print("\n[FAIL] Step 2 (basic verification) failed.")
            return 1
        if args.test_type == "full":
            if not test_runner.run_full_verification():
                print("\n[FAIL] Step 3 (full verification) failed.")
                return 1
        print("\n" + "=" * 80)
        print("[PASS] INSTALLATION TEST PASSED")
        if args.test_type == "sanity":
            print("(sanity: basic verification completed)")
        else:
            print("ROCm has been successfully installed from repository and verified!")
        print("=" * 80 + "\n")
        return 0
    except Exception as e:
        print(f"\n[FAIL] Error during installation test: {e}")
        traceback.print_exc()
        return 1


def main() -> None:
    """Entry point: parse/validate CLI, then run tests."""
    args = parse_cli_arguments()
    sys.exit(run_tests(args))


if __name__ == "__main__":
    main()
