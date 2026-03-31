#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Install ocl-icd-devel on non-SLES RPM hosts: **bundle** (Rocky RPMs + dnf) or **partners**
(RHEL partners .repo + dnf).

Examples (run from ``build_tools/packaging/linux`` or set ``PYTHONPATH``)::

    # Bundle: download only the ocl-icd-devel RPM to ./rpms (default Rocky CRB URL)
    python3 download_ocl_icd_devel_rpm.py -o ./rpms

    # Bundle: download all three RPMs (temp dir) and install with dnf (needs root)
    sudo python3 download_ocl_icd_devel_rpm.py --mode bundle --install

    # Bundle: custom devel RPM URL, then install
    sudo python3 download_ocl_icd_devel_rpm.py --install --url https://example/ocl-icd-devel-2.2.13-4.el9.x86_64.rpm

    # Partners: write RHEL-partners.repo and dnf install ocl-icd-devel (internal network)
    sudo python3 download_ocl_icd_devel_rpm.py --mode partners --install

    sudo python3 download_ocl_icd_devel_rpm.py --mode partners --install \\
        --rhel-partners-release 9.4 --yum-repos-dir /etc/yum.repos.d

Environment (optional)::

    ROCM_OCL_ICD_INSTALL_MODE=bundle|partners
    ROCM_OCL_ICD_RUNTIME_RPM_URL / ROCM_OCL_ICD_OPENCL_HEADERS_RPM_URL / ROCM_OCL_ICD_DEVEL_RPM_URL
    ROCM_RHEL_PARTNERS_RELEASE  ROCM_YUM_REPOS_DIR

See also ``python3 download_ocl_icd_devel_rpm.py --help``."""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Literal

# --- defaults (override with ROCM_* env vars listed in --help) ---

ROCKY = "https://dl.rockylinux.org/pub/rocky/9"


# Environment variable helper: non-empty ROCM_* value wins, else default.
def _env(key: str, default: str) -> str:
    v = os.environ.get(key, "").strip()
    return v if v else default


# CPU arch string used in Rocky RPM paths (x86_64 vs aarch64).
def _rpm_arch() -> str:
    m = platform.machine().lower()
    return "aarch64" if m in ("aarch64", "arm64") else "x86_64"


# Full URL to one RPM under Rocky 9 AppStream or CRB (kind = "AppStream" | "CRB").
def _rocky_url(kind: str, filename: str) -> str:
    a = _rpm_arch()
    return f"{ROCKY}/{kind}/{a}/os/Packages/o/{filename}"


OCL_ICD_RUNTIME_RPM_URL = _env(
    "ROCM_OCL_ICD_RUNTIME_RPM_URL",
    _rocky_url("AppStream", f"ocl-icd-2.2.13-4.el9.{_rpm_arch()}.rpm"),
)
OCL_ICD_OPENCL_HEADERS_RPM_URL = _env(
    "ROCM_OCL_ICD_OPENCL_HEADERS_RPM_URL",
    f"{ROCKY}/AppStream/x86_64/os/Packages/o/"
    "opencl-headers-3.0-6.20201007gitd65bcc5.el9.0.1.noarch.rpm",
)
OCL_ICD_DEVEL_RPM_URL = _env(
    "ROCM_OCL_ICD_DEVEL_RPM_URL",
    _rocky_url("CRB", f"ocl-icd-devel-2.2.13-4.el9.{_rpm_arch()}.rpm"),
)

RHEL_PARTNERS_RELEASE = _env("ROCM_RHEL_PARTNERS_RELEASE", "9.4")
RHEL_PARTNERS_REPO_BASENAME = "RHEL-partners.repo"
YUM_REPOS_DIR = _env("ROCM_YUM_REPOS_DIR", "/etc/yum.repos.d")
ARTIFACTORY_RHEL_ROOT = "http://scvartifactory.amd.com/artifactory/list/rhel-remote"

DOWNLOAD_TIMEOUT_SEC = 300
DNF_TIMEOUT_SEC = 1800
# dnf/config-manager sub-steps (CRB enable path; seconds)
DNF_PLUGIN_INSTALL_TIMEOUT_SEC = 300
DNF_CONFIG_MANAGER_TIMEOUT_SEC = 120
DNF_MAKECACHE_TIMEOUT_SEC = 300

OCL_ICD_DEVEL_DOWNLOAD_TIMEOUT_SEC = DOWNLOAD_TIMEOUT_SEC
OCL_DNF_INSTALL_TIMEOUT_SEC = DNF_TIMEOUT_SEC


# Resolve install strategy: Rocky RPM download (bundle) vs AMD artifactory partners repo (partners).
def install_mode(mode: str | None) -> Literal["bundle", "partners"]:
    """Return ``bundle`` or ``partners`` (arg or ``ROCM_OCL_ICD_INSTALL_MODE``)."""
    m = (mode or _env("ROCM_OCL_ICD_INSTALL_MODE", "bundle")).strip().lower()
    if m in ("bundle", "rocky", "mirror", "download"):
        return "bundle"
    if m in ("partners", "partner", "rhel-partners", "rhel_partners"):
        return "partners"
    raise ValueError(
        "mode must be 'bundle' or 'partners' (or ROCM_OCL_ICD_INSTALL_MODE)"
    )


# Run a subprocess with merged stdout/stderr streamed to the console; return exit code.
def _stream_cmd(cmd: list[str], timeout_sec: int) -> int:
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        # stdout is set when using PIPE (below).
        for line in proc.stdout:
            print(line.rstrip())
            sys.stdout.flush()
        return proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise


# On EL9 only: try to enable CRB/codeready so dnf can resolve deps for a single local devel RPM.
def _enable_el9_crb_if_needed() -> None:
    try:
        ver = (
            Path("/etc/os-release").read_text(encoding="utf-8", errors="ignore").lower()
        )
    except OSError:
        return
    if "el9" not in ver and 'version_id="9' not in ver and "version_id=9" not in ver:
        return
    subprocess.run(
        ["dnf", "install", "-y", "dnf-plugins-core"],
        check=False,
        timeout=DNF_PLUGIN_INSTALL_TIMEOUT_SEC,
    )
    for rid in (
        "crb",
        "powertools",
        "codeready-builder-for-rhel-9-x86_64-rpms",
        "codeready-builder-for-rhel-9-aarch64-rpms",
    ):
        r = subprocess.run(
            ["dnf", "config-manager", "--set-enabled", rid],
            capture_output=True,
            text=True,
            timeout=DNF_CONFIG_MANAGER_TIMEOUT_SEC,
        )
        if r.returncode == 0:
            break
    subprocess.run(
        ["dnf", "makecache"],
        check=False,
        timeout=DNF_MAKECACHE_TIMEOUT_SEC,
    )


# True if this host is not SUSE SLES (OpenCL RPM path is for dnf/yum distros, not zypper SLES).
def is_non_sles_rpm_platform() -> bool:
    try:
        for line in (
            Path("/etc/os-release")
            .read_text(encoding="utf-8", errors="ignore")
            .splitlines()
        ):
            if line.startswith("ID="):
                return line.split("=", 1)[1].strip().strip('"').lower() != "sles"
    except OSError:
        pass
    return True


# Text for yum .repo file with BaseOS, AppStream, and CRB from AMD RHEL partners mirror.
def rhel_partners_repo_body(release_ver: str) -> str:
    root = ARTIFACTORY_RHEL_ROOT
    return (
        f"[RHEL-partners-baseos]\n"
        f"name=Red Hat Enterprise Linux {release_ver} Partners - $basearch\n"
        f"baseurl={root}/{release_ver}/$basearch/os/BaseOS\n"
        "enabled=1\ngpgcheck=0\n\n"
        f"[RHEL-partners-appstream]\n"
        f"name=Red Hat Enterprise Linux {release_ver} Partners (AppStream) - $basearch\n"
        f"baseurl={root}/{release_ver}/$basearch/os/AppStream\n"
        "enabled=1\ngpgcheck=0\n\n"
        "[RHEL-partners-crb]\n"
        "name=Red Hat Enterprise Linux $releasever Partners (CRB) - $basearch\n"
        f"baseurl={root}/{release_ver}/$basearch/os/CRB\n"
        "enabled=1\ngpgcheck=0\n"
    )


# Write partners repo file under yum.repos.d, then dnf install ocl-icd-devel from those repos.
def install_via_rhel_partners_repo(
    yum_repos_dir: str | Path | None = None,
    release_ver: str | None = None,
) -> bool:
    if not is_non_sles_rpm_platform():
        return True
    rel = (release_ver or RHEL_PARTNERS_RELEASE).strip() or "9.4"
    repo_dir = Path(yum_repos_dir or YUM_REPOS_DIR)
    repo_file = repo_dir / RHEL_PARTNERS_REPO_BASENAME
    body = rhel_partners_repo_body(rel)

    print("\n" + "=" * 80 + "\nRHEL partners repo → ocl-icd-devel\n" + "=" * 80)
    print(f"release={rel}\n{repo_file}\n")
    try:
        repo_dir.mkdir(parents=True, exist_ok=True)
        repo_file.write_text(body, encoding="utf-8")
        print(body)
    except OSError as e:
        print(f"[FAIL] write repo: {e}", file=sys.stderr)
        return False

    print("dnf install ocl-icd-devel …")
    try:
        if _stream_cmd(["dnf", "install", "-y", "ocl-icd-devel"], DNF_TIMEOUT_SEC) != 0:
            return False
    except (subprocess.TimeoutExpired, OSError) as e:
        print(f"[FAIL] dnf: {e}", file=sys.stderr)
        return False
    print("[PASS] ocl-icd-devel installed")
    return True


# Fetch one .rpm from HTTP(S) into dest_dir; validates non-empty file.
def download_rpm(
    url: str, dest: Path, *, timeout_sec: int = DOWNLOAD_TIMEOUT_SEC
) -> Path:
    url = url.strip()
    if not url:
        raise RuntimeError("empty RPM URL")
    dest.mkdir(parents=True, exist_ok=True)
    name = url.rsplit("/", 1)[-1]
    if not name.endswith(".rpm"):
        name = "package.rpm"
    out = dest / name
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as r:
            out.write_bytes(r.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"download failed: {e}") from e
    if not out.is_file() or out.stat().st_size == 0:
        out.unlink(missing_ok=True)
        raise RuntimeError("download empty or missing")
    return out


# Download the three Rocky EL9 RPMs (runtime, headers, devel) for offline-capable dnf install.
def download_el9_bundle_rpms(
    dest_dir: Path | str | None = None,
    *,
    devel_url: str | None = None,
    timeout_sec: int = DOWNLOAD_TIMEOUT_SEC,
) -> list[Path]:
    if not is_non_sles_rpm_platform():
        raise RuntimeError("not for SLES hosts")
    d = Path(dest_dir or os.getcwd()).resolve()
    u_devel = (devel_url or OCL_ICD_DEVEL_RPM_URL).strip()
    return [
        download_rpm(OCL_ICD_RUNTIME_RPM_URL, d, timeout_sec=timeout_sec),
        download_rpm(OCL_ICD_OPENCL_HEADERS_RPM_URL, d, timeout_sec=timeout_sec),
        download_rpm(u_devel, d, timeout_sec=timeout_sec),
    ]


# Download only the ocl-icd-devel RPM (e.g. for mirroring); does not install.
def download_ocl_icd_devel_rpm(
    dest_dir: Path | str | None = None,
    url: str | None = None,
    *,
    timeout_sec: int = DOWNLOAD_TIMEOUT_SEC,
) -> Path:
    if not is_non_sles_rpm_platform():
        raise RuntimeError("not for SLES hosts")
    return download_rpm(
        (url or OCL_ICD_DEVEL_RPM_URL).strip(),
        Path(dest_dir or os.getcwd()).resolve(),
        timeout_sec=timeout_sec,
    )


# Single dnf transaction installing multiple local RPM paths (bundle install path).
def install_local_rpms(
    rpm_paths: list[Path], *, timeout_sec: int = DNF_TIMEOUT_SEC
) -> None:
    paths = [p.resolve() for p in rpm_paths]
    for p in paths:
        if not p.is_file():
            raise RuntimeError(f"missing RPM: {p}")
    rc = _stream_cmd(["dnf", "install", "-y"] + [str(p) for p in paths], timeout_sec)
    if rc != 0:
        raise RuntimeError(f"dnf install failed ({rc})")


# Install one ocl-icd-devel RPM from disk; may enable CRB on EL9 so deps resolve from repos.
def install_single_devel_rpm(path: Path, *, timeout_sec: int = DNF_TIMEOUT_SEC) -> None:
    p = path.resolve()
    if not p.is_file():
        raise RuntimeError(f"missing RPM: {p}")
    _enable_el9_crb_if_needed()
    rc = _stream_cmd(["dnf", "install", "-y", str(p)], timeout_sec)
    if rc != 0:
        raise RuntimeError(
            f"dnf install failed ({rc}); try bundle install on unregistered RHEL"
        )


# Full bundle flow: download three Rocky RPMs, copy to temp paths, dnf install, cleanup.
def download_and_install_bundle(rpm_url: str | None = None) -> bool:
    """Rocky EL9 RPMs in one dnf transaction (unregistered RHEL)."""
    if not is_non_sles_rpm_platform():
        return True
    tmp = Path(tempfile.mkdtemp(prefix="rocm-ocl-icd-"))
    staged: list[Path] = []
    try:
        bundle = download_el9_bundle_rpms(dest_dir=tmp, devel_url=rpm_url)
        for src in bundle:
            dst = Path(tempfile.gettempdir()) / f"rocm-oclicd-{os.getpid()}-{src.name}"
            shutil.copy2(src, dst)
            staged.append(dst)
        print("dnf install (local RPMs):", ", ".join(str(s) for s in staged))
        install_local_rpms(staged)
        print("[PASS] ocl-icd stack installed")
        return True
    except RuntimeError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for s in staged:
            s.unlink(missing_ok=True)


# Entry point: choose bundle (download+install) or partners (repo file + dnf) from mode/env.
def install_ocl_icd_devel_stack(
    *,
    mode: str | None = None,
    rpm_url: str | None = None,
    yum_repos_dir: str | Path | None = None,
    release_ver: str | None = None,
) -> bool:
    if not is_non_sles_rpm_platform():
        return True
    try:
        m = install_mode(mode)
    except ValueError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return False
    if m == "partners":
        return install_via_rhel_partners_repo(yum_repos_dir, release_ver)
    return download_and_install_bundle(rpm_url)


CLI_EPILOG = """
examples:
  %(prog)s -o ./rpms
                        Download ocl-icd-devel RPM only (bundle) into ./rpms
  sudo %(prog)s --mode bundle --install
                        Bundle: download Rocky stack (temp dir) + dnf install
  sudo %(prog)s --install --url <devel-rpm-url>
                        Bundle with custom ocl-icd-devel RPM URL
  sudo %(prog)s --mode partners --install
                        Partners: write RHEL-partners.repo + dnf install ocl-icd-devel
  sudo %(prog)s --mode partners --install --rhel-partners-release 9.4
                        Partners with release path segment (see ROCM_RHEL_PARTNERS_RELEASE)
  ROCM_OCL_ICD_INSTALL_MODE=partners sudo %(prog)s --install
                        Same as --mode partners (env overrides when --mode omitted)
"""


# CLI: --install runs install_ocl_icd_devel_stack; without --install, download devel RPM only (bundle).
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="ocl-icd-devel: bundle (Rocky RPMs + dnf) or partners (RHEL partners .repo + dnf).",
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("bundle", "partners"),
        default=None,
        help="default: env or bundle",
    )
    parser.add_argument(
        "-o",
        "--dest-dir",
        type=Path,
        default=Path.cwd(),
        help="download dir (bundle, no --install)",
    )
    parser.add_argument("--url", default=None, help="ocl-icd-devel URL (bundle)")
    parser.add_argument(
        "--yum-repos-dir", default=None, help="partners: repo directory"
    )
    parser.add_argument(
        "--rhel-partners-release", default=None, help="partners: e.g. 9.4"
    )
    parser.add_argument(
        "--install", action="store_true", help="run install for selected mode"
    )
    args = parser.parse_args(argv)

    if args.install:
        ok = install_ocl_icd_devel_stack(
            mode=args.mode,
            rpm_url=args.url,
            yum_repos_dir=args.yum_repos_dir,
            release_ver=args.rhel_partners_release,
        )
        return 0 if ok else 1

    try:
        if install_mode(args.mode) == "partners":
            print("partners mode needs --install", file=sys.stderr)
            return 2
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    try:
        print(download_ocl_icd_devel_rpm(dest_dir=args.dest_dir, url=args.url))
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
