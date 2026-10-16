#!/usr/bin/env python3
"""Compile and run every implemented solution; report failures.

Usage:
    python3 tools/run_solutions.py [python|java|javascript|csharp|all] [tier]
                                   [--jobs N] [--filter SUBSTR] [--range LO-HI]

- python:     python3 solution.py
- java:       javac into a temp dir, then java Solution
- javascript: node solution.js (uses a local lib fallback shim if present)
- csharp:     Roslyn (csc.dll) compile straight against the SDK's targeting pack,
              then dotnet Solution.dll (see tools/csbuild.py); falls back to a
              throwaway `dotnet build` project when no targeting pack is present

`tier` is one of beginner, intermediate, advanced, expert, master and is matched
against any path component, so it works on Windows as well as POSIX.

Every solution must exit 0 and print `PASS`. A solution is only *skipped* when the
language toolchain itself is missing (no python/java/node/dotnet) — never because
of anything a solution prints. Exits non-zero when any solution fails.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csbuild  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FALLBACK = Path("/tmp/simd33lib")
TIERS = ("beginner", "intermediate", "advanced", "expert", "master")
LANGS = ("python", "java", "javascript", "csharp")
LANG_FILES = {
    "python": "solution.py",
    "java": "Solution.java",
    "javascript": "solution.js",
    "csharp": "Solution.cs",
}


PASS = "pass"
FAIL = "fail"
SKIP = "skip"


def base_env() -> dict:
    env = os.environ.copy()
    if FALLBACK.exists():
        env.setdefault("DYLD_FALLBACK_LIBRARY_PATH", str(FALLBACK))
    return env


def run_python(folder: Path) -> tuple[str, str]:
    r = subprocess.run([sys.executable, "solution.py"], cwd=folder,
                       capture_output=True, text=True, timeout=300)
    out = (r.stdout + r.stderr).strip()
    return (PASS if r.returncode == 0 and "PASS" in r.stdout else FAIL), out


def run_java(folder: Path) -> tuple[str, str]:
    if shutil.which("javac") is None or shutil.which("java") is None:
        return SKIP, "java not found — skipped"
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(["javac", "-d", tmp, "Solution.java"], cwd=folder,
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            return FAIL, r.stderr.strip()
        r = subprocess.run(["java", "-cp", tmp, "Solution"], cwd=folder,
                           capture_output=True, text=True, timeout=300)
        out = (r.stdout + r.stderr).strip()
        return (PASS if r.returncode == 0 and "PASS" in r.stdout else FAIL), out


def run_node(folder: Path) -> tuple[str, str]:
    if shutil.which("node") is None:
        return SKIP, "node not found — skipped"
    r = subprocess.run(["node", "solution.js"], cwd=folder, env=base_env(),
                       capture_output=True, text=True, timeout=300)
    out = (r.stdout + r.stderr).strip()
    return (PASS if r.returncode == 0 and "PASS" in r.stdout else FAIL), out


class CSharpRunner:
    """Compiles with Roslyn into a per-thread work dir, falling back to MSBuild."""

    def __init__(self) -> None:
        self.dotnet = csbuild.find_dotnet()
        self.env = csbuild.base_env()
        self.local = threading.local()
        self.toolchain: tuple[Path, Path, str] | None = None
        if self.dotnet is not None:
            try:
                self.toolchain = csbuild.toolchain(self.dotnet)
            except RuntimeError:
                self.toolchain = None

    def work_dir(self) -> Path:
        work = getattr(self.local, "work", None)
        if work is None:
            work = Path(tempfile.mkdtemp(prefix="csrun-"))
            self.local.work = work
        return work

    def __call__(self, folder: Path) -> tuple[str, str]:
        if self.dotnet is None:
            return SKIP, "dotnet not available — skipped"
        source = folder / "Solution.cs"
        try:
            if self.toolchain is not None:
                csc, ref_dir, tfm = self.toolchain
                ok, out = csbuild.run_direct(csc, ref_dir, tfm, source, self.work_dir(),
                                             self.dotnet, self.env)
            else:
                ok, out = csbuild.run_with_msbuild(self.dotnet, "net8.0", source,
                                                   self.work_dir(), self.env)
        except subprocess.TimeoutExpired:
            return FAIL, "timed out"
        return (PASS if ok else FAIL), out


def build_runners() -> dict:
    return {
        "python": run_python,
        "java": run_java,
        "javascript": run_node,
        "csharp": CSharpRunner(),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run every implemented solution.")
    p.add_argument("lang", nargs="?", default="all", choices=[*LANGS, "all"])
    p.add_argument("tier", nargs="?", default=None, choices=TIERS)
    p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 2),
                   help="parallel workers (default: min(8, cpu count))")
    p.add_argument("--filter", default=None, help="only paths containing this substring")
    p.add_argument("--range", dest="range_", default=None, metavar="LO-HI",
                   help="only tasks numbered LO..HI, e.g. --range 601-650")
    return p.parse_args(argv[1:])


def parse_range(spec: str | None) -> tuple[int, int] | None:
    if spec is None:
        return None
    m = re.fullmatch(r"(\d+)-(\d+)", spec.strip())
    if not m:
        raise SystemExit(f"--range expects LO-HI, got {spec!r}")
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi:
        raise SystemExit(f"--range LO must be <= HI, got {spec!r}")
    return lo, hi


TASK_NUMBER = re.compile(r"^(\d{3})-")


def task_number(path: Path) -> int | None:
    for part in path.parts:
        m = TASK_NUMBER.match(part)
        if m:
            return int(m.group(1))
    return None


def collect(lang: str, tier: str | None, filt: str | None,
            number_range: tuple[int, int] | None) -> list[Path]:
    root = ROOT / "solutions" / lang
    if not root.exists():
        return []
    matches = sorted(root.rglob(LANG_FILES[lang]))
    if tier:
        matches = [f for f in matches if tier in f.relative_to(root).parts]
    if number_range:
        lo, hi = number_range
        matches = [f for f in matches
                   if (n := task_number(f.relative_to(root))) is not None and lo <= n <= hi]
    if filt:
        matches = [f for f in matches if filt in str(f)]
    return matches


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    number_range = parse_range(args.range_)
    runners = build_runners()
    langs = list(LANGS) if args.lang == "all" else [args.lang]

    jobs = [(lang, path.parent) for lang in langs
            for path in collect(lang, args.tier, args.filter, number_range)]

    stats = {"pass": 0, "fail": 0, "skip": 0}
    lock = threading.Lock()

    def execute(job: tuple[str, Path]) -> tuple[str, Path, str, str]:
        lang, folder = job
        try:
            status, out = runners[lang](folder)
        except Exception as exc:  # noqa: BLE001 - a runner crash is a failure
            return lang, folder, FAIL, f"{type(exc).__name__}: {exc}"
        return lang, folder, status, out

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for lang, folder, status, out in pool.map(execute, jobs):
            with lock:
                stats[status] += 1
            if status == FAIL:
                print(f"FAILED [{lang}] {folder.relative_to(ROOT)}")
                print("  " + out.replace("\n", "\n  ")[:4000])

    print(f"ran {len(jobs)} solutions, {stats['fail']} failed, "
          f"{stats['skip']} skipped, {stats['pass']} passed")
    return 1 if stats["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
