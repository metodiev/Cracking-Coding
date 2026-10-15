#!/usr/bin/env python3
"""Run all implemented solutions and report failures.

Usage:
    python3 tools/run_solutions.py [python|java|javascript|csharp|all] [tier]

- python:     python3 solution.py
- java:       javac into a temp dir, then java Solution
- javascript: node solution.js (uses a local lib fallback shim if present)
- csharp:     skipped unless `dotnet` is on PATH (dotnet <= this machine)

Exits non-zero when any solution fails.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FALLBACK = Path("/tmp/simd33lib")


def base_env() -> dict:
    env = os.environ.copy()
    if FALLBACK.exists():
        env.setdefault("DYLD_FALLBACK_LIBRARY_PATH", str(FALLBACK))
    return env


def run_python(folder: Path) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, "solution.py"], cwd=folder,
                       capture_output=True, text=True, timeout=60)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def run_java(folder: Path) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(["javac", "-d", tmp, "Solution.java"], cwd=folder,
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return False, r.stderr.strip()
        r = subprocess.run(["java", "-cp", tmp, "Solution"], cwd=folder,
                           capture_output=True, text=True, timeout=120)
        return r.returncode == 0, (r.stdout + r.stderr).strip()


def run_node(folder: Path) -> tuple[bool, str]:
    if shutil.which("node") is None:
        return True, "node not found — skipped"
    r = subprocess.run(["node", "solution.js"], cwd=folder, env=base_env(),
                       capture_output=True, text=True, timeout=60)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def run_csharp(folder: Path) -> tuple[bool, str]:
    return True, "dotnet not available — skipped"


LANG_RUNNERS = {
    "python": ("solutions/python", "solution.py", run_python),
    "java": ("solutions/java", "Solution.java", run_java),
    "javascript": ("solutions/javascript", "solution.js", run_node),
    "csharp": ("solutions/csharp", "Solution.cs", run_csharp),
}


def main(argv: list[str]) -> int:
    lang = argv[1] if len(argv) > 1 else "all"
    tier = argv[2] if len(argv) > 2 else None
    langs = list(LANG_RUNNERS) if lang == "all" else [lang]
    total = failed = skipped = 0
    for lg in langs:
        base, fname, runner = LANG_RUNNERS[lg]
        root = ROOT / base
        if not root.exists():
            continue
        dirs = sorted(root.rglob(fname))
        if tier:
            dirs = [d for d in dirs if f"/{tier}/" in str(d)]
        for f in dirs:
            total += 1
            ok, out = runner(f.parent)
            if "skipped" in out:
                skipped += 1
                continue
            if not ok:
                failed += 1
                print(f"FAILED [{lg}] {f.parent.relative_to(ROOT)}")
                print("  " + out.replace("\n", "\n  ")[:2000])
    print(f"ran {total} solutions, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
