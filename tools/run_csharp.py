#!/usr/bin/env python3
"""Compile and run the C# solutions with the .NET SDK.

Each solution is a bare `Solution.cs` (class `Solution` with `Main`); a solution
passes when it exits 0 and prints `PASS`. Compilation goes through the Roslyn
compiler straight against the SDK's targeting pack (see tools/csbuild.py), which
is ~10x faster than a `dotnet build` per file; when no targeting pack is present
it falls back to a throwaway MSBuild project.

Usage:
    python3 tools/run_csharp.py [path-filter] [--jobs N]

Requires the .NET SDK on PATH or at ~/.dotnet/dotnet (install with:
    bash dotnet-install.sh --channel 8.0 --install-dir ~/.dotnet)
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csbuild  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Compile and run the C# solutions.")
    parser.add_argument("filter", nargs="?", default=None,
                        help="only run solutions whose path contains this substring")
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 2))
    args = parser.parse_args(argv[1:])

    dotnet = csbuild.find_dotnet()
    if dotnet is None:
        print("dotnet not found — skipping C# validation")
        return 0

    env = csbuild.base_env()
    try:
        csc, ref_dir, tfm = csbuild.toolchain(dotnet)
    except RuntimeError:
        csc, ref_dir, tfm = None, None, "net8.0"
        print("no targeting pack found — falling back to per-file `dotnet build`")

    files = sorted((ROOT / "solutions" / "csharp").rglob("Solution.cs"))
    if args.filter:
        files = [f for f in files if args.filter in str(f)]

    local = threading.local()

    def work_dir() -> Path:
        work = getattr(local, "work", None)
        if work is None:
            work = Path(tempfile.mkdtemp(prefix="csrun-"))
            local.work = work
        return work

    def run(f: Path) -> tuple[bool, str]:
        if csc is not None and ref_dir is not None:
            return csbuild.run_direct(csc, ref_dir, tfm, f, work_dir(), dotnet, env)
        return csbuild.run_with_msbuild(dotnet, tfm, f, work_dir(), env)

    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        for f, (ok, out) in zip(files, pool.map(run, files)):
            if not ok:
                failed += 1
                print(f"FAILED {f.relative_to(ROOT)}")
                print("  " + out.replace("\n", "\n  ")[:1500])
    print(f"ran {len(files)} C# solutions, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
