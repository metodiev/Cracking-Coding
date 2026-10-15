#!/usr/bin/env python3
"""Compile and run the C# solutions with dotnet.

Each solution is a bare `Solution.cs` (class `Solution` with `Main`). This script creates a
throwaway console project in a temp dir, copies solutions in one at a time, and runs them.

Usage:
    python3 tools/run_csharp.py [path-filter]

Requires the .NET SDK on PATH or at ~/.dotnet/dotnet (install with:
    bash dotnet-install.sh --channel 8.0 --install-dir ~/.dotnet)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSPROJ = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>disable</ImplicitUsings>
    <AssemblyName>Runner</AssemblyName>
    <InvariantGlobalization>true</InvariantGlobalization>
  </PropertyGroup>
</Project>
"""


def find_dotnet() -> str | None:
    if shutil.which("dotnet"):
        return "dotnet"
    home = Path.home() / ".dotnet" / "dotnet"
    if home.exists():
        return str(home)
    return None


def main(argv: list[str]) -> int:
    dotnet = find_dotnet()
    if dotnet is None:
        print("dotnet not found — skipping C# validation")
        return 0

    filt = argv[1] if len(argv) > 1 else None
    env = os.environ.copy()
    env.update(
        DOTNET_CLI_TELEMETRY_OPTOUT="1",
        DOTNET_NOLOGO="1",
        DOTNET_SKIP_FIRST_TIME_EXPERIENCE="1",
        DOTNET_CLI_HOME=tempfile.gettempdir(),
        HOME=os.environ.get("HOME", str(Path.home())),
    )

    work = Path(tempfile.mkdtemp(prefix="csrun-"))
    try:
        (work / "Runner.csproj").write_text(CSPROJ, encoding="utf-8")
        # restore/build once to warm up
        subprocess.run([dotnet, "build", "-v", "q", "--nologo"], cwd=work, env=env,
                       capture_output=True, text=True, timeout=600)

        files = sorted((ROOT / "solutions" / "csharp").rglob("Solution.cs"))
        if filt:
            files = [f for f in files if filt in str(f)]
        total = failed = 0
        for f in files:
            total += 1
            shutil.copy(f, work / "Solution.cs")
            build = subprocess.run([dotnet, "build", "-v", "q", "--nologo", "-o", "bin"], cwd=work,
                                   env=env, capture_output=True, text=True, timeout=300)
            if build.returncode != 0:
                failed += 1
                print(f"BUILD FAILED {f.relative_to(ROOT)}")
                print("  " + "\n  ".join(line for line in build.stdout.splitlines()
                                         if "error" in line.lower())[:1500])
                continue
            run = subprocess.run([dotnet, "bin/Runner.dll"], cwd=work, env=env,
                                 capture_output=True, text=True, timeout=120)
            if run.returncode != 0 or "PASS" not in run.stdout:
                failed += 1
                print(f"RUN FAILED {f.relative_to(ROOT)}")
                out = (run.stdout + run.stderr).strip().replace("\n", "\n  ")
                print("  " + out[:1500])
        print(f"ran {total} C# solutions, {failed} failed")
        return 1 if failed else 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
