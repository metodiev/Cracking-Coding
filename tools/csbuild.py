#!/usr/bin/env python3
"""Compile and run a single C# `Solution.cs` outside MSBuild.

Every solution is a bare `Solution.cs` with a `Main` that prints `PASS`, so a full
`dotnet build` per file is mostly MSBuild overhead. When a .NET targeting pack is
available we instead invoke the Roslyn compiler (`csc.dll`, shipped inside the SDK)
directly against the `Microsoft.NETCore.App.Ref` reference assemblies and run the
produced assembly with `dotnet`. That is ~10x faster (0.5s vs 4.5s per solution).

When no targeting pack is found the caller can fall back to a throwaway MSBuild
project — see `build_project_csproj()` and `run_with_msbuild()`.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CSPROJ = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>{tfm}</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>disable</ImplicitUsings>
    <AssemblyName>Runner</AssemblyName>
    <InvariantGlobalization>true</InvariantGlobalization>
  </PropertyGroup>
</Project>
"""

# Preferred reference packs, best first: the lowest stable TFMs have the widest
# runtime availability (a CI image usually only carries the LTS runtime).
PREFERRED_TFMS = ["net8.0", "net9.0", "net10.0", "net11.0"]


def base_env() -> dict:
    env = os.environ.copy()
    env.update(
        DOTNET_CLI_TELEMETRY_OPTOUT="1",
        DOTNET_NOLOGO="1",
        DOTNET_SKIP_FIRST_TIME_EXPERIENCE="1",
        DOTNET_CLI_HOME=tempfile.gettempdir(),
    )
    return env


def find_dotnet() -> str | None:
    found = shutil.which("dotnet")
    if found:
        return found
    for fallback in (Path.home() / ".dotnet" / "dotnet",
                     Path.home() / ".dotnet" / "dotnet.exe"):
        if fallback.exists():
            return str(fallback)
    return None


def _version_key(version: str) -> tuple:
    release = "-" not in version
    parts = re.split(r"[.\-]", version)
    numbers = [int(p) if p.isdigit() else 0 for p in parts if p]
    return (1 if release else 0, numbers)


def sdk_roots(dotnet: str, env: dict) -> list[tuple[str, Path]]:
    """(version, dotnet root) pairs from `dotnet --list-sdks`, newest release first."""
    try:
        r = subprocess.run([dotnet, "--list-sdks"], capture_output=True, text=True,
                           timeout=120, env=env)
    except (OSError, subprocess.SubprocessError):
        return []
    roots: list[tuple[str, Path]] = []
    for line in r.stdout.splitlines():
        m = re.match(r"^(\S+)\s+\[(.+)\]\s*$", line.strip())
        if m:
            roots.append((m.group(1), Path(m.group(2)).parent))
    roots.sort(key=lambda item: _version_key(item[0]), reverse=True)
    return roots


def find_csc(dotnet: str, env: dict) -> Path | None:
    for version, root in sdk_roots(dotnet, env):
        csc = root / "sdk" / version / "Roslyn" / "bincore" / "csc.dll"
        if csc.exists():
            return csc
    return None


def find_ref_pack(dotnet: str, env: dict) -> tuple[Path, str] | None:
    """Newest usable targeting pack -> (ref-assembly dir, TFM)."""
    for _version, root in sdk_roots(dotnet, env):
        packs = root / "packs" / "Microsoft.NETCore.App.Ref"
        if not packs.is_dir():
            continue
        candidates: list[tuple[tuple, str, Path]] = []
        for pack in packs.iterdir():
            ref_root = pack / "ref"
            if not ref_root.is_dir():
                continue
            for tfm_dir in ref_root.iterdir():
                if tfm_dir.is_dir() and any(tfm_dir.glob("*.dll")):
                    candidates.append((_version_key(pack.name), tfm_dir.name, tfm_dir))
        if not candidates:
            continue
        preferences = {tfm: i for i, tfm in enumerate(PREFERRED_TFMS)}
        candidates.sort(key=lambda item: (preferences.get(item[1], len(PREFERRED_TFMS)),
                                          [-n for n in item[0][1]]))
        _key, tfm, path = candidates[0]
        return path, tfm
    return None


def _write_response_file(dest: Path, csc_args: list[str]) -> None:
    # csc resolves relative paths against the process CWD, so the file is written
    # into the work dir and csc is invoked with that dir as its CWD.
    dest.write_text("\n".join(csc_args) + "\n", encoding="utf-8")


def _runtimeconfig(tfm: str) -> str:
    m = re.match(r"net(\d+)", tfm)
    major = m.group(1) if m else "8"
    return json.dumps({
        "runtimeOptions": {
            "tfm": tfm,
            "framework": {
                "name": "Microsoft.NETCore.App",
                "version": f"{major}.0.0",
            },
        },
    })


def run_direct(csc: Path, ref_dir: Path, tfm: str, source: Path, work: Path,
               dotnet: str, env: dict, compile_timeout: int = 180,
               run_timeout: int = 180) -> tuple[bool, str]:
    """Roslyn compile + `dotnet` run inside `work`. Returns (ok, output)."""
    work.mkdir(parents=True, exist_ok=True)
    target = work / "Solution.cs"
    shutil.copyfile(source, target)
    args = ["-nologo", "-nostdlib+", "-target:exe", "-langversion:latest",
            "-out:Solution.dll", "-warnaserror-"]
    args += [f'-r:"{dll}"' for dll in sorted(ref_dir.glob("*.dll"))]
    args.append("Solution.cs")
    _write_response_file(work / "Solution.rsp", args)
    (work / "Solution.runtimeconfig.json").write_text(_runtimeconfig(tfm), encoding="utf-8")

    build = subprocess.run([dotnet, str(csc), "@Solution.rsp"], cwd=work, env=env,
                           capture_output=True, text=True, timeout=compile_timeout)
    if build.returncode != 0:
        return False, (build.stdout + build.stderr).strip()
    run = subprocess.run([dotnet, "Solution.dll"], cwd=work, env=env,
                         capture_output=True, text=True, timeout=run_timeout)
    out = (run.stdout + run.stderr).strip()
    return (run.returncode == 0 and "PASS" in run.stdout), out


def run_with_msbuild(dotnet: str, tfm: str, source: Path, work: Path,
                     env: dict, timeout: int = 600) -> tuple[bool, str]:
    """Fallback path: throwaway console project + `dotnet build`."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "Runner.csproj").write_text(CSPROJ.format(tfm=tfm), encoding="utf-8")
    shutil.copyfile(source, work / "Solution.cs")
    build = subprocess.run([dotnet, "build", "-v", "q", "--nologo", "-o", "bin"],
                           cwd=work, env=env, capture_output=True, text=True, timeout=timeout)
    if build.returncode != 0:
        return False, (build.stdout + build.stderr).strip()
    run = subprocess.run([dotnet, os.path.join("bin", "Runner.dll")], cwd=work, env=env,
                         capture_output=True, text=True, timeout=180)
    out = (run.stdout + run.stderr).strip()
    return (run.returncode == 0 and "PASS" in run.stdout), out


def toolchain(dotnet: str) -> tuple[Path, Path, str]:
    """(csc.dll, ref dir, tfm) — raises RuntimeError when unavailable."""
    env = base_env()
    csc = find_csc(dotnet, env)
    pack = find_ref_pack(dotnet, env)
    if csc is None or pack is None:
        raise RuntimeError("no Roslyn compiler or targeting pack found")
    ref_dir, tfm = pack
    return csc, ref_dir, tfm


def main(argv: list[str]) -> int:
    """Debug entry point: python tools/csbuild.py <path/to/Solution.cs>"""
    if len(argv) != 2:
        print(__doc__)
        return 2
    dotnet = find_dotnet()
    if dotnet is None:
        print("dotnet not found")
        return 2
    source = Path(argv[1]).resolve()
    env = base_env()
    work = Path(tempfile.mkdtemp(prefix="csbuild-"))
    try:
        try:
            csc, ref_dir, tfm = toolchain(dotnet)
        except RuntimeError:
            ok, out = run_with_msbuild(dotnet, "net8.0", source, work, env)
        else:
            print(f"using {tfm} ref pack at {ref_dir}")
            ok, out = run_direct(csc, ref_dir, tfm, source, work, dotnet, env)
        print(out)
        return 0 if ok else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
