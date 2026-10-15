# C# Solutions

One folder per task: `solutions/csharp/<tier>/NNN-slug/Solution.cs`.

## Conventions

- A single file: `public static class Solution` with static methods and a `Main` that runs self-checks.
- `using System;` + explicit `using` for collections/LINQ — plain C#, no packages.
- Target: modern C# (top-level-friendly, .NET 6+), avoiding features that break older compilers.

## Run

This machine may not have the .NET SDK installed; the canonical way to run any solution:

```bash
mkdir -p /tmp/scratch && cd /tmp/scratch
dotnet new console -n Runner --force
cp /path/to/solutions/csharp/beginner/001-reverse-a-string/Solution.cs Runner/
rm Runner/Program.cs
dotnet run --project Runner      # prints PASS on success
```

Or paste the file content into https://dotnetfiddle.net / a local IDE.

## Coverage

Tier folders are created as tasks are implemented (staged delivery — see the root README).
Task specs live in [`tasks/`](../../tasks) and link back here once a solution exists.

See also: [C# companion tutorial](../../tutorials/languages/csharp.md).
