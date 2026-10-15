# Java Solutions

One folder per task: `solutions/java/<tier>/NNN-slug/Solution.java`.

## Conventions

- A single file, `public class Solution`, static methods, plus a `main` that runs self-checks.
- No external dependencies — plain JDK (8+ compatible; tested with JDK 25).
- Self-checks throw `AssertionError` on failure so CI-style scripts can detect breakage.

## Run

```bash
cd solutions/java/beginner/001-reverse-a-string
javac Solution.java && java Solution
# prints PASS on success
```

Batch-run all implemented tasks from the repo root:

```bash
find solutions/java -name Solution.java -print0 | while IFS= read -r -d '' f; do
  d=$(dirname "$f")
  (cd "$d" && javac Solution.java && java Solution >/dev/null) || echo "FAILED: $f"
done
echo done
```

## Coverage

Tier folders are created as tasks are implemented (staged delivery — see the root README).
Task specs live in [`tasks/`](../../tasks) and link back here once a solution exists.

See also: [Java companion tutorial](../../tutorials/languages/java.md).
