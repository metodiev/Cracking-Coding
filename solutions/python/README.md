# Python Solutions

One folder per task: `solutions/python/<tier>/NNN-slug/solution.py`.

## Conventions

- A single file per task: functions/classes plus a `main()` with `assert`-based self-checks.
- Standard library only; tested with Python 3.9+.
- No global state; every file runs standalone.

## Run

```bash
cd solutions/python/beginner/001-reverse-a-string
python3 solution.py     # prints PASS on success
```

Batch-run all implemented tasks from the repo root:

```bash
find solutions/python -name solution.py -print0 | while IFS= read -r -d '' f; do
  python3 "$f" >/dev/null || echo "FAILED: $f"
done
echo done
```

## Coverage

Tier folders are created as tasks are implemented (staged delivery — see the root README).
Task specs live in [`tasks/`](../../tasks) and link back here once a solution exists.

See also: [Python companion tutorial](../../tutorials/languages/python.md).
