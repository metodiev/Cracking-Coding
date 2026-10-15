# JavaScript Solutions

One folder per task: `solutions/javascript/<tier>/NNN-slug/solution.js`.

## Conventions

- CommonJS (`module.exports`) so plain `node solution.js` works without config.
- Self-checks with `node:assert`, guarded by `if (require.main === module)`.
- No external packages.

## Run

```bash
cd solutions/javascript/beginner/001-reverse-a-string
node solution.js      # prints PASS on success
```

Batch-run all implemented tasks from the repo root:

```bash
find solutions/javascript -name solution.js -print0 | while IFS= read -r -d '' f; do
  node "$f" >/dev/null || echo "FAILED: $f"
done
echo done
```

## Coverage

Tier folders are created as tasks are implemented (staged delivery — see the root README).
Task specs live in [`tasks/`](../../tasks) and link back here once a solution exists.

See also: [JavaScript companion tutorial](../../tutorials/languages/javascript.md).
