"""Task 988 - Implement cross-repo synchronization and dependency management tool for exercises.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A dependency manager for a fleet of exercise repos.  Exercises declare semver constraints on
each other (`^1.2.0`, `~1.2.0`, `>=1.2.0`, `<2.0.0`, `=1.2.3`); resolution picks the newest
catalog version satisfying every incoming constraint and re-validates each edge, a
deterministic topological order (alphabetical among ready nodes) fixes the sync order, cycles
are reported with their path, and the lockfile is a sorted digest-stable snapshot that a sync
plan diffs.  Simplification: the catalog is an in-memory map (no manifests fetched over the
network), resolution is one greedy pass, the lockfile is a dict.  Verified: constraint
intersection, unsatisfiable/unknown-dependency rejection, topological validity/determinism,
cycle paths, digest stability under permutation, and add/update/remove sync steps.
"""
import hashlib
import json


class UnsatisfiableError(ValueError):
    """No catalog version satisfies every incoming constraint."""


class CycleError(ValueError):
    """The dependency graph contains a cycle."""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_error(action, error_type, needle):
    try:
        action()
    except error_type as exc:
        check(needle in str(exc), "error %r does not mention %r" % (str(exc), needle))
        return
    raise AssertionError("expected %s mentioning %r" % (error_type.__name__, needle))


def parse_version(text):
    parts = text.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError("not a semantic version: %r" % (text,))
    return tuple(int(part) for part in parts)


def format_version(version):
    return "%d.%d.%d" % version


def satisfies(version, spec):
    """Caret allows the same major, tilde the same minor; other specs are plain comparisons."""
    if spec.startswith("^") or spec.startswith("~"):
        low = parse_version(spec[1:])
        high = (low[0] + 1, 0, 0) if spec[0] == "^" else (low[0], low[1] + 1, 0)
        return low <= version < high
    for prefix in (">=", "<=", ">", "<", "="):
        if spec.startswith(prefix):
            bound = parse_version(spec[len(prefix):])
            return {">=": version >= bound, "<=": version <= bound, ">": version > bound,
                    "<": version < bound, "=": version == bound}[prefix]
    return version == parse_version(spec)


def resolve(names, catalog, deps):
    """Newest catalog version per exercise satisfying every incoming constraint."""
    incoming = {name: [] for name in names}
    for name in names:
        for dependency, spec in deps.get(name, ()):
            if dependency not in incoming:
                raise ValueError("exercise %r depends on unknown exercise %r" % (name, dependency))
            incoming[dependency].append((name, spec))
    resolved = {}
    for name in names:
        candidates = sorted((parse_version(v) for v in catalog[name]["versions"]), reverse=True)
        chosen = next((c for c in candidates
                       if all(satisfies(c, spec) for _, spec in incoming[name])), None)
        if chosen is None:
            wanted = ", ".join("%s wants %s" % (src, spec) for src, spec in incoming[name])
            raise UnsatisfiableError("no version of %s satisfies [%s]" % (name, wanted))
        resolved[name] = chosen
    for name in names:  # validation pass: every edge must hold on the chosen combination
        for dependency, spec in deps.get(name, ()):
            if not satisfies(resolved[dependency], spec):
                raise UnsatisfiableError("%s wants %s %s but %s was pinned"
                                         % (name, dependency, spec, format_version(resolved[dependency])))
    return resolved


def topo_order(names, deps):
    """Dependencies first; ties broken alphabetically so the order is reproducible."""
    needs = {name: sorted({dependency for dependency, _ in deps.get(name, ())}) for name in names}
    order, placed = [], set()
    while len(order) < len(names):
        ready = sorted(name for name in names
                       if name not in placed and all(dep in placed for dep in needs[name]))
        if not ready:
            raise CycleError("dependency cycle detected: %s" % " -> ".join(find_cycle(names, needs)))
        order.append(ready[0])
        placed.add(ready[0])
    return order


def find_cycle(names, needs):
    """DFS returning the path of an actual cycle inside the unplaced subgraph."""
    colour, stack = {}, []

    def visit(name):
        colour[name], start = "grey", len(stack)
        stack.append(name)
        for dependency in needs[name]:
            if colour.get(dependency) == "grey":
                return stack[stack.index(dependency):] + [dependency]
            if colour.get(dependency) is None:
                found = visit(dependency)
                if found:
                    return found
        colour[name] = "black"
        stack.pop()
        return None

    for name in sorted(names):
        if colour.get(name) is None:
            found = visit(name)
            if found:
                return found
    return names


def lockfile(names, catalog, deps):
    resolved = resolve(names, catalog, deps)
    entries = [{"exercise": name, "repo": catalog[name]["repo"],
                "version": format_version(resolved[name]),
                "sha": hashlib.sha256(("%s@%s" % (name, format_version(resolved[name])))
                                      .encode("utf-8")).hexdigest()[:12]}
               for name in sorted(names)]
    body = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {"entries": entries, "digest": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "by_exercise": {entry["exercise"]: entry["version"] for entry in entries}}


def sync_plan(current, new):
    """Diff two lock snapshots into add/update/remove steps, sorted by exercise name."""
    steps = []
    for name in sorted(set(current) | set(new)):
        old, fresh = current.get(name), new.get(name)
        if old != fresh:
            steps.append({"op": "add" if old is None else "remove" if fresh is None else "update",
                          "exercise": name, "from": old, "to": fresh})
    return steps


def catalog():
    return {"two-sum": {"repo": "algorithms/easy", "versions": ["1.0.0", "1.2.0", "2.0.0"]},
            "lru-cache": {"repo": "algorithms/data-structures", "versions": ["0.9.0", "1.0.0"]},
            "dijkstra": {"repo": "algorithms/graphs", "versions": ["1.3.0", "1.4.2"]},
            "testing-kit": {"repo": "tooling/tests", "versions": ["3.1.0", "3.2.4"]}}


def sample_deps():
    return {"lru-cache": [("two-sum", ">=1.0.0")],
            "dijkstra": [("two-sum", "<2.0.0"), ("lru-cache", "^1.0.0")],
            "testing-kit": [("two-sum", "^1.0.0"), ("dijkstra", "~1.4.0")]}


def main():
    names = sorted(catalog())
    deps = sample_deps()
    resolved = resolve(names, catalog(), deps)

    # Happy path: the newest satisfying version wins, cross-checked by the spec predicate.
    check(format_version(resolved["two-sum"]) == "1.2.0" and
          format_version(resolved["lru-cache"]) == "1.0.0",
          "two-sum must pick 1.2.0 (^1.0.0, <2.0.0) and lru-cache 1.0.0")
    for name in names:
        for dependency, spec in deps.get(name, ()):
            check(satisfies(resolved[dependency], spec), "resolved %s violates %s" % (dependency, spec))

    # Constraint intersection, boundary inclusivity and exact pins.
    intersected = [v for v in ("1.0.0", "1.2.0", "2.0.0")
                   if satisfies(parse_version(v), "^1.0.0") and satisfies(parse_version(v), "<2.0.0")]
    check(intersected == ["1.0.0", "1.2.0"], "constraint intersection oracle drifted")
    check(format_version(resolve(["two-sum"], catalog(), {})["two-sum"]) == "2.0.0",
          "an unconstrained exercise must resolve to its newest version")
    check(satisfies(parse_version("1.2.0"), "=1.2.0") and not satisfies(parse_version("1.2.1"), "=1.2.0"),
          "an exact pin must accept exactly one version")
    check(satisfies(parse_version("1.2.0"), ">=1.2.0") and not satisfies(parse_version("2.0.0"), "<2.0.0"),
          "inclusive lower and exclusive upper bounds must behave at the boundary")

    # Unsatisfiable graphs and unknown dependencies are rejected, naming the culprit.
    expect_error(lambda: resolve(["dijkstra", "lru-cache", "two-sum"], catalog(),
                                 {"lru-cache": [("two-sum", "^1.0.0")],
                                  "dijkstra": [("two-sum", "=2.0.0")]}),
                 UnsatisfiableError, "two-sum")
    expect_error(lambda: resolve(["lru-cache", "two-sum"], catalog(),
                                 {"lru-cache": [("two-sum", ">=3.0.0")]}),
                 UnsatisfiableError, ">=3.0.0")
    expect_error(lambda: resolve(names, catalog(), {"lru-cache": [("ghost", "^1.0.0")]}),
                 ValueError, "unknown exercise")

    # Topological order: dependencies first, alphabetical tie-break, reproducible.
    order = topo_order(names, deps)
    positions = {name: index for index, name in enumerate(order)}
    edges_ok = all(positions[dep] < positions[name] for name in names for dep, _ in deps.get(name, ()))
    check(edges_ok, "every dependency must sync before the exercise that wants it")
    check(sorted(order) == names and order == topo_order(names, deps),
          "the topological order must be complete and deterministic")
    check(order[:2] == ["two-sum", "lru-cache"], "tie-break must order ready nodes alphabetically")

    # Cycle detection reports a real path rather than a generic failure.
    cycle = {"a": [("c", ">=1.0.0")], "b": [("a", ">=1.0.0")], "c": [("b", ">=1.0.0")]}
    expect_error(lambda: topo_order(["a", "b", "c"], cycle), CycleError, "a -> c -> b -> a")

    # Lock digest: permutation-independent, sorted by exercise, and idempotent.
    first = lockfile(names, catalog(), deps)
    check(lockfile(list(reversed(names)), catalog(), sample_deps())["digest"] == first["digest"]
          and [entry["exercise"] for entry in first["entries"]] == names
          and lockfile(names, catalog(), deps)["digest"] == first["digest"],
          "the lock digest must be permutation-independent, sorted and idempotent")

    # Sync plan: no-op, update, removal and addition.
    check(sync_plan(first["by_exercise"], first["by_exercise"]) == [],
          "an unchanged lock must produce an empty plan")
    bumped = dict(catalog())
    bumped["two-sum"] = {"repo": "algorithms/easy", "versions": ["1.0.0", "1.2.0", "1.3.0", "2.0.0"]}
    plan = sync_plan(first["by_exercise"], lockfile(names, bumped, deps)["by_exercise"])
    check(plan == [{"op": "update", "exercise": "two-sum", "from": "1.2.0", "to": "1.3.0"}],
          "publishing 1.3.0 must yield one update step, got %r" % (plan,))
    trimmed = [name for name in names if name != "testing-kit"]
    removed = sync_plan(first["by_exercise"], lockfile(trimmed, catalog(), deps)["by_exercise"])
    check(removed == [{"op": "remove", "exercise": "testing-kit", "from": "3.2.4", "to": None}],
          "dropping an exercise must yield a remove step, got %r" % (removed,))
    extended = dict(catalog())
    extended["binary-search"] = {"repo": "algorithms/search", "versions": ["1.0.0"]}
    added = sync_plan(first["by_exercise"],
                      lockfile(names + ["binary-search"], extended, deps)["by_exercise"])
    check([step["op"] for step in added] == ["add"], "a new exercise must yield an add step")

    # Boundary: one exercise with no dependencies at all.
    solo = lockfile(["two-sum"], catalog(), {})
    check(topo_order(["two-sum"], {}) == ["two-sum"] and len(solo["entries"]) == 1
          and solo["by_exercise"] == {"two-sum": "2.0.0"},
          "a lone exercise must order alone and lock exactly one entry")

    print("order=%s digest=%s steps=%d" % (",".join(order), first["digest"][:16], len(plan)))
    print("PASS")


if __name__ == "__main__":
    main()
