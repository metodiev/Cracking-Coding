"""Task 989 - Implement offline mode packaging so learners can download problem sets + tests.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Builds a self-contained offline bundle for a learner: problems are selected by tier and tag,
each selected problem contributes statement/solution/tests, and shared fixtures are
content-addressed so identical fixture bytes are stored exactly once no matter how many
problems reference them.  Every file is hashed into a sorted manifest whose digest keys the
download; the bundle refuses to ship anything that still points at the network (an offline
learner could not fetch it) and refuses to exceed the caller's byte budget.

Simplified versus production: bundling happens in memory instead of writing a tarball to disk
and there is no compression step, so sizes are uncompressed byte counts.  Verified: selection
by tier/tag, completeness (tests travel with every problem), fixture de-duplication, an
independent byte-total oracle, reproducibility under permuted input, tamper detection, the
exact budget boundary, network-reference rejection and the empty-selection boundary.
"""
import hashlib
import json
import re

NETWORK_REFERENCE = re.compile(r"https?://[^\s)\"'<>]+")


class OfflineBundleError(ValueError):
    """Raised when a bundle would not be usable offline or would exceed its budget."""


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_error(action, needle):
    try:
        action()
    except OfflineBundleError as exc:
        check(needle in str(exc), "error %r does not mention %r" % (str(exc), needle))
        return
    raise AssertionError("expected an OfflineBundleError mentioning %r" % needle)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def select(problems, tiers=(), tags=()):
    """Tier filter plus any-match tag filter, sorted by id so the choice is reproducible."""
    chosen = [problem for problem in problems
              if (not tiers or problem["tier"] in tiers)
              and (not tags or set(tags) & set(problem["tags"]))]
    return sorted(chosen, key=lambda problem: problem["id"])


def collect_files(chosen):
    """problem files, per-problem tests, and one content-addressed entry per fixture body."""
    files, fixtures = {}, {}
    for problem in chosen:
        pid = problem["id"]
        for name, content in problem.get("files", {}).items():
            files["problems/%s/%s" % (pid, name)] = content
        for name, content in problem.get("tests", {}).items():
            files["problems/%s/tests/%s" % (pid, name)] = content
        for content in problem.get("fixtures", {}).values():
            fixtures.setdefault(sha256_text(content)[:12], content)
    for key, content in fixtures.items():
        files["fixtures/%s.json" % key] = content
    return files


def build_bundle(problems, tiers=(), tags=(), budget=None):
    chosen = select(problems, tiers, tags)
    files = collect_files(chosen)
    for path in sorted(files):
        found = NETWORK_REFERENCE.search(files[path])
        if found:
            raise OfflineBundleError("%s still references %s over the network" % (path, found.group(0)))
    entries = [{"path": path, "sha256": sha256_text(files[path]), "bytes": len(files[path])}
               for path in sorted(files)]
    total = sum(entry["bytes"] for entry in entries)
    if budget is not None and total > budget:
        raise OfflineBundleError("bundle needs %d bytes but the budget is %d" % (total, budget))
    manifest = {"problems": [problem["id"] for problem in chosen],
                "entries": entries, "total_bytes": total}
    manifest["digest"] = sha256_text(canonical(manifest))
    return {"manifest": manifest, "files": files}


def verify_bundle(bundle, current_files):
    """Re-hash the downloaded bytes and report every path that no longer matches the manifest."""
    mismatches = []
    for entry in bundle["manifest"]["entries"]:
        content = current_files.get(entry["path"])
        if content is None:
            mismatches.append("%s (missing)" % entry["path"])
        elif sha256_text(content) != entry["sha256"] or len(content) != entry["bytes"]:
            mismatches.append(entry["path"])
    return mismatches


def fixture_catalog():
    shared = "{\"seed\": 7, \"cases\": [1, 2, 3]}\n"
    return [
        {"id": "two-sum", "tier": "master", "tags": ["arrays", "hashing"],
         "files": {"statement.md": "# Two Sum\n", "solution.py": "def two_sum(xs, k): return None\n"},
         "tests": {"test_two_sum.py": "from solution import two_sum\n"}, "fixtures": {"common": shared}},
        {"id": "lru-cache", "tier": "master", "tags": ["design", "hashing"],
         "files": {"statement.md": "# LRU Cache\n", "solution.py": "class LRU: pass\n"},
         "tests": {"test_lru.py": "from solution import LRU\n"}, "fixtures": {"common": shared}},
        {"id": "bubble-sort", "tier": "advanced", "tags": ["sorting"],
         "files": {"statement.md": "# Bubble Sort\n", "solution.py": "def sort(a): return sorted(a)\n"},
         "tests": {"test_bubble.py": "from solution import sort\n"}, "fixtures": {"common": shared}},
        {"id": "binary-search", "tier": "master", "tags": ["search"],
         "files": {"statement.md": "# Binary Search\n", "solution.py": "def search(a, x): return -1\n"},
         "tests": {"test_search.py": "from solution import search\n"}, "fixtures": {}},
    ]


def main():
    catalog = fixture_catalog()
    bundle = build_bundle(catalog, tiers=("master",))
    manifest = bundle["manifest"]

    # Happy path: tier selection, sorted ids, and completeness (tests travel with the problem).
    check(manifest["problems"] == ["binary-search", "lru-cache", "two-sum"],
          "tier selection must be sorted by id, got %r" % (manifest["problems"],))
    paths = [entry["path"] for entry in manifest["entries"]]
    check(paths == sorted(paths) and len(paths) == len(set(paths)),
          "manifest paths must be unique and sorted")
    for pid in manifest["problems"]:
        check(any(path.startswith("problems/%s/tests/" % pid) for path in paths),
              "every selected problem must ship its tests")
        check(any(path.startswith("problems/%s/" % pid) for path in paths),
              "every selected problem must contribute files")

    # Fixture de-duplication: three problems share one body, so exactly one fixture entry exists.
    fixture_paths = [path for path in paths if path.startswith("fixtures/")]
    distinct_bodies = {sha256_text(content) for problem in catalog for content in problem["fixtures"].values()}
    check(len(fixture_paths) == 1 and len(distinct_bodies) == 1,
          "shared fixture bodies must collapse to one entry, saw %r" % (fixture_paths,))

    # Independent oracle: byte total recomputed straight from the raw catalog, not the manifest.
    chosen = select(catalog, tiers=("master",))
    body = next(iter({sha256_text(content) for problem in chosen
                      for content in problem["fixtures"].values()}))
    shared_body = next(content for problem in chosen for content in problem["fixtures"].values()
                       if sha256_text(content) == body)
    sharers = sum(1 for problem in chosen if shared_body in problem["fixtures"].values())
    per_problem = sum(len(content) for problem in chosen
                      for group in ("files", "tests") for content in problem[group].values())
    naive_total = per_problem + sharers * len(shared_body)
    deduplicated_total = per_problem + len(shared_body)
    check(manifest["total_bytes"] == deduplicated_total == naive_total - (sharers - 1) * len(shared_body),
          "total_bytes %d disagrees with the raw-catalog oracle %d (%d sharers)"
          % (manifest["total_bytes"], deduplicated_total, sharers))
    check(sharers == 2, "the fixture body must really be shared by both selected master problems")
    check(manifest["total_bytes"] == sum(entry["bytes"] for entry in manifest["entries"]),
          "manifest total must equal the sum of its entries")

    # Reproducibility: permuted catalog, tag-union selection, and a rebuilt bundle agree.
    reversed_build = build_bundle(list(reversed(catalog)), tiers=("master",))
    check(reversed_build["manifest"]["digest"] == manifest["digest"],
          "the bundle digest must not depend on catalog order")
    tagged = build_bundle(catalog, tags=("hashing", "sorting"))
    brute_force = sorted(problem["id"] for problem in catalog
                         if {"hashing", "sorting"} & set(problem["tags"]))
    check(tagged["manifest"]["problems"] == brute_force,
          "tag selection must be the any-match union, got %r" % (tagged["manifest"]["problems"],))

    # Integrity: a clean download verifies, a tampered file is reported, a missing file too.
    check(verify_bundle(bundle, dict(bundle["files"])) == [], "a clean bundle must verify")
    tampered = dict(bundle["files"])
    tampered["problems/two-sum/solution.py"] += "# edited\n"
    check(verify_bundle(bundle, tampered) == ["problems/two-sum/solution.py"],
          "an edited file must be reported as a mismatch")
    missing = {path: content for path, content in bundle["files"].items()
               if path != fixture_paths[0]}
    check(verify_bundle(bundle, missing) == ["%s (missing)" % fixture_paths[0]],
          "a missing file must be reported as missing")

    # Boundary: the exact budget fits, one byte less is rejected with both numbers.
    check(build_bundle(catalog, tiers=("master",), budget=manifest["total_bytes"])["manifest"]["digest"]
          == manifest["digest"], "a bundle that exactly fits the budget must still build")
    expect_error(lambda: build_bundle(catalog, tiers=("master",), budget=manifest["total_bytes"] - 1),
                 "budget")

    # Offline hygiene: an absolute URL anywhere in the payload blocks the bundle.
    online = fixture_catalog()
    online[0]["tests"]["test_two_sum.py"] += "# expected from https://example.com/expected.json\n"
    expect_error(lambda: build_bundle(online, tiers=("master",)), "network")
    expect_error(lambda: build_bundle(online, tiers=("master",)), "https://example.com/expected.json")
    relative = build_bundle(catalog, tiers=("master",))
    check("https://" not in "".join(relative["files"].values()),
          "no bundled file may retain an absolute URL")

    # Boundary: a selection that matches nothing yields an empty, well-formed bundle.
    empty = build_bundle(catalog, tags=("no-such-tag",))
    check(empty["manifest"]["problems"] == [] and empty["manifest"]["entries"] == []
          and empty["manifest"]["total_bytes"] == 0 and len(empty["manifest"]["digest"]) == 64,
          "an empty selection must produce an empty but well-formed manifest")
    check(verify_bundle(empty, {}) == [], "an empty bundle must verify trivially")

    print("problems=%d files=%d total_bytes=%d digest=%s"
          % (len(manifest["problems"]), len(manifest["entries"]), manifest["total_bytes"],
             manifest["digest"][:16]))
    print("PASS")


if __name__ == "__main__":
    main()
