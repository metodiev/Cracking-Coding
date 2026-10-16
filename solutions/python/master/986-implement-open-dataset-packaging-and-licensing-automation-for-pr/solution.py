"""Task 986 - Implement open dataset packaging and licensing automation for practice problems.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A deterministic packager for an open problem dataset.  It validates that every member
problem declares an SPDX license the dataset license may redistribute, that attribution (a
non-empty author list) exists and that ids are unique; then it emits a canonical manifest
(entries sorted by id, files sorted by path, authors sorted), a SHA-256 digest over the
canonical serialization, and a NOTICE file grouping problems by member license with
attribution lines.

Simplified versus production: files live in memory instead of a content-addressed store and
license compatibility is a hand-written table, not an SPDX expression solver.  Verified:
order-independence of the digest, equality with an independent hand-written serializer,
rejection of incompatible/missing/unknown licenses and of unattributed or duplicated
entries, notice completeness, tamper detection, and the empty and single-problem boundaries.
"""
import hashlib
import json

# dataset license -> member licenses that may be redistributed under it
COMPATIBILITY = {
    "CC0-1.0": ("CC0-1.0", "MIT", "ISC", "BSD-3-Clause", "Apache-2.0", "CC-BY-4.0"),
    "MIT": ("MIT", "ISC", "BSD-3-Clause", "Apache-2.0", "CC0-1.0"),
    "Apache-2.0": ("Apache-2.0", "MIT", "ISC", "BSD-3-Clause", "CC0-1.0"),
    "CC-BY-4.0": ("CC-BY-4.0", "CC0-1.0"),
    "CC-BY-SA-4.0": ("CC-BY-SA-4.0", "CC-BY-4.0", "CC0-1.0", "GPL-3.0", "MIT", "ISC",
                     "BSD-3-Clause", "Apache-2.0"),
    "GPL-3.0": ("GPL-3.0", "MIT", "ISC", "BSD-3-Clause", "Apache-2.0", "CC-BY-SA-4.0"),
}


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_error(action, needle):
    try:
        action()
    except ValueError as exc:
        check(needle in str(exc), "error %r does not mention %r" % (str(exc), needle))
        return
    raise AssertionError("expected a ValueError mentioning %r" % needle)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(value):
    """Canonical serialization: sorted keys, no incidental whitespace, ASCII only."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def manual_canonical(value):
    """Independent serializer (no json module) used as the digest oracle in the checks."""
    if isinstance(value, dict):
        body = ",".join('"%s":%s' % (key, manual_canonical(value[key])) for key in sorted(value))
        return "{" + body + "}"
    if isinstance(value, list):
        return "[" + ",".join(manual_canonical(item) for item in value) + "]"
    return '"%s"' % value if isinstance(value, str) else str(value)


def validate(dataset):
    """Raise ValueError for any dataset that may not be published under its own license."""
    if not dataset.get("name") or not dataset.get("version"):
        raise ValueError("dataset needs a non-empty name and version")
    license_ = dataset.get("license")
    if license_ not in COMPATIBILITY:
        raise ValueError("unknown dataset license %r" % (license_,))
    allowed = COMPATIBILITY[license_]
    seen = set()
    for problem in dataset.get("problems", ()):
        pid = problem.get("id")
        if not pid:
            raise ValueError("problem without an id")
        if pid in seen:
            raise ValueError("duplicate problem id %r" % (pid,))
        seen.add(pid)
        member = problem.get("license")
        if member not in COMPATIBILITY:
            raise ValueError("problem %r has unknown license %r" % (pid, member))
        if member not in allowed:
            raise ValueError("problem %r license %s cannot be redistributed under %s"
                             % (pid, member, license_))
        authors = problem.get("authors") or []
        if not authors or not all(isinstance(a, str) and a.strip() for a in authors):
            raise ValueError("problem %r lacks attribution (non-empty authors)" % (pid,))
        files = problem.get("files") or {}
        if not files:
            raise ValueError("problem %r has no files" % (pid,))


def build_manifest(dataset):
    validate(dataset)
    entries = []
    for problem in sorted(dataset["problems"], key=lambda item: item["id"]):
        files = {path: {"bytes": len(content), "sha256": sha256_text(content)}
                 for path, content in sorted(problem["files"].items())}
        entries.append({"id": problem["id"],
                        "license": problem["license"],
                        "authors": sorted(problem["authors"]),
                        "files": files,
                        "bytes": sum(entry["bytes"] for entry in files.values())})
    return {"dataset": dataset["name"],
            "version": dataset["version"],
            "license": dataset["license"],
            "entries": entries,
            "total_bytes": sum(entry["bytes"] for entry in entries)}


def manifest_digest(manifest):
    return sha256_text(canonical_json(manifest))


def notice_text(manifest):
    """NOTICE groups problems by member license and names every author once per problem."""
    lines = ["Dataset: %s %s" % (manifest["dataset"], manifest["version"]),
             "Dataset license: %s" % manifest["license"]]
    if not manifest["entries"]:
        lines.append("This dataset contains no entries.")
        return "\n".join(lines)
    groups = {}
    for entry in manifest["entries"]:
        groups.setdefault(entry["license"], []).append(entry)
    for license_ in sorted(groups):
        lines.append("Redistributed under %s:" % license_)
        for entry in groups[license_]:
            lines.append("  - %s by %s" % (entry["id"], ", ".join(entry["authors"])))
    return "\n".join(lines)


def package(dataset):
    manifest = build_manifest(dataset)
    return {"manifest": manifest, "digest": manifest_digest(manifest),
            "notice": notice_text(manifest)}


def verify_package(packaged, current_files):
    """Re-hash the published bytes; report every entry whose digest no longer matches."""
    mismatches = []
    for entry in packaged["manifest"]["entries"]:
        files = current_files.get(entry["id"], {})
        for path, recorded in entry["files"].items():
            content = files.get(path)
            if content is None or sha256_text(content) != recorded["sha256"]:
                mismatches.append("%s/%s" % (entry["id"], path))
    return mismatches


def sample_dataset():
    return {
        "name": "crack-pack", "version": "1.2.0", "license": "CC-BY-SA-4.0",
        "problems": [
            {"id": "two-sum", "license": "MIT", "authors": ["Ada Lovelace"],
             "files": {"statement.md": "# Two Sum\n", "tests.py": "assert solve\n"}},
            {"id": "lru-cache", "license": "GPL-3.0", "authors": ["Grace Hopper", "Ada Lovelace"],
             "files": {"statement.md": "# LRU\n", "solution.py": "class LRU: pass\n"}},
            {"id": "bubble-sort", "license": "CC-BY-4.0", "authors": ["Linus T"],
             "files": {"statement.md": "# Bubble\n", "fixtures.json": "{}\n"}},
        ],
    }


def main():
    dataset = sample_dataset()
    packaged = package(dataset)
    manifest = packaged["manifest"]

    # Happy path: entries sorted by id, authors and file keys canonicalised.
    check([entry["id"] for entry in manifest["entries"]] == ["bubble-sort", "lru-cache", "two-sum"],
          "manifest entries must be sorted by id")
    check(manifest["entries"][1]["authors"] == ["Ada Lovelace", "Grace Hopper"],
          "authors must be canonicalised (sorted)")
    check(manifest["total_bytes"] == sum(entry["bytes"] for entry in manifest["entries"]),
          "manifest total_bytes must equal the per-entry sum")
    check(packaged["digest"] == sha256_text(manual_canonical(manifest)),
          "digest disagrees with the independent canonical serializer")

    # Reproducibility: permuting problems and file insertion order changes nothing.
    shuffled = sample_dataset()
    shuffled["problems"].reverse()
    shuffled["problems"][1]["files"] = {"solution.py": "class LRU: pass\n",
                                       "statement.md": "# LRU\n"}
    repackaged = package(shuffled)
    check(repackaged["digest"] == packaged["digest"],
          "package digest must not depend on input ordering")
    check(repackaged["notice"] == packaged["notice"], "NOTICE must be order-independent")

    # Attribution completeness, brute-forced from the raw input rather than the manifest.
    for problem in dataset["problems"]:
        for author in problem["authors"]:
            mentions = sum(line.count(author) for line in packaged["notice"].splitlines())
            expected = sum(1 for other in dataset["problems"] if author in other["authors"])
            check(mentions == expected,
                  "author %s must be attributed once per problem (%d), saw %d"
                  % (author, expected, mentions))
    check(all(problem["id"] in packaged["notice"] for problem in dataset["problems"]),
          "every packaged problem must appear in the NOTICE")
    check("Redistributed under GPL-3.0:" in packaged["notice"]
          and "  - lru-cache by Ada Lovelace, Grace Hopper" in packaged["notice"],
          "NOTICE must list the copyleft group with sorted attribution")

    # Integrity verification plus tamper detection.
    current = {problem["id"]: dict(problem["files"]) for problem in dataset["problems"]}
    check(verify_package(packaged, current) == [], "untampered bundle must verify clean")
    tampered = {key: dict(value) for key, value in current.items()}
    tampered["lru-cache"]["solution.py"] += "# sneaky\n"
    check(verify_package(packaged, tampered) == ["lru-cache/solution.py"],
          "tampering with a bundled file must be detected")
    changed = sample_dataset()
    changed["problems"][0]["files"]["statement.md"] = "# Two Sum (edited)\n"
    check(package(changed)["digest"] != packaged["digest"],
          "editing file content must change the package digest")

    # Licensing gates: incompatible member, unknown license, missing attribution, duplicate id.
    incompatible = sample_dataset()
    incompatible["license"] = "MIT"  # GPL-3.0 member is not redistributable under MIT
    expect_error(lambda: build_manifest(incompatible), "GPL-3.0")
    unknown = sample_dataset(); unknown["problems"][0]["license"] = "WTFPL"
    expect_error(lambda: build_manifest(unknown), "unknown license")
    unlicensed = sample_dataset(); unlicensed["license"] = "BSD-3-Clause"
    expect_error(lambda: build_manifest(unlicensed), "unknown dataset license")
    unattributed = sample_dataset(); unattributed["problems"][1]["authors"] = []
    expect_error(lambda: build_manifest(unattributed), "attribution")
    duplicated = sample_dataset(); duplicated["problems"].append(dict(duplicated["problems"][0]))
    expect_error(lambda: build_manifest(duplicated), "duplicate problem id")

    # Boundary: a single-problem dataset and the empty dataset both package cleanly.
    solo = package({"name": "solo", "version": "0.1.0", "license": "MIT",
                    "problems": [{"id": "one", "license": "MIT", "authors": ["Solo Author"],
                                  "files": {"statement.md": "# One\n"}}]})
    check(solo["manifest"]["total_bytes"] == len("# One\n")
          and "Redistributed under MIT:" in solo["notice"],
          "single-problem dataset must package with its own size and notice group")
    empty = package({"name": "void", "version": "0.0.1", "license": "CC0-1.0", "problems": []})
    check(empty["manifest"]["entries"] == [] and empty["manifest"]["total_bytes"] == 0
          and empty["notice"].endswith("This dataset contains no entries."),
          "empty dataset must package with no entries and say so")
    check(len(empty["digest"]) == 64 and empty["digest"] != packaged["digest"],
          "the empty digest must be a distinct well-formed sha256")

    print("entries=%d total_bytes=%d digest=%s" %
          (len(manifest["entries"]), manifest["total_bytes"], packaged["digest"][:16]))
    print("PASS")


if __name__ == "__main__":
    main()
