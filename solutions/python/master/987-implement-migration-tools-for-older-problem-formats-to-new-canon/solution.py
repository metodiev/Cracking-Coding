"""Task 987 - Implement migration tools for older problem formats to new canonical schema.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

A registry of step-wise migrators that upgrades legacy problem records to one canonical
schema.  Version detection is structural (v1 = flat title/statement, v2 = meta/body pairs,
canonical = id/sections/solutions); every hop is a pure function of the record, so upgrades
are replayable.  Ids are derived from the title, difficulty/tags are canonicalised
(lower-cased, de-duplicated, sorted) and keys the canonical schema does not model are parked
in an `extra` map instead of being silently dropped.

Simplified versus production: migrations run in memory over dicts rather than through a
versioned migration store with transactional apply/rollback, and tag canonicalisation is
deliberately lossy for case and order (unknown top-level keys are not).  Verified: exact
canonical output for a fixed legacy record, convergence of the v1 and v2 upgrade paths,
idempotence with an unchanged fingerprint, lossless demotion back to v1, extra-field
preservation, defaults for a minimal record, and rejection of unknown/invalid input.
"""
import hashlib
import json

DIFFICULTIES = ("easy", "medium", "hard")
LEGACY_DIFFICULTY = {1: "easy", 2: "medium", 3: "hard"}
V1_KEYS = ("title", "statement", "solution", "lang", "difficulty", "tags")
CANONICAL_KEYS = ("id", "title", "difficulty", "tags", "sections", "solutions", "extra")


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


def fingerprint(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode("utf-8")).hexdigest()


def slugify(title):
    out = []
    for char in title.lower():
        if char.isalnum():
            out.append(char)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")


def canonical_tags(tags):
    return sorted({str(tag).strip().lower() for tag in tags if str(tag).strip()})


def detect_version(record):
    """Structural detection: legacy records carry none of the newer container keys."""
    if all(key in record for key in ("id", "sections", "solutions", "tags")):
        return "canonical"
    if "meta" in record and "body" in record:
        return "v2"
    if "title" in record and "statement" in record:
        return "v1"
    raise ValueError("unknown problem format: cannot detect version for keys %s"
                     % sorted(record))


def to_v2(record):
    """v1 -> v2: split flat fields into meta/body, keep unknown keys for the next hop."""
    extras = {key: value for key, value in record.items() if key not in V1_KEYS}
    solution = record.get("solution", "")
    lang = record.get("lang", "python")
    return {"meta": {"title": record["title"],
                     "difficulty": LEGACY_DIFFICULTY[record.get("difficulty", 2)]},
            "body": {"statement": record["statement"], "solution": solution},
            "tags": list(record.get("tags", [])),
            "langs": {lang: solution} if solution else {},
            "extra": extras}


def to_canonical(record):
    """v2 -> canonical.  Fills defaults so legacy records without difficulty/tags survive."""
    meta, body = record["meta"], record["body"]
    solution = body.get("solution", "")
    return {"id": slugify(meta["title"]),
            "title": meta["title"],
            "difficulty": meta.get("difficulty", "medium"),
            "tags": canonical_tags(record.get("tags", [])),
            "sections": {"statement": body["statement"], "solution": solution},
            "solutions": dict(record.get("langs", {})),
            "extra": dict(record.get("extra", {}))}


def validate_canonical(canonical):
    for key in CANONICAL_KEYS:
        if key not in canonical:
            raise ValueError("canonical record missing %r" % (key,))
    if canonical["difficulty"] not in DIFFICULTIES:
        raise ValueError("unknown difficulty %r" % (canonical["difficulty"],))
    if canonical["id"] != slugify(canonical["title"]) or not canonical["id"]:
        raise ValueError("id %r is not the slug of title %r"
                         % (canonical["id"], canonical["title"]))
    if canonical["tags"] != canonical_tags(canonical["tags"]):
        raise ValueError("tags must be lower-cased, de-duplicated and sorted")
    if set(canonical["sections"]) != {"statement", "solution"}:
        raise ValueError("sections must hold exactly statement and solution")
    if not all(isinstance(code, str) for code in canonical["solutions"].values()):
        raise ValueError("solutions must map language -> source text")


def migrate(record):
    """Detect, hop to canonical, and report the steps and the preserved extra keys."""
    version = detect_version(record)
    if version == "canonical":
        canonical = {key: (dict(record[key]) if isinstance(record[key], dict) else
                           (list(record[key]) if isinstance(record[key], list) else record[key]))
                     for key in CANONICAL_KEYS}
        validate_canonical(canonical)
        steps = []
    else:
        steps = ["v1->v2"] if version == "v1" else []
        canonical = to_canonical(to_v2(record) if version == "v1" else record)
        validate_canonical(canonical)
        steps.append("v2->canonical")
    return {"canonical": canonical,
            "report": {"from": version, "steps": steps,
                       "extra_keys": sorted(canonical["extra"])}}


def demote_to_v1(canonical):
    """Best-effort downgrade used to prove the canonical schema stayed lossless for v1."""
    validate_canonical(canonical)
    lang, code = next(iter(sorted(canonical["solutions"].items())), ("python", ""))
    legacy = {"title": canonical["title"],
              "statement": canonical["sections"]["statement"],
              "solution": code,
              "lang": lang,
              "difficulty": {name: level for level, name in LEGACY_DIFFICULTY.items()}
                            [canonical["difficulty"]]}
    if canonical["tags"]:
        legacy["tags"] = list(canonical["tags"])
    legacy.update(canonical["extra"])
    return legacy


def legacy_record():
    return {"title": "Two Sum", "statement": "Return the indices of two values summing to k.",
            "solution": "def two_sum(xs, k): return (0, 1)\n", "lang": "python",
            "difficulty": 2, "source": "legacy-import-v0"}


def main():
    record = legacy_record()
    result = migrate(record)
    canonical = result["canonical"]

    # Happy path: exact canonical shape for the fixed legacy record.
    expected = {"id": "two-sum",
                "title": "Two Sum",
                "difficulty": "medium",
                "tags": [],
                "sections": {"statement": "Return the indices of two values summing to k.",
                             "solution": "def two_sum(xs, k): return (0, 1)\n"},
                "solutions": {"python": "def two_sum(xs, k): return (0, 1)\n"},
                "extra": {"source": "legacy-import-v0"}}
    check(canonical == expected, "canonical output differs from the hand-written expectation")
    check(result["report"]["steps"] == ["v1->v2", "v2->canonical"],
          "the v1 upgrade path must run exactly the two registered hops")
    check(canonical["extra"] == {"source": "legacy-import-v0"}
          and result["report"]["extra_keys"] == ["source"],
          "unknown legacy keys must be preserved in extra rather than dropped")

    # Idempotence: re-migrating a canonical record is a no-op with an unchanged fingerprint.
    again = migrate(canonical)
    check(again["canonical"] == canonical, "re-migration changed the canonical record")
    check(again["report"]["steps"] == [], "a canonical record must not be migrated again")
    check(fingerprint(again["canonical"]) == fingerprint(canonical),
          "fingerprint drifted across an idempotent migration")
    check(migrate(again["canonical"])["canonical"] == canonical, "migration is not idempotent")

    # Convergence: the v2 path yields the same canonical record as the v1 path.
    v2_record = {"meta": {"title": "Two Sum", "difficulty": "medium"},
                 "body": {"statement": "Return the indices of two values summing to k.",
                          "solution": "def two_sum(xs, k): return (0, 1)\n"},
                 "tags": [],
                 "langs": {"python": "def two_sum(xs, k): return (0, 1)\n"},
                 "extra": {"source": "legacy-import-v0"}}
    check(migrate(v2_record)["canonical"] == canonical,
          "the v1 and v2 upgrade paths must converge on one canonical record")
    check(migrate(v2_record)["report"]["steps"] == ["v2->canonical"],
          "a v2 record needs exactly one hop")

    # Lossless for representable fields: demoting the canonical record restores the v1 dict.
    check(demote_to_v1(canonical) == record,
          "demote(canonical) must reproduce the original legacy record")

    # Canonicalisation of tags and of the legacy difficulty scale.
    tagged = dict(record, title="Two Pointers", tags=["Arrays", "arrays", " Two Pointers "])
    migrated_tags = migrate(tagged)["canonical"]["tags"]
    check(migrated_tags == ["arrays", "two pointers"],
          "tags must be trimmed, lower-cased, de-duplicated and sorted")
    for level, name in ((1, "easy"), (3, "hard")):
        scaled = migrate(dict(record, difficulty=level))["canonical"]
        check(scaled["difficulty"] == name, "legacy difficulty %d must map to %s" % (level, name))

    # Boundary: a minimal record still migrates, taking documented defaults.
    minimal = migrate({"title": "Solo Problem", "statement": "Add one.\n"})["canonical"]
    check({"id": minimal["id"], "difficulty": minimal["difficulty"], "tags": minimal["tags"],
           "solutions": minimal["solutions"]} == {"id": "solo-problem", "difficulty": "medium",
                                                 "tags": [], "solutions": {}},
          "a minimal legacy record must default to medium/python with no tags or solutions")
    check(minimal["sections"]["statement"] == "Add one.\n"
          and minimal["sections"]["solution"] == "", "missing solution must become empty text")

    # Rejections: unknown format, unknown difficulty, unsorted tags, incomplete sections.
    expect_error(lambda: migrate({"notes": "no recognisable keys"}), "unknown problem format")
    expect_error(lambda: migrate(dict(canonical, difficulty="impossible")), "unknown difficulty")
    expect_error(lambda: migrate(dict(canonical, tags=["b", "a"])), "sorted")
    expect_error(lambda: migrate(dict(canonical, tags=["Only", "only"])), "de-duplicated")
    expect_error(lambda: migrate(dict(canonical, sections={"statement": "x"})), "statement")
    expect_error(lambda: migrate(dict(canonical, id="wrong-id")), "slug")
    expect_error(lambda: migrate({}), "unknown problem format")

    print("steps=%s extra_keys=%s fingerprint=%s"
          % (",".join(result["report"]["steps"]), ",".join(result["report"]["extra_keys"]),
             fingerprint(canonical)[:16]))
    print("PASS")


if __name__ == "__main__":
    main()
