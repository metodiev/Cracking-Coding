"""Task 956 - Implement grading rubric generator that assigns difficulty and expected complexity.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Rubric model: a problem profile (input size, algorithm classes the intended solution needs, whether a
proof or a system design is part of the deliverable, edge-case count, judge latency) is scored with a
fixed points table; the score selects a difficulty band, the highest-ranked required class selects
the expected complexity, and the band selects the item weights.  Item weights are integer vectors
that sum to exactly 100, so the rubric is always a complete point distribution.  Scoring is pure:
generating a rubric never mutates the profile and repeated calls return equal rubrics.

Verified: the acceptance path for a mid-size profile, monotonicity in every feature, exact band
boundaries, expected complexity against an independent maximum over a growth-ordered class list,
conservation of the 100 points in all four bands, rejection of malformed profiles, letter-grade
consistency, purity and the empty-profile boundary.
"""
CLASS_ORDER = ("O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(2^n)")
N_THRESHOLDS = ((10_000, 2), (100_000, 6), (10_000_000, 12), (1_000_000_000, 20))
BANDS = (("easy", 25), ("medium", 50), ("hard", 75), ("expert", 101))
GRADES = (("expert", "A"), ("hard", "B"), ("medium", "C"), ("easy", "D"))
RUBRIC_WEIGHTS = {
    "easy": (("correctness", 50), ("complexity", 20), ("clarity", 20), ("tests", 10)),
    "medium": (("correctness", 45), ("complexity", 25), ("clarity", 15), ("tests", 15)),
    "hard": (("correctness", 40), ("complexity", 30), ("clarity", 10), ("tests", 20)),
    "expert": (("correctness", 35), ("complexity", 35), ("clarity", 10), ("tests", 20)),
}
PROFILE_FIELDS = ("size", "required", "has_proof", "has_system_design", "edge_cases",
                  "judge_latency_ms")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def validate_profile(profile):
    """Reject a malformed profile; returns the normalised copy."""
    unknown = sorted(set(profile) - set(PROFILE_FIELDS))
    if unknown:
        raise ValueError(f"unknown profile field(s): {', '.join(unknown)}")
    size = profile.get("size", 1000)
    if not isinstance(size, int) or size < 1:
        raise ValueError(f"size must be a positive integer, got {size!r}")
    required = set(profile.get("required", ()))
    unknown_classes = sorted(required - set(CLASS_ORDER))
    if unknown_classes:
        raise ValueError(f"unknown complexity class(es): {', '.join(unknown_classes)}")
    return {"size": size, "required": required,
            "has_proof": bool(profile.get("has_proof", False)),
            "has_system_design": bool(profile.get("has_system_design", False)),
            "edge_cases": int(profile.get("edge_cases", 0)),
            "judge_latency_ms": int(profile.get("judge_latency_ms", 0))}


def size_points(size):
    """Input size alone is capped at 20 points, so size can never make a problem hard."""
    return max(points for threshold, points in N_THRESHOLDS if size >= threshold) \
        if size >= N_THRESHOLDS[0][0] else 0


def expected_complexity(required):
    """The slowest (highest-ranked) class the intended solution must handle."""
    return max(required, key=lambda label: CLASS_ORDER.index(label)) if required else "O(1)"


def score_profile(profile):
    """Total rubric score; every term is a documented, bounded contribution."""
    clean = validate_profile(profile)
    return {"size": size_points(clean["size"]),
            "algorithms": 6 * CLASS_ORDER.index(expected_complexity(clean["required"])),
            "proof": 10 if clean["has_proof"] else 0,
            "system_design": 12 if clean["has_system_design"] else 0,
            "edge_cases": 2 * min(clean["edge_cases"], 4),
            "latency": 5 if 0 < clean["judge_latency_ms"] <= 100 else 0}


def classify(score):
    """Band lookup; bands are upper-exclusive so the boundary value belongs to the higher band."""
    for band, upper in BANDS:
        if score < upper:
            return band
    return BANDS[-1][0]


def letter(score):
    """Letter grade derived from the same score, independently of `classify`."""
    return next(grade for band, grade in GRADES if band == classify(score))


def rubric(profile):
    """The published rubric: score breakdown, band, expected complexity and item weights."""
    breakdown = score_profile(profile)
    score = sum(breakdown.values())
    band = classify(score)
    items = [{"item": name, "weight": weight, "points": weight}
             for name, weight in RUBRIC_WEIGHTS[band]]
    return {"score": score, "breakdown": breakdown, "band": band,
            "expected_complexity": expected_complexity(validate_profile(profile)["required"]),
            "total_points": sum(item["points"] for item in items), "items": items}


def main():
    # 1) happy path: a mid-size profile with a couple of classes lands in a named band with a
    #    complete point distribution.
    profile = {"size": 200_000, "required": ("O(n log n)",), "edge_cases": 4,
               "judge_latency_ms": 100}
    graded = rubric(profile)
    check(graded["band"] == "medium" and graded["total_points"] == 100
          and graded["expected_complexity"] == "O(n log n)" and graded["score"] == 6 + 18 + 8 + 5
          and [item["item"] for item in graded["items"]] == ["correctness", "complexity", "clarity",
                                                             "tests"],
          f"mid-size profile misgraded: {graded['score']} -> {graded['band']}")

    # 2) exact band boundaries: the boundary value belongs to the higher band.
    boundaries = {score: classify(score) for score in (24, 25, 49, 50, 74, 75, 100)}
    check(boundaries == {24: "easy", 25: "medium", 49: "medium", 50: "hard", 74: "hard",
                         75: "expert", 100: "expert"},
          f"band boundaries wrong: {boundaries}")

    # 3) monotonicity: growing a profile in any feature never lowers its score or its band.
    small = {"size": 500, "required": ("O(n)",), "edge_cases": 1}
    bigger = {"size": 2_000_000_000, "required": ("O(n)", "O(n log n)"), "edge_cases": 3,
              "has_proof": True}
    biggest = {"size": 2_000_000_000, "required": ("O(n log n)", "O(n^2)"), "edge_cases": 4,
               "has_proof": True, "has_system_design": True, "judge_latency_ms": 50}
    scores = [sum(score_profile(item).values()) for item in (small, bigger, biggest)]
    ranks = [CLASS_ORDER.index(expected_complexity(item["required"]))
             for item in (small, bigger, biggest)]
    check(scores[0] < scores[1] < scores[2] and ranks == sorted(ranks)
          and [classify(value) for value in scores] == ["easy", "hard", "expert"],
          f"monotonicity broken: scores {scores} -> {[classify(v) for v in scores]}")

    # 4) independent oracle for the expected complexity: the ranked maximum of the required set.
    sample = ("O(log n)", "O(n^2)")
    indices = sorted(CLASS_ORDER.index(label) for label in sample)
    check(expected_complexity(sample) == CLASS_ORDER[indices[-1]] == "O(n^2)"
          and expected_complexity(()) == "O(1)" and expected_complexity(("O(1)",)) == "O(1)",
          "expected complexity must be the ranked maximum of the required classes")

    # 5) conservation: every band's item weights sum to exactly 100, and the sum of the published
    #    items equals the reported total for any band reachable from real profiles.
    reachable = set()
    for size in (500, 200_000, 2_000_000_000):
        for required in ((), ("O(n)",), ("O(2^n)",)):
            for proof in (False, True):
                for design in (False, True):
                    generated = rubric({"size": size, "required": required, "has_proof": proof,
                                        "has_system_design": design, "edge_cases": 4,
                                        "judge_latency_ms": 80})
                    reachable.add(generated["band"])
                    check(generated["total_points"] == 100, "rubric points must sum to 100")
    check(reachable == {"easy", "medium", "hard", "expert"}
          and all(sum(weight for _, weight in weights) == 100
                  for weights in RUBRIC_WEIGHTS.values()),
          f"not every band is reachable or weights do not sum to 100: {sorted(reachable)}")

    # 6) negative checks: malformed profiles are rejected, and size alone can never make a
    #    problem hard (its contribution is capped).
    rejected = []
    for broken in ({"size": 0}, {"size": -5}, {"required": ("O(n!)",)}, {"difficulty": "easy"}):
        try:
            score_profile(broken)
        except ValueError as error:
            rejected.append(str(error))
    check(len(rejected) == 4 and "O(n!)" in rejected[2] and "difficulty" in rejected[3]
          and classify(size_points(10 ** 12)) == "easy"
          and rubric({"size": 10 ** 12})["band"] == "easy",
          f"malformed profiles must be rejected: {rejected}")

    # 7) letter grades are consistent with the band via an independently written mapping.
    mapping = {"easy": "D", "medium": "C", "hard": "B", "expert": "A"}
    computed = {value: letter(value) for value in (0, 24, 25, 49, 50, 74, 75, 90)}
    check(all(computed[value] == mapping[classify(value)] for value in computed)
          and [computed[value] for value in sorted(computed)] == ["D", "D", "C", "C", "B", "B",
                                                                  "A", "A"],
          f"letter grades inconsistent: {computed}")

    # 8) purity and determinism: generating a rubric leaves the profile untouched and repeated
    #    calls are equal.
    pristine = {"size": 200_000, "required": ("O(n log n)",), "edge_cases": 4,
                "judge_latency_ms": 100}
    snapshot = {key: value for key, value in pristine.items()}
    check(rubric(pristine) == rubric(pristine) and pristine == snapshot
          and rubric(pristine)["expected_complexity"] == "O(n log n)",
          "rubric generation must be pure and deterministic")

    # 9) boundary: an empty profile scores zero, lands in the lowest band with the default
    #    complexity, and rewards the floor of every optional feature.
    empty = rubric({})
    floor = rubric({"size": 1, "required": (), "has_proof": False, "has_system_design": False,
                    "edge_cases": 0, "judge_latency_ms": 0})
    check(empty == floor and empty["score"] == 0 and empty["band"] == "easy"
          and empty["expected_complexity"] == "O(1)" and empty["breakdown"]["size"] == 0
          and empty["breakdown"]["latency"] == 0,
          f"the empty profile must score zero: {empty['breakdown']}")

    print(f"band={graded['band']} score={graded['score']} breakdown={graded['breakdown']} "
          f"boundaries={boundaries}")
    print(f"bands_reachable={sorted(reachable)} letters={computed} "
          f"size_only_band={rubric({'size': 10 ** 12})['band']}")
    print("PASS")


if __name__ == "__main__":
    main()
