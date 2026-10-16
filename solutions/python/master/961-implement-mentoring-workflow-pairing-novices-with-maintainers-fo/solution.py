"""Task 961 - Implement mentoring workflow pairing novices with maintainers for PR review.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

Pairing model: every novice must review with a maintainer whose expertise overlaps their target
areas, and each maintainer can take at most `capacity` novices.  The pairer runs a Kuhn
augmenting-path matching over that capacity-expanded bipartite graph, walking each novice's mentors
in descending affinity order, so the result is maximum-cardinality and biased towards good pairings;
an unmatched novice therefore means the whole mentor capacity is used up.  Each pairing schedules
`REVIEWS_PER_PAIRING` reviews, which gives the maintainer's review load.

Simplified: affinities are a small closed-form score, capacities are integers and no calendar is
modelled.  Verified: capacity feasibility (an assignment is a function novice -> mentor respecting
capacity), maximum cardinality both by formula and against an exhaustive oracle that also compares
total affinity, deterministic tie-breaking under permuted input, load accounting, and the
empty/zero-capacity boundaries.
"""
REVIEWS_PER_PAIRING = 2
WANT_BONUS = 3


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def affinity(novice, mentor):
    """Overlap of target areas plus a bonus when the mentor covers the novice's biggest gap."""
    overlap = len(set(novice["areas"]) & set(mentor["areas"]))
    wants = novice.get("wants", ())
    return 2 * overlap + (WANT_BONUS if wants and wants[0] in mentor["areas"] else 0)


def normalise(people, kind):
    cleaned = {}
    for name, record in people.items():
        capacity = int(record.get("capacity", 1))
        if kind == "mentor" and capacity < 0:
            raise ValueError(f"mentor {name!r} has negative capacity {capacity}")
        cleaned[name] = {"areas": tuple(record.get("areas", ())), "capacity": capacity,
                         "wants": tuple(record.get("wants", ()))}
    return cleaned


def match(novices, mentors):
    """Maximum-cardinality pairing with capacities; returns novice -> mentor."""
    novices, mentors = normalise(novices, "novice"), normalise(mentors, "mentor")
    order = {novice: sorted(mentors, key=lambda name: (-affinity(novices[novice], mentors[name]),
                                                       name))
             for novice in sorted(novices)}
    assigned = {mentor: [] for mentor in sorted(mentors)}

    def augment(novice, visited):
        for mentor in order[novice]:
            if mentor in visited:
                continue
            visited.add(mentor)
            if len(assigned[mentor]) < mentors[mentor]["capacity"]:
                assigned[mentor].append(novice)
                return True
            for displaced in list(assigned[mentor]):
                if augment(displaced, visited):
                    assigned[mentor].remove(displaced)
                    assigned[mentor].append(novice)
                    return True
        return False

    for novice in sorted(novices):
        augment(novice, set())
    return {novice: mentor for mentor in sorted(assigned) for novice in assigned[mentor]}


def total_affinity(novices, mentors, pairing):
    return sum(affinity(normalise(novices, "novice")[novice],
                        normalise(mentors, "mentor")[mentor])
               for novice, mentor in pairing.items())


def brute_force_best(novices, mentors):
    """Exhaustive oracle: maximise (pair count, total affinity) over every legal assignment."""
    novices, mentors = normalise(novices, "novice"), normalise(mentors, "mentor")
    names = sorted(novices)
    best = (0, 0)

    def walk(index, loads, pairs):
        nonlocal best
        if index == len(names):
            best = max(best, (len(pairs), sum(affinity(novices[one], mentors[two])
                                              for one, two in pairs)))
            return
        walk(index + 1, loads, pairs)
        for mentor in sorted(mentors):
            if loads[mentor] < mentors[mentor]["capacity"]:
                loads[mentor] += 1
                walk(index + 1, loads, pairs + [(names[index], mentor)])
                loads[mentor] -= 1

    walk(0, {mentor: 0 for mentor in mentors}, [])
    return best


def review_load(pairing, mentors):
    """Reviews each maintainer must commit to for their assigned novices."""
    loads = {mentor: 0 for mentor in sorted(mentors)}
    for mentor in pairing.values():
        loads[mentor] += REVIEWS_PER_PAIRING
    return loads


def report(pairing, mentors):
    """Per-maintainer view, sorted, with the novices they were paired with."""
    grouped = {mentor: sorted(novice for novice, owner in pairing.items() if owner == mentor)
               for mentor in sorted(mentors)}
    return {mentor: {"novices": novices, "reviews": len(novices) * REVIEWS_PER_PAIRING}
            for mentor, novices in grouped.items()}


NOVICES = {"nina": {"areas": ("graphs",), "wants": ("graphs",)},
           "omar": {"areas": ("parsing",), "wants": ("parsing",)},
           "pia": {"areas": ("graphs", "parsing")},
           "quinn": {"areas": ("math",)},
           "rosa": {"areas": ("graphs",)}}
MENTORS = {"alan": {"areas": ("graphs",), "capacity": 2},
           "bea": {"areas": ("parsing", "math"), "capacity": 2},
           "cid": {"areas": ("systems",), "capacity": 1}}


def main():
    pairing = match(NOVICES, MENTORS)
    mentors = normalise(MENTORS, "mentor")

    # 1) happy path: capacity suffices, so every novice is paired and no mentor is overloaded.
    loads = {mentor: 0 for mentor in mentors}
    for mentor in pairing.values():
        loads[mentor] += 1
    check(len(pairing) == 5 and set(pairing) == set(NOVICES)
          and all(loads[mentor] <= mentors[mentor]["capacity"] for mentor in mentors)
          and loads == {"alan": 2, "bea": 2, "cid": 1},
          f"pairing must be feasible and complete: {pairing}")

    # 2) maximum cardinality by formula: the count is min(novices, total capacity) whenever every
    #    novice is acceptable to every mentor (the graph is complete).
    capacity = sum(record["capacity"] for record in mentors.values())
    tight = {name: record for name, record in list(NOVICES.items())[:2]}
    check(len(match(NOVICES, MENTORS)) == min(len(NOVICES), capacity) == 5
          and len(match(NOVICES, {"alan": {"areas": ("graphs",), "capacity": 1}})) == 1
          and len(match(tight, MENTORS)) == 2,
          f"cardinality must be min(novices, capacity)={min(len(NOVICES), capacity)}")

    # 3) exhaustive oracle on a tiny instance: the augmenting-path result is cardinality-optimal and
    #    matches the best total affinity found by searching every assignment.
    tiny_novices = {"a": {"areas": ("graphs",), "wants": ("graphs",)},
                    "b": {"areas": ("parsing",), "wants": ("parsing",)},
                    "c": {"areas": ("graphs",)}}
    tiny_mentors = {"m1": {"areas": ("graphs",), "capacity": 2},
                    "m2": {"areas": ("parsing",), "capacity": 1}}
    tiny = match(tiny_novices, tiny_mentors)
    check((len(tiny), total_affinity(tiny_novices, tiny_mentors, tiny))
          == brute_force_best(tiny_novices, tiny_mentors),
          f"the match must agree with the exhaustive oracle: {tiny}")

    # 4) capacity exhaustion boundary: with capacity 3 only the first three novices are paired, and
    #    the unmatched tail is deterministic.
    scarce = {"alan": {"areas": ("graphs",), "capacity": 2}, "bea": {"areas": ("parsing",),
                                                                    "capacity": 1}}
    limited = match(NOVICES, scarce)
    check(len(limited) == 3 and sorted(set(NOVICES) - set(limited)) == ["quinn", "rosa"]
          and sorted(limited) == ["nina", "omar", "pia"],
          f"only min(novices, capacity) may be paired: {sorted(limited)}")

    # 5) affinity ordering matters: each novice is sent to the mentor covering their request when
    #    that mentor has room, and the graph specialists stay with the graph maintainer.
    clean = match({"omar": {"areas": ("parsing",), "wants": ("parsing",)}},
                  {"bea": {"areas": ("parsing",), "capacity": 1},
                   "cid": {"areas": ("systems",), "capacity": 1}})
    check(clean == {"omar": "bea"} and pairing["rosa"] == "alan" and pairing["pia"] == "alan"
          and affinity(tiny_novices["b"], tiny_mentors["m2"])
          > affinity(tiny_novices["b"], tiny_mentors["m1"]),
          f"affinity ordering wrong: {clean}")

    # 6) load accounting: reviews are conserved, and every maintainer stays inside their cap.
    loads = review_load(pairing, mentors)
    check(sum(loads.values()) == REVIEWS_PER_PAIRING * len(pairing)
          and loads["alan"] == 4 and loads["cid"] == 2
          and all(loads[mentor] <= 2 * mentors[mentor]["capacity"] for mentor in mentors)
          and report(pairing, mentors)["alan"]["novices"] == ["pia", "rosa"],
          f"review load must be conserved and capped: {loads}")

    # 7) boundaries: no novices pairs nothing, zero capacity pairs nothing, and a single compatible
    #    pair is matched.
    single = match({"solo": {"areas": ("graphs",), "wants": ("graphs",)}},
                   {"one": {"areas": ("graphs",), "capacity": 1}})
    check(match({}, MENTORS) == {} and match(NOVICES, {}) == {}
          and match(NOVICES, {"alan": {"areas": (), "capacity": 0}}) == {}
          and single == {"solo": "one"} and len(pairing) == len(report(pairing, mentors)) + 2,
          f"empty/single boundaries wrong: {single}")

    # 8) determinism under permuted input: reversing every dict yields the same assignment, and the
    #    report is sorted.
    reversed_novices = {name: NOVICES[name] for name in reversed(sorted(NOVICES))}
    reversed_mentors = {name: MENTORS[name] for name in reversed(sorted(MENTORS))}
    check(match(reversed_novices, reversed_mentors) == pairing
          and report(pairing, mentors) == report(pairing, reversed_mentors)
          and list(report(pairing, mentors)) == sorted(MENTORS),
          "pairing must be deterministic and order independent")

    # 9) negative check: a negative capacity is a configuration error, not a silently clamped zero.
    try:
        match(NOVICES, {"bad": {"areas": ("graphs",), "capacity": -1}})
        raise AssertionError("a negative capacity must be rejected")
    except ValueError as error:
        check("negative capacity" in str(error), f"unexpected error text: {error}")

    print(f"pairing={pairing} loads={loads} cardinality={len(pairing)}/{len(NOVICES)}")
    print(f"tiny_matching={tiny} best={brute_force_best(tiny_novices, tiny_mentors)} "
          f"unmatched_under_scarcity={sorted(set(NOVICES) - set(limited))}")
    print("PASS")


if __name__ == "__main__":
    main()
