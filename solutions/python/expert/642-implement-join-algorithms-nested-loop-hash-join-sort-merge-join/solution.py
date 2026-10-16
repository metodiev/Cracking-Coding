"""Task 642 - Implement join algorithms: nested-loop, hash join, sort-merge join.  (Tier: Expert, Ch 14 — Databases)

Three equality-join implementations over in-memory tuple relations, all with SQL NULL
semantics (a NULL key matches nothing, not even another NULL) and bag semantics (duplicate
keys multiply):

  * nested_loop_join  - O(|L| * |R|), needs no memory and no sort order, the correctness
                        reference used to cross-check the others.
  * hash_join         - build a chained hash table on the (smaller) right side, then probe
                        once per left row: |R| inserts + |L| probes, no sort order required.
  * sort_merge_join   - mark-and-scan over two inputs already sorted by their key; each
                        group of equal keys is expanded into its full cross product, and the
                        group cursors never move backwards, i.e. O(|L| + |R|) group steps.

Every function returns (rows, statistics) so the checks can assert the *cost* claims and not
only the relational results: the three algorithms must agree as multisets on random data
with duplicate keys, empty inputs and NULL keys, and the instrumentation must show the
expected probe/comparison counts.
"""

from collections import Counter

NULL = None
ALGORITHMS = ("nested_loop", "hash_join", "sort_merge")


def keys_match(left_key, right_key):
    """SQL equality: NULL is never equal to anything."""
    return left_key is not None and right_key is not None and left_key == right_key


def sort_relation(rows, key_index):
    """Sort by key ascending, NULL keys last (the order sort-merge join expects)."""
    return sorted(rows, key=lambda row: (1, 0) if row[key_index] is None else (0, row[key_index]))


def is_presorted(rows, key_index):
    return list(rows) == sort_relation(rows, key_index)


def nested_loop_join(left, right, left_key, right_key):
    rows, comparisons = [], 0
    for l in left:
        for r in right:
            comparisons += 1
            if keys_match(l[left_key], r[right_key]):
                rows.append(l + r)
    return rows, {"comparisons": comparisons, "probes": len(left) * len(right)}


def hash_join(left, right, left_key, right_key):
    buckets, inserts = {}, 0
    for r in right:
        if r[right_key] is None:
            continue                            # a NULL key can never match: skip the build
        buckets.setdefault(r[right_key], []).append(r)
        inserts += 1
    rows, probes = [], 0
    for l in left:
        probes += 1
        if l[left_key] is None:
            continue
        for r in buckets.get(l[left_key], ()):
            rows.append(l + r)
    return rows, {"comparisons": inserts + probes, "probes": probes, "build": inserts}


def sort_merge_join(left, right, left_key, right_key):
    if not is_presorted(left, left_key) or not is_presorted(right, right_key):
        raise AssertionError("sort-merge join requires inputs sorted by their join key")
    lrows = [r for r in left if r[left_key] is not None]
    rrows = [r for r in right if r[right_key] is not None]
    rows, comparisons, i, j = [], 0, 0, 0
    while i < len(lrows) and j < len(rrows):
        comparisons += 1
        lk, rk = lrows[i][left_key], rrows[j][right_key]
        if lk < rk:
            i += 1
        elif lk > rk:
            j += 1
        else:
            i2 = i
            while i2 < len(lrows) and lrows[i2][left_key] == lk:
                i2 += 1
                comparisons += 1
            j2 = j
            while j2 < len(rrows) and rrows[j2][right_key] == rk:
                j2 += 1
                comparisons += 1
            for a in range(i, i2):
                for b in range(j, j2):
                    rows.append(lrows[a] + rrows[b])
            i, j = i2, j2
    return rows, {"comparisons": comparisons}


def lcg(state):
    return (state * 1103515245 + 12345) % (2 ** 31)


def random_relation(count, key_domain, seed, payload_base, null_modulus=0):
    rows, state = [], seed
    for i in range(count):
        state = lcg(state)
        key = state % key_domain
        if null_modulus and state % null_modulus == 0:
            key = NULL
        rows.append((payload_base + i, key, state % 7))
    return rows, state


def expected_pair_count(left, right, left_key, right_key):
    """Independent oracle: multiplicities multiply per key value."""
    left_counts = Counter(r[left_key] for r in left if r[left_key] is not None)
    right_counts = Counter(r[right_key] for r in right if r[right_key] is not None)
    return sum(count * right_counts.get(key, 0) for key, count in left_counts.items())


def run_all(left, right, left_key, right_key):
    sorted_left = sort_relation(left, left_key)
    sorted_right = sort_relation(right, right_key)
    nl, nl_stats = nested_loop_join(left, right, left_key, right_key)
    hj, hj_stats = hash_join(left, right, left_key, right_key)
    sm, sm_stats = sort_merge_join(sorted_left, sorted_right, left_key, right_key)
    return {"nested_loop": (nl, nl_stats), "hash_join": (hj, hj_stats),
            "sort_merge": (sm, sm_stats)}


def main():
    # hand-checked anchors: the join key is column 0 here ------------------------------
    left = [(1, "a"), (2, "b")]
    right = [(2, "x"), (3, "y")]
    check(nested_loop_join(left, right, 0, 0)[0] == [(2, "b", 2, "x")], "nested loop anchor")
    check(hash_join(left, right, 0, 0)[0] == [(2, "b", 2, "x")], "hash join anchor")
    check(sort_merge_join(left, right, 0, 0)[0] == [(2, "b", 2, "x")], "merge join anchor")

    # all three agree, and agree with the multiplicity oracle, on random data -----------
    for trial in range(20):
        lrel, state = random_relation(30, 6, 1000 + trial * 7919, 0, null_modulus=7)
        rrel, _ = random_relation(25, 6, state, 1000, null_modulus=5)
        results = run_all(lrel, rrel, 1, 1)
        expected = expected_pair_count(lrel, rrel, 1, 1)
        counts = {alg: len(rows) for alg, (rows, _) in results.items()}
        check(len(set(counts.values())) == 1,
              "algorithms disagree on trial %d: %r" % (trial, counts))
        check(counts["nested_loop"] == expected,
              "row count differs from the oracle on trial %d" % trial)
        check(Counter(results["nested_loop"][0]) == Counter(results["hash_join"][0])
              == Counter(results["sort_merge"][0]),
              "algorithm results differ as multisets on trial %d" % trial)
        check(len(results["nested_loop"][0]) == len(set(results["nested_loop"][0])),
              "joined rows are unique thanks to distinct payloads")

    # instrumentation: each algorithm's work matches its analysis ------------------------
    lrel, state = random_relation(40, 5, 555, 0)
    rrel, _ = random_relation(60, 5, state, 1000)
    results = run_all(lrel, rrel, 1, 1)
    check(results["nested_loop"][1]["comparisons"] == 40 * 60, "nested loop compares all pairs")
    check(results["hash_join"][1]["build"] == 60 and results["hash_join"][1]["probes"] == 40,
          "hash join builds 60 buckets and probes 40 times")
    check(results["hash_join"][1]["comparisons"] == 100, "hash join total work is |L| + |R|")
    check(results["sort_merge"][1]["comparisons"] <= 2 * (40 + 60) + 2,
          "merge join runs in O(|L| + |R|): %d" % results["sort_merge"][1]["comparisons"])
    check(results["sort_merge"][1]["comparisons"] < results["nested_loop"][1]["comparisons"],
          "merge join does far less work than the nested loop here")

    # NULL keys never match, in either input, even against another NULL ------------------
    null_left = [(1, NULL), (2, 3)]
    null_right = [(9, NULL), (8, 3)]
    for alg, (rows, _) in run_all(null_left, null_right, 1, 1).items():
        check(rows == [(2, 3, 8, 3)], "%s must ignore NULL keys: %r" % (alg, rows))
    null_stats = {}
    _, null_stats["nested_loop"] = nested_loop_join(null_left, null_right, 1, 1)
    _, null_stats["hash_join"] = hash_join(null_left, null_right, 1, 1)
    check(null_stats["hash_join"]["build"] == 1, "the NULL right key is not hashed")

    # boundary cases --------------------------------------------------------------------
    for alg, (rows, _) in run_all([], rrel, 1, 1).items():
        check(rows == [], "%s must return nothing for an empty left side" % alg)
    for alg, (rows, _) in run_all(lrel, [], 1, 1).items():
        check(rows == [], "%s must return nothing for an empty right side" % alg)
    empty = run_all([], [], 1, 1)
    check(all(len(rows) == 0 for rows, _ in empty.values()), "empty join is empty")

    # all keys equal: the result is the full cross product ------------------------------
    same_left = [(i, 1) for i in range(4)]
    same_right = [(100 + i, 1) for i in range(5)]
    same = run_all(same_left, same_right, 1, 1)
    check(all(len(rows) == 20 for rows, _ in same.values()), "each algorithm pairs 4 x 5 rows")
    check(expected_pair_count(same_left, same_right, 1, 1) == 20, "oracle agrees on 4 x 5")

    # no matches at all ------------------------------------------------------------------
    disjoint = run_all([(1, 1)], [(2, 2)], 1, 1)
    check(all(len(rows) == 0 for rows, _ in disjoint.values()), "disjoint keys join to nothing")
    _, hj_stats = disjoint["hash_join"]
    check(hj_stats["build"] == 1 and hj_stats["probes"] == 1 and len(disjoint["hash_join"][0]) == 0,
          "hash join still does its linear work when nothing matches")

    # self-join: the row count is the sum of squared multiplicities ----------------------
    selfrel = [(i, i % 3) for i in range(12)]
    self_rows, _ = hash_join(selfrel, selfrel, 1, 1)
    counts = Counter(r[1] for r in selfrel)
    check(len(self_rows) == sum(c * c for c in counts.values()),
          "self-join multiplicity: %d vs %d" % (len(self_rows), sum(c * c for c in counts.values())))

    # sorted input is a precondition the merge join really enforces ----------------------
    check(is_presorted(sort_relation(lrel, 1), 1), "the sorted relation reports as presorted")
    check(not is_presorted([(1, 3), (2, 1)], 1), "an unsorted relation is detected")
    try:
        sort_merge_join([(1, 3), (2, 1)], [(1, 3)], 1, 1)
    except AssertionError:
        pass
    else:
        raise AssertionError("sort-merge join must reject unsorted input")

    # larger sorted inputs: merge join is dramatically cheaper than the nested loop ------
    big_left, state = random_relation(300, 50, 77, 0)
    big_right, _ = random_relation(400, 50, state, 1000)
    big = run_all(big_left, big_right, 1, 1)
    check(len(big["nested_loop"][0]) == expected_pair_count(big_left, big_right, 1, 1),
          "oracle agrees on the larger relation")
    check(big["sort_merge"][1]["comparisons"] * 100 < big["nested_loop"][1]["comparisons"],
          "merge join needs two orders of magnitude fewer comparisons: %d vs %d"
          % (big["sort_merge"][1]["comparisons"], big["nested_loop"][1]["comparisons"]))
    print("[642] joins: 300x400 rows, nested %d / hash %d / merge %d comparisons"
          % (big["nested_loop"][1]["comparisons"], big["hash_join"][1]["comparisons"],
             big["sort_merge"][1]["comparisons"]))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
    print("PASS")
