"""Task 645 - Implement materialized view maintenance incremental algorithm.  (Tier: Expert, Ch 14 — Databases)

Incremental maintenance of a join view `V = R(a, b) JOIN S(b, c)` under bag (multiset)
semantics, plus an aggregate view over it.  Instead of rebuilding the view, a transaction
that changes R by `dRi`/`dRd` and S by `dSi`/`dSd` updates the view with the delta

    dV = (dRi JOIN S) + (R JOIN dSi) + (dRi JOIN dSi) + (dRd JOIN dSd)
       - (dRd JOIN S) - (R JOIN dSd) - (dRd JOIN dSi) - (dRi JOIN dSd)

evaluated against the *old* R and S (all eight terms are computed before the bases are
mutated).  The counting-algorithm correction term `+(dRd JOIN dSd)` is what keeps a
transaction that deletes a matching pair of rows from subtracting two view tuples for one
view tuple, and the two mixed terms mean the algorithm never has to assume that the insert
and the delete halves of a transaction are disjoint.  Multiplicities multiply through the
join, which is exactly bag semantics.

Invariants verified: (a) after *every* step the incrementally maintained view equals a full
recompute, (b) the maintained delta equals `recompute(V_new) - recompute(V_old)` as a
multiset, (c) multiplicities are exact (deleting a row held twice removes two contributions
per matching row), (d) removing a view tuple that is not there is rejected, (e) an empty
transaction changes nothing, and (f) incremental maintenance touches far fewer join pairs
than rebuilding the view.
"""

from collections import Counter


def lcg(state):
    return (state * 1103515245 + 12345) % (2 ** 31)


def join_view(r_side, s_side, work):
    """Multiset equi-join of R-side (a, b) tuples with S-side (b, c) tuples -> (a, c)."""
    index = {}
    for (b, c), count in s_side.items():
        index.setdefault(b, []).append((c, count))
    out = Counter()
    for (a, b), rcount in r_side.items():
        for c, scount in index.get(b, ()):
            out[(a, c)] += rcount * scount
            work[0] += 1
    return out


def full_recompute(r_relation, s_relation, work):
    return join_view(r_relation, s_relation, work)


def subtract(base, delta):
    """Remove `delta` from the multiset `base`; refuse to remove what is not there."""
    for key, count in delta.items():
        available = base.get(key, 0)
        if available < count:
            raise AssertionError("cannot delete %r x%d: only %d present" % (key, count, available))
    for key, count in delta.items():
        base[key] -= count
        if base[key] == 0:
            del base[key]


def add(base, delta):
    for key, count in delta.items():
        base[key] += count


def delta_add(accumulator, term):
    """Multiset addition that keeps non-positive counts (Counter operators drop them)."""
    for key, count in term.items():
        accumulator[key] += count


def delta_sub(accumulator, term):
    for key, count in term.items():
        accumulator[key] -= count


def apply_transaction(r_relation, s_relation, view, transaction, work):
    """Compute dV from the eight delta terms, then mutate view and bases.

    Expanding (R - dRd + dRi) JOIN (S - dSd + dSi) against the *old* R and S gives four
    positive and four negative terms; keeping the mixed ones (dRd JOIN dSi and
    dRi JOIN dSd) means the algorithm does not have to assume that the insert and the
    delete parts of a transaction are disjoint.
    """
    d_ri, d_rd, d_si, d_sd = transaction
    delta = Counter()
    for r_part, s_part in ((d_ri, s_relation), (r_relation, d_si),
                           (d_ri, d_si), (d_rd, d_sd)):
        if r_part and s_part:
            delta_add(delta, join_view(r_part, s_part, work))
    for r_part, s_part in ((d_rd, s_relation), (r_relation, d_sd),
                           (d_rd, d_si), (d_ri, d_sd)):
        if r_part and s_part:
            delta_sub(delta, join_view(r_part, s_part, work))
    view.update(delta)
    for key in [k for k, count in view.items() if count <= 0]:
        del view[key]
    subtract(r_relation, d_rd)
    add(r_relation, d_ri)
    subtract(s_relation, d_sd)
    add(s_relation, d_si)
    return delta


def multiset_difference(after, before):
    """Signed multiset difference; Counter's own `-` drops negative counts."""
    out = Counter()
    for key in set(after) | set(before):
        diff = after.get(key, 0) - before.get(key, 0)
        if diff:
            out[key] = diff
    return out


def aggregate_view(view):
    out = Counter()
    for (a, c), count in view.items():
        out[a] += count
    return out


def random_transaction(r_relation, s_relation, state, domain=3):
    """Build a small transaction; insertions and deletions never touch the same tuple."""
    d_ri, d_rd, d_si, d_sd = Counter(), Counter(), Counter(), Counter()
    state = lcg(state)
    mode = state % 3                                   # 0 insert, 1 delete, 2 both
    if mode in (0, 2):
        state = lcg(state)
        for _ in range(1 + state % 2):
            state = lcg(state)
            d_ri[(state % domain, (state // 7) % domain)] += 1
        for _ in range(1 + (state // 13) % 2):
            state = lcg(state)
            d_si[(state % domain, (state // 11) % domain)] += 1
    if mode in (1, 2):
        candidates = sorted(r_relation)
        if candidates:
            state = lcg(state)
            key = candidates[state % len(candidates)]
            d_rd[key] = 1 + (state >> 5) % min(2, r_relation[key])
        candidates = sorted(s_relation)
        if candidates:
            state = lcg(state)
            key = candidates[state % len(candidates)]
            d_sd[key] = 1 + (state >> 5) % min(2, s_relation[key])
    for key in list(d_rd):
        if key in d_ri:
            del d_rd[key]
    for key in list(d_sd):
        if key in d_si:
            del d_sd[key]
    return (d_ri, d_rd, d_si, d_sd), state


def main():
    # hand-checked multiplicity example --------------------------------------------------
    r_relation = Counter({(1, 10): 2})
    s_relation = Counter({(10, 100): 3})
    work = [0]
    view = full_recompute(r_relation, s_relation, work)
    check(view == {(1, 100): 6}, "bag semantics multiply multiplicities: %r" % (view,))
    check(aggregate_view(view) == {1: 6}, "the aggregate view sums the duplicates")

    # the correction term: deleting a matching pair removes exactly one view tuple --------
    pair_r = Counter({(1, 10): 1, (2, 10): 1})
    pair_s = Counter({(10, 100): 1, (10, 200): 1})
    pair_view = full_recompute(pair_r, pair_s, [0])
    check(pair_view == {(1, 100): 1, (1, 200): 1, (2, 100): 1, (2, 200): 1}, "four view tuples")
    delta = apply_transaction(pair_r, pair_s, pair_view,
                              (Counter(), Counter({(1, 10): 1}), Counter(), Counter({(10, 100): 1})),
                              [0])
    check(delta == Counter({(1, 100): -1, (1, 200): -1, (2, 100): -1}),
          "the correction term keeps the shared tuple at -1, not -2: %r" % (delta,))
    check(pair_view == full_recompute(pair_r, pair_s, [0]), "the maintained view is correct")
    check(pair_view == {(2, 200): 1}, "only the tuple without deleted input survives")

    # deleting a tuple held twice ---------------------------------------------------------
    dup_r = Counter({(1, 10): 3})
    dup_s = Counter({(10, 100): 2})
    dup_view = full_recompute(dup_r, dup_s, [0])
    check(dup_view == {(1, 100): 6}, "3 x 2 multiplicities")
    apply_transaction(dup_r, dup_s, dup_view,
                      (Counter(), Counter({(1, 10): 2}), Counter(), Counter()), [0])
    check(dup_view == {(1, 100): 2} and full_recompute(dup_r, dup_s, [0]) == dup_view,
          "removing two of three R rows removes two contributions per matching S row")

    # deleting a tuple that is not there is rejected --------------------------------------
    try:
        subtract(Counter({(9, 9): 1}), Counter({(7, 7): 1}))
    except AssertionError:
        pass
    else:
        raise AssertionError("removing an absent row must fail")

    # the long random sequence: incremental == full recompute after every step ------------
    r_relation = Counter({(a, b): 1 for a in range(5) for b in range(5)})
    s_relation = Counter({(b, c): 1 for b in range(5) for c in range(5)})
    incremental_work, recompute_work = [0], [0]
    view = full_recompute(r_relation, s_relation, incremental_work)
    aggregate = aggregate_view(view)
    check(view == full_recompute(r_relation, s_relation, recompute_work), "initial view")
    check(aggregate == aggregate_view(full_recompute(r_relation, s_relation, [0])),
          "initial aggregate view")

    state, steps, changed = 20240601, 0, 0
    for step in range(60):
        transaction, state = random_transaction(r_relation, s_relation, state, domain=5)
        if not any(transaction):
            continue
        before = Counter(view)
        delta = apply_transaction(r_relation, s_relation, view, transaction, incremental_work)
        recomputed = full_recompute(r_relation, s_relation, recompute_work)
        check(view == recomputed,
              "step %d: maintained view differs from the recomputed view" % step)
        check(multiset_difference(recomputed, before) == delta,
              "step %d: the delta is not the signed view difference" % step)
        for (a, c), count in delta.items():           # maintain the aggregate view too
            aggregate[a] = aggregate.get(a, 0) + count
        aggregate = Counter({a: n for a, n in aggregate.items() if n})
        check(aggregate == aggregate_view(view),
              "step %d: the incremental aggregate view is stale" % step)
        steps += 1
        changed += sum(abs(n) for n in delta.values())
        check(all(count > 0 for count in view.values()), "the view holds no empty tuples")

    # empty transaction: nothing changes --------------------------------------------------
    frozen_r, frozen_s = Counter(r_relation), Counter(s_relation)
    frozen_view = Counter(view)
    empty_delta = apply_transaction(r_relation, s_relation, view,
                                    (Counter(), Counter(), Counter(), Counter()),
                                    incremental_work)
    check(empty_delta == {} and view == frozen_view, "an empty transaction changes nothing")
    check(r_relation == frozen_r and s_relation == frozen_s, "the bases are untouched")

    check(steps > 20, "the random sequence produced a decent number of transactions")
    check(changed > 10, "the sequence really changed the view")
    check(incremental_work[0] * 4 < recompute_work[0],
          "incremental maintenance does far less work: %d vs %d join pairs"
          % (incremental_work[0], recompute_work[0]))
    print("[645] view maintenance: %d transactions, %d delta tuples, %d/%d join pairs"
          % (steps, changed, incremental_work[0], recompute_work[0]))


def check(ok, message):
    if not ok:
        raise AssertionError(message)


if __name__ == "__main__":
    main()
    print("PASS")
