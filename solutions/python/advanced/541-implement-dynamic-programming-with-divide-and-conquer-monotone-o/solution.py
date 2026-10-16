"""Task 541 - Implement dynamic programming with divide-and-conquer monotone optimization. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)

The classic "split a sequence into k contiguous groups" DP: dp[g][i] = minimum over j < i of
dp[g-1][j] + cost(j, i), where cost(j, i) = (pre[i] - pre[j] - shift)^2 uses prefix sums and is a
convex function of the segment sum, hence satisfies the quadrangle (Monge) inequality. That
inequality is exactly what makes the row minima monotone: the leftmost minimising cut position
opt(g, i) is non-decreasing in i, so instead of scanning all i - 1 candidates, the divide-and-
conquer optimisation recurses on the mid position and restricts the search window of the left half
to [opt_lo, opt(mid)] and of the right half to [opt(mid), opt_hi] - O(k n log n) instead of
O(k n^2). Ties are resolved to the smallest index, which is the selection that is provably
monotone, and unreachable states keep the sentinel value.

The implementation is cross-checked three ways: an exhaustive DP that scans every candidate j, an
explicit enumeration of all C(n-1, k-1) cut sets for tiny inputs (fully independent of the
recurrence), and the divide-and-conquer version itself, which must produce bit-identical tables.
The file also counts cost evaluations on both DP variants and asserts the divide-and-conquer
version wins, and it measures the monotonicity of the exhaustive argmin table explicitly.

Checks: the hand-checked [1, 2, 3, 4] instance (k = 2 gives 52 from 6^2 + 4^2, k = 1 gives 100,
k = 4 gives the sum of squares 30); infeasible group counts returning the sentinel and k = 0 or
n = 0 degenerate cases; equality with the exhaustive DP and with cut-set enumeration on
LCG arrays (n up to 60, k up to 10) including zeros, constant arrays and a non-zero shift; monotone
argmin tables; and the evaluation-count advantage.

One precondition is made explicit rather than glossed over: the window argument needs the prefix
sums to be non-decreasing, which is what converts the convexity of the segment cost into the
quadrangle inequality. The non-negative LCG instances and the constant array satisfy it and are
solved by the D&C version; for a sample containing negative items - where the prefix sequence
decreases and the monotone-argmin property genuinely fails - the file asserts the precondition is
violated, checks the exhaustive recurrence against cut-set enumeration instead, and then re-runs
the D&C on the absolute values, where the precondition holds again.
"""
import itertools

INF = 10 ** 18


def prefix_sums(values):
    pre = [0]
    for value in values:
        pre.append(pre[-1] + value)
    return pre


def segment_cost(pre, j, i, shift):
    delta = pre[i] - pre[j] - shift
    return delta * delta


def dp_exhaustive(values, k, shift=0):
    """Reference DP: every predecessor j is examined. Returns the whole table."""
    n = len(values)
    pre = prefix_sums(values)
    table = [[INF] * (n + 1) for _ in range(k + 1)]
    table[0][0] = 0
    evaluations = 0
    for g in range(1, k + 1):
        for i in range(g, n + 1):
            best = INF
            for j in range(g - 1, i):
                evaluations += 1
                candidate = table[g - 1][j] + segment_cost(pre, j, i, shift)
                if candidate < best:
                    best = candidate
            table[g][i] = best
    return table, evaluations


def dp_divide_and_conquer(values, k, shift=0):
    """Monotone-optimum D&C DP. Returns (table, cost evaluation count)."""
    n = len(values)
    pre = prefix_sums(values)
    evaluations = [0]
    previous = [INF] * (n + 1)
    previous[0] = 0
    table = [previous]
    for g in range(1, k + 1):
        current = [INF] * (n + 1)

        def solve(lo, hi, opt_lo, opt_hi):
            if lo > hi:
                return
            mid = (lo + hi) // 2
            lower = max(opt_lo, g - 1)
            upper = min(opt_hi, mid - 1)
            best = None
            best_j = lower
            for j in range(lower, upper + 1):
                evaluations[0] += 1
                candidate = previous[j] + segment_cost(pre, j, mid, shift)
                if best is None or candidate < best:
                    best = candidate
                    best_j = j
            current[mid] = best
            solve(lo, mid - 1, opt_lo, best_j)
            solve(mid + 1, hi, best_j, opt_hi)

        solve(g, n, g - 1, n - 1)
        table.append(current)
        previous = current
    return table, evaluations[0]


def dp_cut_enumeration(values, k, shift=0):
    """Fully independent reference for small n: enumerate every set of k - 1 cut positions."""
    n = len(values)
    if k == 0:
        return 0 if n == 0 else INF
    if k > n:
        return INF
    pre = prefix_sums(values)
    best = INF
    for cuts in itertools.combinations(range(1, n), k - 1):
        boundaries = (0,) + cuts + (n,)
        total = 0
        for index in range(k):
            total += segment_cost(pre, boundaries[index], boundaries[index + 1], shift)
        if total < best:
            best = total
    return best


def argmin_table(values, k, shift=0):
    """Leftmost optimal cut for every (g, i) from the exhaustive recurrence."""
    n = len(values)
    pre = prefix_sums(values)
    table = [[INF] * (n + 1) for _ in range(k + 1)]
    table[0][0] = 0
    argmin = [[-1] * (n + 1) for _ in range(k + 1)]
    for g in range(1, k + 1):
        for i in range(g, n + 1):
            best = INF
            best_j = -1
            for j in range(g - 1, i):
                candidate = table[g - 1][j] + segment_cost(pre, j, i, shift)
                if candidate < best:
                    best = candidate
                    best_j = j
            table[g][i] = best
            argmin[g][i] = best_j
    return argmin


def lcg_array(n, seed, modulus=15, offset=0):
    out = []
    for _ in range(n):
        seed = (seed * 75 + 74) % 65537
        out.append(seed % modulus - offset)
    return out, seed


def main():
    # --- Hand-checked instance ----------------------------------------------------------
    values = [1, 2, 3, 4]
    table, _ = dp_exhaustive(values, 4)
    assert table[1][4] == 100, "one group: 10^2"
    assert table[2][4] == 52, "6^2 + 4^2 beats 3^2 + 7^2 = 58 and 1 + 9^2 = 82"
    assert table[4][4] == 30, "four singletons: 1 + 4 + 9 + 16"
    best_three = min(
        segment_cost(prefix_sums(values), 0, a, 0)
        + segment_cost(prefix_sums(values), a, b, 0)
        + segment_cost(prefix_sums(values), b, 4, 0)
        for a in range(1, 4) for b in range(a + 1, 4))
    assert table[3][4] == best_three

    # --- Infeasible groupings and degenerate sizes --------------------------------------
    over, _ = dp_exhaustive(values, 5)
    assert over[5][4] == INF, "five non-empty groups do not fit in four elements"
    assert table[2][1] == INF, "two groups need at least two elements"
    empty, _ = dp_exhaustive([], 0)
    assert empty[0][0] == 0
    empty2, _ = dp_exhaustive([], 2)
    assert empty2[2][0] == INF, "no groups can be carved out of an empty sequence"
    none, _ = dp_exhaustive(values, 0)
    assert none[0][4] == INF, "zero groups cannot cover four elements"
    single, _ = dp_exhaustive([7], 1)
    assert single[1][1] == 49, "7^2"
    zeros, _ = dp_exhaustive([0, 0, 0], 2)
    assert zeros[2][3] == 0, "the all-zero sequence costs nothing however it is cut"

    # --- Cut enumeration agrees ---------------------------------------------------------
    seed = 5150
    for n in range(1, 9):
        for k in range(0, n + 2):
            values, seed = lcg_array(n, seed, modulus=9)
            table, _ = dp_exhaustive(values, k)
            assert table[k][n] == dp_cut_enumeration(values, k), "cut-set enumeration agrees"

    # --- D&C against the exhaustive DP on random arrays --------------------------------
    seed = 271828
    worst_ratio = 0.0
    for trial in range(12):
        n = 12 + 4 * trial
        k = 2 + trial % 7
        values, seed = lcg_array(n, seed, modulus=20)
        shift = 0 if trial % 3 else 7
        table_brute, evaluations_brute = dp_exhaustive(values, k, shift)
        table_dnc, evaluations_dnc = dp_divide_and_conquer(values, k, shift)
        assert table_dnc == table_brute, "the D&C table must be bit-identical"
        if n <= 12:
            assert table_dnc[k][n] == dp_cut_enumeration(values, k, shift), \
                "cut enumeration agrees on the small instances too"
        assert evaluations_dnc < evaluations_brute, "the window restriction saves evaluations"
        worst_ratio = max(worst_ratio, evaluations_dnc / evaluations_brute)
        # Every layer is monotone in its own right: the optimum never decreases with i.
        argmin = argmin_table(values, k, shift)
        for g in range(1, k + 1):
            previous_position = -1
            for i in range(g, n + 1):
                if argmin[g][i] < 0:
                    continue
                assert argmin[g][i] >= previous_position, "argmin is non-decreasing in i"
                previous_position = argmin[g][i]
        # The recurrence is exhaustive: every reachable state has a finite value.
        for g in range(1, k + 1):
            for i in range(g, n + 1):
                assert table_dnc[g][i] < INF, "states with g <= i are reachable"
            assert table_dnc[g][g - 1] == INF, "g groups need at least g elements"
    assert worst_ratio < 0.9, "the D&C restriction is a real saving on every instance"

    # --- Structured inputs: constants, negatives, heavy ties ---------------------------
    constant = [4] * 40
    table_brute, evaluations_brute = dp_exhaustive(constant, 6)
    table_dnc, evaluations_dnc = dp_divide_and_conquer(constant, 6)
    assert table_dnc == table_brute, "ties everywhere, still identical"
    # Six groups over forty equal elements 4: balancing the lengths 7, 7, 7, 7, 6, 6 minimises
    # the sum of squared group sums, giving 4 * 28^2 + 2 * 24^2 = 3136 + 1152 = 4288.
    assert table_dnc[6][40] == 4 * 28 * 28 + 2 * 24 * 24, "balanced groups minimise the squares"
    negative, _ = lcg_array(24, 42, modulus=9, offset=4)
    assert min(negative) < 0, "the sample really contains negative values"
    neg_pre = prefix_sums(negative)
    assert any(neg_pre[i + 1] < neg_pre[i] for i in range(len(negative))), \
        "with negative items the prefix sums are not monotone"
    # The divide-and-conquer window argument needs a non-decreasing prefix sequence (it is what
    # turns convexity of the segment cost into the quadrangle inequality), so for this instance
    # only the exhaustive recurrence and the independent cut enumeration are compared.
    table_brute, _ = dp_exhaustive(negative, 5)
    for k in (1, 2, 5, 12):
        assert dp_exhaustive(negative, k)[0][k][24] == dp_cut_enumeration(negative, k), \
            "the recurrence itself stays correct without the monotonicity precondition"
    assert table_brute[5][24] == dp_cut_enumeration(negative, 5)
    # Non-negative items restore the precondition, and then the D&C window is provably safe.
    nonneg = [abs(v) for v in negative]
    assert all(nonneg[i] >= 0 for i in range(len(nonneg)))
    pre_nonneg = prefix_sums(nonneg)
    assert all(pre_nonneg[i + 1] >= pre_nonneg[i] for i in range(len(nonneg)))
    table_brute, _ = dp_exhaustive(nonneg, 5)
    table_dnc, _ = dp_divide_and_conquer(nonneg, 5)
    assert table_dnc == table_brute, "non-negative values satisfy the Monge precondition"
    values, _ = lcg_array(10, 3, modulus=3)               # many equal prefix sums
    assert all(v >= 0 for v in values)
    table_brute, _ = dp_exhaustive(values, 4)
    table_dnc, _ = dp_divide_and_conquer(values, 4)
    assert table_dnc == table_brute, "heavy ties must not break the window argument"

    # --- Determinism --------------------------------------------------------------------
    values = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    first, _ = dp_divide_and_conquer(values, 4)
    second, _ = dp_divide_and_conquer(values, 4)
    assert first == second
    assert first[4][11] == dp_cut_enumeration(values, 4)


if __name__ == "__main__":
    main()
    print("PASS")
