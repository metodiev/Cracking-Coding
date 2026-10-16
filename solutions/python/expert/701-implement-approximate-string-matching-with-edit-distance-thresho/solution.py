"""Task 701 - Implement approximate string matching with edit distance threshold using Ukkonen.  (Tier: Expert, CTCI Beyond - Advanced Data Structures & Algorithms)

Approximate string matching: report every end offset j (0-based, exclusive) such that some
substring T[s:j] of the text matches the pattern within k edits, i.e. every j with
    D[m][j] = min_{0 <= s <= j} edit_distance(pattern, T[s:j]) <= k.

The DP is Sellers'/Ukkonen's: D[0][j] = 0 for every j (the match may start anywhere, the text
before it is free) and D[i][0] = i; the recurrence is the usual
    D[i][j] = min(D[i-1][j] + 1, D[i][j-1] + 1, D[i-1][j-1] + (pattern[i-1] != text[j-1])).
Three Ukkonen-style cutoffs turn it from O(m*n) into something far cheaper on real inputs:

  * cell cutoff -- a cell whose value exceeds k can never be an ancestor of a value <= k
    (values only ever grow by 1 per step), so every cell > k is stored as the sentinel k+1;
  * left diagonal prune -- matching i pattern characters inside j text characters costs at
    least |i - j|, hence D[i][j] > k for every j < i - k; those columns are never evaluated;
  * row termination -- the row minima min_j D[i][j] are non-decreasing in i: deleting the last
    pattern character from an optimal alignment of P[0..i+1) against a text suffix leaves an
    alignment of P[0..i) against a text suffix that is no more expensive, so every value of row
    i+1 is at least the minimum of row i.  The first row whose minimum exceeds k therefore
    proves that row m has no value <= k, and the search stops with an empty answer.

The row window [max(0, i-k), last_live + 1] follows from the recurrence: a cell of row i+1 that
is <= k needs a <= k cell of row i to its left (or directly above), so the right edge of the
window can advance by at most one column per row.

Simplified versus Ukkonen's full algorithm: no diagonal/streaming formulation and no bit-parallel
(Myers) inner loop, so the cost bound is the measured one (window cells + early termination)
rather than the textbook O(n*k) band; what is verified is that the pruning changes no answer.
Verified: the pruned DP equals an unpruned full-table run and equals a brute-force oracle that
tries every start offset with a textbook Levenshtein DP, k = 0 degenerates to exact substring
search, the row-monotonicity invariant behind the termination holds on full tables, and the
measured cell savings are real (strictly fewer cells than the full table).
"""
import itertools
import sys


def check(ok, message):
    if not ok:
        raise AssertionError(message)


# --------------------------------------------------------------------------- oracles
def full_distance(a, b):
    """Textbook full Levenshtein DP -- the ground truth used by the brute-force oracle."""
    previous = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i] + [0] * len(b)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1,
                             previous[j - 1] + (ai != b[j - 1]))
        previous = current
    return previous[len(b)]


def brute_matches(pattern, text, k):
    """Every j whose best suffix match T[s:j] of the pattern costs at most k edits."""
    hits = []
    for j in range(len(text) + 1):
        best = min(full_distance(pattern, text[s:j]) for s in range(j + 1))
        if best <= k:
            hits.append(j)
    return hits


# --------------------------------------------------------------------------- the DP
def sellers_table(pattern, text, k):
    """The complete, uncapped-window D table with values above k collapsed to k + 1."""
    m, n = len(pattern), len(text)
    inf = k + 1
    table = [[0] * (n + 1)]
    for i in range(1, m + 1):
        row = [inf] * (n + 1)
        row[0] = i
        for j in range(1, n + 1):
            value = min(table[i - 1][j] + 1, row[j - 1] + 1,
                        table[i - 1][j - 1] + (pattern[i - 1] != text[j - 1]))
            row[j] = value if value <= k else inf
        table.append(row)
    return table


def sellers_matches(pattern, text, k, stats=None, prune=True):
    """All end offsets j with the best suffix-match cost of `pattern` in text[:j] <= k.

    `stats` collects the measured work: window cells evaluated, rows processed, and whether the
    row-minimum termination fired.  With prune=False the identical recurrence is run over the
    full table so the checks can prove the pruning is answer-preserving.
    """
    if k < 0:
        raise ValueError("threshold must be non-negative")
    m, n = len(pattern), len(text)
    inf = k + 1
    stats = stats if stats is not None else {}
    stats["cells"] = 0
    stats["rows"] = 0
    stats["terminated"] = False

    previous = [0] * (n + 1)          # row 0: a match may start at any column, cost 0
    last_live = n                     # rightmost column of row 0 that is <= k (all of them)
    cells = 0
    rows = 0
    if m == 0:                        # the empty pattern matches every offset at cost 0
        stats["rows"] = 0
        return list(range(n + 1))

    for i in range(1, m + 1):
        if prune:
            lo = max(0, i - k)        # left diagonal prune: every j < i - k has cost > k
            hi = min(n, last_live + 1)  # a <= k cell needs a <= k cell of the previous row
        else:
            lo, hi = 0, n
        rows = i
        if lo > hi:
            stats["cells"] = cells
            stats["rows"] = i - 1
            stats["terminated"] = True
            return []
        current = [inf] * (n + 1)
        for j in range(lo, hi + 1):
            if j == 0:
                value = i             # D[i][0] = i: the match starts at the text beginning
            else:
                value = inf
                if previous[j] < inf:                # delete pattern[i-1]
                    candidate = previous[j] + 1
                    if candidate < value:
                        value = candidate
                if j - 1 >= lo:                      # insert text[j-1]
                    candidate = current[j - 1] + 1
                    if candidate < value:
                        value = candidate
                if previous[j - 1] < inf:            # match / substitute
                    candidate = previous[j - 1] + (pattern[i - 1] != text[j - 1])
                    if candidate < value:
                        value = candidate
                if value > inf:
                    value = inf
            current[j] = value
            cells += 1
        row_min = min(current[lo:hi + 1])
        if prune and row_min > k:     # row minima are non-decreasing -> no later match exists
            stats["cells"] = cells
            stats["rows"] = i
            stats["terminated"] = True
            return []
        if prune:
            live = [j for j in range(hi, lo - 1, -1) if current[j] <= k]
            last_live = live[0] if live else lo
        previous = current

    stats["cells"] = cells
    stats["rows"] = rows
    hits = [j for j in range(lo if prune else 0, (hi if prune else n) + 1) if previous[j] <= k]
    return hits


# --------------------------------------------------------------------------- checks
def test_curated_cases():
    # hand-verified small instances; the offsets are the (exclusive) end positions of a match
    cases = [("a", "a", 0, [1]),
             ("ab", "ab", 1, [1, 2]),
             ("abc", "abcabc", 0, [3, 6]),
             ("abc", "abcabc", 1, [2, 3, 4, 5, 6]),
             ("aa", "aaa", 1, [1, 2, 3]),
             ("abcd", "bcd", 1, [3]),
             ("xyz", "abcd", 2, [])]
    for pattern, text, k, expected in cases:
        got = sellers_matches(pattern, text, k)
        check(got == expected, f"curated {pattern!r}/{text!r} k={k}: {got} != {expected}")
        check(got == brute_matches(pattern, text, k), "curated case must match the oracle")


def test_k_zero_is_exact_search():
    text = "the theory of thermometers"
    expected = sorted(i + 3 for i in range(len(text) + 1) if text.startswith("the", i))
    got = sellers_matches("the", text, 0)
    check(got == expected, f"k = 0 must equal exact substring search: {got} != {expected}")
    check(got == brute_matches("the", text, 0), "k = 0 against the oracle")
    check(sellers_matches("zzz", text, 0) == [], "absent pattern has no exact occurrence")


def test_against_brute_force_random():
    mask = (1 << 64) - 1
    seed = 0x9E3779B97F4A7C15
    alphabet = "ab"
    for trial in range(90):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        m = seed % 5
        n = (seed >> 17) % 7
        pattern = "".join(alphabet[(seed >> (5 * i + 1)) & 1] for i in range(m))
        text = "".join(alphabet[(seed >> (3 * i + 2)) & 1] for i in range(n))
        for k in (0, 1, 2):
            got = sellers_matches(pattern, text, k)
            expected = brute_matches(pattern, text, k)
            check(got == expected,
                  f"random trial {trial} {pattern!r}/{text!r} k={k}: {got} != {expected}")


def test_pruning_preserves_answers():
    mask = (1 << 64) - 1
    seed = 0xDEADBEEFCAFEF00D
    alphabet = "abc"
    for trial in range(80):
        seed = (seed * 6364136223846793005 + 1442695040888963407) & mask
        m = 1 + seed % 6
        n = (seed >> 13) % 9
        pattern = "".join(alphabet[(seed >> (4 * i)) % 3] for i in range(m))
        text = "".join(alphabet[(seed >> (4 * i + 2)) % 3] for i in range(n))
        for k in (0, 1, 2, 3):
            pruned = sellers_matches(pattern, text, k)
            unpruned = sellers_matches(pattern, text, k, prune=False)
            check(pruned == unpruned,
                  f"pruning changed the answer {pattern!r}/{text!r} k={k}: "
                  f"{pruned} != {unpruned}")
            check(pruned == brute_matches(pattern, text, k), "full-table run vs oracle")


def test_row_monotonicity_invariant():
    # min_j D[i][j] is non-decreasing in i -- the property behind the row-minimum termination.
    # (The stronger cell-wise claim D[i][j] <= D[i+1][j] is false, e.g. "kitten" vs "sitting".)
    checked = 0
    for m in range(4):
        for n in range(5):
            for pattern in itertools.product("ab", repeat=m):
                for text in itertools.product("ab", repeat=n):
                    pat, txt = "".join(pattern), "".join(text)
                    for k in (0, 1, 2):
                        table = sellers_table(pat, txt, k)
                        minima = [min(row) for row in table]
                        check(all(x <= y for x, y in zip(minima, minima[1:])),
                              f"row minima must be non-decreasing: {pat!r}/{txt!r} k={k} "
                              f"{minima}")
                        checked += 1
    check(checked >= 700, f"exhaustive sweep covered {checked} tables")

    # and the rule it justifies: termination implies the oracle finds nothing
    for pat, txt, k in [("\u03a9a", "bbbb", 0), ("abc", "xyz", 1), ("abab", "aaaa", 1)]:
        stats = {}
        check(sellers_matches(pat, txt, k, stats) == [],
              f"expected no match for {pat!r}/{txt!r} k={k}")
        if stats["terminated"]:
            check(brute_matches(pat, txt, k) == [],
                  f"terminated but the oracle found matches in {pat!r}/{txt!r} k={k}")


def test_boundaries():
    check(sellers_matches("", "", 0) == [0], "empty pattern, empty text: one match at offset 0")
    check(sellers_matches("", "abc", 0) == [0, 1, 2, 3],
          "the empty pattern matches at every offset with cost 0")
    check(sellers_matches("abc", "", 2) == [], "empty text cannot host a 3-character pattern")
    check(sellers_matches("ab", "", 2) == [0], "the whole pattern may be deleted")
    check(sellers_matches("a", "a", 0) == [1], "single-character exact match")
    check(sellers_matches("a", "b", 1) == [0, 1], "single-character substitution, free start")
    check(sellers_matches("aaaa", "aa", 2) == [2], "pattern longer than the text, cost 2")
    check(sellers_matches("abc", "xyz", 3) == [0, 1, 2, 3], "threshold covers the whole pattern")
    try:
        sellers_matches("a", "a", -1)
        raise AssertionError("negative threshold must be rejected")
    except ValueError:
        pass
    # every reported offset is a valid, sorted, de-duplicated end position
    hits = sellers_matches("ab", "ababab", 1)
    check(hits == sorted(set(hits)) and all(0 <= j <= 6 for j in hits),
          f"offsets must be sorted, unique and in range: {hits}")


def test_termination_and_cost_model():
    stats = {}
    text = "g" * 400
    pattern = "\u03a9" * 30                       # a letter that never occurs in the text
    hits = sellers_matches(pattern, text, 2, stats)
    check(hits == [], "no match for an absent pattern")
    check(stats["terminated"], "the row-minimum cutoff must fire")
    check(stats["rows"] <= 3, f"termination after k + 1 = 3 rows, got {stats['rows']}")
    full_cells = (len(pattern) + 1) * (len(text) + 1)
    check(stats["cells"] < full_cells // 5,
          f"terminated run must be much cheaper: {stats['cells']} vs {full_cells}")

    stats = {}
    hits = sellers_matches("gattaca", "g" * 400, 2, stats)
    check(hits == [], "no match once the pattern diverges from the homopolymer")
    check(stats["terminated"], "cutoff fires after the substituted prefix runs out of budget")
    check(stats["rows"] == 4, f"three substitutions exceed k = 2 at row 4, got {stats['rows']}")

    stats = {}
    hits = sellers_matches("abc", "abc" * 100, 1, stats)
    check(hits[0] == 2 and len(hits) == 299,
          f"every offset from 2 on matches within one edit, got first={hits[0]} n={len(hits)}")
    check(stats["cells"] < (len("abc") + 1) * (len("abc" * 100) + 1),
          "the matching case still saves the pruned corner cells")
    check(not stats["terminated"], "a successful search must not terminate early")


def main():
    test_curated_cases()
    test_k_zero_is_exact_search()
    test_against_brute_force_random()
    test_pruning_preserves_answers()
    test_row_monotonicity_invariant()
    test_boundaries()
    test_termination_and_cost_model()

    stats = {}
    text = "secure shell access to the secure server" * 8
    hits = sellers_matches("secure", text, 1, stats)
    print(f"matches={len(hits)} window_cells={stats['cells']} rows={stats['rows']} "
          f"full_cells={(len('secure') + 1) * (len(text) + 1)}")
    print(f"python={sys.version_info.major}.{sys.version_info.minor}")
    print("PASS")


if __name__ == "__main__":
    main()
