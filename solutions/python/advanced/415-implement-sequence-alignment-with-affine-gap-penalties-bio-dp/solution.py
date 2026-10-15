"""Task 415 - Implement sequence alignment with affine gap penalties (bio-DP). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Gotoh's algorithm aligns two strings under affine gap costs. Cost model (documented):
  * substitution: MISMATCH = 2, matched characters cost 0,
  * a gap of length L costs OPEN + (L - 1) * EXTEND, with OPEN = 2 and EXTEND = 1:
    opening the run costs OPEN, every extra character costs EXTEND.
Three DP states drive the recurrence:
  * X - the column ends with a (mis)match of the last two characters,
  * Y - the column ends with a gap in b (a character of a is deleted),
  * Z - the column ends with a gap in a (a character of b is inserted).
Gap-to-gap switches are allowed so that with EXTEND == OPEN the optimum equals
Needleman-Wunsch with a linear gap cost of OPEN per character (asserted below).
An alignment is a list of ops ('=', x, y) match; ('!', x, y) substitution;
('-', x, '') deletion; ('+', '', y) insertion; replaying the ops rebuilds b.
"""

MISMATCH = 2
OPEN = 2
EXTEND = 1
INF = float("inf")


def gotoh(a, b, mismatch=MISMATCH, open_cost=OPEN, extend=EXTEND):
    """Returns (cost, ops) for the optimal affine-gap global alignment."""
    n, m = len(a), len(b)
    X = [[INF] * (m + 1) for _ in range(n + 1)]
    Y = [[INF] * (m + 1) for _ in range(n + 1)]
    Z = [[INF] * (m + 1) for _ in range(n + 1)]
    X[0][0] = 0
    for i in range(1, n + 1):
        Y[i][0] = open_cost + (i - 1) * extend
    for j in range(1, m + 1):
        Z[0][j] = open_cost + (j - 1) * extend
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if a[i - 1] == b[j - 1] else mismatch
            X[i][j] = min(X[i - 1][j - 1], Y[i - 1][j - 1], Z[i - 1][j - 1]) + sub
            Y[i][j] = min(X[i - 1][j] + open_cost, Z[i - 1][j] + open_cost, Y[i - 1][j] + extend)
            Z[i][j] = min(X[i][j - 1] + open_cost, Y[i][j - 1] + open_cost, Z[i][j - 1] + extend)
    end_state = min(range(3), key=lambda s: (X, Y, Z)[s][n][m])
    cost = (X, Y, Z)[end_state][n][m]
    # backtrack (ties: X before Z before Y for X, X before Z before Y overall order below)
    ops = []
    state, i, j = end_state, n, m
    while i > 0 or j > 0:
        if state == 0:
            sub = 0 if a[i - 1] == b[j - 1] else mismatch
            ops.append(("=" if sub == 0 else "!", a[i - 1], b[j - 1]))
            cands = (X[i - 1][j - 1], Y[i - 1][j - 1], Z[i - 1][j - 1])
            state = min(range(3), key=lambda s: cands[s])
            assert cands[state] + sub == X[i][j]
            i, j = i - 1, j - 1
        elif state == 1:
            ops.append(("-", a[i - 1], ""))
            cands = (X[i - 1][j] + open_cost, Z[i - 1][j] + open_cost, Y[i - 1][j] + extend)
            k = min(range(3), key=lambda s: cands[s])
            assert cands[k] == Y[i][j]
            state = (0, 2, 1)[k]
            i -= 1
        else:
            ops.append(("+", "", b[j - 1]))
            cands = (X[i][j - 1] + open_cost, Y[i][j - 1] + open_cost, Z[i][j - 1] + extend)
            state = min(range(3), key=lambda s: cands[s])
            assert cands[state] == Z[i][j]
            j -= 1
    ops.reverse()
    return cost, ops


def needleman_wunsch(a, b, gap, mismatch=MISMATCH):
    """Linear gap cost: every gap character costs `gap` regardless of run length."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap
    for j in range(1, m + 1):
        dp[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if a[i - 1] == b[j - 1] else mismatch
            dp[i][j] = min(dp[i - 1][j] + gap, dp[i][j - 1] + gap, dp[i - 1][j - 1] + sub)
    return dp[n][m]


def replay(a, ops):
    out = []
    ai = 0
    for kind, x, y in ops:
        if kind in ("=", "!", "-"):
            assert a[ai] == x, (ai, x)
            ai += 1
        if kind in ("=", "!", "+"):
            out.append(y)
    assert ai == len(a)
    return "".join(out)


def ops_cost(ops, mismatch=MISMATCH, open_cost=OPEN, extend=EXTEND):
    """Recompute the cost from the op pattern alone, validating the cost model."""
    total = 0
    i = 0
    while i < len(ops):
        kind = ops[i][0]
        if kind == "!":
            total += mismatch
            i += 1
        elif kind in ("-", "+"):
            run = 0
            while i < len(ops) and ops[i][0] == kind:
                run += 1
                i += 1
            total += open_cost + (run - 1) * extend
        else:
            i += 1
    return total


def lcg_string(length, seed, alphabet):
    out = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        out.append(alphabet[seed % len(alphabet)])
    return "".join(out), seed


def main():
    # equal strings align for free
    cost, ops = gotoh("GATTACA", "GATTACA")
    assert cost == 0 and all(op[0] == "=" for op in ops) and len(ops) == 7

    # a single substitution (2) beats delete+insert (2 + 2)
    cost, ops = gotoh("a", "b")
    assert cost == 2 and ops == [("!", "a", "b")]

    # one insertion: a single gap of length 1 costs OPEN = 2
    cost, ops = gotoh("ACGT", "ACCGT")
    assert cost == 2 and ops_cost(ops) == 2 and replay("ACGT", ops) == "ACCGT"

    # one gap of length 2 costs OPEN + EXTEND = 3; two separate gaps would cost 4
    cost, ops = gotoh("acgt", "acgggt")
    assert cost == 3, cost
    assert replay("acgt", ops) == "acgggt"
    assert ops_cost(ops) == 3
    assert needleman_wunsch("acgt", "acgggt", 2) == 4          # two 1-cost-per-char gaps
    assert needleman_wunsch("ab", "aZZb", 2) == 4
    cost_ab, ops_ab = gotoh("ab", "aZZb")
    assert cost_ab == 3 and ops_cost(ops_ab) == 3              # one open + one extend
    assert gotoh("acgt", "acgggt", 2, 2, 2)[0] == needleman_wunsch("acgt", "acgggt", 2)

    # empty-string cases: a gap of length L costs OPEN + (L-1)*EXTEND
    assert gotoh("", "") == (0, [])
    cost, ops = gotoh("abc", "")
    assert cost == OPEN + 2 * EXTEND == 4 and len(ops) == 3 and all(op[0] == "-" for op in ops)
    cost, ops = gotoh("", "abc")
    assert cost == 4 and all(op[0] == "+" for op in ops)
    assert gotoh("", "AAAAA")[0] == OPEN + 4 * EXTEND == 6
    assert needleman_wunsch("", "AAAAA", 2) == 10

    # longer hand case: 3-character insertion in the middle
    cost, ops = gotoh("ABCD", "ABXYZCD")
    assert cost == OPEN + 2 * EXTEND == 4
    assert replay("ABCD", ops) == "ABXYZCD" and ops_cost(ops) == 4

    # replay + cost-model consistency on the hand pairs
    for a_s, b_s in [("kitten", "sitting"), ("sunday", "saturday"), ("flaw", "lawn")]:
        cost, ops = gotoh(a_s, b_s)
        assert replay(a_s, ops) == b_s, (a_s, b_s)
        assert ops_cost(ops) == cost, (a_s, b_s, cost)

    # with EXTEND == OPEN the affine optimum equals Needleman-Wunsch
    seed = 20270
    alphabet = "acgt"
    for _ in range(8):
        la = 25 + seed % 40
        a, seed = lcg_string(la, seed, alphabet)
        lb = 20 + seed % 45
        b, seed = lcg_string(lb, seed, alphabet)
        affine, ops = gotoh(a, b, 2, 3, 3)
        linear = needleman_wunsch(a, b, 3)
        assert affine == linear, (a, b, affine, linear)
        assert replay(a, ops) == b and ops_cost(ops, 2, 3, 3) == affine
        # a cheaper affine extension can never lose against the linear model
        assert gotoh(a, b, 2, 2, 1)[0] <= linear

    # gap-run counting sanity: a long deletion is one open plus many extends
    cost, ops = gotoh("ABCDEFGH", "AB")
    assert cost == OPEN + 5 * EXTEND == 7
    kinds = [op[0] for op in ops]
    assert kinds.count("-") == 6 and ops_cost(ops) == 7
    assert replay("ABCDEFGH", ops) == "AB"


if __name__ == "__main__":
    main()
    print("PASS")
