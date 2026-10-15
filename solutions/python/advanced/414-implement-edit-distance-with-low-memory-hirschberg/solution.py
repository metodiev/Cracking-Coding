"""Task 414 - Implement edit distance with low memory (Hirschberg). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Classic Levenshtein DP uses an O(n*m) table. Hirschberg's algorithm recovers an
optimal alignment with O(n) auxiliary memory and O(n*m) time:
  * split b in half (Myers-Miller variant of the split), compute the forward row
    f[i] = dist(a[:i], b[:mid]) and the backward row g[i] = dist(a[i:], b[mid:])
    with two rolling rows each,
  * pick the split k minimizing f[k] + g[k], recurse on (a[:k], b[:mid]) and
    (a[k:], b[mid:]); base cases (short side, or an empty string) solve directly.
Tie-break: the smallest k wins, so the result is deterministic.
An alignment is a list of ops ('=', x, y): x == y match; ('!', x, y) substitution;
('-', x, '') deletion; ('+', '', y) insertion. Replaying the ops from a rebuilds b.
"""

BASE = 16


def full_dp_dist(a, b):
    """Oracle: full O(n*m) table, traceable but space-hungry."""
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            sub = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub)
        prev = cur
    return prev[m]


def _forward_row(a, prefix):
    """f[i] = dist(a[:i], prefix) for every i; one rolling row over the prefix."""
    m = len(prefix)
    prev = list(range(m + 1))
    out = [prev[m]]
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            sub = 0 if a[i - 1] == prefix[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub)
        prev = cur
        out.append(prev[m])
    return out


def _backward_row(a, suffix):
    """g[i] = dist(a[i:], suffix) for every i.

    Reversal preserves edit distance, so dist(a[i:], suffix) equals
    dist(reverse(a)[: n-i], reverse(suffix)): one rolling row over the reversed
    suffix records the column we need, then the collected values are flipped.
    """
    m = len(suffix)
    ra = a[::-1]
    rs = suffix[::-1]
    prev = list(range(m + 1))
    collected = [prev[m]]                          # p = 0: dist("", reversed suffix)
    for p in range(1, len(ra) + 1):
        cur = [p] + [0] * m
        for j in range(1, m + 1):
            sub = 0 if ra[p - 1] == rs[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub)
        prev = cur
        collected.append(prev[m])
    return collected[::-1]


def direct_align(a, b):
    """Full-table DP with backtracking; used for the recursion base cases."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + sub)
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("=", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("!", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("-", a[i - 1], ""))
            i -= 1
        else:
            ops.append(("+", "", b[j - 1]))
            j -= 1
    ops.reverse()
    return dp[n][m], ops


def hirschberg_align(a, b, base=BASE):
    if not a:
        return len(b), [("+", "", c) for c in b]
    if not b:
        return len(a), [("-", c, "") for c in a]
    if min(len(a), len(b)) <= max(base, 1):
        return direct_align(a, b)
    mid = len(b) // 2
    f = _forward_row(a, b[:mid])
    g = _backward_row(a, b[mid:])
    split = min(range(len(a) + 1), key=lambda i: f[i] + g[i])
    d1, o1 = hirschberg_align(a[:split], b[:mid], base)
    d2, o2 = hirschberg_align(a[split:], b[mid:], base)
    return d1 + d2, o1 + o2


def replay(a, ops):
    """Apply an alignment to a; returns the produced string."""
    out = []
    ai = 0
    for kind, x, y in ops:
        if kind == "=" or kind == "!" or kind == "-":
            assert a[ai] == x, (ai, x)
            ai += 1
        if kind == "=" or kind == "!" or kind == "+":
            out.append(y)
    assert ai == len(a)
    return "".join(out)


def lcg_string(length, seed, alphabet):
    out = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        out.append(alphabet[seed % len(alphabet)])
    return "".join(out), seed


def main():
    # hand-derived distances (Levenshtein, substitution cost 1)
    assert full_dp_dist("kitten", "sitting") == 3
    assert full_dp_dist("flaw", "lawn") == 2
    assert full_dp_dist("sunday", "saturday") == 3
    assert full_dp_dist("", "") == 0
    assert full_dp_dist("a", "b") == 1
    assert full_dp_dist("abc", "") == 3
    assert full_dp_dist("abc", "abc") == 0

    dist, ops = hirschberg_align("kitten", "sitting")
    assert dist == 3, dist
    assert replay("kitten", ops) == "sitting"
    assert [op[0] for op in ops].count("=") == 4      # matches i,t,t,n are forced
    assert [op[0] for op in ops].count("!") == 2      # k->s and e->i
    assert [op[0] for op in ops].count("+") == 1      # trailing g
    assert [op[0] for op in ops].count("-") == 0

    dist, ops = hirschberg_align("flaw", "lawn", base=1)
    assert dist == 2, dist
    assert replay("flaw", ops) == "lawn"
    kinds = [op[0] for op in ops]
    assert kinds.count("!") + kinds.count("-") + kinds.count("+") == 2
    assert kinds.count("=") == 3                      # l, a, w stay matched

    # base-case paths: empty strings, pure insertions/deletions, equal strings
    assert hirschberg_align("", "abc") == (3, [("+", "", "a"), ("+", "", "b"), ("+", "", "c")])
    assert hirschberg_align("abc", "") == (3, [("-", "a", ""), ("-", "b", ""), ("-", "c", "")])
    assert hirschberg_align("", "") == (0, [])
    d, ops = hirschberg_align("same", "same")
    assert d == 0 and replay("same", ops) == "same" and all(op[0] == "=" for op in ops)

    # both base settings must agree with the full table on hand and random pairs
    pairs = [("kitten", "sitting"), ("flaw", "lawn"), ("sunday", "saturday"), ("abc", "yabd")]
    for a, b in pairs:
        for base in (0, 1, 3, 16, 1000):
            d, ops = hirschberg_align(a, b, base)
            assert d == full_dp_dist(a, b), (a, b, base)
            assert replay(a, ops) == b, (a, b, base)
    assert full_dp_dist("abc", "yabd") == 2           # y insertion + c->d substitution

    seed = 4242
    alphabet = "abcd"
    for round_index in range(10):
        la = 30 + seed % 60
        a, seed = lcg_string(la, seed, alphabet)
        lb = 25 + seed % 75
        b, seed = lcg_string(lb, seed, alphabet)
        want = full_dp_dist(a, b)
        for base in (2, 16, 400):
            d, ops = hirschberg_align(a, b, base)
            assert d == want, (round_index, base)
            assert replay(a, ops) == b, (round_index, base)
            kinds = [op[0] for op in ops]
            assert kinds.count("=") + kinds.count("!") + kinds.count("-") == len(a)
            assert kinds.count("=") + kinds.count("!") + kinds.count("+") == len(b)
            assert kinds.count("!") + kinds.count("-") + kinds.count("+") == want
            for kind, x, y in ops:
                if kind == "=":
                    assert x == y
                if kind == "!":
                    assert x != y

    # recurrence sanity: mid rows of the recursion match the full-table boundary values
    a, b = "alignment", "alignmentx"
    mid = len(b) // 2
    f = _forward_row(a, b[:mid])
    g = _backward_row(a, b[mid:])
    assert f[len(a)] == full_dp_dist(a, b[:mid])
    assert g[0] == full_dp_dist(a, b[mid:])
    for i in range(len(a) + 1):
        assert f[i] == full_dp_dist(a[:i], b[:mid]), i
        assert g[i] == full_dp_dist(a[i:], b[mid:]), i
    split = min(range(len(a) + 1), key=lambda i: f[i] + g[i])
    assert f[split] + g[split] == full_dp_dist(a, b)


if __name__ == "__main__":
    main()
    print("PASS")
