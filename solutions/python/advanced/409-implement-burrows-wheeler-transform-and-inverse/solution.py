"""Task 409 - Implement the Burrows-Wheeler transform and its inverse. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Cyclic BWT: sort all n cyclic rotations of s, take the last column, and record the *primary
index* - the row holding the rotation that starts at position 0. Hand anchors: banana ->
("nnbaaa", 3), abab -> ("bbaa", 0), aaaa -> ("aaaa", 0). Reversal is only possible with the
primary index: "ab" and "ba" both transform to "ba" but use primary 0 and 1.

Inverse: with F = sorted(L) (the first column) the LF-mapping row_of(L[i]) -> position in F
(counting equal characters in order) maps a rotation to the rotation starting one position
earlier; walking n steps from the primary row and reversing the collected characters rebuilds
the string. Ties between equal rotations are resolved by row index (stable sort).

Sentinel variant: BWT(s + "$") with a unique smallest character needs no primary index - the
row whose last character is "$" is the rotation starting at 0, so the same LF walk recovers s
exactly ("banana" -> "annb$aa").

Checks: hand values, LF-mapping is a permutation, multiset preserved, both inverses round-trip
on LCG strings including the empty string, and the medium case (300 characters).
"""

SENTINEL = "$"


def bwt_cyclic(s):
    """Return (last_column, primary_index) for the cyclic rotations of s."""
    n = len(s)
    if n == 0:
        return "", 0
    order = sorted(range(n), key=lambda i: s[i:] + s[:i])   # stable: equal rotations keep index order
    last = "".join(s[(i - 1) % n] for i in order)
    primary = order.index(0)
    return last, primary


def lf_mapping(last):
    """LF[i] = row of F holding L[i], with equal characters matched in order."""
    starts = {}
    total = 0
    for ch in sorted(set(last)):
        starts[ch] = total
        total += last.count(ch)
    seen = {}
    lf = [0] * len(last)
    for i, ch in enumerate(last):
        lf[i] = starts[ch] + seen.get(ch, 0)
        seen[ch] = seen.get(ch, 0) + 1
    return lf


def inverse_bwt_cyclic(last, primary):
    n = len(last)
    if n == 0:
        return ""
    lf = lf_mapping(last)
    row = primary
    collected = []
    for _ in range(n):
        collected.append(last[row])
        row = lf[row]
    assert row == primary, "LF walk must return to the starting row"
    return "".join(reversed(collected))


def bwt_sentinel(s):
    if SENTINEL in s:
        raise ValueError("sentinel variant requires texts without '$'")
    last, _ = bwt_cyclic(s + SENTINEL)
    return last


def inverse_bwt_sentinel(last):
    """Rebuild s from BWT(s + '$'); no primary index needed - '$' marks the start."""
    if last.count(SENTINEL) != 1:
        raise ValueError("expected exactly one sentinel")
    n = len(last) - 1
    lf = lf_mapping(last)
    row = last.index(SENTINEL)
    collected = []
    for _ in range(n + 1):
        collected.append(last[row])
        row = lf[row]
    # collected = "$", s[n-1], ..., s[0]; the reversed join is s + "$"
    return "".join(reversed(collected))[:-1]


def lcg_string(length, alphabet, seed):
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append(alphabet[seed % len(alphabet)])
    return "".join(chars), seed


def main():
    assert bwt_cyclic("banana") == ("nnbaaa", 3)
    assert bwt_cyclic("abab") == ("bbaa", 0)
    assert bwt_cyclic("aaaa") == ("aaaa", 0)
    assert bwt_cyclic("") == ("", 0)
    assert bwt_cyclic("a") == ("a", 0)
    assert bwt_cyclic("ba") == ("ba", 1)
    assert bwt_cyclic("ab") == ("ba", 0)

    # same last column, different primary index -> different original
    assert inverse_bwt_cyclic("ba", 0) == "ab"
    assert inverse_bwt_cyclic("ba", 1) == "ba"
    assert inverse_bwt_cyclic(*bwt_cyclic("banana")) == "banana"
    assert inverse_bwt_cyclic(*bwt_cyclic("mississippi")) == "mississippi"

    assert bwt_sentinel("banana") == "annb$aa"
    assert bwt_sentinel("") == "$"
    assert bwt_sentinel("a") == "a$"
    assert inverse_bwt_sentinel("$") == ""
    assert inverse_bwt_sentinel("annb$aa") == "banana"
    assert inverse_bwt_sentinel(bwt_sentinel("abracadabra")) == "abracadabra"

    last, primary = bwt_cyclic("mississippi")
    assert sorted(last) == sorted("mississippi")
    assert sorted(lf_mapping(last)) == list(range(len(last)))

    for alphabet in ("ab", "abc"):
        for trial in range(40):
            s, seed = lcg_string(trial % 41, alphabet, 13000 + trial * 65537)
            last, primary = bwt_cyclic(s)
            assert sorted(last) == sorted(s), s
            assert inverse_bwt_cyclic(last, primary) == s, s
            assert inverse_bwt_sentinel(bwt_sentinel(s)) == s, s

    medium, _ = lcg_string(300, "abc", 777777)
    last, primary = bwt_cyclic(medium)
    assert len(last) == 300
    assert inverse_bwt_cyclic(last, primary) == medium
    assert inverse_bwt_sentinel(bwt_sentinel(medium)) == medium

    try:
        bwt_sentinel("bad$text")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
