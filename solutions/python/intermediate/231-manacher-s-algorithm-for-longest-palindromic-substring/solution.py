"""Task 231 - Manacher's algorithm for longest palindromic substring. (Tier: Intermediate, CTCI Ch 1)

Transforming s into "^#" + "#".join(s) + "#$" makes every palindrome odd length,
so a single radius array suffices: p[i] equals the length of the longest palindrome
in s centred at transformed position i. Reusing the mirror of the current
right-most palindrome keeps the scan linear -> O(n) time, O(n) space.

Tie behaviour: when several palindromes share the maximum length the LEFTMOST one
is returned (the strict ">" comparison keeps the earliest centre).
"""


def longest_palindromic_substring(s):
    """Manacher: leftmost longest palindromic substring of s; "" for empty input."""
    if not s:
        return ""
    t = "^#" + "#".join(s) + "#$"
    n = len(t)
    p = [0] * n
    centre = right = 0
    best_len = 0
    best_start = 0
    for i in range(1, n - 1):
        mirror = 2 * centre - i
        if i < right:
            p[i] = min(right - i, p[mirror])
        while t[i + 1 + p[i]] == t[i - 1 - p[i]]:
            p[i] += 1
        if i + p[i] > right:
            centre, right = i, i + p[i]
        if p[i] > best_len:                       # strict: ties keep the leftmost centre
            best_len = p[i]
            best_start = (i - p[i]) // 2
    return s[best_start:best_start + best_len]


def expand_around_centre_length(s):
    """O(n^2) reference helper: length of the longest palindromic substring of s."""
    best = 0
    for i in range(len(s)):
        for lo, hi in ((i, i), (i, i + 1)):
            while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
                lo -= 1
                hi += 1
            best = max(best, hi - lo - 1)
    return best


def main():
    assert longest_palindromic_substring("babad") == "bab"      # leftmost of "bab"/"aba"
    assert longest_palindromic_substring("cbbd") == "bb"
    assert longest_palindromic_substring("forgeeksskeegfor") == "geeksskeeg"
    assert longest_palindromic_substring("abacaba") == "abacaba"
    assert longest_palindromic_substring("bananas") == "anana"
    assert longest_palindromic_substring("abcde") == "a"
    assert longest_palindromic_substring("ccc") == "ccc"
    assert longest_palindromic_substring("") == ""
    assert longest_palindromic_substring("a") == "a"

    import random

    rng = random.Random(231)
    for _ in range(300):                          # cross-check length vs expand-around-centre
        s = "".join(rng.choice("ab") for _ in range(rng.randint(0, 40)))
        got = longest_palindromic_substring(s)
        assert got == got[::-1]
        assert (got == "" or got in s)
        assert len(got) == expand_around_centre_length(s)
        if s:
            first = next(i for i in range(len(s) - len(got) + 1)
                         if s[i:i + len(got)] == s[i:i + len(got)][::-1])
            assert s.index(got) == first          # leftmost among the maximal palindromes


if __name__ == "__main__":
    main()
    print("PASS")
