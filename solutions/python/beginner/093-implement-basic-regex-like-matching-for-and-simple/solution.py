"""Task 093 - Implement basic regex-like matching for '.' and '*'. (Tier: Beginner, CTCI Ch 8)

'.' matches any single character, '*' matches zero or more of the preceding element.
Classic recursive DP with memoization on (string index, pattern index).
"""


def is_match(s, p):
    memo = {}

    def rec(i, j):
        if (i, j) in memo:
            return memo[(i, j)]
        if j == len(p):
            res = i == len(s)
        else:
            first = i < len(s) and (p[j] == s[i] or p[j] == ".")
            if j + 1 < len(p) and p[j + 1] == "*":
                res = rec(i, j + 2) or (first and rec(i + 1, j))
            else:
                res = first and rec(i + 1, j + 1)
        memo[(i, j)] = res
        return res

    return rec(0, 0)


def main():
    assert is_match("", "") is True
    assert is_match("", "a") is False
    assert is_match("", "a*") is True
    assert is_match("", ".*") is True
    assert is_match("a", "") is False
    assert is_match("a", "a") is True
    assert is_match("a", ".") is True
    assert is_match("a", "b") is False
    assert is_match("aa", "a") is False
    assert is_match("aa", "a*") is True
    assert is_match("aa", ".*") is True
    assert is_match("ab", ".*") is True
    assert is_match("aab", "c*a*b") is True
    assert is_match("abc", "ab.*c") is True
    assert is_match("abXYZc", "ab.*c") is True
    assert is_match("ab", "ab.*c") is False
    assert is_match("aaa", "a*a") is True
    assert is_match("aaa", "ab*a") is False
    assert is_match("mississippi", "mis*is*p*.") is False


if __name__ == "__main__":
    main()
    print("PASS")
