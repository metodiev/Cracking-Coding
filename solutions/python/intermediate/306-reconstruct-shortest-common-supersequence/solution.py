"""Task 306 - Reconstruct the shortest common supersequence. (Tier: Intermediate, CTCI Ch 1)

Build the SCS length table, then walk it backwards: equal characters are emitted once,
otherwise follow the cheaper neighbour. Ties prefer the side of a (documented).
"""

def shortest_common_supersequence(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + 1

    i, j, chars = n, m, []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            chars.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] <= dp[i][j - 1]:       # tie prefers a's character
            chars.append(a[i - 1])
            i -= 1
        else:
            chars.append(b[j - 1])
            j -= 1
    while i > 0:
        chars.append(a[i - 1])
        i -= 1
    while j > 0:
        chars.append(b[j - 1])
        j -= 1
    return "".join(reversed(chars))


def is_subsequence(candidate, text):
    it = iter(text)
    return all(ch in it for ch in candidate)


def lcs_length(a, b):
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            current[j] = previous[j - 1] + 1 if a[i - 1] == b[j - 1] else max(previous[j], current[j - 1])
        previous = current
    return previous[len(b)]


def lcg_strings(length, count, seed=3141):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append("ab"[seed % 2])
        strings.append("".join(chars))
    return strings


def main():
    assert shortest_common_supersequence("ab", "b") == "ab"
    assert shortest_common_supersequence("abc", "abc") == "abc"
    assert shortest_common_supersequence("", "abc") == "abc"
    assert shortest_common_supersequence("abc", "") == "abc"
    assert shortest_common_supersequence("", "") == ""

    a, b = "AGGTAB", "GXTXAYB"
    scs = shortest_common_supersequence(a, b)
    assert len(scs) == 9 and len(scs) == len(a) + len(b) - lcs_length(a, b)
    assert is_subsequence(a, scs) and is_subsequence(b, scs)

    strings = lcg_strings(5, 40)
    for i in range(0, len(strings) - 1, 2):
        x, y = strings[i], strings[i + 1]
        candidate = shortest_common_supersequence(x, y)
        assert len(candidate) == len(x) + len(y) - lcs_length(x, y), (x, y)
        assert is_subsequence(x, candidate) and is_subsequence(y, candidate), (x, y)


if __name__ == "__main__":
    main()
    print("PASS")
