"""Task 297 - Edit distance variants (multi-dimensional DP). (Tier: Intermediate, CTCI Ch 8)

Levenshtein with configurable insert/delete/replace costs: dp[i][j] is the cost of
turning a[:i] into b[:j], with dp[i][0] = i * delete_cost and dp[0][j] = j * insert_cost,
then a min over the substitution (free when characters match), delete and insert
transitions. When replace_cost exceeds insert + delete, two edits beat one replacement:
"abc" -> "axc" costs 2 with replace_cost = 3. Cross-checked against a memoized
recursion written straight from the operation definition.
"""

from functools import lru_cache


def edit_distance(a, b, insert_cost=1, delete_cost=1, replace_cost=1):
    """Minimum cost to turn a into b using insert/delete/replace operations."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * delete_cost
    for j in range(1, m + 1):
        dp[0][j] = j * insert_cost
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            substitution = 0 if a[i - 1] == b[j - 1] else replace_cost
            dp[i][j] = min(
                dp[i - 1][j - 1] + substitution,
                dp[i - 1][j] + delete_cost,
                dp[i][j - 1] + insert_cost,
            )
    return dp[n][m]


def brute_edit(a, b, insert_cost, delete_cost, replace_cost):
    """Independent memoized recursion straight from the operation definition."""
    @lru_cache(maxsize=None)
    def rec(i, j):
        if i == 0:
            return j * insert_cost
        if j == 0:
            return i * delete_cost
        substitution = 0 if a[i - 1] == b[j - 1] else replace_cost
        return min(
            rec(i - 1, j - 1) + substitution,
            rec(i - 1, j) + delete_cost,
            rec(i, j - 1) + insert_cost,
        )

    return rec(len(a), len(b))


def lcg_string(seed, max_length):
    seed = (seed * 75 + 74) % 65537
    length = seed % (max_length + 1)
    chars = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        chars.append("abc"[seed % 3])
    return "".join(chars), seed


def main():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("flaw", "lawn") == 2
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("", "abcd") == 4
    assert edit_distance("abcd", "") == 4
    assert edit_distance("", "") == 0

    # replace 2 vs two edits: "ab" -> "ba" needs two replacements (4) or delete + insert (2)
    assert edit_distance("ab", "ba", replace_cost=2) == 2
    # replace 3 vs delete + insert: two edits beat one replacement
    assert edit_distance("abc", "axc", replace_cost=3) == 2
    assert edit_distance("abc", "axc", replace_cost=1) == 1

    # asymmetric costs: insertion cheap one way, the matching deletion expensive the other
    assert edit_distance("abc", "abcd", insert_cost=1, delete_cost=5) == 1
    assert edit_distance("abcd", "abc", insert_cost=1, delete_cost=5) == 5
    assert edit_distance("abc", "abcd", insert_cost=7, delete_cost=1) == 7

    seed = 31337
    for _ in range(200):
        seed = (seed * 75 + 74) % 65537
        a, seed = lcg_string(seed, 6)
        b, seed = lcg_string(seed, 6)
        seed = (seed * 75 + 74) % 65537
        insert_cost = 1 + seed % 3
        seed = (seed * 75 + 74) % 65537
        delete_cost = 1 + seed % 3
        seed = (seed * 75 + 74) % 65537
        replace_cost = 1 + seed % 3
        assert edit_distance(a, b, insert_cost, delete_cost, replace_cost) == brute_edit(
            a, b, insert_cost, delete_cost, replace_cost
        ), (a, b, insert_cost, delete_cost, replace_cost)


if __name__ == "__main__":
    main()
    print("PASS")
