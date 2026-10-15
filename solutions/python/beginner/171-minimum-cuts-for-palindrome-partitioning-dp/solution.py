
"""Task 171 - Minimum cuts for palindrome partitioning (DP). (Tier: Beginner, CTCI Ch 8)

is_pal[i][j] table + cuts[i] = min cuts for the prefix ending at i.
O(n^2) time and space.
"""


def min_palindrome_cuts(s):
    n = len(s)
    if n <= 1:
        return 0
    is_pal = [[False] * n for _ in range(n)]
    cuts = [0] * n
    for end in range(n):
        best = end                      # worst case: cut between every pair
        for start in range(end + 1):
            if s[start] == s[end] and (end - start <= 1 or is_pal[start + 1][end - 1]):
                is_pal[start][end] = True
                best = 0 if start == 0 else min(best, cuts[start - 1] + 1)
        cuts[end] = best
    return cuts[n - 1]


def main():
    assert min_palindrome_cuts("aab") == 1            # aa | b
    assert min_palindrome_cuts("a") == 0
    assert min_palindrome_cuts("") == 0
    assert min_palindrome_cuts("ab") == 1
    assert min_palindrome_cuts("aabb") == 1           # aa | bb
    assert min_palindrome_cuts("abcba") == 0          # already a palindrome
    assert min_palindrome_cuts("abcd") == 3


if __name__ == "__main__":
    main()
    print("PASS")
