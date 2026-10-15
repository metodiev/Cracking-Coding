
"""Task 170 - Palindrome partitioning backtracking (list partitions). (Tier: Beginner, CTCI Ch 8)

Enumerate every way to cut the string into palindromic pieces.
Palindromes are checked with two pointers; memoizing the check is a later optimization.
"""


def is_palindrome(s):
    i, j = 0, len(s) - 1
    while i < j:
        if s[i] != s[j]:
            return False
        i, j = i + 1, j - 1
    return True


def palindrome_partitions(s):
    """All partitions of s into palindromic substrings; [[]] for the empty string."""
    out = []
    current = []

    def backtrack(start):
        if start == len(s):
            out.append(current[:])
            return
        for end in range(start + 1, len(s) + 1):
            piece = s[start:end]
            if is_palindrome(piece):
                current.append(piece)
                backtrack(end)
                current.pop()

    backtrack(0)
    return out


def main():
    assert palindrome_partitions("aab") == [["a", "a", "b"], ["aa", "b"]]
    assert palindrome_partitions("a") == [["a"]]
    assert palindrome_partitions("") == [[]]
    assert palindrome_partitions("aaa") == [["a", "a", "a"], ["a", "aa"], ["aa", "a"], ["aaa"]]
    assert palindrome_partitions("ab") == [["a", "b"]]
    # every produced partition concatenates back to the input and each piece is a palindrome
    for partition in palindrome_partitions("aabb"):
        assert "".join(partition) == "aabb"
        assert all(is_palindrome(p) for p in partition)


if __name__ == "__main__":
    main()
    print("PASS")
