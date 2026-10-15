"""Task 290 - Palindrome partitioning with backtracking. (Tier: Intermediate, CTCI Ch 8)

All ways to cut a string into palindromic parts. Pieces are extended shortest-first, so
partition lists come out in lexicographic order and results are deterministic. Every
piece is validated with an independent palindrome checker and concatenation must restore
the input; cross-checked against an oracle that enumerates all 2^(n-1) cut sets.
"""

def is_palindrome(text):
    left, right = 0, len(text) - 1
    while left < right:
        if text[left] != text[right]:
            return False
        left += 1
        right -= 1
    return True


def palindrome_partitioning(s):
    """All palindrome partitions, prefixes taken shortest first (lexicographic order)."""
    results = []
    current = []

    def backtrack(start):
        if start == len(s):
            results.append(current[:])
            return
        for end in range(start + 1, len(s) + 1):
            piece = s[start:end]
            if is_palindrome(piece):
                current.append(piece)
                backtrack(end)
                current.pop()

    backtrack(0)
    return results


def brute_force_partitions(s):
    """Oracle: enumerate all 2^(n-1) cut sets and keep the palindrome-only partitions."""
    n = len(s)
    if n == 0:
        return [[]]
    results = []
    for mask in range(1 << (n - 1)):
        parts = []
        start = 0
        for i in range(n - 1):
            if (mask >> i) & 1:
                parts.append(s[start:i + 1])
                start = i + 1
        parts.append(s[start:])
        if all(is_palindrome(part) for part in parts):
            results.append(parts)
    return sorted(results)


def canonical(partitions):
    return sorted(",".join(parts) for parts in partitions)


class Lcg:
    """Deterministic generator: next = (next * 75 + 74) % 65537."""

    def __init__(self, seed=13579):
        self.state = seed

    def next(self):
        self.state = (self.state * 75 + 74) % 65537
        return self.state


def main():
    assert palindrome_partitioning("aab") == [["a", "a", "b"], ["aa", "b"]]
    assert palindrome_partitioning("a") == [["a"]]
    assert palindrome_partitioning("aba") == [["a", "b", "a"], ["aba"]]
    assert palindrome_partitioning("aaa") == [["a", "a", "a"], ["a", "aa"], ["aa", "a"], ["aaa"]]
    assert palindrome_partitioning("aaaa") == brute_force_partitions("aaaa")
    assert len(palindrome_partitioning("aaaa")) == 8     # every substring is palindromic
    assert len(palindrome_partitioning("abba")) == 3
    assert palindrome_partitioning("") == [[]]           # zero parts concatenate to ""

    rng = Lcg()
    for trial in range(200):
        length = 1 + rng.next() % 7
        s = "".join("abc"[rng.next() % 3] for _ in range(length))
        partitions = palindrome_partitioning(s)
        assert partitions == brute_force_partitions(s), s
        assert partitions == sorted(partitions), s         # documented lexicographic order
        assert canonical(partitions) == canonical(brute_force_partitions(s)), s
        seen = set()
        for parts in partitions:
            assert "".join(parts) == s, (s, parts)
            for piece in parts:
                assert piece and is_palindrome(piece), (s, piece)
            key = ",".join(parts)
            assert key not in seen, (s, key)
            seen.add(key)
        assert len(partitions) == len(seen)


if __name__ == "__main__":
    main()
    print("PASS")
