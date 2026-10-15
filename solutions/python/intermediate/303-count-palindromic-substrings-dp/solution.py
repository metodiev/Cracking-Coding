"""Task 303 - Count palindromic substrings. (Tier: Intermediate, CTCI Ch 1)

Expand around all 2n-1 centres (each centre is one character or one gap); every successful
expansion is one palindromic substring. Cross-checked with an all-substrings brute force.
"""

def count_palindromic_substrings(s):
    n = len(s)
    count = 0
    for center in range(n):
        for left, right in ((center, center), (center, center + 1)):
            while left >= 0 and right < n and s[left] == s[right]:
                count += 1
                left -= 1
                right += 1
    return count


def brute_force(s):
    total = 0
    for i in range(len(s)):
        for j in range(i + 1, len(s) + 1):
            part = s[i:j]
            if part == part[::-1]:
                total += 1
    return total


def lcg_strings(length, count, seed=2468):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append("abc"[seed % 3])
        strings.append("".join(chars))
    return strings


def main():
    assert count_palindromic_substrings("abc") == 3
    assert count_palindromic_substrings("aaa") == 6
    assert count_palindromic_substrings("abba") == 6
    assert count_palindromic_substrings("") == 0
    assert count_palindromic_substrings("a") == 1

    for s in ["aab", "aba", "abcba", "abccba", "aaaa"]:
        assert count_palindromic_substrings(s) == brute_force(s), s

    for s in lcg_strings(9, 60):
        assert count_palindromic_substrings(s) == brute_force(s), s


if __name__ == "__main__":
    main()
    print("PASS")
