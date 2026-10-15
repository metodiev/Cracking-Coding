"""Task 230 - Longest palindrome substring (expand center). (Tier: Intermediate, CTCI Ch 1)

Expand around each of the 2n-1 centers: n odd centers on characters and n-1
even centers between them. The scan visits centers left to right and tries the
odd center before the even center; the first palindrome of maximal length wins
ties, so "babad" returns "bab" (seen at center 1) and not "aba" (center 2).
O(n^2) time, O(1) space. Empty input returns "".
"""


def longest_palindrome_substring(s):
    if not s:
        return ""
    start, length = 0, 1
    for c in range(len(s)):
        for l, r in ((c, c), (c, c + 1)):          # odd center, then even center
            a, b = l, r
            while a >= 0 and b < len(s) and s[a] == s[b]:
                a -= 1
                b += 1
            size = b - a - 1
            if size > length:                      # strict: first found wins ties
                start, length = a + 1, size
    return s[start:start + length]


def main():
    assert longest_palindrome_substring("babad") == "bab"
    assert longest_palindrome_substring("cbbd") == "bb"
    assert longest_palindrome_substring("a") == "a"
    assert longest_palindrome_substring("") == ""
    assert longest_palindrome_substring("aaaa") == "aaaa"
    assert longest_palindrome_substring("forgeeksskeegfor") == "geeksskeeg"
    assert longest_palindrome_substring("abaxyzzyxf") == "xyzzyx"
    assert longest_palindrome_substring("ac") == "a"       # tie: first found
    assert longest_palindrome_substring("aac") == "aa"
    assert longest_palindrome_substring("z") == "z"


if __name__ == "__main__":
    main()
    print("PASS")
