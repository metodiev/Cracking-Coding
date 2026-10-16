"""Task 527 - Implement minimal rotation (Booth's algorithm). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Booth's algorithm for the lexicographically least rotation in O(n): the string is doubled, two
candidate starts i < j advance a shared match length k while ss[i + k] == ss[j + k]; the losing
candidate is skipped past the whole matched block (i += k + 1, then pushed to at least j + 1), so
the total work is linear. The result is the *smallest* start index among all starts achieving the
minimum, which is what the brute-force reference returns as well. Documented checks: hand-verified
examples ("baba" -> 1 because "abab" is minimal, "banana" -> 5 for "anaban", "aaa" -> 0,
"abcabcabc" -> 0, "cba" -> 2, "abacaba" -> 6); exhaustive agreement with the brute-force scan of
all n rotations for every string over {a, b} of length 1..12 (8190 strings) and over {a, b, c} of
length 1..8 (9840 strings), and for 400 LCG strings over {a, b, c} of length 1..30, comparing both the returned index and the rotation text;
a third, independent method that takes the lexicographically smallest suffix of s + s whose start
index is below n (compared on rotation text, since it may return a different tie index); the
rotation is a genuine rotation (the multiset of characters is preserved) and its index is in
range; the count of start indices achieving the minimum equals n divided by the length of the
primitive root for constructed Lyndon powers w^k, where w is verified to be a Lyndon word by
brute force (w smaller than all its proper suffixes) and Booth must return 0; and the degenerate
empty / single-character strings.
"""


def booth(text):
    """Least rotation start index (smallest index among the minimal rotations)."""
    n = len(text)
    if n == 0:
        return 0
    doubled = text + text
    i, j, k = 0, 1, 0
    while i < n and j < n and k < n:
        left = doubled[i + k]
        right = doubled[j + k]
        if left == right:
            k += 1
        elif left > right:
            i = i + k + 1
            if i <= j:
                i = j + 1
            k = 0
        else:
            j = j + k + 1
            if j <= i:
                j = i + 1
            k = 0
    start = i if i < j else j
    return start


def minimal_rotation(text):
    return text[booth(text):] + text[:booth(text)]


def brute_force_rotation(text):
    """All rotations scanned directly; returns (smallest index, minimal rotation text)."""
    if not text:
        return 0, ""
    n = len(text)
    rotations = [(text[i:] + text[:i], i) for i in range(n)]
    best = min(rotation for rotation, _ in rotations)
    return min(index for rotation, index in rotations if rotation == best), best


def suffix_minimum_rotation(text):
    """Independent method: the smallest suffix of s + s starting below n gives a minimal rotation."""
    if not text:
        return 0, ""
    n = len(text)
    doubled = text + text
    start = min(range(n), key=lambda i: doubled[i:])
    return start, doubled[start:start + n]


def lcg_text(length, seed, alphabet):
    out = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        out.append("abc"[seed % alphabet])
    return "".join(out)


def is_lyndon(word):
    """A Lyndon word is strictly smaller than every proper non-empty suffix of itself."""
    return bool(word) and all(word < word[i:] for i in range(1, len(word)))


def minimal_rotation_starts(text):
    """Every start index whose rotation equals the minimal rotation."""
    minimal = minimal_rotation(text)
    return [i for i in range(len(text)) if text[i:] + text[:i] == minimal]


def main():
    # Hand-verified examples: 'baba' -> "abab" at index 1; 'banana' -> "anaban" at index 5.
    assert minimal_rotation("baba") == "abab" and booth("baba") == 1
    assert minimal_rotation("banana") == "abanan" and booth("banana") == 5
    assert minimal_rotation("mississippi") == "imississipp" and booth("mississippi") == 10
    assert minimal_rotation("cba") == "acb" and booth("cba") == 2
    assert minimal_rotation("abacaba") == "aabacab" and booth("abacaba") == 6
    assert booth("abcabcabc") == 0 and minimal_rotation("abcabcabc") == "abcabcabc"
    assert booth("aaa") == 0 and booth("abab") == 0 and booth("ab") == 0 and booth("ba") == 1
    assert booth("") == 0 and minimal_rotation("") == ""
    assert booth("z") == 0 and minimal_rotation("z") == "z"
    # sorted("banana") is preserved by every rotation: the result is a true rotation.
    for text in ("baba", "banana", "cba", "abacaba", "mississippi"):
        rotation = minimal_rotation(text)
        assert sorted(rotation) == sorted(text)
        assert 0 <= booth(text) < len(text)
        assert rotation == brute_force_rotation(text)[1]

    # Exhaustive check over the whole {a, b} space up to length 11, plus LCG strings over {a,b,c}.
    total_strings = 0
    for alphabet in ("ab", "abc"):
        limit = 13 if alphabet == "ab" else 9
        for length in range(1, limit):
            for code in range(len(alphabet) ** length):
                value = code
                chars = []
                for _ in range(length):
                    chars.append(alphabet[value % len(alphabet)])
                    value //= len(alphabet)
                text = "".join(chars)
                index = booth(text)
                expected_index, expected_rotation = brute_force_rotation(text)
                assert index == expected_index, (text, index, expected_index)
                assert minimal_rotation(text) == expected_rotation, text
                assert suffix_minimum_rotation(text)[1] == expected_rotation, text
                total_strings += 1
    assert total_strings == (2 ** 13 - 2) + (3 ** 9 - 3) // 2, total_strings
    for seed in range(1, 401):
        text = lcg_text(1 + seed % 30, seed, 3)
        index = booth(text)
        expected_index, expected_rotation = brute_force_rotation(text)
        assert index == expected_index, (seed, text, index, expected_index)
        assert minimal_rotation(text) == expected_rotation
        assert suffix_minimum_rotation(text)[1] == expected_rotation
        assert sorted(minimal_rotation(text)) == sorted(text)

    # Lyndon powers: if w is a Lyndon word then w^k is its own minimal rotation and the minimal
    # rotation is achieved exactly at the multiples of |w|.
    lyndon_words = ["a", "ab", "aab", "abac", "abc", "aabab", "aabac", "abb"]
    for word in lyndon_words:
        assert is_lyndon(word), word
        for power in range(1, 5):
            text = word * power
            assert booth(text) == 0, (text, booth(text))
            assert minimal_rotation(text) == text
            starts = minimal_rotation_starts(text)
            assert starts == [power_index * len(word) for power_index in range(power)], \
                (text, starts)

    # Non-Lyndon controls: a minimal rotation that is not the string itself, and the tie set.
    for text in ("ba", "baba", "cba", "banana", "zyx"):
        assert not is_lyndon(text) or booth(text) == 0
        starts = minimal_rotation_starts(text)
        assert starts, text
        assert starts[0] == booth(text), (text, starts)
    assert minimal_rotation_starts("abab") == [0, 2], "period-2 tie at both starts"
    assert minimal_rotation_starts("aba") == [2], "'aab' is unique"
    assert minimal_rotation_starts("aab") == [0]
    assert minimal_rotation_starts("banana") == [5]

    # Long deterministic input: linear-time behaviour on 20000 characters and a known answer.
    text = lcg_text(20000, 12345, 2)
    index = booth(text)
    assert 0 <= index < 20000
    assert minimal_rotation(text) == suffix_minimum_rotation(text)[1]
    assert booth("b" * 20000) == 0, "all 'b's: every rotation is equal, smallest index wins"
    assert booth("a" * 20000) == 0
    assert booth("ba" * 10000) == 1, "'bababa...' -> smallest rotation is at index 1"
    assert booth("ab" * 10000) == 0

    # Determinism.
    assert booth(text) == booth(text)
    assert minimal_rotation(text) == minimal_rotation(text)


if __name__ == "__main__":
    main()
    print("PASS")
