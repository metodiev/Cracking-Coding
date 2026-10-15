"""Task 186 - Rabin-Karp rolling hash string search. (Tier: Beginner, CTCI Beyond — Advanced Data Structures & Algorithms)

Polynomial window hash h = c0 * base^(m-1) + ... + c(m-1) (mod mod). Sliding one
step drops the leaving character's weighted term and appends the new one, so each
slide is O(1). Equal hashes can also mean a collision, so every hash hit is
confirmed with a direct substring comparison: a false positive never becomes a
reported match. Expected O(n + m) time.
"""

DEFAULT_BASE = 256
DEFAULT_MOD = 1_000_000_007


def pow_mod(base, exponent, mod):
    result = 1
    for _ in range(exponent):
        result = result * base % mod
    return result


def rabin_karp(text, pattern, base=DEFAULT_BASE, mod=DEFAULT_MOD):
    """All start indices of `pattern` in `text`; an empty pattern returns []."""
    if pattern == "":
        return []
    m, n = len(pattern), len(text)
    if m > n:
        return []

    highest = pow_mod(base, m - 1, mod)          # weight of a window's first character
    pattern_hash = 0
    window_hash = 0
    for i in range(m):
        pattern_hash = (pattern_hash * base + ord(pattern[i])) % mod
        window_hash = (window_hash * base + ord(text[i])) % mod

    matches = []
    if window_hash == pattern_hash and text[:m] == pattern:
        matches.append(0)
    for start in range(1, n - m + 1):
        window_hash = (window_hash - ord(text[start - 1]) * highest) % mod
        window_hash = (window_hash * base + ord(text[start + m - 1])) % mod
        if window_hash == pattern_hash and text[start:start + m] == pattern:
            matches.append(start)
    return matches


def rabin_karp_naive(text, pattern):
    """Reference scan used by the self-check."""
    if pattern == "":
        return []
    return [i for i in range(len(text) - len(pattern) + 1) if text.startswith(pattern, i)]


def main():
    assert rabin_karp("abracadabra", "abra") == [0, 7]
    assert rabin_karp("aaaaa", "aa") == [0, 1, 2, 3]
    assert rabin_karp("abcabcabc", "abc") == [0, 3, 6]
    assert rabin_karp("abcd", "bcd") == [1]
    assert rabin_karp("aaaa", "b") == []
    assert rabin_karp("ab", "abc") == []
    assert rabin_karp("", "a") == []
    assert rabin_karp("", "") == []
    assert rabin_karp("anything", "") == []
    assert rabin_karp("a", "a") == [0]

    # repetition-heavy text
    heavy = "a" * 1000 + "b"
    assert rabin_karp(heavy, "a" * 20) == list(range(981))
    assert rabin_karp(heavy, "ab") == [999]
    assert rabin_karp("ab" * 300, "abab") == list(range(0, 597, 2))

    # agreement with the naive scan on pseudo-random strings
    import random
    rng = random.Random(17)
    for _ in range(200):
        text = "".join(rng.choice("ab") for _ in range(rng.randint(0, 40)))
        pattern = "".join(rng.choice("ab") for _ in range(rng.randint(1, 5)))
        assert rabin_karp(text, pattern) == rabin_karp_naive(text, pattern), (text, pattern)

    # a tiny modulus forces collisions; the substring verification must filter them out
    for _ in range(200):
        text = "".join(rng.choice("abc") for _ in range(rng.randint(0, 40)))
        pattern = "".join(rng.choice("abc") for _ in range(rng.randint(1, 4)))
        result = rabin_karp(text, pattern, base=256, mod=101)
        assert result == rabin_karp_naive(text, pattern), (text, pattern)
    colliding_text = "".join(chr(ord("a") + rng.randrange(6)) for _ in range(60))
    seen = {}
    collided = False
    for start in range(len(colliding_text) - 2):
        window = colliding_text[start:start + 3]
        h = 0
        for ch in window:
            h = (h * 256 + ord(ch)) % 101
        if h in seen and seen[h] != window:
            collided = True
        seen.setdefault(h, window)
    assert collided, "expected the tiny modulus to produce a real hash collision"


if __name__ == "__main__":
    main()
    print("PASS")
