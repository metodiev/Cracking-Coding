"""Task 529 - Implement advanced string factoring (Lyndon factorization Duval). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Duval's algorithm for the Chen-Fox-Lyndon factorization of a string into non-increasing Lyndon
words, in linear time and O(1) extra space: the scan keeps the current pre-Lyndon block
s[i..j-1] with period j - k, comparing s[k] against s[j]; if s[k] < s[j] the block repeats
(k = i), if s[k] == s[j] the period continues (k += 1), and if s[k] > s[j] the block cannot be
extended, so s[i .. i + (j - k) - 1] is emitted as a factor and the scan restarts at i + (j - k).
Documented checks: hand-verified factorizations ("abab" -> ab|ab, "banana" -> b|an|an|a,
"dcba" -> d|c|b|a, "abcd" -> abcd, "aaaaa" -> a|a|a|a|a, "" -> no factors); exhaustive agreement
with an independent greedy factorizer (repeatedly take the *longest* Lyndon prefix, verifying
Lyndon-ness by the definition "strictly smaller than every proper suffix") for all 2^1 + ... +
2^10 strings over {a, b} and all strings over {a, b, c} up to length 8; structural invariants for
LCG strings up to length 2000 - concatenation equals the input, every factor is a Lyndon word,
the factor sequence is non-increasing, the first factor is the longest Lyndon prefix of the
input, the last (smallest) factor is the Lyndon word that all factors are >= to, and the input is
a single Lyndon word exactly when exactly one factor is produced; the Lyndon-power identity
w^k -> w repeated k times for brute-force-verified Lyndon words w; and determinism.
"""


def is_lyndon(word):
    """A Lyndon word is strictly smaller than every proper non-empty suffix of itself."""
    if not word:
        return False
    return all(word < word[i:] for i in range(1, len(word)))


def lyndon_factorization(text):
    """Duval's algorithm: non-increasing Lyndon factors in linear time."""
    factors = []
    n = len(text)
    i = 0
    while i < n:
        j = i + 1
        k = i
        while j < n and text[k] <= text[j]:
            if text[k] < text[j]:
                k = i
            else:
                k += 1
            j += 1
        while i <= k:
            factors.append(text[i:i + j - k])
            i += j - k
    return factors


def greedy_factorization(text):
    """Independent method: repeatedly peel off the longest Lyndon prefix."""
    factors = []
    i = 0
    n = len(text)
    while i < n:
        end = i + 1
        for candidate in range(i + 1, n + 1):
            if is_lyndon(text[i:candidate]):
                end = candidate
        factors.append(text[i:end])
        i = end
    return factors


def lcg_text(length, seed, alphabet):
    out = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        out.append("abc"[seed % alphabet])
    return "".join(out)


def main():
    # Hand-verified factorizations.
    assert lyndon_factorization("") == []
    assert lyndon_factorization("a") == ["a"]
    assert lyndon_factorization("abab") == ["ab", "ab"]
    assert lyndon_factorization("banana") == ["b", "an", "an", "a"]
    assert lyndon_factorization("dcba") == ["d", "c", "b", "a"]
    assert lyndon_factorization("abcd") == ["abcd"]
    assert lyndon_factorization("aaaaa") == ["a"] * 5
    assert lyndon_factorization("aab") == ["aab"]
    assert lyndon_factorization("abb") == ["abb"]
    assert lyndon_factorization("abba") == ["abb", "a"]
    assert lyndon_factorization("bac") == ["b", "ac"]
    assert lyndon_factorization("zzy") == ["z", "z", "y"]

    # Exhaustive agreement with the greedy longest-Lyndon-prefix factorizer.
    checked = 0
    for alphabet, limit in (("ab", 11), ("abc", 9)):
        for length in range(1, limit):
            for code in range(len(alphabet) ** length):
                value = code
                chars = []
                for _ in range(length):
                    chars.append(alphabet[value % len(alphabet)])
                    value //= len(alphabet)
                text = "".join(chars)
                factors = lyndon_factorization(text)
                assert factors == greedy_factorization(text), (text, factors,
                                                               greedy_factorization(text))
                assert "".join(factors) == text
                assert all(is_lyndon(factor) for factor in factors), text
                assert all(factors[a] >= factors[a + 1] for a in range(len(factors) - 1)), text
                assert (len(factors) == 1) == is_lyndon(text), text
                checked += 1
    assert checked == (2 ** 11 - 2) + (3 ** 9 - 3) // 2, checked

    # LCG strings: full structural invariants, including the longest-Lyndon-prefix property.
    for length, alphabet, seed in ((1, 2, 3), (9, 2, 11), (33, 2, 23), (80, 3, 37), (150, 2, 53),
                                   (400, 3, 71), (2000, 2, 97)):
        text = lcg_text(length, seed, alphabet)
        factors = lyndon_factorization(text)
        assert factors == greedy_factorization(text), seed
        assert "".join(factors) == text
        assert all(is_lyndon(factor) for factor in factors), seed
        assert all(factors[a] >= factors[a + 1] for a in range(len(factors) - 1)), seed
        # The first factor is the longest Lyndon prefix.
        longest = 0
        for end in range(1, length + 1):
            if is_lyndon(text[:end]):
                longest = end
        assert len(factors[0]) == longest, (seed, factors[0], longest)
        # The last factor is the smallest one and no factor may be smaller than it.
        assert all(factor >= factors[-1] for factor in factors), seed
        assert factors[-1] == min(factors), seed
        # A single factor means the input is itself a Lyndon word.
        assert (len(factors) == 1) == is_lyndon(text), seed
        # The decomposition is maximal: the concatenation of the factors after the first one is
        # not extendable backwards into a bigger Lyndon first factor.
        if len(factors) > 1:
            assert not is_lyndon(factors[0] + factors[1][0]), seed

    # Lyndon powers: w^k factors into k copies of w, and the greedy method agrees.
    lyndon_words = ["a", "ab", "aab", "abac", "abc", "abb", "aabab", "aaab"]
    for word in lyndon_words:
        assert is_lyndon(word), word
        for power in range(1, 6):
            text = word * power
            assert lyndon_factorization(text) == [word] * power, (text, word, power)
            assert greedy_factorization(text) == [word] * power

    # Controls: powers of a Lyndon word are the only strings whose factors are all identical.
    for text in ("abab", "abcabcabc", "aabaab"):
        factors = lyndon_factorization(text)
        assert len(set(factors)) == 1 and factors[0] * len(factors) == text, text
    for text in ("aba", "abacaba", "banana", "mississippi"):
        factors = lyndon_factorization(text)
        assert len(set(factors)) > 1 or len(factors) == 1, text
        assert factors == greedy_factorization(text), text

    # Determinism on a long input.
    text = lcg_text(5000, 2024, 2)
    first = lyndon_factorization(text)
    second = lyndon_factorization(text)
    assert first == second
    assert "".join(first) == text
    assert all(is_lyndon(factor) for factor in first)
    assert all(first[a] >= first[a + 1] for a in range(len(first) - 1))


if __name__ == "__main__":
    main()
    print("PASS")
