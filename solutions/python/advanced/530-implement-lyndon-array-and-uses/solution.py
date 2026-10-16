"""Task 530 - Implement Lyndon array and uses. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

The Lyndon array Lyn[i] = the length of the longest Lyndon word starting at position i, computed
two independent ways: the definition itself (scan L upwards while s[i:i+L] is strictly smaller
than all of its proper suffixes, the Lyndon test) and the Duval factorizer applied to each suffix
(the first factor of the Chen-Fox-Lyndon factorization of s[i:] is precisely the longest Lyndon
prefix of that suffix). Documented uses, each asserted: jumping from 0 by Lyn[0], then by
Lyn[Lyn[0]], ... reproduces Duval's factorization of the whole string, and doing the same from any
i gives the factorization of s[i:]; max Lyn is the length of the longest Lyndon substring; Lyn[i]
== n - i exactly when the suffix s[i:] is itself a Lyndon word (and Lyn[0] == n exactly when s is
Lyndon); and the classic rank certificate - every suffix starting strictly inside the longest
Lyndon word at i is lexicographically *larger* than the suffix at i, so for i < j < i + Lyn[i]
the suffix ranks satisfy rank(i) < rank(j). Documented checks: hand-verified arrays for "abc"
([3,2,1]), "cba" ([1,1,1]), "aab" ([3,2,1]), "aba" ([2,1,1]) and "banana"
([1,2,1,2,1,1], sum 8); exhaustive agreement of the two computations plus all the structural
invariants for every binary string up to length 11 and every ternary string up to length 7;
the same invariants on LCG strings up to length 400 (where the O(n^3) definition is replaced by
the Duval computation); and the boundary cases of the empty string, a single character and a
fully descending string.
"""


def is_lyndon(word):
    """A Lyndon word is non-empty and strictly smaller than every proper non-empty suffix."""
    if not word:
        return False
    return all(word < word[i:] for i in range(1, len(word)))


def lyndon_array_by_definition(text):
    """Lyn[i] = largest L with s[i:i+L] Lyndon, found by scanning L upwards."""
    n = len(text)
    array = []
    for i in range(n):
        longest = 0
        for length in range(1, n - i + 1):
            if is_lyndon(text[i:i + length]):
                longest = length
        array.append(longest)
    return array


def duval(text):
    """Chen-Fox-Lyndon factorization; the first factor is the longest Lyndon prefix."""
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


def lyndon_array_by_duval(text):
    """Lyn[i] = length of the first Duval factor of the suffix s[i:]."""
    return [len(duval(text[i:])[0]) for i in range(len(text))]


def suffix_ranks(text):
    """Lexicographic ranks of the suffixes (rank 0 = smallest suffix)."""
    order = sorted(range(len(text)), key=lambda i: text[i:])
    ranks = [0] * len(text)
    for position, index in enumerate(order):
        ranks[index] = position
    return ranks


def factorization_from_array(text, array):
    """Jump chain i -> i + Lyn[i]: the Lyndon factorization of the whole string."""
    factors = []
    i = 0
    while i < len(text):
        factors.append(text[i:i + array[i]])
        i += array[i]
    return factors


def factorization_of_suffix_from_array(text, array, start):
    factors = []
    i = start
    while i < len(text):
        factors.append(text[i:i + array[i]])
        i += array[i]
    return factors


def lcg_text(length, seed, alphabet):
    out = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        out.append("abc"[seed % alphabet])
    return "".join(out)


def main():
    # Hand-verified arrays.
    assert lyndon_array_by_definition("abc") == [3, 2, 1]
    assert lyndon_array_by_definition("cba") == [1, 1, 1]
    assert lyndon_array_by_definition("aab") == [3, 2, 1]
    assert lyndon_array_by_definition("aba") == [2, 1, 1]
    assert lyndon_array_by_definition("banana") == [1, 2, 1, 2, 1, 1]
    assert sum(lyndon_array_by_definition("banana")) == 8
    assert lyndon_array_by_definition("a") == [1]
    assert lyndon_array_by_definition("zz") == [1, 1]
    # Use 1: the jump chain from 0 reproduces Duval's factorization ("b|an|an|a").
    array = lyndon_array_by_definition("banana")
    assert factorization_from_array("banana", array) == ["b", "an", "an", "a"]
    assert factorization_from_array("banana", array) == duval("banana")
    # Use 2: max Lyn is the longest Lyndon substring ("banana" has "an" as the longest Lyndon word).
    assert max(array) == 2
    assert max(lyndon_array_by_definition("abc")) == 3
    # Use 3: Lyn[i] == n - i exactly for the suffixes that are Lyndon words.
    assert array[5] == 1 and is_lyndon("banana"[5:])
    assert array[1] == 2 and is_lyndon("banana"[1:3])
    assert array[1] != 4 and not is_lyndon("banana"[1:5]), "'anan' is not a Lyndon word"
    assert not is_lyndon("banana")

    # Exhaustive verification: the two computations agree and every documented use holds.
    checked = 0
    for alphabet, limit in (("ab", 12), ("abc", 8)):
        for length in range(1, limit):
            for code in range(len(alphabet) ** length):
                value = code
                chars = []
                for _ in range(length):
                    chars.append(alphabet[value % len(alphabet)])
                    value //= len(alphabet)
                text = "".join(chars)
                by_definition = lyndon_array_by_definition(text)
                by_duval = lyndon_array_by_duval(text)
                assert by_definition == by_duval, (text, by_definition, by_duval)
                assert all(value >= 1 for value in by_duval)
                assert all(is_lyndon(text[i:i + by_duval[i]]) for i in range(length))
                assert factorization_from_array(text, by_duval) == duval(text), text
                for start in range(length):
                    assert factorization_of_suffix_from_array(text, by_duval, start) \
                        == duval(text[start:]), (text, start)
                assert (by_duval[0] == length) == is_lyndon(text), text
                for i in range(length):
                    assert (by_duval[i] == length - i) == is_lyndon(text[i:]), (text, i)
                ranks = suffix_ranks(text)
                for i in range(length):
                    for j in range(i + 1, i + by_duval[i]):
                        assert ranks[i] < ranks[j], (text, i, j)
                checked += 1
    assert checked == (2 ** 12 - 2) + (3 ** 8 - 3) // 2, checked

    # LCG strings up to length 400: the structural invariants stay valid at scale.
    for length, alphabet, seed in ((1, 2, 3), (17, 2, 11), (60, 3, 23), (150, 2, 37), (400, 3, 53)):
        text = lcg_text(length, seed, alphabet)
        array = lyndon_array_by_duval(text)
        assert factorization_from_array(text, array) == duval(text), seed
        assert all(is_lyndon(text[i:i + array[i]]) for i in range(length)), seed
        assert all(1 <= value <= length - i for i, value in enumerate(array)), seed
        assert (array[0] == length) == is_lyndon(text), seed
        ranks = suffix_ranks(text)
        for i in range(length):
            assert (array[i] == length - i) == is_lyndon(text[i:]), (seed, i)
            for j in range(i + 1, i + array[i]):
                assert ranks[i] < ranks[j], (seed, i, j)
        # The longest Lyndon substring length is respected by every position.
        longest = max(array)
        assert any(is_lyndon(text[i:i + longest]) for i in range(length)), seed
        if longest < length:
            assert not is_lyndon(text), seed

    # Boundary cases.
    assert lyndon_array_by_definition("") == [] and lyndon_array_by_duval("") == []
    assert factorization_from_array("", []) == []
    descending = "".join(chr(ord("z") - i) for i in range(26))
    assert lyndon_array_by_definition(descending) == [1] * 26
    assert factorization_from_array(descending, [1] * 26) == list(descending)
    ascending = "".join(chr(ord("a") + i) for i in range(26))
    assert lyndon_array_by_definition(ascending) == list(range(26, 0, -1))
    assert factorization_from_array(ascending, list(range(26, 0, -1))) == [ascending]
    # Periodic strings: the array is periodic and the jump chain still yields the repeat block.
    periodic = "ababababab"
    assert lyndon_array_by_definition(periodic) == [2, 1] * 5
    assert factorization_from_array(periodic, [2, 1] * 5) == ["ab"] * 5
    assert lyndon_array_by_definition("abcabcabc") == [3, 2, 1] * 3
    assert factorization_from_array("abcabcabc", [3, 2, 1] * 3) == ["abc"] * 3
    assert lyndon_array_by_definition("aabbaabb") == [4, 3, 1, 1] * 2
    assert factorization_from_array("aabbaabb", [4, 3, 1, 1] * 2) == ["aabb", "aabb"]

    # Determinism.
    text = lcg_text(120, 97, 2)
    assert lyndon_array_by_duval(text) == lyndon_array_by_duval(text)
    assert suffix_ranks(text) == suffix_ranks(text)


if __name__ == "__main__":
    main()
    print("PASS")
