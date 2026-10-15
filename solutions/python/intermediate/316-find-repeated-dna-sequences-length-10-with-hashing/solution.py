"""Task 316 - Repeated DNA sequences of length 10 (hashing). (Tier: Intermediate, CTCI Ch 1)

Slide a rolling key over the string: each base is packed into 2 bits (A=0, C=1, G=2, T=3), so a
10-letter window has a collision-free key in [0, 4^10) that is updated in O(1) per step. Report
every 10-letter substring whose key repeats, sorted. Cross-checked against a Counter-style naive
implementation on deterministic LCG DNA strings.
"""

from collections import Counter

DNA_LENGTH = 10
CODE = {"A": 0, "C": 1, "G": 2, "T": 3}


def repeated_dna_sequences(s, k=DNA_LENGTH):
    """All k-letter substrings appearing more than once, sorted; [] if none."""
    if k <= 0 or len(s) < k:
        return []
    mask = (1 << (2 * k)) - 1                # keeps only the last k bases
    seen = set()
    repeated = set()
    key = 0
    for i, char in enumerate(s):
        key = ((key << 2) | CODE[char]) & mask
        if i >= k - 1:
            start = i - k + 1
            word = s[start:start + k]
            if key in seen:
                repeated.add(word)
            else:
                seen.add(key)
    return sorted(repeated)


def repeated_dna_naive(s, k=DNA_LENGTH):
    """Counter of every substring; the independent oracle."""
    if k <= 0 or len(s) < k:
        return []
    counts = Counter(s[i:i + k] for i in range(len(s) - k + 1))
    return sorted(word for word, count in counts.items() if count > 1)


def overlapping_count(s, word):
    """Occurrences counted with overlaps (str.count only counts non-overlapping ones)."""
    return sum(1 for i in range(len(s) - len(word) + 1) if s[i:i + len(word)] == word)


def lcg_dna(length, count, seed=3160):
    strings = []
    alphabet = "ACGT"
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(alphabet[seed % 4])
        strings.append("".join(chars))
    return strings


def main():
    classic = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
    assert repeated_dna_sequences(classic) == ["AAAAACCCCC", "CCCCCAAAAA"]
    assert repeated_dna_naive(classic) == ["AAAAACCCCC", "CCCCCAAAAA"]

    assert repeated_dna_sequences("ACGTACGTAC") == []
    assert repeated_dna_sequences("A" * 11) == ["AAAAAAAAAA"]
    assert repeated_dna_sequences("A" * 10) == []
    assert repeated_dna_sequences("ACGT") == []
    assert repeated_dna_sequences("") == []
    assert repeated_dna_sequences("ACGTACGTAC", k=4) == ["ACGT", "CGTA", "GTAC"]
    assert repeated_dna_sequences("ACGTACGTAC", k=0) == []
    assert repeated_dna_sequences("ACGTACGTAC", k=20) == []

    for s in [classic, "A" * 12, "ACGT" * 5] + lcg_dna(40, 40) + lcg_dna(20, 20):
        found = repeated_dna_sequences(s)
        assert found == repeated_dna_naive(s), s
        assert found == sorted(set(found))                    # sorted, no duplicates
        for word in found:                                    # every report really repeats
            assert len(word) == DNA_LENGTH and overlapping_count(s, word) > 1, (s, word)

    # The result set is exactly the set of 10-mers that appear at least twice
    for s in lcg_dna(30, 20):
        from_keys = repeated_dna_sequences(s)
        brute = sorted(w for w in {s[i:i + DNA_LENGTH] for i in range(len(s) - DNA_LENGTH + 1)}
                       if sum(1 for i in range(len(s) - DNA_LENGTH + 1) if s[i:i + DNA_LENGTH] == w) > 1)
        assert from_keys == brute, s


if __name__ == "__main__":
    main()
    print("PASS")
