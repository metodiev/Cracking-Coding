"""Task 315 - Rolling hash with two moduli to reduce collisions. (Tier: Intermediate, CTCI Ch 1)

Polynomial rolling hash h(s) = ((c_0 * B + c_1) * B + ... + c_{k-1}) mod M with prefix arrays, so
any substring hash is O(1): h[l..r) = (pref[r] - pref[l] * B^(r-l)) mod M. Equality uses TWO
moduli, MOD1 = 1_000_000_007 and MOD2 = 998_244_353, with BASE = 131: a false match now needs a
collision in both moduli at once. The tests compare the O(1) queries with naive slicing, and a
deliberately weak single-modulus hash (MOD_WEAK = 65_537 with BASE_WEAK = 65_536 == -1 mod
MOD_WEAK) is constructed to collide on "ab" and "bc" — the double hash separates the pair.
"""

MOD1 = 1_000_000_007
MOD2 = 998_244_353
BASE = 131

MOD_WEAK = 65537
BASE_WEAK = 65536                      # == -1 (mod 65537), a deliberately bad base


class DoubleRollingHash:
    def __init__(self, s):
        n = len(s)
        self.h1 = [0] * (n + 1)
        self.h2 = [0] * (n + 1)
        self.p1 = [1] * (n + 1)
        self.p2 = [1] * (n + 1)
        for i, char in enumerate(s):
            code = ord(char)
            self.h1[i + 1] = (self.h1[i] * BASE + code) % MOD1
            self.h2[i + 1] = (self.h2[i] * BASE + code) % MOD2
            self.p1[i + 1] = self.p1[i] * BASE % MOD1
            self.p2[i + 1] = self.p2[i] * BASE % MOD2

    def hash(self, lo, hi):
        """Hash of s[lo:hi] as the pair (mod MOD1, mod MOD2)."""
        a = (self.h1[hi] - self.h1[lo] * self.p1[hi - lo]) % MOD1
        b = (self.h2[hi] - self.h2[lo] * self.p2[hi - lo]) % MOD2
        return (a, b)

    def equal(self, lo1, hi1, lo2, hi2):
        return self.hash(lo1, hi1) == self.hash(lo2, hi2)


def direct_hash(s):
    """Rolling hash built from scratch, no prefix arrays; used as a cross-check."""
    h1 = h2 = 0
    for char in s:
        h1 = (h1 * BASE + ord(char)) % MOD1
        h2 = (h2 * BASE + ord(char)) % MOD2
    return (h1, h2)


def weak_hash(s):
    """Single-modulus hash with a base that is -1 modulo the modulus (alternating sum)."""
    h = 0
    for char in s:
        h = (h * BASE_WEAK + ord(char)) % MOD_WEAK
    return h


def lcg_strings(length, count, alphabet="abca", seed=3150):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(alphabet[seed % len(alphabet)])
        strings.append("".join(chars))
    return strings


def main():
    assert (MOD1 - 1) % 2 == 0 and (MOD2 - 1) % 2 == 0          # both prime-friendly for bases
    assert MOD1 != MOD2

    s = "banana"
    rolling = DoubleRollingHash(s)
    assert rolling.hash(1, 3) == direct_hash("an")
    assert rolling.hash(0, 2) == direct_hash("ba")
    assert rolling.equal(1, 3, 3, 5)                            # "an" == "an"
    assert not rolling.equal(0, 2, 3, 5)                        # "ba" != "an"
    assert rolling.equal(1, 6, 1, 6)

    # Deliberate collision of a weak single-modulus hash, separated by the double hash
    assert "ab" != "bc"
    assert weak_hash("ab") == weak_hash("bc") == 1              # 97*(-1)+98 == 98*(-1)+99
    assert weak_hash("ab") != weak_hash("ba")
    assert DoubleRollingHash("ab").hash(0, 2) != DoubleRollingHash("bc").hash(0, 2)
    assert direct_hash("ab") != direct_hash("bc")

    # O(1) equality queries must agree with naive slicing, systematically and randomly
    checked = 0
    for text in ["a", "banana", "mississippi", "aaaaaaaa"] \
            + lcg_strings(12, 20) + lcg_strings(16, 10, "abcd"):
        rolling = DoubleRollingHash(text)
        n = len(text)
        for length in range(1, n + 1):
            for lo1 in range(n - length + 1):
                for lo2 in range(lo1, n - length + 1):
                    expected = text[lo1:lo1 + length] == text[lo2:lo2 + length]
                    assert rolling.equal(lo1, lo1 + length, lo2, lo2 + length) == expected, (text, lo1, lo2, length)
                    checked += 1
        # cross-check the prefix-array hash against the naive rebuild for random substrings
        for lo in range(n):
            for hi in range(lo, n + 1):
                assert rolling.hash(lo, hi) == direct_hash(text[lo:hi]), (text, lo, hi)
    assert checked > 1000

    # The "ab"/"bc" pair also survives a single strong modulus with a good base ...
    assert direct_hash("ab")[0] != direct_hash("bc")[0]
    # ... so the weak demo really is about the base being -1, not about a single modulus per se
    assert weak_hash("ab") == weak_hash("bc")


if __name__ == "__main__":
    main()
    print("PASS")
