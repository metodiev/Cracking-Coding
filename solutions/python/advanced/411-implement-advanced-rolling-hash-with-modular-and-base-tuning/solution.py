"""Task 411 - Implement advanced rolling hash with modular and base tuning. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)

Polynomial hashing: hash(s) = sum(s[i] * base^(n-1-i)) mod modulus. Tuning notes:
  * double hashing uses bases 131/137 with the large primes 1_000_000_007 / 1_000_000_009;
    all cross products stay below 2^63 and the two moduli are coprime-friendly.
  * the demo pair modulus 65537 with base 2 is deliberately bad: 2^16 = -1 and 2^32 = 1
    (mod 65537), so same-length strings that only shift a character by 32 positions
    collide — that is the constructed single-hash collision asserted below.
  * base 1 degenerates the hash to the plain character sum, so anagrams collide.
  * prefix hashes make every substring fingerprint O(1); MultiHash combines several
    (base, modulus) pairs so an accidental equality needs to beat all of them at once.
"""

MOD1 = 1_000_000_007
MOD2 = 1_000_000_009
BASE1 = 131
BASE2 = 137
DEMO_MOD = 65537  # small prime paired with DEMO_BASE for the collision demo
DEMO_BASE = 2


def poly_hash(s, base, mod):
    h = 0
    for ch in s:
        h = (h * base + ord(ch)) % mod
    return h


class RollingHash:
    """One (base, modulus) scheme with prefix hashes, so any substring is O(1)."""

    def __init__(self, s, base, mod):
        self.base = base
        self.mod = mod
        self.pref = [0] * (len(s) + 1)
        for i, ch in enumerate(s):
            self.pref[i + 1] = (self.pref[i] * base + ord(ch)) % mod
        self.pow = [1] * (len(s) + 1)
        for i in range(1, len(s) + 1):
            self.pow[i] = self.pow[i - 1] * base % mod

    def get(self, l, r):
        """Fingerprint of s[l:r]."""
        return (self.pref[r] - self.pref[l] * self.pow[r - l]) % self.mod


class MultiHash:
    """Double/triple hashing: equality requires every scheme to agree."""

    def __init__(self, s, specs):
        self.hashes = [RollingHash(s, base, mod) for base, mod in specs]

    def fingerprint(self, l, r):
        return tuple(h.get(l, r) for h in self.hashes)

    def equal(self, l1, r1, l2, r2):
        if r1 - l1 != r2 - l2:
            return False
        return self.fingerprint(l1, r1) == self.fingerprint(l2, r2)


def lcg_string(length, seed, alphabet):
    out = []
    for _ in range(length):
        seed = (seed * 75 + 74) % 65537
        out.append(alphabet[seed % len(alphabet)])
    return "".join(out), seed


def main():
    m1 = MOD1
    # hand values: "ab" = 97*131 + 98, "abc" = 97*131^2 + 98*131 + 99
    assert poly_hash("ab", BASE1, m1) == 12805
    assert poly_hash("abc", BASE1, m1) == 1677554

    h = RollingHash("abc", BASE1, m1)
    assert h.get(0, 3) == 1677554
    assert h.get(0, 1) == 97
    assert h.get(1, 2) == 98
    assert h.get(2, 3) == 99
    assert h.get(1, 3) == poly_hash("bc", BASE1, m1)
    assert h.get(0, 0) == 0 and h.get(3, 3) == 0  # empty substring

    # constructed single-hash collision: base 2, modulus 65537, same length 33,
    # characters 32 positions apart swap places — weights repeat because 2^32 = 1.
    s = "a" + "c" * 31 + "b"
    t = "b" + "c" * 31 + "a"
    assert s != t and len(s) == len(t) == 33
    assert poly_hash(s, DEMO_BASE, DEMO_MOD) == poly_hash(t, DEMO_BASE, DEMO_MOD)
    assert RollingHash(s, DEMO_BASE, DEMO_MOD).get(0, 33) == RollingHash(t, DEMO_BASE, DEMO_MOD).get(0, 33)

    # double hashing with tuned large primes separates them again
    dbl_s = MultiHash(s, [(BASE1, MOD1), (BASE2, MOD2)])
    dbl_t = MultiHash(t, [(BASE1, MOD1), (BASE2, MOD2)])
    assert dbl_s.fingerprint(0, 33) != dbl_t.fingerprint(0, 33)
    assert not (dbl_s.fingerprint(0, 33) == dbl_t.fingerprint(0, 33))

    # base 1 is a degenerate tunings choice: the hash becomes the character sum
    assert poly_hash("ab", 1, m1) == poly_hash("ba", 1, m1)
    assert poly_hash("ab", BASE1, m1) != poly_hash("ba", BASE1, m1)

    # trimming the modulus: with mod = 256 only the low byte survives
    assert poly_hash("A", 131, 256) == ord("A") % 256

    specs = [(BASE1, MOD1), (BASE2, MOD2)]
    seed = 2024
    for _ in range(6):
        text, seed = lcg_string(61, seed, "abcde")
        multi = MultiHash(text, specs)
        for i in range(len(text)):
            for j in range(i, len(text)):
                assert multi.equal(i, j, i, j)
        for _ in range(60):
            seed = (seed * 75 + 74) % 65537
            l1 = seed % len(text)
            seed = (seed * 75 + 74) % 65537
            l2 = seed % len(text)
            seed = (seed * 75 + 74) % 65537
            span = seed % (min(len(text) - l1, len(text) - l2) + 1)
            oracle = text[l1:l1 + span] == text[l2:l2 + span]
            assert multi.equal(l1, l1 + span, l2, l2 + span) == oracle, (l1, l2, span)

    # every substring fingerprint equals the direct polynomial hash
    for (base, mod) in specs:
        rh = RollingHash("mississippi", base, mod)
        for l in range(len("mississippi") + 1):
            for r in range(l, len("mississippi") + 1):
                assert rh.get(l, r) == poly_hash("mississippi"[l:r], base, mod)


if __name__ == "__main__":
    main()
    print("PASS")
