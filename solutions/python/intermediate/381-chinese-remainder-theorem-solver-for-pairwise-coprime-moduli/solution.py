"""Task 381 - Chinese remainder theorem solver for pairwise coprime moduli. (Tier: Intermediate, CTCI Ch 6)

Solves x = a_i (mod m_i) incrementally: merging x = a (mod m1) with x = a2 (mod m2) gives
x = a + m1*t (mod m1*m2) where t = (a2 - a) * m1^(-1) (mod m2); the inverse exists because
gcd(m1, m2) = 1 for pairwise coprime moduli. Non-coprime moduli are rejected with a documented
ValueError -- the general case would first need gcd(m1, m2) to divide (a2 - a).
"""


def extended_gcd(a, b):
    """Return (g, x, y) with a*x + b*y == g == gcd(a, b)."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


def mod_inverse(a, modulus):
    g, x, _ = extended_gcd(a % modulus, modulus)
    if g != 1:
        raise ValueError("modular inverse does not exist")
    return x % modulus


def crt_two(a1, m1, a2, m2):
    """Merge two congruences; raises ValueError when the moduli are not coprime."""
    g, _, _ = extended_gcd(m1, m2)
    if g != 1:
        raise ValueError("moduli %d and %d are not coprime" % (m1, m2))
    t = (a2 - a1) * mod_inverse(m1, m2) % m2
    modulus = m1 * m2
    return (a1 + m1 * t) % modulus, modulus


def crt(congruences):
    """Solve a system with pairwise coprime moduli.

    Returns (x, m): x the smallest non-negative solution, m the product of the moduli.
    Raises ValueError for an empty system or non-coprime moduli.
    """
    if not congruences:
        raise ValueError("empty congruence system")
    a, m = 0, 1
    for residue, modulus in congruences:
        if modulus <= 0:
            raise ValueError("modulus must be positive")
        a, m = crt_two(a, m, residue % modulus, modulus)
    return a, m


def main():
    # classic worked example: 23 = 2 (3), 23 = 3 (5), 23 = 2 (7)
    assert crt([(2, 3), (3, 5), (2, 7)]) == (23, 105)
    x, m = crt([(2, 3), (3, 5), (2, 7)])
    assert x % 3 == 2 and x % 5 == 3 and x % 7 == 2
    assert m == 3 * 5 * 7

    # a single congruence is the identity; merge order must not matter
    assert crt([(4, 9)]) == (4, 9)
    assert crt([(23, 105)]) == (23, 105)
    assert crt([(2, 7), (3, 5), (2, 3)]) == (23, 105)
    assert crt([(0, 2), (1, 3)]) == (4, 6)

    # non-coprime moduli are rejected (documented error), even when consistent
    for bad in ([(1, 4), (2, 6)], [(0, 2), (0, 4)], [(1, 9), (1, 3)]):
        try:
            crt(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for %r" % (bad,))

    # LCG-driven systems: cross-check by substitution and by brute force where feasible
    seed = 4212
    primes = [3, 5, 7, 11, 13, 17, 19, 23]
    checked = 0
    for _ in range(40):
        count = 2 + seed % 3
        chosen = []
        while len(chosen) < count:
            seed = (seed * 75 + 74) % 65537
            p = primes[seed % len(primes)]
            if p not in chosen:
                chosen.append(p)
        residues = []
        for p in chosen:
            seed = (seed * 75 + 74) % 65537
            residues.append(seed % p)
        system = list(zip(residues, chosen))
        x, m = crt(system)
        assert all(x % p == r for r, p in system), system
        product = 1
        for p in chosen:
            product *= p
        assert m == product and x < m
        if m <= 100000:
            brute = next(v for v in range(m) if all(v % p == r for r, p in system))
            assert brute == x, (system, brute, x)
            checked += 1
    assert checked >= 10


if __name__ == "__main__":
    main()
    print("PASS")
