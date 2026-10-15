"""Task 380 - Modular inverse (extended Euclid and Fermat) + fast modular exponentiation. (Tier: Intermediate, CTCI Ch 6)

Two independent inverse implementations:
  * extended Euclid: g = gcd(a, m) and Bezout coefficients x, y with a*x + m*y = g;
    when g == 1, x mod m is the inverse.  Works for ANY modulus m;
  * Fermat: for PRIME p and a not divisible by p, inv = a^(p-2) mod p.
Both must agree on every coprime pair, and a * inv(a, m) % m == 1 on LCG values.

Documented failure: when gcd(a, m) != 1 no inverse exists; the API returns None
(JS null, Java Long null, C# long?) instead of silently returning garbage.  The
m = 1 case is documented as out of scope (every residue is 0 mod 1).

Fast modular exponentiation is binary (square-and-multiply) with the usual contract
pow_mod(a, 0, m) = 1 mod m; hand-checked: 2^10 = 1024, 3^0 = 1, cross-checked
against a naive O(e) loop for small exponents, checked against the LCG identity
pow(a, x+y) = pow(a, x) * pow(a, y) mod m, and Fermat's little theorem
a^(p-1) = 1 mod p is asserted for several primes.

No floating point is used anywhere.
"""

SENTINEL = None      # documented failure marker for non-invertible inputs


def pow_mod(base, exponent, modulus):
    """Fast binary modular exponentiation; exponents >= 0 only (documented)."""
    if exponent < 0:
        raise ValueError("negative exponents are not supported by pow_mod")
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    result = 1 % modulus
    base %= modulus
    while exponent > 0:
        if exponent & 1:
            result = result * base % modulus
        base = base * base % modulus
        exponent >>= 1
    return result


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def extended_gcd(a, b):
    """Iterative extended Euclid: returns (g, x, y) with a*x + b*y = g >= 0."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def mod_inverse_euclid(a, m):
    """Inverse of a modulo m via extended Euclid, or None when gcd(a, m) != 1."""
    if m <= 1:
        raise ValueError("m >= 2 required (m = 1 documented as out of scope)")
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        return SENTINEL
    return x % m


def mod_inverse_fermat(a, p):
    """Inverse of a modulo the PRIME p via a^(p-2) mod p; None when p divides a."""
    if p <= 1:
        raise ValueError("p > 1 required")
    a %= p
    if a == 0:
        return SENTINEL
    return pow_mod(a, p - 2, p)


def pow_naive(base, exponent, modulus):
    """Independent O(exponent) oracle for small exponents."""
    result = 1 % modulus
    for _ in range(exponent):
        result = result * (base % modulus) % modulus
    return result


def lcg_value(seed, bound):
    seed = (seed * 75 + 74) % 65537
    return seed % bound, seed


def main():
    # --- hand-checked small cases ----------------------------------------
    assert mod_inverse_euclid(3, 11) == 4              # 3*4 = 12 = 1 mod 11
    assert mod_inverse_euclid(1, 11) == 1
    assert mod_inverse_euclid(1, 7) == 1
    assert mod_inverse_euclid(1, 1000000000) == 1
    assert mod_inverse_euclid(2, 5) == 3               # 2*3 = 6 = 1 mod 5
    assert mod_inverse_euclid(10, 17) == 12            # 10*12 = 120 = 1 mod 17
    assert mod_inverse_euclid(4, 8) is None            # gcd = 4: documented failure
    assert mod_inverse_euclid(6, 9) is None            # gcd = 3
    assert mod_inverse_euclid(0, 5) is None            # gcd = 5

    assert mod_inverse_fermat(3, 11) == 4
    assert mod_inverse_fermat(5, 101) == 81            # 5*81 = 405 = 4*101 + 1
    assert mod_inverse_fermat(11, 11) is None          # p divides a

    # --- fast exponentiation, hand-checked ---------------------------------
    assert pow_mod(2, 10, 1000000007) == 1024
    assert pow_mod(3, 0, 1000000007) == 1
    assert pow_mod(0, 0, 7) == 1
    assert pow_mod(7, 1, 5) == 2
    assert pow_mod(5, 3, 13) == 125 % 13 == 8
    assert pow_mod(2, 1000, 1000000007) == 688423210    # independently recomputed
    assert pow_mod(3, 200, 1000000007) == 136318165     # independently recomputed

    # binary vs naive loop for small exponents
    seed = 38013
    for _ in range(60):
        base, seed = lcg_value(seed, 1000)
        exponent, seed = lcg_value(seed, 40)
        modulus, seed = lcg_value(seed, 999)
        modulus += 2
        assert pow_mod(base, exponent, modulus) == pow_naive(base, exponent, modulus), (base, exponent, modulus)

    # pow(a, x + y) = pow(a, x) * pow(a, y) mod m
    for _ in range(40):
        base, seed = lcg_value(seed, 1000000006)
        base += 1
        x, seed = lcg_value(seed, 60)
        y, seed = lcg_value(seed, 60)
        modulus = 1000000007
        assert pow_mod(base, x + y, modulus) == pow_mod(base, x, modulus) * pow_mod(base, y, modulus) % modulus

    # --- extended Euclid identity a*inv = 1 (mod m) on LCG values ----------
    for _ in range(60):
        m, seed = lcg_value(seed, 998)
        m += 2                                          # m in [2, 999]
        a, seed = lcg_value(seed, 1000)
        inverse = mod_inverse_euclid(a, m)
        if gcd(a, m) == 1:
            assert inverse is not None, (a, m)
            assert a * inverse % m == 1, (a, m, inverse)
        else:
            assert inverse is None, (a, m)

    # --- Euclid and Fermat agree for primes --------------------------------
    for prime in (7, 11, 13, 101, 65537, 1000000007):
        for _ in range(12):
            a, seed = lcg_value(seed, min(prime - 1, 65536))
            a += 1
            if a % prime == 0:
                continue
            euclid = mod_inverse_euclid(a, prime)
            fermat = mod_inverse_fermat(a, prime)
            assert euclid == fermat, (a, prime)
            assert a * euclid % prime == 1, (a, prime)

    # Fermat's little theorem: a^(p-1) = 1 mod p for p prime, p not dividing a
    for prime in (7, 11, 13, 101, 1000000007):
        for a in (1, 2, 3, prime - 1, prime - 2):
            if a % prime == 0:
                continue
            assert pow_mod(a, prime - 1, prime) == 1, (a, prime)

    # Wilson-flavoured sanity: (p-1)! = -1 mod p checked for a small prime
    factorial = 1
    for i in range(1, 19):
        factorial = factorial * i % 19
    assert factorial == 18


if __name__ == "__main__":
    main()
    print("PASS")
