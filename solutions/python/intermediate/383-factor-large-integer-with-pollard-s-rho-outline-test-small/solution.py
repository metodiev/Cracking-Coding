"""Task 383 - Factor a large integer with Pollard's rho (outline / small tests). (Tier: Intermediate, CTCI Ch 6)

Floyd cycle-finding variant of Pollard's rho: iterate f(x) = x^2 + c (mod n) with a tortoise
(one step) and a hare (two steps), take gcd(|x - y|, n) each round, and retry with the next
c = 1, 2, ... when the gcd collapses to n (a degenerate cycle). Primality of every found
factor is certified by deterministic Miller-Rabin, so the recursion splits composites and
stops at primes. All test numbers stay below ~2^40.
"""

MILLER_RABIN_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def is_prime(n):
    """Deterministic Miller-Rabin (same twelve-base set as task 382)."""
    if n < 2:
        return False
    for p in MILLER_RABIN_BASES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in MILLER_RABIN_BASES:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def pollard_rho(n):
    """Return a non-trivial factor of a composite n, or raise when the outline gives up."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    c = 1
    while c < 100:
        x, y, d = 2, 2, 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = gcd(abs(x - y), n)
        if d != n:
            return d
        c += 1
    raise RuntimeError("Pollard's rho failed to find a factor")


def factorize(n):
    """Complete prime factorisation in ascending order ([] for 1)."""
    out = []

    def split(value):
        if value == 1:
            return
        if is_prime(value):
            out.append(value)
            return
        factor = pollard_rho(value)
        split(factor)
        split(value // factor)

    split(n)
    return sorted(out)


def main():
    # hand-checked factorisations
    assert factorize(8051) == [83, 97]                       # 83 * 97 = 8051
    assert factorize(10403) == [101, 103]                    # 101 * 103 = 10403
    assert factorize(2 ** 32 + 1) == [641, 6700417]          # Euler's factorisation of F5
    assert factorize(123456789) == [3, 3, 3607, 3803]        # 3607 * 3803 = 13717421
    assert factorize(600851475143) == [71, 839, 1471, 6857]  # Project Euler 3
    assert factorize(2 ** 40 + 1) == [257, 4278255361]
    assert 83 * 97 == 8051 and 101 * 103 == 10403
    assert 641 * 6700417 == 4294967297 == 2 ** 32 + 1
    assert 9 * 13717421 == 123456789 and 3607 * 3803 == 13717421
    assert 71 * 839 * 1471 * 6857 == 600851475143
    assert 257 * 4278255361 == 2 ** 40 + 1

    # primes and 1 are handled without recursion
    assert factorize(1) == []
    assert factorize(2) == [2]
    assert factorize(97) == [97]
    assert factorize(97 * 97) == [97, 97]

    # cross-check: product of factors == n, every factor prime, list ascending
    samples = [8051, 10403, 2 ** 32 + 1, 123456789, 600851475143, 2 ** 40 + 1, 999983,
               1, 4, 51, 65536, 1048576 + 1]
    for n in samples:
        factors = factorize(n)
        product = 1
        for f in factors:
            assert is_prime(f), (n, f)
            product *= f
        assert product == n, (n, factors)
        assert factors == sorted(factors), (n, factors)
    assert factorize(999983) == [999983]                     # prime
    assert factorize(65536) == [2] * 16
    assert factorize(1048577) == [17, 61681]                 # 2^20 + 1


if __name__ == "__main__":
    main()
    print("PASS")
