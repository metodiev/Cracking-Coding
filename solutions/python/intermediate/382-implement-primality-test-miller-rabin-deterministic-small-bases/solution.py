"""Task 382 - Miller-Rabin primality test with deterministic small bases. (Tier: Intermediate, CTCI Ch 6)

Strong-probable-prime test: write n-1 = d * 2^s with d odd. A base a is a witness of
compositeness unless a^d = 1 (mod n) or a^(d*2^r) = -1 (mod n) for some 0 <= r < s.
Testing all twelve bases {2,3,5,7,11,13,17,19,23,29,31,37} is deterministic for every
n < 3.3e24 (Sorenson & Webster); a single base is not enough: 2047 = 23*89 is the smallest
strong pseudoprime to base 2, and 3215031751 = 151*751*28351 passes bases 2, 3, 5 and 7.
"""

DETERMINISTIC_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_strong_probable_prime(n, base):
    """One-base strong test: True when the base fails to prove n composite."""
    if n < 2:
        return False
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    x = pow(base % n, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False


def is_prime(n):
    """Deterministic Miller-Rabin for n < 3.3e24 using the twelve fixed bases."""
    if n < 2:
        return False
    for p in DETERMINISTIC_BASES:
        if n % p == 0:
            return n == p
    return all(is_strong_probable_prime(n, a) for a in DETERMINISTIC_BASES)


def is_prime_trial_division(n):
    """Reference oracle for the cross-check; O(sqrt(n))."""
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def main():
    # small primes: the base set itself must be recognised
    assert all(is_prime(p) for p in DETERMINISTIC_BASES)

    # known primes (including the Mersenne primes 2^31-1 and 2^61-1)
    known_primes = [41, 97, 541, 7919, 104729, 1299709, 2147483647, 2305843009213693951,
                    67280421310721]
    assert all(is_prime(p) for p in known_primes), "known primes"

    # known composites incl. Fermat pseudoprime 341 = 11*31 and Carmichael numbers 561, 1105, 1729
    known_composites = [0, 1, 4, 9, 15, 21, 25, 49, 91, 121, 341, 561, 1105, 1729,
                        4294967297]
    assert all(not is_prime(c) for c in known_composites), "known composites"
    assert 11 * 31 == 341 and 3 * 11 * 17 == 561 and 5 * 13 * 17 == 1105 and 7 * 13 * 19 == 1729
    assert 641 * 6700417 == 4294967297

    # 2047 = 23*89: the base-2 test alone would call it prime; extra bases expose it
    assert 23 * 89 == 2047
    assert is_strong_probable_prime(2047, 2)
    assert not is_strong_probable_prime(2047, 3)
    assert not is_prime(2047)
    assert not is_prime_trial_division(2047)

    # 3215031751 = 151*751*28351 passes the first four bases but not the whole set
    assert 151 * 751 * 28351 == 3215031751
    assert all(is_strong_probable_prime(3215031751, b) for b in (2, 3, 5, 7))
    assert not is_strong_probable_prime(3215031751, 11)
    assert not is_prime(3215031751)

    # cross-check against trial division for every n <= 10000
    for n in range(10001):
        assert is_prime(n) == is_prime_trial_division(n), n
    assert sum(1 for n in range(10001) if is_prime(n)) == 1229  # pi(10000) = 1229


if __name__ == "__main__":
    main()
    print("PASS")
