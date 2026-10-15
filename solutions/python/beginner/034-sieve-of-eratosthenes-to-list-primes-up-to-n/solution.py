"""Task 034 - Sieve of Eratosthenes to list primes up to N. (Tier: Beginner, CTCI Ch 6)"""

def sieve(n):
    """O(n log log n)."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    p = 2
    while p * p <= n:
        if is_prime[p]:
            for multiple in range(p * p, n + 1, p):
                is_prime[multiple] = False
        p += 1
    return [x for x, ok in enumerate(is_prime) if ok]


def main():
    assert sieve(1) == []
    assert sieve(30) == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    assert sieve(2) == [2]


if __name__ == "__main__":
    main()
    print("PASS")
