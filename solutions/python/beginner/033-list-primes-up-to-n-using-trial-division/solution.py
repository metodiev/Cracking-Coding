"""Task 033 - List primes up to N using trial division. (Tier: Beginner, CTCI Ch 6)"""

def primes_upto(n):
    def prime(x):
        if x < 2:
            return False
        i = 2
        while i * i <= x:
            if x % i == 0:
                return False
            i += 1
        return True

    return [x for x in range(2, n + 1) if prime(x)]


def main():
    assert primes_upto(1) == []
    assert primes_upto(10) == [2, 3, 5, 7]
    assert primes_upto(20) == [2, 3, 5, 7, 11, 13, 17, 19]


if __name__ == "__main__":
    main()
    print("PASS")
