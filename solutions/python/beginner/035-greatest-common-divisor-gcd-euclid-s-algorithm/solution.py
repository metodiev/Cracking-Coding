"""Task 035 - Greatest common divisor (GCD) Euclid's algorithm. (Tier: Beginner, CTCI Ch 6)"""

def gcd(a, b):
    a, b = abs(a), abs(b)
    while b:
        a, b = b, a % b
    return a


def main():
    assert gcd(48, 18) == 6
    assert gcd(17, 5) == 1
    assert gcd(0, 9) == 9
    assert gcd(-24, 36) == 12


if __name__ == "__main__":
    main()
    print("PASS")
