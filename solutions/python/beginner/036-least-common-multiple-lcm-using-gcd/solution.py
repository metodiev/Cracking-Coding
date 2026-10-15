"""Task 036 - Least common multiple (LCM) using GCD. (Tier: Beginner, CTCI Ch 6)"""

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def lcm(a, b):
    if a == 0 or b == 0:
        return 0
    return abs(a // gcd(a, b) * b)   # divide first to limit overflow


def main():
    assert lcm(4, 6) == 12
    assert lcm(21, 6) == 42
    assert lcm(0, 5) == 0
    assert lcm(-3, 5) == 15


if __name__ == "__main__":
    main()
    print("PASS")
