"""Task 032 - Check prime number naive. (Tier: Beginner, CTCI Ch 6)"""

def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def main():
    assert not is_prime(0) and not is_prime(1)
    assert is_prime(2) and is_prime(97)
    assert not is_prime(100)


if __name__ == "__main__":
    main()
    print("PASS")
