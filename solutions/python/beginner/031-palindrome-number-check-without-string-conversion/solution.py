"""Task 031 - Palindrome number check (without string conversion). (Tier: Beginner, CTCI Ch 6)"""

def is_palindrome_number(n):
    if n < 0:
        return False
    original, reversed_half = n, 0
    while n:
        reversed_half = reversed_half * 10 + n % 10
        n //= 10
    return original == reversed_half


def main():
    assert is_palindrome_number(121)
    assert not is_palindrome_number(-121)
    assert is_palindrome_number(0)
    assert not is_palindrome_number(123)


if __name__ == "__main__":
    main()
    print("PASS")
