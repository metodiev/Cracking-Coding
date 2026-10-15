"""Task 018 - Convert integer to string (manual itoa).  (Tier: Beginner, CTCI Ch 1)"""


def itoa(n):
    if n == 0:
        return "0"
    negative = n < 0
    n = abs(n)
    digits = []
    while n > 0:
        digits.append(chr(ord("0") + n % 10))
        n //= 10
    digits.reverse()
    return ("-" if negative else "") + "".join(digits)


def main():
    assert itoa(0) == "0"
    assert itoa(123) == "123"
    assert itoa(-456) == "-456"
    assert itoa(2147483647) == "2147483647"
    print("PASS")


if __name__ == "__main__":
    main()
