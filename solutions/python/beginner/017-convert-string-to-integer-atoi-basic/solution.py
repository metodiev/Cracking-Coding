"""Task 017 - Convert string to integer (atoi) basic.  (Tier: Beginner, CTCI Ch 1)

Supports an optional leading sign and decimal digits; raises on empty/invalid input.
"""


def atoi(s):
    if not s:
        raise ValueError("empty input")
    i, sign = 0, 1
    if s[0] in "+-":
        sign = -1 if s[0] == "-" else 1
        i = 1
    if i == len(s):
        raise ValueError("no digits")
    value = 0
    while i < len(s):
        if not s[i].isdigit():
            raise ValueError(f"invalid character: {s[i]!r}")
        value = value * 10 + (ord(s[i]) - ord("0"))
        i += 1
    return sign * value


def main():
    assert atoi("42") == 42
    assert atoi("-123") == -123
    assert atoi("+7") == 7
    assert atoi("0") == 0
    print("PASS")


if __name__ == "__main__":
    main()
