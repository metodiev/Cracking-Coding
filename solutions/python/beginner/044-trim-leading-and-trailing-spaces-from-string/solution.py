"""Task 044 - Trim leading and trailing spaces from string. (Tier: Beginner, CTCI Ch 1)"""

def trim(s):
    lo, hi = 0, len(s)
    while lo < hi and s[lo] == " ":
        lo += 1
    while hi > lo and s[hi - 1] == " ":
        hi -= 1
    return s[lo:hi]


def main():
    assert trim("  hello  ") == "hello"
    assert trim("hello") == "hello"
    assert trim("   ") == ""
    assert trim("  a b ") == "a b"


if __name__ == "__main__":
    main()
    print("PASS")
