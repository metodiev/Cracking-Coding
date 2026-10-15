"""Task 001 - Reverse a string.  (Tier: Beginner, CTCI Ch 1)"""


def reverse_string(s):
    return s[::-1]


def main():
    assert reverse_string("hello") == "olleh"
    assert reverse_string("") == ""
    assert reverse_string("a") == "a"
    assert reverse_string("ab") == "ba"
    print("PASS")


if __name__ == "__main__":
    main()
