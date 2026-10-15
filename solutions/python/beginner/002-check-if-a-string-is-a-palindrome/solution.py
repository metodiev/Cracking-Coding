"""Task 002 - Check if a string is a palindrome.  (Tier: Beginner, CTCI Ch 1)"""


def is_palindrome(s):
    i, j = 0, len(s) - 1
    while i < j:
        if s[i] != s[j]:
            return False
        i, j = i + 1, j - 1
    return True


def main():
    assert is_palindrome("racecar")
    assert is_palindrome("")
    assert is_palindrome("a")
    assert not is_palindrome("hello")
    print("PASS")


if __name__ == "__main__":
    main()
