"""Task 043 - Count words in a string. (Tier: Beginner, CTCI Ch 1)"""

def count_words(s):
    return len(s.split())


def main():
    assert count_words("the sky is blue") == 4
    assert count_words("   ") == 0
    assert count_words("") == 0
    assert count_words("one") == 1


if __name__ == "__main__":
    main()
    print("PASS")
