"""Task 003 - Count vowels and consonants in a string.  (Tier: Beginner, CTCI Ch 1)"""


def count_vowels_and_consonants(s):
    vowels = consonants = 0
    for ch in s.lower():
        if not ch.isalpha():
            continue
        if ch in "aeiou":
            vowels += 1
        else:
            consonants += 1
    return vowels, consonants


def main():
    assert count_vowels_and_consonants("hello") == (2, 3)
    assert count_vowels_and_consonants("") == (0, 0)
    assert count_vowels_and_consonants("AEIOU") == (5, 0)
    assert count_vowels_and_consonants("hi! 42") == (1, 1)
    print("PASS")


if __name__ == "__main__":
    main()
