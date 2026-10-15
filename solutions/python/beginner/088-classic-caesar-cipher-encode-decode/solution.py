"""Task 088 - Classic Caesar cipher encode/decode. (Tier: Beginner, CTCI Ch 1)

Shifts lowercase a-z with wrap-around; every shift is reduced mod 26 so large
and negative shifts work. Non-lowercase characters (including uppercase letters,
digits, spaces, and punctuation) are left unchanged, so
decode(encode(text, k), k) == text.
"""


def _shift_char(ch, shift):
    if "a" <= ch <= "z":
        return chr((ord(ch) - ord("a") + shift) % 26 + ord("a"))
    return ch


def caesar_encode(text, shift):
    normalized = shift % 26
    return "".join(_shift_char(ch, normalized) for ch in text)


def caesar_decode(text, shift):
    return caesar_encode(text, -shift)


def main():
    assert caesar_encode("abc", 3) == "def"
    assert caesar_encode("xyz", 3) == "abc"
    assert caesar_encode("hello, world!", 3) == "khoor, zruog!"
    assert caesar_encode("abc", -3) == "xyz"
    assert caesar_encode("abc", 29) == "def"
    assert caesar_encode("abc", 26) == "abc"
    assert caesar_encode("abc", -29) == "xyz"
    assert caesar_encode("", 5) == ""

    assert caesar_decode("khoor", 3) == "hello"
    assert caesar_decode("def", 3) == "abc"
    assert caesar_decode(caesar_encode("attack at dawn!", 7), 7) == "attack at dawn!"
    assert caesar_decode(caesar_encode("the quick brown fox 123", 13), 13) == "the quick brown fox 123"

    # non-letters are unchanged; uppercase is not lowercase, so it is left alone
    assert caesar_encode("a1 B2 c!", 1) == "b1 B2 d!"
    assert caesar_encode("Hello", 1) == "Hfmmp"


if __name__ == "__main__":
    main()
    print("PASS")
