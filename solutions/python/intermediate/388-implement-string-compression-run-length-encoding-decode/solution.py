"""Task 388 - Implement string compression (run-length encoding + decode). (Tier: Intermediate, CTCI Ch 1)

Convention (identical in all four languages): a maximal run of one character is written as
the bare character when its length is 1, and as the character followed by its decimal count
when the length is >= 2 -- "aaabbc" encodes to "a3b2c" and "a" stays "a". Decoding reads a
letter, then the optional digits that follow it. Inputs are restricted to letters (a digit
used as data would be ambiguous), and decoding rejects non-canonical counts 0 and 1.
"""


def is_letter(ch):
    return "a" <= ch <= "z" or "A" <= ch <= "Z"


def is_digit(ch):
    return "0" <= ch <= "9"


def rle_encode(text):
    out = []
    i = 0
    while i < len(text):
        j = i
        while j < len(text) and text[j] == text[i]:
            j += 1
        run = j - i
        out.append(text[i] if run == 1 else text[i] + str(run))
        i = j
    return "".join(out)


def rle_decode(encoded):
    out = []
    i = 0
    while i < len(encoded):
        ch = encoded[i]
        if not is_letter(ch):
            raise ValueError("expected a letter at position %d" % i)
        i += 1
        digits = []
        while i < len(encoded) and is_digit(encoded[i]):
            digits.append(encoded[i])
            i += 1
        if not digits:
            out.append(ch)
            continue
        count = int("".join(digits))
        if count < 2:
            raise ValueError("count %d is not canonical" % count)
        out.append(ch * count)
    return "".join(out)


def run_characters(text):
    """First character of every maximal run, in order (the letters of the encoding)."""
    return "".join(text[i] for i in range(len(text)) if i == 0 or text[i] != text[i - 1])


def main():
    assert rle_encode("aaabbc") == "a3b2c"
    assert rle_decode("a3b2c") == "aaabbc"

    # empty string, single characters, no runs
    assert rle_encode("") == "" and rle_decode("") == ""
    assert rle_encode("a") == "a" and rle_decode("a") == "a"
    assert rle_encode("abcd") == "abcd" and rle_decode("abcd") == "abcd"

    # runs of exactly two keep their length; runs of three or more shrink
    assert rle_encode("aab") == "a2b"
    assert rle_encode("abb") == "ab2"
    assert rle_encode("aabbaa") == "a2b2a2"
    assert rle_decode("a2b2a2") == "aabbaa"
    assert rle_encode("aaabccccd") == "a3bc4d"
    assert rle_decode("a3bc4d") == "aaabccccd"
    assert rle_decode("z2") == "zz"

    # multi-digit counts (the 9 -> 10 boundary)
    assert rle_encode("a" * 9 + "b") == "a9b"
    assert rle_encode("a" * 10) == "a10" and rle_decode("a10") == "a" * 10
    assert rle_encode("a" * 12) == "a12" and rle_decode("a12") == "a" * 12
    assert rle_decode("a99") == "a" * 99

    # canonical outputs round trip through decode then encode
    for encoded in ("a3b2c", "a2b2a2", "a9b", "a10", "a12", "a3bc4d", "abcd", "a", "a99"):
        assert rle_encode(rle_decode(encoded)) == encoded, encoded

    # malformed input is rejected (documented)
    for bad in ("3", "a0", "a1", "ab1"):
        try:
            rle_decode(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for %r" % (bad,))

    # LCG strings: round trips, canonical idempotence, and the letters of the encoding
    seed = 5150
    for _ in range(200):
        seed = (seed * 75 + 74) % 65537
        length = seed % 60
        seed = (seed * 75 + 74) % 65537
        alphabet = 2 + seed % 4
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(chr(ord("a") + seed % alphabet))
        text = "".join(chars)
        encoded = rle_encode(text)
        assert len(encoded) <= len(text), text
        assert run_characters(text) == "".join(ch for ch in encoded if is_letter(ch)), text
        assert rle_decode(encoded) == text, text
        assert rle_encode(rle_decode(encoded)) == encoded, text


if __name__ == "__main__":
    main()
    print("PASS")
