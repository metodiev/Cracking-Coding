"""Task 392 - Implement LZW compression (encoder/decoder skeleton). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)

The encoder keeps the longest dictionary match and emits its code; the decoder rebuilds the
same dictionary on the fly, using the "cSc" rule for the one entry the encoder adds ahead.
Dictionary ids are assigned in first-seen order, so the code stream is deterministic and
asserted exactly in all four languages.
"""

def lzw_encode(text, alphabet):
    """alphabet: an iterable of distinct single-character symbols (its order fixes the ids)."""
    dictionary = {ch: i for i, ch in enumerate(alphabet)}
    next_code = len(dictionary)
    codes = []
    current = ""
    for ch in text:
        if ch not in dictionary and current == "":
            raise ValueError("symbol outside the alphabet: " + ch)
        candidate = current + ch
        if candidate in dictionary:
            current = candidate
        else:
            codes.append(dictionary[current])
            dictionary[candidate] = next_code
            next_code += 1
            current = ch
            if ch not in dictionary:
                raise ValueError("symbol outside the alphabet: " + ch)
    if current:
        codes.append(dictionary[current])
    return codes


def lzw_decode(codes, alphabet):
    symbols = list(alphabet)
    dictionary = {i: ch for i, ch in enumerate(symbols)}
    next_code = len(dictionary)
    if not codes:
        return ""
    first = codes[0]
    if first not in dictionary:
        raise ValueError("invalid first code")
    previous = dictionary[first]
    out = [previous]
    for code in codes[1:]:
        if code in dictionary:
            entry = dictionary[code]
        elif code == next_code:
            entry = previous + previous[0]           # the cSc rule
        else:
            raise ValueError("invalid code: " + str(code))
        out.append(entry)
        dictionary[next_code] = previous + entry[0]
        next_code += 1
        previous = entry
    return "".join(out)


def lcg_strings(alphabet, length, count, seed=5150):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(alphabet[seed % len(alphabet)])
        strings.append("".join(chars))
    return strings


def main():
    assert lzw_encode("abababa", "ab") == [0, 1, 2, 4]
    assert lzw_encode("aaaa", "ab") == [0, 2, 0]
    assert lzw_encode("abcabcabcabc", "abc") == [0, 1, 2, 3, 5, 4, 6]
    assert lzw_encode("", "ab") == []
    assert lzw_encode("a", "ab") == [0]

    assert lzw_decode([0, 1, 2, 4], "ab") == "abababa"
    assert lzw_decode([0, 2, 0], "ab") == "aaaa"
    assert lzw_decode([0, 1, 2, 3, 5, 4, 6], "abc") == "abcabcabcabc"
    assert lzw_decode([], "ab") == ""
    assert lzw_decode([0], "ab") == "a"

    samples = ["abababa", "aaaa", "", "a"] + lcg_strings("ab", 30, 40)
    for s in samples:
        assert lzw_decode(lzw_encode(s, "ab"), "ab") == s, s
    for s in lcg_strings("abcd", 40, 40):
        assert lzw_decode(lzw_encode(s, "abcd"), "abcd") == s, s

    # LZW shrinks highly repetitive text: "ab"*20 -> fewer codes than symbols
    repetitive = "ab" * 20
    assert len(lzw_encode(repetitive, "ab")) < len(repetitive)

    # a stream that exercises the cSc rule on the decoder side
    csc = lzw_encode("abababab", "ab")
    assert lzw_decode(csc, "ab") == "abababab"


if __name__ == "__main__":
    main()
    print("PASS")
