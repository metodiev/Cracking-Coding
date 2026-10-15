"""Task 393 - Implement a basic DEFLATE/Brotli idea demo (LZ77 + Huffman, simplified). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)

DEFLATE stacks two ideas: LZ77 removes repetition by emitting (length, distance) matches,
then Huffman coding shortens the remaining bytes. This demo does both in miniature:
token 0 = literal byte, token 1 = (length, distance) with both fields fitting one byte
(window <= 255, match length <= 255). Real DEFLATE additionally uses fixed/dynamic code
tables and 258-length caps; the structure here is the same pipeline.
"""

def lz77_tokens(text, window):
    """Returns tokens: (0, byte, 0) literals and (1, length, distance) matches."""
    data = [ord(ch) for ch in text]
    n = len(data)
    tokens = []
    i = 0
    while i < n:
        best_len, best_dist = 0, 0
        for dist in range(1, min(window, i) + 1):
            length = 0
            while i + length < n and data[i + length] == data[i - dist + length] and length < 255:
                length += 1
            if length > best_len:
                best_len, best_dist = length, dist
        if best_len >= 3:
            tokens.append((1, best_len, best_dist))
            i += best_len
        else:
            tokens.append((0, data[i], 0))
            i += 1
    return tokens


def lz77_decompress(tokens):
    out = []
    for kind, a, b in tokens:
        if kind == 0:
            out.append(a)
        else:
            for _ in range(a):
                out.append(out[len(out) - b])
    return "".join(chr(byte) for byte in out)


def tokens_to_bytes(tokens):
    stream = []
    for kind, a, b in tokens:
        stream.append(kind)
        stream.append(a)
        stream.append(b) if kind == 1 else None
    return stream


def bytes_to_tokens(stream):
    tokens = []
    i = 0
    while i < len(stream):
        kind = stream[i]
        if kind == 0:
            tokens.append((0, stream[i + 1], 0))
            i += 2
        else:
            tokens.append((1, stream[i + 1], stream[i + 2]))
            i += 3
    return tokens


class HuffNode:
    __slots__ = ("weight", "seq", "byte", "left", "right")

    def __init__(self, weight, seq, byte=None, left=None, right=None):
        self.weight = weight
        self.seq = seq
        self.byte = byte
        self.left = left
        self.right = right


def build_huffman(freq):
    work = []
    seq = 0
    for byte in sorted(freq):
        work.append(HuffNode(freq[byte], seq, byte))
        seq += 1
    while len(work) > 1:
        work.sort(key=lambda node: (node.weight, node.seq))
        x, y = work[0], work[1]
        work = work[2:]
        work.append(HuffNode(x.weight + y.weight, seq, None, x, y))
        seq += 1
    return work[0]


def collect_codes(root, prefix="", out=None):
    out = {} if out is None else out
    if root.byte is not None:
        out[root.byte] = prefix
        return out
    collect_codes(root.left, prefix + "0", out)
    collect_codes(root.right, prefix + "1", out)
    return out


def huffman_encode(data):
    if not data:
        return "", {}
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    codes = collect_codes(build_huffman(freq))
    return "".join(codes[byte] for byte in data), codes


def huffman_decode(bits, codes):
    reverse = {code: byte for byte, code in codes.items()}
    out = []
    current = ""
    for bit in bits:
        current += bit
        if current in reverse:
            out.append(reverse[current])
            current = ""
    assert current == "", "truncated code stream"
    return out


def is_prefix_free(codes):
    table = list(codes.values())
    for i in range(len(table)):
        for j in range(len(table)):
            if i != j and table[j].startswith(table[i]):
                return False
    return True


def compress_text(text, window=8):
    return huffman_encode(tokens_to_bytes(lz77_tokens(text, window)))


def decompress_text(bits, codes):
    return lz77_decompress(bytes_to_tokens(huffman_decode(bits, codes)))


def lcg_strings(alphabet, length, count, seed=3131):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(alphabet[seed % len(alphabet)])
        strings.append("".join(chars))
    return strings


def main():
    # Hand-derived tokens (window 8, min match 3):
    assert lz77_tokens("abcabcabc", 8) == [(0, 97, 0), (0, 98, 0), (0, 99, 0), (1, 6, 3)]
    assert lz77_tokens("aaaaaaaa", 8) == [(0, 97, 0), (1, 7, 1)]
    assert lz77_tokens("abc" * 10, 8) == [(0, 97, 0), (0, 98, 0), (0, 99, 0), (1, 27, 3)]
    assert lz77_tokens("", 8) == []
    assert lz77_tokens("ab", 8) == [(0, 97, 0), (0, 98, 0)]

    assert lz77_decompress([(0, 97, 0), (0, 98, 0), (0, 99, 0), (1, 6, 3)]) == "abcabcabc"
    assert lz77_decompress([(0, 97, 0), (1, 7, 1)]) == "aaaaaaaa"

    # serialization round trip
    assert tokens_to_bytes([(0, 97, 0), (1, 6, 3)]) == [0, 97, 1, 6, 3]
    assert bytes_to_tokens([0, 97, 1, 6, 3]) == [(0, 97, 0), (1, 6, 3)]

    samples = ["abcabcabc", "aaaaaaaa", "abc" * 10, "", "ab", "aabaabaa"] + lcg_strings("abc", 18, 30)
    for s in samples:
        assert lz77_decompress(lz77_tokens(s, 8)) == s, s
        assert bytes_to_tokens(tokens_to_bytes(lz77_tokens(s, 8))) == lz77_tokens(s, 8), s

    # full pipeline round trip (the table stands in for DEFLATE's stored code table)
    for s in samples:
        bits, codes = compress_text(s, 8)
        assert is_prefix_free(codes), s
        assert len(bits) == sum(len(codes[b]) for b in tokens_to_bytes(lz77_tokens(s, 8))), s
        assert decompress_text(bits, codes) == s, s

    # the two layers really compress the repetitive examples
    for text in ["abcabcabc", "aaaaaaaa", "abc" * 10]:
        bits, _ = compress_text(text, 8)
        assert len(bits) < 8 * len(text), text

    # a shorter window finds shorter matches
    assert lz77_tokens("abcabcabc", 2) == [(0, 97, 0), (0, 98, 0), (0, 99, 0), (0, 97, 0), (0, 98, 0), (0, 99, 0), (0, 97, 0), (0, 98, 0), (0, 99, 0)]


if __name__ == "__main__":
    main()
    print("PASS")
