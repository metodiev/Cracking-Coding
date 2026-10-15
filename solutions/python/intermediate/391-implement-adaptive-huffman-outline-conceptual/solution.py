"""Task 391 - Implement adaptive Huffman (outline, conceptual). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)

Adaptive Huffman rebuilds the code tree from the running frequencies: both encoder and
decoder rebuild the tree before every symbol, so the codes evolve with the data and no
frequency table is transmitted. FGK maintains the same tree incrementally in O(1) per
symbol by keeping the sibling property; the rebuild here is the O(n log n)-per-symbol
conceptual equivalent. Node tie-break is (weight, creation order) so the tree shape is
identical in every language and can be asserted exactly.
"""

class Node:
    __slots__ = ("weight", "seq", "ch", "left", "right")

    def __init__(self, weight, seq, ch=None, left=None, right=None):
        self.weight = weight
        self.seq = seq
        self.ch = ch
        self.left = left
        self.right = right


def build_tree(freq, alphabet):
    """Builds the Huffman tree; ties broken by (weight, creation order)."""
    work = []
    seq = 0
    for ch in alphabet:
        work.append(Node(freq[ch], seq, ch))
        seq += 1
    while len(work) > 1:
        work.sort(key=lambda node: (node.weight, node.seq))
        x, y = work[0], work[1]
        work = work[2:]
        work.append(Node(x.weight + y.weight, seq, None, x, y))
        seq += 1
    return work[0]


def collect_codes(root, prefix="", out=None):
    out = {} if out is None else out
    if root.ch is not None:
        out[root.ch] = prefix
        return out
    collect_codes(root.left, prefix + "0", out)
    collect_codes(root.right, prefix + "1", out)
    return out


def adaptive_encode(text, alphabet):
    freq = {ch: 1 for ch in alphabet}
    bits = []
    for ch in text:
        if ch not in freq:
            raise ValueError("symbol outside the alphabet: " + ch)
        codes = collect_codes(build_tree(freq, alphabet))
        bits.append(codes[ch])
        freq[ch] += 1
    return "".join(bits)


def adaptive_decode(bits, alphabet):
    freq = {ch: 1 for ch in alphabet}
    out = []
    i = 0
    while i < len(bits):
        node = build_tree(freq, alphabet)
        while node.ch is None:
            node = node.left if bits[i] == "0" else node.right
            i += 1
        out.append(node.ch)
        freq[node.ch] += 1
    return "".join(out)


def lcg_strings(alphabet, length, count, seed=6060):
    strings = []
    for _ in range(count):
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(alphabet[seed % len(alphabet)])
        strings.append("".join(chars))
    return strings


def main():
    # Hand-derived: alphabet "abc" starts at weights 1/1/1 -> a="10", b="11", c="0".
    assert adaptive_encode("a", "abc") == "10"
    # After each symbol the tree is rebuilt from the running counts; the ten bits below
    # come from the five trees of "aaabbc" (derivation in the task folder docs).
    assert adaptive_encode("aaabbc", "abc") == "1001000110"
    assert adaptive_encode("aaa", "abc") == "1001"
    assert adaptive_encode("", "abc") == ""
    assert adaptive_decode("1001000110", "abc") == "aaabbc"

    for s in ["a", "abc", "aaabbc", "abababab", "cccbaaa"] + lcg_strings("abc", 24, 40):
        assert adaptive_decode(adaptive_encode(s, "abc"), "abc") == s, s

    # adaptivity: the most frequent symbol ends up with the shortest possible code
    text = "a" * 40 + "b" + "c"
    codes = collect_codes(build_tree({"a": 41, "b": 2, "c": 2}, "abc"))
    assert codes["a"] == "1" and len(codes["b"]) == len(codes["c"]) == 2

    try:
        adaptive_encode("d", "abc")
        raise AssertionError("expected a ValueError for out-of-alphabet symbols")
    except ValueError:
        pass


if __name__ == "__main__":
    main()
    print("PASS")
