"""Task 389 - Implement Huffman coding for characters frequency. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)

Huffman's greedy prefix code: repeatedly merge the two least frequent nodes of a min-heap.
Deterministic tie-break (documented): nodes are ordered by (frequency, creation id); leaves
get ids in sorted character order and internal nodes get ids in creation order. On merge the
first popped node becomes the left child (bit '0') and the second the right child (bit '1').
A single distinct character yields the one code "0" (nothing to discriminate). The weighted
path length sum(freq * len(code)) equals the encoded bit length.
"""
import heapq


def character_frequencies(text):
    frequencies = {}
    for ch in text:
        frequencies[ch] = frequencies.get(ch, 0) + 1
    return frequencies


def huffman_codes(frequencies):
    """Return (codes, weighted_length); codes maps character -> bit string."""
    if not frequencies:
        return {}, 0
    heap = [(freq, node_id, ch, None, None)
            for node_id, (ch, freq) in enumerate(sorted(frequencies.items()))]
    heapq.heapify(heap)
    next_id = len(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        heapq.heappush(heap, (left[0] + right[0], next_id, None, left, right))
        next_id += 1
    codes = {}

    def walk(node, prefix):
        _, _, ch, left, right = node
        if ch is not None:
            codes[ch] = prefix if prefix else "0"
            return
        walk(left, prefix + "0")
        walk(right, prefix + "1")

    walk(heap[0], "")
    weighted = sum(frequencies[ch] * len(code) for ch, code in codes.items())
    return codes, weighted


def huffman_encode(text):
    """Return (bits, codes, weighted_length)."""
    codes, weighted = huffman_codes(character_frequencies(text))
    return "".join(codes[ch] for ch in text), codes, weighted


def huffman_decode(bits, codes):
    reverse = {code: ch for ch, code in codes.items()}
    out = []
    current = ""
    for bit in bits:
        current += bit
        if current in reverse:
            out.append(reverse[current])
            current = ""
    if current:
        raise ValueError("bits end with an incomplete code")
    return "".join(out)


def is_prefix_free(codes):
    for code in codes.values():
        for other in codes.values():
            if code != other and other.startswith(code):
                return False
    return True


def kraft_sum_ok(codes):
    """Prefix codes of a complete binary tree satisfy sum 2^(-len) == 1. The degenerate
    single-symbol table is a lone leaf, so its sum is 1/2 and it is checked separately."""
    if not codes:
        return True
    if len(codes) == 1:
        return list(codes.values()) == ["0"]
    longest = max(len(code) for code in codes.values())
    return sum(2 ** (longest - len(code)) for code in codes.values()) == 2 ** longest


def ceil_log2(value):
    bits = 0
    while (1 << bits) < value:
        bits += 1
    return bits


def main():
    # hand-verified small example: a:2, b:3, c:4 -> {c: "0", a: "10", b: "11"}
    text = "aabbbcccc"
    assert character_frequencies(text) == {"a": 2, "b": 3, "c": 4}
    codes, weighted = huffman_codes({"a": 2, "b": 3, "c": 4})
    assert codes == {"c": "0", "a": "10", "b": "11"}
    assert weighted == 14 == 2 * 2 + 3 * 2 + 4 * 1
    bits, codes2, weighted2 = huffman_encode(text)
    assert codes2 == codes and weighted2 == 14
    assert bits == "10101111110000" and len(bits) == weighted
    assert huffman_decode(bits, codes) == text

    # single distinct character has the single code "0"
    codes1, weighted1 = huffman_codes({"a": 4})
    assert codes1 == {"a": "0"} and weighted1 == 4
    bits1, _, _ = huffman_encode("aaaa")
    assert bits1 == "0000" and huffman_decode(bits1, codes1) == "aaaa"

    # empty text
    assert huffman_encode("") == ("", {}, 0)
    assert huffman_decode("", {}) == ""

    # second hand-verified structure, built straight from frequencies
    codes_big, weighted_big = huffman_codes({"a": 35, "b": 17, "c": 17, "d": 16, "e": 15})
    assert codes_big == {"a": "0", "e": "100", "d": "101", "b": "110", "c": "111"}
    assert weighted_big == 230 == 35 * 1 + (17 + 17 + 16 + 15) * 3

    # prefix-free property and the Kraft equality (a complete binary tree)
    for table in (codes, codes1, codes_big):
        assert is_prefix_free(table)
        assert kraft_sum_ok(table)

    # decoding rejects a dangling prefix (documented error)
    try:
        huffman_decode("1", {"c": "0", "a": "10", "b": "11"})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an incomplete bit string")

    # LCG strings: round trips, bit length equals the weighted path length
    seed = 2718
    for _ in range(120):
        seed = (seed * 75 + 74) % 65537
        length = seed % 80
        seed = (seed * 75 + 74) % 65537
        alphabet = 1 + seed % 6
        chars = []
        for _ in range(length):
            seed = (seed * 75 + 74) % 65537
            chars.append(chr(ord("a") + seed % alphabet))
        sample = "".join(chars)
        encoded, table, weight = huffman_encode(sample)
        assert huffman_decode(encoded, table) == sample, sample
        assert len(encoded) == weight, sample
        assert is_prefix_free(table) and kraft_sum_ok(table), sample
        if sample:
            distinct = len(set(sample))
            assert weight <= len(sample) * max(1, ceil_log2(distinct)), sample


if __name__ == "__main__":
    main()
    print("PASS")
