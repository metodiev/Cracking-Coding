"""Task 390 - Implement Shannon-Fano coding (contrast with Huffman). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms)

Shannon-Fano (top-down): sort symbols by frequency descending (ties by character ascending,
documented), then split the list at the point that minimises the absolute difference of the
two halves' total frequencies -- the earliest split wins ties -- assign '0' to the left part
and '1' to the right, and recurse. A local Huffman implementation (bottom-up greedy merging,
tie-break by (frequency, creation id)) is included for the contrast. Both are prefix codes and
Huffman's weighted length sum(freq * len) is always <= Shannon-Fano's, because Huffman is
optimal among prefix codes. Examples: "aabbbcccc" ties (both 14 bits), while
{a:35, b:17, c:17, d:16, e:15} is strictly better for Huffman (230 vs 231 bits).
"""
import heapq


def character_frequencies(text):
    frequencies = {}
    for ch in text:
        frequencies[ch] = frequencies.get(ch, 0) + 1
    return frequencies


def shannon_fano_codes(frequencies):
    """Return codes mapping character -> bit string ({} for empty input)."""
    codes = {}
    if not frequencies:
        return codes
    items = sorted(frequencies.items(), key=lambda kv: (-kv[1], kv[0]))

    def rec(part, prefix):
        if len(part) == 1:
            codes[part[0][0]] = prefix if prefix else "0"
            return
        total = sum(freq for _, freq in part)
        best_split, best_diff, left_sum = 1, None, 0
        for k in range(1, len(part)):
            left_sum += part[k - 1][1]
            diff = abs(total - 2 * left_sum)
            if best_diff is None or diff < best_diff:
                best_split, best_diff = k, diff
        rec(part[:best_split], prefix + "0")
        rec(part[best_split:], prefix + "1")

    rec(items, "")
    return codes


def huffman_codes(frequencies):
    """Local Huffman re-implementation (same tie-break as task 389)."""
    if not frequencies:
        return {}
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
        _, _, ch, lft, rgt = node
        if ch is not None:
            codes[ch] = prefix if prefix else "0"
            return
        walk(lft, prefix + "0")
        walk(rgt, prefix + "1")

    walk(heap[0], "")
    return codes


def weighted_length(codes, frequencies):
    return sum(frequencies[ch] * len(code) for ch, code in codes.items())


def encode(text, codes):
    return "".join(codes[ch] for ch in text)


def decode(bits, codes):
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
    """sum 2^(-len) == 1 for a complete binary tree; single-symbol table "0" is a lone leaf."""
    if not codes:
        return True
    if len(codes) == 1:
        return list(codes.values()) == ["0"]
    longest = max(len(code) for code in codes.values())
    return sum(2 ** (longest - len(code)) for code in codes.values()) == 2 ** longest


def main():
    # tie example: aabbbcccc -> Shannon-Fano and Huffman both cost 14 bits
    tie_freq = {"a": 2, "b": 3, "c": 4}
    sf = shannon_fano_codes(tie_freq)
    huff = huffman_codes(tie_freq)
    assert sf == {"c": "0", "b": "10", "a": "11"}
    assert huff == {"c": "0", "a": "10", "b": "11"}
    assert weighted_length(sf, tie_freq) == 14
    assert weighted_length(huff, tie_freq) == 14
    assert weighted_length(sf, tie_freq) == weighted_length(huff, tie_freq)  # documented tie
    for codes in (sf, huff):
        assert is_prefix_free(codes) and kraft_sum_ok(codes)
        assert decode(encode("aabbbcccc", codes), codes) == "aabbbcccc"

    # strict example: Huffman 230 < Shannon-Fano 231
    big_freq = {"a": 35, "b": 17, "c": 17, "d": 16, "e": 15}
    sf_big = shannon_fano_codes(big_freq)
    huff_big = huffman_codes(big_freq)
    assert sf_big == {"a": "00", "b": "01", "c": "10", "d": "110", "e": "111"}
    assert huff_big == {"a": "0", "e": "100", "d": "101", "b": "110", "c": "111"}
    assert weighted_length(sf_big, big_freq) == 231
    assert weighted_length(huff_big, big_freq) == 230
    assert weighted_length(huff_big, big_freq) < weighted_length(sf_big, big_freq)
    assert len(sf_big["a"]) == 2 and len(huff_big["a"]) == 1
    for codes in (sf_big, huff_big):
        assert is_prefix_free(codes) and kraft_sum_ok(codes)

    # edge cases
    assert shannon_fano_codes({}) == {} and huffman_codes({}) == {}
    assert shannon_fano_codes({"a": 4}) == {"a": "0"} == huffman_codes({"a": 4})
    assert decode("", {}) == "" and encode("", {}) == ""

    # LCG strings: both codecs round trip, prefix-free, and Huffman <= Shannon-Fano
    seed = 8642
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
        freq = character_frequencies(sample)
        sf_codes = shannon_fano_codes(freq)
        huff_codes = huffman_codes(freq)
        for codes in (sf_codes, huff_codes):
            assert is_prefix_free(codes) and kraft_sum_ok(codes), sample
            bits = encode(sample, codes)
            assert decode(bits, codes) == sample, sample
            assert len(bits) == weighted_length(codes, freq), sample
        assert weighted_length(huff_codes, freq) <= weighted_length(sf_codes, freq), sample


if __name__ == "__main__":
    main()
    print("PASS")
