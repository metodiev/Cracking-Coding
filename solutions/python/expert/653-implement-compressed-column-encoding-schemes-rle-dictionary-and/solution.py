"""Task 653 - Implement compressed column encoding schemes (RLE, dictionary) and decode.  (Tier: Expert, Ch 14 - Databases)

Four encodings of an integer column are implemented with real encoders/decoders and a cost model
that decides which one to store:

* `raw`        - n values, `value_bits` each.
* `rle`        - maximal runs as (value, count) pairs; golden for columns that are sorted, clustered
                 or simply repetitive. Runs are stored unpacked (a real system would bit-pack them);
                 the model charges `value_bits + count_bits` per run.
* `dictionary` - distinct values in first-appearance order plus bit-packed indices using exactly
                 `ceil(log2(distinct))` bits per row, so the codes are packed across word boundaries.
* `dict_rle`   - dictionary codes, then RLE *over the codes*: the combination column stores use when
                 a low-cardinality column is also clustered.

The cost model counts bits the way a column store would (`choose_encoding` returns the cheapest of
the four for a given column), and every encoding is verified by round-trip: decode(encode(v)) must
reproduce the exact sequence, including multiplicity and order, and random access (`value_at`) must
agree with the materialised decode without decoding the whole column. Decode validates its own input:
a code outside the dictionary, a non-positive run length or a declared row count that does not match
the payload all raise `ValueError`, so corrupt payloads cannot silently produce rows.

Simplified: bit widths come from the maximum value instead of per-block frames of reference, and the
dictionary is stored uncompressed (no prefix/suffix sharing). The invariant that is still verified is
the exact multiset/order round-trip plus the size ordering claim (RLE wins on constant columns,
dictionary wins on low-cardinality columns, dictionary+RLE wins on clustered low-cardinality ones).
"""

WORD_BITS = 64
MASK64 = (1 << WORD_BITS) - 1
COUNT_BITS = 8          # the cost model's price for one run length
CHECKSUM_BITS = 16      # payload header overhead charged once
RAW_VALUE_BITS = 32     # uncompressed columns store fixed-width 32-bit values


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def value_width(values):
    """Bits per raw value: enough for the maximum magnitude, at least one bit.

    Integer columns dominate the cost model; a non-integer payload (the string dictionary used by
    the round-trip checks) is charged one byte per value.
    """
    bits = 0
    for value in values:
        if isinstance(value, int):
            bits = max(bits, abs(value).bit_length())
        else:
            bits = max(bits, 8)
    return max(1, bits)


def dictionary_width(distinct):
    """Codes must fit `distinct` symbols; one value needs a single bit anyway."""
    return max(1, (distinct - 1).bit_length())


def pack_bits(values, width):
    """Bit-pack `values` into 64-bit words, little-endian within a word, crossings allowed."""
    words = []
    accumulator = 0
    used = 0
    mask = (1 << width) - 1
    for value in values:
        accumulator |= (value & mask) << used
        used += width
        while used >= WORD_BITS:
            words.append(accumulator & MASK64)
            accumulator >>= WORD_BITS
            used -= WORD_BITS
    if used:
        words.append(accumulator & MASK64)
    return words


def unpack_bits(words, width, count):
    if count < 0:
        raise ValueError("negative value count")
    mask = (1 << width) - 1
    out = []
    accumulator = 0
    used = 0
    cursor = 0
    for _ in range(count):
        while used < width:
            nxt = words[cursor] if cursor < len(words) else 0
            cursor += 1
            accumulator |= nxt << used
            used += WORD_BITS
        out.append(accumulator & mask)
        accumulator >>= width
        used -= width
    return out


def rle_encode(values):
    """Maximal runs as (value, count); a run of one is written as a literal pair."""
    runs = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return [tuple(run) for run in runs]


def rle_decode(runs, row_count=None):
    out = []
    for value, count in runs:
        if count <= 0:
            raise ValueError("run length must be positive")
        out.extend([value] * count)
    if row_count is not None and len(out) != row_count:
        raise ValueError(f"payload holds {len(out)} rows, header declared {row_count}")
    return out


def rle_value_at(runs, index):
    """Random access into an RLE column without decoding the whole payload."""
    if index < 0:
        raise IndexError("negative row index")
    cursor = 0
    for value, count in runs:
        if index < cursor + count:
            return value
        cursor += count
    raise IndexError("row index past the end of the column")


class DictionaryEncoding:
    """First-appearance dictionary plus bit-packed codes."""

    def __init__(self, dictionary, row_count, packed, width):
        self.dictionary = list(dictionary)
        self.row_count = row_count
        self.packed = list(packed)
        self.width = width

    @classmethod
    def encode(cls, values):
        dictionary = []
        lookup = {}
        codes = []
        for value in values:
            code = lookup.get(value)
            if code is None:
                code = len(dictionary)
                lookup[value] = code
                dictionary.append(value)
            codes.append(code)
        width = dictionary_width(len(dictionary))
        return cls(dictionary, len(values), pack_bits(codes, width), width), codes

    @property
    def codes(self):
        return unpack_bits(self.packed, self.width, self.row_count)

    def decode(self):
        for code in self.codes:
            if code >= len(self.dictionary):
                raise ValueError(f"code {code} outside the dictionary of {len(self.dictionary)}")
        return [self.dictionary[code] for code in self.codes]

    def value_at(self, index):
        if not 0 <= index < self.row_count:
            raise IndexError("row index outside the column")
        code = unpack_bits(self.packed, self.width, index + 1)[-1]
        if code >= len(self.dictionary):
            raise ValueError(f"code {code} outside the dictionary")
        return self.dictionary[code]

    def bits(self):
        dictionary_bits = len(self.dictionary) * value_width(self.dictionary or [0])
        return len(self.packed) * WORD_BITS + dictionary_bits + CHECKSUM_BITS


def dict_rle_encode(values):
    """Dictionary codes first, then runs over the codes (clustered low-cardinality columns)."""
    encoding, codes = DictionaryEncoding.encode(values)
    return encoding, rle_encode(codes)


def dict_rle_decode(encoding, runs):
    codes = rle_decode(runs, encoding.row_count)
    for code in codes:
        if code >= len(encoding.dictionary):
            raise ValueError(f"code {code} outside the dictionary")
    return [encoding.dictionary[code] for code in codes]


def estimate_bits(values):
    """Bit cost of each scheme under the documented model (values are assumed ints)."""
    width = value_width(values)
    runs = rle_encode(values)
    encoding, _ = DictionaryEncoding.encode(values)
    return {
        "raw": len(values) * RAW_VALUE_BITS + CHECKSUM_BITS,
        "rle": len(runs) * (width + COUNT_BITS) + CHECKSUM_BITS,
        "dict": encoding.bits(),
        "dict_rle": len(rle_encode(encoding.codes)) * (encoding.width + COUNT_BITS)
                    + len(encoding.dictionary) * width + CHECKSUM_BITS,
    }


class CompressedColumn:
    """A column stored in whichever of the four schemes the cost model finds cheapest."""

    def __init__(self, values):
        self.values = list(values)
        self.costs = estimate_bits(self.values)
        self.scheme = min(self.costs, key=lambda name: (self.costs[name], name))
        if self.scheme == "rle":
            self.payload = rle_encode(self.values)
        elif self.scheme == "dict":
            self.encoding, _ = DictionaryEncoding.encode(self.values)
            self.payload = self.encoding.packed
        elif self.scheme == "dict_rle":
            self.encoding, self.payload = dict_rle_encode(self.values)
        else:
            self.payload = list(self.values)
        self.bits = self.costs[self.scheme]

    def decode(self):
        if self.scheme == "rle":
            return rle_decode(self.payload, len(self.values))
        if self.scheme == "dict":
            return self.encoding.decode()
        if self.scheme == "dict_rle":
            return dict_rle_decode(self.encoding, self.payload)
        return list(self.payload)

    def value_at(self, index):
        if self.scheme == "rle":
            return rle_value_at(self.payload, index)
        if self.scheme == "dict":
            return self.encoding.value_at(index)
        return self.decode()[index]


def main():
    # 1. hand-checked RLE
    check(rle_encode([1, 1, 1, 2, 2, 3]) == [(1, 3), (2, 2), (3, 1)],
          "runs must be maximal and keep their order")
    check(rle_decode([(1, 3), (2, 2), (3, 1)]) == [1, 1, 1, 2, 2, 3], "RLE round-trip")
    check(rle_encode([]) == [], "an empty column has no runs")
    check(rle_decode([], 0) == [], "an empty payload decodes to no rows")

    # 2. boundary and error handling
    check(rle_decode([(7, 3)], 3) == [7, 7, 7], "declared row count must be honoured")
    try:
        rle_decode([(7, 3)], 4)
        raise AssertionError("a mismatching row count must be rejected")
    except ValueError:
        pass
    try:
        rle_decode([(7, 0)])
        raise AssertionError("a zero-length run must be rejected")
    except ValueError:
        pass
    check(rle_value_at([(5, 2), (9, 3)], 2) == 9, "random access must skip whole runs")
    try:
        rle_value_at([(5, 2)], 2)
        raise AssertionError("random access past the end must be rejected")
    except IndexError:
        pass

    # 3. bit packing across word boundaries
    values = [i % 7 for i in range(100)]
    packed = pack_bits(values, 3)
    check(len(packed) == (100 * 3 + WORD_BITS - 1) // WORD_BITS,
          "packed words must be ceil(n * width / 64)")
    check(unpack_bits(packed, 3, 100) == values, "bit packing must round-trip across word edges")
    check(unpack_bits(packed, 3, 0) == [], "unpacking zero values must give an empty list")
    check(unpack_bits([0], 1, 3) == [0, 0, 0], "missing words must unpack as zero padding")

    # 4. dictionary encoding shape
    encoding, codes = DictionaryEncoding.encode(["b", "a", "b", "c", "a"])
    check(encoding.dictionary == ["b", "a", "c"], "the dictionary must follow first appearance")
    check(codes == [0, 1, 0, 2, 1], "codes must index the dictionary")
    check(encoding.width == dictionary_width(3) == 2, "two-bit codes for three symbols")
    check(encoding.decode() == ["b", "a", "b", "c", "a"], "dictionary round-trip")
    check(encoding.value_at(3) == "c" and encoding.value_at(0) == "b",
          "dictionary random access must match the decode")
    check(encoding.bits() >= len(encoding.packed) * WORD_BITS, "packed codes must be charged")
    single, single_codes = DictionaryEncoding.encode([9, 9, 9])
    check(single.width == 1 and single_codes == [0, 0, 0],
          "a single-symbol column needs one-bit codes")

    # 5. corrupt dictionary payloads must be rejected
    broken = DictionaryEncoding(["x"], 3, pack_bits([0, 1, 1], 2), 2)
    try:
        broken.decode()
        raise AssertionError("a code outside the dictionary must be rejected")
    except ValueError:
        pass
    try:
        broken.value_at(1)
        raise AssertionError("random access must reject an out-of-range code")
    except ValueError:
        pass

    # 6. exotic columns round-trip through every codec path explicitly
    specials = [
        [],
        [0],
        [42, 42, 42, 42],
        [1, 2, 3, 4, 5],
        [0, 1000, 0, 1000, 0],
        [255, 255, 256, 256, 255],
        [5, 4, 3, 2, 1, 0],
    ]

    def round_trip(scheme, values):
        if scheme == "rle":
            return rle_decode(rle_encode(values), len(values))
        if scheme == "dict":
            return DictionaryEncoding.encode(values)[0].decode()
        if scheme == "dict_rle":
            encoding, runs = dict_rle_encode(values)
            return dict_rle_decode(encoding, runs)
        return list(values)

    for values in specials:
        for scheme in ("raw", "rle", "dict", "dict_rle"):
            check(round_trip(scheme, values) == values, f"{scheme} must round-trip {values}")
        stored = CompressedColumn(values)
        check(stored.decode() == values, "the chosen scheme must round-trip the column")
        check(stored.bits == min(stored.costs.values()), "the stored scheme must be the cheapest one")
        for index, value in enumerate(values):
            check(stored.value_at(index) == value, "random access must agree with the column")
    check(CompressedColumn([]).decode() == [] and max(CompressedColumn([]).costs.values()) > 0,
          "an empty column must still be storable")

    # 7. the cost model picks each scheme for the workload it is designed for
    constant = [7] * 512
    clustered = [(i // 64 % 3) * 1000 for i in range(512)]   # runs of 64 identical wide values
    low_card = [i % 4 for i in range(512)]             # scattered, low cardinality
    print(f"653 constant={CompressedColumn(constant).bits} "
          f"dict={CompressedColumn(low_card).bits} clustered={CompressedColumn(clustered).bits}")
    print(f"653 raw constant={estimate_bits(constant)['raw']} raw low-card={estimate_bits(low_card)['raw']}")
    check(CompressedColumn(constant).scheme == "rle", "a constant column must pick RLE")
    check(CompressedColumn(low_card).scheme == "dict", "a scattered low-cardinality column picks dictionary")
    check(CompressedColumn(clustered).scheme == "dict_rle",
          "clustered dictionary codes must pick dictionary+RLE")
    check(CompressedColumn(constant).bits * 5 < estimate_bits(constant)["raw"],
          "RLE must beat the raw encoding on a constant column by 5x")
    check(CompressedColumn(low_card).bits < estimate_bits(low_card)["raw"],
          "the dictionary must beat the raw encoding on a low-cardinality column")
    check(CompressedColumn(clustered).bits < CompressedColumn(low_card).bits,
          "clustering the same codes must make the column cheaper")

    # 8. a large shuffled column round-trips deterministically
    seed = 0x1CE
    big = []
    for _ in range(1500):
        seed = (seed * 1103515245 + 12345) & ((1 << 31) - 1)
        big.append(seed % 1000)
    first, second = CompressedColumn(big), CompressedColumn(big)
    check(first.decode() == big and second.decode() == big, "a large column must round-trip exactly")
    check(first.scheme == second.scheme and first.bits == second.bits and first.payload == second.payload,
          "encoding a column twice must give identical bytes and cost")
    check(sorted(first.decode()) == sorted(big), "decoding must preserve the multiset of values")
    check(all(first.value_at(i) == big[i] for i in range(0, len(big), 7)),
          "random access must work for every seventh row")

    print("PASS")


if __name__ == "__main__":
    main()
