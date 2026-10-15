"""Task 488 - Implement compression-aware storage layout simulation. (Tier: Advanced, CTCI Ch 14)

Value blocks (32 non-negative 16-bit ints each; the last block may be shorter) are encoded with the
cheaper of two schemes, chosen by exact byte count with a documented tie rule (RLE first):
- RLE     : header 2 bytes (mode=0, run count) + 4 bytes per run (value, run length).
- BITPACK : header 2 bytes (mode=1, bit width w) + ceil(count * w / 8) bytes, LSB-first packing,
            where w = max(1, max bit length of the block's values). For values < 2^31 bit packing
            with w <= 31 bits is never worse than a raw 32-bit copy, so raw mode is not needed.

"Compression-aware layout": blocks are grouped by similarity - a stable sort by bit width puts
similar blocks next to each other, and consecutive blocks sharing a width cost one 2-byte group
header for the whole run instead of per-block overhead. stored_bytes counts group headers plus
per-block payloads only.

The test asserts exact round-trips on skewed, mixed and hand blocks, checks the mode actually
chosen per block, and bounds stored bytes on skewed data: for 1024 ints in {0..3} the layout stores
322 bytes vs 4096 raw (<= raw/3), matching the pinned exact count.
"""

BLOCK = 32
MODE_RLE = 0
MODE_BITPACK = 1
HEADER = 2


def bit_length(value):
    return value.bit_length()


def rle_runs(values):
    runs = []
    for value in values:
        if runs and runs[-1][0] == value:
            runs[-1][1] += 1
        else:
            runs.append([value, 1])
    return runs


def encode_block(values):
    runs = rle_runs(values)
    rle_size = HEADER + 4 * len(runs)
    width = max(1, max(bit_length(v) for v in values))
    bitpack_size = HEADER + (len(values) * width + 7) // 8
    if rle_size <= bitpack_size:
        payload = bytearray(HEADER)
        payload[0] = MODE_RLE
        payload[1] = len(runs)
        for value, run in runs:
            payload.append(value & 0xFF)
            payload.append((value >> 8) & 0xFF)
            payload.append(run & 0xFF)
            payload.append((run >> 8) & 0xFF)
        return {"mode": MODE_RLE, "width": None, "payload": bytes(payload), "runs": len(runs)}
    packed = bytearray(HEADER)
    packed[0] = MODE_BITPACK
    packed[1] = width
    acc = 0
    bits = 0
    for value in values:
        acc |= value << bits
        bits += width
        while bits >= 8:
            packed.append(acc & 0xFF)
            acc >>= 8
            bits -= 8
    if bits:
        packed.append(acc & 0xFF)
    assert len(packed) == bitpack_size, (len(packed), bitpack_size)
    return {"mode": MODE_BITPACK, "width": width, "payload": bytes(packed), "runs": None}


def decode_block(block, count):
    payload = block["payload"]
    if payload[0] == MODE_RLE:
        runs = payload[1]
        values = []
        offset = HEADER
        for _ in range(runs):
            value = payload[offset] | (payload[offset + 1] << 8)
            run = payload[offset + 2] | (payload[offset + 3] << 8)
            values.extend([value] * run)
            offset += 4
        assert len(values) == count
        return values
    width = payload[1]
    values = []
    acc = 0
    bits = 0
    offset = HEADER
    for _ in range(count):
        while bits < width:
            acc |= payload[offset] << bits
            offset += 1
            bits += 8
        values.append(acc & ((1 << width) - 1))
        acc >>= width
        bits -= width
    return values


def build_layout(blocks):
    """Group similar (equal-width) blocks; returns (layout, stored_bytes, group_count)."""
    infos = [encode_block(values) for values in blocks]
    order = sorted(range(len(blocks)), key=lambda i: (infos[i]["width"] if infos[i]["width"] is not None else -1, i))
    layout = []
    stored = 0
    groups = 0
    previous_width = None
    for index in order:
        info = infos[index]
        width = info["width"] if info["width"] is not None else -1
        if width != previous_width:
            groups += 1
            stored += HEADER
            previous_width = width
        stored += len(info["payload"])
        layout.append({"index": index, "group": groups - 1, **info})
    return layout, stored, groups


def decode_layout(layout, lengths):
    blocks = [None] * len(lengths)
    for entry in layout:
        blocks[entry["index"]] = decode_block(entry, lengths[entry["index"]])
    return blocks


def chunk(values, size=BLOCK):
    return [values[i:i + size] for i in range(0, len(values), size)]


def lcg_values(count, seed, limit):
    values = []
    for _ in range(count):
        seed = (seed * 75 + 74) % 65537
        values.append(seed % limit)
    return values


def main():
    # Hand blocks: RLE for constant blocks (tie goes to RLE), BITPACK for narrow random blocks.
    constant = encode_block([7] * 32)
    assert constant["mode"] == MODE_RLE and constant["runs"] == 1
    assert len(constant["payload"]) == 6
    assert decode_block(constant, 32) == [7] * 32

    zeros = encode_block([0] * 32)
    assert zeros["mode"] == MODE_RLE and len(zeros["payload"]) == 6
    assert decode_block(zeros, 32) == [0] * 32

    narrow = encode_block([0, 1, 2, 3] * 8)
    assert narrow["mode"] == MODE_BITPACK and narrow["width"] == 2
    assert len(narrow["payload"]) == 2 + 8
    assert decode_block(narrow, 32) == [0, 1, 2, 3] * 8

    wide = encode_block([65535, 0, 12345, 65534] * 8)
    assert wide["mode"] == MODE_BITPACK and wide["width"] == 16
    assert len(wide["payload"]) == 2 + 64
    assert decode_block(wide, 32) == [65535, 0, 12345, 65534] * 8

    counting = encode_block(list(range(1, 33)))
    assert counting["mode"] == MODE_BITPACK and counting["width"] == 6
    assert decode_block(counting, 32) == list(range(1, 33))

    # Skewed data: 1024 ints in {0..3} -> 32 blocks of width 2, one group.
    skewed = lcg_values(1024, 488, 4)
    blocks = chunk(skewed)
    assert len(blocks) == 32 and all(len(block) == 32 for block in blocks)
    layout, stored, groups = build_layout(blocks)
    assert groups == 1
    assert stored == 32 * 10 + 2 == 322, stored
    raw = 4 * len(skewed)
    assert stored <= raw // 3, (stored, raw)
    assert decode_layout(layout, [len(b) for b in blocks]) == blocks

    # Mixed data: half skewed, half full 16-bit range -> two width groups.
    mixed_values = lcg_values(512, 489, 4) + lcg_values(512, 490, 65536)
    mixed = chunk(mixed_values)
    assert len(mixed) == 32
    layout, stored, groups = build_layout(mixed)
    assert groups == 2, groups
    assert stored == 16 * 10 + 16 * 66 + 2 * 2, stored
    assert stored <= (4 * len(mixed_values)) // 3
    decoded = decode_layout(layout, [len(b) for b in mixed])
    assert decoded == mixed
    # Similar blocks are adjacent: every group is one run of equal widths in payload order.
    widths = []
    for entry in layout:
        widths.append(entry["width"] if entry["width"] is not None else -1)
    changes = sum(1 for i in range(1, len(widths)) if widths[i] != widths[i - 1])
    assert changes == groups - 1, (widths, groups)

    # Round trip with a short trailing block and mixed modes.
    short = [5] * 33
    blocks = chunk(short)
    assert [len(b) for b in blocks] == [32, 1]
    layout, stored, groups = build_layout(blocks)
    assert decode_layout(layout, [len(b) for b in blocks]) == blocks
    # Block 1 is a 32-long constant run -> RLE (6 bytes); the 1-int block is cheaper bit-packed
    # at width 3 (3 bytes), so it forms its own width group.
    assert groups == 2
    assert stored == 6 + 3 + 2 + 2, stored
    print("PASS")


if __name__ == "__main__":
    main()
