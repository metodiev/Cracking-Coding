"""Task 197 - Implement checksum-based file difference (simplified rsync idea). (Tier: Beginner, CTCI Beyond)

Split file A into fixed-size blocks (the last one may be shorter). Every block gets a
cheap "weak" checksum (two byte sums, like the classic rsync rolling checksum) and is
confirmed with a strong SHA-256 hash when a weak match shows up on the B side. B is
scanned with a sliding window (recomputed from scratch - no rolling update, which is
the documented simplification) and the report maps each block index of A to the first
position in B where that block's content occurs. Partial trailing blocks are looked up
directly, also documented as a simplification.
"""
import hashlib

MASK16 = 0xFFFF


def weak_checksum(block):
    a = 0
    b = 0
    length = len(block)
    for i, byte in enumerate(block):
        a += byte
        b += (length - i) * byte
    return ((a & MASK16) << 16) | (b & MASK16)


def strong_hash(block):
    return hashlib.sha256(block).hexdigest()


def split_blocks(data, block_size):
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return [data[i:i + block_size] for i in range(0, len(data), block_size)]


def find_block_matches(a, b, block_size):
    blocks = split_blocks(a, block_size)
    index_by_weak = {}
    for index, block in enumerate(blocks):
        index_by_weak.setdefault(weak_checksum(block), []).append(index)

    matches = {}
    for start in range(0, len(b) - block_size + 1):
        window = b[start:start + block_size]
        for index in index_by_weak.get(weak_checksum(window), ()):
            if index in matches or len(blocks[index]) != block_size:
                continue
            if strong_hash(window) == strong_hash(blocks[index]):     # weak -> strong confirmation
                matches[index] = start

    if blocks:
        tail_index = len(blocks) - 1
        tail = blocks[tail_index]
        if len(tail) < block_size and tail_index not in matches:
            position = b.find(tail)                                   # simplified tail lookup
            if position >= 0:
                matches[tail_index] = position
    return matches


def main():
    a = bytes(range(40))                                              # 40 bytes -> 5 blocks of 8
    assert find_block_matches(a, a, 8) == {0: 0, 1: 8, 2: 16, 3: 24, 4: 32}

    changed = bytearray(a)
    changed[10] ^= 0xFF                                               # one byte inside block 1
    assert find_block_matches(a, bytes(changed), 8) == {0: 0, 2: 16, 3: 24, 4: 32}

    assert find_block_matches(a, bytes([9]) * 64, 8) == {}            # completely different content

    shifted = b"HEADER" + a
    assert find_block_matches(a, shifted, 8) == {0: 6, 1: 14, 2: 22, 3: 30, 4: 38}

    assert find_block_matches(b"ABABABAB", b"ZZABABABAB", 4) == {0: 2, 1: 2}   # duplicate blocks
    assert find_block_matches(b"abc", b"xxabcxx", 8) == {0: 2}                  # partial tail
    assert find_block_matches(b"", a, 8) == {}
    assert weak_checksum(b"AB") != weak_checksum(b"BA")

    try:
        find_block_matches(a, a, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected error for block_size == 0")


if __name__ == "__main__":
    main()
    print("PASS")
