"""Task 721 - Implement checkpoint compression and incremental checkpointing.  (Tier: Expert, CTCI Beyond — Machine Learning & AI)

A training checkpoint is a dictionary of parameter tensors; writing it in full after every
epoch dominates the storage budget of long runs.  Two standard tricks are implemented:

* blocking plus delta encoding - each tensor is split into fixed-size blocks (the final block
  of a tensor is ragged) and a new version stores only the blocks whose bytes differ from the
  previous version, plus the tensor names that appeared or disappeared.  The first version is
  therefore a full checkpoint and every later one a small delta.
* run-length encoding of the payload - a payload is mostly repeated bytes (block ids, zero
  regions of sparse updates), so a varint run-length codec shrinks it; the codec is only used
  when it actually helps, so a payload never costs more than the raw bytes plus a header.

Restoring version k replays the base payload and the k-1 deltas that follow it.  The invariant
the checks lean on is exact replay: `struct` round-trips doubles bit for bit, so a restored
version must compare *equal* to the saved state, not merely close.  Independent oracles: the
dirty-block set of each delta is compared with a brute-force block-by-block comparison of the
two versions, the stored sizes are compared with a hand-written full-checkpoint cost, and
integrity is checked by mutating a payload byte and requiring a rejection.

Simplified versus a real framework (PyTorch DCP, TensorFlow checkpointing): nothing touches
the filesystem - payloads stay in memory as byte strings - there is no background writer, no
multi-rank sharding, no deduplication beyond block deltas, and the codec is RLE rather than a
general-purpose compressor.  What is real and checked: block-level dirty detection, delta
payloads, the RLE codec and its round trip, a FNV-1a checksum that rejects corrupted payloads,
ragged final blocks, tensors added and removed between versions, the measured storage ratio of
incremental versus full checkpoints, and determinism.
"""
import struct


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def expect_value_error(call, message):
    try: call()
    except ValueError: return
    raise AssertionError(message)


def pack(values):
    """Exact double encoding: unpack(pack(v)) == v bit for bit."""
    return struct.pack(f"<{len(values)}d", *values)


def unpack(blob):
    return list(struct.unpack(f"<{len(blob) // 8}d", blob))


def fnv1a(data):
    digest = 0x811C9DC5
    for byte in data:
        digest = ((digest ^ byte) * 0x01000193) & 0xFFFFFFFF
    return digest


def varint(value):
    if value < 0:
        raise ValueError("varints are non-negative")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | 0x80 if value else byte)
        if not value:
            return bytes(out)


def read_varint(data, pos):
    shift, value = 0, 0
    while True:
        if pos >= len(data):
            raise ValueError("truncated varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
        shift += 7


def rle_encode(data):
    """Run-length codec: varint(run length) followed by the repeated byte."""
    out = bytearray()
    index = 0
    while index < len(data):
        run = 1
        while index + run < len(data) and data[index + run] == data[index]:
            run += 1
        out += varint(run)
        out.append(data[index])
        index += run
    return bytes(out)


def rle_decode(data):
    out = bytearray()
    pos = 0
    while pos < len(data):
        run, pos = read_varint(data, pos)
        if pos >= len(data):
            raise ValueError("truncated run")
        out += bytes([data[pos]]) * run
        pos += 1
    return bytes(out)


def pack_payload(body):
    """Checksum + length + a flag saying whether the RLE stream was worth using."""
    compressed = rle_encode(body)
    header = varint(fnv1a(body)) + varint(len(body))
    return header + (b"\x01" + compressed if len(compressed) < len(body) else b"\x00" + body)


def unpack_payload(encoded):
    """Inverse of pack_payload; raises when the payload is truncated or corrupted."""
    checksum, pos = read_varint(encoded, 0)
    length, pos = read_varint(encoded, pos)
    if pos >= len(encoded):
        raise ValueError("truncated payload header")
    flag, pos = encoded[pos], pos + 1
    if flag not in (0, 1):
        raise ValueError("unknown payload codec flag")
    body = rle_decode(encoded[pos:]) if flag else encoded[pos:]
    if len(body) != length:
        raise ValueError("payload length mismatch")
    if fnv1a(body) != checksum:
        raise ValueError("payload checksum mismatch - the checkpoint is corrupt")
    return body


class CheckpointStore:
    """In-memory incremental checkpoint store: one payload per version.

    Payload layout: entries for every tensor present in the version (name, block count, the
    dirty blocks), then the names that were deleted.  A block only appears when its bytes
    differ from the same offset in the previous version, which is what makes version 0 a full
    checkpoint and every later version a delta.
    """

    def __init__(self, block_size=8):
        if block_size < 1:
            raise ValueError("block size must be at least one element")
        self.block_size = block_size
        self.block_bytes = block_size * 8
        self.payloads = []            # one encoded payload per version
        self.blocks = []              # name -> {block index: block bytes} per version
        self.raw_sizes = []
        self.changed_blocks = []      # (name, block index) dirtied by the last save
        self.added = []
        self.removed = []

    def save(self, state):
        """Append a version; returns (version, stored bytes, uncompressed payload bytes)."""
        for name, values in state.items():
            if not isinstance(values, list):
                raise ValueError("a tensor must be a list of floats")
        previous = self.blocks[-1] if self.blocks else {}
        current = {}
        for name, values in state.items():
            blob = pack(values)
            current[name] = {i: blob[i * self.block_bytes:(i + 1) * self.block_bytes]
                             for i in range((len(blob) + self.block_bytes - 1) // self.block_bytes)}
        body, changed, added = self._encode(previous, current)
        self.payloads.append(pack_payload(body))
        self.blocks.append(current)
        self.raw_sizes.append(len(body))
        self.changed_blocks = changed
        self.added = added
        self.removed = sorted(set(previous) - set(current))
        return len(self.payloads) - 1, len(self.payloads[-1]), len(body)

    def _encode(self, previous, current):
        body = bytearray(varint(len(current)))
        changed, added = [], []
        for name in sorted(current):
            dirty = []
            for index in sorted(current[name]):
                chunk = current[name][index]
                if previous.get(name, {}).get(index) != chunk:
                    dirty.append((index, chunk))
                    changed.append((name, index))
            if name not in previous:
                added.append(name)
            encoded = name.encode("utf-8")
            body += varint(len(encoded)) + encoded + varint(len(dirty))
            for index, chunk in dirty:
                body += varint(index) + varint(len(chunk)) + chunk
        gone = sorted(set(previous) - set(current))
        body += varint(len(gone))
        for name in gone:
            encoded = name.encode("utf-8")
            body += varint(len(encoded)) + encoded
        return bytes(body), changed, added

    def _decode_payload(self, body):
        """Turn a payload body into (entries, removed names)."""
        count, pos = read_varint(body, 0)
        entries, removed = {}, []
        for _ in range(count):
            name_len, pos = read_varint(body, pos)
            name = body[pos:pos + name_len].decode("utf-8")
            pos += name_len
            dirty_count, pos = read_varint(body, pos)
            records = {}
            for _ in range(dirty_count):
                index, pos = read_varint(body, pos)
                size, pos = read_varint(body, pos)
                records[index] = body[pos:pos + size]
                pos += size
            entries[name] = records
        removed_count, pos = read_varint(body, pos)
        for _ in range(removed_count):
            name_len, pos = read_varint(body, pos)
            removed.append(body[pos:pos + name_len].decode("utf-8"))
            pos += name_len
        if pos != len(body):
            raise ValueError("trailing bytes in the payload")
        return entries, removed

    def restore(self, version):
        """Replay the base plus every delta up to `version` and unpack the block map."""
        if not 0 <= version < len(self.payloads):
            raise ValueError("unknown checkpoint version")
        blocks = {}
        for encoded in self.payloads[:version + 1]:
            entries, removed = self._decode_payload(unpack_payload(encoded))
            for name, records in entries.items():
                blocks.setdefault(name, {}).update(records)
            for name in removed:
                blocks.pop(name, None)
        restored = {}
        for name, records in blocks.items():
            restored[name] = unpack(b"".join(records[i] for i in sorted(records)))
        return restored

    def stored_bytes(self, version):
        """Bytes resident to restore `version`: the base payload plus its deltas."""
        return sum(len(payload) for payload in self.payloads[:version + 1])

    def full_bytes(self, version):
        """Cost of writing `version` as a standalone uncompressed checkpoint."""
        return sum(len(pack(values)) for values in self.restore(version).values())


def changed_blocks_bruteforce(previous, current, block_size):
    """Independent oracle: compare two states block by block without using the store."""
    def blocks(state):
        out = {}
        for name, values in state.items():
            blob = pack(values)
            width = block_size * 8
            out[name] = [blob[i:i + width] for i in range(0, len(blob), width)]
        return out
    left, right = blocks(previous), blocks(current)
    total = 0
    for name in sorted(set(left) | set(right)):
        old, new = left.get(name, []), right.get(name, [])
        for index in range(max(len(old), len(new))):
            if (old[index] if index < len(old) else b"") != (new[index] if index < len(new) else b""):
                total += 1
    return total


def main():
    # 1. the codec layer: RLE round-trips arbitrary bytes and only ever shrinks a payload.
    sample = bytes(range(256)) * 2
    check(rle_decode(rle_encode(sample)) == sample, "RLE must round-trip arbitrary bytes")
    check(rle_decode(rle_encode(b"")) == b"", "RLE must handle an empty payload")
    sparse = b"\x00" * 200 + b"\x07" * 50 + b"\x00" * 100
    check(rle_decode(rle_encode(sparse)) == sparse, "RLE must round-trip sparse bytes")
    check(len(rle_encode(sparse)) < len(sparse) // 10, "RLE must shrink long runs")
    check(unpack_payload(pack_payload(sparse)) == sparse, "pack_payload must round-trip")
    check(unpack_payload(pack_payload(sample)) == sample, "pack_payload must round-trip raw too")
    check(pack([1.5, -2.0]) == struct.pack("<2d", 1.5, -2.0), "pack must be little-endian doubles")
    check(unpack(pack([0.1, 1e300, -1e-300])) == [0.1, 1e300, -1e-300],
          "doubles must round-trip bit for bit")

    # 2. version replay: restoring version k must reproduce exactly what was saved.
    store = CheckpointStore(block_size=8)
    versions = []
    weights = [0.5 * i for i in range(96)]
    bias = [0.25] * 32
    versions.append({"w": list(weights), "b": list(bias)})
    for step in range(1, 6):
        weights[2 * step] += 1.0 + step                  # exactly one block of w changes
        if step % 2 == 0:
            bias[step] = 3.5
        versions.append({"w": list(weights), "b": list(bias)})
    for index, state in enumerate(versions):
        version, stored, raw = store.save(state)
        check(version == index, "versions must be numbered in save order")
        check(store.restore(index) == state, "a restored checkpoint must equal the saved state")
        check(store.full_bytes(index) == len(pack(state["w"])) + len(pack(state["b"])),
              "the full-checkpoint cost is the raw tensor size")

    # 3. delta economics: the base is a full checkpoint, each delta a small fraction of it,
    #    and the whole delta chain costs less than one more full checkpoint.
    base_stored = len(store.payloads[0])
    deltas = [len(p) for p in store.payloads[1:]]
    check(len(deltas) == 5, "five saves after the base must produce five deltas")
    check(all(size < base_stored // 5 for size in deltas),
          "each incremental payload must be under a fifth of the base")
    check(store.stored_bytes(5) - base_stored < base_stored,
          "the whole delta chain must cost less than a second full checkpoint")
    check(store.stored_bytes(5) < 2 * base_stored,
          "six versions must fit in two full checkpoints")
    check(store.raw_sizes[0] <= store.full_bytes(0) + 64,
          "the base payload body must be the raw checkpoint plus a small header")
    check(base_stored < store.full_bytes(0),
          "the run-length codec must already shrink the base checkpoint")
    check(all(len(p) <= raw + 8 for p, raw in zip(store.payloads, store.raw_sizes)),
          "the codec header must stay tiny when RLE is not worth using")

    # 4. dirty-block detection against the brute-force oracle, version by version.
    probe = CheckpointStore(block_size=8)
    for index, state in enumerate(versions):
        probe.save(state)
        reference = changed_blocks_bruteforce(versions[index - 1] if index else {}, state, 8)
        check(len(probe.changed_blocks) == reference,
              "the delta must record exactly the blocks that changed")
        check(sorted(probe.changed_blocks) == sorted(set(probe.changed_blocks)),
              "a block may only be recorded once per delta")

    # 5. an unchanged version produces a header-only payload yet still restores correctly.
    steady = CheckpointStore(block_size=8)
    steady.save({"w": [1.0, 2.0, 3.0]})
    version, stored, raw = steady.save({"w": [1.0, 2.0, 3.0]})
    check(steady.changed_blocks == [], "an unchanged state must produce no dirty blocks")
    check(raw < 12, "an unchanged state must produce a header-only payload")
    check(steady.restore(version) == {"w": [1.0, 2.0, 3.0]}, "a no-op delta must still restore")

    # 6. boundary cases: ragged final block, empty state, block size 1, added and removed
    #    tensors, and a tensor whose tail is the only thing that changed.
    ragged = CheckpointStore(block_size=8)
    ragged.save({"t": [float(i) for i in range(13)]})
    tail = [float(i) for i in range(13)]
    tail[12] = 99.0
    version, stored, raw = ragged.save({"t": tail})
    check(ragged.changed_blocks == [("t", 1)], "only the ragged tail block must be dirty")
    check(ragged.restore(version) == {"t": tail}, "the ragged block must restore exactly")
    empty = CheckpointStore(block_size=4)
    empty.save({})
    check(empty.restore(0) == {}, "an empty state must restore to an empty dict")
    check(len(empty.payloads[0]) <= 12, "an empty state must encode to a tiny payload")
    fine = CheckpointStore(block_size=1)
    fine.save({"v": [1.0, 2.0]})
    fine.save({"v": [1.0, 5.0]})
    check(fine.changed_blocks == [("v", 1)], "with block size 1 every changed value is a block")
    grow = CheckpointStore(block_size=8)
    grow.save({"a": [1.0], "b": [2.0]})
    version, _, _ = grow.save({"a": [1.0], "c": [3.0]})
    check("c" in grow.added and grow.removed == ["b"], "added and removed tensors must be known")
    check(grow.restore(version) == {"a": [1.0], "c": [3.0]},
          "a delta must be able to add and drop tensors")
    check(grow.restore(0) == {"a": [1.0], "b": [2.0]}, "the old version must stay intact")

    # 7. integrity: a flipped payload byte is rejected instead of silently restored, and the
    #    malformed inputs are refused.
    broken = bytearray(store.payloads[1])
    broken[-1] ^= 0xFF
    expect_value_error(lambda: unpack_payload(bytes(broken)), "a corrupted payload must be caught")
    expect_value_error(lambda: store.restore(99), "an unknown version must be rejected")
    expect_value_error(lambda: store.restore(-1), "a negative version must be rejected")
    expect_value_error(lambda: CheckpointStore(block_size=0), "block size 0 is invalid")
    expect_value_error(lambda: CheckpointStore().save({"w": 3.0}), "a tensor must be a list")

    # 8. determinism and the storage report.
    twin = CheckpointStore(block_size=8)
    for state in versions:
        twin.save(state)
    check(twin.payloads == store.payloads, "identical input must produce identical bytes")
    check(twin.restore(5) == store.restore(5), "identical input must restore identically")
    ratio = store.stored_bytes(5) / (6 * store.full_bytes(0))
    check(0.0 < ratio < 0.5, "incremental storage must be well under six full checkpoints")

    print(f"721 base={base_stored}B deltas={deltas}B raw_first={store.raw_sizes[0]}B")
    print(f"721 stored(5)={store.stored_bytes(5)}B full={store.full_bytes(0)}B ratio={ratio:.3f}")
    print("PASS")


if __name__ == "__main__":
    main()
