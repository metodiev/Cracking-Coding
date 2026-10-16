"""Task 914 - Implement cross-language optimized RPC with zero-copy serialization and backpressure.
(Tier: Master, CTCI Beyond - Software Engineering & Tooling)

What is implemented (in-process, deterministic; no sockets, no threads, the network is an
explicitly driven in-memory schedule):

  * `encode_frame` - a length-prefixed little-endian wire format: a varint message count, then per
    message a varint field count, per field a varint byte length and the raw bytes.  Everything is
    fixed-width or varint, so any language with a byte buffer can read it.
  * `decode_frame` - a *zero-copy* reader: it slices the received buffer into memoryviews and
    builds a metadata "tablet" of (offset, length) pairs once, so fields are interpreted lazily
    and no payload bytes are copied.  `decode_struct` is an independent second reader written
    against the same wire spec with `struct` unpacking - the cross-language oracle.
  * `CreditWindow` - flow control: a sender may only transmit while in-flight messages are below
    the receiver's advertised window; acknowledgements return credit.  `send_with_backpressure`
    drives the schedule and checks the invariant after every send and ack.
  * `batch` - coalesces small messages into frames that respect the MTU.

Simplified versus production: one process, one transport link, no compression, no timeouts or
retransmission, and the "second language" is a second reader rather than another runtime.

Verified below: golden byte vectors for the format, varint boundaries, round-trip equality,
zero-copy identity of the decoded views, agreement with the independent struct reader, the
in-flight window invariant under a driven schedule, MTU batching, and the empty / single-byte /
zero-window boundaries.
"""

import struct

MAGIC = 0x9140
MTU = 64


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def put_varint(out, value):
    """Little-endian base-128 varint, 7 bits per byte with a continuation flag."""
    check(value >= 0, "varints are unsigned")
    while True:
        chunk = value & 0x7F
        value >>= 7
        if value:
            out.append(chunk | 0x80)
        else:
            out.append(chunk)
            return


def get_varint(buffer, offset):
    """Returns (value, next_offset); the buffer is indexed but never copied."""
    value, shift = 0, 0
    while True:
        byte = buffer[offset]
        value |= (byte & 0x7F) << shift
        offset += 1
        if not byte & 0x80:
            return value, offset
        shift += 7
        check(shift < 70, "a varint may not exceed ten bytes")


def encode_frame(messages):
    """messages: iterable of lists of byte fields."""
    body = bytearray()
    messages = list(messages)
    put_varint(body, len(messages))
    for fields in messages:
        put_varint(body, len(fields))
        for field in fields:
            payload = bytes(field)
            put_varint(body, len(payload))
            body += payload
    header = bytearray()
    header += struct.pack("<HI", MAGIC, len(body))
    return bytes(header) + bytes(body)


def decode_frame(data):
    """Zero-copy decode: returns (tablet, payload) where tablet[i] is a list of (offset, length)
    pairs into `data` constructed once, and payload is a memoryview of the same object."""
    view = memoryview(data)
    check(len(view) >= 6, "a frame needs at least the magic and the length")
    magic, length = struct.unpack_from("<HI", view, 0)
    check(magic == MAGIC, "bad magic: 0x%04x" % magic)
    check(6 + length == len(view), "the frame length %d does not match the buffer" % length)
    cursor = 6
    count, cursor = get_varint(view, cursor)
    tablet = []
    for _ in range(count):
        fields, cursor = get_varint(view, cursor)
        row = []
        for _ in range(fields):
            size, cursor = get_varint(view, cursor)
            row.append((cursor, size))
            cursor += size
        tablet.append(row)
    check(cursor == len(view), "trailing bytes after the last field")
    return tablet, view


def read_field(tablet, view, message, field):
    offset, size = tablet[message][field]
    return view[offset:offset + size]


def decode_struct(data):
    """Independent reader for the same spec, built on struct/bytes (the cross-language oracle)."""
    magic, length = struct.unpack_from("<HI", data, 0)
    check(magic == MAGIC and 6 + length == len(data), "the struct reader rejects the frame")
    cursor = 6

    def take():
        nonlocal cursor
        value, shift = 0, 0
        while True:
            byte = data[cursor]
            cursor += 1
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
            shift += 7

    messages = []
    for _ in range(take()):
        row = []
        for _ in range(take()):
            size = take()
            row.append(data[cursor:cursor + size])
            cursor += size
        messages.append(row)
    check(cursor == len(data), "the struct reader must consume the frame exactly")
    return messages


class CreditWindow:
    """Receiver-advertised window: in-flight messages may never exceed `window`."""

    def __init__(self, window):
        self.window = window
        self.inflight = 0
        self.sent = 0
        self.stalled = 0

    def can_send(self):
        return self.inflight < self.window

    def send(self):
        check(self.can_send(), "backpressure violated: %d in flight of %d" % (self.inflight, self.window))
        self.inflight += 1
        self.sent += 1
        return self.inflight

    def ack(self, count=1):
        take = min(count, self.inflight)
        self.inflight -= take
        check(self.inflight >= 0, "credits cannot go negative")


def send_with_backpressure(window, messages):
    """Drive the schedule: send while credit remains, ack one message per step."""
    delivered = []
    index = 0
    while index < len(messages) or window.inflight > 0:
        check(window.inflight <= window.window, "the in-flight count must respect the window")
        if index < len(messages) and window.can_send():
            window.send()
            delivered.append(messages[index])
            index += 1
        elif window.inflight > 0:
            window.ack()
        else:
            window.stalled += 1
            break
        check(window.inflight <= window.window, "the window invariant must survive every step")
    return delivered


def batch(messages, mtu=MTU):
    """Coalesce messages into frames that fit the MTU; return (frames, rejected)."""
    frames, current, rejected = [], [], []
    for fields in messages:
        probe = encode_frame(current + [fields])
        if len(probe) - 6 <= mtu:
            current.append(fields)
        elif current:
            frames.append(current)
            current = [fields]
            check(len(encode_frame(current)) - 6 <= mtu, "a single message must fit the MTU")
        else:
            rejected.append(fields)
    if current:
        frames.append(current)
    return frames, rejected


def main():
    check(struct.calcsize("<HI") == 6, "the header must be six bytes")

    # ---- 1. varint boundaries against the byte-length table
    out = bytearray()
    for value, expected in ((0, 1), (127, 1), (128, 2), (16383, 2), (16384, 3), (1 << 21, 4)):
        out.clear()
        put_varint(out, value)
        check(len(out) == expected, "varint(%d) must be %d bytes, got %d" % (value, expected, len(out)))
        view = memoryview(bytes(out))
        check(get_varint(view, 0) == (value, expected), "varint round-trip for %d" % value)
    out = bytearray()
    put_varint(out, 300)
    check(bytes(out) == b"\xac\x02", "varint(300) must be 0xac 0x02 (golden vector)")

    # ---- 2. golden frame vector
    golden = encode_frame([[b"hi"], [b"a", b"bc"]])
    body = bytes([2, 1, 2]) + b"hi" + bytes([2, 1]) + b"a" + bytes([2]) + b"bc"
    check(golden == struct.pack("<HI", MAGIC, len(body)) + body,
          "the frame layout must match the hand-written golden vector")

    # ---- 3. round-trip and zero-copy
    messages = [[b"hello"], [b"a", b"b" * 40], [], [b"unicode-\xe2\x9c\x93", bytes(range(10))]]
    frame = encode_frame(messages)
    tablet, view = decode_frame(frame)
    check(len(tablet) == len(messages), "every message must appear in the tablet")
    for index, fields in enumerate(messages):
        check(len(tablet[index]) == len(fields), "field count mismatch in message %d" % index)
        for slot, field in enumerate(fields):
            check(bytes(read_field(tablet, view, index, slot)) == field,
                  "field (%d, %d) must round-trip" % (index, slot))
    check(view.obj is frame, "the decoded payload must be a view of the received buffer")
    check(read_field(tablet, view, 1, 1).obj is frame, "fields must be views, not copies")
    before = len(frame)
    for index, fields in enumerate(messages):
        for slot in range(len(fields)):
            read_field(tablet, view, index, slot)
    check(len(frame) == before, "reading fields must not copy the buffer")
    check(decode_struct(frame) == [list(fields) for fields in messages],
          "the independent struct reader must agree with the zero-copy reader")

    # ---- 4. the zero-copy reader rejects malformed frames
    for broken, why in ((frame[:-1], "a truncated frame"),
                        (frame[:6] + b"\x03" + frame[7:], "a lying message count"),
                        (struct.pack("<HI", 0x1234, len(frame) - 6) + frame[6:], "a foreign magic")):
        try:
            decode_frame(broken)
            raise AssertionError("%s must be rejected" % why)
        except (AssertionError, IndexError, struct.error) as error:
            check(bool(str(error)), "the rejection must be reported")

    # ---- 5. backpressure: the window invariant under a driven schedule
    window = CreditWindow(3)
    delivered = send_with_backpressure(window, messages * 4)
    check(len(delivered) == len(messages) * 4, "every message must eventually be delivered")
    check(window.sent == len(messages) * 4 and window.inflight == 0,
          "the schedule must drain to zero in flight")
    check(window.inflight <= window.window, "the window must never be exceeded")
    heavy = CreditWindow(2)
    check(send_with_backpressure(heavy, messages * 3) == messages * 3,
          "a small window must still deliver everything, just slower")
    frozen = CreditWindow(0)
    check(not frozen.can_send() and frozen.stalled == 0, "a zero window offers no credit")
    try:
        frozen.send()
        raise AssertionError("sending without credit must be rejected")
    except AssertionError as error:
        check("backpressure" in str(error), "the rejection must name backpressure")
    frozen.sent = 0
    stalled = send_with_backpressure(CreditWindow(0), messages)
    check(stalled == [], "a zero window must stall the whole schedule")
    one = CreditWindow(1)
    check(send_with_backpressure(one, messages[:1]) == messages[:1] and one.inflight == 0,
          "a single message must pass through a window of one")

    # ---- 6. batching respects the MTU
    frames, rejected = batch([[b"x" * 10] * 1 for _ in range(9)], mtu=32)
    check(all(len(encode_frame(frame)) - 6 <= 32 for frame in frames),
          "every frame must fit the MTU")
    check(sum(len(frame) for frame in frames) == 9, "batching may not drop or duplicate messages")
    check(rejected == [], "small messages must all fit")
    oversized, refused = batch([[b"y" * 100]], mtu=32)
    check(oversized == [] and len(refused) == 1, "an oversized message must be reported, not split")
    empty_frames, empty_refused = batch([], mtu=32)
    check(empty_frames == [] and empty_refused == [], "batching nothing must produce nothing")

    # ---- 7. boundaries
    blank = encode_frame([])
    blank_tablet, blank_view = decode_frame(blank)
    check(blank_tablet == [] and len(blank_view) == 7, "an empty frame is header plus one varint")
    single = encode_frame([[b"\x00"]])
    single_tablet, single_view = decode_frame(single)
    check(bytes(read_field(single_tablet, single_view, 0, 0)) == b"\x00",
          "a single zero byte must round-trip")
    wide = encode_frame([[]])
    wide_tablet, _ = decode_frame(wide)
    check(wide_tablet == [[]], "a message with no fields must be representable")
    check(decode_struct(encode_frame([[]])) == [[]], "the struct reader agrees on empty messages")

    # ---- 8. determinism
    check(encode_frame(messages) == frame, "encoding must be reproducible")
    check(decode_struct(frame) == decode_struct(encode_frame(messages)),
          "decoding must be reproducible")

    print("frame %d bytes, tablet %r" % (len(frame), tablet[:2]))
    print("window %d: sent %d, delivered %d" % (window.window, window.sent, len(delivered)))
    print("golden body %d bytes, independent reader agrees" % len(body))
    print("PASS")


if __name__ == "__main__":
    main()
