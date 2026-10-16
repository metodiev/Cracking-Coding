/** Task 914 - Implement cross-language optimized RPC with zero-copy serialization and backpressure.
 *  (Tier: Master, CTCI Beyond — Distributed Systems & Wire Protocols)
 *
 * A length-prefixed binary framing layer with a hand-written codec: every frame is
 * [u8 type][u32le length][payload], and decoding returns Buffer *views* over the caller's frame (Node's
 * subarray shares the ArrayBuffer), so a decoded message costs no copy — verified by mutating the frame
 * and watching the view change, and by checking the shared buffer identity. The transport on top is
 * credit-based: a receiver grants window credits, the sender may keep at most `window` messages in flight,
 * and when the receiver stalls the sender queues up to `queueLimit` and then refuses rather than
 * dropping, which is the backpressure contract the checks enforce. A byte-stream receiver reassembles
 * frames from arbitrarily fragmented chunks.
 *
 * Checks: round-trip fidelity against a JSON oracle for every message shape, byte-stable re-encoding,
 * the zero-copy property, the 5-byte framing overhead, credit accounting under a stalled receiver and a
 * subsequent resume, one-byte-at-a-time reassembly, degenerate payloads (empty, huge, truncated,
 * oversize) and determinism.
 */
"use strict";
const assert = require("node:assert");
const HEADER_BYTES = 5;
const MAX_FRAME = 1 << 20;
const TYPES = { number: 0, string: 1, numbers: 2, record: 3 };

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.state = x;
    return x;
  }
  float() { return this.next() / 4294967296; }
  int(bound) { return this.next() % bound; }
}
/** Encode a message into a fresh frame: header + payload, nothing else. */
function encode(message) {
  let type;
  let payload;
  if (typeof message === "number") {
    type = TYPES.number;
    payload = Buffer.alloc(8);
    payload.writeDoubleLE(message);
  } else if (typeof message === "string") {
    type = TYPES.string;
    payload = Buffer.from(message, "utf8");
  } else if (Array.isArray(message)) {
    type = TYPES.numbers;
    payload = Buffer.alloc(message.length * 8);
    message.forEach((value, i) => payload.writeDoubleLE(value, i * 8));
  } else if (message && typeof message === "object") {
    type = TYPES.record;
    const keyPayload = Buffer.from(String(message.key), "utf8");
    const head = Buffer.alloc(4);
    head.writeUInt32LE(keyPayload.length);
    const tail = Buffer.alloc(8);
    tail.writeDoubleLE(Number(message.value));
    payload = Buffer.concat([head, keyPayload, tail]);
  } else {
    throw new Error(`unsupported message ${JSON.stringify(message)}`);
  }
  if (payload.length > MAX_FRAME) throw new Error(`frame of ${payload.length} bytes exceeds the limit`);
  const frame = Buffer.alloc(HEADER_BYTES + payload.length);
  frame.writeUInt8(type, 0);
  frame.writeUInt32LE(payload.length, 1);
  payload.copy(frame, HEADER_BYTES);
  return frame;
}
/** Decode one frame; the payload is a view over `frame`, never a copy. */
function decode(frame) {
  if (frame.length < HEADER_BYTES) return { need: HEADER_BYTES - frame.length };
  const type = frame.readUInt8(0);
  const length = frame.readUInt32LE(1);
  if (length > MAX_FRAME) throw new Error(`frame of ${length} bytes exceeds the limit`);
  if (frame.length < HEADER_BYTES + length) return { need: HEADER_BYTES + length - frame.length };
  const payload = frame.subarray(HEADER_BYTES, HEADER_BYTES + length);
  let message;
  if (type === TYPES.number) message = payload.readDoubleLE(0);
  else if (type === TYPES.string) message = payload.toString("utf8");
  else if (type === TYPES.numbers) {
    message = Array.from({ length: payload.length / 8 }, (_, i) => payload.readDoubleLE(i * 8));
  } else if (type === TYPES.record) {
    const keyLength = payload.readUInt32LE(0);
    message = { key: payload.subarray(4, 4 + keyLength).toString("utf8"), value: payload.readDoubleLE(4 + keyLength) };
  } else {
    throw new Error(`unknown frame type ${type}`);
  }
  return { message, payload, consumed: HEADER_BYTES + length };
}
/** Reassembles frames from a chunked byte stream (a socket read returns arbitrary boundaries). */
class FrameReader {
  constructor() { this.pending = Buffer.alloc(0); this.frames = 0; }
  push(chunk) {
    this.pending = this.pending.length === 0 ? Buffer.from(chunk) : Buffer.concat([this.pending, chunk]);
    const messages = [];
    let offset = 0;
    while (offset < this.pending.length) {
      const result = decode(this.pending.subarray(offset));
      if (result.need !== undefined) break;
      messages.push(result.message);
      offset += result.consumed;
      this.frames += 1;
    }
    this.pending = this.pending.subarray(offset);
    return messages;
  }
}
/** Credit-based sender: at most `window` unacknowledged messages, bounded queue, no drops. */
class CreditSender {
  constructor(window, queueLimit) {
    this.window = window;
    this.queueLimit = queueLimit;
    this.queue = [];
    this.inFlight = [];
    this.credits = window;
    this.refused = 0;
    this.sent = 0;
  }
  send(message) {
    this.queue.push(message);
    const promoted = this.pump();
    if (this.queue.length > this.queueLimit) {
      this.queue.pop();
      this.refused += 1;
      throw new Error("backpressure: the queue limit was reached");
    }
    return promoted;
  }
  /** Move queued messages into the window while credits last. */
  pump() {
    const promoted = [];
    while (this.credits > 0 && this.queue.length > 0) {
      const next = this.queue.shift();
      this.credits -= 1;
      this.inFlight.push(next);
      this.sent += 1;
      promoted.push(next);
    }
    return promoted;
  }
  acknowledge(count) {
    const released = this.inFlight.splice(0, count);
    this.credits = Math.min(this.window, this.credits + released.length);
    this.pump();                                  // an acknowledgement immediately refills the window
    return released;
  }
  /** Move queued messages into the window while credits last. */
  pump() {
    const promoted = [];
    while (this.credits > 0 && this.queue.length > 0) {
      const next = this.queue.shift();
      this.credits -= 1;
      this.inFlight.push(next);
      this.sent += 1;
      promoted.push(next);
    }
    return promoted;
  }
  acknowledge(count) {
    const released = this.inFlight.splice(0, count);
    this.credits = Math.min(this.window, this.credits + released.length);
    this.pump();                                  // an acknowledgement immediately refills the window
    return released;
  }
}
const roundTrip = (message) => decode(encode(message)).message;

function main() {
  const prng = new Prng(0x5eed0914);
  // 1. round-trip fidelity — every shape survives encoding, checked against a JSON oracle of the value.
  {
    const messages = [
      0, -0.0 + 0, 3.141592653589793, -1234.5, Number.EPSILON, 1e12, "", "hello", "unicode ✓ ünïcödé 漢字", "x".repeat(5000), [], [1, 2, 3], Array.from({ length: 100 }, (_, i) => i * 0.5 - 25), { key: "user-1", value: 42.5 }, { key: "", value: -1 }, ];
    for (const message of messages) {
      const decoded = roundTrip(message);
      assert.deepStrictEqual(decoded, message, `round trip of ${JSON.stringify(message).slice(0, 40)}`);
      assert.deepStrictEqual(JSON.parse(JSON.stringify(decoded)), JSON.parse(JSON.stringify(message)), "the decoded value matches the JSON oracle");
    }
    for (let i = 0; i < 200; i++) {
      const message = prng.float() * 1e6 - 5e5;
      assert.strictEqual(roundTrip(message), message, "random doubles are exact");
    }
  }
  // 2. the codec is byte-stable and the framing overhead is exactly the header size.
  {
    const message = { key: "alpha", value: 7.25 };
    const frame = encode(message);
    assert.strictEqual(frame.length - HEADER_BYTES - frame.readUInt32LE(1), 0, "no trailing bytes");
    assert.deepStrictEqual(encode(roundTrip(message)), frame, "re-encoding is byte-stable");
    for (const sample of [1, "abc", [1.5, 2.5], { key: "k", value: 9 }]) {
      const bytes = encode(sample).length - Buffer.byteLength(JSON.stringify(sample), "utf8");
      assert.ok(encode(sample).length >= HEADER_BYTES, "every frame carries the header");
      assert.ok(bytes < 64, `framing overhead ${bytes} for ${JSON.stringify(sample)}`);
    }
    const stream = Buffer.concat([encode(1), encode("two"), encode([3, 4])]);
    const reader = new FrameReader();
    assert.deepStrictEqual(reader.push(stream), [1, "two", [3, 4]], "a concatenated stream decodes in order");
    assert.strictEqual(reader.frames, 3);
    assert.strictEqual(reader.pending.length, 0, "everything was consumed");
  }
  // 3. zero copy — the decoded payload is a view, so in-place edits are visible without re-decoding.
  {
    const frame = encode([1.5, 2.5, 3.5]);
    const { payload, message } = decode(frame);
    assert.strictEqual(payload.buffer, frame.buffer, "the payload shares the frame's ArrayBuffer");
    assert.ok(payload.byteOffset >= frame.byteOffset, "the payload is an offset into the frame");
    assert.notStrictEqual(payload.byteOffset, frame.byteOffset, "the header is skipped, not copied");
    const before = message.slice();
    payload.writeDoubleLE(99.5, 0);
    const reread = decode(frame).message;
    assert.strictEqual(reread[0], 99.5, "writing through the view changes the frame's bytes");
    assert.deepStrictEqual(reread.slice(1), before.slice(1), "only the touched slot changed");
    const stringFrame = encode("copy me");
    const view = decode(stringFrame).payload;
    assert.strictEqual(view.buffer, stringFrame.buffer, "string payloads are views as well");
    assert.strictEqual(decode(stringFrame).message, "copy me");
  }
  // 4. backpressure — the sender never exceeds its window, queues under a stalled receiver, resumes.
  {
    const sender = new CreditSender(3, 10);
    const delivered = [];
    const total = 25;
    let stalled = true;
    for (let i = 0; i < total; i++) {
      try {
        const promoted = sender.send(`message-${i}`);
        if (!stalled) {
          for (const message of sender.acknowledge(promoted.length)) delivered.push(message);
        }
        assert.ok(sender.inFlight.length <= sender.window, "in flight must stay inside the window");
      } catch (error) {
        assert.match(error.message, /backpressure/, `unexpected error at ${i}`);
        assert.ok(sender.queue.length <= sender.queueLimit, "the queue stays bounded under pressure");
        assert.strictEqual(sender.refused, 1, "the refusal is counted");
        break;
      }
    }
    assert.ok(sender.refused === 1, "a stalled receiver eventually forces a refusal");
    assert.ok(sender.inFlight.length === 3, "exactly the window is in flight while stalled");
    stalled = false;
    let guard = 0;
    while ((sender.inFlight.length > 0 || sender.queue.length > 0) && guard < 1000) {
      delivered.push(...sender.acknowledge(1));
      guard += 1;
    }
    assert.ok(guard < 1000, "the transfer must drain once the receiver resumes");
    assert.ok(delivered.every((message) => typeof message === "string"), "everything delivered is a string");
    assert.ok(delivered.indexOf("message-0") < delivered.indexOf("message-1"), "order is preserved");
    assert.strictEqual(sender.credits, sender.window, "credits are fully replenished at the end");
    assert.strictEqual(sender.inFlight.length, 0, "nothing is left in flight");
    assert.strictEqual(sender.queue.length, 0, "the queue drained completely");
    assert.ok(!stalled && delivered.length >= 3, "the receiver saw real traffic");
  }
  // 5. fragmentation — one byte at a time reassembles exactly the same messages.
  {
    const messages = [7, "fragmented payload", [1, 2, 3], { key: "frag", value: 13.5 }];
    const stream = Buffer.concat(messages.map(encode));
    const reader = new FrameReader();
    const seen = [];
    for (const byte of stream) seen.push(...reader.push(Buffer.from([byte])));
    assert.deepStrictEqual(seen, messages, "byte-wise reassembly is exact");
    assert.strictEqual(reader.pending.length, 0);
    const partial = stream.subarray(0, stream.length - 3);
    const partialReader = new FrameReader();
    const complete = partialReader.push(Buffer.from(partial));
    assert.deepStrictEqual(complete, messages.slice(0, 3), "only complete frames are released");
    assert.ok(partialReader.pending.length > 0, "the tail is retained for the next chunk");
    assert.deepStrictEqual(partialReader.push(Buffer.from(stream.subarray(stream.length - 3))), [messages[3]], "the remainder completes the last frame");
  }
  // 6. boundary — empty payloads, truncated headers, an unknown type and the size limit.
  {
    assert.strictEqual(roundTrip(""), "", "an empty string round trips");
    assert.deepStrictEqual(roundTrip([]), [], "an empty list round trips");
    assert.deepStrictEqual(decode(Buffer.alloc(2)), { need: 3 }, "a partial header asks for more bytes");
    const headerOnly = Buffer.alloc(HEADER_BYTES);
    headerOnly.writeUInt8(1, 0);
    headerOnly.writeUInt32LE(4, 1);
    assert.deepStrictEqual(decode(headerOnly), { need: 4 }, "a header without a payload asks for more");
    const bogus = Buffer.alloc(HEADER_BYTES);
    bogus.writeUInt8(9, 0);
    bogus.writeUInt32LE(0, 1);
    assert.throws(() => decode(bogus), /unknown frame type/);
    assert.throws(() => encode(undefined), /unsupported message/);
    assert.throws(() => encode(null), /unsupported message/);
    const oversize = Buffer.alloc(HEADER_BYTES);
    oversize.writeUInt8(1, 0);
    oversize.writeUInt32LE(MAX_FRAME + 1, 1);
    assert.throws(() => decode(oversize), /exceeds the limit/);
    assert.throws(() => encode("x".repeat(MAX_FRAME + 1)), /exceeds the limit/);
    const huge = Array.from({ length: 20000 }, (_, i) => i / 3);
    assert.deepStrictEqual(roundTrip(huge), huge, "a large message still round trips");
  }
  // 7. credit accounting — credits are conserved and one message is one credit.
  {
    const sender = new CreditSender(2, 5);
    assert.deepStrictEqual(sender.send("a"), ["a"]);
    assert.deepStrictEqual(sender.send("b"), ["b"]);
    assert.deepStrictEqual(sender.send("c"), [], "the window is full, so it queues");
    assert.strictEqual(sender.inFlight.length, 2);
    assert.strictEqual(sender.queue.length, 1);
    assert.deepStrictEqual(sender.acknowledge(1), ["a"]);
    assert.deepStrictEqual(sender.inFlight, ["b", "c"], "the freed credit pulled the queued message in");
    assert.deepStrictEqual(sender.send("d"), [], "the window refilled, so d waits");
    assert.deepStrictEqual(sender.acknowledge(5), ["b", "c"], "over-acknowledging releases what exists");
    assert.deepStrictEqual(sender.inFlight, ["d"], "the last credit promotes d");
    assert.strictEqual(sender.credits, 1, "credits never exceed the window");
    assert.strictEqual(sender.queue.length, 0);
  }
  // 8. determinism — the same messages produce the same bytes, and the payload generator is seeded.
  {
    const build = () => {
      const local = new Prng(99);
      return Array.from({ length: 50 }, () => Array.from({ length: 4 }, () => local.float()));
    };
    assert.deepStrictEqual(build(), build());
    assert.deepStrictEqual(encode(build()[0]), encode(build()[0]));
    assert.notDeepStrictEqual(encode([1]), encode([2]));
    const first = new Prng(1).float();
    assert.strictEqual(first, new Prng(1).float());
  }
  console.log(`rpc: ${Object.keys(TYPES).length} message shapes, ${HEADER_BYTES}-byte header, payload views share ` + `the frame buffer; credit window 3 stalled after ${3} in-flight messages with 1 refusal and full ` + `recovery; 1-byte-at-a-time reassembly of 4 frames verified`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = {
  HEADER_BYTES, MAX_FRAME, TYPES, Prng, encode, decode, FrameReader, CreditSender, roundTrip, };
