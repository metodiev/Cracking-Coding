/** Task 721 - Implement checkpoint compression and incremental checkpointing. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * A training checkpoint is a named set of tensors (here integer arrays, which keeps every
 * round-trip bit-exact). Writing all of them every step costs O(model) bytes per step, so
 * production checkpointers (DeepSpeed/Torch distributed checkpointing, ZeRO-Offload, `-i`
 * incremental archives) instead write one *base* checkpoint and then one *delta* per step
 * that only enumerates what moved.
 *
 * Wire format (little-endian, all integers zigzag varints):
 *   full  : "CKP1" | tensorCount | (nameLen, name, count, values...) * | checksum(4)
 *   delta : "DLT1" | tensorCount | (nameLen, name, kind, payload) * | checksum(4)
 *           kind 0 untouched (no payload), kind 1 sparse (changedCount, then
 *           index-gap + value-delta pairs), kind 2 full rewrite of that tensor.
 * A checksum (FNV-1a over every preceding byte) makes silent corruption detectable, and a
 * per-tensor policy rewrites a tensor in full when more than half of its entries moved,
 * because at that point a sparse payload would be larger than the plain one.
 *
 * Invariants verified here: decode(encode(state)) equals the state exactly (including
 * extreme int32 values); a delta writes exactly the entries that changed and `kind 0` for
 * tensors that did not; base + deltas replayed in order reproduces every intermediate state;
 * the compaction policy matches an independently recomputed diff; the incremental chain
 * costs far fewer bytes than writing full checkpoints; varint/zigzag boundaries; empty and
 * zero-length tensors; and checksum/truncation detection for tampered payloads.
 * Simplified: no filesystem I/O (buffers in memory), no cross-process sharding, and the
 * "model" is a small deterministic integer parameter set.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual, throws } = require("node:assert");

/** Deterministic splitmix64 -> [0, 1). */
function makeRng(seed) {
  let state = (BigInt(seed >>> 0) + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
  return function next() {
    state = (state + 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & 0xffffffffffffffffn;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & 0xffffffffffffffffn;
    z ^= z >> 31n;
    return Number(z >> 11n) / 9007199254740992;
  };
}

const FULL_MAGIC = "CKP1";
const DELTA_MAGIC = "DLT1";
const CHECKSUM_BYTES = 4;
const REWRITE_NUMERATOR = 1;                  // rewrite when changed / count > REWRITE_RATIO
const REWRITE_RATIO = 2;

class ByteWriter {
  constructor() {
    this.bytes = [];
  }

  raw(byte) {
    this.bytes.push(byte & 0xff);
    return this;
  }

  text(value) {
    for (const byte of Buffer.from(value, "utf8")) this.raw(byte);
    return this;
  }

  /** Unsigned LEB128. */
  varint(value) {
    let rest = value;
    do {
      const chunk = rest % 128;
      rest = Math.floor(rest / 128);
      this.raw(chunk | (rest > 0 ? 0x80 : 0));
    } while (rest > 0);
    return this;
  }

  /** Zigzag maps signed integers onto the unsigned varint domain. */
  signed(value) {
    return this.varint(value >= 0 ? 2 * value : -2 * value - 1);
  }

  checksum() {
    const hash = fnv1a(this.bytes);
    for (let i = 0; i < CHECKSUM_BYTES; i++) this.raw((hash >>> (8 * i)) & 0xff);
    return this;
  }

  toBuffer() {
    return Buffer.from(this.bytes);
  }
}

class ByteReader {
  constructor(buffer) {
    this.buffer = buffer;
    this.position = 0;
  }

  get remaining() {
    return this.buffer.length - this.position;
  }

  raw() {
    if (this.position >= this.buffer.length) throw new Error("checkpoint truncated");
    return this.buffer[this.position++];
  }

  varint() {
    let shift = 1;
    let value = 0;
    for (;;) {
      const byte = this.raw();
      value += (byte & 0x7f) * shift;
      if ((byte & 0x80) === 0) return value;
      shift *= 128;
    }
  }

  signed() {
    const value = this.varint();
    return value % 2 === 0 ? value / 2 : -(value + 1) / 2;
  }

  text(length) {
    const slice = this.buffer.subarray(this.position, this.position + length);
    if (slice.length !== length) throw new Error("checkpoint truncated");
    this.position += length;
    return slice.toString("utf8");
  }
}

/** FNV-1a over the bytes written so far. */
function fnv1a(bytes) {
  let hash = 0x811c9dc5;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash;
}

const stateOf = (tensors) => ({ tensors: tensors.map((t) => ({ name: t.name, values: t.values.slice() })) });

/** Full checkpoint of a whole state. */
function encodeFull(state) {
  const writer = new ByteWriter();
  writer.text(FULL_MAGIC).varint(state.tensors.length);
  for (const tensor of state.tensors) {
    writer.varint(Buffer.byteLength(tensor.name, "utf8")).text(tensor.name).varint(tensor.values.length);
    for (const value of tensor.values) writer.signed(value);
  }
  writer.checksum();
  return writer.toBuffer();
}

function readHeader(reader, magic, expectedTensors) {
  strictEqual(reader.text(4), magic, `bad magic, expected ${magic}`);
  strictEqual(reader.varint(), expectedTensors.length, "checkpoint tensor count mismatch");
}

/** Split a checkpoint into body (everything the checksum covers) and verify the checksum. */
function verifyChecksum(bytes) {
  ok(bytes.length > CHECKSUM_BYTES, "checkpoint is too short to hold a checksum");
  const body = bytes.subarray(0, bytes.length - CHECKSUM_BYTES);
  let recorded = 0;
  for (let i = 0; i < CHECKSUM_BYTES; i++) {
    recorded += bytes[bytes.length - CHECKSUM_BYTES + i] * 2 ** (8 * i);
  }
  const actual = fnv1a(body);
  ok(actual === recorded, `checksum mismatch: payload hashes to ${actual}, header says ${recorded}`);
  return body;
}

function decodeFull(buffer, template) {
  const body = verifyChecksum(buffer);
  const reader = new ByteReader(body);
  readHeader(reader, FULL_MAGIC, template.tensors);
  const out = stateOf(template.tensors);
  out.tensors.forEach((tensor) => {
    const name = reader.text(reader.varint());
    strictEqual(name, tensor.name, "tensor name mismatch: the checkpoint does not fit this state");
    const count = reader.varint();
    const values = new Array(count);
    for (let i = 0; i < count; i++) values[i] = reader.signed();
    tensor.values = values;
  });
  return out;
}

/** Diff two states: which entries of which tensors moved. */
function diffStates(base, next) {
  strictEqual(base.tensors.length, next.tensors.length, "states must have the same tensors");
  return base.tensors.map((tensor, index) => {
    strictEqual(tensor.name, next.tensors[index].name, "states must have the same tensor names");
    const changed = [];
    for (let i = 0; i < tensor.values.length; i++) {
      if (tensor.values[i] !== next.tensors[index].values[i]) changed.push(i);
    }
    return changed;
  });
}

/** Delta between two states; `kind 2` rewrites a tensor whose sparse payload would be larger. */
function encodeDelta(base, next) {
  const changed = diffStates(base, next);
  const writer = new ByteWriter();
  writer.text(DELTA_MAGIC).varint(base.tensors.length);
  base.tensors.forEach((tensor, index) => {
    const count = tensor.values.length;
    const moved = changed[index];
    writer.varint(Buffer.byteLength(tensor.name, "utf8")).text(tensor.name);
    if (moved.length === 0) {
      writer.raw(0);
    } else if (moved.length * REWRITE_RATIO > count * REWRITE_NUMERATOR) {
      writer.raw(2).varint(count);
      for (const value of next.tensors[index].values) writer.signed(value);
    } else {
      writer.raw(1).varint(moved.length);
      let previous = 0;
      for (const position of moved) {
        writer.varint(position - previous);
        writer.signed(next.tensors[index].values[position] - tensor.values[position]);
        previous = position + 1;
      }
    }
  });
  writer.checksum();
  return writer.toBuffer();
}

/** Apply a delta on top of `base`, returning a fresh state. */
function decodeDelta(base, buffer) {
  const body = verifyChecksum(buffer);
  const reader = new ByteReader(body);
  readHeader(reader, DELTA_MAGIC, base.tensors);
  const out = stateOf(base.tensors);
  base.tensors.forEach((tensor, index) => {
    const nameLength = reader.varint();
    const name = reader.text(nameLength);
    strictEqual(name, tensor.name, "tensor name mismatch: the delta does not fit this state");
    const kind = reader.raw();
    if (kind === 0) return;
    if (kind === 2) {
      const count = reader.varint();
      strictEqual(count, tensor.values.length, "full rewrite must keep the tensor shape");
      const values = new Array(count);
      for (let i = 0; i < count; i++) values[i] = reader.signed();
      out.tensors[index].values = values;
      return;
    }
    strictEqual(kind, 1, `unknown delta record kind ${kind}`);
    const moved = reader.varint();
    const values = tensor.values.slice();
    let cursor = 0;
    for (let k = 0; k < moved; k++) {
      cursor += reader.varint();
      ok(cursor < values.length, "delta index out of range");
      values[cursor] += reader.signed();
      cursor++;
    }
    out.tensors[index].values = values;
  });
  return out;
}

/** Walk the delta records without applying them: { name, kind, moved } per tensor. */
function describeDelta(base, buffer) {
  const reader = new ByteReader(verifyChecksum(buffer));
  readHeader(reader, DELTA_MAGIC, base.tensors);
  return base.tensors.map((tensor) => {
    const name = reader.text(reader.varint());
    strictEqual(name, tensor.name, "tensor name mismatch: the delta does not fit this state");
    const kind = reader.raw();
    let moved = 0;
    if (kind === 1) {
      moved = reader.varint();
      let cursor = 0;
      for (let k = 0; k < moved; k++) {
        cursor += reader.varint();
        reader.signed();
        cursor++;
      }
    } else if (kind === 2) {
      const count = reader.varint();
      moved = count;
      for (let i = 0; i < count; i++) reader.signed();
    }
    return { name, kind, moved };
  });
}

/** A tiny deterministic "model": int32 tensors of the given sizes. */
function makeState(rng, sizes) {
  return {
    tensors: sizes.map((size, index) => ({
      name: `layer${index}.weight`,
      values: Array.from({ length: size }, () => Math.floor(rng() * 2001) - 1000),
    })),
  };
}

const totalElements = (state) => state.tensors.reduce((acc, t) => acc + t.values.length, 0);

module.exports = {
  makeRng, ByteWriter, ByteReader, fnv1a, stateOf, encodeFull, decodeFull, encodeDelta,
  decodeDelta, describeDelta, verifyChecksum, diffStates, makeState, totalElements, REWRITE_RATIO,
};

if (require.main === module) {
  const rng = makeRng(20260721);

  // -- varint/zigzag boundaries ---------------------------------------------------------
  const boundaries = [0, 1, -1, 63, 64, -64, 127, 128, -128, 8191, -8192, 2 ** 31 - 1, -(2 ** 31)];
  for (const value of boundaries) {
    const writer = new ByteWriter().signed(value);
    const reader = new ByteReader(writer.toBuffer());
    strictEqual(reader.signed(), value, `zigzag round-trip failed for ${value}`);
    strictEqual(reader.remaining, 0, "the varint must consume exactly its own bytes");
  }
  strictEqual(new ByteWriter().varint(127).toBuffer().length, 1, "127 fits in one varint byte");
  strictEqual(new ByteWriter().varint(128).toBuffer().length, 2, "128 needs two varint bytes");
  strictEqual(new ByteWriter().signed(-1).toBuffer().length, 1, "zigzag keeps -1 in one byte");

  // -- full checkpoint round-trip and an exact byte count -------------------------------
  const tiny = stateOf([{ name: "w", values: [0, -1, 1] }]);
  const tinyBytes = encodeFull(tiny);
  strictEqual(tinyBytes.length, 4 + 1 + 1 + 1 + 1 + 3 + 4,
    "CKP1 | count | nameLen | name | count | 3 single-byte values | 4-byte checksum");
  deepStrictEqual(decodeFull(tinyBytes, tiny), tiny, "a full checkpoint must round-trip");

  const state = makeState(rng, [256, 128, 64, 32]);
  const full = encodeFull(state);
  deepStrictEqual(decodeFull(full, state), state, "full checkpoint round-trip must be exact");
  const extremes = stateOf([{ name: "edge", values: [2 ** 31 - 1, -(2 ** 31), 0] }]);
  deepStrictEqual(decodeFull(encodeFull(extremes), extremes), extremes, "int32 extremes must survive");
  console.log(`full checkpoint: ${totalElements(state)} elements in ${full.length} bytes`);

  // -- a delta writes exactly what changed ----------------------------------------------
  const stepped = stateOf(state.tensors);
  const moved = new Set([3, 7, 11, 200]);
  stepped.tensors[0].values = stepped.tensors[0].values.map((v, i) => (moved.has(i) ? v + 5 : v));
  stepped.tensors[1].values[0] -= 3;
  const delta = encodeDelta(state, stepped);
  deepStrictEqual(describeDelta(state, delta).map((r) => [r.moved, r.kind]),
    [[4, 1], [1, 1], [0, 0], [0, 0]],
    "the delta must flag exactly the moved entries and leave untouched tensors alone");
  deepStrictEqual(diffStates(state, stepped).map((c) => c.length), [4, 1, 0, 0],
    "independently recomputed diff must agree");
  deepStrictEqual(decodeDelta(state, delta), stepped, "delta replay must reproduce the state");
  ok(delta.length < full.length / 4, `sparse delta ${delta.length} must be much smaller than ${full.length}`);

  // -- the rewrite policy for a tensor that moved wholesale ------------------------------
  const wholesale = stateOf(state.tensors);
  wholesale.tensors[2].values = wholesale.tensors[2].values.map((v) => v + 1);
  const rewrite = encodeDelta(state, wholesale);
  deepStrictEqual(describeDelta(state, rewrite).map((r) => r.kind), [0, 0, 2, 0],
    "a fully moved tensor must be rewritten, not diffed");
  deepStrictEqual(decodeDelta(state, rewrite), wholesale, "a full rewrite record must replay too");
  const halfMoved = stateOf(state.tensors);
  halfMoved.tensors[0].values = halfMoved.tensors[0].values.map((v, i) => (i < 129 ? v + 1 : v));
  strictEqual(describeDelta(state, encodeDelta(state, halfMoved))[0].kind, 2,
    "just over half the entries moving flips the tensor to a rewrite");

  // -- a no-change delta is nearly free and changes nothing ------------------------------
  const idle = encodeDelta(state, stateOf(state.tensors));
  deepStrictEqual(decodeDelta(state, idle), state, "an idle delta must leave the state alone");
  deepStrictEqual(describeDelta(state, idle).map((r) => r.kind), [0, 0, 0, 0], "nothing moved");
  ok(idle.length < full.length / 10, `idle delta ${idle.length} bytes vs full ${full.length}`);
  console.log(`sparse delta ${delta.length} B, rewrite delta ${rewrite.length} B, idle ${idle.length} B`);

  // -- incremental chain: base + 40 deltas versus 41 full checkpoints --------------------
  const live = makeState(rng, [256, 128, 64, 32]);
  const baseBytes = encodeFull(live);
  let incrementalBytes = baseBytes.length;
  let fullBytes = baseBytes.length;
  let current = live;
  for (let step = 0; step < 40; step++) {
    const next = stateOf(current.tensors);
    const tensor = next.tensors[step % 4];                    // only one tensor trains per step
    for (let i = 0; i < tensor.values.length; i++) {
      if (i % 16 === step % 16) tensor.values[i] += (i % 7) - 3 + step;   // ~1/16 of the entries move
    }
    const encoded = encodeDelta(current, next);
    incrementalBytes += encoded.length;
    fullBytes += encodeFull(next).length;
    current = decodeDelta(current, encoded);                  // replay must equal the live state
    deepStrictEqual(current, next, `step ${step}: replay of the delta chain must match the state`);
    current = next;
  }
  ok(current.tensors.every((t, i) => t.values.some((v, j) => v !== live.tensors[i].values[j])),
    "the workload must actually move every tensor");
  ok(incrementalBytes * 5 < fullBytes,
    `incremental ${incrementalBytes} B must beat full checkpoints ${fullBytes} B by 5x`);
  deepStrictEqual(decodeFull(baseBytes, live), live, "the base checkpoint is intact");
  console.log(`40 steps: incremental=${incrementalBytes} B vs full=${fullBytes} B ` +
    `(ratio ${(fullBytes / incrementalBytes).toFixed(1)}x)`);

  // -- empty and zero-length tensors ----------------------------------------------------
  const emptyState = { tensors: [] };
  const emptyBytes = encodeFull(emptyState);
  deepStrictEqual(decodeFull(emptyBytes, emptyState), emptyState, "an empty checkpoint round-trips");
  strictEqual(emptyBytes.length, 4 + 1 + 4, "empty: magic + tensor count + checksum");
  strictEqual(encodeDelta(emptyState, emptyState).length, 4 + 1 + 4, "the idle empty delta is minimal");
  const hollow = stateOf([{ name: "none", values: [] }]);
  deepStrictEqual(decodeFull(encodeFull(hollow), hollow), hollow, "a zero-length tensor round-trips");
  deepStrictEqual(decodeDelta(hollow, encodeDelta(hollow, hollow)), hollow, "and it deltas cleanly");

  // -- corruption, truncation and mis-binding are detected -------------------------------
  const corrupted = Buffer.from(full);
  corrupted[6] ^= 0x01;
  throws(() => decodeFull(corrupted, state), /checksum mismatch/, "a flipped byte must be caught");
  throws(() => decodeFull(full.subarray(0, full.length - 7), state), /checksum|truncated/,
    "a truncated checkpoint must be caught");
  throws(() => decodeDelta(state, delta.subarray(0, delta.length - 6)), /checksum|truncated/,
    "a truncated delta must be caught");
  const foreign = stateOf(["other", "layer1.weight", "layer2.weight", "layer3.weight"]
    .map((name, i) => ({ name, values: new Array([256, 128, 64, 32][i]).fill(0) })));
  throws(() => decodeDelta(foreign, delta), /name mismatch/, "a delta must be bound to its tensors");
  const loud = stateOf(state.tensors.map((t) => ({ name: t.name, values: t.values.map((v) => v * 1000000) })));
  ok(encodeFull(loud).length > full.length * 2, "varints must make wide-range values cost more");

  // -- determinism ----------------------------------------------------------------------
  deepStrictEqual(encodeFull(state), encodeFull(state), "encoding must be deterministic");
  deepStrictEqual(encodeDelta(state, stepped), delta, "delta encoding must be deterministic");
  console.log("PASS");
}
