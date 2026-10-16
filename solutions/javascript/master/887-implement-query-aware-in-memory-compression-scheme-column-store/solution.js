/** Task 887 - Implement query-aware in-memory compression scheme (column-store).
 *  (Tier: Master, CTCI Ch 14 — Databases)
 *
 * A column store where every column is stored with one of four integer encodings and where the
 * *encoding is chosen per column from the observed workload*:
 *
 *   plain  — presence bitmap + bit-packed zigzag values (bits = max bit width),
 *   dict   — sorted dictionary with one code per distinct value, bit-packed codes, plus a code
 *            frequency histogram so count/sum/avg/min/max are answered from the dictionary alone,
 *   rle    — runs of equal entries (null runs included) as (value, length) pairs, so a predicate is
 *            evaluated once per run and the aggregate is weighted by the run length,
 *   delta  — presence bitmap + first value + bit-packed zigzag deltas with a checkpoint index every
 *            32 entries, which gives O(log n) random access for sorted columns (range predicates via
 *            binary search, no full decode).
 *
 * Query-aware planning: a workload is a list of (predicate, aggregate, weight) triples. The planner
 * enumerates every encoding assignment for the touched columns and picks the one minimising
 * Σ weight · (payload words read + values decoded), using the per-encoding cost rules above. Because
 * the number of candidate assignments is small the enumeration is exhaustive, so the chosen plan is
 * provably the best *according to the model*; the model itself is cross-checked against the cost the
 * real evaluator reports (words touched and values decoded are counted by the evaluator, not
 * assumed).
 *
 * Invariants verified below: (1) every encoding round-trips exactly, including nulls, empty columns,
 * negatives and constant columns; (2) every aggregate/predicate combination agrees with an
 * independent plain-array scan (the oracle) on all four encodings; (3) the planner's choice equals
 * the exhaustive optimum of the model and its *measured* cost is never worse than the plain layout;
 * (4) compressed-domain evaluation really saves work (a constant column's sum decodes 1 value, a
 * dictionary equality count decodes 0 values, a sorted range scan touches O(log n) values);
 * (5) memory bounds hold for the encodings that are supposed to help; (6) null semantics are exact;
 * (7) the planner is deterministic.
 *
 * Simplifications: integers only (no floats, no strings), one in-process column store, no I/O — the
 * "compression" is measured in real packed bits and the "cost" in words touched / values decoded,
 * which is exactly the quantity a query planner optimises, but there is no disk.
 */
"use strict";

/* ------------------------------------------------------------------ deterministic randomness */

class Prng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (this.state + 0x9e3779b9) >>> 0;
    let z = this.state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return (z ^ (z >>> 15)) >>> 0;
  }
  below(limit) { return this.next() % limit; }
}

/* ------------------------------------------------------------------ bit packing */

const bitsFor = (maxValue) => (maxValue <= 0 ? 0 : 32 - Math.clz32(maxValue));

/** Zigzag maps signed integers to unsigned ones so small magnitudes stay narrow. */
const zigzag = (v) => (v >= 0 ? 2 * v : -2 * v - 1);
const unzigzag = (u) => (u % 2 === 0 ? u / 2 : -(u + 1) / 2);

/** LSB-first bit writer over 32-bit words: bit i of the stream lives in word i>>5, bit i&31. */
class BitWriter {
  constructor() { this.words = []; this.bitLength = 0; }
  write(value, bits) {
    let remaining = bits;
    let v = value >>> 0;
    while (remaining > 0) {
      const wordIndex = this.bitLength >>> 5;
      const offset = this.bitLength & 31;
      const take = Math.min(32 - offset, remaining);
      const mask = take === 32 ? 0xffffffff : (0xffffffff >>> (32 - take));
      const slice = (v & mask) >>> 0;
      this.words[wordIndex] = ((this.words[wordIndex] || 0) | ((slice << offset) >>> 0)) >>> 0;
      v >>>= take;
      this.bitLength += take;
      remaining -= take;
    }
    return this;
  }
  get bytes() { return Math.ceil(this.bitLength / 8); }
}

class BitReader {
  constructor(words, bitLength) { this.words = words; this.bitLength = bitLength; this.bitPos = 0; }
  read(bits) {
    let value = 0;
    let filled = 0;
    let remaining = bits;
    while (remaining > 0) {
      const wordIndex = this.bitPos >>> 5;
      const offset = this.bitPos & 31;
      const take = Math.min(32 - offset, remaining);
      const mask = take === 32 ? 0xffffffff : (0xffffffff >>> (32 - take));
      const slice = ((this.words[wordIndex] || 0) >>> offset) & mask;
      value = ((value | ((slice << filled) >>> 0)) >>> 0);
      filled += take;
      this.bitPos += take;
      remaining -= take;
    }
    return value >>> 0;
  }
}

/* ------------------------------------------------------------------ encodings */

/**
 * One encoded column. `payload` holds the packed bits; `meta` holds the side information that makes
 * compressed-domain evaluation possible (dictionary, run table, delta checkpoints).
 */
class EncodedColumn {
  constructor(name, encoding, values) {
    this.name = name;
    this.encoding = encoding;
    this.n = values.length;
    this.stats = analyse(values);
    this.build(values);
  }

  build(values) {
    const writer = new BitWriter();
    if (this.encoding === "plain") {
      this.valueBits = bitsFor(Math.max(0, ...values.filter((v) => v !== null).map((v) => zigzag(v))));
      for (const v of values) writer.write(v === null ? 0 : 1, 1).write(v === null ? 0 : zigzag(v), this.valueBits);
    } else if (this.encoding === "dict") {
      const distinct = [...new Set(values.filter((v) => v !== null))].sort((a, b) => a - b);
      this.dictionary = distinct;
      const codes = new Map(distinct.map((v, i) => [v, i]));
      this.nullCode = distinct.length; // null gets its own trailing code
      this.codeBits = bitsFor(distinct.length);
      this.frequency = new Int32Array(distinct.length);
      for (const v of values) {
        const code = v === null ? this.nullCode : codes.get(v);
        if (v !== null) this.frequency[code]++;
        writer.write(code, this.codeBits);
      }
    } else if (this.encoding === "rle") {
      const runs = [];
      for (const v of values) {
        const last = runs[runs.length - 1];
        if (last && last.value === v) last.length++;
        else runs.push({ value: v, length: 1 });
      }
      this.runs = runs;
      this.runValueBits = bitsFor(Math.max(0, ...runs.filter((r) => r.value !== null).map((r) => zigzag(r.value))));
      this.runLengthBits = bitsFor(Math.max(0, ...runs.map((r) => r.length)));
      for (const run of runs) {
        writer.write(run.value === null ? 0 : 1, 1)
          .write(run.value === null ? 0 : zigzag(run.value), this.runValueBits)
          .write(run.length, this.runLengthBits);
      }
    } else if (this.encoding === "delta") {
      const present = values.filter((v) => v !== null);
      this.deltaStart = present.length > 0 ? present[0] : 0;
      const deltas = [];
      for (let i = 1; i < present.length; i++) deltas.push(present[i] - present[i - 1]);
      this.deltaBits = bitsFor(Math.max(0, ...deltas.map((d) => zigzag(d))));
      this.firstBits = bitsFor(zigzag(this.deltaStart));
      this.checkpoints = [{ index: 0, value: null, bitPos: 0 }];
      let running = this.deltaStart;
      writer.write(zigzag(this.deltaStart), this.firstBits);
      for (let i = 0; i < deltas.length; i++) {
        writer.write(zigzag(deltas[i]), this.deltaBits);
        running += deltas[i];
        if ((i + 1) % 32 === 0) this.checkpoints.push({ index: i + 2, value: running, bitPos: writer.bitLength });
      }
      this.deltaCount = deltas.length;
      for (const v of values) writer.write(v === null ? 0 : 1, 1);
    } else {
      throw new Error(`unknown encoding ${this.encoding}`);
    }
    this.words = writer.words;
    this.bitLength = writer.bitLength;
    this.bytes = writer.bytes;
  }

  /** Materialises the column; used by the oracle and by encodings without fast paths. */
  decode() {
    if (this.encoding === "plain") {
      const reader = new BitReader(this.words, this.bitLength);
      const out = [];
      for (let i = 0; i < this.n; i++) {
        const present = reader.read(1);
        const raw = reader.read(this.valueBits);
        out.push(present ? unzigzag(raw) : null);
      }
      return out;
    }
    if (this.encoding === "dict") {
      const reader = new BitReader(this.words, this.bitLength);
      const out = [];
      for (let i = 0; i < this.n; i++) {
        const code = reader.read(this.codeBits);
        out.push(code === this.nullCode ? null : this.dictionary[code]);
      }
      return out;
    }
    if (this.encoding === "rle") {
      const reader = new BitReader(this.words, this.bitLength);
      const out = [];
      for (let i = 0; i < this.runs.length; i++) {
        // The value field is always present in the stream, even for null runs, so it must always be
        // consumed — a short-circuit here would desynchronise the bit cursor.
        const present = reader.read(1);
        const raw = reader.read(this.runValueBits);
        const length = reader.read(this.runLengthBits);
        const value = present ? unzigzag(raw) : null;
        for (let k = 0; k < length; k++) out.push(value);
      }
      return out;
    }
    const present = this.decodePresence();
    const values = [];
    const reader = new BitReader(this.words, this.bitLength);
    let running = unzigzag(reader.read(this.firstBits));
    values.push(running);
    for (let i = 0; i < this.deltaCount; i++) {
      running += unzigzag(reader.read(this.deltaBits));
      values.push(running);
    }
    let cursor = 0;
    return present.map((flag) => (flag ? values[cursor++] : null));
  }

  decodePresence() {
    const stride = this.firstBits + this.deltaCount * this.deltaBits;
    const reader = new BitReader(this.words, this.bitLength);
    reader.bitPos = stride;
    const out = [];
    for (let i = 0; i < this.n; i++) out.push(reader.read(1) === 1);
    return out;
  }

  /** O(log n) random access into a delta column using the 32-entry value checkpoints. */
  decodeAt(index) {
    const present = this.decodePresence();
    if (!present[index]) return null;
    let position = 0;
    for (let i = 0; i <= index; i++) position += present[i] ? 1 : 0;
    position -= 1; // 0-based index inside the non-null subsequence
    let checkpoint = this.checkpoints[0];
    for (const candidate of this.checkpoints) if (candidate.index <= position) checkpoint = candidate;
    const reader = new BitReader(this.words, this.bitLength);
    let running;
    let cursor;
    if (checkpoint.index === 0) {
      reader.bitPos = 0;
      running = unzigzag(reader.read(this.firstBits));
      cursor = 0;
    } else {
      reader.bitPos = checkpoint.bitPos;
      running = checkpoint.value;
      cursor = checkpoint.index - 1;
    }
    while (cursor < position) {
      running += unzigzag(reader.read(this.deltaBits));
      cursor++;
    }
    return running;
  }

  /** Decoded values touched by an aggregate; explicit, so checks can assert on it. */
  evaluate(predicate, aggregate) {
    const runner = new AggregateRunner(aggregate);
    if (this.encoding === "dict") return this.evaluateDict(predicate, runner);
    if (this.encoding === "rle") return this.evaluateRle(predicate, runner);
    if (this.encoding === "delta") return this.evaluateDelta(predicate, runner);
    const values = this.decode();
    let decoded = 0;
    for (const value of values) {
      if (value === null) continue;
      decoded++;
      if (!testPredicate(predicate, value)) continue;
      runner.push(value);
    }
    runner.valuesDecoded = decoded;
    runner.wordsTouched = this.words.length;
    return runner.finish();
  }

  evaluateDict(predicate, runner) {
    // The predicate is mapped over the dictionary only; the packed codes are never materialised.
    runner.valuesDecoded = this.dictionary.length;
    runner.wordsTouched = this.words.length;
    for (let code = 0; code < this.dictionary.length; code++) {
      const value = this.dictionary[code];
      if (!testPredicate(predicate, value)) continue;
      runner.addWeighted(value, this.frequency[code]);
    }
    return runner.finish();
  }

  evaluateRle(predicate, runner) {
    // One predicate evaluation per run, weighted by the run length.
    runner.valuesDecoded = 0;
    runner.wordsTouched = this.words.length;
    for (const run of this.runs) {
      runner.runsVisited++;
      if (run.value === null) continue;
      runner.valuesDecoded++;
      if (!testPredicate(predicate, run.value)) continue;
      runner.addWeighted(run.value, run.length);
    }
    return runner.finish();
  }

  evaluateDelta(predicate, runner) {
    if (!this.stats.sorted) {
      // Deltas only give random access on sorted columns; otherwise fall back to a full decode.
      const values = this.decode();
      let decoded = 0;
      for (const value of values) {
        if (value === null) continue;
        decoded++;
        if (testPredicate(predicate, value)) runner.push(value);
      }
      runner.valuesDecoded = decoded;
      runner.wordsTouched = this.words.length;
      return runner.finish();
    }
    // Sorted: binary search the ends of the matching range via decodeAt, so only O(log n) values
    // plus the matched ones are touched.
    const present = this.decodePresence();
    const positions = [];
    for (let i = 0; i < this.n; i++) if (present[i]) positions.push(i);
    const byValue = (index) => this.decodeAt(positions[index]);
    /** First position whose value is >= target. */
    const lowerBound = (target) => {
      let lo = 0;
      let hi = positions.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (byValue(mid) < target) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    };
    /** First position whose value is > target. */
    const upperBound = (target) => {
      let lo = 0;
      let hi = positions.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (byValue(mid) <= target) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    };
    let from = 0;
    let to = positions.length;
    if (predicate.kind === "eq") { from = lowerBound(predicate.value); to = upperBound(predicate.value); }
    else if (predicate.kind === "lt") { to = lowerBound(predicate.value); }
    else if (predicate.kind === "gt") { from = upperBound(predicate.value); }
    else if (predicate.kind === "range") {
      from = lowerBound(predicate.lo);
      to = upperBound(predicate.hi);
    }
    let decoded = 0;
    for (let i = from; i < to; i++) {
      const value = this.decodeAt(positions[i]);
      decoded++;
      runner.push(value);
    }
    runner.valuesDecoded = decoded + Math.ceil(Math.log2(positions.length + 1)) * 2;
    runner.wordsTouched = Math.max(1, Math.ceil(this.bytes / 8));
    runner.rangeScan = true;
    return runner.finish();
  }
}

/* ------------------------------------------------------------------ predicates and aggregates */

const PREDICATE_KINDS = new Set(["all", "eq", "lt", "gt", "range"]);

function testPredicate(predicate, value) {
  switch (predicate.kind) {
    case "all": return true;
    case "eq": return value === predicate.value;
    case "lt": return value < predicate.value;
    case "gt": return value > predicate.value;
    case "range": return value >= predicate.lo && value <= predicate.hi;
    default: throw new Error(`unknown predicate ${predicate.kind}`);
  }
}

function normalisePredicate(predicate) {
  if (!predicate || !PREDICATE_KINDS.has(predicate.kind)) throw new Error(`invalid predicate`);
  if (predicate.kind === "eq" && !Number.isInteger(predicate.value)) throw new Error("eq needs an integer");
  if (predicate.kind === "range" && !(Number.isInteger(predicate.lo) && Number.isInteger(predicate.hi))) {
    throw new Error("range needs integer bounds");
  }
  if (predicate.kind === "range" && predicate.lo > predicate.hi) throw new Error("range is empty");
  return predicate;
}

const AGGREGATES = new Set(["count", "sum", "min", "max", "avg"]);

/** Accumulates aggregates once, independently of the encoding that feeds it. */
class AggregateRunner {
  constructor(aggregate) {
    if (!AGGREGATES.has(aggregate)) throw new Error(`unknown aggregate ${aggregate}`);
    this.aggregate = aggregate;
    this.count = 0;
    this.total = 0;
    this.low = null;
    this.high = null;
    this.valuesDecoded = 0;
    this.wordsTouched = 0;
    this.runsVisited = 0;
    this.rangeScan = false;
  }
  push(value) { this.addWeighted(value, 1); }
  /** Grouped update: a run of `weight` equal values costs one decision. */
  addWeighted(value, weight) {
    this.count += weight;
    this.total += value * weight;
    if (this.low === null || value < this.low) this.low = value;
    if (this.high === null || value > this.high) this.high = value;
  }
  finish() {
    let result;
    if (this.aggregate === "count") result = this.count;
    else if (this.aggregate === "sum") result = this.total;
    else if (this.aggregate === "min") result = this.low;
    else if (this.aggregate === "max") result = this.high;
    else result = this.count === 0 ? null : this.total / this.count;
    return {
      result, count: this.count, valuesDecoded: this.valuesDecoded, wordsTouched: this.wordsTouched,
      runsVisited: this.runsVisited, rangeScan: this.rangeScan,
    };
  }
}

/** Independent oracle: plain scan over the raw array, no compression involved. */
function plainScan(values, predicate, aggregate) {
  normalisePredicate(predicate);
  const runner = new AggregateRunner(aggregate);
  for (const value of values) {
    if (value === null) continue;
    if (testPredicate(predicate, value)) runner.push(value);
  }
  return runner.finish().result;
}

/* ------------------------------------------------------------------ statistics and planner */

function analyse(values) {
  const present = values.filter((v) => v !== null);
  let runs = 0;
  for (let i = 0; i < values.length; i++) if (i === 0 || values[i] !== values[i - 1]) runs++;
  let sorted = true;
  for (let i = 1; i < present.length; i++) if (present[i] < present[i - 1]) { sorted = false; break; }
  return {
    n: values.length,
    nullCount: values.length - present.length,
    distinct: new Set(present).size,
    runs,
    sorted,
    runRatio: values.length === 0 ? 0 : runs / values.length,
  };
}

const ENCODINGS = ["plain", "dict", "rle", "delta"];

/**
 * Encoding applicability — the same pruning a real column store applies before costing a plan:
 * a dictionary only pays off while the cardinality is well below the row count, run-length encoding
 * only while runs are long, and deltas only give random access on sorted columns.
 */
function candidatesFor(values) {
  const stats = analyse(values);
  const candidates = ["plain"];
  if (stats.n > 0 && stats.distinct * 2 <= stats.n) candidates.push("dict");
  if (stats.runRatio <= 0.5) candidates.push("rle");
  if (stats.sorted) candidates.push("delta");
  return candidates;
}

/**
 * Cost model. Both terms are counted in the same units the evaluator reports, so the model can be
 * validated against measurement: a query costs (payload words read) + (values decoded), times the
 * workload weight. Value-decoding dominates on wide columns, which is how the planner learns that a
 * dictionary or a run table beats a full scan.
 */
function estimateCost(column, queries) {
  let cost = 0;
  for (const query of queries) {
    const words = Math.max(1, Math.ceil(column.bytes / 4));
    let decode;
    if (column.encoding === "dict") decode = column.dictionary.length;
    else if (column.encoding === "rle") decode = column.runs.length;
    else if (column.encoding === "delta") decode = column.stats.sorted ? Math.ceil(Math.log2(column.n + 1)) * 2 : column.n;
    else decode = column.n;
    cost += query.weight * (words + decode);
  }
  return cost;
}

/** Exhaustive planner: every applicable assignment over the columns is a candidate. */
function planColumns(specs, workload, options = {}) {
  let best = null;
  const current = new Array(specs.length).fill(null);
  const recurse = (index, estimated) => {
    if (index === specs.length) {
      if (best === null || estimated < best.estimated) best = { plan: current.slice(), estimated };
      return;
    }
    const candidates = options.candidateEncodings || candidatesFor(specs[index].values);
    const queries = workload.filter((q) => q.column === specs[index].name);
    for (const encoding of candidates) {
      const encoded = new EncodedColumn(specs[index].name, encoding, specs[index].values);
      current[index] = encoded;
      recurse(index + 1, estimated + estimateCost(encoded, queries));
    }
  };
  recurse(0, 0);
  return best;
}

/* ------------------------------------------------------------------ column shapes */

function makeColumn(shape, count, seed) {
  const prng = new Prng(seed);
  const values = [];
  for (let i = 0; i < count; i++) {
    if (shape === "constant") values.push(42);
    else if (shape === "sorted") values.push(i * 3 + 1);
    else if (shape === "sorted-with-gaps") values.push(i * i);
    else if (shape === "clustered") values.push(Math.floor(i / 25) * 10);
    else if (shape === "lowcard") values.push(prng.below(4) * 5);
    else if (shape === "highcard") values.push(prng.below(1 << 20));
    else if (shape === "nulls") values.push(i % 3 === 0 ? null : prng.below(50));
    else if (shape === "empty") { /* nothing */ }
    else throw new Error(`unknown shape ${shape}`);
  }
  if (shape === "nulls" && count > 8) { values[1] = null; values[2] = null; }
  return values;
}

module.exports = {
  Prng, BitWriter, BitReader, bitsFor, zigzag, unzigzag, EncodedColumn, AggregateRunner,
  testPredicate, normalisePredicate, plainScan, analyse, ENCODINGS, candidatesFor, estimateCost,
  planColumns, makeColumn,
};

/* ------------------------------------------------------------------ self-checks */

if (require.main === module) {
  const assert = require("node:assert");

  // --- packing primitives -----------------------------------------------------------------
  for (let bits = 0; bits <= 32; bits++) {
    const writer = new BitWriter();
    const values = [];
    const limit = bits === 32 ? 0xffffffff : Math.pow(2, bits) - 1;
    for (let trial = 0; trial < 12; trial++) {
      const value = limit === 0 ? 0 : (trial * 0x9e3779b1) % (limit + 1);
      values.push(value >>> 0);
      writer.write(value, bits);
    }
    const reader = new BitReader(writer.words, writer.bitLength);
    assert.deepStrictEqual(values.map(() => reader.read(bits)), values, `bit round trip at ${bits} bits`);
  }
  for (const value of [0, 1, -1, 32767, -32768, 1000000, -1000000]) {
    assert.strictEqual(unzigzag(zigzag(value)), value, `zigzag ${value}`);
  }
  assert.strictEqual(bitsFor(0), 0);
  assert.strictEqual(bitsFor(255), 8);
  assert.strictEqual(bitsFor(256), 9);
  const handPacked = new BitWriter().write(0b101, 3).write(0b1111, 4);
  assert.strictEqual(handPacked.words[0], 0b1111101, "LSB-first packing is as documented");
  assert.strictEqual(handPacked.bytes, 1);

  // --- round trip of every encoding over every shape --------------------------------------
  const shapes = ["constant", "sorted", "sorted-with-gaps", "clustered", "lowcard", "highcard", "nulls"];
  for (const shape of shapes) {
    const values = makeColumn(shape, 200, 887);
    for (const encoding of ENCODINGS) {
      const column = new EncodedColumn(shape, encoding, values);
      assert.deepStrictEqual(column.decode(), values, `${shape}/${encoding} round trip`);
    }
  }
  assert.deepStrictEqual(new EncodedColumn("e", "plain", []).decode(), [], "empty column round trip");
  assert.deepStrictEqual(new EncodedColumn("e", "rle", []).decode(), []);
  assert.deepStrictEqual(new EncodedColumn("e", "delta", []).decode(), []);
  assert.deepStrictEqual(new EncodedColumn("e", "dict", []).decode(), []);
  assert.deepStrictEqual(new EncodedColumn("s", "dict", [null]).decode(), [null], "all-null column");

  // --- memory bounds: the right encoding really is smaller --------------------------------
  const constant = makeColumn("constant", 200, 1);
  assert.ok(new EncodedColumn("c", "rle", constant).bytes <= 8, "a constant column is O(1) bytes");
  assert.ok(new EncodedColumn("c", "rle", constant).bytes < new EncodedColumn("c", "plain", constant).bytes);
  const lowcard = makeColumn("lowcard", 400, 2);
  assert.ok(new EncodedColumn("l", "dict", lowcard).bytes < new EncodedColumn("l", "plain", lowcard).bytes);
  const sorted = makeColumn("sorted", 400, 3);
  assert.ok(new EncodedColumn("s", "delta", sorted).bytes < new EncodedColumn("s", "plain", sorted).bytes);
  const clustered = makeColumn("clustered", 400, 4);
  assert.ok(new EncodedColumn("k", "rle", clustered).bytes < new EncodedColumn("k", "plain", clustered).bytes * 8 / 100);

  // --- aggregate correctness against the oracle, for every encoding and predicate ----------
  const predicates = [
    { kind: "all" }, { kind: "eq", value: 10 }, { kind: "lt", value: 100 },
    { kind: "gt", value: 5 }, { kind: "range", lo: 3, hi: 60 },
  ];
  let comparisons = 0;
  for (const shape of shapes) {
    const values = makeColumn(shape, 150, 4242);
    for (const encoding of ENCODINGS) {
      const column = new EncodedColumn(shape, encoding, values);
      for (const predicate of predicates) {
        for (const aggregate of ["count", "sum", "min", "max", "avg"]) {
          const expected = plainScan(values, predicate, aggregate);
          const actual = column.evaluate(predicate, aggregate).result;
          if (expected === null) assert.strictEqual(actual, null, `${shape}/${encoding}/${predicate.kind}/${aggregate}`);
          else assert.ok(Math.abs(actual - expected) < 1e-9,
            `${shape}/${encoding}/${predicate.kind}/${aggregate}: ${actual} != ${expected}`);
          comparisons++;
        }
      }
    }
  }
  assert.ok(comparisons >= 600, `only ${comparisons} aggregate comparisons ran`);

  // --- compressed-domain evaluation really saves work --------------------------------------
  const bigConstant = new EncodedColumn("c", "rle", makeColumn("constant", 5000, 5));
  const constantSum = bigConstant.evaluate({ kind: "all" }, "sum");
  assert.strictEqual(constantSum.result, 42 * 5000);
  assert.strictEqual(constantSum.runsVisited, 1, "one run for a constant column");
  assert.strictEqual(constantSum.valuesDecoded, 1, "one decoded value, not 5000");

  const bigLowCard = new EncodedColumn("d", "dict", makeColumn("lowcard", 5000, 6));
  const dictCount = bigLowCard.evaluate({ kind: "eq", value: 10 }, "count");
  assert.strictEqual(dictCount.result, plainScan(bigLowCard.decode(), { kind: "eq", value: 10 }, "count"));
  assert.strictEqual(dictCount.valuesDecoded, bigLowCard.dictionary.length, "dictionary only");
  assert.ok(dictCount.valuesDecoded <= 4);

  const bigSorted = new EncodedColumn("s", "delta", makeColumn("sorted", 5000, 7));
  const rangeCount = bigSorted.evaluate({ kind: "range", lo: 300, hi: 600 }, "count");
  assert.strictEqual(rangeCount.result, plainScan(bigSorted.decode(), { kind: "range", lo: 300, hi: 600 }, "count"));
  assert.ok(rangeCount.result > 90, "the range does match roughly 100 values");
  assert.ok(rangeCount.valuesDecoded < 300, `range scan touched ${rangeCount.valuesDecoded} values`);
  assert.ok(rangeCount.rangeScan, "sorted columns take the binary-search path");
  const narrow = bigSorted.evaluate({ kind: "range", lo: 300, hi: 315 }, "sum");
  assert.strictEqual(narrow.result, plainScan(bigSorted.decode(), { kind: "range", lo: 300, hi: 315 }, "sum"));
  assert.ok(narrow.valuesDecoded < 60, `narrow range touched only ${narrow.valuesDecoded} values`);
  const tail = bigSorted.evaluate({ kind: "gt", value: 14900 }, "count");
  assert.strictEqual(tail.result, plainScan(bigSorted.decode(), { kind: "gt", value: 14900 }, "count"));
  assert.ok(tail.valuesDecoded < 60, `tail range touched ${tail.valuesDecoded} values`);
  assert.strictEqual(bigSorted.decodeAt(4999), plainScan(bigSorted.decode(), { kind: "all" }, "max"));
  assert.strictEqual(bigSorted.decodeAt(0), 1);
  assert.strictEqual(bigSorted.decodeAt(3), 10);

  const plainColumn = new EncodedColumn("p", "plain", makeColumn("highcard", 1000, 8));
  const plainAgg = plainColumn.evaluate({ kind: "range", lo: 0, hi: 1 << 20 }, "max");
  assert.strictEqual(plainAgg.valuesDecoded, 1000, "plain must decode everything");

  // --- null semantics ---------------------------------------------------------------------
  const nulls = makeColumn("nulls", 120, 9);
  const allNull = new Array(20).fill(null);
  const nullColumn = new EncodedColumn("n", "plain", allNull);
  assert.strictEqual(nullColumn.evaluate({ kind: "all" }, "count").result, 0);
  assert.strictEqual(nullColumn.evaluate({ kind: "all" }, "sum").result, 0);
  assert.strictEqual(nullColumn.evaluate({ kind: "all" }, "min").result, null);
  assert.strictEqual(nullColumn.evaluate({ kind: "all" }, "avg").result, null);
  const nullDict = new EncodedColumn("n2", "dict", allNull);
  assert.strictEqual(nullDict.evaluate({ kind: "all" }, "min").result, null);
  assert.strictEqual(nullDict.decode().length, 20);
  assert.strictEqual(new EncodedColumn("n3", "rle", nulls).evaluate({ kind: "all" }, "count").result,
    plainScan(nulls, { kind: "all" }, "count"));

  // --- edge cases -------------------------------------------------------------------------
  const empty = new EncodedColumn("e", "dict", []);
  assert.strictEqual(empty.evaluate({ kind: "all" }, "count").result, 0);
  assert.strictEqual(empty.evaluate({ kind: "all" }, "avg").result, null);
  assert.strictEqual(plainScan([], { kind: "eq", value: 1 }, "count"), 0);
  assert.strictEqual(plainScan([7], { kind: "eq", value: 7 }, "count"), 1);
  assert.strictEqual(plainScan([7], { kind: "gt", value: 7 }, "count"), 0);
  assert.strictEqual(plainScan([7], { kind: "range", lo: 7, hi: 7 }, "max"), 7);
  assert.strictEqual(plainScan([1, 2], { kind: "eq", value: 99 }, "min"), null, "no match is null");
  assert.throws(() => plainScan([1], { kind: "between" }, "count"), /invalid predicate/);
  assert.throws(() => plainScan([1], { kind: "range", lo: 5, hi: 1 }, "count"), /empty/);
  assert.throws(() => plainScan([1], { kind: "all" }, "median"), /unknown aggregate/);
  assert.throws(() => new EncodedColumn("x", "huffman", [1]), /unknown encoding/);

  // --- planner: exhaustive optimum and workload sensitivity -------------------------------
  const workload = [
    { column: "code", predicate: { kind: "eq", value: 10 }, aggregate: "count", weight: 4 },
    { column: "code", predicate: { kind: "all" }, aggregate: "sum", weight: 1 },
    { column: "ts", predicate: { kind: "range", lo: 100, hi: 300 }, aggregate: "count", weight: 3 },
    { column: "value", predicate: { kind: "gt", value: 1000 }, aggregate: "avg", weight: 2 },
  ];
  const specs = [
    { name: "code", values: makeColumn("lowcard", 400, 11) },
    { name: "ts", values: makeColumn("sorted", 400, 12) },
    { name: "value", values: makeColumn("highcard", 400, 13) },
  ];
  const chosen = planColumns(specs, workload);
  const layout = new Map(chosen.plan.map((column) => [column.name, column.encoding]));
  assert.strictEqual(layout.get("code"), "dict", "low-cardinality column with equality workload -> dict");
  assert.strictEqual(layout.get("ts"), "delta", "sorted column with range workload -> delta");
  assert.strictEqual(layout.get("value"), "plain", "high-cardinality unsorted column -> plain");

  // The chosen plan must be the model optimum, verified by brute force over all assignments.
  let bruteBest = Infinity;
  for (const a of candidatesFor(specs[0].values)) {
    for (const b of candidatesFor(specs[1].values)) {
      for (const c of candidatesFor(specs[2].values)) {
        const candidate = [a, b, c].map((encoding, i) =>
          new EncodedColumn(specs[i].name, encoding, specs[i].values));
        const cost = candidate.reduce((total, column) =>
          total + estimateCost(column, workload.filter((q) => q.column === column.name)), 0);
        bruteBest = Math.min(bruteBest, cost);
      }
    }
  }
  assert.strictEqual(chosen.estimated, bruteBest, "planner must find the model optimum");

  // --- measured cost: the chosen plan beats the plain layout on the real evaluator ---------
  const measure = (columns) => {
    let cost = 0;
    for (const query of workload) {
      const column = columns.find((c) => c.name === query.column);
      const outcome = estimateCost(column, [query]);
      const measured = column.evaluate(query.predicate, query.aggregate);
      const actual = query.weight * (measured.wordsTouched + measured.valuesDecoded);
      assert.ok(outcome > 0 && actual > 0);
      cost += actual;
    }
    return cost;
  };
  const chosenCost = measure(chosen.plan);
  const plainPlan = specs.map((spec) => new EncodedColumn(spec.name, "plain", spec.values));
  const plainCost = measure(plainPlan);
  assert.ok(chosenCost < plainCost, `query-aware plan ${chosenCost} must beat plain ${plainCost}`);

  // A cluster-oriented workload picks run-length encoding instead.
  const clusteredSpecs = [{ name: "region", values: makeColumn("clustered", 400, 14) }];
  const clusteredWorkload = [
    { column: "region", predicate: { kind: "eq", value: 20 }, aggregate: "count", weight: 5 },
    { column: "region", predicate: { kind: "range", lo: 0, hi: 40 }, aggregate: "sum", weight: 5 },
  ];
  const clusteredPlan = planColumns(clusteredSpecs, clusteredWorkload);
  assert.strictEqual(clusteredPlan.plan[0].encoding, "rle");
  assert.ok(measure2(clusteredPlan.plan, clusteredWorkload) <
    measure2([new EncodedColumn("region", "plain", clusteredSpecs[0].values)], clusteredWorkload));
  function measure2(columns, queries) {
    return columns.reduce((total, column) => {
      for (const query of queries) {
        const measured = column.evaluate(query.predicate, query.aggregate);
        total += query.weight * (measured.wordsTouched + measured.valuesDecoded);
      }
      return total;
    }, 0);
  }

  // --- determinism ------------------------------------------------------------------------
  const again = planColumns(specs, workload);
  assert.deepStrictEqual(again.plan.map((c) => c.encoding), chosen.plan.map((c) => c.encoding));
  assert.strictEqual(again.estimated, chosen.estimated);

  // --- cost model sanity: estimates order the encodings the way measurement does ----------
  const sortedWorkload = [{ column: "ts", predicate: { kind: "range", lo: 0, hi: 100 }, aggregate: "count", weight: 1 }];
  const estDelta = estimateCost(new EncodedColumn("ts", "delta", specs[1].values), sortedWorkload);
  const estPlain = estimateCost(new EncodedColumn("ts", "plain", specs[1].values), sortedWorkload);
  assert.ok(estDelta < estPlain, "the model must predict the delta advantage");
  const deltaMeasured = new EncodedColumn("ts", "delta", specs[1].values)
    .evaluate({ kind: "range", lo: 0, hi: 100 }, "count");
  const plainMeasured = new EncodedColumn("ts", "plain", specs[1].values)
    .evaluate({ kind: "range", lo: 0, hi: 100 }, "count");
  assert.ok(deltaMeasured.valuesDecoded < plainMeasured.valuesDecoded, "and the measurement must agree");

  console.log(`column-store: ${comparisons} aggregate comparisons, chosen layout ` +
    `${specs.map((s) => `${s.name}=${layout.get(s.name)}`).join(" ")}, ` +
    `measured cost ${chosenCost} vs plain ${plainCost}`);
  console.log("PASS");
}
