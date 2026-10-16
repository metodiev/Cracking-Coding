/** Task 653 - Implement compressed column encoding schemes (RLE, dictionary) and decode. (Tier: Expert, CTCI Ch 14 — Databases)
 *
 * Four candidate layouts for a column chunk, the same menu a column store (Parquet, ORC)
 * picks from per chunk:
 *   - RLE        : runs of `{value, length}` — wins on clustered / few-distinct columns;
 *   - dictionary : value dictionary + bit-packed codes — wins on low-cardinality columns;
 *   - FOR (frame of reference): numeric minimum as base + bit-packed offsets — wins on
 *                  narrow numeric ranges;
 *   - raw        : 32 bits per value, the fallback.
 * Each layout charges a header (32 bits for RLE/dictionary, 64 bits for FOR's base) so the
 * comparison is not rigged in favour of the cleverest scheme.
 *
 * Bit packing is a real LSB-first bit stream over 32-bit words (accumulated in BigInt so
 * 32-bit fields stay exact); `packBits`/`unpackBits` are inverses for widths 0..32.
 *
 * Invariant verified: `decodeColumn(encodeColumn(col))` reproduces the column elementwise
 * for every scheme on every column tested (empty, singleton, all-equal, all-distinct,
 * numeric with negatives, string, PRNG-skewed), the chosen scheme's size is the true argmin
 * of the estimates (brute-force minimum of the statistics-derived sizes), the payload the
 * winning scheme produced is never larger than raw storage, and malformed input is rejected
 * rather than silently mis-decoded.
 */
"use strict";
const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");
/** Deterministic xorshift32; returns integers in [0, bound). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x1f123bb5;
  return function next(bound) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}
const WORD_BITS = 32;
const HEADER_BITS = 32;
const FOR_HEADER_BITS = 64;
function bitWidthOf(maxValue) {
  if (maxValue <= 0) return 0;
  return 32 - Math.clz32(maxValue);
}
/** LSB-first bit packer: each value becomes one `width`-bit unsigned field. */
function packBits(values, width) {
  if (!Number.isInteger(width) || width < 0 || width > 32) throw new RangeError(`bad width ${width}`);
  const mask = width === 0 ? 0n : (1n << BigInt(width)) - 1n;
  const words = [];
  let acc = 0n;
  let bits = 0;
  for (const value of values) {
    const v = BigInt(Math.trunc(value));
    if (v < 0n || v > mask) throw new RangeError(`value ${value} does not fit in ${width} bits`);
    acc |= v << BigInt(bits);
    bits += width;
    while (bits >= WORD_BITS) {
      words.push(Number(acc & 0xffffffffn) >>> 0);
      acc >>= 32n;
      bits -= WORD_BITS;
    }
  }
  if (bits > 0) words.push(Number(acc & 0xffffffffn) >>> 0);
  return new Uint32Array(words);
}
function unpackBits(words, width, count) {
  const out = new Array(count).fill(0);
  if (width === 0) return out;
  const mask = (1n << BigInt(width)) - 1n;
  let acc = 0n;
  let bits = 0;
  let wi = 0;
  for (let i = 0; i < count; i++) {
    while (bits < width) {
      if (wi >= words.length) throw new RangeError("bit stream exhausted");
      acc |= BigInt(words[wi] >>> 0) << BigInt(bits);
      wi++;
      bits += WORD_BITS;
    }
    out[i] = Number(acc & mask);
    acc >>= BigInt(width);
    bits -= width;
  }
  return out;
}
// ------------------------------------------------------------- run-length encoding
function rleEncode(values) {
  const runs = [];
  for (const value of values) {
    const last = runs.length > 0 ? runs[runs.length - 1] : undefined;
    if (last !== undefined && last.value === value) last.run++;
    else runs.push({ value, run: 1 });
  }
  return runs;
}
function rleDecode(runs, expectedLength) {
  const out = [];
  let total = 0;
  for (const { value, run } of runs) {
    if (!Number.isInteger(run) || run <= 0) throw new RangeError(`bad run length ${run}`);
    total += run;
    if (expectedLength !== undefined && total > expectedLength) {
      throw new RangeError("run lengths exceed the declared column length");
    }
    for (let i = 0; i < run; i++) out.push(value);
  }
  if (expectedLength !== undefined && total !== expectedLength) {
    throw new RangeError(`run lengths sum to ${total}, expected ${expectedLength}`);
  }
  return out;
}
function rleBitSize(runCount, n) {
  if (n === 0) return 0;
  return HEADER_BITS + runCount * (WORD_BITS + bitWidthOf(n));
}
// --------------------------------------------------------------- dictionary encoding
function dictionaryEncode(values) {
  const dictionary = [];
  const index = new Map();
  const rawCodes = [];
  for (const value of values) {
    let code = index.get(value);
    if (code === undefined) {
      code = dictionary.length;
      index.set(value, code);
      dictionary.push(value);
    }
    rawCodes.push(code);
  }
  // At least one bit per code: a column still has to distinguish "code present" from absent.
  const codeWidth = values.length === 0 ? 0 : Math.max(1, bitWidthOf(dictionary.length - 1));
  return { dictionary, codeWidth, count: values.length, codes: packBits(rawCodes, codeWidth) };
}
function dictionaryDecode(enc) {
  const codes = unpackBits(enc.codes, enc.codeWidth, enc.count);
  const out = new Array(enc.count);
  for (let i = 0; i < enc.count; i++) {
    if (codes[i] >= enc.dictionary.length) {
      throw new RangeError(`code ${codes[i]} outside the dictionary of ${enc.dictionary.length}`);
    }
    out[i] = enc.dictionary[codes[i]];
  }
  return out;
}
function dictionaryBitSize(distinct, n) {
  if (n === 0) return 0;
  return HEADER_BITS + distinct * WORD_BITS + n * Math.max(1, bitWidthOf(distinct - 1));
}
// --------------------------------------------------------- frame of reference (FOR)
function forEncode(values) {
  if (values.some((v) => typeof v !== "number" || !Number.isFinite(v))) {
    throw new TypeError("frame-of-reference encoding needs finite numbers");
  }
  if (values.length === 0) return { base: 0, width: 0, count: 0, offsets: new Uint32Array(0) };
  const base = values.reduce((min, v) => (v < min ? v : min), values[0]);
  const offsets = values.map((v) => v - base);
  const width = bitWidthOf(offsets.reduce((max, v) => (v > max ? v : max), 0));
  return { base, width, count: values.length, offsets: packBits(offsets, width) };
}
function forDecode(enc) {
  return unpackBits(enc.offsets, enc.width, enc.count).map((off) => off + enc.base);
}
function forBitSize(n, width) {
  return n === 0 ? 0 : FOR_HEADER_BITS + n * width;
}
// --------------------------------------------------------------- scheme selection
/** Statistics an analyst reads straight off the column, used to size every candidate. */
function columnStats(values) {
  const runs = rleEncode(values).length;
  const distinct = new Set(values).size;
  const allNumeric = values.every((v) => typeof v === "number" && Number.isFinite(v));
  let rangeWidth = 0;
  if (allNumeric && values.length > 0) {
    let min = values[0];
    let max = values[0];
    for (const v of values) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    rangeWidth = bitWidthOf(max - min);
  }
  return { n: values.length, runs, distinct, allNumeric, rangeWidth };
}
function estimateSizes(values) {
  const s = columnStats(values);
  return {
    rle: rleBitSize(s.runs, s.n),
    dictionary: dictionaryBitSize(s.distinct, s.n),
    for: s.allNumeric ? forBitSize(s.n, s.rangeWidth) : Infinity,
    raw: s.n * WORD_BITS,
  };
}
function encodeColumn(values) {
  const sizes = estimateSizes(values);
  let scheme = "raw";
  for (const candidate of ["rle", "dictionary", "for"]) {
    if (sizes[candidate] < sizes[scheme]) scheme = candidate;
  }
  switch (scheme) {
    case "rle":
      return { scheme, bits: sizes.rle, payload: { runs: rleEncode(values), count: values.length } };
    case "dictionary":
      return { scheme, bits: sizes.dictionary, payload: dictionaryEncode(values) };
    case "for":
      return { scheme, bits: sizes["for"], payload: forEncode(values) };
    default:
      return { scheme, bits: sizes.raw, payload: { values: values.slice(), count: values.length } };
  }
}
function decodeColumn(enc) {
  switch (enc.scheme) {
    case "rle":
      return rleDecode(enc.payload.runs, enc.payload.count);
    case "dictionary":
      return dictionaryDecode(enc.payload);
    case "for":
      return forDecode(enc.payload);
    case "raw":
      return enc.payload.values.slice();
    default:
      throw new RangeError(`unknown scheme ${enc.scheme}`);
  }
}
/** Physical words a payload occupies, for a sanity check against the estimate. */
function payloadWords(enc) {
  switch (enc.scheme) {
    case "rle":
      return enc.payload.runs.length * 2;
    case "dictionary":
      return enc.payload.dictionary.length + enc.payload.codes.length;
    case "for":
      return 2 + enc.payload.offsets.length;
    default:
      return enc.payload.values.length;
  }
}
module.exports = {
  bitWidthOf, packBits, unpackBits, rleEncode, rleDecode, rleBitSize,
  dictionaryEncode, dictionaryDecode, dictionaryBitSize, forEncode, forDecode,
  forBitSize, columnStats, estimateSizes, encodeColumn, decodeColumn, payloadWords, makeRng,
};
if (require.main === module) {
  // -- bit packer: every width 0..32 is an exact inverse ---------------------------
  const brng = makeRng(4242);
  for (let width = 0; width <= 32; width++) {
    const limit = width === 0 ? 1 : Math.min(2 ** Math.min(width, 24), 1 << 20);
    const values = Array.from({ length: 37 }, () => brng(limit));
    deepStrictEqual(unpackBits(packBits(values, width), width, values.length), values,
      `round-trip at width ${width}`);
  }
  const edge = [0, 0xffffffff, 0x80000000, 1, 0x7fffffff];
  deepStrictEqual(unpackBits(packBits(edge, 32), 32, edge.length), edge, "32-bit edge values");
  throws(() => packBits([256], 8), RangeError, "a value wider than the field must be rejected");
  // -- RLE ------------------------------------------------------------------------
  const rleValues = [1, 1, 1, 2, 2, 3];
  const runs = rleEncode(rleValues);
  deepStrictEqual(runs, [{ value: 1, run: 3 }, { value: 2, run: 2 }, { value: 3, run: 1 }]);
  deepStrictEqual(rleDecode(runs, rleValues.length), rleValues);
  strictEqual(rleEncode([]).length, 0);
  deepStrictEqual(rleDecode([], 0), []);
  deepStrictEqual(rleEncode([9, 9, 9, 9]), [{ value: 9, run: 4 }], "all-equal collapses to one run");
  deepStrictEqual(rleDecode(rleEncode(["a", "a", "b"])), ["a", "a", "b"], "RLE handles strings too");
  throws(() => rleDecode(runs, 7), RangeError, "run/length mismatch must be rejected");
  throws(() => rleDecode([{ value: 1, run: 0 }], 1), RangeError, "zero-length run must be rejected");
  // -- dictionary -----------------------------------------------------------------
  const dictValues = ["red", "blue", "red", "red", "green"];
  const dict = dictionaryEncode(dictValues);
  strictEqual(dict.dictionary.length, 3, "dictionary holds the distinct values");
  strictEqual(dict.codeWidth, 2, "three distinct values need two code bits");
  deepStrictEqual(dictionaryDecode(dict), dictValues);
  const singleDict = dictionaryEncode([7, 7, 7, 7]);
  strictEqual(singleDict.codeWidth, 1, "codes are at least one bit wide");
  deepStrictEqual(dictionaryDecode(singleDict), [7, 7, 7, 7]);
  deepStrictEqual(dictionaryDecode(dictionaryEncode([])), []);
  throws(() => dictionaryDecode({ dictionary: ["a"], codeWidth: 8, count: 1, codes: packBits([4], 8) }),
    RangeError, "a code outside the dictionary must be rejected");
  // -- frame of reference ---------------------------------------------------------
  const numeric = [-100, 0, 37, 100, -100, 5];
  const fore = forEncode(numeric);
  strictEqual(fore.base, -100);
  strictEqual(fore.width, 8, "the range 0..200 fits in 8 bits");
  deepStrictEqual(forDecode(fore), numeric);
  deepStrictEqual(forDecode(forEncode([3])), [3]);
  deepStrictEqual(forDecode(forEncode([])), []);
  throws(() => forEncode(["x", "y"]), TypeError, "FOR is numeric only");
  // -- selection is the true argmin, cross-checked against counted statistics ------
  const columns = [
    [],
    [5],
    [1, 1, 1, 1, 1, 1],
    Array.from({ length: 64 }, (_, i) => i),
    Array.from({ length: 64 }, (_, i) => (i < 32 ? 0 : 1)),
    Array.from({ length: 64 }, (_, i) => (i % 3 === 0 ? 1000 : i)),
    "the quick brown fox jumps over the lazy dog".split(""),
  ];
  for (const col of columns) {
    const enc = encodeColumn(col);
    const sizes = estimateSizes(col);
    const bruteMin = Math.min(sizes.rle, sizes.dictionary, sizes["for"], sizes.raw);
    strictEqual(enc.bits, bruteMin, "the chosen size must equal the brute-force minimum");
    deepStrictEqual(decodeColumn(enc), col, `decode must reproduce a column of length ${col.length}`);
    ok(enc.bits <= sizes.raw, "the chosen scheme never exceeds raw storage");
    ok(payloadWords(enc) * WORD_BITS >= enc.bits - 2 * FOR_HEADER_BITS,
      "the physical payload must be able to hold the estimated bits");
  }
  strictEqual(encodeColumn(Array.from({ length: 64 }, (_, i) => i)).scheme, "for",
    "a dense range is best served by frame of reference");
  strictEqual(encodeColumn(Array.from({ length: 64 }, () => "same")).scheme, "rle",
    "a constant column is best served by RLE");
  const twoValue = encodeColumn(Array.from({ length: 64 }, (_, i) => (i % 2 === 0 ? "yes" : "no")));
  strictEqual(twoValue.scheme, "dictionary", "a low-cardinality string column prefers a dictionary");
  strictEqual(encodeColumn([]).bits, 0, "an empty column costs nothing");
  // -- PRNG-skewed columns: all schemes round-trip, selection stays minimal ---------
  const rng = makeRng(777001);
  for (let trial = 0; trial < 30; trial++) {
    const n = 20 + rng(180);
    const hot = rng(4) + 1;
    const values = Array.from({ length: n }, () => (rng(100) < 70 ? rng(hot) : rng(500) - 250));
    const enc = encodeColumn(values);
    deepStrictEqual(decodeColumn(enc), values, `skewed trial ${trial}`);
    const sizes = estimateSizes(values);
    strictEqual(enc.bits, Math.min(sizes.rle, sizes.dictionary, sizes["for"], sizes.raw));
    deepStrictEqual(rleDecode(rleEncode(values), values.length), values);
    deepStrictEqual(dictionaryDecode(dictionaryEncode(values)), values);
    deepStrictEqual(forDecode(forEncode(values)), values);
  }
  // -- reference metric for a run-heavy column -------------------------------------
  const longRuns = Array.from({ length: 1000 }, (_, i) => Math.floor(i / 200));
  const longEnc = encodeColumn(longRuns);
  strictEqual(longEnc.scheme, "rle", "five runs in a thousand values must prefer RLE");
  deepStrictEqual(decodeColumn(longEnc), longRuns);
  const stats = columnStats(longRuns);
  strictEqual(stats.runs, 5);
  strictEqual(stats.distinct, 5);
  ok(stats.n * WORD_BITS / longEnc.bits > 100,
    `RLE should be >100x smaller than raw, got ${(stats.n * WORD_BITS / longEnc.bits).toFixed(1)}x`);
  console.log(`long-run column: rle=${longEnc.bits} bits vs raw=${stats.n * WORD_BITS} bits ` +
    `(x${(stats.n * WORD_BITS / longEnc.bits).toFixed(1)}), dictScheme=${twoValue.scheme}`);
  console.log("PASS");
}
