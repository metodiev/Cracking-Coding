/** Task 488 - Implement compression-aware storage layout simulation. (Tier: Advanced, CTCI Ch 14)
 *
 * Value blocks (32 non-negative 16-bit ints each; the last block may be shorter) are encoded with
 * the cheaper of two schemes, chosen by exact byte count with a documented tie rule (RLE first):
 * - RLE     : header 2 bytes (mode=0, run count) + 4 bytes per run (value, run length).
 * - BITPACK : header 2 bytes (mode=1, bit width w) + ceil(count * w / 8) bytes, LSB-first packing,
 *             where w = max(1, max bit length of the block's values). For values < 2^31 bit packing
 *             with w <= 31 bits is never worse than a raw 32-bit copy, so raw mode is not needed.
 *
 * "Compression-aware layout": blocks are grouped by similarity - a stable sort by bit width puts
 * similar blocks next to each other, and consecutive blocks sharing a width cost one 2-byte group
 * header for the whole run instead of per-block overhead. storedBytes counts group headers plus
 * per-block payloads only.
 *
 * The test asserts exact round-trips on skewed, mixed and hand blocks, checks the mode actually
 * chosen per block, and bounds stored bytes on skewed data: for 1024 ints in {0..3} the layout
 * stores 322 bytes vs 4096 raw (<= raw/3), matching the pinned exact count.
 */
"use strict";

const MODE_RLE = 0;
const MODE_BITPACK = 1;
const HEADER = 2;

function bitLength(value) {
  return value === 0 ? 0 : 32 - Math.clz32(value);
}

function encodeBlock(values) {
  const runs = [];
  for (const value of values) {
    if (runs.length > 0 && runs[runs.length - 1][0] === value) runs[runs.length - 1][1] += 1;
    else runs.push([value, 1]);
  }
  const rleSize = HEADER + 4 * runs.length;
  let width = 1;
  for (const value of values) width = Math.max(width, bitLength(value));
  const bitpackSize = HEADER + Math.ceil((values.length * width) / 8);
  if (rleSize <= bitpackSize) {
    const payload = [];
    payload.push(MODE_RLE, runs.length);
    for (const [value, run] of runs) {
      payload.push(value & 0xFF, (value >> 8) & 0xFF, run & 0xFF, (run >> 8) & 0xFF);
    }
    return { mode: MODE_RLE, width: -1, payload, runs: runs.length };
  }
  const payload = [];
  payload.push(MODE_BITPACK, width);
  let acc = 0;
  let bits = 0;
  for (const value of values) {
    acc |= value << bits;
    bits += width;
    while (bits >= 8) {
      payload.push(acc & 0xFF);
      acc >>>= 8;
      bits -= 8;
    }
  }
  if (bits > 0) payload.push(acc & 0xFF);
  if (payload.length !== bitpackSize) throw new Error(`FAILED: bitpack size ${payload.length} ${bitpackSize}`);
  return { mode: MODE_BITPACK, width, payload, runs: -1 };
}

function decodeBlock(info, count) {
  const payload = info.payload;
  if (payload[0] === MODE_RLE) {
    const values = [];
    let at = HEADER;
    for (let i = 0; i < payload[1]; i++) {
      const value = payload[at] | (payload[at + 1] << 8);
      const run = payload[at + 2] | (payload[at + 3] << 8);
      at += 4;
      for (let j = 0; j < run; j++) values.push(value);
    }
    if (values.length !== count) throw new Error("FAILED: rle count");
    return values;
  }
  const width = payload[1];
  const values = [];
  let acc = 0;
  let bits = 0;
  let at = HEADER;
  for (let i = 0; i < count; i++) {
    while (bits < width) {
      acc |= payload[at] << bits;
      at += 1;
      bits += 8;
    }
    values.push(acc & ((1 << width) - 1));
    acc >>>= width;
    bits -= width;
  }
  return values;
}

function buildLayout(blocks) {
  const infos = blocks.map((values) => encodeBlock(values));
  const order = infos.map((_, i) => i);
  order.sort((a, b) => (infos[a].width !== infos[b].width ? infos[a].width - infos[b].width : a - b));
  const entries = [];
  let stored = 0;
  let groups = 0;
  let previousWidth = null;
  for (const index of order) {
    const info = infos[index];
    if (previousWidth === null || info.width !== previousWidth) {
      groups += 1;
      stored += HEADER;
      previousWidth = info.width;
    }
    stored += info.payload.length;
    entries.push({ index, group: groups - 1, ...info });
  }
  return { entries, storedBytes: stored, groups };
}

function decodeLayout(layout, lengths) {
  const blocks = new Array(lengths.length).fill(null);
  for (const entry of layout.entries) {
    if (blocks[entry.index] !== null) throw new Error("FAILED: duplicate index");
    blocks[entry.index] = decodeBlock(entry, lengths[entry.index]);
  }
  return blocks;
}

function chunk(values, size) {
  const blocks = [];
  for (let i = 0; i < values.length; i += size) blocks.push(values.slice(i, i + size));
  return blocks;
}

function lcgValues(count, seed, limit) {
  const values = [];
  for (let i = 0; i < count; i++) {
    seed = (seed * 75 + 74) % 65537;
    values.push(seed % limit);
  }
  return values;
}

module.exports = { encodeBlock, decodeBlock, buildLayout, decodeLayout };

if (require.main === module) {
  const assert = require("node:assert");

  const constant = encodeBlock(new Array(32).fill(7));
  assert.strictEqual(constant.mode, MODE_RLE);
  assert.strictEqual(constant.runs, 1);
  assert.strictEqual(constant.payload.length, 6);
  assert.deepStrictEqual(decodeBlock(constant, 32), new Array(32).fill(7));

  const zeros = encodeBlock(new Array(32).fill(0));
  assert.strictEqual(zeros.mode, MODE_RLE);
  assert.strictEqual(zeros.payload.length, 6);
  assert.deepStrictEqual(decodeBlock(zeros, 32), new Array(32).fill(0));

  const narrowValues = Array.from({ length: 32 }, (_, i) => i % 4);
  const narrow = encodeBlock(narrowValues);
  assert.strictEqual(narrow.mode, MODE_BITPACK);
  assert.strictEqual(narrow.width, 2);
  assert.strictEqual(narrow.payload.length, 2 + 8);
  assert.deepStrictEqual(decodeBlock(narrow, 32), narrowValues);

  const wideValues = Array.from({ length: 32 }, (_, i) => [65535, 0, 12345, 65534][i % 4]);
  const wide = encodeBlock(wideValues);
  assert.strictEqual(wide.mode, MODE_BITPACK);
  assert.strictEqual(wide.width, 16);
  assert.strictEqual(wide.payload.length, 2 + 64);
  assert.deepStrictEqual(decodeBlock(wide, 32), wideValues);

  const countingValues = Array.from({ length: 32 }, (_, i) => i + 1);
  const counting = encodeBlock(countingValues);
  assert.strictEqual(counting.mode, MODE_BITPACK);
  assert.strictEqual(counting.width, 6);
  assert.deepStrictEqual(decodeBlock(counting, 32), countingValues);

  const skewed = lcgValues(1024, 488, 4);
  const blocks = chunk(skewed, 32);
  assert.strictEqual(blocks.length, 32);
  let layout = buildLayout(blocks);
  assert.strictEqual(layout.groups, 1);
  assert.strictEqual(layout.storedBytes, 322);
  assert.ok(layout.storedBytes <= (4 * skewed.length) / 3);
  assert.deepStrictEqual(decodeLayout(layout, blocks.map((b) => b.length)), blocks);

  const mixedValues = lcgValues(512, 489, 4).concat(lcgValues(512, 490, 65536));
  const mixed = chunk(mixedValues, 32);
  assert.strictEqual(mixed.length, 32);
  layout = buildLayout(mixed);
  assert.strictEqual(layout.groups, 2);
  assert.strictEqual(layout.storedBytes, 16 * 10 + 16 * 66 + 2 * 2);
  assert.ok(layout.storedBytes <= (4 * mixedValues.length) / 3);
  assert.deepStrictEqual(decodeLayout(layout, mixed.map((b) => b.length)), mixed);
  let previousWidth = null;
  let groupCount = 0;
  for (const entry of layout.entries) {
    if (previousWidth === null || entry.width !== previousWidth) {
      groupCount += 1;
      previousWidth = entry.width;
    }
  }
  assert.strictEqual(groupCount, layout.groups);

  const shortValues = new Array(33).fill(5);
  const shorts = chunk(shortValues, 32);
  assert.deepStrictEqual(shorts.map((b) => b.length), [32, 1]);
  layout = buildLayout(shorts);
  assert.deepStrictEqual(decodeLayout(layout, [32, 1]), shorts);
  assert.strictEqual(layout.groups, 2);
  assert.strictEqual(layout.storedBytes, 6 + 3 + 2 + 2);
  console.log("PASS");
}
