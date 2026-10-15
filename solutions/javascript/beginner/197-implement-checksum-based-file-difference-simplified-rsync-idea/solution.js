/** Task 197 - Implement checksum-based file difference (simplified rsync idea). (Tier: Beginner, CTCI Beyond) */
"use strict";

const crypto = require("node:crypto");

const MASK16 = 0xFFFF;

function weakChecksum(data, offset, length) {
  let a = 0;
  let b = 0;
  for (let i = 0; i < length; i++) {
    const value = data[offset + i];
    a = (a + value) & MASK16;
    b = (b + (length - i) * value) & MASK16;
  }
  return a * 65536 + b;
}

function strongHash(data, offset, length) {
  return crypto.createHash("sha256").update(data.subarray(offset, offset + length)).digest("hex");
}

function splitBlocks(data, blockSize) {
  if (blockSize <= 0) throw new Error("block_size must be positive");
  const blocks = [];
  for (let i = 0; i < data.length; i += blockSize) {
    blocks.push(data.subarray(i, Math.min(i + blockSize, data.length)));
  }
  return blocks;
}

function findBlockMatches(a, b, blockSize) {
  const blocks = splitBlocks(a, blockSize);
  const indexByWeak = new Map();
  blocks.forEach((block, index) => {
    const weak = weakChecksum(block, 0, block.length);
    if (!indexByWeak.has(weak)) indexByWeak.set(weak, []);
    indexByWeak.get(weak).push(index);
  });

  const matches = new Map();
  for (let start = 0; start + blockSize <= b.length; start++) {
    const candidates = indexByWeak.get(weakChecksum(b, start, blockSize));
    if (candidates === undefined) continue;
    for (const index of candidates) {
      if (matches.has(index) || blocks[index].length !== blockSize) continue;
      if (strongHash(b, start, blockSize) === strongHash(blocks[index], 0, blockSize)) {
        matches.set(index, start);                           // weak -> strong confirmation
      }
    }
  }

  if (blocks.length > 0) {
    const tailIndex = blocks.length - 1;
    const tail = blocks[tailIndex];
    if (tail.length < blockSize && !matches.has(tailIndex)) {
      const position = b.indexOf(tail);                      // simplified tail lookup
      if (position >= 0) matches.set(tailIndex, position);
    }
  }
  return matches;
}

module.exports = { weakChecksum, strongHash, findBlockMatches };

if (require.main === module) {
  const assert = require("node:assert");
  const a = Buffer.from(Array.from({ length: 40 }, (_, i) => i));
  const sorted = (map) => [...map.entries()].sort((x, y) => x[0] - y[0]);

  assert.deepStrictEqual(sorted(findBlockMatches(a, a, 8)),
    [[0, 0], [1, 8], [2, 16], [3, 24], [4, 32]]);

  const changed = Buffer.from(a);
  changed[10] ^= 0xFF;                                       // one byte inside block 1
  assert.deepStrictEqual(sorted(findBlockMatches(a, changed, 8)),
    [[0, 0], [2, 16], [3, 24], [4, 32]]);

  assert.deepStrictEqual(sorted(findBlockMatches(a, Buffer.alloc(64, 9), 8)), []);
  assert.deepStrictEqual(sorted(findBlockMatches(a, Buffer.concat([Buffer.from("HEADER"), a]), 8)),
    [[0, 6], [1, 14], [2, 22], [3, 30], [4, 38]]);

  assert.deepStrictEqual(sorted(findBlockMatches(Buffer.from("ABABABAB"), Buffer.from("ZZABABABAB"), 4)),
    [[0, 2], [1, 2]]);
  assert.deepStrictEqual(sorted(findBlockMatches(Buffer.from("abc"), Buffer.from("xxabcxx"), 8)), [[0, 2]]);
  assert.deepStrictEqual(sorted(findBlockMatches(Buffer.alloc(0), a, 8)), []);
  assert.notStrictEqual(weakChecksum(Buffer.from("AB"), 0, 2), weakChecksum(Buffer.from("BA"), 0, 2));
  assert.throws(() => findBlockMatches(a, a, 0), /block_size must be positive/);
  console.log("PASS");
}
