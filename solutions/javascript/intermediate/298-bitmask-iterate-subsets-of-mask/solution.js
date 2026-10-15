/** Task 298 - Bitmask: iterate subsets of a mask. (Tier: Intermediate, CTCI Ch 5) */
"use strict";

function enumerateSubmasks(mask) {
  const result = [];
  let sub = mask;
  for (;;) {
    result.push(sub);
    if (sub === 0) break;
    sub = (sub - 1) & mask;
  }
  return result;
}

function popcount(value) {
  let count = 0;
  while (value !== 0) {
    count += value & 1;
    value >>>= 1;
  }
  return count;
}

module.exports = { enumerateSubmasks, popcount };

if (require.main === module) {
  const assert = require("node:assert");

  assert.deepStrictEqual(enumerateSubmasks(0), [0]);
  assert.deepStrictEqual(enumerateSubmasks(1), [1, 0]);
  assert.deepStrictEqual(enumerateSubmasks(0b1011), [11, 10, 9, 8, 3, 2, 1, 0]);
  assert.deepStrictEqual(enumerateSubmasks(0b11111), Array.from({ length: 32 }, (_, i) => 31 - i));

  const masks = [0, 1, 0b1011, 0b11111, 0b1000, 0b101010];
  let seed = 9090;
  for (let i = 0; i < 40; i++) {
    seed = (seed * 75 + 74) % 65537;
    masks.push(seed % 4096);
  }
  for (const mask of masks) {
    const submasks = enumerateSubmasks(mask);
    assert.strictEqual(submasks.length, 2 ** popcount(mask), `count ${mask}`);
    assert.strictEqual(new Set(submasks).size, submasks.length, `duplicates ${mask}`);
    assert.ok(submasks.every((sub) => (sub & ~mask) === 0), `submask property ${mask}`);
    assert.strictEqual(submasks[0], mask, `first ${mask}`);
    assert.strictEqual(submasks[submasks.length - 1], 0, `last ${mask}`);
    for (let i = 0; i + 1 < submasks.length; i++) {
      assert.ok(submasks[i] > submasks[i + 1], `descending ${mask}`);
    }
    const oracle = [];
    for (let value = mask; value >= 0; value--) if ((value & ~mask) === 0) oracle.push(value);
    assert.deepStrictEqual(submasks, oracle, `oracle ${mask}`);
  }
  console.log("PASS");
}
