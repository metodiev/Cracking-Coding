/** Task 416 - Implement bitset DP optimizations for subset problems. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Subset-sum reachability with a bitset: bit s of `mask` is set iff some subset of the
 * processed values sums to s. Adding value v shifts the whole reachable set by v:
 *     mask |= mask << v        (masked to the [0, limit] window after every step)
 * JavaScript numbers cannot address bit 4000 or 20000 (and Number arithmetic cannot
 * hold those integers), so the bitset is a BigInt — one never mixes Number literals
 * into BigInt arithmetic here. The classic boolean DP is the oracle and the reachable
 * sets must match exactly.
 */
"use strict";

function reachableMask(values, limit) {
  let mask = 1n;                                  // the empty subset sums to 0
  const window = (1n << BigInt(limit + 1)) - 1n;
  for (const v of values) {
    if (v <= limit) {
      mask = (mask | (mask << BigInt(v))) & window;
    }
  }
  return mask & window;
}

function canMake(values, target, limit = target) {
  if (target < 0 || target > limit) return false;
  const mask = reachableMask(values, limit);
  return ((mask >> BigInt(target)) & 1n) === 1n;
}

function canPartition(values) {
  const total = values.reduce((acc, v) => acc + v, 0);
  if (total % 2 !== 0) return false;
  return canMake(values, total / 2);
}

/** Oracle: plain boolean DP over sums 0..limit. */
function classicDp(values, limit) {
  const reach = new Array(limit + 1).fill(false);
  reach[0] = true;
  for (const v of values) {
    if (v > limit) continue;
    for (let s = limit; s >= v; s--) {
      if (reach[s - v]) reach[s] = true;
    }
  }
  return reach;
}

let seed = 416;

function lcgValues(count, lo, hi) {
  const values = [];
  for (let k = 0; k < count; k++) {
    seed = (seed * 75 + 74) % 65537;
    values.push(lo + (seed % (hi - lo + 1)));
  }
  return values;
}

function maskBitLength(mask) {
  return mask.toString(2).length;
}

function maskCount(mask) {
  return mask.toString(2).split("").filter((c) => c === "1").length;
}

module.exports = { reachableMask, canMake, canPartition, classicDp, lcgValues, maskBitLength, maskCount };

if (require.main === module) {
  const assert = require("node:assert");

  const base = [3, 34, 4, 12, 5, 2];
  // classic subset-sum hand cases
  assert.strictEqual(canMake(base, 9), true);      // 3 + 4 + 2 (or 5 + 4)
  assert.strictEqual(canMake(base, 30), false);    // sums jump from 26 to 34+
  assert.strictEqual(canMake(base, 34), true);     // the single element 34
  assert.strictEqual(canMake(base, 0), true);      // the empty subset
  assert.strictEqual(canMake(base, 60), true);     // everything
  assert.strictEqual(canMake(base, 59), false);
  assert.strictEqual(canMake(base, 50), true);     // 34 + 12 + 4

  // hand-derived reachable set of {2, 3, 4, 5, 12, 34}: {2,3,4,5} reaches
  // {0, 2..12, 14} (1 and 13 stay empty), adding 12 fills {0, 2..12, 14..24, 26}
  // (13 and 25 stay empty), and adding 34 mirrors that set 34 higher:
  // {0, 2..12, 14..24, 26, 34, 36..46, 48..58, 60} — 48 values.
  const mask = reachableMask(base, 60);
  assert.strictEqual(maskCount(mask), 48);
  for (const s of [0, 2, 12, 14, 24, 26, 34, 36, 46, 48, 58, 60]) {
    assert.strictEqual((mask >> BigInt(s)) & 1n, 1n, `hand present ${s}`);
  }
  for (const s of [1, 13, 25, 35, 47, 59]) {
    assert.strictEqual((mask >> BigInt(s)) & 1n, 0n, `hand absent ${s}`);
  }
  assert.strictEqual(canMake(base, 25), false);

  // equal-partition hand cases
  assert.strictEqual(canPartition([1, 5, 11, 5]), true);   // 11 vs 1 + 5 + 5
  assert.strictEqual(canPartition([1, 5, 3]), false);      // odd total
  assert.strictEqual(canPartition([2, 2, 2, 2]), true);

  // oracle agreement on hand and LCG sets
  const handSets = [[3, 34, 4, 12, 5, 2], [1, 5, 11, 5], [2, 2, 2, 2], [7], []];
  const handLimits = [60, 22, 8, 7, 5];
  for (let h = 0; h < handSets.length; h++) {
    const m = reachableMask(handSets[h], handLimits[h]);
    const oracle = classicDp(handSets[h], handLimits[h]);
    for (let s = 0; s <= handLimits[h]; s++) {
      assert.strictEqual(((m >> BigInt(s)) & 1n) === 1n, oracle[s], `hand oracle set ${h} sum ${s}`);
    }
  }

  for (let round = 0; round < 8; round++) {
    const n = 10 + (seed % 31);
    const values = lcgValues(n, 1, 60);
    const sum = values.reduce((acc, v) => acc + v, 0);
    const limit = Math.min(sum, 1500);
    const m = reachableMask(values, limit);
    const oracle = classicDp(values, limit);
    for (let s = 0; s <= limit; s++) {
      assert.strictEqual(((m >> BigInt(s)) & 1n) === 1n, oracle[s], `lcg ${round} sum ${s}`);
    }
    assert.ok(maskBitLength(m) <= limit + 1, `lcg mask length ${round}`);
    let running = 0;
    for (const v of values) {
      running += v;
      if (running <= limit) assert.ok(((m >> BigInt(running)) & 1n) === 1n, `prefix sum ${round}`);
    }
    assert.strictEqual(canMake(values, limit + 1, limit), false, `truncated target ${round}`);
  }

  // limits beyond 63 bits: a single 4000-value jump needs bit 4000
  const big = reachableMask([4000], 5000);
  assert.strictEqual((big >> 4000n) & 1n, 1n);
  assert.strictEqual((big >> 4001n) & 1n, 0n);
  assert.strictEqual(maskBitLength(big), 4001);
  assert.strictEqual(canMake([4000], 4000), true);
  assert.strictEqual(canMake([4000], 3999), false);

  // n = 40 stress: no 32/64-bit integer could hold these masks
  seed = Math.floor(65537 / 2);
  for (let round = 0; round < 3; round++) {
    const values = lcgValues(40, 1, 1000);
    const limit = 20000;
    const m = reachableMask(values, limit);
    const oracle = classicDp(values, limit);
    let reached = 0;
    for (let s = 0; s <= limit; s++) {
      const bit = ((m >> BigInt(s)) & 1n) === 1n;
      assert.strictEqual(bit, oracle[s], `stress ${round} sum ${s}`);
      if (bit) reached++;
    }
    const subsetSum = values.slice(0, 20).reduce((acc, v) => acc + v, 0);
    if (subsetSum <= limit) assert.ok(((m >> BigInt(subsetSum)) & 1n) === 1n, "stress subset sum");
    assert.ok(maskBitLength(m) <= limit + 1, "stress length");
    let running = 0;
    for (const v of values.slice(0, 5)) {
      running += v;
      assert.ok(((m >> BigInt(running)) & 1n) === 1n, "stress prefix");
    }
    assert.ok(reached >= 1, "stress reached");
  }

  // partition on n = 40 inputs (both parities exercised)
  const values40 = Array.from({ length: 40 }, (_, i) => i + 1);   // total 820, even
  assert.strictEqual(canPartition(values40), true);               // 1..n reaches every sum
  const values41 = Array.from({ length: 41 }, (_, i) => i + 1);   // total 861, odd
  assert.strictEqual(canPartition(values41), false);

  console.log("PASS");
}
