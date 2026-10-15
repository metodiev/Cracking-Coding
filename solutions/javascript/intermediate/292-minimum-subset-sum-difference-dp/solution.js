/** Task 292 - Minimum subset sum difference (DP, with reconstruction). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function minDifference(nums) {
  let total = 0;
  for (const value of nums) total += value;
  const half = Math.floor(total / 2);
  const n = nums.length;
  const reachable = Array.from({ length: n + 1 }, () => new Array(half + 1).fill(false));
  reachable[0][0] = true;
  for (let i = 1; i <= n; i++) {
    const value = nums[i - 1];
    for (let s = 0; s <= half; s++) {
      reachable[i][s] = reachable[i - 1][s] || (s >= value && reachable[i - 1][s - value]);
    }
  }
  let best = half;
  while (best > 0 && !reachable[n][best]) best--;
  const side = [];
  let s = best;
  for (let i = n; i >= 1; i--) {
    const value = nums[i - 1];
    if (s >= value && reachable[i - 1][s - value]) {
      side.push(value);
      s -= value;
    }
  }
  side.reverse();
  return [total - 2 * best, side];
}

function bruteMinDifference(nums) {
  let total = 0;
  for (const value of nums) total += value;
  const n = nums.length;
  let best = total;
  for (let mask = 0; mask < 1 << n; mask++) {
    let s = 0;
    for (let i = 0; i < n; i++) if ((mask >> i) & 1) s += nums[i];
    best = Math.min(best, Math.abs(total - 2 * s));
  }
  return best;
}

function lcgNumbers(n, seed) {
  const nums = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    nums.push(1 + (seed % 25));
  }
  return [nums, seed];
}

module.exports = { minDifference, bruteMinDifference, lcgNumbers };

if (require.main === module) {
  const assert = require("node:assert");

  let [diff, side] = minDifference([1, 6, 11, 5]);
  assert.strictEqual(diff, 1);
  assert.strictEqual(side.reduce((a, b) => a + b, 0), 11);
  assert.deepStrictEqual(side.slice().sort((a, b) => a - b), [5, 6]);

  [diff, side] = minDifference([3, 3, 3]);
  assert.strictEqual(diff, 3);
  assert.deepStrictEqual(side, [3]);

  [diff, side] = minDifference([1, 2, 3, 4]);
  assert.strictEqual(diff, 0);
  assert.deepStrictEqual(side.slice().sort((a, b) => a - b), [1, 4]);

  [diff, side] = minDifference([7]);
  assert.strictEqual(diff, 7);
  assert.deepStrictEqual(side, []);

  [diff, side] = minDifference([]);
  assert.strictEqual(diff, 0);
  assert.deepStrictEqual(side, []);

  let seed = 777;
  for (let trial = 0; trial < 200; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const count = 1 + (seed % 14);
    let nums;
    [nums, seed] = lcgNumbers(count, seed);
    const [answer, chosen] = minDifference(nums);
    const total = nums.reduce((a, b) => a + b, 0);
    const chosenSum = chosen.reduce((a, b) => a + b, 0);
    assert.strictEqual(answer, bruteMinDifference(nums), `trial ${trial}`);
    assert.strictEqual(total - 2 * chosenSum, answer, `reconstruction ${trial}`);
  }
  console.log("PASS");
}
