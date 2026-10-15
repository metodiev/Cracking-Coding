/** Task 291 - Partition equal subset (DP, with reconstruction). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function partition(nums) {
  let total = 0;
  for (const value of nums) total += value;
  if (total % 2 !== 0) return null;
  const target = total / 2;
  const n = nums.length;
  const reachable = Array.from({ length: n + 1 }, () => new Array(target + 1).fill(false));
  reachable[0][0] = true;
  for (let i = 1; i <= n; i++) {
    const value = nums[i - 1];
    for (let s = 0; s <= target; s++) {
      reachable[i][s] = reachable[i - 1][s] || (s >= value && reachable[i - 1][s - value]);
    }
  }
  if (!reachable[n][target]) return null;
  const side = [];
  let s = target;
  for (let i = n; i >= 1; i--) {
    const value = nums[i - 1];
    if (s >= value && reachable[i - 1][s - value]) {
      side.push(value);
      s -= value;
    }
  }
  side.reverse();
  return side;
}

function validate(nums, side) {
  const remaining = nums.slice();
  for (const value of side) {
    const index = remaining.indexOf(value);
    if (index === -1) return false;
    remaining.splice(index, 1);
  }
  let sideSum = 0;
  for (const value of side) sideSum += value;
  let restSum = 0;
  for (const value of remaining) restSum += value;
  return sideSum === restSum;
}

function brutePartitionExists(nums) {
  let total = 0;
  for (const value of nums) total += value;
  if (total % 2 !== 0) return false;
  const target = total / 2;
  const n = nums.length;
  for (let mask = 0; mask < 1 << n; mask++) {
    let sum = 0;
    for (let i = 0; i < n; i++) if ((mask >> i) & 1) sum += nums[i];
    if (sum === target) return true;
  }
  return false;
}

function lcgNumbers(n, seed) {
  const nums = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    nums.push(1 + (seed % 15));
  }
  return [nums, seed];
}

module.exports = { partition, validate, brutePartitionExists, lcgNumbers };

if (require.main === module) {
  const assert = require("node:assert");

  const sample = [1, 5, 11, 5];
  const side = partition(sample);
  assert.deepStrictEqual(side, [1, 5, 5]);
  assert.strictEqual(validate(sample, side), true);

  assert.strictEqual(partition([1, 2, 3, 5]), null);
  assert.strictEqual(partition([1, 2]), null);
  assert.deepStrictEqual(partition([]), []);
  assert.strictEqual(partition([2, 2, 2]), null);

  const four = [1, 2, 3, 4];
  const fourSide = partition(four);
  assert.strictEqual(fourSide.reduce((a, b) => a + b, 0), 5);
  assert.strictEqual(validate(four, fourSide), true);

  let seed = 1234;
  for (let trial = 0; trial < 200; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const count = 2 + (seed % 8);
    let nums;
    [nums, seed] = lcgNumbers(count, seed);
    const randomSide = partition(nums);
    assert.strictEqual(randomSide !== null, brutePartitionExists(nums), `trial ${trial}`);
    if (randomSide !== null) assert.strictEqual(validate(nums, randomSide), true, `validation ${trial}`);
  }
  console.log("PASS");
}
