/** Task 219 - Implement meet-in-the-middle for subset problems (n ~ 40). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

function subsetSums(values) {
  const sums = [0];
  for (const x of values) {
    const current = sums.length;
    for (let i = 0; i < current; i++) sums.push(sums[i] + x);
  }
  return sums;
}

function lowerBound(a, value) {
  let lo = 0, hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (a[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

class MeetInTheMiddle {
  constructor(nums) {
    const mid = Math.floor(nums.length / 2);
    this.left = subsetSums(nums.slice(0, mid));
    this.right = subsetSums(nums.slice(mid)).sort((a, b) => a - b);
  }

  exists(target) {
    for (const s of this.left) {
      const i = lowerBound(this.right, target - s);
      if (i < this.right.length && this.right[i] === target - s) return true;
    }
    return false;
  }

  closest(target) {
    let best = null;
    for (const s of this.left) {
      const i = lowerBound(this.right, target - s);
      for (const j of [i - 1, i]) {
        if (j >= 0 && j < this.right.length) {
          const candidate = s + this.right[j];
          if (best === null || Math.abs(candidate - target) < Math.abs(best - target)) best = candidate;
        }
      }
    }
    return best;
  }
}

function bruteExists(nums, target) {
  return subsetSums(nums).includes(target);
}

module.exports = { MeetInTheMiddle, subsetSums, bruteExists };

if (require.main === module) {
  const assert = require("node:assert");
  let seed = 4;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };

  const small = Array.from({ length: 16 }, () => 1 + Math.floor(rng() * 99));
  const mitmSmall = new MeetInTheMiddle(small);
  for (let t = 0; t < 200; t++) {
    const target = Math.floor(rng() * 600);
    assert.strictEqual(mitmSmall.exists(target), bruteExists(small, target), `target ${target}`);
  }

  const big = Array.from({ length: 40 }, () => 1 + Math.floor(rng() * 9));
  const mitmBig = new MeetInTheMiddle(big);
  const total = big.reduce((a, b) => a + b, 0);
  assert.strictEqual(mitmBig.exists(total), true);
  assert.strictEqual(mitmBig.exists(total + 1), false);
  assert.strictEqual(mitmBig.exists(0), true);
  console.log("PASS");
}
