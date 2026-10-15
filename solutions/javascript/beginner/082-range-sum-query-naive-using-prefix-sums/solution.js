/** Task 082 - Range sum query naive using prefix sums. (Tier: Beginner, CTCI Ch 1)
 *
 * Build the prefix sums once in O(n); then every inclusive rangeSum(lo, hi) query
 * is O(1) via prefix[hi + 1] - prefix[lo]. Invalid ranges (lo < 0, hi >= n,
 * lo > hi) throw an Error.
 */
"use strict";

class RangeSumQuery {
  constructor(nums) {
    this.prefix = new Array(nums.length + 1).fill(0);
    for (let i = 0; i < nums.length; i++) this.prefix[i + 1] = this.prefix[i] + nums[i];
  }

  rangeSum(lo, hi) {
    const n = this.prefix.length - 1;
    if (lo < 0 || hi >= n || lo > hi) {
      throw new Error(`invalid range [${lo}, ${hi}] for n=${n}`);
    }
    return this.prefix[hi + 1] - this.prefix[lo];
  }
}

module.exports = { RangeSumQuery };

if (require.main === module) {
  const assert = require("node:assert");
  const rs = new RangeSumQuery([3, 1, 4, 1, 5]);
  assert.strictEqual(rs.rangeSum(0, 4), 14);
  assert.strictEqual(rs.rangeSum(1, 3), 6);
  assert.strictEqual(rs.rangeSum(2, 2), 4);
  assert.strictEqual(rs.rangeSum(0, 0), 3);
  assert.strictEqual(rs.rangeSum(4, 4), 5);
  assert.strictEqual(new RangeSumQuery([7]).rangeSum(0, 0), 7);
  assert.throws(() => rs.rangeSum(-1, 2), "lo < 0");
  assert.throws(() => rs.rangeSum(2, 1), "lo > hi");
  assert.throws(() => rs.rangeSum(1, 5), "hi >= n");
  assert.throws(() => new RangeSumQuery([]).rangeSum(0, 0), "empty array");
  console.log("PASS");
}
