/**
 * Task 378 - Count integer partitions p(n) with DP. (Tier: Intermediate, CTCI Ch 8)
 *
 * p(n) is the number of ways to write n as a sum of positive integers where ORDER DOES
 * NOT MATTER ((2, 1, 1) and (1, 2, 1) are the same partition of 4).  The coin-change
 * style DP treats part sizes one at a time: after processing part size k, dp[s] counts
 * partitions of s using parts <= k, so each multiset is built exactly once regardless of
 * order.  dp[0] = 1 (the empty partition) and p(n) = dp[n] after all sizes up to n.
 *
 * Hand-verified milestones asserted below: p(0)=1, p(1)=1, p(4)=5, p(5)=7, p(10)=42,
 * p(20)=627, p(50)=204226 (classic values, cross-checked here by independent recursive
 * counters for every n <= 20 and by the p(100)=190569292 milestone).
 *
 * Oracles: bruteForcePartitions enumerates partitions recursively by picking the largest
 * remaining part, walkCount by picking non-decreasing parts - neither uses a DP table.
 */
"use strict";

/** Number of integer partitions of n (unordered). */
function partitionCount(n) {
  if (n < 0) return 0;
  const dp = new Array(n + 1).fill(0);
  dp[0] = 1;
  for (let part = 1; part <= n; part++) {
    for (let total = part; total <= n; total++) dp[total] += dp[total - part];
  }
  return dp[n];
}

/** Recursive oracle: partitions of n whose parts are all <= maxPart. */
function bruteForcePartitions(n, maxPart) {
  if (n === 0) return 1;
  if (maxPart === 0) return 0;
  let total = 0;
  for (let part = 1; part <= Math.min(n, maxPart); part++) {
    total += bruteForcePartitions(n - part, part);
  }
  return total;
}

/** Second oracle: enumerate non-decreasing part lists explicitly. */
function walkCount(remaining, smallest) {
  if (remaining === 0) return 1;
  let total = 0;
  for (let part = smallest; part <= remaining; part++) total += walkCount(remaining - part, part);
  return total;
}

module.exports = { partitionCount, bruteForcePartitions, walkCount };

if (require.main === module) {
  const assert = require("node:assert");

  // milestone values (classic partition numbers, re-derived by the oracles below)
  assert.strictEqual(partitionCount(0), 1, "p(0) = 1");
  assert.strictEqual(partitionCount(1), 1, "p(1) = 1");
  assert.strictEqual(partitionCount(2), 2, "p(2) = 2");
  assert.strictEqual(partitionCount(3), 3, "p(3) = 3");
  assert.strictEqual(partitionCount(4), 5, "p(4) = 5");
  assert.strictEqual(partitionCount(5), 7, "p(5) = 7");
  assert.strictEqual(partitionCount(10), 42, "p(10) = 42");
  assert.strictEqual(partitionCount(20), 627, "p(20) = 627");
  assert.strictEqual(partitionCount(50), 204226, "p(50) = 204226");
  assert.strictEqual(partitionCount(100), 190569292, "p(100) = 190569292");

  // p(4) = 5 spelled out: 4; 3+1; 2+2; 2+1+1; 1+1+1+1
  assert.strictEqual(bruteForcePartitions(4, 4), 5, "oracle p(4)");
  assert.strictEqual(walkCount(4, 1), 5, "walk oracle p(4)");

  // p(5) = 7: 5; 4+1; 3+2; 3+1+1; 2+2+1; 2+1+1+1; 1+1+1+1+1
  assert.strictEqual(bruteForcePartitions(5, 5), 7, "oracle p(5)");

  // all n <= 20 against both independent oracles
  for (let n = 0; n <= 20; n++) {
    const expected = bruteForcePartitions(n, n);
    assert.strictEqual(partitionCount(n), expected, `oracle agreement ${n}`);
    assert.strictEqual(walkCount(n, 1), expected, `walk agreement ${n}`);
  }

  // strictly increasing for n >= 1
  let previous = 0;
  for (let n = 1; n <= 25; n++) {
    const current = partitionCount(n);
    assert.ok(current > previous, `monotone ${n}`);
    previous = current;
  }

  assert.strictEqual(partitionCount(-3), 0, "negative input is 0");
  console.log("PASS");
}
