/** Task 417 - Implement DP with monotone queue optimization. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Sliding-window minimum DP: jump from j to i at cost a[i], landing windows of size k:
 *     dp[0] = a[0],   dp[i] = a[i] + min(dp[j] for j in [max(0, i - k), i - 1]).
 * The naive transition scans k predecessors (O(n*k) overall). A monotone deque keeps the
 * candidate indices with STRICTLY increasing dp values: expired indices (j < i - k) leave
 * from the front, and while adding i every index with dp >= dp[i] is popped from the back,
 * because it is both older and worse. Every step is amortized O(1).
 * Tie-break: newer index wins (the pop condition keeps equal-valued indices newest-first),
 * which the tests only use through value equality.
 */
"use strict";

/** Returns { dp, pred }; pred[i] is the j achieving the window minimum. */
function slidingMinDp(a, k) {
  const n = a.length;
  const dp = new Array(n).fill(0);
  const pred = new Array(n).fill(-1);
  if (n === 0) return { dp, pred };
  dp[0] = a[0];
  const dq = [0];
  let head = 0;
  for (let i = 1; i < n; i++) {
    while (dq[head] < i - k) head++;
    const j = dq[head];
    dp[i] = a[i] + dp[j];
    pred[i] = j;
    while (dq.length > head && dp[dq[dq.length - 1]] >= dp[i]) dq.pop();
    dq.push(i);
  }
  return { dp, pred };
}

/** Oracle: O(n*k) direct evaluation of the recurrence. */
function slidingMinDpNaive(a, k) {
  const n = a.length;
  if (n === 0) return [];
  const dp = [a[0]];
  for (let i = 1; i < n; i++) {
    let best = Infinity;
    for (let j = Math.max(0, i - k); j < i; j++) best = Math.min(best, dp[j]);
    dp.push(a[i] + best);
  }
  return dp;
}

let seed = 417;

function lcgArray(length, lo, hi) {
  const out = [];
  for (let k = 0; k < length; k++) {
    seed = (seed * 75 + 74) % 65537;
    out.push(lo + (seed % (hi - lo + 1)));
  }
  return out;
}

module.exports = { slidingMinDp, slidingMinDpNaive, lcgArray };

if (require.main === module) {
  const assert = require("node:assert");

  // hand-derived values, a = [10, 1, 5, 7, 3, 2]
  const a = [10, 1, 5, 7, 3, 2];
  const r2 = slidingMinDp(a, 2);
  assert.deepStrictEqual(r2.dp, [10, 11, 15, 18, 18, 20]);
  // at i = 4 the equal-valued index 3 is dropped (newer ties win), so pred[5] = 4
  assert.deepStrictEqual(r2.pred, [-1, 0, 0, 1, 2, 4]);
  assert.strictEqual(r2.dp[4], a[4] + Math.min(r2.dp[2], r2.dp[3]));
  assert.strictEqual(r2.dp[5], a[5] + Math.min(r2.dp[3], r2.dp[4]));

  const r3 = slidingMinDp(a, 3);
  assert.deepStrictEqual(r3.dp, [10, 11, 15, 17, 14, 16]);
  const r1 = slidingMinDp(a, 1);
  assert.deepStrictEqual(r1.dp, [10, 11, 16, 23, 26, 28]);
  assert.deepStrictEqual(r1.pred, [-1, 0, 1, 2, 3, 4]);

  // k = 1 makes dp a running prefix sum of a
  let running = 0;
  for (let i = 0; i < a.length; i++) {
    running += a[i];
    assert.strictEqual(r1.dp[i], running, `prefix sum ${i}`);
  }
  // k >= n opens the whole history
  const big = slidingMinDp(a, 100);
  assert.strictEqual(big.dp[0], 10);
  assert.strictEqual(big.dp[1], 11);
  assert.strictEqual(big.dp[2], 15);
  assert.strictEqual(big.dp[3], 17);
  assert.strictEqual(big.dp[4], 13);
  assert.strictEqual(big.dp[5], 12);

  // every hand value also matches the naive oracle and the predecessor optimality
  for (let k = 1; k <= 7; k++) {
    const res = slidingMinDp(a, k);
    assert.deepStrictEqual(res.dp, slidingMinDpNaive(a, k), `oracle ${k}`);
    for (let i = 1; i < a.length; i++) {
      assert.ok(res.pred[i] >= Math.max(0, i - k) && res.pred[i] < i, `pred range ${k} ${i}`);
      const windowMin = Math.min(...res.dp.slice(Math.max(0, i - k), i));
      assert.strictEqual(res.dp[res.pred[i]], windowMin, `pred optimal ${k} ${i}`);
    }
  }

  // degenerate sizes
  assert.deepStrictEqual(slidingMinDp([], 5).dp, []);
  const single = slidingMinDp([7], 3);
  assert.deepStrictEqual(single.dp, [7]);
  assert.deepStrictEqual(single.pred, [-1]);
  assert.deepStrictEqual(slidingMinDpNaive([], 5), []);
  assert.strictEqual(slidingMinDp([-4, -8], 2).dp[1], -12);

  for (let round = 0; round < 8; round++) {
    const n = 120 + (seed % 180);
    const values = lcgArray(n, -50, 50);
    const k = 1 + (seed % 15);
    const res = slidingMinDp(values, k);
    const dp = res.dp, pred = res.pred;
    assert.deepStrictEqual(dp, slidingMinDpNaive(values, k), `lcg oracle ${round}`);
    for (let i = 1; i < n; i++) {
      const j = pred[i];
      assert.ok(j >= Math.max(0, i - k) && j < i, `lcg pred range ${round}`);
      assert.strictEqual(dp[i], values[i] + dp[j], `lcg recurrence ${round}`);
      const windowMin = Math.min(...dp.slice(Math.max(0, i - k), i));
      assert.strictEqual(dp[j], windowMin, `lcg pred optimal ${round}`);
    }
    // monotone deque invariant: candidate values strictly increase front to back
    const dq = [0];
    let head = 0;
    for (let i = 1; i < n; i++) {
      while (dq[head] < i - k) head++;
      const windowMin = Math.min(...dp.slice(Math.max(0, i - k), i));
      assert.strictEqual(dp[dq[head]], windowMin, `deque front ${round} ${i}`);
      while (dq.length > head && dp[dq[dq.length - 1]] >= dp[i]) dq.pop();
      dq.push(i);
    }
  }

  // window bound: widening k never increases any dp value (only relaxes the minimum)
  const values = lcgArray(150, 0, 30);
  let previous = slidingMinDpNaive(values, 1);
  for (let k = 2; k <= 10; k++) {
    const current = slidingMinDpNaive(values, k);
    for (let i = 0; i < values.length; i++) {
      assert.ok(current[i] <= previous[i], `monotone k ${k} i ${i}`);
    }
    previous = current;
  }

  console.log("PASS");
}
