/**
 * Task 541 - Implement dynamic programming with divide-and-conquer monotone optimization. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)
 *
 * The classic partition problem: split an array of positive integers into k contiguous,
 * non-empty segments minimising the sum of the squared segment sums.  With prefix sums the
 * transition is dp[t][i] = min over j < i of (dp[t-1][j] + (pre[i] - pre[j])^2), which is the
 * textbook setting for the divide-and-conquer optimisation (a.k.a. monotone-opt D&C, Knuth-style
 * "divide and conquer DP"): the cost is quadrangle-convex, so opt[t][i] is non-decreasing in i
 * and each layer can be filled by a recursive midpoint scheme that only scans the candidate range
 * [optLo, optHi] implied by its neighbours, turning the O(k n^2) table into O(k n log n).
 *
 * Documented checks: the D&C table equals the naive O(k n^2) table entry-for-entry (including the
 * recorded argmins, which are tie-broken by the smallest j in both engines), both equal exhaustive
 * enumeration of every (n-1 choose k-1) cut set on small inputs, the quadrangle inequality
 * cost(a,c) + cost(b,d) <= cost(a,d) + cost(b,c) is verified directly on the prefix sums, the
 * argmin table is asserted monotone non-decreasing in i, the D&C candidate count is asserted
 * strictly smaller than the naive count, reconstructed partitions are audited (k segments,
 * contiguous, cost matching the DP value), and the closed-form cases k = 1 (one big square) and
 * k = n (sum of squares) plus infeasible k > n are covered.
 */
"use strict";

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

function prefixSums(values) {
  const pre = new Array(values.length + 1).fill(0);
  for (let i = 0; i < values.length; i++) pre[i + 1] = pre[i] + values[i];
  return pre;
}

/** Cost of the segment covering indices [a, b) - the squared segment sum (convex in the prefix). */
function segmentCost(pre, a, b) {
  const width = pre[b] - pre[a];
  return width * width;
}

/** Naive O(k n^2) DP; also records argmins and how many candidates it scored. */
function naiveDP(values, k) {
  const n = values.length;
  const pre = prefixSums(values);
  const dp = Array.from({ length: k + 1 }, () => new Array(n + 1).fill(Infinity));
  const opt = Array.from({ length: k + 1 }, () => new Array(n + 1).fill(-1));
  const stats = { evaluations: 0 };
  dp[0][0] = 0;
  for (let t = 1; t <= k; t++) {
    for (let i = 1; i <= n; i++) {
      let best = Infinity;
      let bestJ = -1;
      for (let j = 0; j < i; j++) {
        if (dp[t - 1][j] === Infinity) continue;
        stats.evaluations += 1;
        const candidate = dp[t - 1][j] + segmentCost(pre, j, i);
        if (candidate < best) {
          best = candidate;
          bestJ = j;
        }
      }
      dp[t][i] = best;
      opt[t][i] = bestJ;
    }
  }
  return { dp, opt, pre, stats };
}

/**
 * Divide-and-conquer optimisation.  For layer t the argmin is monotone in i, so the midpoint of a
 * range is solved by a linear scan inside [optLo, optHi] and its argmin splits the recursion.
 */
function divideConquerDP(values, k) {
  const n = values.length;
  const pre = prefixSums(values);
  const dp = Array.from({ length: k + 1 }, () => new Array(n + 1).fill(Infinity));
  const opt = Array.from({ length: k + 1 }, () => new Array(n + 1).fill(-1));
  const stats = { evaluations: 0 };
  dp[0][0] = 0;
  for (let t = 1; t <= k; t++) {
    const solve = (lo, hi, optLo, optHi) => {
      if (lo > hi) return;
      const mid = (lo + hi) >> 1;
      const upper = Math.min(optHi, mid - 1);
      let best = Infinity;
      let bestJ = -1;
      for (let j = optLo; j <= upper; j++) {
        if (dp[t - 1][j] === Infinity) continue;
        stats.evaluations += 1;
        const candidate = dp[t - 1][j] + segmentCost(pre, j, mid);
        if (candidate < best) {
          best = candidate;
          bestJ = j;
        }
      }
      dp[t][mid] = best;
      opt[t][mid] = bestJ;
      solve(lo, mid - 1, optLo, bestJ);
      solve(mid + 1, hi, bestJ, optHi);
    };
    solve(1, n, 0, n - 1);
  }
  return { dp, opt, pre, stats };
}

/** Recovers the cut positions from an argmin table. */
function reconstruct(opt, k, n) {
  const cuts = [];
  let i = n;
  for (let t = k; t >= 1; t--) {
    const j = opt[t][i];
    if (j < 0) return null;
    cuts.push([j, i]);
    i = j;
  }
  cuts.reverse();
  return cuts;
}

function partitionCost(values, cuts) {
  const pre = prefixSums(values);
  let total = 0;
  for (const [a, b] of cuts) total += segmentCost(pre, a, b);
  return total;
}

/** Exhaustive enumeration of every way to cut into exactly k non-empty segments. */
function brutePartition(values, k) {
  const n = values.length;
  if (k <= 0) return k === 0 ? 0 : Infinity;
  if (k > n) return Infinity;
  const pre = prefixSums(values);
  let best = Infinity;
  let bestCuts = null;
  const cuts = [];
  const rec = (start, remaining) => {
    if (remaining === 1) {
      cuts.push([start, n]);
      const total = partitionCost(values, cuts);
      if (total < best) {
        best = total;
        bestCuts = cuts.slice();
      }
      cuts.pop();
      return;
    }
    for (let end = start + 1; end <= n - remaining + 1; end++) {
      cuts.push([start, end]);
      rec(end, remaining - 1);
      cuts.pop();
    }
  };
  rec(0, k);
  return { cost: best, cuts: bestCuts, pre };
}

function randomValues(n, seed, limit = 9) {
  const values = [];
  let s = seed % 65537;
  for (let i = 0; i < n; i++) {
    s = lcgNext(s);
    values.push((s % limit) + 1);
  }
  return values;
}

function quadrangleHolds(pre, a, b, c, d) {
  return segmentCost(pre, a, c) + segmentCost(pre, b, d) <=
    segmentCost(pre, a, d) + segmentCost(pre, b, c) + 1e-9;
}

function main() {
  const assert = require("node:assert");

  // (1) Hand-checked: [1,2,3] with k = 2 -> 1^2 + 5^2 = 26 versus 3^2 + 3^2 = 18.
  const hand = [1, 2, 3];
  assert.strictEqual(naiveDP(hand, 2).dp[2][3], 18);
  assert.strictEqual(divideConquerDP(hand, 2).dp[2][3], 18);
  assert.strictEqual(brutePartition(hand, 2).cost, 18);
  assert.strictEqual(naiveDP(hand, 1).dp[1][3], 36, "one segment: (1+2+3)^2");
  assert.strictEqual(divideConquerDP(hand, 3).dp[3][3], 14, "singletons: 1 + 4 + 9");

  // (2) Equivalence of the two engines, against exhaustive partitioning, on small inputs.
  for (let seed = 1; seed <= 6; seed++) {
    const n = 7 + (seed % 3);
    const values = randomValues(n, seed * 31);
    for (let k = 1; k <= n; k++) {
      const fast = divideConquerDP(values, k);
      const slow = naiveDP(values, k);
      const brute = brutePartition(values, k);
      assert.strictEqual(fast.dp[k][n], slow.dp[k][n],
        `seed ${seed} k=${k}: D&C matches the naive table`);
      assert.strictEqual(fast.dp[k][n], brute.cost, `seed ${seed} k=${k}: naive matches brute force`);
      for (let i = 1; i <= n; i++) {
        assert.strictEqual(fast.dp[k][i], slow.dp[k][i], `seed ${seed} k=${k} i=${i}: cell`);
        assert.strictEqual(fast.opt[k][i], slow.opt[k][i],
          `seed ${seed} k=${k} i=${i}: argmin tie-break matches`);
      }
      const cuts = reconstruct(slow.opt, k, n);
      assert.strictEqual(cuts.length, k, "the reconstruction yields exactly k segments");
      assert.strictEqual(cuts[0][0], 0);
      assert.strictEqual(cuts[k - 1][1], n);
      for (let t = 0; t + 1 < k; t++) assert.strictEqual(cuts[t][1], cuts[t + 1][0],
        "segments are contiguous");
      for (const [a, b] of cuts) assert.ok(b > a, "segments are non-empty");
      assert.strictEqual(partitionCost(values, cuts), slow.dp[k][n],
        "the reconstructed partition costs exactly the DP optimum");
    }
  }

  // (3) The monotone-argmin precondition: the quadrangle inequality on the squared-sum cost.
  for (let seed = 11; seed <= 14; seed++) {
    const values = randomValues(9, seed * 17);
    const pre = prefixSums(values);
    for (let a = 0; a < pre.length; a++) {
      for (let b = a; b < pre.length; b++) {
        for (let c = b; c < pre.length; c++) {
          for (let d = c; d < pre.length; d++) {
            assert.ok(quadrangleHolds(pre, a, b, c, d),
              `seed ${seed}: quadrangle inequality fails for (${a},${b},${c},${d})`);
          }
        }
      }
    }
  }

  // (4) The argmin table itself is non-decreasing in i (what D&C exploits).
  for (let seed = 21; seed <= 24; seed++) {
    const values = randomValues(12, seed * 23);
    const { opt } = naiveDP(values, 5);
    for (let t = 1; t <= 5; t++) {
      for (let i = 2; i <= 12; i++) {
        assert.ok(opt[t][i] >= opt[t][i - 1],
          `seed ${seed} t=${t}: argmin must be monotone at i=${i}`);
      }
    }
  }

  // (5) The D&C scheme really is cheaper: strictly fewer candidate evaluations.
  for (let seed = 31; seed <= 33; seed++) {
    const values = randomValues(60, seed * 41);
    const fast = divideConquerDP(values, 6);
    const slow = naiveDP(values, 6);
    assert.strictEqual(fast.dp[6][60], slow.dp[6][60], "large instance: same optimum");
    assert.ok(fast.stats.evaluations < slow.stats.evaluations,
      `D&C must scan fewer candidates (${fast.stats.evaluations} < ${slow.stats.evaluations})`);
    assert.ok(fast.stats.evaluations <= slow.stats.evaluations / 3,
      "and by a clear margin on this instance");
    assert.ok(fast.stats.evaluations <= 2 * 6 * 60 * Math.ceil(Math.log2(60)),
      `and stay within an O(k n log n) budget (${fast.stats.evaluations})`);
  }

  // (6) Closed forms and infeasible inputs.
  const values = [3, 1, 4, 1, 5, 9, 2, 6];
  const total = values.reduce((acc, v) => acc + v, 0);
  assert.strictEqual(naiveDP(values, 1).dp[1][values.length], total * total);
  assert.strictEqual(divideConquerDP(values, 1).dp[1][values.length], total * total);
  const sumSquares = values.reduce((acc, v) => acc + v * v, 0);
  assert.strictEqual(naiveDP(values, values.length).dp[values.length][values.length], sumSquares);
  assert.strictEqual(divideConquerDP(values, values.length).dp[values.length][values.length],
    sumSquares);
  assert.strictEqual(divideConquerDP(values, values.length + 1).dp[values.length + 1][values.length],
    Infinity, "more segments than items is infeasible");
  assert.strictEqual(reconstruct(naiveDP(values, values.length + 1).opt,
    values.length + 1, values.length), null);
  assert.ok(naiveDP([5], 1).dp[1][1] === 25 && divideConquerDP([5], 1).dp[1][1] === 25);
  assert.strictEqual(divideConquerDP([], 0).dp[0][0], 0, "no values, no segments");

  // (7) Determinism.
  const repeat = randomValues(20, 909, 7);
  assert.deepStrictEqual(divideConquerDP(repeat, 4), divideConquerDP(repeat, 4));
  assert.deepStrictEqual(naiveDP(repeat, 4), naiveDP(repeat, 4));
}

module.exports = { lcgNext, prefixSums, segmentCost, naiveDP, divideConquerDP, reconstruct,
  partitionCost, brutePartition, randomValues, quadrangleHolds };

if (require.main === module) {
  main();
  console.log("PASS");
}
