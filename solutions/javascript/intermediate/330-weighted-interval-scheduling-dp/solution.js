/** Task 330 - Weighted interval scheduling (DP). (Tier: Intermediate, CTCI Ch 8) */
"use strict";

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
  constructor(seed) {
    this.seed = seed;
  }

  next(modulus) {
    this.seed = (this.seed * 75 + 74) % 65537;
    return this.seed % modulus;
  }
}

/**
 * Jobs are [start, finish, weight]; compatible when one finishes at or before the next
 * starts (finish <= start). Returns { total, chosen }.
 */
function weightedIntervalScheduling(jobs) {
  if (jobs.length === 0) return { total: 0, chosen: [] };
  const ordered = jobs
    .map((job) => job.slice())
    .sort((u, v) => u[1] - v[1] || u[0] - v[0]);
  const finishes = ordered.map((job) => job[1]);

  /** Largest index j < i with finishes[j] <= start; -1 when none exists. */
  const lastCompatible = (start, i) => {
    let lo = 0;
    let hi = i - 1;
    let best = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (finishes[mid] <= start) {
        best = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return best;
  };

  const best = new Array(ordered.length + 1).fill(0);
  const choose = new Array(ordered.length + 1).fill(false);
  for (let i = 1; i <= ordered.length; i++) {
    const without = best[i - 1];
    const withJob = ordered[i - 1][2] + best[lastCompatible(ordered[i - 1][0], i - 1) + 1];
    if (withJob > without) {
      best[i] = withJob;
      choose[i] = true;
    } else {
      best[i] = without;
    }
  }

  const chosen = [];
  let i = ordered.length;
  while (i > 0) {
    if (choose[i]) {
      chosen.push(ordered[i - 1]);
      i = lastCompatible(ordered[i - 1][0], i - 1) + 1;
    } else {
      i--;
    }
  }
  chosen.reverse();
  return { total: best[ordered.length], chosen };
}

/** Exhaustive subset enumeration; the independent oracle for small inputs. */
function bruteForce(jobs) {
  const n = jobs.length;
  let best = 0;
  for (let mask = 0; mask < 1 << n; mask++) {
    const chosen = [];
    let total = 0;
    for (let i = 0; i < n; i++) {
      if ((mask >> i) & 1) {
        chosen.push(jobs[i]);
        total += jobs[i][2];
      }
    }
    if (total <= best) continue;
    chosen.sort((u, v) => u[0] - v[0] || u[1] - v[1]);
    let compatible = true;
    for (let k = 0; k + 1 < chosen.length; k++) {
      if (chosen[k][1] > chosen[k + 1][0]) compatible = false;
    }
    if (compatible) best = total;
  }
  return best;
}

function lcgJobs(count, lcg) {
  const jobs = [];
  for (let i = 0; i < count; i++) {
    const start = lcg.next(30);
    jobs.push([start, start + 1 + lcg.next(8), 1 + lcg.next(20)]);
  }
  return jobs;
}

module.exports = { weightedIntervalScheduling, bruteForce, lcgJobs, Lcg };

if (require.main === module) {
  const assert = require("node:assert");
  // the job set from the task description: (1,3)+(4,6)+(6,7) = 5+5+4 = 14, not 12
  const briefJobs = [[1, 3, 5], [2, 5, 6], [4, 6, 5], [6, 7, 4]];
  let result = weightedIntervalScheduling(briefJobs);
  assert.strictEqual(result.total, 14, "brief job set is 14");
  assert.deepStrictEqual(result.chosen, [[1, 3, 5], [4, 6, 5], [6, 7, 4]]);
  assert.strictEqual(bruteForce(briefJobs), 14);

  // hand-solved example: (1,3,5)+(3,5,4)+(5,7,3) = 12; the heavy decoys lose
  const hand = [[1, 3, 5], [3, 5, 4], [5, 7, 3], [1, 7, 10], [2, 3, 2]];
  result = weightedIntervalScheduling(hand);
  assert.strictEqual(result.total, 12, "hand example totals 12");
  assert.deepStrictEqual(result.chosen, [[1, 3, 5], [3, 5, 4], [5, 7, 3]]);
  assert.strictEqual(bruteForce(hand), 12);

  assert.deepStrictEqual(weightedIntervalScheduling([]), { total: 0, chosen: [] });
  result = weightedIntervalScheduling([[4, 9, 7]]);
  assert.strictEqual(result.total, 7);
  assert.deepStrictEqual(result.chosen, [[4, 9, 7]]);

  const lcg = new Lcg(57721);
  for (let trial = 0; trial < 60; trial++) {
    const count = 4 + lcg.next(6);
    const jobs = lcgJobs(count, lcg);
    result = weightedIntervalScheduling(jobs);
    assert.strictEqual(result.total, bruteForce(jobs), `trial ${trial}`);
    const weightSum = result.chosen.reduce((sum, job) => sum + job[2], 0);
    assert.strictEqual(weightSum, result.total);
    for (let k = 0; k + 1 < result.chosen.length; k++) {
      assert.ok(result.chosen[k][1] <= result.chosen[k + 1][0], `compatible ${trial}`);
    }
  }
  console.log("PASS");
}
