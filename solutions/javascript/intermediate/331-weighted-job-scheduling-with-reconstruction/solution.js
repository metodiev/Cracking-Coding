/** Task 331 - Weighted job scheduling with reconstruction. (Tier: Intermediate, CTCI Ch 8) */
"use strict";

/**
 * Jobs are (start, end, weight) half-open intervals, so end == next start is not an overlap.
 * Sort by end time, let p[i] be the last job finishing at or before job i starts, then
 * dp[i] = max(dp[i-1], weight[i] + dp[p[i]]). Descending through the same dp/pred pair
 * reconstructs one optimal set. Cross-checked against exhaustive subset search.
 */
function weightedSchedule(jobs) {
  const n = jobs.length;
  if (n === 0) return { value: 0, chosen: [] };
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) =>
    jobs[a][1] !== jobs[b][1] ? jobs[a][1] - jobs[b][1] : jobs[a][0] - jobs[b][0]);
  const ends = order.map((i) => jobs[i][1]);

  const pred = new Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    let lo = 0, hi = i;
    while (lo < hi) {                              // first index whose end exceeds this start
      const mid = (lo + hi) >> 1;
      if (ends[mid] <= jobs[order[i]][0]) lo = mid + 1;
      else hi = mid;
    }
    pred[i] = lo - 1;
  }

  const dp = new Array(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    dp[i] = Math.max(dp[i - 1], jobs[order[i - 1]][2] + dp[pred[i - 1] + 1]);
  }

  const chosen = [];
  let i = n;
  while (i > 0) {
    if (dp[i] > dp[i - 1]) {                       // job order[i-1] belongs to this optimum
      chosen.push(order[i - 1]);
      i = pred[i - 1] + 1;
    } else {
      i -= 1;
    }
  }
  chosen.reverse();
  return { value: dp[n], chosen };
}

/** Exhaustive best weight over all subsets (n <= 12). */
function bruteSchedule(jobs) {
  const n = jobs.length;
  let best = 0;
  for (let mask = 0; mask < 1 << n; mask++) {
    let weight = 0;
    const picked = [];
    let valid = true;
    for (let i = 0; i < n && valid; i++) {
      if (!((mask >> i) & 1)) continue;
      for (const j of picked) {
        if (overlaps(jobs[i], jobs[j])) { valid = false; break; }
      }
      if (valid) { picked.push(i); weight += jobs[i][2]; }
    }
    if (valid) best = Math.max(best, weight);
  }
  return best;
}

function overlaps(a, b) {
  return a[0] < b[1] && b[0] < a[1];
}

let seed = 9091;

function lcgJobs(count) {
  const jobs = [];
  for (let i = 0; i < count; i++) {
    seed = (seed * 75 + 74) % 65537;
    const start = seed % 30;
    seed = (seed * 75 + 74) % 65537;
    const length = 1 + (seed % 8);
    seed = (seed * 75 + 74) % 65537;
    jobs.push([start, start + length, 1 + (seed % 20)]);
  }
  return jobs;
}

function isNonOverlapping(jobs, indices) {
  for (let i = 0; i < indices.length; i++) {
    for (let j = i + 1; j < indices.length; j++) {
      if (overlaps(jobs[indices[i]], jobs[indices[j]])) return false;
    }
  }
  return true;
}

module.exports = { weightedSchedule, bruteSchedule, overlaps };

if (require.main === module) {
  const assert = require("node:assert");

  // Hand-checked: (1,3,w5) + (4,6,w5) + (6,7,w4) = 14.
  const hand = [[1, 3, 5], [2, 5, 6], [4, 6, 5], [6, 7, 4]];
  const result = weightedSchedule(hand);
  assert.strictEqual(result.value, 14);
  assert.deepStrictEqual(result.chosen, [0, 2, 3]);
  assert.strictEqual(result.chosen.reduce((sum, i) => sum + hand[i][2], 0), 14);
  assert.ok(isNonOverlapping(hand, result.chosen));
  assert.strictEqual(result.value, bruteSchedule(hand));

  assert.deepStrictEqual(weightedSchedule([]), { value: 0, chosen: [] });
  assert.deepStrictEqual(weightedSchedule([[5, 9, 7]]), { value: 7, chosen: [0] });

  // Touching intervals (end == next start) do not overlap.
  assert.strictEqual(weightedSchedule([[0, 5, 3], [5, 10, 4]]).value, 7);

  // A single heavy job beats two light ones.
  const heavy = weightedSchedule([[0, 10, 100], [0, 4, 3], [6, 10, 3]]);
  assert.deepStrictEqual(heavy, { value: 100, chosen: [0] });

  for (let trial = 0; trial < 200; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const n = 1 + (seed % 12);
    const jobs = lcgJobs(n);
    const schedule = weightedSchedule(jobs);
    assert.strictEqual(schedule.value, bruteSchedule(jobs), `random value ${trial}`);
    assert.ok(isNonOverlapping(jobs, schedule.chosen), `random feasibility ${trial}`);
    assert.strictEqual(schedule.chosen.reduce((sum, i) => sum + jobs[i][2], 0), schedule.value,
      `random weight sum ${trial}`);
  }
  console.log("PASS");
}
