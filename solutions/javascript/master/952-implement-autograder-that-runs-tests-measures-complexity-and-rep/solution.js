/** Task 952 - Implement autograder that runs tests, measures complexity, and reports regression.
 * (Tier: Master, CTCI Ch 11 - Testing)
 *
 * Three independent stages for one graded problem (does a sorted array hold two distinct indices whose
 * values sum to the target?):
 *   1. correctness - the problem's hand-written test battery, each failure reported with its case index;
 *   2. complexity  - the submission runs on a geometric ladder of sizes against a step Counter and the
 *      (n, ops) points are fitted to {1, log n, n, n log n, n^2, n^3} by least-squares scaling; the
 *      smallest worst relative residual wins, so classes are told apart with no timing at all (no wall
 *      clock, no noise, no flakiness);
 *   3. regression  - the profile is compared with a stored baseline: a worse class, or the same class
 *      above (1 + tolerance) x baseline operations, is a regression, so the run is not "pass".
 * Simplified: submissions are in-process functions, not source files, and cost is counted steps rather
 * than nanoseconds. Verified: all six complexity labels, a duplicate-blind submission caught by one test,
 * complexity and cost regressions plus the inclusive tolerance boundary, no false regression against the
 * submission's own baseline, differential agreement with a brute-force oracle, rejection of a problem
 * without tests, deterministic re-grading, and the report's counts.
 */
"use strict";
const assert = require("node:assert");
/** splitmix64 - the only source of randomness in this file; identical stream on every run. */
function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}
/** Counted steps replace the clock: one array probe or comparison is one step. */
class Counter {
  constructor() { this.ops = 0; }
  step(n = 1) { this.ops += n; return this; }
  compare(a, b) { this.step(); return a < b ? -1 : a > b ? 1 : 0; }
}
// --- submissions for the graded problem, each with a complexity known by construction ------------
/** Golden: two pointers over the sorted array, one step per iteration -> Theta(n). */
function twoPointer([xs, target], ctr) {
  let i = 0, j = xs.length - 1;
  while (i < j) {
    ctr.step();
    const sum = xs[i] + xs[j];
    if (sum === target) return true;
    if (sum < target) i += 1; else j -= 1;
  }
  return false;
}
/** Naive: every pair, no early exit on the ladder input -> Theta(n^2). */
function quadratic([xs, target], ctr) {
  for (let i = 0; i < xs.length; i += 1) {
    for (let j = i + 1; j < xs.length; j += 1) { ctr.step(); if (xs[i] + xs[j] === target) return true; }
  }
  return false;
}
/** Correct but padded: the golden scan plus half a redundant pass -> same class, 1.5x the work. */
function padded([xs, target], ctr) {
  const verdict = twoPointer([xs, target], ctr);
  ctr.step(Math.ceil(xs.length / 2));
  return verdict;
}
/** The classic bug: de-duplicating first destroys "two distinct indices holding equal values". */
const dedupeBug = ([xs, target], ctr) => twoPointer([[...new Set(xs)], target], ctr);
/** Independent oracle for differential testing: every ordered pair of distinct indices, brute force. */
const twoSumBrute = (xs, target) => xs.some((x, i) => xs.some((y, j) => i !== j && x + y === target));
// --- complexity fixtures: implementations whose true class the grader must rediscover ------------
function lowerBound([xs, target], ctr) {
  let lo = 0, hi = xs.length;
  while (lo < hi) { ctr.step(); const mid = (lo + hi) >> 1; if (xs[mid] < target) lo = mid + 1; else hi = mid; }
  return lo;
}
function mergeCount([xs], ctr) {
  const a = xs.slice();
  const buffer = new Array(a.length);
  const sortRange = (lo, hi) => {
    if (hi - lo < 2) return;
    const mid = (lo + hi) >> 1;
    sortRange(lo, mid); sortRange(mid, hi);
    let i = lo, j = mid, k = lo;
    while (i < mid && j < hi) { if (ctr.compare(a[i], a[j]) <= 0) { buffer[k] = a[i]; i += 1; } else { buffer[k] = a[j]; j += 1; } k += 1; }
    while (i < mid) { buffer[k] = a[i]; i += 1; k += 1; }
    while (j < hi) { buffer[k] = a[j]; j += 1; k += 1; }
    for (let t = lo; t < hi; t += 1) a[t] = buffer[t];
  };
  sortRange(0, a.length);
  return a;
}
const countZeroTriples = ([xs], ctr) => {
  let count = 0;
  for (let i = 0; i < xs.length; i += 1) for (let j = i + 1; j < xs.length; j += 1) {
    for (let k = j + 1; k < xs.length; k += 1) { ctr.step(); if (xs[i] + xs[j] + xs[k] === 0) count += 1; }
  }
  return count;
};
/** Worst case for the two-sum family: sorted distinct values and an unreachable target. */
function makeArgs(n, rng) {
  const xs = [];
  let v = -n;
  for (let i = 0; i < n; i += 1) { v += 1 + rng.int(3); xs.push(v); }
  return [xs, xs.length ? xs[xs.length - 1] * 2 + 5 : 0];
}
/** A target that is always present, so a binary search probes log2(n) times. */
const logArgs = (n, rng) => { const [xs] = makeArgs(n, rng); return [xs, xs.length ? xs[xs.length >> 1] : 0]; };
const cubicArgs = (n, rng) => [Array.from({ length: n }, () => rng.int(2001) - 1000)];
const LADDER = [32, 64, 128, 256, 512];
const LOG_LADDER = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
const CUBIC_LADDER = [8, 16, 32, 64];
const PROBLEM = {
  id: "two-sum-sorted", sizes: LADDER, makeArgs,
  tests: [
    { args: [[1, 2, 3], 5], expected: true }, { args: [[1, 1, 2], 2], expected: true }, // equal values, distinct indices
    { args: [[1, 2], 3], expected: true }, { args: [[], 0], expected: false }, // boundary: empty array
    { args: [[7], 14], expected: false }, { args: [[-3, 0, 4], 1], expected: true }, // boundary: single element
    { args: [[2, 4, 6], 11], expected: false },
  ],
};
const MODELS = [
  { label: "O(1)", f: () => 1 },
  { label: "O(log n)", f: (n) => Math.log2(n) },
  { label: "O(n)", f: (n) => n },
  { label: "O(n log n)", f: (n) => n * Math.log2(n) },
  { label: "O(n^2)", f: (n) => n * n },
  { label: "O(n^3)", f: (n) => n ** 3 },
];
const RANKS = new Map(MODELS.map((model, index) => [model.label, index]));
const complexityRank = (label) => RANKS.get(label);/** Fit every model by least squares in its scaling constant, keep the smallest worst residual. */
function classifyComplexity(points) {
  let best = null;
  for (const model of MODELS) {
    const scale = points.reduce((a, p) => a + p.ops / model.f(p.n), 0) / points.length;
    const err = Math.max(...points.map((p) => Math.abs(p.ops - scale * model.f(p.n)) / Math.max(1, p.ops)));
    if (!best || err < best.err) best = { label: model.label, scale, err };
  }
  return best;
}
/** Measure one implementation on a size ladder; inputs are derived from (seed, size) only. */
function measure(sizes, build, impl, seed) {
  return sizes.map((n) => {
    const ctr = new Counter();
    impl(build(n, splitmix64((seed ^ Math.imul(n, 0x9e3779b1)) >>> 0)), ctr);
    return { n, ops: ctr.ops };
  });
}
/** Pure regression predicate; `>` (not `>=`) is what makes the tolerance an inclusive budget. */
function isRegression(label, opsAtMax, baseline, tolerance) {
  if (complexityRank(label) > complexityRank(baseline.label)) {
    return { regression: true, reason: `complexity ${baseline.label} -> ${label}` };
  }
  if (opsAtMax > baseline.opsAtMax * (1 + tolerance)) {
    return { regression: true, reason: `cost ${opsAtMax} > ${baseline.opsAtMax} * ${(1 + tolerance).toFixed(2)}` };
  }
  return { regression: false, reason: opsAtMax < baseline.opsAtMax ? "improvement" : "unchanged" };
}
/** Run the tests, measure the profile, then compare it with the baseline when one is supplied. */
function grade(problem, submission, { baseline = null, tolerance = 0.25, sizes = problem.sizes, seed = 20240952 } = {}) {
  if (problem.tests.length === 0) return { id: submission.id, error: "problem has no tests" };
  const failures = [];
  problem.tests.forEach((test, index) => {
    try {
      const got = submission.impl(test.args, new Counter());
      if (got !== test.expected) failures.push({ index, expected: test.expected, got });
    } catch (error) { failures.push({ index, error: error.message }); }
  });
  const points = measure(sizes, problem.makeArgs, submission.impl, seed);
  const fit = classifyComplexity(points);
  const opsAtMax = points[points.length - 1].ops;
  const verdict = baseline ? isRegression(fit.label, opsAtMax, baseline, tolerance) : { regression: false, reason: "no baseline" };
  const passed = failures.length === 0;
  return { id: submission.id, passed, failures, points, label: fit.label, residual: Number(fit.err.toFixed(6)),
    opsAtMax, regression: verdict.regression, reason: verdict.reason,
    status: !passed ? "tests-failed" : verdict.regression ? "regression" : "pass" };
}
/** Human-readable report plus machine counts: the autograder's output surface. */
function summarize(results) {
  const counts = { passed: 0, testFailures: 0, regressions: 0, errors: 0 };
  const lines = results.map((r, index) => {
    if (r.error) { counts.errors += 1; return `#${index + 1} ${r.id}: ERROR ${r.error}`; }
    counts[r.status === "pass" ? "passed" : r.status === "regression" ? "regressions" : "testFailures"] += 1;
    return `#${index + 1} ${r.id}: ${r.status.toUpperCase()} class=${r.label} `
      + `ops@${r.points[r.points.length - 1].n}=${r.opsAtMax} (${r.reason})`;
  });
  return { counts, lines, text: lines.join("\n") };
}
function main() {
  const golden = grade(PROBLEM, { id: "golden", impl: twoPointer });
  const baseline = { label: golden.label, opsAtMax: golden.opsAtMax };
  const regolden = grade(PROBLEM, { id: "golden", impl: twoPointer }, { baseline });
  // 1. Happy path: the two-pointer submission clears every test and does not regress against itself.
  assert(golden.status === "pass" && golden.failures.length === 0 && golden.label === "O(n)",
    `golden: ${golden.status} ${golden.label} failures=${golden.failures.length}`);
  assert(regolden.status === "pass" && regolden.reason === "unchanged", `own baseline: ${regolden.reason}`);
  // 2. The duplicate-blind submission fails exactly the equal-values case (index 1).
  const buggy = grade(PROBLEM, { id: "dedupe-bug", impl: dedupeBug }, { baseline });
  assert(buggy.status === "tests-failed" && buggy.failures.length === 1 && buggy.failures[0].index === 1,
    `equal-values case caught: ${JSON.stringify(buggy.failures)}`);
  assert(buggy.failures[0].expected === true && buggy.failures[0].got === false && buggy.points.length === LADDER.length,
    "expected true / got false, and a failing submission is still profiled");
  // 3. Naive passes the tests but its measured class is a regression against the baseline.
  const naive = grade(PROBLEM, { id: "naive", impl: quadratic }, { baseline });
  assert(naive.passed && naive.regression && naive.status === "regression" && naive.reason.includes("O(n^2)"),
    `complexity regression: ${naive.label} ${naive.reason}`);
  // 4. Padded keeps the class but spends 1.5x the budget, so it is a cost regression.
  const padding = grade(PROBLEM, { id: "padded", impl: padded }, { baseline });
  assert(padding.label === "O(n)" && padding.regression && padding.reason.startsWith("cost"),
    `cost regression: ${padding.label} ${padding.reason}`);
  // 5. Tolerance boundary: exactly 1 + tolerance is inside the budget, one step above is not.
  const budget = Math.floor((1 + 0.25) * golden.opsAtMax);
  assert(isRegression("O(n)", budget, baseline, 0.25).regression === false, "exactly at the budget passes");
  assert(isRegression("O(n)", budget + 1, baseline, 0.25).regression === true, "one step above the budget fails");
  assert(isRegression("O(n)", Math.floor(golden.opsAtMax / 2), baseline, 0.25).reason === "improvement",
    "a cheaper same-class run is an improvement");
  // 6. Complexity labels for six fixtures with known classes, including the log and cubic extremes.
  const fixtures = [
    ["O(1)", LADDER, makeArgs, (args, ctr) => { ctr.step(1); return 1; }],
    ["O(log n)", LOG_LADDER, logArgs, lowerBound],
    ["O(n)", LADDER, makeArgs, twoPointer],
    ["O(n log n)", LADDER, makeArgs, mergeCount],
    ["O(n^2)", LADDER, makeArgs, quadratic],
    ["O(n^3)", CUBIC_LADDER, cubicArgs, countZeroTriples],
  ];
  const labels = fixtures.map(([, ladder, build, impl]) => classifyComplexity(measure(ladder, build, impl, 20240952)).label);
  assert(labels.join("|") === fixtures.map(([label]) => label).join("|"), `labels ${labels.join(", ")}`);
  // 7. Independent oracle: brute force agrees with the golden implementation on 120 seeded cases.
  const rng = splitmix64(20240952);
  let agreements = 0;
  for (let trial = 0; trial < 120; trial += 1) {
    const [xs] = makeArgs(4 + rng.int(20), rng);
    const target = trial % 3 === 0 ? xs[xs.length - 1] + xs[0] : xs[rng.int(xs.length)] + xs[rng.int(xs.length)];
    if (twoPointer([xs, target], new Counter()) === twoSumBrute(xs, target)) agreements += 1;
  }
  assert.strictEqual(agreements, 120, `brute-force oracle disagrees on ${120 - agreements} cases`);
  // 8. Boundary: a problem without tests is rejected instead of being graded green.
  const empty = grade({ id: "empty", sizes: LADDER, makeArgs, tests: [] }, { id: "any", impl: twoPointer });
  assert.strictEqual(empty.error, "problem has no tests", "an empty battery cannot pass");
  // 9. Report + determinism: counts add up, the text names the regression, re-grading is identical.
  const report = summarize([golden, naive, buggy, padding, empty]);
  assert(report.counts.passed === 1 && report.counts.regressions === 2 && report.counts.testFailures === 1
    && report.counts.errors === 1, `report counts ${JSON.stringify(report.counts)}`);
  assert(report.text.includes("REGRESSION") && report.text.includes("ERROR problem has no tests"), "report text");
  assert.deepStrictEqual(grade(PROBLEM, { id: "naive", impl: quadratic }, { baseline }), naive, "deterministic re-grade");
  console.log(`autograder: ${report.counts.passed} passed, ${report.counts.regressions} regressions, `
    + `${report.counts.testFailures} test failures; labels=${labels.join("/")}`);
  console.log("PASS");
}
module.exports = { splitmix64, Counter, twoPointer, quadratic, padded, dedupeBug, twoSumBrute, lowerBound,
  mergeCount, countZeroTriples, makeArgs, logArgs, cubicArgs, LADDER, LOG_LADDER, CUBIC_LADDER, PROBLEM,
  MODELS, complexityRank, classifyComplexity, measure, isRegression, grade, summarize };
if (require.main === module) main();
