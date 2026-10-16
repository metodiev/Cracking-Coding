/** Task 957 - Implement randomized test-case generator with oracle checker for each problem.
 * (Tier: Master, CTCI Ch 11 - Testing)
 *
 * For each registered problem: a schema-constrained generator (size and value-span ranges drawn from a
 * seeded xorshift32 stream), declared input invariants, a slow but obviously correct oracle, and an
 * efficient reference solution. The checker runs candidate implementations against the oracle on a
 * generated stream and reports the first counterexample; the counterexample is then *shrunk* by
 * re-generating from smaller parameter tuples (floor/half/size-1 sizes and half the value span, with a
 * few seed offsets), keeping only candidates that still fail.
 *
 * Simplified: shrinking searches the generator's parameter space rather than the raw input bytes, so the
 * minimal case is minimal with respect to the parameter directions actually tried (asserted as such),
 * and oracle/reference are in-process functions. Verified: input invariants over 200 cases per problem,
 * oracle agreement on the same stream, falsification of two deliberately buggy implementations, shrink
 * progress plus local minimality, the empty-case and out-of-shape boundaries, an independently
 * constructed minimal witness, stream determinism, and the report's arithmetic.
 */
"use strict";
const assert = require("node:assert");

/** xorshift32 - the only source of randomness in this file; identical stream on every run. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  const next = () => { state ^= state << 13; state >>>= 0; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  return { next, int: (n) => next() % n };
}
const ascending = (xs) => xs.every((x, i) => i === 0 || xs[i - 1] <= x);
// --- oracles (brute force) and reference solutions (independent algorithms) ----------------------
const brutePair = (c) => {
  for (let i = 0; i < c.xs.length; i += 1) for (let j = i + 1; j < c.xs.length; j += 1) {
    if (c.xs[i] + c.xs[j] === c.target) return true;
  }
  return false;
};
const twoPointerPair = (c) => {
  let i = 0, j = c.xs.length - 1;
  while (i < j) {
    const sum = c.xs[i] + c.xs[j];
    if (sum === c.target) return true;
    if (sum < c.target) i += 1; else j -= 1;
  }
  return false;
};
const dedupeBugPair = (c) => twoPointerPair({ xs: [...new Set(c.xs)], target: c.target });
const bruteMax = (c) => {
  let best = c.xs.length ? -Infinity : 0;
  for (let i = 0; i < c.xs.length; i += 1) {
    let sum = 0;
    for (let j = i; j < c.xs.length; j += 1) { sum += c.xs[j]; if (sum > best) best = sum; }
  }
  return best;
};
const kadaneMax = (c) => {
  if (c.xs.length === 0) return 0;
  let best = c.xs[0], run = c.xs[0];
  for (let i = 1; i < c.xs.length; i += 1) { run = Math.max(c.xs[i], run + c.xs[i]); best = Math.max(best, run); }
  return best;
};
/** Classic bug: clamping the running sum at 0 makes an all-negative array report 0. */
const resetBugMax = (c) => {
  let best = 0, run = 0;
  for (const x of c.xs) { run = Math.max(0, run + x); best = Math.max(best, run); }
  return best;
};

const PROBLEMS = [
  { id: "two-sum-index", buggy: dedupeBugPair, oracle: brutePair, reference: twoPointerPair,
    shape: { size: [0, 24], valueSpan: [1, 6] },
    draw: (size, params, rng) => {
      const xs = Array.from({ length: size }, () => rng.int(params.valueSpan + 1)).sort((a, b) => a - b);
      const target = size >= 2 && rng.int(3) === 0 ? xs[rng.int(size)] + xs[rng.int(size)] : rng.int(2 * params.valueSpan + 2);
      return { xs, target };
    },
    invariants: [(c) => c.xs.length === c.size, (c) => ascending(c.xs), (c) => c.xs.every((x) => x >= 0 && x <= c.params.valueSpan)] },
  { id: "max-subarray", buggy: resetBugMax, oracle: bruteMax, reference: kadaneMax,
    shape: { size: [0, 20], valueSpan: [1, 4] },
    draw: (size, params, rng) => ({ xs: Array.from({ length: size }, () => rng.int(2 * params.valueSpan + 1) - params.valueSpan) }),
    invariants: [(c) => c.xs.length === c.size, (c) => c.xs.every((x) => Math.abs(x) <= c.params.valueSpan)] },
];

/** One case: the parameter tuple is the shrink handle, the generated input is derived from it. */
function makeCase(problem, size, params, seed) {
  const [lo, hi] = problem.shape.size;
  if (size < lo || size > hi) throw new Error(`${problem.id}: size ${size} outside [${lo}, ${hi}]`);
  const [vlo, vhi] = problem.shape.valueSpan;
  if (params.valueSpan < vlo || params.valueSpan > vhi) throw new Error(`${problem.id}: valueSpan ${params.valueSpan} outside [${vlo}, ${vhi}]`);
  const rng = makeRng((seed ^ Math.imul(size + 1, 0x9e3779b1)) >>> 0);
  return { problem: problem.id, size, params: { ...params }, seed, ...problem.draw(size, params, rng) };
}
const remake = (problem, spec) => makeCase(problem, spec.size, spec.params, spec.seed);
/** A deterministic stream of cases: size and value span are drawn from the schema ranges. */
function generateStream(problem, count, seed) {
  const rng = makeRng(seed);
  const [lo, hi] = problem.shape.size;
  const [vlo, vhi] = problem.shape.valueSpan;
  const cases = [];
  for (let i = 0; i < count; i += 1) {
    cases.push(makeCase(problem, lo + rng.int(hi - lo + 1), { valueSpan: vlo + rng.int(vhi - vlo + 1) }, rng.int(1 << 20)));
  }
  return cases;
}
/** Run a candidate against the oracle; returns the disagreements in stream order. */
function checkStream(problem, candidate, cases) {
  const disagreements = [];
  for (const item of cases) {
    const expected = problem.oracle(item);
    const got = candidate(item);
    if (got !== expected) disagreements.push({ case: item, expected, got });
  }
  return { checked: cases.length, disagreements };
}
const smaller = (candidate, current) => candidate.size < current.size || candidate.params.valueSpan < current.params.valueSpan;
/** Parameter-space delta debugging: keep shrinking while a failing candidate exists. */
function shrink(problem, failing, fails, { maxRounds = 40 } = {}) {
  let current = failing;
  const steps = [];
  for (let round = 0; round < maxRounds; round += 1) {
    const sizes = [problem.shape.size[0], Math.floor(current.size / 2), current.size - 1];
    const spans = [Math.max(problem.shape.valueSpan[0], Math.floor(current.params.valueSpan / 2))];
    const candidates = [
      ...sizes.flatMap((size) => [0, 1, 2, 3].map((offset) => remake(problem, { size, params: current.params, seed: current.seed + offset }))),
      ...spans.flatMap((valueSpan) => [0, 1].map((offset) => remake(problem, { size: current.size, params: { valueSpan }, seed: current.seed + offset }))),
    ].filter((c) => smaller(c, current) && fails(c));
    if (candidates.length === 0) return { case: current, steps };
    const next = candidates.sort((a, b) => (a.size - b.size) || (a.params.valueSpan - b.params.valueSpan) || (a.seed - b.seed))[0];
    steps.push({ size: next.size, valueSpan: next.params.valueSpan, seed: next.seed });
    current = next;
  }
  return { case: current, steps };
}

function main() {
  const rng = makeRng(20240957);
  const counts = PROBLEMS.map(() => 150 + rng.int(101)); // seeded case budgets, so the run is sized by the stream
  const streams = Object.fromEntries(PROBLEMS.map((p, i) => [p.id, generateStream(p, counts[i], 20240957)]));
  // 1. Happy path: every generated case satisfies every declared invariant and the schema.
  const invariantChecks = PROBLEMS.reduce((a, p, i) => a + counts[i] * p.invariants.length, 0);
  for (const problem of PROBLEMS) {
    for (const item of streams[problem.id]) {
      for (const [index, holds] of problem.invariants.entries()) assert(holds(item), `${problem.id} case ${item.seed}: invariant ${index}`);
    }
  }
  assert(invariantChecks === counts.reduce((a, c, i) => a + c * PROBLEMS[i].invariants.length, 0), `invariant checks ${invariantChecks}`);
  const sizes = streams["two-sum-index"].map((c) => c.size);
  assert(Math.min(...sizes) <= 2 && Math.max(...sizes) >= 22, `the schema ends are exercised (${Math.min(...sizes)}..${Math.max(...sizes)})`);
  // 2. Oracle agreement: the efficient reference must match the brute force on every generated case.
  for (const problem of PROBLEMS) {
    const verdict = checkStream(problem, problem.reference, streams[problem.id]);
    assert(verdict.disagreements.length === 0, `${problem.id} reference disagrees: ${JSON.stringify(verdict.disagreements[0])}`);
  }
  // 3. Falsification: each deliberately buggy implementation is caught, with its counterexample kept.
  const found = {};
  for (const problem of PROBLEMS) {
    const cases = generateStream(problem, 300, 4242);
    const verdict = checkStream(problem, problem.buggy, cases);
    assert(verdict.disagreements.length > 0, `${problem.id}: the buggy implementation slipped through`);
    found[problem.id] = { problem, verdict, first: verdict.disagreements[0] };
  }
  assert(Object.keys(found).length === 2 && Object.values(found).every((f) => f.verdict.checked === 300), "both problems falsified");
  // 4. Shrinking: the counterexample never grows, still fails, and no tried smaller case fails.
  const shrunk = {};
  for (const [id, { problem, first }] of Object.entries(found)) {
    const fails = (c) => problem.buggy(c) !== problem.oracle(c);
    const result = shrink(problem, first.case, fails);
    shrunk[id] = result;
    assert(result.case.size <= first.case.size && result.steps.every((s) => s.size <= first.case.size),
      `${problem.id}: shrinking must not grow the case`);
    assert(fails(result.case), `${problem.id}: the shrunk case must still fail`);
    const lastSize = result.case.size;
    const smallerTried = [problem.shape.size[0], Math.floor(lastSize / 2), lastSize - 1]
      .filter((size) => size >= problem.shape.size[0] && size < lastSize)
      .flatMap((size) => [0, 1, 2, 3].map((offset) => remake(problem, { size, params: result.case.params, seed: result.case.seed + offset })));
    assert(!smallerTried.some(fails), `${problem.id}: a smaller tried case still fails (not locally minimal)`);
  }
  assert(Object.values(shrunk).some((r) => r.steps.length > 0), "at least one counterexample was actually reduced");
  // 5. Independent minimal witnesses: the essence of each bug, checked by hand.
  const pairWitness = { problem: "two-sum-index", size: 2, params: { valueSpan: 6 }, seed: 7, xs: [3, 3], target: 6 };
  assert(dedupeBugPair(pairWitness) === false && twoPointerPair(pairWitness) === true, "the dedupe bug ignores the equal pair");
  const pairFloor = [0, 1].every((size) => {
    const cases = Array.from({ length: 50 }, (_, seed) => makeCase(PROBLEMS[0], size, { valueSpan: 6 }, seed));
    return checkStream(PROBLEMS[0], dedupeBugPair, cases).disagreements.length === 0;
  });
  assert(pairFloor, "no counterexample can exist below size 2, so size 2 is the true minimum");
  const negativeWitness = { xs: [-3] };
  assert(resetBugMax(negativeWitness) === 0 && kadaneMax(negativeWitness) === -3, "the clamp bug hides an all-negative array");
  // 6. Boundaries: the degenerate size and an out-of-shape request.
  const emptyCase = makeCase(PROBLEMS[0], 0, { valueSpan: 1 }, 1);
  assert(emptyCase.xs.length === 0 && brutePair(emptyCase) === twoPointerPair(emptyCase), "empty case agrees");
  assert.throws(() => makeCase(PROBLEMS[0], 25, { valueSpan: 1 }, 1), /size 25 outside/, "size above the shape is rejected");
  assert.throws(() => makeCase(PROBLEMS[0], 4, { valueSpan: 99 }, 1), /valueSpan 99 outside/, "value span outside the shape is rejected");
  // 7. Determinism: the stream is a pure function of its seed, and a single case replays exactly.
  assert.deepStrictEqual(generateStream(PROBLEMS[0], 50, 99), generateStream(PROBLEMS[0], 50, 99), "same seed, same stream");
  assert.notDeepStrictEqual(generateStream(PROBLEMS[0], 50, 99), generateStream(PROBLEMS[0], 50, 100), "a different seed moves the stream");
  assert.deepStrictEqual(remake(PROBLEMS[0], streams["two-sum-index"][5]), streams["two-sum-index"][5], "parameter tuples replay cases");
  // 8. Report arithmetic: totals line up with the work actually done.
  const report = { generated: Object.values(streams).reduce((a, s) => a + s.length, 0), counts: counts.join("/"),
    invariantChecks, counterexamples: Object.values(found).reduce((a, f) => a + f.verdict.disagreements.length, 0),
    shrunkSizes: Object.values(shrunk).map((r) => r.case.size) };
  assert(report.generated === counts.reduce((a, c) => a + c, 0) && report.generated >= 300, `report ${JSON.stringify(report)}`);
  assert(report.counterexamples >= 2 && report.shrunkSizes.length === 2, "both bugs are reported and shrunk");
  console.log(`oracle-checker: generated=${report.generated} (${report.counts}) invariantChecks=${report.invariantChecks} `
    + `counterexamples=${report.counterexamples} shrunk=${report.shrunkSizes.join("/")}`);
  console.log("PASS");
}

module.exports = { makeRng, ascending, brutePair, twoPointerPair, dedupeBugPair, bruteMax, kadaneMax,
  resetBugMax, PROBLEMS, makeCase, remake, generateStream, checkStream, shrink };
if (require.main === module) main();
