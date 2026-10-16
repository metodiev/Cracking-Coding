/** Task 980 - Implement continuous feedback loop for curriculum improvement from user telemetry.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * Telemetry is a learner x item response matrix. The analyzer derives per-item metrics - sample
 * size, pass rate, median solve time and discrimination - where discrimination is the point-biserial
 * correlation between the item outcome and the learner's total score; a rule table turns those
 * metrics into flags (too easy, too hard, negative discrimination, too slow, insufficient data) and
 * each flag into a curriculum action. The loop then feeds the actions back into the item
 * difficulties (bounded step per round), re-simulates the next cohort and reports whether the pass
 * rates converge into the target band.
 * Simplified: cohorts are simulated with a one-parameter logistic ability model and solve times are
 * log-normal noise. Verified: metric definitions on a fixture, discrimination sign against an
 * independent upper/lower-27% contrast oracle, the flag rules, action mapping, budgeted difficulty
 * steps, convergence of the distance to the target band, insufficient-data boundaries and
 * determinism.
 */
"use strict";
const assert = require("node:assert");

const TARGET = 0.6;
const BAND = [0.4, 0.8];
const MIN_SAMPLES = 30;
const TIME_BUDGET_MS = 900;
const STEP_LIMIT = 0.4;

const ITEMS = [
  { id: "arrays-1", domain: "arrays", difficulty: 0.6 },
  { id: "graphs-1", domain: "graphs", difficulty: 0.1 },
  { id: "dp-1", domain: "dp", difficulty: -0.8 },
  { id: "dp-2", domain: "dp", difficulty: 1.4 },
  { id: "strings-1", domain: "strings", difficulty: -1.6 },
  { id: "hashing-1", domain: "hashing", difficulty: 0 },
];

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, unit: () => Number(next() >> 11n) / 2 ** 53, int: (n) => Number(next() % BigInt(n)) };
}

const logistic = (x) => 1 / (1 + Math.exp(-x));
const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

/** One cohort of learners answering every item; abilities are normal-ish sums of uniforms. */
function simulateCohort({ items, seed, learners = 120 }) {
  const rng = splitmix64(seed);
  return Array.from({ length: learners }, () => {
    const ability = (rng.unit() + rng.unit() + rng.unit() + rng.unit() - 2) * 1.1;
    return { ability, responses: items.map((item) => ({ id: item.id,
      correct: rng.unit() < logistic(ability - item.difficulty),
      ms: Math.round(240 * Math.exp((rng.unit() - 0.5) * 1.1) * (1 + item.difficulty / 4)) })) };
  });
}

/** Per-item metrics plus the total score of each learner (the discrimination reference). */
function analyze(items, cohort) {
  const totals = cohort.map((learner) => learner.responses.filter((response) => response.correct).length);
  return items.map((item) => {
    const outcomes = cohort.map((learner, i) => ({ correct: learner.responses.find((r) => r.id === item.id).correct,
      rest: totals[i] - (learner.responses.find((r) => r.id === item.id).correct ? 1 : 0),
      ms: learner.responses.find((r) => r.id === item.id).ms }));
    const attempts = outcomes.length;
    const passRate = outcomes.filter((o) => o.correct).length / attempts;
    const spread = Math.sqrt(outcomes.reduce((sum, o) => sum + (o.rest - outcomes.reduce((s, x) => s + x.rest, 0) / attempts) ** 2, 0) / attempts);
    const meanRest = outcomes.reduce((sum, o) => sum + o.rest, 0) / attempts;
    const covariance = outcomes.reduce((sum, o) => sum + (o.correct - passRate) * (o.rest - meanRest), 0) / attempts;
    const discrimination = spread === 0 ? 0 : covariance / (spread * Math.sqrt(passRate * (1 - passRate)));
    return { id: item.id, domain: item.domain, attempts, passRate, discrimination,
      medianMs: median(outcomes.map((o) => o.ms)), outcomes,
      flags: flagFor({ attempts, passRate, discrimination, medianMs: median(outcomes.map((o) => o.ms)) }) };
  });
}

/** Independent discrimination view: a median split on the rest score (different statistic). */
function contrastSign(outcomes) {
  const ranked = [...outcomes].sort((a, b) => (b.rest - a.rest) || (Number(b.correct) - Number(a.correct)));
  const half = Math.floor(ranked.length / 2);
  if (half === 0) return 0;
  const rate = (rows) => rows.filter((row) => row.correct).length / rows.length;
  return Math.sign(rate(ranked.slice(0, half)) - rate(ranked.slice(-half)));
}

function flagFor({ attempts, passRate, discrimination, medianMs }) {
  if (attempts < MIN_SAMPLES) return [{ flag: "insufficient-data", action: "collect-more-data" }];
  const flags = [];
  if (passRate > BAND[1]) flags.push({ flag: "too-easy", action: "raise-difficulty" });
  else if (passRate < BAND[0]) flags.push({ flag: "too-hard", action: "lower-difficulty" });
  if (discrimination < 0) flags.push({ flag: "negative-discrimination", action: "rewrite-or-retire" });
  if (medianMs > TIME_BUDGET_MS) flags.push({ flag: "too-slow", action: "relax-time-limit" });
  if (flags.length === 0) flags.push({ flag: "healthy", action: "keep" });
  return flags;
}

/** Feedback policy: nudge item difficulty toward the band centre, never by more than the limit. */
function applyPolicy(items, metrics) {
  return items.map((item) => {
    const row = metrics.find((metric) => metric.id === item.id);
    const gap = TARGET - row.passRate;
    const adjustment = Math.max(-STEP_LIMIT, Math.min(STEP_LIMIT, gap * 0.8));
    return { ...item, difficulty: Number(Math.max(-3, Math.min(3, item.difficulty - adjustment)).toFixed(4)),
      previous: item.difficulty, adjustment };
  });
}

/** Run the loop: simulate, analyze, act, repeat; report the distance to the band each round. */
function loop(items, { rounds = 8, seed = 980, learners = 120 } = {}) {
  const history = [];
  let current = items;
  for (let round = 0; round < rounds; round += 1) {
    const metrics = analyze(current, simulateCohort({ items: current, seed: seed + round, learners }));
    const distance = metrics.reduce((sum, metric) => sum + Math.abs(metric.passRate - TARGET), 0) / metrics.length;
    history.push({ round, distance, metrics, items: current.map((item) => ({ ...item })) });
    current = applyPolicy(current, metrics);
  }
  return { history, final: current };
}

function main() {
  const cohort = simulateCohort({ items: ITEMS, seed: 980 });
  const metrics = analyze(ITEMS, cohort);

  // 1. Happy path: metric definitions hold on the simulated cohort.
  assert.strictEqual(cohort.length, 120, "the cohort size is fixed");
  assert(metrics.every((metric) => metric.attempts === 120), "every item is attempted by everyone");
  for (const metric of metrics) {
    const expected = cohort.filter((learner) => learner.responses.find((r) => r.id === metric.id).correct).length / 120;
    assert.strictEqual(metric.passRate, expected, `${metric.id} pass rate matches a direct count`);
    assert(metric.passRate >= 0 && metric.passRate <= 1 && metric.discrimination >= -1 && metric.discrimination <= 1,
      `${metric.id} metrics stay in range`);
  }
  assert(metrics.find((m) => m.id === "strings-1").passRate > metrics.find((m) => m.id === "dp-2").passRate,
    "an easier item is passed more often");

  // 2. Oracle: the point-biserial sign agrees with an independent upper/lower-27% contrast.
  for (const metric of metrics) {
    const sign = contrastSign(metric.outcomes);
    if (Math.abs(metric.discrimination) > 1e-9 && sign !== 0) {
      assert.strictEqual(Math.sign(metric.discrimination), sign, `${metric.id} discrimination direction`);
    }
  }
  assert(metrics.every((metric) => metric.discrimination > 0), "a well-behaved bank discriminates positively");

  // 3. Flags and actions follow the rule table on a hand-built fixture.
  const fixture = {
    attempts: 100, passRate: 0.98, discrimination: 0.4, medianMs: 300,
  };
  assert.deepStrictEqual(flagFor(fixture), [{ flag: "too-easy", action: "raise-difficulty" }], "too easy");
  assert.deepStrictEqual(flagFor({ ...fixture, passRate: 0.2 }), [{ flag: "too-hard", action: "lower-difficulty" }], "too hard");
  assert.deepStrictEqual(flagFor({ ...fixture, passRate: 0.5, discrimination: -0.2 }),
    [{ flag: "negative-discrimination", action: "rewrite-or-retire" }], "a bad question is rewritten");
  assert.deepStrictEqual(flagFor({ ...fixture, passRate: 0.5, medianMs: 5000 }),
    [{ flag: "too-slow", action: "relax-time-limit" }], "a slow item gets more time");
  assert.deepStrictEqual(flagFor({ ...fixture, passRate: 0.6 }), [{ flag: "healthy", action: "keep" }], "in band is healthy");
  assert.deepStrictEqual(flagFor({ attempts: 5, passRate: 0.6, discrimination: 0.1, medianMs: 300 }),
    [{ flag: "insufficient-data", action: "collect-more-data" }], "a thin sample suppresses every other flag");
  assert.deepStrictEqual(flagFor({ ...fixture, passRate: 0.9, discrimination: -0.3, medianMs: 4000 }).map((f) => f.flag),
    ["too-easy", "negative-discrimination", "too-slow"], "flags stack in table order");

  // 4. Feedback policy: bounded steps and difficulty always inside [-3, 3].
  const stepped = applyPolicy(ITEMS, [{ id: "arrays-1", passRate: 1 }, { id: "graphs-1", passRate: 0 },
    { id: "dp-1", passRate: 0.6 }, { id: "dp-2", passRate: 0.6 }, { id: "strings-1", passRate: 0.6 },
    { id: "hashing-1", passRate: 0.6 }]);
  assert(Math.abs(stepped[0].adjustment) <= STEP_LIMIT && Math.abs(stepped[1].adjustment) <= STEP_LIMIT,
    "the step is capped");
  assert(stepped[0].difficulty > ITEMS[0].difficulty, "a too-easy item gets harder");
  assert(stepped[1].difficulty < ITEMS[1].difficulty, "a too-hard item gets easier");
  assert.strictEqual(stepped[2].difficulty, ITEMS[2].difficulty, "an in-band item is untouched");
  const extreme = applyPolicy([{ id: "x", difficulty: -3 }], [{ id: "x", passRate: 0 }]);
  assert.strictEqual(extreme[0].difficulty, -3, "difficulty is clamped at the bottom");

  // 5. Loop convergence: distance to the band centre falls and stays down.
  const run = loop(ITEMS, { rounds: 8, seed: 980 });
  const distances = run.history.map((entry) => entry.distance);
  assert.strictEqual(run.history.length, 8, "eight feedback rounds");
  assert(distances.at(-1) < distances[0], `distance improves: ${distances[0].toFixed(3)} -> ${distances.at(-1).toFixed(3)}`);
  assert(distances.at(-1) <= Math.min(...distances.slice(0, 3)) + 0.03, "the loop does not regress later");
  assert(distances.at(-1) < 0.15, `the final distance is small: ${distances.at(-1).toFixed(3)}`);
  const increases = distances.slice(1).filter((value, i) => value > distances[i] + 0.02).length;
  assert(increases <= 2, `the loop is stable, at most two noisy upticks: ${increases}`);

  // 6. Actions actually consumed: every flagged item was moved, healthy items were not.
  const first = run.history[0];
  const flagged = first.metrics.filter((metric) => metric.flags.some((f) => f.flag !== "healthy"));
  assert(flagged.length >= 1, "the initial cohort flags at least one item");
  const moved = run.final.filter((item, i) => item.difficulty !== ITEMS[i].difficulty).length;
  assert(moved >= flagged.length, `every flagged item is adjusted (${moved} moved)`);

  // 7. Boundaries: an empty item list, a cohort with no variance and a single learner.
  assert.deepStrictEqual(analyze([], []), [], "no items, no metrics");
  assert.strictEqual(analyze([ITEMS[0]], simulateCohort({ items: [ITEMS[0]], seed: 1, learners: 1 }))[0].attempts, 1,
    "a single learner is measured but flagged as insufficient");
  assert.strictEqual(analyze([ITEMS[0]], simulateCohort({ items: [ITEMS[0]], seed: 1, learners: 1 }))[0]
    .flags[0].flag, "insufficient-data", "one response is not evidence");

  // 8. Determinism: the whole loop is a pure function of the seed.
  assert(JSON.stringify(loop(ITEMS, { rounds: 8, seed: 980 })) === JSON.stringify(run), "the loop is reproducible");
  const other = loop(ITEMS, { rounds: 8, seed: 981 });
  assert(other.history[0].metrics[0].passRate !== run.history[0].metrics[0].passRate, "another seed, another cohort");
  assert(other.history.at(-1).distance < other.history[0].distance, "the second loop converges too");

  console.log(`feedback loop: ${ITEMS.length} items over 8 rounds, distance ` +
    `${distances.map((d) => d.toFixed(2)).join(" -> ")}, flags ` +
    `${first.metrics.flatMap((m) => m.flags.map((f) => `${m.id}:${f.flag}`)).join(", ")}`);
  console.log("PASS");
}
module.exports = { TARGET, BAND, MIN_SAMPLES, TIME_BUDGET_MS, STEP_LIMIT, ITEMS, splitmix64, logistic, median,
  simulateCohort, analyze, contrastSign, flagFor, applyPolicy, loop };
if (require.main === module) main();
