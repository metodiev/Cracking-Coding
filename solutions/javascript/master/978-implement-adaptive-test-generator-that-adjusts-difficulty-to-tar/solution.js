/** Task 978 - Implement adaptive test generator that adjusts difficulty to target percentile.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * Items follow the Rasch model, p(correct) = 1/(1+exp(-(theta - b))). The generator estimates the
 * examinee's ability by Newton iterations on the Rasch score equation, picks the next item with the
 * maximum Fisher information at that estimate (the difficulty closest to theta), respects a
 * blueprint that forces at least one item per required topic, and stops when the standard error of
 * the estimate (1/sqrt(sum of information)) drops below a target or the form is full. A target
 * percentile is converted to an ability with the normal quantile, which drives the "harder items
 * for a higher target" property.
 * Simplified: 1PL/Rasch only (no discrimination or guessing parameters) and abilities are clamped
 * to [-4, 4] rather than reported as infinite. Verified: ability estimates against a bisection
 * oracle and the score residual, the normal quantile against a table, blueprint satisfaction,
 * non-increasing standard error, adaptive-versus-fixed accuracy, boundaries and determinism.
 */
"use strict";
const assert = require("node:assert");

/** Abramowitz & Stegun 7.1.26 - the normal CDF behind the ability scale. */
function erf(x) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const y = 1 - poly * Math.exp(-x * x);
  return x < 0 ? -y : y;
}
const normCdf = (z) => 0.5 * (1 + erf(z / Math.SQRT2));

/** Independent normal quantile: bisection on the CDF, cross-checked against a hand table below. */
function normQuantile(percentile, { low = -8, high = 8 } = {}) {
  assert(percentile > 0 && percentile < 1, `percentile ${percentile} must lie strictly inside (0,1)`);
  let lo = low;
  let hi = high;
  for (let i = 0; i < 200 && hi - lo > 1e-12; i += 1) {
    const mid = (lo + hi) / 2;
    if (normCdf(mid) < percentile) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}
const QUANTILE_TABLE = [[0.5, 0], [0.75, 0.6745], [0.9, 1.2816], [0.95, 1.6449], [0.99, 2.3263]];

const RASCH = {
  p: (theta, b) => 1 / (1 + Math.exp(-(theta - b))),
  info: (theta, b) => { const p = RASCH.p(theta, b); return p * (1 - p); },
};

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

/** Bank with difficulties spread over the scale and topics cycling through the blueprint. */
function makeBank(seed, count = 24) {
  const rng = splitmix64(seed);
  const topics = ["arrays", "graphs", "dp", "strings"];
  return Array.from({ length: count }, (_, i) => ({
    id: `i${String(i).padStart(2, "0")}`,
    difficulty: Number((-2.4 + (4.8 * i) / (count - 1) + (rng.unit() - 0.5) * 0.3).toFixed(4)),
    topic: topics[i % topics.length],
  }));
}

/** Newton on the score equation, clamped; all-correct/all-wrong patterns saturate at the ends. */
function estimateAbility(responses, { low = -4, high = 4 } = {}) {
  if (responses.length === 0) return { theta: 0, se: Infinity, clamped: false, fitted: false, residual: null };
  const correct = responses.filter((entry) => entry.correct).length;
  const score = (theta) => correct - responses.reduce((sum, entry) => sum + RASCH.p(theta, entry.item.difficulty), 0);
  const information = (theta) => responses.reduce((sum, entry) => sum + RASCH.info(theta, entry.item.difficulty), 0);
  const saturated = correct === responses.length || correct === 0;
  let theta = 0;
  if (saturated) theta = correct === responses.length ? high : low;
  else {
    for (let i = 0; i < 100; i += 1) {
      const step = score(theta) / Math.max(1e-9, information(theta));
      const nextTheta = Math.min(high, Math.max(low, theta + step));
      const done = Math.abs(nextTheta - theta) < 1e-12;
      theta = nextTheta;
      if (done) break;
    }
  }
  return { theta, se: 1 / Math.sqrt(Math.max(1e-9, information(theta))), clamped: saturated, fitted: true,
    residual: score(theta) };
}

/** Oracle: bisection on the same score equation, without Newton heuristics. */
function oracleAbility(responses, { low = -8, high = 8 } = {}) {
  if (responses.length === 0) return null;
  const correct = responses.filter((entry) => entry.correct).length;
  const f = (theta) => responses.reduce((sum, entry) => sum + RASCH.p(theta, entry.item.difficulty), 0) - correct;
  if (f(low) > 0) return low;
  if (f(high) < 0) return high;
  let lo = low;
  let hi = high;
  for (let i = 0; i < 200 && hi - lo > 1e-12; i += 1) {
    const mid = (lo + hi) / 2;
    if (f(mid) < 0) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/** Maximum-information selection, restricted while the blueprint still has uncovered topics.
 * Returns `null` when the remaining bank is empty. */
function selectItem({ bank, used, theta, requiredTopics, covered, slotsLeft }) {
  const remaining = bank.filter((item) => !used.has(item.id));
  const uncovered = requiredTopics.filter((topic) => !covered.has(topic));
  const pool = uncovered.length >= slotsLeft ? remaining.filter((item) => uncovered.includes(item.topic)) : remaining;
  if (pool.length === 0) return null;
  return pool.map((item) => ({ item, info: RASCH.info(theta, item.difficulty) }))
    .sort((a, b) => (b.info - a.info) || (a.item.id < b.item.id ? -1 : 1))[0].item;
}

/** Build a form for a simulated examinee and report the ability estimate it yields. */
function generateForm({ bank, thetaTrue, seed = 978, length = 10, targetSe = 0.45,
  requiredTopics = ["arrays", "graphs", "dp"] }) {
  const rng = splitmix64(seed);
  const used = new Set();
  const covered = new Set();
  const responses = [];
  for (let step = 0; step < length; step += 1) {
    const estimate = estimateAbility(responses);
    if (responses.length > 0 && estimate.se <= targetSe) break;
    const item = selectItem({ bank, used, theta: estimate.theta, requiredTopics, covered,
      slotsLeft: length - responses.length });
    if (item === null) {
      if (responses.length === 0) throw new Error("the bank cannot satisfy the blueprint");
      break;
    }
    used.add(item.id);
    covered.add(item.topic);
    responses.push({ item, correct: rng.unit() < RASCH.p(thetaTrue, item.difficulty) });
  }
  return { responses, items: responses.map((entry) => entry.item), covered: [...covered].sort(),
    estimate: estimateAbility(responses), thetaTrue };
}

/** Baseline: the same examinee answering `length` items drawn uniformly at random from the bank. */
function fixedForm({ bank, thetaTrue, seed = 978, length = 10 }) {
  const rng = splitmix64(seed);
  const responses = [];
  const used = new Set();
  while (responses.length < length) {
    const item = bank[rng.int(bank.length)];
    if (used.has(item.id)) continue;
    used.add(item.id);
    responses.push({ item, correct: rng.unit() < RASCH.p(thetaTrue, item.difficulty) });
  }
  return { responses, estimate: estimateAbility(responses) };
}

function main() {
  const bank = makeBank(978, 24);
  const shortBank = makeBank(978, 6);

  // 1. Happy path: a form is blueprint-complete and lands near the true ability.
  const form = generateForm({ bank, thetaTrue: 1, seed: 978, length: 10 });
  assert(form.responses.length <= 10 && form.responses.length >= 3, `form length ${form.responses.length}`);
  assert(["arrays", "dp", "graphs"].every((topic) => form.covered.includes(topic)), "the blueprint topics are covered");
  assert.strictEqual(new Set(form.items.map((item) => item.id)).size, form.items.length, "no repeated items");
  assert(Math.abs(form.estimate.theta - 1) < 1.5 * form.estimate.se + 0.2,
    `estimate ${form.estimate.theta.toFixed(2)} +/- ${form.estimate.se.toFixed(2)} vs truth 1`);

  // 2. Oracle: the estimate satisfies the score equation and matches bisection.
  for (const entry of [form, generateForm({ bank, thetaTrue: -1.5, seed: 7, length: 8 })]) {
    const oracle = oracleAbility(entry.responses);
    assert(Math.abs(entry.estimate.theta - oracle) < 1e-3, `estimate vs oracle: ${entry.estimate.theta} / ${oracle}`);
    assert(Math.abs(entry.estimate.residual) < 1e-6, `score residual ${entry.estimate.residual}`);
  }
  const even = [{ item: { difficulty: 0 }, correct: true }, { item: { difficulty: 0 }, correct: true },
    { item: { difficulty: 0 }, correct: false }, { item: { difficulty: 0 }, correct: false }];
  assert(Math.abs(estimateAbility(even).theta) < 1e-9, "two of four at b=0 imply theta 0");
  assert(Math.abs(oracleAbility(even)) < 1e-9, "the oracle agrees on the symmetric pattern");

  // 3. Independent quantile oracle: a hand-transcribed table of standard normal quantiles.
  for (const [percentile, expected] of QUANTILE_TABLE) {
    assert(Math.abs(normQuantile(percentile) - expected) < 1e-3, `z(${percentile}) = ${normQuantile(percentile)}`);
    assert(Math.abs(normCdf(normQuantile(percentile)) - percentile) < 1e-9, "CDF/quantile round trip");
  }
  assert(normQuantile(0.5) > -1e-9 && normQuantile(0.5) < 1e-9, "the median is zero");
  assert(normQuantile(0.9) > normQuantile(0.75), "quantiles increase with the percentile");

  // 4. Target percentile drives item difficulty: a 90th-percentile form is harder than a median one.
  const hard = generateForm({ bank, thetaTrue: normQuantile(0.9), seed: 978, length: 10 });
  const middling = generateForm({ bank, thetaTrue: normQuantile(0.5), seed: 978, length: 10 });
  const meanDifficulty = (entry) => entry.items.reduce((sum, item) => sum + item.difficulty, 0) / entry.items.length;
  assert(meanDifficulty(hard) > meanDifficulty(middling) + 0.5,
    `hard form ${meanDifficulty(hard).toFixed(2)} vs median form ${meanDifficulty(middling).toFixed(2)}`);
  assert(hard.estimate.theta > middling.estimate.theta, "the estimates order with the targets");
  assert(Math.abs(middling.estimate.theta - normQuantile(0.5)) < 1.0, "the median learner stays near zero");

  // 5. Information invariant: the standard error never grows as items accumulate.
  const running = [];
  const wedges = form.responses.map((entry) => {
    running.push(entry);
    return estimateAbility(running).se;
  });
  for (let i = 1; i < wedges.length; i += 1) assert(wedges[i] <= wedges[i - 1] + 1e-12, `se grows at item ${i}`);
  assert(wedges.at(-1) <= 0.45 || form.responses.length === 10, "the form stops on target or at full length");

  // 6. Adaptivity pays off: adaptive forms beat fixed-length random forms across six examinees.
  const truths = [-2, -1, 0, 0.5, 1, 2];
  const adaptiveError = truths.reduce((sum, thetaTrue) => sum
    + Math.abs(generateForm({ bank, thetaTrue, seed: 978, length: 10 }).estimate.theta - thetaTrue), 0) / truths.length;
  const fixedError = truths.reduce((sum, thetaTrue) => sum
    + Math.abs(fixedForm({ bank, thetaTrue, seed: 978, length: 10 }).estimate.theta - thetaTrue), 0) / truths.length;
  assert(adaptiveError < fixedError, `adaptive ${adaptiveError.toFixed(2)} < fixed ${fixedError.toFixed(2)}`);
  assert(adaptiveError < 0.9, `adaptive error stays small: ${adaptiveError.toFixed(2)}`);

  // 7. Boundaries: saturation, empty patterns, tiny banks and illegal percentiles.
  const allCorrect = [{ item: { difficulty: 0 }, correct: true }, { item: { difficulty: 0.5 }, correct: true }];
  const top = estimateAbility(allCorrect);
  assert(top.clamped && top.theta === 4, "nobody-is-wrong saturates at the top");
  assert(estimateAbility([{ item: { difficulty: 0 }, correct: false }]).theta === -4, "nobody-is-right saturates low");
  assert.deepStrictEqual(estimateAbility([]), { theta: 0, se: Infinity, clamped: false, fitted: false, residual: null },
    "no responses fall back to the prior");
  assert.strictEqual(oracleAbility([]), null, "the oracle has no opinion on an empty pattern");
  assert.throws(() => normQuantile(0), /strictly inside/, "the 0th percentile is rejected");
  assert.throws(() => normQuantile(1), /strictly inside/, "the 100th percentile is rejected");
  assert.throws(() => generateForm({ bank: [], thetaTrue: 0 }), /cannot satisfy the blueprint/,
    "an empty bank cannot build a form");
  assert.strictEqual(selectItem({ bank: [{ id: "x", difficulty: 0, topic: "arrays" }], used: new Set(["x"]),
    theta: 0, requiredTopics: [], covered: new Set(["arrays"]), slotsLeft: 1 }), null, "an exhausted bank returns null");

  // 8. Determinism: the same seed produces an identical form for the same examinee.
  const replay = generateForm({ bank, thetaTrue: 1, seed: 978, length: 10 });
  assert(JSON.stringify(replay) === JSON.stringify(form), "forms are reproducible");
  const other = generateForm({ bank, thetaTrue: 1, seed: 979, length: 10 });
  assert.strictEqual(other.items[0].id, form.items[0].id, "the first item follows the prior ability, not the seed");
  assert(JSON.stringify(other.responses) !== JSON.stringify(form.responses), "another seed runs another session");
  const shortForm = generateForm({ bank: shortBank, thetaTrue: 0, seed: 3, length: 10 });
  assert.strictEqual(shortForm.items.length, 6, "a short bank caps the form at its size");
  assert(shortForm.covered.length >= 3, "the short form still covers the blueprint topics");

  console.log(`adaptive test: ${form.responses.length} items, theta ${form.estimate.theta.toFixed(2)} ` +
    `+/- ${form.estimate.se.toFixed(2)} (truth 1.00), blueprint ${form.covered.join("/")}, ` +
    `adaptive error ${adaptiveError.toFixed(2)} vs fixed ${fixedError.toFixed(2)}`);
  console.log("PASS");
}
module.exports = { erf, normCdf, normQuantile, QUANTILE_TABLE, RASCH, splitmix64, makeBank, estimateAbility,
  oracleAbility, selectItem, generateForm, fixedForm };
if (require.main === module) main();
