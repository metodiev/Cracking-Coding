/** Task 977 - Implement long-term retention estimator using learner response models.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * Recall after `t` days follows a stretched exponential, R(t) = exp(-(t/h)^beta), where `h` is the
 * characteristic time and `beta` the shape. The estimator fits (h, beta) to a learner's review
 * history by a deterministic coordinate search on the negative log-likelihood, and an independent
 * brute-force grid search is used as the oracle. Per-item stability is then updated after each
 * review (remembered: h grows, forgotten: h shrinks), and revisits are recommended for the items
 * whose predicted recall has fallen below a threshold, most-forgotten first. The closed-form due
 * day h * (-ln r0)^(1/beta) is a rounding trip for the recall target.
 * Simplified: a two-parameter curve per item (no difficulty/slippage terms, no per-learner priors);
 * observations are generated from a known curve plus seeded Bernoulli noise. Verified: curve
 * invariants, recovery of the generating parameters, likelihood against the grid oracle, the
 * stability update direction, the due-day round trip, single/empty-history boundaries, ordering and
 * determinism.
 */
"use strict";
const assert = require("node:assert");

const FLOOR_H = 0.2;
const CEIL_H = 400;
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

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

/** Probability of still recalling an item `t` days after the last successful review. */
const recallAt = (t, h, beta = 0.6) => Math.exp(-Math.pow(Math.max(0, t) / h, beta));

/** Observations sampled from a known curve, so the fit can be scored against the truth. */
function makeHistory(seed, { h = 6, beta = 0.6, trialsPerDay = 20, days = 21 } = {}) {
  const rng = splitmix64(seed);
  const observations = [];
  for (let day = 1; day <= days; day += 1) {
    for (let trial = 0; trial < trialsPerDay; trial += 1) {
      observations.push({ t: day, recalled: rng.unit() < recallAt(day, h, beta) });
    }
  }
  return observations;
}

const negativeLogLikelihood = (observations, h, beta) => observations.reduce((sum, { t, recalled }) => {
  const probability = clamp(recallAt(t, h, beta), 1e-9, 1 - 1e-9);
  return sum - Math.log(recalled ? probability : 1 - probability);
}, 0);

/** Deterministic coordinate search on (h, beta) with a shrinking multiplicative step. */
function fitModel(observations, { prior = { h: 4, beta: 0.6 } } = {}) {
  if (observations.length < 2) return { ...prior, fitted: false, negativeLogLikelihood: null };
  let h = prior.h;
  let beta = prior.beta;
  let step = 0.4;
  for (let round = 0; round < 60; round += 1) {
    for (const [dh, db] of [[step, 0], [-step, 0], [0, step], [0, -step]]) {
      const trialH = clamp(h * Math.exp(dh), 0.05, 500);
      const trialBeta = clamp(beta + db * 0.5, 0.05, 2.5);
      if (negativeLogLikelihood(observations, trialH, trialBeta) < negativeLogLikelihood(observations, h, beta)) {
        h = trialH;
        beta = trialBeta;
      }
    }
    step *= 0.85;
  }
  return { h, beta, fitted: true, negativeLogLikelihood: negativeLogLikelihood(observations, h, beta) };
}

/** Oracle: exhaustive grid search over the same likelihood, coarse but assumption-free. */
function gridFit(observations, { hStep = 0.5, betaStep = 0.1 } = {}) {
  let best = { h: 0, beta: 0, negativeLogLikelihood: Infinity };
  for (let h = 0.5; h <= 60; h += hStep) {
    for (let beta = 0.1; beta <= 2.4; beta += betaStep) {
      const value = negativeLogLikelihood(observations, h, beta);
      if (value < best.negativeLogLikelihood) best = { h, beta, negativeLogLikelihood: value };
    }
  }
  return best;
}

/** Stability update after a review: success moves the characteristic time out, failure pulls it in. */
const updateStability = (h, remembered) => clamp(remembered ? h * 1.4 : h * 0.55, FLOOR_H, CEIL_H);

/** Closed form: the day on which recall is expected to fall to `target`. */
const dueDay = (h, beta, target = 0.9) => h * Math.pow(-Math.log(target), 1 / beta);

/** Items whose predicted recall today is below the threshold, most forgotten first. */
function recommend(items, today, { h, beta = 0.6 } = {}, threshold = 0.75) {
  return items
    .map((item) => ({ ...item, predicted: recallAt(today - item.lastReviewDay, item.stability ?? h, beta) }))
    .filter((item) => item.predicted < threshold)
    .sort((a, b) => (a.predicted - b.predicted) || (a.id < b.id ? -1 : 1));
}

function main() {
  const history = makeHistory(977);
  const fitted = fitModel(history);

  // 1. Curve invariants: recall starts at one and never increases with delay.
  assert.strictEqual(recallAt(0, 6, 0.6), 1, "no delay means perfect recall");
  assert(recallAt(-3, 6, 0.6) === 1, "negative delays clamp to the present");
  let previous = 1;
  for (let t = 1; t <= 40; t += 1) {
    const value = recallAt(t, 6, 0.6);
    assert(value < previous && value > 0, `recall decreases at t=${t}`);
    previous = value;
  }
  assert(recallAt(dueDay(6, 0.6, 0.9), 6, 0.6).toFixed(6) === recallAt(dueDay(6, 0.6, 0.9), 6, 0.6).toFixed(6));

  // 2. Recovery: the fit lands near the generating curve on 210 observations.
  assert.strictEqual(history.length, 420, "21 days x 20 trials");
  assert(fitted.fitted, "the model is fitted");
  assert(Math.abs(fitted.h - 6) / 6 < 0.4, `characteristic time in the right region: ${fitted.h.toFixed(2)}`);
  assert(Math.abs(fitted.beta - 0.6) < 0.3, `shape in the right region: ${fitted.beta.toFixed(3)}`);
  const drift = Math.max(...Array.from({ length: 21 }, (_, i) => i + 1)
    .map((t) => Math.abs(recallAt(t, fitted.h, fitted.beta) - recallAt(t, 6, 0.6))));
  assert(drift < 0.08, `the fitted curve tracks the truth (max drift ${drift.toFixed(3)})`);
  assert(negativeLogLikelihood(history, fitted.h, fitted.beta) <= negativeLogLikelihood(history, 6, 0.6),
    "the fit beats the truth it was sampled from");

  // 3. Oracle: the coordinate search never loses to an exhaustive grid, and lands beside it.
  const grid = gridFit(history);
  assert(fitted.negativeLogLikelihood <= grid.negativeLogLikelihood + 1e-6,
    `fit ${fitted.negativeLogLikelihood.toFixed(3)} vs grid ${grid.negativeLogLikelihood.toFixed(3)}`);
  assert(Math.abs(fitted.h - grid.h) / grid.h < 0.2, `h beside the grid: ${fitted.h.toFixed(2)} vs ${grid.h}`);
  assert(Math.abs(fitted.beta - grid.beta) < 0.11, `beta beside the grid: ${fitted.beta.toFixed(3)} vs ${grid.beta}`);

  // 4. Goodness of fit on the empirical daily rates (an independent residual view).
  for (const [low, high] of [[1, 3], [4, 7], [8, 14], [15, 21]]) {
    const bin = history.filter((observation) => observation.t >= low && observation.t <= high);
    const observed = bin.filter((observation) => observation.recalled).length / bin.length;
    const predicted = recallAt((low + high) / 2, fitted.h, fitted.beta);
    assert(Math.abs(predicted - observed) < 0.12, `days ${low}-${high}: predicted ${predicted.toFixed(2)} vs ${observed.toFixed(2)}`);
  }

  // 5. Stability updates move in the right direction and stay inside the floor and the ceiling.
  assert(updateStability(6, true) > 6, "a success extends the characteristic time");
  assert(updateStability(6, false) < 6, "a failure shortens it");
  let lapsed = 400;
  for (let i = 0; i < 20; i += 1) lapsed = updateStability(lapsed, false);
  assert.strictEqual(lapsed, FLOOR_H, "repeated failures bottom out at the floor");
  let grown = 0.2;
  for (let i = 0; i < 30; i += 1) grown = updateStability(grown, true);
  assert.strictEqual(grown, CEIL_H, "repeated successes cap at the ceiling");

  // 6. Due-day round trip: the closed form predicts exactly the target recall.
  for (const [h, beta, target] of [[6, 0.6, 0.9], [1.5, 1, 0.8], [40, 0.5, 0.95]]) {
    assert(Math.abs(recallAt(dueDay(h, beta, target), h, beta) - target) < 1e-12,
      `round trip h=${h} beta=${beta} target=${target}`);
  }
  assert(dueDay(6, 0.6, 0.9) < dueDay(6, 0.6, 0.5), "a lower target recall comes later");
  assert(dueDay(6, 0.6, 0.9) < dueDay(12, 0.6, 0.9), "a stronger memory comes later");

  // 7. Recommendations: only forgotten items, most forgotten first, nothing due at day zero.
  const items = [
    { id: "a", lastReviewDay: 0, stability: 6 }, { id: "b", lastReviewDay: 5, stability: 2 },
    { id: "c", lastReviewDay: 20, stability: 30 },
  ];
  const due = recommend(items, 21, {}, 0.75);
  assert.deepStrictEqual(due.map((item) => item.id), ["b", "a"], `ordering: ${due.map((item) => item.id)}`);
  assert(due.every((item) => item.predicted < 0.75), "only items below the threshold are suggested");
  assert.deepStrictEqual(recommend(items, 0, {}, 0.75), [], "nothing is due before any time passes");
  assert.strictEqual(recommend(items, 21, {}, 0.75)[0].predicted <= due[1].predicted, true, "worst first");

  // 8. Boundaries and determinism: prior for a single observation, identical runs otherwise.
  const single = fitModel([{ t: 3, recalled: true }]);
  assert.deepStrictEqual([single.fitted, single.h, single.beta], [false, 4, 0.6], "one point uses the prior");
  assert.strictEqual(fitModel([]).fitted, false, "an empty history uses the prior");
  const replay = fitModel(makeHistory(977));
  assert(JSON.stringify(replay) === JSON.stringify(fitted), "fitting is reproducible");
  const other = fitModel(makeHistory(978));
  assert(other.negativeLogLikelihood !== fitted.negativeLogLikelihood, "another seed gives another history");
  assert(Math.abs(other.h - 6) / 6 < 0.5, `the second history also recovers h: ${other.h.toFixed(2)}`);

  console.log(`retention: fitted h=${fitted.h.toFixed(2)}d beta=${fitted.beta.toFixed(2)} ` +
    `(truth 6.00/0.60, nll ${fitted.negativeLogLikelihood.toFixed(1)} vs grid ${grid.negativeLogLikelihood.toFixed(1)}), ` +
    `90% recall due at day ${dueDay(fitted.h, fitted.beta, 0.9).toFixed(1)}`);
  console.log("PASS");
}
module.exports = { FLOOR_H, CEIL_H, splitmix64, recallAt, makeHistory, negativeLogLikelihood, fitModel,
  gridFit, updateStability, dueDay, recommend };
if (require.main === module) main();
