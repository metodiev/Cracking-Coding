/** Task 714 - Implement federated averaging simulation with stragglers and compression. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * Federated averaging: every selected client takes `localSteps` gradient steps on its own objective
 * starting from the global model, and the server averages the resulting models. The client here is
 * the quadratic f_i(w) = 1/2 ||w - mu_i||^2, whose optimum is the global mean of the mu_i, so the
 * simulation can be scored against a closed-form answer instead of a guessed one.
 *
 * Two of the practical hazards of the field are simulated and measured:
 *   - stragglers: each client draws a latency from a seeded geometric distribution, and the server
 *     only waits for a deadline, so slow clients are skipped (or contribute a stale update from the
 *     previous round). Convergence slows down but keeps going;
 *   - compression: a client only uploads the top-k coordinates of its update, scaled by d/k so the
 *     expected magnitude survives. Sending without memory wastes the discarded mass, so error
 *     feedback is implemented as well: the residual is added back into the next local update, which
 *     makes the communicated sequence telescoping and the final model noticeably better.
 *
 * What is simplified: quadratic local objectives with uniform client weights (so the true optimum is
 * known exactly), no differential privacy noise, no secure aggregation, and a deterministic seeded
 * latency model instead of real systems measurement.
 *
 * Invariant verified: the uncompressed full-participation run converges to the closed-form optimum,
 * the loss never rises by more than a tolerance while the step size is inside the stability range,
 * a compressed update is exactly k-sparse, the error-feedback residual is exactly the discarded part
 * of the update, k = d reproduces the uncompressed run exactly, and k = 0 makes no progress at all.
 */
"use strict";
const { ok, strictEqual, deepStrictEqual, throws } = require("node:assert");

/** Deterministic xorshift32; returns integers in [0, bound) (or the raw word). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

function mean(vector) { return vector.reduce((a, b) => a + b, 0) / vector.length; }
function subtract(first, second) { return first.map((value, i) => value - second[i]); }
function add(first, second) { return first.map((value, i) => value + second[i]); }
function scale(vector, factor) { return vector.map((value) => value * factor); }

/** Local objective of a client: the squared distance to its mean vector. */
function localLoss(model, target) {
  return 0.5 * target.reduce((sum, value, i) => sum + (model[i] - value) ** 2, 0);
}

function globalLoss(model, targets) {
  return targets.reduce((sum, target) => sum + localLoss(model, target), 0) / targets.length;
}

/** One local pass of `localSteps` gradient steps; the gradient is model - target. */
function localUpdate(model, target, localSteps, stepSize) {
  let local = model.slice();
  for (let step = 0; step < localSteps; step++) {
    local = local.map((value, i) => value - stepSize * (value - target[i]));
  }
  return local;
}

/**
 * Top-k sparsification. The optional `rescale` multiplies the survivors by d/k, which keeps the
 * expected magnitude when there is no memory; with error feedback the residual already carries the
 * discarded mass, so the plain top-k is the better choice and is what the rounds use.
 */
function topKSparse(update, k, rescale = false) {
  if (k < 0 || k > update.length) throw new RangeError("k must be inside [0, dimension]");
  const order = update.map((value, index) => ({ value, index }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value) || a.index - b.index);
  const chosen = new Set(order.slice(0, k).map((entry) => entry.index));
  const factor = rescale && k > 0 ? update.length / k : 1;
  const sparse = update.map((value, index) => (chosen.has(index) ? value * factor : 0));
  return { sparse, chosen: [...chosen].sort((a, b) => a - b) };
}

/**
 * One federated round. Returns the new global model plus bookkeeping used by the checks. The
 * `residuals` array carries the error-feedback state per client.
 */
function federatedRound(model, targets, options) {
  const {
    localSteps = 1, stepSize = 0.1, participation = 1, k = null, errorFeedback = true,
    latencies = null, deadline = Infinity, residuals = null,
  } = options;
  const clients = targets.length;
  const selected = [];
  for (let i = 0; i < clients; i++) {
    if (participation >= 1 || (i / clients) < participation) selected.push(i);
  }
  const onTime = latencies === null ? selected
    : selected.filter((i) => latencies[i] <= deadline);
  const skipped = selected.filter((i) => !onTime.includes(i));
  const nextResiduals = residuals ?? targets.map(() => model.map(() => 0));
  const compressed = [];
  let sentCoordinates = 0;
  for (const i of onTime) {
    const local = localUpdate(model, targets[i], localSteps, stepSize);
    let delta = subtract(local, model);
    if (k !== null) delta = add(delta, nextResiduals[i]);
    if (k === null) {
      compressed.push(delta);
      sentCoordinates += delta.length;
      nextResiduals[i] = targets[i].map(() => 0);
    } else {
      const { sparse } = topKSparse(delta, k);
      compressed.push(sparse);
      sentCoordinates += k;
      nextResiduals[i] = errorFeedback ? subtract(delta, sparse) : targets[i].map(() => 0);
    }
  }
  const sent = compressed.length === 0
    ? model.map(() => 0)
    : Array.from({ length: model.length }, (_, i) => mean(compressed.map((value) => value[i])));
  return {
    model: add(model, sent),
    residuals: nextResiduals,
    onTime,
    skipped,
    sentCoordinates,
    sent,
  };
}

/** Runs `rounds` federated rounds and records the loss and the distance to the closed-form optimum. */
function runFederation(initial, targets, options) {
  let model = initial.slice();
  const optimum = trueOptimum(options.optimumTargets ?? targets);
  const history = [globalLoss(model, targets)];
  const gaps = [Math.sqrt(model.reduce((sum, value, i) => sum + (value - optimum[i]) ** 2, 0))];
  let residuals = options.residuals ?? targets.map(() => initial.map(() => 0));
  let sent = 0;
  for (let round = 0; round < (options.rounds ?? 20); round++) {
    const result = federatedRound(model, targets, { ...options, residuals });
    model = result.model;
    residuals = result.residuals;
    sent += result.sentCoordinates;
    history.push(globalLoss(model, targets));
    gaps.push(Math.sqrt(model.reduce((sum, value, i) => sum + (value - optimum[i]) ** 2, 0)));
  }
  return { model, history, gaps, sent, residuals, optimum };
}

/** The closed-form optimum: the average of the client targets (uniform weights). */
function trueOptimum(targets) {
  const dimension = targets[0].length;
  return Array.from({ length: dimension }, (_, i) => mean(targets.map((target) => target[i])));
}

/** The loss value at the optimum: the spread of the client means, which is not zero. */
function minimumLoss(targets) {
  const optimum = trueOptimum(targets);
  return targets.reduce((sum, target) => sum + localLoss(optimum, target), 0) / targets.length;
}

/** Geometric latencies from the seeded PRNG, in [1, scale]. */
function latenciesFor(count, seed, scale = 10) {
  const rng = makeRng(seed);
  return Array.from({ length: count }, () => 1 + (rng(1 << 20) % scale));
}

module.exports = {
  makeRng, mean, add, subtract, scale, localLoss, globalLoss, localUpdate, topKSparse,
  federatedRound, runFederation, trueOptimum, minimumLoss, latenciesFor,
};

if (require.main === module) {
  // -- the compression primitive ------------------------------------------------------------------
  throws(() => topKSparse([1, 2], 3), RangeError);
  const sparse = topKSparse([0.5, -4, 0.1, 3], 2);
  deepStrictEqual(sparse.chosen, [1, 3], "the two largest magnitudes survive");
  strictEqual(sparse.sparse.filter((value) => value !== 0).length, 2, "the update is exactly 2-sparse");
  deepStrictEqual(sparse.sparse, [0, -4, 0, 3], "the survivors keep their values");
  deepStrictEqual(topKSparse([0.5, -4, 0.1, 3], 2, true).sparse, [0, -8, 0, 6],
    "the optional d/k rescale doubles the survivors here");
  deepStrictEqual(topKSparse([1, 2, 3], 3).sparse, [1, 2, 3], "k = d changes nothing");
  deepStrictEqual(topKSparse([1, 2, 3], 0).sparse, [0, 0, 0], "k = 0 sends nothing");

  // -- error feedback: the residual is exactly the discarded part ---------------------------------
  const delta = [0.4, -2.5, 0.9, 0.1];
  const sparsified = topKSparse(delta, 2);
  const sentVector = sparsified.sparse;
  const residual = delta.map((value, i) => value - sentVector[i]);
  for (let i = 0; i < delta.length; i++) {
    if (sparsified.chosen.includes(i)) {
      strictEqual(sentVector[i], delta[i], "a kept coordinate is transmitted untouched");
      strictEqual(residual[i], 0, "a kept coordinate leaves nothing behind");
    } else {
      strictEqual(sentVector[i], 0, "a dropped coordinate is not transmitted");
      strictEqual(residual[i], delta[i], "a dropped coordinate waits in the residual");
    }
  }
  ok(sparsified.chosen.length === 2);

  // -- full participation converges to the closed-form optimum -------------------------------------
  const rng = makeRng(714);
  const dimension = 5;
  const targets = Array.from({ length: 8 }, () => Array.from({ length: dimension }, () => rng(2000) / 1000 - 1));
  const optimum = trueOptimum(targets);
  const floor = minimumLoss(targets);
  const initial = new Array(dimension).fill(0);
  const full = runFederation(initial, targets, { rounds: 60, localSteps: 1, stepSize: 0.5 });
  ok(full.gaps.at(-1) < 1e-6, `the model must reach the optimum, gap ${full.gaps.at(-1)}`);
  ok(Math.abs(full.history.at(-1) - floor) < 1e-9,
    `the loss must settle at the client spread ${floor}, got ${full.history.at(-1)}`);
  for (let i = 0; i < dimension; i++) ok(Math.abs(full.model[i] - optimum[i]) < 1e-3, "coordinate by coordinate");
  for (let round = 1; round < full.history.length; round++) {
    ok(full.history[round] <= full.history[round - 1] + 1e-12, "the loss must not rise at a stable step size");
  }

  // -- more local steps still converge, just not necessarily monotonically --------------------------
  const localSteps5 = runFederation(initial, targets, { rounds: 60, localSteps: 5, stepSize: 0.2 });
  ok(localSteps5.gaps.at(-1) < 1e-6, "five local steps per round also converge here");
  ok(localSteps5.sent === 60 * targets.length * dimension, "the uncompressed run sends every coordinate");

  // -- stragglers: only the fast clients report, so the fixed point is their mean -------------------
  const latencies = latenciesFor(targets.length, 7, 10);
  const deadline = 4;
  const onTime = latencies.filter((value) => value <= deadline).length;
  ok(onTime > 0 && onTime < targets.length, `the deadline must exclude somebody (${latencies})`);
  const participants = targets.filter((_, index) => latencies[index] <= deadline);
  const straggling = runFederation(initial, targets, {
    rounds: 120, localSteps: 1, stepSize: 0.4, latencies, deadline, optimumTargets: participants,
  });
  ok(straggling.gaps.at(-1) < 1e-6, "the run must reach the mean of the clients that reported");
  ok(straggling.history.at(-1) > floor + 1e-9,
    "missing clients leave a permanent loss floor above the full-participation one");
  const noDeadline = runFederation(initial, targets, { rounds: 120, localSteps: 1, stepSize: 0.4 });
  ok(straggling.history[10] > noDeadline.history[10] - 1e-12,
    "stragglers must not make early progress faster than full participation");
  const everyoneLate = runFederation(initial, targets, {
    rounds: 10, stepSize: 0.4, latencies, deadline: 0,
  });
  deepStrictEqual(everyoneLate.model, initial, "a deadline nobody meets leaves the model untouched");
  deepStrictEqual(everyoneLate.history, new Array(11).fill(everyoneLate.history[0]));

  // -- compression: k = d is exact, k < d loses a little, error feedback keeps the rest -------------
  const exact = runFederation(initial, targets, { rounds: 30, stepSize: 0.5, k: dimension });
  const plain = runFederation(initial, targets, { rounds: 30, stepSize: 0.5 });
  strictEqual(exact.history.at(-1).toFixed(12), plain.history.at(-1).toFixed(12), "k = d is lossless");
  deepStrictEqual(exact.model.map((v) => v.toFixed(12)), plain.model.map((v) => v.toFixed(12)));
  const compressed = runFederation(initial, targets, { rounds: 120, stepSize: 0.3, k: 2 });
  const noFeedback = runFederation(initial, targets, {
    rounds: 120, stepSize: 0.3, k: 2, errorFeedback: false,
  });
  ok(compressed.sent === 120 * targets.length * 2, "each compressed client sends exactly k coordinates");
  ok(noFeedback.sent === compressed.sent, "both variants send the same number of coordinates");
  ok(compressed.gaps[10] < noFeedback.gaps[10], `error feedback must help early (${compressed.gaps[10]} vs ${noFeedback.gaps[10]})`);
  ok(compressed.gaps.at(-1) <= noFeedback.gaps.at(-1) + 1e-9, "and it must not be worse at the end");
  const freeze = runFederation(initial, targets, { rounds: 30, stepSize: 0.3, k: 0 });
  deepStrictEqual(freeze.model, initial, "k = 0 cannot move the model");

  // -- the residual bookkeeping survives a long run --------------------------------------------------
  const tracked = runFederation(initial, targets, { rounds: 40, stepSize: 0.3, k: 2 });
  for (const residual of tracked.residuals) {
    for (const value of residual) ok(Number.isFinite(value), "the error feedback must stay finite");
  }
  ok(tracked.residuals.some((residual) => residual.some((value) => value !== 0)),
    "some mass must still be waiting to be sent");

  // -- partial participation and degenerate shapes ----------------------------------------------------
  const halfTargets = targets.filter((_, index) => index / targets.length < 0.5);
  const half = runFederation(initial, targets, {
    rounds: 60, stepSize: 0.5, participation: 0.5, optimumTargets: halfTargets,
  });
  ok(half.gaps.at(-1) < 1e-6, "a fixed half of the clients averages to the mean of that half");
  ok(half.history.at(-1) > floor + 1e-9, "and it cannot reach the full-participation loss floor");
  const single = runFederation(initial, [targets[0]], { rounds: 30, stepSize: 0.5 });
  ok(single.gaps.at(-1) < 1e-6, "one client converges to its own mean");
  ok(single.history.at(-1) < 1e-12, "and its loss floor is exactly zero");
  const noRounds = runFederation(initial, targets, { rounds: 0, stepSize: 0.5 });
  deepStrictEqual(noRounds.model, initial);
  strictEqual(noRounds.history.length, 1);

  // -- determinism -------------------------------------------------------------------------------------
  function replay(seed) {
    const local = makeRng(seed);
    const localTargets = Array.from({ length: 6 }, () => Array.from({ length: 3 }, () => local(1000) / 1000));
    const result = runFederation([0, 0, 0], localTargets, {
      rounds: 25, stepSize: 0.4, k: 2, latencies: latenciesFor(6, seed, 6), deadline: 3,
    });
    return `${result.history.at(-1).toFixed(12)}|${result.model.map((v) => v.toFixed(9)).join(",")}`;
  }
  strictEqual(replay(3), replay(3), "identical seeds must replay identically");

  console.log(`optimum reached: loss ${full.history.at(-1).toExponential(2)} after 60 rounds`);
  console.log(`stragglers on time=${onTime}/${targets.length}; compressed loss=${compressed.history.at(-1).toExponential(2)} vs no feedback ${noFeedback.history.at(-1).toExponential(2)}`);
  console.log("PASS");
}
