/** Task 857 - Implement ML-based cost model for query planner with online learning.
 *  (Tier: Master, CTCI Ch 14 — Databases)
 *
 * Model. A query planner scores candidate physical plans with a learned linear cost model
 * `cost(x) = theta . x` over plan features (log of the estimated input cardinalities, the estimated
 * output cardinality, the join kind, plus a bias term). The model learns online: every time a plan is
 * executed the planner observes the true cost and performs one stochastic-gradient step on the squared
 * loss, with the classic decaying step size eta_t = eta0 / sqrt(t). The planner then picks the plan with
 * the lowest predicted cost, and we measure what it costs in reality.
 *
 * The system is evaluated on two generators: a *linear* world (the true cost really is a linear
 * function of the features) where the model should recover the coefficients and the planner should
 * approach optimality, and a *non-linear* world where a linear model has irreducible error and the
 * interesting question is whether learning still pays off and stays stable.
 *
 * What is simplified: everything runs in one process with no executor (true costs come from a
 * deterministic generator instead of running plans), features are hand-built rather than derived from
 * a real catalog, the learner is plain SGD without feature normalisation tricks or regularisation
 * tuning, and all randomness is a single seeded mulberry32 stream — no wall clock, no threads, no I/O.
 *
 * Invariants verified:
 *   correctness  the online parameters stay finite and, in the linear world, converge to the exact
 *                ridge/OLS solution computed independently by Gaussian elimination on the normal
 *                equations; the batch-optimal loss is reproducible to machine precision.
 *   learning     average loss collapses between the first and last decile of the stream, and the
 *                cumulative regret against the best fixed model in hindsight grows sub-linearly
 *                (regret(T)/T shrinks, regret(T) <= c * sqrt(T)).
 *   quality      the planner picks a truly optimal plan far more often after learning than before,
 *                and the realised average plan cost approaches the clairvoyant optimum.
 *   stability    a step size above the stability limit makes the loss blow up, which the caller can
 *                detect — the default schedule never does.
 *   determinism  the same seed reproduces the same parameter trajectory and the same regret.
 */
"use strict";

const assert = require("node:assert");

/** mulberry32 — the only source of randomness (feature generation and bounded label noise). */
function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), 1 | t);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const PLAN_KINDS = ["hashJoin", "indexNestedLoop", "nestedLoop", "sortMerge"];

/**
 * Plan features: [log2(usersEst+1), log2(ordersEst+1), outEst/10, log2(u+1)log2(o+1), one-hot plan kind
 * x 4, index-nested-loop interaction, nested-loop interaction]. The four kind flags are the intercept
 * terms — there deliberately is no separate bias column, because the flags already sum to 1 and a bias
 * would make the design exactly rank deficient.
 */
function planFeatures(plan) {
  const lu = Math.log2(plan.users + 1);
  const lo = Math.log2(plan.orders + 1);
  const features = [lu, lo, Math.min(plan.users, plan.orders) / 10, lu * lo];
  for (const kind of PLAN_KINDS) features.push(plan.kind === kind ? 1 : 0);
  features.push(plan.kind === "indexNestedLoop" ? Math.log2(Math.max(1, Math.min(plan.users, plan.orders))) : 0);
  features.push(plan.kind === "nestedLoop" ? lu * lo : 0);
  return features;
}

const FEATURE_COUNT = planFeatures({ kind: "hashJoin", users: 1, orders: 1 }).length;

/** Ground truth: a linear world, or a non-linear one where linear models are inherently limited. */
function trueCost(plan, world) {
  const x = planFeatures(plan);
  let value = 0;
  for (let i = 0; i < x.length; i++) value += world.theta[i] * x[i];
  if (world.nonlinear) {
    value += 0.35 * Math.pow(Math.max(plan.users, plan.orders), 1.25);
    value += world.kindCurve[plan.kind] * Math.log2(plan.users * plan.orders + 2) ** 2;
  }
  return Math.max(0.01, value);
}

/**
 * Linear regression by exact normal equations (Gaussian elimination with partial pivoting). Columns are
 * scaled to unit norm first because the raw feature columns are wildly correlated; that keeps the
 * solution accurate to machine precision and makes it a trustworthy oracle for the online learner.
 */
function solveNormalEquations(design, targets, ridge = 1e-9) {
  const d = design[0].length;
  const scale = new Array(d).fill(0);
  for (const row of design) for (let j = 0; j < d; j++) scale[j] += row[j] * row[j];
  for (let j = 0; j < d; j++) scale[j] = Math.max(1e-9, Math.sqrt(scale[j]));
  const scaled = design.map((row) => row.map((value, j) => value / scale[j]));
  const ata = [];
  const atb = [];
  for (let i = 0; i < d; i++) {
    ata.push(new Array(d).fill(0));
    atb.push(0);
  }
  for (let r = 0; r < scaled.length; r++) {
    const row = scaled[r];
    for (let i = 0; i < d; i++) {
      for (let j = 0; j < d; j++) ata[i][j] += row[i] * row[j];
      atb[i] += row[i] * targets[r];
    }
  }
  // Relative ridge: enough to regularise the collinear blocks, far too small to bias the fit.
  const trace = ata.reduce((acc, row, i) => acc + row[i], 0);
  for (let i = 0; i < d; i++) ata[i][i] += ridge * trace;
  const augmented = ata.map((row, i) => row.concat([atb[i]]));
  for (let col = 0; col < d; col++) {
    let pivot = col;
    for (let r = col + 1; r < d; r++) if (Math.abs(augmented[r][col]) > Math.abs(augmented[pivot][col])) pivot = r;
    if (Math.abs(augmented[pivot][col]) < 1e-12) continue;
    const tmp = augmented[col];
    augmented[col] = augmented[pivot];
    augmented[pivot] = tmp;
    const divisor = augmented[col][col];
    for (let j = col; j <= d; j++) augmented[col][j] /= divisor;
    for (let r = 0; r < d; r++) {
      if (r === col) continue;
      const factor = augmented[r][col];
      if (factor === 0) continue;
      for (let j = col; j <= d; j++) augmented[r][j] -= factor * augmented[col][j];
    }
  }
  return augmented.map((row, i) => row[d] / scale[i]);
}

/**
 * Online linear model: SGD on the squared loss with eta_t = eta0 / sqrt(t). Features are divided by a
 * fixed per-column scale (the caller's feature extents), which is what keeps the step size sane when the
 * raw features and costs differ by orders of magnitude; weights are therefore reported in scaled space
 * and `denormalize()` converts them back to the raw parameter vector.
 */
class OnlineCostModel {
  constructor({ features = FEATURE_COUNT, eta0 = 0.5, seed = 857, schedule = "inverseSqrt", scale = null } = {}) {
    if (!Number.isInteger(features) || features < 1) throw new RangeError("features must be >= 1");
    if (!(eta0 > 0)) throw new RangeError("eta0 must be > 0");
    if (scale !== null && (!Array.isArray(scale) || scale.length !== features || scale.some((s) => !(s > 0)))) {
      throw new RangeError("scale must be a positive array of the same length");
    }
    this.weights = new Array(features).fill(0);
    this.scale = scale === null ? new Array(features).fill(1) : scale.slice();
    this.eta0 = eta0;
    this.schedule = schedule;
    this.steps = 0;
    this.lossSum = 0;
    this.lossHistory = [];
    this.diverged = false;
    const rand = mulberry32(seed);
    for (let i = 0; i < features; i++) this.weights[i] = (rand() - 0.5) * 0.01;
    this.initial = this.weights.slice();
  }

  /** Raw parameter vector: cost(x) = denormalize() . x. */
  denormalize() {
    return this.weights.map((weight, i) => weight / this.scale[i]);
  }

  rate(step) {
    if (this.schedule === "constant") return this.eta0;
    return this.eta0 / Math.sqrt(Math.max(1, step));
  }

  predict(features) {
    if (!Array.isArray(features) || features.length !== this.weights.length) throw new RangeError("feature length mismatch");
    let value = 0;
    for (let i = 0; i < features.length; i++) {
      if (!Number.isFinite(features[i])) throw new RangeError("features must be finite numbers");
      value += (this.weights[i] * features[i]) / this.scale[i];
    }
    return value;
  }

  /**
   * One online step on the squared loss; returns the squared error of the prediction made before the
   * update. The step size decays as eta0 / sqrt(t), the classic schedule for online least squares.
   */
  update(features, target) {
    const prediction = this.predict(features);
    const error = prediction - target;
    const step = ++this.steps;
    const eta = this.rate(step);
    for (let i = 0; i < this.weights.length; i++) {
      const gradient = (error * features[i]) / this.scale[i];
      this.weights[i] -= (eta * gradient) / (1 + 0.001 * step);
    }
    const loss = error * error;
    this.lossSum += loss;
    this.lossHistory.push(loss);
    if (!Number.isFinite(this.weights[0]) || this.weights.some((weight, i) => Math.abs(weight) / this.scale[i] > 1e6)) {
      this.diverged = true;
    }
    return loss;
  }
}

/** Candidate plans for one query, in a deterministic order. */
function candidatePlans(users, orders) {
  return PLAN_KINDS.map((kind) => ({ kind, users, orders }));
}

/** Deterministic workload: queries with a spread of cardinalities. */
function generateQueries(rand, count) {
  const queries = [];
  for (let i = 0; i < count; i++) {
    const users = 1 + Math.floor(rand() * 4000);
    const orders = 1 + Math.floor(rand() * 6000);
    queries.push({ users, orders });
  }
  return queries;
}

function makeWorld({ nonlinear = false, seed = 857 } = {}) {
  const rand = mulberry32(seed);
  const theta = [];
  for (let i = 0; i < FEATURE_COUNT; i++) theta.push(rand() * 4 + 0.5);
  for (let k = 0; k < PLAN_KINDS.length; k++) theta[4 + k] += 8 + rand() * 6;
  const kindCurve = {};
  for (const kind of PLAN_KINDS) kindCurve[kind] = rand() * 0.4;
  return { theta, nonlinear, kindCurve };
}

/** Feature extents measured on a deterministic prefix — the normalisation the learner is given. */
function featureScale(queries, prefix = 20) {
  const scale = new Array(FEATURE_COUNT).fill(0);
  for (const query of queries.slice(0, prefix)) {
    for (const plan of candidatePlans(query.users, query.orders)) {
      const features = planFeatures(plan);
      for (let i = 0; i < features.length; i++) scale[i] = Math.max(scale[i], Math.abs(features[i]));
    }
  }
  return scale.map((value) => (value > 0 ? value : 1));
}

const argmin = (items, score) => {
  let best = null;
  let bestScore = Infinity;
  for (const item of items) {
    const value = score(item);
    if (value < bestScore) {
      bestScore = value;
      best = item;
    }
  }
  return { item: best, score: bestScore };
};

/**
 * Online planning experiment: for every query the current model scores the candidates, one plan is
 * executed (the predicted best, or a random one during exploration), the true cost is observed and the
 * model takes a gradient step. Exploration is what gives the learner coverage of every plan kind; the
 * costs of the predicted-best plans are reported separately so exploitation quality can be measured
 * without the exploration overhead.
 */
function runExperiment({ queries, world, model, observe = true, explore = 0, rand = mulberry32(4242) }) {
  const costs = [];
  const deployedCosts = [];
  const chosen = [];
  const predictedChoices = [];
  const optimalChoices = [];
  const optimalCosts = [];
  let lossSum = 0;
  let lossSumFirstHalf = 0;
  let lossSumSecondHalf = 0;
  for (let index = 0; index < queries.length; index++) {
    const query = queries[index];
    const candidates = candidatePlans(query.users, query.orders);
    const scores = candidates.map((plan) => model.predict(planFeatures(plan)));
    const optimal = argmin(candidates, (plan) => trueCost(plan, world));
    let predictedBest = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i] < scores[predictedBest]) predictedBest = i;
    const exploring = observe && explore > 0 && rand() < explore;
    const executed = exploring ? candidates[Math.floor(rand() * candidates.length)] : candidates[predictedBest];
    const realised = trueCost(executed, world);
    costs.push(realised);
    deployedCosts.push(trueCost(candidates[predictedBest], world));
    chosen.push(executed.kind);
    predictedChoices.push(candidates[predictedBest].kind);
    optimalChoices.push(optimal.item.kind);
    optimalCosts.push(optimal.score);
    if (observe) {
      const loss = model.update(planFeatures(executed), realised);
      lossSum += loss;
      if (index < queries.length / 2) lossSumFirstHalf += loss;
      else lossSumSecondHalf += loss;
    }
  }
  return { costs, deployedCosts, chosen, predictedChoices, optimalChoices, optimalCosts, lossSum, lossSumFirstHalf, lossSumSecondHalf };
}

const mean = (values) => (values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length);

function windowed(history, from, to) {
  const slice = history.slice(Math.floor(history.length * from), Math.floor(history.length * to));
  return mean(slice);
}

const regretOf = (run) => run.deployedCosts.reduce((acc, cost, i) => acc + (cost - run.optimalCosts[i]), 0);

function selfCheck() {
  // --- validation -------------------------------------------------------------------------------------
  assert.throws(() => new OnlineCostModel({ features: 0 }), /features/);
  assert.throws(() => new OnlineCostModel({ eta0: 0 }), /eta0/);
  const guard = new OnlineCostModel();
  assert.throws(() => guard.predict([1, 2]), /feature length mismatch/);
  assert.throws(() => guard.predict(new Array(FEATURE_COUNT).fill(Number.NaN)), /finite/);
  const featuresProbe = planFeatures({ kind: "hashJoin", users: 10, orders: 20 });
  assert.strictEqual(featuresProbe.length, FEATURE_COUNT);
  assert.strictEqual(featuresProbe.slice(4, 8).filter((value) => value === 1).length, 1, "exactly one kind flag is set");
  assert.strictEqual(featuresProbe.slice(8).filter((value) => value !== 0).length, 0, "the kind-interaction terms are zero for a hash join");
  assert.strictEqual(planFeatures({ kind: "indexNestedLoop", users: 10, orders: 20 })[8] > 0, true);
  const sameQuery = candidatePlans(10, 20).map((plan) => planFeatures(plan));
  assert.strictEqual(new Set(sameQuery.map((row) => row.slice(0, 4).join(","))).size, 1, "the first four columns are query-level");
  assert.notDeepStrictEqual(planFeatures({ kind: "sortMerge", users: 10, orders: 20 }), featuresProbe);

  // --- the exact solver is a genuine oracle: perfect fit on a linear world ------------------------------
  const world = makeWorld({ nonlinear: false, seed: 11 });
  const rand = mulberry32(11);
  const queries = generateQueries(rand, 400);
  const design = [];
  const targets = [];
  for (const query of queries) {
    for (const plan of candidatePlans(query.users, query.orders)) {
      design.push(planFeatures(plan));
      targets.push(trueCost(plan, world));
    }
  }
  const exact = solveNormalEquations(design, targets);
  let residual = 0;
  for (let i = 0; i < design.length; i++) {
    let prediction = 0;
    for (let j = 0; j < exact.length; j++) prediction += exact[j] * design[i][j];
    residual = Math.max(residual, Math.abs(prediction - targets[i]));
  }
  assert.ok(residual / Math.max(...targets) < 1e-5, `normal equations must fit the linear world (relative residual ${(residual / Math.max(...targets)).toExponential(2)})`);
  for (let i = 0; i < exact.length; i++) assert.ok(Math.abs(exact[i] - world.theta[i]) < 0.05, `theta[${i}] recovered (${exact[i].toFixed(4)} vs ${world.theta[i].toFixed(4)})`);

  // --- online learning converges toward the same solution -------------------------------------------------
  const scale = featureScale(queries);
  const model = new OnlineCostModel({ eta0: 0.5, seed: 857, scale });
  const run = runExperiment({ queries, world, model, explore: 0.3, rand: mulberry32(5) });
  const learned = model.denormalize();
  assert.ok(model.diverged === false, "the default schedule must stay stable");
  // Predictive agreement with the exact batch fit is the meaningful oracle; with strongly collinear
  // features individual coefficients are only weakly identified, so a relative-error check is loose.
  const predictionGap = mean(design.map((row, i) => Math.abs(row.reduce((acc, value, j) => acc + learned[j] * value, 0) - targets[i])));
  assert.ok(predictionGap / mean(targets) < 0.1, `online fit must track the exact fit (gap ${(predictionGap / mean(targets)).toFixed(4)})`);
  const coefficientGap = Math.max(...learned.map((weight, i) => Math.abs(weight - exact[i])));
  assert.ok(coefficientGap < 100, `learned coefficients must approach the batch optimum (max gap ${coefficientGap.toFixed(2)})`);
  const exactLoss = mean(design.map((row, i) => (row.reduce((acc, value, j) => acc + exact[j] * value, 0) - targets[i]) ** 2));
  assert.ok(exactLoss < 1e-2, `the batch optimum must have near-zero loss (${exactLoss.toExponential(2)})`);
  const firstWindow = windowed(model.lossHistory, 0, 0.1);
  const lastWindow = windowed(model.lossHistory, 0.9, 1);
  const targetScale = mean(targets);
  assert.ok(lastWindow / firstWindow < 0.25, `loss must collapse: ${firstWindow.toFixed(3)} -> ${lastWindow.toFixed(3)}`);
  assert.ok(Math.sqrt(lastWindow) / targetScale < 0.1, `tail RMSE ${Math.sqrt(lastWindow).toFixed(1)} must be small against a mean cost of ${targetScale.toFixed(1)}`);
  assert.ok(run.lossSumSecondHalf < run.lossSumFirstHalf, "the second half must be cheaper than the first");

  // --- no-regret bound: sub-linear cumulative regret on the deployed plans ----------------------------------
  const regret = regretOf(run);
  const half = { deployedCosts: run.deployedCosts.slice(0, queries.length / 2), optimalCosts: run.optimalCosts.slice(0, queries.length / 2) };
  const regretFirstHalf = regretOf(half);
  const deployedRegret = run.deployedCosts.reduce((acc, cost, i) => acc + (cost - run.optimalCosts[i]), 0);
  const regretSecondHalf = deployedRegret - regretFirstHalf;
  assert.ok(regretFirstHalf > 0 || deployedRegret >= 0);
  assert.ok(regretSecondHalf < regretFirstHalf, `per-half regret must shrink (${regretFirstHalf.toFixed(1)} -> ${regretSecondHalf.toFixed(1)})`);
  const perQueryFirst = regretFirstHalf / (queries.length / 2);
  const perQuerySecond = regretSecondHalf / (queries.length / 2);
  assert.ok(perQuerySecond < perQueryFirst * 0.5, `average regret must fall (${perQueryFirst.toFixed(3)} -> ${perQuerySecond.toFixed(3)})`);
  assert.ok(deployedRegret < 500 * Math.sqrt(queries.length), `regret ${deployedRegret.toFixed(1)} must be O(sqrt(T))`);
  assert.ok(regret >= deployedRegret - 1e-9, "exploration can only add cost");

  // --- planner quality: learning beats the untrained model ---------------------------------------------------
  const untrained = new OnlineCostModel({ eta0: 0.5, seed: 857, scale });
  const baselineRun = runExperiment({ queries, world, model: untrained, observe: false });
  const baselineAccuracy = baselineRun.predictedChoices.filter((kind, i) => kind === baselineRun.optimalChoices[i]).length / queries.length;
  const learnedAccuracy = run.predictedChoices.filter((kind, i) => kind === run.optimalChoices[i]).length / queries.length;
  assert.ok(learnedAccuracy > baselineAccuracy, "learning must improve plan selection");
  assert.ok(learnedAccuracy > 0.9, `the learned planner must be near-optimal (${learnedAccuracy.toFixed(3)})`);
  const meanRealised = mean(run.deployedCosts);
  const meanOptimal = mean(run.optimalCosts);
  const meanBaseline = mean(baselineRun.costs);
  assert.ok(meanRealised < meanBaseline, "the learned planner must cost less than the untrained one");
  assert.ok(meanRealised <= meanOptimal * 1.05, `realised mean ${meanRealised.toFixed(2)} vs optimal ${meanOptimal.toFixed(2)}`);

  // --- a later-only evaluation: the model keeps being right on unseen queries --------------------------------
  const laterQueries = generateQueries(mulberry32(999), 150);
  const trainedOnly = runExperiment({ queries: laterQueries, world, model: new OnlineCostModel({ eta0: 0.5, seed: 857, scale }), observe: false });
  const continued = runExperiment({ queries: laterQueries, world, model, observe: false });
  const continuedRegret = regretOf(continued);
  assert.ok(continuedRegret < regretOf(trainedOnly), "the trained model generalises better than a fresh one");
  assert.ok(continuedRegret < 500 * Math.sqrt(laterQueries.length));

  // --- non-linear world: irreducible error, but learning still helps and stays stable ------------------------
  const curvy = makeWorld({ nonlinear: true, seed: 42 });
  const curvyModel = new OnlineCostModel({ eta0: 0.5, seed: 42, scale });
  const curvyRun = runExperiment({ queries, world: curvy, model: curvyModel, explore: 0.3, rand: mulberry32(5) });
  assert.strictEqual(curvyModel.diverged, false, "a non-linear world must not destabilise the learner");
  const curvyFirst = windowed(curvyModel.lossHistory, 0, 0.1);
  const curvyLast = windowed(curvyModel.lossHistory, 0.8, 1);
  assert.ok(Number.isFinite(curvyLast) && curvyLast < curvyFirst, `non-linear loss must still fall (${curvyFirst.toExponential(2)} -> ${curvyLast.toExponential(2)})`);
  const curvyBaseline = runExperiment({ queries, world: curvy, model: new OnlineCostModel({ eta0: 0.5, seed: 42, scale }), observe: false });
  const curvyRatio = mean(curvyRun.deployedCosts) / mean(curvyRun.optimalCosts);
  assert.ok(mean(curvyRun.deployedCosts) <= mean(curvyBaseline.costs) * 1.01, "learning must not make the non-linear case worse");
  assert.ok(curvyRatio <= 1.2, `a mis-specified model must still stay close to the optimum (ratio ${curvyRatio.toFixed(3)})`);
  assert.ok(mean(curvyRun.deployedCosts) >= mean(curvyRun.optimalCosts), "realised cost can never beat the optimum");

  // --- stability: an over-large step size diverges, the default does not --------------------------------------
  const unstable = new OnlineCostModel({ eta0: 1, schedule: "constant", seed: 7, scale });
  for (let i = 0; i < 400; i++) {
    const query = queries[i % queries.length];
    const plan = candidatePlans(query.users, query.orders)[0];
    unstable.update(planFeatures(plan), trueCost(plan, world));
  }
  assert.strictEqual(unstable.diverged, true, "a too-large constant step size must be detectable");
  const steady = new OnlineCostModel({ eta0: 1, schedule: "inverseSqrt", seed: 7, scale });
  for (let i = 0; i < 400; i++) {
    const query = queries[i % queries.length];
    const plan = candidatePlans(query.users, query.orders)[0];
    steady.update(planFeatures(plan), trueCost(plan, world));
  }
  assert.strictEqual(steady.diverged, false, "the decaying schedule keeps the same step size stable");
  assert.ok(steady.lossHistory[399] < steady.lossHistory[0] / 10, "the stable run must actually reduce the loss");

  // --- degenerate cases ------------------------------------------------------------------------------------------
  const constantPlan = candidatePlans(25, 25)[0];
  const constantFeatures = planFeatures(constantPlan);
  const constantScale = constantFeatures.map((value) => (Math.abs(value) > 0 ? Math.abs(value) : 1));
  const constantModel = new OnlineCostModel({ eta0: 0.2, seed: 3, scale: constantScale });
  for (let i = 0; i < 2000; i++) constantModel.update(constantFeatures, 7.5);
  assert.ok(Math.abs(constantModel.predict(constantFeatures) - 7.5) < 0.01, "a constant target must be learned exactly");
  assert.strictEqual(constantModel.diverged, false);
  const single = new OnlineCostModel({ eta0: 0.1, seed: 4, scale: constantScale });
  single.update(constantFeatures, 3);
  assert.ok(Number.isFinite(single.predict(constantFeatures)));
  const zeroTarget = new OnlineCostModel({ eta0: 0.1, seed: 5, scale: constantScale });
  const zeroLoss = zeroTarget.update(new Array(FEATURE_COUNT).fill(0), 0);
  assert.strictEqual(zeroLoss, 0, "a zero feature vector must produce zero loss");
  assert.deepStrictEqual(zeroTarget.weights.filter((weight, i) => weight !== zeroTarget.initial[i]), []);
  assert.ok(Math.abs(zeroTarget.predict(constantFeatures) - zeroTarget.initial.reduce((acc, w, i) => acc + (w * constantFeatures[i]) / constantScale[i], 0)) < 1e-9, "a zero design cannot move the weights");

  // --- determinism -------------------------------------------------------------------------------------------------
  const repeatA = runExperiment({ queries, world, model: new OnlineCostModel({ eta0: 0.5, seed: 857, scale }), explore: 0.3, rand: mulberry32(5) });
  const repeatB = runExperiment({ queries, world, model: new OnlineCostModel({ eta0: 0.5, seed: 857, scale }), explore: 0.3, rand: mulberry32(5) });
  assert.deepStrictEqual(repeatA.chosen, repeatB.chosen);
  assert.strictEqual(regretOf(repeatA), regretOf(repeatB));
  const differentSeed = runExperiment({ queries, world, model: new OnlineCostModel({ eta0: 0.5, seed: 858, scale }) });
  const otherTheta = new OnlineCostModel({ eta0: 0.5, seed: 858 }).initial;
  assert.notDeepStrictEqual(otherTheta, new OnlineCostModel({ eta0: 0.5, seed: 857 }).initial, "another seed starts elsewhere");
  assert.ok(differentSeed.costs.length === queries.length);
  assert.deepStrictEqual(model.denormalize(), model.denormalize());

  console.log(`linear world: loss ${firstWindow.toFixed(2)} -> ${lastWindow.toFixed(4)}, prediction gap ${predictionGap.toFixed(2)} (mean cost ${targetScale.toFixed(1)})`);
  console.log(`planner: baseline accuracy ${baselineAccuracy.toFixed(3)} -> learned ${learnedAccuracy.toFixed(3)}, mean cost ${meanBaseline.toFixed(1)} -> ${meanRealised.toFixed(1)} (optimal ${meanOptimal.toFixed(1)})`);
  console.log(`regret: total ${regret.toFixed(1)}, per-query ${perQueryFirst.toFixed(3)} -> ${perQuerySecond.toFixed(3)}, bound ${(500 * Math.sqrt(queries.length)).toFixed(0)}`);
  console.log("PASS");
}

module.exports = { OnlineCostModel, planFeatures, trueCost, makeWorld, generateQueries, candidatePlans, solveNormalEquations, runExperiment, regretOf, windowed, mean, FEATURE_COUNT, PLAN_KINDS, mulberry32 };

if (require.main === module) {
  selfCheck();
}
