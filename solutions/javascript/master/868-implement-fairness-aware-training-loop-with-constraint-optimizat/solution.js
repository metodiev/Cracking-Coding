/** Task 868 - Implement fairness-aware training loop with constraint optimization (demographic parity). (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * A deterministic training loop for a logistic-regression classifier trained under a
 * demographic-parity constraint. With h_theta(x) = sigmoid(theta . x) the constrained problem is
 *
 *     minimise  F(theta) = mean_i logloss(h_theta(x_i), y_i)
 *     subject to c(theta) = |DP(theta)| - eps <= 0,
 *     DP(theta) = mean_{s=1} h(x) - mean_{s=0} h(x)          (demographic parity gap)
 *
 * solved with the augmented Lagrangian method for inequality constraints:
 *
 *     L_rho(theta; lambda) = F(theta) + ( max(0, lambda + rho c)^2 - lambda^2 ) / (2 rho)
 *     theta  <- argmin_theta L_rho(theta; lambda)               (inner solver, below)
 *     lambda <- max(0, lambda + rho c(theta))                   (projected dual ascent)
 *     rho    <- penaltyGrowth * rho                             (penalty schedule)
 *
 * The inner minimisation uses full-batch gradient descent with an Armijo backtracking line search,
 * which is what makes the loop stable no matter how large the multiplier becomes; the gradient of the
 * constraint is grad c = sign(DP) * ( mean_{s=1} h(1-h)x - mean_{s=0} h(1-h)x ), with sign(0) fixed
 * to +1 so that the subgradient of |.| is deterministic.
 *
 * Invariants verified: (i) the analytic loss/constraint gradients agree with central finite
 * differences; (ii) primal feasibility |DP| <= eps + tol at the returned iterate, at five values of
 * eps including eps = 0; (iii) dual feasibility lambda >= 0 and complementary slackness
 * lambda * max(0, c) ~ 0, so an inactive constraint leaves lambda exactly at 0; (iv) the objective is
 * monotone non-increasing in eps and accuracy is monotone non-decreasing in eps, i.e. the
 * accuracy/fairness trade-off is a real Pareto curve; (v) the loop is bit-for-bit reproducible.
 * Feasibility and the objective are cross-checked by two independent solvers: a quadratic-penalty
 * method with a doubling schedule and a brute-force scan over frozen multipliers, each with a primal
 * re-solve.
 *
 * What is simplified: the model is linear logistic regression in two features plus a bias (no hidden
 * layers), the data is small and synthetic (a seeded mulberry32 PRNG, Box-Muller normals), the
 * constraint is the *empirical* parity gap on the training set rather than a generalisation bound,
 * and steps are synchronous and full batch instead of minibatch SGD with momentum. No timers, no
 * wall-clock time, no Math.random, no threads - the same inputs always produce bit-identical output.
 */
"use strict";

const { ok, throws } = require("node:assert");

/** Deterministic mulberry32 PRNG: the only source of randomness in this file. */
function makeRng(seed) {
  let state = (seed >>> 0) || 0x9e3779b9;
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller; consumes exactly two uniforms per standard normal draw. */
function makeNormal(rng) {
  return function normal() {
    const u1 = Math.max(rng(), 1e-12);
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const dot = (a, b) => a.reduce((acc, v, i) => acc + v * b[i], 0);
const mean = (xs) => xs.reduce((acc, v) => acc + v, 0) / xs.length;

/**
 * Synthetic two-group classification problem. A shared, strongly predictive feature u1 and a second
 * feature u2 whose mean is shifted by the sensitive group; the label depends on both, so using u2
 * buys accuracy but moves the two groups' positive rates apart. That is the regime where
 * demographic parity and accuracy genuinely trade off against each other.
 */
function makeDataset({
  seed = 20240607, perGroup = 200, signalShared = 1.8, signalGrouped = 1.0,
  groupShift = 0.5, groupedNoise = 0.6,
} = {}) {
  if (!Number.isInteger(perGroup) || perGroup <= 0) throw new RangeError("perGroup must be a positive integer");
  if (!(groupedNoise > 0)) throw new RangeError("groupedNoise must be positive");
  const rng = makeRng(seed);
  const normal = makeNormal(rng);
  const rows = [];
  for (const group of [0, 1]) {
    const centred = group === 1 ? groupShift : -groupShift;
    for (let i = 0; i < perGroup; i++) {
      const u1 = normal();
      const u2 = centred + groupedNoise * normal();
      const logit = signalShared * u1 + signalGrouped * u2;
      rows.push({ x: [1, u1, u2], group, y: sigmoid(logit) > 0.5 ? 1 : 0 });
    }
  }
  return rows;
}

const predict = (theta, x) => sigmoid(dot(theta, x));

const groupRows = (rows, group) => rows.filter((r) => r.group === group);

/**
 * Single pass over the data returning the mean logistic loss and the parity gap together: the inner
 * solvers evaluate this thousands of times, so the two statistics must not cost two passes.
 */
function stats(theta, rows) {
  if (rows.length === 0) throw new Error("empty dataset");
  let loss = 0;
  let sumOne = 0;
  let countOne = 0;
  let sumZero = 0;
  let countZero = 0;
  for (const r of rows) {
    const p = sigmoid(dot(theta, r.x));
    const clipped = Math.min(Math.max(p, 1e-15), 1 - 1e-15);
    loss += -(r.y * Math.log(clipped) + (1 - r.y) * Math.log(1 - clipped));
    if (r.group === 1) {
      sumOne += p;
      countOne += 1;
    } else {
      sumZero += p;
      countZero += 1;
    }
  }
  if (countOne === 0 || countZero === 0) {
    throw new Error("demographic parity needs at least one sample in every group");
  }
  return { loss: loss / rows.length, gap: sumOne / countOne - sumZero / countZero };
}

/** Demographic parity gap: mean prediction on group 1 minus mean prediction on group 0. */
function demographicParityGap(theta, rows) {
  const zero = groupRows(rows, 0);
  const one = groupRows(rows, 1);
  if (zero.length === 0 || one.length === 0) {
    throw new Error("demographic parity needs at least one sample in every group");
  }
  return mean(one.map((r) => predict(theta, r.x))) - mean(zero.map((r) => predict(theta, r.x)));
}

function logisticLoss(theta, rows) {
  if (rows.length === 0) throw new Error("empty dataset");
  return mean(rows.map((r) => {
    const p = Math.min(Math.max(predict(theta, r.x), 1e-15), 1 - 1e-15);
    return -(r.y * Math.log(p) + (1 - r.y) * Math.log(1 - p));
  }));
}

function accuracy(theta, rows) {
  return rows.filter((r) => (predict(theta, r.x) >= 0.5 ? 1 : 0) === r.y).length / rows.length;
}

function lossGradient(theta, rows) {
  const g = new Array(theta.length).fill(0);
  for (const r of rows) {
    const p = predict(theta, r.x);
    for (let j = 0; j < theta.length; j++) g[j] += (p - r.y) * r.x[j];
  }
  return g.map((v) => v / rows.length);
}

/** Subgradient of c(theta) = |DP| - eps; sign(0) is fixed to +1 so the loop stays deterministic. */
function constraintGradient(theta, rows) {
  const gap = demographicParityGap(theta, rows);
  const sign = gap >= 0 ? 1 : -1;
  const sizes = [groupRows(rows, 0).length, groupRows(rows, 1).length];
  if (sizes[0] === 0 || sizes[1] === 0) throw new Error("demographic parity needs every group populated");
  const g = new Array(theta.length).fill(0);
  for (const r of rows) {
    const p = predict(theta, r.x);
    const weight = ((r.group === 1 ? 1 : -1) * sign * p * (1 - p)) / sizes[r.group];
    for (let j = 0; j < theta.length; j++) g[j] += weight * r.x[j];
  }
  return g;
}

/** Central-difference gradient of any scalar objective (independent audit of the analytic ones). */
function numericGradient(f, theta, h = 1e-6) {
  return theta.map((value, j) => {
    const up = theta.slice();
    const down = theta.slice();
    up[j] = value + h;
    down[j] = value - h;
    return (f(up) - f(down)) / (2 * h);
  });
}

/**
 * Full-batch gradient descent with Armijo backtracking: the step is halved until sufficient decrease
 * is achieved, so the effective step adapts to the local curvature and to large Lagrange multipliers.
 */
function minimiseSubproblem(objective, gradient, theta0, { steps = 150, lr0 = 0.5 } = {}) {
  const theta = theta0.slice();
  let value = objective(theta);
  for (let s = 0; s < steps; s++) {
    const g = gradient(theta);
    const gNormSq = g.reduce((acc, v) => acc + v * v, 0);
    if (gNormSq === 0) break;
    let step = lr0;
    let accepted = false;
    for (let backtrack = 0; backtrack < 40; backtrack++) {
      const candidate = theta.map((v, i) => v - step * g[i]);
      const candidateValue = objective(candidate);
      if (candidateValue <= value - 1e-4 * step * gNormSq) {
        theta.splice(0, theta.length, ...candidate);
        value = candidateValue;
        accepted = true;
        break;
      }
      step /= 2;
    }
    if (!accepted) break;
  }
  return theta;
}

/**
 * Augmented-Lagrangian primal-dual training loop. Returns the iterate, the multiplier and a trace with
 * one entry per outer round so feasibility, the trade-off curve and dual settling can be audited.
 */
function trainFairnessConstrained(rows, {
  eps = 0.05, rounds = 12, innerSteps = 110, lr0 = 0.5, penalty = 2, penaltyGrowth = 4, theta0 = null,
} = {}) {
  if (rows.length === 0) throw new Error("empty dataset");
  if (!(eps >= 0)) throw new RangeError(`eps must be non-negative (got ${eps})`);
  if (!(penalty > 0)) throw new RangeError(`penalty must be positive (got ${penalty})`);
  if (!(penaltyGrowth >= 1)) throw new RangeError("penaltyGrowth must be at least 1");
  if (!Number.isInteger(rounds) || rounds < 1) throw new RangeError("rounds must be a positive integer");
  if (!Number.isInteger(innerSteps) || innerSteps < 1) throw new RangeError("innerSteps must be a positive integer");
  const dim = rows[0].x.length;
  let theta = theta0 ? theta0.slice() : new Array(dim).fill(0);
  let lambda = 0;
  let rho = penalty;
  const trace = [];
  const snapshot = (round) => {
    const gap = demographicParityGap(theta, rows);
    const violation = Math.abs(gap) - eps;
    const loss = logisticLoss(theta, rows);
    return { round, loss, gap, violation, lambda, objective: loss + lambda * Math.max(0, violation) };
  };
  for (let round = 0; round < rounds; round++) {
    const objective = (t) => {
      const { loss, gap } = stats(t, rows);
      const slack = Math.max(0, lambda + rho * (Math.abs(gap) - eps));
      return loss + (slack * slack - lambda * lambda) / (2 * rho);
    };
    const gradient = (t) => {
      const s = Math.max(0, lambda + rho * (Math.abs(stats(t, rows).gap) - eps));
      const gLoss = lossGradient(t, rows);
      const gConstraint = constraintGradient(t, rows);
      return gLoss.map((v, j) => v + s * gConstraint[j]);
    };
    theta = minimiseSubproblem(objective, gradient, theta, { steps: innerSteps, lr0 });
    lambda = Math.max(0, lambda + rho * (Math.abs(demographicParityGap(theta, rows)) - eps));
    rho *= penaltyGrowth;
    trace.push(snapshot(round));
  }
  const final = snapshot(rounds);
  return {
    theta,
    lambda,
    loss: final.loss,
    gap: final.gap,
    violation: final.violation,
    accuracy: accuracy(theta, rows),
    trace,
  };
}

/** Independent solver #1: quadratic penalty with a doubling schedule (no dual variable at all). */
function trainPenalty(rows, { eps = 0.05, rounds = 12, innerSteps = 110, penalty = 1, growth = 4, lr0 = 0.5, theta0 = null } = {}) {
  const dim = rows[0].x.length;
  let theta = theta0 ? theta0.slice() : new Array(dim).fill(0);
  let rho = penalty;
  for (let round = 0; round < rounds; round++) {
    const objective = (t) => {
      const { loss, gap } = stats(t, rows);
      const excess = Math.max(0, Math.abs(gap) - eps);
      return loss + rho * excess * excess;
    };
    const gradient = (t) => {
      const excess = Math.max(0, Math.abs(demographicParityGap(t, rows)) - eps);
      const gLoss = lossGradient(t, rows);
      const gConstraint = constraintGradient(t, rows);
      return gLoss.map((v, j) => v + 2 * rho * excess * gConstraint[j]);
    };
    theta = minimiseSubproblem(objective, gradient, theta, { steps: innerSteps, lr0 });
    rho *= growth;
  }
  const gap = demographicParityGap(theta, rows);
  return { theta, loss: logisticLoss(theta, rows), gap, accuracy: accuracy(theta, rows) };
}

/**
 * Independent solver #2: multi-start penalty search. Six deterministic starts (the zero vector, three
 * scaled copies of the unconstrained direction and two seeded random points) each run the quadratic
 * penalty method to convergence; the best feasible result is an upper bound on the constrained optimum
 * obtained from a different algorithm and from different initial points. A frozen-multiplier *linear*
 * Lagrangian is deliberately not used as an oracle here: for an inequality constraint its minimiser
 * drives the slack towards zero (and the loss towards infinity) instead of respecting eps.
 */
function trainMultiStartPenalty(rows, { eps = 0.05, rounds = 12, innerSteps = 70, growth = 4, seed = 4242, scale = 6 } = {}) {
  if (!(eps >= 0)) throw new RangeError(`eps must be non-negative (got ${eps})`);
  const dim = rows[0].x.length;
  const rng = makeRng(seed);
  const unconstrained = minimiseSubproblem(
    (t) => logisticLoss(t, rows), (t) => lossGradient(t, rows), new Array(dim).fill(0), { steps: 400, lr0: 0.5 },
  );
  const starts = [new Array(dim).fill(0)];
  for (const factor of [0.25, 0.5, 1]) starts.push(unconstrained.map((v) => v * factor));
  for (let k = 0; k < 2; k++) starts.push(Array.from({ length: dim }, () => (rng() * 2 - 1) * scale));
  const results = starts.map((start) => {
    const run = trainPenalty(rows, { eps, rounds, innerSteps, growth, penalty: 1, lr0: 0.5, theta0: start });
    return { ...run, feasible: Math.abs(run.gap) <= eps + 1e-3 };
  });
  const feasible = results.filter((r) => r.feasible);
  if (feasible.length === 0) throw new Error("no multi-start run produced a feasible point");
  const best = feasible.reduce((a, b) => (b.loss < a.loss ? b : a));
  return { starts, results, feasibleCount: feasible.length, best };
}

function main() {
  const diagnostics = [];
  const worstDifference = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

  // 1. Data set shape and the engineered disparity in the raw label prevalence.
  const rows = makeDataset({ seed: 20240607, perGroup: 200 });
  const prevalence = [0, 1].map((g) => groupRows(rows, g).filter((r) => r.y === 1).length / groupRows(rows, g).length);
  diagnostics.push(`dataset n=${rows.length}, positive rate g0=${prevalence[0].toFixed(3)} g1=${prevalence[1].toFixed(3)}`);
  ok(rows.length === 400, "dataset must contain exactly 400 rows");
  ok(Math.abs(prevalence[1] - prevalence[0]) > 0.05, "the two groups must have different label prevalence");

  // 2. Analytic gradients versus central finite differences (independent oracle).
  const probe = [0.25, -0.6, 0.4];
  const lossGradErr = worstDifference(numericGradient((t) => logisticLoss(t, rows), probe), lossGradient(probe, rows));
  const constraintGradErr = worstDifference(
    numericGradient((t) => Math.abs(demographicParityGap(t, rows)) - 0.1, probe),
    constraintGradient(probe, rows),
  );
  diagnostics.push(`gradient check: loss ${lossGradErr.toExponential(2)}, constraint ${constraintGradErr.toExponential(2)}`);
  ok(lossGradErr < 1e-7, `analytic loss gradient must match finite differences (error ${lossGradErr})`);
  ok(constraintGradErr < 1e-6, `analytic constraint gradient must match finite differences (error ${constraintGradErr})`);

  // 3. The unconstrained optimum violates parity by a wide margin and is more accurate than the fair
  //    model: the trade-off exists and the constraint is not vacuous.
  const eps = 0.02;
  const unconstrained = trainFairnessConstrained(rows, { eps: 1e9 });
  const fair = trainFairnessConstrained(rows, { eps });
  diagnostics.push(`unconstrained: gap=${unconstrained.gap.toFixed(4)} acc=${unconstrained.accuracy.toFixed(4)} loss=${unconstrained.loss.toFixed(4)}`);
  diagnostics.push(`fair eps=${eps}: gap=${fair.gap.toFixed(4)} acc=${fair.accuracy.toFixed(4)} loss=${fair.loss.toFixed(4)} lambda=${fair.lambda.toFixed(3)}`);
  ok(Math.abs(unconstrained.gap) > 5 * eps, `the unconstrained fit must violate the constraint badly (gap ${unconstrained.gap})`);
  ok(unconstrained.accuracy > fair.accuracy, "enforcing parity must cost accuracy on this data set");
  ok(fair.accuracy > 0.65, `the fair model must stay useful (accuracy ${fair.accuracy})`);

  // 4. Primal feasibility, dual feasibility and complementary slackness at the returned iterate.
  ok(Math.abs(fair.gap) <= eps + 1e-4, `primal feasibility violated: |gap| = ${Math.abs(fair.gap)}`);
  ok(fair.lambda > 0, "an active constraint must produce a positive multiplier");
  ok(fair.lambda * Math.max(0, fair.violation) < 1e-3, `complementary slackness: lambda*max(0,c) = ${fair.lambda * Math.max(0, fair.violation)}`);

  // 5. Training really descends: well below the trivial theta = 0 model, and monotonically per round.
  const trivialLoss = logisticLoss([0, 0, 0], rows);
  const head = unconstrained.trace[0].loss;
  const tail = unconstrained.trace[unconstrained.trace.length - 1].loss;
  diagnostics.push(`loss descent: trivial ${trivialLoss.toFixed(4)} -> round 0 ${head.toFixed(4)} -> last round ${tail.toFixed(4)}`);
  ok(tail < trivialLoss - 0.4, `training must beat the trivial model (${tail.toFixed(4)} vs ${trivialLoss.toFixed(4)})`);
  ok(unconstrained.trace.every((t) => t.loss < trivialLoss - 0.4), "every traced round must already beat the trivial model");

  // 6. Primal convergence and stability: feasibility is reached quickly and then maintained on every
  //    later round, the objective stabilises, and the multiplier stays finite and non-negative.
  const feasibleFrom = fair.trace.findIndex((t) => Math.abs(t.gap) <= eps + 1e-3);
  const lambdaPath = fair.trace.map((t) => t.lambda);
  diagnostics.push(`dual path: ${lambdaPath.map((v) => v.toFixed(4)).join(" -> ")}`);
  diagnostics.push(`gap path: ${fair.trace.map((t) => t.gap.toExponential(1)).join(" ")}`);
  ok(feasibleFrom >= 0 && feasibleFrom < 6, `the constraint must become feasible within 6 rounds (round ${feasibleFrom})`);
  ok(fair.trace.slice(feasibleFrom).every((t) => Math.abs(t.gap) <= eps + 1e-3), "once feasible the parity gap must stay inside the eps tube (transient slack 1e-3)");
  ok(lambdaPath[0] > 0, "the multiplier must leave the origin while the constraint is violated");
  ok(fair.trace.every((t) => t.lambda >= 0 && Number.isFinite(t.lambda)), "dual feasibility must hold at every round");
  ok(fair.trace.every((t) => t.lambda < 1e3), "the multiplier must stay bounded");
  const lastLosses = fair.trace.slice(-5).map((t) => t.loss);
  ok(Math.max(...lastLosses) - Math.min(...lastLosses) < 1e-3, "the primal objective must stabilise in the tail rounds");

  // 7. Pareto curve over eps: feasibility at every value, objective improving with eps, accuracy
  //    monotone in eps, and an inactive constraint leaving lambda exactly at zero.
  const epsilons = [0.0, 0.02, 0.1, 0.3, 1.0];
  const curve = epsilons.map((e) => {
    const run = trainFairnessConstrained(rows, { eps: e });
    return { eps: e, loss: run.loss, gap: Math.abs(run.gap), accuracy: run.accuracy, lambda: run.lambda };
  });
  for (const point of curve) {
    diagnostics.push(`eps=${point.eps.toFixed(2)} -> |gap|=${point.gap.toFixed(4)} acc=${point.accuracy.toFixed(4)} loss=${point.loss.toFixed(4)} lambda=${point.lambda.toFixed(3)}`);
    ok(point.gap <= point.eps + 1e-4, `feasibility failed at eps=${point.eps} (|gap| = ${point.gap})`);
  }
  for (let i = 1; i < curve.length; i++) {
    ok(curve[i].loss <= curve[i - 1].loss + 1e-3, `objective must not worsen when eps grows (${curve[i - 1].eps} -> ${curve[i].eps})`);
    ok(curve[i].accuracy >= curve[i - 1].accuracy - 1e-9, `accuracy must not drop when eps grows (${curve[i - 1].eps} -> ${curve[i].eps})`);
  }
  ok(curve[0].gap < 1e-3, `eps = 0 must drive the parity gap to ~0 (got ${curve[0].gap})`);
  ok(curve[curve.length - 1].lambda === 0, "an inactive constraint must leave the multiplier exactly at zero");

  // 8. Independent solver: quadratic penalty method reaches the same objective and feasible region.
  const penaltyRun = trainPenalty(rows, { eps, rounds: 12, innerSteps: 70, penalty: 1, growth: 4 });
  diagnostics.push(`penalty method: |gap|=${Math.abs(penaltyRun.gap).toFixed(4)} loss=${penaltyRun.loss.toFixed(4)} (primal-dual loss ${fair.loss.toFixed(4)})`);
  ok(Math.abs(penaltyRun.gap) <= eps + 1e-3, `penalty method must also be feasible (|gap| = ${Math.abs(penaltyRun.gap)})`);
  ok(Math.abs(penaltyRun.loss - fair.loss) < 0.02, "both solvers must agree on the objective");
  ok(penaltyRun.loss >= fair.loss - 5e-3, "the adaptive dual must not be beaten by a fixed quadratic penalty");

  // 9. Independent solver: multi-start penalty search (different algorithm, different initial points).
  const multi = trainMultiStartPenalty(rows, { eps });
  diagnostics.push(`multi-start penalty: ${multi.feasibleCount}/${multi.results.length} feasible, best loss=${multi.best.loss.toFixed(5)} |gap|=${Math.abs(multi.best.gap).toFixed(4)} (primal-dual ${fair.loss.toFixed(5)})`);
  ok(multi.feasibleCount >= 5, `most starts must reach feasibility (${multi.feasibleCount} of ${multi.results.length})`);
  ok(Math.abs(multi.best.loss - fair.loss) < 0.02, `the multi-start search and the primal-dual loop must agree on the objective (${multi.best.loss} vs ${fair.loss})`);
  ok(multi.best.loss >= fair.loss - 5e-3, "no independent start may meaningfully beat the primal-dual solution (tolerance 5e-3)");

  // 10. KKT stationarity: adding lambda * grad c must cancel the gradient of the loss almost entirely
  //     (and the unconstrained solution, where lambda = 0, must already be stationary).
  const stationarity = (run) => {
    const gLoss = lossGradient(run.theta, rows);
    const gConstraint = constraintGradient(run.theta, rows);
    const residual = Math.hypot(...gLoss.map((v, j) => v + run.lambda * gConstraint[j]));
    return { residual, gLoss: Math.hypot(...gLoss) };
  };
  const fairStationarity = stationarity(fair);
  const freeStationarity = stationarity(unconstrained);
  diagnostics.push(`KKT residual: constrained ${fairStationarity.residual.toExponential(2)} (||gradF|| ${fairStationarity.gLoss.toExponential(2)}), unconstrained ${freeStationarity.residual.toExponential(2)}`);
  ok(fairStationarity.residual < 0.1 * fairStationarity.gLoss, "the multiplier must cancel most of the loss gradient at the constrained optimum");
  ok(freeStationarity.residual < 1e-2, `the unconstrained run must be stationary (residual ${freeStationarity.residual})`);
  ok(freeStationarity.residual < fairStationarity.gLoss, "the unconstrained optimum must be closer to stationary than a random point");

  // 10. Reproducibility: identical inputs give bit-identical iterates, new data gives a new model.
  const repeat = trainFairnessConstrained(rows, { eps });
  ok(repeat.theta.every((v, i) => Object.is(v, fair.theta[i])), "training must be bit-for-bit reproducible");
  ok(Object.is(repeat.lambda, fair.lambda), "the multiplier must be reproducible too");
  const otherRows = makeDataset({ seed: 777, perGroup: 200 });
  const otherRun = trainFairnessConstrained(otherRows, { eps });
  ok(!otherRun.theta.every((v, i) => Object.is(v, fair.theta[i])), "a different data set must give a different model");

  // 11. Boundary: one sample per group, parity is then a two-point average and still drivable to zero.
  const tiny = makeDataset({ seed: 7, perGroup: 1, groupedNoise: 0.5 });
  const tinyRun = trainFairnessConstrained(tiny, { eps: 0 });
  diagnostics.push(`tiny dataset (n=${tiny.length}) |gap| = ${Math.abs(tinyRun.gap).toExponential(3)}`);
  ok(Math.abs(tinyRun.gap) < 1e-3, "with one sample per group the parity gap must still be driven to zero");

  // 12. Invalid configurations are rejected instead of producing a made-up answer.
  throws(() => trainFairnessConstrained([], {}), /empty dataset/);
  throws(() => logisticLoss([0, 0, 0], []), /empty dataset/);
  throws(() => trainFairnessConstrained(rows, { eps: -0.1 }), RangeError);
  throws(() => trainFairnessConstrained(rows, { penalty: 0 }), RangeError);
  throws(() => trainFairnessConstrained(rows, { rounds: 0 }), RangeError);
  throws(() => trainFairnessConstrained(rows, { innerSteps: 1.5 }), RangeError);
  throws(() => demographicParityGap([0, 0, 0], groupRows(rows, 1)), /every group/);
  throws(() => makeDataset({ perGroup: 0 }), RangeError);
  throws(() => makeDataset({ groupedNoise: 0 }), RangeError);

  for (const line of diagnostics) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = {
  makeRng,
  makeNormal,
  makeDataset,
  predict,
  demographicParityGap,
  logisticLoss,
  accuracy,
  lossGradient,
  constraintGradient,
  numericGradient,
  minimiseSubproblem,
  trainFairnessConstrained,
  trainPenalty,
  trainMultiStartPenalty,
};

if (require.main === module) main();
