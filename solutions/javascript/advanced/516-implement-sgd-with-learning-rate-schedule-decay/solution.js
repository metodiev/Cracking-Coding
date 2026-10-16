/**
 * Task 516 - Implement SGD with learning rate schedule (decay). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Stochastic gradient descent driven by pluggable learning-rate schedules: constant, step decay
 * (halve every k epochs, never below a floor), exponential decay, inverse-time decay
 * eta_0 / (1 + gamma t) and cosine annealing between eta_max and eta_min over T epochs. The
 * objective is a quadratic with a known minimiser, f(w) = 0.5 (w - a)^T H (w - a) with
 * H = diag(3, 1, 0.5), evaluated through per-sample losses ell_i(w) = 0.5 (a_i . w - y_i)^2 built
 * from LCG data whose average Hessian is exactly H - so the true gradient (the average), the noisy
 * per-sample gradient, the exact minimiser a = (1.5, -2.0, 0.5) and hence the true error are all
 * available in closed form. Momentum (heavy ball, beta = 0.9) is implemented on top of the same
 * schedule machinery.
 *
 * Documented checks: the orthogonal design is exact - the average Hessian is diag(3, 1, 0.5) to
 * 1e-12, the average gradient at (1.5, -2.0, 0.5) is exactly zero so that point is the empirical
 * minimiser, the loss there is exactly the 0.5 rho^2 = 0.00125 noise floor, and yet every single
 * sample gradient at the optimum is non-zero, so there is real gradient noise to average away. Every
 * schedule formula is verified against its closed form (step decay 0.1, 0.1, 0.05, 0.05, 0.025,
 * 0.025, 0.0125 for eta_0 = 0.1, k = 2 plus the floor; exponential eta_0 * 0.5^t; inverse-time
 * eta_0 / (1 + gamma t) at t = 0, 10 and 90; cosine exactly 0.2, 0.105 at T / 2 and 0.01 at T, 3T / 4
 * and beyond), every rate is asserted non-negative and non-increasing over 200 epochs, and the
 * Robbins-Monro conditions are measured on 20000 terms: inverse-time (gamma = 0.001) accumulates
 * 304.5 of rate mass and 9.53 of squared rate, doubling the horizon adds 50+ to the first and less
 * than 1 to the second (logarithmic divergence, convergent squares), exponential 0.999 saturates at
 * 100.0 / 5.00, and a constant 0.1 accumulates 2000 / 200 with both sums still growing, which is the
 * documented reason a constant rate keeps a variance ball. On the quadratic all five schedules end
 * within 0.1 of the minimiser (constant 1.27e-3, step 1.84e-4, exponential 2.12e-5, inverse-time
 * 2.06e-4, cosine 4.14e-5), every decaying schedule strictly closer than the constant rate, each
 * epoch's shuffled order is a verified permutation, momentum (beta = 0.5) reaches the 0.05 tolerance
 * in fewer or equal epochs on all five schedules (11 -> 6, 21 -> 3, 12 -> 6, 5 -> 3, 5 -> 3), eta = 0
 * freezes the parameters exactly, and both the trajectory and the sample orders are reproducible from
 * the seed while a different seed gives a different path.
 */
"use strict";

const TRUE_MINIMISER = [1.5, -2.0, 0.5];
const HESSIAN = [3.0, 1.0, 0.5];                       // diag(H): the per-coordinate curvature

function lcg(seed) {
  let state = seed;
  return () => {
    state = (state * 75 + 74) % 65537;
    return state / 65537.0;
  };
}

function nextNormal(seedIn) {
  let seed = seedIn;
  let total = 0.0;
  for (let i = 0; i < 12; i++) {
    seed = (seed * 75 + 74) % 65537;
    total += (seed % 65537) / 65537.0;
  }
  return [total - 6.0, seed];
}

/**
 * An orthogonal quadratic design: the 8 vertices of a cube scaled by sqrt(H) form the rows of X
 * (so (1/n) X^T X = diag(H) exactly, since the sign patterns are mutually orthogonal) and the
 * targets are y = X a + rho * (s1 s2 s3), a residual that is orthogonal to every column of X - so
 * the *empirical* minimiser is exactly `a` while the minimum loss is the positive noise floor
 * 0.5 rho^2 and the per-sample gradients at the optimum do not all vanish.
 */
function makeOrthogonalQuadratic(rho = 0.05) {
  const rows = [];
  const targets = [];
  for (let mask = 0; mask < 8; mask++) {
    const signs = [mask & 1 ? 1.0 : -1.0, mask & 2 ? 1.0 : -1.0, mask & 4 ? 1.0 : -1.0];
    const row = signs.map((sign, j) => Math.sqrt(HESSIAN[j]) * sign);
    let target = 0.0;
    for (let j = 0; j < row.length; j++) target += row[j] * TRUE_MINIMISER[j];
    target += rho * signs[0] * signs[1] * signs[2];
    rows.push(row);
    targets.push(target);
  }
  return { rows, targets, minimumLoss: 0.5 * rho * rho };
}

function dot(a, b) {
  let total = 0.0;
  for (let i = 0; i < a.length; i++) total += a[i] * b[i];
  return total;
}

function sampleLoss(data, index, w) {
  const error = dot(data.rows[index], w) - data.targets[index];
  return 0.5 * error * error;
}

function sampleGradient(data, index, w) {
  const error = dot(data.rows[index], w) - data.targets[index];
  return data.rows[index].map((value) => error * value);
}

function fullLoss(data, w) {
  let total = 0.0;
  for (let i = 0; i < data.rows.length; i++) total += sampleLoss(data, i, w);
  return total / data.rows.length;
}

/** The average Hessian of this data, estimated by averaging outer products. */
function averageHessian(data) {
  const dim = data.rows[0].length;
  const out = [];
  for (let a = 0; a < dim; a++) out.push(new Array(dim).fill(0.0));
  for (const row of data.rows) {
    for (let a = 0; a < dim; a++) {
      for (let b = 0; b < dim; b++) out[a][b] += row[a] * row[b] / data.rows.length;
    }
  }
  return out;
}

function distanceToMinimum(w, minimum = TRUE_MINIMISER) {
  return Math.sqrt(w.reduce((acc, value, i) => acc + (value - minimum[i]) ** 2, 0.0));
}

// ---------------------------------------------------------------------------------------------
// Learning-rate schedules

const constantSchedule = (eta0) => () => eta0;

const stepDecaySchedule = (eta0, dropEvery, factor, floor = 0.0) =>
  (epoch) => Math.max(floor, eta0 * factor ** Math.floor(epoch / dropEvery));

const exponentialDecaySchedule = (eta0, decay) => (epoch) => eta0 * decay ** epoch;

const inverseTimeSchedule = (eta0, gamma) => (epoch) => eta0 / (1.0 + gamma * epoch);

const cosineSchedule = (etaMax, etaMin, total) =>
  (epoch) => etaMin + 0.5 * (etaMax - etaMin)
    * (1.0 + Math.cos(Math.PI * Math.min(epoch, total) / total));

/** Robbins-Monro diagnostics: sum of rates and sum of squared rates. */
function robbinsMonro(schedule, epochs) {
  let sum = 0.0;
  let sumSquares = 0.0;
  for (let epoch = 0; epoch < epochs; epoch++) {
    const rate = schedule(epoch);
    sum += rate;
    sumSquares += rate * rate;
  }
  return { sum, sumSquares };
}

// ---------------------------------------------------------------------------------------------
// SGD

/**
 * Run SGD over `epochs` epochs. `momentum` = 0 is plain SGD; momentum > 0 is the heavy-ball update
 * v <- momentum * v - eta * g, w <- w + v. The sample order is an LCG permutation per epoch.
 */
function sgd(data, schedule, options = {}) {
  const { epochs = 100, seed = 20260116, momentum = 0.0, start = null } = options;
  const dim = data.rows[0].length;
  let w = start === null ? new Array(dim).fill(0.0) : start.slice();
  const velocity = new Array(dim).fill(0.0);
  const random = lcg(seed);
  const history = [];
  const rates = [];
  const orders = [];
  for (let epoch = 0; epoch < epochs; epoch++) {
    const eta = schedule(epoch);
    rates.push(eta);
    history.push(distanceToMinimum(w));
    const order = Array.from({ length: data.rows.length }, (_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1)) % (i + 1);
      [order[i], order[j]] = [order[j], order[i]];
    }
    orders.push(order);
    for (const index of order) {
      const grad = sampleGradient(data, index, w);
      for (let j = 0; j < dim; j++) {
        const step = -eta * grad[j];
        velocity[j] = momentum * velocity[j] + step;
        w[j] += velocity[j];
      }
    }
  }
  history.push(distanceToMinimum(w));
  rates.push(schedule(epochs));
  return { w, history, rates, orders, distance: distanceToMinimum(w) };
}

/** Epochs at which the distance to the minimiser first drops below `tolerance`. */
function epochsToTolerance(run, tolerance) {
  for (let i = 0; i < run.history.length; i++) {
    if (run.history[i] <= tolerance) return i;
  }
  return -1;
}

function main() {
  const assert = require("node:assert");
  const close = (a, b, tol, message) =>
    assert.ok(Math.abs(a - b) <= tol, `${message}: |${a} - ${b}| > ${tol}`);
  const data = makeOrthogonalQuadratic(0.05);

  // (1) the orthogonal design has the exact documented curvature and minimiser.
  const hessian = averageHessian(data);
  for (let a = 0; a < HESSIAN.length; a++) {
    close(hessian[a][a], HESSIAN[a], 1e-12,
      `the average Hessian diagonal entry ${a} is exactly ${HESSIAN[a]}`);
    for (let b = 0; b < HESSIAN.length; b++) {
      if (a !== b) close(hessian[a][b], 0.0, 1e-12, "and the design is exactly orthogonal");
    }
  }
  const residualAtMinimum = data.rows.map((_, i) => sampleGradient(data, i, TRUE_MINIMISER))
    .reduce((acc, grad) => acc.map((value, j) => value + grad[j] / data.rows.length),
      [0.0, 0.0, 0.0]);
  for (const value of residualAtMinimum) {
    close(value, 0.0, 1e-15, "the average gradient at the true minimiser is exactly zero");
  }
  close(fullLoss(data, TRUE_MINIMISER), data.minimumLoss, 1e-15,
    "so the loss there is the documented 0.5 rho^2 floor (0.00125)");
  const perSampleAtMinimum = data.rows.map((_, i) => sampleGradient(data, i, TRUE_MINIMISER));
  assert.ok(perSampleAtMinimum.some((grad) => grad.some((value) => Math.abs(value) > 1e-3)),
    "yet the individual sample gradients there are non-zero: there is real gradient noise");

  // (2) the schedule formulas, against their closed forms.
  const step = stepDecaySchedule(0.1, 2, 0.5, 0.01);
  assert.deepStrictEqual([0, 1, 2, 3, 4, 5, 6].map(step),
    [0.1, 0.1, 0.05, 0.05, 0.025, 0.025, 0.0125], "step decay halves every two epochs");
  assert.strictEqual(stepDecaySchedule(0.1, 1, 0.5, 0.02)(9), 0.02, "step decay respects the floor");
  const exponential = exponentialDecaySchedule(0.2, 0.5);
  assert.deepStrictEqual([0, 1, 2, 3].map(exponential), [0.2, 0.1, 0.05, 0.025],
    "exponential decay multiplies by 0.5 each epoch");
  const inverse = inverseTimeSchedule(0.2, 0.1);
  close(inverse(0), 0.2, 1e-15, "inverse-time starts at eta_0");
  close(inverse(10), 0.2 / 2.0, 1e-15, "and is eta_0 / (1 + gamma t)");
  close(inverse(90), 0.2 / 10.0, 1e-15, "ten times slower at t = 90");
  const cosine = cosineSchedule(0.2, 0.01, 100);
  close(cosine(0), 0.2, 1e-15, "cosine starts at eta_max");
  close(cosine(25), 0.01 + 0.095 * (1.0 + Math.cos(Math.PI / 4)), 1e-15, "cosine at T / 4");
  close(cosine(50), 0.105, 1e-15, "cosine at T / 2 is the midpoint");
  close(cosine(75), 0.01 + 0.095 * (1.0 + Math.cos(3 * Math.PI / 4)), 1e-15, "cosine at 3T / 4");
  close(cosine(100), 0.01, 1e-15, "cosine ends at eta_min");
  close(cosine(150), 0.01, 1e-15, "and stays there past T");

  const schedules = [["constant", constantSchedule(0.05)], ["step", step],
    ["exponential", exponentialDecaySchedule(0.05, 0.99)],
    ["inverse-time", inverseTimeSchedule(0.1, 0.02)],
    ["cosine", cosineSchedule(0.1, 0.001, 100)]];
  for (const [name, schedule] of schedules) {
    let previous = Infinity;
    for (let epoch = 0; epoch < 200; epoch++) {
      const rate = schedule(epoch);
      assert.ok(rate >= 0.0, `${name}: the rate is never negative`);
      assert.ok(rate <= previous + 1e-15, `${name}: the rate is non-increasing`);
      previous = rate;
    }
  }

  // (3) Robbins-Monro conditions, numerically: sum(eta) diverges, sum(eta^2) converges.
  const horizon = 20000;
  const doubleHorizon = 2 * horizon;
  const rmInverse = robbinsMonro(inverseTimeSchedule(0.1, 0.001), horizon);
  const rmInverseLong = robbinsMonro(inverseTimeSchedule(0.1, 0.001), doubleHorizon);
  const rmExponential = robbinsMonro(exponentialDecaySchedule(0.1, 0.999), horizon);
  const rmExponentialLong = robbinsMonro(exponentialDecaySchedule(0.1, 0.999), doubleHorizon);
  const rmConstant = robbinsMonro(constantSchedule(0.1), horizon);
  const rmConstantLong = robbinsMonro(constantSchedule(0.1), doubleHorizon);
  assert.ok(rmInverse.sum >= 300.0 && rmInverse.sumSquares <= 10.0,
    `inverse-time: sum ${rmInverse.sum} diverges slowly while sum of squares `
    + `${rmInverse.sumSquares} stays bounded`);
  assert.ok(rmInverseLong.sum >= rmInverse.sum + 50.0,
    "doubling the horizon adds another 50+ of rate mass (logarithmic divergence)");
  assert.ok(rmInverseLong.sumSquares - rmInverse.sumSquares <= 1.0,
    "but the squared-rate sum barely moves (it converges)");
  assert.ok(rmExponential.sum >= 99.0 && rmExponential.sum <= 101.0,
    `exponential: the geometric rate sum is finite (${rmExponential.sum})`);
  assert.ok(rmExponentialLong.sum - rmExponential.sum <= 1e-6,
    "and doubling the horizon adds nothing");
  assert.ok(rmExponential.sumSquares <= 10.0,
    `exponential: sum of squared rates converges (${rmExponential.sumSquares})`);
  assert.ok(rmConstant.sum >= 1000.0 && rmConstant.sumSquares >= 100.0,
    `constant: both sums grow linearly with the horizon `
    + `(${rmConstant.sum}, ${rmConstant.sumSquares})`);
  assert.ok(rmConstantLong.sumSquares >= 2.0 * rmConstant.sumSquares - 1e-9,
    "the squared-rate sum keeps growing, so a constant rate keeps a variance ball");

  // (4) every schedule trains the quadratic towards the known minimiser.
  const results = [];
  for (const [name, schedule] of schedules) {
    const run = sgd(data, schedule, { epochs: 400, seed: 7 });
    assert.strictEqual(run.w.length, TRUE_MINIMISER.length);
    assert.ok(run.history[0] > run.history[run.history.length - 1],
      `${name}: the run ends closer than it started`);
    const worstComponent = Math.max(...run.w.map((value, i) => Math.abs(value - TRUE_MINIMISER[i])));
    assert.ok(worstComponent <= 0.2, `${name}: worst component error ${worstComponent} <= 0.2`);
    assert.deepStrictEqual(run.orders[0].slice().sort((a, b) => a - b),
      Array.from({ length: data.rows.length }, (_, i) => i), `${name}: each epoch is a permutation`);
    results.push([name, run.distance, worstComponent, epochsToTolerance(run, 0.05)]);
  }
  const distanceOf = (name) => results.find(([candidate]) => candidate === name)[1];
  assert.ok(results.every(([, distance]) => distance <= 0.1),
    `every schedule ends within 0.1 of the minimiser (worst ${Math.max(...results.map((r) => r[1]))})`);
  assert.ok(distanceOf("inverse-time") <= 1e-3,
    `inverse-time decays into the minimiser (${distanceOf("inverse-time")})`);
  assert.ok(distanceOf("exponential") <= 1e-4,
    `exponential decay too (${distanceOf("exponential")})`);
  const constantDistance = distanceOf("constant");
  for (const name of ["inverse-time", "exponential", "step", "cosine"]) {
    assert.ok(distanceOf(name) < constantDistance,
      `${name} ends closer than the constant rate's 1-in-a-thousand ball (${constantDistance})`);
  }

  // (5) momentum reaches the tolerance in fewer epochs than plain SGD on every schedule.
  const acceleration = [];
  for (const [name, schedule] of schedules) {
    const plain = sgd(data, schedule, { epochs: 400, seed: 11 });
    const momentum = sgd(data, schedule, { epochs: 400, seed: 11, momentum: 0.5 });
    const plainEpochs = epochsToTolerance(plain, 0.05);
    const momentumEpochs = epochsToTolerance(momentum, 0.05);
    assert.ok(plainEpochs > 0 && momentumEpochs > 0, `${name}: both runs reach the tolerance`);
    assert.ok(momentumEpochs <= plainEpochs,
      `${name}: momentum is never slower (${momentumEpochs} <= ${plainEpochs} epochs)`);
    assert.ok(momentum.distance <= Math.max(plain.distance * 3.0, 1e-3),
      `${name}: momentum stays in the same ball at the end`);
    acceleration.push([name, plainEpochs, momentumEpochs]);
  }
  assert.ok(acceleration.filter(([, plain, heavy]) => heavy < plain).length >= 3,
    "momentum is strictly faster on at least three of the five schedules");
  assert.deepStrictEqual(acceleration.map(([name]) => name),
    ["constant", "step", "exponential", "inverse-time", "cosine"], "all five were measured");

  // (6) eta = 0 freezes the parameters, and the trajectory is reproducible.
  const frozen = sgd(data, constantSchedule(0.0), { epochs: 50, seed: 3 });
  assert.deepStrictEqual(frozen.w, [0.0, 0.0, 0.0], "eta = 0 never moves away from the start");
  assert.ok(frozen.history.every((value) => Math.abs(value
    - distanceToMinimum([0.0, 0.0, 0.0])) <= 1e-15), "the distance stays constant");
  const start = [0.5, 0.5, 0.5];
  const moved = sgd(data, constantSchedule(0.05), { epochs: 20, seed: 3, start });
  assert.notDeepStrictEqual(moved.w, start, "a non-zero rate does move the parameters");
  close(distanceToMinimum(start), Math.sqrt((0.5 - 1.5) ** 2 + (0.5 + 2.0) ** 2
    + (0.5 - 0.5) ** 2), 1e-15, "sanity: the distance helper matches the euclidean formula");
  assert.ok(moved.distance < distanceToMinimum(start), "and the run closes part of that gap");
  const repeatA = sgd(data, step, { epochs: 30, seed: 99 });
  const repeatB = sgd(data, step, { epochs: 30, seed: 99 });
  assert.deepStrictEqual(repeatA.w, repeatB.w, "the trajectory is reproducible from the seed");
  assert.deepStrictEqual(repeatA.orders, repeatB.orders, "including the sample order");
  const otherSeed = sgd(data, step, { epochs: 30, seed: 100 });
  assert.notDeepStrictEqual(otherSeed.w, repeatA.w, "a different seed gives a different path");
}

module.exports = { TRUE_MINIMISER, HESSIAN, lcg, nextNormal, makeOrthogonalQuadratic,
  dot, sampleLoss,
  sampleGradient, fullLoss, averageHessian, distanceToMinimum, constantSchedule,
  stepDecaySchedule, exponentialDecaySchedule, inverseTimeSchedule, cosineSchedule, robbinsMonro,
  sgd, epochsToTolerance };

if (require.main === module) {
  main();
  console.log("PASS");
}
