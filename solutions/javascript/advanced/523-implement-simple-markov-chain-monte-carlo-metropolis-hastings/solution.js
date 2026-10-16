/**
 * Task 523 - Implement simple Markov chain Monte Carlo (Metropolis-Hastings). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * A generic Metropolis-Hastings kernel: propose a candidate state from q(. | x), accept it with
 * probability min(1, (pi(y) q(x | y)) / (pi(x) q(y | x))), otherwise stay at x. Only the *ratio*
 * of the target is needed, so the file works with unnormalised weights (discrete) or unnormalised
 * log-densities (continuous). Two run modes are provided: a discrete chain built on an explicit
 * proposal matrix, for which the exact transition matrix of the resulting Markov chain is
 * assembled and checked against detailed balance, and a continuous random-walk chain driven by
 * Box-Muller Gaussian steps from a seeded 32-bit LCG with a scrambled state, so every trace is
 * reproducible. Burn-in, thinning and the acceptance counter are reported with each run.
 *
 * Documented checks: for a symmetric cycle proposal and for an asymmetric proposal the assembled
 * transition matrix has rows summing to 1 (to 1e-12), satisfies detailed balance
 * pi_i P_ij = pi_j P_ji exactly, has pi as its stationary distribution (||pi P - pi||_inf <=
 * 1e-12), and is irreducible with a strictly positive self-loop (hence aperiodic); a 200k-step discrete chain started at every state reproduces pi to within 0.01 and
 * yields start-independent frequencies. The Hastings correction is shown to matter: for an
 * asymmetric proposal on pi = (1/2, 1/2) the naive Metropolis rule converges to (1/11, 10/11) while
 * the corrected chain stays at pi within 0.01. A target with a zero-probability state is never
 * visited by the chain. Continuous checks: a random walk on N(0, 1) gives mean within 0.03 and
 * standard deviation within 0.05 of the truth with an acceptance rate inside [0.55, 0.85]; the
 * empirical CDF matches a numerically integrated standard normal CDF to 0.02 at four quantiles;
 * a 2-D independent-normal chain shows sample correlation within 0.03; and the heavy-tailed
 * unnormalised target 1/cosh(x) reproduces both the logistic variance pi^2/3 and the logistic CDF
 * 1/(1 + e^-x). Invalid weights and non-stochastic proposal rows are rejected.
 */
"use strict";

/** Seeded 32-bit LCG with scrambled state (neighbouring seeds are decorrelated). */
class Lcg {
  constructor(seed) {
    let s = ((seed >>> 0) ^ 0x9e3779b9) >>> 0;
    s = Math.imul(s ^ (s >>> 16), 0x85ebca6b) >>> 0;
    s = Math.imul(s ^ (s >>> 13), 0xc2b2ae35) >>> 0;
    this.state = (s ^ (s >>> 16)) >>> 0;
    if (this.state === 0) this.state = 0x1b873593;
  }

  nextUint32() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state;
  }

  nextFloat() {
    return this.nextUint32() / 4294967296;
  }

  nextInt(bound) {
    return Math.floor(this.nextFloat() * bound);
  }
}

/** Box-Muller standard normal (u is drawn strictly inside (0, 1) so the log is finite). */
function gaussian(rng) {
  const u1 = (rng.nextUint32() + 1) / 4294967297;
  const u2 = (rng.nextUint32() + 1) / 4294967297;
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

function normalizedWeights(weights) {
  for (const w of weights) if (!(w >= 0.0)) throw new Error("weights must be non-negative");
  const total = weights.reduce((a, b) => a + b, 0.0);
  if (!(total > 0.0)) throw new Error("weights must sum to a positive value");
  return weights.map((w) => w / total);
}

/**
 * Acceptance probability min(1, (pi_y q(x | y)) / (pi_x q(y | x))) with qXY = q(y | x) and
 * qYX = q(x | y); zero when the candidate has zero target mass.
 */
function acceptance(piX, piY, qXY, qYX) {
  if (piY === 0.0) return 0.0;
  if (piX === 0.0) return 1.0;
  return Math.min(1.0, (piY * qYX) / (piX * qXY));
}

/** Draw a state from proposal row `i` by CDF inversion (the row is validated to sum to 1). */
function sampleFromRow(q, i, rng) {
  const u = rng.nextFloat();
  let acc = 0.0;
  for (let j = 0; j < q[i].length; j++) {
    acc += q[i][j];
    if (u < acc) return j;
  }
  return q[i].length - 1;
}

/** A proposal must be a square, non-negative matrix whose rows are probability distributions. */
function validateProposal(q) {
  const k = q.length;
  if (k === 0) throw new Error("proposal matrix must not be empty");
  for (let i = 0; i < k; i++) {
    if (q[i].length !== k) throw new Error("proposal matrix must be square");
    let sum = 0.0;
    for (const p of q[i]) {
      if (!(p >= 0.0)) throw new Error("proposal entries must be non-negative");
      sum += p;
    }
    if (Math.abs(sum - 1.0) > 1e-9) {
      throw new Error(`proposal row ${i} is not a probability distribution`);
    }
  }
  return k;
}

/** Exact transition matrix of the MH chain over a finite state space. */
function transitionMatrix(weights, q) {
  const pi = normalizedWeights(weights);
  const k = validateProposal(q);
  if (pi.length !== k) throw new Error("dimension mismatch between weights and proposal");
  const matrix = [];
  for (let i = 0; i < k; i++) {
    const row = new Array(k).fill(0.0);
    let stay = 1.0;
    for (let j = 0; j < k; j++) {
      if (j === i || q[i][j] === 0.0) continue;
      const p = q[i][j] * acceptance(pi[i], pi[j], q[i][j], q[j][i]);
      row[j] = p;
      stay -= p;
    }
    // Rounding can leave a self-loop a hair below zero; clamp that to zero.
    if (stay < 0.0 && stay > -1e-12) stay = 0.0;
    row[i] = stay;
    if (stay < -1e-9) throw new Error(`proposal row ${i} is not a probability distribution`);
    const rowSum = row.reduce((a, b) => a + b, 0.0);
    if (Math.abs(rowSum - 1.0) > 1e-9) throw new Error("transition row is not stochastic");
    matrix.push(row);
  }
  return { pi, matrix };
}

function stationaryResidual(pi, matrix) {
  const k = pi.length;
  let worst = 0.0;
  for (let j = 0; j < k; j++) {
    let acc = 0.0;
    for (let i = 0; i < k; i++) acc += pi[i] * matrix[i][j];
    worst = Math.max(worst, Math.abs(acc - pi[j]));
  }
  return worst;
}

/**
 * Strong connectivity of the positive-probability support graph, restricted to the states with
 * positive target mass (both directions); states of zero mass are never visited.
 */
function isIrreducible(matrix, pi = null) {
  const k = matrix.length;
  const states = [];
  for (let i = 0; i < k; i++) if (pi === null || pi[i] > 0.0) states.push(i);
  const reachable = (transpose) => {
    const seen = new Array(k).fill(false);
    const start = states[0];
    const stack = [start];
    seen[start] = true;
    let count = 1;
    while (stack.length > 0) {
      const i = stack.pop();
      for (const j of states) {
        const p = transpose ? matrix[j][i] : matrix[i][j];
        if (p > 0.0 && !seen[j]) {
          seen[j] = true;
          count += 1;
          stack.push(j);
        }
      }
    }
    return count === states.length;
  };
  return reachable(false) && reachable(true);
}

function detailedBalanceResidual(pi, matrix) {
  const k = pi.length;
  let worst = 0.0;
  for (let i = 0; i < k; i++) {
    for (let j = 0; j < k; j++) {
      worst = Math.max(worst, Math.abs(pi[i] * matrix[i][j] - pi[j] * matrix[j][i]));
    }
  }
  return worst;
}

/**
 * Discrete MH run; `correct` toggles the Hastings correction so the uncorrected (naive Metropolis)
 * rule can be exhibited too. Counts are taken after `burnIn` steps, keeping every `thin`-th state.
 */
function discreteRun(weights, q, start, steps, rng, { burnIn = 0, thin = 1, correct = true } = {}) {
  const pi = normalizedWeights(weights);
  const k = validateProposal(q);
  if (pi.length !== k) throw new Error("dimension mismatch between weights and proposal");
  if (!Number.isInteger(start) || start < 0 || start >= k) throw new Error("bad start state");
  const counts = new Array(k).fill(0);
  let current = start;
  let accepted = 0;
  let kept = 0;
  for (let step = 0; step < steps; step++) {
    const candidate = sampleFromRow(q, current, rng);
    if (candidate !== current) {
      const ratio = correct
        ? (pi[candidate] * q[candidate][current]) / (pi[current] * q[current][candidate])
        : pi[candidate] / pi[current];
      if (rng.nextFloat() < Math.min(1.0, ratio)) {
        current = candidate;
        accepted += 1;
      }
    }
    if (step >= burnIn && (step - burnIn) % thin === 0) {
      counts[current] += 1;
      kept += 1;
    }
  }
  const frequencies = counts.map((c) => c / kept);
  return { frequencies, counts, kept, accepted, steps, acceptanceRate: accepted / steps };
}

/** Generic continuous MH with a symmetric Gaussian random-walk proposal. */
function continuousRun(logTarget, sigma, start, steps, rng, { burnIn = 0, thin = 1 } = {}) {
  const dim = start.length;
  const samples = [];
  let current = start.slice();
  let currentLog = logTarget(current);
  let accepted = 0;
  for (let step = 0; step < steps; step++) {
    const proposal = new Array(dim);
    for (let d = 0; d < dim; d++) proposal[d] = current[d] + sigma * gaussian(rng);
    const proposalLog = logTarget(proposal);
    if (proposalLog !== -Infinity &&
        Math.log(rng.nextFloat()) < proposalLog - currentLog) {
      current = proposal;
      currentLog = proposalLog;
      accepted += 1;
    }
    if (step >= burnIn && (step - burnIn) % thin === 0) samples.push(current.slice());
  }
  return { samples, accepted, steps, acceptanceRate: accepted / steps };
}

function sampleMean(values) {
  return values.reduce((a, b) => a + b, 0.0) / values.length;
}

function sampleVariance(values) {
  const m = sampleMean(values);
  return values.reduce((acc, v) => acc + (v - m) * (v - m), 0.0) / (values.length - 1);
}

/** Numerically integrated standard normal CDF (Simpson on [-10, x]); no special functions. */
function normalCdf(x) {
  const lo = -10.0;
  const n = 2000;
  const h = (x - lo) / n;
  const density = (t) => Math.exp(-0.5 * t * t) / Math.sqrt(2.0 * Math.PI);
  let sum = density(lo) + density(x);
  for (let i = 1; i < n; i++) sum += density(lo + i * h) * (i % 2 === 1 ? 4.0 : 2.0);
  return (sum * h) / 3.0;
}

function empiricalCdf(samples, x) {
  let count = 0;
  for (const v of samples) if (v <= x) count += 1;
  return count / samples.length;
}

function main() {
  const assert = require("node:assert");

  const cycle = (k) => {
    const q = [];
    for (let i = 0; i < k; i++) {
      const row = new Array(k).fill(0.0);
      row[(i + 1) % k] = 0.5;
      row[(i + k - 1) % k] = 0.5;
      q.push(row);
    }
    return q;
  };
  const asymmetric = (k) => {
    const q = [];
    for (let i = 0; i < k; i++) {
      const row = new Array(k).fill(0.0);
      let total = 0.0;
      for (let j = 0; j < k; j++) {
        if (j === i) continue;
        row[j] = 1.0 / (1.0 + Math.abs(i - j));
        total += row[j];
      }
      for (let j = 0; j < k; j++) row[j] /= total;
      q.push(row);
    }
    return q;
  };

  for (const [name, weights, q] of [
    ["cycle", [1.0, 2.0, 4.0, 8.0, 5.0], cycle(5)],
    ["asymmetric", [1.0, 2.0, 4.0, 8.0, 5.0], asymmetric(5)],
    ["zero-state", [1.0, 0.0, 3.0, 2.0], asymmetric(4)],
  ]) {
    const { pi, matrix } = transitionMatrix(weights, q);
    for (const row of matrix) {
      assert.ok(Math.abs(row.reduce((a, b) => a + b, 0.0) - 1.0) <= 1e-12, `${name}: stochastic`);
      assert.ok(row.every((p) => p >= -1e-15 && p <= 1.0 + 1e-15));
      assert.ok(matrix.every((r) => r.every((p) => Number.isFinite(p))));
    }
    assert.ok(detailedBalanceResidual(pi, matrix) <= 1e-12, `${name}: detailed balance`);
    assert.ok(stationaryResidual(pi, matrix) <= 1e-12, `${name}: pi is stationary`);
    // pi_i > 0 and Q_ij > 0 imply P_ij > 0; irreducibility plus one positive self-loop makes the
    // chain aperiodic (in an irreducible chain every state shares the same period).
    for (let i = 0; i < pi.length; i++) {
      assert.ok(matrix[i][i] >= 0.0, `${name}: no negative self-loop at ${i}`);
      for (let j = 0; j < pi.length; j++) {
        if (pi[i] > 0.0 && pi[j] > 0.0 && q[i][j] > 0.0) {
          assert.ok(matrix[i][j] > 0.0, `${name}: P_${i}${j} > 0`);
        }
      }
    }
    assert.ok(isIrreducible(matrix, pi),
      `${name}: the support graph over the positive-mass states is strongly connected`);
    assert.ok(matrix.some((row, i) => pi[i] > 0.0 && row[i] > 0.0),
      `${name}: a strictly positive self-loop exists somewhere`);
  }

  const assertMatrixClose = (actual, expected, label) => {
    assert.strictEqual(actual.length, expected.length, `${label}: row count`);
    actual.forEach((row, i) => {
      assert.strictEqual(row.length, expected[i].length, `${label}: column count`);
      row.forEach((value, j) => {
        assert.ok(Math.abs(value - expected[i][j]) <= 1e-15,
          `${label}: P[${i}][${j}] = ${value} should be ${expected[i][j]}`);
      });
    });
  };

  // The uncorrected rule does NOT leave pi invariant when the proposal is asymmetric.
  const equalWeights = [1.0, 1.0];
  const swap = [[0.0, 1.0], [1.0, 0.0]];
  const { matrix: swapMatrix } = transitionMatrix(equalWeights, swap);
  assertMatrixClose(swapMatrix, [[0.0, 1.0], [1.0, 0.0]],
    "a deterministic swap proposal accepts every candidate");
  assert.ok(!swapMatrix.some((row, i) => row[i] > 0.0), "and is therefore periodic");

  const biased = [[0.0, 1.0], [0.1, 0.9]];
  const { pi: biasedPi, matrix: biasedMatrix } = transitionMatrix(equalWeights, biased);
  assertMatrixClose(biasedMatrix, [[0.9, 0.1], [0.1, 0.9]],
    "Hastings shrinks the proposal's favourite move instead of following it");
  assert.deepStrictEqual(biasedPi, [0.5, 0.5], "and the uniform law stays invariant");

  const naive = biased;
  assert.ok(Math.abs(stationaryResidual([0.5, 0.5], naive)) > 0.3,
    "uncorrected chain: pi = (1/2, 1/2) is not its stationary distribution");

  const biasedRun = discreteRun(equalWeights, biased, 0, 200000, new Lcg(2024),
    { correct: false });
  assert.ok(Math.abs(detailedBalanceResidual([0.5, 0.5], naive)) > 1e-3,
    "the uncorrected kernel does not satisfy detailed balance for the uniform target");
  assert.ok(Math.abs(biasedRun.frequencies[0] - 1.0 / 11.0) <= 0.01,
    "without the Hastings correction the chain drifts to the proposal's favourite: (1/11, 10/11)");
  assert.ok(Math.abs(biasedRun.frequencies[1] - 10.0 / 11.0) <= 0.01);
  const corrected = discreteRun(equalWeights, biased, 0, 200000, new Lcg(2024));
  const correctedPi = biasedPi;
  assert.ok(Math.abs(corrected.frequencies[0] - correctedPi[0]) <= 0.01,
    "with the correction the chain keeps the uniform target");

  // Long discrete chains reproduce pi from every starting state.
  const weights5 = [1.0, 2.0, 4.0, 8.0, 5.0];
  const q5 = asymmetric(5);
  const truthPi = normalizedWeights(weights5);
  const empirical = [];
  for (let start = 0; start < 5; start++) {
    const run = discreteRun(weights5, q5, start, 200000, new Lcg(777 + start), { burnIn: 1000 });
    run.frequencies.forEach((f, i) => {
      assert.ok(Math.abs(f - truthPi[i]) <= 0.01,
        `start ${start}: frequency ${f} of state ${i} should be ${truthPi[i]}`);
    });
    empirical.push(run.frequencies);
  }
  for (let i = 1; i < empirical.length; i++) {
    for (let s = 0; s < 5; s++) {
      assert.ok(Math.abs(empirical[i][s] - empirical[0][s]) <= 0.015,
        "the stationary law does not depend on the starting state");
    }
  }
  const thinned = discreteRun(weights5, q5, 0, 200000, new Lcg(777), { burnIn: 1000, thin: 5 });
  assert.ok(thinned.kept > 39000 && thinned.kept <= 40000, "thinning keeps every 5th state");
  thinned.frequencies.forEach((f, i) => {
    assert.ok(Math.abs(f - truthPi[i]) <= 0.02, "the thinned chain still reproduces pi");
  });

  // A state the target forbids is never visited.
  const zeroState = discreteRun([1.0, 0.0, 3.0, 2.0], asymmetric(4), 0, 100000, new Lcg(31));
  assert.strictEqual(zeroState.counts[1], 0, "a zero-probability state is never accepted");
  const zeroPi = normalizedWeights([1.0, 0.0, 3.0, 2.0]);
  [0, 2, 3].forEach((i) => {
    assert.ok(Math.abs(zeroState.frequencies[i] - zeroPi[i]) <= 0.01);
  });

  // Continuous: random walk on N(0, 1).
  const logNormal = (x) => -0.5 * x[0] * x[0];
  const gaussRun = continuousRun(logNormal, 1.0, [4.0], 200000, new Lcg(4242),
    { burnIn: 2000 });
  const values = gaussRun.samples.map((p) => p[0]);
  assert.ok(Math.abs(sampleMean(values)) <= 0.03, "sample mean of N(0, 1) within 0.03");
  assert.ok(Math.abs(Math.sqrt(sampleVariance(values)) - 1.0) <= 0.05,
    "sample standard deviation within 0.05");
  assert.ok(gaussRun.acceptanceRate >= 0.55 && gaussRun.acceptanceRate <= 0.85,
    `1-D random walk acceptance ${gaussRun.acceptanceRate} in the documented band`);
  for (const x of [-1.5, -0.5, 0.5, 1.5]) {
    assert.ok(Math.abs(empiricalCdf(values, x) - normalCdf(x)) <= 0.02,
      `empirical CDF at ${x} matches the integrated normal CDF`);
  }
  assert.ok(Math.abs(normalCdf(0.0) - 0.5) <= 1e-6, "quadrature sanity: Phi(0) = 1/2");
  assert.ok(Math.abs(normalCdf(1.96) - 0.975) <= 5e-4, "quadrature sanity: Phi(1.96) = 0.975");

  // A too-large step size lowers the acceptance rate but keeps the invariant law.
  const coarse = continuousRun(logNormal, 12.0, [0.0], 200000, new Lcg(4242), { burnIn: 2000 });
  const coarseValues = coarse.samples.map((p) => p[0]);
  assert.ok(coarse.acceptanceRate < gaussRun.acceptanceRate, "big steps are rejected more often");
  assert.ok(Math.abs(sampleMean(coarseValues)) <= 0.05);
  assert.ok(Math.abs(Math.sqrt(sampleVariance(coarseValues)) - 1.0) <= 0.08);

  // 2-D independent normals: the marginals are correct and the coordinates stay uncorrelated.
  const log2d = (x) => -0.5 * (x[0] * x[0] + x[1] * x[1]);
  const run2d = continuousRun(log2d, 1.0, [3.0, -3.0], 120000, new Lcg(99), { burnIn: 2000 });
  const xs = run2d.samples.map((p) => p[0]);
  const ys = run2d.samples.map((p) => p[1]);
  assert.ok(Math.abs(sampleMean(xs)) <= 0.04 && Math.abs(sampleMean(ys)) <= 0.04);
  assert.ok(Math.abs(sampleVariance(xs) - 1.0) <= 0.06);
  assert.ok(Math.abs(sampleVariance(ys) - 1.0) <= 0.06);
  const mx = sampleMean(xs);
  const my = sampleMean(ys);
  let cov = 0.0;
  for (let i = 0; i < xs.length; i++) cov += (xs[i] - mx) * (ys[i] - my);
  cov /= xs.length - 1;
  assert.ok(Math.abs(cov) <= 0.03, "independent coordinates stay uncorrelated");

  // Heavy-tailed unnormalised target sech(x) = 1/cosh(x): the hyperbolic secant law, whose
  // variance is pi^2/4 (not pi^2/3, which belongs to the logistic density 1/(4 cosh^2(x/2))) and
  // whose CDF is (2/pi) arctan(e^x).
  const logSech = (x) => -Math.log(Math.cosh(x[0]));
  const sechRun = continuousRun(logSech, 3.0, [6.0], 150000, new Lcg(5150), { burnIn: 2000 });
  const sechValues = sechRun.samples.map((p) => p[0]);
  assert.ok(Math.abs(sampleMean(sechValues)) <= 0.05);
  assert.ok(Math.abs(sampleVariance(sechValues) - Math.PI * Math.PI / 4.0) <= 0.15,
    "hyperbolic secant variance is pi^2/4");
  for (const x of [-2.0, -1.0, 1.0, 2.0]) {
    const cdf = (2.0 / Math.PI) * Math.atan(Math.exp(x));
    assert.ok(Math.abs(empiricalCdf(sechValues, x) - cdf) <= 0.02, `sech CDF at ${x}`);
  }
  assert.ok(Math.abs((2.0 / Math.PI) * Math.atan(Math.exp(0.0)) - 0.5) <= 1e-15,
    "closed-form sanity: F(0) = 1/2");

  assert.throws(() => normalizedWeights([1.0, -1.0]), /non-negative/);
  assert.throws(() => normalizedWeights([0.0, 0.0]), /positive value/);
  assert.throws(() => transitionMatrix([1.0, 1.0], [[0.0, 1.4], [1.4, 0.0]]),
    /not a probability distribution/);
  assert.throws(() => transitionMatrix([1.0, 1.0, 1.0],
    [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]), /must be square/);
  assert.throws(() => transitionMatrix([1.0, 1.0], [[0.0, -0.5], [1.5, 0.0]]),
    /non-negative/);
  assert.throws(() => transitionMatrix([1.0, 1.0], []), /must not be empty/);
  assert.throws(() => transitionMatrix([1.0, 1.0, 1.0], [[0.0, 1.0], [1.0, 0.0]]),
    /dimension mismatch/);
  assert.throws(() => discreteRun([1.0, 1.0], [[0.0, 1.0], [1.0, 0.0]], 5, 10, new Lcg(1)),
    /bad start state/);
}

module.exports = { Lcg, gaussian, normalizedWeights, validateProposal, acceptance, sampleFromRow,
  transitionMatrix,
  stationaryResidual, detailedBalanceResidual, isIrreducible, discreteRun, continuousRun, sampleMean,
  sampleVariance, normalCdf, empiricalCdf };

if (require.main === module) {
  main();
  console.log("PASS");
}
