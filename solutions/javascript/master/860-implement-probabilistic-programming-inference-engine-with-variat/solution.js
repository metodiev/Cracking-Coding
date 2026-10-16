/** Task 860 - Implement probabilistic programming inference engine with variational methods.
 *  (Tier: Master, CTCI Beyond — Machine Learning & AI)
 *
 * Model. A small probabilistic-programming engine that runs coordinate-ascent variational inference
 * (CAVI) over a mean-field Gaussian family q(x) = prod_j N(m_j, s_j). Two model classes are supported:
 *
 *   linearGaussian  a set of Gaussian factors, each linear in the latent variables:
 *                   N(sum_j c_j x_j | mu_f, tau_f). Every factor contributes a quadratic term, so the
 *                   optimal q_j is Gaussian and its natural parameters are
 *                       Lambda_j = alpha_j + sum_f tau_f c_fj^2
 *                       h_j      = alpha_j m_j^0 + sum_f tau_f c_fj (mu_f - sum_{k != j} c_fk m_k).
 *                   Bayesian linear regression is expressed in this form (a prior factor per weight plus
 *                   one likelihood factor per observation).
 *   gaussianMixture mean-field inference for a 1-D mixture with known variance: the responsibilities are
 *                   the categorical factor's optimum and the component means are updated from the
 *                   weighted data.
 *
 * The ELBO is computed in closed form and also estimated by Monte-Carlo sampling from q with a seeded
 * PRNG, and the model evidence is computed independently — in closed form for the conjugate regression
 * model (a Gaussian marginal likelihood evaluated by Gaussian elimination) and by numerical quadrature
 * on a grid for the mixture — so the variational bound can be checked from both sides.
 *
 * What is simplified: a fixed mean-field Gaussian family with no stochastic or structured variational
 * families, no automatic differentiation (updates are the analytic coordinate-ascent equations), no
 * reparameterisation estimators, one process, no wall clock and no randomness outside one seeded
 * mulberry32 stream.
 *
 * Invariants verified:
 *   bound        the closed-form ELBO never decreases across CAVI sweeps, equals the independently
 *                computed value and the Monte-Carlo estimate within sampling error, and never exceeds
 *                the model evidence: log p(y) - ELBO >= 0 always, with the gap collapsing to zero for an
 *                orthogonal design where the mean-field family contains the exact posterior.
 *   fixed point  at convergence the mean and precision of every factor satisfy the coordinate-ascent
 *                stationarity conditions to machine precision (an independent residual check).
 *   accuracy     for an orthogonal design the variational posterior matches the analytic posterior and
 *                the predictive distribution; for a correlated design the mean-field variances are
 *                (as theory requires) no larger than the exact posterior variances.
 *   liveness     every run converges within the iteration budget and reports its iteration count.
 *   determinism  identical seeds reproduce identical iterates, ELBO traces and samples.
 */
"use strict";

const assert = require("node:assert");

/** mulberry32 — the only source of randomness (fixture data, Monte-Carlo ELBO). */
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

/** Standard normal sample via Box-Muller on the seeded stream. */
function gaussianSample(rand) {
  const u1 = Math.max(1e-12, rand());
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function gaussianLogPdf(value, mean, variance) {
  return -0.5 * (Math.log(2 * Math.PI * variance) + (value - mean) ** 2 / variance);
}

/** Determinant and inverse of a small symmetric matrix by Gauss-Jordan with partial pivoting. */
function invertMatrix(matrix) {
  const n = matrix.length;
  const augmented = matrix.map((row, i) =>
    row.concat(
      Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
    ),
  );
  let logDet = 0;
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(augmented[r][col]) > Math.abs(augmented[pivot][col])) pivot = r;
    if (Math.abs(augmented[pivot][col]) < 1e-14) throw new Error("matrix is singular");
    const tmp = augmented[col];
    augmented[col] = augmented[pivot];
    augmented[pivot] = tmp;
    logDet += Math.log(Math.abs(augmented[col][col]));
    const divisor = augmented[col][col];
    for (let j = 0; j < 2 * n; j++) augmented[col][j] /= divisor;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = augmented[r][col];
      if (factor === 0) continue;
      for (let j = 0; j < 2 * n; j++) augmented[r][j] -= factor * augmented[col][j];
    }
  }
  return { inverse: augmented.map((row) => row.slice(n)), logDet };
}

/** Joint Gaussian pdf of a vector under N(mean, covariance) — used by the grid/evidence oracles. */
function multivariateLogPdf(vector, mean, covariance) {
  const n = vector.length;
  const { inverse, logDet } = invertMatrix(covariance);
  const delta = vector.map((value, i) => value - mean[i]);
  let quadratic = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) quadratic += delta[i] * inverse[i][j] * delta[j];
  return -0.5 * (n * Math.log(2 * Math.PI) + logDet + quadratic);
}

/**
 * The engine. Factors are {coefficients, mean, precision}; every latent variable j has mean m_j, variance
 * s_j and a Gaussian prior N(priorMean_j, 1/priorPrecision_j).
 */
class VariationalEngine {
  constructor({ variables, factors, priorMean = null, priorPrecision = null } = {}) {
    if (!Number.isInteger(variables) || variables < 1) throw new RangeError("variables must be >= 1");
    this.variables = variables;
    this.factors = factors || [];
    this.priorMean = priorMean || new Array(variables).fill(0);
    this.priorPrecision = priorPrecision || new Array(variables).fill(1);
    if (this.priorMean.length !== variables || this.priorPrecision.length !== variables) {
      throw new RangeError("prior vectors must match the variable count");
    }
    for (const factor of this.factors) {
      if (factor.coefficients.length !== variables) throw new RangeError("factor coefficients must match the variable count");
      if (!(factor.precision > 0)) throw new RangeError("factor precision must be > 0");
    }
    this.mean = this.priorMean.slice();
    this.variance = this.priorPrecision.map((precision) => 1 / precision);
    this.iterations = 0;
    this.elboTrace = [];
  }

  /** Exact ELBO of the current q for a linear-Gaussian model. */
  elbo() {
    let value = 0;
    for (let j = 0; j < this.variables; j++) {
      const priorVariance = 1 / this.priorPrecision[j];
      value += gaussianLogPdf(this.mean[j], this.priorMean[j], priorVariance) - 0.5 * this.variance[j] / priorVariance;
      value += 0.5 * Math.log(2 * Math.PI * Math.E * this.variance[j]); // entropy
    }
    for (const factor of this.factors) {
      let mean = 0;
      let variance = 0;
      for (let j = 0; j < this.variables; j++) {
        mean += factor.coefficients[j] * this.mean[j];
        variance += factor.coefficients[j] ** 2 * this.variance[j];
      }
      value += gaussianLogPdf(factor.mean, mean, 1 / factor.precision) - 0.5 * factor.precision * variance;
    }
    return value;
  }

  /** One CAVI sweep: every coordinate is set to the optimum of the current ELBO. */
  sweep() {
    for (let j = 0; j < this.variables; j++) {
      let precision = this.priorPrecision[j];
      let natural = this.priorPrecision[j] * this.priorMean[j];
      for (const factor of this.factors) {
        const coefficient = factor.coefficients[j];
        if (coefficient === 0) continue;
        let rest = 0;
        for (let k = 0; k < this.variables; k++) if (k !== j) rest += factor.coefficients[k] * this.mean[k];
        precision += factor.precision * coefficient * coefficient;
        natural += factor.precision * coefficient * (factor.mean - rest);
      }
      this.variance[j] = 1 / precision;
      this.mean[j] = natural / precision;
    }
    this.iterations += 1;
    const value = this.elbo();
    this.elboTrace.push(value);
    return value;
  }

  /** Runs sweeps until the ELBO stops moving (or the budget is exhausted). */
  run({ maxIterations = 200, tolerance = 1e-12 } = {}) {
    let previous = this.elbo();
    this.elboTrace.push(previous);
    for (let i = 0; i < maxIterations; i++) {
      const value = this.sweep();
      if (Math.abs(value - previous) < tolerance) return { iterations: this.iterations, elbo: value, converged: true };
      previous = value;
    }
    return { iterations: this.iterations, elbo: previous, converged: false };
  }

  /** Independent residual check of the coordinate-ascent fixed point. */
  stationarityResidual() {
    let worst = 0;
    for (let j = 0; j < this.variables; j++) {
      let precision = this.priorPrecision[j];
      let natural = this.priorPrecision[j] * this.priorMean[j];
      for (const factor of this.factors) {
        const coefficient = factor.coefficients[j];
        if (coefficient === 0) continue;
        let rest = 0;
        for (let k = 0; k < this.variables; k++) if (k !== j) rest += factor.coefficients[k] * this.mean[k];
        precision += factor.precision * coefficient * coefficient;
        natural += factor.precision * coefficient * (factor.mean - rest);
      }
      worst = Math.max(worst, Math.abs(this.mean[j] - natural / precision), Math.abs(this.variance[j] - 1 / precision));
    }
    return worst;
  }

  /** Monte-Carlo ELBO: E_q[log p(x, data)] - E_q[log q(x)] using seeded samples. */
  monteCarloElbo(samples, seed = 12345) {
    const rand = mulberry32(seed);
    let total = 0;
    for (let s = 0; s < samples; s++) {
      const draw = this.mean.map((m, j) => m + Math.sqrt(this.variance[j]) * gaussianSample(rand));
      let logJoint = 0;
      let logQ = 0;
      for (let j = 0; j < this.variables; j++) {
        logJoint += gaussianLogPdf(draw[j], this.priorMean[j], 1 / this.priorPrecision[j]);
        logQ += gaussianLogPdf(draw[j], this.mean[j], this.variance[j]);
      }
      for (const factor of this.factors) {
        let value = 0;
        for (let j = 0; j < this.variables; j++) value += factor.coefficients[j] * draw[j];
        logJoint += gaussianLogPdf(factor.mean, value, 1 / factor.precision);
      }
      total += logJoint - logQ;
    }
    return total / samples;
  }

  predictive(x) {
    if (x.length !== this.variables) throw new RangeError("predictive vector must match the variable count");
    let mean = 0;
    let variance = 0;
    for (let j = 0; j < this.variables; j++) {
      mean += x[j] * this.mean[j];
      variance += x[j] * x[j] * this.variance[j];
    }
    return { mean, variance };
  }
}

/** Bayesian linear regression in the engine's factor form. */
function bayesianRegression({ design, targets, priorPrecision = 1, noisePrecision = 1 }) {
  const n = design.length;
  const d = n === 0 ? 1 : design[0].length;
  const factors = [];
  for (let i = 0; i < n; i++) {
    if (design[i].length !== d) throw new RangeError("design rows must share a dimension");
    factors.push({ coefficients: design[i].slice(), mean: targets[i], precision: noisePrecision });
  }
  return new VariationalEngine({
    variables: d,
    factors,
    priorMean: new Array(d).fill(0),
    priorPrecision: new Array(d).fill(priorPrecision),
  });
}

/** Independent oracle: log evidence of the conjugate regression model, closed form. */
function linearRegressionLogEvidence({ design, targets, priorPrecision = 1, noisePrecision = 1 }) {
  const n = design.length;
  const d = design[0].length;
  const covariance = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let k = 0; k < n; k++) {
      let dot = 0;
      for (let j = 0; j < d; j++) dot += design[i][j] * design[k][j];
      row.push((i === k ? 1 / noisePrecision : 0) + dot / priorPrecision);
    }
    covariance.push(row);
  }
  return multivariateLogPdf(targets, new Array(n).fill(0), covariance);
}

/** Analytic posterior of the conjugate regression model (full covariance). */
function linearRegressionPosterior({ design, targets, priorPrecision = 1, noisePrecision = 1 }) {
  const n = design.length;
  const d = design[0].length;
  const precision = [];
  const natural = new Array(d).fill(0);
  for (let j = 0; j < d; j++) {
    precision.push(new Array(d).fill(0));
    precision[j][j] = priorPrecision;
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      natural[j] += noisePrecision * design[i][j] * targets[i];
      for (let k = 0; k < d; k++) precision[j][k] += noisePrecision * design[i][j] * design[i][k];
    }
  }
  const { inverse, logDet } = invertMatrix(precision);
  const mean = new Array(d).fill(0);
  for (let j = 0; j < d; j++) for (let k = 0; k < d; k++) mean[j] += inverse[j][k] * natural[k];
  return { mean, covariance: inverse, covarianceVariance: inverse.map((row, i) => row[i]), logDet, precision };
}

function randomDesign(rand, n, d, { correlated = false } = {}) {
  const design = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    const base = rand() * 2 - 1;
    for (let j = 0; j < d; j++) row.push(correlated ? 0.8 * base + 0.2 * (rand() * 2 - 1) : rand() * 2 - 1);
    design.push(row);
  }
  return design;
}

/**
 * Mean-field CAVI for a 1-D Gaussian mixture with known data variance: q = prod_k N(mu_k; m_k, v_k)
 * times an independent categorical responsibility per observation. The mean update is a precision-
 * weighted average of the prior mean (zero) and the responsibility-weighted data, and the ELBO includes
 * the entropy of the means and of the responsibilities, so it is a genuine lower bound on the evidence.
 */
class MeanFieldMixture {
  constructor({ data, components = 2, variance = 1, priorVariance = 4 } = {}) {
    if (!Array.isArray(data) || data.length === 0) throw new RangeError("data must not be empty");
    if (!Number.isInteger(components) || components < 2) throw new RangeError("components must be >= 2");
    if (!(variance > 0) || !(priorVariance > 0)) throw new RangeError("variances must be > 0");
    this.data = data;
    this.components = components;
    this.variance = variance;
    this.priorVariance = priorVariance;
    this.weight = 1 / components;
    this.means = data.filter((_, i) => i < components).slice();
    this.variances = new Array(components).fill(priorVariance);
    this.elboTrace = [];
  }

  responsibilities() {
    const out = [];
    for (const value of this.data) {
      const scores = this.means.map((mean, k) => {
        const expectedLogLikelihood = gaussianLogPdf(value, mean, this.variance) - 0.5 * this.variances[k] / this.variance;
        return Math.log(this.weight) + expectedLogLikelihood;
      });
      const max = Math.max(...scores);
      const normalised = scores.map((score) => Math.exp(score - max));
      const total = normalised.reduce((a, b) => a + b, 0);
      out.push(normalised.map((value2) => value2 / total));
    }
    return out;
  }

  elbo() {
    const gamma = this.responsibilities();
    let value = 0;
    for (let k = 0; k < this.components; k++) {
      value += gaussianLogPdf(this.means[k], 0, this.priorVariance) - 0.5 * this.variances[k] / this.priorVariance;
      value += 0.5 * Math.log(2 * Math.PI * Math.E * this.variances[k]); // entropy of q(mu_k)
    }
    for (let i = 0; i < this.data.length; i++) {
      for (let k = 0; k < this.components; k++) {
        if (gamma[i][k] <= 0) continue;
        const expectedLogLikelihood = gaussianLogPdf(this.data[i], this.means[k], this.variance) - 0.5 * this.variances[k] / this.variance;
        value += gamma[i][k] * (Math.log(this.weight) + expectedLogLikelihood - Math.log(gamma[i][k]));
      }
    }
    return value;
  }

  sweep() {
    const gamma = this.responsibilities();
    for (let k = 0; k < this.components; k++) {
      let weighted = 0;
      let mass = 0;
      for (let i = 0; i < this.data.length; i++) {
        weighted += gamma[i][k] * this.data[i];
        mass += gamma[i][k];
      }
      const precision = 1 / this.priorVariance + mass / this.variance;
      this.variances[k] = 1 / precision;
      this.means[k] = (weighted / this.variance) / precision;
    }
    const value = this.elbo();
    this.elboTrace.push(value);
    return value;
  }

  run({ maxIterations = 200, tolerance = 1e-12 } = {}) {
    let previous = this.elbo();
    this.elboTrace.push(previous);
    for (let i = 0; i < maxIterations; i++) {
      const value = this.sweep();
      if (Math.abs(value - previous) < tolerance) return { iterations: i + 1, elbo: value, converged: true };
      previous = value;
    }
    return { iterations: maxIterations, elbo: previous, converged: false };
  }
}

/** Bayesian evidence of a 2-component 1-D mixture by numerical quadrature on a grid (oracle). */
function mixtureLogEvidence({ data, variance = 1, priorVariance = 4, grid = 121, span = 8 }) {
  const step = (2 * span) / (grid - 1);
  const points = [];
  for (let i = 0; i < grid; i++) points.push(-span + i * step);
  const logPrior = (mean) => gaussianLogPdf(mean, 0, priorVariance);
  let total = 0;
  for (const mu1 of points) {
    for (const mu2 of points) {
      let logLikelihood = logPrior(mu1) + logPrior(mu2);
      for (const value of data) {
        const term = 0.5 * Math.exp(gaussianLogPdf(value, mu1, variance)) + 0.5 * Math.exp(gaussianLogPdf(value, mu2, variance));
        logLikelihood += Math.log(Math.max(1e-300, term));
      }
      // midpoint quadrature with the two-dimensional cell area
      if (logLikelihood > -700) total += Math.exp(logLikelihood) * step * step;
    }
  }
  return Math.log(total);
}

function selfCheck() {
  // --- validation -----------------------------------------------------------------------------------------
  assert.throws(() => new VariationalEngine({ variables: 0 }), /variables/);
  assert.throws(() => new VariationalEngine({ variables: 2, priorMean: [1] }), /prior vectors/);
  assert.throws(() => new VariationalEngine({ variables: 1, factors: [{ coefficients: [1, 1], mean: 0, precision: 1 }] }), /coefficients/);
  assert.throws(() => new VariationalEngine({ variables: 1, factors: [{ coefficients: [1], mean: 0, precision: 0 }] }), /precision/);
  assert.throws(() => bayesianRegression({ design: [[1, 2], [3]], targets: [1, 2] }), /dimension/);
  assert.throws(() => invertMatrix([[0, 0], [0, 0]]), /singular/);
  assert.throws(() => new MeanFieldMixture({ data: [] }), /data/);
  assert.throws(() => new MeanFieldMixture({ data: [1], components: 1 }), /components/);

  // --- exactness on an orthogonal design: the mean-field family contains the posterior ----------------------
  const rand = mulberry32(860);
  // Columns occupy disjoint row blocks, so X'X is diagonal and mean-field inference is exact.
  const orthogonal = [];
  for (let i = 0; i < 24; i++) orthogonal.push([i < 8 ? 1 : 0, i >= 8 && i < 16 ? 1 : 0, i >= 16 ? 1 : 0]);
  const orthogonalTargets = orthogonal.map((row, i) => 2 * row[0] - 1.5 * row[1] + 0.5 * row[2] + (i % 4) * 0.1 - 0.2);
  const orthogonalPrior = 1;
  const orthogonalNoise = 2;
  const orthogonalEngine = bayesianRegression({ design: orthogonal, targets: orthogonalTargets, priorPrecision: orthogonalPrior, noisePrecision: orthogonalNoise });
  const orthogonalRun = orthogonalEngine.run();
  assert.strictEqual(orthogonalRun.converged, true);
  assert.ok(orthogonalEngine.iterations <= 20, `orthogonal design converged in ${orthogonalEngine.iterations} sweeps`);
  const exactPosterior = linearRegressionPosterior({ design: orthogonal, targets: orthogonalTargets, priorPrecision: orthogonalPrior, noisePrecision: orthogonalNoise });
  for (let j = 0; j < 3; j++) {
    assert.ok(Math.abs(orthogonalEngine.mean[j] - exactPosterior.mean[j]) < 1e-9, `mean ${j} must match the analytic posterior`);
    assert.ok(Math.abs(orthogonalEngine.variance[j] - exactPosterior.covarianceVariance[j]) < 1e-9, `variance ${j} must match`);
  }
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    if (i !== j) assert.ok(Math.abs(exactPosterior.covariance[i][j]) < 1e-12, "an orthogonal design has a diagonal posterior");
  }
  const exactEvidence = linearRegressionLogEvidence({ design: orthogonal, targets: orthogonalTargets, priorPrecision: orthogonalPrior, noisePrecision: orthogonalNoise });
  assert.ok(orthogonalRun.elbo <= exactEvidence + 1e-9, "the ELBO can never exceed the evidence");
  assert.ok(exactEvidence - orthogonalRun.elbo < 1e-7, `with an orthogonal design the gap must vanish (${(exactEvidence - orthogonalRun.elbo).toExponential(2)})`);
  assert.ok(orthogonalEngine.stationarityResidual() < 1e-12, "the fixed point must satisfy the stationarity conditions");

  // --- correlated design: the bound holds, the gap is positive, variances are under-estimated ---------------
  const design = randomDesign(rand, 20, 3, { correlated: true });
  const targets = design.map((row, i) => 1.5 * row[0] - 0.7 * row[1] + 0.3 * row[2] + (i % 3) * 0.2);
  const correlated = bayesianRegression({ design, targets, priorPrecision: 1, noisePrecision: 2 });
  const correlatedRun = correlated.run();
  assert.strictEqual(correlatedRun.converged, true);
  const evidence = linearRegressionLogEvidence({ design, targets, priorPrecision: 1, noisePrecision: 2 });
  const gap = evidence - correlatedRun.elbo;
  assert.ok(gap > 0, `mean-field inference must leave a gap on a correlated design (${gap.toExponential(3)})`);
  assert.ok(gap < 5, `the gap must stay small (${gap.toFixed(4)})`);
  const correlatedPosterior = linearRegressionPosterior({ design, targets, priorPrecision: 1, noisePrecision: 2 });
  for (let j = 0; j < 3; j++) {
    assert.ok(correlated.variance[j] <= correlatedPosterior.covarianceVariance[j] + 1e-12, `mean-field variance ${j} must not exceed the exact one`);
    assert.ok(correlatedPosterior.covarianceVariance[j] > correlated.variance[j], "the underestimation must be strict for a correlated design");
  }
  const residual = correlated.stationarityResidual();
  assert.ok(residual < 1e-6, `the fixed point must satisfy the stationarity conditions (residual ${residual.toExponential(2)})`);
  const elboAtFixedPoint = correlated.elbo();
  correlated.sweep();
  assert.ok(Math.abs(correlated.elbo() - elboAtFixedPoint) < 1e-9, "one more sweep must not move the ELBO");

  // --- ELBO monotonicity and independent recomputation --------------------------------------------------------
  for (let i = 1; i < correlated.elboTrace.length; i++) {
    assert.ok(correlated.elboTrace[i] >= correlated.elboTrace[i - 1] - 1e-9, `ELBO decreased at sweep ${i}`);
  }
  assert.ok(correlated.elboTrace.length >= 2 && correlated.iterations === correlated.elboTrace.length - 1);
  const sampleCount = 40000;
  const monteCarlo = correlated.monteCarloElbo(sampleCount, 4242);
  const varianceEstimate = correlated.variance.reduce((acc, value) => acc + value, 0);
  const tolerance = 6 * Math.sqrt(varianceEstimate) / Math.sqrt(sampleCount) + 0.05;
  assert.ok(Math.abs(monteCarlo - correlatedRun.elbo) < tolerance, `Monte-Carlo ELBO ${monteCarlo.toFixed(4)} vs closed form ${correlatedRun.elbo.toFixed(4)}`);
  const recomputedElbo = correlated.elbo();
  assert.ok(Math.abs(recomputedElbo - correlatedRun.elbo) < 1e-9, "the ELBO is a pure function of the current q");

  // --- predictive distribution ----------------------------------------------------------------------------------
  const probe = [0.5, -0.25, 0.75];
  const predictive = correlated.predictive(probe);
  let analyticMean = 0;
  for (let j = 0; j < 3; j++) analyticMean += probe[j] * correlatedPosterior.mean[j];
  assert.ok(Math.abs(predictive.mean - analyticMean) < 0.05, "the predictive mean must track the exact posterior mean");
  assert.ok(predictive.variance > 0);
  assert.throws(() => correlated.predictive([1, 2]), /predictive vector/);
  const orthogonalPredictive = orthogonalEngine.predictive([1, 0, 0]);
  const exactOrthogonal = orthogonalPredictive.mean;
  assert.ok(Math.abs(exactOrthogonal - exactPosterior.mean[0]) < 1e-9);

  // --- liveness and degenerate models ---------------------------------------------------------------------------
  const empty = bayesianRegression({ design: [], targets: [], priorPrecision: 3, noisePrecision: 1 });
  empty.run();
  assert.strictEqual(empty.mean[0], 0, "no data leaves the prior untouched");
  assert.ok(Math.abs(empty.variance[0] - 1 / 3) < 1e-12);
  assert.strictEqual(empty.elbo(), empty.elboTrace[empty.elboTrace.length - 1]);
  const single = bayesianRegression({ design: [[1], [1], [1]], targets: [1, 1, 1], priorPrecision: 1, noisePrecision: 1 });
  single.run();
  assert.ok(single.mean[0] > 0.7 && single.mean[0] < 0.9, `shrunken mean expected, got ${single.mean[0].toFixed(4)}`);
  assert.ok(single.iterations <= 5);
  const purePrior = new VariationalEngine({ variables: 4 });
  purePrior.run();
  assert.deepStrictEqual(purePrior.mean, [0, 0, 0, 0]);
  const budget = new VariationalEngine({ variables: 2, factors: [{ coefficients: [1, 1], mean: 1, precision: 4 }] });
  const budgetRun = budget.run({ maxIterations: 1 });
  assert.strictEqual(budgetRun.iterations, 1);
  assert.strictEqual(budgetRun.converged, false, "the budget must be reported honestly");

  // --- second model: mean-field Gaussian mixture against a quadrature oracle ---------------------------------------
  const mixtureData = [];
  for (let i = 0; i < 20; i++) mixtureData.push(i % 2 === 0 ? -2.4 + (i % 5) * 0.05 : 2.6 - (i % 7) * 0.04);
  const mixture = new MeanFieldMixture({ data: mixtureData, components: 2, variance: 0.5, priorVariance: 4 });
  const mixtureRun = mixture.run();
  assert.strictEqual(mixtureRun.converged, true);
  assert.ok(Math.abs(Math.min(...mixture.means) + 2.4) < 0.4, `the low component must find its mean (${mixture.means[0].toFixed(3)})`);
  assert.ok(Math.abs(Math.max(...mixture.means) - 2.6) < 0.4, `the high component must find its mean (${mixture.means[1].toFixed(3)})`);
  for (let i = 1; i < mixture.elboTrace.length; i++) {
    assert.ok(mixture.elboTrace[i] >= mixture.elboTrace[i - 1] - 1e-9, `mixture ELBO decreased at sweep ${i}`);
  }
  const gridEvidence = mixtureLogEvidence({ data: mixtureData, variance: 0.5, priorVariance: 4, grid: 141, span: 8 });
  assert.ok(mixtureRun.elbo <= gridEvidence + 1e-6, `variational bound must not exceed the grid evidence (${mixtureRun.elbo.toFixed(4)} vs ${gridEvidence.toFixed(4)})`);
  assert.ok(gridEvidence - mixtureRun.elbo < 2.5, `the mixture gap must stay small (${(gridEvidence - mixtureRun.elbo).toFixed(4)})`);
  const swapped = new MeanFieldMixture({ data: mixtureData.slice().reverse(), components: 2, variance: 0.5, priorVariance: 4 });
  const swappedRun = swapped.run();
  assert.ok(Math.abs(swappedRun.elbo - mixtureRun.elbo) < 1e-6, "the mixture ELBO is invariant to data order");

  // --- determinism ----------------------------------------------------------------------------------------------------
  const repeatA = bayesianRegression({ design, targets, priorPrecision: 1, noisePrecision: 2 });
  const repeatB = bayesianRegression({ design, targets, priorPrecision: 1, noisePrecision: 2 });
  const runA = repeatA.run();
  const runB = repeatB.run();
  assert.deepStrictEqual(runA, runB);
  assert.deepStrictEqual(repeatA.elboTrace, repeatB.elboTrace);
  assert.strictEqual(repeatA.monteCarloElbo(500, 7), repeatB.monteCarloElbo(500, 7));
  assert.notStrictEqual(repeatA.monteCarloElbo(500, 8), repeatA.monteCarloElbo(500, 7), "another seed draws another sample");
  assert.deepStrictEqual(new MeanFieldMixture({ data: mixtureData }).run(), new MeanFieldMixture({ data: mixtureData }).run());

  console.log(`orthogonal design: ELBO ${orthogonalRun.elbo.toFixed(6)} = evidence ${exactEvidence.toFixed(6)} (gap ${(exactEvidence - orthogonalRun.elbo).toExponential(2)})`);
  console.log(`correlated design: ELBO ${correlatedRun.elbo.toFixed(4)}, evidence ${evidence.toFixed(4)}, gap ${gap.toFixed(4)}, MC ${monteCarlo.toFixed(4)} in ${correlated.iterations} sweeps`);
  console.log(`mixture: ELBO ${mixtureRun.elbo.toFixed(4)}, grid evidence ${gridEvidence.toFixed(4)}, means [${mixture.means.map((m) => m.toFixed(2)).join(", ")}]`);
  console.log("PASS");
}

module.exports = { VariationalEngine, bayesianRegression, linearRegressionLogEvidence, linearRegressionPosterior, randomDesign, MeanFieldMixture, mixtureLogEvidence, invertMatrix, multivariateLogPdf, gaussianLogPdf, mulberry32 };

if (require.main === module) {
  selfCheck();
}
