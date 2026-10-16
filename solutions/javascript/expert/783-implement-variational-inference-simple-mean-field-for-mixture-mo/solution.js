/** Task 783 - Implement variational inference simple mean-field for mixture models. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * A Bayesian Gaussian mixture with *known* component parameters (means and standard deviations), an
 * unknown mixing vector π with a symmetric Dirichlet(alpha) prior, and a latent cluster assignment
 * z_n per observation. The exact posterior p(z, π | x) is intractable in general, so mean-field
 * coordinate ascent (CAVI) optimises the factorised family
 *
 *     q(z, pi) = Dirichlet(pi; lambda) * prod_n Categorical(z_n; phi_n)
 *
 * with the closed-form updates
 *
 *     phi_nk ~ exp(psi(lambda_k) - psi(sum lambda)) * N(x_n; mu_k, sigma_k),
 *     lambda_k = alpha + sum_n phi_nk,
 *
 * and the ELBO computed with digamma/trigamma-free identities. Coordinate ascent guarantees the
 * ELBO never decreases - that monotonicity is asserted after *every* update, not just at the end.
 *
 * The exact oracle here is brute force: for small data every one of the K^N assignments is weighted
 * by the collapsed Dirichlet-multinomial (topics) times the product of the Gaussian likelihoods, so
 * log p(x) and the exact posterior over assignments are both available in closed form. The ELBO must
 * sit below log p(x) (the fundamental variational bound) and climb towards it, and the mean-field
 * responsibilities must approach the exact marginals even though the factorisation makes them
 * inexact for a finite alpha.
 *
 * What is verified: log-gamma against factorials and digamma against a numerical derivative of
 * log-gamma; the ELBO is monotone after every single coordinate update; the ELBO never exceeds the
 * exact log evidence over several data sets, and the final gap is smaller than the gap after the
 * first sweep; the converged responsibilities match the exact posterior marginals within a measured
 * tolerance; the K = 1 case collapses to the exact evidence; degenerate priors and data (extreme
 * alpha, one observation, identical observations) behave; and everything is deterministic.
 */
"use strict";

/** Lanczos log-gamma. */
function logGamma(x) {
  if (x <= 0) throw new RangeError("logGamma needs a positive argument");
  const coefficients = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  let z = x;
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  z -= 1;
  let series = 0.99999999999980993;
  for (let i = 0; i < coefficients.length; i++) series += coefficients[i] / (z + i + 1);
  const t = z + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(series);
}

/** Digamma by recurrence plus the classic asymptotic series. */
function digamma(x) {
  if (x <= 0) throw new RangeError("digamma needs a positive argument");
  let result = 0;
  let value = x;
  while (value < 6) { result -= 1 / value; value += 1; }
  const inverse = 1 / value;
  const square = inverse * inverse;
  return result + Math.log(value) - 0.5 * inverse
    - square * (1 / 12 - square * (1 / 120 - square * (1 / 252 - square / 240)));
}

/** log N(x; mu, sigma) - the Gaussian component likelihood. */
function logGaussian(x, mu, sigma) {
  if (!(sigma > 0)) throw new RangeError("sigma must be positive");
  const delta = x - mu;
  return -0.5 * Math.log(2 * Math.PI) - Math.log(sigma) - (delta * delta) / (2 * sigma * sigma);
}

/**
 * log of the Dirichlet count integral  integral prod_k pi_k^{n_k} Dir(pi; alpha) dpi .  The
 * multinomial coefficient N!/prod n_k! would turn it into the probability of the counts, but that
 * factor is constant across assignments and therefore irrelevant to the collapsed likelihood; the
 * self-check adds it back to test the normalisation.
 */
function logDirichletMultinomial(counts, alpha) {
  let total = counts.reduce((sum, count) => sum + count, 0);
  let out = logGamma(counts.length * alpha) - logGamma(total + counts.length * alpha) - counts.length * logGamma(alpha);
  for (const count of counts) out += logGamma(count + alpha);
  return out;
}

/** Exact log evidence and exact assignment marginals by enumerating every z in {0..K-1}^N. */
function exactPosterior(data, { means, sigmas, alpha, topics }) {
  const n = data.length;
  const assignments = [];
  const logWeights = [];
  for (let code = 0; code < topics ** n; code++) {
    const assignment = [];
    let rest = code;
    for (let i = 0; i < n; i++) { assignment.push(rest % topics); rest = Math.floor(rest / topics); }
    const counts = new Array(topics).fill(0);
    for (const k of assignment) counts[k]++;
    let logWeight = logDirichletMultinomial(counts, alpha);
    for (let i = 0; i < n; i++) logWeight += logGaussian(data[i], means[assignment[i]], sigmas[assignment[i]]);
    assignments.push(assignment);
    logWeights.push(logWeight);
  }
  const max = Math.max(...logWeights);
  const scaled = logWeights.map((value) => Math.exp(value - max));
  const total = scaled.reduce((sum, value) => sum + value, 0);
  const probabilities = scaled.map((value) => value / total);
  const logEvidence = max + Math.log(total);
  const marginals = Array.from({ length: n }, () => new Array(topics).fill(0));
  for (let index = 0; index < assignments.length; index++) {
    for (let i = 0; i < n; i++) marginals[i][assignments[index][i]] += probabilities[index];
  }
  return { logEvidence, marginals, assignments, probabilities };
}

/** Mean-field CAVI for the mixture; `onUpdate` sees the ELBO after every coordinate update. */
function meanField(data, { means, sigmas, alpha, topics, iterations = 200, tolerance = 1e-12, onUpdate = null }) {
  const n = data.length;
  const logLikelihoods = data.map((x) => means.map((mu, k) => logGaussian(x, mu, sigmas[k])));
  const phi = Array.from({ length: n }, () => new Array(topics).fill(1 / topics)); // symmetric init
  const lambda = new Array(topics).fill(alpha + n / topics);
  const elboHistory = [];

  const elbo = () => {
    const sumLambda = lambda.reduce((sum, value) => sum + value, 0);
    const digammaSum = digamma(sumLambda);
    let value = logGamma(topics * alpha) - topics * logGamma(alpha);
    for (let k = 0; k < topics; k++) value += (alpha - 1) * (digamma(lambda[k]) - digammaSum);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < topics; k++) {
        const probability = Math.max(phi[i][k], 1e-300);
        value += phi[i][k] * (logLikelihoods[i][k] + digamma(lambda[k]) - digammaSum - Math.log(probability));
      }
    }
    // q(pi) entropy
    let logBeta = 0;
    for (let k = 0; k < topics; k++) logBeta += logGamma(lambda[k]);
    logBeta -= logGamma(sumLambda);
    let entropy = logBeta + (sumLambda - topics) * digammaSum;
    for (let k = 0; k < topics; k++) entropy -= (lambda[k] - 1) * digamma(lambda[k]);
    return value + entropy;
  };

  let previous = elbo();
  elboHistory.push(previous);
  let converged = false;
  let sweeps = 0;
  for (let iteration = 0; iteration < iterations; iteration++) {
    sweeps = iteration + 1;
    // phi updates
    const sumLambda = lambda.reduce((sum, value) => sum + value, 0);
    const digammaSum = digamma(sumLambda);
    for (let i = 0; i < n; i++) {
      const logits = new Array(topics);
      let maxLogit = -Infinity;
      for (let k = 0; k < topics; k++) {
        logits[k] = logLikelihoods[i][k] + digamma(lambda[k]) - digammaSum;
        maxLogit = Math.max(maxLogit, logits[k]);
      }
      let total = 0;
      for (let k = 0; k < topics; k++) { logits[k] = Math.exp(logits[k] - maxLogit); total += logits[k]; }
      for (let k = 0; k < topics; k++) phi[i][k] = logits[k] / total;
      const now = elbo();
      if (now < previous - 1e-9) throw new Error(`ELBO decreased to ${now} from ${previous}`);
      previous = now;
      elboHistory.push(now);
      if (onUpdate !== null) onUpdate(now, "phi");
    }
    // lambda update
    const counts = new Array(topics).fill(0);
    for (let i = 0; i < n; i++) for (let k = 0; k < topics; k++) counts[k] += phi[i][k];
    for (let k = 0; k < topics; k++) lambda[k] = alpha + counts[k];
    const now = elbo();
    if (now < previous - 1e-9) throw new Error(`ELBO decreased to ${now} from ${previous}`);
    const change = now - previous;
    previous = now;
    elboHistory.push(now);
    if (onUpdate !== null) onUpdate(now, "lambda");
    if (change < tolerance) { converged = true; break; }
  }
  return { phi, lambda, elbo: previous, elboHistory, sweeps, converged };
}

/** Assignments of the mean-field responsibilities and the resulting collapsed log likelihood. */
function collapsedLogLikelihood(data, assignment, { means, sigmas, alpha, topics }) {
  const counts = new Array(topics).fill(0);
  for (const k of assignment) counts[k]++;
  let value = logDirichletMultinomial(counts, alpha);
  for (let i = 0; i < data.length; i++) value += logGaussian(data[i], means[assignment[i]], sigmas[assignment[i]]);
  return value;
}

function makeData(rng, count, gaussianNoise = 1) {
  const out = [];
  for (let i = 0; i < count; i++) {
    const centre = rng(2) === 0 ? -3 : 3;
    const u = Math.max(rng(1e9) / 1e9, 1e-12);
    const v = rng(1e9) / 1e9;
    out.push(centre + gaussianNoise * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v));
  }
  return out;
}

function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return (bound) => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

module.exports = {
  logGamma, digamma, logGaussian, logDirichletMultinomial, exactPosterior, meanField,
  collapsedLogLikelihood, makeData, makeRng,
};

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, deepStrictEqual, ok, throws } = assert;
  const rng = makeRng(20260783);

  // -- special functions -------------------------------------------------------------
  ok(Math.abs(logGamma(1)) < 1e-9, "logGamma(1) = 0");
  for (const n of [1, 2, 3, 4, 5, 6, 7, 8]) {
    let factorial = 1;
    for (let i = 1; i < n; i++) factorial *= i;
    ok(Math.abs(logGamma(n) - Math.log(factorial)) < 1e-9, `logGamma(${n}) must equal log(${factorial})`);
  }
  ok(Math.abs(logGamma(0.5) - Math.log(Math.sqrt(Math.PI))) < 1e-9, "logGamma(1/2) = log(sqrt(pi))");
  for (const x of [0.7, 1.3, 3.1, 7.5, 20.2]) {
    const h = 1e-5;
    const derivative = (logGamma(x + h) - logGamma(x - h)) / (2 * h);
    ok(Math.abs(digamma(x) - derivative) < 1e-6,
      `digamma(${x}) must match the numerical derivative (${digamma(x)} vs ${derivative})`);
  }
  throws(() => logGamma(0), RangeError, "logGamma rejects zero");
  throws(() => digamma(-1), RangeError, "digamma rejects negatives");
  throws(() => logGaussian(0, 0, 0), RangeError, "the standard deviation must be positive");
  /** N!/prod n_k! - the missing factor that turns the count integral into a probability. */
  const multinomial = (counts) => {
    let factor = 1;
    const total = counts.reduce((sum, count) => sum + count, 0);
    for (let i = 2; i <= total; i++) factor *= i;
    for (const count of counts) for (let i = 2; i <= count; i++) factor /= i;
    return factor;
  };
  let countMass = 0;
  for (let a = 0; a <= 6; a++) countMass += multinomial([a, 6 - a]) * Math.exp(logDirichletMultinomial([a, 6 - a], 1.5));
  ok(Math.abs(countMass - 1) < 1e-9,
    "the Dirichlet-multinomial over the count vectors of size 6 must sum to one");
  let threeMass = 0;
  for (let a = 0; a <= 5; a++) {
    for (let b = 0; b + a <= 5; b++) {
      const counts = [a, b, 5 - a - b];
      threeMass += multinomial(counts) * Math.exp(logDirichletMultinomial(counts, 0.7));
    }
  }
  ok(Math.abs(threeMass - 1) < 1e-9, "the same identity must hold for three bins and a sharp prior");
  const dmProbability = (counts, alpha) => multinomial(counts) * Math.exp(logDirichletMultinomial(counts, alpha));
  ok(Math.abs(dmProbability([10, 0], 1) - dmProbability([5, 5], 1)) < 1e-12,
    "with alpha = 1 the count distribution is uniform, so both compositions carry the same mass");
  ok(Math.abs(dmProbability([10, 0], 1) - 1 / 11) < 1e-12, "that mass is 1/11 for N = 10 and K = 2");
  ok(dmProbability([10, 0], 0.1) > dmProbability([5, 5], 0.1),
    "a sharp prior concentrates the mass on sparse count vectors, not on balanced ones");
  ok(dmProbability([10, 0], 5) < dmProbability([5, 5], 5),
    "a smooth prior concentrates the mass on balanced count vectors");
  ok(dmProbability([10, 0], 1) < dmProbability([10, 0], 0.1),
    "shrinking alpha must raise the mass of the extreme composition");

  // -- the ELBO is monotone after every single coordinate update ---------------------
  const model = { means: [-3, 3], sigmas: [1, 1], alpha: 1, topics: 2 };
  const data = makeData(rng, 8);
  const updates = [];
  const fit = meanField(data, { ...model, iterations: 60, onUpdate: (value, kind) => updates.push({ value, kind }) });
  ok(updates.length >= 10, "the check must observe a real number of coordinate updates");
  strictEqual(updates.length, fit.sweeps * (data.length + 1),
    "every sweep must consist of one update per observation plus one for the mixing weights");
  ok(fit.converged, "this data set must converge within the sweep budget");
  ok(updates.filter((u) => u.kind === "lambda").length === fit.sweeps, "one mixing update per sweep");
  for (let i = 1; i < updates.length; i++) {
    ok(updates[i].value >= updates[i - 1].value - 1e-9,
      `${updates[i].kind} update ${i} must not lower the ELBO`);
  }
  ok(fit.elbo >= fit.elboHistory[0], "the final ELBO must be at least the initial one");
  ok(updates.some((u) => u.kind === "lambda") && updates.some((u) => u.kind === "phi"),
    "both coordinate blocks must have been updated");

  // -- the variational bound: ELBO <= exact log evidence -----------------------------
  let worstGap = 0;
  let closest = Infinity;
  let comparisons = 0;
  for (const count of [4, 6, 8, 9]) {
    const sample = makeData(rng, count);
    for (const alpha of [0.2, 1, 3]) {
      const settings = { ...model, alpha };
      const exact = exactPosterior(sample, settings);
      const variational = meanField(sample, { ...settings, iterations: 400 });
      ok(variational.elbo <= exact.logEvidence + 1e-6,
        `the ELBO must not exceed log p(x) (${variational.elbo} vs ${exact.logEvidence})`);
      const gap = exact.logEvidence - variational.elbo;
      ok(gap >= -1e-6, "the variational gap must be non-negative");
      const initialGap = exact.logEvidence - variational.elboHistory[0];
      ok(gap <= initialGap + 1e-9, "optimising must not widen the gap it started with");
      worstGap = Math.max(worstGap, gap);
      closest = Math.min(closest, gap);
      comparisons++;
    }
  }
  ok(comparisons === 12, "twelve bound comparisons were performed");
  ok(worstGap < 3, `the mean-field gap must stay modest (worst ${worstGap.toFixed(4)})`);

  // -- the ELBO climbs towards the evidence and the responsibilities approach the truth
  const sample = makeData(rng, 7);
  const exact = exactPosterior(sample, model);
  const variational = meanField(sample, { ...model, iterations: 400 });
  const firstSweepGap = exact.logEvidence - variational.elboHistory[1];
  const finalGap = exact.logEvidence - variational.elbo;
  ok(finalGap < firstSweepGap, `the gap must shrink (${finalGap.toFixed(4)} < ${firstSweepGap.toFixed(4)})`);
  let worstMarginal = 0;
  for (let i = 0; i < sample.length; i++) {
    let totalVariation = 0;
    for (let k = 0; k < model.topics; k++) {
      totalVariation += Math.abs(variational.phi[i][k] - exact.marginals[i][k]);
    }
    worstMarginal = Math.max(worstMarginal, totalVariation / 2);
  }
  ok(worstMarginal < 0.05,
    `the mean-field responsibilities must approach the exact marginals (worst TV ${worstMarginal.toFixed(4)})`);
  ok(worstMarginal > 1e-12, "the responsibilities cannot be exactly the exact posterior (the gap is real)");
  const lambdaSum = variational.lambda.reduce((sum, value) => sum + value, 0);
  ok(Math.abs(lambdaSum - (model.alpha * model.topics + sample.length)) < 1e-9,
    "the Dirichlet concentration must account for every observation");

  // -- K = 1 collapses to the exact evidence ----------------------------------------
  const single = { means: [0], sigmas: [1], alpha: 1, topics: 1 };
  const singleData = [0.4, -1.2, 2.2];
  const singleFit = meanField(singleData, { ...single, iterations: 5 });
  const singleExact = exactPosterior(singleData, single);
  ok(Math.abs(singleFit.elbo - singleExact.logEvidence) < 1e-9,
    "with a single topic the mean-field ELBO is the exact evidence");
  deepStrictEqual(singleFit.phi.map((row) => row[0]), [1, 1, 1], "one topic means certain responsibilities");

  // -- degenerate data ---------------------------------------------------------------
  const oneObservation = meanField([0.1], { ...model, iterations: 50 });
  ok(oneObservation.phi[0].reduce((sum, value) => sum + value, 0) - 1 < 1e-12, "a single point has normalised responsibilities");
  ok(meanField([], { ...model, iterations: 5 }).elbo <= exactPosterior([], model).logEvidence + 1e-9,
    "an empty data set still satisfies the bound");
  const identical = [2, 2, 2, 2];
  const identicalFit = meanField(identical, { ...model, iterations: 100 });
  ok(identicalFit.phi.every((row) => row[1] > 0.99), "points near the second component must be assigned to it");
  const sharp = meanField(sample, { ...model, alpha: 0.01, iterations: 200 });
  ok(sharp.lambda.every((value) => value >= 0.01 - 1e-12), "a sharp prior keeps the concentrations positive");
  ok(Number.isFinite(sharp.elbo), "the sharp prior still yields a finite ELBO");

  // -- the collapsed likelihood of the variational assignment beats a random one -----
  const chosen = variational.phi.map((row) => (row[0] >= row[1] ? 0 : 1));
  const chosenScore = collapsedLogLikelihood(sample, chosen, model);
  const bestExact = Math.max(...exact.assignments.map((assignment) => collapsedLogLikelihood(sample, assignment, model)));
  ok(Number.isFinite(chosenScore), "the chosen assignment must be scoreable");
  ok(chosenScore >= bestExact - 1e-9,
    `CAVI must find an assignment as good as the best enumeration (${chosenScore} vs ${bestExact})`);

  // -- determinism -------------------------------------------------------------------
  const replay = () => meanField(sample, { ...model, iterations: 100 });
  strictEqual(JSON.stringify(replay()), JSON.stringify(replay()), "CAVI must be deterministic");

  console.log(`ELBO monotone over ${updates.length} updates | gap range ${closest.toFixed(5)}..${worstGap.toFixed(5)} | worst responsibility TV ${worstMarginal.toFixed(5)} | K=1 gap ${Math.abs(singleFit.elbo - singleExact.logEvidence).toExponential(2)}`);
  console.log("PASS");
}
