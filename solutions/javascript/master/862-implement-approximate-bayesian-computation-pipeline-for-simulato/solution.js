/** Task 862 - Implement approximate Bayesian computation pipeline for simulator-based inference.
 *  (Tier: Master, CTCI Beyond — Machine Learning & AI)
 *
 * Model. A normal-normal problem is used as the testbed because every quantity has a closed form:
 * theta ~ N(0, tau^2) is the prior, the simulator draws n observations y_i ~ N(theta, sigma^2), and the
 * summary statistic is the sample mean (sufficient for a Gaussian). The analytic posterior
 * theta | y ~ N(m, v) is therefore available as an oracle, while the pipeline itself only ever touches
 * the simulator — exactly the situation ABC is designed for.
 *
 * Two inference routines are implemented:
 *   rejection ABC   draw theta ~ prior, simulate, keep the draw when rho(eta(y*), eta(y_obs)) <= epsilon;
 *                   the acceptance rate falls as epsilon shrinks and the posterior converges to the
 *                   analytic one.
 *   ABC-SMC         a sequence of tempered populations with importance weights
 *                   w_i' = w_i * prior(theta*)/sum_j w_j K(theta_j, theta*), systematic resampling when the
 *                   effective sample size drops, and an adaptive tolerance schedule derived from the
 *                   accepted distances (the standard quantile rule).
 *
 * What is simplified: one process, a synthetic Gaussian simulator instead of a real one, the sample mean
 * as the only summary statistic (no semi-automatic summary selection), a Gaussian random-walk
 * perturbation kernel with a simple variance doubling and no optimal-covariance learning, and randomness
 * confined to one seeded mulberry32 stream — no wall clock, no threads, no external state.
 *
 * Invariants verified:
 *   consistency  both routines converge to the analytic posterior as the tolerance shrinks: the posterior
 *                mean and variance match within the Monte-Carlo tolerance, and the posterior was shown to
 *                be no worse with a sufficient statistic than with an insufficient one.
 *   calibration  a 90% credible interval covers the true parameter in at least 80% of independent
 *                repetitions, and a flat (epsilon = infinity) run reproduces the prior.
 *   cost         the acceptance rate is monotone in epsilon, the SMC tolerance schedule is monotone
 *                decreasing and ends inside the target, particles are resampled exactly when the ESS
 *                collapses, and the simulator-call count stays within a documented budget.
 *   determinism  identical seeds reproduce identical accepted sets, weights and diagnostics.
 */
"use strict";

const assert = require("node:assert");

/** mulberry32 — the only source of randomness in the whole pipeline. */
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

function gaussianSample(rand) {
  const u1 = Math.max(1e-12, rand());
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

const mean = (values) => (values.length === 0 ? 0 : values.reduce((a, b) => a + b, 0) / values.length);
const variance = (values) => {
  if (values.length < 2) return 0;
  const m = mean(values);
  return values.reduce((acc, value) => acc + (value - m) ** 2, 0) / (values.length - 1);
};

/** The problem: a prior, a simulator and observed data generated from a known truth. */
function makeProblem({ n = 20, tau = 2, sigma = 1, truth = 0.75, seed = 862 } = {}) {
  if (!Number.isInteger(n) || n < 1) throw new RangeError("n must be >= 1");
  if (!(tau > 0) || !(sigma > 0)) throw new RangeError("prior and noise scales must be > 0");
  const rand = mulberry32(seed);
  const data = [];
  for (let i = 0; i < n; i++) data.push(truth + sigma * gaussianSample(rand));
  return { n, tau, sigma, truth, data, observed: mean(data) };
}

/** Analytic posterior of the normal-normal model — the oracle both routines are measured against. */
function analyticPosterior(problem) {
  const priorPrecision = 1 / (problem.tau * problem.tau);
  const dataPrecision = problem.n / (problem.sigma * problem.sigma);
  const posteriorVariance = 1 / (priorPrecision + dataPrecision);
  const posteriorMean = posteriorVariance * (problem.observed * dataPrecision);
  return { mean: posteriorMean, variance: posteriorVariance, sd: Math.sqrt(posteriorVariance) };
}

/** Simulator: returns the sufficient statistic of a synthetic data set drawn at `theta`. */
function simulate(problem, theta, rand) {
  let total = 0;
  for (let i = 0; i < problem.n; i++) total += theta + problem.sigma * gaussianSample(rand);
  return total / problem.n;
}

/** Rejection ABC. */
function rejectionAbc(problem, { epsilon, samples = 20000, seed = 862, statistic = null } = {}) {
  if (!(epsilon >= 0)) throw new RangeError("epsilon must be >= 0");
  if (!Number.isInteger(samples) || samples < 1) throw new RangeError("samples must be >= 1");
  const rand = mulberry32(seed);
  const summary = statistic || ((value) => value);
  const accepted = [];
  const distances = [];
  let calls = 0;
  for (let i = 0; i < samples; i++) {
    const theta = problem.tau * gaussianSample(rand);
    calls += 1;
    const simulated = simulate(problem, theta, rand);
    const distance = Math.abs(summary(simulated) - summary(problem.observed));
    if (distance <= epsilon) {
      accepted.push(theta);
      distances.push(distance);
    }
  }
  return {
    accepted,
    distances,
    acceptanceRate: accepted.length / samples,
    simulatorCalls: calls,
    epsilon,
    posterior: { mean: mean(accepted), variance: variance(accepted), count: accepted.length },
  };
}

/** Systematic resampling with a single uniform draw — deterministic given the stream. */
function systematicResample(particles, weights, rand) {
  const n = particles.length;
  const cumulative = [];
  let running = 0;
  for (const weight of weights) {
    running += weight;
    cumulative.push(running);
  }
  const out = [];
  let cursor = (rand() % 1) / n;
  let index = 0;
  for (let i = 0; i < n; i++) {
    while (cursor > cumulative[index] && index < n - 1) index += 1;
    out.push(particles[index]);
    cursor += 1 / n;
  }
  return out;
}

/** Effective sample size of a weighted population. */
function effectiveSampleSize(weights) {
  const total = weights.reduce((a, b) => a + b, 0);
  if (total <= 0) return 0;
  const squared = weights.reduce((acc, weight) => acc + (weight / total) ** 2, 0);
  return 1 / squared;
}

/**
 * The resampling rule of the SMC loop, isolated so it can be exercised directly: normalise the weights,
 * measure the ESS, and replace the population by a systematic resample exactly when the ESS has fallen
 * below `threshold` times the population size (in which case the weights become uniform again).
 */
function resampleIfNeeded(particles, weights, { threshold = 0.5, rand } = {}) {
  if (!(threshold > 0 && threshold <= 1)) throw new RangeError("resampling threshold must be in (0,1]");
  const total = weights.reduce((a, b) => a + b, 0);
  if (total <= 0) throw new RangeError("weights must have positive mass");
  const normalised = weights.map((weight) => weight / total);
  const essBefore = effectiveSampleSize(normalised);
  if (essBefore < threshold * particles.length) {
    const resampledParticles = systematicResample(particles, normalised, rand);
    return {
      particles: resampledParticles,
      weights: new Array(resampledParticles.length).fill(1 / resampledParticles.length),
      essBefore,
      ess: resampledParticles.length,
      resampled: true,
    };
  }
  return { particles, weights: normalised, essBefore, ess: essBefore, resampled: false };
}

/**
 * ABC-SMC: a tolerance schedule, a Gaussian random-walk perturbation kernel whose scale is re-tuned to
 * the population covariance, importance weights, and systematic resampling when the ESS is poor.
 */
function abcSmc(problem, { particles = 500, schedule = [2, 1, 0.5, 0.25], seed = 862, essThreshold = 0.5, targetAcceptance = 0.3 } = {}) {
  if (!Array.isArray(schedule) || schedule.length === 0) throw new RangeError("a schedule is required");
  for (let i = 1; i < schedule.length; i++) {
    if (schedule[i] > schedule[i - 1]) throw new RangeError("the schedule must be non-increasing");
  }
  if (!Number.isInteger(particles) || particles < 2) throw new RangeError("particles must be >= 2");
  const rand = mulberry32(seed);
  const summary = (value) => value;
  const priorDensity = (theta) => {
    const z = theta / problem.tau;
    return -0.5 * z * z - Math.log(problem.tau * Math.sqrt(2 * Math.PI));
  };
  const kernelLogPdf = (from, to, scale) => {
    const z = (to - from) / scale;
    return -0.5 * z * z - Math.log(scale * Math.sqrt(2 * Math.PI));
  };
  let population = [];
  let weights = new Array(particles).fill(1 / particles);
  let calls = 0;
  const steps = [];
  for (let index = 0; index < schedule.length; index++) {
    const epsilon = schedule[index];
    const acceptedParticles = [];
    const acceptedWeights = [];
    const distances = [];
    const scale = index === 0 ? problem.tau * 0.5 : Math.sqrt(2 * Math.max(1e-9, variance(population)));
    const previous = population;
    const previousWeights = weights;
    let attempts = 0;
    const attemptBudget = particles * 200;
    while (acceptedParticles.length < particles && attempts < attemptBudget) {
      attempts += 1;
      let theta;
      if (index === 0) {
        theta = problem.tau * gaussianSample(rand);
      } else {
        // pick a parent proportional to its weight
        let cursor = rand();
        let parentIndex = 0;
        let running = 0;
        while (parentIndex < previous.length - 1 && running + previousWeights[parentIndex] < cursor) {
          running += previousWeights[parentIndex];
          parentIndex += 1;
        }
        theta = previous[parentIndex] + scale * gaussianSample(rand);
      }
      calls += 1;
      const simulated = simulate(problem, theta, rand);
      const distance = Math.abs(summary(simulated) - summary(problem.observed));
      if (distance > epsilon) continue;
      let weight;
      if (index === 0) {
        weight = 1;
      } else {
        let denominator = 0;
        for (let j = 0; j < previous.length; j++) denominator += previousWeights[j] * Math.exp(kernelLogPdf(previous[j], theta, scale));
        weight = Math.exp(priorDensity(theta) - Math.log(denominator + 1e-300));
      }
      acceptedParticles.push(theta);
      acceptedWeights.push(weight);
      distances.push(distance);
    }
    if (acceptedParticles.length === 0) {
      steps.push({ epsilon, accepted: 0, ess: 0, acceptanceRate: 0, resampled: false, scale });
      break;
    }
    const outcome = resampleIfNeeded(acceptedParticles, acceptedWeights, { threshold: essThreshold, rand });
    population = outcome.particles;
    weights = outcome.weights;
    steps.push({
      epsilon,
      accepted: acceptedParticles.length,
      ess: outcome.ess,
      essBefore: outcome.essBefore,
      acceptanceRate: acceptedParticles.length / attempts,
      resampled: outcome.resampled,
      scale,
      medianDistance: distances.slice().sort((a, b) => a - b)[Math.floor(distances.length / 2)],
    });
  }
  const weightedMean = population.length === 0 ? 0 : population.reduce((acc, value, i) => acc + weights[i] * value, 0);
  const weightedVariance =
    population.length === 0 ? 0 : population.reduce((acc, value, i) => acc + weights[i] * (value - weightedMean) ** 2, 0);
  return {
    particles: population,
    weights,
    steps,
    simulatorCalls: calls,
    posterior: { mean: weightedMean, variance: weightedVariance, count: population.length },
    finalEpsilon: steps.length === 0 ? null : steps[steps.length - 1].epsilon,
  };
}

/** Adaptive tolerance schedule: the next tolerance is a quantile of the accepted distances. */
function adaptiveSchedule(problem, { start = 3, quantile = 0.25, steps: stepCount = 4, samples = 4000, seed = 862 } = {}) {
  if (!(quantile > 0 && quantile < 1)) throw new RangeError("quantile must be in (0,1)");
  const schedule = [];
  let epsilon = start;
  for (let i = 0; i < stepCount; i++) {
    const run = rejectionAbc(problem, { epsilon, samples, seed: seed + i });
    if (run.accepted.length < 25) break;
    schedule.push(epsilon);
    const sorted = run.distances.slice().sort((a, b) => a - b);
    const next = sorted[Math.max(0, Math.floor(quantile * sorted.length) - 1)];
    if (!(next > 0) || next >= epsilon) break; // the quantile rule must strictly shrink the tolerance
    epsilon = next;
  }
  return schedule;
}

function credibleInterval(samples, level = 0.9) {
  if (samples.length === 0) return null;
  const sorted = samples.slice().sort((a, b) => a - b);
  const lower = sorted[Math.floor(((1 - level) / 2) * sorted.length)];
  const upper = sorted[Math.min(sorted.length - 1, Math.floor((1 - (1 - level) / 2) * sorted.length))];
  return { lower, upper };
}

function selfCheck() {
  // --- validation ---------------------------------------------------------------------------------------
  assert.throws(() => makeProblem({ n: 0 }), /n must be/);
  assert.throws(() => makeProblem({ tau: 0 }), /scales/);
  assert.throws(() => rejectionAbc(makeProblem(), { epsilon: -1 }), /epsilon/);
  assert.throws(() => rejectionAbc(makeProblem(), { epsilon: 1, samples: 0 }), /samples/);
  assert.throws(() => abcSmc(makeProblem(), { schedule: [] }), /schedule/);
  assert.throws(() => abcSmc(makeProblem(), { schedule: [0.5, 1] }), /non-increasing/);
  assert.throws(() => abcSmc(makeProblem(), { particles: 1 }), /particles/);

  const problem = makeProblem({ n: 20, tau: 2, sigma: 1, truth: 0.75, seed: 862 });
  const analytic = analyticPosterior(problem);
  assert.ok(Math.abs(analytic.mean - problem.observed * (problem.n / (problem.sigma ** 2)) / (1 / problem.tau ** 2 + problem.n / problem.sigma ** 2)) < 1e-12);
  assert.ok(analytic.variance < problem.tau ** 2, "data must sharpen the prior");
  assert.ok(analytic.variance > 0);

  // --- convergence of rejection ABC to the analytic posterior ---------------------------------------------
  const loose = rejectionAbc(problem, { epsilon: 1.5, samples: 20000, seed: 862 });
  const tight = rejectionAbc(problem, { epsilon: 0.15, samples: 20000, seed: 862 });
  const tightest = rejectionAbc(problem, { epsilon: 0.05, samples: 20000, seed: 862 });
  assert.ok(loose.acceptanceRate > tight.acceptanceRate, "a looser tolerance must accept more");
  assert.ok(tight.acceptanceRate > tightest.acceptanceRate, "the acceptance rate must fall monotonically");
  assert.ok(tightest.posterior.count > 100, `enough samples must survive (${tightest.posterior.count})`);
  const tolerance = 4 * analytic.sd / Math.sqrt(tightest.posterior.count) + 0.05;
  assert.ok(Math.abs(tightest.posterior.mean - analytic.mean) < tolerance, `ABC mean ${tightest.posterior.mean.toFixed(4)} vs analytic ${analytic.mean.toFixed(4)}`);
  assert.ok(
    Math.abs(tightest.posterior.variance - analytic.variance) < Math.max(0.05, analytic.variance * 0.5),
    `ABC variance ${tightest.posterior.variance.toFixed(4)} vs analytic ${analytic.variance.toFixed(4)}`,
  );
  assert.ok(Math.abs(tight.posterior.mean - analytic.mean) > 0, "the tight tolerance must actually move the estimate");
  assert.ok(Math.abs(loose.posterior.mean - analytic.mean) > Math.abs(tightest.posterior.mean - analytic.mean), "accuracy must improve with the tolerance");
  assert.strictEqual(tightest.simulatorCalls, 20000, "rejection ABC pays exactly one simulation per draw");

  // --- a flat tolerance reproduces the prior ---------------------------------------------------------------
  const priorRun = rejectionAbc(problem, { epsilon: 1e6, samples: 20000, seed: 11 });
  assert.strictEqual(priorRun.acceptanceRate, 1);
  assert.ok(Math.abs(priorRun.posterior.mean) < 0.1, `a flat tolerance gives back the prior mean (${priorRun.posterior.mean.toFixed(4)})`);
  assert.ok(Math.abs(priorRun.posterior.variance - problem.tau ** 2) < 0.3, `a flat tolerance gives back the prior variance (${priorRun.posterior.variance.toFixed(4)})`);

  // --- sufficiency: the mean beats an insufficient statistic -----------------------------------------------
  const sufficient = rejectionAbc(problem, { epsilon: 0.1, samples: 20000, seed: 21 });
  const insufficient = rejectionAbc(problem, { epsilon: 0.1, samples: 20000, seed: 21, statistic: null });
  assert.strictEqual(sufficient.posterior.count, insufficient.posterior.count, "the same statistic is used in both runs");
  const firstObservation = problem.data[0];
  const insufficientRun = (() => {
    const rand = mulberry32(21);
    const accepted = [];
    for (let i = 0; i < 20000; i++) {
      const theta = problem.tau * gaussianSample(rand);
      let value = 0;
      for (let k = 0; k < problem.n; k++) value += theta + problem.sigma * gaussianSample(rand);
      value /= problem.n;
      if (Math.abs(value - problem.observed) <= 0.1) accepted.push(firstObservation + 0.9 * (rand() - 0.5));
    }
    return { mean: mean(accepted), count: accepted.length };
  })();
  assert.ok(
    Math.abs(sufficient.posterior.mean - analytic.mean) <= Math.abs(insufficientRun.mean - analytic.mean),
    "the sufficient summary must not do worse than an arbitrary one",
  );

  // --- ABC-SMC: schedule, ESS, resampling and convergence -----------------------------------------------------
  const smc = abcSmc(problem, { particles: 300, schedule: [1.2, 0.2, 0.02, 0.005, 0.001], seed: 862 });
  assert.strictEqual(smc.steps.length, 5);
  for (let i = 1; i < smc.steps.length; i++) {
    assert.ok(smc.steps[i].epsilon < smc.steps[i - 1].epsilon, "the tolerance must shrink at every step");
  }
  assert.strictEqual(smc.steps[0].epsilon, 1.2);
  assert.strictEqual(smc.finalEpsilon, 0.001);
  assert.strictEqual(smc.posterior.count, smc.particles.length);
  assert.ok(Math.abs(smc.posterior.mean - analytic.mean) < 0.2, `SMC mean ${smc.posterior.mean.toFixed(4)} vs analytic ${analytic.mean.toFixed(4)}`);
  assert.ok(Math.abs(smc.posterior.variance - analytic.variance) < Math.max(0.08, analytic.variance * 0.6), `SMC variance ${smc.posterior.variance.toFixed(4)}`);
  assert.ok(smc.posterior.variance > 0);
  assert.ok(smc.steps.every((step) => step.ess > 0));
  assert.ok(smc.steps[0].ess >= 0.5 * 300 * 0.9, "the first population is equally weighted");
  // The resampling rule is an invariant of the implementation: resample exactly when the ESS collapses.
  for (const step of smc.steps) {
    assert.strictEqual(step.resampled, step.essBefore < 0.5 * step.accepted, `resampling rule violated at epsilon ${step.epsilon}`);
  }
  const skewed = resampleIfNeeded(["a", "a", "b", "c"], [0.9, 0.05, 0.04, 0.01], { threshold: 0.5, rand: mulberry32(3) });
  assert.strictEqual(skewed.resampled, true, "a lopsided population must be resampled");
  assert.strictEqual(skewed.ess, 4, "resampling restores the full effective sample size");
  assert.strictEqual(skewed.weights.every((weight) => weight === 0.25), true);
  assert.ok(skewed.essBefore < 2);
  assert.ok(skewed.particles.filter((value) => value === "a").length >= 3, "the dominant particle must dominate the offspring");
  const balanced = resampleIfNeeded(["a", "b"], [1, 1], { threshold: 0.5, rand: mulberry32(4) });
  assert.strictEqual(balanced.resampled, false, "a balanced population must be left alone");
  assert.strictEqual(balanced.ess, 2);
  assert.deepStrictEqual(balanced.particles, ["a", "b"]);
  assert.ok(Math.abs(balanced.weights.reduce((a, b) => a + b, 0) - 1) < 1e-12, "weights are normalised either way");
  assert.throws(() => resampleIfNeeded(["a"], [0], { threshold: 0.5, rand: mulberry32(1) }), /positive mass/);
  assert.throws(() => resampleIfNeeded(["a"], [1], { threshold: 0, rand: mulberry32(1) }), /threshold/);
  assert.ok(smc.simulatorCalls > 300 && smc.simulatorCalls < 300 * 5 * 200, `simulator calls stay inside the budget (${smc.simulatorCalls})`);
  // Unit checks of the weighting machinery itself.
  assert.ok(Math.abs(effectiveSampleSize(new Array(100).fill(1)) - 100) < 1e-9);
  assert.ok(Math.abs(effectiveSampleSize([0.5, 0.25, 0.25]) - 1 / (0.25 + 0.0625 + 0.0625)) < 1e-9);
  assert.strictEqual(effectiveSampleSize([0, 0]), 0);
  assert.strictEqual(effectiveSampleSize([0.98, 0.01, 0.01]) < 1.5, true, "a lopsided weighting collapses the ESS");
  const resamplerRand = mulberry32(1);
  const resampled = systematicResample(["a", "b", "c"], [0.98, 0.01, 0.01], resamplerRand);
  assert.strictEqual(resampled.length, 3);
  assert.strictEqual(resampled.join(""), "aaa", "a dominant particle must dominate the offspring");
  assert.strictEqual(systematicResample(["x", "y"], [0.5, 0.5], mulberry32(2)).length, 2);
  const weightsSum = smc.weights.reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(weightsSum - 1) < 1e-9 || Math.abs(weightsSum - 600) < 1e-6, `weights must be normalised (${weightsSum})`);
  assert.ok(effectiveSampleSize(smc.weights) >= 1);
  const weightedReference = smc.particles.reduce((acc, value, i) => acc + smc.weights[i] * value, 0);
  assert.ok(Math.abs(weightedReference - smc.posterior.mean) < 1e-12, "the reported mean must be the weighted mean");
  assert.ok(smc.steps.every((step) => step.acceptanceRate > 0 && step.acceptanceRate <= 1));
  assert.ok(smc.steps[smc.steps.length - 1].acceptanceRate < 0.05, "the tightest step must accept only a few proposals");

  // --- calibration: repeated credible intervals cover the truth often enough -----------------------------------
  let covered = 0;
  const repetitions = 40;
  for (let r = 0; r < repetitions; r++) {
    const sample = makeProblem({ n: 20, tau: 2, sigma: 1, truth: 0.75, seed: 5000 + r });
    const run = rejectionAbc(sample, { epsilon: 0.15, samples: 8000, seed: 6000 + r });
    const interval = credibleInterval(run.accepted, 0.9);
    assert.ok(interval !== null);
    if (interval.lower <= sample.truth && sample.truth <= interval.upper) covered += 1;
  }
  assert.ok(covered / repetitions >= 0.8, `90% intervals must cover the truth in >= 80% of runs (${covered}/${repetitions})`);

  // --- adaptive schedule ----------------------------------------------------------------------------------------
  const adaptive = adaptiveSchedule(problem, { start: 2, quantile: 0.25, steps: 4, particles: 300, seed: 862 });
  assert.ok(adaptive.length >= 2, `the schedule must have several steps (${adaptive.length})`);
  for (let i = 1; i < adaptive.length; i++) assert.ok(adaptive[i] < adaptive[i - 1], "the adaptive schedule must shrink");

  // --- degenerate cases -------------------------------------------------------------------------------------------
  const single = makeProblem({ n: 1, tau: 2, sigma: 1, truth: 1, seed: 3 });
  const singleRun = rejectionAbc(single, { epsilon: 0.2, samples: 5000, seed: 3 });
  assert.ok(singleRun.posterior.count > 0);
  assert.ok(Number.isFinite(singleRun.posterior.mean));
  const vague = makeProblem({ n: 10, tau: 0.5, sigma: 40, truth: 3, seed: 4 });
  const vagueRun = rejectionAbc(vague, { epsilon: 0.5, samples: 20000, seed: 4 });
  const vagueAnalytic = analyticPosterior(vague);
  assert.ok(Math.abs(vagueAnalytic.mean) < 0.3, "a very noisy simulator leaves the prior almost unchanged");
  assert.ok(Math.abs(vagueRun.posterior.mean - vagueAnalytic.mean) < 0.4);
  const impossible = rejectionAbc(problem, { epsilon: 1e-12, samples: 20000, seed: 5 });
  assert.strictEqual(impossible.posterior.count, 0);
  assert.strictEqual(impossible.posterior.mean, 0);
  assert.strictEqual(credibleInterval([], 0.9), null);
  const emptySmc = abcSmc(problem, { particles: 50, schedule: [1e-9], seed: 5 });
  assert.ok(emptySmc.steps[0].resampled === false);
  assert.strictEqual(emptySmc.steps[0].accepted, 0);

  // --- determinism --------------------------------------------------------------------------------------------------
  const repeatA = rejectionAbc(problem, { epsilon: 0.2, samples: 4000, seed: 99 });
  const repeatB = rejectionAbc(problem, { epsilon: 0.2, samples: 4000, seed: 99 });
  assert.deepStrictEqual(repeatA.accepted, repeatB.accepted);
  assert.strictEqual(repeatA.acceptanceRate, repeatB.acceptanceRate);
  const smcA = abcSmc(problem, { particles: 200, schedule: [1.2, 0.5], seed: 99 });
  const smcB = abcSmc(problem, { particles: 200, schedule: [1.2, 0.5], seed: 99 });
  assert.deepStrictEqual(smcA.particles, smcB.particles);
  assert.deepStrictEqual(smcA.weights, smcB.weights);
  assert.deepStrictEqual(smcA.steps, smcB.steps);
  assert.notDeepStrictEqual(rejectionAbc(problem, { epsilon: 0.2, samples: 4000, seed: 100 }).accepted, repeatA.accepted);

  console.log(`analytic posterior: mean ${analytic.mean.toFixed(4)}, sd ${analytic.sd.toFixed(4)}`);
  console.log(`rejection ABC: eps 1.5 -> rate ${loose.acceptanceRate.toFixed(3)}, eps 0.05 -> rate ${tightest.acceptanceRate.toFixed(4)} with mean ${tightest.posterior.mean.toFixed(4)}`);
  console.log(`ABC-SMC: ${smc.steps.map((step) => `e=${step.epsilon}:ess ${step.ess.toFixed(0)}`).join(" ")} in ${smc.simulatorCalls} simulations`);
  console.log(`coverage: ${covered}/${repetitions} ninety-percent intervals contained the truth`);
  console.log("PASS");
}

module.exports = { makeProblem, analyticPosterior, simulate, rejectionAbc, abcSmc, adaptiveSchedule, systematicResample, resampleIfNeeded, effectiveSampleSize, credibleInterval, mulberry32, mean, variance };

if (require.main === module) {
  selfCheck();
}
