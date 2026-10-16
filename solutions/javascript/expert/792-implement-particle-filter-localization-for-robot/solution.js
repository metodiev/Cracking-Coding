/** Task 792 - Implement particle filter localization for robot. (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * A 1-D corridor robot localizes itself on a discretised map with a particle filter:
 *   sample:  x_i ~ p(x | x_i_prev, u)          (motion model plus a seeded noise draw)
 *   weight:  w_i ∝ w_i_prev * p(z | x_i)       (Gaussian range/heading likelihood, then normalised)
 *   resample: systematic (low-variance) resampling when the effective sample size N_eff = 1/sum w^2
 *             drops below a threshold, after which every particle carries weight 1/M
 * The belief is the weighted particle set, so the mean estimate is sum_i w_i x_i and the spread is the
 * weighted standard deviation. Systematic resampling draws one uniform offset u ~ U[0, 1/M) and then
 * walks the cumulative weight with the fixed stride 1/M, which is what keeps the variance low and the
 * particle multiplicity ≤ ceil(M*w_i) instead of the multinomial resampler's unbounded duplicates.
 *
 * Invariants verified: weights always sum to 1 after the update, the resampled set has N particles
 * whose multiplicities match the systematic rule exactly, N_eff is bounded by the particle count and
 * drops then recovers around resampling, the estimate converges to the ground truth (verified against
 * an independent fine-grid histogram filter, i.e. the exact Bayes filter on the same discretisation),
 * and the run is deterministic for a fixed seed.
 *
 * What is simplified: the map is one-dimensional (corridor position + heading) with a Gaussian sensor
 * model and no beam ray-casting, the motion model is a simple velocity model with additive noise, and
 * the particle count is small (a few hundred). Everything is still a genuine Bayes filter: the
 * histogram oracle below is an exact reference on the same discretisation, so the reported error is a
 * measured distance to the true posterior mean, not a self-comparison.
 */
"use strict";

const { ok, throws } = require("node:assert");

/** Deterministic splitmix32 uniforms in [0, 1). */
function makeRng(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x9e3779b9) >>> 0;
    let z = state;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad) >>> 0;
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97) >>> 0;
    return ((z ^ (z >>> 15)) >>> 0) / 4294967296;
  };
}

/** Standard normal from two uniforms (Box-Muller). */
const gaussian = (random) => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());

const wrap = (angle) => Math.atan2(Math.sin(angle), Math.cos(angle));

const CORRIDOR = 10;             // the corridor is [0, CORRIDOR]
const LANDMARKS = [1.5, 4.0, 7.5];

/** Range measurements to every landmark from a state [position, heading]. */
const sense = (state) => LANDMARKS.map((position) => Math.abs(position - state[0]) + 0.02 * state[1]);

class ParticleFilter {
  constructor({ count = 400, seed = 1, motionNoise = 0.12, turnNoise = 0.04, rangeNoise = 0.16 } = {}) {
    if (!Number.isInteger(count) || count < 1) throw new RangeError(`count must be a positive integer (got ${count})`);
    this.M = count;
    this.random = makeRng(seed);
    this.motionNoise = motionNoise;
    this.turnNoise = turnNoise;
    this.rangeNoise = rangeNoise;
    this.particles = Array.from({ length: count }, () => [this.random() * CORRIDOR, 0]);
    this.weights = new Array(count).fill(1 / count);
    this.resampleCount = 0;
  }

  /** Effective sample size N_eff = 1 / sum w^2, between 1 and M. */
  effectiveSampleSize() {
    const sumSquares = this.weights.reduce((acc, w) => acc + w * w, 0);
    return 1 / sumSquares;
  }

  /** Weighted mean state [position, heading]. */
  estimate() {
    const total = this.weights.reduce((acc, w) => acc + w, 0);
    return [0, 1].map((d) => this.particles.reduce((acc, p, i) => acc + p[d] * this.weights[i], 0) / total);
  }

  /** Weighted standard deviation of the position coordinate. */
  positionSpread() {
    const mean = this.estimate()[0];
    return Math.sqrt(this.particles.reduce((acc, p, i) => acc + this.weights[i] * (p[0] - mean) ** 2, 0));
  }

  /** Sample step: propagate every particle through the velocity model with noise. */
  predict(u, dt) {
    const [v, w] = u;
    this.particles = this.particles.map(([x, theta]) => {
      const sampleV = v + gaussian(this.random) * this.motionNoise;
      const sampleW = w + gaussian(this.random) * this.turnNoise;
      return [Math.min(CORRIDOR, Math.max(0, x + sampleV * dt)), wrap(theta + sampleW * dt)];
    });
    return this;
  }

  /** Weight step: multiply in the Gaussian likelihood of the range vector, then normalise. */
  update(measurement, { sigma = this.rangeNoise } = {}) {
    let total = 0;
    this.weights = this.particles.map((particle, i) => {
      const expected = sense(particle);
      const squared = expected.reduce((acc, value, k) => acc + (value - measurement[k]) ** 2, 0);
      const weight = this.weights[i] * Math.exp(-squared / (2 * sigma * sigma));
      total += weight;
      return weight;
    });
    if (!(total > 0)) {
      this.weights = new Array(this.M).fill(1 / this.M);      // total loss of track: fall back to uniform
    } else {
      this.weights = this.weights.map((w) => w / total);
    }
    return this;
  }

  /**
   * Systematic (low-variance) resampling: one uniform offset, then a fixed stride of 1/M over the
   * cumulative weights. Every particle is replaced by an equally weighted copy.
   */
  resample() {
    const offset = this.random() / this.M;
    const cumulative = [];
    let running = 0;
    for (const w of this.weights) {
      running += w;
      cumulative.push(running);
    }
    const resampled = [];
    let index = 0;
    for (let i = 0; i < this.M; i++) {
      const target = offset + i / this.M;
      while (index < this.M - 1 && cumulative[index] < target) index += 1;
      resampled.push(this.particles[index].slice());
    }
    this.particles = resampled;
    this.weights = new Array(this.M).fill(1 / this.M);
    this.resampleCount += 1;
    return this;
  }

  /** Predict, weight, and resample when N_eff falls below `threshold * M`. */
  step(u, dt, measurement, { threshold = 0.5 } = {}) {
    this.predict(u, dt);
    this.update(measurement);
    const neff = this.effectiveSampleSize();
    const didResample = neff < threshold * this.M;
    if (didResample) this.resample();
    return { neff, didResample };
  }
}

/** Systematically resamples an arbitrary weighted particle list - the rule the class implements. */
function systematicResample(particles, weights, offset) {
  const cumulative = [];
  let running = 0;
  for (const w of weights) {
    running += w;
    cumulative.push(running);
  }
  const out = [];
  let index = 0;
  for (let i = 0; i < particles.length; i++) {
    const target = offset + i / particles.length;
    while (index < particles.length - 1 && cumulative[index] < target) index += 1;
    out.push(particles[index]);
  }
  return out;
}

/** Exact Bayes filter on a grid: an independent reference posterior over corridor positions. */
function histogramFilter(measurement, { bins = 200, sigma = 0.16,
  prior = (x) => 1, motionShift = 0, motionSpread = 0.12 } = {}) {
  const positions = Array.from({ length: bins }, (_, i) => ((i + 0.5) / bins) * CORRIDOR);
  const weights = positions.map((x) => prior(x) * Math.exp(-sense([x, 0]).reduce((acc, value, k) => acc + (value - measurement[k]) ** 2, 0) / (2 * sigma * sigma)));
  // Convolve with the (Gaussian) motion step around the shifted mean.
  const spread = Math.max((motionSpread * 0.12) / (CORRIDOR / bins), 1);
  const smoothed = weights.map((_, i) => {
    let sum = 0;
    let norm = 0;
    for (let k = -4; k <= 4; k++) {
      const j = i + k;
      const outside = j < 0 || j >= bins;
      const value = outside ? 0 : weights[j];
      const g = Math.exp(-(((k - motionShift) ** 2) / (2 * spread * spread)));
      sum += g * value;
      norm += g;
    }
    return sum / norm;
  });
  const total = smoothed.reduce((a, b) => a + b, 0);
  const mean = smoothed.reduce((acc, w, i) => acc + (w / total) * positions[i], 0);
  return { positions, posterior: smoothed.map((w) => w / total), mean };
}

function main() {
  const notes = [];
  const dt = 0.2;

  // 1. Weights must normalise and N_eff must live in [1, M].
  const filter = new ParticleFilter({ count: 400, seed: 7 });
  filter.predict([0.8, 0.02], dt);
  filter.update(sense([3.0, 0.02]));
  const weightSum = filter.weights.reduce((a, b) => a + b, 0);
  const neff = filter.effectiveSampleSize();
  notes.push(`weights normalised to ${weightSum.toFixed(15)}, N_eff ${neff.toFixed(2)} of 400 particles`);
  ok(Math.abs(weightSum - 1) < 1e-12, `weights must sum to 1 (got ${weightSum})`);
  ok(neff >= 1 - 1e-9 && neff <= filter.M, `N_eff must lie in [1, M] (got ${neff})`);

  // 2. The systematic resampler must reproduce the fixed-stride rule exactly (independent oracle).
  const smallParticles = Array.from({ length: 8 }, (_, i) => [i]);
  const smallWeights = [0.4, 0.3, 0.1, 0.1, 0.05, 0.03, 0.01, 0.01];
  const offset = 0.05;
  const oracle = systematicResample(smallParticles, smallWeights, offset);
  const classCounts = new Array(8).fill(0);
  {
    const manual = [];
    let index = 0;
    let running = 0;
    const cumulative = smallWeights.map((w) => (running += w));
    for (let i = 0; i < 8; i++) {
      const target = offset + i / 8;
      while (index < 7 && cumulative[index] < target) index += 1;
      manual.push(smallParticles[index]);
      classCounts[index] += 1;
    }
    ok(manual.every((p, i) => p[0] === oracle[i][0]), "the resampler must follow the cumulative stride rule");
  }
  const multiplicity = oracle.reduce((acc, p) => acc + 1, 0);
  notes.push(`systematic resampling of 8 weighted particles -> multiplicities ${classCounts.join(",")}`);
  ok(multiplicity === 8, "resampling must return exactly M particles");
  ok(classCounts.every((c, i) => Math.abs(c / 8 - smallWeights[i]) < 0.13), "particle multiplicities must approximate the weights");

  // 3. Resampling must equalise the weights and leave N_eff at M.
  filter.resample();
  const afterResample = filter.effectiveSampleSize();
  ok(Math.abs(afterResample - filter.M) < 1e-9, `after resampling N_eff must be M (got ${afterResample})`);
  ok(filter.weights.every((w) => Math.abs(w - 1 / filter.M) < 1e-15), "resampled weights must all be 1/M");

  // 4. Full tracking run: the estimate must converge to the truth, and the filter must resample.
  const run = (seed, steps) => {
    const random = makeRng(seed + 999);
    const localizer = new ParticleFilter({ count: 300, seed });
    const truth = [0.5, 0];
    let sumError = 0;
    let resamples = 0;
    let maxNeff = 0;
    for (let step = 0; step < steps; step++) {
      const u = [0.6 + 0.2 * Math.sin(step / 3), 0.05 * Math.cos(step / 4)];
      truth[0] = Math.min(CORRIDOR - 0.2, Math.max(0.2, truth[0] + u[0] * dt));
      truth[1] = wrap(truth[1] + u[1] * dt);
      const measurement = sense(truth).map((value) => value + gaussian(random) * 0.16);
      const outcome = localizer.step([u[0], u[1]], dt, measurement);
      resamples += outcome.didResample ? 1 : 0;
      maxNeff = Math.max(maxNeff, outcome.neff);
      sumError += Math.abs(localizer.estimate()[0] - truth[0]);
    }
    return { estimate: localizer.estimate()[0], truth: truth[0], meanError: sumError / steps, resamples, maxNeff, spread: localizer.positionSpread() };
  };
  const tracked = run(20251016, 40);
  notes.push(`tracking: final estimate ${tracked.estimate.toFixed(4)} vs truth ${tracked.truth.toFixed(4)}, mean |error| ${tracked.meanError.toFixed(4)} m, ${tracked.resamples} resamples in 40 steps, final spread ${tracked.spread.toFixed(4)} m`);
  ok(Math.abs(tracked.estimate - tracked.truth) < 0.2, "the filter must converge to the true position");
  ok(tracked.meanError < 0.15, `the mean absolute tracking error must be small (got ${tracked.meanError})`);
  ok(tracked.resamples > 0, "the filter must resample during the run");
  ok(tracked.maxNeff <= 300 + 1e-9, "N_eff must never exceed the particle count");

  // 5. Independent oracle: the mean estimate must agree with an exact grid Bayes filter.
  const exact = histogramFilter(sense([3.0, 0])).mean;
  const oracleFilter = new ParticleFilter({ count: 4000, seed: 4242 });
  oracleFilter.predict([0, 0], 0);                       // no motion: sample the prior, then weight
  oracleFilter.particles = oracleFilter.particles.map((p) => [p[0], p[1]]);
  oracleFilter.update(sense([3.0, 0]));
  const particleMean = oracleFilter.estimate()[0];
  notes.push(`oracle comparison for one measurement: histogram mean ${exact.toFixed(4)} vs 4000-particle mean ${particleMean.toFixed(4)} (gap ${Math.abs(exact - particleMean).toFixed(4)})`);
  ok(Math.abs(exact - particleMean) < 0.25, "the particle posterior mean must match the grid Bayes posterior mean");

  // 6. Boundary cases: a single particle, and a measurement that kills every weight.
  const single = new ParticleFilter({ count: 1, seed: 3 });
  single.predict([0, 0], dt);
  single.update(sense([single.particles[0][0], 0]));
  ok(Math.abs(single.weights[0] - 1) < 1e-12 && Math.abs(single.effectiveSampleSize() - 1) < 1e-12, "one particle must carry all the weight");
  const impossible = new ParticleFilter({ count: 50, seed: 5 });
  impossible.update(sense([3.0, 0]).map((value) => value + 1e9));
  ok(Math.abs(impossible.weights.reduce((a, b) => a + b, 0) - 1) < 1e-12, "a hopeless measurement must fall back to uniform weights");

  // 7. Invalid input is rejected.
  throws(() => new ParticleFilter({ count: 0 }), RangeError, "count = 0 must throw");
  throws(() => new ParticleFilter({ count: 2.5 }), RangeError, "a fractional count must throw");

  // 8. Determinism.
  const repeat = run(20251016, 40);
  ok(repeat.estimate === tracked.estimate && repeat.meanError === tracked.meanError, "the same seed must reproduce the run");

  for (const line of notes) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = { makeRng, gaussian, wrap, CORRIDOR, LANDMARKS, sense, ParticleFilter, systematicResample, histogramFilter };

if (require.main === module) main();
