/** Task 798 - Implement Thompson sampling with hierarchical prior simulation. (Tier: Expert, CTCI Beyond - Machine Learning & AI)
 *
 * A hierarchical bandit: the K arms have unknown means mu_k drawn from a common prior
 *   mu_k ~ N(mu_0, tau^2),   reward_t ~ N(mu_{a_t}, sigma^2)   with sigma known
 * while the hyperparameters (mu_0, tau) are themselves unknown. The agent runs a Gibbs sampler over the
 * joint posterior, whose blocks all have closed forms:
 *   mu_k | mu_0, tau, data ~ N( (n_k x_bar_k/sigma^2 + mu_0/tau^2) / (n_k/sigma^2 + 1/tau^2),
 *                               1 / (n_k/sigma^2 + 1/tau^2) )
 *   mu_0 | mu, tau         ~ N( precision-weighted mean of (prior, mu_k), 1/(1/priorVar + K/tau^2) )
 *   tau^2 | mu, mu_0       ~ InverseGamma( shape + K/2,  scale + sum_k (mu_k - mu_0)^2 / 2 )
 * Thompson sampling then plays the arm with the largest sampled mu_k. Pooling is the point: an arm with
 * few observations is shrunk toward mu_0, so the sampler borrows strength across arms.
 *
 * Invariants verified: with the hyperparameters frozen the sampler's draws match the analytic normal
 * posterior in mean and variance (the independent oracle); with only the variance block active the
 * tau^2 draws match the closed-form inverse-gamma mean; the posterior mean of a sparsely observed arm
 * lies strictly between its sample mean and mu_0 and shrinks monotonically less as data arrives; the
 * hierarchical policy beats random and pooling does not hurt against a fixed-prior (unpooled) policy
 * with equal means; regret is sublinear; single-arm, identical-arm and one-round cases give zero
 * regret; and invalid hyperparameters are rejected.
 *
 * What is simplified: a few hundred Gibbs sweeps on three arms, sigma assumed known, the standard
 * weakly informative hyperprior, and no Rao-Blackwellisation. The chain is driven by a seeded splitmix32
 * (with Box-Muller and Marsaglia-Tsang draws), so every run is deterministic and all gaps are measured.
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

/** Gamma(shape, 1) draw by Marsaglia-Tsang, with the shape < 1 boost identity. */
function gammaDraw(shape, random) {
  if (!(shape > 0)) throw new RangeError("the gamma shape must be > 0");
  if (shape < 1) return gammaDraw(shape + 1, random) * Math.max(random(), 1e-12) ** (1 / shape);
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (let attempt = 0; attempt < 10000; attempt++) {
    const x = gaussian(random);
    const v = (1 + c * x) ** 3;
    if (v <= 0) continue;
    if (Math.log(Math.max(random(), 1e-12)) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v;
  }
  throw new Error("the gamma sampler failed to accept a draw");
}

/** Analytic posterior of mu_k when the hyperparameters are frozen: the sampler's oracle. */
function frozenPosterior(count, sampleMean, { sigma, mu0, tau2 }) {
  const precision = count / (sigma * sigma) + 1 / tau2;
  return { mean: (count * sampleMean / (sigma * sigma) + mu0 / tau2) / precision, variance: 1 / precision };
}

class HierarchicalThompson {
  constructor(arms, { sigma = 0.5, priorMean = 0, priorVar = 100, shape = 1, scale = 1,
    learnHyper = true, learnTau2 = null, freezeMus = false, seed = 1 } = {}) {
    if (!Number.isInteger(arms) || arms < 1) throw new RangeError("the number of arms must be a positive integer");
    if (!(sigma > 0)) throw new RangeError("sigma must be > 0");
    if (!(priorVar > 0)) throw new RangeError("the prior variance must be > 0");
    if (!(shape > 0) || !(scale > 0)) throw new RangeError("the inverse-gamma parameters must be > 0");
    this.arms = arms;
    this.sigma = sigma;
    this.priorMean = priorMean;
    this.priorVar = priorVar;
    this.shape = shape;
    this.scale = scale;
    this.learnHyper = learnHyper;
    this.learnTau2 = learnTau2 === null ? learnHyper : learnTau2;
    this.freezeMus = freezeMus;
    this.random = makeRng(seed);
    this.counts = new Array(arms).fill(0);
    this.sums = new Array(arms).fill(0);
    this.state = { mus: new Array(arms).fill(priorMean), mu0: priorMean, tau2: 1 };
    this.samples = { mus: [], mu0: [], tau2: [] };
  }

  sampleMean(arm) {
    return this.counts[arm] === 0 ? 0 : this.sums[arm] / this.counts[arm];
  }

  /** One Gibbs sweep over mu_k, mu_0 and tau^2; records every draw for the posterior diagnostics. */
  sweep() {
    const { sigma, priorMean, priorVar, shape, scale } = this;
    if (!this.freezeMus) {
      for (let arm = 0; arm < this.arms; arm++) {
        const precision = this.counts[arm] / (sigma * sigma) + 1 / this.state.tau2;
        const weighted = this.counts[arm] * this.sampleMean(arm) / (sigma * sigma) + this.state.mu0 / this.state.tau2;
        this.state.mus[arm] = weighted / precision + gaussian(this.random) / Math.sqrt(precision);
      }
    }
    if (this.learnHyper) {
      const precision = 1 / priorVar + this.arms / this.state.tau2;
      const weighted = priorMean / priorVar + this.state.mus.reduce((acc, mu) => acc + mu / this.state.tau2, 0);
      this.state.mu0 = weighted / precision + gaussian(this.random) / Math.sqrt(precision);
    } else {
      this.state.mu0 = priorMean;
    }
    if (this.learnTau2) {
      const spread = this.state.mus.reduce((acc, mu) => acc + (mu - this.state.mu0) ** 2, 0);
      this.state.tau2 = (scale + spread / 2) / gammaDraw(shape + this.arms / 2, this.random);
    }
    this.samples.mus.push(this.state.mus.slice());
    this.samples.mu0.push(this.state.mu0);
    this.samples.tau2.push(this.state.tau2);
    return this.state;
  }

  /** Runs the chain (with burn-in and thinning) and returns posterior means plus the retained draws. */
  posterior({ sweeps = 300, burnIn = 50, thin = 1 } = {}) {
    for (let k = 0; k < burnIn + sweeps; k++) this.sweep();
    const keep = (values) => values.slice(burnIn).filter((_, i) => i % thin === 0);
    const mus = keep(this.samples.mus);
    const mu0 = keep(this.samples.mu0);
    const tau2 = keep(this.samples.tau2);
    const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;
    return {
      arms: Array.from({ length: this.arms }, (_, arm) => mean(mus.map((draw) => draw[arm]))),
      mu0: mean(mu0),
      tau2: mean(tau2),
      samples: mus,
    };
  }

  /** Thompson draw: the arm with the largest current sampled mean. */
  select() {
    let best = 0;
    for (let arm = 1; arm < this.arms; arm++) if (this.state.mus[arm] > this.state.mus[best]) best = arm;
    return best;
  }

  observe(arm, reward) {
    this.counts[arm] += 1;
    this.sums[arm] += reward;
    return this;
  }
}

/** Simulates one policy over a fixed arm set; returns the exact cumulative regret and per-round gaps. */
function simulate(policy, truths, { horizon = 300, sigma = 0.5, seed = 7, sweepsPer = 5, priorMean = 0.6 } = {}) {
  const best = Math.max(...truths);
  const draw = makeRng(seed + 2);
  const uniform = makeRng(seed + 1);
  const agent = policy === "random" ? null : new HierarchicalThompson(truths.length, {
    sigma, seed, priorMean, learnHyper: policy === "hierarchical",
  });
  const gaps = [];
  const counts = new Array(truths.length).fill(0);
  for (let t = 0; t < horizon; t++) {
    let arm;
    if (policy === "random") {
      arm = Math.floor(uniform() * truths.length) % truths.length;
    } else {
      for (let k = 0; k < sweepsPer; k++) agent.sweep();
      arm = agent.select();
      agent.observe(arm, truths[arm] + gaussian(draw) * sigma);
    }
    gaps.push(best - truths[arm]);
    counts[arm] += 1;
  }
  return { regret: gaps.reduce((a, b) => a + b, 0), gaps, counts, agent };
}

function main() {
  const notes = [];

  // 1. Frozen hyperparameters: the sampler must match the analytic normal posterior in mean and variance.
  const oracleSetup = { sigma: 0.5, mu0: 0.4, tau2: 0.25 };
  const sampler = new HierarchicalThompson(3, { sigma: oracleSetup.sigma, priorMean: oracleSetup.mu0, learnHyper: false, seed: 11 });
  sampler.state.tau2 = oracleSetup.tau2;
  sampler.observe(0, 1.1);
  sampler.observe(0, 0.9);
  sampler.observe(1, -0.3);
  sampler.observe(2, 0.5);
  const draws = [[], []];
  for (let k = 0; k < 20000; k++) {
    const state = sampler.sweep();
    draws[0].push(state.mus[0]);
    draws[1].push(state.mus[1]);
  }
  const oracleReport = [0, 1].map((arm) => {
    const analytic = frozenPosterior(sampler.counts[arm], sampler.sampleMean(arm), oracleSetup);
    const mean = draws[arm].reduce((a, b) => a + b, 0) / draws[arm].length;
    const variance = draws[arm].reduce((acc, v) => acc + (v - mean) ** 2, 0) / draws[arm].length;
    const standardError = Math.sqrt(variance / draws[arm].length);
    return { arm, analytic, mean, variance, standardError, sigmas: Math.abs(mean - analytic.mean) / standardError };
  });
  notes.push(`frozen-hyperparameter oracle: ${oracleReport.map((r) => `arm ${r.arm} mean ${r.mean.toFixed(5)} vs ${r.analytic.mean.toFixed(5)} (${r.sigmas.toFixed(2)} MC sigma), variance ${r.variance.toFixed(6)} vs ${r.analytic.variance.toFixed(6)}`).join("; ")}`);
  ok(oracleReport.every((r) => r.sigmas < 4), "the sampler mean must match the analytic posterior mean within Monte-Carlo error");
  ok(oracleReport.every((r) => Math.abs(r.variance / r.analytic.variance - 1) < 0.05), "the sampler variance must match the analytic posterior variance");

  // 2. Tau^2 only: with the means frozen the draws are i.i.d. InverseGamma, so the closed-form mean applies.
  const tauSampler = new HierarchicalThompson(3, { sigma: 0.5, shape: 1, scale: 1, learnHyper: false, learnTau2: true, freezeMus: true, priorMean: 0.4, seed: 29 });
  tauSampler.state.mus = [0.3, 0.9, 0.6];
  tauSampler.state.mu0 = 0.4;
  const posterior = tauSampler.posterior({ sweeps: 3000, burnIn: 100, thin: 1 });
  const spread = tauSampler.state.mus.reduce((acc, mu) => acc + (mu - tauSampler.state.mu0) ** 2, 0);
  const closedFormMean = (1 + spread / 2) / (1 + 3 / 2 - 1);
  notes.push(`tau^2 block: sampled mean ${posterior.tau2.toFixed(5)} vs closed-form inverse-gamma mean ${closedFormMean.toFixed(5)} (relative error ${(Math.abs(posterior.tau2 / closedFormMean - 1) * 100).toFixed(2)}%)`);
  ok(Math.abs(posterior.tau2 / closedFormMean - 1) < 0.1, "the tau^2 draws must match the closed-form inverse-gamma mean");
  ok(posterior.samples.every((draw) => draw.join(",") === "0.3,0.9,0.6"), "frozen means must stay frozen across sweeps");

  // 3. Shrinkage: the posterior mean of a sparse arm lies between mu_0 and its sample mean, and moves
  //    toward the sample mean as the count grows.
  const observed = 1.5;
  const shrinkage = [1, 4, 16, 64].map((count) => ({
    count,
    mean: frozenPosterior(count, observed, { sigma: 0.5, mu0: 0.5, tau2: 0.25 }).mean,
  }));
  notes.push(`shrinkage toward mu_0 = 0.5 with a sample mean of ${observed}: ${shrinkage.map((r) => `n = ${r.count} -> ${r.mean.toFixed(4)}`).join(", ")}`);
  ok(shrinkage.every((r) => r.mean > 0.5 && r.mean < observed), "the posterior mean must be shrunk toward mu_0");
  ok(shrinkage.every((r, i) => i === 0 || r.mean > shrinkage[i - 1].mean), "more data must shrink less");

  // 4. Policy comparison. (a) With well-separated arms the hierarchical policy crushes random.
  const separated = [0.9, 0.6, 0.3];
  const seeds = [23, 37, 51, 65, 79];
  const separatedTotals = seeds.map((seed) => ({ hierarchical: simulate("hierarchical", separated, { seed }).regret, random: simulate("random", separated, { seed }).regret }));
  const meanOf = (rows, key) => rows.reduce((acc, row) => acc + row[key], 0) / rows.length;
  notes.push(`separated arms [0.9, 0.6, 0.3] over ${seeds.length} seeds: hierarchical ${meanOf(separatedTotals, "hierarchical").toFixed(2)} vs random ${meanOf(separatedTotals, "random").toFixed(2)}`);
  ok(meanOf(separatedTotals, "hierarchical") < meanOf(separatedTotals, "random") / 3, "the hierarchical policy must be far better than random");

  // (b) Sparse many-arm regime (10 arms, 40 rounds, 4 observations per arm): pooling borrows strength,
  //     so the hierarchical policy should match or beat a per-arm policy with a vague prior.
  const manyArms = [0.55, 0.6, 0.5, 0.58, 0.52, 0.62, 0.56, 0.54, 0.59, 0.51];
  const sparseSeeds = [23, 37, 51, 65, 79, 91, 103, 115];
  const sparseTotals = sparseSeeds.map((seed) => ({
    hierarchical: simulate("hierarchical", manyArms, { horizon: 40, sweepsPer: 15, sigma: 0.3, seed }).regret,
    unpooled: simulate("unpooled", manyArms, { horizon: 40, sweepsPer: 15, sigma: 0.3, priorMean: 0.5, seed }).regret,
  }));
  const pooledWins = sparseTotals.filter((row) => row.hierarchical <= row.unpooled).length;
  notes.push(`sparse regime (10 arms, 40 rounds) over ${sparseSeeds.length} seeds: pooled ${meanOf(sparseTotals, "hierarchical").toFixed(3)} vs unpooled ${meanOf(sparseTotals, "unpooled").toFixed(3)}, pooled wins ${pooledWins}/${sparseSeeds.length}`);
  ok(meanOf(sparseTotals, "hierarchical") <= 1.05 * meanOf(sparseTotals, "unpooled"), "pooling must not hurt in the sparse regime");
  ok(pooledWins >= 4, "pooling must win on at least half of the seeds");

  // 5. Regret must be sublinear: the last quarter of the run is cheaper than the first quarter.
  const tracked = simulate("hierarchical", separated, { horizon: 400, seed: 23 });
  const quarters = [0, 1, 2, 3].map((q) => tracked.gaps.slice((q * 400) / 4, ((q + 1) * 400) / 4).reduce((a, b) => a + b, 0));
  notes.push(`regret by quarter: ${quarters.map((q) => q.toFixed(3)).join(" -> ")}, total ${tracked.regret.toFixed(3)} over 400 rounds`);
  ok(quarters[3] < quarters[0], "the final quarter must be cheaper than the first");
  ok(tracked.counts.reduce((a, b) => a + b, 0) === 400, "every round must be counted exactly once");

  // 6. Boundaries: one arm, identical arms and a one-round horizon.
  ok(simulate("hierarchical", [0.3], { horizon: 50, seed: 3 }).regret === 0, "one arm must give zero regret");
  ok(simulate("hierarchical", [0.3, 0.3], { horizon: 50, seed: 3 }).regret === 0, "identical arms must give zero regret");
  ok(simulate("hierarchical", [0.3, 0.5], { horizon: 1, seed: 3 }).gaps.length === 1, "a one-round horizon must produce a single gap");

  // 7. Validation and determinism.
  throws(() => new HierarchicalThompson(3, { sigma: 0 }), RangeError, "sigma = 0 must throw");
  throws(() => new HierarchicalThompson(3, { priorVar: 0 }), RangeError, "a zero prior variance must throw");
  throws(() => new HierarchicalThompson(0), RangeError, "zero arms must throw");
  throws(() => new HierarchicalThompson(2, { shape: -1 }), RangeError, "a negative shape must throw");
  throws(() => gammaDraw(0, makeRng(1)), RangeError, "a zero gamma shape must throw");
  const repeat = simulate("hierarchical", separated, { horizon: 400, seed: 23 });
  ok(repeat.regret === tracked.regret && repeat.counts.join(",") === tracked.counts.join(","), "the run must be reproducible");

  for (const line of notes) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = { makeRng, gaussian, gammaDraw, frozenPosterior, HierarchicalThompson, simulate };

if (require.main === module) main();
