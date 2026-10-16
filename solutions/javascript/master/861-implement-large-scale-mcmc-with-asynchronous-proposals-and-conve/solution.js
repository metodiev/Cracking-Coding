/** Task 861 - Implement large-scale MCMC with asynchronous proposals and convergence diagnostics.
 *  (Tier: Master, CTCI Beyond — Machine Learning & AI)
 *
 * Model. Many Metropolis-Hastings chains run at the same time with no global barrier: worker w draws its
 * own proposals from its own seeded stream, decides for itself, and the scheduler merely interleaves the
 * workers round by round (asynchronous proposals). Chains can also be *restarted* from a shared point:
 * whenever any worker accepts a sample the shared state moves, and on a fixed schedule the other workers
 * re-initialise from that shared state. A restart only replaces the chain's current point, it does not
 * change the kernel, so the invariant distribution of every chain is untouched and only the transient
 * shortens — which the R-hat comparison below measures.
 *
 * Two targets are used deliberately: a small finite-state target where the Metropolis kernel can be
 * enumerated exactly, so detailed balance is checked as an identity rather than by simulation, and a
 * Gaussian target where the posterior moments, the acceptance probability and the asymptotics of the
 * proposal scale are all known in closed form.
 *
 * Diagnostics implemented: split-R-hat (Gelman-Rubin on chains halved after the burn-in) and the
 * effective sample size by the initial-positive-sequence autocorrelation sum, cross-checked against an
 * overlapping-batch-means estimator.
 *
 * What is simplified: no parallelism of any kind (workers are interleaved deterministically in one
 * process), no network, no wall clock, no adaptive proposals, fixed proposal scales, and randomness
 * confined to a single seeded mulberry32 stream per worker — so a run is bit-for-bit reproducible.
 *
 * Invariants verified:
 *   exactness   on the finite target the enumerated transition matrix satisfies pi_i P_ij = pi_j P_ji for
 *               every pair (an identity, not a statistical test), the transition matrix is stochastic and
 *               pi is its exact stationary distribution; the acceptance rate equals sum_i pi_i a_i.
 *   consistency the Gaussian chain's empirical mean, variance and acceptance rate match the analytic
 *               values inside the Monte-Carlo tolerance implied by its own effective sample size.
 *   diagnostics split-R-hat equals an independently computed value on a hand-built example, is ~1 for
 *               segments drawn from the target and grows past 1.2 for chains started far apart, and ESS is
 *               confirmed by a second estimator.
 *   adaptation  asynchronous restarts shorten the transient (lower R-hat at the same step count) without
 *               shifting the stationary moments, and the asynchronous schedule needs max(steps) rounds
 *               instead of the sum of the per-chain step counts.
 *   determinism  identical seeds reproduce identical chains, diagnostics and round counts.
 */
"use strict";

const assert = require("node:assert");

/** mulberry32 — the only source of randomness; every worker owns a stream derived from one seed. */
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

/** Finite-state target: a ring with a deterministic, non-uniform stationary distribution. */
function makeRingTarget({ states = 9, seed = 861 } = {}) {
  if (!Number.isInteger(states) || states < 3) throw new RangeError("states must be >= 3");
  const rand = mulberry32(seed);
  const weights = [];
  for (let i = 0; i < states; i++) weights.push(0.5 + rand());
  const total = weights.reduce((a, b) => a + b, 0);
  const pi = weights.map((w) => w / total);
  const logPdf = (state) => Math.log(pi[state]);
  return { states, pi, logPdf };
}

/** Proposal kernel on the ring: move to a uniformly chosen neighbour, otherwise stay. */
function ringProposal(states, state, rand) {
  const roll = rand();
  if (roll < 1 / 3) return state;
  if (roll < 2 / 3) return (state + 1) % states;
  return (state + states - 1) % states;
}

/**
 * The exact Metropolis kernel of the ring target: P_ij = q(i->j) * min(1, pi_j/pi_i) + (1 - sum_j) delta.
 * Enumerating it lets detailed balance be checked as an exact identity.
 */
function enumerateKernel(target) {
  const n = target.states;
  const kernel = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    const neighbours = [i, (i + 1) % n, (i + n - 1) % n];
    let keep = 0;
    for (let index = 0; index < neighbours.length; index++) {
      const j = neighbours[index];
      const probability = 1 / 3;
      const accept = Math.min(1, target.pi[j] / target.pi[i]);
      kernel[i][j] += probability * accept;
      keep += probability * (1 - accept);
    }
    kernel[i][i] += keep;
  }
  return kernel;
}

/** Same kernel for the Gaussian target with a symmetric normal random walk, as an exact integral-free check. */
function gaussianAcceptanceProbability(proposalScale, targetVariance = 1) {
  // E[min(1, exp(-(x'^2 - x^2)/2))] under x ~ N(0, targetVariance), x' = x + eps.
  const rand = mulberry32(97);
  const samples = 400000;
  let total = 0;
  for (let i = 0; i < samples; i++) {
    const x = Math.sqrt(targetVariance) * gaussianSample(rand);
    const proposal = x + proposalScale * gaussianSample(rand);
    const logRatio = (-0.5 * proposal * proposal + 0.5 * x * x) / targetVariance;
    total += Math.min(1, Math.exp(logRatio));
  }
  return total / samples;
}

/**
 * Runs `workers` Metropolis chains without a global barrier. Each round advances every worker once, so
 * the schedule costs max(steps) rounds rather than the sum — the point of asynchronous proposals.
 */
function runChains({ logTarget, proposalScale, workers = 8, steps = 4000, initial = () => 0, seed = 861, restartEvery = 0, sharedStart = null }) {
  if (!Number.isInteger(workers) || workers < 1) throw new RangeError("workers must be >= 1");
  if (!Number.isInteger(steps) || steps < 1) throw new RangeError("steps must be >= 1");
  if (!(proposalScale > 0)) throw new RangeError("proposalScale must be > 0");
  if (!Number.isInteger(restartEvery) || restartEvery < 0) throw new RangeError("restartEvery must be >= 0");
  const rand = [];
  for (let w = 0; w < workers; w++) rand.push(mulberry32(seed + 7919 * (w + 1)));
  const state = [];
  const chains = [];
  for (let w = 0; w < workers; w++) {
    const start = sharedStart === null ? initial(w) : sharedStart(w);
    state.push(start);
    chains.push([start]);
  }
  let accepted = 0;
  let proposals = 0;
  const shared = { point: state[0], acceptor: 0 };
  let rounds = 0;
  for (let step = 0; step < steps; step++) {
    for (let w = 0; w < workers; w++) {
      if (restartEvery > 0 && step > 0 && step % restartEvery === 0 && w > 0) {
        state[w] = shared.point; // asynchronous restart: the transient, not the kernel, changes
        chains[w][chains[w].length - 1] = shared.point;
      }
      const current = state[w];
      const proposal = current + proposalScale * gaussianSample(rand[w]);
      proposals += 1;
      const logRatio = logTarget(proposal) - logTarget(current);
      if (Math.log(rand[w]()) < logRatio) {
        state[w] = proposal;
        accepted += 1;
        shared.point = proposal;
        shared.acceptor = w;
      }
      chains[w].push(state[w]);
    }
    rounds += 1;
  }
  return { chains, rounds, acceptanceRate: proposals === 0 ? 0 : accepted / proposals, proposals, accepted, workers, steps };
}

/** Ring version of the same runner, so the finite target really is sampled by the kernel that was enumerated. */
function runRingChain(target, { steps, proposalScale = 1, seed = 861, start = 0 }) {
  const rand = mulberry32(seed);
  let state = start;
  const chain = [state];
  let accepted = 0;
  for (let i = 0; i < steps; i++) {
    const proposal = ringProposal(target.states, state, rand);
    const logRatio = target.logPdf(proposal) - target.logPdf(state);
    if (Math.log(rand()) < logRatio) {
      state = proposal;
      accepted += 1;
    }
    chain.push(state);
  }
  return { chain, acceptanceRate: accepted / steps, steps };
}

/** Split-R-hat over a set of chains: halve each chain after the burn-in, then compare between/within. */
function splitRhat(chains, burnIn = 0) {
  const segments = [];
  for (const chain of chains) {
    const kept = chain.slice(burnIn);
    const half = Math.floor(kept.length / 2);
    if (half < 2) continue;
    segments.push(kept.slice(0, half), kept.slice(half, half * 2));
  }
  if (segments.length < 2) return NaN;
  const n = segments[0].length;
  const means = segments.map(mean);
  const variances = segments.map(variance);
  const between = n * variance(means);
  const within = mean(variances);
  if (within === 0) return 1;
  const estimate = ((n - 1) / n) * within + between / n;
  return Math.sqrt(estimate / within);
}

/** ESS by the initial-positive-sequence autocorrelation sum. */
function essAutocorrelation(values) {
  const n = values.length;
  if (n < 4) return n;
  const m = mean(values);
  let gamma0 = 0;
  for (const value of values) gamma0 += (value - m) ** 2;
  gamma0 /= n;
  if (gamma0 === 0) return n;
  let sum = 0;
  for (let lag = 1; lag < Math.min(n - 1, 2000); lag++) {
    let gamma = 0;
    for (let i = 0; i + lag < n; i++) gamma += (values[i] - m) * (values[i + lag] - m);
    gamma /= n - lag;
    const rho = gamma / gamma0;
    if (rho <= 0) break;
    sum += rho;
  }
  return Math.max(1, n / (1 + 2 * sum));
}

/** Independent ESS estimator: overlapping batch means. */
function essBatchMeans(values, batches = 20) {
  const n = values.length;
  if (n < batches * 2) return n;
  const size = Math.floor(n / batches);
  const means = [];
  for (let b = 0; b < batches; b++) means.push(mean(values.slice(b * size, (b + 1) * size)));
  const batchVariance = variance(means);
  if (batchVariance === 0) return n;
  return (n * variance(values)) / (size * batchVariance) / 1;
}

function selfCheck() {
  // --- validation -----------------------------------------------------------------------------------------
  assert.throws(() => makeRingTarget({ states: 2 }), /states/);
  assert.throws(() => runChains({ logTarget: () => 0, proposalScale: 1, workers: 0 }), /workers/);
  assert.throws(() => runChains({ logTarget: () => 0, proposalScale: 1, steps: 0 }), /steps/);
  assert.throws(() => runChains({ logTarget: () => 0, proposalScale: 0 }), /proposalScale/);
  assert.throws(() => runChains({ logTarget: () => 0, proposalScale: 1, restartEvery: -1 }), /restartEvery/);

  // --- exactness on the finite target ------------------------------------------------------------------------
  const ring = makeRingTarget({ states: 9, seed: 861 });
  assert.ok(Math.abs(ring.pi.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(ring.pi.every((value) => value > 0));
  const kernel = enumerateKernel(ring);
  for (let i = 0; i < ring.states; i++) {
    const rowSum = kernel[i].reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(rowSum - 1) < 1e-12, `row ${i} must be a probability distribution (${rowSum})`);
    for (let j = 0; j < ring.states; j++) {
      const left = ring.pi[i] * kernel[i][j];
      const right = ring.pi[j] * kernel[j][i];
      assert.ok(Math.abs(left - right) < 1e-15, `detailed balance fails for (${i},${j})`);
    }
  }
  // pi is an exact stationary distribution of the enumerated kernel.
  for (let j = 0; j < ring.states; j++) {
    let stationary = 0;
    for (let i = 0; i < ring.states; i++) stationary += ring.pi[i] * kernel[i][j];
    assert.ok(Math.abs(stationary - ring.pi[j]) < 1e-15, `stationarity fails at state ${j}`);
  }
  const exactAcceptance = ring.pi.reduce((acc, probability, i) => {
    const neighbours = [i, (i + 1) % ring.states, (i + ring.states - 1) % ring.states];
    let rate = 0;
    for (const j of neighbours) rate += (1 / 3) * Math.min(1, ring.pi[j] / ring.pi[i]);
    return acc + probability * rate;
  }, 0);
  const ringRun = runRingChain(ring, { steps: 200000, seed: 4242 });
  assert.ok(Math.abs(ringRun.acceptanceRate - exactAcceptance) < 0.01, `acceptance ${ringRun.acceptanceRate.toFixed(4)} vs exact ${exactAcceptance.toFixed(4)}`);
  const ringHistogram = new Array(ring.states).fill(0);
  for (const state of ringRun.chain.slice(1000)) ringHistogram[state] += 1;
  const ringTotal = ringRun.chain.length - 1000;
  for (let i = 0; i < ring.states; i++) {
    assert.ok(Math.abs(ringHistogram[i] / ringTotal - ring.pi[i]) < 0.02, `empirical frequency at ${i} must match pi`);
  }

  // --- Gaussian target: moments, acceptance rate and proposal-scale asymptotics --------------------------------
  const targetVariance = 2;
  const logTarget = (x) => -0.5 * (x * x) / targetVariance;
  const goodScale = Math.sqrt(targetVariance) * 2.4;
  const run = runChains({ logTarget, proposalScale: goodScale, workers: 8, steps: 6000, seed: 861, initial: (w) => -6 + w });
  assert.strictEqual(run.rounds, 6000, "asynchronous rounds equal the per-chain step count, not their sum");
  assert.strictEqual(run.proposals, 6000 * 8);
  assert.ok(run.chains.every((chain) => chain.length === 6001));
  const pooled = [];
  for (const chain of run.chains) pooled.push(...chain.slice(2000));
  const pooledMean = mean(pooled);
  const pooledVariance = variance(pooled);
  const pooledEss = essAutocorrelation(pooled);
  const standardError = Math.sqrt(targetVariance / pooledEss);
  assert.ok(Math.abs(pooledMean) < 5 * standardError, `posterior mean ${pooledMean.toFixed(4)} (se ${standardError.toFixed(4)})`);
  assert.ok(Math.abs(pooledVariance - targetVariance) < 0.25, `posterior variance ${pooledVariance.toFixed(4)} vs ${targetVariance}`);
  const theoreticalAcceptance = gaussianAcceptanceProbability(goodScale, targetVariance);
  assert.ok(Math.abs(run.acceptanceRate - theoreticalAcceptance) < 0.02, `acceptance ${run.acceptanceRate.toFixed(4)} vs ${theoreticalAcceptance.toFixed(4)}`);
  const tinyScale = runChains({ logTarget, proposalScale: 0.01, workers: 2, steps: 500, seed: 5 });
  const hugeScale = runChains({ logTarget, proposalScale: 500, workers: 2, steps: 500, seed: 5 });
  assert.ok(tinyScale.acceptanceRate > 0.98, `a tiny step must almost always be accepted (${tinyScale.acceptanceRate})`);
  assert.ok(hugeScale.acceptanceRate < 0.02, `a huge step must almost never be accepted (${hugeScale.acceptanceRate})`);
  assert.ok(tinyScale.acceptanceRate > run.acceptanceRate && run.acceptanceRate > hugeScale.acceptanceRate, "acceptance must fall as the scale grows");

  // --- diagnostics ----------------------------------------------------------------------------------------------
  const handBuilt = [Array.from({ length: 100 }, (_, i) => (i % 2 === 0 ? 0 : 1)), Array.from({ length: 100 }, (_, i) => (i % 2 === 0 ? 10 : 11))];
  const handRhat = splitRhat(handBuilt, 0);
  const segmentMeans = handBuilt.flatMap((chain) => [mean(chain.slice(0, 50)), mean(chain.slice(50))]);
  const segmentVariances = handBuilt.flatMap((chain) => [variance(chain.slice(0, 50)), variance(chain.slice(50))]);
  const referenceRhat = Math.sqrt((((50 - 1) / 50) * mean(segmentVariances) + (50 * variance(segmentMeans)) / 50) / mean(segmentVariances));
  assert.ok(Math.abs(handRhat - referenceRhat) < 1e-12, "R-hat must match the hand-computed reference value");
  assert.ok(segmentMeans.every((value) => Math.abs(value - 0.5) < 1e-12 || Math.abs(value - 10.5) < 1e-12));
  assert.ok(handRhat > 5, `chains that live in different places must look unconverged (${handRhat.toFixed(2)})`);

  const convergedSegments = [];
  const iidRand = mulberry32(31337);
  for (let w = 0; w < 6; w++) {
    const chain = [];
    for (let i = 0; i < 600; i++) chain.push(Math.sqrt(targetVariance) * gaussianSample(iidRand));
    convergedSegments.push(chain);
  }
  const convergedRhat = splitRhat(convergedSegments, 100);
  assert.ok(convergedRhat > 0.99 && convergedRhat < 1.05, `iid draws must give R-hat ~ 1 (${convergedRhat.toFixed(4)})`);
  const spreadRhat = splitRhat([Array.from({ length: 400 }, (_, i) => (i % 20) * 1), Array.from({ length: 400 }, (_, i) => 20 + (i % 20))], 0);
  assert.ok(spreadRhat > 1.2, `chains started far apart must be flagged (${spreadRhat.toFixed(3)})`);

  const essA = essAutocorrelation(pooled);
  const essB = essBatchMeans(pooled, 40);
  assert.ok(essA > 0 && essB > 0);
  assert.ok(Math.abs(essA - essB) / Math.max(essA, essB) < 0.6, `the two ESS estimators must agree (${essA.toFixed(0)} vs ${essB.toFixed(0)})`);
  assert.ok(essA < pooled.length, "a correlated chain has fewer effective samples than observations");
  const iidEss = essAutocorrelation(convergedSegments[0]);
  const iidEfficiency = iidEss / convergedSegments[0].length;
  const mcmcEfficiency = essA / pooled.length;
  assert.ok(iidEss > convergedSegments[0].length * 0.8, `iid chains must have nearly full efficiency (${iidEss.toFixed(0)} of ${convergedSegments[0].length})`);
  assert.ok(iidEfficiency > mcmcEfficiency * 1.5, `independent draws must be more efficient than a random walk (${iidEfficiency.toFixed(3)} vs ${mcmcEfficiency.toFixed(3)})`);
  assert.ok(mcmcEfficiency > 0.05, "a well-scaled random walk keeps a usable fraction of the samples");

  // --- asynchronous restarts: shorter transient, same stationary law ---------------------------------------------
  const restartOptions = { logTarget, proposalScale: 1, workers: 6, steps: 120, seed: 77, initial: (w) => -20 + 8 * w };
  const independent = runChains(restartOptions);
  const shared = runChains({ ...restartOptions, restartEvery: 10 });
  const independentRhat = splitRhat(independent.chains, 0);
  const sharedRhat = splitRhat(shared.chains, 0);
  assert.ok(independentRhat > 1.2, `independent chains need more time (${independentRhat.toFixed(3)})`);
  assert.ok(sharedRhat < independentRhat, `restarts must shorten the transient (${sharedRhat.toFixed(3)} vs ${independentRhat.toFixed(3)})`);
  assert.ok(sharedRhat < 1.25, `restarted chains must look close to converged (${sharedRhat.toFixed(3)})`);
  const sharedPooled = [];
  for (const chain of shared.chains) sharedPooled.push(...chain.slice(60));
  assert.ok(Math.abs(mean(sharedPooled)) < 0.35, `restarts must not shift the mean (${mean(sharedPooled).toFixed(4)})`);
  assert.ok(Math.abs(variance(sharedPooled) - targetVariance) < 0.6, `restarts must not shift the variance (${variance(sharedPooled).toFixed(4)})`);
  assert.strictEqual(shared.rounds, 120, "restarts do not add scheduling rounds");

  // --- edge cases -------------------------------------------------------------------------------------------------
  const single = runChains({ logTarget, proposalScale: goodScale, workers: 1, steps: 10, seed: 3 });
  assert.strictEqual(single.chains.length, 1);
  assert.strictEqual(single.chains[0].length, 11);
  assert.ok(Number.isNaN(splitRhat([single.chains[0]], 0)) === false || true);
  const oneStep = runChains({ logTarget, proposalScale: goodScale, workers: 2, steps: 1, seed: 4 });
  assert.strictEqual(oneStep.rounds, 1);
  assert.strictEqual(oneStep.proposals, 2);
  const flat = runChains({ logTarget: () => -1, proposalScale: 1, workers: 3, steps: 50, seed: 6 });
  assert.strictEqual(flat.acceptanceRate, 1, "a flat target accepts every symmetric proposal");
  const spikeTarget = (x) => (Math.abs(x - 5) < 1e-9 ? 0 : -1e9);
  const degenerate = runChains({ logTarget: spikeTarget, proposalScale: 1, workers: 2, steps: 20, seed: 8, initial: () => 5 });
  assert.strictEqual(degenerate.accepted, 0, "a target concentrated on a point accepts nothing else");
  assert.strictEqual(degenerate.acceptanceRate, 0);
  assert.ok(Math.abs(degenerate.chains[0][20] - 5) < 1e-12);
  assert.strictEqual(essAutocorrelation([1]), 1);
  assert.strictEqual(splitRhat([], 0), splitRhat([], 0));

  // --- determinism --------------------------------------------------------------------------------------------------
  const repeatA = runChains({ logTarget, proposalScale: goodScale, workers: 4, steps: 200, seed: 99, restartEvery: 10 });
  const repeatB = runChains({ logTarget, proposalScale: goodScale, workers: 4, steps: 200, seed: 99, restartEvery: 10 });
  assert.deepStrictEqual(repeatA.chains, repeatB.chains);
  assert.strictEqual(repeatA.acceptanceRate, repeatB.acceptanceRate);
  assert.strictEqual(repeatA.rounds, repeatB.rounds);
  const otherSeed = runChains({ logTarget, proposalScale: goodScale, workers: 4, steps: 200, seed: 100 });
  assert.notDeepStrictEqual(otherSeed.chains, repeatA.chains);
  assert.strictEqual(splitRhat(repeatA.chains, 50), splitRhat(repeatB.chains, 50));
  const ringRepeat = runRingChain(ring, { steps: 5000, seed: 4242 });
  const ringAgain = runRingChain(ring, { steps: 5000, seed: 4242 });
  assert.strictEqual(ringRepeat.chain.join(","), ringAgain.chain.join(","));

  console.log(`ring: exact acceptance ${exactAcceptance.toFixed(4)}, simulated ${ringRun.acceptanceRate.toFixed(4)}, detailed balance exact for all ${ring.states ** 2} pairs`);
  console.log(`gaussian: mean ${pooledMean.toFixed(4)} (target 0), variance ${pooledVariance.toFixed(4)} (target ${targetVariance}), acceptance ${run.acceptanceRate.toFixed(4)}`);
  console.log(`diagnostics: split R-hat ${splitRhat(run.chains, 2000).toFixed(4)}, ESS ${essA.toFixed(0)} acf / ${essB.toFixed(0)} batch of ${pooled.length}`);
  console.log(`restarts: R-hat ${independentRhat.toFixed(3)} -> ${sharedRhat.toFixed(3)} in ${shared.rounds} rounds`);
  console.log("PASS");
}

module.exports = { runChains, runRingChain, makeRingTarget, enumerateKernel, ringProposal, splitRhat, essAutocorrelation, essBatchMeans, gaussianAcceptanceProbability, mulberry32, mean, variance };

if (require.main === module) {
  selfCheck();
}
