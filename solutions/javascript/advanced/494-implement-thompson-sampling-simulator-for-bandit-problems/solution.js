/**
 * Task 494 - Implement Thompson sampling simulator for bandit problems. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Beta-Bernoulli Thompson sampling on a planted three-armed bandit: arm i pays 1 with the known
 * probability arms[i], and every trial samples each arm's success rate from its Beta posterior
 * (prior Beta(1,1)), plays the sampled argmax (ties to the lowest index), observes a Bernoulli
 * reward drawn from the fixed LCG and updates the chosen arm's posterior. Beta(a,b) with integer
 * parameters is exact: two Gamma(a,1)/Gamma(b,1) draws built as sums of a (resp. b) unit
 * exponentials -ln(u), u uniform; that only needs the fixed LCG, no random source. Every draw
 * steps seed = (seed*75 + 74) % 65537 and returns (seed + 0.5)/65537 in (0,1). Documented
 * expectations for arms [0.2, 0.5, 0.8] over 1200 fixed trials: the best arm is picked at least
 * 85% of the time, it is picked more often than each other arm, and the average regret
 * R = sum(max(mu) - mu[a_t]) stays under 100; the posterior mean of the best arm sits within
 * 0.05 of 0.8. Re-running with the same seed reproduces the pull vector exactly.
 */
"use strict";

const ARMS = [0.2, 0.5, 0.8];
const TRIALS = 1200;
const SEED = 42;

class ThompsonSampler {
  constructor(arms, seed) {
    this.arms = arms.slice();
    this.seed = seed;
    this.alpha = this.arms.map(() => 1);            // integer parameters: exact Beta draws
    this.beta = this.arms.map(() => 1);
    this.pulls = this.arms.map(() => 0);
  }

  uniform() {
    this.seed = (this.seed * 75 + 74) % 65537;
    return (this.seed + 0.5) / 65537.0;
  }

  /** Gamma(shape, 1) for integer shape >= 1: a sum of shape unit exponentials. */
  gammaUnit(shape) {
    let total = 0.0;
    for (let i = 0; i < shape; i++) total += -Math.log(this.uniform());
    return total;
  }

  betaSample(a, b) {
    const x = this.gammaUnit(a);
    const y = this.gammaUnit(b);
    return x / (x + y);
  }

  run(trials) {
    const best = Math.max(...this.arms);
    let rewardTotal = 0;
    let regret = 0.0;
    for (let t = 0; t < trials; t++) {
      const samples = this.arms.map((_, i) => this.betaSample(this.alpha[i], this.beta[i]));
      let choice = 0;
      for (let i = 1; i < samples.length; i++) {
        if (samples[i] > samples[choice]) choice = i;
      }
      const reward = this.uniform() < this.arms[choice] ? 1 : 0;
      this.alpha[choice] += reward;
      this.beta[choice] += 1 - reward;
      this.pulls[choice] += 1;
      rewardTotal += reward;
      regret += best - this.arms[choice];
    }
    return [rewardTotal, regret];
  }

  posteriorMean(arm) {
    return this.alpha[arm] / (this.alpha[arm] + this.beta[arm]);
  }
}

function main() {
  const assert = require("node:assert");

  const sampler = new ThompsonSampler(ARMS, SEED);
  const [rewardTotal, regret] = sampler.run(TRIALS);

  assert.strictEqual(sampler.pulls.reduce((a, b) => a + b, 0), TRIALS);
  const bestFrequency = sampler.pulls[2] / TRIALS;
  assert.ok(bestFrequency >= 0.85, "documented: best arm picked >= 85% of the time");
  assert.ok(sampler.pulls[2] > sampler.pulls[1] && sampler.pulls[1] > sampler.pulls[0],
    "arm ordering");
  assert.ok(regret <= 100.0, "documented: average regret bound");
  const posterior = sampler.posteriorMean(2);
  assert.ok(posterior >= 0.75 && posterior <= 0.85, "posterior mean of the best arm");
  assert.ok(Math.abs(posterior - ARMS[2]) <= 0.05);
  assert.ok(rewardTotal >= 0 && rewardTotal <= TRIALS);

  const repeat = new ThompsonSampler(ARMS, SEED);
  repeat.run(TRIALS);
  assert.deepStrictEqual(repeat.pulls, sampler.pulls, "fixed seed must reproduce the pulls");
  assert.strictEqual(repeat.posteriorMean(0), sampler.posteriorMean(0));

  const trivial = new ThompsonSampler([0.9], SEED + 1);
  trivial.run(200);
  assert.deepStrictEqual(trivial.pulls, [200]);

  const duet = new ThompsonSampler([0.3, 0.7], SEED + 2);
  duet.run(600);
  assert.ok(duet.pulls[1] >= 0.8 * 600, "larger gap concentrates faster");

  const tied = new ThompsonSampler([0.5, 0.5], SEED + 3);
  tied.run(600);
  assert.strictEqual(tied.pulls[0] + tied.pulls[1], 600);
  assert.ok(Math.min(...tied.pulls) >= 100, "equal arms must both keep being explored");
}

module.exports = { ARMS, TRIALS, SEED, ThompsonSampler };

if (require.main === module) {
  main();
  console.log("PASS");
}
