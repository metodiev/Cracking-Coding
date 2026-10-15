/**
 * Task 495 - Implement UCB (upper confidence bound) algorithm simulation. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * UCB1 on a planted three-armed bandit: each arm is pulled once to initialise its averages
 * (arms in index order), then every trial pulls argmax(mu_hat[i] + sqrt(2 ln t / n_i)), ties to
 * the lowest index, t counting trials so far. Rewards are Bernoulli draws from the fixed LCG
 * (seed = (seed*75 + 74) % 65537, reward when (seed + 0.5)/65537 < p_arm), so the whole run is
 * deterministic. The pseudo-regret uses the known true means, R(T) = sum over trials of
 * (max(mu) - mu[a_t]) with mu = [0.2, 0.5, 0.8] as planted. Documented expectations over 2000
 * fixed trials: the best arm is pulled at least 80% of the time, pulls are strictly ordered
 * best > middle > worst, total regret stays under 100, and the mean per-trial regret of the
 * second half is strictly below that of the first half (the regret rate decreases as UCB1
 * learns). A uniform-random baseline using the same reward stream is used to show UCB1's
 * advantage.
 */
"use strict";

const ARMS = [0.2, 0.5, 0.8];
const TRIALS = 2000;
const SEED = 99;

class UCB1 {
  constructor(arms, seed) {
    this.arms = arms.slice();
    this.seed = seed;
    this.counts = this.arms.map(() => 0);
    this.sums = this.arms.map(() => 0.0);
    this.pulls = this.arms.map(() => 0);
  }

  reward(p) {
    this.seed = (this.seed * 75 + 74) % 65537;
    return (this.seed + 0.5) / 65537.0 < p ? 1 : 0;
  }

  index(arm, t) {
    if (this.counts[arm] === 0) return Infinity;    // untried arms are taken first
    const mean = this.sums[arm] / this.counts[arm];
    return mean + Math.sqrt((2.0 * Math.log(t)) / this.counts[arm]);
  }

  run(trials) {
    const best = Math.max(...this.arms);
    let regret = 0.0;
    const regretTrace = [];
    for (let t = 1; t <= trials; t++) {
      let choice = 0;
      for (let i = 1; i < this.arms.length; i++) {
        if (this.index(i, t) > this.index(choice, t)) choice = i;
      }
      const r = this.reward(this.arms[choice]);
      this.counts[choice] += 1;
      this.sums[choice] += r;
      this.pulls[choice] += 1;
      regret += best - this.arms[choice];
      regretTrace.push(regret);
    }
    return [regret, regretTrace];
  }
}

/** Uniform-random arm choice over the same LCG reward stream (control experiment). */
class RandomBaseline {
  constructor(arms, seed) {
    this.arms = arms.slice();
    this.seed = seed;
  }

  run(trials) {
    const best = Math.max(...this.arms);
    let regret = 0.0;
    for (let t = 0; t < trials; t++) {
      this.seed = (this.seed * 75 + 74) % 65537;
      const choice = this.seed % this.arms.length;
      this.seed = (this.seed * 75 + 74) % 65537;
      regret += best - this.arms[choice];
    }
    return regret;
  }
}

function main() {
  const assert = require("node:assert");

  const ucb = new UCB1(ARMS, SEED);
  const [regret, trace] = ucb.run(TRIALS);

  assert.strictEqual(ucb.pulls.reduce((a, b) => a + b, 0), TRIALS);
  assert.deepStrictEqual(ucb.pulls, ucb.counts);

  const bestFrequency = ucb.pulls[2] / TRIALS;
  assert.ok(bestFrequency >= 0.80, "documented: best arm picked >= 80% of the time");
  assert.ok(ucb.pulls[2] > ucb.pulls[1] && ucb.pulls[1] > ucb.pulls[0], "strict ordering");
  assert.ok(regret <= 100.0, "documented: total pseudo-regret bound");

  for (let i = 1; i < trace.length; i++) {
    assert.ok(trace[i] >= trace[i - 1], "regret is a cumulative sum of non-negative gaps");
  }
  const half = TRIALS >> 1;
  const firstRate = trace[half - 1] / half;
  const secondRate = (trace[trace.length - 1] - trace[half - 1]) / (TRIALS - half);
  assert.ok(secondRate < firstRate, "documented: regret rate decreases");

  const baseline = new RandomBaseline(ARMS, SEED);
  const baselineRegret = baseline.run(TRIALS);
  assert.ok(baselineRegret > 2.0 * regret, "UCB1 must beat uniform sampling");

  const repeat = new UCB1(ARMS, SEED);
  const [repeatRegret] = repeat.run(TRIALS);
  assert.deepStrictEqual(repeat.pulls, ucb.pulls);
  assert.strictEqual(repeatRegret, regret);

  const warm = new UCB1(ARMS, SEED);
  warm.run(3);
  assert.deepStrictEqual(warm.counts, [1, 1, 1], "warm-up visits each arm exactly once");
  warm.run(7);
  assert.ok(Math.min(...warm.counts) >= 1, "no arm is skipped during warm-up");
  assert.strictEqual(warm.counts.reduce((a, b) => a + b, 0), 10);

  const duet = new UCB1([0.3, 0.7], SEED + 1);
  duet.run(1000);
  assert.ok(duet.pulls[1] >= 0.9 * 1000);
  assert.ok(duet.pulls[1] > duet.pulls[0]);
}

module.exports = { ARMS, TRIALS, SEED, UCB1, RandomBaseline };

if (require.main === module) {
  main();
  console.log("PASS");
}
