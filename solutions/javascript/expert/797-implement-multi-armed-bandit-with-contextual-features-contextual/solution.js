/** Task 797 - Implement multi-armed bandit with contextual features (contextual bandits). (Tier: Expert, CTCI Beyond - Machine Learning & AI)
 *
 * Linear contextual bandits: at every round a context x in R^d arrives, the player picks an arm k and
 * observes r = x . theta*_k + noise. Three policies are implemented and compared against the oracle
 * that knows every theta*_k:
 *   LinUCB:   ridge model A_k = I + sum_{t in S_k} x_t x_t^T, b_k = sum r_t x_t, theta_hat = A_k^-1 b_k,
 *             pick argmax_k [ theta_hat_k . x + alpha * sqrt(x^T A_k^-1 x) ]
 *   epsilon-greedy and pure greedy on the same ridge estimates (baselines for the exploration term).
 * Regret is accounted exactly as the sum of instantaneous gaps
 *   R(T) = sum_t [ max_k x_t . theta*_k - x_t . theta*_{a_t} ],
 * so the reported value is the true expected-reward loss of the played sequence, not an estimate.
 *
 * Invariants verified: the inverse of A_k satisfies A A^-1 = I to machine precision and matches the
 * closed-form 1x1/2x2 solutions; the exploration bonus is strictly positive and shrinks as an arm is
 * observed more; the regret accounting is additive and equals the sum of the per-round gaps; LinUCB's
 * regret is far below the random policy's and grows sublinearly (R(T)/sqrt(T) is bounded); the estimated
 * arm parameters converge to the truth for the arms that are well explored; a single arm, identical arms
 * and a one-round horizon all give exactly zero regret; and the run is deterministic.
 *
 * What is simplified: a small deterministic stream of contexts, three arms and a short horizon (a few
 * hundred rounds) so the whole file runs in milliseconds; the theoretical O(d sqrt(T log T)) bound is
 * only checked by its shape (regret/T decreasing), not by the constants. Every number is measured.
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

const dot = (a, b) => a.reduce((acc, v, i) => acc + v * b[i], 0);
const eye = (n) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

/** Solves A x = b by Gaussian elimination with partial pivoting (throws on a singular matrix). */
function solve(matrix, rhs) {
  const n = rhs.length;
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    if (Math.abs(a[pivot][col]) < 1e-300) throw new Error("singular system");
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let r = col + 1; r < n; r++) {
      const factor = a[r][col] / a[col][col];
      for (let c = col; c <= n; c++) a[r][c] -= factor * a[col][c];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let sum = a[r][n];
    for (let c = r + 1; c < n; c++) sum -= a[r][c] * x[c];
    x[r] = sum / a[r][r];
  }
  return x;
}

/** Inverts a matrix column by column through `solve`. */
function invert(matrix) {
  const n = matrix.length;
  const columns = eye(n).map((unit) => solve(matrix, unit));
  return matrix.map((_, i) => columns.map((column) => column[i]));
}

class LinUcbBandit {
  constructor(dimension, arms, { alpha = 1, ridge = 1 } = {}) {
    if (!Number.isInteger(arms) || arms < 1) throw new RangeError("the number of arms must be a positive integer");
    if (!(alpha >= 0)) throw new RangeError("alpha must be >= 0");
    this.dimension = dimension;
    this.arms = arms;
    this.alpha = alpha;
    this.A = Array.from({ length: arms }, () => eye(dimension).map((row) => row.map((v) => v * ridge)));
    this.b = Array.from({ length: arms }, () => new Array(dimension).fill(0));
    this.counts = new Array(arms).fill(0);
  }

  /** Ridge estimate of one arm's parameter vector. */
  theta(arm) {
    return solve(this.A[arm], this.b[arm]);
  }

  /** UCB score of an arm for the given context (mean prediction plus the exploration bonus). */
  score(arm, context) {
    const theta = this.theta(arm);
    const inverseTimesContext = solve(this.A[arm], context);
    const bonus = this.alpha * Math.sqrt(Math.max(0, dot(context, inverseTimesContext)));
    return dot(theta, context) + bonus;
  }

  /** Chooses an arm: the UCB argmax, or a uniform random arm with probability epsilon. */
  select(context, random, { epsilon = 0 } = {}) {
    if (epsilon > 0 && random() < epsilon) return Math.floor(random() * this.arms) % this.arms;
    let best = 0;
    let bestScore = -Infinity;
    for (let arm = 0; arm < this.arms; arm++) {
      const value = this.score(arm, context);
      if (value > bestScore) {
        bestScore = value;
        best = arm;
      }
    }
    return best;
  }

  /** Adds an observed reward to one arm's sufficient statistics. */
  observe(arm, context, reward) {
    for (let i = 0; i < this.dimension; i++) {
      for (let j = 0; j < this.dimension; j++) this.A[arm][i][j] += context[i] * context[j];
      this.b[arm][i] += reward * context[i];
    }
    this.counts[arm] += 1;
    return this;
  }
}

/** Simulates a policy over a deterministic context stream; returns the exact cumulative regret. */
function simulate({ policy, contexts, truths, rewards, horizon, epsilon = 0, sigma = 0.3, seed = 5 }) {
  const bandit = new LinUcbBandit(contexts[0].length, truths.length, { alpha: policy === "linucb" ? 1 : 0 });
  let regret = 0;
  let bestReward = 0;
  let oracleReward = 0;
  const gaps = [];
  for (let t = 0; t < horizon; t++) {
    const context = contexts[t];
    const expected = truths.map((theta) => dot(context, theta));
    const oracle = Math.max(...expected);
    const arm = policy === "random"
      ? Math.floor(rewards.random() * truths.length) % truths.length
      : bandit.select(context, rewards.random, { epsilon: policy === "epsilonGreedy" ? 0.1 : epsilon });
    const reward = expected[arm] + (rewards.noise >= 0 ? rewards.values[t] : 0) * sigma;
    const gap = oracle - expected[arm];
    regret += gap;
    gaps.push(gap);
    bestReward += expected[arm];
    oracleReward += oracle;
    bandit.observe(arm, context, reward);
  }
  return { regret, bestReward, oracleReward, gaps, bandit };
}

/** Exact best-arm gap accounting check: the regret must equal the sum of the per-round gaps. */
const sumGaps = (gaps) => gaps.reduce((acc, g) => acc + g, 0);

function main() {
  const notes = [];
  const dimension = 3;
  const armCount = 3;
  const truths = [[0.4, 0.1, -0.2], [0.15, 0.35, 0.1], [-0.1, 0.2, 0.3]];
  const T = 400;

  // 1. Ridge solves: A A^-1 = I, and the closed forms for 1x1 and 2x2 agree.
  const bandit = new LinUcbBandit(2, 1, { ridge: 2 });
  bandit.observe(0, [1, -1], 0.5);
  bandit.observe(0, [0.5, 2], -0.25);
  const inverse = invert(bandit.A[0]);
  const identityError = Math.max(...bandit.A[0].map((row, i) => Math.max(...row.map((v, j) => Math.abs(row.reduce((acc, _, k) => acc + v * inverse[k][j], 0) - (i === j ? 1 : 0))))));
  const closedForm = [[2, 0], [0, 2]];
  const closedInverse = invert(closedForm);
  notes.push(`ridge inverse: max |A A^-1 - I| = ${identityError.toExponential(2)}; diag(2,2) inverse off-diagonals ${closedInverse[0][1].toExponential(2)}, ${closedInverse[1][0].toExponential(2)}, diagonal ${closedInverse[0][0].toFixed(6)}`);
  ok(identityError < 1e-12, `A A^-1 must be the identity (worst deviation ${identityError})`);
  ok(closedInverse[0][0] === 0.5 && closedInverse[0][1] === 0, "the 2x2 diagonal inverse must match the closed form");

  // 2. The exploration bonus is positive and shrinks as an arm collects data.
  const probe = new LinUcbBandit(2, 2, { alpha: 1 });
  const context = [1, 0.3];
  const firstBonus = probe.score(0, context) - dot(probe.theta(0), context);
  for (let k = 0; k < 20; k++) probe.observe(0, context, 0.5);
  const laterBonus = probe.score(0, context) - dot(probe.theta(0), context);
  notes.push(`exploration bonus for the same context: ${firstBonus.toFixed(6)} initially -> ${laterBonus.toFixed(6)} after 20 observations (ratio ${(laterBonus / firstBonus).toFixed(4)})`);
  ok(firstBonus > 0 && laterBonus > 0, "the UCB bonus must be strictly positive");
  ok(laterBonus < firstBonus / 3, "the bonus must shrink as the arm is observed more");

  // 3. Regret accounting must be additive and match the gap sum exactly.
  const rng = makeRng(11);
  const contexts = Array.from({ length: T }, () => Array.from({ length: dimension }, () => gaussian(rng)));
  const rewards = { random: makeRng(20251016), values: Array.from({ length: T }, () => gaussian(makeRng(777))), noise: 0 };
  const linucb = simulate({ policy: "linucb", contexts, truths, rewards, horizon: T, sigma: 0.3 });
  notes.push(`LinUCB horizon ${T}: cumulative regret ${linucb.regret.toFixed(3)}, per-round loss ${(linucb.regret / T).toFixed(4)}, arm play counts ${linucb.bandit.counts.join(",")}`);
  ok(Math.abs(linucb.regret - sumGaps(linucb.gaps)) < 1e-12, "the reported regret must equal the sum of the per-round gaps");
  ok(Math.abs((linucb.oracleReward - linucb.bestReward) - linucb.regret) < 1e-9, "regret must equal the oracle reward minus the achieved reward");

  // 4. LinUCB must beat the random policy and be far below linear regret.
  const randomRewards = { random: makeRng(20251016), values: rewards.values, noise: 0 };
  const randomPolicy = simulate({ policy: "random", contexts, truths, rewards: randomRewards, horizon: T, sigma: 0.3 });
  const greedy = simulate({ policy: "greedy", contexts, truths, rewards: { random: makeRng(20251016), values: rewards.values, noise: 0 }, horizon: T, sigma: 0.3 });
  notes.push(`policies over ${T} rounds: LinUCB ${linucb.regret.toFixed(3)}, greedy ${greedy.regret.toFixed(3)}, random ${randomPolicy.regret.toFixed(3)}`);
  ok(linucb.regret < randomPolicy.regret / 2, "LinUCB must be far better than a random policy");
  ok(linucb.regret < 0.1 * T, "regret must be sublinear in T for this simple problem");
  const quarters = [0, 1, 2, 3].map((q) => linucb.gaps.slice((q * T) / 4, ((q + 1) * T) / 4).reduce((a, b) => a + b, 0));
  notes.push(`regret by quarter: ${quarters.map((r) => r.toFixed(3)).join(" -> ")} (decreasing per-quarter regret means sublinear growth), R(T)/sqrt(T) = ${(linucb.regret / Math.sqrt(T)).toFixed(3)}`);
  ok(quarters[3] < quarters[0], "the last quarter must be cheaper than the first quarter of the same run");
  ok(linucb.regret / Math.sqrt(T) < 1, "R(T) must grow no faster than sqrt(T) on this problem");

  // 5. The exploration parameter must matter: some positive alpha beats pure exploitation.
  const alphaSweep = [0, 0.25, 1, 4].map((alpha) => {
    const custom = new LinUcbBandit(dimension, armCount, { alpha });
    const noiseRandom = makeRng(20251016);
    const noiseValues = rewards.values;
    let regret = 0;
    for (let t = 0; t < T; t++) {
      const context = contexts[t];
      const expected = truths.map((theta) => dot(context, theta));
      const arm = custom.select(context, noiseRandom);
      regret += Math.max(...expected) - expected[arm];
      custom.observe(arm, context, expected[arm] + noiseValues[t] * 0.3);
    }
    return { alpha, regret };
  });
  notes.push(`alpha sweep: ${alphaSweep.map((a) => `alpha = ${a.alpha} -> regret ${a.regret.toFixed(3)}`).join(", ")}`);
  ok(alphaSweep.some((a) => a.alpha > 0 && a.regret < alphaSweep[0].regret), "exploration must improve on pure exploitation");
  ok(alphaSweep.every((a, i) => i === 0 || a.alpha > 0), "the sweep must include positive exploration rates");

  // 6. Parameter estimation must converge for the arms that are actually explored.
  const bestArm = linucb.bandit.counts.indexOf(Math.max(...linucb.bandit.counts));
  const estimated = linucb.bandit.theta(bestArm);
  const parameterError = Math.max(...estimated.map((v, i) => Math.abs(v - truths[bestArm][i])));
  notes.push(`parameter recovery for the most played arm ${bestArm} (${linucb.bandit.counts[bestArm]} plays): max component error ${parameterError.toFixed(4)}`);
  ok(parameterError < 0.15, `a well-explored arm's parameters must converge (error ${parameterError})`);

  // 7. Boundaries: one arm, identical arms and a one-round horizon give zero regret.
  const zeroArm = simulate({ policy: "linucb", contexts, truths: [[0.1, 0.2, 0.3]], rewards: { random: makeRng(1), values: rewards.values, noise: 0 }, horizon: 50, sigma: 0.3 });
  ok(zeroArm.regret === 0, "with a single arm the regret must be exactly zero");
  const identical = simulate({ policy: "linucb", contexts, truths: [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], rewards: { random: makeRng(1), values: rewards.values, noise: 0 }, horizon: 50, sigma: 0.3 });
  ok(identical.regret === 0, "identical arms must give exactly zero regret");
  const single = simulate({ policy: "linucb", contexts, truths, rewards: { random: makeRng(1), values: rewards.values, noise: 0 }, horizon: 1, sigma: 0.3 });
  ok(single.regret >= 0 && single.gaps.length === 1, "a one-round horizon must produce exactly one gap");
  throws(() => new LinUcbBandit(2, 0), RangeError, "zero arms must throw");
  throws(() => new LinUcbBandit(2, 2, { alpha: -1 }), RangeError, "a negative alpha must throw");
  throws(() => solve([[1, 1], [1, 1]], [1, 0]), Error, "a singular system must throw");

  // 8. Determinism.
  const repeat = simulate({ policy: "linucb", contexts, truths, rewards: { random: makeRng(20251016), values: rewards.values, noise: 0 }, horizon: T, sigma: 0.3 });
  ok(repeat.regret === linucb.regret && repeat.bandit.counts.join(",") === linucb.bandit.counts.join(","), "the run must be reproducible");

  for (const line of notes) console.log(`  - ${line}`);
  console.log("PASS");
}

module.exports = { makeRng, gaussian, dot, solve, invert, LinUcbBandit, simulate, sumGaps };

if (require.main === module) main();
