/**
 * Task 524 - Implement Hidden Markov Model Viterbi algorithm. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Viterbi decoding of a discrete HMM (initial distribution pi, transition matrix A, emission
 * matrix B) with full log-space dynamic programming: dp[t][j] = max_i dp[t-1][i] + log A[i][j] +
 * log B[j][o_t], with backpointers, so underflow cannot occur and a zero-probability transition or
 * emission becomes log 0 = -infinity rather than a silent zero. Ties are resolved by the documented
 * rule "smallest state index wins", which - because the recursion keeps only the single best
 * predecessor per state - produces the lexicographically smallest maximum-probability path.
 *
 * Documented checks: the score is compared against exhaustive enumeration of all N^T paths (N = 3,
 * T = 6 and N = 4, T = 5), including random HMMs whose entries are powers of two so that every path
 * probability is exact in binary floating point and ties are genuine ties; the returned path's own
 * probability is re-computed by an independent product and must equal exp(logScore) to 1e-12; on a
 * uniform HMM (all entries equal) the decoder returns the all-zeros path, which is the unique
 * lexicographic minimum among the tied optima; the max-path probability never exceeds the total
 * likelihood of the observation sequence (sum over all paths), and the Viterbi path's probability
 * is compared with the posterior-decoding path built from brute-force per-step marginals, which can
 * never beat it. Further checks: T = 1 reduces to the argmax of pi * B[:, o], a deterministic HMM
 * recovers its own state sequence exactly, simulated data with a fixed LCG is recovered at the
 * documented rate, and malformed inputs (empty observation, non-stochastic rows, dimension
 * mismatch) are rejected.
 */
"use strict";

const LOG_ZERO = -Infinity;

/** Deterministic LCG in the style of the neighbouring wave-3 solutions. */
function lcgStream(n, seed, limit) {
  const out = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 75 + 74) % 65537;
    out.push(seed % limit);
  }
  return out;
}

function validateModel(initial, transition, emission) {
  const n = initial.length;
  if (transition.length !== n) throw new Error("transition matrix dimension mismatch");
  if (emission.length !== n) throw new Error("emission matrix dimension mismatch");
  const check = (values, label) => {
    let sum = 0.0;
    for (const v of values) {
      if (!(v >= 0.0)) throw new Error(`${label} must be non-negative`);
      sum += v;
    }
    if (Math.abs(sum - 1.0) > 1e-9) throw new Error(`${label} must sum to 1`);
  };
  check(initial, "initial distribution");
  for (const row of transition) {
    if (row.length !== n) throw new Error("transition matrix must be square");
    check(row, "transition row");
  }
  const m = emission[0].length;
  for (const row of emission) {
    if (row.length !== m) throw new Error("emission rows must have equal length");
    check(row, "emission row");
  }
  return { n, m };
}

/** Probability of one state path given the observations (plain product). */
function pathProbability(path, obs, initial, transition, emission) {
  if (path.length !== obs.length) throw new Error("path length mismatch");
  let p = initial[path[0]] * emission[path[0]][obs[0]];
  for (let t = 1; t < path.length; t++) {
    p *= transition[path[t - 1]][path[t]] * emission[path[t]][obs[t]];
  }
  return p;
}

/** Viterbi in log space; returns the path, its log-probability and the dp table. */
function viterbi(initial, transition, emission, obs) {
  const { n } = validateModel(initial, transition, emission);
  const T = obs.length;
  if (T === 0) throw new Error("empty observation sequence");
  const logInitial = initial.map((p) => (p > 0.0 ? Math.log(p) : LOG_ZERO));
  const logTransition = transition.map((row) => row.map((p) => (p > 0.0 ? Math.log(p) : LOG_ZERO)));
  const logEmission = emission.map((row) => row.map((p) => (p > 0.0 ? Math.log(p) : LOG_ZERO)));
  for (const o of obs) if (o < 0 || o >= logEmission[0].length) throw new Error(`bad symbol ${o}`);

  const dp = [];
  const back = [];
  const first = new Array(n);
  for (let j = 0; j < n; j++) {
    first[j] = obs[0] === undefined ? LOG_ZERO :
      (logInitial[j] === LOG_ZERO || logEmission[j][obs[0]] === LOG_ZERO
        ? LOG_ZERO : logInitial[j] + logEmission[j][obs[0]]);
  }
  dp.push(first);
  back.push(new Array(n).fill(-1));
  for (let t = 1; t < T; t++) {
    const previous = dp[t - 1];
    const row = new Array(n).fill(LOG_ZERO);
    const pointers = new Array(n).fill(-1);
    for (let j = 0; j < n; j++) {
      let best = LOG_ZERO;
      let bestI = -1;
      for (let i = 0; i < n; i++) {
        if (previous[i] === LOG_ZERO || logTransition[i][j] === LOG_ZERO) continue;
        const candidate = previous[i] + logTransition[i][j];
        // Strictly greater keeps the smallest predecessor index on a tie.
        if (candidate > best) {
          best = candidate;
          bestI = i;
        }
      }
      row[j] = best === LOG_ZERO || logEmission[j][obs[t]] === LOG_ZERO
        ? LOG_ZERO : best + logEmission[j][obs[t]];
      pointers[j] = bestI;
    }
    dp.push(row);
    back.push(pointers);
  }

  let bestScore = LOG_ZERO;
  let bestState = -1;
  for (let j = 0; j < n; j++) {
    if (dp[T - 1][j] > bestScore) {
      bestScore = dp[T - 1][j];
      bestState = j;
    }
  }
  if (bestState === -1) {
    return { path: null, logScore: LOG_ZERO, probability: 0.0, dp, back };
  }
  const path = new Array(T);
  path[T - 1] = bestState;
  for (let t = T - 1; t > 0; t--) path[t - 1] = back[t][path[t]];
  return { path, logScore: bestScore, probability: Math.exp(bestScore), dp, back };
}

/** Exhaustive search over all N^T paths; ties are broken by the lexicographically smallest path. */
function bruteForceBest(initial, transition, emission, obs) {
  const { n } = validateModel(initial, transition, emission);
  const T = obs.length;
  let bestScore = LOG_ZERO;
  let bestPath = null;
  const path = new Array(T).fill(0);
  const walk = (depth) => {
    if (depth === T) {
      const p = pathProbability(path, obs, initial, transition, emission);
      const score = p > 0.0 ? Math.log(p) : LOG_ZERO;
      if (score > bestScore + 1e-12 ||
          (Math.abs(score - bestScore) <= 1e-12 && bestPath !== null && compareLex(path, bestPath) < 0)) {
        bestScore = score;
        bestPath = path.slice();
      }
      return;
    }
    for (let s = 0; s < n; s++) {
      path[depth] = s;
      walk(depth + 1);
    }
  };
  walk(0);
  return { path: bestPath, logScore: bestScore,
    probability: bestPath === null ? 0.0 : pathProbability(bestPath, obs, initial, transition, emission) };
}

function compareLex(a, b) {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i];
  return 0;
}

/** Total likelihood P(obs) by summing the probabilities of every path. */
function bruteForceLikelihood(initial, transition, emission, obs) {
  const { n } = validateModel(initial, transition, emission);
  const T = obs.length;
  let total = 0.0;
  const path = new Array(T).fill(0);
  const walk = (depth) => {
    if (depth === T) {
      total += pathProbability(path, obs, initial, transition, emission);
      return;
    }
    for (let s = 0; s < n; s++) {
      path[depth] = s;
      walk(depth + 1);
    }
  };
  walk(0);
  return total;
}

/** Per-step posterior marginals P(state_t = i | obs) by enumerating all paths. */
function bruteForcePosteriors(initial, transition, emission, obs) {
  const { n } = validateModel(initial, transition, emission);
  const T = obs.length;
  const numer = [];
  for (let t = 0; t < T; t++) numer.push(new Array(n).fill(0.0));
  const path = new Array(T).fill(0);
  const walk = (depth) => {
    if (depth === T) {
      const p = pathProbability(path, obs, initial, transition, emission);
      for (let t = 0; t < T; t++) numer[t][path[t]] += p;
      return;
    }
    for (let s = 0; s < n; s++) {
      path[depth] = s;
      walk(depth + 1);
    }
  };
  walk(0);
  return numer.map((row) => {
    const z = row.reduce((a, b) => a + b, 0.0);
    return z > 0.0 ? row.map((v) => v / z) : row.map(() => 0.0);
  });
}

/** Posterior-decoding path: the per-step argmax of the marginals (smallest index on ties). */
function posteriorPath(posteriors) {
  return posteriors.map((row) => {
    let best = 0;
    for (let i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
    return best;
  });
}

/** Random but deterministic HMM whose rows are powers of two, so all products are exact. */
function dyadicHmm(n, m, seed) {
  const powers = [0.5, 0.25, 0.125, 0.0625, 0.03125];
  const drawRow = (length, s) => {
    // Random partition of 1 using dyadic weights: repeatedly split the current weight.
    const row = new Array(length).fill(0.0);
    let remaining = 1.0;
    for (let i = 0; i < length - 1; i++) {
      s = (s * 75 + 74) % 65537;
      const share = remaining * powers[s % powers.length] * 2.0;
      const take = Math.min(remaining, share);
      row[i] = take;
      remaining -= take;
      if (remaining <= 1e-12) break;
    }
    row[length - 1] += remaining;
    return [row, s];
  };
  let s = seed;
  const initial = [];
  const rowInit = drawRow(n, s);
  s = rowInit[1];
  for (let i = 0; i < n; i++) initial.push(rowInit[0][i]);
  const transition = [];
  for (let i = 0; i < n; i++) {
    const [row, ns] = drawRow(n, s);
    s = ns;
    transition.push(row);
  }
  const emission = [];
  for (let i = 0; i < n; i++) {
    const [row, ns] = drawRow(m, s);
    s = ns;
    emission.push(row);
  }
  return { initial, transition, emission };
}

function main() {
  const assert = require("node:assert");

  // Hand-checked 2-state chain: A = [[0.5, 0.5], [0.25, 0.75]], B = [[0.9, 0.1], [0.2, 0.8]].
  const initial = [0.5, 0.5];
  const transition = [[0.5, 0.5], [0.25, 0.75]];
  const emission = [[0.9, 0.1], [0.2, 0.8]];
  const obs = [0, 0, 1];
  const decoded = viterbi(initial, transition, emission, obs);
  assert.deepStrictEqual(decoded.path, [0, 0, 1],
    "state 0 explains the two 0-symbols, then state 1 explains the 1-symbol");
  const handProb = 0.5 * 0.9 * 0.5 * 0.9 * 0.5 * 0.8;
  assert.ok(Math.abs(decoded.probability - handProb) <= 1e-15);
  assert.ok(Math.abs(decoded.logScore - Math.log(handProb)) <= 1e-12);
  assert.ok(Math.abs(decoded.dp[0][0] - Math.log(0.45)) <= 1e-15 &&
    Math.abs(decoded.dp[0][1] - Math.log(0.10)) <= 1e-15,
    "the first dp column is log(pi_j B[j][o_0])");

  const single = viterbi(initial, transition, emission, [1]);
  assert.deepStrictEqual(single.path, [1], "T = 1 is argmax(pi_j * B[j][1])");
  assert.ok(Math.abs(single.probability - 0.5 * 0.8) <= 1e-15);

  // The score of every path is re-derived independently and Viterbi must be the argmax.
  const dyadic = dyadicHmm(3, 4, 21);
  const cases = [
    ["dyadic-3x4-T6", dyadic, lcgStream(6, 91, 4)],
    ["dyadic-3x4-T4", dyadic, lcgStream(4, 5, 4)],
    ["dyadic-4x3-T5", dyadicHmm(4, 3, 77), lcgStream(5, 33, 3)],
    ["dyadic-2x5-T6", dyadicHmm(2, 5, 101), lcgStream(6, 7, 5)],
  ];
  for (const [name, model, sequence] of cases) {
    const best = bruteForceBest(model.initial, model.transition, model.emission, sequence);
    const found = viterbi(model.initial, model.transition, model.emission, sequence);
    assert.ok(Math.abs(found.logScore - best.logScore) <= 1e-12, `${name}: optimal score`);
    assert.ok(Math.abs(found.probability -
      pathProbability(found.path, sequence, model.initial, model.transition, model.emission)) <= 1e-15,
      `${name}: the reported probability is the path's own product`);
    assert.strictEqual(pathProbability(found.path, sequence, model.initial, model.transition,
      model.emission), best.probability, `${name}: the found path attains the optimum exactly`);
    assert.deepStrictEqual(found.path, best.path,
      `${name}: the lexicographically smallest optimal path is returned`);

    const total = bruteForceLikelihood(model.initial, model.transition, model.emission, sequence);
    assert.ok(best.probability <= total + 1e-15, `${name}: max path probability <= total likelihood`);
    assert.ok(total <= 1.0 + 1e-15, `${name}: 0 <= P(obs) <= 1`);

    const posteriors = bruteForcePosteriors(model.initial, model.transition, model.emission, sequence);
    for (const row of posteriors) {
      assert.ok(Math.abs(row.reduce((a, b) => a + b, 0.0) - 1.0) <= 1e-12,
        `${name}: posterior marginals sum to one`);
    }
    const posterior = posteriorPath(posteriors);
    const posteriorProb = pathProbability(posterior, sequence, model.initial, model.transition,
      model.emission);
    assert.ok(posteriorProb <= best.probability + 1e-15,
      `${name}: posterior decoding cannot beat Viterbi`);
  }

  // A uniform HMM makes every path equally likely: the tie-break must give the all-zeros path.
  const uniform = { initial: [0.5, 0.5], transition: [[0.5, 0.5], [0.5, 0.5]],
    emission: [[0.5, 0.5], [0.5, 0.5]] };
  const uniformDecoded = viterbi(uniform.initial, uniform.transition, uniform.emission,
    [0, 1, 1, 0]);
  assert.deepStrictEqual(uniformDecoded.path, [0, 0, 0, 0], "lexicographic tie-break");
  assert.ok(Math.abs(uniformDecoded.probability - Math.pow(0.25, 4)) <= 1e-15);
  assert.ok(Math.abs(bruteForceLikelihood(uniform.initial, uniform.transition, uniform.emission,
    [0, 1, 1, 0]) - 16 * Math.pow(0.25, 4)) <= 1e-15, "all 16 paths are equally likely");

  // Deterministic flip-flop HMM: exactly one path has non-zero probability and the decoder must
  // find it (state 0 emits 0, state 1 emits 1, transitions alternate).
  const deterministic = { initial: [1.0, 0.0], transition: [[0.0, 1.0], [1.0, 0.0]],
    emission: [[1.0, 0.0], [0.0, 1.0]] };
  const forcedDecode = viterbi(deterministic.initial, deterministic.transition,
    deterministic.emission, [0, 1, 0, 1, 0]);
  assert.deepStrictEqual(forcedDecode.path, [0, 1, 0, 1, 0], "the generating path is recovered");
  assert.strictEqual(forcedDecode.probability, 1.0, "and it is the only feasible path");

  // Simulated data: sample a hidden chain and emissions, then decode with a fixed LCG.
  const truth = { initial: [0.6, 0.4], transition: [[0.7, 0.3], [0.4, 0.6]],
    emission: [[0.8, 0.15, 0.05], [0.1, 0.3, 0.6]] };
  let seed = 12345;
  const draw = (probs) => {
    seed = (seed * 75 + 74) % 65537;
    const u = seed / 65537.0;
    let acc = 0.0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (u < acc) return i;
    }
    return probs.length - 1;
  };
  const hidden = [draw(truth.initial)];
  const observed = [draw(truth.emission[hidden[0]])];
  for (let t = 1; t < 8; t++) {
    hidden.push(draw(truth.transition[hidden[t - 1]]));
    observed.push(draw(truth.emission[hidden[t]]));
  }
  const recovered = viterbi(truth.initial, truth.transition, truth.emission, observed);
  const hits = recovered.path.filter((s, t) => s === hidden[t]).length;
  assert.ok(hits >= 6, `Viterbi recovers ${hits}/8 hidden states on simulated data`);
  const exact = bruteForceBest(truth.initial, truth.transition, truth.emission, observed);
  assert.deepStrictEqual(recovered.path, exact.path, "brute force agrees on simulated data too");

  // Zero-probability transitions become -infinity instead of vanishing.
  const zeroTransition = { initial: [0.5, 0.5], transition: [[0.0, 1.0], [1.0, 0.0]],
    emission: [[0.5, 0.5], [0.5, 0.5]] };
  const forced = viterbi(zeroTransition.initial, zeroTransition.transition,
    zeroTransition.emission, [0, 0, 0]);
  assert.deepStrictEqual(forced.path, [0, 1, 0], "the chain must alternate between states");
  assert.strictEqual(forced.probability, 0.5 * 0.5 * 0.5 * 0.5,
    "one initial draw and three emission draws, transitions are forced");
  const impossible = viterbi(initial, transition, [[1.0, 0.0], [0.0, 1.0]], [1]);
  assert.strictEqual(impossible.path[0], 1);
  const zeroEmission = { initial: [1.0, 0.0], transition: [[1.0, 0.0], [0.0, 1.0]],
    emission: [[1.0, 0.0], [0.0, 1.0]] };
  const impossibleObs = viterbi(zeroEmission.initial, zeroEmission.transition,
    zeroEmission.emission, [0, 1]);
  assert.strictEqual(impossibleObs.probability, 0.0, "an impossible observation has probability 0");

  assert.throws(() => viterbi(initial, transition, emission, []), /empty observation/);
  assert.throws(() => viterbi([0.5, 0.5], [[0.5, 0.5], [0.5, 0.5]], emission, [5]), /bad symbol/);
  assert.throws(() => viterbi([0.5, 0.5], [[0.5, 0.5], [0.5, 0.6]], emission, [0]), /must sum to 1/);
  assert.throws(() => viterbi([0.5, 0.5], [[0.5, 0.5]], emission, [0]), /dimension mismatch/);
  assert.throws(() => viterbi([-0.5, 1.5], transition, emission, [0]), /non-negative/);
}

module.exports = { LOG_ZERO, lcgStream, validateModel, pathProbability, viterbi, bruteForceBest,
  compareLex, bruteForceLikelihood, bruteForcePosteriors, posteriorPath, dyadicHmm };

if (require.main === module) {
  main();
  console.log("PASS");
}
