/**
 * Task 568 - Implement submodular maximization greedy guarantee example. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * A generic monotone-submodular maximization harness under a cardinality constraint: the function
 * is a black box f(S) on subsets of a ground set of size n, greedy picks the element with the
 * largest marginal gain f(S + e) - f(S) for k rounds (ties to the smallest index), and the result
 * is compared with the exhaustive optimum over all C(n, k) subsets.  Three concrete monotone
 * submodular families are supplied and tested: weighted coverage
 * (f(S) = sum_{e in union(S)} w_e), modular weights (f(S) = sum_{e in S} w_e), and a
 * facility-location style function (f(S) = sum_j max_{i in S} w[i][j]).
 *
 * The submodularity checker enumerates every pair A subset of B and every x not in B and asserts
 * the diminishing-returns inequality f(A+x) - f(A) >= f(B+x) - f(B), together with monotonicity
 * and non-negativity; it is itself validated by feeding it a deliberately supermodular function
 * (f(S) = (sum of weights)^2), which it must reject.  All randomness is drawn from a documented
 * 64-bit splitmix64 generator rather than the repo's 16-bit LCG, because the searches consume tens
 * of thousands of independent draws (per the wave-3 note about the LCG's short cycles and poor
 * low-bit structure).
 *
 * Documented checks: greedy >= (1 - (1 - 1/k)^k) * OPT >= (1 - 1/e) * OPT on every instance of
 * the shipped 300-search (120 seeds x k = 1..4 over random weighted-coverage instances), where the
 * *worst* observed ratio is 0.9 (seed 54, n = 6, k = 2: greedy 18 vs OPT 20) - above both the
 * 0.75 that the k = 2 tight family attains and the 0.632121 floor, i.e. designed instances beat
 * random ones; the first greedy marginal is at least OPT / k (the first step of the classic
 * proof); greedy is exactly optimal for modular f and whenever k >= n; k = 0 gives value 0; and
 * the greedy marginal gains are non-increasing along its own trajectory.  The (1 - (1 - 1/k)^k)
 * bound is *attained exactly* by a shipped instantiation of the canonical tight family built from
 * k^(k+1) unit atoms: L = k^k, layer t (t < k) holds k groups of g_t - g_(t+1) = g_t / k atoms with
 * g_t = L (1 - 1/k)^(t-1), decoy t covers layer t entirely, and block set i takes group i of every
 * layer plus P = g_k private atoms (so each block set is worth exactly L and the k block sets are
 * disjoint).  Every one of the k rounds is an exact tie between a decoy and all block sets; the
 * documented lowest-index tie-break takes the decoys, giving greedy = L k (1 - (1 - 1/k)^k) against
 * OPT = k L, i.e. greedy / OPT = 1 - (1 - 1/k)^k to 1e-12 for k = 2, 3, 4 (universe sizes
 * k^(k+1) = 8, 81, 1024, all values exact integers).  With the mirrored tie-break the block sets
 * win every tie and greedy reaches OPT, which is exactly why the guarantee must be a worst-case
 * statement.  Values and structures are asserted against the exhaustive optimum.  Finally the
 * (1 - 1/k)^k sequence - which increases towards 1/e from below, so the guarantee decreases towards
 * 1 - 1/e from above - is asserted numerically: the guarantee is 0.75, 0.703704 and 0.683594 at
 * k = 2, 3, 4 and 0.6323046 at k = 1000, always above 1 - 1/e = 0.6321206.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;
const EPS = 1e-9;

/** Documented 64-bit splitmix64 generator (long period, well-mixed low and high bits). */
class Rng {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    const z = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    this.state = z;
    let x = z;
    x = ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    x = ((x ^ (x >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return x ^ (x >> 31n);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

/* ------------------------------------------------------------ function families */

/** Weighted coverage: each element of the universe has a weight, sets cover element lists. */
function coverageFunction(weights, sets) {
  return (S) => {
    const seen = new Set();
    for (const i of S) for (const e of sets[i]) seen.add(e);
    let total = 0.0;
    for (const e of seen) total += weights[e];
    return total;
  };
}

/** Modular: f(S) = sum of the weights of the chosen ground elements. */
function modularFunction(weights) {
  return (S) => {
    let total = 0.0;
    for (const i of S) total += weights[i];
    return total;
  };
}

/** Facility-location style monotone submodular function: f(S) = sum_j max_{i in S} w[i][j]. */
function facilityLocationFunction(w) {
  return (S) => {
    let total = 0.0;
    for (let j = 0; j < w[0].length; j++) {
      let best = 0.0;
      for (const i of S) best = Math.max(best, w[i][j]);
      total += best;
    }
    return total;
  };
}

/** Deliberately supermodular, used only to prove the diminishing-returns checker has teeth. */
function supermodularFunction(weights) {
  return (S) => {
    let total = 0.0;
    for (const i of S) total += weights[i];
    return total * total;
  };
}

/* ------------------------------------------------------------------- algorithms */

function value(f, S) {
  return f(S);
}

/** Greedy: k rounds of the largest marginal gain; ties resolved to the smallest index. */
function greedyMaximize(f, n, k) {
  const chosen = [];
  const gains = [];
  let current = f(chosen);
  for (let step = 0; step < k; step++) {
    let pick = -1;
    let pickGain = -Infinity;
    for (let i = 0; i < n; i++) {
      if (chosen.indexOf(i) >= 0) continue;
      const gain = f(chosen.concat([i])) - current;
      if (gain > pickGain + EPS) {
        pick = i;
        pickGain = gain;
      }
    }
    if (pick < 0) break;
    chosen.push(pick);
    gains.push(pickGain);
    current = f(chosen);
  }
  return { S: chosen, value: current, gains };
}

/** Exhaustive optimum over all C(n, k) subsets. */
function exhaustiveMaximize(f, n, k) {
  if (k < 0 || k > n) throw new Error("0 <= k <= n required");
  let best = -Infinity;
  let bestSet = null;
  let evaluated = 0;
  const current = [];
  const rec = (start) => {
    if (current.length === k) {
      evaluated += 1;
      const v = f(current);
      if (v > best + EPS) {
        best = v;
        bestSet = current.slice();
      }
      return;
    }
    for (let i = start; i < n; i++) {
      current.push(i);
      rec(i + 1);
      current.pop();
    }
  };
  rec(0);
  return { S: bestSet, value: best, evaluated };
}

function binomial(n, k) {
  if (k < 0 || k > n) return 0;
  let result = 1;
  for (let i = 1; i <= k; i++) result = (result * (n - k + i)) / i;
  return Math.round(result);
}

/**
 * Every violation of the diminishing-returns inequality (and of monotonicity) for the ground set
 * {0..n-1}: for all A subset of B and all x not in B, f(A+x) - f(A) must be >= f(B+x) - f(B).
 */
function submodularityViolations(f, n) {
  const asList = (mask) => {
    const list = [];
    for (let i = 0; i < n; i++) if (mask & (1 << i)) list.push(i);
    return list;
  };
  const violations = [];
  for (let B = 0; B < (1 << n); B++) {
    const bList = asList(B);
    const fB = f(bList);
    for (let A = B; ; A = (A - 1) & B) {
      const aList = asList(A);
      const fA = f(aList);
      if (fA < -EPS) violations.push({ kind: "negative", A, value: fA });
      if (fA > fB + EPS) violations.push({ kind: "non-monotone", A, B });
      for (let x = 0; x < n; x++) {
        if (B & (1 << x)) continue;
        const deltaA = f(aList.concat([x])) - fA;
        const deltaB = f(bList.concat([x])) - fB;
        if (deltaA < deltaB - EPS) violations.push({ kind: "diminishing", A, B, x, deltaA, deltaB });
      }
      if (A === 0) break;
    }
  }
  return violations;
}

/**
 * The canonical tight family for the (1 - (1 - 1/k)^k) guarantee, in unweighted-coverage form
 * (every atom is a separate element of weight 1, so all values below are exact integers).
 *
 * Normalise OPT = k * L with L = k^k.  The graded greedy gains are g_t = L (1 - 1/k)^(t-1), so the
 * universe is split into k - 1 "layers": layer t holds k groups of n_t = g_t - g_{t+1} = g_t / k
 * atoms (one group per block set), and each block set additionally owns P = g_k private atoms.
 * Decoy t (t = 1..k-1) covers layer t entirely, so its marginal gain when picked is g_t - exactly
 * the fresh value of every still-uncovered block set at that moment (block i holds its group of
 * each remaining layer plus its P private atoms = g_t).  Every round is therefore a tie between a
 * decoy and all k block sets, and the documented lowest-index tie-break takes the decoys: greedy
 * accumulates g_1 + ... + g_(k-1) and then one block set's private atoms g_k, i.e.
 * L * k (1 - (1 - 1/k)^k), while the k disjoint block sets - each worth exactly L - attain
 * OPT = k L.  Hence greedy / OPT = 1 - (1 - 1/k)^k exactly.  With the mirrored tie-break the three
 * (or k) block sets win every tie and greedy reaches OPT, which is why the guarantee is a
 * worst-case statement.
 */
function tightCoverageInstance(k) {
  if (k < 2) throw new Error("the tight family needs k >= 2");
  const L = Math.pow(k, k);
  const g = (t) => L * Math.pow(1 - 1 / k, t - 1);
  let next = 0;
  const layers = [];
  for (let t = 1; t <= k - 1; t++) {
    const perGroup = Math.round(g(t) - g(t + 1));
    const row = [];
    for (let i = 0; i < k; i++) {
      const list = [];
      for (let a = 0; a < perGroup; a++) list.push(next++);
      row.push(list);
    }
    layers.push(row);
  }
  const privateAtoms = Math.round(g(k));
  const priv = [];
  for (let i = 0; i < k; i++) {
    const list = [];
    for (let a = 0; a < privateAtoms; a++) list.push(next++);
    priv.push(list);
  }
  const sets = [];
  for (let t = 0; t < k - 1; t++) {
    const covered = [];
    for (let i = 0; i < k; i++) covered.push(...layers[t][i]);
    sets.push(covered);
  }
  for (let i = 0; i < k; i++) {
    const covered = [];
    for (let t = 0; t < k - 1; t++) covered.push(...layers[t][i]);
    covered.push(...priv[i]);
    sets.push(covered);
  }
  const weights = new Array(next).fill(1.0);
  return { weights, sets, universeSize: next, perGroupAtoms: layers.map((row) => row[0].length),
    privateAtoms, k, L };
}

/** Independent tie-break experiment: prefer the *last* index among equal gains (block sets). */
function greedyMaximizeLastTieBreak(f, n, k) {
  const chosen = [];
  let current = f(chosen);
  for (let step = 0; step < k; step++) {
    let pick = -1;
    let pickGain = -Infinity;
    for (let i = 0; i < n; i++) {
      if (chosen.indexOf(i) >= 0) continue;
      const gain = f(chosen.concat([i])) - current;
      if (gain >= pickGain - EPS) {
        pick = i;
        pickGain = gain;
      }
    }
    if (pick < 0) break;
    chosen.push(pick);
    current = f(chosen);
  }
  return { S: chosen, value: current };
}

function main() {
  const assert = require("node:assert");

  // (1) Submodularity / monotonicity / non-negativity of the three families, plus a negative
  //     control proving the checker rejects a non-submodular function.
  {
    const rng = new Rng(20240516);
    const n = 7;
    const universe = 9;
    const weights = [];
    for (let e = 0; e < universe; e++) weights.push(0.5 + rng.nextInt(5));
    const sets = [];
    for (let i = 0; i < n; i++) {
      const s = [];
      for (let e = 0; e < universe; e++) if (rng.nextInt(3) === 0) s.push(e);
      sets.push(s);
    }
    const coverage = coverageFunction(weights, sets);
    const modular = modularFunction(weights.slice(0, n));
    const table = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < universe; j++) row.push(0.5 + rng.nextInt(5));
      table.push(row);
    }
    const facility = facilityLocationFunction(table);
    for (const [name, f] of [["coverage", coverage], ["modular", modular],
      ["facility", facility]]) {
      const violations = submodularityViolations(f, n);
      assert.deepStrictEqual(violations, [], `${name} must be monotone and submodular`);
      assert.ok(f([]) >= 0, `${name} is non-negative`);
    }
    assert.strictEqual(coverage([0, 1]) >= coverage([0]) && coverage([0]) >= 0, true);
    const bad = submodularityViolations(supermodularFunction(new Array(n).fill(1.0)), n);
    assert.ok(bad.length > 0, "the checker has teeth: (sum of weights)^2 violates it");
    assert.strictEqual(bad[0].kind, "diminishing");
    assert.ok(bad.some((v) => Math.abs(v.deltaA - 1.0) <= EPS && Math.abs(v.deltaB - 3.0) <= EPS),
      "f(S) = (sum)^2: after {0} the gain of 1 is 3, while from the empty set it is only 1");
  }

  // (2) The (1 - (1-1/k)^k) >= (1 - 1/e) guarantee on random weighted-coverage instances.
  let worstRatio = 1;
  let worstRecord = null;
  let instances = 0;
  for (let seed = 1; seed <= 120; seed++) {
    const rng = new Rng(5000 + seed);
    const n = 4 + rng.nextInt(5);
    const universe = 6 + rng.nextInt(9);
    const weights = [];
    for (let e = 0; e < universe; e++) weights.push(1 + rng.nextInt(4));
    const sets = [];
    for (let i = 0; i < n; i++) {
      const s = [];
      for (let e = 0; e < universe; e++) if (rng.nextInt(2) === 0) s.push(e);
      sets.push(s);
    }
    const f = coverageFunction(weights, sets);
    for (let k = 1; k <= 4; k++) {
      if (k > n) continue;
      const greedy = greedyMaximize(f, n, k);
      const optimal = exhaustiveMaximize(f, n, k);
      const bound = 1 - Math.pow(1 - 1 / k, k);
      instances += 1;
      assert.strictEqual(optimal.evaluated, binomial(n, k));
      assert.ok(greedy.S.length <= k);
      assert.ok(greedy.value <= optimal.value + EPS, "greedy cannot beat the optimum");
      assert.ok(greedy.value >= bound * optimal.value - EPS,
        `seed ${seed} k ${k}: ${greedy.value} vs bound ${bound * optimal.value}`);
      assert.ok(greedy.value >= (1 - 1 / Math.E) * optimal.value - EPS,
        "and certainly the k-independent 1 - 1/e floor");
      // The first step of the classic proof: the best singleton covers at least OPT / k.
      const bestSingleton = Math.max(...[...Array(n).keys()].map((i) => f([i])));
      assert.ok(bestSingleton >= optimal.value / k - EPS,
        `best singleton ${bestSingleton} >= OPT/k = ${optimal.value / k}`);
      assert.ok(Math.abs(greedy.gains[0] - bestSingleton) <= EPS,
        "greedy's first pick is the best singleton");
      if (optimal.value > EPS) {
        const ratio = greedy.value / optimal.value;
        if (ratio < worstRatio) {
          worstRatio = ratio;
          worstRecord = { seed, n, k, ratio, bound, greedy: greedy.value, optimal: optimal.value };
        }
      }
    }
  }
  assert.ok(worstRatio >= 1 - 1 / Math.E,
    `worst ratio ${worstRatio} must respect 1 - 1/e = ${1 - 1 / Math.E}`);
  assert.ok(worstRatio >= 0.85,
    `documented: the worst of ${instances} searches is ${worstRatio} (${JSON.stringify(worstRecord)})`);
  assert.ok(worstRatio >= 0.9 - 1e-12,
    "the shipped search never falls below the documented worst ratio 0.9");
  assert.ok(worstRatio > 1 - Math.pow(1 - 1 / 2, 2),
    "and the random family stays strictly above what the designed k = 2 tight family attains");

  // (3) Modular f: greedy is exactly optimal (it takes the k largest weights).
  {
    const rng = new Rng(999);
    for (let trial = 0; trial < 40; trial++) {
      const n = 5 + rng.nextInt(6);
      const weights = [];
      for (let i = 0; i < n; i++) weights.push(1 + rng.nextInt(20));
      const f = modularFunction(weights);
      for (const k of [1, 2, 3]) {
        const greedy = greedyMaximize(f, n, k);
        const optimal = exhaustiveMaximize(f, n, k);
        assert.ok(Math.abs(greedy.value - optimal.value) <= EPS,
          "greedy is exact on modular functions");
        const sorted = weights.slice().sort((a, b) => b - a);
        let closed = 0;
        for (let i = 0; i < k; i++) closed += sorted[i];
        assert.ok(Math.abs(greedy.value - closed) <= EPS,
          "and equals the sum of the k largest weights (closed form)");
      }
    }
  }

  // (4) Boundary cases: k >= n, k = 0, empty ground set.
  {
    const weights = [3.0, 1.0, 2.0];
    const f = modularFunction(weights);
    assert.strictEqual(greedyMaximize(f, 3, 0).value, 0, "zero picks give value 0");
    assert.strictEqual(greedyMaximize(f, 3, 3).value, 6);
    assert.strictEqual(greedyMaximize(f, 3, 5).value, 6, "k > n stops at the ground set");
    assert.strictEqual(exhaustiveMaximize(f, 3, 3).value, 6);
    assert.deepStrictEqual(exhaustiveMaximize(f, 0, 0).S, []);
    assert.throws(() => exhaustiveMaximize(f, 3, 4), /0 <= k <= n/);
    const coverage = coverageFunction([1.0, 1.0], [[], []]);
    assert.strictEqual(greedyMaximize(coverage, 2, 2).value, 0, "empty sets cover nothing");
    assert.strictEqual(greedyMaximize(coverageFunction([1.0], [[0], [0]]), 2, 1).value, 1,
      "duplicate sets cannot both be useful");
    assert.strictEqual(greedyMaximize(coverageFunction([1.0], [[0], [0]]), 2, 2).value, 1);
  }

  // (5) Facility-location function: the guarantee also holds there, by the same argument.
  {
    const rng = new Rng(31415);
    let checked = 0;
    for (let trial = 0; trial < 25; trial++) {
      const n = 5 + rng.nextInt(4);
      const columns = 5 + rng.nextInt(5);
      const table = [];
      for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < columns; j++) row.push(rng.nextInt(10) / 2.0);
        table.push(row);
      }
      const f = facilityLocationFunction(table);
      const k = 2 + (trial % 3);
      const greedy = greedyMaximize(f, n, k);
      const optimal = exhaustiveMaximize(f, n, k);
      const bound = 1 - Math.pow(1 - 1 / k, k);
      assert.ok(greedy.value >= bound * optimal.value - EPS,
        `facility-location function: ${greedy.value} vs ${bound * optimal.value}`);
      checked += 1;
    }
    assert.strictEqual(checked, 25);
  }

  // (6) The canonical tight family: greedy / OPT = 1 - (1 - 1/k)^k exactly (k = 2, 3, 4).
  {
    const ratios = [];
    for (const k of [2, 3, 4]) {
      const family = tightCoverageInstance(k);
      const { weights, sets } = family;
      const n = sets.length;
      const f = coverageFunction(weights, sets);
      assert.strictEqual(family.universeSize, Math.pow(k, k + 1),
        "the family uses k^(k+1) unit atoms");
      assert.strictEqual(n, 2 * k - 1, "k - 1 decoy sets plus k block sets");
      const greedy = greedyMaximize(f, n, k);
      const optimal = exhaustiveMaximize(f, n, k);
      assert.strictEqual(optimal.evaluated, binomial(2 * k - 1, k));
      const L = family.L;
      const ratio = greedy.value / optimal.value;
      const bound = 1 - Math.pow(1 - 1 / k, k);
      assert.ok(Math.abs(optimal.value - k * L) <= 1e-9,
        "the k disjoint block sets are worth L each, so OPT = k * L");
      for (let i = 0; i < k; i++) {
        assert.strictEqual(f([k - 1 + i]), L, "every block set is worth exactly L");
      }
      assert.ok(Math.abs(greedy.value - bound * k * L) <= 1e-9,
        `k = ${k}: greedy value ${greedy.value} must equal ${bound} * k * L`);
      assert.ok(Math.abs(ratio - bound) <= 1e-12,
        `k = ${k}: greedy/OPT = ${ratio} must equal the bound ${bound}`);
      assert.deepStrictEqual(greedy.S, [...Array(k).keys()],
        "the documented lowest-index tie-break takes the k - 1 decoys and then one block set");
      assert.ok(greedy.value < optimal.value - 1e-9, "so the greedy is strictly suboptimal here");
      // With the opposite tie-break the block sets win every tie and greedy reaches OPT.
      const other = greedyMaximizeLastTieBreak(f, n, k);
      assert.ok(Math.abs(other.value - optimal.value) <= 1e-9,
        "the mirrored tie-break attains OPT, hence the guarantee must be worst-case");
      ratios.push(ratio);
    }
    assert.ok(ratios[0] > ratios[1] && ratios[1] > ratios[2],
      `the tight ratios decrease towards 1 - 1/e: ${ratios.join(", ")}`);
    assert.ok(ratios[2] >= 1 - 1 / Math.E, "and stay above the 1 - 1/e floor");
  }

  // (7) The (1 - 1/k)^k sequence increases towards 1/e from below (so the guarantee
  //     1 - (1 - 1/k)^k decreases towards the 1 - 1/e floor from above).
  {
    let previous = -Infinity;
    for (const k of [2, 3, 4, 5, 10, 100, 1000]) {
      const value = Math.pow(1 - 1 / k, k);
      assert.ok(value > previous, `(1 - 1/k)^k must increase in k (k = ${k})`);
      assert.ok(value <= 1 / Math.E + 1e-15, "and stay below 1/e");
      assert.ok(1 - value >= 1 - 1 / Math.E - 1e-15, "so the guarantee stays above 1 - 1/e");
      previous = value;
    }
    assert.ok(Math.abs(Math.pow(1 - 1 / 2, 2) - 0.25) <= 1e-15);
    assert.ok(Math.abs(1 - Math.pow(1 - 1 / 2, 2) - 0.75) <= 1e-15);
    assert.ok(Math.abs(1 - Math.pow(1 - 1 / 3, 3) - 0.7037037037037037) <= 1e-15);
    assert.ok(Math.abs(1 - Math.pow(1 - 1 / 4, 4) - 0.68359375) <= 1e-15);
    assert.ok(Math.abs(Math.pow(1 - 1 / 1000, 1000) - 0.3676954248) <= 1e-9,
      "at k = 1000 the sequence is 0.3676954 vs 1/e = 0.3678794");
    assert.ok(Math.abs(1 - Math.pow(1 - 1 / 1000, 1000) - 0.6323045752) <= 1e-9,
      "the guarantee at k = 1000 is 0.6323046, just above 1 - 1/e = 0.6321206");
    assert.ok(Math.abs(1 / Math.E - 0.36787944117144233) <= 1e-15);
  }

  // (8) Greedy's own marginal gains are non-increasing (a consequence of submodularity).
  {
    const rng = new Rng(2718);
    for (let trial = 0; trial < 30; trial++) {
      const n = 6 + rng.nextInt(5);
      const universe = 8 + rng.nextInt(6);
      const sets = [];
      for (let i = 0; i < n; i++) {
        const s = [];
        for (let e = 0; e < universe; e++) if (rng.nextInt(2) === 0) s.push(e);
        sets.push(s);
      }
      const f = coverageFunction(new Array(universe).fill(1.0), sets);
      const greedy = greedyMaximize(f, n, 4);
      for (let i = 1; i < greedy.gains.length; i++) {
        assert.ok(greedy.gains[i] <= greedy.gains[i - 1] + EPS,
          "greedy marginal gains must be non-increasing");
      }
      assert.ok(greedy.gains.every((gain) => gain >= -EPS));
      assert.ok(Math.abs(greedy.value - f(greedy.S)) <= EPS, "value matches the chosen set");
    }
  }

  // (9) Determinism.
  {
    const rng = new Rng(1234567);
    const sets = [];
    const universe = 12;
    for (let i = 0; i < 9; i++) {
      const s = [];
      for (let e = 0; e < universe; e++) if (rng.nextInt(3) === 0) s.push(e);
      sets.push(s);
    }
    const f = coverageFunction(new Array(universe).fill(1.0), sets);
    assert.deepStrictEqual(greedyMaximize(f, 9, 3), greedyMaximize(f, 9, 3));
    assert.deepStrictEqual(tightCoverageInstance(3), tightCoverageInstance(3));
    assert.notDeepStrictEqual(tightCoverageInstance(3), tightCoverageInstance(4));
    assert.throws(() => tightCoverageInstance(1), /k >= 2/);
  }
}

module.exports = { MASK64, Rng, coverageFunction, modularFunction, facilityLocationFunction,
  supermodularFunction, value, greedyMaximize, exhaustiveMaximize, binomial,
  submodularityViolations, tightCoverageInstance, greedyMaximizeLastTieBreak };

if (require.main === module) {
  main();
  console.log("PASS");
}
