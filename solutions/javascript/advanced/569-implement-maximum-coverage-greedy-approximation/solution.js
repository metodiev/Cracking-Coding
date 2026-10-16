/**
 * Task 569 - Implement maximum coverage greedy approximation. (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Maximum coverage: given m sets over a universe of size u and a budget k, choose k sets maximising
 * the number of covered elements.  The classical greedy picks at every round the set covering the
 * most *uncovered* elements (ties to the smallest set index) and is a (1 - 1/e)-approximation.  Two
 * independent implementations are written and cross-checked on every instance: one keeps a
 * Uint8Array of covered flags and rescans the sets each round (O(sum of set sizes) per round, so it
 * works for the 1024-element tight family), and one keeps the covered elements in a 32-bit bitmask
 * and evaluates each candidate with a single `popcount(~covered & mask)` (only for u <= 31).
 * Ground truth is exhaustive enumeration of all C(m, k) budgets, plus a closed form for the
 * planned instances.  All randomness comes from a documented 64-bit splitmix64 generator rather
 * than the repo's 16-bit LCG, per the wave-3 note on that LCG's short cycles and poor low-bit
 * structure.
 *
 * Documented checks: greedy is optimal for k = 1 (it picks the largest set), for k >= m (it
 * recovers the full union) and on families of pairwise disjoint sets; OPT <= min(u, k * max |S|)
 * and greedy <= OPT on every instance; on the shipped random family of 300 searches the greedy/OPT
 * ratio never drops below 0.833333 (worst case: seed 37, m = 7, u = 7, k = 2, greedy 5 vs OPT 6)
 * while a shipped instantiation of the canonical tight family attains exactly
 * 1 - (1 - 1/k)^k for k = 2, 3, 4, i.e. 0.75, 0.703704, 0.683594 (there the greedy takes the k - 1
 * "decoy" layer sets on index tie-breaks and is provably suboptimal: 57 of 81 covered at k = 3).
 * The tight family is built from u = k^(k+1) unit atoms with L = k^k, layer t (< k) holding k
 * groups of g_t - g_(t+1) = g_t / k atoms for g_t = L (1 - 1/k)^(t-1), decoy t covering layer t
 * and block set i taking group i of every layer plus g_k private atoms; each block set is worth
 * exactly L and the k block sets are disjoint, so OPT = k L.  Edge cases covered: k = 0, empty sets
 * and empty universe, duplicate sets, repeated elements inside a set, and k > m.  Both greedy
 * implementations are asserted to agree set-for-set, the reported cover is recomputed independently
 * with a Set, and reruns are bit-identical.
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

function popcount32(value) {
  let x = value >>> 0;
  x = x - ((x >>> 1) & 0x55555555);
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  x = (x + (x >>> 4)) & 0x0f0f0f0f;
  return Math.imul(x, 0x01010101) >>> 24;
}

/** Greedy with a covered-flag array; works for large universes. */
function coverageGreedy(rawSets, universeSize, k) {
  // Sets are de-duplicated first: the 32-bit variant is inherently duplicate-free, and without
  // this the flag version would count a repeated element twice and disagree with it.
  const sets = rawSets.map((set) => {
    const unique = new Set(set);
    return unique.size === set.length ? set : [...unique];
  });
  const covered = new Uint8Array(universeSize);
  const chosen = [];
  const gains = [];
  let count = 0;
  for (let step = 0; step < k; step++) {
    let pick = -1;
    let pickGain = 0;
    for (let i = 0; i < sets.length; i++) {
      if (chosen.indexOf(i) >= 0) continue;
      let fresh = 0;
      for (const e of sets[i]) if (!covered[e]) fresh += 1;
      if (fresh > pickGain) {
        pick = i;
        pickGain = fresh;
      }
    }
    if (pick < 0) break;
    chosen.push(pick);
    gains.push(pickGain);
    for (const e of sets[pick]) covered[e] = 1;
    count += pickGain;
  }
  const elements = [];
  for (let e = 0; e < universeSize; e++) if (covered[e]) elements.push(e);
  return { chosen, count, gains, elements };
}

/** Independent greedy over 32-bit bitmasks (universe size <= 31). */
function coverageGreedyMask(masks, k) {
  const chosen = [];
  const gains = [];
  let covered = 0;
  for (let step = 0; step < k; step++) {
    let pick = -1;
    let pickGain = 0;
    for (let i = 0; i < masks.length; i++) {
      if (chosen.indexOf(i) >= 0) continue;
      const fresh = popcount32(~covered & masks[i]);
      if (fresh > pickGain) {
        pick = i;
        pickGain = fresh;
      }
    }
    if (pick < 0) break;
    chosen.push(pick);
    gains.push(pickGain);
    covered = (covered | masks[pick]) >>> 0;
  }
  return { chosen, count: popcount32(covered), covered, gains };
}

function buildMasks(sets) {
  return sets.map((set) => {
    let mask = 0;
    for (const e of set) mask |= 1 << e;
    return mask >>> 0;
  });
}

function unionSize(sets, chosen) {
  const seen = new Set();
  for (const i of chosen) for (const e of sets[i]) seen.add(e);
  return seen.size;
}

/** Exhaustive optimum over all C(m, k) budgets. */
function coverageOptimal(sets, k) {
  let best = -Infinity;
  let bestSet = null;
  let evaluated = 0;
  const current = [];
  const rec = (start) => {
    if (current.length === k) {
      evaluated += 1;
      const value = unionSize(sets, current);
      if (value > best) {
        best = value;
        bestSet = current.slice();
      }
      return;
    }
    for (let i = start; i < sets.length; i++) {
      current.push(i);
      rec(i + 1);
      current.pop();
    }
  };
  rec(0);
  return { chosen: bestSet, count: best, evaluated };
}

function binomial(n, k) {
  if (k < 0 || k > n) return 0;
  let result = 1;
  for (let i = 1; i <= k; i++) result = (result * (n - k + i)) / i;
  return Math.round(result);
}

/** Random family: m sets of random subsets of a universe run through splitmix64. */
function randomFamily(rng, m, universeSize, percent) {
  const sets = [];
  for (let i = 0; i < m; i++) {
    const set = [];
    for (let e = 0; e < universeSize; e++) if (rng.nextInt(100) < percent) set.push(e);
    sets.push(set);
  }
  return { sets, universeSize };
}

/**
 * The canonical tight family, in unweighted coverage form (every atom is one element, so all
 * counts are exact integers).  See the file header for the construction.
 */
function tightLayeredFamily(k) {
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
    const set = [];
    for (let i = 0; i < k; i++) set.push(...layers[t][i]);
    sets.push(set);
  }
  for (let i = 0; i < k; i++) {
    const set = [];
    for (let t = 0; t < k - 1; t++) set.push(...layers[t][i]);
    set.push(...priv[i]);
    sets.push(set);
  }
  return { sets, universeSize: next, L, k, privateAtoms };
}

function main() {
  const assert = require("node:assert");

  // (1) Hand cases: single pick, unlimited budget, disjoint sets - greedy is optimal.
  {
    const sets = [[0, 1, 2], [2, 3], [3, 4, 5, 6], [0], []];
    const greedy1 = coverageGreedy(sets, 7, 1);
    assert.deepStrictEqual(greedy1.chosen, [2], "k = 1 takes the largest set");
    assert.strictEqual(greedy1.count, 4);
    assert.strictEqual(coverageOptimal(sets, 1).count, 4);
    const all = coverageGreedy(sets, 7, 5);
    assert.strictEqual(all.count, 7, "k >= m covers the whole universe");
    assert.strictEqual(coverageOptimal(sets, 5).count, 7);
    const disjoint = [[0, 3], [1, 2], [4, 5], [], []];
    const budgetOpt = coverageOptimal(disjoint, 3).count;
    assert.strictEqual(coverageGreedy(disjoint, 6, 3).count, budgetOpt,
      "pairwise disjoint sets: greedy is optimal");
    assert.strictEqual(budgetOpt, 6);
    // subsets of each other: the largest set wins immediately, and greedy then stops because no
    // remaining set can add a fresh element (documented early exit when the best gain is 0)
    const nested = [[0, 1], [0, 1, 2], [0, 1, 2, 3]];
    const nestedGreedy = coverageGreedy(nested, 4, 2);
    assert.deepStrictEqual(nestedGreedy.chosen, [2]);
    assert.strictEqual(nestedGreedy.count, 4);
    assert.strictEqual(coverageOptimal(nested, 2).count, 4);
  }

  // (2) Both implementations agree, and the reported cover is recomputed independently.
  {
    const rng = new Rng(8675309);
    for (let trial = 0; trial < 60; trial++) {
      const m = 3 + rng.nextInt(7);
      const u = 1 + rng.nextInt(20);
      const { sets } = randomFamily(rng, m, u, 10 + rng.nextInt(60));
      for (let k = 1; k <= m; k++) {
        const flags = coverageGreedy(sets, u, k);
        const masks = coverageGreedyMask(buildMasks(sets), k);
        assert.deepStrictEqual(flags.chosen, masks.chosen,
          `both greedy variants must make the same choices (trial ${trial}, k = ${k})`);
        assert.deepStrictEqual(flags.gains, masks.gains);
        assert.strictEqual(flags.count, masks.count);
        assert.strictEqual(flags.count, unionSize(sets, flags.chosen),
          "the reported cover matches a Set union of the chosen sets");
        assert.strictEqual(flags.elements.length, flags.count);
        assert.deepStrictEqual(flags.elements,
          [...new Set(sets.flatMap((set, i) => (flags.chosen.includes(i) ? set : [])))]
            .sort((a, b) => a - b));
      }
    }
  }

  // (3) The (1 - 1/e) guarantee against exhaustive search, plus OPT <= k * max |S|.
  {
    let worstRatio = 1;
    let worstRecord = null;
    let instances = 0;
    for (let seed = 1; seed <= 100; seed++) {
      const rng = new Rng(4242 + seed);
      const m = 5 + rng.nextInt(6);
      const u = 6 + rng.nextInt(10);
      const { sets } = randomFamily(rng, m, u, 15 + rng.nextInt(50));
      for (let k = 1; k <= 3; k++) {
        if (k > m) continue;
        const greedy = coverageGreedy(sets, u, k);
        const optimal = coverageOptimal(sets, k);
        const bound = 1 - Math.pow(1 - 1 / k, k);
        instances += 1;
        assert.strictEqual(optimal.evaluated, binomial(m, k));
        assert.ok(greedy.count <= optimal.count, "greedy cannot beat the optimum");
        assert.ok(greedy.count >= bound * optimal.count - EPS,
          `seed ${seed} k ${k}: ${greedy.count} vs ${(bound * optimal.count).toFixed(3)}`);
        assert.ok(greedy.count >= (1 - 1 / Math.E) * optimal.count - EPS,
          "and certainly the k-independent 1 - 1/e floor");
        const maxSetSize = Math.max(...sets.map((s) => s.length));
        const upperBound = Math.min(u, k * maxSetSize);
        assert.ok(optimal.count <= upperBound,
          `OPT ${optimal.count} <= min(u, k * max|S|) = ${upperBound}`);
        assert.ok(greedy.count >= Math.min(u, maxSetSize),
          "even one greedy pick covers at least the largest set");
        const ratio = optimal.count > 0 ? greedy.count / optimal.count : 1.0;
        if (ratio < worstRatio) {
          worstRatio = ratio;
          worstRecord = { seed, m, u, k, ratio, greedy: greedy.count, optimal: optimal.count };
        }
      }
    }
    assert.ok(worstRatio >= 1 - 1 / Math.E, "the 1 - 1/e floor is never crossed");
    assert.ok(worstRatio >= 0.83,
      `documented: the worst of ${instances} searches is ${worstRatio} (${JSON.stringify(worstRecord)})`);
    assert.ok(worstRatio > 1 - Math.pow(1 - 1 / 2, 2),
      "the random family stays strictly above the designed k = 2 tight value 0.75");
  }

  // (4) The tight family attains 1 - (1 - 1/k)^k exactly for k = 2, 3, 4.
  {
    const ratios = [];
    for (const k of [2, 3, 4]) {
      const family = tightLayeredFamily(k);
      const { sets, universeSize } = family;
      assert.strictEqual(universeSize, Math.pow(k, k + 1), "u = k^(k+1) unit atoms");
      assert.strictEqual(sets.length, 2 * k - 1, "k - 1 decoy sets plus k block sets");
      assert.strictEqual(unionSize(sets, [...Array(sets.length).keys()]), universeSize,
        "the whole universe is covered by the family");
      const greedy = coverageGreedy(sets, universeSize, k);
      const optimal = coverageOptimal(sets, k);
      assert.strictEqual(optimal.evaluated, binomial(2 * k - 1, k));
      const bound = 1 - Math.pow(1 - 1 / k, k);
      assert.strictEqual(optimal.count, k * family.L, "the k disjoint block sets give OPT = k L");
      assert.strictEqual(greedy.count, Math.round(bound * k * family.L));
      assert.ok(Math.abs(greedy.count / optimal.count - bound) <= 1e-12,
        `k = ${k}: greedy/OPT = ${greedy.count / optimal.count} must equal ${bound}`);
      assert.deepStrictEqual(greedy.chosen, [...Array(k).keys()],
        "greedy takes the k - 1 decoys on index tie-breaks, then one block set");
      assert.ok(greedy.count < optimal.count, "so the greedy is strictly suboptimal here");
      assert.ok(Math.abs(family.L - Math.pow(k, k)) <= 1e-9);
      assert.strictEqual(family.privateAtoms, Math.round(family.L * Math.pow(1 - 1 / k, k - 1)));
      ratios.push(greedy.count / optimal.count);
    }
    assert.deepStrictEqual(ratios.map((r) => Number(r.toFixed(6))), [0.75, 0.703704, 0.683594]);
    assert.ok(ratios[0] > ratios[1] && ratios[1] > ratios[2]);
    assert.ok(ratios[2] >= 1 - 1 / Math.E);
    // k = 3 spot value: 57 of 81 atoms covered.
    const k3 = tightLayeredFamily(3);
    assert.strictEqual(coverageGreedy(k3.sets, k3.universeSize, 3).count, 57);
    assert.strictEqual(coverageOptimal(k3.sets, 3).count, 81);
  }

  // (5) Boundary cases.
  {
    assert.deepStrictEqual(coverageGreedy([[0, 1]], 2, 0), { chosen: [], count: 0, gains: [],
      elements: [] });
    assert.strictEqual(coverageOptimal([[0, 1]], 0).count, 0);
    assert.strictEqual(coverageGreedy([], 0, 3).count, 0, "no sets, nothing to cover");
    assert.strictEqual(coverageGreedy([[]], 5, 1).count, 0, "the only set is empty");
    assert.strictEqual(coverageGreedy([[], [], []], 5, 3).count, 0);
    const duplicates = [[0, 1], [0, 1], [0, 1]];
    assert.strictEqual(coverageGreedy(duplicates, 2, 3).count, 2);
    assert.deepStrictEqual(coverageGreedy(duplicates, 2, 3).chosen, [0]);
    assert.deepStrictEqual(coverageGreedy(duplicates, 2, 3).gains, [2],
      "greedy stops as soon as no set adds a fresh element");
    assert.strictEqual(coverageOptimal(duplicates, 2).count, 2);
    const repeated = [[0, 0, 1, 1, 1], [1, 1, 2]]; // duplicate elements inside a set
    assert.strictEqual(coverageGreedy(repeated, 3, 1).count, 2,
      "the de-duplicated first set covers exactly {0, 1}");
    assert.deepStrictEqual(coverageGreedy(repeated, 3, 2).chosen, [0, 1]);
    assert.strictEqual(coverageGreedy(repeated, 3, 2).count, 3);
    assert.deepStrictEqual(coverageGreedyMask(buildMasks(repeated), 2).chosen,
      coverageGreedy(repeated, 3, 2).chosen,
      "both variants agree despite the repeated elements");
    assert.strictEqual(buildMasks(repeated)[0], 3);
    assert.strictEqual(popcount32(buildMasks(repeated)[0]), 2);
    assert.strictEqual(popcount32(0xffffffff), 32);
    assert.strictEqual(popcount32(0), 0);
    const biggerBudget = coverageGreedy([[0], [1], [2]], 3, 7);
    assert.strictEqual(biggerBudget.count, 3, "k > m stops after m sets");
    assert.strictEqual(biggerBudget.chosen.length, 3);
    assert.throws(() => tightLayeredFamily(1), /k >= 2/);
  }

  // (6) Monotonicity in k, deterministic reruns, and mask variant on the tight family's small case.
  {
    const rng = new Rng(2024);
    const { sets, universeSize } = randomFamily(rng, 8, 14, 30);
    let previous = -1;
    for (let k = 0; k <= 8; k++) {
      const value = coverageGreedy(sets, universeSize, k).count;
      assert.ok(value >= previous, `the covered count must not decrease (k = ${k})`);
      previous = value;
    }
    assert.deepStrictEqual(coverageGreedy(sets, universeSize, 3), coverageGreedy(sets, universeSize, 3));
    assert.deepStrictEqual(coverageOptimal(sets, 3), coverageOptimal(sets, 3));
    const family2 = tightLayeredFamily(2);
    const masked = coverageGreedyMask(buildMasks(family2.sets), 2);
    const flagged = coverageGreedy(family2.sets, family2.universeSize, 2);
    assert.deepStrictEqual(masked.chosen, flagged.chosen);
    assert.strictEqual(masked.count, flagged.count);
    assert.strictEqual(masked.count, 6, "k = 2 tight family: greedy covers 6 of the 8 atoms");
    assert.strictEqual(family2.universeSize, 8);
    assert.strictEqual(unionSize(family2.sets, [1, 2]), 8, "the two block sets cover everything");
  }
}

module.exports = { MASK64, Rng, popcount32, coverageGreedy, coverageGreedyMask, buildMasks,
  unionSize, coverageOptimal, binomial, randomFamily, tightLayeredFamily };

if (require.main === module) {
  main();
  console.log("PASS");
}
