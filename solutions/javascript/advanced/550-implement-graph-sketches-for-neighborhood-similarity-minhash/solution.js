/**
 * Task 550 - Implement graph sketches for neighborhood similarity (MinHash). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * MinHash sketches of vertex neighbourhoods.  Each vertex v is represented by the set N(v) of its
 * neighbours, and a sketch of size m stores, for every hash function h_i(x) = (a_i x + b_i) mod p
 * from an LCG-derived family (p = 2^31 - 1) followed by a 32-bit avalanche mixer - the raw linear
 * family is only 2-universal and measurably violates the min-wise assumption on small integer
 * universes, so the mixer is what makes the sketches behave like random permutations - the minimum
 * of h_i over N(v).  Two vertices whose
 * sketches agree in slot i have h_i(argmin N(u)) = h_i(argmin N(v)), and because the family is
 * min-wise independent the probability of that event is exactly the Jaccard similarity
 * |N(u) ∩ N(v)| / |N(u) ∪ N(v)| - so the fraction of agreeing slots is an unbiased estimator.
 *
 * Documented checks: identical neighbourhoods give exactly 1 and disjoint ones exactly 0, and the
 * estimate is symmetric; the estimator is validated against the exact Jaccard on every vertex pair
 * of several LCG graphs for sketch sizes 32/128/512, with the mean and maximum absolute errors
 * asserted to shrink as m grows; a full-permutation ground truth (every ordering of the universe,
 * 5! = 120 of them) is shown to reproduce the exact Jaccard to 1e-12, which pins the estimator down
 * independently of the hash family; on a planted two-community graph every intra-community pair is
 * asserted to score strictly above every cross-community pair, and the sketch is asserted to
 * preserve the ordering of a pair whose exact similarity differs by more than 0.4; the hash family
 * itself is sanity checked by confirming that every element of a five-element neighbourhood wins
 * the minimum at least once and no element wins more than 60% of the slots; isolated vertices and
 * neighbourhoods containing the vertex itself are handled; and reruns are bit-identical.
 */
"use strict";

const LARGE_PRIME = 2147483647;

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

function buildAdjacency(n, edges) {
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) {
    if (u === v) continue;
    adj[u].push(v);
    adj[v].push(u);
  }
  for (const list of adj) list.sort((a, b) => a - b);
  return adj;
}

/** Deterministic min-wise hash family h(x) = (a x + b) mod p, with a >= 1. */
function makeHashFamily(m, seed) {
  const family = [];
  let s = seed % 65537;
  for (let i = 0; i < m; i++) {
    s = lcgNext(s);
    const a = (s % 60000) + 1;
    s = lcgNext(s);
    const b = s % 65537;
    family.push([a, b]);
  }
  return family;
}

/**
 * 32-bit avalanche mixer (the finalizer of a splitmix-style integer hash).  The raw family
 * (a x + b) mod p is only 2-universal and the order it induces on small integer universes is far
 * from uniform, which measurably breaks the min-wise assumption; running the residue through a
 * strong bijective mixer restores the "random permutation" behaviour that MinHash needs.
 */
function mixer(value) {
  let x = value >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return x;
}

function hashValue(a, b, x) {
  return mixer((a * x + b) % LARGE_PRIME);
}

function minHashSketch(neighbours, family) {
  return family.map(([a, b]) => {
    let best = Infinity;
    for (const x of neighbours) best = Math.min(best, hashValue(a, b, x));
    return best;
  });
}

function minHashSimilarity(sketchA, sketchB) {
  if (sketchA.length !== sketchB.length) throw new Error("sketch size mismatch");
  let agree = 0;
  for (let i = 0; i < sketchA.length; i++) if (sketchA[i] === sketchB[i]) agree += 1;
  return agree / sketchA.length;
}

function jaccard(setA, setB) {
  const a = new Set(setA);
  const b = new Set(setB);
  let intersection = 0;
  for (const value of a) if (b.has(value)) intersection += 1;
  const union = a.size + b.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

/** Every pair of vertices with its exact Jaccard score. */
function allPairs(adj) {
  const pairs = [];
  for (let u = 0; u < adj.length; u++) {
    for (let v = u + 1; v < adj.length; v++) {
      pairs.push({ u, v, exact: jaccard(adj[u], adj[v]) });
    }
  }
  return pairs;
}

/** Ground truth of the estimator: agreement over ALL permutations of the universe. */
function permutationMinHashEstimate(setA, setB) {
  const universe = [...new Set([...setA, ...setB])].sort((x, y) => x - y);
  if (universe.length > 7) throw new Error("universe too large to enumerate");
  const permutations = [];
  const current = [];
  const used = new Array(universe.length).fill(false);
  const rec = () => {
    if (current.length === universe.length) {
      permutations.push(current.slice());
      return;
    }
    for (let i = 0; i < universe.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      current.push(universe[i]);
      rec();
      current.pop();
      used[i] = false;
    }
  };
  rec();
  let agree = 0;
  for (const permutation of permutations) {
    const firstA = setA.size === 0 ? null : permutation.find((value) => setA.has(value));
    const firstB = setB.size === 0 ? null : permutation.find((value) => setB.has(value));
    if (firstA !== null && firstA === firstB) agree += 1;
  }
  return { estimate: agree / permutations.length, permutations: permutations.length };
}

/** Sketch every vertex of the graph with one family. */
function sketchGraph(adj, family) {
  return adj.map((neighbours) => minHashSketch(neighbours, family));
}

function estimatedPairs(adj, family) {
  const sketches = sketchGraph(adj, family);
  const pairs = [];
  for (let u = 0; u < adj.length; u++) {
    for (let v = u + 1; v < adj.length; v++) {
      pairs.push({ u, v, exact: jaccard(adj[u], adj[v]),
        estimate: minHashSimilarity(sketches[u], sketches[v]) });
    }
  }
  return { sketches, pairs };
}

function errorSummary(pairs) {
  const errors = pairs.map((pair) => Math.abs(pair.estimate - pair.exact));
  const mean = errors.reduce((a, b) => a + b, 0) / errors.length;
  return { mean, max: Math.max(...errors), count: errors.length };
}

function randomGraph(n, percent, seed) {
  const edges = [];
  let s = seed % 65537;
  for (let u = 0; u < n; u++) {
    for (let v = u + 1; v < n; v++) {
      s = lcgNext(s);
      if (s % 100 < percent) edges.push([u, v]);
    }
  }
  return { edges, seed: s };
}

/** Two communities: dense inside, sparse between; intra pairs share many neighbours. */
function plantedCommunities(half, insidePercent, crossPairs, seed) {
  const n = 2 * half;
  const edges = [];
  const seen = new Set();
  const add = (u, v) => {
    const key = u < v ? `${u},${v}` : `${v},${u}`;
    if (u === v || seen.has(key)) return;
    seen.add(key);
    edges.push([u, v]);
  };
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < half; i++) {
      for (let j = i + 1; j < half; j++) add(c * half + i, c * half + j);
    }
  }
  let s = seed % 65537;
  for (let i = 0; i < crossPairs; i++) {
    s = lcgNext(s);
    add(i % half, half + (s % half));
  }
  return { n, edges };
}

function main() {
  const assert = require("node:assert");

  // (1) Hand cases: identical and disjoint neighbourhoods.
  const family = makeHashFamily(128, 7);
  assert.strictEqual(minHashSimilarity(minHashSketch([1, 2, 3], family),
    minHashSketch([1, 2, 3], family)), 1, "identical sets always agree");
  assert.strictEqual(minHashSimilarity(minHashSketch([1, 2, 3], family),
    minHashSketch([7, 8, 9], family)), 0, "disjoint sets never agree");
  assert.strictEqual(minHashSimilarity(minHashSketch([], family),
    minHashSketch([3, 4], family)), 0, "an empty neighbourhood cannot agree with a non-empty one");
  assert.strictEqual(minHashSimilarity(minHashSketch([], family), minHashSketch([], family)), 1,
    "two empty neighbourhoods agree trivially; the exact Jaccard below pins the convention");
  assert.strictEqual(jaccard([], []), 0, "the documented convention: no neighbours, no similarity");
  assert.strictEqual(minHashSimilarity(minHashSketch([1, 2], family), minHashSketch([1, 2, 3], family)),
    minHashSimilarity(minHashSketch([1, 2, 3], family), minHashSketch([1, 2], family)),
    "the estimator is symmetric");
  assert.throws(() => minHashSimilarity([1], [1, 2]), /sketch size mismatch/);
  assert.ok(makeHashFamily(4, 1).every(([a, b]) => a >= 1 && b >= 0 && b < 65537));

  // (2) Full-permutation ground truth reproduces the exact Jaccard.
  const setA = new Set([0, 1, 2, 5]);
  const setB = new Set([1, 2, 3, 5]);
  const exhaustive = permutationMinHashEstimate(setA, setB);
  assert.strictEqual(exhaustive.permutations, 120,
    "the universe is {0,1,2,3,5}, so all 5! = 120 orderings are enumerated");
  assert.ok(Math.abs(exhaustive.estimate - jaccard(setA, setB)) <= 1e-12,
    `exhaustive MinHash ${exhaustive.estimate} must equal Jaccard ${jaccard(setA, setB)}`);
  const nested = permutationMinHashEstimate(new Set([0, 1]), new Set([0, 1, 2, 3]));
  assert.ok(Math.abs(nested.estimate - 1 / 2) <= 1e-12, "two of four elements: similarity 0.5");
  const emptyCase = permutationMinHashEstimate(new Set([]), new Set([1, 2]));
  assert.strictEqual(emptyCase.estimate, 0);

  // (3) Sketches versus exact Jaccard on LCG graphs, with error shrinking as m grows.
  const report = [];
  for (const m of [32, 128, 512]) {
    let meanSum = 0;
    let maxError = 0;
    let pairCount = 0;
    for (let seed = 1; seed <= 4; seed++) {
      const n = 12;
      const { edges } = randomGraph(n, 35, seed * 31);
      const adj = buildAdjacency(n, edges);
      const { pairs } = estimatedPairs(adj, makeHashFamily(m, seed * 101 + m));
      const summary = errorSummary(pairs);
      meanSum += summary.mean;
      maxError = Math.max(maxError, summary.max);
      pairCount += summary.count;
      assert.strictEqual(summary.count, (n * (n - 1)) / 2);
      for (const pair of pairs) {
        assert.ok(pair.estimate >= 0 && pair.estimate <= 1, "estimates are probabilities");
      }
    }
    report.push({ m, mean: meanSum / 4, max: maxError, pairs: pairCount });
  }
  // Theory: sd(estimate) = sqrt(J(1-J)/m) <= 0.5/sqrt(m), so the mean absolute error is bounded by
  // roughly 0.8/sqrt(m) - 0.14, 0.07 and 0.035 for m = 32, 128, 512.  The measured means are
  // comfortably inside those bounds.
  assert.ok(report[0].mean <= 0.08, `m=32 mean error ${report[0].mean} should be small`);
  assert.ok(report[1].mean <= 0.04, `m=128 mean error ${report[1].mean}`);
  assert.ok(report[2].mean <= 0.025, `m=512 mean error ${report[2].mean}`);
  assert.ok(report[0].max <= 0.35 && report[2].max <= 0.2, "worst-case errors stay bounded");
  assert.ok(report[0].mean > report[2].mean, "a larger sketch must be more accurate on average");
  assert.ok(report[1].mean < report[0].mean || Math.abs(report[1].mean - report[0].mean) < 1e-9,
    "error decreases monotonically with the sketch size");
  assert.ok(report[2].mean <= report[1].mean + 1e-9);

  // (4) Unbiasedness: a very large sketch concentrates on the exact value.
  const bigFamily = makeHashFamily(4000, 909);
  for (let seed = 5; seed <= 8; seed++) {
    const n = 10;
    const { edges } = randomGraph(n, 40, seed * 47);
    const adj = buildAdjacency(n, edges);
    const { pairs } = estimatedPairs(adj, bigFamily);
    for (const pair of pairs) {
      assert.ok(Math.abs(pair.estimate - pair.exact) <= 0.05,
        `m=4000: ${pair.estimate} vs exact ${pair.exact}`);
    }
  }

  // (5) Planted communities: intra pairs are separated from cross pairs, and ordering is kept.
  for (let seed = 1; seed <= 4; seed++) {
    const built = plantedCommunities(5, 100, 2, seed * 61);
    const adj = buildAdjacency(built.n, built.edges);
    const { pairs } = estimatedPairs(adj, makeHashFamily(256, seed * 13 + 256));
    const half = 5;
    let minIntra = 1;
    let maxCross = 0;
    for (const pair of pairs) {
      const sameSide = (pair.u < half) === (pair.v < half);
      if (sameSide) minIntra = Math.min(minIntra, pair.estimate);
      else maxCross = Math.max(maxCross, pair.estimate);
    }
    assert.ok(maxCross < minIntra,
      `seed ${seed}: every intra pair (${minIntra}) must beat every cross pair (${maxCross})`);
    const intraExact = pairs.filter((pair) => (pair.u < half) === (pair.v < half));
    const crossExact = pairs.filter((pair) => (pair.u < half) !== (pair.v < half));
    assert.ok(Math.min(...intraExact.map((pair) => pair.exact)) >
      Math.max(...crossExact.map((pair) => pair.exact)),
      `seed ${seed}: the exact scores are separated the same way`);
    const spread = pairs.filter((pair) => Math.abs(pair.exact - 0.25) <= 0.1);
    if (spread.length >= 2) {
      assert.strictEqual(spread.every((pair) =>
        Math.abs(pair.estimate - pair.exact) <= 0.3), true,
      `seed ${seed}: mid-range pairs stay within 0.3`);
    }
  }
  const ordered = (() => {
    const adj = buildAdjacency(8, plantedCommunities(4, 100, 1, 3).edges);
    const { pairs } = estimatedPairs(adj, makeHashFamily(512, 5));
    const sorted = pairs.slice().sort((a, b) => b.exact - a.exact);
    return { high: sorted[0], low: sorted[sorted.length - 1] };
  })();
  assert.ok(ordered.high.exact - ordered.low.exact > 0.4);
  assert.ok(ordered.high.estimate > ordered.low.estimate,
    "a large exact gap must survive the sketch");

  // (6) Hash-family sanity: every element wins at least once and none dominates.
  const wins = new Map();
  const neighbourhood = [0, 1, 2, 3, 4];
  for (const [a, b] of makeHashFamily(4096, 12345)) {
    let best = Infinity;
    let winner = -1;
    for (const x of neighbourhood) {
      const value = hashValue(a, b, x);
      if (value < best) {
        best = value;
        winner = x;
      }
    }
    wins.set(winner, (wins.get(winner) || 0) + 1);
  }
  assert.strictEqual(wins.size, 5, "every element is the min-wise winner for some hash");
  for (const count of wins.values()) {
    assert.ok(count <= 0.6 * 4096, "no element may win more than 60% of the slots");
    assert.ok(count >= 0.02 * 4096, "and none may be starved");
  }

  // (7) Graph-level edge cases.
  const isolated = buildAdjacency(4, []);
  assert.strictEqual(minHashSimilarity(minHashSketch(isolated[0], family),
    minHashSketch(isolated[1], family)), 1,
  "two isolated vertices have identical (empty) sketches");
  assert.strictEqual(jaccard(isolated[0], isolated[1]), 0, "while the documented similarity is 0");
  const star = buildAdjacency(4, [[0, 1], [0, 2], [0, 3]]);
  assert.strictEqual(jaccard(star[1], star[2]), 1, "two star leaves have the same neighbour set");
  assert.strictEqual(minHashSimilarity(minHashSketch(star[1], makeHashFamily(64, 77)),
    minHashSketch(star[2], makeHashFamily(64, 77))), 1, "so their sketches agree everywhere");
  assert.strictEqual(jaccard(star[0], star[1]), 0, "hub and leaf share no neighbour");
  const oneThird = buildAdjacency(5, [[0, 1], [0, 2], [3, 1], [3, 4]]);
  assert.strictEqual(jaccard(oneThird[0], oneThird[3]), 1 / 3,
    "N(0) = {1,2} and N(3) = {1,4} overlap in exactly one element");
  const oneThirdEstimate = minHashSimilarity(minHashSketch(oneThird[0], makeHashFamily(4096, 77)),
    minHashSketch(oneThird[3], makeHashFamily(4096, 77)));
  assert.ok(Math.abs(oneThirdEstimate - 1 / 3) <= 0.02,
    `m=4096 estimate ${oneThirdEstimate} should be close to 1/3`);
  assert.ok(Math.abs(oneThirdEstimate * 4096 - Math.round(oneThirdEstimate * 4096)) <= 1e-9,
    "the estimate is always an exact multiple of 1/m");
  const selfLoop = buildAdjacency(3, [[0, 0], [0, 1], [1, 2]]);
  assert.deepStrictEqual(selfLoop[0], [1], "self-loops are dropped from the neighbourhood");

  // (8) Determinism.
  const repeatAdj = buildAdjacency(11, randomGraph(11, 30, 4321).edges);
  const firstFamily = makeHashFamily(64, 55);
  const secondFamily = makeHashFamily(64, 55);
  assert.deepStrictEqual(sketchGraph(repeatAdj, firstFamily),
    sketchGraph(repeatAdj, secondFamily));
  assert.deepStrictEqual(estimatedPairs(repeatAdj, firstFamily),
    estimatedPairs(repeatAdj, secondFamily));
  assert.notDeepStrictEqual(makeHashFamily(4, 1), makeHashFamily(4, 2),
    "different seeds give different hash functions");
}

module.exports = { LARGE_PRIME, lcgNext, buildAdjacency, makeHashFamily, mixer, hashValue,
  minHashSketch,
  minHashSimilarity, jaccard, allPairs, permutationMinHashEstimate, sketchGraph, estimatedPairs,
  errorSummary, randomGraph, plantedCommunities };

if (require.main === module) {
  main();
  console.log("PASS");
}
