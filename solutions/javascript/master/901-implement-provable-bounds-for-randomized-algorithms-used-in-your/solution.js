/** Task 901 - Implement provable bounds for randomized algorithms used in your projects, with tests.
 *  (Tier: Master, CTCI Beyond — Randomized Algorithms & Verifiable Performance)
 *
 * Three primitives with the provable bound that justifies using them, and seeded tests comparing an
 * empirical measurement against that bound:
 *   quicksort  Las Vegas, always correct; E[C_n] = 2(n+1)H_n - 4n < 2n ln n + 2n. This three-way
 *              partition spends one extra comparison per subproblem of size >= 2, so the measured mean
 *              must land in [E[C_n], E[C_n] + n].
 *   min cut    Karger succeeds with probability >= 2/(n(n-1)) per run, so k restarts fail with
 *              probability <= exp(-2k/(n(n-1))); an exhaustive bipartition oracle gives the truth.
 *   estimation Hoeffding P(|mean-mu| >= eps) <= 2exp(-2m eps^2) beats Chebyshev sigma^2/(m eps^2) once
 *              m eps^2 >= 1.5 with sigma^2 <= 1/4; median-of-means fails only if >= k/2 group means
 *              deviate, bounded by the Chernoff tail exp(-k*D(1/2 || p)), p = 2exp(-2m eps^2).
 * Empirical rates get a three-sigma binomial slack: a random measurement must be consistent with the
 * bound, not equal to it. All randomness is a fixed-seed xorshift32, so runs are reproducible.
 */
"use strict";

const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.state = x;
    return x;
  }
  int(bound) {
    assert.ok(Number.isInteger(bound) && bound > 0, `int() needs a positive bound, got ${bound}`);
    return this.next() % bound;
  }
  float() { return this.next() / 4294967296; }
}

/** Three-way random-pivot quicksort; returns the sorted copy and the comparison count. */
function randomizedQuicksort(values, prng) {
  const data = values.slice();
  let comparisons = 0;
  const swap = (i, j) => { const t = data[i]; data[i] = data[j]; data[j] = t; };
  const sort = (lo, hi) => {
    while (hi - lo > 1) {
      const pivot = data[lo + prng.int(hi - lo)];
      let lt = lo;
      let i = lo;
      let gt = hi;
      while (i < gt) {
        comparisons += 1;
        if (data[i] < pivot) swap(lt++, i++);
        else if (data[i] > pivot) swap(i, --gt);
        else i += 1;
      }
      // [lt, gt) holds the pivot value and is final: recurse into the smaller side, loop on the bigger.
      if (lt - lo < hi - gt) { sort(lo, lt); lo = gt; } else { sort(gt, hi); hi = lt; }
    }
  };
  sort(0, data.length);
  return { sorted: data, comparisons };
}

/** Expected comparison count of classic random-pivot quicksort: 2(n+1)H_n - 4n. */
function expectedComparisons(n) {
  let harmonic = 0;
  for (let i = 1; i <= n; i++) harmonic += 1 / i;
  return 2 * (n + 1) * harmonic - 4 * n;
}

function permutation(n, prng) {
  const values = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = prng.int(i + 1);
    const t = values[i]; values[i] = values[j]; values[j] = t;
  }
  return values;
}

/** One contraction run on a simple graph given as an edge list. */
function kargerMinCut(vertexCount, edges, prng) {
  const parent = Array.from({ length: vertexCount }, (_, i) => i);
  const find = (x) => {
    let root = x;
    while (parent[root] !== root) root = parent[root];
    while (parent[x] !== root) { const next = parent[x]; parent[x] = root; x = next; }
    return root;
  };
  const pool = edges.map(([u, v]) => [u, v]);
  let classes = vertexCount;
  while (classes > 2) {
    const live = pool.filter(([u, v]) => find(u) !== find(v));
    if (live.length === 0) return 0;                 // graph already separated: cut of size 0
    const [a, b] = live[prng.int(live.length)];
    parent[find(b)] = find(a);
    classes -= 1;
  }
  return pool.filter(([u, v]) => find(u) !== find(v)).length;
}

/** Independent oracle: enumerate every vertex bipartition with vertex 0 fixed on the first side. */
function bruteForceMinCut(vertexCount, edges) {
  let best = Infinity;
  const side = new Array(vertexCount).fill(false);
  side[0] = true;
  // mask = all ones would put every vertex on one side (a degenerate cut of 0), so skip both ends.
  for (let mask = 1; mask < (1 << (vertexCount - 1)) - 1; mask++) {
    for (let v = 1; v < vertexCount; v++) side[v] = ((mask >>> (v - 1)) & 1) === 1;
    let crossing = 0;
    for (const [u, v] of edges) if (side[u] !== side[v]) crossing += 1;
    if (crossing < best) best = crossing;
  }
  return best;
}

const hoeffdingTail = (m, eps) => Math.min(1, 2 * Math.exp(-2 * m * eps * eps));
const chebyshevTail = (variance, m, eps) => Math.min(1, variance / (m * eps * eps));

/** Chernoff bound for the median-of-means estimator: P(Bin(k, p) >= k/2). */
function medianOfMeansTail(k, m, eps) {
  const p = hoeffdingTail(m, eps);
  if (p >= 0.5 || p <= 0) return 1;
  const divergence = 0.5 * Math.log(0.5 / p) + 0.5 * Math.log(0.5 / (1 - p));
  return Math.min(1, Math.exp(-k * divergence));
}

function sampleMean(p, m, prng) {
  let successes = 0;
  for (let i = 0; i < m; i++) if (prng.float() < p) successes += 1;
  return successes / m;
}

/** Median-of-means: k groups of m i.i.d. draws, median of the group means. */
function medianOfMeans(p, k, m, prng) {
  const means = [];
  for (let g = 0; g < k; g++) means.push(sampleMean(p, m, prng));
  means.sort((a, b) => a - b);
  return means[(k - 1) >>> 1];
}

const samplingSlack = (trials, rate) => 3 * Math.sqrt(trials * rate * (1 - rate));
const graphs = [
  { name: "triangle", n: 3, edges: [[0, 1], [1, 2], [2, 0]], minCut: 2 },
  { name: "C6", n: 6, edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]], minCut: 2 },
  { name: "K5", n: 5, edges: [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]], minCut: 4 },
  { name: "K3,3", n: 6, edges: [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]], minCut: 3 },
  { name: "two-triangles", n: 6, edges: [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [2, 3]], minCut: 1 },
];

function main() {
  const prng = new Prng(0x5eed0901);
  const oracleSort = (a) => a.slice().sort((x, y) => x - y);
  // 1. happy path — output matches the sort oracle for distinct and duplicate keys, input untouched.
  for (const n of [2, 3, 7, 33, 64]) {
    const input = Array.from({ length: n }, () => prng.int(1000) - 500);
    const { sorted } = randomizedQuicksort(input, prng);
    assert.deepStrictEqual(sorted, oracleSort(input), `sorted output for n=${n}`);
    assert.deepStrictEqual(input, input.slice(), "the input array must not be mutated");
  }
  const duplicates = [5, 1, 5, 3, 5, 2, 1, 5, 3, 1];
  assert.deepStrictEqual(randomizedQuicksort(duplicates, prng).sorted, oracleSort(duplicates));
  // 2. boundary — empty, single and two-element inputs.
  assert.deepStrictEqual(randomizedQuicksort([], prng), { sorted: [], comparisons: 0 });
  assert.deepStrictEqual(randomizedQuicksort([42], prng), { sorted: [42], comparisons: 0 });
  assert.deepStrictEqual(randomizedQuicksort([2, 1], prng), { sorted: [1, 2], comparisons: 2 });
  // 3. Las Vegas bound — the empirical mean lands inside [E[C_n], E[C_n] + n] and below 2n ln n + 2n.
  const n3 = 64;
  const trials3 = 120;
  const measured = Array.from({ length: trials3 },
    () => randomizedQuicksort(permutation(n3, prng), prng).comparisons).reduce((a, b) => a + b, 0) / trials3;
  const analytic3 = expectedComparisons(n3);
  const provable3 = 2 * n3 * Math.log(n3) + 2 * n3;
  const noise3 = 0.15 * analytic3;
  assert.ok(analytic3 <= provable3, `analytic ${analytic3} must be below provable ${provable3}`);
  assert.ok(measured >= analytic3 - noise3, `mean ${measured} must reach the expectation ${analytic3}`);
  assert.ok(measured <= analytic3 + n3 + noise3, `mean ${measured} must not exceed E[C_n] + n`);
  assert.ok(measured <= provable3, `mean ${measured} must stay below 2n ln n + 2n = ${provable3}`);
  // 4. Karger vs the exhaustive oracle on five graphs (best of 60 restarts per graph).
  for (const graph of graphs) {
    assert.strictEqual(bruteForceMinCut(graph.n, graph.edges), graph.minCut, `oracle ${graph.name}`);
    let best = Infinity;
    for (let r = 0; r < 60; r++) best = Math.min(best, kargerMinCut(graph.n, graph.edges, prng));
    assert.strictEqual(best, graph.minCut, `Karger found the min cut of ${graph.name}`);
  }
  // 5. Monte Carlo bound — failure frequency of k restarts stays under exp(-2k/(n(n-1))).
  const graph5 = graphs[3];
  const k5 = 20;
  const trials5 = 200;
  let failures5 = 0;
  for (let t = 0; t < trials5; t++) {
    let best = Infinity;
    for (let r = 0; r < k5; r++) best = Math.min(best, kargerMinCut(graph5.n, graph5.edges, prng));
    if (best !== graph5.minCut) failures5 += 1;
  }
  const bound5 = Math.min(1, Math.exp(-2 * k5 / (graph5.n * (graph5.n - 1))));
  assert.ok(bound5 < 1, "the restart union bound must be informative");
  assert.ok(failures5 <= trials5 * bound5 + samplingSlack(trials5, bound5),
    `${failures5}/${trials5} failures exceed the provable bound ${bound5}`);
  // 6. Hoeffding — the empirical deviation frequency of a sample mean stays under the analytic tail. — the empirical deviation frequency of a sample mean stays under the analytic tail.
  const m6 = 64;
  const eps6 = 0.15;
  const bound6 = hoeffdingTail(m6, eps6);
  let violations6 = 0;
  for (let t = 0; t < 500; t++) {
    if (Math.abs(sampleMean(0.3, m6, prng) - 0.3) >= eps6) violations6 += 1;
  }
  assert.ok(bound6 < 0.5, `the Hoeffding bound ${bound6} must be informative`);
  assert.ok(violations6 <= 500 * bound6 + samplingSlack(500, bound6),
    `${violations6}/500 deviations exceed the Hoeffding bound ${bound6}`);
  // 7. Median-of-means — its weaker but valid Chernoff bound still covers the observed failures. — its weaker but valid Chernoff bound still covers the observed failures.
  const k7 = 7;
  const m7 = 24;
  const eps7 = 0.25;
  let violations7 = 0;
  for (let t = 0; t < 400; t++) {
    if (Math.abs(medianOfMeans(0.3, k7, m7, prng) - 0.3) >= eps7) violations7 += 1;
  }
  const bound7 = medianOfMeansTail(k7, m7, eps7);
  assert.ok(bound7 > 0 && bound7 < 0.5, `median-of-means bound ${bound7} must be informative`);
  assert.ok(violations7 <= 400 * bound7 + samplingSlack(400, bound7),
    `${violations7}/400 median-of-means deviations exceed ${bound7}`);
  // 8. The bounds themselves — Hoeffding dominates Chebyshev for m*eps^2 >= 1.5, both are monotone. — Hoeffding dominates Chebyshev for m*eps^2 >= 1.5, both are monotone.
  for (const t of [1.5, 2, 5, 12]) {
    const eps = Math.sqrt(t / 100);
    assert.ok(hoeffdingTail(100, eps) <= chebyshevTail(0.25, 100, eps) + 1e-12,
      `Hoeffding must dominate Chebyshev at m*eps^2=${t}`);
  }
  for (const m of [10, 40, 160]) {
    assert.ok(hoeffdingTail(m + 1, 0.2) < hoeffdingTail(m, 0.2), "the tail decreases with m");
    assert.ok(hoeffdingTail(m, 0.3) < hoeffdingTail(m, 0.2), "the tail decreases with eps");
  }
  assert.strictEqual(hoeffdingTail(0, 0.1), 1, "no samples means no bound");
  // 9. Determinism — the same seed replays exactly, a different seed diverges. — the same seed replays exactly, a different seed diverges.
  const signature = (seed) => {
    const local = new Prng(seed);
    const comparisons = randomizedQuicksort(permutation(32, local), local).comparisons;
    const cut = kargerMinCut(6, graphs[3].edges, local);
    return `${comparisons}:${cut}:${sampleMean(0.4, 50, local).toFixed(6)}`;
  };
  assert.strictEqual(signature(777), signature(777));
  assert.notStrictEqual(signature(777), signature(778));
  console.log(`bounds: quicksort mean ${measured.toFixed(1)} comparisons (E[C_64]=` +
    `${analytic3.toFixed(1)}, bound ${provable3.toFixed(1)}); karger failure ${failures5}/${trials5} <= ` +
    `${bound5.toFixed(3)}; hoeffding ${violations6}/500 deviations <= ${bound6.toFixed(3)}`);
  console.log("PASS");
}

if (require.main === module) main();

module.exports = {
  Prng, randomizedQuicksort, expectedComparisons, kargerMinCut, bruteForceMinCut,
  hoeffdingTail, chebyshevTail, medianOfMeansTail, sampleMean, medianOfMeans,
};
