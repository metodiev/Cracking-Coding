/**
 * Task 553 - Implement induced subgraph frequency sampling. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * The induced-k-subgraph frequencies of a graph are the relative counts of the isomorphism classes of
 * the subgraphs induced by its vertex k-subsets; on a large graph the exact C(n,k) enumeration is the
 * bottleneck, so the standard practice is to sample k-subsets and rescale.  This file implements both
 * sides.  Exactly: every k-subset is classified by a canonical form obtained by minimising the upper
 * triangle bitstring over all k! vertex orderings, so the classes are genuine isomorphism classes (3
 * vertices: empty / single edge / path / triangle; 4 vertices: the 11 unlabelled graphs on 4 vertices)
 * and the exhaustive pass gives the ground-truth counts and frequencies count / C(n,k).
 *
 * Sampling: S k-subsets are drawn and the count of a class is estimated by multiplicities *if* the
 * draws are with replacement, namely (hits / S) * C(n,k) - an unbiased estimator whose variance is
 * exactly C(n,k)^2 p(1-p)/S - and the same rescaling is used when the S distinct subsets are drawn
 * without replacement (partial Fisher-Yates over the C(n,k) subset indices), which is the usual
 * practical choice.  Randomness comes from a documented 64-bit LCG (Knuth's MMIX constants, state
 * mod 2^64 in BigInt) rather than the repo's 16-bit LCG, whose short cycle and weak low bits are
 * unsuitable for tens of thousands of independent draws; the uniform is taken from the top 32 bits of
 * the state.
 *
 * Documented checks: exhaustive frequencies sum to 1 and the counts sum to C(n,k) exactly, and the
 * class multiplicities reproduce an independent brute-force triangle count (3-subsets with 3 edges);
 * for 3 vertices the four canonical forms are pinned by hand ("000", "001", "011", "111") and each is
 * invariant under the input ordering, while the eleven 4-vertex classes are characterised
 * independently by canonicalising all 2^6 = 64 labelled graphs on 4 vertices (exactly 11 distinct
 * forms) and a 12-vertex random graph is asserted to realise all of them; drawing all C(n,k) subsets
 * without replacement reproduces the exhaustive counts exactly, which pins the rescaling and the
 * subset-index bijection, while the same budget with replacement only meets a 3-sigma bound; the
 * estimator is asserted unbiased by averaging 40 independent runs (S = 2000 draws on a fixed graph)
 * and requiring each class mean to sit within 3 sigma of the exact count; single-run deviations are
 * bounded by 3 sigma computed from the exact p; every run's per-class estimates sum back to C(n,k);
 * sampling without replacement is asserted to be at least as accurate as with replacement on the same
 * budget over 30 paired replicates; k > n, an edgeless graph, k = n and k = 0 are handled; and reruns
 * are bit-identical.
 */
"use strict";

const TWO_64 = 1n << 64n;
const LCG64_MULT = 6364136223846793005n;
const LCG64_ADD = 1442695040888963407n;

/** 64-bit LCG (Knuth MMIX), state reduced mod 2^64. */
function lcg64Next(state) {
  return (state * LCG64_MULT + LCG64_ADD) % TWO_64;
}

/** Uniform in [0, 1) from the top 32 bits of the 64-bit state. */
function lcg64Uniform(state) {
  return Number(state >> 32n) / 4294967296;
}

function buildAdjacency(n, edges) {
  const adj = new Array(n).fill(0);
  for (const [u, v] of edges) {
    if (u === v) continue;
    adj[u] |= 1 << v;
    adj[v] |= 1 << u;
  }
  return adj;
}

function popcount(mask) {
  let m = mask;
  let count = 0;
  while (m) {
    m &= m - 1;
    count += 1;
  }
  return count;
}

function randomGraph(n, percent, seedState) {
  const edges = [];
  let state = seedState % TWO_64;
  for (let u = 0; u < n; u++) {
    for (let v = u + 1; v < n; v++) {
      state = lcg64Next(state);
      if (lcg64Uniform(state) * 100 < percent) edges.push([u, v]);
    }
  }
  return { edges, state };
}

/** All k-subsets of {0..n-1} in lexicographic order. */
function subsets(n, k) {
  const out = [];
  const current = [];
  const rec = (start) => {
    if (current.length === k) {
      out.push(current.slice());
      return;
    }
    for (let v = start; v < n; v++) {
      current.push(v);
      rec(v + 1);
      current.pop();
    }
  };
  rec(0);
  return out;
}

function binom(n, k) {
  if (k < 0 || k > n) return 0;
  let result = 1;
  for (let i = 0; i < k; i++) result = (result * (n - i)) / (i + 1);
  return Math.round(result);
}

/**
 * Canonical form of the subgraph induced by `vertices`: the lexicographically smallest upper-triangle
 * bitstring over all orderings of the vertices, as a binary string.  Two induced subgraphs are
 * isomorphic exactly when their canonical forms are equal.
 */
function canonicalForm(adj, vertices) {
  const k = vertices.length;
  let best = null;
  const order = [];
  const used = new Array(k).fill(false);
  // Local adjacency between the positions of `vertices`.
  const local = vertices.map((v) => vertices.map((w) => (adj[v] & (1 << w)) !== 0));
  const rec = () => {
    if (order.length === k) {
      let code = "";
      for (let i = 0; i < k; i++) {
        for (let j = i + 1; j < k; j++) code += local[order[i]][order[j]] ? "1" : "0";
      }
      if (best === null || code < best) best = code;
      return;
    }
    for (let i = 0; i < k; i++) {
      if (used[i]) continue;
      used[i] = true;
      order.push(i);
      rec();
      order.pop();
      used[i] = false;
    }
  };
  rec();
  return best;
}

function classify(adj, vertices) {
  return canonicalForm(adj, vertices);
}

/** Exact induced-k-subgraph class counts. */
function exactCounts(adj, k) {
  const counts = new Map();
  for (const vertices of subsets(adj.length, k)) {
    const form = classify(adj, vertices);
    counts.set(form, (counts.get(form) ?? 0) + 1);
  }
  return counts;
}

function frequencies(counts, total) {
  const out = new Map();
  for (const [form, count] of counts) out.set(form, count / total);
  return out;
}

/**
 * Draw S k-subsets and accumulate class hits.
 * replacement = true draws S subsets independently (uniform over the C(n,k) subsets);
 * replacement = false draws S distinct subsets via a partial Fisher-Yates over the subset indices.
 */
function sampleCounts(adj, k, samples, seedState, replacement = true) {
  const n = adj.length;
  const total = binom(n, k);
  const all = subsets(n, k);
  const hits = new Map();
  let state = seedState % TWO_64;
  if (replacement) {
    for (let i = 0; i < samples; i++) {
      state = lcg64Next(state);
      const index = Math.min(total - 1, Math.floor(lcg64Uniform(state) * total));
      const form = classify(adj, all[index]);
      hits.set(form, (hits.get(form) ?? 0) + 1);
    }
  } else {
    if (samples > total) throw new Error("cannot draw more distinct subsets than exist");
    const indices = Array.from({ length: total }, (_, i) => i);
    for (let i = 0; i < samples; i++) {
      state = lcg64Next(state);
      const j = i + Math.min(total - 1 - i, Math.floor(lcg64Uniform(state) * (total - i)));
      const tmp = indices[i];
      indices[i] = indices[j];
      indices[j] = tmp;
      const form = classify(adj, all[indices[i]]);
      hits.set(form, (hits.get(form) ?? 0) + 1);
    }
  }
  return { hits, state, total, subsetCount: total };
}

/** Rescale hits to counts: hits / S * C(n,k). */
function estimateCounts(hits, samples, total) {
  const out = new Map();
  for (const [form, hit] of hits) out.set(form, (hit / samples) * total);
  return out;
}

/** Total absolute deviation between an estimate map and the exact counts. */
function totalDeviation(estimate, exact) {
  const forms = new Set([...estimate.keys(), ...exact.keys()]);
  let deviation = 0;
  for (const form of forms) deviation += Math.abs((estimate.get(form) ?? 0) - (exact.get(form) ?? 0));
  return deviation;
}

function main() {
  const assert = require("node:assert");

  // (1) Canonical forms for the 3-vertex classes and their exact frequencies on a hand graph.
  const hand = buildAdjacency(6, [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 5], [5, 3]]);
  assert.strictEqual(canonicalForm(buildAdjacency(3, [[0, 1]]), [0, 1, 2]), "001",
    "minimising the bitstring pushes the single edge to the last slot");
  assert.strictEqual(canonicalForm(buildAdjacency(3, []), [0, 1, 2]), "000");
  assert.strictEqual(canonicalForm(buildAdjacency(3, [[0, 1], [1, 2]]), [0, 1, 2]), "011");
  assert.strictEqual(canonicalForm(buildAdjacency(3, [[0, 1], [1, 2], [0, 2]]), [0, 1, 2]), "111");
  // The canonical form does not depend on the input ordering of the vertices.
  const triangle3 = buildAdjacency(3, [[0, 1], [1, 2], [0, 2]]);
  assert.strictEqual(canonicalForm(triangle3, [0, 1, 2]), canonicalForm(triangle3, [2, 0, 1]));
  assert.strictEqual(canonicalForm(triangle3, [1, 2, 0]), "111");
  const path3 = buildAdjacency(3, [[0, 1], [1, 2]]);
  assert.strictEqual(canonicalForm(path3, [0, 1, 2]), canonicalForm(path3, [2, 1, 0]));

  const exact3 = exactCounts(hand, 3);
  const total3 = new Set(subsets(6, 3)).size;
  assert.strictEqual(total3, binom(6, 3));
  assert.strictEqual(total3, 20);
  let sum3 = 0;
  for (const count of exact3.values()) sum3 += count;
  assert.strictEqual(sum3, 20, "the class counts partition the 20 vertex triples");
  // Triangle class: 3-subsets with all three edges; the two K3's contribute 1 each.
  assert.strictEqual(exact3.get("111"), 2, "the graph is K3 + K3 with a path of four between them");
  // Independent triangle count by a triple loop.
  let bruteTriangles = 0;
  for (let a = 0; a < 6; a++) {
    for (let b = a + 1; b < 6; b++) {
      for (let c = b + 1; c < 6; c++) {
        if ((hand[a] & (1 << b)) && (hand[a] & (1 << c)) && (hand[b] & (1 << c))) bruteTriangles += 1;
      }
    }
  }
  assert.strictEqual(bruteTriangles, 2, "brute force agrees with the canonical-form count");
  const freq3 = frequencies(exact3, 20);
  let freqSum = 0;
  for (const value of freq3.values()) freqSum += value;
  assert.ok(Math.abs(freqSum - 1) <= 1e-12);

  // (2) Four vertices: the 11 unlabelled graphs on 4 vertices are all realised by an LCG graph.
  const dense = buildAdjacency(12, randomGraph(12, 50, 20240607n).edges);
  const exact4 = exactCounts(dense, 4);
  assert.strictEqual([...exact4.values()].reduce((a, b) => a + b, 0), binom(12, 4));
  assert.strictEqual(binom(12, 4), 495);
  assert.strictEqual(exact4.size, 11,
    "a 12-vertex random graph with 50% density realises every unlabelled 4-vertex graph");
  // Independent characterisation of the 11 classes: canonicalise all 2^6 labelled graphs on 4
  // vertices - 64 labelled graphs collapse onto exactly 11 unlabelled ones.
  const universe = new Set();
  for (let mask = 0; mask < 64; mask++) {
    const edges = [];
    let bit = 0;
    for (let i = 0; i < 4; i++) {
      for (let j = i + 1; j < 4; j++) {
        if (mask & (1 << bit)) edges.push([i, j]);
        bit += 1;
      }
    }
    universe.add(canonicalForm(buildAdjacency(4, edges), [0, 1, 2, 3]));
  }
  assert.strictEqual(universe.size, 11, "there are exactly 11 unlabelled graphs on 4 vertices");
  for (const form of exact4.keys()) assert.ok(universe.has(form), `class ${form} is a 4-vertex class`);
  assert.strictEqual(universe.size, exact4.size, "the dense graph realises all of them");
  assert.ok(universe.has("000000") && universe.has("111111"), "both the empty and the complete one");
  assert.strictEqual(universe.has("100100"), false,
    "100100 is not canonical: the canonical form of 2K2 (two disjoint edges) is 001100");

  // (3) Sampling every subset without replacement reproduces the exact enumeration; with
  //     replacement the same budget is a random sample, so only a 3-sigma bound can be claimed.
  for (const k of [3, 4]) {
    const exact = k === 3 ? exact3 : exact4;
    const subject = k === 3 ? hand : dense;
    const total = binom(subject.length, k);

    const sweep = sampleCounts(subject, k, total, 1234567n, false);
    assert.strictEqual(totalDeviation(estimateCounts(sweep.hits, total, total), exact), 0,
      `k=${k}: drawing all ${total} subsets without replacement is exactly the exhaustive pass`);
    assert.strictEqual(sweep.hits.size, exact.size, "and every class is hit");

    const withReplacement = sampleCounts(subject, k, total, 1234567n, true);
    const estimate = estimateCounts(withReplacement.hits, total, total);
    let threeSigma = 0;
    for (const [, exactCount] of exact) {
      const p = exactCount / total;
      threeSigma += 3 * Math.sqrt(total * total * p * (1 - p) / total);
    }
    assert.ok(totalDeviation(estimate, exact) <= threeSigma,
      `k=${k}: deviation ${totalDeviation(estimate, exact)} within 3 sigma ${threeSigma}`);
  }

  // (4) Unbiasedness of the with-replacement estimator: the mean of 40 runs sits within 3 sigma.
  const subject = hand;
  const exactGrand = exact3;
  const grand = binom(subject.length, 3);
  const S = 2000;
  const runs = 40;
  const meanOverRuns = new Map();
  let sigmaWorst = 0;
  for (let run = 0; run < runs; run++) {
    const { hits } = sampleCounts(subject, 3, S, 991n + BigInt(run) * 7919n, true);
    const estimate = estimateCounts(hits, S, grand);
    let running = 0;
    for (const [form, value] of estimate) running += value;
    assert.ok(Math.abs(running - grand) <= 1e-9,
      "each run's per-class estimates sum back to C(n,k) exactly");
    for (const [form, value] of estimate) {
      meanOverRuns.set(form, (meanOverRuns.get(form) ?? 0) + value / runs);
    }
  }
  for (const [form, exactCount] of exactGrand) {
    const p = exactCount / grand;
    const sigma = Math.sqrt(grand * grand * p * (1 - p) / S);
    sigmaWorst = Math.max(sigmaWorst, sigma);
    const mean = meanOverRuns.get(form) ?? 0;
    assert.ok(Math.abs(mean - exactCount) <= 3 * sigma,
      `class ${form}: mean estimate ${mean} vs exact ${exactCount} (3 sigma = ${3 * sigma})`);
  }
  assert.ok(sigmaWorst > 0, "the variance is non-zero, otherwise the bias check is vacuous");

  // (5) A single run's deviation is bounded by 3 sigma on every class.
  for (let run = 0; run < 20; run++) {
    const { hits } = sampleCounts(subject, 3, S, 555n + BigInt(run) * 104729n, true);
    const estimate = estimateCounts(hits, S, grand);
    for (const [form, exactCount] of exactGrand) {
      const p = exactCount / grand;
      const sigma = Math.sqrt(grand * grand * p * (1 - p) / S);
      const deviation = Math.abs((estimate.get(form) ?? 0) - exactCount);
      assert.ok(deviation <= 3 * sigma + 1e-9,
        `run ${run} class ${form}: deviation ${deviation} exceeds 3 sigma ${3 * sigma}`);
    }
  }

  // (6) Sampling without replacement beats sampling with replacement on the same budget: the mean
  //     absolute deviation over 30 paired replicates is smaller (the variance of a without-replacement
  //     sample of size S is (1 - S/N) times the with-replacement variance).
  const budget = 400;
  const bigSubject = buildAdjacency(20, randomGraph(20, 40, 55555n).edges);
  const bigExact = exactCounts(bigSubject, 3);
  const bigTotal = binom(20, 3);
  let withError = 0;
  let withoutError = 0;
  for (let replicate = 0; replicate < 30; replicate++) {
    const seed = 31337n + BigInt(replicate) * 61681n;
    const withHits = sampleCounts(bigSubject, 3, budget, seed, true);
    const withoutHits = sampleCounts(bigSubject, 3, budget, seed, false);
    withError += totalDeviation(estimateCounts(withHits.hits, budget, bigTotal), bigExact);
    withoutError += totalDeviation(estimateCounts(withoutHits.hits, budget, bigTotal), bigExact);
  }
  assert.strictEqual(bigTotal, 1140, "C(20,3) = 1140 subsets, comfortably more than the budget");
  assert.ok(withoutError <= withError,
    `without replacement ${withoutError} should be at least as accurate as with replacement ${withError}`);

  // (7) Boundaries: k > n, an edgeless graph, k = n, and the degenerate k = 0 subset.
  const emptyGraph = buildAdjacency(7, []);
  assert.strictEqual(exactCounts(emptyGraph, 3).size, 1, "every triple is edgeless");
  assert.strictEqual(exactCounts(emptyGraph, 3).get("000"), 35);
  assert.strictEqual(exactCounts(emptyGraph, 4).get("000000"), binom(7, 4));
  assert.strictEqual(subsets(7, 8).length, 0);
  assert.strictEqual(binom(7, 8), 0);
  const kEqualsN = exactCounts(hand, 6);
  assert.strictEqual([...kEqualsN.values()].reduce((a, b) => a + b, 0), 1);
  assert.strictEqual(subsets(6, 0).length, 1);
  assert.strictEqual(canonicalForm(hand, []), "");
  assert.throws(() => sampleCounts(hand, 3, 21, 1n, false), /distinct subsets/);
  assert.ok(sampleCounts(hand, 3, 20, 1n, false).hits.size <= 4,
    "a full sweep visits at most the four 3-vertex classes");

  // (8) Determinism.
  const repeatGraph = buildAdjacency(10, randomGraph(10, 45, 777n).edges);
  assert.deepStrictEqual(sampleCounts(repeatGraph, 3, 500, 42n, true),
    sampleCounts(repeatGraph, 3, 500, 42n, true));
  assert.deepStrictEqual(exactCounts(repeatGraph, 3), exactCounts(repeatGraph, 3));
  assert.deepStrictEqual(estimateCounts(sampleCounts(repeatGraph, 3, 100, 42n, false).hits, 100,
    binom(10, 3)), estimateCounts(sampleCounts(repeatGraph, 3, 100, 42n, false).hits, 100,
    binom(10, 3)));
  assert.notDeepStrictEqual(sampleCounts(repeatGraph, 3, 100, 42n, false),
    sampleCounts(repeatGraph, 3, 100, 43n, false), "different seeds give different draws");
}

module.exports = { TWO_64, lcg64Next, lcg64Uniform, buildAdjacency, popcount, randomGraph, subsets,
  binom, canonicalForm, classify, exactCounts, frequencies, sampleCounts, estimateCounts,
  totalDeviation };

if (require.main === module) {
  main();
  console.log("PASS");
}
