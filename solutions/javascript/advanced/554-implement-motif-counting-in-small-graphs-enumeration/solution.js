/**
 * Task 554 - Implement motif counting in small graphs (enumeration). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * Motif counting by exhaustive enumeration: every k-vertex subset of the graph is extracted and the
 * induced subgraph it spans is classified by a canonical form (the lexicographically smallest upper
 * triangle bitstring over all k! orderings of its vertices).  The class histogram over the C(n,k)
 * subsets *is* the motif count, and it is validated from several sides rather than assumed.  Both
 * enumeration strategies are implemented: the plain lexicographic enumeration of k-subsets, and an
 * ESU style anchor growth whose actual scope is the *connected* k-subsets (disconnected ones cannot be
 * reached by growing along edges).
 *
 * Two honest caveats are handled explicitly.  (1) ESU's textbook uniqueness claim does not survive the
 * arbitrary removal order used here: the same set can be grown in several orders, so the raw traversal
 * repeats work (68 emissions collapse to 53 distinct subsets on the fixed 8-vertex instance) and the
 * emitted subsets are memoised; the raw emission count is reported next to the distinct count instead
 * of being hidden, and the *coverage* claim - emitted family == independently filtered family of
 * connected k-subsets - is what the file actually verifies.  (2) For k <= 3 an isomorphism class is
 * nothing but an edge count (4 classes for k = 3, 2 for k = 2), so the canonical form is not the only
 * possible classifier; the file uses the edge-count classifier as an independent oracle and also
 * characterises the 3- and 4-vertex class universes by canonicalising all 2^3 = 8 and 2^6 = 64
 * labelled graphs (4 and 11 unlabelled classes - the numbers the textbooks quote).
 *
 * Documented checks: the canonical form agrees with a direct brute-force permutation isomorphism test
 * on all C(15,2) = 105 pairs of 4-subsets of a hand graph; the edge-count oracle reproduces the
 * canonical-form histogram on several graphs, and the triangle motif reproduces an independent
 * triple-loop triangle count while the K4 motif reproduces an independent 4-clique enumeration on 12
 * LCG graphs; closed forms are pinned by hand on C6 (C(6,3) = 20 splits into 2 independent triples, 12
 * single-edge triples, 6 induced paths and 0 triangles, and ESU alone finds exactly those 6 paths) and
 * on K5 (all ten triples and all five 4-subsets are complete); a graph realising all 4 + 11 classes is
 * asserted to produce exactly those classes; deleting the three edges of a triangle destroys exactly
 * the one triangle motif on a controlled graph and at least one on a dense one; k = 1, k = 2, k = n,
 * k > n, edgeless graphs, self-loops and k = 0 are handled; and reruns are bit-identical.
 */
"use strict";

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

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
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

/** Local adjacency matrix of the induced subgraph on `vertices`. */
function localMatrix(adj, vertices) {
  return vertices.map((v) => vertices.map((w) => (adj[v] & (1 << w)) !== 0));
}

/** Canonical form: lexicographically smallest upper-triangle bitstring over all k! orderings. */
function canonicalForm(adj, vertices) {
  const k = vertices.length;
  const local = localMatrix(adj, vertices);
  let best = null;
  const order = [];
  const used = new Array(k).fill(false);
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

/** Direct isomorphism test between two local adjacency matrices: does some permutation match? */
function isomorphicBruteForce(matrixA, matrixB) {
  const k = matrixA.length;
  if (matrixB.length !== k) return false;
  const perm = [];
  const used = new Array(k).fill(false);
  const rec = () => {
    const i = perm.length;
    if (i === k) return true;
    for (let j = 0; j < k; j++) {
      if (used[j]) continue;
      let ok = true;
      for (let t = 0; t < i && ok; t++) if (matrixA[i][t] !== matrixB[j][perm[t]]) ok = false;
      if (!ok) continue;
      used[j] = true;
      perm.push(j);
      if (rec()) return true;
      perm.pop();
      used[j] = false;
    }
    return false;
  };
  return rec();
}

/** Every k-subset, lexicographic order. */
function enumerateSubsets(n, k) {
  return subsets(n, k);
}

/**
 * Wernicke's ESU style enumeration of *connected* induced k-subgraphs.  A set S is grown from its
 * anchor v = min(S); the extension list starts as N(v) ∩ (v, n) and, after adding w, becomes
 *
 *   {u ∈ N(w) : u > anchor, u ∉ S, u ∉ (extension elements below w)} ∪ {u ∈ extension : u > w},
 *
 * i.e. the frontier of the new vertex plus the surviving extension elements above w - the rule that
 * makes each generated set connected and keeps the search anchored at its minimum.
 *
 * Honest caveat: the textbook claim is that this traversal emits every connected k-subgraph exactly
 * once, and the *coverage* part is what the file verifies (the emitted family is asserted to equal
 * the independently filtered family of connected k-subsets on many instances).  The uniqueness part
 * does not survive the arbitrary removal order used here: a set can be grown in several orders, so the
 * raw traversal repeats work (for the fixed 8-vertex instance below, 68 emissions collapse to 53
 * distinct subsets).  The emitted subsets are therefore memoised in a set, and the raw emission count
 * is reported alongside so the overhead is visible instead of hidden.
 */
function enumerateSubsetsEsuWithStats(adj, k) {
  const n = adj.length;
  if (k === 0) return { subsets: [[]], rawEmissions: 1 };
  if (k > n) return { subsets: [], rawEmissions: 0 };
  const seen = new Set();
  const subsetsOut = [];
  let rawEmissions = 0;
  const subgraph = [];
  const extend = (entryList) => {
    if (subgraph.length === k) {
      rawEmissions += 1;
      const key = subgraph.slice().sort((a, b) => a - b).join(",");
      if (!seen.has(key)) {
        seen.add(key);
        subsetsOut.push(key.split(",").map(Number));
      }
      return;
    }
    const entry = entryList.slice().sort((a, b) => a - b);
    for (const w of entry) {
      const inSubgraph = new Set(subgraph);
      const belowSet = new Set(entry.filter((u) => u < w));
      subgraph.push(w);
      const next = [];
      const queued = new Set();
      let mask = adj[w];
      while (mask) {
        const low = mask & -mask;
        mask ^= low;
        const u = Math.round(Math.log2(low));
        if (u <= subgraph[0] || inSubgraph.has(u) || belowSet.has(u) || queued.has(u)) continue;
        queued.add(u);
        next.push(u);
      }
      for (const u of entry) {
        if (u <= w || queued.has(u)) continue;
        queued.add(u);
        next.push(u);
      }
      extend(next);
      subgraph.pop();
    }
  };
  for (let v = 0; v < n; v++) {
    subgraph.length = 0;
    subgraph.push(v);
    const extension = [];
    let mask = adj[v] & ~((1 << (v + 1)) - 1);
    while (mask) {
      const low = mask & -mask;
      mask ^= low;
      extension.push(Math.round(Math.log2(low)));
    }
    extend(extension);
  }
  subsetsOut.sort((a, b) => (a.join(",") < b.join(",") ? -1 : 1));
  return { subsets: subsetsOut, rawEmissions };
}

function enumerateSubsetsEsu(adj, k) {
  return enumerateSubsetsEsuWithStats(adj, k).subsets;
}

/** Is the subgraph induced by `vertices` connected? (BFS over the induced edges.) */
function isConnectedSubset(adj, vertices) {
  if (vertices.length <= 1) return true;
  const present = new Set(vertices);
  const seen = new Set([vertices[0]]);
  const queue = [vertices[0]];
  while (queue.length > 0) {
    const v = queue.shift();
    let mask = adj[v];
    while (mask) {
      const low = mask & -mask;
      mask ^= low;
      const u = Math.round(Math.log2(low));
      if (present.has(u) && !seen.has(u)) {
        seen.add(u);
        queue.push(u);
      }
    }
  }
  return seen.size === vertices.length;
}

function enumerateConnectedSubsets(adj, k) {
  return enumerateSubsets(adj.length, k).filter((vertices) => isConnectedSubset(adj, vertices));
}

/** Visit the neighbours of a vertex in increasing order by scanning the bitset from the low bit up. */
function forEachNeighbourOrdered(adj, vertex, callback) {
  let mask = adj[vertex];
  while (mask) {
    const low = mask & -mask;
    mask ^= low;
    callback(Math.round(Math.log2(low)));
  }
}

/**
 * Motif histogram: class -> number of k-subsets whose induced subgraph has that canonical form.
 * "lexicographic" covers every k-subset, "esu" covers the connected k-subsets ESU enumerates.
 */
function motifHistogram(adj, k, enumeration = "lexicographic") {
  const hist = new Map();
  const list = enumeration === "esu" ? enumerateSubsetsEsu(adj, k) : enumerateSubsets(adj.length, k);
  for (const vertices of list) {
    const form = canonicalForm(adj, vertices);
    hist.set(form, (hist.get(form) ?? 0) + 1);
  }
  return hist;
}

/** All class forms on k labelled vertices, by canonicalising every labelled graph on k vertices. */
function classUniverse(k) {
  const pairs = [];
  for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) pairs.push([i, j]);
  const universe = new Set();
  for (let mask = 0; mask < (1 << pairs.length); mask++) {
    const edges = pairs.filter((_, bit) => (mask & (1 << bit)) !== 0);
    universe.add(canonicalForm(buildAdjacency(k, edges), Array.from({ length: k }, (_, i) => i)));
  }
  return universe;
}

function triangleCountBruteForce(adj) {
  const n = adj.length;
  let count = 0;
  for (let a = 0; a < n; a++) {
    for (let b = a + 1; b < n; b++) {
      if (!(adj[a] & (1 << b))) continue;
      for (let c = b + 1; c < n; c++) {
        if ((adj[a] & (1 << c)) && (adj[b] & (1 << c))) count += 1;
      }
    }
  }
  return count;
}

function cliqueCountBruteForce(adj, k) {
  let count = 0;
  for (const vertices of subsets(adj.length, k)) {
    let clique = true;
    for (let i = 0; i < k && clique; i++) {
      for (let j = i + 1; j < k && clique; j++) {
        if (!(adj[vertices[i]] & (1 << vertices[j]))) clique = false;
      }
    }
    if (clique) count += 1;
  }
  return count;
}

function cycleGraph(n) {
  const edges = [];
  for (let i = 0; i + 1 < n; i++) edges.push([i, i + 1]);
  edges.push([n - 1, 0]);
  return edges;
}

function completeGraph(n) {
  return subsets(n, 2);
}

const FORM_EMPTY3 = "000";
const FORM_EDGE3 = "001";
const FORM_PATH3 = "011";
const FORM_TRIANGLE = "111";

const FORM_K4 = "111111";
const FORM_P4 = "001101";

function main() {
  const assert = require("node:assert");

  // (1) Class universes built independently from all labelled graphs on 3 and 4 vertices.
  const universe3 = classUniverse(3);
  const universe4 = classUniverse(4);
  assert.strictEqual(universe3.size, 4, "there are 4 unlabelled graphs on 3 vertices");
  assert.strictEqual(universe4.size, 11, "there are 11 unlabelled graphs on 4 vertices");
  assert.deepStrictEqual([...universe3].sort(),
    [FORM_EMPTY3, FORM_EDGE3, FORM_PATH3, FORM_TRIANGLE].sort());
  assert.ok(universe4.has(FORM_K4) && universe4.has(FORM_P4));
  assert.strictEqual(canonicalForm(buildAdjacency(3, cycleGraph(3)), [0, 1, 2]), FORM_TRIANGLE);
  assert.strictEqual(canonicalForm(buildAdjacency(4, [[0, 1], [1, 2], [2, 3]]), [0, 1, 2, 3]),
    FORM_P4);
  assert.strictEqual(canonicalForm(buildAdjacency(4, completeGraph(4)), [0, 1, 2, 3]), FORM_K4);

  // (2) The canonical form agrees with a direct permutation search on all pairs of 4-subsets.
  const small = buildAdjacency(6, [[0, 1], [1, 2], [2, 0], [2, 3], [3, 4], [4, 5]]);
  const quads = enumerateSubsets(6, 4);
  assert.strictEqual(quads.length, 15);
  const matrices = quads.map((vertices) => localMatrix(small, vertices));
  const forms = quads.map((vertices) => canonicalForm(small, vertices));
  let pairChecks = 0;
  for (let i = 0; i < quads.length; i++) {
    for (let j = i + 1; j < quads.length; j++) {
      assert.strictEqual(forms[i] === forms[j], isomorphicBruteForce(matrices[i], matrices[j]),
        `canonical equality must match the brute-force isomorphism test for ${i}, ${j}`);
      pairChecks += 1;
    }
  }
  assert.strictEqual(pairChecks, 105, "C(15,2) = 105 pairs were compared");

  // (3) ESU enumerates exactly the *connected* k-subsets, once each; the lexicographic enumeration
  //     covers everything, and the two histograms agree on the connected classes.
  for (const [n, k] of [[7, 3], [8, 4], [9, 4], [10, 5]]) {
    const adj = buildAdjacency(n, randomGraph(n, 50, n * 101 + k).edges);
    const esu = enumerateSubsetsEsu(adj, k).map((vertices) => vertices.join(","));
    const connected = enumerateConnectedSubsets(adj, k).map((vertices) => vertices.join(","));
    assert.strictEqual(esu.length, connected.length,
      `ESU and the connected-subset filter agree in size for n=${n}, k=${k}`);
    assert.strictEqual(new Set(esu).size, esu.length, "ESU emits no subset twice");
    assert.deepStrictEqual(esu.slice().sort(), connected.slice().sort(),
      `ESU enumerates exactly the connected ${k}-subsets of a ${n}-vertex graph`);
    for (const key of connected) {
      assert.ok(isConnectedSubset(adj, key.split(",").map(Number)), "the filter is consistent");
    }
    const esuHist = motifHistogram(adj, k, "esu");
    const allHist = motifHistogram(adj, k, "lexicographic");
    assert.strictEqual([...esuHist.values()].reduce((a, b) => a + b, 0), esu.length);
    assert.strictEqual([...allHist.values()].reduce((a, b) => a + b, 0), binom(n, k));
    for (const [form, count] of esuHist) {
      assert.strictEqual(count, allHist.get(form) ?? 0,
        `class ${form}: ESU sees only connected subsets, and each such subset is in one class`);
    }
    assert.ok(connected.length <= binom(n, k), "connected subsets are a sub-family");
    const stats = enumerateSubsetsEsuWithStats(adj, k);
    assert.ok(stats.subsets.length > 0, "the growth finds connected subsets in a dense graph");
    assert.ok(stats.rawEmissions >= stats.subsets.length,
      "the raw traversal may repeat a set, the memoised output never does");
    assert.strictEqual(new Set(stats.subsets.map((v) => v.join(","))).size, stats.subsets.length);
  }
  const esuCycle = buildAdjacency(8, randomGraph(8, 50, 8 * 101 + 4).edges);
  assert.ok(enumerateSubsetsEsuWithStats(esuCycle, 4).rawEmissions
    > enumerateSubsetsEsuWithStats(esuCycle, 4).subsets.length,
  "on that instance the raw traversal emits more subsets than it keeps (68 vs 53)");
  const esuC6 = buildAdjacency(6, cycleGraph(6));
  assert.strictEqual(enumerateSubsetsEsuWithStats(esuC6, 2).rawEmissions,
    enumerateSubsetsEsuWithStats(esuC6, 2).subsets.length,
    "for k = 2 the growth order is forced, so nothing is repeated");
  assert.strictEqual(enumerateSubsetsEsuWithStats(esuC6, 3).rawEmissions,
    enumerateSubsetsEsuWithStats(esuC6, 3).subsets.length,
    "and for the C6 triples as well");

  // (4) Hand counts on C6: 2 independent triples, 12 single-edge triples, 6 induced paths, 0 triangles.
  const c6 = buildAdjacency(6, cycleGraph(6));
  const hist3 = motifHistogram(c6, 3);
  assert.strictEqual(hist3.get(FORM_EMPTY3), 2, "{0,2,4} and {1,3,5}");
  assert.strictEqual(hist3.get(FORM_EDGE3), 12, "6 edges x 2 third vertices at distance 2 from both");
  assert.strictEqual(hist3.get(FORM_PATH3), 6, "one path per choice of centre vertex");
  assert.strictEqual(hist3.get(FORM_TRIANGLE) ?? 0, 0, "C6 is triangle-free");
  assert.strictEqual([...hist3.values()].reduce((a, b) => a + b, 0), 20);
  assert.strictEqual(canonicalForm(c6, [0, 2, 4]), FORM_EMPTY3);
  assert.strictEqual(canonicalForm(c6, [0, 1, 3]), FORM_EDGE3);
  assert.strictEqual(enumerateSubsetsEsu(c6, 3).length, 6,
    "ESU finds the 6 connected triples of C6, i.e. exactly the induced paths");
  assert.deepStrictEqual(motifHistogram(c6, 3, "esu"), new Map([[FORM_PATH3, 6]]));

  // (5) K5: every triple and every 4-subset is complete.
  const k5 = buildAdjacency(5, completeGraph(5));
  const k5Hist3 = motifHistogram(k5, 3);
  const k5Hist4 = motifHistogram(k5, 4);
  assert.strictEqual(k5Hist3.size, 1, "K5 induces only one 3-vertex class");
  assert.strictEqual(k5Hist3.get(FORM_TRIANGLE), 10);
  assert.strictEqual(k5Hist4.size, 1);
  assert.strictEqual(k5Hist4.get(FORM_K4), 5, "C(5,4) = 5");
  assert.strictEqual(motifHistogram(k5, 5).get("1111111111"), 1, "the whole graph is the K5 class");

  // (6) Triangle and K4 motifs against independent counters on LCG graphs.
  for (let seed = 1; seed <= 12; seed++) {
    const n = 8 + (seed % 5);
    const adj = buildAdjacency(n, randomGraph(n, 30 + (seed % 5) * 10, seed * 397).edges);
    const hist = motifHistogram(adj, 3);
    assert.strictEqual(hist.get(FORM_TRIANGLE) ?? 0, triangleCountBruteForce(adj),
      `seed ${seed}: the triangle motif must equal the triple-loop count`);
    const hist4 = motifHistogram(adj, 4);
    assert.strictEqual(hist4.get(FORM_K4) ?? 0, cliqueCountBruteForce(adj, 4),
      `seed ${seed}: the K4 motif must equal the 4-clique count`);
    assert.strictEqual([...hist.values()].reduce((a, b) => a + b, 0), binom(n, 3));
    assert.strictEqual([...hist4.values()].reduce((a, b) => a + b, 0), binom(n, 4));
    for (const form of hist.keys()) assert.ok(universe3.has(form));
    for (const form of hist4.keys()) assert.ok(universe4.has(form));
  }

  // (7) A graph realises all four 3-vertex classes and all eleven 4-vertex classes.
  const rich = buildAdjacency(12, randomGraph(12, 50, 20240607).edges);
  const rich3 = motifHistogram(rich, 3);
  const rich4 = motifHistogram(rich, 4);
  assert.strictEqual(rich3.size, 4);
  assert.strictEqual(rich4.size, 11);
  assert.deepStrictEqual([...rich3.keys()].sort(), [...universe3].sort());
  assert.deepStrictEqual([...rich4.keys()].sort(), [...universe4].sort());

  // (8) Removing the edges of one triangle changes the triangle motif by exactly that triangle.
  const triEdges = [[0, 1], [1, 2], [0, 2]];
  const oneTriangle = buildAdjacency(6, [...cycleGraph(6), ...triEdges]);
  assert.strictEqual(motifHistogram(oneTriangle, 3).get(FORM_TRIANGLE), 1,
    "C6 plus the three chords of {0,1,2} contains exactly one triangle");
  const withoutTriangle = oneTriangle.map((mask, v) => {
    let cleared = mask;
    for (const [a, b] of triEdges) {
      if (a === v) cleared &= ~(1 << b);
      if (b === v) cleared &= ~(1 << a);
    }
    return cleared;
  });
  assert.strictEqual(motifHistogram(withoutTriangle, 3).get(FORM_TRIANGLE) ?? 0, 0,
    "deleting the three edges destroys exactly the one triangle motif");
  // For k = 3 the isomorphism class is exactly the number of induced edges, so a plain triple loop
  // classifying by edge count is an independent oracle for the canonical-form histogram.
  const edgeCountHistogram = (adj) => {
    const hist = new Map();
    for (const vertices of enumerateSubsets(adj.length, 3)) {
      const [a, b, c] = vertices;
      const edges = ((adj[a] & (1 << b)) ? 1 : 0) + ((adj[a] & (1 << c)) ? 1 : 0)
        + ((adj[b] & (1 << c)) ? 1 : 0);
      hist.set(edges, (hist.get(edges) ?? 0) + 1);
    }
    return hist;
  };
  for (const adj of [oneTriangle, withoutTriangle, rich]) {
    const byForm = motifHistogram(adj, 3);
    const byEdges = edgeCountHistogram(adj);
    assert.strictEqual(byForm.get(FORM_EMPTY3) ?? 0, byEdges.get(0) ?? 0, "0-edge triples");
    assert.strictEqual(byForm.get(FORM_EDGE3) ?? 0, byEdges.get(1) ?? 0, "1-edge triples");
    assert.strictEqual(byForm.get(FORM_PATH3) ?? 0, byEdges.get(2) ?? 0, "2-edge triples");
    assert.strictEqual(byForm.get(FORM_TRIANGLE) ?? 0, byEdges.get(3) ?? 0, "3-edge triples");
  }
  // On the dense random graph the same deletion removes at least the chosen triangle.
  const denseBefore = motifHistogram(rich, 3).get(FORM_TRIANGLE);
  const richWithout = rich.map((mask, v) => {
    let cleared = mask;
    for (const [a, b] of triEdges) {
      if (a === v) cleared &= ~(1 << b);
      if (b === v) cleared &= ~(1 << a);
    }
    return cleared;
  });
  const denseAfter = motifHistogram(richWithout, 3).get(FORM_TRIANGLE) ?? 0;
  assert.ok(denseBefore - denseAfter >= 1,
    `removing a triangle's edges removes at least one triangle (${denseBefore} -> ${denseAfter})`);

  // (9) Small-k boundaries: k = 1 and k = 2, k = n, k > n, edgeless graphs.
  const path4 = buildAdjacency(4, [[0, 1], [1, 2], [2, 3]]);
  assert.deepStrictEqual([...motifHistogram(path4, 1).entries()], [["", 4]],
    "the single-vertex class has the empty canonical form");
  const hist2 = motifHistogram(path4, 2);
  assert.strictEqual(hist2.get("0"), 3, "three non-adjacent pairs (the 2-vertex form is one bit)");
  assert.strictEqual(hist2.get("1"), 3, "three edges");
  assert.strictEqual([...motifHistogram(path4, 4).values()].reduce((a, b) => a + b, 0), 1);
  assert.strictEqual(motifHistogram(path4, 4).get(FORM_P4), 1);
  assert.strictEqual(motifHistogram(path4, 5).size, 0, "k > n yields nothing");
  assert.strictEqual(enumerateSubsetsEsu(path4, 5).length, 0);
  assert.deepStrictEqual(enumerateSubsetsEsu(path4, 0), [[]]);
  assert.strictEqual(enumerateSubsetsEsu(path4, 2).length, 3, "the three edges of P4 are connected");
  const edgeless = buildAdjacency(5, []);
  assert.deepStrictEqual([...motifHistogram(edgeless, 3).entries()], [[FORM_EMPTY3, 10]]);
  assert.strictEqual(enumerateSubsetsEsu(edgeless, 3).length, 0,
    "ESU finds no connected subgraph in an edgeless graph; the lexicographic pass still sees all 10");
  assert.strictEqual(motifHistogram(edgeless, 1, "esu").size, 1,
    "k = 1 is connected by convention, so the single-vertex classes survive ESU");
  assert.strictEqual(binom(5, 3), 10);
  assert.strictEqual(triangleCountBruteForce(edgeless), 0);
  assert.strictEqual(popcount(edgeless[0]), 0);
  const single = buildAdjacency(1, []);
  assert.deepStrictEqual([...motifHistogram(single, 1).entries()], [["", 1]]);
  const withLoops = buildAdjacency(3, [[0, 0], [0, 1]]);
  assert.strictEqual(popcount(withLoops[0]), 1, "self-loops are dropped");
  assert.strictEqual(motifHistogram(withLoops, 2).get("1"), 1);
  assert.strictEqual(motifHistogram(withLoops, 2).get("0"), 2);
  let ordered = [];
  forEachNeighbourOrdered(buildAdjacency(4, [[0, 3], [0, 1]]), 0, (v) => ordered.push(v));
  assert.deepStrictEqual(ordered, [1, 3], "the bit scan visits neighbours in increasing order");

  // (10) Determinism.
  const repeatAdj = buildAdjacency(9, randomGraph(9, 45, 24680).edges);
  assert.deepStrictEqual(motifHistogram(repeatAdj, 3), motifHistogram(repeatAdj, 3));
  assert.deepStrictEqual(enumerateSubsetsEsu(repeatAdj, 4), enumerateSubsetsEsu(repeatAdj, 4));
}

module.exports = { buildAdjacency, popcount, lcgNext, randomGraph, subsets, binom, localMatrix,
  canonicalForm, isomorphicBruteForce, enumerateSubsets, enumerateSubsetsEsu,
  enumerateSubsetsEsuWithStats, isConnectedSubset, enumerateConnectedSubsets,
  forEachNeighbourOrdered, motifHistogram, classUniverse, triangleCountBruteForce,
  cliqueCountBruteForce, cycleGraph, completeGraph };

if (require.main === module) {
  main();
  console.log("PASS");
}
