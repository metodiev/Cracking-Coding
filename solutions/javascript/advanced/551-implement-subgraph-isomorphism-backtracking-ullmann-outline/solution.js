/**
 * Task 551 - Implement subgraph isomorphism backtracking (Ullmann outline). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * Ullmann's backtracking algorithm, implemented in full rather than as an outline.  The pattern P
 * (k vertices) and the target G (n vertices) are stored as bitset adjacencies; the search keeps a
 * candidate matrix M[v] = {u : u may host pattern vertex v} and (1) prunes each candidate whose
 * target degree is below the pattern degree, (2) refines the matrix to a fixed point with Ullmann's
 * look-ahead rule - M[v] keeps u only if every pattern neighbour w of v still has a candidate inside
 * the target neighbourhood of u, and, in the induced relation, every pattern non-neighbour of v still
 * has a candidate outside it (refinement against the complement, which is where the "two-matrix"
 * reading of Ullmann comes from) - and (3) backtracks over the pattern vertex with the fewest
 * surviving candidates, re-running the refinement after every assignment.  Two matching relations
 * are supported, because mixing them up is the classic source of wrong counts: monomorphism
 * (non-induced - pattern edges must map to target edges) and induced/isomorphism (pattern edges map
 * to target edges *and* pattern non-edges map to target non-edges).
 *
 * Documented checks.  A brute-force oracle enumerates every injective mapping (all P(n,k) ordered
 * k-tuples) and every C(n,k) vertex subset, so the backtracking result is never trusted on its own:
 * on K3 in K5 the monomorphism count is asserted to be 5*4*3 = 60 (24 for K4) while the induced
 * copy count is C(5,3) = 10 (C(4,3) = 4 for K4).  The textbook identity #induced embeddings =
 * #induced copies * |Aut(P)| is verified against an independently computed |Aut(P)| (itself
 * brute-forced over permutations), and the monomorphism identity #mono(P,G) = sum over k-subsets S
 * of #mono(P, G[S]) is checked as well.  The P3-versus-K4 instance pins the two relations apart:
 * 24 monomorphisms but 0 induced copies.  Soundness of the look-ahead refinement is asserted on 48
 * LCG graph/pattern pairs (24 seeds x 2 relations) by comparing search with refinement against the
 * same search with the refinement disabled and against brute force; the degree prune is shown to be
 * active on a 4-regular pattern vertex inside a 3-regular target; empty patterns, k > n, edgeless
 * patterns, self-loops and disconnected patterns are handled; and reruns are bit-identical.
 */
"use strict";

/** Adjacency bitsets: adj[u] has bit v set iff u-v is an edge. Self-loops are ignored. */
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

function neighbours(mask) {
  const out = [];
  for (let v = 0; v < 32; v++) if (mask & (1 << v)) out.push(v);
  return out;
}

function edgeCount(adj) {
  let count = 0;
  for (const mask of adj) count += popcount(mask);
  return count / 2;
}

/** Deterministic 16-bit LCG used by waves 1-3; sufficient for structural (not statistical) draws. */
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

/** Ullmann's initial candidate matrix: u hosts v only if deg_G(u) >= deg_P(v). */
function initialCandidates(patternAdj, targetAdj) {
  return patternAdj.map((mask) => {
    const need = popcount(mask);
    const row = [];
    for (let u = 0; u < targetAdj.length; u++) {
      if (popcount(targetAdj[u]) >= need) row.push(u);
    }
    return row;
  });
}

/** How many (pattern vertex, target vertex) pairs the degree rule rejects outright. */
function degreePruneRemoves(patternAdj, targetAdj) {
  let removed = 0;
  patternAdj.forEach((mask) => {
    const need = popcount(mask);
    for (let u = 0; u < targetAdj.length; u++) {
      if (popcount(targetAdj[u]) < need) removed += 1;
    }
  });
  return removed;
}

/**
 * Ullmann look-ahead refinement to a fixed point.  A candidate u for pattern vertex v survives only
 * if every pattern neighbour w of v has at least one surviving candidate inside N_G(u).  For the
 * induced relation the very same rule is applied to the complement of G: every pattern non-neighbour
 * w of v must have at least one surviving candidate that is *not* adjacent to u (this is Ullmann's
 * two-matrix refinement; the tempting stronger rule "every surviving candidate of w must be
 * non-adjacent" is unsound - it prunes the identity embedding of P3 into itself, because vertex 1 is
 * a surviving candidate of vertex 2 that happens to be adjacent to u = 1).
 */
function refine(candidates, patternAdj, targetAdj, induced, maxRounds = 100) {
  const k = patternAdj.length;
  let current = candidates.map((row) => row.slice());
  let rounds = 0;
  for (;;) {
    rounds += 1;
    const next = current.map((row) => row.slice());
    let changed = false;
    for (let v = 0; v < k; v++) {
      const kept = [];
      for (const u of current[v]) {
        let ok = true;
        for (let w = 0; w < k && ok; w++) {
          if (w === v) continue;
          const required = (patternAdj[v] & (1 << w)) !== 0;
          const local = current[w];
          if (required) {
            ok = local.some((x) => (targetAdj[u] & (1 << x)) !== 0);
          } else if (induced) {
            ok = local.some((x) => (targetAdj[u] & (1 << x)) === 0);
          }
        }
        if (ok) kept.push(u);
      }
      if (kept.length !== current[v].length) {
        next[v] = kept;
        changed = true;
      }
    }
    current = next;
    if (!changed || rounds >= maxRounds) break;
  }
  return { candidates: current, rounds };
}

function compatibleAssignment(patternAdj, targetAdj, mapping, induced) {
  const k = patternAdj.length;
  for (let v = 0; v < k; v++) {
    if (mapping[v] < 0) continue;
    for (let w = v + 1; w < k; w++) {
      if (mapping[w] < 0) continue;
      const patternEdge = (patternAdj[v] & (1 << w)) !== 0;
      const targetEdge = (targetAdj[mapping[v]] & (1 << mapping[w])) !== 0;
      if (patternEdge !== targetEdge && (patternEdge || induced)) return false;
    }
  }
  return true;
}

/**
 * Count injective mappings phi such that, for every pair (v, w):
 *   non-induced: pattern edge => target edge
 *   induced:     pattern edge <=> target edge
 */
function countEmbeddings(patternAdj, targetAdj, { induced = false, useRefinement = true,
  degreePrune = true } = {}) {
  const k = patternAdj.length;
  const n = targetAdj.length;
  if (k > n) return 0;
  if (k === 0) return 1;

  let candidates = degreePrune ? initialCandidates(patternAdj, targetAdj)
    : patternAdj.map(() => Array.from({ length: n }, (_, u) => u));
  if (candidates.some((row) => row.length === 0)) return 0;

  let total = 0;
  const mapping = new Array(k).fill(-1);
  const used = new Array(n).fill(false);

  const search = (matrix) => {
    let best = -1;
    for (let v = 0; v < k; v++) {
      if (mapping[v] >= 0) continue;
      if (best < 0 || matrix[v].length < matrix[best].length) best = v;
    }
    if (best < 0) {
      total += 1;
      return;
    }
    for (const u of matrix[best]) {
      if (used[u]) continue;
      mapping[best] = u;
      used[u] = true;
      if (compatibleAssignment(patternAdj, targetAdj, mapping, induced)) {
        if (useRefinement) {
          const restricted = matrix.map((row, idx) => (idx === best ? [u] : row.slice()));
          const next = refine(restricted, patternAdj, targetAdj, induced).candidates;
          if (!next.some((row) => row.length === 0)) search(next);
        } else {
          search(matrix);
        }
      }
      mapping[best] = -1;
      used[u] = false;
    }
  };

  search(useRefinement ? refine(candidates, patternAdj, targetAdj, induced).candidates
    : candidates);
  return total;
}

/** Brute-force oracle: every injective mapping, checked directly. */
function countEmbeddingsBruteForce(patternAdj, targetAdj, induced = false) {
  const k = patternAdj.length;
  const n = targetAdj.length;
  if (k > n) return 0;
  if (k === 0) return 1;
  let total = 0;
  const mapping = new Array(k).fill(-1);
  const used = new Array(n).fill(false);
  const rec = (v) => {
    if (v === k) {
      total += 1;
      return;
    }
    for (let u = 0; u < n; u++) {
      if (used[u]) continue;
      mapping[v] = u;
      used[u] = true;
      if (compatibleAssignment(patternAdj, targetAdj, mapping, induced)) rec(v + 1);
      used[u] = false;
      mapping[v] = -1;
    }
  };
  rec(0);
  return total;
}

/** All k-subsets of {0..n-1}, as sorted arrays. */
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

/** Induced subgraph on the given vertex set, re-indexed 0..k-1. */
function inducedAdjacency(adj, vertices) {
  const local = new Map(vertices.map((v, i) => [v, i]));
  return vertices.map((v) => {
    let mask = 0;
    for (const w of vertices) {
      if (w !== v && (adj[v] & (1 << w)) !== 0) mask |= 1 << local.get(w);
    }
    return mask;
  });
}

/** Number of k-subsets S with G[S] isomorphic to P, by brute force over subsets. */
function countInducedCopies(adj, patternAdj, useRefinement = true) {
  const k = patternAdj.length;
  if (k === 0) return 1;
  if (k > adj.length) return 0;
  let copies = 0;
  for (const vertexSet of subsets(adj.length, k)) {
    const sub = inducedAdjacency(adj, vertexSet);
    if (countEmbeddings(patternAdj, sub, { induced: true, useRefinement }) > 0) copies += 1;
  }
  return copies;
}

/** |Aut(P)|: the number of edge-preserving permutations of P onto itself. */
function automorphismCount(patternAdj) {
  return countEmbeddings(patternAdj, patternAdj, { induced: true });
}

function patternFromEdges(k, edges) {
  return buildAdjacency(k, edges);
}

function main() {
  const assert = require("node:assert");

  const triangle = patternFromEdges(3, [[0, 1], [1, 2], [0, 2]]);
  const path3 = patternFromEdges(3, [[0, 1], [1, 2]]);
  const edge = patternFromEdges(2, [[0, 1]]);
  const twoEdgesDisjoint = patternFromEdges(4, [[0, 1], [2, 3]]);

  const k5 = buildAdjacency(5, subsets(5, 2));
  const k4 = buildAdjacency(4, subsets(4, 2));
  const k3 = buildAdjacency(3, subsets(3, 2));

  // (1) Hand cases with known closed forms: K3 and P3 inside K4 / K5.
  assert.strictEqual(countEmbeddings(triangle, k5), 60,
    "K3 in K5: 5 choices for v0, 4 for v1, 3 for v2 = 60 monomorphisms");
  assert.strictEqual(countEmbeddings(triangle, k4), 24);
  assert.strictEqual(countEmbeddings(path3, k4), 24, "24 ordered (a, b, c) with a~b, b~c, a!=c");
  assert.strictEqual(countInducedCopies(k4, triangle), 4,
    "every 3-subset of K4 induces a triangle, so there are C(4,3) = 4 induced copies");
  assert.strictEqual(countInducedCopies(k5, triangle), 10);
  assert.strictEqual(countInducedCopies(k4, path3), 0,
    "K4 has no independent pair, so P3 (which needs one) does not occur induced");
  assert.strictEqual(countEmbeddings(path3, k4, { induced: true }), 0,
    "the induced relation agrees: 24 monomorphisms but no induced embedding");
  assert.strictEqual(countEmbeddings(edge, k5), 20, "ordered pairs of adjacent vertices in K5");
  assert.strictEqual(countEmbeddings(triangle, k3), 6);
  assert.strictEqual(countEmbeddings(triangle, k3, { induced: true }), 6);
  assert.strictEqual(countEmbeddings(triangle, path3), 0, "a triangle is not a subgraph of P3");

  // (2) The two textbook counting identities, with the automorphism group brute-forced.
  assert.strictEqual(automorphismCount(triangle), 6);
  assert.strictEqual(automorphismCount(path3), 2);
  assert.strictEqual(automorphismCount(edge), 2);
  assert.strictEqual(automorphismCount(twoEdgesDisjoint), 8, "4! / 2^2 = 8 automorphisms");
  const target2K2 = buildAdjacency(6, [[0, 1], [2, 3]]);
  for (const [pattern, target] of [[triangle, k5], [triangle, k4], [path3, k4],
    [twoEdgesDisjoint, target2K2], [edge, k5]]) {
    const copies = countInducedCopies(target, pattern);
    const embeddings = countEmbeddings(pattern, target, { induced: true });
    assert.strictEqual(embeddings, copies * automorphismCount(pattern),
      "#induced embeddings = #copies * |Aut(P)|");
  }
  assert.strictEqual(countInducedCopies(target2K2, twoEdgesDisjoint), 1,
    "the only 4-subset inducing two disjoint edges is {0,1,2,3}");
  assert.strictEqual(countEmbeddings(twoEdgesDisjoint, target2K2, { induced: true }), 8);

  // The monomorphism variant: #mono(P, G) = sum over k-subsets S of #mono(P, G[S]).
  for (const [pattern, target] of [[path3, k4], [triangle, k5], [twoEdgesDisjoint, target2K2]]) {
    const k = pattern.length;
    let sum = 0;
    for (const vertexSet of subsets(target.length, k)) {
      sum += countEmbeddings(pattern, inducedAdjacency(target, vertexSet), { induced: false });
    }
    assert.strictEqual(sum, countEmbeddings(pattern, target, { induced: false }),
      "#mono = sum over k-subsets of #mono(P, G[S])");
  }
  // A target mixing two disjoint edges and two isolated vertices, checked against brute force.
  for (const induced of [false, true]) {
    assert.strictEqual(countEmbeddings(twoEdgesDisjoint, target2K2, { induced }),
      countEmbeddingsBruteForce(twoEdgesDisjoint, target2K2, induced),
      `2K2 in a 6-vertex target, induced=${induced}`);
    assert.strictEqual(countEmbeddings(triangle, target2K2, { induced }),
      countEmbeddingsBruteForce(triangle, target2K2, induced));
  }
  assert.strictEqual(countEmbeddings(triangle, target2K2), 0,
    "the only edges of the target are two disjoint ones, so it is triangle-free");

  // (3) Brute force versus Ullmann on many LCG instances, with and without the refinements.
  let instances = 0;
  for (let seed = 1; seed <= 24; seed++) {
    const n = 7;
    const target = buildAdjacency(n, randomGraph(n, 45, seed * 977).edges);
    const pn = 3 + (seed % 2);
    const pattern = buildAdjacency(pn, randomGraph(pn, 40 + (seed % 5) * 10, seed * 131).edges);
    for (const induced of [false, true]) {
      const fast = countEmbeddings(pattern, target, { induced });
      const unrefined = countEmbeddings(pattern, target, { induced, useRefinement: false });
      const brute = countEmbeddingsBruteForce(pattern, target, induced);
      assert.strictEqual(fast, brute,
        `seed ${seed} induced=${induced}: Ullmann ${fast} should equal brute force ${brute}`);
      assert.strictEqual(unrefined, brute,
        `seed ${seed} induced=${induced}: the look-ahead refinement must not change the count`);
      instances += 1;
    }
  }
  assert.strictEqual(instances, 48);

  // (4) The degree prune is genuinely active, and a 4-regular pattern vertex cannot live in a
  //     3-regular target.
  const star4 = patternFromEdges(5, [[0, 1], [0, 2], [0, 3], [0, 4]]);
  const twoTrianglesMatched = buildAdjacency(6, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3],
    [0, 3], [1, 4], [2, 5]]);
  assert.strictEqual(countEmbeddings(star4, twoTrianglesMatched), 0,
    "every target vertex has degree 3, the star's centre needs 4");
  assert.ok(degreePruneRemoves(star4, twoTrianglesMatched) > 0, "the prune rejects candidates");
  assert.strictEqual(degreePruneRemoves(triangle, k5), 0, "an equally dense target prunes nothing");
  assert.strictEqual(countEmbeddings(star4, twoTrianglesMatched, { degreePrune: false }), 0,
    "without the degree prune the search still finds nothing, it is only slower");
  const cycle6 = buildAdjacency(6, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]);
  assert.strictEqual(countInducedCopies(cycle6, path3), 6,
    "C6 induces P3 on exactly 6 vertex triples (pick the centre, then its two neighbours)");
  assert.strictEqual(countEmbeddings(path3, cycle6, { induced: true }), 12,
    "12 = 6 copies x |Aut(P3)| = 2");

  // (5) Boundaries: empty pattern, k > n, edgeless patterns, self-loops.
  assert.strictEqual(countEmbeddings([], k5), 1, "the empty pattern has exactly one empty map");
  assert.strictEqual(countEmbeddings(star4, k4), 0, "k = 5 > n = 4");
  assert.strictEqual(countEmbeddings(patternFromEdges(2, []), k5), 20,
    "an edgeless 2-vertex pattern matches every ordered pair, adjacent or not");
  assert.strictEqual(countEmbeddings(patternFromEdges(2, []), cycle6, { induced: true }), 18,
    "induced, an edgeless pair needs a non-adjacent pair: C6 has 9 of them, each in 2 orders");
  assert.strictEqual(countEmbeddings(new Array(6).fill(0), cycle6), 720,
    "an edgeless 6-vertex pattern uses every ordering of C6: 6! = 720");
  assert.strictEqual(countEmbeddings(new Array(6).fill(0), cycle6, { induced: true }), 0,
    "C6 has no independent set of size 6");
  assert.strictEqual(countEmbeddings(new Array(6).fill(0), twoTrianglesMatched, { induced: true }),
    0, "an independent set can take at most one vertex from each triangle, never six");
  const selfLoopTarget = buildAdjacency(3, [[0, 0], [0, 1], [1, 2]]);
  assert.strictEqual(selfLoopTarget[0] & 1, 0, "self-loops are dropped when building adjacency");
  assert.strictEqual(edgeCount(selfLoopTarget), 2);
  assert.strictEqual(countEmbeddings(triangle, selfLoopTarget), 0,
    "the target is a path, there is no triangle");
  assert.strictEqual(edgeCount(buildAdjacency(4, [[0, 1], [1, 2], [2, 3], [3, 0], [0, 0]])), 4);
  assert.deepStrictEqual(neighbours(buildAdjacency(3, [[0, 2]])[0]), [2]);

  // (6) Triangle counts via the pattern machinery against other structural facts.
  const triangleCount = (adj) => countEmbeddings(triangle, adj);
  assert.strictEqual(triangleCount(k5), 60);
  assert.strictEqual(triangleCount(cycle6), 0, "C6 has no triangles");
  assert.strictEqual(triangleCount(twoTrianglesMatched), 12,
    "K3 + K3 + a perfect matching: 3! = 6 monomorphisms per triangle, the matching adds none");
  assert.strictEqual(countInducedCopies(twoTrianglesMatched, triangle), 2);

  // (7) Determinism of the whole pipeline.
  const repeatAdj = buildAdjacency(8, randomGraph(8, 40, 4242).edges);
  assert.deepStrictEqual(refine(initialCandidates(path3, repeatAdj), path3, repeatAdj, true),
    refine(initialCandidates(path3, repeatAdj), path3, repeatAdj, true));
  assert.strictEqual(countEmbeddings(path3, repeatAdj), countEmbeddings(path3, repeatAdj));
  assert.strictEqual(countInducedCopies(repeatAdj, path3), countInducedCopies(repeatAdj, path3));
}

module.exports = { buildAdjacency, popcount, neighbours, edgeCount, lcgNext, randomGraph,
  initialCandidates, degreePruneRemoves, refine, compatibleAssignment, countEmbeddings,
  countEmbeddingsBruteForce, subsets, inducedAdjacency, countInducedCopies, automorphismCount,
  patternFromEdges };

if (require.main === module) {
  main();
  console.log("PASS");
}
