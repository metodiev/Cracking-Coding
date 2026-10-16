/**
 * Task 552 - Implement VF2 algorithm skeleton for subgraph iso. (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * A working VF2 engine rather than a skeleton of one.  VF2 (Cordella, Foggia, Sansone, Vento) grows a
 * partial mapping phi: V(P) -> V(G) and keeps the mapped "core" of both graphs plus the frontier
 * information VF2 is built around: in every state the next pattern vertex is chosen from the frontier
 * (an unmapped vertex adjacent to the core, higher degree first, so the search never wanders into a
 * disconnected piece that cannot be related to what is already mapped) and its candidates are
 * filtered by two families of rules:
 *   - semantic rules: adjacency to already-mapped vertices must agree with the pattern, and in the
 *     induced relation non-adjacency must agree as well;
 *   - look-ahead rules: R_succ/R_pred (every unmapped pattern neighbour of v needs a feasible free
 *     target neighbour of u, and every unmapped pattern non-neighbour needs a feasible free target
 *     non-neighbour when the relation is induced), the counting form of R_succ/R_pred that compares
 *     the number of free target neighbours of u with the number of unmapped pattern neighbours of v,
 *     and R_new, which requires every free target neighbour of u to be hostable by some unmapped
 *     pattern neighbour of v.
 * Two honest caveats are implemented rather than papered over.  (1) R_new is only applied when every
 * remaining target vertex must be used (#unmapped pattern vertices == #free target vertices, the
 * induced-*isomorphism* setting it was formulated for); applying it unconditionally is unsound and the
 * file demonstrates the exact counterexample - K3 in K5 after 0->0, 1->1 leaves vertex 2 -> 2 with
 * free neighbours 3, 4 and no unmapped pattern neighbour to host them, and the forced rule collapses
 * the true count of 60 embeddings to 0.  A `forceRNew` switch asserts that collapse.  (2) The
 * existence rules are the sound direction only: R_pred for monomorphism asks for a *free* target
 * neighbour of u of sufficient degree, not the full VF2 T_in/T_out bookkeeping.  A three-rung rule
 * ladder (level 0 = semantic only, 1 = existence rules, 2 = + counting rules and the guarded R_new)
 * turns the usual hand-waving about VF2's pruning into a measured quantity.
 *
 * Documented checks: hand cases with closed forms (K3 in K5 and P3 in K5 each have 60 embeddings; C6
 * has 2 * 6 = 12 automorphisms, the dihedral group); a deterministic relabelling of C6 is still
 * reported isomorphic, while C6 versus 2*C3 - same order, same degree sequence - is correctly
 * rejected; every returned mapping is validated explicitly against both relations; 96 LCG
 * pattern/target pairs (pattern 3-5 vertices, target 6-9) are decided identically to an exhaustive
 * injective-mapping oracle at all three levels, with embedding counts matching the oracle exactly, so
 * no look-ahead rule changes an answer; node visits are asserted non-increasing down the ladder and
 * strictly decreasing overall; degree-infeasible instances are rejected before the first node
 * (visits == 0 at every level); empty patterns, k > n, isolated vertices, self-loops and edgeless
 * patterns are handled; and reruns are bit-identical.
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

function degrees(adj) {
  return adj.map(popcount);
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

/** Deterministic Fisher-Yates relabelling (16-bit LCG, adequate for permutations up to n = 32). */
function relabel(edges, n, seed) {
  const perm = Array.from({ length: n }, (_, i) => i);
  let s = seed % 65537;
  for (let i = n - 1; i > 0; i--) {
    s = lcgNext(s);
    const j = s % (i + 1);
    const tmp = perm[i];
    perm[i] = perm[j];
    perm[j] = tmp;
  }
  const inverse = new Array(n);
  perm.forEach((value, index) => {
    inverse[value] = index;
  });
  return { edges: edges.map(([u, v]) => [inverse[u], inverse[v]]), perm };
}

/**
 * Semantic feasibility of assigning pattern vertex v to target vertex u.
 * Non-induced: every mapped neighbour of v must be adjacent to u.
 * Induced: additionally every mapped non-neighbour of v must be non-adjacent to u.
 */
function semanticFeasible(patternAdj, targetAdj, mapping, v, u, induced) {
  const k = patternAdj.length;
  for (let w = 0; w < k; w++) {
    const image = mapping[w];
    if (image < 0) continue;
    const patternEdge = (patternAdj[v] & (1 << w)) !== 0;
    const targetEdge = (targetAdj[u] & (1 << image)) !== 0;
    if (patternEdge && !targetEdge) return false;
    if (induced && !patternEdge && targetEdge) return false;
  }
  return true;
}

/** Bits of the free (still unmapped) target vertices, as a mask. */
function freeMask(usedTarget) {
  let mask = 0;
  for (let x = 0; x < usedTarget.length; x++) if (!usedTarget[x]) mask |= 1 << x;
  return mask;
}

/**
 * R_new may only be applied when every remaining target vertex has to be used: the number of
 * unmapped pattern vertices after v -> u must equal the number of free target vertices after u is
 * consumed.  That is the induced-isomorphism setting (k == n); see the header counterexample.
 */
function rNewApplicable(patternAdj, mapping, v, usedTarget) {
  let unmappedPattern = 1;
  for (let w = 0; w < patternAdj.length; w++) if (w !== v && mapping[w] < 0) unmappedPattern += 1;
  let freeTarget = 0;
  for (const used of usedTarget) if (!used) freeTarget += 1;
  return unmappedPattern === freeTarget;
}

/**
 * Look-ahead (syntactic) feasibility for v -> u.
 * level 0: no look-ahead (semantic rules only).
 * level 1: existence rules (R_succ/R_pred, plus the induced complement variant).
 * level 2: level 1 plus the counting rules and the guarded R_new.
 * forceRNew: apply R_new even when rNewApplicable says it is unsound (used to exhibit the collapse).
 */
function lookaheadFeasible(patternAdj, targetAdj, mapping, v, u, usedTarget, induced, level,
  forceRNew = false) {
  if (level <= 0) return true;
  const k = patternAdj.length;
  const free = freeMask(usedTarget);

  let unmappedNeighbours = 0;
  for (let w = 0; w < k; w++) {
    if (w === v || mapping[w] >= 0) continue;
    if ((patternAdj[v] & (1 << w)) !== 0) unmappedNeighbours += 1;
  }
  let freeNeighbours = 0;
  let bit = free;
  while (bit) {
    const low = bit & -bit;
    bit ^= low;
    if ((targetAdj[u] & low) !== 0) freeNeighbours += 1;
  }

  // Existence rules: each unmapped pattern neighbour (non-neighbour, when induced) needs a free
  // target vertex of the same relation to u whose degree can host the pattern degree.
  for (let w = 0; w < k; w++) {
    if (w === v || mapping[w] >= 0) continue;
    const required = (patternAdj[v] & (1 << w)) !== 0;
    if (!required && !induced) continue;
    const need = popcount(patternAdj[w]);
    let found = false;
    let rest = free;
    while (rest) {
      const low = rest & -rest;
      rest ^= low;
      const candidate = Math.round(Math.log2(low));
      if (popcount(targetAdj[candidate]) < need) continue;
      if (required === ((targetAdj[u] & low) !== 0)) {
        found = true;
        break;
      }
    }
    if (!found) return false;
  }

  if (level >= 2) {
    if (freeNeighbours < unmappedNeighbours) return false;
    const useRNew = induced && (forceRNew || rNewApplicable(patternAdj, mapping, v, usedTarget));
    if (useRNew) {
      let rest = free;
      while (rest) {
        const low = rest & -rest;
        rest ^= low;
        if ((targetAdj[u] & low) === 0) continue;
        const candidate = Math.round(Math.log2(low));
        let usable = false;
        for (let w = 0; w < k && !usable; w++) {
          if (w === v || mapping[w] >= 0) continue;
          if ((patternAdj[v] & (1 << w)) === 0) continue;
          if (popcount(targetAdj[candidate]) >= popcount(patternAdj[w])) usable = true;
        }
        if (!usable) return false;
      }
    }
  }
  return true;
}

/** Frontier-aware vertex selection: prefer unmapped vertices adjacent to the core, then high degree. */
function selectPatternVertex(patternAdj, mapping, k) {
  let best = -1;
  let bestInFrontier = -1;
  let bestDegree = -1;
  for (let v = 0; v < k; v++) {
    if (mapping[v] >= 0) continue;
    let inFrontier = 0;
    for (let w = 0; w < k; w++) {
      if (w !== v && mapping[w] >= 0 && (patternAdj[v] & (1 << w)) !== 0) {
        inFrontier = 1;
        break;
      }
    }
    const degree = popcount(patternAdj[v]);
    if (best < 0 || inFrontier > bestInFrontier
      || (inFrontier === bestInFrontier && degree > bestDegree)) {
      best = v;
      bestInFrontier = inFrontier;
      bestDegree = degree;
    }
  }
  return best;
}

/**
 * Enumerate every embedding of the pattern into the target.
 * Returns { count, firstMapping, visits }; visits counts the nodes the search actually entered.
 */
function vf2Enumerate(patternAdj, targetAdj, { induced = false, level = 2, stopAfterFirst = false,
  forceRNew = false } = {}) {
  const k = patternAdj.length;
  const n = targetAdj.length;
  const result = { count: 0, firstMapping: null, visits: 0 };
  if (k > n) return result;
  if (k === 0) {
    result.count = 1;
    result.firstMapping = [];
    return result;
  }

  const patternDeg = degrees(patternAdj);
  const targetDeg = degrees(targetAdj);
  const candidates = patternAdj.map((_, v) => {
    const row = [];
    for (let u = 0; u < n; u++) if (targetDeg[u] >= patternDeg[v]) row.push(u);
    return row;
  });
  if (candidates.some((row) => row.length === 0)) return result;

  const mapping = new Array(k).fill(-1);
  const usedTarget = new Array(n).fill(false);
  let done = false;
  const search = (depth) => {
    result.visits += 1;
    if (depth === k) {
      result.count += 1;
      if (result.firstMapping === null) result.firstMapping = mapping.slice();
      if (stopAfterFirst) done = true;
      return;
    }
    const v = selectPatternVertex(patternAdj, mapping, k);
    for (const u of candidates[v]) {
      if (done) return;
      if (usedTarget[u]) continue;
      if (!semanticFeasible(patternAdj, targetAdj, mapping, v, u, induced)) continue;
      if (!lookaheadFeasible(patternAdj, targetAdj, mapping, v, u, usedTarget, induced, level,
        forceRNew)) {
        continue;
      }
      mapping[v] = u;
      usedTarget[u] = true;
      search(depth + 1);
      usedTarget[u] = false;
      mapping[v] = -1;
    }
  };
  search(0);
  return result;
}

function vf2IsSubgraph(patternAdj, targetAdj, induced = false) {
  return vf2Enumerate(patternAdj, targetAdj, { induced, stopAfterFirst: true }).count > 0;
}

function verifyMapping(patternAdj, targetAdj, mapping, induced) {
  const k = patternAdj.length;
  if (mapping.length !== k) return false;
  if (new Set(mapping).size !== k) return false;
  for (let v = 0; v < k; v++) {
    if (mapping[v] < 0 || mapping[v] >= targetAdj.length) return false;
    for (let w = v + 1; w < k; w++) {
      const patternEdge = (patternAdj[v] & (1 << w)) !== 0;
      const targetEdge = (targetAdj[mapping[v]] & (1 << mapping[w])) !== 0;
      if (patternEdge && !targetEdge) return false;
      if (induced && !patternEdge && targetEdge) return false;
    }
  }
  return true;
}

/** Exhaustive oracle: enumerate all injective mappings and check the relation directly. */
function bruteForceCount(patternAdj, targetAdj, induced = false) {
  const k = patternAdj.length;
  const n = targetAdj.length;
  if (k > n) return 0;
  if (k === 0) return 1;
  let count = 0;
  const mapping = new Array(k).fill(-1);
  const used = new Array(n).fill(false);
  const rec = (v) => {
    if (v === k) {
      count += 1;
      return;
    }
    for (let u = 0; u < n; u++) {
      if (used[u]) continue;
      mapping[v] = u;
      used[u] = true;
      if (semanticFeasible(patternAdj, targetAdj, mapping, v, u, induced)) rec(v + 1);
      used[u] = false;
      mapping[v] = -1;
    }
  };
  rec(0);
  return count;
}

function patternFromEdges(k, edges) {
  return buildAdjacency(k, edges);
}

function pathGraph(n) {
  const edges = [];
  for (let i = 0; i + 1 < n; i++) edges.push([i, i + 1]);
  return edges;
}

function cycleGraph(n) {
  const edges = pathGraph(n);
  edges.push([n - 1, 0]);
  return edges;
}

function main() {
  const assert = require("node:assert");

  const triangle = patternFromEdges(3, cycleGraph(3));
  const path3 = patternFromEdges(3, pathGraph(3));
  const k5 = buildAdjacency(5, [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3],
    [2, 4], [3, 4]]);
  const k3 = patternFromEdges(3, cycleGraph(3));

  // (1) Hand cases with closed forms.
  assert.ok(vf2IsSubgraph(k3, k5), "K3 is a subgraph of K5");
  assert.ok(vf2IsSubgraph(k3, k5, true), "and an induced subgraph too");
  assert.ok(!vf2IsSubgraph(k5, k3), "K5 does not fit inside K3");
  assert.ok(!vf2IsSubgraph(triangle, path3), "a triangle is not a subgraph of P3");
  assert.strictEqual(vf2Enumerate(k3, k5).count, 60,
    "5 choices for phi(0), 4 for phi(1), 3 for phi(2)");
  assert.strictEqual(vf2Enumerate(path3, k5).count, 60, "5 centres x 4 x 3 = 60");
  assert.strictEqual(vf2Enumerate(path3, k5).count, bruteForceCount(path3, k5));
  assert.strictEqual(vf2Enumerate(k3, k5, { induced: true }).count, 60);
  assert.strictEqual(vf2Enumerate(triangle, path3).count, 0);

  // (2) R_new is sound only when every remaining target vertex must be used; forcing it otherwise
  //     destroys correct answers, which is asserted here rather than hidden.
  assert.strictEqual(rNewApplicable(k3, [-1, -1, -1], 0, [false, false, false, false, false]),
    false, "K3 has 3 vertices, K5 has 5: the vertices of K5 need not all be used");
  assert.strictEqual(vf2Enumerate(k3, k5, { induced: true, forceRNew: true }).count, 0,
    "the unguarded R_new collapses the true 60 embeddings of K3 in K5 to 0");
  const c6same = patternFromEdges(6, cycleGraph(6));
  assert.strictEqual(rNewApplicable(c6same, [-1, -1, -1, -1, -1, -1], 0,
    [false, false, false, false, false, false]), true,
    "on an isomorphic pair every remaining target vertex must be used");
  assert.strictEqual(vf2Enumerate(c6same, c6same, { induced: true }).count,
    vf2Enumerate(c6same, c6same, { induced: true, forceRNew: true }).count,
    "where it applies, R_new guards an already-sound rule: the count is unchanged");

  // (3) Isomorphism versus non-isomorphism with equal order and degree sequence.
  const c6 = patternFromEdges(6, cycleGraph(6));
  const twoTriangles = buildAdjacency(6, [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]]);
  assert.ok(vf2IsSubgraph(c6, c6, true), "a graph is isomorphic to itself");
  assert.strictEqual(vf2Enumerate(c6, c6, { induced: true }).count, 12,
    "C6 has 2 * 6 = 12 automorphisms: the dihedral group of order 12");
  assert.strictEqual(vf2Enumerate(c6, c6).count, 12,
    "non-induced it is still only the isomorphisms, C6 has no chords");
  assert.ok(!vf2IsSubgraph(c6, twoTriangles, true),
    "C6 and 2*C3 share order and degree sequence but are not isomorphic");
  assert.ok(!vf2IsSubgraph(twoTriangles, c6, true));
  assert.ok(!vf2IsSubgraph(c6, twoTriangles),
    "a 6-cycle cannot even map into two triangles: the image of a connected graph is connected");
  const relabelled = relabel(cycleGraph(6), 6, 99);
  assert.ok(vf2IsSubgraph(c6, buildAdjacency(6, relabelled.edges), true),
    "a deterministic relabelling must not change the answer");
  assert.strictEqual(vf2Enumerate(c6, buildAdjacency(6, relabelled.edges), { induced: true }).count,
    12);
  assert.notDeepStrictEqual(relabelled.perm, [0, 1, 2, 3, 4, 5], "the permutation is non-trivial");

  // (4) The returned mapping is a genuine embedding for both relations.
  const found = vf2Enumerate(path3, k5, { stopAfterFirst: true });
  assert.strictEqual(found.count, 1);
  assert.ok(verifyMapping(path3, k5, found.firstMapping, false));
  const inducedFound = vf2Enumerate(triangle, k5, { induced: true, stopAfterFirst: true });
  assert.ok(verifyMapping(triangle, k5, inducedFound.firstMapping, true));
  assert.ok(!verifyMapping(triangle, k5, [0, 1, 1], true), "a mapping must be injective");
  const matching8 = buildAdjacency(8, [[0, 1], [2, 3], [4, 5], [6, 7]]);
  assert.ok(!verifyMapping(triangle, matching8, [0, 1, 4], false), "0-1 is an edge, 1-4 is not");

  // (5) VF2 versus the exhaustive oracle on 96 random pairs, across the whole rule ladder.
  let decided = 0;
  const visits = [0, 0, 0];
  for (let seed = 1; seed <= 48; seed++) {
    const pn = 3 + (seed % 3);
    const n = 6 + (seed % 4);
    const pattern = buildAdjacency(pn, randomGraph(pn, 40 + (seed % 4) * 12, seed * 613).edges);
    const target = buildAdjacency(n, randomGraph(n, 45, seed * 271).edges);
    for (const useInduced of [false, true]) {
      const oracle = bruteForceCount(pattern, target, useInduced);
      for (const level of [0, 1, 2]) {
        const run = vf2Enumerate(pattern, target, { induced: useInduced, level });
        assert.strictEqual(run.count, oracle,
          `seed ${seed} induced=${useInduced} level=${level}: VF2 ${run.count} vs oracle ${oracle}`);
        if (run.count > 0) {
          assert.ok(verifyMapping(pattern, target, run.firstMapping, useInduced),
            `seed ${seed} induced=${useInduced} level=${level}: the mapping must be valid`);
        } else {
          assert.strictEqual(run.firstMapping, null);
        }
        visits[level] += run.visits;
      }
      decided += 1;
    }
  }
  assert.strictEqual(decided, 96);
  assert.ok(visits[1] <= visits[0],
    `the existence rules must not add nodes (${visits[1]} vs ${visits[0]})`);
  assert.ok(visits[2] <= visits[1],
    `the counting rules must not add nodes (${visits[2]} vs ${visits[1]})`);
  assert.ok(visits[2] < visits[0], "overall the rule ladder prunes strictly on this corpus");
  assert.ok(visits[2] < visits[1] || visits[1] < visits[0], "at least one rung does real work");

  // (6) Degree-infeasible instances die before the first node.
  const star = patternFromEdges(4, [[0, 1], [0, 2], [0, 3]]);
  assert.ok(!vf2IsSubgraph(star, matching8), "no vertex of a perfect matching has degree 3");
  assert.strictEqual(vf2Enumerate(star, matching8).visits, 0,
    "the initial degree filter rejects the whole instance before any node is entered");
  assert.strictEqual(vf2Enumerate(star, matching8, { level: 0 }).visits, 0,
    "the degree filter works independently of the look-ahead level");

  // (7) Boundaries.
  assert.strictEqual(vf2Enumerate([], k5).count, 1, "the empty pattern embeds exactly once");
  assert.deepStrictEqual(vf2Enumerate([], k5).firstMapping, []);
  assert.strictEqual(vf2Enumerate(star, patternFromEdges(2, [[0, 1]])).count, 0, "k = 4 > n = 2");
  const single = patternFromEdges(1, []);
  assert.strictEqual(vf2Enumerate(single, matching8).count, 8, "one pattern vertex, any host");
  assert.strictEqual(vf2Enumerate(single, matching8, { induced: true }).count, 8);
  assert.deepStrictEqual(vf2Enumerate(single, matching8).firstMapping, [0]);
  assert.strictEqual(vf2Enumerate(patternFromEdges(2, []), k5).count, 20,
    "an edgeless pair matches every ordered pair in K5");
  assert.strictEqual(vf2Enumerate(patternFromEdges(2, []), k5, { induced: true }).count, 0,
    "K5 has no non-adjacent pair, so the induced reading finds nothing");
  assert.strictEqual(vf2Enumerate(patternFromEdges(2, []), c6, { induced: true }).count, 18,
    "C6 has 9 non-adjacent pairs, each obtainable in 2 orders");
  assert.ok(verifyMapping([], k5, [], false));
  const withLoops = buildAdjacency(4, [[0, 0], [0, 1], [1, 2], [2, 3]]);
  assert.strictEqual(popcount(withLoops[0]), 1, "self-loops are dropped when building adjacency");
  assert.strictEqual(vf2Enumerate(triangle, withLoops).count, 0);
  assert.strictEqual(vf2Enumerate(path3, twoTriangles, { induced: true }).count, 0,
    "2*C3 induces no P3: every vertex has degree 2 inside its own triangle");

  // (8) Determinism.
  const repeatTarget = buildAdjacency(7, randomGraph(7, 50, 8080).edges);
  const repeatPattern = patternFromEdges(4, randomGraph(4, 50, 5150).edges);
  assert.deepStrictEqual(vf2Enumerate(repeatPattern, repeatTarget),
    vf2Enumerate(repeatPattern, repeatTarget));
  assert.strictEqual(vf2IsSubgraph(repeatPattern, repeatTarget),
    vf2IsSubgraph(repeatPattern, repeatTarget));
}

module.exports = { buildAdjacency, popcount, degrees, lcgNext, randomGraph, relabel,
  semanticFeasible, freeMask, rNewApplicable, lookaheadFeasible, selectPatternVertex, vf2Enumerate,
  vf2IsSubgraph, verifyMapping, bruteForceCount, patternFromEdges, pathGraph, cycleGraph };

if (require.main === module) {
  main();
  console.log("PASS");
}
