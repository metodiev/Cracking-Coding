/**
 * Task 543 - Implement fingerprinting for graph similarity (Weisfeiler-Lehman sketch). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * Colour refinement (1-dimensional Weisfeiler-Lehman, the "naive vertex classification" of
 * Weisfeiler-Leman): vertices start coloured by their degree, and each round replaces a vertex's
 * colour with the rank of the pair (its own colour, the sorted multiset of its neighbours'
 * colours).  Two engines compute the same refinement: a hashing engine that ranks the keys of a
 * round in sorted order (so colour names are canonical and directly comparable across graphs) and
 * a hash-free partition-refinement engine that groups vertices by explicit pairwise multiset
 * comparison; the self-test asserts that both produce identical partitions in every round.  The
 * WL sketch of a graph is the sequence of per-round sorted colour multisets, i.e. the histogram
 * refinement used by the WL subtree kernel.
 *
 * Documented checks: isomorphic graphs always share the sketch (asserted on LCG graphs relabelled
 * by random permutations, cross-checked against a backtracking isomorphism search); the sketch
 * separates the classic non-isomorphic pairs C6 versus two triangles - a pair 1-WL provably
 * cannot distinguish - is asserted to have EQUAL sketches while the brute-force search reports
 * non-isomorphism, and K3,3 versus the triangular prism (both 3-regular on 6 vertices) is the
 * second such witness, so the limitation of 1-WL is documented rather than hidden; the number of
 * colour classes is non-decreasing and refinement stops exactly when the partition is stable; and
 * stars, paths, complete bipartite graphs and empty graphs are hand-verified.
 */
"use strict";

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

function buildAdjacency(n, edges) {
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) {
    adj[u].push(v);
    adj[v].push(u);
  }
  for (const list of adj) list.sort((a, b) => a - b);
  return adj;
}

function edgesOf(adj) {
  const edges = [];
  for (let u = 0; u < adj.length; u++) for (const v of adj[u]) if (u < v) edges.push([u, v]);
  return edges;
}

function relabel(n, edges, permutation) {
  return edges.map(([u, v]) => [permutation[u], permutation[v]]);
}

function randomPermutation(n, seed) {
  const out = [...Array(n).keys()];
  let s = seed % 65537;
  for (let i = n - 1; i > 0; i--) {
    s = lcgNext(s);
    const j = s % (i + 1);
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
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

function degreeSequence(adj) {
  return adj.map((list) => list.length).sort((a, b) => a - b).join(",");
}

/**
 * Colour refinement with canonical colour names.  Round 0 colours by degree; each later round
 * keys a vertex by (own colour, sorted neighbour colours) and new keys receive ranks in sorted
 * key order, which makes the numbering independent of vertex numbering and adjacency order.
 */
function refineByHashing(adj, maxRounds = 100) {
  const n = adj.length;
  let colors = adj.map((list) => list.length);
  const canonicalize = (values) => {
    const distinct = [...new Set(values)].sort((a, b) => a - b);
    const rank = new Map(distinct.map((value, index) => [value, index]));
    return values.map((value) => rank.get(value));
  };
  colors = canonicalize(colors);
  const rounds = [colors.slice()];
  let stable = false;
  for (let round = 1; round <= maxRounds && !stable; round++) {
    const keys = new Array(n);
    for (let u = 0; u < n; u++) {
      const neighbourColors = adj[u].map((v) => colors[v]).sort((a, b) => a - b);
      keys[u] = `${colors[u]}:${neighbourColors.join(",")}`;
    }
    const distinct = [...new Set(keys)].sort();
    const rank = new Map(distinct.map((key, index) => [key, index]));
    const next = keys.map((key) => rank.get(key));
    stable = partitionToString(next) === partitionToString(colors);
    colors = next;
    rounds.push(colors.slice());
  }
  return { rounds, colors, stable };
}

function partitionToString(colors) {
  const groups = new Map();
  colors.forEach((color, vertex) => {
    if (!groups.has(color)) groups.set(color, []);
    groups.get(color).push(vertex);
  });
  return [...groups.values()].map((members) => members.join(",")).sort().join("|");
}

function classCount(colors) {
  return new Set(colors).size;
}

/**
 * Hash-free reference implementation of the same refinement: vertices are grouped by explicit
 * pairwise comparison of (own class, multiset of neighbour classes), so no colour names are
 * invented at all - the resulting partition is what the hashing engine must reproduce.
 */
function refineByDefinition(adj, maxRounds = 100) {
  const n = adj.length;
  const classesOf = (colors) => {
    const classes = new Map();
    colors.forEach((color, vertex) => {
      if (!classes.has(color)) classes.set(color, []);
      classes.get(color).push(vertex);
    });
    return [...classes.values()].map((members) => members.slice().sort((a, b) => a - b))
      .sort((a, b) => (a[0] - b[0]));
  };
  const multisetEqual = (a, b) => {
    if (a.length !== b.length) return false;
    const left = a.slice().sort((x, y) => x - y);
    const right = b.slice().sort((x, y) => x - y);
    return left.every((value, index) => value === right[index]);
  };
  let colors = adj.map((list) => list.length);
  let labels = new Array(n).fill(-1);
  let next = 0;
  const byDegree = new Map();
  colors.forEach((degree, vertex) => {
    if (!byDegree.has(degree)) byDegree.set(degree, next++);
    labels[vertex] = byDegree.get(degree);
  });
  colors = labels.slice();
  const rounds = [colors.slice()];
  for (let round = 1; round <= maxRounds; round++) {
    const representatives = [];
    const fresh = new Array(n).fill(-1);
    for (let u = 0; u < n; u++) {
      const key = [colors[u], adj[u].map((v) => colors[v])];
      let found = -1;
      for (let r = 0; r < representatives.length; r++) {
        if (representatives[r].color === key[0] &&
            multisetEqual(representatives[r].neighbours, key[1])) {
          found = r;
          break;
        }
      }
      if (found === -1) {
        representatives.push({ color: key[0], neighbours: key[1].slice() });
        found = representatives.length - 1;
      }
      fresh[u] = found;
    }
    if (partitionToString(fresh) === partitionToString(colors)) {
      rounds.push(fresh.slice());
      break;
    }
    colors = fresh;
    rounds.push(colors.slice());
  }
  return { rounds, colors, classes: classesOf(colors) };
}

/** The WL sketch: per-round sorted colour multisets, joined. */
function wlSignature(adj, maxRounds = 100) {
  const { rounds } = refineByHashing(adj, maxRounds);
  return rounds.map((colors) => colors.slice().sort((a, b) => a - b).join(",")).join("|");
}

function roundProfile(adj, maxRounds = 100) {
  const { rounds } = refineByHashing(adj, maxRounds);
  return rounds.map((colors) => {
    const sizes = new Map();
    for (const color of colors) sizes.set(color, (sizes.get(color) || 0) + 1);
    return [...sizes.values()].sort((a, b) => a - b);
  });
}

/** Backtracking isomorphism search - the independent ground truth on tiny graphs. */
function isomorphicBrute(adjA, adjB) {
  const n = adjA.length;
  if (adjB.length !== n) return false;
  const setsA = adjA.map((list) => new Set(list));
  const setsB = adjB.map((list) => new Set(list));
  const degA = adjA.map((list) => list.length);
  const degB = adjB.map((list) => list.length);
  const map = new Array(n).fill(-1);
  const used = new Array(n).fill(false);
  const rec = (i) => {
    if (i === n) return true;
    for (let v = 0; v < n; v++) {
      if (used[v] || degA[i] !== degB[v]) continue;
      let ok = true;
      for (let j = 0; j < i; j++) {
        if (setsA[i].has(j) !== setsB[v].has(map[j])) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;
      map[i] = v;
      used[v] = true;
      if (rec(i + 1)) return true;
      used[v] = false;
      map[i] = -1;
    }
    return false;
  };
  return rec(0);
}

function cycleEdges(n, offset = 0, components = 1) {
  const edges = [];
  const size = n / components;
  for (let c = 0; c < components; c++) {
    const base = offset + c * size;
    for (let i = 0; i < size; i++) edges.push([base + i, base + ((i + 1) % size)]);
  }
  return edges;
}

function prismEdges() {
  return [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 3], [1, 4], [2, 5]];
}

function k33Edges() {
  return [[0, 3], [0, 4], [0, 5], [1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]];
}

function main() {
  const assert = require("node:assert");

  // (1) Hand-verified sketches.
  const k3 = buildAdjacency(3, [[0, 1], [1, 2], [2, 0]]);
  assert.strictEqual(wlSignature(k3), "0,0,0|0,0,0", "a triangle is one colour class");
  assert.strictEqual(classCount(refineByHashing(k3).colors), 1);
  const star3 = buildAdjacency(4, [[0, 1], [0, 2], [0, 3]]);
  const refinedStar = refineByHashing(star3);
  assert.strictEqual(classCount(refinedStar.colors), 2, "centre and leaves split apart");
  assert.deepStrictEqual(refinedStar.colors[0], 1, "the centre is the hub");
  const empty = buildAdjacency(5, []);
  assert.strictEqual(classCount(refineByHashing(empty).colors), 1, "no edges: one class");
  const path4 = buildAdjacency(4, [[0, 1], [1, 2], [2, 3]]);
  assert.strictEqual(classCount(refineByHashing(path4).colors), 2,
    "1-WL keeps the two ends together and the two inner vertices together, and no more");
  assert.strictEqual(partitionToString(refineByHashing(path4).colors), "0,3|1,2");

  // (2) The hashing engine and the pairwise partition-refinement engine agree round for round.
  for (let seed = 1; seed <= 5; seed++) {
    const n = 7;
    const { edges } = randomGraph(n, 45, seed * 19);
    const adj = buildAdjacency(n, edges);
    const hashed = refineByHashing(adj);
    const definition = refineByDefinition(adj);
    assert.strictEqual(hashed.rounds.length, definition.rounds.length,
      `seed ${seed}: both engines refine for the same number of rounds`);
    assert.strictEqual(partitionToString(hashed.colors), partitionToString(definition.colors),
      `seed ${seed}: identical final partitions`);
    for (let round = 0; round < hashed.rounds.length; round++) {
      assert.strictEqual(hashed.rounds[round].length, n);
      assert.strictEqual(partitionToString(hashed.rounds[round]),
        partitionToString(definition.rounds[round]), `seed ${seed} round ${round}: partition`);
    }
    assert.deepStrictEqual(refineByDefinition(adj).classes,
      definition.classes, `seed ${seed}: deterministic`);
  }

  // (3) Colour classes only ever get finer, and refinement stops when the partition is stable.
  for (let seed = 11; seed <= 14; seed++) {
    const n = 8;
    const adj = buildAdjacency(n, randomGraph(n, 40, seed * 23).edges);
    const { rounds, stable } = refineByHashing(adj);
    assert.ok(stable, `seed ${seed}: refinement must reach a fixed point`);
    for (let round = 1; round < rounds.length; round++) {
      assert.ok(classCount(rounds[round]) >= classCount(rounds[round - 1]),
        `seed ${seed}: classes are monotone at round ${round}`);
    }
    assert.strictEqual(partitionToString(rounds[rounds.length - 1]),
      partitionToString(rounds[rounds.length - 2]),
      `seed ${seed}: the last recorded round is exactly the stable one`);
    const profiles = roundProfile(adj);
    assert.strictEqual(profiles.length, rounds.length);
    for (const profile of profiles) {
      assert.strictEqual(profile.reduce((a, b) => a + b, 0), n);
    }
  }

  // (4) Isomorphism preserves the sketch; random relabelling never changes it.
  for (let seed = 21; seed <= 28; seed++) {
    const n = 6;
    const { edges } = randomGraph(n, 50, seed * 29);
    const base = buildAdjacency(n, edges);
    const copy = buildAdjacency(n, relabel(n, edges, randomPermutation(n, seed * 31)));
    assert.strictEqual(wlSignature(base), wlSignature(copy),
      `seed ${seed}: relabelling must not change the sketch`);
    assert.ok(isomorphicBrute(base, copy), `seed ${seed}: the brute-force search agrees`);
    assert.strictEqual(degreeSequence(base), degreeSequence(copy));
  }

  // (5) Documented limitation: 1-WL cannot separate these non-isomorphic pairs.
  const c6 = buildAdjacency(6, cycleEdges(6));
  const twoTriangles = buildAdjacency(6, cycleEdges(6, 0, 2));
  assert.strictEqual(wlSignature(c6), wlSignature(twoTriangles),
    "1-WL provably fails on C6 versus 2 x C3");
  assert.ok(!isomorphicBrute(c6, twoTriangles),
    "but the graphs really are non-isomorphic (different component counts)");
  assert.strictEqual(degreeSequence(c6), degreeSequence(twoTriangles));

  const prism = buildAdjacency(6, prismEdges());
  const k33 = buildAdjacency(6, k33Edges());
  assert.strictEqual(wlSignature(prism), wlSignature(k33),
    "1-WL also fails on the prism versus K3,3");
  assert.ok(!isomorphicBrute(prism, k33), "the prism has triangles, K3,3 does not");
  assert.strictEqual(classCount(refineByHashing(prism).colors), 1, "both are 3-regular");
  assert.strictEqual(classCount(refineByHashing(k33).colors), 1);

  // (6) Where 1-WL does separate graphs, the sketch differs.
  const k4 = buildAdjacency(4, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]);
  const paw = buildAdjacency(4, [[0, 1], [1, 2], [2, 0], [2, 3]]);
  assert.notStrictEqual(wlSignature(k4), wlSignature(paw), "K4 versus a triangle plus a leaf");
  assert.ok(!isomorphicBrute(k4, paw));
  let agreements = 0;
  let tested = 0;
  for (let seed = 41; seed <= 56; seed++) {
    const n = 6;
    const first = buildAdjacency(n, randomGraph(n, 40, seed * 37).edges);
    const second = buildAdjacency(n, randomGraph(n, 40, seed * 41 + 7).edges);
    const sameSketch = wlSignature(first) === wlSignature(second);
    const isomorphic = isomorphicBrute(first, second);
    tested += 1;
    if (sameSketch === isomorphic) agreements += 1;
    if (isomorphic) {
      assert.strictEqual(sameSketch, true, `seed ${seed}: isomorphism implies equal sketches`);
    }
  }
  assert.ok(agreements >= tested - 2,
    `the sketch is a strong necessary condition (agreed on ${agreements}/${tested} pairs)`);

  // (7) Reflexivity of the sketch (an equivalence-relation sanity check) and determinism.
  const sample = buildAdjacency(6, prismEdges());
  assert.strictEqual(wlSignature(sample), wlSignature(sample));
  assert.strictEqual(wlSignature(sample), wlSignature(buildAdjacency(6,
    relabel(6, prismEdges(), [3, 0, 4, 1, 5, 2]))));
  assert.deepStrictEqual(refineByHashing(sample), refineByHashing(sample));
  assert.strictEqual(wlSignature(buildAdjacency(1, [])), "0|0",
    "a lone vertex is coloured, then stays put for the recorded stable round");
}

module.exports = { lcgNext, buildAdjacency, edgesOf, relabel, randomPermutation, randomGraph,
  degreeSequence, refineByHashing, refineByDefinition, partitionToString, classCount, wlSignature,
  roundProfile, isomorphicBrute, cycleEdges, prismEdges, k33Edges };

if (require.main === module) {
  main();
  console.log("PASS");
}
