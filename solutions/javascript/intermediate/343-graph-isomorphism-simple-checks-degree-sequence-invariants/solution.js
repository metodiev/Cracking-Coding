/**
 * Task 343 - Graph isomorphism: simple checks with degree-sequence invariants. (Tier: Intermediate, CTCI Ch 4)
 *
 * Necessary conditions, deliberately not sufficient: equal sorted degree sequences, equal triangle
 * counts and equal connected-component size multisets.  C6 vs C3+C3 fails on components, K3,3 vs
 * the triangular prism fails on triangles, and the documented counterexample (cube Q3 vs Moebius
 * ladder V8) passes all three checks yet the graphs are not isomorphic - Q3 is bipartite and V8
 * is not (asserted below).
 */
"use strict";

function degrees(adj) {
  return adj.map((neighbours) => new Set(neighbours).size).sort((a, b) => a - b);
}

function triangleCount(adj) {
  const sets = adj.map((neighbours) => new Set(neighbours));
  let count = 0;
  for (let u = 0; u < adj.length; u++) {
    for (const v of sets[u]) {
      if (v <= u) continue;
      for (const w of sets[u]) {
        if (w > v && sets[v].has(w)) count++;
      }
    }
  }
  return count;
}

function componentSizes(adj) {
  const seen = new Array(adj.length).fill(false);
  const sizes = [];
  for (let start = 0; start < adj.length; start++) {
    if (seen[start]) continue;
    seen[start] = true;
    const stack = [start];
    let size = 0;
    while (stack.length > 0) {
      const u = stack.pop();
      size++;
      for (const v of adj[u]) {
        if (!seen[v]) {
          seen[v] = true;
          stack.push(v);
        }
      }
    }
    sizes.push(size);
  }
  return sizes.sort((a, b) => a - b);
}

/** Independent oracle for the component multiset. */
function componentSizesUnionFind(adj) {
  const parent = Array.from({ length: adj.length }, (_, i) => i);
  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  for (let u = 0; u < adj.length; u++) {
    for (const v of adj[u]) {
      const rootU = find(u);
      const rootV = find(v);
      if (rootU !== rootV) parent[rootU] = rootV;
    }
  }
  const counts = new Map();
  for (let i = 0; i < adj.length; i++) {
    const root = find(i);
    counts.set(root, (counts.get(root) || 0) + 1);
  }
  return [...counts.values()].sort((a, b) => a - b);
}

function checkIsomorphic(a, b) {
  if (a.length !== b.length) return false;
  if (!arraysEqual(degrees(a), degrees(b))) return false;
  if (triangleCount(a) !== triangleCount(b)) return false;
  if (!arraysEqual(componentSizes(a), componentSizes(b))) return false;
  return true;
}

function diameter(adj) {
  let best = 0;
  for (let start = 0; start < adj.length; start++) {
    const dist = new Array(adj.length).fill(-1);
    dist[start] = 0;
    const queue = [start];
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head];
      for (const v of adj[u]) {
        if (dist[v] < 0) {
          dist[v] = dist[u] + 1;
          queue.push(v);
        }
      }
    }
    for (const d of dist) {
      if (d < 0) return -1;
      best = Math.max(best, d);
    }
  }
  return best;
}

/** perm maps old vertex -> new vertex. */
function relabel(adj, perm) {
  const out = new Array(adj.length);
  for (let u = 0; u < adj.length; u++) {
    out[perm[u]] = [...new Set(adj[u].map((v) => perm[v]))].sort((a, b) => a - b);
  }
  return out;
}

function inverse(perm) {
  const out = new Array(perm.length);
  for (let old = 0; old < perm.length; old++) out[perm[old]] = old;
  return out;
}

function isBipartite(adj) {
  const color = new Array(adj.length).fill(-1);
  for (let start = 0; start < adj.length; start++) {
    if (color[start] >= 0) continue;
    color[start] = 0;
    const queue = [start];
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head];
      for (const v of adj[u]) {
        if (color[v] < 0) {
          color[v] = 1 - color[u];
          queue.push(v);
        } else if (color[v] === color[u]) {
          return false;
        }
      }
    }
  }
  return true;
}

function arraysEqual(a, b) {
  return a.length === b.length && a.every((value, i) => value === b[i]);
}

function normalize(adj) {
  return adj.map((neighbours) => [...new Set(neighbours)].sort((a, b) => a - b));
}

function cycleGraph(n) {
  return normalize(Array.from({ length: n }, (_, i) => [(i - 1 + n) % n, (i + 1) % n]));
}

function cubeQ3() {
  return normalize(Array.from({ length: 8 }, (_, i) => [i ^ 1, i ^ 2, i ^ 4]));
}

function mobiusV8() {
  const adj = Array.from({ length: 8 }, () => []);
  for (let i = 0; i < 8; i++) {
    adj[i].push((i + 1) % 8);
    adj[(i + 1) % 8].push(i);
  }
  for (let i = 0; i < 4; i++) {
    adj[i].push(i + 4);
    adj[i + 4].push(i);
  }
  return normalize(adj);
}

class Lcg {
  constructor(seed) {
    this.state = seed;
  }

  next() {
    this.state = (this.state * 75 + 74) % 65537;
    return this.state;
  }
}

function lcgGraph(n, rng) {
  const adj = Array.from({ length: n }, () => []);
  for (let u = 0; u < n; u++) {
    for (let v = u + 1; v < n; v++) {
      if (rng.next() % 2 === 0) {
        adj[u].push(v);
        adj[v].push(u);
      }
    }
  }
  return normalize(adj);
}

function lcgPermutation(n, rng) {
  const perm = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = rng.next() % (i + 1);
    [perm[i], perm[j]] = [perm[j], perm[i]];
  }
  return perm;
}

module.exports = { degrees, triangleCount, componentSizes, componentSizesUnionFind, checkIsomorphic, diameter,
  relabel, inverse, isBipartite, cycleGraph, cubeQ3, mobiusV8, lcgGraph, lcgPermutation, Lcg };

if (require.main === module) {
  const assert = require("node:assert");

  const c5 = cycleGraph(5);
  const c6 = cycleGraph(6);
  const path4 = normalize([[1], [0, 2], [1, 3], [2]]);
  const star4 = normalize([[1, 2, 3, 4], [0], [0], [0], [0]]);
  const bowtie = normalize([[1, 2, 3, 4], [0, 2], [0, 1], [0, 4], [0, 3]]);

  assert.ok(checkIsomorphic(c5, c5));
  const pentagram = relabel(c5, [0, 2, 4, 1, 3]);
  assert.deepStrictEqual(pentagram, [[2, 3], [3, 4], [0, 4], [0, 1], [1, 2]]);
  assert.notDeepStrictEqual(pentagram, c5);
  assert.ok(checkIsomorphic(c5, pentagram), "C5 vs pentagram");
  assert.ok(checkIsomorphic(path4, relabel(path4, [2, 0, 3, 1])));
  assert.ok(checkIsomorphic(star4, relabel(star4, [4, 0, 1, 2, 3])));
  assert.ok(checkIsomorphic(bowtie, relabel(bowtie, [3, 1, 4, 0, 2])));
  assert.ok(checkIsomorphic([], []));
  assert.ok(checkIsomorphic([[]], [[]]));

  const star3 = normalize([[1, 2, 3], [0], [0], [0]]);
  assert.ok(!checkIsomorphic(star3, path4), "star K1,3 vs P4");
  assert.ok(!checkIsomorphic(bowtie, c5), "bowtie vs C5");

  const twoTriangles = normalize([[1, 2], [0, 2], [0, 1], [4, 5], [3, 5], [3, 4]]);
  assert.deepStrictEqual(degrees(c6), degrees(twoTriangles));
  assert.deepStrictEqual(degrees(c6), [2, 2, 2, 2, 2, 2]);
  assert.strictEqual(triangleCount(c6), 0);
  assert.strictEqual(triangleCount(twoTriangles), 2);
  assert.deepStrictEqual(componentSizes(c6), [6]);
  assert.deepStrictEqual(componentSizes(twoTriangles), [3, 3]);
  assert.ok(!checkIsomorphic(c6, twoTriangles), "C6 not isomorphic to two triangles");

  const k33 = normalize([[3, 4, 5], [3, 4, 5], [3, 4, 5], [0, 1, 2], [0, 1, 2], [0, 1, 2]]);
  const prism = normalize([[1, 2, 3], [0, 2, 4], [0, 1, 5], [4, 5, 0], [3, 5, 1], [3, 4, 2]]);
  assert.deepStrictEqual(degrees(k33), degrees(prism));
  assert.deepStrictEqual(degrees(k33), [3, 3, 3, 3, 3, 3]);
  assert.deepStrictEqual(componentSizes(k33), componentSizes(prism));
  assert.strictEqual(triangleCount(k33), 0);
  assert.strictEqual(triangleCount(prism), 2);
  assert.ok(!checkIsomorphic(k33, prism), "K3,3 not isomorphic to the prism");

  const cube = cubeQ3();
  const v8 = mobiusV8();
  assert.ok(checkIsomorphic(cube, v8), "cube vs V8 passes all three necessary checks");
  assert.deepStrictEqual(degrees(cube), degrees(v8));
  assert.strictEqual(triangleCount(cube), 0);
  assert.strictEqual(triangleCount(v8), 0);
  assert.deepStrictEqual(componentSizes(cube), componentSizes(v8));
  assert.ok(isBipartite(cube), "the cube is bipartite");
  assert.ok(!isBipartite(v8), "V8 is not - the documented insufficiency");
  assert.strictEqual(diameter(cube), 3);
  assert.strictEqual(diameter(v8), 2);

  const rng = new Lcg(20261016);
  for (let trial = 0; trial < 40; trial++) {
    const graph = lcgGraph(7, rng);
    const perm = lcgPermutation(7, rng);
    const relabelled = relabel(graph, perm);
    assert.ok(checkIsomorphic(graph, relabelled), `random relabel ${trial}`);
    assert.deepStrictEqual(degrees(graph), degrees(relabelled), `random degrees ${trial}`);
    assert.strictEqual(triangleCount(graph), triangleCount(relabelled), `random triangles ${trial}`);
    assert.deepStrictEqual(componentSizes(graph), componentSizesUnionFind(graph), `union-find oracle ${trial}`);
    assert.deepStrictEqual(componentSizes(relabelled), componentSizesUnionFind(relabelled), `union-find relabelled ${trial}`);
    assert.deepStrictEqual(relabel(relabelled, inverse(perm)), graph, `double relabel ${trial}`);
  }
  console.log("PASS");
}
