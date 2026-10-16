/**
 * Task 534 - Implement matching in general graphs improved heuristics (Edmonds practical). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * A practical implementation of Edmonds' blossom algorithm for maximum cardinality matching in a
 * general (non-bipartite) graph, in the compact O(V^3) "BFS + blossom contraction" form: for every
 * still-free vertex a BFS grows an alternating forest, tracking for each vertex its tree parent p
 * and its current blossom base; when an edge joins two even-level vertices the odd cycle is
 * detected, its lowest common ancestor is computed by walking both halves to the root, and the
 * blossom is *contracted* by relabelling every base inside it - exactly the piece that a bipartite
 * augmenting-path search lacks. When the BFS reaches a free vertex the augmenting path is flipped
 * along the parent pointers. A greedy maximal matching accompanies the exact algorithm: it is not
 * sufficient on its own (a locally attractive first choice can block the optimum, which the checks
 * exhibit) but it supplies a cheap maximal-matching baseline and the structural invariants the
 * final result is validated against.
 *
 * Documented checks: the matching is always verified structurally (symmetric partner array, every
 * partner an actual neighbour, size = matched/2, no edge between two unmatched vertices) and
 * compared with an exact bitmask-DP maximum matching on hundreds of small graphs - LCG random graphs
 * of 2..14 vertices at several edge densities, odd cycles, triangles, complete graphs - and with an
 * independent tree dynamic program on 300-vertex random trees. Closed forms are asserted for paths
 * (floor(n/2)), cycles (floor(n/2)), complete graphs (floor(n/2)), stars (1) and edgeless graphs
 * (0), and the C5 example (2, no perfect matching) is fixed explicitly. The greedy heuristic is
 * shown to be strictly worse on a seven-vertex example where a locally good first choice blocks an
 * optimal matching, while the blossom result still matches the DP optimum there. The augmentation
 * phase count never exceeds floor(n/2), results are invariant under vertex permutation, and invalid
 * edges are rejected.
 */
"use strict";

class Lcg {
  constructor(seed) {
    let s = ((seed >>> 0) ^ 0x9e3779b9) >>> 0;
    s = Math.imul(s ^ (s >>> 16), 0x85ebca6b) >>> 0;
    s = Math.imul(s ^ (s >>> 13), 0xc2b2ae35) >>> 0;
    this.state = (s ^ (s >>> 16)) >>> 0;
    if (this.state === 0) this.state = 0x1b873593;
  }

  nextUint32() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state;
  }

  nextInt(bound) {
    return Math.floor((this.nextUint32() / 4294967296) * bound);
  }
}

class Blossom {
  constructor(vertexCount) {
    if (!Number.isInteger(vertexCount) || vertexCount < 0) {
      throw new Error("the vertex count must be a non-negative integer");
    }
    this.n = vertexCount;
    this.adj = Array.from({ length: vertexCount }, () => []);
    this.edges = [];
  }

  addEdge(u, v) {
    if (!Number.isInteger(u) || !Number.isInteger(v) || u < 0 || v < 0 ||
        u >= this.n || v >= this.n) {
      throw new Error(`edge (${u}, ${v}) is out of range`);
    }
    if (u === v) throw new Error("self-loops are not allowed");
    this.adj[u].push(v);
    this.adj[v].push(u);
    this.edges.push([u, v]);
  }

  /** Maximum cardinality matching; returns the partner array (or -1) and the phase count. */
  solve() {
    const n = this.n;
    const adj = this.adj;
    const match = new Array(n).fill(-1);
    let phases = 0;

    const findPath = (root) => {
      const used = new Array(n).fill(false);
      const p = new Array(n).fill(-1);
      const base = Array.from({ length: n }, (_, i) => i);
      const blossom = new Array(n).fill(false);
      const queue = [root];
      used[root] = true;

      const lca = (a, b) => {
        const seen = new Array(n).fill(false);
        let x = a;
        for (;;) {
          x = base[x];
          seen[x] = true;
          if (match[x] === -1) break;
          x = p[match[x]];
        }
        let y = b;
        for (;;) {
          y = base[y];
          if (seen[y]) return y;
          y = p[match[y]];
        }
      };

      const markPath = (start, curbase, child) => {
        let v = start;
        let c = child;
        while (base[v] !== curbase) {
          blossom[base[v]] = true;
          blossom[base[match[v]]] = true;
          p[v] = c;
          c = match[v];
          v = p[match[v]];
        }
      };

      for (let head = 0; head < queue.length; head++) {
        const v = queue[head];
        for (const to of adj[v]) {
          if (base[v] === base[to] || match[v] === to) continue;
          if (to === root || (match[to] !== -1 && p[match[to]] !== -1)) {
            const curbase = lca(v, to);
            blossom.fill(false);
            markPath(v, curbase, to);
            markPath(to, curbase, v);
            for (let i = 0; i < n; i++) {
              if (blossom[base[i]]) {
                base[i] = curbase;
                if (!used[i]) {
                  used[i] = true;
                  queue.push(i);
                }
              }
            }
          } else if (p[to] === -1) {
            p[to] = v;
            if (match[to] === -1) return { end: to, p };
            const partner = match[to];
            used[partner] = true;
            queue.push(partner);
          }
        }
      }
      return null;
    };

    for (let root = 0; root < n; root++) {
      if (match[root] !== -1) continue;
      const found = findPath(root);
      if (found === null) continue;
      phases += 1;
      let v = found.end;
      while (v !== -1) {
        const pv = found.p[v];
        const next = match[pv];
        match[v] = pv;
        match[pv] = v;
        v = next;
      }
    }

    let size = 0;
    for (const partner of match) if (partner !== -1) size += 1;
    return { match, size: size / 2, phases };
  }
}

/** Greedy maximal matching: walk the edges in the given order and take both free endpoints. */
function greedyMatching(n, edges) {
  const match = new Array(n).fill(-1);
  for (const [u, v] of edges) {
    if (match[u] === -1 && match[v] === -1) {
      match[u] = v;
      match[v] = u;
    }
  }
  let size = 0;
  for (const partner of match) if (partner !== -1) size += 1;
  return { match, size: size / 2 };
}

/** Exact maximum matching by bitmask DP over the set of unmatched vertices (n <= 20). */
function bruteForceMatching(n, edges) {
  const adjacency = new Array(n).fill(0);
  for (const [u, v] of edges) {
    adjacency[u] |= 1 << v;
    adjacency[v] |= 1 << u;
  }
  const memo = new Map();
  const search = (free) => {
    if (free === 0) return 0;
    if (memo.has(free)) return memo.get(free);
    const low = free & -free;
    const v = 31 - Math.clz32(low);
    let best = search(free & ~low);
    let candidates = free & adjacency[v] & ~low;
    while (candidates !== 0) {
      const bit = candidates & -candidates;
      const w = 31 - Math.clz32(bit);
      const withPair = 1 + search(free & ~low & ~bit);
      if (withPair > best) best = withPair;
      candidates &= candidates - 1;
    }
    memo.set(free, best);
    return best;
  };
  return search((1 << n) - 1);
}

/**
 * Independent maximum matching on a tree by dynamic programming (no matching machinery):
 * skip[v] = best in the subtree of v with v unmatched to any child,
 * used[v] = best with v matched to one of its children, which forbids that child from being matched
 * to one of its own children (hence the dp0 term) while the others stay free to do either.
 */
function treeMatchingDp(n, edges) {
  const adj = Array.from({ length: n }, () => []);
  for (const [u, v] of edges) {
    adj[u].push(v);
    adj[v].push(u);
  }
  if (n === 0) return 0;
  const parent = new Array(n).fill(-2);
  const order = [];
  const stack = [0];
  parent[0] = -1;
  while (stack.length > 0) {
    const v = stack.pop();
    order.push(v);
    for (const to of adj[v]) {
      if (parent[to] === -2) {
        parent[to] = v;
        stack.push(to);
      }
    }
  }
  const skip = new Array(n).fill(0);
  const usedNode = new Array(n).fill(0);
  for (let i = order.length - 1; i >= 0; i--) {
    const v = order[i];
    let total = 0;
    let bestTrade = -Infinity;
    let children = 0;
    for (const to of adj[v]) {
      if (to === parent[v]) continue;
      children += 1;
      const best = Math.max(skip[to], usedNode[to]);
      total += best;
      const trade = skip[to] - best;
      if (trade > bestTrade) bestTrade = trade;
    }
    skip[v] = total;
    usedNode[v] = children === 0 ? -Infinity : total + 1 + bestTrade;
  }
  return Math.max(skip[0], usedNode[0] === -Infinity ? 0 : usedNode[0]);
}

function verifyMatching(n, edges, match, size) {
  const edgeSet = new Set();
  for (const [u, v] of edges) {
    edgeSet.add(`${Math.min(u, v)},${Math.max(u, v)}`);
  }
  let matched = 0;
  for (let v = 0; v < n; v++) {
    if (match[v] === -1) continue;
    matched += 1;
    if (match[match[v]] !== v) throw new Error(`matching is not symmetric at ${v}`);
    if (!edgeSet.has(`${Math.min(v, match[v])},${Math.max(v, match[v])}`)) {
      throw new Error(`matched pair (${v}, ${match[v]}) is not an edge`);
    }
  }
  if (matched / 2 !== size) throw new Error("reported size does not match the partner array");
  for (const [u, v] of edges) {
    if (match[u] === -1 && match[v] === -1) {
      throw new Error(`edge (${u}, ${v}) joins two unmatched vertices: not maximal`);
    }
  }
  return size;
}

function solveGraph(n, edges) {
  const blossom = new Blossom(n);
  for (const [u, v] of edges) blossom.addEdge(u, v);
  return blossom.solve();
}

function pathEdges(n) {
  const edges = [];
  for (let i = 0; i + 1 < n; i++) edges.push([i, i + 1]);
  return edges;
}

function cycleEdges(n) {
  const edges = pathEdges(n);
  if (n >= 3) edges.push([n - 1, 0]);
  return edges;
}

function completeEdges(n) {
  const edges = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) edges.push([i, j]);
  return edges;
}

function starEdges(leaves) {
  const edges = [];
  for (let i = 1; i <= leaves; i++) edges.push([0, i]);
  return edges;
}

function randomGraph(n, seed, probability) {
  const rng = new Lcg(seed);
  const edges = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (rng.nextInt(100) < probability) edges.push([i, j]);
    }
  }
  return edges;
}

function randomTree(n, seed) {
  const rng = new Lcg(seed);
  const edges = [];
  for (let v = 1; v < n; v++) edges.push([v, rng.nextInt(v)]);
  return edges;
}

function main() {
  const assert = require("node:assert");

  // Closed forms.
  assert.strictEqual(solveGraph(0, []).size, 0);
  assert.strictEqual(solveGraph(1, []).size, 0);
  assert.strictEqual(solveGraph(2, []).size, 0, "an edgeless graph has no matching");
  assert.strictEqual(solveGraph(5, []).size, 0);
  assert.strictEqual(solveGraph(2, [[0, 1]]).size, 1);
  assert.strictEqual(solveGraph(6, starEdges(5)).size, 1, "a star admits exactly one edge");
  for (let n = 1; n <= 30; n++) {
    assert.strictEqual(solveGraph(n, pathEdges(n)).size, Math.floor(n / 2),
      `path P_${n}`);
    assert.strictEqual(solveGraph(n, cycleEdges(n)).size, Math.floor(n / 2), `cycle C_${n}`);
    assert.strictEqual(solveGraph(n, completeEdges(n)).size, Math.floor(n / 2), `complete K_${n}`);
  }
  const c5 = solveGraph(5, cycleEdges(5));
  assert.strictEqual(c5.size, 2, "C5 has no perfect matching");
  verifyMatching(5, cycleEdges(5), c5.match, c5.size);
  assert.strictEqual(c5.phases, 2, "two augmentations are needed");
  assert.ok(c5.match.every((partner, v) => partner === -1 || c5.match[partner] === v));

  // A blossom that must be contracted: C5 with a pendant leaf on each of two adjacent vertices.
  const blossomGraph = cycleEdges(5).concat([[0, 5], [1, 6]]);
  const blossomResult = solveGraph(7, blossomGraph);
  assert.strictEqual(blossomResult.size, 3,
    "C5 with two pendant leaves matches three pairs, which is the maximum for seven vertices");
  assert.strictEqual(blossomResult.size, bruteForceMatching(7, blossomGraph));
  verifyMatching(7, blossomGraph, blossomResult.match, blossomResult.size);

  // Greedy is maximal but not maximum: one bad first choice blocks the optimum.
  const greedyFail = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 4]];
  const greedy = greedyMatching(5, greedyFail);
  assert.strictEqual(greedy.size, 1, "the greedy order takes (0, 1) and then blocks everything");
  const optimal = solveGraph(5, greedyFail);
  assert.strictEqual(optimal.size, 2, "Edmonds finds (0, 3) and (1, 4)");
  assert.strictEqual(optimal.size, bruteForceMatching(5, greedyFail));
  verifyMatching(5, greedyFail, optimal.match, optimal.size);

  // Exact comparison with the bitmask DP on many small graphs.
  let compared = 0;
  let greedyBeat = 0;
  for (const n of [2, 3, 4, 5, 6, 7, 8, 11, 14]) {
    for (const probability of [15, 35, 55, 80, 100]) {
      for (let seed = 1; seed <= 4; seed++) {
        const edges = randomGraph(n, seed * 31 + n * 7 + probability, probability);
        const result = solveGraph(n, edges);
        const expected = bruteForceMatching(n, edges);
        assert.strictEqual(result.size, expected,
          `n = ${n}, p = ${probability}, seed = ${seed}: blossom size`);
        verifyMatching(n, edges, result.match, result.size);
        assert.ok(result.phases <= Math.floor(n / 2), "the phase count stays within floor(n/2)");
        const greedySize = greedyMatching(n, edges).size;
        assert.ok(greedySize <= expected);
        if (greedySize < expected) greedyBeat += 1;
        compared += 1;
      }
    }
  }
  assert.ok(compared >= 180, `compared ${compared} graphs against the exact DP`);
  assert.ok(greedyBeat >= 10, `greedy is suboptimal on ${greedyBeat} of them`);

  // Vertex permutation invariance.
  for (let seed = 1; seed <= 20; seed++) {
    const n = 8;
    const edges = randomGraph(n, seed * 13, 45);
    const rng = new Lcg(seed * 977);
    const order = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = rng.nextInt(i + 1);
      const tmp = order[i];
      order[i] = order[j];
      order[j] = tmp;
    }
    const permuted = edges.map(([u, v]) => [order[u], order[v]]);
    assert.strictEqual(solveGraph(n, permuted).size, solveGraph(n, edges).size,
      "the matching size does not depend on vertex labels");
  }

  // Trees: an independent DP for the maximum matching.
  for (let seed = 1; seed <= 12; seed++) {
    for (const n of [7, 40, 150, 300]) {
      const edges = randomTree(n, seed * 101 + n);
      const result = solveGraph(n, edges);
      const expected = treeMatchingDp(n, edges);
      assert.strictEqual(result.size, expected,
        `tree n = ${n}, seed = ${seed}: blossom matches the tree DP`);
      verifyMatching(n, edges, result.match, result.size);
      assert.ok(result.size <= Math.floor(n / 2));
    }
  }

  // Larger sparse graphs: structural validation plus the C4/odd-cycle stress.
  for (let seed = 1; seed <= 6; seed++) {
    const n = 200;
    const rng = new Lcg(seed * 7919);
    const edges = [];
    const seen = new Set();
    for (let v = 0; v < n; v++) {
      for (let k = 0; k < 3; k++) {
        const to = rng.nextInt(n);
        if (to === v) continue;
        const key = `${Math.min(v, to)},${Math.max(v, to)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        edges.push([v, to]);
      }
    }
    const result = solveGraph(n, edges);
    verifyMatching(n, edges, result.match, result.size);
    assert.ok(result.size >= greedyMatching(n, edges).size, "never worse than greedy");
    assert.ok(result.size <= Math.floor(n / 2));
    assert.ok(result.size >= 40, `sparse n = 200 graph: matching of size ${result.size}`);
  }

  // Duplicate edges and triangles are handled.
  const withDuplicates = [[0, 1], [0, 1], [1, 2], [2, 0], [2, 3]];
  assert.strictEqual(solveGraph(4, withDuplicates).size, 2);
  assert.strictEqual(solveGraph(3, [[0, 1], [1, 2], [2, 0]]).size, 1, "a triangle matches one pair");

  assert.throws(() => new Blossom(-1), /non-negative integer/);
  assert.throws(() => new Blossom(3).addEdge(0, 0), /self-loops/);
  assert.throws(() => new Blossom(3).addEdge(0, 3), /out of range/);
  assert.throws(() => new Blossom(3).addEdge(-1, 1), /out of range/);
}

module.exports = { Lcg, Blossom, greedyMatching, bruteForceMatching, treeMatchingDp,
  verifyMatching, solveGraph, pathEdges, cycleEdges, completeEdges, starEdges, randomGraph,
  randomTree };

if (require.main === module) {
  main();
  console.log("PASS");
}
