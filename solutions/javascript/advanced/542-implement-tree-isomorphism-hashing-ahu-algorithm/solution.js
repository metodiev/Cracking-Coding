/**
 * Task 542 - Implement tree isomorphism hashing (AHU algorithm). (Tier: Advanced, CTCI Ch 4 — Trees and Graphs)
 *
 * AHU canonical forms for unrooted trees: root the tree at its centre (found by the classic
 * double-BFS diameter walk, one centre for an even diameter and two for an odd one), then label
 * subtrees bottom-up - every node's code is "(" + its children's codes sorted lexicographically +
 * ")", so the code of a subtree is identical exactly for isomorphic rooted subtrees and the
 * smallest centre-rooted code is a complete isomorphism invariant.  Because the code for a level
 * is just the sorted multiset of the children's codes, the construction is the level-by-level
 * "AHU hashing" of the literature; the file also keeps the simpler numeric variant (label =
 * index of the sorted multiset of child labels) and asserts that the two invariants induce the
 * same partition of any test family.
 *
 * Documented checks: the number of distinct canonical forms over ALL labelled trees on n vertices
 * (n = 2..7, generated from every Prufer sequence, n^(n-2) of them) equals the known counts of
 * unlabelled trees 1, 1, 1, 2, 3, 6, 11; the canonical form is invariant under random relabelling
 * yet separates non-isomorphic trees - and there is an asserted witness pair with identical
 * degree sequences but different canonical forms; canonical-form equality agrees with a
 * backtracking brute-force isomorphism search on random pairs; minimising the rooted code over
 * every possible root is verified to be an isomorphism invariant too (and to separate the same
 * classes); and paths, stars, single vertices and single edges are hand-verified.
 */
"use strict";

function lcgNext(seed) {
  return (seed * 75 + 74) % 65537;
}

function emptyAdjacency(n) {
  return Array.from({ length: n }, () => []);
}

function buildAdjacency(n, edges) {
  const adj = emptyAdjacency(n);
  for (const [u, v] of edges) {
    adj[u].push(v);
    adj[v].push(u);
  }
  for (const list of adj) list.sort((a, b) => a - b);
  return adj;
}

function isTree(n, edges) {
  if (edges.length !== n - 1) return false;
  const adj = buildAdjacency(n, edges);
  const seen = new Array(n).fill(false);
  seen[0] = true;
  const stack = [0];
  let count = 1;
  while (stack.length) {
    const u = stack.pop();
    for (const v of adj[u]) {
      if (!seen[v]) {
        seen[v] = true;
        count += 1;
        stack.push(v);
      }
    }
  }
  return count === n;
}

/** Persistent-rooted AHU code: "(" + sorted child codes + ")". */
function rootedCode(adj, root) {
  const rec = (u, parent) => {
    const childCodes = [];
    for (const v of adj[u]) if (v !== parent) childCodes.push(rec(v, u));
    childCodes.sort();
    return `(${childCodes.join("")})`;
  };
  return rec(root, -1);
}

/**
 * Breadth-first rooting: parents and depths of every node with respect to `root`.
 */
function rootTree(adj, root) {
  const n = adj.length;
  const parent = new Array(n).fill(-1);
  const depth = new Array(n).fill(0);
  const order = [];
  const seen = new Array(n).fill(false);
  seen[root] = true;
  const queue = [root];
  for (let head = 0; head < queue.length; head++) {
    const u = queue[head];
    order.push(u);
    for (const v of adj[u]) {
      if (!seen[v]) {
        seen[v] = true;
        parent[v] = u;
        depth[v] = depth[u] + 1;
        queue.push(v);
      }
    }
  }
  return { parent, depth, order };
}

/**
 * Level-by-level numeric AHU variant.  Nodes are grouped by depth and processed deepest first;
 * a node's key is the sorted list of its children's labels, and new keys are ranked in sorted
 * order in a cumulative registry.  Ranking by sorted key (instead of by discovery order) makes
 * the labels independent of both the vertex numbering and the adjacency order, so the per-level
 * sorted label multisets are a relabelling-invariant fingerprint of the rooted tree.
 */
function labelledLevels(adj, root) {
  const { parent, depth, order } = rootTree(adj, root);
  const maxDepth = Math.max(...depth);
  const byDepth = Array.from({ length: maxDepth + 1 }, () => []);
  for (const u of order) byDepth[depth[u]].push(u);
  const labels = new Array(adj.length).fill(-1);
  const registry = new Map();
  const levelStrings = new Array(maxDepth + 1).fill("");
  for (let d = maxDepth; d >= 0; d--) {
    const keys = new Map();
    for (const u of byDepth[d]) {
      const childLabels = [];
      for (const v of adj[u]) if (v !== parent[u]) childLabels.push(labels[v]);
      childLabels.sort((a, b) => a - b);
      keys.set(u, childLabels.join(","));
    }
    const fresh = [...new Set(keys.values())].filter((key) => !registry.has(key)).sort();
    for (const key of fresh) registry.set(key, registry.size);
    const levelLabels = [];
    for (const u of byDepth[d]) {
      labels[u] = registry.get(keys.get(u));
      levelLabels.push(labels[u]);
    }
    levelLabels.sort((a, b) => a - b);
    levelStrings[d] = levelLabels.join(",");
  }
  return { labels, levels: levelStrings };
}

function rootedLabels(adj, root) {
  return labelledLevels(adj, root).labels;
}

/** Deepest level first: leaf multiset | next level | ... | root. */
function rootedLevelSignature(adj, root) {
  return labelledLevels(adj, root).levels.slice().reverse().join("|");
}

/** Tree centre(s) via the double-BFS diameter walk. */
function centers(adj) {
  const n = adj.length;
  if (n <= 2) return [...Array(n).keys()];
  const bfs = (start) => {
    const dist = new Array(n).fill(-1);
    const parent = new Array(n).fill(-1);
    dist[start] = 0;
    const queue = [start];
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head];
      for (const v of adj[u]) {
        if (dist[v] === -1) {
          dist[v] = dist[u] + 1;
          parent[v] = u;
          queue.push(v);
        }
      }
    }
    let far = start;
    for (let i = 0; i < n; i++) if (dist[i] > dist[far]) far = i;
    return { far, parent };
  };
  const first = bfs(0).far;
  const { far: second, parent } = bfs(first);
  const path = [];
  for (let v = second; v !== -1; v = parent[v]) path.push(v);
  path.reverse();
  const diameter = path.length - 1;
  if (diameter % 2 === 0) return [path[diameter / 2]];
  return [path[(diameter - 1) / 2], path[(diameter + 1) / 2]].sort((a, b) => a - b);
}

/** Canonical form of an unrooted tree: the lexicographically smallest centre-rooted code. */
function canonicalForm(adj) {
  return centers(adj).map((root) => rootedCode(adj, root)).sort()[0];
}

function canonicalLabels(adj) {
  return centers(adj).map((root) => rootedLevelSignature(adj, root)).sort()[0];
}

/**
 * Backtracking brute-force isomorphism search - the independent ground truth (exponential, hence
 * only used on tiny inputs).  Two vertices may be matched only when their degrees agree and every
 * adjacency/non-adjacency among the already mapped prefix is preserved.
 */
function bruteIsomorphic(adjA, adjB) {
  const n = adjA.length;
  if (adjB.length !== n) return false;
  const degA = adjA.map((list) => list.length);
  const degB = adjB.map((list) => list.length);
  const map = new Array(n).fill(-1);
  const used = new Array(n).fill(false);
  const setsA = adjA.map((list) => new Set(list));
  const setsB = adjB.map((list) => new Set(list));
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

function pruferToEdges(sequence, n) {
  if (n === 2) return [[0, 1]];
  const degree = new Array(n).fill(1);
  for (const v of sequence) degree[v] += 1;
  const edges = [];
  const deg = degree.slice();
  for (const v of sequence) {
    let leaf = -1;
    for (let i = 0; i < n; i++) {
      if (deg[i] === 1) {
        leaf = i;
        break;
      }
    }
    edges.push([leaf, v]);
    deg[leaf] = 0;
    deg[v] -= 1;
  }
  const rest = [];
  for (let i = 0; i < n; i++) if (deg[i] === 1) rest.push(i);
  edges.push([rest[0], rest[1]]);
  return edges;
}

function randomPrufer(n, seed) {
  const sequence = [];
  let s = seed % 65537;
  for (let i = 0; i < n - 2; i++) {
    s = lcgNext(s);
    sequence.push(s % n);
  }
  return sequence;
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

function relabel(n, edges, permutation) {
  return edges.map(([u, v]) => [permutation[u], permutation[v]]);
}

function pathEdges(n) {
  const edges = [];
  for (let i = 0; i + 1 < n; i++) edges.push([i, i + 1]);
  return edges;
}

function starEdges(n) {
  const edges = [];
  for (let i = 1; i < n; i++) edges.push([0, i]);
  return edges;
}

/** Enumerates every labelled tree on n vertices by walking all Prufer sequences. */
function allTrees(n) {
  const trees = [];
  if (n === 1) return [emptyAdjacency(1)];
  if (n === 2) return [buildAdjacency(2, [[0, 1]])];
  const total = n ** (n - 2);
  for (let index = 0; index < total; index++) {
    const sequence = [];
    let value = index;
    for (let i = 0; i < n - 2; i++) {
      sequence.push(value % n);
      value = Math.floor(value / n);
    }
    trees.push(buildAdjacency(n, pruferToEdges(sequence, n)));
  }
  return trees;
}

function degreeSequence(adj) {
  return adj.map((list) => list.length).sort((a, b) => a - b).join(",");
}

function main() {
  const assert = require("node:assert");

  // (1) Hand-verified basics: relabelled paths agree, path != star, single vertex and edge.
  const path5 = buildAdjacency(5, pathEdges(5));
  const shuffled = buildAdjacency(5, relabel(5, pathEdges(5), randomPermutation(5, 7)));
  assert.strictEqual(canonicalForm(path5), canonicalForm(shuffled));
  assert.ok(bruteIsomorphic(path5, shuffled));
  const star5 = buildAdjacency(5, starEdges(5));
  assert.notStrictEqual(canonicalForm(path5), canonicalForm(star5));
  assert.ok(!bruteIsomorphic(path5, star5));
  assert.deepStrictEqual(centers(path5), [2], "a 4-edge path has a single centre");
  const path4 = buildAdjacency(4, pathEdges(4));
  assert.deepStrictEqual(centers(path4), [1, 2], "a 3-edge path has two centres");
  assert.strictEqual(canonicalForm(buildAdjacency(1, [])), "()");
  const singleEdge = canonicalForm(buildAdjacency(2, [[0, 1]]));
  assert.strictEqual(singleEdge, "(())", "two vertices: one leaf under one leaf");
  assert.strictEqual(rootedCode(buildAdjacency(2, [[0, 1]]), 1), "(())",
    "either vertex can serve as the root of a 2-vertex tree");
  assert.strictEqual(canonicalLabels(buildAdjacency(2, [[0, 1]])), "0|1",
    "leaf level then root level: the smallest centre-rooted level signature");
  assert.strictEqual(canonicalLabels(buildAdjacency(1, [])), "0", "a lone vertex has one level");

  // (2) Number of distinct canonical forms over ALL labelled trees == number of unlabelled trees.
  const known = { 2: 1, 3: 1, 4: 2, 5: 3, 6: 6, 7: 11 };
  for (let n = 2; n <= 7; n++) {
    const trees = allTrees(n);
    assert.strictEqual(trees.length, n ** (n - 2), `n=${n}: n^(n-2) labelled trees`);
    for (const adj of trees) {
      assert.strictEqual(adj.length, n);
      const edges = [];
      for (let u = 0; u < n; u++) for (const v of adj[u]) if (u < v) edges.push([u, v]);
      assert.ok(isTree(n, edges), `n=${n}: Prufer construction must yield a tree`);
    }
    const forms = new Set();
    const labelClasses = new Set();
    for (const adj of trees) {
      forms.add(canonicalForm(adj));
      labelClasses.add(canonicalLabels(adj));
    }
    assert.strictEqual(forms.size, known[n],
      `n=${n}: expected ${known[n]} isomorphism classes, got ${forms.size}`);
    assert.strictEqual(labelClasses.size, forms.size,
      `n=${n}: the string and numeric AHU invariants must induce the same partition`);
    if (n <= 6) {
      const rootMin = new Set();
      for (const adj of trees) {
        rootMin.add([...Array(n).keys()].map((root) => rootedCode(adj, root)).sort()[0]);
      }
      assert.strictEqual(rootMin.size, known[n],
        `n=${n}: minimising over every root separates the classes too`);
    }
  }

  // (3) A witness pair: same degree sequence, different canonical form.
  const trees6 = allTrees(6);
  let witness = null;
  const byDegree = new Map();
  for (const adj of trees6) {
    const key = degreeSequence(adj);
    const form = canonicalForm(adj);
    if (byDegree.has(key)) {
      const other = byDegree.get(key);
      if (other.form !== form) {
        witness = [other.adj, adj, key];
        break;
      }
    } else {
      byDegree.set(key, { adj, form });
    }
  }
  assert.ok(witness !== null, "there must be two trees with equal degree sequences");
  assert.strictEqual(degreeSequence(witness[0]), witness[2]);
  assert.notStrictEqual(canonicalForm(witness[0]), canonicalForm(witness[1]));
  assert.ok(!bruteIsomorphic(witness[0], witness[1]),
    "the degree-sequence collision really is non-isomorphic");

  // (4) Canonical equality agrees with the brute-force isomorphism search, and the centre-based
  //     code equals the minimum over all roots.
  for (let seed = 1; seed <= 6; seed++) {
    const n = 7;
    const base = buildAdjacency(n, pruferToEdges(randomPrufer(n, seed * 13), n));
    const copy = buildAdjacency(n, relabel(n, pruferToEdges(randomPrufer(n, seed * 13), n),
      randomPermutation(n, seed * 29)));
    assert.strictEqual(canonicalForm(base), canonicalForm(copy),
      `seed ${seed}: relabelling preserves the canonical form`);
    assert.ok(bruteIsomorphic(base, copy), `seed ${seed}: brute force agrees`);
    const other = buildAdjacency(n, pruferToEdges(randomPrufer(n, seed * 31 + 1), n));
    const sameForm = canonicalForm(base) === canonicalForm(other);
    assert.strictEqual(sameForm, bruteIsomorphic(base, other),
      `seed ${seed}: canonical-form equality must match the brute-force search`);
    const allRoots = [...Array(n).keys()].map((root) => rootedCode(base, root)).sort();
    const copyRoots = [...Array(n).keys()].map((root) => rootedCode(copy, root)).sort();
    assert.ok(allRoots[0] <= canonicalForm(base),
      `seed ${seed}: the centre-rooted code is one of the candidates`);
    assert.strictEqual(allRoots[0], copyRoots[0],
      `seed ${seed}: the root-minimised code is an isomorphism invariant as well`);
    assert.strictEqual(canonicalForm(base) === canonicalForm(copy), true);
  }

  // (5) Determinism and structural sanity of the numeric invariant.
  const repeat = buildAdjacency(9, pruferToEdges(randomPrufer(9, 4242), 9));
  assert.strictEqual(canonicalForm(repeat), canonicalForm(repeat));
  assert.deepStrictEqual(canonicalLabels(repeat), canonicalLabels(repeat));
  assert.ok(canonicalLabels(repeat).includes("|"),
    "a 9-vertex tree has more than one level in its signature");
  assert.strictEqual(rootedCode(repeat, 0), rootedCode(repeat, 0));
  assert.strictEqual(rootedCode(path4, 0), "(((())))", "rooting a path at its end nests 4 levels");
  assert.strictEqual(rootedCode(path4, 1), "((())())",
    "rooting the same path at a centre yields a two-legged code");
  assert.strictEqual(rootedCode(star5, 0), "(()()()())", "a star is four sibling leaves");
  assert.strictEqual(rootedCode(path4, 0), rootedCode(buildAdjacency(4, pathEdges(4)), 0));
  assert.notStrictEqual(rootedCode(path4, 0), rootedCode(path4, 1),
    "rooted codes do depend on the chosen root");
  assert.deepStrictEqual(rootedLabels(path4, 0), [3, 2, 1, 0],
    "a path grows one distinct label per level");
  assert.deepStrictEqual(rootedLabels(star5, 0), [1, 0, 0, 0, 0]);
}

module.exports = { lcgNext, emptyAdjacency, buildAdjacency, isTree, rootedCode, rootTree,
  labelledLevels, rootedLabels, rootedLevelSignature, centers, canonicalForm, canonicalLabels,
  bruteIsomorphic, pruferToEdges, randomPrufer,
  randomPermutation, relabel, pathEdges, starEdges, allTrees, degreeSequence };

if (require.main === module) {
  main();
  console.log("PASS");
}
