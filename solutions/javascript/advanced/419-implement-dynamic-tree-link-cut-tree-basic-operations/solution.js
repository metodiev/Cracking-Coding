/** Task 419 - Implement dynamic tree (link-cut tree) basic operations. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A link-cut tree represents a forest of rooted trees with splay trees over preferred
 * paths. Nodes are 1-based here, 0 is the null child. Arrays: left/right/parent (the
 * parent entry doubles as the splay parent and the path-parent), rev (lazy subtree flip),
 * val (node value) and total (splay-subtree aggregate = path sum of the preferred path).
 *
 * Splay details:
 *   * isRoot(x): parent[x] == 0 or x is neither the left nor the right child of parent[x]
 *     — that is where a preferred path ends and a path-parent edge starts,
 *   * every splay first pushes the rev flags from the splay root down to x, so rotations
 *     never see stale orientations,
 *   * rotations fix three families of links (b-subtree, p, g) and pull p then x.
 * Operations: access (make the x-root path preferred), makeroot (access + rev toggle),
 * findroot (access + descend leftmost + splay), link/cut (both guarded), connected, and
 * pathSum(u, v) = makeroot(u) + access(v) -> total[v]. Ops in the tests stay in the
 * hundreds; every result is validated against an explicit adjacency oracle.
 * All values are small (< 2^53), so plain Numbers stay exact — no BigInt needed.
 */
"use strict";

class LinkCutTree {
  constructor(values) {
    const n = values.length;
    this.left = new Array(n + 1).fill(0);
    this.right = new Array(n + 1).fill(0);
    this.parent = new Array(n + 1).fill(0);
    this.rev = new Array(n + 1).fill(false);
    this.val = [0].concat(values);
    this.total = [0].concat(values);
  }

  #isRoot(x) {
    const p = this.parent[x];
    return p === 0 || (this.left[p] !== x && this.right[p] !== x);
  }

  #push(x) {
    if (this.rev[x]) {
      const t = this.left[x];
      this.left[x] = this.right[x];
      this.right[x] = t;
      if (this.left[x] !== 0) this.rev[this.left[x]] = !this.rev[this.left[x]];
      if (this.right[x] !== 0) this.rev[this.right[x]] = !this.rev[this.right[x]];
      this.rev[x] = false;
    }
  }

  #pull(x) {
    this.total[x] = this.total[this.left[x]] + this.total[this.right[x]] + this.val[x];
  }

  #rotate(x) {
    const p = this.parent[x], g = this.parent[p];
    if (this.left[p] === x) {
      const b = this.right[x];
      this.right[x] = p;
      this.left[p] = b;
      if (b !== 0) this.parent[b] = p;
    } else {
      const b = this.left[x];
      this.left[x] = p;
      this.right[p] = b;
      if (b !== 0) this.parent[b] = p;
    }
    this.parent[p] = x;
    this.parent[x] = g;
    if (g !== 0) {
      if (this.left[g] === p) this.left[g] = x;
      else if (this.right[g] === p) this.right[g] = x;
    }
    this.#pull(p);
    this.#pull(x);
  }

  #splay(x) {
    const stack = [x];
    let y = x;
    while (!this.#isRoot(y)) {
      y = this.parent[y];
      stack.push(y);
    }
    while (stack.length > 0) this.#push(stack.pop());
    while (!this.#isRoot(x)) {
      const p = this.parent[x], g = this.parent[p];
      if (!this.#isRoot(p)) {
        if ((this.left[p] === x) === (this.left[g] === p)) this.#rotate(p);
        else this.#rotate(x);
      }
      this.#rotate(x);
    }
  }

  #access(x) {
    let last = 0;
    let y = x;
    while (y !== 0) {
      this.#splay(y);
      this.right[y] = last;
      if (last !== 0) this.parent[last] = y;
      this.#pull(y);
      last = y;
      y = this.parent[y];
    }
    this.#splay(x);
  }

  makeroot(x) {
    this.#access(x);
    this.rev[x] = !this.rev[x];
  }

  findroot(x) {
    this.#access(x);
    for (;;) {
      this.#push(x);
      if (this.left[x] === 0) break;
      x = this.left[x];
    }
    this.#splay(x);
    return x;
  }

  connected(u, v) {
    if (u === v) return true;
    return this.findroot(u) === this.findroot(v);
  }

  link(u, v) {
    this.makeroot(u);
    if (this.findroot(v) !== u) this.parent[u] = v;
  }

  cut(u, v) {
    this.makeroot(u);
    this.#access(v);
    if (this.left[v] === u && this.right[u] === 0) {
      this.left[v] = 0;
      this.parent[u] = 0;
      this.#pull(v);
    }
  }

  pathSum(u, v) {
    this.makeroot(u);
    this.#access(v);
    return this.total[v];
  }
}

/** Oracle: explicit adjacency plus BFS for connectivity and path sums. */
class Forest {
  constructor(n) {
    this.adj = [];
    for (let i = 0; i <= n; i++) this.adj.push(new Set());
  }

  link(u, v) {
    this.adj[u].add(v);
    this.adj[v].add(u);
  }

  cut(u, v) {
    this.adj[u].delete(v);
    this.adj[v].delete(u);
  }

  hasEdge(u, v) {
    return this.adj[u].has(v);
  }

  /** Unique forest path u -> v as a vertex list, or null when disconnected. */
  path(u, v) {
    const prev = new Map([[u, 0]]);
    const queue = [u];
    while (queue.length > 0) {
      const cur = queue.shift();
      if (cur === v) {
        const path = [];
        let node = v;
        while (node !== 0) {
          path.push(node);
          node = prev.get(node);
        }
        path.reverse();
        return path;
      }
      for (const nxt of [...this.adj[cur]].sort((a, b) => a - b)) {
        if (!prev.has(nxt)) {
          prev.set(nxt, cur);
          queue.push(nxt);
        }
      }
    }
    return null;
  }

  connected(u, v) {
    return this.path(u, v) !== null;
  }
}

let seed = 0;

function nextRand() {
  seed = (seed * 75 + 74) % 65537;
  return seed;
}

function pathSumOracle(forest, values, u, v) {
  const path = forest.path(u, v);
  return path.reduce((acc, node) => acc + values[node - 1], 0);
}

module.exports = { LinkCutTree, Forest, nextRand, pathSumOracle };

if (require.main === module) {
  const assert = require("node:assert");

  const values = [3, 1, 4, 1, 5, 9];
  let lct = new LinkCutTree(values);

  // isolated vertices: every node is its own root, no paths connect them
  for (let v = 1; v <= 6; v++) assert.strictEqual(lct.findroot(v), v, `isolated root ${v}`);
  assert.strictEqual(lct.connected(1, 2), false);
  assert.strictEqual(lct.connected(2, 3), false);
  assert.strictEqual(lct.connected(4, 5), false);

  // link two singletons and read the path
  lct.link(1, 2);
  assert.strictEqual(lct.connected(1, 2), true);
  assert.strictEqual(lct.connected(2, 1), true);
  assert.strictEqual(lct.pathSum(1, 2), 4);          // 3 + 1
  assert.strictEqual(lct.pathSum(2, 1), 4);
  assert.strictEqual(lct.findroot(2), lct.findroot(1));
  assert.strictEqual(lct.pathSum(1, 1), 3);

  // extend to a chain 1 - 2 - 3 and query across it
  lct.link(2, 3);
  assert.strictEqual(lct.pathSum(1, 3), 8);          // 3 + 1 + 4
  assert.strictEqual(lct.pathSum(3, 1), 8);
  assert.strictEqual(lct.pathSum(2, 2), 1);
  // pathSum calls makeroot internally; an explicit makeroot(3) makes 3 the root
  lct.makeroot(3);
  assert.strictEqual(lct.findroot(1), 3);
  assert.strictEqual(lct.findroot(3), 3);

  // cutting an existing edge splits the tree; missing edges are left untouched
  lct.cut(2, 3);
  assert.strictEqual(lct.connected(1, 3), false);
  assert.strictEqual(lct.connected(1, 2), true);
  assert.strictEqual(lct.findroot(3), 3);
  lct.cut(1, 3);                    // no such edge: guarded no-op
  assert.strictEqual(lct.connected(1, 2), true);
  assert.strictEqual(lct.connected(1, 3), false);

  // build two trees, then join them through single edges
  lct.link(3, 4);
  lct.link(5, 6);
  assert.strictEqual(lct.connected(3, 4), true);
  assert.strictEqual(lct.connected(5, 6), true);
  assert.strictEqual(lct.connected(1, 4), false);
  lct.link(2, 3);                    // re-join {1,2} with {3,4}
  assert.strictEqual(lct.connected(1, 4), true);
  assert.strictEqual(lct.pathSum(1, 4), 9);          // 3+1+4+1
  lct.link(4, 5);
  assert.strictEqual(lct.connected(1, 6), true);
  assert.strictEqual(lct.pathSum(1, 6), 23);         // 3+1+4+1+5+9
  assert.strictEqual(lct.pathSum(6, 1), 23);
  assert.strictEqual(lct.pathSum(3, 5), 10);         // 4+1+5

  // cutting the bridge splits the forest again
  lct.cut(4, 5);
  assert.strictEqual(lct.connected(1, 4), true);
  assert.strictEqual(lct.connected(5, 6), true);
  assert.strictEqual(lct.connected(1, 5), false);

  // LCG-driven script against the adjacency oracle (hundreds of operations)
  const n = 30;
  seed = 4190;
  const vals = [];
  for (let i = 0; i < n; i++) vals.push(1 + (nextRand() % 100));
  lct = new LinkCutTree(vals);
  const forest = new Forest(n);

  for (let step = 0; step < 350; step++) {
    const kind = nextRand() % 4;
    if (kind === 0) {
      const u = 1 + (nextRand() % n);
      const v = 1 + (nextRand() % n);
      if (u !== v && forest.connected(u, v) && forest.hasEdge(u, v)) {
        lct.cut(u, v);
        forest.cut(u, v);
        assert.strictEqual(lct.connected(u, v), false, `script cut ${step}`);
      }
    } else if (kind === 1) {
      const u = 1 + (nextRand() % n);
      const v = 1 + (nextRand() % n);
      if (u !== v && !forest.connected(u, v)) {
        lct.link(u, v);
        forest.link(u, v);
        assert.strictEqual(lct.connected(u, v), true, `script link ${step}`);
      }
    } else if (kind === 2) {
      const u = 1 + (nextRand() % n);
      const v = 1 + (nextRand() % n);
      assert.strictEqual(lct.connected(u, v), forest.connected(u, v), `script conn ${step}`);
      if (forest.connected(u, v)) {
        assert.strictEqual(lct.pathSum(u, v), pathSumOracle(forest, vals, u, v), `script sum ${step}`);
        assert.strictEqual(lct.pathSum(v, u), lct.pathSum(u, v), `script sum sym ${step}`);
      }
    } else {
      // full pairwise connectivity audit
      for (let u = 1; u <= n; u++) {
        for (let v = u + 1; v <= n; v++) {
          assert.strictEqual(lct.connected(u, v), forest.connected(u, v), `audit ${step} ${u} ${v}`);
        }
      }
    }
  }

  // final audit: every path sum across the whole forest matches the oracle
  let checks = 0;
  for (let u = 1; u <= n; u++) {
    for (let v = u; v <= n; v++) {
      assert.strictEqual(lct.connected(u, v), forest.connected(u, v), "final conn");
      if (forest.connected(u, v)) {
        assert.strictEqual(lct.pathSum(u, v), pathSumOracle(forest, vals, u, v), `final sum ${u} ${v}`);
        checks++;
      }
    }
  }
  assert.ok(checks > n, "final coverage");

  // a star built by links from one hub answers all leaf-to-leaf paths through it
  seed = 777;
  const hubValues = [];
  for (let i = 0; i < 9; i++) hubValues.push(1 + (nextRand() % 9));
  lct = new LinkCutTree(hubValues);
  for (let leaf = 2; leaf <= 9; leaf++) lct.link(1, leaf);
  for (let leaf = 2; leaf <= 9; leaf++) {
    assert.strictEqual(lct.pathSum(1, leaf), hubValues[0] + hubValues[leaf - 1], `star hub ${leaf}`);
  }
  for (let a = 2; a <= 9; a++) {
    for (let b = a + 1; b <= 9; b++) {
      const want = hubValues[0] + hubValues[a - 1] + hubValues[b - 1];
      assert.strictEqual(lct.pathSum(a, b), want, `star leaf ${a} ${b}`);
    }
  }
  // each path query re-roots through makeroot; after makeroot(9), 9 is the tree root
  lct.makeroot(9);
  assert.strictEqual(lct.findroot(1), 9);
  assert.strictEqual(lct.findroot(9), 9);

  // cutting every leaf leaves a single node
  for (let leaf = 2; leaf <= 9; leaf++) lct.cut(1, leaf);
  for (let leaf = 2; leaf <= 9; leaf++) {
    assert.strictEqual(lct.connected(1, leaf), false, `leaf cut ${leaf}`);
    assert.strictEqual(lct.findroot(leaf), leaf, `leaf root ${leaf}`);
  }

  console.log("PASS");
}
