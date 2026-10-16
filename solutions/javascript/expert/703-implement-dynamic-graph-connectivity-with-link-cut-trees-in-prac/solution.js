/** Task 703 - Implement dynamic graph connectivity with link-cut trees in practice. (Tier: Expert, CTCI Ch 4 — Trees and Graphs)
 *
 * A link-cut tree keeps a forest under link/cut with an Euler-tour-free decomposition: the forest
 * is split into preferred paths, every path is one splay tree whose in-order sequence is the path
 * from top to bottom, and the splay root holds a path pointer to the parent vertex of the path.
 * `access(x)` splices the root-to-x preferred path into a single splay tree; makeRoot, findRoot,
 * link, cut, path aggregates and LCA then cost O(log n) amortised - dynamic connectivity that a
 * static union-find cannot offer, while path queries keep the tree structure intact.
 *
 * What is simplified: a fixed vertex set (vertices are never inserted), values are aggregated on
 * vertices only (no edge weights), and the amortised bound is not proved here, only observed by
 * counting splay rotations.
 *
 * Invariant verified: after each of hundreds of operations the edge set read back out of the
 * auxiliary trees equals the brute-force forest, and connectivity, path sum, path max and LCA
 * agree with BFS oracles on that same forest; rejected operations (cycle-creating link, cut of a
 * non-edge, out-of-range vertex) leave the structure untouched.
 *
 * NB: a path query calls makeRoot(first argument), so it re-roots the forest - the checks pin the
 * root before every LCA comparison, because an LCA is only defined relative to a root.
 */
"use strict";
const { strictEqual, ok, throws } = require("node:assert");
/** Deterministic xorshift32; returns integers in [0, bound) (or the raw word). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

class LinkCutTree {
  /** Vertices 0..values.length-1 carry a value; only vertex values are aggregated. */
  constructor(values) {
    this.nodes = values.map((value, id) => ({
      id, value, sum: value, max: value, left: null, right: null, parent: null, reversed: false,
    }));
    this.rotations = 0;
  }
  get length() { return this.nodes.length; }
  #node(x) {
    if (!Number.isInteger(x) || x < 0 || x >= this.nodes.length) {
      throw new RangeError(`vertex ${x} is outside 0..${this.nodes.length - 1}`);
    }
    return this.nodes[x];
  }
  /** True when x roots its splay tree (its parent is a path pointer, not a splay child). */
  #isAuxRoot(x) { return x.parent === null || (x.parent.left !== x && x.parent.right !== x); }
  #push(x) {
    if (!x.reversed) return;
    const swap = x.left;
    x.left = x.right;
    x.right = swap;
    if (x.left !== null) x.left.reversed = !x.left.reversed;
    if (x.right !== null) x.right.reversed = !x.right.reversed;
    x.reversed = false;
  }
  #pull(x) {
    x.sum = x.value + (x.left === null ? 0 : x.left.sum) + (x.right === null ? 0 : x.right.sum);
    let best = x.value;
    if (x.left !== null && x.left.max > best) best = x.left.max;
    if (x.right !== null && x.right.max > best) best = x.right.max;
    x.max = best;
  }
  #rotate(x) {
    const p = x.parent;
    const g = p.parent;
    this.rotations++;
    if (p.left === x) {
      p.left = x.right;
      if (x.right !== null) x.right.parent = p;
      x.right = p;
    } else {
      p.right = x.left;
      if (x.left !== null) x.left.parent = p;
      x.left = p;
    }
    p.parent = x;
    x.parent = g;
    if (g !== null) {
      if (g.left === p) g.left = x;
      else if (g.right === p) g.right = x;
    }
    this.#pull(p);
    this.#pull(x);
  }
  /** Splay x to its auxiliary root, pushing the reversal flags down the spine first. */
  #splay(x) {
    const spine = [x];
    for (let y = x; !this.#isAuxRoot(y);) spine.push(y = y.parent);
    for (let i = spine.length - 1; i >= 0; i--) this.#push(spine[i]);
    while (!this.#isAuxRoot(x)) {
      const p = x.parent;
      const g = p.parent;
      if (!this.#isAuxRoot(p)) this.#rotate((p.left === x) === (g.left === p) ? p : x);
      this.#rotate(x);
    }
  }
  /** Splices the root-to-x path into one splay tree; returns the root of that tree. */
  #access(x) {
    let previous = null;
    for (let y = x; y !== null; y = y.parent) {
      this.#splay(y);
      y.right = previous;
      this.#pull(y);
      previous = y;
    }
    this.#splay(x);
    return previous;
  }
  makeRoot(x) {
    const node = this.#node(x);
    this.#access(node);
    node.reversed = !node.reversed;
    return x;
  }
  /** Root of x's represented tree - a real vertex index, unlike a union-find representative. */
  findRoot(x) {
    let node = this.#node(x);
    this.#access(node);
    while (node.left !== null) {
      this.#push(node);
      node = node.left;
    }
    this.#splay(node);
    return node.id;
  }
  connected(x, y) {
    this.#node(x);
    this.#node(y);
    return this.findRoot(x) === this.findRoot(y);
  }
  link(x, y) {
    this.#node(x);
    const node = this.#node(y);
    if (x === y || this.connected(x, y)) throw new Error(`link(${x},${y}) would create a cycle`);
    this.makeRoot(x);
    this.nodes[x].parent = node;
    return this;
  }
  cut(x, y) {
    this.#node(x);
    this.#node(y);
    this.makeRoot(x);
    this.#access(this.nodes[y]); // the path x..y is now a single splay tree, x leftmost
    const xNode = this.nodes[x];
    const yNode = this.nodes[y];
    if (yNode.left !== xNode || xNode.right !== null) throw new Error(`cut(${x},${y}): no such edge`);
    yNode.left = null;
    xNode.parent = null;
    this.#pull(yNode);
    return this;
  }
  pathSum(x, y) {
    this.#node(y);
    this.makeRoot(x);
    this.#access(this.nodes[y]);
    return this.nodes[y].sum;
  }
  pathMax(x, y) {
    this.#node(y);
    this.makeRoot(x);
    this.#access(this.nodes[y]);
    return this.nodes[y].max;
  }
  /**
   * Lowest common ancestor w.r.t. the current root. After access(x) the preferred path is exactly
   * root..x, so the ancestor of y that the upward walk of the second access meets first while its
   * auxiliary tree already contains x is the LCA.
   */
  lca(x, y) {
    this.#node(x);
    this.#node(y);
    if (!this.connected(x, y)) return -1;
    this.#access(this.nodes[x]);
    const target = this.nodes[x];
    let answer = null;
    let previous = null;
    for (let v = this.nodes[y]; v !== null; v = v.parent) {
      this.#splay(v);
      if (answer === null && this.#auxRootOf(target) === v) answer = v;
      v.right = previous;
      this.#pull(v);
      previous = v;
    }
    this.#splay(this.nodes[y]);
    return (answer === null ? this.nodes[this.findRoot(y)] : answer).id;
  }
  #auxRootOf(node) {
    let walk = node;
    while (!this.#isAuxRoot(walk)) walk = walk.parent;
    return walk;
  }
  /**
   * Reads the represented forest back out of the auxiliary trees: an in-order walk of a splay tree
   * gives its preferred path (consecutive vertices are adjacent) and a splay root's parent is the
   * parent of the topmost vertex of that path.
   */
  edgeList() {
    const edges = new Set();
    const record = (a, b) => {
      if (a !== null && b !== null) edges.add(a.id < b.id ? `${a.id}-${b.id}` : `${b.id}-${a.id}`);
    };
    for (const root of this.nodes) {
      if (!this.#isAuxRoot(root)) continue;
      let first = null;
      let previous = null;
      this.#inOrder(root, (node) => {
        if (first === null) first = node;
        else record(previous, node);
        previous = node;
      });
      record(first, root.parent);
    }
    return [...edges].sort().join(",");
  }
  /** In-order walk of a splay tree, pushing the reversal flags down on the way. */
  #inOrder(root, visit) {
    const stack = [];
    for (let current = root; stack.length > 0 || current !== null;) {
      while (current !== null) {
        this.#push(current);
        stack.push(current);
        current = current.left;
      }
      const node = stack.pop();
      visit(node);
      current = node.right;
    }
  }
}
/**
 * Brute-force oracle. `adjacency` is maintained by the test itself; rooting the component at
 * `root` gives depths and parents, the LCA follows by lifting both endpoints to the same depth.
 */
function oracleQuery(adjacency, x, y, root, values) {
  const size = adjacency.length;
  const parent = new Array(size).fill(-1);
  const depth = new Array(size).fill(-1);
  depth[root] = 0;
  const queue = [root];
  while (queue.length > 0) {
    const current = queue.shift();
    for (const next of adjacency[current]) {
      if (depth[next] === -1) { depth[next] = depth[current] + 1; parent[next] = current; queue.push(next); }
    }
  }
  if (depth[x] === -1 || depth[y] === -1) return { connected: false };
  let a = x;
  let b = y;
  while (depth[a] > depth[b]) a = parent[a];
  while (depth[b] > depth[a]) b = parent[b];
  while (a !== b) { a = parent[a]; b = parent[b]; }
  const route = [a];
  for (let v = x; v !== a; v = parent[v]) route.push(v);
  for (let v = y; v !== a; v = parent[v]) route.push(v);
  return {
    connected: true,
    lca: a,
    sum: route.reduce((s, v) => s + values[v], 0),
    max: Math.max(...route.map((v) => values[v])),
  };
}
function edgesOf(adjacency) {
  const edges = new Set();
  for (let x = 0; x < adjacency.length; x++) {
    for (const y of adjacency[x]) edges.add(x < y ? `${x}-${y}` : `${y}-${x}`);
  }
  return [...edges].sort().join(",");
}

module.exports = { makeRng, LinkCutTree, oracleQuery, edgesOf };

if (require.main === module) {
  // -- single vertex, and every rejected operation ---------------------------------------------
  const solo = new LinkCutTree([7]);
  ok(solo.connected(0, 0));
  strictEqual(solo.pathSum(0, 0), 7);
  strictEqual(solo.pathMax(0, 0), 7);
  strictEqual(solo.findRoot(0), 0);
  strictEqual(solo.lca(0, 0), 0);
  strictEqual(solo.edgeList(), "", "a lone vertex has no edges");
  throws(() => solo.link(0, 0), Error);
  for (const bad of [() => solo.cut(0, 0), () => solo.pathSum(0, 1), () => solo.makeRoot(-1), () => solo.link(0, -2)]) {
    throws(bad, Error);
  }

  // -- hand-checked tree 0-1-2-3 plus 1-4 with values 1..5 -------------------------------------
  const hand = new LinkCutTree([1, 2, 3, 4, 5]);
  hand.link(0, 1).link(1, 2).link(2, 3).link(1, 4);
  strictEqual(hand.edgeList(), "0-1,1-2,1-4,2-3");
  strictEqual(hand.pathSum(0, 3), 10);
  strictEqual(hand.pathSum(0, 1), 3);
  strictEqual(hand.pathMax(4, 3), 5);
  strictEqual(hand.pathMax(0, 1), 2);
  hand.makeRoot(1); // an LCA is only defined relative to a root, so pin it
  strictEqual(hand.lca(0, 3), 1);
  strictEqual(hand.lca(4, 3), 1);
  strictEqual(hand.lca(0, 4), 1);
  strictEqual(hand.lca(0, 1), 1);
  strictEqual(hand.lca(2, 3), 2);
  throws(() => hand.link(0, 3), Error);
  hand.cut(2, 3);
  strictEqual(hand.edgeList(), "0-1,1-2,1-4");
  ok(!hand.connected(0, 3));
  throws(() => hand.cut(2, 3), Error);

  // -- mixed sequence against the BFS oracle ---------------------------------------------------
  const rng = makeRng(703);
  const size = 12;
  const values = Array.from({ length: size }, () => rng(20) - 10);
  const lct = new LinkCutTree(values);
  const adjacency = Array.from({ length: size }, () => new Set());
  let links = 0, cuts = 0, queries = 0;
  for (let step = 0; step < 400; step++) {
    const x = rng(size);
    const y = rng(size);
    const action = rng(10);
    const truth = oracleQuery(adjacency, x, y, lct.findRoot(x), values);
    if (action < 4) {
      if (x !== y && !truth.connected) {
        lct.link(x, y);
        adjacency[x].add(y);
        adjacency[y].add(x);
        links++;
      } else if (x !== y) {
        throws(() => lct.link(x, y), Error); // a cycle-creating link is rejected
      }
    } else if (action < 7) {
      if (adjacency[x].has(y)) {
        lct.cut(x, y);
        adjacency[x].delete(y);
        adjacency[y].delete(x);
        cuts++;
      } else {
        throws(() => lct.cut(x, y), Error); // cutting a non-edge is rejected
      }
    } else {
      strictEqual(lct.connected(x, y), x === y || truth.connected, `connected(${x},${y}) step ${step}`);
      if (truth.connected) {
        strictEqual(lct.lca(x, y), truth.lca, `lca(${x},${y}) at step ${step}`);
        strictEqual(lct.pathSum(x, y), truth.sum, `pathSum(${x},${y}) at step ${step}`);
        strictEqual(lct.pathMax(x, y), truth.max, `pathMax(${x},${y}) at step ${step}`);
      }
      queries++;
    }
    strictEqual(lct.edgeList(), edgesOf(adjacency), `the forest must match after step ${step}`);
  }
  ok(links > 10 && cuts > 5, `the workload must really link and cut (${links}/${cuts})`);

  // -- long chain: path queries at scale, plus an amortised rotation count ----------------------
  const chainValues = Array.from({ length: 400 }, (_, i) => i % 7);
  const chain = new LinkCutTree(chainValues);
  for (let i = 1; i < 400; i++) chain.link(i - 1, i);
  strictEqual(chain.pathSum(0, 399), chainValues.reduce((a, b) => a + b, 0));
  strictEqual(chain.pathMax(0, 399), 6);
  for (let i = 0; i < 50; i++) {
    const left = (i * 37) % 400;
    const right = 399 - left;
    const segment = chainValues.slice(Math.min(left, right), Math.max(left, right) + 1);
    ok(chain.connected(left, right));
    strictEqual(chain.pathSum(left, right), segment.reduce((a, b) => a + b, 0), `chain sum ${left}..${right}`);
    strictEqual(chain.pathMax(left, right), Math.max(...segment), `chain max ${left}..${right}`);
  }
  ok(chain.rotations < 400 * 50, `splay rotations stay amortised (${chain.rotations})`);
  const reroot = new LinkCutTree([3, 1, 4, 1, 5, 9]);
  reroot.link(0, 1).link(1, 2).link(2, 3);
  const pinned = [reroot.pathSum(0, 3), reroot.pathSum(3, 0), reroot.pathMax(0, 3)];
  reroot.makeRoot(3);
  strictEqual(reroot.pathSum(0, 3), pinned[0], "path sum is root-independent");
  strictEqual(reroot.pathSum(3, 0), pinned[1]);
  strictEqual(reroot.pathMax(0, 3), pinned[2]);

  // -- determinism -----------------------------------------------------------------------------
  function replay(seed) {
    const r = makeRng(seed);
    const tree = new LinkCutTree([1, 2, 3, 4, 5, 6, 7, 8]);
    const log = [];
    for (let step = 0; step < 60; step++) {
      const x = r(8);
      const y = r(8);
      if (x === y) { log.push("same"); continue; }
      if (!tree.connected(x, y)) tree.link(x, y);
      else { try { tree.cut(x, y); } catch { /* (x,y) is not an edge */ } }
      log.push(`${tree.pathSum(x, y)}:${tree.edgeList()}`);
    }
    return log.join(";");
  }
  strictEqual(replay(11), replay(11), "identical seeds must replay identically");

  console.log(`random ops: links=${links} cuts=${cuts} queries=${queries}`);
  console.log(`chain rotations=${chain.rotations} vertices=${chain.length}`);
  console.log("PASS");
}
