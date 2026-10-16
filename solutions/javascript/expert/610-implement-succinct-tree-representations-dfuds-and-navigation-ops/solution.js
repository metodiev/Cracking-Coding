/** Task 610 - Implement succinct tree representations (DFUDS) and navigation ops. (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Depth-First Unary Degree Sequence (DFUDS). A node is encoded as `(`^degree followed by `)`,
 * the encodings are concatenated in preorder, and one leading `(` makes the whole thing a
 * balanced 2n-bit vector. Every navigation op is then rank/select/excess on that vector, with
 * no pointers and no per-node records:
 *
 *   degree(v)      = run of `(` before the `)` that identifies v (the artificial leading `(`
 *                    is discounted for the root),   start(v) = v - degree(v)
 *   firstChild(v)  = the first `)` after v (child(v, t+1) = first `)` after subtreeEnd(child t))
 *   subtreeEnd(v)  = re-parse v's encoding and its descendants' encodings (memoized)
 *   parent(v)      = nearest preorder predecessor whose subtreeEnd reaches subtreeEnd(v)
 *   subtreeSize(v) = rank_)(subtreeEnd(v)) - rank_)(start(v) - 1)
 *
 * What is simplified: subtreeEnd() re-parses the encoding and parent() walks back over preorder
 * predecessors (O(nodes) worst case) instead of using the constant-time excess/minExcess jump
 * of a production DFUDS, and rank/select sit on a plain word-prefix directory. Verified: the
 * encoding is balanced and valid, decoding the vector reproduces the original children lists
 * exactly, every navigation op matches a plain-tree oracle for every node of hand-built and
 * random trees, subtreeEnd agrees with a second definition descending the child chain, and
 * malformed vectors are rejected.
 */
"use strict";

class Bitvector {          // bit 1 means `)`, bit 0 means `(`
  constructor(digits) {
    this.bits = typeof digits === "string" ? Array.from(digits, (c) => (c === ")" ? 1 : 0)) : Array.from(digits, Number);
    this.n = this.bits.length;
    this.ones = 0;
    this.#prefix = new Array(Math.ceil(this.n / 32) + 1).fill(0);
    for (let i = 0; i < this.n; i++) {
      if (i % 32 === 0) this.#prefix[i / 32] = this.ones;
      this.ones += this.bits[i];
    }
    this.#prefix[Math.ceil(this.n / 32)] = this.ones;
  }

  #prefix;

  get length() { return this.n; }

  rank1(i) {
    if (i < -1 || i > this.n - 1) throw new RangeError(`rank1 index ${i} out of range`);
    if (i < 0) return 0;
    const word = Math.floor(i / 32);
    let count = this.#prefix[word];
    for (let k = word * 32; k <= i; k++) count += this.bits[k];
    return count;
  }

  rank0(i) { return i < 0 ? 0 : i + 1 - this.rank1(i); }

  select1(j) {
    if (j < 0 || j >= this.ones) throw new RangeError(`select1(${j}) out of range`);
    const word = this.#prefix.findIndex((count) => count > j) - 1;
    let seen = this.#prefix[word];
    for (let k = word * 32; k < this.n; k++) {
      if (this.bits[k] === 1 && seen++ === j) return k;
    }
    throw new RangeError(`select1(${j}) out of range`);
  }

  /** Opens minus closes in positions 0..i (the DFUDS excess). */
  excess(i) { return 2 * this.rank0(i) - (i + 1); }

  toDigits() { return this.bits.map((b) => (b ? ")" : "(")).join(""); }
}

/** Concatenates one `(` per node plus its `)` in preorder, after the leading artificial `(`. */
function buildDfuuds(children) {
  if (children.length === 0) return "";
  const order = preorderOrder(children);
  const out = ["("];
  for (const v of order) {
    for (let i = 0; i < children[v].length; i++) out.push("(");
    out.push(")");
  }
  return out.join("");
}

function preorderOrder(children) {
  const order = [];
  const stack = children.length === 0 ? [] : [0];
  while (stack.length > 0) {
    const v = stack.pop();
    order.push(v);
    for (let i = children[v].length - 1; i >= 0; i--) stack.push(children[v][i]);
  }
  return order;
}

class DfudsTree {
  /** Node identifiers are *positions of the `)` that closes the node's encoding* (0-based). */
  constructor(bitvector) {
    this.bv = bitvector;
    this.size = bitvector.length / 2;
    this.#ends = new Map();
    this.#validate();
  }

  #ends;

  static fromChildren(children) {
    return new DfudsTree(new Bitvector(buildDfuuds(children)));
  }

  static fromString(digits) { return new DfudsTree(new Bitvector(digits)); }

  #validate() {
    const n = this.bv.length;
    if (n === 0) return;                        // empty forest
    if (n % 2 !== 0) throw new Error("DFUDS length must be even");
    if (this.bv.bits[0] !== 0) throw new Error("DFUDS must start with an opening parenthesis");
    if (this.bv.ones !== n / 2) throw new Error("DFUDS must have as many ')' as '('");
    for (let i = 0; i < n - 1; i++) {
      if (this.bv.excess(i) < 1) throw new Error(`DFUDS is not balanced at position ${i}`);
    }
    if (this.bv.excess(n - 1) !== 0) throw new Error("DFUDS must close exactly at the end");
  }

  get bitvector() { return this.bv; }

  /** Close positions in preorder == the node ids of an equally-ordered plain tree. */
  nodeClose(index) { return this.bv.select1(index); }

  nodeIndex(v) { return this.bv.rank1(v) - 1; }

  rootNode() { return this.size === 0 ? null : this.bv.select1(0); }

  isRoot(v) { return this.bv.rank1(v) === 1; }

  isLeaf(v) { return this.degree(v) === 0; }

  degree(v) {
    let previousOne = -1;
    const seen = this.bv.rank1(v - 1);
    if (seen > 0) previousOne = this.bv.select1(seen - 1);
    return v - 1 - previousOne - (this.isRoot(v) ? 1 : 0);
  }

  startPos(v) { return v - this.degree(v); }

  /** First `)` strictly after position `pos`, i.e. the close of the next node. */
  #closeAfter(pos) {
    const seen = this.bv.rank1(pos);
    if (seen >= this.bv.ones) return null;
    return this.bv.select1(seen);
  }

  firstChild(v) { return this.degree(v) === 0 ? null : this.#closeAfter(v); }

  childAt(v, index) {
    const kids = this.children(v);
    return index >= 0 && index < kids.length ? kids[index] : null;
  }

  children(v) {
    const count = this.degree(v);
    if (count === 0) return [];
    const kids = [this.#closeAfter(v)];
    while (kids.length < count) kids.push(this.#closeAfter(this.subtreeEnd(kids[kids.length - 1])));
    return kids;
  }

  /** Definitional re-parse: greedily read encodings from start(v) until the subtree is done.
   *  A production DFUDS would jump straight to that position with an excess/minExcess index;
   *  here the parse is memoized per node, which keeps the whole battery well under a second. */
  subtreeEnd(v) {
    const cached = this.#ends.get(v);
    if (cached !== undefined) return cached;
    let pos = this.startPos(v);
    let pending = 1;
    let close = pos;
    while (pending > 0) {
      let degree = 0;
      while (this.bv.bits[pos + degree] === 0) degree++;
      close = pos + degree;
      pending += degree - 1;
      pos = close + 1;
    }
    this.#ends.set(v, close);
    return close;
  }

  /** Independent second definition: the subtree of v ends with the subtree of its last child. */
  subtreeEndByChildChain(v) {
    const kids = this.children(v);
    return kids.length === 0 ? v : this.subtreeEndByChildChain(kids[kids.length - 1]);
  }

  subtreeSize(v) { return this.bv.rank1(this.subtreeEnd(v)) - this.bv.rank1(this.startPos(v) - 1); }

  /** Previous node in preorder (its close is the last `)` before v's encoding). */
  #prevPreorder(v) {
    const seen = this.bv.rank1(this.startPos(v) - 1);
    return seen === 0 ? null : this.bv.select1(seen - 1);
  }

  parent(v) {
    if (this.size === 0 || this.isRoot(v)) return null;
    const end = this.subtreeEnd(v);
    let candidate = this.#prevPreorder(v);
    while (candidate !== null && this.subtreeEnd(candidate) < end) candidate = this.#prevPreorder(candidate);
    return candidate;
  }

  sibling(v) {
    const u = this.parent(v);
    if (u === null) return null;
    const kids = this.children(u);
    const index = kids.indexOf(v);
    return index + 1 < kids.length ? kids[index + 1] : null;
  }

  previousSibling(v) {
    const u = this.parent(v);
    if (u === null) return null;
    const kids = this.children(u);
    const index = kids.indexOf(v);
    return index > 0 ? kids[index - 1] : null;
  }

  depth(v) {
    let d = 0;
    for (let u = this.parent(v); u !== null; u = this.parent(u)) d++;
    return d;
  }

  /** Decode the bit vector back into a plain children-list representation (ids in preorder). */
  toChildren() {
    const children = Array.from({ length: this.size }, () => []);
    for (let index = 0; index < this.size; index++) {
      const v = this.nodeClose(index);
      const u = this.parent(v);
      if (u === null) {
        if (index !== 0) throw new Error("DFUDS encodes more than one root");
      } else {
        children[this.nodeIndex(u)].push(index);
      }
    }
    return children;
  }
}

/** Plain-tree oracle used to pin every navigation result down. */
function treeOracle(children) {
  const n = children.length;
  const parent = new Array(n).fill(-1);
  const sibling = new Array(n).fill(-1);
  for (let v = 0; v < n; v++) {
    for (const [i, c] of children[v].entries()) {
      parent[c] = v;
      if (i + 1 < children[v].length) sibling[c] = children[v][i + 1];
    }
  }
  const depth = new Array(n).fill(0);
  for (let v = 0; v < n; v++) depth[v] = parent[v] === -1 ? 0 : depth[parent[v]] + 1;
  const size = new Array(n).fill(1);
  for (let v = n - 1; v >= 0; v--) if (parent[v] !== -1) size[parent[v]] += size[v];
  return { parent, sibling, depth, size };
}

/** Random tree with preorder-numbered ids (parent[i] < i, so ids must be renumbered). */
function randomTree(nodeCount, maxParents, next) {
  const rawChildren = Array.from({ length: nodeCount }, () => []);
  for (let v = 1; v < nodeCount; v++) {
    const limit = Math.min(v, maxParents);
    rawChildren[Math.floor(next() * limit)].push(v);
  }
  const renumbered = Array.from({ length: nodeCount }, () => []);
  const oldToNew = new Array(nodeCount).fill(-1);
  let counter = 0;
  const stack = [0];
  while (stack.length > 0) {
    const v = stack.pop();
    oldToNew[v] = counter++;
    const kids = rawChildren[v];
    for (let i = kids.length - 1; i >= 0; i--) stack.push(kids[i]);
  }
  for (let v = 0; v < nodeCount; v++) {
    for (const c of rawChildren[v]) renumbered[oldToNew[v]].push(oldToNew[c]);
  }
  for (const kids of renumbered) kids.sort((a, b) => a - b);
  return renumbered;
}

function xorshift32(seed) {
  let state = seed >>> 0 || 1;
  return () => {
    state ^= (state << 13) >>> 0; state >>>= 0;
    state ^= state >>> 17;
    state ^= (state << 5) >>> 0; state >>>= 0;
    return state / 4294967296;
  };
}

/** Cross-checks the DFUDS navigation of one plain tree against its oracle. */
function checkTree(children) {
  const assert = require("node:assert");
  const order = preorderOrder(children);
  assert.deepStrictEqual(order, Array.from({ length: children.length }, (_, i) => i), "test tree must be numbered in preorder");
  const tree = DfudsTree.fromChildren(children);
  const oracle = treeOracle(children);
  const n = children.length;
  assert.strictEqual(tree.size, n, "node count");
  assert.strictEqual(tree.bitvector.length, 2 * n, "bit vector length");
  assert.deepStrictEqual(tree.toChildren(), children, "DFUDS round-trip");
  const degreeSum = children.reduce((acc, kids) => acc + kids.length, 0);
  assert.strictEqual(degreeSum, n - 1, "edges = nodes - 1");
  for (let index = 0; index < n; index++) {
    const v = tree.nodeClose(index);
    assert.strictEqual(tree.nodeIndex(v), index, "nodeIndex");
    assert.strictEqual(tree.degree(v), children[index].length, `degree of ${index}`);
    assert.strictEqual(tree.isLeaf(v), children[index].length === 0, `isLeaf of ${index}`);
    assert.deepStrictEqual(tree.children(v), children[index].map((c) => tree.nodeClose(c)), `children of ${index}`);
    assert.strictEqual(tree.subtreeSize(v), oracle.size[index], `subtreeSize of ${index}`);
    assert.strictEqual(tree.depth(v), oracle.depth[index], `depth of ${index}`);
    const parent = tree.parent(v);
    assert.strictEqual(parent, oracle.parent[index] === -1 ? null : tree.nodeClose(oracle.parent[index]), `parent of ${index}`);
    const sibling = tree.sibling(v);
    assert.strictEqual(sibling, oracle.sibling[index] === -1 ? null : tree.nodeClose(oracle.sibling[index]), `sibling of ${index}`);
    const previous = tree.previousSibling(v);
    if (previous !== null) assert.strictEqual(tree.sibling(previous), v, `sibling round-trip at ${index}`);
    for (let t = 0; t < children[index].length; t++) {
      assert.strictEqual(tree.childAt(v, t), tree.nodeClose(children[index][t]), `childAt(${index},${t})`);
    }
    assert.strictEqual(tree.childAt(v, children[index].length), null, `childAt out of range at ${index}`);
    if (children[index].length > 0) assert.strictEqual(tree.parent(tree.firstChild(v)), v, `parent(firstChild(${index}))`);
    assert.strictEqual(tree.subtreeEnd(v), tree.subtreeEndByChildChain(v), `subtreeEnd vs child chain at ${index}`);
  }
  return tree;
}

module.exports = { DfudsTree, Bitvector, buildDfuuds, preorderOrder, treeOracle, randomTree };

if (require.main === module) {
  const assert = require("node:assert");

  checkTree([[]]);
  checkTree([[1], []]);                                  // root with one leaf
  checkTree([[1, 4], [2, 3], [], [], [5, 6], [], [7], []]);
  checkTree([Array.from({ length: 7 }, (_, i) => i + 1), ...Array.from({ length: 7 }, () => [])]);   // star
  checkTree(Array.from({ length: 6 }, (_, i) => (i + 1 < 6 ? [i + 1] : [])));                        // chain
  checkTree([[1, 4], [2, 3], [], [], [5, 6], [], []]);                                               // full binary

  // The first shape from the analysis above, spelled out as a vector.
  const exampleOne = DfudsTree.fromString("((()()))");
  assert.strictEqual(exampleOne.size, 4);
  assert.deepStrictEqual(exampleOne.toChildren(), [[1, 3], [2], [], []]);
  assert.strictEqual(exampleOne.bitvector.excess(exampleOne.bitvector.length - 1), 0);

  // Rank/select dictionary properties on the encoded vector of a non-trivial tree.
  const sample = checkTree(randomTree(40, 3, xorshift32(0x5eed)));
  const bv = sample.bitvector;
  assert.strictEqual(bv.ones, sample.size, "one ')' per node");
  assert.strictEqual(bv.toDigits(), buildDfuuds(sample.toChildren()), "re-encoding is identical");
  for (let i = 0; i < bv.length; i++) {
    const rank = bv.rank1(i);
    assert.strictEqual(bv.rank0(i) + rank, i + 1, `rank0 + rank1 at ${i}`);
    if (rank > 0) assert.ok(bv.select1(rank - 1) <= i, `select1(rank1(${i}) - 1) <= ${i}`);
    if (bv.bits[i] === 1) assert.strictEqual(bv.rank1(bv.select1(rank - 1)), rank, `inverse at ${i}`);
  }
  for (let j = 0; j < bv.ones; j++) assert.strictEqual(bv.rank1(bv.select1(j)), j + 1, `select/rank at ${j}`);
  assert.throws(() => bv.rank1(bv.length), RangeError);
  assert.throws(() => bv.select1(bv.ones), RangeError);

  // Random trees: full navigation cross-check.
  const next = xorshift32(0xc0ffee);
  for (let trial = 0; trial < 40; trial++) {
    const nodes = 2 + Math.floor(next() * 58);
    checkTree(randomTree(nodes, 1 + Math.floor(next() * 5), next));
  }

  // Boundary shapes: empty forest, single node, leaf root, and a deep chain.
  const empty = DfudsTree.fromChildren([]);
  assert.strictEqual(empty.size, 0);
  assert.strictEqual(empty.rootNode(), null);
  assert.deepStrictEqual(empty.toChildren(), []);
  const single = checkTree([[]]);
  assert.strictEqual(single.degree(single.rootNode()), 0);
  assert.strictEqual(single.parent(single.rootNode()), null);
  assert.strictEqual(single.subtreeEnd(single.rootNode()), 1);
  assert.strictEqual(single.firstChild(single.rootNode()), null);
  const chain = checkTree(Array.from({ length: 30 }, (_, i) => (i + 1 < 30 ? [i + 1] : [])));
  assert.strictEqual(chain.depth(chain.nodeClose(29)), 29);
  assert.strictEqual(chain.subtreeSize(chain.rootNode()), 30);

  // Malformed vectors must be rejected, valid ones accepted.
  assert.throws(() => DfudsTree.fromString(")("), /opening parenthesis/);
  assert.throws(() => DfudsTree.fromString("((("), /even/);
  assert.throws(() => DfudsTree.fromString("(()"), /even/);
  assert.throws(() => DfudsTree.fromString("()()"), /not balanced at position/);
  assert.throws(() => DfudsTree.fromString("[["), /as many/);
  assert.throws(() => DfudsTree.fromString("(("), /as many/);
  const chainOfTwo = DfudsTree.fromString("(())");
  assert.strictEqual(chainOfTwo.size, 2);
  assert.deepStrictEqual(chainOfTwo.toChildren(), [[1], []]);
  console.log("PASS");
}
