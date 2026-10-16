/** Task 611 - Implement dynamic succinct trees (conceptual). (Tier: Expert, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A balanced-parentheses (BP) dynamic succinct tree: the tree is the sequence `(` children `)`
 * per node flattened into one bit vector, kept as a list of small blocks that each cache their
 * open/close counts and relative min/max excess so rank/select/excess never touch more blocks
 * than they need. Nodes are addressed by *preorder index*, resolved with select on the block
 * list, so the representation stays n + o(n) shaped instead of carrying per-node records - and
 * because indices shift on every insertion the harness re-derives its labels after each
 * mutation, which is what a real dynamic succinct tree needs a handle update for.
 *
 *   addLeaf(parentIndex, t) - insert `()` inside the parent after its t-th child (t === degree
 *                             appends as the last child, t === 0 inserts first)
 *   removeLeaf(index)       - delete a leaf pair; every splice is followed by a normalisation
 *                             pass that splits blocks above 2*BLOCK_TARGET and merges adjacent
 *                             blocks that fit again, keeping the block list O(n / BLOCK_TARGET)
 *
 * What is simplified (this is the conceptual version): rank/select within and across blocks is a
 * linear scan rather than a two-level index of O(log n) trees, the backwards excess search is
 * linear rather than block-skipping, and only leaf insert/delete is supported. Verified after
 * every single mutation: the sequence is well formed (excess never negative, zero exactly at the
 * end), every block cache matches a fresh recomputation, block sizes obey the split/merge
 * bounds, and every navigation op plus a full decode agree with an independent children-list
 * oracle whose labels the harness tracks across the index shifts.
 */
"use strict";

const BLOCK_TARGET = 8;              // blocks hold 1..2*BLOCK_TARGET bits (bit 0 = `(`, 1 = `)`)

class Block {
  constructor(bits) { this.bits = bits; this.recompute(); }

  recompute() {
    let rel = 0;
    this.opens = 0; this.closes = 0; this.minRel = 0; this.maxRel = 0;
    for (const bit of this.bits) {
      if (bit === 0) { this.opens++; rel++; } else { this.closes++; rel--; }
      if (rel < this.minRel) this.minRel = rel;
      if (rel > this.maxRel) this.maxRel = rel;
    }
    this.rel = rel;
  }

  get length() { return this.bits.length; }
}

class DynamicBpTree {
  constructor(blocks) { this.blocks = blocks; }

  static fromChildren(children) {
    const bits = [];
    const walk = (v) => { bits.push(0); for (const c of children[v]) walk(c); bits.push(1); };
    if (children.length > 0) walk(0);
    const blocks = [];
    for (let i = 0; i < bits.length; i += BLOCK_TARGET) blocks.push(new Block(bits.slice(i, i + BLOCK_TARGET)));
    if (blocks.length === 0) blocks.push(new Block([]));
    return new DynamicBpTree(blocks);
  }

  get length() { return this.blocks.reduce((acc, b) => acc + b.length, 0); }
  get size() { return this.length / 2; }

  #locate(i) {
    let remaining = i;
    for (const block of this.blocks) {
      if (remaining < block.length) return { block, offset: remaining };
      remaining -= block.length;
    }
    throw new RangeError(`bit index ${i} out of range`);
  }

  bitAt(i) {
    if (i < 0 || i >= this.length) throw new RangeError(`bit index ${i} out of range`);
    const { block, offset } = this.#locate(i);
    return block.bits[offset];
  }

  rank1(i) {                                       // `)` in positions 0..i
    if (i < -1 || i > this.length - 1) throw new RangeError(`rank1 index ${i} out of range`);
    let remaining = i + 1;
    let count = 0;
    for (const block of this.blocks) {
      if (remaining <= 0) break;
      if (remaining >= block.length) { count += block.closes; remaining -= block.length; continue; }
      for (let k = 0; k < remaining; k++) count += block.bits[k];
      remaining = 0;
    }
    return count;
  }

  rank0(i) { return i < 0 ? 0 : i + 1 - this.rank1(i); }

  select0(j) {                                     // position of the (j + 1)-th `(`
    if (j < 0 || j >= this.size) throw new RangeError(`select0(${j}) out of range`);
    let seen = 0;
    let base = 0;
    for (const block of this.blocks) {
      if (seen + block.opens > j) {
        let local = j - seen;
        for (let k = 0; k < block.length; k++) if (block.bits[k] === 0 && local-- === 0) return base + k;
      }
      seen += block.opens;
      base += block.length;
    }
    throw new RangeError(`select0(${j}) out of range`);
  }

  excess(i) { return this.rank0(i) - this.rank1(i); }

  /** Matching `)` of the node whose `(` is at `open`: the excess returns to its level. */
  closeOf(open) {
    const level = this.excess(open - 1);
    for (let j = open + 1; j < this.length; j++) if (this.excess(j) === level) return j;
    throw new Error("unbalanced parenthesis sequence");
  }

  openPos(index) { return this.select0(index); }
  indexOf(open) { return this.rank0(open) - 1; }
  parentPos(index) { return this.openPos(index) === 0 ? -1 : this.#lastOpenAtLevel(this.openPos(index) - 1, this.excess(this.openPos(index) - 1) - 1); }
  parentIndex(index) { const parent = this.parentPos(index); return parent === -1 ? null : this.indexOf(parent); }
  firstChildIndex(index) { return this.childIndexAt(index, 0); }
  subtreeSize(index) { return (this.closeOf(this.openPos(index)) - this.openPos(index) + 1) / 2; }
  depth(index) { return this.excess(this.openPos(index) - 1); }

  /** Last `(` before `pos` whose own level is exactly `level` (the ancestor chain search). */
  #lastOpenAtLevel(pos, level) {
    for (let j = pos; j >= 0; j--) if (this.bitAt(j) === 0 && this.excess(j - 1) === level) return j;
    return -1;
  }

  childPosAt(index, t) {
    if (t < 0) return -1;
    const open = this.openPos(index);
    const close = this.closeOf(open);
    let pos = open + 1;
    for (let seen = 0; seen < t; seen++) {
      if (pos >= close || this.bitAt(pos) === 1) return -1;
      pos = this.closeOf(pos) + 1;
    }
    return pos < close && this.bitAt(pos) === 0 ? pos : -1;
  }

  childIndexAt(index, t) { const pos = this.childPosAt(index, t); return pos === -1 ? null : this.indexOf(pos); }

  degree(index) { let count = 0; while (this.childIndexAt(index, count) !== null) count++; return count; }

  siblingIndex(index) {
    const parent = this.parentPos(index);
    if (parent === -1) return null;
    const parentIndex = this.indexOf(parent);
    const degree = this.degree(parentIndex);
    for (let t = 0; t < degree; t++) {
      if (this.childIndexAt(parentIndex, t) === index) return t + 1 < degree ? this.childIndexAt(parentIndex, t + 1) : null;
    }
    return null;
  }

  /** Insert a leaf pair as the t-th child of parentIndex; returns its new preorder index. */
  addLeaf(parentIndex, t) {
    const degree = this.degree(parentIndex);
    if (t < 0 || t > degree) throw new RangeError(`cannot insert at child slot ${t} of ${parentIndex}`);
    const open = this.openPos(parentIndex);
    let position = open + 1;
    if (t > 0) {
      const previous = this.childPosAt(parentIndex, t - 1);
      if (previous === -1) throw new RangeError(`parent ${parentIndex} has no child ${t - 1}`);
      position = this.closeOf(previous) + 1;
    }
    let newIndex = parentIndex + 1;
    for (let seen = 0; seen < t; seen++) newIndex += this.subtreeSize(this.childIndexAt(parentIndex, seen));
    this.#splice(position, [0, 1], 0);
    return newIndex;
  }

  removeLeaf(index) {
    if (this.degree(index) !== 0) throw new Error(`node ${index} is not a leaf`);
    this.#splice(this.openPos(index), [], 2);
  }

  /** Replace `removed` bits at `pos` with `inserted`, then normalise the block list. */
  #splice(pos, inserted, removed) {
    for (let k = 0; k < removed; k++) {
      const { block, offset } = this.#locate(pos);
      block.bits.splice(offset, 1);
      block.recompute();
    }
    for (let k = 0; k < inserted.length; k++) {
      const { block, offset } = this.#locate(pos + k);
      block.bits.splice(offset, 0, inserted[k]);
      block.recompute();
    }
    this.#normalise();
  }

  #normalise() {
    this.blocks = this.blocks.filter((block) => block.length > 0);
    if (this.blocks.length === 0) this.blocks.push(new Block([]));
    for (let i = 0; i < this.blocks.length - 1; i++) {
      const merged = this.blocks[i].length + this.blocks[i + 1].length;
      if (this.blocks[i].length < BLOCK_TARGET && merged <= 2 * BLOCK_TARGET) {
        this.blocks[i] = new Block(this.blocks[i].bits.concat(this.blocks[i + 1].bits));
        this.blocks.splice(i + 1, 1);
        i--;
      }
    }
    for (let i = 0; i < this.blocks.length; i++) {
      while (this.blocks[i].length > 2 * BLOCK_TARGET) {
        const half = Math.floor(this.blocks[i].length / 2);
        this.blocks.splice(i, 1, new Block(this.blocks[i].bits.slice(0, half)), new Block(this.blocks[i].bits.slice(half)));
      }
    }
  }

  /** Structural invariants; throws with the broken invariant instead of returning false. */
  validate() {
    if (this.blocks.length === 0) throw new Error("block list must not be empty");
    let excess = 0;
    let total = 0;
    for (const block of this.blocks) {
      const fresh = new Block(block.bits);
      if (fresh.rel !== block.rel || fresh.minRel !== block.minRel || fresh.maxRel !== block.maxRel ||
          fresh.opens !== block.opens || fresh.closes !== block.closes) throw new Error("stale block cache");
      if (block.length === 0 && this.length > 0) throw new Error("empty block");
      if (block.length > 2 * BLOCK_TARGET) throw new Error("block above the split bound");
      for (const bit of block.bits) {
        excess += bit === 0 ? 1 : -1;
        total++;
        if (excess < 0) throw new Error("excess became negative: not a balanced sequence");
      }
    }
    if (excess !== 0) throw new Error("sequence does not close: excess != 0 at the end");
    if (total % 2 !== 0) throw new Error("sequence length must be even");
    return true;
  }

  /** Decode into children lists addressed by preorder index. */
  toChildren() {
    const children = Array.from({ length: this.size }, () => []);
    for (let index = 1; index < this.size; index++) children[this.parentIndex(index)].push(index);
    return children;
  }
}

/** Independent oracle: a children list mutated with the same ops and preorder renumbering. */
class ListTree {
  constructor() { this.children = [[]]; this.labels = [0]; this.nextLabel = 1; }

  subtreeSize(v) {
    let total = 1;
    for (const c of this.children[v]) total += this.subtreeSize(c);
    return total;
  }

  addLeaf(parent, t) {
    if (t < 0 || t > this.children[parent].length) throw new RangeError("child slot out of range");
    let index = parent + 1;
    for (let i = 0; i < t; i++) index += this.subtreeSize(this.children[parent][i]);
    const label = this.nextLabel++;
    for (const kids of this.children) for (let i = 0; i < kids.length; i++) if (kids[i] >= index) kids[i]++;
    this.children.splice(index, 0, []);
    this.labels.splice(index, 0, label);
    this.children[parent].splice(t, 0, index);
    return index;
  }

  removeLeaf(index) {
    if (this.children[index].length !== 0) throw new Error("not a leaf");
    const parent = this.children.findIndex((kids) => kids.includes(index));
    this.children[parent].splice(this.children[parent].indexOf(index), 1);
    this.children.splice(index, 1);
    this.labels.splice(index, 1);
    for (const kids of this.children) for (let i = 0; i < kids.length; i++) if (kids[i] > index) kids[i]--;
  }
}

/** Canonical description of a tree shape (labels included) so index shifts are tracked. */
function describeShape(children, labels) {
  const parent = new Array(children.length).fill(-1);
  for (let v = 0; v < children.length; v++) for (const c of children[v]) parent[c] = v;
  const depth = new Array(children.length).fill(0);
  for (let v = 0; v < children.length; v++) depth[v] = parent[v] === -1 ? 0 : depth[parent[v]] + 1;
  const size = new Array(children.length).fill(1);
  for (let v = children.length - 1; v >= 0; v--) if (parent[v] !== -1) size[parent[v]] += size[v];
  return children.map((kids, v) => [labels[v], depth[v], size[v], parent[v] === -1 ? null : labels[parent[v]],
    kids.map((c) => labels[c])]);
}

/** Compares the dynamic BP tree against the oracle: structure, navigation, block invariants. */
function checkAgainstOracle(tree, oracle) {
  const assert = require("node:assert");
  tree.validate();
  assert.strictEqual(tree.size, oracle.children.length, "node count vs oracle");
  assert.deepStrictEqual(describeShape(tree.toChildren(), oracle.labels),
    describeShape(oracle.children, oracle.labels), "navigation/decode signature vs oracle");
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

module.exports = { DynamicBpTree, ListTree, Block, describeShape, BLOCK_TARGET };

if (require.main === module) {
  const assert = require("node:assert");

  // Hand-built shape: 0 -> {1, 3}, 1 -> {2}; navigation pinned value by value.
  const tree = DynamicBpTree.fromChildren([[1, 3], [2], [], []]);
  const oracle = new ListTree();
  oracle.addLeaf(0, 0); oracle.addLeaf(1, 0); oracle.addLeaf(0, 1);
  checkAgainstOracle(tree, oracle);
  for (const [actual, expected, label] of [
    [tree.size, 4, "size"], [tree.depth(1), 1, "depth(1)"], [tree.depth(2), 2, "depth(2)"],
    [tree.subtreeSize(0), 4, "subtreeSize(0)"], [tree.subtreeSize(1), 2, "subtreeSize(1)"],
    [tree.parentIndex(0), null, "root parent"], [tree.parentIndex(2), 1, "parent(2)"],
    [tree.siblingIndex(1), 3, "sibling(1)"], [tree.siblingIndex(3), null, "sibling(3)"],
    [tree.childIndexAt(0, 0), 1, "child(0,0)"], [tree.childIndexAt(0, 1), 3, "child(0,1)"],
    [tree.childIndexAt(0, 2), null, "child(0,2)"], [tree.childIndexAt(2, 0), null, "child(2,0)"],
  ]) assert.strictEqual(actual, expected, label);

  // Insertion at every child slot, then a nested insert and a removal, all oracle checked.
  const grown = DynamicBpTree.fromChildren([[]]);
  const grownOracle = new ListTree();
  for (let t = 0; t < 5; t++) {
    assert.strictEqual(grown.addLeaf(0, t), 1 + t, `new leaf lands at preorder slot ${1 + t}`);
    grownOracle.addLeaf(0, t);
    checkAgainstOracle(grown, grownOracle);
  }
  assert.strictEqual(grown.degree(0), 5);
  assert.strictEqual(grown.addLeaf(2, 0), 3, "nested leaf takes preorder slot 3");
  grownOracle.addLeaf(2, 0);
  checkAgainstOracle(grown, grownOracle);
  assert.strictEqual(grown.depth(3), 2);
  assert.strictEqual(grown.parentIndex(3), 2);
  grown.removeLeaf(3);
  grownOracle.removeLeaf(3);
  checkAgainstOracle(grown, grownOracle);
  assert.strictEqual(grown.size, 6);

  // Randomised insert/delete schedule (append and interior slots) against the oracle.
  const next = xorshift32(0x5eed);
  const random = DynamicBpTree.fromChildren([[]]);
  const randomOracle = new ListTree();
  for (let step = 0; step < 160; step++) {
    if (step % 3 !== 2) {
      const target = Math.floor(next() * random.size);
      const slot = Math.floor(next() * (random.degree(target) + 1));
      const sizeBefore = random.subtreeSize(target);
      if (slot === random.degree(target)) assert.strictEqual(random.addLeaf(target, slot), target + sizeBefore);
      else random.addLeaf(target, slot);
      randomOracle.addLeaf(target, slot);
    } else {
      const leaves = [];
      for (let index = 0; index < random.size; index++) if (random.degree(index) === 0) leaves.push(index);
      if (leaves.length > 0) {
        const victim = leaves[Math.floor(next() * leaves.length)];
        random.removeLeaf(victim);
        randomOracle.removeLeaf(victim);
      }
    }
    checkAgainstOracle(random, randomOracle);
  }
  assert.ok(random.blocks.every((b) => b.length <= 2 * BLOCK_TARGET), "split bound");
  assert.ok(random.blocks.length < random.size, "blocks stay fewer than nodes");

  // Degenerate and boundary cases.
  const empty = DynamicBpTree.fromChildren([]);
  empty.validate();
  assert.strictEqual(empty.size, 0);
  assert.deepStrictEqual(empty.toChildren(), []);
  assert.throws(() => empty.select0(0), RangeError);
  const single = DynamicBpTree.fromChildren([[]]);
  assert.strictEqual(single.size, 1);
  assert.strictEqual(single.bitAt(0), 0);
  assert.strictEqual(single.bitAt(1), 1);
  assert.throws(() => single.bitAt(2), RangeError);
  assert.throws(() => single.rank1(2), RangeError);
  const pair = DynamicBpTree.fromChildren([[1], []]);
  assert.throws(() => pair.removeLeaf(0), /not a leaf/);
  assert.throws(() => pair.addLeaf(0, 3), RangeError);
  assert.throws(() => pair.addLeaf(1, -1), RangeError);
  pair.removeLeaf(1);
  pair.validate();
  assert.deepStrictEqual(pair.toChildren(), [[]]);
  assert.throws(() => new DynamicBpTree([new Block([0])]).validate(), /close|negative/);
  console.log("PASS");
}
