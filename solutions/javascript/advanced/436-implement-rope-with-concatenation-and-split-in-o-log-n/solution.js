/** Task 436 - Implement rope with concatenation and split in O(log n). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Rope = balanced binary tree over text chunks. Documented thresholds: every leaf holds at most
 * LEAF_LIMIT = 32 characters (a longer input string is pre-split into such leaves), and after every
 * concat/split the root height is checked against 2 * ceil(log2(#leaves + 1)) + 1 - if a rope grows
 * taller than that it is rebuilt from its leaves into a perfectly balanced tree. Each node caches its
 * character size, height and leaf count, so concat is O(1) plus an amortised rebuild, and split
 * recurses along one root-to-leaf path (plus an amortised rebuild of the two results). The rebuild
 * counter is documented and asserted: building a 200-character rope by 200 single-character concats
 * triggers at least one rebalance. Assertions compare toString() with a plain-string oracle across
 * LCG operation scripts including splits at 0 and at the full length, and check leaf, size and height
 * invariants after every operation.
 */
"use strict";

const LEAF_LIMIT = 32;

class Node {
  constructor(text = null, left = null, right = null) {
    this.text = text;
    this.left = left;
    this.right = right;
    if (text !== null) {
      this.size = text.length;
      this.height = 1;
      this.leaves = 1;
    } else {
      this.size = left.size + right.size;
      this.height = 1 + Math.max(left.height, right.height);
      this.leaves = left.leaves + right.leaves;
    }
  }
}

function leavesOf(text) {
  const chunks = [];
  for (let i = 0; i < text.length; i += LEAF_LIMIT) chunks.push(text.slice(i, i + LEAF_LIMIT));
  return chunks;
}

function buildBalanced(chunks, lo, hi) {
  if (lo > hi) return null;
  if (lo === hi) return new Node(chunks[lo]);
  const mid = (lo + hi) >> 1;
  return new Node(null, buildBalanced(chunks, lo, mid), buildBalanced(chunks, mid + 1, hi));
}

function heightBound(leafCount) {
  return 2 * Math.ceil(Math.log2(leafCount + 1)) + 1;
}

function rawConcat(left, right) {
  if (left === null) return right;
  if (right === null) return left;
  return new Node(null, left, right);
}

function splitRoot(node, index) {
  if (node === null) return [null, null];
  if (node.text !== null) {
    const head = node.text.slice(0, index);
    const tail = node.text.slice(index);
    return [head.length > 0 ? new Node(head) : null, tail.length > 0 ? new Node(tail) : null];
  }
  const leftSize = node.left.size;
  if (index < leftSize) {
    const [first, second] = splitRoot(node.left, index);
    return [first, rawConcat(second, node.right)];
  }
  if (index === leftSize) return [node.left, node.right];
  const [first, second] = splitRoot(node.right, index - leftSize);
  return [rawConcat(node.left, first), second];
}

class Rope {
  constructor(text = "") {
    const chunks = leavesOf(text);
    this.root = buildBalanced(chunks, 0, chunks.length - 1);
    this.rebuilds = 0;
  }

  static _fromRoot(root, rebuilds) {
    const rope = Object.create(Rope.prototype);
    rope.root = root;
    rope.rebuilds = rebuilds;
    return rope;
  }

  size() {
    return this.root === null ? 0 : this.root.size;
  }

  height() {
    return this.root === null ? 0 : this.root.height;
  }

  leafCount() {
    return this.root === null ? 0 : this.root.leaves;
  }

  toString() {
    const parts = [];
    const stack = [this.root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (node === null) continue;
      if (node.text !== null) parts.push(node.text);
      else {
        stack.push(node.right);
        stack.push(node.left);
      }
    }
    return parts.join("");
  }

  leafTexts() {
    const out = [];
    const stack = [this.root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (node === null) continue;
      if (node.text !== null) out.push(node.text);
      else {
        stack.push(node.right);
        stack.push(node.left);
      }
    }
    return out;
  }

  validate() {
    const walk = (node) => {
      if (node === null) return [0, 0, 0];
      if (node.text !== null) {
        if (node.text.length > LEAF_LIMIT || node.text.length === 0) return null;
        return [node.size, node.height, node.leaves];
      }
      const left = walk(node.left);
      const right = walk(node.right);
      if (left === null || right === null) return null;
      if (node.size !== left[0] + right[0]) return null;
      if (node.height !== 1 + Math.max(left[1], right[1])) return null;
      if (node.leaves !== left[2] + right[2]) return null;
      return [node.size, node.height, node.leaves];
    };
    const result = walk(this.root);
    if (result === null) return false;
    return result[0] === this.size() && result[1] === this.height() && result[2] === this.leafCount();
  }

  _balanced(root) {
    if (root === null || root.height <= heightBound(root.leaves)) return [root, 0];
    const chunks = [];
    const stack = [root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (node.text !== null) chunks.push(node.text);
      else {
        stack.push(node.right);
        stack.push(node.left);
      }
    }
    return [buildBalanced(chunks, 0, chunks.length - 1), 1];
  }

  concat(other) {
    if (this.root === null) return Rope._fromRoot(other.root, this.rebuilds + other.rebuilds);
    if (other.root === null) return Rope._fromRoot(this.root, this.rebuilds + other.rebuilds);
    const merged = rawConcat(this.root, other.root);
    const [root, rebuilt] = this._balanced(merged);
    return Rope._fromRoot(root, this.rebuilds + other.rebuilds + rebuilt);
  }

  split(index) {
    if (index < 0 || index > this.size()) throw new RangeError("split index out of bounds");
    let [first, second] = splitRoot(this.root, index);
    let firstRebuilt;
    let secondRebuilt;
    [first, firstRebuilt] = this._balanced(first);
    [second, secondRebuilt] = this._balanced(second);
    return [
      Rope._fromRoot(first, this.rebuilds + firstRebuilt),
      Rope._fromRoot(second, this.rebuilds + secondRebuilt),
    ];
  }
}

module.exports = { LEAF_LIMIT, Node, Rope, leavesOf, buildBalanced, heightBound, rawConcat, splitRoot };

if (require.main === module) {
  const assert = require("node:assert");
  let rope = new Rope("hello world");
  assert.strictEqual(rope.toString(), "hello world");
  assert.strictEqual(rope.size(), 11);
  assert.strictEqual(rope.leafCount(), 1);
  assert.strictEqual(rope.height(), 1);
  assert.ok(rope.validate());

  let parts = rope.split(0);
  assert.strictEqual(parts[0].toString(), "");
  assert.strictEqual(parts[0].size(), 0);
  assert.strictEqual(parts[1].toString(), "hello world");
  parts = rope.split(11);
  assert.strictEqual(parts[0].toString(), "hello world");
  assert.strictEqual(parts[1].toString(), "");
  assert.strictEqual(parts[1].size(), 0);
  parts = rope.split(5);
  assert.strictEqual(parts[0].toString(), "hello");
  assert.strictEqual(parts[1].toString(), " world");
  assert.strictEqual(parts[0].concat(parts[1]).toString(), "hello world");
  assert.ok(parts[0].height() <= heightBound(parts[0].leafCount()));

  const dataChars = [];
  for (let i = 0; i < 2000; i++) dataChars.push(String.fromCharCode(97 + ((i * 7 + i * i) % 26)));
  const data = dataChars.join("");
  rope = new Rope(data);
  assert.strictEqual(rope.toString(), data);
  assert.strictEqual(rope.size(), 2000);
  assert.strictEqual(rope.leafCount(), Math.ceil(2000 / LEAF_LIMIT));
  assert.ok(rope.height() <= heightBound(rope.leafCount()));
  assert.ok(rope.validate());
  assert.ok(Math.max(...rope.leafTexts().map((chunk) => chunk.length)) <= LEAF_LIMIT);
  parts = rope.split(999);
  assert.strictEqual(parts[0].toString() + parts[1].toString(), data);
  assert.strictEqual(parts[0].concat(parts[1]).toString(), data);

  rope = new Rope();
  for (let index = 0; index < 200; index++) {
    rope = rope.concat(new Rope(String.fromCharCode(97 + (index % 26))));
  }
  assert.strictEqual(rope.size(), 200);
  assert.ok(rope.rebuilds >= 1, `build rebuilds ${rope.rebuilds}`);
  const expectedBuilt = [];
  for (let i = 0; i < 200; i++) expectedBuilt.push(String.fromCharCode(97 + (i % 26)));
  assert.strictEqual(rope.toString(), expectedBuilt.join(""));
  assert.ok(rope.height() <= heightBound(rope.leafCount()));
  assert.ok(rope.validate());

  const initialChars = [];
  for (let i = 0; i < 64; i++) initialChars.push(String.fromCharCode(97 + ((i * 13 + 5) % 26)));
  rope = new Rope(initialChars.join(""));
  let oracle = initialChars.join("");
  let seed = 4360;
  for (let step = 0; step < 100; step++) {
    seed = (seed * 75 + 74) % 65537;
    const op = seed % 4;
    if (op === 0) {
      seed = (seed * 75 + 74) % 65537;
      const index = seed % (oracle.length + 1);
      const [first, second] = rope.split(index);
      assert.strictEqual(first.toString(), oracle.slice(0, index), `split left ${step}`);
      assert.strictEqual(second.toString(), oracle.slice(index), `split right ${step}`);
      if (seed % 2 === 0) {
        rope = first.concat(second);
      } else {
        rope = second.concat(first);
        oracle = oracle.slice(index) + oracle.slice(0, index);
      }
    } else if (op === 1) {
      seed = (seed * 75 + 74) % 65537;
      const index = seed % (oracle.length + 1);
      rope = rope.split(index)[0];
      oracle = oracle.slice(0, index);
    } else if (op === 2) {
      seed = (seed * 75 + 74) % 65537;
      const index = seed % (oracle.length + 1);
      rope = rope.split(index)[1];
      oracle = oracle.slice(index);
    } else {
      const extraChars = [];
      const length = seed % 20;
      for (let i = 0; i < length; i++) extraChars.push(String.fromCharCode(97 + ((step * 3 + i) % 26)));
      const extra = extraChars.join("");
      rope = rope.concat(new Rope(extra));
      oracle = oracle + extra;
    }
    assert.strictEqual(rope.toString(), oracle, `text ${step}`);
    assert.strictEqual(rope.size(), oracle.length, `size ${step}`);
    assert.ok(rope.height() <= heightBound(rope.leafCount()), `height ${step}`);
    assert.ok(rope.validate(), `validate ${step}`);
    assert.strictEqual(rope.leafTexts().join(""), oracle, `leaves ${step}`);
  }
  parts = rope.split(0);
  assert.strictEqual(parts[0].toString(), "");
  assert.strictEqual(parts[1].toString(), oracle);
  parts = rope.split(rope.size());
  assert.strictEqual(parts[1].toString(), "");
  assert.strictEqual(parts[0].toString(), oracle);
  console.log("PASS");
}
