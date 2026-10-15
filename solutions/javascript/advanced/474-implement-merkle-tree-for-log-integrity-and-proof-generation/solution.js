/** Task 474 - Implement Merkle tree for log integrity and proof generation. (Tier: Advanced, CTCI Beyond — Security & Privacy) */
"use strict";

const MASK = 0xFFFFFFFF;

function fnv1a(bytes) {
  let h = 2166136261;
  for (const b of bytes) {
    h ^= b;
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

function hashLeaf(value) {
  const data = [0];
  for (let i = 0; i < value.length; i++) data.push(value.charCodeAt(i));
  return fnv1a(data);
}

function hashNode(left, right) {
  const data = [
    (left >>> 24) & 0xff, (left >>> 16) & 0xff, (left >>> 8) & 0xff, left & 0xff,
    (right >>> 24) & 0xff, (right >>> 16) & 0xff, (right >>> 8) & 0xff, right & 0xff,
  ];
  return fnv1a(data);
}

class MerkleTree {
  constructor(values) {
    this.leaves = values.slice();
    this.levels = [this.leaves.map((v) => hashLeaf(v))];
    let level = this.levels[0];
    while (level.length > 1) {
      const next = [];
      for (let i = 0; i + 1 < level.length; i += 2) next.push(hashNode(level[i], level[i + 1]));
      if (level.length % 2 === 1) next.push(level[level.length - 1]); // promote
      this.levels.push(next);
      level = next;
    }
  }

  root() {
    return this.levels[this.levels.length - 1][0];
  }

  proof(index) {
    const path = [];
    let idx = index;
    for (let li = 0; li < this.levels.length - 1; li++) {
      const level = this.levels[li];
      if (idx % 2 === 1) path.push([level[idx - 1], 0]); // sibling on the left
      else if (idx + 1 < level.length) path.push([level[idx + 1], 1]);
      idx = Math.floor(idx / 2);
    }
    return path;
  }
}

function verify(root, value, path) {
  let h = hashLeaf(value);
  for (const [sibling, side] of path) {
    h = side === 0 ? hashNode(sibling, h) : hashNode(h, sibling);
  }
  return h === root;
}

module.exports = { MerkleTree, verify, hashLeaf, hashNode, fnv1a };

if (require.main === module) {
  const assert = require("node:assert");
  const leaves = [];
  for (let i = 0; i < 7; i++) leaves.push(`tx${i}`);
  const tree = new MerkleTree(leaves);
  assert.strictEqual(tree.root(), 818430076, "root of the 7-record log");
  for (let i = 0; i < 7; i++) assert.ok(verify(tree.root(), leaves[i], tree.proof(i)), `proof must verify ${i}`);
  assert.strictEqual(tree.proof(0).length, 3);
  assert.strictEqual(tree.proof(6).length, 2);
  assert.ok(!verify(tree.root(), "tx3x", tree.proof(3)), "tampered leaf must fail");
  assert.ok(!verify(tree.root() ^ 1, leaves[3], tree.proof(3)), "wrong root must fail");
  const bad = tree.proof(3).map((step) => step.slice());
  bad[0][0] ^= 1;
  assert.ok(!verify(tree.root(), leaves[3], bad), "tampered sibling must fail");
  assert.strictEqual(new MerkleTree(leaves).root(), tree.root());
  const rebuilt = new MerkleTree(leaves);
  for (let i = 0; i < 7; i++) assert.ok(verify(rebuilt.root(), leaves[i], tree.proof(i)));

  const leaves8 = [];
  for (let i = 0; i < 8; i++) leaves8.push(`tx${i}`);
  const tree8 = new MerkleTree(leaves8);
  assert.strictEqual(tree8.root(), 955777637, "root of the 8-record log");
  for (let i = 0; i < 8; i++) {
    assert.strictEqual(tree8.proof(i).length, 3);
    assert.ok(verify(tree8.root(), leaves8[i], tree8.proof(i)));
  }
  assert.ok(!verify(tree8.root(), "tx0!", tree8.proof(0)));
  assert.notStrictEqual(tree8.root(), tree.root());

  const single = new MerkleTree(["only"]);
  assert.strictEqual(single.root(), hashLeaf("only"));
  assert.deepStrictEqual(single.proof(0), []);
  assert.ok(verify(single.root(), "only", []));
  console.log("PASS");
}
