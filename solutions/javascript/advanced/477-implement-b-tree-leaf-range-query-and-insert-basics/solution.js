/** Task 477 - Implement B+ tree leaf-range query and insert basics. (Tier: Advanced, CTCI Ch 14 — Databases) */
"use strict";

class BNode {
  constructor(isLeaf) {
    this.isLeaf = isLeaf;
    this.keys = [];
    this.children = []; // internal nodes only
    this.next = null; // leaf nodes only
  }
}

class BPlusTree {
  constructor(maxKeys) {
    this.maxKeys = maxKeys;
    this.root = new BNode(true);
    this.size = 0;
  }

  leafFor(key) {
    let node = this.root;
    while (!node.isLeaf) {
      let idx = 0;
      while (idx < node.keys.length && node.keys[idx] <= key) idx++;
      node = node.children[idx];
    }
    return node;
  }

  contains(key) {
    return this.leafFor(key).keys.includes(key);
  }

  allKeys() {
    let leaf = this.leafFor(-Infinity);
    const out = [];
    while (leaf !== null) {
      out.push(...leaf.keys);
      leaf = leaf.next;
    }
    return out;
  }

  range(lo, hi) {
    const out = [];
    if (lo > hi) return out;
    let leaf = this.leafFor(lo);
    while (leaf !== null) {
      for (const key of leaf.keys) {
        if (key > hi) return out;
        if (key >= lo) out.push(key);
      }
      leaf = leaf.next;
    }
    return out;
  }

  insert(key) {
    if (this.contains(key)) return false;
    const res = this.insertInto(this.root, key);
    if (res !== null) {
      const newRoot = new BNode(false);
      newRoot.keys = [res.sep];
      newRoot.children = [this.root, res.right];
      this.root = newRoot;
    }
    this.size += 1;
    return true;
  }

  insertInto(node, key) {
    if (node.isLeaf) {
      let pos = 0;
      while (pos < node.keys.length && node.keys[pos] < key) pos++;
      node.keys.splice(pos, 0, key);
      if (node.keys.length <= this.maxKeys) return null;
      const mid = Math.floor(node.keys.length / 2);
      const right = new BNode(true);
      right.keys = node.keys.slice(mid);
      node.keys = node.keys.slice(0, mid);
      right.next = node.next;
      node.next = right;
      return { sep: right.keys[0], right };
    }
    let idx = 0;
    while (idx < node.keys.length && node.keys[idx] <= key) idx++;
    const res = this.insertInto(node.children[idx], key);
    if (res === null) return null;
    node.keys.splice(idx, 0, res.sep);
    node.children.splice(idx + 1, 0, res.right);
    if (node.keys.length <= this.maxKeys) return null;
    const mid = Math.floor(node.children.length / 2);
    const promoted = node.keys[mid - 1];
    const rightNode = new BNode(false);
    rightNode.children = node.children.slice(mid);
    rightNode.keys = node.keys.slice(mid);
    node.children = node.children.slice(0, mid);
    node.keys = node.keys.slice(0, mid - 1);
    return { sep: promoted, right: rightNode };
  }
}

function lcg(seed) {
  return (seed * 75 + 74) % 65537;
}

function minKey(node) {
  while (!node.isLeaf) node = node.children[0];
  return node.keys[0];
}

function collectLeaves(node) {
  if (node.isLeaf) return [node];
  const out = [];
  for (const child of node.children) out.push(...collectLeaves(child));
  return out;
}

function chainLeaves(tree) {
  let leaf = tree.leafFor(-Infinity);
  const out = [];
  while (leaf !== null) {
    out.push(leaf);
    leaf = leaf.next;
  }
  return out;
}

function checkInvariants(tree) {
  const depths = [];
  const walk = (node, depth) => {
    const sorted = node.keys.slice().sort((a, b) => a - b);
    if (JSON.stringify(node.keys) !== JSON.stringify(sorted)) throw new Error("keys sorted");
    if (node.keys.length > tree.maxKeys) throw new Error("node capacity");
    if (node.isLeaf) {
      depths.push(depth);
      return;
    }
    if (node.children.length !== node.keys.length + 1) throw new Error("children = keys + 1");
    for (let i = 0; i < node.keys.length; i++) {
      if (minKey(node.children[i + 1]) !== node.keys[i]) throw new Error("separator equals right subtree minimum");
    }
    for (const child of node.children) walk(child, depth + 1);
  };
  walk(tree.root, 0);
  if (new Set(depths).size !== 1) throw new Error("all leaves at equal depth");
  const byChain = chainLeaves(tree);
  const byDfs = collectLeaves(tree.root);
  if (byChain.length !== byDfs.length) throw new Error("leaf chain covers every leaf");
  for (let i = 0; i < byChain.length; i++) {
    if (JSON.stringify(byChain[i].keys) !== JSON.stringify(byDfs[i].keys)) throw new Error("leaf chain order");
  }
}

function leafCount(tree) {
  return collectLeaves(tree.root).length;
}

function height(tree) {
  let depth = 1;
  let node = tree.root;
  while (!node.isLeaf) {
    node = node.children[0];
    depth += 1;
  }
  return depth;
}

function filterKeys(values, lo, hi) {
  return values.filter((v) => v >= lo && v <= hi);
}

module.exports = { BPlusTree, lcg, checkInvariants, leafCount, height, collectLeaves };

if (require.main === module) {
  const assert = require("node:assert");
  const tree = new BPlusTree(3);
  const brute = [];
  const probes = [[1, 12], [4, 9], [5, 5], [3, 3], [10, 20], [0, 0], [12, 12], [2, 11]];
  for (let key = 1; key <= 12; key++) {
    assert.ok(tree.insert(key));
    brute.push(key);
    checkInvariants(tree);
    for (const [lo, hi] of probes) {
      assert.deepStrictEqual(tree.range(lo, hi), filterKeys(brute, lo, hi), `${lo}..${hi}`);
    }
  }
  assert.deepStrictEqual(tree.root.keys, [5], "root separator hand-traced to [5]");
  assert.strictEqual(leafCount(tree), 6);
  assert.strictEqual(height(tree), 3);
  assert.deepStrictEqual(tree.allKeys(), brute);
  assert.ok(!tree.insert(7), "duplicates are rejected");
  assert.deepStrictEqual(tree.allKeys(), brute);
  assert.strictEqual(tree.size, 12);

  const tree2 = new BPlusTree(3);
  const brute2 = [];
  for (let key = 12; key >= 1; key--) {
    assert.ok(tree2.insert(key));
    brute2.push(key);
    checkInvariants(tree2);
    assert.deepStrictEqual(tree2.range(1, 12), brute2.slice().sort((a, b) => a - b));
  }
  assert.deepStrictEqual(tree2.allKeys(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

  const tree3 = new BPlusTree(4);
  const inserted = new Set();
  let seed = 4242;
  for (let i = 0; i < 40; i++) {
    seed = lcg(seed);
    const key = seed % 64;
    if (inserted.has(key)) {
      assert.ok(!tree3.insert(key));
    } else {
      assert.ok(tree3.insert(key));
      inserted.add(key);
    }
    checkInvariants(tree3);
  }
  const keysSorted = [...inserted].sort((a, b) => a - b);
  assert.ok(keysSorted.length > 25);
  assert.deepStrictEqual(tree3.allKeys(), keysSorted);
  for (let i = 0; i < 80; i++) {
    seed = lcg(seed);
    const lo = seed % 70;
    seed = lcg(seed);
    const hi = lo + (seed % (70 - lo));
    assert.deepStrictEqual(tree3.range(lo, hi), filterKeys(keysSorted, lo, hi), `${lo}..${hi}`);
  }
  assert.deepStrictEqual(tree3.range(65, 63), []);
  assert.deepStrictEqual(tree3.range(200, 300), []);
  assert.deepStrictEqual(tree3.range(0, 0), inserted.has(0) ? [0] : []);
  console.log("PASS");
}
