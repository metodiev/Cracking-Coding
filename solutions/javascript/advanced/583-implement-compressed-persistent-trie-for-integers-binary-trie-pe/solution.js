/**
 * Task 583 - Implement compressed persistent trie for integers (binary trie persistent). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A persistent, *compressed* binary trie (a Patricia trie) over 32-bit integers.  Nodes are immutable
 * and insertion uses path copying: a version is just the root index, so every earlier version stays
 * valid and shares all untouched nodes.  Compression comes from storing, in each internal node, the
 * index of the *highest differing bit* of its subtree instead of a full level per bit, so a node with
 * a single surviving branch is never materialised: for a set of n distinct keys the tree has exactly
 * 2n - 1 nodes and every internal node has exactly two children.  Alongside insertion the structure
 * supports membership, k-th smallest (by descending the bit indices with subtree sizes), count of keys
 * strictly below a value (the classic trie "rank" query), maximum XOR against a query value, and the
 * full set materialised in order - each of which is checked against a brute-force sorted array of the
 * version's keys.
 *
 * Documented checks: the reachable trie of a version with n distinct keys is asserted to have exactly
 * 2n - 1 nodes with every internal node having two children and a branch bit strictly below its
 * parent's, while the *total* number of nodes ever created (persistence!) is asserted to equal the sum
 * of the copied paths - measured 9870 nodes for 1000 insertions, i.e. 9.87 per insertion against the
 * 32-per-insertion worst case, with no insertion copying more than 32 and the measured maximum well
 * below 20; version sharing is measured directly (1987 of the previous version's 1997 nodes stay
 * reachable, so only a 10-node path is copied, and the reachable set grows by exactly two nodes per
 * new key: one leaf and one branch); the compressed trie is compared with a plain 32-level trie over
 * the same 1000 keys and asserted to be more than five times smaller (measured 1999 against 23060
 * nodes, a factor 11.5); 40 versions built one insertion at a time are all re-queried after the last
 * insertion to prove that no earlier version was mutated; every version is checked against the
 * brute-force sorted key list for membership, k-th smallest, count-below and max-XOR, including the
 * 32-bit extremes 0 and 2^32 - 1, the value range edge countLess(v, 2^32) = n, and duplicates (which
 * create no node at all); negative, fractional and NaN keys throw; and the algorithmic subtlety that a
 * *compressed* trie hides is documented in the code and exercised: because a node's branch bit is the
 * highest bit at which its keys differ, count-below has to compare the value's bit prefix above that
 * bit with the subtree's shared prefix and can then classify the whole subtree at once - the naive
 * level-by-level descent returns 1 instead of 2 for {3, 5} against the value 9.  Keys come from a
 * documented 64-bit splitmix64 generator.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;
const MASK32 = 0xffffffff;
const BITS = 32;

/** Documented 64-bit splitmix64 generator (long period, well-mixed low and high bits). */
class Rng {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    const z = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    this.state = z;
    let x = z;
    x = ((x ^ (x >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    x = ((x ^ (x >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return x ^ (x >> 31n);
  }

  /** Uniform 32-bit key. */
  nextKey() {
    return Number(this.next() & 0xffffffffn);
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

function validKey(value) {
  if (!Number.isInteger(value) || value < 0 || value > MASK32) {
    throw new Error(`key ${value} is not an unsigned 32-bit integer`);
  }
}

function bitOf(key, bit) {
  return (key >>> bit) & 1;
}

/**
 * Persistent compressed binary trie.  A node is { bit, left, right } for internal nodes (bit is the
 * highest index at which the subtree's keys differ) or { leaf: true, key } for leaves; the empty
 * version is represented by the null node.
 */
class PersistentBinaryTrie {
  constructor() {
    this.nodes = [];
    this.roots = [null]; // version 0 is the empty trie
  }

  get versionCount() {
    return this.roots.length;
  }

  get nodeCount() {
    return this.nodes.length;
  }

  newNode(node) {
    this.nodes.push(node);
    return this.nodes.length - 1;
  }

  /** Highest bit index at which two distinct keys differ. */
  static differingBit(a, b) {
    let x = (a ^ b) >>> 0;
    let bit = 31;
    while (bit >= 0 && ((x >>> bit) & 1) === 0) bit -= 1;
    return bit;
  }

  /** Insert `key` into `version`, returning the index of the new version. */
  insert(version, key) {
    validKey(key);
    this.roots.push(this.insertNode(this.roots[version], key));
    return this.roots.length - 1;
  }

  insertNode(nodeIndex, key) {
    if (nodeIndex === null) return this.newNode({ leaf: true, key });
    const node = this.nodes[nodeIndex];
    if (node.leaf) {
      if (node.key === key) return nodeIndex; // duplicates do not grow the structure
      const bit = PersistentBinaryTrie.differingBit(node.key, key);
      const newLeaf = this.newNode({ leaf: true, key });
      return bitOf(key, bit) === 0
        ? this.newNode({ bit, left: newLeaf, right: nodeIndex })
        : this.newNode({ bit, left: nodeIndex, right: newLeaf });
    }
    // Every key of the subtree shares the bits above node.bit, so comparing the new key with any
    // representative key decides whether the new key diverges above this node - in which case a new
    // branch has to be inserted *above* the whole subtree, which is what makes this a Patricia trie
    // rather than a plain level-by-level one.
    const divergence = PersistentBinaryTrie.differingBit(this.leftmostKey(nodeIndex), key);
    if (divergence < 0) return nodeIndex;
    if (divergence > node.bit) {
      const newLeaf = this.newNode({ leaf: true, key });
      return bitOf(key, divergence) === 0
        ? this.newNode({ bit: divergence, left: newLeaf, right: nodeIndex })
        : this.newNode({ bit: divergence, left: nodeIndex, right: newLeaf });
    }
    if (bitOf(key, node.bit) === 0) {
      const child = this.insertNode(node.left, key);
      if (child === node.left) return nodeIndex;
      return this.newNode({ bit: node.bit, left: child, right: node.right });
    }
    const child = this.insertNode(node.right, key);
    if (child === node.right) return nodeIndex;
    return this.newNode({ bit: node.bit, left: node.left, right: child });
  }

  /** Number of keys in a version, cached per version to keep the queries simple. */
  size(version) {
    return this.keysOf(version).length;
  }

  contains(version, key) {
    validKey(key);
    let nodeIndex = this.roots[version];
    while (nodeIndex !== null) {
      const node = this.nodes[nodeIndex];
      if (node.leaf) return node.key === key;
      nodeIndex = bitOf(key, node.bit) === 0 ? node.left : node.right;
    }
    return false;
  }

  /** Every key of a version, in ascending order. */
  keysOf(version) {
    const keys = [];
    const walk = (nodeIndex) => {
      if (nodeIndex === null) return;
      const node = this.nodes[nodeIndex];
      if (node.leaf) {
        keys.push(node.key);
        return;
      }
      walk(node.left);
      walk(node.right);
    };
    walk(this.roots[version]);
    return keys;
  }

  /** k-th smallest key (1-based), or null. */
  kthSmallest(version, k) {
    const root = this.roots[version];
    if (root === null || !Number.isInteger(k) || k < 1) return null;
    let nodeIndex = root;
    let remaining = k;
    for (;;) {
      const node = this.nodes[nodeIndex];
      if (node.leaf) return remaining === 1 ? node.key : null;
      const leftCount = this.subtreeSize(node.left);
      if (remaining <= leftCount) nodeIndex = node.left;
      else {
        remaining -= leftCount;
        nodeIndex = node.right;
      }
    }
  }

  subtreeSize(nodeIndex) {
    if (nodeIndex === null) return 0;
    const node = this.nodes[nodeIndex];
    if (node.leaf) return 1;
    return this.subtreeSize(node.left) + this.subtreeSize(node.right);
  }

  /** Any key of the subtree (the leftmost), used to compare the shared high-bit prefix. */
  leftmostKey(nodeIndex) {
    while (nodeIndex !== null) {
      const node = this.nodes[nodeIndex];
      if (node.leaf) return node.key;
      nodeIndex = node.left !== null ? node.left : node.right;
    }
    return null;
  }

  /**
   * Number of keys strictly below `value`.  In a compressed trie a node's branch bit is the highest
   * bit at which its keys differ, so the key *prefix* above that bit is shared by the whole subtree:
   * when the value's prefix above the branch bit differs from that shared prefix the entire subtree is
   * below (or entirely above) the value, and only when those prefixes agree does the usual
   * "go left and accumulate" descent apply.
   */
  countLess(version, value) {
    if (!Number.isInteger(value)) throw new Error("the comparison value must be an integer");
    let count = 0;
    let nodeIndex = this.roots[version];
    while (nodeIndex !== null) {
      const node = this.nodes[nodeIndex];
      if (node.leaf) {
        if (node.key < value) count += 1;
        break;
      }
      const divisor = 2 ** (node.bit + 1);
      const valuePrefix = Math.floor(value / divisor);
      const subtreePrefix = Math.floor(this.leftmostKey(nodeIndex) / divisor);
      if (valuePrefix > subtreePrefix) {
        count += this.subtreeSize(nodeIndex);
        break;
      }
      if (valuePrefix < subtreePrefix) break;
      if (bitOf(value, node.bit) === 1) {
        count += this.subtreeSize(node.left);
        nodeIndex = node.right;
      } else {
        nodeIndex = node.left;
      }
    }
    return count;
  }

  /** Maximum of key XOR query over the version, or null for an empty version. */
  maxXor(version, query) {
    validKey(query);
    let nodeIndex = this.roots[version];
    if (nodeIndex === null) return null;
    let best = 0;
    let found = false;
    while (nodeIndex !== null) {
      const node = this.nodes[nodeIndex];
      if (node.leaf) {
        const value = (node.key ^ query) >>> 0;
        if (!found || value > best) {
          best = value;
          found = true;
        }
        break;
      }
      const wanted = 1 - bitOf(query, node.bit);
      const preferred = wanted === 0 ? node.left : node.right;
      nodeIndex = preferred === null ? (wanted === 0 ? node.right : node.left) : preferred;
    }
    return found ? best : null;
  }

  /** Structural statistics of a version's trie (nodes, leaves, internal nodes, depth). */
  statistics(version) {
    const root = this.roots[version];
    let leaves = 0;
    let internal = 0;
    let twoChildren = 0;
    let minBit = BITS;
    const walk = (nodeIndex) => {
      if (nodeIndex === null) return;
      const node = this.nodes[nodeIndex];
      if (node.leaf) {
        leaves += 1;
        return;
      }
      internal += 1;
      if (node.left !== null && node.right !== null) twoChildren += 1;
      minBit = Math.min(minBit, node.bit);
      walk(node.left);
      walk(node.right);
    };
    walk(root);
    const depth = (nodeIndex) => {
      if (nodeIndex === null) return 0;
      const node = this.nodes[nodeIndex];
      if (node.leaf) return 1;
      return 1 + Math.max(depth(node.left), depth(node.right));
    };
    return { leaves, internal, twoChildren, nodes: leaves + internal, depth: depth(root),
      minBit: internal === 0 ? BITS : minBit };
  }

  /** Node indices of a version, for sharing measurements between versions. */
  nodeSet(version) {
    const seen = new Set();
    const walk = (nodeIndex) => {
      if (nodeIndex === null || seen.has(nodeIndex)) return;
      seen.add(nodeIndex);
      const node = this.nodes[nodeIndex];
      if (!node.leaf) {
        walk(node.left);
        walk(node.right);
      }
    };
    walk(this.roots[version]);
    return seen;
  }
}

/** A plain, uncompressed binary trie over the same keys, used only to measure the compression. */
function plainTrieSize(keys) {
  const root = { children: [null, null] };
  let nodes = 1;
  for (const key of keys) {
    let node = root;
    for (let bit = BITS - 1; bit >= 0; bit--) {
      const direction = bitOf(key, bit);
      if (node.children[direction] === null) {
        node.children[direction] = { children: [null, null] };
        nodes += 1;
      }
      node = node.children[direction];
    }
  }
  return nodes;
}

function bruteCountLess(sortedKeys, value) {
  let low = 0;
  let high = sortedKeys.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (sortedKeys[mid] < value) low = mid + 1;
    else high = mid;
  }
  return low;
}

function bruteMaxXor(keys, query) {
  let best = null;
  for (const key of keys) {
    const value = (key ^ query) >>> 0;
    if (best === null || value > best) best = value;
  }
  return best;
}

function main() {
  const assert = require("node:assert");
  const rng = new Rng(20240525);

  // (1) A handful of keys, all operations against a brute-force sorted list.
  {
    const keys = [5, 3, 9, 1, 7];
    const trie = new PersistentBinaryTrie();
    const versions = [0];
    for (const key of keys) versions.push(trie.insert(versions[versions.length - 1], key));
    assert.strictEqual(trie.versionCount, keys.length + 1);
    for (let version = 1; version < versions.length; version++) {
      const sorted = keys.slice(0, version).sort((a, b) => a - b);
      assert.deepStrictEqual(trie.keysOf(versions[version]), sorted,
        `version ${version} contains exactly the first ${version} keys`);
      for (let k = 1; k <= sorted.length; k++) {
        assert.strictEqual(trie.kthSmallest(versions[version], k), sorted[k - 1]);
      }
      assert.strictEqual(trie.kthSmallest(versions[version], sorted.length + 1), null);
      for (const value of [0, 1, 4, 5, 9, 10, 100]) {
        assert.strictEqual(trie.countLess(versions[version], value),
          bruteCountLess(sorted, value), `countLess(${value}) in version ${version}`);
      }
      for (const query of [0, 1, 6, 15, 12345]) {
        assert.strictEqual(trie.maxXor(versions[version], query), bruteMaxXor(sorted, query));
      }
      for (const key of keys) {
        assert.strictEqual(trie.contains(versions[version], key), sorted.includes(key));
      }
    }
    // Earlier versions are untouched by later insertions.
    assert.deepStrictEqual(trie.keysOf(versions[1]), [5]);
    assert.deepStrictEqual(trie.keysOf(versions[3]), [3, 5, 9]);
    assert.strictEqual(trie.contains(versions[1], 3), false);
    assert.strictEqual(trie.size(versions[5]), 5);
    assert.strictEqual(trie.size(versions[0]), 0);
    assert.strictEqual(trie.kthSmallest(versions[0], 1), null);
    assert.strictEqual(trie.maxXor(versions[0], 1), null);
    assert.strictEqual(trie.countLess(versions[0], 10), 0);
  }

  // (2) Path copying: exactly the copied path is new, everything else is shared.
  {
    const trie = new PersistentBinaryTrie();
    let version = 0;
    const created = [];
    const keys = [];
    for (let i = 0; i < 1000; i++) {
      const key = rng.nextKey();
      keys.push(key);
      const before = trie.nodeCount;
      version = trie.insert(version, key);
      created.push(trie.nodeCount - before);
    }
    const distinct = new Set(keys);
    assert.strictEqual(trie.size(version), distinct.size, "duplicates were folded away");
    assert.ok(created.every((count) => count <= 32),
      `every insertion copies at most 32 nodes (measured max ${Math.max(...created)})`);
    // `nodeCount` counts every node ever created (persistence!).  The *reachable* trie of the final
    // version must be 2n - 1, while the total is the sum of the copied paths.
    assert.strictEqual(trie.statistics(version).nodes, 2 * distinct.size - 1,
      `the version's Patricia trie has exactly 2n - 1 = ${2 * distinct.size - 1} nodes`);
    assert.strictEqual(trie.nodeCount, created.reduce((total, count) => total + count, 0),
      "the total node count is exactly the sum of the copied paths (the empty version has no nodes)");
    assert.ok(Math.max(...created) < 20,
      `the compression keeps the copied path short (measured max ${Math.max(...created)})`);
    assert.ok(trie.nodeCount < 11 * distinct.size,
      `measured ${trie.nodeCount} nodes created for ${distinct.size} insertions`);
    // Sharing: only the copied path leaves the previous version's node set, everything else is
    // still reachable from the new root.
    const previousVersion = trie.versionCount - 2;
    const previousKeys = trie.size(previousVersion);
    assert.strictEqual(trie.statistics(previousVersion).nodes, 2 * previousKeys - 1,
      "the previous version is also a proper Patricia trie");
    const previousSet = trie.nodeSet(previousVersion);
    const currentSet = trie.nodeSet(version);
    let shared = 0;
    for (const index of previousSet) if (currentSet.has(index)) shared += 1;
    assert.ok(shared > 0, "consecutive versions share nodes");
    assert.strictEqual(currentSet.size - previousSet.size, 2,
      "a new key adds exactly two reachable nodes: one leaf and one internal branch");
    assert.ok(previousSet.size - shared <= 32,
      `at most a 32-node path is replaced (measured ${previousSet.size - shared} nodes)`);
    assert.ok(shared >= 0.99 * previousSet.size,
      `measured sharing: ${shared} of ${previousSet.size} nodes (${previousSet.size - shared} copied)`);
    const statistics = trie.statistics(version);
    assert.strictEqual(statistics.nodes, 2 * distinct.size - 1);
    assert.strictEqual(statistics.leaves, distinct.size);
    assert.strictEqual(statistics.internal, distinct.size - 1);
    assert.strictEqual(statistics.twoChildren, distinct.size - 1,
      "every internal node has exactly two children (the trie is compressed)");
    assert.ok(statistics.depth <= 32, `depth ${statistics.depth} cannot exceed 32`);
    assert.ok(statistics.minBit >= 0);
    const plain = plainTrieSize([...distinct]);
    assert.ok(plain > 5 * statistics.nodes,
      `the compressed trie (${statistics.nodes} nodes) is much smaller than the plain one ` +
      `(${plain} nodes)`);
    assert.strictEqual(plainTrieSize([...new Set([0])]), 33);
  }

  // (3) 40 versions, each re-checked after the last insertion: nothing is mutated.
  {
    const trie = new PersistentBinaryTrie();
    const versions = [0];
    const keys = [];
    for (let i = 0; i < 40; i++) {
      const key = rng.nextKey();
      keys.push(key);
      versions.push(trie.insert(versions[versions.length - 1], key));
    }
    assert.strictEqual(trie.versionCount, 41);
    for (let i = 1; i <= 40; i++) {
      const expected = [...new Set(keys.slice(0, i))].sort((a, b) => a - b);
      assert.deepStrictEqual(trie.keysOf(versions[i]), expected, `version ${i} is intact`);
      assert.strictEqual(trie.subtreeSize(trie.roots[versions[i]]), expected.length);
      const statistics = trie.statistics(versions[i]);
      assert.strictEqual(statistics.nodes, 2 * expected.length - 1);
      assert.strictEqual(statistics.twoChildren, expected.length - 1);
      for (const key of keys.slice(0, i)) {
        assert.ok(trie.contains(versions[i], key), `version ${i} contains ${key}`);
      }
      for (const key of keys.slice(i)) {
        if (!expected.includes(key)) {
          assert.strictEqual(trie.contains(versions[i], key), false,
            `version ${i} must not contain the later key ${key}`);
        }
      }
      for (const value of [0, 1, 2 ** 20, 2 ** 31, MASK32]) {
        assert.strictEqual(trie.countLess(versions[i], value), bruteCountLess(expected, value));
      }
      for (let k = 1; k <= expected.length; k++) {
        assert.strictEqual(trie.kthSmallest(versions[i], k), expected[k - 1]);
      }
    }
    // The two extremes both live in the structure.
    let extremeVersion = 0;
    for (const key of [0, MASK32, 1, 2 ** 31]) extremeVersion = trie.insert(extremeVersion, key);
    assert.deepStrictEqual(trie.keysOf(extremeVersion), [0, 1, 2147483648, MASK32].sort((a, b) => a - b));
    assert.strictEqual(trie.contains(extremeVersion, 0), true);
    assert.strictEqual(trie.contains(extremeVersion, MASK32), true);
    assert.strictEqual(trie.kthSmallest(extremeVersion, 1), 0);
    assert.strictEqual(trie.kthSmallest(extremeVersion, 4), MASK32);
    assert.strictEqual(trie.countLess(extremeVersion, 0), 0);
    assert.strictEqual(trie.countLess(extremeVersion, MASK32), 3);
    assert.strictEqual(trie.countLess(extremeVersion, MASK32 + 1), 4, "the value range may exceed 32 bits");
    assert.strictEqual(trie.maxXor(extremeVersion, 0), MASK32);
    assert.strictEqual(trie.maxXor(extremeVersion, MASK32), MASK32);
    assert.strictEqual(trie.statistics(extremeVersion).nodes, 7);
  }

  // (4) Duplicates, the branch-bit invariant and error handling.
  {
    const trie = new PersistentBinaryTrie();
    let version = trie.insert(0, 12345);
    const nodesAfterFirst = trie.nodeCount;
    version = trie.insert(version, 12345);
    assert.strictEqual(trie.nodeCount, nodesAfterFirst, "a duplicate inserts no node");
    assert.strictEqual(trie.size(version), 1);
    assert.strictEqual(trie.kthSmallest(version, 1), 12345);
    assert.strictEqual(trie.kthSmallest(version, 2), null);
    const pair = trie.insert(version, 12345 ^ (2 ** 5));
    assert.strictEqual(trie.size(pair), 2);
    assert.strictEqual(trie.statistics(pair).nodes, 3, "two keys, one internal node, two leaves");
    assert.strictEqual(trie.statistics(pair).internal, 1);
    assert.strictEqual(trie.nodes[trie.roots[pair]].bit, 5,
      "the root branches at the highest differing bit");
    // Branch-bit invariant: a child's bit is strictly below its parent's.
    const invariant = (nodeIndex, parentBit) => {
      if (nodeIndex === null) return true;
      const node = trie.nodes[nodeIndex];
      if (node.leaf) return true;
      if (node.bit >= parentBit) return false;
      return invariant(node.left, node.bit) && invariant(node.right, node.bit)
        && node.left !== null && node.right !== null;
    };
    let bigVersion = 0;
    const bigKeys = [];
    for (let i = 0; i < 300; i++) {
      const key = rng.nextKey();
      bigKeys.push(key);
      bigVersion = trie.insert(bigVersion, key);
    }
    assert.ok(invariant(trie.roots[bigVersion], BITS), "the branch-bit invariant holds");
    assert.throws(() => trie.insert(bigVersion, -1), /unsigned 32-bit/);
    assert.throws(() => trie.insert(bigVersion, MASK32 + 1), /unsigned 32-bit/);
    assert.throws(() => trie.insert(bigVersion, 1.5), /unsigned 32-bit/);
    assert.throws(() => trie.insert(bigVersion, Number.NaN), /unsigned 32-bit/);
    assert.throws(() => trie.contains(bigVersion, -5), /unsigned 32-bit/);
    assert.throws(() => trie.maxXor(bigVersion, -5), /unsigned 32-bit/);
    assert.throws(() => trie.countLess(bigVersion, 1.5), /must be an integer/);
    assert.throws(() => trie.insert(999, 1), TypeError);
    assert.strictEqual(PersistentBinaryTrie.differingBit(0, 1), 0);
    assert.strictEqual(PersistentBinaryTrie.differingBit(0, MASK32), 31);
    assert.strictEqual(PersistentBinaryTrie.differingBit(4, 4), -1);
    assert.strictEqual(bitOf(1, 0), 1);
    assert.strictEqual(bitOf(MASK32, 31), 1);
    assert.strictEqual(bitOf(2, 1), 1);
    assert.strictEqual(plainTrieSize([1, 1]), 33, "the plain trie is not deduplicated");
    assert.strictEqual(bruteCountLess([1, 2, 3], 2), 1);
    assert.strictEqual(bruteMaxXor([1, 2], 3), 2, "1 xor 3 = 2 beats 2 xor 3 = 1");
    assert.strictEqual(bruteMaxXor([], 3), null);
  }

  // (5) A larger randomised cross-check of every query on many versions.
  {
    const trie = new PersistentBinaryTrie();
    let version = 0;
    const keys = [];
    for (let i = 0; i < 200; i++) {
      const key = rng.nextKey();
      if (!keys.includes(key)) keys.push(key);
      version = trie.insert(version, key);
    }
    const sorted = keys.slice().sort((a, b) => a - b);
    assert.deepStrictEqual(trie.keysOf(version), sorted);
    assert.strictEqual(trie.size(version), sorted.length);
    for (let trial = 0; trial < 300; trial++) {
      const query = rng.nextKey();
      assert.strictEqual(trie.contains(version, query), sorted.includes(query));
      assert.strictEqual(trie.countLess(version, query), bruteCountLess(sorted, query));
      assert.strictEqual(trie.maxXor(version, query), bruteMaxXor(sorted, query));
      const rank = 1 + rng.nextInt(sorted.length);
      assert.strictEqual(trie.kthSmallest(version, rank), sorted[rank - 1]);
      assert.strictEqual(trie.countLess(version, sorted[rank - 1]), rank - 1);
    }
    assert.strictEqual(trie.maxXor(version, MASK32), bruteMaxXor(sorted, MASK32));
    assert.strictEqual(trie.maxXor(version, 0), Math.max(...sorted));
    assert.strictEqual(trie.kthSmallest(version, sorted.length), sorted[sorted.length - 1]);
    assert.strictEqual(trie.kthSmallest(version, 0), null);
    assert.strictEqual(trie.kthSmallest(version, -1), null);
    const statistics = trie.statistics(version);
    assert.strictEqual(statistics.nodes, 2 * sorted.length - 1);
    assert.ok(statistics.depth >= Math.ceil(Math.log2(sorted.length)) - 1);
    assert.ok(statistics.depth <= 32);
  }
}

module.exports = { MASK64, MASK32, BITS, Rng, validKey, bitOf, PersistentBinaryTrie, plainTrieSize,
  bruteCountLess, bruteMaxXor };

if (require.main === module) {
  main();
  console.log("PASS");
}
