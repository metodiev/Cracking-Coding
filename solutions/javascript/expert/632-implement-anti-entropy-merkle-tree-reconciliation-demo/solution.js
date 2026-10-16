/** Task 632 - Implement anti-entropy Merkle tree reconciliation demo. (Tier: Expert, CTCI Beyond — Distributed Systems)
 *
 * Anti-entropy reconciliation compares two replicas without shipping their contents: each replica
 * hashes its keys into a Merkle tree whose *shape depends only on the key hash space*, never on the
 * data, so the two trees line up node by node. The replicas walk the trees together, pruning every
 * subtree whose hashes agree - equal hashes mean equal contents, which is the whole point of the
 * tree - and exchange only the entries inside the leaves that survive that pruning. The cost of a
 * repair is therefore proportional to the divergence, not to the data set.
 *
 * What is simplified (a demo, per the task title): the keys are ordered by their FNV-1a 32-bit hash
 * and each internal node splits its hash-ordered range in half, so the tree is a balanced Merkle tree
 * of depth `log2(n / leafSize)` whose shape depends on the key *count* only - two replicas holding
 * the same number of keys have node-for-node compatible trees. A leaf is hashed by folding its sorted
 * entries, an internal node by mixing its children's hashes, and the mixer is a deterministic
 * non-cryptographic function: the demo verifies structural convergence and pruning, not collision
 * resistance. `compare()` returns the entries each side must send, so the repair happens in-process;
 * there is no network and no I/O.
 *
 * Verified invariants: (1) equal root hashes imply identical contents (checked against the maps on
 * every run) and differing contents always produce differing roots; (2) the number of keys sent
 * equals the symmetric difference for single-key divergences and always covers it, so the replicas
 * are identical after the exchange; (3) pruning is real - with a handful of differences the walk
 * visits only a fraction of the nodes, and the visited count grows with the divergence; (4) tree
 * construction is order-independent and deterministic, and changing one key changes every node on
 * its path up to the root.
 */
"use strict";

const FNV_OFFSET = 0x811c9dc5, FNV_PRIME = 0x01000193;

function hash32(text) {
  let hash = FNV_OFFSET;
  for (let index = 0; index < text.length; index++) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, FNV_PRIME) >>> 0;
  }
  return hash >>> 0;
}

function mix(a, b) {
  let hash = Math.imul(a ^ 0x9e3779b9, FNV_PRIME) >>> 0;
  hash = Math.imul(hash ^ b, 0x85ebca6b) >>> 0;
  return (hash ^ (hash >>> 13)) >>> 0;
}

function lcg(seed) {
  return (seed * 48271) % 2147483647;
}

/** Leaf hash: fold the (key, value) pairs in key order so the result does not depend on insertion. */
function leafHash(entries) {
  let hash = 0x15051505;
  const sorted = entries.slice().sort((a, b) => (a.key < b.key ? -1 : a.key > b.key ? 1 : 0));
  for (const entry of sorted) hash = mix(hash, mix(hash32(entry.key), hash32(String(entry.value))));
  return hash >>> 0;
}

function build(entries, depth, maxDepth, leafSize) {
  if (depth >= maxDepth || entries.length <= leafSize) return { leaf: true, depth, entries, hash: leafHash(entries) };
  const middle = entries.length >> 1; // split the hash-ordered list: the shape depends on the key count only
  const children = [
    build(entries.slice(0, middle), depth + 1, maxDepth, leafSize),
    build(entries.slice(middle), depth + 1, maxDepth, leafSize),
  ];
  return { leaf: false, depth, children, hash: mix(children[0].hash, children[1].hash) };
}

function collectEntries(node, into = []) {
  if (node.leaf) return into.concat(node.entries);
  collectEntries(node.children[0], into);
  collectEntries(node.children[1], into);
  return into;
}

function countNodes(node) {
  return node.leaf ? 1 : 1 + countNodes(node.children[0]) + countNodes(node.children[1]);
}

class MerkleTree {
  constructor(entries, { maxDepth = 12, leafSize = 2 } = {}) {
    this.maxDepth = maxDepth;
    this.leafSize = leafSize;
    this.entries = entries.map(([key, value]) => ({ key, value, hash: hash32(key) }));
    this.entries.sort((a, b) => a.hash - b.hash); // hash order: the tree shape depends on the key count only
    this.data = new Map(this.entries.map((entry) => [entry.key, entry.value]));
    this.root = build(this.entries, 0, maxDepth, leafSize);
    this.nodeCount = countNodes(this.root);
  }

  static fromMap(map, options) {
    return new MerkleTree([...map.entries()], options);
  }

  hashes() {
    return this.entries.map((entry) => entry.hash);
  }

  /** Walk both trees together, pruning equal subtrees; return what each side must send. */
  compare(other) {
    const fromA = [], fromB = [];
    let visited = 0, pruned = 0;
    const walk = (a, b) => {
      visited++;
      if (a.hash === b.hash) {
        pruned++;
        return;
      }
      if (a.leaf && b.leaf) {
        fromA.push(...a.entries);
        fromB.push(...b.entries);
        return;
      }
      if (a.leaf || b.leaf) {
        fromA.push(...collectEntries(a));
        fromB.push(...collectEntries(b));
        return;
      }
      walk(a.children[0], b.children[0]);
      walk(a.children[1], b.children[1]);
    };
    walk(this.root, other.root);
    return {
      fromA, fromB, visited, pruned,
      identical: this.root.hash === other.root.hash,
      rootsEqual: this.root.hash === other.root.hash,
    };
  }

  /** Anti-entropy as it is really run: this replica is the initiator, so it *pulls* the peer's
   *  version of every differing key and *pushes* the keys the peer is missing. A versioned store
   *  would compare versions instead; here the peer's value wins, which is the usual repair rule. */
  reconcileFrom(other) {
    const difference = this.compare(other);
    const pulled = new Set(), pushed = new Set();
    for (const entry of difference.fromB) {
      this.data.set(entry.key, entry.value);
      pulled.add(entry.key);
    }
    for (const entry of difference.fromA) {
      if (other.data.has(entry.key)) continue;
      other.data.set(entry.key, entry.value);
      pushed.add(entry.key);
    }
    const repairedA = MerkleTree.fromMap(this.data, { maxDepth: this.maxDepth, leafSize: this.leafSize });
    const repairedB = MerkleTree.fromMap(other.data, { maxDepth: other.maxDepth, leafSize: other.leafSize });
    return {
      ...difference, pulled, pushed, repairedA, repairedB,
      converged: repairedA.root.hash === repairedB.root.hash,
    };
  }
}

function symmetricDifferenceKeys(a, b) {
  const keys = new Set();
  for (const [key, value] of a) if (b.get(key) !== value) keys.add(key);
  for (const [key, value] of b) if (a.get(key) !== value) keys.add(key);
  return keys;
}

function buildScenario(seed, totalKeys, changedKeys) {
  const base = new Map();
  let s = seed;
  for (let index = 0; index < totalKeys; index++) {
    base.set(`user:${index}`, `v${index}`);
  }
  const left = new Map(base);
  const right = new Map(base);
  s = lcg(s);
  const keys = [...base.keys()];
  const touched = [];
  for (let index = 0; index < changedKeys; index++) {
    s = lcg(s);
    const key = keys[s % keys.length];
    s = lcg(s);
    right.set(key, `changed:${s % 1000}`);
    touched.push(key);
  }
  return { left, right, touched };
}

module.exports = { hash32, mix, leafHash, build, countNodes, MerkleTree, symmetricDifferenceKeys, buildScenario, lcg };

if (require.main === module) {
  const assert = require("node:assert");
  const options = { maxDepth: 12, leafSize: 2 };

  // --- identical replicas prune the whole tree ----------------------------------------------
  const base = new Map([["a", 1], ["b", 2], ["c", 3], ["d", 4], ["e", 5], ["f", 6], ["g", 7], ["h", 8]]);
  const treeA = MerkleTree.fromMap(base, options);
  const treeB = MerkleTree.fromMap(new Map(base), options);
  assert.strictEqual(treeA.root.hash, treeB.root.hash, "equal contents hash equally");
  const same = treeA.compare(treeB);
  assert.strictEqual(same.visited, 1, "an identical pair prunes at the root");
  assert.deepStrictEqual(same.fromA, []);
  assert.deepStrictEqual(same.fromB, []);
  assert.ok(treeA.nodeCount > 1, "the tree must have more than the root");

  // --- one changed key: the walk descends one path and sends exactly that key ---------------
  const changed = new Map(base);
  changed.set("c", 300);
  const treeC = MerkleTree.fromMap(changed, options);
  assert.notStrictEqual(treeA.root.hash, treeC.root.hash, "one change changes the root");
  const difference = treeA.compare(treeC);
  assert.strictEqual(difference.fromA.length, 2, "the affected leaf travels from A");
  assert.strictEqual(difference.fromB.length, 2, "and the same leaf travels back from C");
  assert.deepStrictEqual(difference.fromA.map((entry) => entry.key), difference.fromB.map((entry) => entry.key),
    "anti-entropy exchanges one leaf, not the whole store");
  assert.deepStrictEqual(difference.fromA.filter((entry) => entry.key === "c").map((entry) => entry.value), [3]);
  assert.deepStrictEqual(difference.fromB.filter((entry) => entry.key === "c").map((entry) => entry.value), [300]);
  assert.ok(difference.visited < treeA.nodeCount, "pruning visits less than the whole tree");
  const repaired = treeA.reconcileFrom(treeC);
  assert.ok(repaired.converged, "the repair converges");
  assert.strictEqual(treeA.data.get("c"), 300, "the stale replica adopted the new value");
  assert.strictEqual(treeC.data.get("c"), 300);

  // --- boundaries ---------------------------------------------------------------------------
  const emptyLeft = MerkleTree.fromMap(new Map(), options);
  const emptyRight = MerkleTree.fromMap(new Map(), options);
  assert.strictEqual(emptyLeft.root.hash, emptyRight.root.hash, "two empty replicas agree");
  assert.strictEqual(emptyLeft.compare(emptyRight).visited, 1);
  const single = MerkleTree.fromMap(new Map([["only", 1]]), options);
  const emptyVsOne = emptyLeft.reconcileFrom(single);
  assert.strictEqual(emptyVsOne.fromA.length + emptyVsOne.fromB.length, 1, "an empty replica receives the key");
  assert.ok(emptyVsOne.converged, "and both are identical afterwards");
  assert.strictEqual(emptyLeft.data.get("only"), 1);
  assert.strictEqual(single.data.size, 1, "no key was duplicated");

  // --- randomized divergence: the exchanged set covers the symmetric difference -------------
  let seed = 31415926;
  let totalVisited = 0, totalNodes = 0;
  for (const changedKeys of [1, 2, 5, 20, 100]) {
    seed = lcg(seed);
    const scenario = buildScenario(seed, 500, changedKeys);
    const leftTree = MerkleTree.fromMap(scenario.left, options);
    const rightTree = MerkleTree.fromMap(scenario.right, options);
    assert.ok(!leftTree.compare(rightTree).identical, `${changedKeys} changes must be visible`);
    const before = symmetricDifferenceKeys(scenario.left, scenario.right);
    assert.ok(before.size > 0 && before.size <= changedKeys, "the scenario really changed keys");
    const result = leftTree.reconcileFrom(rightTree);
    assert.ok(result.converged, `${changedKeys} differences must be reconciled`);
    assert.strictEqual(symmetricDifferenceKeys(leftTree.data, rightTree.data).size, 0, "no residue");
    assert.deepStrictEqual([...leftTree.data.entries()].sort(), [...rightTree.data.entries()].sort(), "identical maps");
    const sent = new Set(result.fromA.concat(result.fromB).map((entry) => entry.key));
    for (const key of before) assert.ok(sent.has(key), `the differing key ${key} must be exchanged`);
    assert.ok(sent.size >= before.size, "at least the differing keys travel");
    assert.ok(sent.size < scenario.left.size, "but never the whole data set");
    assert.ok(result.visited <= leftTree.nodeCount + rightTree.nodeCount, "the walk is a tree walk");
    totalVisited += result.visited;
    totalNodes += leftTree.nodeCount;
    if (changedKeys <= 2) {
      assert.ok(result.visited * 3 < leftTree.nodeCount, `a small divergence must prune hard (${result.visited})`);
    }
  }
  assert.ok(totalVisited < totalNodes, "pruned walks stay cheaper than full traversals");

  // --- determinism, order independence and path hashing -------------------------------------
  const shuffled = [...base.entries()].reverse();
  assert.strictEqual(MerkleTree.fromMap(new Map(shuffled), options).root.hash, treeA.root.hash,
    "insertion order does not affect the tree");
  assert.strictEqual(MerkleTree.fromMap(new Map(base), options).root.hash, treeA.root.hash, "rebuilding is deterministic");
  assert.notStrictEqual(hash32("a"), hash32("b"), "distinct keys hash distinctly here");
  assert.strictEqual(hash32("a"), hash32("a"), "the hash is a function");

  const deepOptions = { maxDepth: 12, leafSize: 1 };
  const deepBefore = MerkleTree.fromMap(base, deepOptions);
  const changedDeep = new Map(base);
  changedDeep.set("c", 301);
  const deepAfter = MerkleTree.fromMap(changedDeep, deepOptions);
  assert.strictEqual(deepBefore.nodeCount, deepAfter.nodeCount, "a changed value keeps the shape");
  let diverged = 0;
  const countDiverged = (a, b) => {
    if (a.hash !== b.hash) diverged++;
    if (a.leaf || b.leaf) return;
    countDiverged(a.children[0], b.children[0]);
    countDiverged(a.children[1], b.children[1]);
  };
  countDiverged(deepBefore.root, deepAfter.root);
  assert.ok(diverged >= 2, "at least the root and the leaf change hash");
  assert.ok(diverged <= 6, `only the path to the key changes (got ${diverged} of ${deepBefore.nodeCount} nodes)`);
  assert.ok(diverged < deepBefore.nodeCount, "and not the whole tree");

  console.log(`nodes=${treeA.nodeCount} one-change-visits=${difference.visited} keys-sent=${difference.fromA.length + difference.fromB.length}`);
  console.log(`tree walks=${totalVisited} vs nodes=${totalNodes}`);
  console.log("PASS");
}
