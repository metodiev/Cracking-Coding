/** Task 806 - Implement fully persistent B-tree with copy-on-write pages and concurrency. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A page-oriented, fully persistent B-tree: the pager hands out page ids from a free list, every
 * mutation copies the pages on the root-to-leaf path and publishes a brand new root, so a root is a
 * complete immutable snapshot of the tree. Readers therefore need no locks at all - a reader keeps
 * the root it started with and keeps seeing a frozen, self-consistent tree while writers move on
 * (the same trick as a copy-on-write file system or a versioned B-tree index). Deletes follow the
 * classic CLRS rebalancing rules: borrow from a sibling or merge, all on freshly copied pages. A
 * write-ahead log records every page image, so the tree can be rebuilt from the log alone, and a
 * cleaner reclaims pages that no live root can reach (verified by brute-force reachability).
 *
 * What is simplified: pages live in an array instead of a file, "fsync" is appending a page image to
 * the log, node fan-out is a typed parameter (so the tree is 3-4 levels instead of 6-7 at full
 * scale), and concurrency is modelled by interleaved explicit reader/writer steps rather than
 * threads, which is exactly what a copy-on-write tree needs to be correct. The scale knobs are typed
 * parameters (order, key count, snapshot count, operations per phase).
 *
 * Invariants verified:
 *   - B-tree invariants after every mutation: keys sorted inside a node, every leaf at the same
 *     depth, node occupancy between t-1 and 2t-1 keys, and the key sets of all separators between
 *     the child subtrees;
 *   - oracle equivalence: point lookups, range scans and the sorted key list always agree with a
 *     plain sorted-array/Map oracle, over random insert and delete workloads including edge keys;
 *   - persistence: every snapshot root reproduces exactly the key/value set of the version it was
 *     taken at, re-read after hundreds of later mutations, and identical queries on the same root
 *     give identical answers;
 *   - copy-on-write: pages already published are never mutated (per-page checksums stay valid), the
 *     pages written per insert stay on the O(height) path-copying budget, page ids are only reused
 *     after a sweep proves the page unreachable from every live root;
 *   - durability: replaying the write-ahead log into a fresh pager rebuilds a tree with the same
 *     page images, the same root and the same query answers;
 *   - concurrency: a long-lived reader is unaffected by hundreds of interleaved writes, and its
 *     snapshot invariants hold while writers rebalance underneath it.
 */
"use strict";

const assert = require("node:assert");

/** Deterministic priority/mix used for the operation schedule and for page checksums. */
function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}

class TreeError extends Error {}

/** Append-only redo log: one record per page image (plus a checkpoint per published root). */
class WriteAheadLog {
  constructor() { this.records = []; }
  append(record) { this.records.push({ lsn: this.records.length + 1, ...record }); return this.records.length; }
  pageRecords() { return this.records.filter((record) => record.op === "page"); }
  latestRoot() {
    let rootId = null;
    for (const record of this.records) if (record.op === "checkpoint") rootId = record.rootId;
    return rootId;
  }
  checksum() { return this.records.map((record) => `${record.lsn}:${record.op}:${record.pageId ?? ""}:${record.image ?? ""}`).join("|"); }
}

class Page {
  constructor(id, leaf) {
    this.id = id;
    this.leaf = leaf;
    this.keys = [];
    this.values = [];        // leaf only
    this.children = [];      // internal only
    this.image = "";         // frozen image, set when the page is published
  }
}

const pageImage = (page) => (page.leaf ? `L[${page.keys.join(",")}]=[${page.values.join(",")}]` : `I[${page.keys.join(",")}](${page.children.join(",")})`);

class Pager {
  constructor(order, log = new WriteAheadLog()) {
    this.order = order;
    this.log = log;
    this.pages = new Map();
    this.freeList = [];
    this.nextId = 1;
    this.allocations = 0;
    this.reuses = 0;
    this.touched = new Set();
    this.checksums = new Map();          // pageId -> image hash the first time it was published
    this.violations = [];
  }

  allocate(leaf) {
    if (this.freeList.length > 0) { this.reuses++; return this.allocateSpecific(this.freeList.shift(), leaf); }
    return this.allocateSpecific(this.nextId++, leaf);
  }

  allocateSpecific(id, leaf) {
    if (this.pages.has(id)) this.violations.push(`page ${id} allocated while still reachable from a live root`);
    const page = new Page(id, leaf);
    this.pages.set(id, page);
    this.allocations++;
    this.touched.add(id);
    return page;
  }

  /** Pages written by one logical operation: only these are published (and logged). */
  beginWrite() { this.touched = new Set(); }

  /** Publish the written pages in id order and checkpoint the root that they belong to. */
  commitWrite(rootId) {
    for (const id of [...this.touched].sort((a, b) => a - b)) {
      const page = this.pages.get(id);
      if (page !== undefined) this.publish(page);
    }
    this.touched = new Set();
    if (rootId !== undefined && rootId !== null) this.log.append({ op: "checkpoint", rootId });
  }

  /** A copy-on-write page: a fresh id carrying the old contents. */
  copyOf(page) {
    const fresh = this.allocate(page.leaf);
    fresh.keys = [...page.keys];
    fresh.values = [...page.values];
    fresh.children = [...page.children];
    return fresh;
  }

  /** Publishing freezes the image, logs the full page contents, and detects in-place mutation. */
  publish(page) {
    const image = pageImage(page);
    const previous = this.checksums.get(page.id);
    if (previous !== undefined && previous !== image) {
      this.violations.push(`page ${page.id} was mutated after being published (${previous} -> ${image})`);
    }
    this.checksums.set(page.id, image);
    page.image = image;
    this.log.append({
      op: "page", pageId: page.id, leaf: page.leaf, image,
      keys: [...page.keys], values: [...page.values], children: [...page.children],
    });
    return image;
  }

  free(id) {
    this.pages.delete(id);
    this.checksums.delete(id);
    this.freeList.push(id);
  }

  imageOf(id) { const page = this.pages.get(id); return page === undefined ? null : pageImage(page); }

  livePageCount() { return this.pages.size; }
}

// --------------------------------------------------------------------------- persistent B-tree

class PersistentBTree {
  constructor(order, pager = new Pager(order)) {
    this.order = order;                       // t = minimum degree: nodes hold t-1 .. 2t-1 keys
    this.pager = pager;
    this.rootId = null;
    this.versions = [];                       // roots, oldest first
  }

  /** A version handle is just a root id - the whole point of persistence. */
  snapshot() {
    const handle = { rootId: this.rootId, size: this.size ?? 0, at: this.versions.length };
    this.versions.push(handle);
    return handle;
  }

  node(id) {
    if (id === null) return null;
    const page = this.pager.pages.get(id);
    if (page === undefined) throw new TreeError(`page ${id} is not allocated`);
    return page;
  }

  static create(order, pager) {
    const tree = new PersistentBTree(order, pager);
    return tree;
  }

  get t() { return this.order; }

  get size() { return this.inOrder(this.rootId).length; }

  /** In-order walk over every node: in this variant separators carry values too. */
  inOrder(rootId = this.rootId) {
    const out = [];
    const walk = (id) => {
      if (id === null || id === undefined) return;
      const node = this.node(id);
      if (node.leaf) {
        for (let i = 0; i < node.keys.length; i++) out.push([node.keys[i], node.values[i]]);
        return;
      }
      for (let i = 0; i < node.keys.length; i++) {
        walk(node.children[i]);
        out.push([node.keys[i], node.values[i]]);
      }
      walk(node.children[node.keys.length]);
    };
    walk(rootId);
    return out;
  }

  /** Public mutators always copy the path, then publish (and log) exactly the pages they wrote. */
  set(key, value) {
    this.pager.beginWrite();
    if (this.rootId === null) {
      const root = this.pager.allocate(true);
      root.keys = [key];
      root.values = [value];
      this.rootId = root.id;
      this.pager.commitWrite(root.id);
      return this.rootId;
    }
    let root = this.node(this.rootId);
    if (root.keys.length === 2 * this.order - 1) {
      const fresh = this.pager.allocate(false);
      fresh.children = [root.id];
      this.splitChild(fresh, 0, root);
      root = fresh;
    } else {
      root = this.pager.copyOf(root);
    }
    this.insertNonFull(root, key, value);
    this.rootId = root.id;
    this.pager.commitWrite(root.id);
    return root.id;
  }

  /** CLRS split: the child at `index` is already copied by the caller, both halves are new pages. */
  splitChild(parent, index, child) {
    const t = this.order;
    const median = child.keys[t - 1];
    const medianValue = child.values[t - 1];

    const right = this.pager.allocate(child.leaf);
    right.keys = child.keys.slice(t);
    right.values = child.values.slice(t);
    if (!child.leaf) right.children = child.children.slice(t);

    const left = this.pager.allocate(child.leaf);
    left.keys = child.keys.slice(0, t - 1);
    left.values = child.values.slice(0, t - 1);
    if (!child.leaf) left.children = child.children.slice(0, t);

    parent.keys.splice(index, 0, median);
    if (parent.leaf) throw new TreeError("an internal node must not be a leaf");
    parent.values.splice(index, 0, medianValue);
    parent.children.splice(index, 1, left.id, right.id);

    return { left, right };
  }

  insertNonFull(node, key, value) {
    const t = this.order;
    let index = node.keys.findIndex((existing) => existing >= key);
    if (index === -1) index = node.keys.length;
    if (index < node.keys.length && node.keys[index] === key) { node.values[index] = value; return false; }   // overwrite
    if (node.leaf) {
      node.keys.splice(index, 0, key);
      node.values.splice(index, 0, value);
      return true;
    }
    let child = this.pager.copyOf(this.node(node.children[index]));
    node.children[index] = child.id;
    if (child.keys.length === 2 * t - 1) {
      this.splitChild(node, index, child);
      if (key === node.keys[index]) { node.values[index] = value; return false; }     // the median moved up
      index = key > node.keys[index] ? index + 1 : index;
      child = this.node(node.children[index]);
    }
    return this.insertNonFull(child, key, value);
  }

  /** Lookup against a specific root - the reader's snapshot, never the live root. */
  get(key, rootId = this.rootId) {
    let id = rootId;
    while (id !== null && id !== undefined) {
      const node = this.node(id);
      let index = 0;
      while (index < node.keys.length && key > node.keys[index]) index++;
      if (index < node.keys.length && node.keys[index] === key) return { found: true, value: node.values[index] };
      if (node.leaf) return { found: false, value: undefined };
      id = node.children[index];
    }
    return { found: false, value: undefined };
  }

  /** Range scan against any root: in-order walk that prunes subtrees outside [lo, hi]. */
  range(lo, hi, rootId = this.rootId) {
    const out = [];
    const walk = (id, lowBound, highBound) => {
      if (id === null || id === undefined) return;
      const node = this.node(id);
      if (node.leaf) {
        for (let i = 0; i < node.keys.length; i++) if (node.keys[i] >= lo && node.keys[i] <= hi) out.push([node.keys[i], node.values[i]]);
        return;
      }
      const count = node.keys.length;
      for (let i = 0; i <= count; i++) {
        const low = i === 0 ? lowBound : node.keys[i - 1];
        const high = i === count ? highBound : node.keys[i];
        const outside = (low !== null && low >= hi) || (high !== null && high <= lo);
        if (!outside) walk(node.children[i], low, high);
        // The separator live in this node, so it is emitted even when its neighbours are pruned.
        if (i < count && node.keys[i] >= lo && node.keys[i] <= hi) out.push([node.keys[i], node.values[i]]);
      }
    };
    walk(rootId, null, null);
    return out;
  }

  /** delete is the classic three-case recursion; every visited page is copied on the way down. */
  remove(key, rootId = this.rootId) {
    if (rootId === null) return { rootId, removed: false };
    this.pager.beginWrite();
    const present = this.get(key, rootId).found;
    const root = this.pager.copyOf(this.node(rootId));
    if (present) this.removeFrom(root, key);
    let newRoot = root;
    if (root.keys.length === 0 && !root.leaf && root.children.length > 0) newRoot = this.node(root.children[0]);
    this.rootId = newRoot.id;
    this.pager.commitWrite(newRoot.id);
    return { rootId: newRoot.id, removed: present };
  }

  contains(key, rootId) { return this.get(key, rootId).found; }

  /** Returns true when the key was present in this subtree (CLRS delete with borrow/merge). */
  removeFrom(node, key) {
    const t = this.order;
    let index = 0;
    while (index < node.keys.length && key > node.keys[index]) index++;
    const foundHere = index < node.keys.length && node.keys[index] === key;

    if (node.leaf) {
      if (foundHere) { node.keys.splice(index, 1); node.values.splice(index, 1); }
      return foundHere;
    }
    if (foundHere) {
      const leftId = node.children[index];
      const rightId = node.children[index + 1];
      const left = this.node(leftId);
      const right = this.node(rightId);
      if (left.keys.length >= t) {
        const predecessor = this.maxKey(leftId);
        node.keys[index] = predecessor.key;
        node.values[index] = predecessor.value;
        const leftCopy = this.pager.copyOf(left);
        node.children[index] = leftCopy.id;
        this.removeFrom(leftCopy, predecessor.key);
        return true;
      }
      if (right.keys.length >= t) {
        const successor = this.minKey(rightId);
        node.keys[index] = successor.key;
        node.values[index] = successor.value;
        const rightCopy = this.pager.copyOf(right);
        node.children[index + 1] = rightCopy.id;
        this.removeFrom(rightCopy, successor.key);
        return true;
      }
      const separatorValue = node.values[index];
      const merged = this.pager.copyOf(left);
      merged.keys = [...left.keys, key, ...right.keys];
      merged.values = [...left.values, separatorValue, ...right.values];
      if (!left.leaf) merged.children = [...left.children, ...right.children];
      node.keys.splice(index, 1);
      node.values.splice(index, 1);
      node.children.splice(index, 2, merged.id);
      this.discard(rightId);
      return this.removeFrom(merged, key);
    }
    return this.removeFromChild(node, index, key);
  }

  removeFromChild(node, index, key) {
    const t = this.order;
    let child = this.pager.copyOf(this.node(node.children[index]));
    node.children[index] = child.id;
    if (child.keys.length < t) {
      const hasLeft = index > 0;
      const hasRight = index < node.children.length - 1;
      const leftSibling = hasLeft ? this.pager.copyOf(this.pager.pages.get(node.children[index - 1])) : null;
      const rightSibling = hasRight ? this.pager.copyOf(this.pager.pages.get(node.children[index + 1])) : null;
      if (leftSibling !== null && leftSibling.keys.length >= t) {
        node.children[index - 1] = leftSibling.id;
        child.keys.unshift(node.keys[index - 1]);
        node.keys[index - 1] = leftSibling.keys.pop();
        child.values.unshift(node.values[index - 1]);
        node.values[index - 1] = leftSibling.values.pop();
        if (!child.leaf) child.children.unshift(leftSibling.children.pop());
      } else if (rightSibling !== null && rightSibling.keys.length >= t) {
        node.children[index + 1] = rightSibling.id;
        child.keys.push(node.keys[index]);
        node.keys[index] = rightSibling.keys.shift();
        child.values.push(node.values[index]);
        node.values[index] = rightSibling.values.shift();
        if (!child.leaf) child.children.push(rightSibling.children.shift());
      } else if (leftSibling !== null) {
        node.children[index - 1] = leftSibling.id;
        const separator = node.keys[index - 1];
        const separatorValue = node.values[index - 1];
        leftSibling.keys.push(separator);
        leftSibling.values.push(separatorValue);
        leftSibling.keys.push(...child.keys);
        leftSibling.values.push(...child.values);
        if (!leftSibling.leaf) leftSibling.children.push(...child.children);
        node.keys.splice(index - 1, 1);
        node.values.splice(index - 1, 1);
        node.children.splice(index, 1);
        this.discard(child.id);
        child = leftSibling;
        index -= 1;
      } else if (rightSibling !== null) {
        node.children[index + 1] = rightSibling.id;
        child.keys.push(node.keys[index]);
        child.values.push(node.values[index]);
        child.keys.push(...rightSibling.keys);
        child.values.push(...rightSibling.values);
        if (!child.leaf) child.children.push(...rightSibling.children);
        node.keys.splice(index, 1);
        node.values.splice(index, 1);
        node.children.splice(index + 1, 1);
        this.discard(rightSibling.id);
      }
    }
    return this.removeFrom(child, key);
  }

  /**
   * A page dropped by a merge stays allocated: an older snapshot may still reach it, so only a
   * sweep that knows every live root may return it to the free list.
   */
  discard(id) { void id; }

  minKey(rootId) {
    let node = this.node(rootId);
    while (!node.leaf) node = this.node(node.children[0]);
    return { key: node.keys[0], value: node.values[0] };
  }

  maxKey(rootId) {
    let node = this.node(rootId);
    while (!node.leaf) node = this.node(node.children[node.children.length - 1]);
    return { key: node.keys[node.keys.length - 1], value: node.values[node.values.length - 1] };
  }

  keyList(rootId = this.rootId) { return this.inOrder(rootId).map(([key]) => key); }

  /** Full structural check: order, occupancy, uniform leaf depth and subtree key ranges. */
  validate(rootId = this.rootId) {
    if (rootId === null) return true;
    const t = this.order;
    const depths = new Set();
    const check = (id, depth, low, high) => {
      const node = this.node(id);
      for (let i = 0; i < node.keys.length; i++) {
        if (i > 0 && !(node.keys[i - 1] < node.keys[i])) throw new TreeError(`keys unsorted in page ${id}`);
        if (low !== null && !(node.keys[i] > low)) throw new TreeError(`key ${node.keys[i]} below the separator in page ${id}`);
        if (high !== null && !(node.keys[i] < high)) throw new TreeError(`key ${node.keys[i]} above the separator in page ${id}`);
      }
      if (node.leaf) {
        if (node.keys.length !== node.values.length) throw new TreeError(`page ${id} has mismatched keys/values`);
        if (depth > 0 && (node.keys.length < t - 1 || node.keys.length > 2 * t - 1)) throw new TreeError(`page ${id} occupancy ${node.keys.length} out of bounds`);
        if (id === rootId && node.keys.length < 1 && this.size > 0) throw new TreeError("an empty non-empty root");
        depths.add(depth);
        return;
      }
      if (node.keys.length + 1 !== node.children.length) throw new TreeError(`page ${id} has an inconsistent child count`);
      if (depth > 0 && (node.keys.length < t - 1 || node.keys.length > 2 * t - 1)) throw new TreeError(`page ${id} occupancy ${node.keys.length} out of bounds`);
      for (let i = 0; i < node.children.length; i++) {
        check(node.children[i], depth + 1, i === 0 ? low : node.keys[i - 1], i === node.keys.length ? high : node.keys[i]);
      }
    };
    check(rootId, 0, null, null);
    if (depths.size !== 1) throw new TreeError(`leaves at ${[...depths].join("/")} depths`);
    return true;
  }

  /** Publish every page currently reachable, freezing the images of the live version. */
  publishAll(rootId = this.rootId) {
    const seen = new Set();
    const walk = (id) => {
      if (id === null || id === undefined || seen.has(id)) return;
      seen.add(id);
      const node = this.node(id);
      if (!node.leaf) for (const child of node.children) walk(child);
      this.pager.publish(node);
    };
    walk(rootId);
    return seen;
  }

  reachable(rootIds) {
    const seen = new Set();
    const walk = (id) => {
      if (id === null || id === undefined || seen.has(id)) return;
      seen.add(id);
      const node = this.pager.pages.get(id);
      if (node === undefined || node.leaf) return;
      for (const child of node.children) walk(child);
    };
    for (const rootId of rootIds) walk(rootId);
    return seen;
  }

  /** Sweep: every page no live root can reach is returned to the free list (and unmapped). */
  sweep(liveRoots) {
    const reachable = this.reachable(liveRoots);
    const reclaimed = [];
    for (const id of [...this.pager.pages.keys()]) {
      if (reachable.has(id)) continue;
      reclaimed.push(id);
      this.pager.free(id);
    }
    return reclaimed;
  }

  /** Recovery: redo every page image in LSN order, then take the last checkpoint as the root. */
  static recover(order, log) {
    const rebuilt = new Pager(order);
    let rootId = null;
    for (const record of log.records) {
      if (record.op === "checkpoint") { rootId = record.rootId; continue; }
      const page = rebuilt.allocateSpecific(record.pageId, record.leaf);
      page.keys = [...record.keys];
      page.values = [...record.values];
      page.children = [...record.children];
      rebuilt.publish(page);
    }
    const tree = new PersistentBTree(order, rebuilt);
    tree.rootId = rootId;
    return tree;
  }
}

// --------------------------------------------------------------------------- oracle

/** Plain sorted map oracle: no pages, no structure, just the semantics. */
class Oracle {
  constructor() { this.map = new Map(); }

  set(key, value) { const existed = this.map.has(key); this.map.set(key, value); return !existed; }

  remove(key) { return this.map.delete(key); }

  get(key) { return this.map.has(key) ? { found: true, value: this.map.get(key) } : { found: false, value: undefined }; }

  range(lo, hi) { return [...this.map.entries()].filter(([key]) => key >= lo && key <= hi).sort((a, b) => a[0] - b[0]); }

  keys() { return [...this.map.keys()].sort((a, b) => a - b); }

  clone() { const copy = new Oracle(); copy.map = new Map(this.map); return copy; }
}

if (require.main === module) {
  const rng = splitmix64(0x806);
  const order = 3;                                   // t = 3: 2..5 keys per node

  // --- 1. sequential inserts match the oracle, with invariants after every single insert --------
  const tree = new PersistentBTree(order);
  const oracle = new Oracle();
  for (let key = 0; key < 120; key++) {
    tree.set(key, key * 7);
    oracle.set(key, key * 7);
    tree.validate();
    assert.strictEqual(tree.size, oracle.keys().length, `size after inserting ${key}`);
  }
  assert.deepStrictEqual(tree.keyList(), oracle.keys(), "the in-order walk equals the oracle key list");
  for (let key = 0; key < 120; key += 7) assert.deepStrictEqual(tree.get(key), oracle.get(key), `lookup of ${key}`);
  assert.deepStrictEqual(tree.get(1000), { found: false, value: undefined }, "a missing key is reported as missing");
  assert.deepStrictEqual(tree.range(30, 39), oracle.range(30, 39), "range scan equals the oracle");
  assert.deepStrictEqual(tree.range(500, 600), [], "a range above every key is empty");
  const height = (() => { let depth = 1; let id = tree.rootId; while (!tree.node(id).leaf) { id = tree.node(id).children[0]; depth++; } return depth; })();
  assert.ok(height >= 4, `120 keys at t=3 build a real multi-level tree (height ${height})`);

  // --- 2. random insert/delete workload against the oracle ---------------------------------------
  const randomTree = new PersistentBTree(order);
  const randomOracle = new Oracle();
  for (let step = 0; step < 1500; step++) {
    const key = rng.int(300);
    if (rng.int(4) === 0) {
      randomTree.remove(key);
      randomOracle.remove(key);
    } else {
      const value = rng.int(1000);
      randomTree.set(key, value);
      randomOracle.set(key, value);
    }
    if (step % 25 === 0) {
      randomTree.validate();
      assert.deepStrictEqual(randomTree.keyList(), randomOracle.keys(), `key set after step ${step}`);
    }
  }
  randomTree.validate();
  assert.deepStrictEqual(randomTree.keyList(), randomOracle.keys(), "the key set survived 1500 mixed operations");
  for (let key = 0; key < 300; key++) assert.deepStrictEqual(randomTree.get(key), randomOracle.get(key), `value of ${key} after the random workload`);
  for (let i = 0; i < 60; i++) {
    const lo = rng.int(300);
    const hi = lo + rng.int(40);
    assert.deepStrictEqual(randomTree.range(lo, hi), randomOracle.range(lo, hi), `range ${lo}..${hi} after the random workload`);
  }
  assert.ok(randomTree.size > 100, `the tree still holds ${randomTree.size} keys`);

  // --- 3. persistence: snapshots keep answering after hundreds of later mutations ----------------
  const versioned = new PersistentBTree(order);
  const history = new Oracle();
  const snapshots = [];
  for (let step = 0; step < 400; step++) {
    if (step % 20 === 0) snapshots.push({ handle: versioned.snapshot(), state: history.clone() });
    const key = rng.int(150);
    if (rng.int(5) === 0) { versioned.remove(key); history.remove(key); }
    else { const value = rng.int(500); versioned.set(key, value); history.set(key, value); }
  }
  snapshots.push({ handle: versioned.snapshot(), state: history.clone() });
  assert.ok(snapshots.length >= 20, `${snapshots.length} snapshots were taken`);
  for (const [index, snapshot] of snapshots.entries()) {
    const keys = versioned.range(-1, 10 ** 9, snapshot.handle.rootId).map(([key]) => key);
    assert.deepStrictEqual(keys, snapshot.state.keys(), `snapshot ${index} still reproduces its own key set`);
    for (let probe = 0; probe < 12; probe++) {
      const key = rng.int(150);
      const expected = snapshot.state.get(key);
      assert.strictEqual(versioned.get(key, snapshot.handle.rootId).value, expected.value, `snapshot ${index} value of ${key}`);
      assert.strictEqual(versioned.get(key, snapshot.handle.rootId).found, expected.found, `snapshot ${index} presence of ${key}`);
    }
    assert.strictEqual(snapshot.handle.size, snapshot.state.keys().length, `snapshot ${index} recorded size`);
    versioned.validate(snapshot.handle.rootId);
  }
  const oldest = snapshots[0];
  const newest = snapshots[snapshots.length - 1];
  assert.notStrictEqual(oldest.handle.rootId, newest.handle.rootId, "each version has its own root");
  const oldestKeys = versioned.range(-1, 10 ** 9, oldest.handle.rootId).map(([key]) => `${key}:${versioned.get(key, oldest.handle.rootId).value}`);
  versioned.set(9999, "later");
  versioned.remove(0);
  const oldestAgain = versioned.range(-1, 10 ** 9, oldest.handle.rootId).map(([key]) => `${key}:${versioned.get(key, oldest.handle.rootId).value}`);
  assert.deepStrictEqual(oldestAgain, oldestKeys, "further writes never change an old snapshot");

  // --- 4. copy-on-write accounting and page immutability -----------------------------------------
  const cow = new PersistentBTree(order);
  for (let key = 0; key < 200; key++) cow.set(key, key);
  cow.publishAll();
  assert.deepStrictEqual(cow.pager.violations, [], "no published page was ever mutated");
  cow.pager.allocations = 0;
  let cowHeight = 1;
  for (let id = cow.rootId; !cow.node(id).leaf;) { id = cow.node(id).children[0]; cowHeight++; }
  for (let i = 0; i < 50; i++) cow.set(1000 + i, i);
  const allocationsForFiftyWrites = cow.pager.allocations;
  const pathBudget = 50 * (cowHeight + 2);
  assert.ok(allocationsForFiftyWrites <= pathBudget, `path copying wrote ${allocationsForFiftyWrites} pages, budget ${pathBudget}`);
  const inPlaceCost = 50 * (cow.pager.livePageCount());
  assert.ok(allocationsForFiftyWrites < inPlaceCost / 4, `copy-on-write is far cheaper than rewriting every live page (${allocationsForFiftyWrites} vs ${inPlaceCost})`);
  cow.publishAll();
  assert.deepStrictEqual(cow.pager.violations, [], "the writes only created new pages");
  const checksumBefore = new Map(cow.pager.checksums);
  const snapshotBefore = cow.snapshot();
  const snapshotKeys = cow.keyList(snapshotBefore.rootId);
  for (let i = 0; i < 80; i++) cow.set(rng.int(400), i);
  cow.publishAll();
  for (const [id, image] of checksumBefore) {
    if (!cow.pager.pages.has(id)) continue;
    assert.strictEqual(cow.pager.imageOf(id), image, `page ${id} kept its published image`);
  }
  assert.deepStrictEqual(cow.keyList(snapshotBefore.rootId), snapshotKeys, "the snapshot root is frozen while the live tree moves on");

  // --- 5. garbage collection: only unreachable pages are reclaimed --------------------------------
  const gc = new PersistentBTree(order);
  for (let key = 0; key < 150; key++) gc.set(key, key);
  gc.publishAll();
  const liveSnapshot = gc.snapshot();
  for (let key = 150; key < 300; key++) gc.set(key, key);
  gc.publishAll();
  const pagesBefore = gc.pager.livePageCount();
  const reclaimedA = gc.sweep([gc.rootId, liveSnapshot.rootId]);
  assert.ok(reclaimedA.length > 0, `the sweep reclaimed ${reclaimedA.length} unreachable pages`);
  assert.strictEqual(gc.pager.livePageCount(), pagesBefore - reclaimedA.length, "the page count dropped by exactly the reclaimed pages");
  gc.validate();
  gc.validate(liveSnapshot.rootId);
  assert.deepStrictEqual(gc.keyList(liveSnapshot.rootId), Array.from({ length: 150 }, (_, index) => index), "the live snapshot survived the sweep intact");
  const reachable = gc.reachable([gc.rootId, liveSnapshot.rootId]);
  for (const id of reclaimedA) assert.strictEqual(reachable.has(id), false, `reclaimed page ${id} is reachable from no live root`);
  const keptPages = [...gc.pager.pages.keys()];
  const reclaimedB = gc.sweep([gc.rootId]);
  assert.ok(reclaimedB.length > 0, "dropping the old snapshot frees more pages");
  for (const id of keptPages) if (gc.pager.pages.has(id)) assert.strictEqual(gc.reachable([gc.rootId]).has(id), true, `kept page ${id} is still reachable`);
  gc.validate();
  assert.deepStrictEqual(gc.keyList(), Array.from({ length: 300 }, (_, index) => index), "and the live tree is still complete");

  // --- 6. durability: rebuild the tree from the write-ahead log ----------------------------------
  const durable = new PersistentBTree(order);
  for (let key = 0; key < 90; key++) durable.set(key, `v${key}`);
  durable.remove(3);
  durable.remove(40);
  durable.set(90, "tail");
  durable.publishAll();
  const logChecksum = durable.pager.log.checksum();
  const pageRecords = durable.pager.log.pageRecords().length;
  const recovered = PersistentBTree.recover(order, durable.pager.log);
  assert.strictEqual(recovered.rootId, durable.rootId, "recovery finds the checkpointed root");
  assert.strictEqual(recovered.rootId, durable.pager.log.latestRoot(), "and it is the last checkpoint in the log");
  assert.deepStrictEqual(recovered.keyList(), durable.keyList(), "the recovered tree has the same keys");
  recovered.validate();
  for (let key = 0; key < 91; key++) assert.deepStrictEqual(recovered.get(key), durable.get(key), `recovered value of ${key}`);
  assert.deepStrictEqual(recovered.get(3), { found: false, value: undefined }, "the deleted key stayed deleted after recovery");
  assert.deepStrictEqual(recovered.range(80, 90), durable.range(80, 90), "ranges survive recovery");
  const livePages = durable.reachable([durable.rootId]);
  for (const id of livePages) {
    assert.strictEqual(recovered.pager.imageOf(id), durable.pager.imageOf(id), `recovered page ${id} matches (undo/redo image)`);
  }
  assert.strictEqual(durable.pager.log.checksum(), logChecksum, "the log is append-only and unchanged");
  assert.ok(pageRecords >= livePages.size, `the log holds at least one page image per live page (${pageRecords} >= ${livePages.size})`);
  assert.ok(durable.pager.log.records.length >= pageRecords, "checkpoints are interleaved with page records");

  // --- 7. concurrency: a reader keeps its snapshot while writers rebalance -----------------------
  const concurrent = new PersistentBTree(order);
  for (let key = 0; key < 260; key++) concurrent.set(key, key * 2);
  const reader = concurrent.snapshot();
  const readerKeys = concurrent.keyList(reader.rootId);
  const readerValues = new Map(concurrent.range(-1, 10 ** 9, reader.rootId));
  let reads = 0;
  for (let step = 0; step < 400; step++) {
    const key = rng.int(500);
    concurrent.remove(key);
    concurrent.set(500 + rng.int(200), rng.int(1000));
    if (step % 40 === 0) {
      concurrent.validate();
      assert.deepStrictEqual(concurrent.keyList(reader.rootId), readerKeys, `reader key list stable at step ${step}`);
      for (let probe = 0; probe < 5; probe++) {
        const probeKey = rng.int(260);
        assert.strictEqual(concurrent.get(probeKey, reader.rootId).value, readerValues.get(probeKey), `reader value of ${probeKey} stable at step ${step}`);
        reads++;
      }
      concurrent.validate(reader.rootId);
    }
  }
  assert.strictEqual(concurrent.pager.violations.length, 0, concurrent.pager.violations.join("; "));
  assert.deepStrictEqual(concurrent.keyList(reader.rootId), readerKeys, "400 writes later the reader still sees its tree");
  const writerSize = concurrent.size;
  assert.strictEqual(concurrent.keyList(reader.rootId).length, readerKeys.length, "the reader's tree did not grow");
  assert.ok(writerSize !== readerKeys.length, `the writer's tree moved on (${writerSize} vs ${readerKeys.length} keys)`);

  // --- 8. edges ------------------------------------------------------------------------------------
  const single = new PersistentBTree(order);
  single.set(5, "five");
  single.validate();
  assert.deepStrictEqual(single.keyList(), [5], "a single key tree is valid");
  assert.deepStrictEqual(single.get(5), { found: true, value: "five" });
  const overwritten = single.set(5, "FIVE");
  assert.deepStrictEqual(single.get(5), { found: true, value: "FIVE" }, "re-setting a key overwrites the value");
  assert.strictEqual(overwritten, single.rootId);
  assert.strictEqual(single.size, 1, "overwriting does not duplicate the key");
  assert.throws(() => single.node(99999), /not allocated/, "an unknown page id is rejected");
  const drain = new PersistentBTree(order, new Pager(order));
  for (let key = 0; key < 40; key++) drain.set(key, key);
  for (let key = 0; key < 40; key++) drain.remove(key);
  drain.validate();
  assert.deepStrictEqual(drain.keyList(), [], "removing every key empties the tree");
  assert.strictEqual(drain.size, 0, "and the size is zero");
  assert.deepStrictEqual(drain.get(0), { found: false, value: undefined }, "lookups in an emptied tree miss");
  drain.set(1, "back");
  drain.validate();
  assert.deepStrictEqual(drain.keyList(), [1], "the emptied tree accepts new keys");
  const decrement = splitmix64(0x807);
  const shrink = new PersistentBTree(2);              // t=2: the tightest legal occupancy
  const shrinkOracle = new Oracle();
  for (let step = 0; step < 400; step++) {
    const key = decrement.int(60);
    if (decrement.int(3) === 0) { shrink.remove(key); shrinkOracle.remove(key); }
    else { const value = decrement.int(50); shrink.set(key, value); shrinkOracle.set(key, value); }
    if (step % 20 === 0) { shrink.validate(); assert.deepStrictEqual(shrink.keyList(), shrinkOracle.keys(), `t=2 key set after step ${step}`); }
  }
  shrink.validate();
  assert.deepStrictEqual(shrink.keyList(), shrinkOracle.keys(), "t=2 trees stay valid under mixed operations");

  console.log(`btree: order ${order}, height ${height} for 120 keys, ${randomTree.pager.allocations} allocations over 1500 random ops`);
  console.log(`persistence: ${snapshots.length} snapshots all still readable, ${allocationsForFiftyWrites} pages written for 50 inserts (budget ${pathBudget}, in-place would be ${inPlaceCost})`);
  console.log(`gc: ${reclaimedA.length + reclaimedB.length} pages reclaimed, ${gc.pager.livePageCount()} live pages; cow violations ${concurrent.pager.violations.length}`);
  console.log(`reader: ${reads} probe reads served from a frozen snapshot across 400 concurrent writes`);
  console.log("PASS");
}

module.exports = { PersistentBTree, Pager, Page, Oracle, WriteAheadLog, TreeError, pageImage, splitmix64 };
