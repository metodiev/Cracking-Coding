/** Task 475 - Implement authenticated data structure example (Merkle+proof). (Tier: Advanced, CTCI Beyond — Security & Privacy) */
"use strict";

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

class AuthenticatedLog {
  constructor(records) {
    this.records = records.slice();
    this.levels = [this.records.map((v) => hashLeaf(v))];
    let level = this.levels[0];
    while (level.length > 1) {
      const next = [];
      for (let i = 0; i + 1 < level.length; i += 2) next.push(hashNode(level[i], level[i + 1]));
      if (level.length % 2 === 1) next.push(level[level.length - 1]);
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
      if (idx % 2 === 1) path.push([level[idx - 1], 0]);
      else if (idx + 1 < level.length) path.push([level[idx + 1], 1]);
      idx = Math.floor(idx / 2);
    }
    return path;
  }

  multiProof(targets) {
    let level = this.levels[0];
    let needed = new Set(targets);
    const out = [];
    while (level.length > 1) {
      const siblings = [];
      for (const i of [...needed].sort((a, b) => a - b)) {
        if (i % 2 === 1 && !needed.has(i - 1)) siblings.push([i - 1, level[i - 1]]);
        else if (i % 2 === 0 && i + 1 < level.length && !needed.has(i + 1)) siblings.push([i + 1, level[i + 1]]);
      }
      out.push(siblings);
      const next = [];
      for (let i = 0; i + 1 < level.length; i += 2) next.push(hashNode(level[i], level[i + 1]));
      if (level.length % 2 === 1) next.push(level[level.length - 1]);
      level = next;
      needed = new Set([...needed].map((i) => Math.floor(i / 2)));
    }
    return out;
  }
}

function verify(root, value, path) {
  let h = hashLeaf(value);
  for (const [sibling, side] of path) {
    h = side === 0 ? hashNode(sibling, h) : hashNode(h, sibling);
  }
  return h === root;
}

function verifyMulti(root, n, targets, proof) {
  let known = new Map();
  for (const [i, v] of targets) known.set(i, hashLeaf(v));
  let levelLen = n;
  for (const siblings of proof) {
    const sibMap = new Map(siblings.map(([pos, h]) => [pos, h]));
    const parents = new Map();
    for (let i = 0; i + 1 < levelLen; i += 2) {
      const haveLeft = known.has(i);
      const haveRight = known.has(i + 1);
      if (!haveLeft && !haveRight) continue;
      if (haveLeft && haveRight) {
        parents.set(i / 2, hashNode(known.get(i), known.get(i + 1)));
      } else {
        const pos = haveLeft ? i + 1 : i;
        if (!sibMap.has(pos)) return false;
        const left = haveLeft ? known.get(i) : sibMap.get(i);
        const right = haveRight ? known.get(i + 1) : sibMap.get(i + 1);
        parents.set(i / 2, hashNode(left, right));
      }
    }
    if (levelLen % 2 === 1 && known.has(levelLen - 1)) {
      parents.set((levelLen - 1) / 2, known.get(levelLen - 1));
    }
    known = parents;
    levelLen = (levelLen + 1) / 2;
  }
  return known.has(0) && known.get(0) === root;
}

function updatedRoot(newValue, path) {
  let h = hashLeaf(newValue);
  for (const [sibling, side] of path) {
    h = side === 0 ? hashNode(sibling, h) : hashNode(h, sibling);
  }
  return h;
}

module.exports = { AuthenticatedLog, verify, verifyMulti, updatedRoot, hashLeaf, hashNode };

if (require.main === module) {
  const assert = require("node:assert");
  const leaves8 = [];
  for (let i = 0; i < 8; i++) leaves8.push(`tx${i}`);
  const log = new AuthenticatedLog(leaves8);
  assert.strictEqual(log.root(), 955777637);

  const path2 = log.proof(2);
  assert.deepStrictEqual(path2.map(([h]) => h), [1947293384, 240787827, 3271786783]);
  assert.deepStrictEqual(path2.map(([, s]) => s), [1, 0, 1]);

  const mp01 = log.multiProof([0, 1]);
  assert.deepStrictEqual(mp01.map((s) => s.length), [0, 1, 1]);
  assert.deepStrictEqual(mp01.flat().map(([, h]) => h), [1737906683, 3271786783]);
  assert.ok(verifyMulti(log.root(), 8, new Map([[0, "tx0"], [1, "tx1"]]), mp01));
  assert.ok(2 < log.proof(0).length + log.proof(1).length, "batching saves hashes");

  const mp03 = log.multiProof([0, 3]);
  assert.deepStrictEqual(mp03.map((s) => s.length), [2, 0, 1]);
  assert.ok(verifyMulti(log.root(), 8, new Map([[0, "tx0"], [3, "tx3"]]), mp03));

  const mpAll = log.multiProof([0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepStrictEqual(mpAll.map((s) => s.length), [0, 0, 0]);
  const allTargets = new Map();
  for (let i = 0; i < 8; i++) allTargets.set(i, `tx${i}`);
  assert.ok(verifyMulti(log.root(), 8, allTargets, mpAll));

  const mp246 = log.multiProof([2, 4, 6]);
  assert.deepStrictEqual(mp246.map((s) => s.length), [3, 1, 0]);
  assert.ok(verifyMulti(log.root(), 8, new Map([[2, "tx2"], [4, "tx4"], [6, "tx6"]]), mp246));

  const mp5 = log.multiProof([5]);
  const mp5Hashes = mp5.flat().map(([, h]) => h).sort((a, b) => a - b);
  const path5Hashes = log.proof(5).map(([h]) => h).sort((a, b) => a - b);
  assert.deepStrictEqual(mp5Hashes, path5Hashes);
  assert.ok(verifyMulti(log.root(), 8, new Map([[5, "tx5"]]), mp5));

  const badSib = mp01.map((level) => level.map((s) => s.slice()));
  badSib[1][0][1] ^= 1;
  assert.ok(!verifyMulti(log.root(), 8, new Map([[0, "tx0"], [1, "tx1"]]), badSib));
  assert.ok(!verifyMulti(log.root(), 8, new Map([[0, "tx0x"], [1, "tx1"]]), mp01));

  const leaves7 = [];
  for (let i = 0; i < 7; i++) leaves7.push(`tx${i}`);
  const log7 = new AuthenticatedLog(leaves7);
  const mp56 = log7.multiProof([5, 6]);
  assert.deepStrictEqual(mp56.map((s) => s.length), [1, 0, 1]);
  assert.ok(verifyMulti(log7.root(), 7, new Map([[5, "tx5"], [6, "tx6"]]), mp56));
  const mp5b = log7.multiProof([5]);
  assert.deepStrictEqual(mp5b.map((s) => s.length), [1, 1, 1]);
  assert.ok(verifyMulti(log7.root(), 7, new Map([[5, "tx5"]]), mp5b));

  const updated = leaves8.slice();
  updated[2] = "tx2*";
  const logNew = new AuthenticatedLog(updated);
  assert.strictEqual(logNew.root(), 3100337682);
  assert.strictEqual(updatedRoot("tx2*", path2), 3100337682, "update path yields new root");
  assert.ok(!verify(logNew.root(), "tx2", path2), "stale proof fails");
  assert.ok(verify(logNew.root(), "tx2*", logNew.proof(2)));
  assert.ok(!verifyMulti(logNew.root(), 8, new Map([[0, "tx0"], [1, "tx1"]]), mp01), "stale batch fails");
  const mp25New = logNew.multiProof([2, 5]);
  assert.ok(verifyMulti(logNew.root(), 8, new Map([[2, "tx2*"], [5, "tx5"]]), mp25New));
  assert.ok(!verify(logNew.root(), "tx5", log.proof(5)), "old proof of unaffected record fails");
  console.log("PASS");
}
