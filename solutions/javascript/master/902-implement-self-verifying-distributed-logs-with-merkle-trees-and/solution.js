/** Task 902 - Implement self-verifying distributed logs with Merkle trees and authenticated queries.
 *  (Tier: Master, CTCI Beyond — Distributed Systems & Verifiable Data)
 *
 * Append-only logs hashed with the RFC 6962 construction: leaf = H(0x00 || data), interior node =
 * H(0x01 || left || right), a range [a,b) splits at a + (largest power of two below b-a), and an empty
 * log hashes to H(""). Inclusion paths prove that a value is entry i of the audited log; consistency
 * proofs let a holder of root_m accept root_n iff the m-entry log is a genuine prefix — the verifier
 * re-derives every range and side from (m, n) alone and recomputes *both* roots, so a forged proof
 * cannot recombine, and two replicas disagreeing at equal size are caught forking.
 * Simplification: RFC 6962 sends O(log n) hashes per consistency proof; this demo sends the explicit
 * per-piece audit paths (O(log^2 n)) but derives ranges instead of trusting them. Sizes up to 17.
 */
"use strict";
const assert = require("node:assert");
const crypto = require("node:crypto");
const sha256 = (buf) => crypto.createHash("sha256").update(buf).digest();
const hashLeaf = (data) => sha256(Buffer.concat([Buffer.from([0x00]), Buffer.from(data)]));
const hashNode = (left, right) => sha256(Buffer.concat([Buffer.from([0x01]), left, right]));
const EMPTY_ROOT = sha256(Buffer.alloc(0));
const recordHash = (record) => hashLeaf(JSON.stringify(record));
const isPowerOfTwo = (value) => value > 0 && (value & (value - 1)) === 0;
/** Largest power of two strictly below `length` (callers guarantee length >= 2). */
function splitPoint(length) {
  let k = 1;
  while (k * 2 < length) k *= 2;
  return k;
}
/** RFC 6962 root of the leaf-hash range [lo, hi). */
function rangeRoot(lo, hi, hashes) {
  if (hi - lo === 1) return hashes[lo];
  const mid = lo + splitPoint(hi - lo);
  return hashNode(rangeRoot(lo, mid, hashes), rangeRoot(mid, hi, hashes));
}
const merkleRoot = (hashes) => (hashes.length === 0 ? EMPTY_ROOT : rangeRoot(0, hashes.length, hashes));
/** Independent oracle: incremental view — merge equal-sized subtrees, then fold the stack right to left. */
function incrementalRoot(hashes) {
  const stack = [];
  for (const hash of hashes) {
    stack.push({ size: 1, hash });
    while (stack.length >= 2 && stack[stack.length - 1].size === stack[stack.length - 2].size) {
      const right = stack.pop();
      const left = stack.pop();
      stack.push({ size: left.size + right.size, hash: hashNode(left.hash, right.hash) });
    }
  }
  let acc = null;
  for (let i = stack.length - 1; i >= 0; i--) acc = acc === null ? stack[i].hash : hashNode(stack[i].hash, acc);
  return acc === null ? EMPTY_ROOT : acc;
}
/** Maximal perfect subtrees covering [lo, hi), largest first (binary decomposition of hi - lo). */
function decompose(lo, hi) {
  const length = hi - lo;
  if (length <= 0) return [];
  if (isPowerOfTwo(length)) return [[lo, hi]];
  const mid = lo + splitPoint(length);
  return [[lo, mid]].concat(decompose(mid, hi));
}
/** Sibling sides from node [lo, hi) up to the root of [0, size), derived from ranges and sizes only. */
function pieceSides(lo, hi, size) {
  const sides = [];
  const walk = (a, b) => {
    if (a === lo && b === hi) return;
    assert.ok(lo >= a && hi <= b, `[${lo},${hi}) is not a node of the size-${size} tree`);
    const mid = a + splitPoint(b - a);
    if (hi <= mid) { walk(a, mid); sides.push("right"); } else { walk(mid, b); sides.push("left"); }
  };
  walk(0, size);
  return sides;
}
/** Prover side: the same walk plus the sibling hashes it passes, bottom-up. */
function pieceSteps(lo, hi, size, hashes) {
  const steps = [];
  const walk = (a, b) => {
    if (a === lo && b === hi) return;
    const mid = a + splitPoint(b - a);
    if (hi <= mid) { walk(a, mid); steps.push({ hash: rangeRoot(mid, b, hashes), side: "right" }); }
    else { walk(mid, b); steps.push({ hash: rangeRoot(a, mid, hashes), side: "left" }); }
  };
  walk(0, size);
  return steps;
}
/** Audit path for one leaf, bottom-up, each step tagged with the side its sibling sits on. */
function inclusionProof(hashes, index) {
  assert.ok(Number.isInteger(index) && index >= 0 && index < hashes.length, `index ${index} is out of range for a log of ${hashes.length} entries`);
  return pieceSteps(index, index + 1, hashes.length, hashes);
}

function verifyInclusion(leafHash, path, root) {
  let acc = Buffer.from(leafHash);
  for (const step of path) {
    if (!Buffer.isBuffer(step.hash) || (step.side !== "left" && step.side !== "right")) return false;
    acc = step.side === "right" ? hashNode(acc, step.hash) : hashNode(step.hash, acc);
  }
  return acc.equals(root);
}
/** Flat proof: piece hash followed by its audit-path hashes, once per piece of decompose(0, m). */
function consistencyProof(hashes, m) {
  const n = hashes.length;
  assert.ok(Number.isInteger(m) && m >= 1 && m <= n, `history size ${m} must satisfy 1 <= m <= ${n}`);
  if (m === n) return [];                            // a log is trivially consistent with itself
  const proof = [];
  for (const [lo, hi] of decompose(0, m)) {
    proof.push(rangeRoot(lo, hi, hashes));
    for (const step of pieceSteps(lo, hi, n, hashes)) proof.push(step.hash);
  }
  return proof;
}
/** Recompute both roots from the proof; every range and side is re-derived from (m, n). */
function verifyConsistency(m, rootM, rootN, proof, n) {
  if (!Number.isInteger(m) || !Number.isInteger(n) || m < 1 || n < m) return false;
  if (!Buffer.isBuffer(rootM) || !Buffer.isBuffer(rootN) || !Array.isArray(proof)) return false;
  if (m === n) return proof.length === 0 && rootM.equals(rootN);
  const pieces = decompose(0, m);
  const expected = pieces.reduce((total, [lo, hi]) => total + 1 + pieceSides(lo, hi, n).length, 0);
  if (proof.length !== expected) return false;
  let cursor = 0;
  const pieceHashes = [];
  for (const [lo, hi] of pieces) {
    let acc = proof[cursor++];
    if (!Buffer.isBuffer(acc)) return false;
    pieceHashes.push(acc);
    for (const side of pieceSides(lo, hi, n)) {
      const sibling = proof[cursor++];
      if (!Buffer.isBuffer(sibling)) return false;
      acc = side === "right" ? hashNode(acc, sibling) : hashNode(sibling, acc);
    }
    if (!acc.equals(rootN)) return false;            // this piece is not a subtree of root_n
  }
  let acc = pieceHashes[pieceHashes.length - 1];
  for (let i = pieceHashes.length - 2; i >= 0; i--) acc = hashNode(pieceHashes[i], acc);
  return acc.equals(rootM);                          // folding the pieces must reproduce the old root
}

class LogReplica {
  constructor(name) { this.name = name; this.records = []; this.leaves = []; }
  append(key, value) {
    const record = { index: this.records.length, key, value };
    this.records.push(record);
    this.leaves.push(recordHash(record));
    return record;
  }
  get size() { return this.leaves.length; }
  root() { return merkleRoot(this.leaves); }
  proof(index) { return inclusionProof(this.leaves, index); }
  /** Authenticated query: the value plus its inclusion proof, or null when the key is absent. */
  query(key) {
    const index = this.records.findIndex((record) => record.key === key);
    return index < 0 ? null : { index, record: this.records[index], path: this.proof(index) };
  }
}
/** Client side: an answer counts only if its record hashes to the leaf under the audited root. */
const verifyQuery = (root, answer) => Boolean(answer) && answer.record.index === answer.index && verifyInclusion(recordHash(answer.record), answer.path, root);
/** One gossip exchange: equal sizes compare roots, otherwise the smaller side checks consistency. */
function gossip(a, b) {
  if (a.size === b.size) return { kind: "roots", agreed: a.root().equals(b.root()) };
  const [small, large] = a.size < b.size ? [a, b] : [b, a];
  const proof = consistencyProof(large.leaves, small.size);
  return { kind: "consistency", agreed: verifyConsistency(small.size, small.root(), large.root(), proof, large.size) };
}

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0;
    this.state = x;
    return x;
  }
  int(bound) { return this.next() % bound; }
}
function main() {
  const build = (size) => {                          // same size => same content, so roots are comparable
    const prng = new Prng(0x5eed0902 + size);
    const replica = new LogReplica(`r${size}`);
    for (let i = 0; i < size; i++) replica.append(`key-${i}`, `value-${prng.int(1 << 20)}`);
    return replica;
  };
  // 1. happy path — every entry of every size proves inclusion with a short path folding to the root.
  for (let size = 1; size <= 17; size++) {
    const replica = build(size);
    for (let index = 0; index < size; index++) {
      const path = replica.proof(index);
      assert.ok(verifyInclusion(replica.leaves[index], path, replica.root()), `inclusion ${index}/${size}`);
      assert.ok(path.length <= Math.ceil(Math.log2(size)) + 1, `path length ${index}/${size}`);
    }
  }
  // 2. tampering — flipped siblings, wrong sides, truncation and foreign leaves are all rejected.
  {
    const replica = build(9);
    const root = replica.root();
    const path = replica.proof(5);
    const forgeries = path.map((_, i) => path.map((step, j) => (j === i ? { hash: hashLeaf(`t${i}`), side: step.side } : step)))
      .concat([path.slice(1), [], path.map((step) => ({ hash: step.hash, side: step.side === "left" ? "right" : "left" }))]);
    for (const forged of forgeries) assert.ok(!verifyInclusion(replica.leaves[5], forged, root), "tampered path");
    assert.ok(!verifyInclusion(hashLeaf("not-in-log"), path, root), "a foreign leaf is rejected");
    assert.ok(!verifyInclusion(replica.leaves[5], path, hashLeaf("wrong-root")), "a wrong root is rejected");
  }
  // 3. boundary — empty log, single entry, bad indices and bad history sizes.
  {
    const empty = new LogReplica("empty");
    assert.ok(empty.root().equals(sha256(Buffer.alloc(0))) && empty.root().equals(EMPTY_ROOT), "empty root");
    const single = new LogReplica("single");
    single.append("k", "v");
    assert.deepStrictEqual(single.proof(0), [], "a single-entry log has an empty audit path");
    assert.ok(verifyInclusion(single.leaves[0], [], single.root()));
    assert.throws(() => single.proof(1), /out of range/);
    assert.throws(() => single.proof(-1), /out of range/);
    assert.throws(() => consistencyProof(single.leaves, 0), /1 <= m/);
  }
  // 4. consistency — every pair with 1 <= m <= n <= 17 verifies against both roots.
  for (let n = 1; n <= 17; n++) {
    const replica = build(n);
    for (let m = 1; m <= n; m++) {
      assert.ok(verifyConsistency(m, merkleRoot(replica.leaves.slice(0, m)), replica.root(), consistencyProof(replica.leaves, m), n), `consistency ${m} -> ${n}`);
    }
  }
  // 5. consistency tampering — any flipped hash, truncation, padding or wrong size/root is rejected.
  {
    const replica = build(13);
    const oldRoot = merkleRoot(replica.leaves.slice(0, 5));
    const newRoot = replica.root();
    const proof = consistencyProof(replica.leaves, 5);
    const forgeries = proof.map((_, i) => proof.map((h, j) => (j === i ? hashLeaf(`forged-${i}`) : h)))
      .concat([proof.slice(1), proof.concat([EMPTY_ROOT]), proof.slice(0, -1)]);
    for (const forged of forgeries) assert.ok(!verifyConsistency(5, oldRoot, newRoot, forged, 13), "forged proof");
    assert.ok(!verifyConsistency(4, oldRoot, newRoot, proof, 13), "wrong history size");
    assert.ok(!verifyConsistency(5, newRoot, newRoot, proof, 13), "wrong old root");
    assert.ok(!verifyConsistency(5, oldRoot, oldRoot, proof, 13), "wrong new root");
    assert.ok(verifyConsistency(13, newRoot, newRoot, [], 13), "a log is consistent with itself");
    assert.ok(!verifyConsistency(13, oldRoot, newRoot, [], 13), "self-consistency needs equal roots");
    assert.ok(!verifyConsistency(14, newRoot, newRoot, [], 13), "m > n is invalid");
  }
  // 6. independent oracle — the incremental stack fold agrees with the recursive RFC 6962 root.
  for (let size = 0; size <= 17; size++) {
    const replica = build(size);
    assert.ok(incrementalRoot(replica.leaves).equals(replica.root()), `stack vs batch root at n=${size}`);
  }
  assert.ok(incrementalRoot([]).equals(EMPTY_ROOT), "the empty stack root");
  // 7. gossip — equal logs agree, extensions verify, forks are caught at equal and at unequal size.
  {
    const a = build(7);
    assert.deepStrictEqual(gossip(a, build(7)), { kind: "roots", agreed: true }, "identical logs agree");
    const longer = new LogReplica("longer");
    for (const record of a.records) longer.append(record.key, record.value);
    for (let i = 7; i < 11; i++) longer.append(`key-${i}`, `extra-${i}`);
    assert.ok(gossip(a, longer).agreed && gossip(longer, a).agreed, "a prefix is consistent with its extension");
    const forked = build(7);
    forked.records[3].value = "rewritten";
    forked.leaves[3] = recordHash(forked.records[3]);
    assert.ok(!gossip(a, forked).agreed, "a rewritten entry is caught at equal size");
    const spliced = new LogReplica("spliced");
    for (let i = 0; i < 7; i++) spliced.append(a.records[i].key, i === 2 ? "rewritten" : a.records[i].value);
    spliced.append("tail", "x");
    assert.ok(!gossip(a, spliced).agreed, "a divergent longer log is rejected");
  }
  // 8. authenticated query — honest answers verify, absent keys and forgeries are not accepted.
  {
    const replica = build(12);
    const root = replica.root();
    const answer = replica.query("key-4");
    assert.ok(verifyQuery(root, answer), "an honest answer verifies against the audited root");
    assert.strictEqual(answer.record.value, replica.records[4].value);
    assert.strictEqual(replica.query("key-999"), null, "an absent key has no answer");
    assert.ok(!verifyQuery(root, { ...answer, record: { ...answer.record, value: "forged" } }), "forged value");
    assert.ok(!verifyQuery(root, { ...answer, index: 5 }), "a mismatched index fails");
    assert.ok(!verifyQuery(root, null), "no answer is never accepted");
    assert.ok(!verifyQuery(EMPTY_ROOT, answer), "nothing verifies against the empty root");
  }
  // 9. determinism — identical records give identical roots, any change moves the root.
  {
    const first = new LogReplica("a");
    const second = new LogReplica("b");
    const third = new LogReplica("c");
    for (let i = 0; i < 8; i++) { first.append(`k${i}`, i); second.append(`k${i}`, i); third.append(`k${i}`, i + 1); }
    assert.ok(first.root().equals(second.root()), "same records hash to the same root");
    assert.ok(!first.root().equals(third.root()), "a changed payload changes the root");
    second.append("k8", 8);
    assert.ok(!first.root().equals(second.root()), "appending changes the root");
  }
  const proofSizes = [1, 4, 7, 13, 17].map((m) => consistencyProof(build(17).leaves, m).length);
  console.log(`merkle-rfc6962: inclusion verified for n = 1..17, consistency for all 1 <= m <= n <= 17; ` + `at n = 17 proof hashes for m in [1,4,7,13,17] = [${proofSizes}]`);
  console.log("PASS");
}
if (require.main === module) main();

module.exports = {
  sha256, hashLeaf, hashNode, recordHash, EMPTY_ROOT, merkleRoot, incrementalRoot, decompose, pieceSides, inclusionProof, verifyInclusion, consistencyProof, verifyConsistency, LogReplica, verifyQuery, gossip, Prng, };
