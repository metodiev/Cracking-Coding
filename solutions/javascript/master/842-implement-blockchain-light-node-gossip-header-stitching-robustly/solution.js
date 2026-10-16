/** Task 842 - Implement blockchain light node gossip + header stitching robustly.
 *  (Tier: Master, CTCI Ch 9 — System Design and Scalability)
 *
 * A light (SPV) node keeps only block *headers*, yet it must (a) decide which chain is real and
 * (b) prove that a transaction is inside a block. Both halves are implemented:
 *
 *  - header = { height, parent, merkleRoot, time, difficulty, nonce }. Proof of work means the first
 *    64 bits of SHA-256(canonical header) are <= floor((2^64 - 1) / difficulty). Difficulty is
 *    retargeted every `retargetEvery` blocks from the *logical* block times, and every validator
 *    recomputes that rule, so a peer cannot simply publish an easier block;
 *  - chain selection is by highest cumulative work, where one header contributes
 *    floor(2^64 / (target + 1)) — with retargeting this is not the same as "longest chain";
 *  - SPV proofs are Merkle inclusion paths against the header's merkleRoot, verified with the leaf
 *    index deciding each left/right concatenation, so a path cannot be replayed at another position;
 *  - gossip + stitching: the node talks to peers over an explicit in-memory message network, asks for
 *    headers after a given hash, keeps headers whose parent is still unknown *pending*, and adopts a
 *    header only once parent link, height, time, difficulty and proof of work all validate.
 *
 * Byzantine peers are modelled directly: one that publishes a broken proof of work, one that re-mines
 * a block at a lower difficulty, one that re-mines with a wrong height, one that mine a header whose
 * parent does not exist, and one that withholds headers. The light node must reject and blame the
 * first three, keep the orphan pending without blaming anyone, keep working when a liar is in the
 * peer set — and *fail* to make progress when the only peer withholds, which is the honest liveness
 * statement about a headers-only node.
 *
 * Simplifications, stated honestly: difficulties are tiny (a block costs a few hundred hashes), the
 * retarget window is four blocks, there is no mempool, fees, finality rule or consensus voting beyond
 * "most work", and transactions are validated only for Merkle inclusion. Time is a logical integer
 * carried in the header (never the wall clock), all randomness comes from seeded splitmix64, and
 * there are no timers, threads or sockets.
 *
 * Invariants verified below:
 *  - PoW and structural safety: every accepted header meets its target, and height/parent/time/
 *    difficulty are recomputed independently, including the retarget rule;
 *  - chain selection: the head equals the maximum-cumulative-work tip found by a brute-force
 *    enumeration of every branch in the header graph, including a branch that is *shorter but
 *    heavier* than its rival;
 *  - SPV soundness: proofs verify for every transaction in every block, while a tampered transaction,
 *    a replayed index, a truncated path and a wrong root are rejected;
 *  - liveness under gossip: five delivery orders (in order, reverse, shuffled, duplicated, gapped)
 *    stitch to the same head within a bounded number of rounds and messages;
 *  - Byzantine robustness: forged headers are rejected with their specific reason, blamed on the peer
 *    that served them, never adopted, and the honest tip is still reached; an orphan never enters the
 *    chain without its parent, and a withholding peer cannot stop progress while an honest peer
 *    exists;
 *  - determinism: the same seeds reproduce the same tip, the same verdicts and the same rejections.
 */
"use strict";

const crypto = require("node:crypto");

const MASK64 = (1n << 64n) - 1n;
const HASH_SPACE = 1n << 64n; // the space the truncated header hash lives in
const sha256Hex = (text) => crypto.createHash("sha256").update(text).digest("hex");

/** Deterministic serialisation for hashing: sorted keys, big integers in decimal. */
function canonical(value) {
  if (value === null || value === undefined) return "null";
  if (typeof value === "bigint") return value.toString(10);
  if (typeof value === "number" || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${canonical(value[k])}`)
    .join(",")}}`;
}

/** splitmix64: the only source of pseudo-randomness. */
class SplitMix64 {
  constructor(seed) {
    this.state = (BigInt(seed) & MASK64) || 1n;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return (z ^ (z >> 31n)) & MASK64;
  }

  upto(bound) {
    return Number(this.next() % BigInt(bound));
  }
}

// --- Merkle trees and SPV proofs --------------------------------------------
const leafHash = (txid) => sha256Hex(`leaf:${txid}`);
const nodeHash = (left, right) => sha256Hex(`node:${left}|${right}`);

/** Iterative Merkle root; an odd level duplicates its last node (Bitcoin's rule). */
function merkleRoot(txids) {
  if (txids.length === 0) return sha256Hex("empty-block");
  let level = txids.map(leafHash);
  while (level.length > 1) {
    if (level.length % 2 === 1) level.push(level[level.length - 1]);
    const next = [];
    for (let i = 0; i < level.length; i += 2) next.push(nodeHash(level[i], level[i + 1]));
    level = next;
  }
  return level[0];
}

/** Independent recursive implementation of the same pairing rule (no padding pass, no loop). */
function merkleRootPairs(txids) {
  if (txids.length === 0) return sha256Hex("empty-block");
  const pairUp = (nodes) => {
    if (nodes.length === 1) return nodes;
    const next = nodes.map((node, index) => (index % 2 === 0
      ? nodeHash(node, nodes[Math.min(index + 1, nodes.length - 1)])
      : null)).filter((node) => node !== null);
    return pairUp(next);
  };
  return pairUp(txids.map(leafHash))[0];
}

/** Sibling path for the leaf at `index`; the verifier derives left/right from the index bits. */
function merkleProof(txids, index) {
  if (index < 0 || index >= txids.length) throw new Error("index out of range");
  let level = txids.map(leafHash);
  let position = index;
  const proof = [];
  while (level.length > 1) {
    if (level.length % 2 === 1) level.push(level[level.length - 1]);
    proof.push(level[position % 2 === 0 ? position + 1 : position - 1]);
    const next = [];
    for (let i = 0; i < level.length; i += 2) next.push(nodeHash(level[i], level[i + 1]));
    level = next;
    position = Math.floor(position / 2);
  }
  return proof;
}

/** SPV verification: the leaf index decides the order of every concatenation. */
function verifyMerkleProof(txid, index, proof, root) {
  let running = leafHash(txid);
  let position = index;
  for (const sibling of proof) {
    if (typeof sibling !== "string" || !/^[0-9a-f]{64}$/.test(sibling)) return false;
    running = position % 2 === 0 ? nodeHash(running, sibling) : nodeHash(sibling, running);
    position = Math.floor(position / 2);
  }
  return running === root;
}

// --- proof of work and consensus rules --------------------------------------
/** Target for a difficulty: floor((2^64 - 1) / difficulty); more difficulty = smaller target. */
const targetFor = (difficulty) => (HASH_SPACE - 1n) / BigInt(difficulty);

const hashHeader = (header) => sha256Hex(canonical(header));
const powValue = (header) => BigInt(`0x${hashHeader(header).slice(0, 16)}`);
const powOk = (header) => powValue(header) <= targetFor(header.difficulty);
const headerWork = (header) => HASH_SPACE / (targetFor(header.difficulty) + 1n);

const DEFAULT_CONSENSUS = {
  retargetEvery: 4,
  spacing: 3,
  minDifficulty: 8n,
  maxDifficulty: 1024n,
  genesisDifficulty: 32n,
  batchSize: 6,
};

const clamp = (value, low, high) => (value < low ? low : value > high ? high : value);

/** Walks a header map back to the ancestor at `height` (null when it is not loaded yet). */
function ancestorAtHeight(chain, header, height) {
  let current = header;
  for (let guard = 0; current !== null && current.height > height && guard < 10000; guard++) {
    current = chain.get(current.parent) ?? null;
  }
  return current !== null && current.height === height ? current : null;
}

/** The retarget rule: miners apply it, validators recompute it. */
function expectedDifficulty(chain, parent, consensus) {
  const height = parent.height + 1;
  if (height <= consensus.retargetEvery || height % consensus.retargetEvery !== 0) {
    return parent.difficulty;
  }
  const ancestor = ancestorAtHeight(chain, parent, height - consensus.retargetEvery);
  if (ancestor === null) return null; // the window is not loaded yet: undecidable, not invalid
  const expected = BigInt(consensus.retargetEvery * consensus.spacing);
  const actual = BigInt(Math.max(1, parent.time - ancestor.time));
  return clamp((parent.difficulty * expected) / actual, consensus.minDifficulty,
    consensus.maxDifficulty);
}

/** Mines PoW over an arbitrary field set: the same primitive serves honest miners and liars. */
function mineFields(fields, nonceSeed) {
  const rng = new SplitMix64(nonceSeed);
  const base = { ...fields, nonce: rng.upto(1 << 20) };
  for (let attempt = 0; attempt < 500000; attempt++) {
    const candidate = { ...base, nonce: base.nonce + attempt };
    if (powOk(candidate)) return candidate;
  }
  throw new Error("mining failed");
}

/** Mines one honest header on top of `parent`. */
function mineHeader({ parent, txids, time, difficulty, nonceSeed }) {
  return mineFields({
    height: parent === null ? 0 : parent.height + 1,
    parent: parent === null ? "0".repeat(64) : hashHeader(parent),
    merkleRoot: merkleRoot(txids),
    time,
    difficulty,
  }, nonceSeed);
}

/**
 * Builds an honest chain. Block times come from the seeded PRNG, so the retarget rule genuinely moves
 * the difficulty and the cumulative work differs between branches.
 */
function buildHonestChain(options = {}) {
  const consensus = { ...DEFAULT_CONSENSUS, ...(options.consensus ?? {}) };
  const blocks = options.blocks ?? 24;
  const seed = options.seed ?? 842;
  const rng = new SplitMix64(seed);
  const chain = new Map();
  const transactions = new Map();
  const list = [];
  let parent = null;
  for (let i = 0; i < blocks; i++) {
    const difficulty = parent === null ? consensus.genesisDifficulty
      : expectedDifficulty(chain, parent, consensus);
    const time = parent === null ? 100 : parent.time + 1 + rng.upto(2 * consensus.spacing);
    const txids = Array.from({ length: 1 + rng.upto(5) }, (_, k) => `tx-${i}-${k}`);
    const header = mineHeader({ parent, txids, time, difficulty, nonceSeed: seed * 1000 + i });
    chain.set(hashHeader(header), header);
    transactions.set(hashHeader(header), txids);
    list.push(header);
    parent = header;
  }
  return { consensus, chain, transactions, list, tip: parent };
}

/** Extends `chain` from `from` with a competing branch. `timeDelta` controls block speed. */
function extendFork({ chain, transactions, from, length, seed, consensus, timeDelta = null }) {
  const rng = new SplitMix64(seed);
  let parent = from;
  const list = [];
  for (let i = 0; i < length; i++) {
    const difficulty = expectedDifficulty(chain, parent, consensus);
    const time = parent.time + (timeDelta === null ? 1 + rng.upto(2 * consensus.spacing) : timeDelta);
    const txids = [`fork-${seed}-${i}`];
    const header = mineHeader({ parent, txids, time, difficulty, nonceSeed: seed * 77 + i + 1 });
    chain.set(hashHeader(header), header);
    transactions.set(hashHeader(header), txids);
    list.push(header);
    parent = header;
  }
  return { list, tip: parent };
}

/** Reason codes that mean "this peer lied", as opposed to benign gossip states. */
const BLAMEWORTHY = new Set(["pow", "difficulty", "height", "time", "bad-height",
  "bad-time-or-nonce", "bad-merkle-root", "bad-difficulty", "malformed"]);

/** Cheap shape checks that need no chain view. */
function structuralCheck(header) {
  if (typeof header !== "object" || header === null) return "malformed";
  for (const field of ["height", "parent", "merkleRoot", "time", "difficulty", "nonce"]) {
    if (!(field in header)) return `missing-${field}`;
  }
  if (!Number.isInteger(header.height) || header.height < 0) return "bad-height";
  if (!Number.isInteger(header.time) || !Number.isInteger(header.nonce)) return "bad-time-or-nonce";
  if (typeof header.merkleRoot !== "string" || header.merkleRoot.length !== 64) return "bad-merkle-root";
  if (typeof header.difficulty !== "bigint" || header.difficulty < 1n) return "bad-difficulty";
  if (typeof header.parent !== "string" || header.parent.length !== 64) return "bad-parent";
  return null;
}

/**
 * Validates one header against the node's view. Returns null when it is valid, "orphan" when its
 * parent is not loaded, "unverifiable" when its retarget window is not loaded, or a lie code when the
 * header provably breaks a rule.
 */
function validateHeader(chain, header, consensus) {
  const structural = structuralCheck(header);
  if (structural !== null) return structural;
  if (!powOk(header)) return "pow";
  const parent = chain.get(header.parent);
  if (parent === undefined) return "orphan";
  if (header.height !== parent.height + 1) return "height";
  if (header.time <= parent.time) return "time";
  const expected = expectedDifficulty(chain, parent, consensus);
  if (expected === null) return "unverifiable";
  if (header.difficulty !== expected) return "difficulty";
  return null;
}

/** Explicit in-memory message network between the light node and its peers. */
class Network {
  constructor() {
    this.queues = new Map();
    this.sent = 0;
  }

  send(from, to, type, payload) {
    const key = `${from}->${to}`;
    if (!this.queues.has(key)) this.queues.set(key, []);
    this.queues.get(key).push({ type, from, payload });
    this.sent++;
  }

  drain(to) {
    const inbox = [];
    for (const [key, queue] of this.queues) {
      if (key.split("->")[1] !== to) continue;
      inbox.push(...queue);
      queue.length = 0;
    }
    return inbox;
  }
}

/**
 * A full node serving headers-first batches. `defect` turns it Byzantine; the forged headers are
 * *re-mined* so their proof of work is valid and only the intended rule is broken.
 */
class FullNode {
  constructor(id, chain, transactions, consensus, defect = null) {
    this.id = id;
    this.chain = chain;
    this.transactions = transactions;
    this.consensus = consensus;
    this.defect = defect;
    this.served = 0;
    this.forged = 0;
  }

  /** Up to `batchSize` headers starting at `fromHash` (inclusive) plus their successors. */
  serve(fromHash) {
    const start = this.chain.get(fromHash);
    this.served++;
    if (start === undefined) return [];
    if (this.defect === "withhold" && this.served > 4) return [];
    const batch = [];
    let current = start;
    for (let i = 0; i < this.consensus.batchSize; i++) {
      batch.push(this.forge(current));
      const successor = [...this.chain.values()]
        .find((candidate) => candidate.parent === hashHeader(current));
      if (successor === undefined) break;
      current = successor;
    }
    return batch;
  }

  /** Re-mines a header so that only the rule named by `defect` is violated. */
  forge(header) {
    if (this.defect === null || this.defect === "withhold") return header;
    this.forged++;
    const seed = this.forged * 31 + header.height;
    if (this.defect === "pow") return { ...header, nonce: header.nonce + 1 };
    if (this.defect === "link") {
      return mineFields({ ...header, parent: "f".repeat(64) }, seed);
    }
    if (this.defect === "difficulty") {
      return mineFields({ ...header, difficulty: this.consensus.minDifficulty }, seed);
    }
    if (this.defect === "height") {
      return mineFields({ ...header, height: header.height + 2 }, seed);
    }
    return header;
  }
}

/**
 * The light node: validated headers, stitched gaps, most-work head, and blame for peers that serve
 * headers which provably break a rule. Headers whose parent or retarget window is still missing are
 * parked instead of being discarded, and re-checked whenever new headers arrive.
 */
class LightNode {
  constructor(id, anchor, consensus) {
    this.id = id;
    this.consensus = consensus;
    this.headers = new Map([[hashHeader(anchor), anchor]]);
    this.work = new Map([[hashHeader(anchor), 0n]]);
    this.pending = new Map();
    this.orphans = new Set();
    this.waiting = new Map();
    this.head = anchor;
    this.badPeers = new Map();
    this.rejected = [];
    this.rounds = 0;
  }

  get headHash() {
    return hashHeader(this.head);
  }

  /** Adopted plus parked headers: enough to check parent links and retarget windows. */
  knownView() {
    return new Map([...this.headers, ...this.pending]);
  }

  /**
   * Accepts one header and resolves as much as possible. Returns null once it is on the chain,
   * "orphan" / "unverifiable" while it waits for context, or the reason code that condemns it.
   */
  adopt(header, peerId = null) {
    const hash = hashHeader(header);
    if (this.headers.has(hash)) return null;
    if (!this.pending.has(hash)) {
      const structural = structuralCheck(header);
      if (structural !== null) {
        this.record(hash, structural, peerId);
        return structural;
      }
      this.pending.set(hash, header);
      this.orphans.add(hash);
    }
    this.resolve(peerId);
    if (this.headers.has(hash)) return null;
    if (this.pending.has(hash)) return this.waiting.get(hash) ?? "orphan";
    return this.rejected.find((entry) => entry.hash === hash)?.reason ?? null;
  }

  /** Re-checks every parked header until no further progress is possible. */
  resolve(peerId) {
    let progressed = true;
    while (progressed) {
      progressed = false;
      for (const [hash, header] of [...this.pending]) {
        const reason = validateHeader(this.knownView(), header, this.consensus);
        if (reason === "orphan" || reason === "unverifiable") {
          this.waiting.set(hash, reason);
          continue;
        }
        if (reason !== null) {
          this.discard(hash);
          this.record(hash, reason, peerId);
          continue;
        }
        const parentWork = this.work.get(header.parent);
        if (parentWork === undefined) {
          this.waiting.set(hash, "orphan"); // valid, but its parent is not on the chain yet
          continue;
        }
        const candidateWork = parentWork + headerWork(header);
        const headWork = this.work.get(this.headHash) ?? 0n;
        this.headers.set(hash, header);
        this.work.set(hash, candidateWork);
        this.discard(hash);
        progressed = true;
        if (candidateWork > headWork ||
            (candidateWork === headWork && header.height > this.head.height)) {
          this.head = header;
        }
      }
    }
  }

  discard(hash) {
    this.pending.delete(hash);
    this.orphans.delete(hash);
    this.waiting.delete(hash);
  }

  record(hash, reason, peerId) {
    this.rejected.push({ hash, reason, peerId });
    if (peerId !== null && BLAMEWORTHY.has(reason)) {
      this.badPeers.set(peerId, (this.badPeers.get(peerId) ?? 0) + 1);
    }
  }

  learnFrom(peerId, batch) {
    let accepted = 0;
    for (const header of batch ?? []) if (this.adopt(header, peerId) !== null) accepted++;
    return accepted;
  }

  /** Parent hashes of parked headers that the node still needs. */
  missingParents() {
    const wanted = new Set();
    for (const header of this.pending.values()) {
      if (!this.headers.has(header.parent)) wanted.add(header.parent);
    }
    return [...wanted].sort();
  }

  /**
   * Gossip and stitching: one request/response round per iteration. The node asks every peer for each
   * missing parent (or for the successor of its current head), each peer answers from its own chain,
   * and the loop stops as soon as a round learns nothing new.
   */
  sync(peers, network, maxRounds = 100) {
    while (this.rounds < maxRounds) {
      this.rounds++;
      const before = this.headers.size + this.pending.size;
      // Always pull the successor of the head as well, so unresolvable orphans cannot stall the node.
      const requests = [...new Set([...this.missingParents(), this.headHash])];
      for (const fromHash of requests) {
        for (const peer of peers) {
          network.send(this.id, peer.id, "getheaders", { fromHash });
        }
        for (const peer of peers) {
          for (const message of network.drain(peer.id)) {
            if (message.type !== "getheaders") continue;
            network.send(peer.id, this.id, "headers", peer.serve(message.payload.fromHash));
          }
        }
        for (const reply of network.drain(this.id)) {
          this.learnFrom(reply.from, reply.payload ?? []);
        }
      }
      if (this.headers.size + this.pending.size === before) break;
    }
    return this.head;
  }
}

/** Brute force: enumerate every branch of the header graph and return the maximum-work tip. */
function bruteForceBestChain(headers, anchorHash) {
  const byHash = new Map(headers.map((header) => [hashHeader(header), header]));
  const anchor = byHash.get(anchorHash);
  if (anchor === undefined) return null;
  const children = new Map();
  for (const [hash, header] of byHash) {
    if (!children.has(header.parent)) children.set(header.parent, []);
    children.get(header.parent).push(hash);
  }
  let best = { work: 0n, tip: anchorHash, height: anchor.height };
  const walk = (hash, work, height) => {
    if (work > best.work || (work === best.work && height > best.height)) {
      best = { work, tip: hash, height };
    }
    for (const child of children.get(hash) ?? []) {
      const header = byHash.get(child);
      walk(child, work + headerWork(header), header.height);
    }
  };
  walk(anchorHash, 0n, anchor.height);
  return best;
}

module.exports = { MASK64, HASH_SPACE, sha256Hex, canonical, SplitMix64, leafHash, nodeHash,
  merkleRoot, merkleRootPairs, merkleProof, verifyMerkleProof, targetFor, hashHeader, powValue,
  powOk, headerWork, DEFAULT_CONSENSUS, BLAMEWORTHY, clamp, ancestorAtHeight, expectedDifficulty,
  mineFields, mineHeader, buildHonestChain, extendFork, structuralCheck, validateHeader, Network,
  FullNode, LightNode, bruteForceBestChain };

if (require.main === module) {
  const assert = require("node:assert");
  const consensus = DEFAULT_CONSENSUS;
  const built = buildHonestChain({ blocks: 24, seed: 842 });
  const hashOf = (header) => hashHeader(header);
  const workOf = (headers) => headers.reduce((acc, header) => acc + headerWork(header), 0n);

  // --- proof of work ---------------------------------------------------------
  assert.strictEqual(built.list.length, 24, "24 blocks were mined");
  assert.strictEqual(built.list[0].height, 0, "the chain starts at height 0");
  for (const header of built.list) {
    assert.ok(powOk(header), `header ${header.height} meets its target`);
  }
  assert.strictEqual(powOk({ ...built.list[10], nonce: built.list[10].nonce + 1 }), false,
    "a bumped nonce breaks the proof of work");
  assert.ok(targetFor(1024n) < targetFor(64n), "higher difficulty means a smaller target");
  assert.strictEqual(headerWork({ difficulty: 64n }), 64n, "work is the expected hashes per block");

  // --- structure and the retarget rule --------------------------------------
  let retargets = 0;
  for (let i = 1; i < built.list.length; i++) {
    const header = built.list[i];
    const parent = built.list[i - 1];
    assert.strictEqual(header.parent, hashOf(parent), `block ${i} links to its parent`);
    assert.strictEqual(header.height, parent.height + 1, `block ${i} height is contiguous`);
    assert.ok(header.time > parent.time, `block ${i} time strictly increases`);
    assert.strictEqual(header.difficulty, expectedDifficulty(built.chain, parent, consensus),
      `block ${i} difficulty follows the retarget rule`);
    assert.ok(header.difficulty >= consensus.minDifficulty &&
      header.difficulty <= consensus.maxDifficulty, "difficulty stays inside its clamps");
    if (header.difficulty !== parent.difficulty) retargets++;
  }
  assert.ok(retargets >= 2, `the retarget rule fired ${retargets} times`);
  const difficulties = new Set(built.list.map((header) => header.difficulty.toString()));
  assert.ok(difficulties.size >= 2, "difficulty is not constant, so work varies per block");

  // --- Merkle roots and SPV proofs ------------------------------------------
  const sampleTxs = ["a", "b", "c", "d", "e"];
  assert.strictEqual(merkleRoot(sampleTxs), merkleRootPairs(sampleTxs),
    "the iterative root matches the recursive oracle");
  for (let size = 1; size <= 9; size++) {
    const txids = Array.from({ length: size }, (_, i) => `t${i}`);
    assert.strictEqual(merkleRoot(txids), merkleRootPairs(txids), `odd/even sizes agree (${size})`);
  }
  assert.strictEqual(merkleRoot([]), sha256Hex("empty-block"), "an empty block has a fixed root");
  let proofsChecked = 0;
  for (const header of built.list) {
    const txids = built.transactions.get(hashOf(header));
    assert.strictEqual(merkleRoot(txids), header.merkleRoot, `block ${header.height} commits to its txs`);
    for (let index = 0; index < txids.length; index++) {
      const proof = merkleProof(txids, index);
      assert.strictEqual(verifyMerkleProof(txids[index], index, proof, header.merkleRoot), true,
        `proof ${index} of block ${header.height} verifies`);
      assert.strictEqual(verifyMerkleProof(txids[index], index, proof, hashOf(header)), false,
        "the same proof fails against a different root");
      assert.strictEqual(verifyMerkleProof(`${txids[index]}x`, index, proof, header.merkleRoot), false,
        "a tampered transaction fails");
      if (proof.length > 0) {
        assert.strictEqual(verifyMerkleProof(txids[index], (index + 1) % txids.length, proof,
          header.merkleRoot), false, "a proof cannot be replayed at another index");
        assert.strictEqual(verifyMerkleProof(txids[index], index, proof.slice(0, -1),
          header.merkleRoot), false, "a truncated path fails");
      }
      proofsChecked++;
    }
  }
  assert.ok(proofsChecked >= 48, `SPV proofs checked: ${proofsChecked}`);

  // --- gossip: five delivery orders stitch to the same head ------------------
  const allHeaders = built.list.slice(1);
  const orders = {
    "in order": () => allHeaders.map((header) => [header]),
    "reverse": () => allHeaders.slice().reverse().map((header) => [header]),
    "shuffled": () => allHeaders.map((header, index) => ({ header, key: (index * 7919) % 23 }))
      .sort((a, b) => a.key - b.key).map(({ header }) => [header]),
    "duplicated batches": () => allHeaders.flatMap((header) => [[header], [header]]),
    "gapped then stitched": () => allHeaders.filter((_, index) => index % 3 !== 1)
      .map((header) => [header]),
  };
  let stitched = 0;
  for (const [label, plan] of Object.entries(orders)) {
    const node = new LightNode("light", built.list[0], consensus);
    const network = new Network();
    const peers = [new FullNode("p0", built.chain, built.transactions, consensus)];
    for (const batch of plan()) node.learnFrom("p0", batch);
    const head = node.sync(peers, network);
    assert.strictEqual(hashOf(head), hashOf(built.tip), `${label}: stitched to the honest tip`);
    assert.strictEqual(node.pending.size, 0, `${label}: nothing is left pending`);
    assert.deepStrictEqual(node.rejected, [], `${label}: no spurious rejections`);
    assert.ok(node.rounds <= 40, `${label}: sync used ${node.rounds} rounds`);
    stitched++;
  }
  assert.strictEqual(stitched, 5, "all five delivery orders were exercised");

  // --- most work wins: shorter but heavier branch ---------------------------
  const forked = buildHonestChain({ blocks: 20, seed: 950 });
  const slowBranch = extendFork({ chain: forked.chain, transactions: forked.transactions,
    from: forked.tip, length: 9, seed: 61, timeDelta: 12, consensus });
  const fastBranch = extendFork({ chain: forked.chain, transactions: forked.transactions,
    from: forked.tip, length: 8, seed: 62, timeDelta: 1, consensus });
  const commonWork = workOf(forked.list);
  assert.ok(commonWork > 0n, "the common prefix carries work");
  assert.ok(workOf(fastBranch.list) !== workOf(slowBranch.list),
    "the two branches differ in cumulative work");
  const branchHeaders = forked.list.slice(1).concat(slowBranch.list, fastBranch.list);
  const forkNode = new LightNode("light", forked.list[0], consensus);
  for (const header of branchHeaders.slice().reverse()) forkNode.adopt(header, "seed");
  const best = bruteForceBestChain([forked.list[0]].concat(branchHeaders), hashOf(forked.list[0]));
  assert.strictEqual(hashOf(forkNode.head), best.tip, "the head equals the brute-force best tip");
  assert.strictEqual(forkNode.head.height, best.height, "and it is at the brute-force height");
  assert.strictEqual(forkNode.work.get(forkNode.headHash), best.work,
    "the node's cumulative work matches the brute force");
  assert.strictEqual(forkNode.pending.size, 0, "the fork stitched completely");
  assert.ok(forkNode.work.has(hashOf(fastBranch.tip)) && forkNode.work.has(hashOf(slowBranch.tip)),
    "both branch tips are remembered even though only one is the head");
  const chosen = forkNode.head;
  const rival = chosen === fastBranch.tip ? slowBranch.tip : fastBranch.tip;
  assert.ok(forkNode.work.get(hashOf(chosen)) >= forkNode.work.get(hashOf(rival)),
    "the chosen branch carries at least as much work as its rival");
  assert.ok(fastBranch.tip.height < slowBranch.tip.height &&
    workOf(fastBranch.list) > workOf(slowBranch.list),
  "the faster branch is shorter yet heavier");
  assert.strictEqual(hashOf(chosen), hashOf(fastBranch.tip),
    "so the head is the shorter, heavier branch");

  // --- Byzantine peers -------------------------------------------------------
  const honestPeer = () => new FullNode("honest", built.chain, built.transactions, consensus);
  const runWithLiar = (defect, peerIds) => {
    const node = new LightNode("light", built.list[0], consensus);
    const network = new Network();
    const liar = new FullNode("liar", built.chain, built.transactions, consensus, defect);
    const peers = [liar, ...peerIds.map((id) => (id === "honest" ? honestPeer()
      : new FullNode(id, built.chain, built.transactions, consensus, defect)))];
    for (let i = 0; i < allHeaders.length; i++) {
      for (const peer of peers) {
        // one header per delivery, so a withholding peer really starves the node
        node.learnFrom(peer.id, peer.serve(hashOf(allHeaders[Math.max(0, i - 1)])).slice(0, 1));
      }
    }
    const head = node.sync(peers, network);
    return { node, head, network, liar, peers };
  };
  const forged = {
    "bumped nonce": "pow",
    "lower difficulty": "difficulty",
    "wrong height": "height",
  };
  let byzantineCases = 0;
  for (const [label, reason] of Object.entries(forged)) {
    const { node, head, liar } = runWithLiar(reason, ["honest"]);
    assert.strictEqual(hashOf(head), hashOf(built.tip), `${label}: the honest tip is still reached`);
    assert.ok(node.rejected.some((entry) => entry.reason === reason),
      `${label}: rejected with reason ${reason}`);
    assert.ok((node.badPeers.get("liar") ?? 0) > 0, `${label}: the liar is blamed`);
    assert.ok(liar.forged > 0, `${label}: the liar really forged headers`);
    for (const header of node.headers.values()) {
      if (header.height === 0) continue; // the anchor's parent is the conventional zero hash
      assert.strictEqual(validateHeader(node.knownView(), header, consensus), null,
        `${label}: every adopted header is valid under the full view`);
    }
    byzantineCases++;
  }
  // A forged orphan with valid proof of work must wait for its parent and never be adopted.
  const orphanRun = runWithLiar("link", ["honest"]);
  assert.strictEqual(hashOf(orphanRun.head), hashOf(built.tip),
    "an orphaned forgery cannot move the head");
  assert.ok(orphanRun.node.orphans.size > 0, "the forgery is parked as an orphan");
  assert.ok(orphanRun.node.rejected.every((entry) => entry.reason !== "orphan"),
    "waiting for a parent is never recorded as a lie");
  assert.strictEqual(orphanRun.node.badPeers.get("liar") ?? 0, 0,
    "orphans are legitimate gossip, so nobody is blamed");
  // A withholding peer cannot stop progress while an honest peer exists, but it can block a peer set
  // of one: that is the real liveness statement for a headers-only node.
  const withholdCovered = runWithLiar("withhold", ["honest"]);
  assert.strictEqual(hashOf(withholdCovered.head), hashOf(built.tip),
    "a withholding peer does not stop a node with another peer");
  const withholdAlone = runWithLiar("withhold", []);
  assert.notStrictEqual(hashOf(withholdAlone.head), hashOf(built.tip),
    "a node whose only peer withholds cannot reach the tip");
  assert.ok(withholdAlone.node.headers.size < built.list.length / 2,
    `and it is left with only ${withholdAlone.node.headers.size} of ${built.list.length} headers`);
  byzantineCases += 2;
  assert.strictEqual(byzantineCases, 5, "all five Byzantine shapes were exercised");

  // --- precise validation reasons -------------------------------------------
  const view = new LightNode("light", built.list[0], consensus);
  const target = built.list[1]; // its parent is the anchor, so only the rule under test can fail
  assert.strictEqual(view.adopt({ ...target, nonce: target.nonce + 1 }, 7), "pow",
    "a broken proof of work is reported as such");
  const orphanCandidate = mineFields({ ...target, parent: "f".repeat(64) }, 4242);
  assert.strictEqual(powOk(orphanCandidate), true, "the orphan candidate has valid proof of work");
  assert.strictEqual(view.adopt(orphanCandidate, 7), "orphan",
    "an unknown parent parks the header as an orphan");
  const wrongHeight = mineFields({ ...target, height: target.height + 2 }, 77);
  assert.strictEqual(view.adopt(wrongHeight, 7), "height",
    "a re-mined header with a wrong height is condemned");
  const wrongTime = mineFields({ ...target, time: target.time - 100 }, 78);
  assert.strictEqual(view.adopt(wrongTime, 7), "time", "an older timestamp is condemned");
  assert.strictEqual(view.adopt({ height: 1 }, 7).startsWith("missing-"), true,
    "malformed headers are diagnosed");
  assert.ok(view.badPeers.get(7) >= 3, "the peer that served the lies is blamed");
  assert.strictEqual(view.adopt(built.list[0], 7), null, "an already known header is ignored");
  assert.ok(view.rejected.every((entry) => entry.peerId === 7), "rejections carry their peer");

  // --- liveness and cost ----------------------------------------------------
  const costNode = new LightNode("light", built.list[0], consensus);
  const costNetwork = new Network();
  const costPeers = [honestPeer()];
  for (const header of allHeaders) costNode.learnFrom("p0", [header]);
  const costHead = costNode.sync(costPeers, costNetwork);
  assert.strictEqual(hashOf(costHead), hashOf(built.tip), "sync reaches the tip");
  assert.ok(costNode.rounds <= 3, `a complete view needs ${costNode.rounds} rounds`);
  assert.ok(costNetwork.sent <= allHeaders.length + 4, `messages stay bounded (${costNetwork.sent})`);
  assert.strictEqual(costNode.rejected.length, 0, "no spurious rejections");
  assert.strictEqual(costNode.headers.size, built.list.length, "every honest header was adopted");

  // --- determinism ----------------------------------------------------------
  const replay = buildHonestChain({ blocks: 24, seed: 842 });
  assert.strictEqual(hashOf(replay.tip), hashOf(built.tip), "the same seed mines the same tip");
  assert.deepStrictEqual(replay.list.map(hashOf), built.list.map(hashOf), "every header matches");
  const firstOrder = orders["in order"]();
  const replayNode = new LightNode("light", built.list[0], consensus);
  const replayNetwork = new Network();
  for (const batch of firstOrder) replayNode.learnFrom("p0", batch);
  replayNode.sync([honestPeer()], replayNetwork);
  assert.strictEqual(hashOf(replayNode.head), hashOf(built.tip), "replay stitches to the same tip");
  assert.deepStrictEqual(replayNode.rejected, [], "replay rejects nothing");

  console.log(`blocks=${built.list.length} retargets=${retargets} difficulties=${difficulties.size} ` +
    `spv-proofs=${proofsChecked}`);
  console.log(`delivery orders stitched= ${stitched} byzantine shapes survived= ${byzantineCases} ` +
    `fork work: fast=${workOf(fastBranch.list)} slow=${workOf(slowBranch.list)}`);
  console.log(`gossip cost: rounds<=${costNode.rounds} messages=${costNetwork.sent} ` +
    `tip work=${workOf(built.list)}`);
  console.log("PASS");
}
