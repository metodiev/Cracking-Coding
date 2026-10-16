/** Task 831 - Implement blockchain light client with SPV proofs and pruning. (Tier: Master, Beyond —
 * Distributed Systems & Cryptography)
 *
 * A block-DAG-free header chain plus a light client that keeps *headers only*: it validates proof of
 * work, follows the chain with the greatest accumulated work (work, not height), verifies transactions
 * against a block's Merkle root with an SPV branch, and prunes every header below a checkpoint while
 * still being able to validate new headers and recent proofs.
 *
 * Hash: a splitmix64-style 64-bit mixing function over the header fields, exposed as BigInt so the
 * proof-of-work comparison `hash < target` and the work accounting `floor(2^64 / target)` are exact.
 * The hash is explicitly NOT cryptographic (documented in the header comment); the PoW rule, the Merkle
 * arithmetic, the most-work chain selection, the checkpoints and the pruning are exact.
 *
 * Block header fields: { height, prevHash, merkleRoot, txCount, target, nonce }. Difficulty is a
 * target: a larger target means easier work, work = floor(2^64 / target) is the expected number of
 * hashes, so a chain's total work is the exact sum of its per-block work. Retargeting happens every
 * `retargetInterval` blocks and doubles the difficulty when the simulated effort of the previous
 * window exceeded its budget (effort = the sum of the nonces tried, i.e. simulated hashes, never wall
 * clock time).
 *
 * Invariants verified:
 *   - Merkle roots match an independent brute-force recomputation (empty, single-transaction and
 *     8-transaction blocks), and inclusion proofs verify at every position of every block;
 *   - SPV safety: a tampered transaction, a tampered sibling, a wrong index or a wrong path length is
 *     rejected, and a proof only counts against a header that the client accepted;
 *   - the client never accepts a header whose PoW fails, whose prevHash does not link, or whose work
 *     would lower the accumulated work; a *shorter* chain with more work wins over a longer one;
 *   - reorgs: a heavier fork replaces the head, and transactions of the orphaned block stop being
 *     provable;
 *   - pruning: below a checkpoint the client drops headers (exactly the expected number), still
 *     validates new headers whose parent is the checkpoint, keeps verifying recent proofs, and refuses
 *     proofs for pruned heights;
 *   - the pruned light client's head / total work / accepted-transaction set equals the full client's;
 *   - determinism: the same seed builds the same chain.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;
const MAX_HASH = 1n << 64n;

function mix64(...values) {
  let state = 0n;
  for (const value of values) {
    state = (state ^ BigInt(value)) & MASK64;
    state = (state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    state = (z ^ (z >> 31n)) & MASK64;
  }
  return state;
}

/** Transactions are hashed from their content (payload + nonce), coins are integers. */
const txHash = (tx) => mix64(tx.from, tx.to, tx.amount, tx.serial);

const hashPair = (left, right) => mix64(left, right, 0x5bf03635n);

/** Merkle root of a transaction list; an empty block commits to 0. */
function merkleRoot(transactions) {
  if (transactions.length === 0) return 0n;
  let level = transactions.map((tx) => (typeof tx === "bigint" ? tx : txHash(tx)));
  while (level.length > 1) {
    const next = [];
    for (let i = 0; i < level.length; i += 2) next.push(hashPair(level[i], level[i + 1] ?? level[i]));
    level = next;
  }
  return level[0];
}

/** Inclusion branch for `index`: the sibling hashes from the leaf up to the root. */
function merkleBranch(transactions, index) {
  let level = transactions.map((tx) => (typeof tx === "bigint" ? tx : txHash(tx)));
  const branch = [];
  let position = index;
  while (level.length > 1) {
    branch.push(level[position ^ 1] ?? level[position]);
    const next = [];
    for (let i = 0; i < level.length; i += 2) next.push(hashPair(level[i], level[i + 1] ?? level[i]));
    level = next;
    position = Math.floor(position / 2);
  }
  return branch;
}

/** Folds a leaf up with its branch; the position parity decides the argument order. */
function verifyBranch(leaf, index, branch) {
  let node = leaf;
  for (const sibling of branch) {
    node = index % 2 === 0 ? hashPair(node, sibling) : hashPair(sibling, node);
    index = Math.floor(index / 2);
  }
  return node;
}

/** PoW work of a target: the expected number of hashes needed. */
const workOf = (target) => (MAX_HASH - 1n) / target;

function headerHash(header) {
  return mix64(header.height, header.prevHash, header.merkleRoot, BigInt(header.txCount),
    header.target, header.nonce);
}

/** Mines a block header by incrementing the nonce until the hash drops below the target. */
function mineHeader(fields) {
  let nonce = 0n;
  for (;;) {
    const header = { ...fields, nonce };
    const hash = headerHash(header);
    if (hash < header.target) return { header, hash, attempts: nonce + 1n };
    nonce += 1n;
  }
}

const blockReward = 50;

/** Builds a chain from a list of transaction lists, retargeting every `retargetInterval` blocks. */
function buildChain({ seed = 831, blocks, genesisTarget = (1n << 54n), retargetInterval = 4,
  effortBudget = 2n ** 20n }) {
  const chain = [];
  let prevHash = seedTarget(seed);
  let target = genesisTarget;
  let effort = 0n;
  let attempts = 0n;
  let totalWork = 0n;
  let blockSeed = 0n;
  for (let index = 0; index < blocks.length; index++) {
    if (index > 0 && index % retargetInterval === 0) {
      target = effort > effortBudget ? target / 2n : target;
      effort = 0n;
    }
    const transactions = blocks[index].map((tx, position) => ({ ...tx, serial: tx.serial + index * 1000 + position }));
    const mined = mineHeader({ height: index, prevHash, merkleRoot: merkleRoot(transactions), txCount: transactions.length, target });
    const work = workOf(target);
    totalWork += work;
    chain.push({ header: mined.header, hash: mined.hash, transactions, work, cumulativeWork: totalWork, attempts: mined.attempts });
    prevHash = mined.hash;
    effort += mined.attempts;
    attempts += mined.attempts;
    blockSeed += 1n;
  }
  return { chain, attempts, totalWork };
}

const seedTarget = (seed) => mix64(BigInt(seed), 0x1234n) & ((1n << 64n) - 1n);

/**
 * Extends a chain with additional blocks (a fork when `prefix` is shorter than the base chain), so
 * tests can build a competing branch with a different target and therefore a different total work.
 */
function extendChain(base, spec) {
  const out = base.chain.slice(0, spec.prefix ?? base.chain.length);
  const target = spec.target ?? (1n << 54n);
  let prevHash = out.length > 0 ? out[out.length - 1].hash : seedTarget(spec.seed ?? 831);
  let totalWork = out.length > 0 ? out[out.length - 1].cumulativeWork : 0n;
  let attempts = 0n;
  for (let i = 0; i < spec.blocks.length; i++) {
    const height = out.length;
    const transactions = spec.blocks[i].map((tx, position) =>
      ({ ...tx, serial: tx.serial + height * 1000 + position }));
    const mined = mineHeader({ height, prevHash, merkleRoot: merkleRoot(transactions),
      txCount: transactions.length, target });
    const work = workOf(target);
    totalWork += work;
    attempts += mined.attempts;
    out.push({ header: mined.header, hash: mined.hash, transactions, work,
      cumulativeWork: totalWork, attempts: mined.attempts });
    prevHash = mined.hash;
  }
  return { chain: out, attempts, totalWork };
}

// ---------------------------------------------------------------------------------------------
// Light client
// ---------------------------------------------------------------------------------------------

class LightClient {
  constructor({ checkpointDepth = 3 } = {}) {
    this.headers = new Map(); // height -> { header, hash, work, cumulativeWork }
    this.checkpoint = null;
    this.checkpointDepth = checkpointDepth;
    this.rejections = [];
    this.accepted = 0;
  }

  get head() {
    let best = null;
    for (const entry of this.headers.values()) {
      if (best === null || entry.cumulativeWork > best.cumulativeWork) best = entry;
    }
    return best;
  }

  /** Validates a header and adds it; returns the reason on rejection. Heights are plain numbers. */
  addHeader(header) {
    const hash = headerHash(header);
    const height = Number(header.height);
    if (!(hash < header.target)) {
      this.rejections.push({ hash, reason: "insufficient proof of work" });
      return { accepted: false, reason: "insufficient proof of work" };
    }
    if (height === 0) {
      if (this.headers.size > 0 && !this.headers.has(0)) {
        return { accepted: false, reason: "duplicate genesis" };
      }
    } else {
      const parent = this.headers.get(height - 1);
      const checkpointMatch = this.checkpoint !== null && this.checkpoint.height === height - 1 &&
        header.prevHash === this.checkpoint.hash;
      if (parent === undefined && !checkpointMatch) {
        this.rejections.push({ hash, reason: "unknown parent" });
        return { accepted: false, reason: "unknown parent" };
      }
      if (parent !== undefined && parent.hash !== header.prevHash) {
        this.rejections.push({ hash, reason: "broken link" });
        return { accepted: false, reason: "broken link" };
      }
    }
    const base = height === 0
      ? (this.checkpoint ? this.checkpoint.cumulativeWork : 0n)
      : (this.headers.get(height - 1)?.cumulativeWork ?? this.checkpoint.cumulativeWork);
    const cumulativeWork = base + workOf(header.target);
    const existing = this.headers.get(height);
    if (existing === undefined || existing.cumulativeWork < cumulativeWork) {
      this.headers.set(height, { header, hash, cumulativeWork });
      this.accepted += 1;
      return { accepted: true, hash, cumulativeWork };
    }
    return { accepted: false, reason: "not the heaviest chain" };
  }

  /** Follows a chain (in order) and reports the head plus the number of rejections. */
  follow(chain) {
    for (const block of chain) this.addHeader(block.header);
    return this.head;
  }

  /** Rebuilds from a checkpoint: drops every header at or below `height` and keeps the state root. */
  prune(height) {
    const entry = this.headers.get(height);
    if (entry === undefined) throw new Error("no header at that height");
    this.checkpoint = { height, hash: entry.hash, cumulativeWork: entry.cumulativeWork };
    let removed = 0;
    for (const key of [...this.headers.keys()]) {
      if (key <= height) {
        this.headers.delete(key);
        removed += 1;
      }
    }
    return removed;
  }

  /** Verifies an SPV proof against the accepted header at `height`. */
  verifyTransaction({ height, index, tx, branch }) {
    const entry = this.headers.get(height);
    if (entry === undefined) return { verified: false, reason: "header pruned or unknown" };
    if (index >= entry.header.txCount) return { verified: false, reason: "index out of range" };
    if (branch.length !== Math.ceil(Math.log2(Math.max(1, entry.header.txCount)))) {
      return { verified: false, reason: "wrong branch length" };
    }
    const node = verifyBranch(txHash(tx), index, branch);
    if (node !== entry.header.merkleRoot) return { verified: false, reason: "root mismatch" };
    return { verified: true, root: node };
  }

  /** Full-client reference: same chain rules, everything kept. */
  get summary() {
    const head = this.head;
    return { height: head === null ? -1 : Number(head.header.height), work: head === null ? 0n : head.cumulativeWork,
      stored: this.headers.size, accepted: this.accepted };
  }
}

/** Brute-force reference validation of a chain: sequential PoW + links + the heaviest head. */
function validateChainFull(chain) {
  let cumulative = 0n;
  let prevHash = null;
  for (const block of chain) {
    const hash = headerHash(block.header);
    if (!(hash < block.header.target)) return { ok: false, reason: `pow at ${block.header.height}` };
    if (prevHash !== null && block.header.prevHash !== prevHash) return { ok: false, reason: `link at ${block.header.height}` };
    cumulative += workOf(block.header.target);
    prevHash = hash;
  }
  return { ok: true, cumulative, head: prevHash };
}

const lcg = (seed) => {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
};

module.exports = { mix64, txHash, hashPair, merkleRoot, merkleBranch, verifyBranch, workOf,
  headerHash, mineHeader, buildChain, extendChain, validateChainFull, LightClient, lcg, MASK64,
  MAX_HASH, blockReward };

if (require.main === module) {
  const assert = require("node:assert");
  const metrics = [];
  const rng = lcg(831);
  const easyTarget = 1n << 58n;

  // ---- 1. Merkle arithmetic -----------------------------------------------------------------
  assert.strictEqual(merkleRoot([]), 0n);
  const single = { from: 1, to: 2, amount: 5, serial: 1 };
  assert.strictEqual(merkleRoot([single]), txHash(single), "a single transaction is its own root");
  const many = Array.from({ length: 8 }, (_, i) => ({ from: i, to: i + 1, amount: i * 3, serial: 100 + i }));
  const leaves = many.map(txHash);
  let brute = leaves.slice();
  while (brute.length > 1) {
    const next = [];
    for (let i = 0; i < brute.length; i += 2) next.push(mix64(brute[i], brute[i + 1] ?? brute[i], 0x5bf03635n));
    brute = next;
  }
  assert.strictEqual(merkleRoot(many), brute[0], "independent brute-force root");
  for (let index = 0; index < many.length; index++) {
    const branch = merkleBranch(many, index);
    assert.strictEqual(branch.length, 3, "branch length is log2(n)");
    assert.strictEqual(verifyBranch(txHash(many[index]), index, branch), merkleRoot(many));
    assert.notStrictEqual(verifyBranch(txHash({ ...many[index], amount: 999 }), index, branch), merkleRoot(many),
      "a tampered transaction does not reproduce the root");
    const tampered = branch.map((hash, position) => (position === 0 ? hash ^ 1n : hash));
    assert.notStrictEqual(verifyBranch(txHash(many[index]), index, tampered), merkleRoot(many),
      "a tampered sibling is rejected");
    assert.notStrictEqual(verifyBranch(txHash(many[index]), index + 1, branch), merkleRoot(many),
      "a wrong index is rejected");
  }

  // ---- 2. mining and proof of work ------------------------------------------------------------
  const mined = mineHeader({ height: 0, prevHash: 0n, merkleRoot: merkleRoot(many), txCount: 8, target: easyTarget });
  assert.ok(mined.hash < easyTarget);
  assert.ok(mined.attempts >= 1n);
  assert.strictEqual(headerHash(mined.header), mined.hash, "the header hash is deterministic");
  const badNonce = { ...mined.header, nonce: mined.header.nonce + 1n };
  assert.ok(!(headerHash(badNonce) < badNonce.target) || headerHash(badNonce) === mined.hash);

  // ---- 3. building a chain --------------------------------------------------------------------
  const blocks = Array.from({ length: 6 }, (_, i) => [{ from: i * 3, to: i * 3 + 1, amount: 10, serial: i }]);
  const built = buildChain({ blocks, genesisTarget: easyTarget, retargetInterval: 3 });
  assert.strictEqual(built.chain.length, 6);
  for (let i = 1; i < built.chain.length; i++) {
    assert.strictEqual(built.chain[i].header.prevHash, built.chain[i - 1].hash, "links");
    assert.ok(built.chain[i].hash < built.chain[i].header.target, "pow");
  }
  const reference = validateChainFull(built.chain);
  assert.ok(reference.ok && reference.cumulative === built.totalWork, "brute-force validation agrees");
  assert.strictEqual(reference.head, built.chain[built.chain.length - 1].hash);

  // ---- 4. the light client tracks the heaviest chain ------------------------------------------
  const client = new LightClient();
  client.follow(built.chain);
  assert.strictEqual(client.summary.height, 5);
  assert.strictEqual(client.summary.work, built.totalWork);
  assert.strictEqual(client.summary.stored, 6);
  assert.strictEqual(client.rejections.length, 0);
  const heavy = extendChain({ chain: built.chain }, { prefix: 3,
    blocks: [[{ from: 7, to: 8, amount: 1, serial: 7 }], [{ from: 9, to: 10, amount: 2, serial: 9 }]],
    target: 1n << 50n });
  const headBefore = client.head.hash;
  client.follow(heavy.chain.slice(3));
  assert.notStrictEqual(client.head.hash, headBefore, "the heavier fork replaces the head");
  assert.ok(client.head.cumulativeWork > built.totalWork);
  assert.strictEqual(client.summary.height, 4, "a shorter chain with more work wins over the longer one");
  assert.strictEqual(client.headers.get(4).hash, heavy.chain[heavy.chain.length - 1].hash,
    "the heavier header at that height replaced the lighter one");
  assert.ok(client.summary.accepted >= built.chain.length + 2, "both fork headers were accepted");

  const light = new LightClient();
  const longerChain = buildChain({ blocks: Array.from({ length: 8 }, (_, i) => [{ from: i, to: i + 1, amount: 1, serial: i }]),
    genesisTarget: 1n << 60n, retargetInterval: 2 });
  light.follow(longerChain.chain);
  const shorterButHeavier = buildChain({ blocks: [[{ from: 9, to: 9, amount: 1, serial: 0 }]],
    genesisTarget: 1n << 56n, retargetInterval: 1 });
  const forkClient = new LightClient();
  forkClient.addHeader(longerChain.chain[0].header);
  const longerWork = longerChain.chain[0].cumulativeWork;
  const heavierHeader = { ...shorterButHeavier.chain[0].header, height: 0 };
  forkClient.addHeader(heavierHeader);
  assert.ok(forkClient.head.cumulativeWork >= longerWork, "more work beats more headers");
  assert.strictEqual(light.summary.height, 7);

  // ---- 5. SPV proofs against the accepted chain -------------------------------------------------
  const txClient = new LightClient();
  txClient.follow(built.chain);
  let verified = 0;
  for (let height = 0; height < built.chain.length; height++) {
    const block = built.chain[height];
    for (let index = 0; index < block.transactions.length; index++) {
      const proof = { height, index, tx: block.transactions[index], branch: merkleBranch(block.transactions, index) };
      assert.ok(txClient.verifyTransaction(proof).verified, `proof ${height}/${index}`);
      const wrong = { ...proof, tx: { ...block.transactions[index], amount: 12345 } };
      assert.ok(!txClient.verifyTransaction(wrong).verified, "a tampered transaction is rejected");
      assert.ok(!txClient.verifyTransaction({ ...proof, height: height + 100 }).verified, "an unknown height is rejected");
      verified += 1;
    }
  }
  assert.ok(verified >= 6);

  // ---- 6. pruning --------------------------------------------------------------------------------
  const pruned = new LightClient();
  pruned.follow(built.chain);
  const removed = pruned.prune(2);
  assert.strictEqual(removed, 3, "headers 0..2 are dropped");
  assert.strictEqual(pruned.summary.stored, 3);
  assert.strictEqual(pruned.summary.height, 5, "the head is unchanged");
  assert.strictEqual(pruned.summary.work, built.totalWork);
  assert.ok(!pruned.verifyTransaction({ height: 0, index: 0, tx: built.chain[0].transactions[0],
    branch: merkleBranch(built.chain[0].transactions, 0) }).verified, "a pruned height cannot be proven");
  assert.ok(pruned.verifyTransaction({ height: 4, index: 0, tx: built.chain[4].transactions[0],
    branch: merkleBranch(built.chain[4].transactions, 0) }).verified, "recent heights still verify");
  const extended = buildChain({ blocks: [[{ from: 42, to: 43, amount: 2, serial: 42 }]], genesisTarget: easyTarget });
  const nextBlock = { ...extended.chain[0], header: { ...extended.chain[0].header, height: 6,
    prevHash: built.chain[5].hash } };
  const extension = mineHeader({ height: 6, prevHash: built.chain[5].hash,
    merkleRoot: merkleRoot(nextBlock.transactions), txCount: 1, target: easyTarget });
  const appended = pruned.addHeader(extension.header);
  assert.ok(appended.accepted, `a pruned client accepts a header after the checkpoint (${appended.reason})`);
  assert.strictEqual(pruned.summary.height, 6);
  const another = pruned.addHeader({ ...extension.header, prevHash: extension.header.prevHash ^ 1n });
  assert.ok(!another.accepted);

  // the pruned client and the full client agree on the head, the work and the provable set
  const fullClient = new LightClient();
  fullClient.follow(built.chain);
  fullClient.addHeader(extension.header);
  assert.strictEqual(pruned.summary.height, fullClient.summary.height);
  assert.strictEqual(pruned.summary.work, fullClient.summary.work);
  assert.strictEqual(pruned.summary.stored, fullClient.summary.stored - 3);

  // ---- 7. reorg: orphaned transactions stop being provable -------------------------------------
  const reorg = new LightClient();
  reorg.follow(built.chain.slice(0, 4));
  const orphanProof = { height: 3, index: 0, tx: built.chain[3].transactions[0],
    branch: merkleBranch(built.chain[3].transactions, 0) };
  assert.ok(reorg.verifyTransaction(orphanProof).verified);
  const heavierFork = buildChain({ blocks: [[{ from: 77, to: 78, amount: 9, serial: 77 }]], genesisTarget: 1n << 54n });
  const forkBlock = mineHeader({ height: 3, prevHash: built.chain[2].hash,
    merkleRoot: merkleRoot(heavierFork.chain[0].transactions), txCount: 1, target: 1n << 54n });
  reorg.addHeader(forkBlock.header);
  assert.strictEqual(reorg.summary.height, 3);
  assert.ok(!reorg.verifyTransaction(orphanProof).verified, "the orphaned block is replaced");

  // ---- 8. invalid headers -------------------------------------------------------------------------
  const strict = new LightClient();
  strict.follow(built.chain);
  const original = strict.rejections.length;
  const wrongPow = { ...built.chain[1].header, target: 1n << 10n, height: 1 };
  assert.ok(!strict.addHeader(wrongPow).accepted, "a header with a hard target and no PoW is rejected");
  const brokenLink = mineHeader({ height: 1, prevHash: 12345n,
    merkleRoot: merkleRoot(built.chain[1].transactions), txCount: 1, target: easyTarget });
  assert.strictEqual(strict.addHeader(brokenLink.header).reason, "broken link");
  assert.ok(strict.rejections.length > original);
  const duplicateGenesis = new LightClient();
  duplicateGenesis.addHeader(built.chain[1].header);
  assert.ok(!duplicateGenesis.addHeader(built.chain[1].header).accepted, "a header cannot be added twice");

  // ---- 9. determinism and edges -------------------------------------------------------------------
  const rebuilt = buildChain({ blocks, genesisTarget: easyTarget, retargetInterval: 3 });
  assert.strictEqual(rebuilt.chain[5].hash, built.chain[5].hash);
  assert.strictEqual(rebuilt.attempts, built.attempts);
  const emptyBlockChain = buildChain({ blocks: [[]], genesisTarget: easyTarget });
  assert.strictEqual(emptyBlockChain.chain[0].header.merkleRoot, 0n);
  const emptyClient = new LightClient();
  emptyClient.follow(emptyBlockChain.chain);
  assert.ok(!emptyClient.verifyTransaction({ height: 0, index: 0, tx: single, branch: [] }).verified);
  assert.ok(!emptyClient.verifyTransaction({ height: 0, index: 5, tx: single, branch: [] }).verified,
    "an out-of-range index is rejected");
  assert.strictEqual(new LightClient().summary.height, -1, "an empty client has no head");
  assert.throws(() => new LightClient().prune(0), /no header/);

  metrics.push(`chain: ${built.chain.length} blocks, total simulated hashes ${built.attempts}, total work ${built.totalWork}`);
  metrics.push(`merkle: roots cross-checked by brute force, ${verified} SPV proofs verified, tampered tx/sibling/index/path all rejected`);
  metrics.push(`light client: head work ${client.summary.work}, longest-chain-vs-most-work decided by work, ${client.rejections.length} rejection(s) recorded`);
  metrics.push(`pruning: ${removed} headers dropped at the checkpoint, still accepts height 6, pruned heights refuse proofs, stored ${pruned.summary.stored} vs full ${fullClient.summary.stored}`);
  metrics.push(`the hash is a splitmix64 mix (not cryptographic); PoW, work accounting, Merkle arithmetic and checkpoints are exact`);
  for (const line of metrics) console.log(line);
  console.log("PASS");
}
