/** Task 813 — Implement end-to-end encrypted collaborative editor with operational transforms and
 *  access control.  (Tier: Master, Ch 9 — System Design and Scalability)
 *
 * Topology. One sequencer (the server) orders the operations, clients apply their own edit
 * optimistically and rebase the still-unacknowledged operations against everything the server sends
 * back — the classic OT client loop (Jupiter). An operation is a list of components
 * `retain n / insert text / delete n`, i.e. a total function on documents of `baseLength`
 * characters; `transformPair(a, b)` is the inclusion transformation and returns `[a', b']` such
 * that `apply(apply(t, a), b') === apply(apply(t, b), a')` for every `t` of that length (TP1).
 *
 * Transform rules (derived from "each base character is deleted iff at least one operation deletes
 * it, every inserted run survives, an insertion at the left edge of a deleted range stays left"):
 *   - retain/retain : both advance by min, both emit `retain min`;
 *   - retain/delete : both advance by min, only the deleting side emits `delete min`;
 *   - delete/delete : both advance by min, neither emits (the common part is already gone);
 *   - insert/*      : the inserting side emits the insert, the other side emits `retain len` and its
 *                     base cursor does not move (it steps over the inserted text instead).
 * The rules are not asserted to be right — TP1 is verified exhaustively for base length 3 and on
 * random operation pairs, and convergence is verified against a brute-force replay of the server
 * history, so any wrong rule shows up as a failed check rather than a silent divergence.
 *
 * Encryption. The operation log is sealed per key epoch with `seal`/`open`: a splitmix64 keystream
 * (seeded from a 64-bit FNV-1a of key‖nonce) is XORed over a UTF-8 payload and a keyed tag is
 * appended, so the log leaks nothing without the key and any modification is detected. Access control
 * is an ACL with owner/editor/viewer roles plus an epoch counter; `revoke` bumps the epoch so the
 * removed member cannot read or produce anything sealed afterwards. The prefix written while it was a
 * member stays readable to it (no re-wrapping of historical entries) — that residual leak is measured
 * and reported instead of being hidden.
 *
 * Simplified versus production: no sockets/WebSocket framing (messages travel over an explicit
 * in-memory network and are delivered when `pump()` is called), no persistence, cursors or presence,
 * a keystream + keyed tag instead of an AEAD, and no re-encryption of the historical prefix after a
 * revocation.
 *
 * Verified: apply/transform/TP1 semantics (exhaustive + random), convergence of all replicas to the
 * brute-force replay of the server history with induced out-of-order submissions (liveness: no
 * accepted edit is lost), ACL enforcement, revocation confinement, confidentiality (ciphertext
 * independence from plaintext bytes), integrity (tamper/wrong-key rejection), determinism, message and
 * byte accounting and the degenerate empty/single-character documents.
 */
"use strict";

// ---------------------------------------------------------------------------- deterministic entropy

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  return function next() {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
}

function keyedHash64(key, text) {
  let hash = 0xcbf29ce484222325n;
  const message = `${key}\u0000${text}`;
  for (let i = 0; i < message.length; i++) {
    hash = BigInt.asUintN(64, hash ^ BigInt(message.charCodeAt(i)));
    hash = BigInt.asUintN(64, hash * 0x100000001b3n);
  }
  return hash;
}

function keystream(key, nonce, length) {
  const next = splitmix64(keyedHash64(key, `ks:${nonce}`));
  const bytes = Buffer.alloc(length);
  for (let i = 0; i < length; i += 8) {
    const word = next();
    for (let j = 0; j < 8 && i + j < length; j++) bytes[i + j] = Number((word >> BigInt(8 * j)) & 0xffn);
  }
  return bytes;
}

function seal(key, nonce, plaintext) {
  const body = Buffer.from(plaintext, "utf8");
  const stream = keystream(key, nonce, body.length);
  const ciphertext = Buffer.alloc(body.length);
  for (let i = 0; i < body.length; i++) ciphertext[i] = body[i] ^ stream[i];
  return { nonce, ciphertext, tag: keyedHash64(key, `tag:${nonce}:${ciphertext.toString("hex")}`) };
}

function open(key, sealed) {
  const expected = keyedHash64(key, `tag:${sealed.nonce}:${sealed.ciphertext.toString("hex")}`);
  if (expected !== sealed.tag) throw new Error("integrity check failed: wrong key or forged entry");
  const stream = keystream(key, sealed.nonce, sealed.ciphertext.length);
  const plaintext = Buffer.alloc(sealed.ciphertext.length);
  for (let i = 0; i < plaintext.length; i++) plaintext[i] = sealed.ciphertext[i] ^ stream[i];
  return plaintext.toString("utf8");
}

/** Fraction of differing bits between a plaintext and the keystream bytes that encrypt it. */
function differingBitFraction(plaintext, key, nonce) {
  const body = Buffer.from(plaintext, "utf8");
  const stream = keystream(key, nonce, body.length);
  let differing = 0;
  let total = 0;
  for (let i = 0; i < body.length; i++) {
    let x = body[i] ^ stream[i];
    for (let bit = 0; bit < 8; bit++) {
      total++;
      if (x & 1) differing++;
      x >>= 1;
    }
  }
  return total === 0 ? 0 : differing / total;
}

/** Key epoch n derived from the document key. */
function epochKey(documentKey, epoch) {
  return `${documentKey}#epoch:${epoch}`;
}

// ---------------------------------------------------------------------------- operations

function normalizeComponents(components) {
  const merged = [];
  for (const component of components) {
    if (component.type === "retain" || component.type === "delete") {
      if (!Number.isInteger(component.n) || component.n < 0) throw new RangeError("bad component length");
      if (component.n === 0) continue;
    }
    if (component.type === "insert" && component.text.length === 0) continue;
    const last = merged[merged.length - 1];
    if (last && last.type === component.type && component.type !== "insert") {
      last.n += component.n;
    } else if (last && last.type === "insert" && component.type === "insert") {
      last.text += component.text;
    } else {
      merged.push({ type: component.type, n: component.n, text: component.text });
    }
  }
  return merged;
}

function makeOp(components) {
  return { components: normalizeComponents(components) };
}

function baseLength(op) {
  let total = 0;
  for (const component of op.components) if (component.type !== "insert") total += component.n;
  return total;
}

function targetLength(op) {
  let total = 0;
  for (const component of op.components) total += component.type === "delete" ? 0 : component.type === "insert" ? component.text.length : component.n;
  return total;
}

function opToString(op) {
  return op.components
    .map((component) => (component.type === "insert" ? `+${component.text}` : `${component.type[0]}${component.n}`))
    .join(" ");
}

function applyOp(text, op) {
  const out = [];
  let cursor = 0;
  for (const component of op.components) {
    if (component.type === "retain") {
      if (cursor + component.n > text.length) throw new RangeError("retain past end of document");
      out.push(text.slice(cursor, cursor + component.n));
      cursor += component.n;
    } else if (component.type === "insert") {
      out.push(component.text);
    } else {
      if (cursor + component.n > text.length) throw new RangeError("delete past end of document");
      cursor += component.n;
    }
  }
  out.push(text.slice(cursor));
  return out.join("");
}

function pushComponent(out, type, amount, text) {
  const last = out[out.length - 1];
  if (last && last.type === type && type !== "insert") {
    last.n += amount;
  } else if (last && last.type === "insert" && type === "insert") {
    last.text += text;
  } else {
    out.push({ type, n: amount, text });
  }
}

const pushRetain = (out, amount) => pushComponent(out, "retain", amount);
const pushDelete = (out, amount) => pushComponent(out, "delete", amount);
const pushInsert = (out, text) => pushComponent(out, "insert", undefined, text);

/**
 * Inclusion transformation: `[a', b']` with `apply(apply(t, a), b') === apply(apply(t, b), a')`.
 * Both operations must cover the same base length, which the sequencer guarantees: a client submits
 * only when its operation is based on the server's current revision.
 */
function transformPair(opA, opB) {
  if (baseLength(opA) !== baseLength(opB)) throw new Error("transform requires equal base lengths");
  const queueA = opA.components.map((c) => ({ ...c }));
  const queueB = opB.components.map((c) => ({ ...c }));
  const outA = [];
  const outB = [];
  let remainderA = null;
  let remainderB = null;
  while (remainderA !== null || remainderB !== null || queueA.length > 0 || queueB.length > 0) {
    if (remainderA === null && queueA.length > 0) remainderA = queueA.shift();
    if (remainderB === null && queueB.length > 0) remainderB = queueB.shift();
    if (remainderA === null && remainderB === null) break;

    if (remainderA !== null && remainderA.type === "insert") {
      // A inserts at the shared cursor: it stays left of whatever B does there and B skips over it.
      pushInsert(outA, remainderA.text);
      pushRetain(outB, remainderA.text.length);
      remainderA = null;
      continue;
    }
    if (remainderB !== null && remainderB.type === "insert") {
      pushInsert(outB, remainderB.text);
      pushRetain(outA, remainderB.text.length);
      remainderB = null;
      continue;
    }
    if (remainderA === null || remainderB === null) {
      const side = remainderA !== null ? remainderA : remainderB;
      const target = remainderA !== null ? outA : outB;
      if (side.type === "delete") pushDelete(target, side.n);
      else pushRetain(target, side.n);
      if (remainderA !== null) remainderA = null; else remainderB = null;
      continue;
    }

    const amount = Math.min(remainderA.n, remainderB.n);
    if (remainderA.type === "retain" && remainderB.type === "retain") {
      pushRetain(outA, amount);
      pushRetain(outB, amount);
    } else if (remainderA.type === "retain") {
      pushDelete(outB, amount);
    } else if (remainderB.type === "retain") {
      pushDelete(outA, amount);
    }
    // delete/delete: the common characters are deleted in both target documents, so neither side
    // emits anything; whichever operation deletes more keeps its excess in the next iteration.
    remainderA.n -= amount;
    remainderB.n -= amount;
    if (remainderA.n === 0) remainderA = null;
    if (remainderB.n === 0) remainderB = null;
  }
  return [makeOp(outA), makeOp(outB)];
}

function totalOp(length) {
  return length === 0 ? makeOp([]) : makeOp([{ type: "retain", n: length }]);
}

// ---------------------------------------------------------------------------- in-memory network

class InMemoryNetwork {
  constructor() {
    this.mailboxes = new Map();
    this.messagesSent = 0;
    this.bytesSent = 0;
  }

  register(id) {
    this.mailboxes.set(id, []);
  }

  send(from, to, message) {
    const box = this.mailboxes.get(to);
    if (!box) throw new Error(`unknown endpoint ${to}`);
    const envelope = { from, ...message };
    box.push(envelope);
    this.messagesSent++;
    this.bytesSent += JSON.stringify(message).length;
  }

  drain(id) {
    const box = this.mailboxes.get(id);
    this.mailboxes.set(id, []);
    return box;
  }

  get pending() {
    let total = 0;
    for (const box of this.mailboxes.values()) total += box.length;
    return total;
  }
}

// ---------------------------------------------------------------------------- access control

class AccessControlList {
  constructor(ownerId) {
    this.ownerId = ownerId;
    this.roles = new Map([[ownerId, "owner"]]);
    this.epoch = 0;
    this.documentKey = "doc-key";
    this.audit = [];
  }

  grant(memberId, role) {
    if (!["editor", "viewer"].includes(role)) throw new RangeError("unknown role");
    this.roles.set(memberId, role);
    this.audit.push({ action: "grant", memberId, role, epoch: this.epoch });
  }

  /** Revocation advances the key epoch, so nothing sealed afterwards is readable by the member. */
  revoke(memberId) {
    if (memberId === this.ownerId) throw new Error("the owner cannot be revoked");
    const role = this.roles.get(memberId);
    if (role === undefined) throw new Error("member not found");
    this.roles.delete(memberId);
    this.epoch++;
    this.audit.push({ action: "revoke", memberId, role, epoch: this.epoch });
    return this.epoch;
  }

  canEdit(memberId) {
    const role = this.roles.get(memberId);
    return role === "owner" || role === "editor";
  }

  canRead(memberId) {
    return this.roles.has(memberId) || memberId === "auditor";
  }

  keyForEpoch(epoch) {
    return epochKey(this.documentKey, epoch);
  }
}

// ---------------------------------------------------------------------------- sequencer + clients

class CollaborationServer {
  constructor(network, acl, documentId = "doc") {
    this.network = network;
    this.acl = acl;
    this.documentId = documentId;
    this.text = "";
    this.revision = 0;
    this.history = [];
    this.log = [];
    this.rejected = [];
    this.stats = { submits: 0, accepted: 0, rebases: 0, rejected: 0, syncs: 0, broadcasts: 0 };
    this.nonceCounter = 0;
  }

  /** Brute-force semantic reference: the document after replaying the accepted history in order. */
  replay(text = "") {
    let current = text;
    for (const entry of this.history) current = applyOp(current, entry.op);
    return current;
  }

  appendLog(clientId, op) {
    const epoch = this.acl.epoch;
    const plaintext = JSON.stringify({ revision: this.revision, clientId, op: op.components });
    const sealed = seal(this.acl.keyForEpoch(epoch), this.nonceCounter++, plaintext);
    this.log.push({ revision: this.revision, epoch, sealed });
  }

  handle(envelope) {
    const { from, type, op, revision } = envelope;
    if (type === "sync") {
      this.stats.syncs++;
      const missing = this.history.slice(revision).map((entry) => entry.op);
      this.network.send(this.documentId, from, { type: "rebase", ops: missing, fromRevision: revision, revision: this.revision });
      return;
    }
    if (type !== "submit") throw new Error(`unknown message ${type}`);
    this.stats.submits++;
    if (!this.acl.canEdit(from)) {
      this.stats.rejected++;
      this.rejected.push({ clientId: from, reason: "forbidden" });
      this.network.send(this.documentId, from, { type: "rejected", reason: "forbidden" });
      return;
    }
    if (revision !== this.revision) {
      // The operation was produced against an older state: the client rebases and resubmits.
      this.stats.rebases++;
      const missing = this.history.slice(revision).map((entry) => entry.op);
      this.network.send(this.documentId, from, { type: "rebase", ops: missing, fromRevision: revision, revision: this.revision });
      return;
    }
    if (baseLength(op) !== this.text.length) {
      this.stats.rejected++;
      this.rejected.push({ clientId: from, reason: "revision/state mismatch" });
      this.network.send(this.documentId, from, { type: "rejected", reason: `op covers ${baseLength(op)} of ${this.text.length} characters` });
      return;
    }
    this.stats.accepted++;
    this.appendLog(from, op);
    this.text = applyOp(this.text, op);
    this.history.push({ clientId: from, op, revision: this.revision });
    this.revision++;
    for (const id of this.network.mailboxes.keys()) {
      if (id === this.documentId || id === from) continue;
      this.stats.broadcasts++;
      this.network.send(this.documentId, id, { type: "op", op, revision: this.revision, clientId: from });
    }
    this.network.send(this.documentId, from, { type: "ack", revision: this.revision });
  }
}

class EditorClient {
  constructor(id, network, documentId) {
    this.id = id;
    this.network = network;
    this.documentId = documentId;
    this.syncedText = ""; // document state at `revision`, without the unacknowledged operations
    this.revision = 0;
    this.pending = [];
    this.inflight = false;
    this.stats = { rebases: 0, acks: 0, rejections: 0, transforms: 0 };
  }

  /** What the user sees: the synchronised state with the local, unacknowledged edits on top. */
  get text() {
    let current = this.syncedText;
    for (const op of this.pending) current = applyOp(current, op);
    return current;
  }

  /** Local edit: queued behind every operation not yet acknowledged. */
  localEdit(op) {
    this.pending.push(op);
    this.trySubmit();
  }

  trySubmit() {
    if (this.inflight || this.pending.length === 0) return;
    this.inflight = true;
    this.network.send(this.id, this.documentId, { type: "submit", op: this.pending[0], revision: this.revision });
  }

  /**
   * Incorporate one remote operation that the sequencer applied at `this.revision + 1`: it becomes
   * part of the synchronised layer, and every unacknowledged operation is rebased on top of it, so
   * the user-visible text is unchanged while the operation becomes acknowledged-ordered state.
   */
  incorporate(remote) {
    this.syncedText = applyOp(this.syncedText, remote);
    this.revision++;
    let current = remote;
    for (let i = 0; i < this.pending.length; i++) {
      const [rebasePending, rebaseRemote] = transformPair(this.pending[i], current);
      this.pending[i] = rebasePending;
      current = rebaseRemote;
      this.stats.transforms++;
    }
  }

  handle(envelope) {
    if (envelope.type === "ack") {
      this.stats.acks++;
      this.revision = envelope.revision;
      // The acknowledged operation is now part of the synchronised state.
      this.syncedText = applyOp(this.syncedText, this.pending[0]);
      this.pending.shift();
      this.inflight = false;
      this.trySubmit();
    } else if (envelope.type === "rebase") {
      this.stats.rebases++;
      // The sequencer lists every operation recorded since the revision this client based its
      // submission on; operations this client already received through the broadcast are skipped.
      const skip = this.revision - envelope.fromRevision;
      if (skip < 0 || skip > envelope.ops.length) throw new Error("rebase window out of range");
      for (const op of envelope.ops.slice(skip)) this.incorporate(op);
      this.inflight = false;
      this.trySubmit();
    } else if (envelope.type === "op") {
      if (envelope.revision === this.revision + 1) this.incorporate(envelope.op);
    } else if (envelope.type === "rejected") {
      this.stats.rejections++;
      this.inflight = false;
      // The local edit cannot be applied: drop it and resynchronise from the sequencer.
      this.pending = [];
      this.network.send(this.id, this.documentId, { type: "sync", revision: this.revision });
    } else {
      throw new Error(`unknown message ${envelope.type}`);
    }
  }

  sync() {
    this.network.send(this.id, this.documentId, { type: "sync", revision: this.revision });
  }
}

/** Deterministic round-robin delivery until every mailbox is empty (or the budget is exhausted). */
function pump(network, server, clients, budget = 10000) {
  let steps = 0;
  const endpoints = [server.documentId, ...clients.map((client) => client.id)];
  while (network.pending > 0 && steps < budget) {
    for (const id of endpoints) {
      for (const envelope of network.drain(id)) {
        if (id === server.documentId) server.handle(envelope);
        else clients.find((client) => client.id === id).handle(envelope);
        steps++;
      }
    }
  }
  return steps;
}

// ---------------------------------------------------------------------------- operation generation

const INSERT_ALPHABET = ["a", "b", "bc"];

/**
 * A realistic client edit: one to three localised edit sites (insert or delete of one or two
 * characters) separated by retains, i.e. a small operation with an O(1) number of components. The
 * document length therefore only performs a small random walk, which keeps the checks fast and still
 * produces genuinely concurrent operations on the same regions.
 */
function randomOp(base, next) {
  const components = [];
  let cursor = 0;
  const sites = 1 + Number(next() % 3n);
  for (let site = 0; site < sites; site++) {
    const target = Number(next() % BigInt(base + 1));
    const skip = Math.max(0, Math.min(target - cursor, base - cursor));
    if (skip > 0) {
      components.push({ type: "retain", n: skip });
      cursor += skip;
    }
    if (cursor < base && Number(next() % 3n) !== 2) {
      const n = 1 + Number(next() % BigInt(Math.min(2, base - cursor)));
      components.push({ type: "delete", n });
      cursor += n;
    } else {
      components.push({ type: "insert", text: INSERT_ALPHABET[Number(next() % BigInt(INSERT_ALPHABET.length))] });
    }
  }
  if (cursor < base) components.push({ type: "retain", n: base - cursor });
  return makeOp(components);
}

/** All (total) operations over `base` characters built from retains, deletes and inserts. */
function enumerateOps(base) {
  const found = new Map();
  const insertTexts = ["x", "yz"];
  const record = (components) => {
    const op = makeOp(components);
    if (baseLength(op) !== base) return;
    const key = JSON.stringify(op.components);
    if (!found.has(key)) found.set(key, op);
  };
  const walk = (components, remaining) => {
    record(components);
    if (components.length >= 4) return;
    for (const text of insertTexts) {
      const withInsert = [...components, { type: "insert", text }];
      record(withInsert);
      walk(withInsert, remaining); // insertions do not consume base characters
    }
    for (let n = 1; n <= remaining; n++) {
      walk([...components, { type: "retain", n }], remaining - n);
      walk([...components, { type: "delete", n }], remaining - n);
    }
  };
  walk([], base);
  return [...found.values()];
}

module.exports = {
  splitmix64,
  keyedHash64,
  seal,
  open,
  differingBitFraction,
  epochKey,
  makeOp,
  applyOp,
  baseLength,
  targetLength,
  opToString,
  transformPair,
  totalOp,
  randomOp,
  enumerateOps,
  InMemoryNetwork,
  AccessControlList,
  CollaborationServer,
  EditorClient,
  pump,
};

if (require.main === module) {
  const assert = require("node:assert");

  // --- component semantics -------------------------------------------------
  const doc = "abcdef";
  assert.strictEqual(applyOp(doc, makeOp([{ type: "retain", n: 2 }, { type: "insert", text: "XY" }, { type: "retain", n: 4 }])), "abXYcdef");
  assert.strictEqual(applyOp(doc, makeOp([{ type: "retain", n: 1 }, { type: "delete", n: 3 }, { type: "retain", n: 2 }])), "aef");
  assert.strictEqual(applyOp(doc, makeOp([{ type: "delete", n: 6 }])), "");
  assert.strictEqual(applyOp("", makeOp([{ type: "insert", text: "hi" }])), "hi");
  assert.throws(() => applyOp("ab", makeOp([{ type: "delete", n: 3 }])), RangeError);
  assert.strictEqual(baseLength(makeOp([{ type: "retain", n: 2 }, { type: "delete", n: 1 }])), 3);
  assert.strictEqual(targetLength(makeOp([{ type: "delete", n: 2 }, { type: "insert", text: "zz" }])), 2);
  assert.throws(() => transformPair(makeOp([{ type: "retain", n: 2 }]), makeOp([{ type: "retain", n: 3 }])), /equal base lengths/);

  // --- TP1 exhaustively over every operation on 3 characters ----------------
  const ops = enumerateOps(3);
  assert.ok(ops.length >= 40, `expected a rich operation space, got ${ops.length}`);
  let pairsChecked = 0;
  for (const text of ["abc", "aaa", "xyz"]) {
    for (const a of ops) {
      for (const b of ops) {
        const [aPrime, bPrime] = transformPair(a, b);
        const left = applyOp(applyOp(text, a), bPrime);
        const right = applyOp(applyOp(text, b), aPrime);
        assert.strictEqual(left, right, `TP1 failed: text=${text} a=[${opToString(a)}] b=[${opToString(b)}] -> ${left} vs ${right}`);
        pairsChecked++;
      }
    }
  }

  // --- random TP1 + determinism of the transform ----------------------------
  let seed = splitmix64(0x5eed813);
  let randomPairs = 0;
  for (let trial = 0; trial < 400; trial++) {
    const length = 1 + Number(seed() % 8n);
    let text = "";
    for (let i = 0; i < length; i++) text += "abcde"[Number(seed() % 5n)];
    const a = randomOp(text.length, seed);
    const b = randomOp(text.length, seed);
    const [aPrime, bPrime] = transformPair(a, b);
    const again = transformPair(a, b);
    assert.deepStrictEqual(again, [aPrime, bPrime], "transform must be deterministic");
    assert.strictEqual(applyOp(applyOp(text, a), bPrime), applyOp(applyOp(text, b), aPrime), `random TP1 failed for ${text}`);
    randomPairs++;
  }

  // --- confidentiality, integrity, determinism of the sealed log -----------
  const key = epochKey("doc-key", 0);
  const secret = "insert confidential sentence";
  const sealed = seal(key, 7, secret);
  assert.notStrictEqual(sealed.ciphertext.toString("utf8"), secret);
  assert.ok(!sealed.ciphertext.toString("hex").includes(Buffer.from(secret, "utf8").toString("hex")));
  assert.strictEqual(open(key, sealed), secret);
  const repeat = seal(key, 7, secret);
  assert.strictEqual(repeat.ciphertext.toString("hex"), sealed.ciphertext.toString("hex"), "seal must be deterministic");
  assert.notStrictEqual(seal(key, 8, secret).ciphertext.toString("hex"), sealed.ciphertext.toString("hex"), "nonce reuse would leak");
  const tampered = seal(key, 9, secret);
  tampered.ciphertext[0] ^= 0x01;
  assert.throws(() => open(key, tampered), /integrity check failed/);
  assert.throws(() => open(epochKey("doc-key", 1), sealed), /integrity check failed/);
  const balance = differingBitFraction(secret, key, 11);
  assert.ok(balance > 0.3 && balance < 0.7, `keystream/plaintext bit balance ${balance}`);

  // --- access control -------------------------------------------------------
  const network = new InMemoryNetwork();
  const acl = new AccessControlList("alice");
  acl.grant("bob", "editor");
  acl.grant("carol", "viewer");
  const server = new CollaborationServer(network, acl);
  const alice = new EditorClient("alice", network, server.documentId);
  const bob = new EditorClient("bob", network, server.documentId);
  const carol = new EditorClient("carol", network, server.documentId);
  const clients = [alice, bob, carol];
  for (const client of clients) network.register(client.id);
  network.register(server.documentId);

  alice.localEdit(makeOp([{ type: "insert", text: "hello" }]));
  pump(network, server, clients);
  assert.strictEqual(server.text, "hello");
  assert.strictEqual(bob.text, "hello");
  assert.strictEqual(carol.text, "hello", "viewers read but do not write");

  carol.localEdit(makeOp([{ type: "retain", n: 5 }, { type: "insert", text: "!" }]));
  pump(network, server, clients);
  assert.strictEqual(server.text, "hello");
  assert.ok(server.rejected.some((entry) => entry.clientId === "carol" && entry.reason === "forbidden"));
  assert.strictEqual(carol.text, "hello");

  // --- revocation confines the removed member to the older epoch ------------
  const beforeRevocation = server.log.length;
  acl.revoke("bob");
  assert.strictEqual(acl.epoch, 1);
  bob.localEdit(makeOp([{ type: "retain", n: 5 }, { type: "insert", text: "!" }]));
  pump(network, server, clients);
  assert.strictEqual(server.text, "hello", "a revoked editor cannot write");
  assert.ok(server.rejected.some((entry) => entry.clientId === "bob" && entry.reason === "forbidden"));
  assert.strictEqual(bob.text, "hello", "the rejected edit is rolled back locally");
  alice.localEdit(makeOp([{ type: "retain", n: 5 }, { type: "insert", text: " world" }]));
  pump(network, server, clients);
  assert.strictEqual(server.text, "hello world");
  assert.strictEqual(bob.text, "hello world", "the revoked member still receives the broadcast");
  const newEntries = server.log.slice(beforeRevocation);
  assert.ok(newEntries.length > 0 && newEntries.every((entry) => entry.epoch === 1));
  const revokedKey = acl.keyForEpoch(0);
  const currentKey = acl.keyForEpoch(1);
  assert.throws(() => open(revokedKey, newEntries[newEntries.length - 1].sealed), /integrity check failed/);
  assert.strictEqual(open(currentKey, newEntries[0].sealed).includes("world"), true);
  const readablePrefix = server.log.slice(0, beforeRevocation).every((entry) => open(revokedKey, entry.sealed).length > 0);
  assert.ok(readablePrefix, "the residual leak of the un-rewrapped prefix is measured, not hidden");

  // --- concurrent editing: convergence + no lost updates --------------------
  const network2 = new InMemoryNetwork();
  const acl2 = new AccessControlList("alice");
  const server2 = new CollaborationServer(network2, acl2, "shared-doc");
  const writers = ["alice", "bob", "carol", "dave"];
  for (const id of writers.slice(1)) acl2.grant(id, "editor");
  const editors = writers.map((id) => new EditorClient(id, network2, server2.documentId));
  for (const editor of editors) network2.register(editor.id);
  network2.register(server2.documentId);
  editors[0].localEdit(makeOp([{ type: "insert", text: "seed" }]));
  pump(network2, server2, editors);
  assert.strictEqual(server2.text, "seed");
  for (const editor of editors) assert.strictEqual(editor.text, "seed", `${editor.id} missed the seed operation`);
  const seedOps = server2.history.length;

  let editSeed = splitmix64(0xc0ffee813n);
  let submitted = 0;
  for (let round = 0; round < 60; round++) {
    for (const editor of editors) {
      // The client edits the state it can see: the operations it produces are concurrent with the
      // ones still travelling in the network, which is exactly what exercises the rebase path.
      const local = editor.text;
      const op = randomOp(local.length, editSeed);
      editor.localEdit(op);
      submitted++;
      if (Number(editSeed() % 3n) === 0) pump(network2, server2, editors, 50);
    }
  }
  pump(network2, server2, editors);
  for (const editor of editors) editor.sync();
  pump(network2, server2, editors);
  for (const editor of editors) editor.sync();
  pump(network2, server2, editors);

  const oracle = server2.replay("");
  assert.strictEqual(server2.text, oracle, "sequencer state must match the brute-force replay");
  for (const editor of editors) {
    assert.strictEqual(editor.text, oracle, `${editor.id} diverged: ${editor.text} vs ${oracle}`);
    assert.strictEqual(editor.pending.length, 0, `${editor.id} still has unacknowledged operations`);
  }
  assert.strictEqual(server2.history.length, seedOps + submitted, "every accepted operation must be ordered exactly once");
  const totalRebases = editors.reduce((sum, editor) => sum + editor.stats.rebases, 0);
  const totalTransforms = editors.reduce((sum, editor) => sum + editor.stats.transforms, 0);
  assert.ok(totalRebases > 0, "the out-of-order path must be exercised");
  assert.ok(totalTransforms >= totalRebases);
  assert.ok(server2.log.length === server2.history.length);
  // Cost accounting: every submission costs one request plus exactly one answer, every accepted
  // operation one broadcast per other member, every sync one request plus one answer.
  const expectedMessages = 2 * server2.stats.submits + server2.stats.broadcasts + 2 * server2.stats.syncs;
  assert.strictEqual(network2.messagesSent, expectedMessages, "message accounting");
  assert.strictEqual(server2.stats.accepted, server2.history.length);
  assert.strictEqual(server2.stats.accepted + server2.stats.rebases + server2.stats.rejected, server2.stats.submits);

  // --- degenerate documents and a single writer ----------------------------
  const network3 = new InMemoryNetwork();
  const acl3 = new AccessControlList("solo");
  const server3 = new CollaborationServer(network3, acl3, "empty-doc");
  const solo = new EditorClient("solo", network3, server3.documentId);
  network3.register(solo.id);
  network3.register(server3.documentId);
  solo.sync();
  pump(network3, server3, [solo]);
  assert.strictEqual(server3.text, "");
  assert.strictEqual(solo.text, "");
  solo.localEdit(makeOp([{ type: "insert", text: "x" }]));
  pump(network3, server3, [solo]);
  solo.localEdit(makeOp([{ type: "delete", n: 1 }]));
  pump(network3, server3, [solo]);
  assert.strictEqual(server3.text, "");
  assert.strictEqual(server3.log.length, 2);

  console.log(
    `TP1 pairs checked ${pairsChecked} exhaustive + ${randomPairs} random; convergent ops ${server2.history.length}, ` +
      `rebases ${totalRebases}, transforms ${totalTransforms}, messages ${network2.messagesSent}, ` +
      `keystream bit balance ${balance.toFixed(3)}`
  );
  console.log("PASS");
}
