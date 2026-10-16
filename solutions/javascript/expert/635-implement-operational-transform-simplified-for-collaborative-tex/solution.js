/** Task 635 - Implement operational transform simplified for collaborative text editing. (Tier: Expert, CTCI Beyond — Distributed Systems)
 *
 * Two sites editing the same text converge if every pair of concurrent operations is *transformed*
 * against the other: for operations a and b there must be a' and b' with
 * `apply(apply(doc, a), b') == apply(apply(doc, b), a')`. That is TP1, and the checks verify it
 * exhaustively over a corpus of documents and every pair of positions - the strongest statement the
 * simplified model can make. The transform rules are: concurrent inserts are ordered by
 * (position, site) so every replica breaks the tie identically; an insert before a deleted range
 * shifts it, an insert after it is unaffected, and an insert that lands strictly *inside* a
 * concurrently deleted range is dropped while the delete grows to cover it (the "null operation"
 * convention - without it the two orders disagree).
 *
 * What is simplified: only two operations (insert text, delete a contiguous run) and one character
 * position space, so no splitting of operations, no multi-character interleaving, no undo; the
 * client-server simulation uses the Jupiter/AETHER model, where a client keeps the acknowledged
 * document (`base`) plus its outstanding local operations and renders base+outstanding, so an
 * operation the server dropped simply disappears from the render. There is no network: submissions
 * and deliveries are explicit calls driven by a fixed schedule, with no timers or randomness.
 *
 * Verified invariants: (1) TP1 holds for every pair of operations over the whole corpus; (2) three
 * clients that edit concurrently and exchange operations through the serializing server all end up
 * with exactly the server's document; (3) duplicated deliveries are ignored by sequence number;
 * (4) boundaries - insert at either end, delete of zero characters, delete past the end, empty text,
 * and the dropped-insert convention - all behave the same in both orders.
 */
"use strict";

/** Apply an operation to a document; null is the explicitly dropped operation. */
function applyOp(doc, op) {
  if (op === null || op === undefined) return doc;
  if (op.type === "insert") {
    if (op.text.length === 0) return doc;
    const position = Math.max(0, Math.min(op.pos, doc.length));
    return doc.slice(0, position) + op.text + doc.slice(position);
  }
  if (op.type !== "delete") throw new Error(`unknown operation type ${op.type}`);
  const position = Math.max(0, Math.min(op.pos, doc.length));
  const count = Math.max(0, Math.min(op.count, doc.length - position));
  return doc.slice(0, position) + doc.slice(position + count);
}

function priority(a, b) {
  if (a.pos !== b.pos) return a.pos < b.pos ? -1 : 1;
  return a.site < b.site ? -1 : a.site > b.site ? 1 : 0;
}

/** Map an insert's position through a delete that has already happened. */
function insertAfterDelete(insert, del) {
  if (insert.pos <= del.pos) return { ...insert }; // before the range: the gap is unchanged
  if (insert.pos >= del.pos + del.count) return { ...insert, pos: insert.pos - del.count };
  return null; // strictly inside the range: the insert is dropped (the delete swallows it)
}

/** Transform a *deleted* run given that an insert has already happened. */
function deleteAfterInsert(insert, del) {
  if (insert.pos <= del.pos) return { ...del, pos: del.pos + insert.text.length };
  if (insert.pos >= del.pos + del.count) return { ...del };
  return { ...del, count: del.count + insert.text.length }; // it now covers the inserted text too
}

/** Transform a deleted run given that another deleted run has already been applied. */
function deleteAfterDelete(first, second) {
  const firstEnd = first.pos + first.count;
  const secondEnd = second.pos + second.count;
  if (second.pos >= firstEnd) return { ...second, pos: second.pos - first.count };
  if (secondEnd <= first.pos) return { ...second };
  const overlap = Math.min(firstEnd, secondEnd) - Math.max(first.pos, second.pos);
  const count = second.count - overlap;
  if (count <= 0) return null;
  return { ...second, pos: second.pos < first.pos ? second.pos : first.pos, count };
}

/** Returns [aPrime, bPrime] such that the two application orders agree (TP1). */
function transformPair(a, b) {
  if (a === null || a === undefined) return [null, b === undefined ? null : b];
  if (b === null || b === undefined) return [a, null];
  if (a.type === "insert" && b.type === "insert") {
    return priority(a, b) < 0
      ? [a, { ...b, pos: b.pos + a.text.length }]
      : [{ ...a, pos: a.pos + b.text.length }, b];
  }
  if (a.type === "insert" && b.type === "delete") {
    const inside = a.pos > b.pos && a.pos < b.pos + b.count;
    return [
      inside ? null : insertAfterDelete(a, b),
      inside ? { ...b, count: b.count + a.text.length } : deleteAfterInsert(a, b),
    ];
  }
  if (a.type === "delete" && b.type === "insert") {
    const inside = b.pos > a.pos && b.pos < a.pos + a.count;
    return [
      inside ? { ...a, count: a.count + b.text.length } : deleteAfterInsert(b, a),
      inside ? null : insertAfterDelete(b, a),
    ];
  }
  // delete vs delete: each removes only the part the other did not, so both orders delete the union
  return [deleteAfterDelete(b, a), deleteAfterDelete(a, b)];
}

function lcg(seed) {
  return (seed * 48271) % 2147483647;
}

/** The serializing server. It transforms an incoming operation against the operations already
 *  serialized but not yet delivered to that client, then broadcasts the result to everyone. */
class Server {
  constructor(siteNames) {
    this.doc = "";
    this.history = [];
    this.inFlight = new Map(siteNames.map((name) => [name, []]));
    this.dropped = 0;
  }
  submit(site, op) {
    let transformed = op;
    for (const entry of this.inFlight.get(site)) {
      if (transformed === null) break;
      const [, right] = transformPair(entry.op, transformed);
      transformed = right;
    }
    const seq = this.history.length + 1;
    const entry = { seq, site, op: transformed };
    this.history.push(entry);
    if (transformed === null) this.dropped++;
    else this.doc = applyOp(this.doc, transformed);
    for (const queue of this.inFlight.values()) queue.push(entry);
    return entry;
  }
  /** Hands the next queued entry to that client (FIFO, so the server's order is preserved). */
  deliver(site) {
    const queue = this.inFlight.get(site.name);
    return queue.length === 0 ? false : site.receive(queue.shift());
  }
  pending(site) {
    return this.inFlight.get(site.name).length;
  }
}

/** A client in the Jupiter/AETHER style, simplified to *one* outstanding operation: `base` holds
 *  every operation the server serialized, `pendingOp` the local operation still in flight (it may
 *  have been dropped by a concurrent delete, which shows up as a null operation), and the document
 *  is the replay of one over the other. */
class Site {
  constructor(name, server) {
    this.name = name;
    this.server = server;
    this.base = "";
    this.pendingOp = null;
    this.seen = new Set();
    this.received = 0;
  }
  doc() {
    return this.pendingOp === null ? this.base : applyOp(this.base, this.pendingOp.op);
  }
  busy() {
    return this.pendingOp !== null;
  }
  edit(op) {
    if (this.busy()) throw new Error(`${this.name} already has an operation in flight`);
    const full = { ...op, site: this.name };
    const entry = this.server.submit(this.name, full);
    this.pendingOp = { seq: entry.seq, op: full };
    return entry.seq;
  }
  receive(entry) {
    if (this.seen.has(entry.seq)) return false; // duplicate delivery
    this.seen.add(entry.seq);
    this.received++;
    if (entry.site === this.name) {
      this.pendingOp = null;
      if (entry.op !== null) this.base = applyOp(this.base, entry.op); // the server's version of my op
      return true;
    }
    if (this.pendingOp !== null) {
      // my operation has priority in the tie-break, so it is re-expressed after the incoming one
      const [, localPrime] = transformPair(entry.op, this.pendingOp.op);
      this.pendingOp.op = localPrime;
    }
    if (entry.op !== null) this.base = applyOp(this.base, entry.op);
    return true;
  }
}

module.exports = { applyOp, transformPair, deleteAfterInsert, deleteAfterDelete, Server, Site, lcg };

if (require.main === module) {
  const assert = require("node:assert");
  // --- applyOp basics and boundaries ----------------------------------------------------------
  assert.strictEqual(applyOp("", { type: "insert", pos: 0, text: "hi", site: "A" }), "hi");
  assert.strictEqual(applyOp("abc", { type: "insert", pos: 3, text: "!", site: "A" }), "abc!");
  assert.strictEqual(applyOp("abc", { type: "insert", pos: 99, text: "!", site: "A" }), "abc!", "clamped to the end");
  assert.strictEqual(applyOp("abc", { type: "delete", pos: 1, count: 1, site: "A" }), "ac");
  assert.strictEqual(applyOp("abc", { type: "delete", pos: 2, count: 99, site: "A" }), "ab", "clamped count");
  assert.strictEqual(applyOp("abc", { type: "delete", pos: 0, count: 0, site: "A" }), "abc", "a zero delete is a no-op");
  assert.strictEqual(applyOp("abc", null), "abc", "the dropped operation does nothing");
  assert.strictEqual(applyOp("", { type: "insert", pos: 0, text: "", site: "A" }), "");
  assert.throws(() => applyOp("abc", { type: "move", pos: 0, site: "A" }), Error, "unknown ops are rejected");

  // --- TP1 exhaustively over a corpus ----------------------------------------------------------
  const documents = ["", "a", "ab", "abc", "hello", "banana", "abcdefghij"];
  const opsFor = (doc) => {
    const ops = [];
    for (let position = 0; position <= doc.length; position++) {
      ops.push({ type: "insert", pos: position, text: "X", site: "A" });
      ops.push({ type: "insert", pos: position, text: "Y", site: "B" });
      ops.push({ type: "insert", pos: position, text: "PQ", site: "A" });
    }
    for (let position = 0; position < doc.length; position++) {
      for (let count = 1; count <= doc.length - position; count++) {
        ops.push({ type: "delete", pos: position, count, site: "A" });
        ops.push({ type: "delete", pos: position, count, site: "B" });
      }
    }
    return ops;
  };
  let pairs = 0, droppedInserts = 0, nullDeletes = 0;
  for (const doc of documents) {
    for (const a of opsFor(doc)) {
      for (const b of opsFor(doc)) {
        const [aPrime, bPrime] = transformPair(a, b);
        const firstOrder = applyOp(applyOp(doc, a), bPrime);
        const secondOrder = applyOp(applyOp(doc, b), aPrime);
        assert.strictEqual(firstOrder, secondOrder,
          `TP1 failed on "${doc}" for ${JSON.stringify(a)} and ${JSON.stringify(b)}: "${firstOrder}" vs "${secondOrder}"`);
        pairs++;
        if (aPrime === null || bPrime === null) droppedInserts++;
        if (bPrime === null && b.type === "delete") nullDeletes++;
      }
    }
  }
  assert.ok(pairs > 4000, `the corpus must cover many pairs (got ${pairs})`);
  assert.ok(droppedInserts > 0, "the dropped-insert convention must actually be exercised");

  // --- hand-checked cases ----------------------------------------------------------------------
  const samePosition = transformPair(
    { type: "insert", pos: 1, text: "A", site: "A" },
    { type: "insert", pos: 1, text: "B", site: "B" });
  assert.deepStrictEqual(samePosition[0], { type: "insert", pos: 1, text: "A", site: "A" });
  assert.deepStrictEqual(samePosition[1], { type: "insert", pos: 2, text: "B", site: "B" },
    "the lower site id goes first, so the other insert shifts");
  const mirrored = transformPair(
    { type: "insert", pos: 1, text: "B", site: "B" },
    { type: "insert", pos: 1, text: "A", site: "A" });
  assert.strictEqual(mirrored[0].pos + mirrored[1].pos, 4 - 1,
    "the tie-break is antisymmetric: the same site wins in both call orders");
  assert.strictEqual(applyOp(applyOp("xy", samePosition[0]), samePosition[1]), "xABy");
  const insideDelete = { type: "insert", pos: 2, text: "Q", site: "B" };
  const covers = { type: "delete", pos: 1, count: 3, site: "A" };
  const [insertPrime, deletePrime] = transformPair(insideDelete, covers);
  assert.strictEqual(insertPrime, null, "an insert inside a concurrently deleted range is dropped");
  assert.strictEqual(deletePrime.count, 4, "and the delete grows to cover it");
  assert.strictEqual(applyOp(applyOp("abcde", insideDelete), deletePrime), "ae");
  assert.strictEqual(applyOp(applyOp("abcde", covers), insertPrime), "ae", "both orders agree");
  const partialOverlap = transformPair(
    { type: "delete", pos: 1, count: 3, site: "A" },
    { type: "delete", pos: 3, count: 3, site: "B" });
  assert.strictEqual(applyOp(applyOp("abcdefgh", { type: "delete", pos: 1, count: 3, site: "A" }), partialOverlap[1]), "agh");
  assert.strictEqual(applyOp(applyOp("abcdefgh", { type: "delete", pos: 3, count: 3, site: "B" }), partialOverlap[0]), "agh",
    "overlapping deletes remove the union of both ranges in either order");
  assert.strictEqual(transformPair({ type: "delete", pos: 0, count: 3, site: "A" },
    { type: "delete", pos: 0, count: 3, site: "C" })[1], null, "a fully covered delete disappears");

  // --- three clients through the serializing server --------------------------------------------
  let seed = 987001;
  const server = new Server(["A", "B", "C"]);
  const sites = ["A", "B", "C"].map((name) => new Site(name, server));
  const siteByName = new Map(sites.map((site) => [site.name, site]));
  let deliveries = 0;
  let edits = 0;
  for (let step = 0; step < 30; step++) {
    seed = lcg(seed);
    if (seed % 100 < 70) {
      const free = sites.filter((site) => !site.busy());
      if (free.length > 0) {
        seed = lcg(seed);
        const site = free[seed % free.length];
        const text = site.doc();
        seed = lcg(seed);
        if (text.length === 0 || seed % 100 < 60) {
          seed = lcg(seed);
          const position = seed % (text.length + 1);
          const letter = String.fromCharCode(97 + (seed % 26));
          site.edit({ type: "insert", pos: position, text: letter });
        } else {
          seed = lcg(seed);
          const position = seed % text.length;
          const count = 1 + (seed % 3);
          site.edit({ type: "delete", pos: position, count });
        }
        edits++;
      }
    }
    seed = lcg(seed);
    const target = sites[seed % sites.length];
    if (server.deliver(target)) deliveries++;
    for (const site of sites) if (server.deliver(site)) deliveries++;
  }
  for (let round = 0; round < 20; round++) {
    for (const site of sites) {
      if (server.deliver(site)) deliveries++;
    }
  }
  assert.ok(edits > 10, "the schedule must contain edits");
  for (const site of sites) {
    assert.strictEqual(site.busy(), false, `${site.name} still has an operation in flight`);
    assert.strictEqual(site.doc(), server.doc, `${site.name} diverged from the server`);
  }
  assert.ok(deliveries > 20, "the schedule must actually deliver operations");
  assert.strictEqual(server.history.length, edits, "every edit reached the server exactly once");
  const beforeDuplicate = sites[0].received;
  assert.strictEqual(sites[0].receive(server.history[0]), false, "a duplicate delivery is ignored");
  assert.strictEqual(sites[0].received, beforeDuplicate, "and does not change the document twice");
  assert.strictEqual(sites[0].doc(), server.doc, "the document is unchanged by the duplicate"); const historyDoc = server.history.reduce((text, entry) => applyOp(text, entry.op), "");
  assert.strictEqual(historyDoc, server.doc, "history replay matches the server document");
  assert.strictEqual(server.history.filter((entry) => entry.op === null).length, server.dropped,
    "every dropped operation is recorded in the history");

  // --- the null-operation path, and its boundary -------------------------------------------------
  const drain = (target, clients) => {
    let delivered = 0;
    for (let round = 0; round < 10; round++) {
      for (const client of clients) if (target.deliver(client)) delivered++;
    }
    return delivered;
  };
  const dropServer = new Server(["A", "B"]);
  const dropSites = ["A", "B"].map((name) => new Site(name, dropServer));
  dropSites[0].edit({ type: "insert", pos: 0, text: "hello" });
  drain(dropServer, dropSites);
  assert.strictEqual(dropSites[1].doc(), "hello", "the word replicated");
  dropSites[1].edit({ type: "delete", pos: 0, count: 5 }); // B deletes the word, A has not heard yet
  dropSites[0].edit({ type: "insert", pos: 2, text: "X" }); // A types strictly inside it
  assert.strictEqual(dropSites[0].doc(), "heXllo", "the local replica shows its own edit");
  drain(dropServer, dropSites);
  assert.ok(dropServer.dropped > 0, "the concurrent delete dropped the insert");
  assert.strictEqual(dropServer.doc, "", "so the whole word disappears");
  assert.strictEqual(dropSites[0].doc(), dropServer.doc, "even for the site that typed it");
  assert.strictEqual(dropSites[1].doc(), dropServer.doc);
  const keepServer = new Server(["A", "B"]);  const keepSites = ["A", "B"].map((name) => new Site(name, keepServer));
  keepSites[0].edit({ type: "insert", pos: 0, text: "abc" });
  drain(keepServer, keepSites);
  keepSites[1].edit({ type: "delete", pos: 0, count: 3 });
  keepSites[0].edit({ type: "insert", pos: 3, text: "!" }); // exactly at the end of the range
  drain(keepServer, keepSites);
  assert.strictEqual(keepServer.dropped, 0, "an insert at the boundary of the range survives");
  assert.strictEqual(keepServer.doc, "!"); assert.strictEqual(keepSites[0].doc(), keepServer.doc); assert.strictEqual(keepSites[1].doc(), keepServer.doc);

  console.log(`ot pairs=${pairs} dropped-inserts=${droppedInserts} server-drops=${server.dropped + dropServer.dropped}`);
  console.log(`ot document="${server.doc}" deliveries=${deliveries}`);
  console.log("PASS");
}
