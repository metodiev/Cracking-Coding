/** Task 665 - Implement operational transform with intention preservation tests. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * String OT over the two classic operations, `Insert(position, text, site)` and
 * `Delete(position, length)`. `transform(b, a)` returns the *list* of operations equivalent to b
 * when a has already been applied, so that a; transform(b, a) and b; transform(a, b) reach the
 * same document. Returning a list matters: when an insert lands strictly inside a deleted range,
 * no single delete can express the remainder, so the delete is split in two — that split is what
 * makes the transform satisfy TP1 on the awkward cases instead of silently losing or resurrecting
 * text.
 *
 * Rules: concurrent inserts at the same anchor are ordered by site id (higher id first), which is
 * a total order both replicas compute identically, so no text is lost (insert intention). An
 * insert whose anchor lies inside a deleted range collapses to the deletion start; a delete
 * transformed by an insert shifts, or splits when the insert is inside its range.
 *
 * Verified here: TP1 checked *exhaustively* over a generated corpus of documents and every valid
 * operation pair (thousands of pairs, including same-anchor collisions and insert-inside-delete),
 * intention preservation (both concurrent characters survive, deleted text never returns, no
 * spurious deletion), and two end-to-end client/server scenarios that converge — the two-client
 * case is exactly the pairwise TP1 property applied through two different code paths, and the
 * third client joins causally afterwards. TP2 (the three-way transpose property) is explicitly
 * NOT claimed: the plain pair-wise transform does not satisfy it, which is why the server model
 * serialises operations and clients rebase only what is still in flight.
 */
"use strict";
const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");
const insert = (position, text, site = "s0") => {
  if (!Number.isInteger(position) || position < 0) throw new RangeError("insert position must be a non-negative integer");
  if (typeof text !== "string") throw new TypeError("inserted text must be a string");
  return { type: "insert", position, text, site };
};
const remove = (position, length) => {
  if (!Number.isInteger(position) || position < 0) throw new RangeError("delete position must be non-negative");
  if (!Number.isInteger(length) || length < 0) throw new RangeError("delete length must be non-negative");
  return { type: "delete", position, length };
};
function apply(doc, operation) {
  if (operation.type === "insert") {
    if (operation.position > doc.length) throw new RangeError("insert position past the end");
    return doc.slice(0, operation.position) + operation.text + doc.slice(operation.position);
  }
  if (operation.type === "delete") {
    if (operation.position + operation.length > doc.length) throw new RangeError("delete range past the end");
    return doc.slice(0, operation.position) + doc.slice(operation.position + operation.length);
  }
  throw new TypeError(`unknown operation ${operation.type}`);
}
const applyAll = (doc, operations) => operations.reduce((current, operation) => apply(current, operation), doc);
const span = (operation) => operation.position + (operation.type === "insert" ? operation.text.length : operation.length);
const rank = (site) => (site === undefined ? 0 : site.charCodeAt(0) * 1024 + site.charCodeAt(site.length - 1));
/**
 * Same-anchor ordering. Sites are ordered by id; two operations of the same site can only be
 * causally ordered, but the transform must still be antisymmetric, so the inserted text breaks the
 * remaining tie. `tieBreak(x, y) === true` means x takes the left slot (y shifts right).
 */
function tieBreak(other, operation) {
  if (rank(other.site) !== rank(operation.site)) return rank(other.site) > rank(operation.site);
  return other.text > operation.text;
}
/**
 * Transforms `operation` (produced against the document *before* `other` was applied) so it can be
 * applied *after* `other`. Always returns an array: deletes may be split in two.
 */
function transform(operation, other) {
  if (operation.type !== "insert" && operation.type !== "delete") {
    throw new TypeError(`unknown operation ${operation.type}`);
  }
  if (other.type !== "insert" && other.type !== "delete") throw new TypeError(`unknown operation ${other.type}`);
  if (other.type === "insert") {
    if (operation.type === "insert") {
      if (other.position < operation.position) return [{ ...operation, position: operation.position + other.text.length }];
      if (other.position > operation.position) return [{ ...operation }];
      // Same anchor: the higher site id wins the left slot, the loser shifts right.
      return tieBreak(other, operation)
        ? [{ ...operation, position: operation.position + other.text.length }]
        : [{ ...operation }];
    }
    // Mapping rule: an old index i maps to i when i < other.position, and to i + k otherwise.
    if (operation.position >= other.position) {
      return [{ ...operation, position: operation.position + other.text.length }];
    }
    if (span(operation) <= other.position) return [{ ...operation }];
    // The inserted text sits inside the deleted range and is not part of it: split around it.
    // The second range's position is expressed after the first range has already been removed.
    const left = other.position - operation.position;
    const right = span(operation) - other.position;
    const parts = [];
    if (left > 0) parts.push(remove(operation.position, left));
    if (right > 0) parts.push(remove(other.position + other.text.length - left, right));
    return parts;
  }
  if (operation.type === "insert") {
    if (operation.position <= other.position) return [{ ...operation }];
    if (operation.position >= span(other)) return [{ ...operation, position: operation.position - other.length }];
    return [{ ...operation, position: other.position }]; // the anchor is gone: collapse to the range start
  }
  const start = Math.max(operation.position, other.position);
  const stop = Math.min(span(operation), span(other));
  if (start >= stop) {
    return operation.position > other.position
      ? [{ ...operation, position: Math.max(other.position, operation.position - other.length) }]
      : [{ ...operation }];
  }
  return [{ ...operation, position: Math.min(operation.position, other.position), length: operation.length - (stop - start) }];
}
/** Server-side rebase: fold the pair-wise transform over a history of already-committed work. */
function transformAgainst(operations, history) {
  let current = operations.map((operation) => ({ ...operation }));
  for (const other of history) current = current.flatMap((operation) => transform(operation, other));
  return current;
}
/** Pairwise convergence check (TP1) for one document and two concurrent operations. */
function converges(doc, a, b) {
  return applyAll(apply(doc, a), transform(b, a)) === applyAll(apply(doc, b), transform(a, b));
}
class Site {
  constructor(id, doc = "") {
    this.id = id;
    this.doc = doc;
    this.pending = []; // locally applied operations the server has not committed yet
    this.seen = 0;     // how many server operations this site has already applied
  }
  localInsert(position, text) {
    const operation = insert(position, text, this.id);
    operation.base = this.seen; // revision this operation was created against
    this.doc = apply(this.doc, operation);
    this.pending.push(operation);
    return operation;
  }
  localDelete(position, length) {
    const operation = remove(position, length);
    operation.base = this.seen;
    this.doc = apply(this.doc, operation);
    this.pending.push(operation);
    return operation;
  }
  /** Applies operations committed by other sites, rebased against what is still in flight here. */
  receive(operations) {
    const batch = Array.isArray(operations) ? operations : [operations];
    const applied = [];
    for (const operation of batch) {
      const rebased = transformAgainst([operation], this.pending);
      for (const part of rebased) {
        this.doc = apply(this.doc, part);
        applied.push(part);
      }
    }
    return applied;
  }
  /** Catches up on every committed operation this site has not applied yet. */
  sync(server) {
    const applied = [];
    for (let index = this.seen; index < server.history.length; index++) {
      applied.push(...this.receive([server.history[index]]));
    }
    this.seen = server.history.length;
    return applied;
  }
}
class Server {
  constructor(doc = "") {
    this.doc = doc;
    this.history = [];
    this.sites = [];
  }
  attach(site) {
    this.sites.push(site);
    site.seen = this.history.length;
  }
  text() {
    return this.doc;
  }
  /**
   * Transforms a site's in-flight operations against everything they were not created against,
   * commits them, tells the author which earlier commits it still owes itself, and broadcasts the
   * committed work to the other sites.
   */
  submit(site) {
    const committed = [];
    for (const local of site.pending) {
      const unseen = this.history.slice(local.base ?? site.seen);
      const transformed = transformAgainst([local], unseen);
      const firstNew = this.history.length;
      for (const operation of transformed) {
        this.doc = apply(this.doc, operation);
        this.history.push(operation);
        committed.push(operation);
      }
      // The author applied its own operation against its base revision, so it still owes the
      // operations committed after that base — rebased against its own operation.
      for (let index = site.seen; index < firstNew; index++) {
        for (const part of transform(this.history[index], local)) site.doc = apply(site.doc, part);
      }
      site.seen = this.history.length;
    }
    site.pending = [];
    for (const other of this.sites) {
      if (other === site || committed.length === 0) continue;
      other.receive(committed);
      other.seen = this.history.length;
    }
    return committed;
  }
}
module.exports = { insert, remove, apply, applyAll, transform, transformAgainst, converges, Site, Server, span };
if (require.main === module) {
  // -- operation validation --------------------------------------------------------
  strictEqual(apply("abc", insert(1, "XY")), "aXYbc");
  strictEqual(apply("abc", remove(1, 2)), "a");
  strictEqual(apply("abc", remove(0, 0)), "abc", "a zero-length delete is a no-op");
  strictEqual(apply("", insert(0, "hi")), "hi");
  throws(() => apply("abc", insert(9, "z")), RangeError, "an out-of-range insert is rejected");
  throws(() => apply("abc", remove(2, 5)), RangeError, "an out-of-range delete is rejected");
  throws(() => insert(-1, "z"), RangeError, "a negative insert position is rejected");
  throws(() => remove(0, -1), RangeError, "a negative delete length is rejected");
  throws(() => insert(0, 5), TypeError, "inserted text must be a string");
  throws(() => transform({ type: "move" }, insert(0, "x")), TypeError, "unknown operations are rejected");
  // -- transform, by hand ----------------------------------------------------------
  deepStrictEqual(transform(insert(1, "B"), insert(0, "A")), [insert(2, "B")], "a later insert shifts right");
  deepStrictEqual(transform(insert(1, "A"), insert(3, "B")), [insert(1, "A")], "an earlier insert is untouched");
  deepStrictEqual(transform(insert(5, "x"), remove(0, 3)), [insert(2, "x")], "an insert after a delete shifts left");
  deepStrictEqual(transform(insert(2, "x"), remove(0, 3)), [insert(0, "x")],
    "an insert inside the deleted range collapses to its start");
  deepStrictEqual(transform(remove(0, 3), insert(0, "A")), [remove(1, 3)], "an insert before a delete pushes it right");
  const split = transform(remove(1, 3), insert(2, "XY"));
  deepStrictEqual(split, [remove(1, 1), remove(3, 2)], "an insert inside a delete splits the delete around it");
  strictEqual(applyAll(apply("abcde", insert(2, "XY")), split), "aXYe",
    "the split deletes exactly the original range and keeps the inserted text");
  deepStrictEqual(transform(remove(0, 4), remove(2, 4)), [remove(0, 2)], "overlapping deletes keep only the excess");
  deepStrictEqual(transform(remove(6, 2), remove(0, 3)), [remove(3, 2)], "a later delete shifts left");
  deepStrictEqual(transform(remove(0, 3), remove(5, 2)), [remove(0, 3)], "an earlier delete is untouched");
  // -- exhaustive TP1 over a generated corpus --------------------------------------
  const documents = ["", "a", "ab", "hello", "aaaa", "abcdefghij", "banana", "xyzw"];
  const operations = [];  for (let position = 0; position <= 10; position++) {
    operations.push(insert(position, "Q", `s${position}`), insert(position, "QR", `s${position}`));
    operations.push(remove(position, 0), remove(position, 1), remove(position, 3));
  }
  let pairs = 0;
  for (const doc of documents) {
    for (const a of operations) {
      if (a.type === "insert" ? a.position > doc.length : span(a) > doc.length) continue;
      for (const b of operations) {
        if (b.type === "insert" ? b.position > doc.length : span(b) > doc.length) continue;
        const left = applyAll(apply(doc, a), transform(b, a));
        const right = applyAll(apply(doc, b), transform(a, b));
        if (left !== right) {
          throw new Error(`TP1 violation on "${doc}": ${JSON.stringify(a)} then ${JSON.stringify(b)} gave "${left}" and "${right}"`);
        }
        pairs++;
      }
    }
  }
  ok(pairs > 4000, `the sweep must cover a real corpus, ran ${pairs} pairs`);
  // -- same-anchor insert collisions (intention preservation for inserts) -----------
  let collisions = 0;
  const sites = ["alpha", "beta", "gamma"];  for (const doc of documents) {
    for (let position = 0; position <= doc.length; position++) {
      for (const left of sites) {
        for (const right of sites) {
          const a = insert(position, `${left[0]}1`, left);
          const b = insert(position, `${right[0]}2`, right);
          const forward = applyAll(apply(doc, a), transform(b, a));
          const backward = applyAll(apply(doc, b), transform(a, b));
          strictEqual(forward, backward, `collision at ${position} on "${doc}" must converge`);
          ok(forward.includes(`${left[0]}1`) && forward.includes(`${right[0]}2`), "both insertions survive");
          strictEqual(forward.length, doc.length + 4, "no character is duplicated or lost");
          collisions++;
        }
      }
    }
  }
  ok(collisions > 200, `the collision sweep must cover real cases, ran ${collisions}`);
  // -- insert inside a deleted range: text is neither resurrected nor swallowed -----
  let insideChecks = 0;
  for (const doc of ["abcdef", "hello world"]) {
    for (let start = 0; start < doc.length; start++) {
      for (let length = 1; start + length <= doc.length; length++) {
        for (let position = start + 1; position < start + length; position++) {
          const deletion = remove(start, length);
          const insertion = insert(position, "@", "ins");
          ok(converges(doc, deletion, insertion), `insert inside delete must converge (${start},${length},${position})`);
          const result = applyAll(apply(doc, deletion), transform(insertion, deletion));
          ok(result.includes("@"), "the inserted text survives the concurrent delete");
          strictEqual(result.length, doc.length - length + 1, "exactly the deleted range disappears");
          insideChecks++;
        }
      }
    }
  }
  ok(insideChecks > 50, `the inside-delete sweep must cover real cases, ran ${insideChecks}`);
  // -- two clients editing concurrently --------------------------------------------
  function twoClientScenario() {
    const server = new Server("hello");
    const a = new Site("alpha", "hello");
    const b = new Site("beta", "hello");
    server.attach(a);
    server.attach(b);
    a.localInsert(0, "A");      // both edit before either syncs: genuinely concurrent
    b.localDelete(1, 3);
    server.submit(a);
    server.submit(b);
    return { server: server.text(), a: a.doc, b: b.doc, history: server.history };
  }
  const two = twoClientScenario();
  strictEqual(two.a, two.b, "the two clients must converge on the same text");
  strictEqual(two.a, two.server, "and agree with the server");
  ok(two.server.includes("A"), "the concurrent insert survived");
  ok(!two.server.includes("ell"), "the concurrent delete really removed its range");
  deepStrictEqual(twoClientScenario(), two, "the scenario must be deterministic");
  // -- a third client joins causally after the others -------------------------------
  const server = new Server("hello");
  const first = new Site("alpha", "hello");
  const second = new Site("beta", "hello");
  server.attach(first);
  server.attach(second);
  first.localInsert(0, "A");
  server.submit(first);
  second.localDelete(0, 1);
  server.submit(second);
  const third = new Site("gamma", "hello");
  third.sync(server);
  strictEqual(third.doc, server.text(), "the late joiner catches up through the history");
  third.localInsert(third.doc.length, "!");
  server.submit(third);
  for (const site of [first, second]) site.sync(server);
  ok([first.doc, second.doc, third.doc, server.text()].every((doc) => doc === server.text()),
    `all sites must agree after the causal join: ${[first.doc, second.doc, third.doc, server.text()].join(" | ")}`);
  ok(server.text().endsWith("!"), "the late joiner's insert is preserved at the end");
  // -- transformAgainst is exactly the sequential fold ------------------------------
  const history = [insert(0, "A", "s1"), insert(3, "B", "s2"), remove(1, 1)];
  const incoming = insert(1, "Z", "s3");
  deepStrictEqual(transformAgainst([incoming], history),
    history.reduce((current, other) => current.flatMap((operation) => transform(operation, other)), [{ ...incoming }]),
    "the fold must equal the sequential pair-wise transform");  const historyDoc = applyAll("xyz", history);
  const folded = transformAgainst([incoming], history);
  strictEqual(applyAll(historyDoc, folded).length, historyDoc.length + 1, "the folded insert adds one character");
  ok(applyAll(historyDoc, folded).includes("Z"));
  for (const doc of documents) {    for (const operation of operations) {
      if (operation.type === "insert" && operation.position > doc.length) continue;
      if (operation.type === "delete" && span(operation) > doc.length) continue;
      const transformed = transform(operation, insert(0, "##", "zz"));
      const total = transformed.reduce((sum, part) => sum + (part.type === "insert" ? part.text.length : -part.length), 0);
      const originalDelta = operation.type === "insert" ? operation.text.length : -operation.length;
      ok(total === originalDelta, "the transformed batch must keep the original net length change");
    }
  }
  console.log(`tp1Pairs=${pairs} collisions=${collisions} insertInsideDelete=${insideChecks} ` +
    `twoClients=[${two.server}] threeClients=[${server.text()}]`);
  console.log("PASS");
}
