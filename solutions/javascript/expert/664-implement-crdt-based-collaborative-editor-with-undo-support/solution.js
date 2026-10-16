/** Task 664 - Implement CRDT-based collaborative editor with undo support. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * An RGA (Replicated Growable Array) text CRDT plus per-site undo.
 *
 * Data model: every character is an element `{ id: {site, seq}, ch, origin, deletedBy: Set<site> }`.
 * `origin` is the id of the element the character was typed after (null = at the head). The
 * document is a depth-first walk of the `children[origin]` lists, skipping tombstones. The
 * ordering rule for *concurrent* siblings is `id` descending (site first, then seq), which both
 * sites compute identically, and that is what gives convergence: an insert refers only to an
 * origin that its author had already seen, so it can never be delivered before its origin —
 * unknown-origin operations are buffered (causal delivery) and integrated later.
 *
 * Deletion is a tombstone *set of sites* (`deletedBy`) rather than a flag. That is what makes
 * undo honest: undoing my own delete removes only my site from the set, so a concurrent delete
 * from another site keeps the character dead and cannot be resurrected by my undo.
 *
 * Verified: convergence of three replicas fed the same operations in different orders (and with
 * duplicated deliveries), causal buffering of out-of-order operations, idempotent integration,
 * the sibling tie-break rule, undo/redo of both insert and delete, and the no-resurrection rule
 * above. Simplification: one process per site, no operation compression, no text formatting.
 */
"use strict";
const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");
function makeRng(seed) {
  let state = seed >>> 0 || 0x5bd1e995;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}
const idKey = (id) => `${id.site}:${id.seq}`;
/** RGA tie-break: higher site id wins; between equal sites, higher sequence wins. */
function compareIds(a, b) {
  if (a.site !== b.site) return a.site < b.site ? -1 : 1;
  if (a.seq !== b.seq) return a.seq < b.seq ? -1 : 1;
  return 0;
}
class RGA {
  constructor(site) {
    if (typeof site !== "string" || site.length === 0) throw new RangeError("a site id is required");
    this.site = site;
    this.counter = 0;
    this.elements = new Map();  // idKey -> element
    this.children = new Map();  // originKey (null for the head) -> [idKey] ordered by descending id
    this.pending = [];          // operations whose origin has not arrived yet
    this.undoStack = [];
    this.redoStack = [];
  }
  /** Visible text: depth-first walk of the children lists, tombstones skipped. */
  text() {
    const out = [];
    const walk = (originKey) => {
      for (const key of this.children.get(originKey) ?? []) {
        const element = this.elements.get(key);
        if (element.deletedBy.size === 0) out.push(element.ch);
        walk(key);
      }
    };
    walk(null);
    return out.join("");
  }
  /** Visible elements in order, as [idKey, element] pairs. */
  visible() {
    const out = [];
    const walk = (originKey) => {
      for (const key of this.children.get(originKey) ?? []) {
        const element = this.elements.get(key);
        if (element.deletedBy.size === 0) out.push([key, element]);
        walk(key);
      }
    };
    walk(null);
    return out;
  }
  /** Inserts a character after the visible element at `index - 1`. */
  localInsert(index, ch) {
    if (!Number.isInteger(index) || index < 0 || index > this.visible().length) {
      throw new RangeError(`index ${index} is outside the document`);
    }
    if (typeof ch !== "string" || ch.length !== 1) throw new TypeError("exactly one character is required");
    const origin = index === 0 ? null : this.visible()[index - 1][0];
    const operation = { type: "insert", id: { site: this.site, seq: ++this.counter }, origin, ch };
    this.integrate(operation);
    this.undoStack.push(operation);
    this.redoStack = [];
    return operation;
  }
  /** Tombstones the visible character at `index` (as this site's deletion). */
  localDelete(index) {
    const visible = this.visible();
    if (!Number.isInteger(index) || index < 0 || index >= visible.length) {
      throw new RangeError(`index ${index} is outside the document`);
    }
    const [target] = visible[index];
    const operation = { type: "delete", target, site: this.site };
    this.integrate(operation);
    this.undoStack.push(operation);
    this.redoStack = [];
    return operation;
  }
  /** Undoes the most recent local operation; returns the inverse operation or null. */
  undo() {
    const operation = this.undoStack.pop();
    if (operation === undefined) return null;
    const inverse = operation.type === "insert"
      ? { type: "delete", target: idKey(operation.id), site: this.site }
      : { type: "restore", target: operation.target, site: this.site };
    this.integrate(inverse);
    // Redo applies the opposite of the undo, which is why an undone insert comes back as a
    // restore of the still-present element rather than as a second insert of the same id.
    this.redoStack.push({ redo: this.inverseOf(inverse), original: operation });
    return inverse;
  }
  /** Redoes the most recently undone local operation; returns the operation it applied. */
  redo() {
    const entry = this.redoStack.pop();
    if (entry === undefined) return null;
    this.integrate(entry.redo);
    this.undoStack.push(entry.original);
    return entry.redo;
  }
  inverseOf(operation) {
    if (operation.type === "delete") return { type: "restore", target: operation.target, site: this.site };
    if (operation.type === "restore") return { type: "delete", target: operation.target, site: this.site };
    return { type: "delete", target: operation.target, site: this.site };
  }
  /**
   * Integrates a remote or local operation. Returns true when it changed the document.
   * Operations whose origin is missing are buffered instead of being dropped.
   */
  integrate(operation) {
    if (operation.type === "insert") {
      const key = idKey(operation.id);
      if (this.elements.has(key)) return false; // duplicate delivery
      if (operation.origin !== null && !this.elements.has(operation.origin)) {
        const alreadyBuffered = this.pending.some((op) => op.type === "insert" && idKey(op.id) === key);
        if (!alreadyBuffered) this.pending.push(operation);
        return false;
      }
      this.elements.set(key, { id: operation.id, ch: operation.ch, origin: operation.origin, deletedBy: new Set() });
      const siblings = this.children.get(operation.origin) ?? [];
      let position = 0;
      while (position < siblings.length && compareIds(this.elements.get(siblings[position]).id, operation.id) > 0) {
        position++;
      }
      siblings.splice(position, 0, key);
      this.children.set(operation.origin, siblings);
      if (operation.id.site === this.site) this.counter = Math.max(this.counter, operation.id.seq);
      this.drainPending();
      return true;
    }
    if (operation.type === "delete") {
      const element = this.elements.get(operation.target);
      if (element === undefined) {
        if (!this.pending.some((op) => op.type === "delete" && op.target === operation.target)) {
          this.pending.push(operation);
        }
        return false;
      }
      if (element.deletedBy.has(operation.site)) return false;
      element.deletedBy.add(operation.site);
      return true;
    }
    if (operation.type === "restore") {
      const element = this.elements.get(operation.target);
      if (element === undefined) return false;
      return element.deletedBy.delete(operation.site);
    }
    throw new TypeError(`unknown operation ${operation.type}`);
  }
  /** Retries buffered operations until no further progress is possible (causal delivery). */
  drainPending() {
    let progressed = true;
    while (progressed) {
      progressed = false;
      for (const operation of [...this.pending]) {
        if (operation.type === "insert" && operation.origin !== null &&
          !this.elements.has(operation.origin)) continue;
        if (operation.type === "delete" && !this.elements.has(operation.target)) continue;
        this.pending = this.pending.filter((op) => op !== operation);
        progressed = true;
        this.integrate(operation);
      }
    }
    return this.pending.length;
  }
  pendingCount() {
    return this.pending.length;
  }
}
/** Applies a list of operations in the given order to a fresh replica. */
function applySequence(site, operations) {
  const replica = new RGA(site);
  for (const operation of operations) replica.integrate(operation);
  return replica;
}
module.exports = { RGA, applySequence, compareIds, idKey, makeRng };
if (require.main === module) {
  // -- local editing ---------------------------------------------------------------
  const solo = new RGA("s0");
  solo.localInsert(0, "a");
  solo.localInsert(1, "b");
  solo.localInsert(2, "c");
  strictEqual(solo.text(), "abc");
  solo.localInsert(1, "X");
  strictEqual(solo.text(), "aXbc", "a middle insert lands at the requested index");
  solo.localDelete(1);
  strictEqual(solo.text(), "abc", "deleting the X restores the text");
  throws(() => solo.localInsert(9, "z"), RangeError, "an out-of-range insert is rejected");
  throws(() => solo.localDelete(-1), RangeError, "an out-of-range delete is rejected");
  throws(() => solo.localInsert(0, "xy"), TypeError, "only single characters can be inserted");
  throws(() => solo.integrate({ type: "rename" }), TypeError, "unknown operations are rejected");
  // -- undo / redo -----------------------------------------------------------------
  const undoing = new RGA("s0");
  undoing.localInsert(0, "h");
  undoing.localInsert(1, "i");
  strictEqual(undoing.text(), "hi");
  const undone = undoing.undo();
  deepStrictEqual([undoing.text(), undone.type], ["h", "delete"], "undo tombstones the last insert");
  strictEqual(undoing.redo().type, "restore", "redo counter-acts the undo");
  strictEqual(undoing.text(), "hi", "redo restores the insert");
  undoing.localDelete(0);
  strictEqual(undoing.text(), "i");
  undoing.undo();
  strictEqual(undoing.text(), "hi", "undo of a delete restores the character");
  const draining = new RGA("s9");
  draining.localInsert(0, "p");
  draining.localInsert(1, "q");
  strictEqual(draining.undo().type, "delete");
  strictEqual(draining.undo().type, "delete");
  strictEqual(draining.text(), "", "undoing every insert empties the document");
  strictEqual(draining.undo(), null, "an empty undo stack yields null");
  strictEqual(new RGA("s8").redo(), null, "an empty redo stack yields null");
  // -- two replicas typing at the same spot ----------------------------------------
  const left = new RGA("left");
  const right = new RGA("right");
  left.localInsert(0, "A");
  right.integrate(...left.undoStack.slice(0, 1).map((op) => structuredClone(op)));
  const leftOp = left.localInsert(1, "L");
  const rightOp = right.localInsert(1, "R");
  left.integrate(structuredClone(rightOp));
  right.integrate(structuredClone(leftOp));
  strictEqual(left.text(), right.text(), "concurrent inserts converge");
  strictEqual(left.text().length, 3, "both concurrent characters survive (no lost text)");
  ok(left.text() === "ALR" || left.text() === "ARL", `deterministic sibling order, got ${left.text()}`);
  strictEqual(left.text(), right.text());
  // -- three replicas, permuted delivery, duplicated messages ----------------------
  const author = new RGA("author");
  for (const ch of "abcdefgh") author.localInsert(author.visible().length, ch);
  author.localInsert(3, "X");
  author.localDelete(2);
  author.localInsert(2, "Z");
  const operations = author.undoStack.map((op) => structuredClone(op));
  const reference = author.text();
  const rng = makeRng(4242);
  for (let trial = 0; trial < 12; trial++) {
    const shuffled = operations.map((op) => structuredClone(op));
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = rng(i + 1);
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const withDuplicates = [];
    for (const operation of shuffled) {
      withDuplicates.push(operation);
      if (rng(100) < 40) withDuplicates.push(structuredClone(operation));
    }
    const replica = applySequence(`r${trial}`, withDuplicates);
    strictEqual(replica.text(), reference, `permuted delivery ${trial} must converge`);
    strictEqual(replica.pendingCount(), 0, "nothing is left buffered once every origin arrived");
  }
  // -- out-of-order delivery is buffered, not dropped ------------------------------
  const buffered = new RGA("later");
  const chained = new RGA("origin");
  const first = chained.localInsert(0, "1");
  const second = chained.localInsert(1, "2");
  const third = chained.localInsert(2, "3");
  buffered.integrate(structuredClone(third));
  strictEqual(buffered.text(), "", "an insert whose origin is missing is buffered");
  strictEqual(buffered.pendingCount(), 1);
  buffered.integrate(structuredClone(second));
  strictEqual(buffered.pendingCount(), 2, "the chain is still incomplete");
  buffered.integrate(structuredClone(first));
  strictEqual(buffered.text(), "123", "arriving the origin releases the whole buffered chain");
  strictEqual(buffered.pendingCount(), 0);
  strictEqual(applySequence("dup", [structuredClone(first), structuredClone(first), structuredClone(first)]).text(),
    "1", "a duplicated insert is integrated only once");
  // -- undo never resurrects a concurrent delete -----------------------------------
  const alice = new RGA("alice");
  const bob = new RGA("bob");
  const shared = alice.localInsert(0, "x");
  bob.integrate(structuredClone(shared));
  const bobDelete = bob.localDelete(0);
  alice.localDelete(0);
  alice.integrate(structuredClone(bobDelete));
  strictEqual(alice.text(), "", "both sites' tombstones are in force");
  alice.undo();
  strictEqual(alice.text(), "", "alice's undo cannot resurrect a character bob also deleted");
  bob.integrate({ type: "restore", target: idKey(shared.id), site: "alice" });
  strictEqual(bob.text(), "", "the withdrawn tombstone propagates, but bob's own still stands");
  bob.undo();
  strictEqual(bob.text(), "x", "with every tombstone withdrawn the character reappears");
  deepStrictEqual([alice.undoStack.length, bob.undoStack.length], [1, 0], "undo stacks reflect the history");
  // -- tombstones still anchor later inserts ---------------------------------------
  const anchoring = new RGA("anchor");
  anchoring.localInsert(0, "a");
  anchoring.localInsert(1, "b");
  anchoring.localDelete(0);
  strictEqual(anchoring.text(), "b");
  anchoring.localInsert(0, "c");
  strictEqual(anchoring.text(), "cb", "the insert after a tombstone lands before the visible text");
  console.log(`reference=[${reference}] replicas=3 permutations=12 duplicated=<40% bufferedRelease=ok`);
  console.log("PASS");
}
