/** Task 634 - Implement CRDT sequence (RGA / Logoot) basics for collaborative editing. (Tier: Expert, CTCI Beyond — Distributed Systems)
 *
 * RGA (Replicated Growable Array) makes a text sequence a CRDT by giving every character a unique
 * identifier `(counter, replica)` and an *origin*: the identifier of the character it was inserted
 * after. Insertion only ever names an existing character, and the document order is defined as a
 * pre-order walk of the resulting insertion tree with siblings ordered by identifier - a pure
 * function of the set of characters, so any two replicas that received the same operations render the
 * same text whatever the delivery order. Deleted characters stay behind as *tombstones* so that later
 * inserts can still name them as anchors, which is what keeps concurrent edits in place.
 *
 * Logoot is the position-identifier alternative: a position is a list of digits plus a suffix, and
 * `logootBetween(left, right, suffix)` mints a fresh position strictly between two neighbours, so a
 * plain sorted list of (position, character) converges without any tree.
 *
 * What is simplified: identifiers are small structs rather than compressed (counter, replica) pairs,
 * operations are plain objects delivered by explicit calls, and the walk rebuilds the order on each
 * render instead of maintaining a linked list - the demo verifies convergence and the anchoring
 * rules, not the asymptotics. There are no timers, threads or randomness beyond a fixed-seed LCG.
 *
 * Verified invariants: (1) every permutation of the same operation set renders the same text;
 * (2) concurrent runs of appends stay contiguous (no interleaving) and the tie-break is deterministic;
 * (3) a delete removes the character from the text but keeps its anchor role, and applying a delete or
 * an insert twice changes nothing; (4) Logoot positions are unique, strictly ordered and placed
 * between their neighbours, so a sorted list converges as well.
 */
"use strict";

const ROOT = { counter: 0, replica: "" };
const LOGOOT_BASE = 10;

function compareIds(a, b) {
  if (a.counter !== b.counter) return a.counter - b.counter;
  return a.replica < b.replica ? -1 : a.replica > b.replica ? 1 : 0;
}

function idKey(id) {
  if (id === null) return "root";
  return `${id.counter}@${id.replica}`;
}

class RgaDocument {
  constructor(replica) {
    this.replica = replica;
    this.counter = 0;
    this.pending = [];
    this.nodes = new Map(); // idKey -> { id, originKey, value, deleted, children: [] }
    this.nodes.set("root", { id: null, originKey: null, value: null, deleted: false, children: [] });
  }
  /** Local insert: a fresh identifier, placed after `afterId` (null = document start). */
  insertAfter(value, afterId = null) {
    this.counter++;
    const id = { counter: this.counter, replica: this.replica };
    this.apply({ kind: "insert", id, origin: afterId, value });
    return id;
  }
  /** Applies a remote operation; operations whose origin has not arrived are buffered, and calling
   *  this with the same operation twice changes nothing (identifiers are unique). */
  apply(operation) {
    const key = idKey(operation.id);
    if (operation.kind === "delete") {
      const node = this.nodes.get(key);
      if (node === undefined) {
        this.pending.push(operation); // the insert has not arrived yet: remember the removal
        return false;
      }
      if (node.id === null) return false;
      const already = node.deleted;
      node.deleted = true;
      return !already;
    }
    if (this.nodes.has(key)) return false; // already applied
    if (!this.nodes.has(idKey(operation.origin))) {
      this.pending.push(operation); // out-of-order delivery: wait for the anchor
      return false;
    }
    const originParent = this.nodes.get(idKey(operation.origin));
    const node = { id: operation.id, originKey: idKey(operation.origin), value: operation.value, deleted: false, children: [] };
    this.nodes.set(key, node);
    originParent.children.push(operation.id);
    // siblings are ordered by identifier, larger first: RGA keeps a replica's run contiguous
    originParent.children.sort((a, b) => compareIds(b, a));
    this.flushPending();
    return true;
  }
  flushPending() {
    let progress = true;
    while (progress) {
      progress = false;
      for (let index = 0; index < this.pending.length; index++) {
        const operation = this.pending[index];
        if (operation.kind === "insert" && this.nodes.has(idKey(operation.id))) {
          this.pending.splice(index--, 1);
          continue;
        }
        if (operation.kind === "insert" && !this.nodes.has(idKey(operation.origin))) continue;
        if (operation.kind === "delete" && !this.nodes.has(idKey(operation.id))) continue;
        this.pending.splice(index--, 1);
        progress = true;
        this.apply(operation);
      }
    }
  }
  remove(id) {
    return this.apply({ kind: "delete", id });
  }
  contains(id) {
    return this.nodes.has(idKey(id)) && this.nodes.get(idKey(id)).id !== null;
  }
  render() {
    let text = "";
    const walk = (key) => {
      for (const child of this.nodes.get(key).children) {
        const node = this.nodes.get(idKey(child));
        if (!node.deleted) text += node.value;
        walk(idKey(child));
      }
    };
    walk("root");
    return text;
  }
  liveCount() {
    let count = 0;
    for (const node of this.nodes.values()) if (node.id !== null && !node.deleted) count++;
    return count;
  }
  tombstoneCount() {
    let count = 0;
    for (const node of this.nodes.values()) if (node.id !== null && node.deleted) count++;
    return count;
  }
  operations(ops = []) {
    for (const node of this.nodes.values()) {
      if (node.id === null) continue;
      ops.push({
        kind: "insert", id: node.id,
        origin: node.originKey === "root" ? null : this.nodes.get(node.originKey).id,
        value: node.value,
      });
      if (node.deleted) ops.push({ kind: "delete", id: node.id });
    }
    return ops;
  }
}

function lcg(seed) {
  return (seed * 48271) % 2147483647;
}

function permutations(items, seed, count) {
  const result = [];
  let s = seed;
  for (let trial = 0; trial < count; trial++) {
    const order = items.slice();
    for (let index = order.length - 1; index > 0; index--) {
      s = lcg(s);
      const swap = s % (index + 1);
      [order[index], order[swap]] = [order[swap], order[index]];
    }
    result.push(order);
    s = lcg(s);
  }
  return result;
}

function hashSuffix(suffix) {
  let hash = 3;
  for (let index = 0; index < suffix.length; index++) hash = (hash * 31 + suffix.charCodeAt(index)) % 9973;
  return hash;
}

/** Logoot position generation: the returned path lies strictly between `left` and `right`. */
function logootBetween(left, right, suffix, base = LOGOOT_BASE) {
  const positions = [];
  for (let index = 0; ; index++) {
    const low = index < left.length ? left[index] : 0;
    const high = right !== null && index < right.length ? right[index] : base;
    if (high - low > 1) {
      const span = high - low - 1;
      positions.push(low + 1 + (hashSuffix(suffix) + index) % span);
      return { positions, suffix };
    }
    positions.push(low);
  }
}

function comparePositions(a, b) {
  const length = Math.max(a.positions.length, b.positions.length);
  for (let index = 0; index < length; index++) {
    const left = a.positions[index] === undefined ? -1 : a.positions[index];
    const right = b.positions[index] === undefined ? -1 : b.positions[index];
    if (left !== right) return left - right;
  }
  return a.suffix < b.suffix ? -1 : a.suffix > b.suffix ? 1 : 0;
}

module.exports = { ROOT, compareIds, idKey, RgaDocument, logootBetween, comparePositions, permutations, lcg };

if (require.main === module) {
  const assert = require("node:assert");
  // --- hand-checked convergence: two replicas append concurrently --------------------------
  const alpha = new RgaDocument("A");
  const beta = new RgaDocument("B");
  const a1 = alpha.insertAfter("a");
  const a2 = alpha.insertAfter("a", a1);
  const b1 = beta.insertAfter("b");
  const b2 = beta.insertAfter("b", b1);
  const operations = [
    { kind: "insert", id: a1, origin: null, value: "a" },
    { kind: "insert", id: a2, origin: a1, value: "a" },
    { kind: "insert", id: b1, origin: null, value: "b" },
    { kind: "insert", id: b2, origin: b1, value: "b" },
  ];
  const expected = "bbaa"; // sibling order is by (counter, replica) descending: B:1 before A:1
  assert.strictEqual(alpha.render(), "aa");
  assert.strictEqual(beta.render(), "bb");
  for (const order of permutations(operations, 4242, 12)) {
    const doc = new RgaDocument("X");
    for (const operation of order) doc.apply(operation);
    assert.strictEqual(doc.render(), expected, "the same operations must render the same text");
    assert.strictEqual(doc.liveCount(), 4);
  }
  const merged = new RgaDocument("A");
  for (const operation of alpha.operations()) merged.apply(operation);
  for (const operation of beta.operations()) merged.apply(operation);
  assert.strictEqual(merged.render(), expected, "merging both replicas gives the same text");
  assert.ok(merged.render().includes("bb") && merged.render().includes("aa"), "each run stays contiguous");

  // --- interleaved edits in the middle of an existing document --------------------------------
  const doc = new RgaDocument("A");
  const first = doc.insertAfter("x");
  const second = doc.insertAfter("y", first);
  assert.strictEqual(doc.render(), "xy");
  const other = new RgaDocument("B");
  for (const operation of doc.operations()) other.apply(operation);
  const middle = doc.insertAfter("m", first); // A inserts between x and y
  const tail = other.insertAfter("z", second); // B appends at the end
  const all = doc.operations().concat(other.operations());
  const texts = new Set();
  for (const order of permutations(all, 777, 20)) {
    const replica = new RgaDocument("Z");
    for (const operation of order) replica.apply(operation);
    texts.add(replica.render());
    assert.strictEqual(replica.liveCount(), 4);
  }
  assert.strictEqual(texts.size, 1, "all delivery orders converge");
  const converged = [...texts][0];
  assert.strictEqual(converged[0], "x");
  assert.strictEqual(converged[converged.length - 1], "z", "the concurrent tail insert stays at the end");
  assert.ok(converged.includes("m") && converged.includes("y"));
  assert.ok(converged.indexOf("m") < converged.indexOf("z"));

  // --- tombstones anchor later inserts --------------------------------------------------------
  const anchored = new RgaDocument("A");
  const anchor = anchored.insertAfter("p");
  const afterAnchor = anchored.insertAfter("q", anchor);
  assert.strictEqual(anchored.render(), "pq");
  anchored.remove(anchor);
  assert.strictEqual(anchored.render(), "q", "the deleted character disappears");
  const revived = anchored.insertAfter("r", anchor); // inserting after a tombstone still works
  assert.strictEqual(anchored.tombstoneCount(), 1);
  assert.strictEqual(anchored.render(), "rq", "the new character takes the tombstone's place");
  assert.strictEqual(anchored.apply({ kind: "delete", id: anchor }), false, "a repeated delete changes nothing");
  assert.strictEqual(anchored.render(), "rq");
  assert.strictEqual(anchored.apply({ kind: "insert", id: revived, origin: anchor, value: "r" }), false,
    "a duplicated insert is ignored");
  assert.strictEqual(anchored.liveCount(), 2);
  assert.strictEqual(compareIds(anchor, afterAnchor) !== 0, true, "identifiers are distinct");
  assert.strictEqual(compareIds(revived, revived), 0, "and compare with themselves");

  // --- boundaries -----------------------------------------------------------------------------
  const empty = new RgaDocument("A");
  assert.strictEqual(empty.render(), "", "an empty document is empty");
  assert.strictEqual(empty.liveCount(), 0);
  const source = new RgaDocument("Q");
  const q1 = source.insertAfter("1");
  const q2 = source.insertAfter("2", q1);
  const sourceOps = source.operations();
  assert.strictEqual(sourceOps.length, 2);
  assert.strictEqual(empty.apply(sourceOps[1]), false, "an insert whose anchor has not arrived is buffered");
  assert.strictEqual(empty.pending.length, 1);
  assert.strictEqual(empty.render(), "", "and does not disturb the text");
  assert.strictEqual(empty.apply(sourceOps[0]), true, "the anchor arrives");
  assert.strictEqual(empty.pending.length, 0, "the buffered insert is applied immediately");
  assert.strictEqual(empty.render(), "12", "in the right place");
  assert.strictEqual(empty.apply({ kind: "delete", id: { counter: 99, replica: "Z" } }), false,
    "a delete for an unknown character is buffered");
  assert.strictEqual(empty.pending.length, 1);
  assert.strictEqual(empty.render(), "12", "and changes nothing");
  empty.remove(q2);
  assert.strictEqual(empty.render(), "1", "the known character is deleted");
  assert.strictEqual(empty.pending.length, 1, "the unknown delete is still waiting");
  assert.strictEqual(empty.apply({ kind: "insert", id: { counter: 99, replica: "Z" }, origin: null, value: "z" }), true);
  assert.strictEqual(empty.pending.length, 0, "once it arrives the buffered delete applies");
  assert.strictEqual(empty.render(), "1", "and removes exactly that character");
  const deleted = new RgaDocument("A");
  const only = deleted.insertAfter("solo");
  deleted.remove(only);
  assert.strictEqual(deleted.render(), "", "deleting everything empties the text");
  assert.strictEqual(deleted.liveCount(), 0);
  assert.strictEqual(deleted.apply({ kind: "delete", id: { counter: 99, replica: "A" } }), false, "unknown deletes are ignored");
  const headInsert = deleted.insertAfter("head");
  assert.strictEqual(deleted.render(), "head", "the tombstone is not rendered");
  assert.strictEqual(compareIds(headInsert, only) > 0, true, "the newer identifier sorts later");

  // --- Logoot positions -----------------------------------------------------------------------
  const start = logootBetween([], null, "A0");
  assert.ok(start.positions.length === 1 && start.positions[0] > 0, "the first position is positive");
  const middlePos = logootBetween(start.positions, null, "A0");
  assert.ok(comparePositions(start, middlePos) < 0, "a generated position is strictly after its left neighbour");
  const before = logootBetween([], start.positions, "A1");
  assert.ok(comparePositions(before, start) < 0, "and strictly before its right neighbour");
  const tightLeft = { positions: [3], suffix: "L0" };
  const tightRight = { positions: [4], suffix: "R0" };
  const tight = logootBetween(tightLeft.positions, tightRight.positions, "M0");
  assert.ok(comparePositions(tightLeft, tight) < 0 && comparePositions(tight, tightRight) < 0,
    "a position can be squeezed between adjacent digits");
  assert.ok(tight.positions.length > 1, "which requires a longer path");
  const positions = [];
  for (let index = 0; index < 12; index++) positions.push(logootBetween([], null, `replica${index}`));
  const unique = new Set(positions.map((position) => `${position.positions.join(",")}#${position.suffix}`));
  assert.strictEqual(unique.size, positions.length, "generated positions are unique");
  const characters = ["h", "e", "l", "l", "o"];
  const logootDoc = characters.map((character, index) => ({ character, position: positions[index] }));
  const ordered = logootDoc.slice().sort((a, b) => comparePositions(a.position, b.position));
  assert.ok(ordered.every((entry) => characters.includes(entry.character)));
  const sortedTwice = logootDoc.slice().reverse().sort((a, b) => comparePositions(a.position, b.position));
  assert.deepStrictEqual(sortedTwice.map((entry) => entry.character), ordered.map((entry) => entry.character),
    "sorting a Logoot list is order of arrival independent");
  assert.notStrictEqual(comparePositions(before, tight), 0, "distinct positions never compare equal");

  console.log(`rga permutations=${12 + 20} text=${converged} tombstones=${anchored.tombstoneCount()}`);
  console.log(`logoot positions=${positions.length} tight-path=${tight.positions.join(".")}`);
  console.log("PASS");
}
