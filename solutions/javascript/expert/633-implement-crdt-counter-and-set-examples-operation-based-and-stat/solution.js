/** Task 633 - Implement CRDT counter and set examples (operation-based and state-based). (Tier: Expert, CTCI Beyond — Distributed Systems)
 *
 * Conflict-free replicated data types converge because their merge is commutative, associative and
 * idempotent. Three examples are built here:
 *   - `GCounter`, a grow-only counter: each replica owns a slot and merging takes the per-slot
 *     maximum, so the value is the sum of the maxima and never double-counts a replica.
 *   - `PNCounter`, the +/- extension: two grow-only counters, value = P - N.
 *   - `ORSet`, an observed-remove set: an add is a fresh unique *tag* (replica:counter) and a remove
 *     deletes exactly the tags it has seen, never tags from a concurrent add. Merging is set union of
 *     the tag sets, which is why a concurrent add survives a remove (add-wins).
 * The same ORSet state is also driven as an *operation-based* CRDT: replicas exchange the operations
 * themselves (each delivered exactly once) instead of the state, and the checks assert that the two
 * formulations end up in the same state.
 *
 * What is simplified: replicas, partitions and merges are explicit calls driven by a fixed-seed LCG,
 * with no network and no timers; tags are `replica:counter` strings because the demo needs unique
 * identifiers, not globally unique 128-bit ones. The op-based side deliberately ignores causal-order
 * buffering (task 630 covers that): operations are delivered exactly once, in any order, which the
 * OR-set tolerates because every op carries the tags it acts on.
 *
 * Verified invariants: (1) merge is idempotent, commutative and associative for all three types;
 * (2) replicated increments/decrements are never double-counted: a counter merged in any order and
 * any number of times equals the total of the increments minus the decrements; (3) an OR-set remove
 * only removes what it observed, so a concurrent add wins - and the check shows a naive
 * "remove deletes forever" set losing that element; (4) state-based merge and op-based replay agree.
 */
"use strict";

function lcg(seed) {
  return (seed * 48271) % 2147483647;
}

function sortedSignature(map) {
  return [...map.entries()]
    .filter(([, tags]) => tags.size > 0)
    .map(([element, tags]) => `${element}:${[...tags].sort().join(",")}`)
    .sort()
    .join("|");
}

class GCounter {
  constructor(counts = new Map()) {
    this.counts = new Map(counts);
  }
  increment(replica, amount = 1) {
    if (amount < 0) throw new Error("a grow-only counter refuses negative increments");
    this.counts.set(replica, (this.counts.get(replica) || 0) + amount);
    return this;
  }
  value() {
    return [...this.counts.values()].reduce((total, count) => total + count, 0);
  }
  merge(other) {
    const merged = new GCounter(this.counts);
    for (const [replica, count] of other.counts) {
      merged.counts.set(replica, Math.max(merged.counts.get(replica) || 0, count));
    }
    return merged;
  }
  equals(other) {
    const mine = [...this.counts.keys()].sort();
    const theirs = [...other.counts.keys()].sort();
    if (mine.length !== theirs.length) return false;
    return mine.every((replica, index) => replica === theirs[index]
      && this.counts.get(replica) === other.counts.get(replica));
  }
}

class PNCounter {
  constructor(positive = new GCounter(), negative = new GCounter()) {
    this.positive = positive;
    this.negative = negative;
  }
  increment(replica, amount = 1) {
    this.positive.increment(replica, amount);
    return this;
  }
  decrement(replica, amount = 1) {
    this.negative.increment(replica, amount);
    return this;
  }
  value() {
    return this.positive.value() - this.negative.value();
  }
  merge(other) {
    return new PNCounter(this.positive.merge(other.positive), this.negative.merge(other.negative));
  }
  equals(other) {
    return this.positive.equals(other.positive) && this.negative.equals(other.negative);
  }
}

class ORSet {
  constructor(state = new Map(), tombstones = []) {
    this.state = new Map([...state.entries()].map(([element, tags]) => [element, new Set(tags)]));
    this.tombstones = new Set(tombstones); // removes are remembered as tombstones, so they survive a merge
    this.clock = 0;
  }
  /** Every add gets a globally unique tag, so it can never be confused with an earlier add. */
  add(element, replica) {
    this.clock++;
    const tag = `${replica}:${this.clock}`;
    if (!this.state.has(element)) this.state.set(element, new Set());
    this.state.get(element).add(tag);
    return tag;
  }
  /** Remove only the tags this replica has observed; concurrent add tags survive. */
  remove(element) {
    const observed = this.tags(element);
    for (const tag of observed) this.tombstones.add(tag);
    return observed;
  }
  removeTags(element, tags) {
    for (const tag of tags) this.tombstones.add(tag);
  }
  liveTags(element) {
    return [...(this.state.get(element) || new Set())].filter((tag) => !this.tombstones.has(tag)).sort();
  }
  tags(element) {
    return this.liveTags(element);
  }
  contains(element) {
    return this.liveTags(element).length > 0;
  }
  elements() {
    return [...this.state.keys()].filter((element) => this.contains(element)).sort();
  }
  merge(other) {
    const merged = new ORSet(this.state, this.tombstones);
    for (const [element, tags] of other.state) {
      if (!merged.state.has(element)) merged.state.set(element, new Set());
      const target = merged.state.get(element);
      for (const tag of tags) target.add(tag);
    }
    for (const tag of other.tombstones) merged.tombstones.add(tag);
    return merged;
  }
  signature() {
    const live = new Map();
    for (const element of this.state.keys()) {
      const tags = this.liveTags(element);
      if (tags.length > 0) live.set(element, tags);
    }
    return sortedSignature(live);
  }
}

/** Op-based replica: instead of merging state, it replays operations exactly once. */
class ORSetReplica {
  constructor(name) {
    this.name = name;
    this.set = new ORSet();
    this.ops = [];
    this.applied = new Set();
  }
  add(element) {
    const tag = this.set.add(element, this.name);
    return this.record({ kind: "add", element, tag });
  }
  remove(element) {
    const tags = this.set.remove(element);
    return this.record({ kind: "remove", element, tags });
  }
  record(op) {
    const entry = { ...op, id: `${this.name}#${this.ops.length + 1}` };
    this.ops.push(entry);
    this.applied.add(entry.id);
    return entry;
  }
  apply(op) {
    if (this.applied.has(op.id)) return false;
    if (op.kind === "add") {
      if (!this.set.state.has(op.element)) this.set.state.set(op.element, new Set());
      this.set.state.get(op.element).add(op.tag);
    } else {
      this.set.removeTags(op.element, op.tags);
    }
    this.applied.add(op.id);
    this.ops.push(op);
    return true;
  }
}

/** A "naive set" that forgets concurrent adds - the negative control for add-wins. */
class NaiveLwwSet {
  constructor() {
    this.state = new Map();
  }
  add(element) {
    this.state.set(element, true);
  }
  remove(element) {
    this.state.set(element, false);
  }
  contains(element) {
    return this.state.get(element) === true;
  }
  merge(other) {
    // deliberately order dependent: whichever replica is passed last wins
    const merged = new NaiveLwwSet();
    for (const [element, present] of this.state) merged.state.set(element, present);
    for (const [element, present] of other.state) merged.state.set(element, present);
    return merged;
  }
}

module.exports = { GCounter, PNCounter, ORSet, ORSetReplica, NaiveLwwSet, sortedSignature, lcg };

if (require.main === module) {
  const assert = require("node:assert");
  // --- grow-only counter: no double counting -------------------------------------------------
  const a = new GCounter().increment("A", 3).increment("A", 2);
  const b = new GCounter().increment("B", 7);
  const c = new GCounter().increment("C", 1).increment("C", 1);
  assert.strictEqual(a.value(), 5);
  assert.strictEqual(a.merge(b).merge(a).merge(b).value(), 12, "idempotent and commutative");
  assert.strictEqual(a.merge(b).merge(c).value(), 14);
  assert.ok(a.merge(b).merge(c).equals(c.merge(a).merge(b)), "association order does not matter");
  assert.ok(!a.merge(c).equals(b), "different counters stay different");
  assert.throws(() => a.increment("A", -1), Error, "a grow-only counter rejects decrements");
  assert.strictEqual(new GCounter().value(), 0, "an empty counter is zero");
  assert.strictEqual(new GCounter().merge(new GCounter()).value(), 0);
  assert.ok(a.merge(a).equals(a), "merge with itself is a no-op");

  // --- PN counter ----------------------------------------------------------------------------
  const pn = new PNCounter().increment("A", 5).decrement("B", 2).increment("B", 1);
  assert.strictEqual(pn.value(), 4);
  const other = new PNCounter().decrement("A", 10).increment("C", 3);
  assert.strictEqual(pn.merge(other).value(), -3, "concurrent increments and decrements add up");
  assert.ok(pn.merge(other).equals(other.merge(pn)), "PN merges commute");
  assert.strictEqual(pn.value(), 4, "merging does not mutate the operands");
  assert.ok(pn.merge(other).merge(other).equals(pn.merge(other)), "and are idempotent");

  // --- observed-remove set: add-wins ---------------------------------------------------------
  const left = new ORSet();
  let right = new ORSet();
  const tagA1 = left.add("x", "A");
  right = right.merge(left); // the add replicated to the other replica
  assert.strictEqual(right.contains("x"), true, "the add replicated");
  assert.deepStrictEqual(right.tags("x"), [tagA1]);
  const concurrentAdd = left.add("x", "A"); // a fresh tag, created without seeing right's remove
  assert.deepStrictEqual(right.remove("x"), [tagA1], "the remove deletes exactly the tag it observed");
  assert.strictEqual(right.contains("x"), false);
  const merged = left.merge(right);
  assert.strictEqual(merged.contains("x"), true, "the concurrent add wins over the remove");
  assert.deepStrictEqual(merged.tags("x"), [concurrentAdd], "only the surviving tag remains");
  assert.strictEqual(merged.merge(right).signature(), merged.signature(), "merging the loser again changes nothing");
  assert.ok(merged.merge(new ORSet(new Map([["x", [concurrentAdd]]]))).contains("x"), "a redundant replica keeps it");

  // --- a remove only affects the tags that replica observed -----------------------------------
  const first = new ORSet();
  let second = new ORSet();
  const tagY = first.add("y", "A");
  second = second.merge(first);
  assert.deepStrictEqual(first.remove("y"), [tagY], "the remove acts on the observed tag");
  assert.strictEqual(first.contains("y"), false, "so this replica no longer holds the element");
  assert.strictEqual(second.contains("y"), true, "the other replica never observed the remove");
  const reconciled = second.merge(first);
  assert.strictEqual(reconciled.contains("y"), false, "merging the two states applies the observed remove");
  assert.deepStrictEqual(reconciled.elements(), []);
  assert.deepStrictEqual(second.remove("y"), [tagY], "the second replica can remove its own observed tag");

  // --- randomized partitions and merges: everything converges --------------------------------
  let seed = 24681012;
  let signatureChecks = 0;
  let lastReference = null;
  for (let trial = 0; trial < 25; trial++) {
    seed = lcg(seed);
    const replicas = ["A", "B", "C"].map(() => new ORSet());
    const elements = ["e0", "e1", "e2", "e3"];
    for (let step = 0; step < 40; step++) {
      seed = lcg(seed);
      const replicaIndex = seed % replicas.length;
      seed = lcg(seed);
      const element = elements[seed % elements.length];
      seed = lcg(seed);
      if (seed % 100 < 60) replicas[replicaIndex].add(element, ["A", "B", "C"][replicaIndex]);
      else replicas[replicaIndex].remove(element);
    }
    const reference = replicas.reduce((acc, replica) => acc.merge(replica), new ORSet());
    lastReference = reference;
    for (const replica of replicas) {
      assert.strictEqual(replica.merge(reference).signature(), reference.signature(),
        `trial ${trial}: a replica diverged after merging the reference`);
    }
    const shuffledMerge = [replicas[2], replicas[1], replicas[0], replicas[2]]
      .reduce((acc, replica) => acc.merge(replica), new ORSet());
    assert.strictEqual(shuffledMerge.signature(), reference.signature(), `trial ${trial}: merge order matters?`);
    signatureChecks++;
    assert.ok(reference.elements().every((element) => reference.tags(element).length > 0), "no empty elements survive");
  }
  assert.strictEqual(signatureChecks, 25);

  // --- operation-based replicas replay the same operations ------------------------------------
  seed = lcg(seed);
  const opReplicas = ["A", "B", "C"].map((name) => new ORSetReplica(name));
  for (const [index, element] of ["op0", "op1", "op2"].entries()) {
    opReplicas[index % 3].add(element);
    seed = lcg(seed);
    if (seed % 2 === 0) opReplicas[(index + 1) % 3].add(element);
  }
  opReplicas[0].remove("op0"); // A observed its own add only
  for (const source of opReplicas) {
    for (const target of opReplicas) {
      for (const op of source.ops) target.apply(op);
    }
  }
  const stateBased = opReplicas.reduce((acc, replica) => acc.merge(replica.set), new ORSet());
  for (const replica of opReplicas) {
    assert.strictEqual(replica.set.signature(), stateBased.signature(),
      `${replica.name} state-based and op-based views disagree`);
    assert.strictEqual(replica.set.contains("op0"), stateBased.contains("op0"));
  }
  assert.deepStrictEqual(opReplicas[0].set.elements(), stateBased.elements());
  const beforeReplay = opReplicas[0].set.signature();
  for (const op of opReplicas[1].ops) opReplicas[0].apply(op);
  assert.strictEqual(opReplicas[0].set.signature(), beforeReplay, "applying an op twice is a no-op");
  assert.strictEqual(opReplicas[0].applied.size, opReplicas[0].ops.length, "no operation is ever applied twice");

  // --- the naive control loses the concurrent add ---------------------------------------------
  const naiveLeft = new NaiveLwwSet();
  const naiveRight = new NaiveLwwSet();
  naiveLeft.add("z"); // replica A adds z
  naiveRight.remove("z"); // replica B concurrently removes z (it never saw the add)
  const naiveMerged = naiveLeft.merge(naiveRight);
  assert.strictEqual(naiveMerged.contains("z"), false, "the naive set loses the concurrent add");
  assert.strictEqual(naiveRight.merge(naiveLeft).contains("z"), true, "and is order dependent");
  const orSetLeft = new ORSet();
  const orSetRight = new ORSet().merge(orSetLeft);
  orSetLeft.add("z", "A");
  assert.strictEqual(orSetLeft.merge(orSetRight).contains("z"), true, "the OR-set keeps it");

  // --- boundaries ------------------------------------------------------------------------------
  const empty = new ORSet();
  assert.deepStrictEqual(empty.elements(), []);
  assert.strictEqual(empty.contains("nothing"), false);
  assert.strictEqual(empty.remove("nothing").length, 0, "removing an unseen element observes nothing");
  assert.strictEqual(empty.merge(new ORSet()).signature(), "", "the empty signature is empty");
  const onlyRemove = empty.merge(new ORSet(new Map([["ghost", []]])));
  assert.strictEqual(onlyRemove.contains("ghost"), false, "an empty tag set is not a membership");

  console.log(`crdt counters=${a.merge(b).merge(c).value()} or-set elements=${lastReference.elements().length} trials=${signatureChecks}`);
  console.log(`op-based ops=${opReplicas.reduce((total, replica) => total + replica.ops.length, 0)}`);
  console.log("PASS");
}
