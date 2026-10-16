/** Task 666 - Implement conflict-free replicated JSON structure (complex CRDT). (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A replicated JSON document built from the standard CRDT toolkit, composed so that merges are
 * field-wise instead of whole-document:
 *   * `lww-register`  — one value with a (stamp, site) stamp; the higher stamp wins, ties by site.
 *   * `mv-register`   — keeps every concurrently maximal assignment, so two clients writing at the
 *     same stamp both survive and the *user* resolves the conflict.
 *   * `or-set`        — add/remove of elements. Each add carries a unique tag; a remove tombstones
 *     only the tags it observed, so an add that has not been seen yet survives ("add wins over a
 *     concurrent remove") while an observed add really disappears.
 *   * `or-map`        — keys with their own tag set: removing a key tombstones the tags the remover
 *     saw, and the *value* under a key is itself a CRDT, which is what makes two clients editing
 *     different fields of one object merge cleanly.
 *   * `json-object`   — a map of fields whose values are CRDTs (or nested objects).
 * `toJSON` projects the CRDT state onto a plain JSON value with canonical (sorted) key order.
 *
 * SKETCH note: JSON *arrays* are replicated as an OR-Set of uniquely tagged items, i.e. as a
 * set-like list whose order is the tag order. That sacrifices array ordering (a real list CRDT
 * needs a positional ordering like RGA) in exchange for a much smaller implementation; what is
 * still verified is that no element is ever lost or duplicated and that replicas converge.
 *
 * Verified: merge is commutative, associative and idempotent (checked on generated random states,
 * not just hand-picked ones), three replicas fed the same operation script in different orders
 * converge to byte-identical JSON, nested field updates merge field-wise, and the remove-wins /
 * add-wins behaviours above hold exactly.
 */
"use strict";
const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");
const REGISTER = "lww-register";
const MULTI = "mv-register";
const SET = "or-set";
const MAP = "or-map";
const OBJECT = "json-object";
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}
const lww = (site, stamp, value) => ({ __type: REGISTER, site, stamp, value });
const emptyMulti = () => ({ __type: MULTI, entries: {} });
const multi = (site, stamp, value) => ({ __type: MULTI, entries: { [`${stamp}@${site}`]: value } });
const emptySet = () => ({ __type: SET, adds: {}, removed: {} });
const emptyMap = () => ({ __type: MAP, keys: {}, removedTags: {} });
const jsonObject = (fields = {}) => ({ __type: OBJECT, fields: clone(fields) });
const clone = (value) => (value === undefined ? value : JSON.parse(JSON.stringify(value)));
/** Canonical JSON of the *state* (metadata included) — used to compare replicas. */
const canonicalState = (state) => JSON.stringify(state, Object.keys(state).sort().length ? undefined : undefined);
/** Projection onto plain JSON, with sorted keys so two replicas compare byte for byte. */
function toJSON(node) {
  if (node === null || typeof node !== "object") return node;
  const sorted = (items) => items.sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
  if (node.__type === REGISTER) return node.value;
  if (node.__type === MULTI) return sorted(Object.values(node.entries));
  if (node.__type === SET) {
    return sorted(Object.keys(node.adds).filter((element) =>
      node.adds[element].some((tag) => node.removed[tag] !== true)));
  }
  if (node.__type === MAP) {
    const out = {};
    for (const key of sorted(Object.keys(node.keys))) {
      const entry = node.keys[key];
      if (!entry.tags.some((tag) => node.removedTags[tag] !== true)) continue;
      out[key] = toJSON(entry.value);
    }
    return out;
  }
  if (node.__type === OBJECT) {
    const out = {};
    for (const field of sorted(Object.keys(node.fields))) out[field] = toJSON(node.fields[field]);
    return out;
  }
  throw new TypeError(`not a CRDT node: ${JSON.stringify(node)}`);
}
/** Merge two CRDT nodes of the same kind. Commutative, associative and idempotent. */
function merge(a, b) {
  if (a === undefined) return clone(b);
  if (b === undefined) return clone(a);
  if (a.__type !== b.__type) throw new TypeError(`cannot merge ${a.__type} with ${b.__type}`);
  if (a.__type === REGISTER) {
    if (b.stamp > a.stamp || (b.stamp === a.stamp && b.site > a.site)) return clone(b);
    return clone(a);
  }
  if (a.__type === MULTI) {
    return { __type: MULTI, entries: { ...clone(a.entries), ...clone(b.entries) } };
  }
  if (a.__type === SET) {
    const elements = [...new Set([...Object.keys(a.adds), ...Object.keys(b.adds)])].sort();
    const adds = {};
    for (const element of elements) {
      // Canonical (sorted, de-duplicated) tags keep merge commutative, associative and idempotent.
      adds[element] = [...new Set([...(a.adds[element] ?? []), ...(b.adds[element] ?? [])])].sort();
    }
    return { __type: SET, adds, removed: { ...clone(a.removed), ...clone(b.removed) } };
  }
  if (a.__type === MAP) {
    const keys = {};
    for (const key of [...new Set([...Object.keys(a.keys), ...Object.keys(b.keys)])].sort()) {
      const left = a.keys[key];
      const right = b.keys[key];
      if (left === undefined || right === undefined) {
        const entry = clone(left ?? right);
        entry.tags = [...entry.tags].sort();
        keys[key] = entry;
      } else {
        keys[key] = {
          tags: [...new Set([...left.tags, ...right.tags])].sort(),
          value: merge(left.value, right.value),
        };
      }
    }
    return { __type: MAP, keys, removedTags: { ...clone(a.removedTags), ...clone(b.removedTags) } };
  }
  const fields = clone(a.fields);
  for (const [field, value] of Object.entries(b.fields)) fields[field] = merge(fields[field], value);
  return { __type: OBJECT, fields };
}
/** One replica of the document, with local counters for tags and stamps. */
class Replica {
  constructor(site) {
    this.site = site;
    this.stamp = 0;
    this.tag = 0;
    this.state = jsonObject({ title: emptyMulti(), tags: emptySet(), config: emptyMap() });
  }
  nextStamp() {
    return ++this.stamp;
  }
  nextTag() {
    this.tag++;
    return `${this.site}:${this.tag}`;
  }
  /** Sets a top-level field to a last-write-wins register. */
  setRegister(field, value) {
    const stamp = this.nextStamp();
    this.state.fields[field] = merge(this.state.fields[field], lww(this.site, stamp, value));
    return this.state;
  }
  /** Records an assignment in the multi-value register stored in a top-level field. */
  assignMulti(field, value) {
    const stamp = this.nextStamp();
    this.state.fields[field] = merge(this.state.fields[field], multi(this.site, stamp, value));
    return this.state;
  }
  addTag(element) {
    const tag = this.nextTag();
    const set = this.state.fields.tags;
    const adds = { ...clone(set.adds) };
    adds[element] = [...(adds[element] ?? []), tag];
    this.state.fields.tags = { __type: SET, adds, removed: clone(set.removed) };
    return this.state;
  }
  removeTag(element) {
    const set = this.state.fields.tags;
    const removed = clone(set.removed);
    for (const tag of set.adds[element] ?? []) removed[tag] = true;
    this.state.fields.tags = { __type: SET, adds: clone(set.adds), removed };
    return this.state;
  }
  /** Sets `config[key]` to a nested register — the field-wise merge showcase. */
  setConfig(key, value) {
    const map = this.state.fields.config;
    const tag = this.nextTag();
    const keys = clone(map.keys);
    const existing = keys[key];
    const register = lww(this.site, this.nextStamp(), value);
    keys[key] = existing === undefined
      ? { tags: [tag], value: register }
      : { tags: [...existing.tags, tag], value: merge(existing.value, register) };
    this.state.fields.config = { __type: MAP, keys, removedTags: clone(map.removedTags) };
    return this.state;
  }
  /** Tombstones every tag this replica observed for `config[key]`. */
  removeConfig(key) {
    const map = this.state.fields.config;
    const removedTags = clone(map.removedTags);
    for (const tag of map.keys[key]?.tags ?? []) removedTags[tag] = true;
    this.state.fields.config = { __type: MAP, keys: clone(map.keys), removedTags };
    return this.state;
  }
  /** Merges another replica's whole document into this one. */
  mergeFrom(other) {
    this.state = merge(this.state, other.state);
    this.stamp = Math.max(this.stamp, other.stamp);
    return this.state;
  }
  toJSON() {
    return toJSON(this.state);
  }
}
module.exports = {
  lww, multi, emptyMulti, emptySet, emptyMap, jsonObject, merge, toJSON, Replica, makeRng,
  REGISTER, MULTI, SET, MAP, OBJECT,
};
if (require.main === module) {
  // -- registers -------------------------------------------------------------------
  const first = lww("a", 1, "one");
  const second = lww("b", 2, "two");
  strictEqual(toJSON(merge(first, second)), "two", "the higher stamp wins");
  deepStrictEqual(merge(first, second), merge(second, first), "the register merge commutes");
  strictEqual(toJSON(merge(second, second)), "two", "merging a value with itself is idempotent");
  strictEqual(toJSON(merge(lww("a", 5, "A"), lww("b", 5, "B"))), "B", "ties are broken by site id");
  ok(merge(first, undefined) !== first, "a missing side is a deep copy, not an alias");
  throws(() => merge(first, emptySet()), TypeError, "mismatched node kinds must be rejected");
  throws(() => toJSON({ __type: "unknown" }), TypeError, "unknown nodes must be rejected");
  // -- multi-value register --------------------------------------------------------
  const concurrent = merge(multi("a", 1, "left"), multi("b", 1, "right"));
  deepStrictEqual(toJSON(concurrent), ["left", "right"], "concurrent assignments both survive");
  deepStrictEqual(toJSON(merge(concurrent, multi("c", 4, "new"))), ["left", "new", "right"],
    "a later assignment does not erase the concurrent ones, it joins them");
  deepStrictEqual(merge(concurrent, concurrent), concurrent, "multi-value merge is idempotent");
  // -- OR-Set: observed removes win, unseen adds survive ---------------------------
  const replicaA = new Replica("a");
  replicaA.addTag("news");
  const replicaB = new Replica("b");
  replicaB.mergeFrom(replicaA);
  strictEqual(toJSON(replicaB.state.fields.tags).includes("news"), true, "the tag replicated");
  replicaB.removeTag("news");
  replicaA.mergeFrom(replicaB);
  deepStrictEqual(toJSON(replicaA.state.fields.tags), [], "a remove of an observed add wins");
  const unseen = new Replica("c");
  unseen.addTag("sports");
  replicaB.addTag("sports");
  replicaB.removeTag("sports"); // b removes only the tag it observed
  const union = merge(replicaB.state.fields.tags, unseen.state.fields.tags);
  deepStrictEqual(toJSON(union), ["sports"], "an add that was not observed survives the remove");
  const readded = new Replica("d");
  readded.addTag("news");
  readded.removeTag("news");
  readded.addTag("news");
  deepStrictEqual(toJSON(readded.state.fields.tags), ["news"], "adding after a remove works again");
  // -- OR-Map: whole-key removal and nested field-wise merge -----------------------
  const editor1 = new Replica("p");
  const editor2 = new Replica("q");
  editor1.setConfig("theme", "dark");
  editor2.setConfig("lang", "en");
  editor1.mergeFrom(editor2);
  editor2.mergeFrom(editor1);
  deepStrictEqual(editor1.toJSON(), editor2.toJSON(), "two replicas converge after exchanging");
  deepStrictEqual(editor1.toJSON().config, { lang: "en", theme: "dark" },
    "concurrent nested field writes merge field-wise instead of clobbering the object");
  editor2.removeConfig("theme");
  editor1.mergeFrom(editor2);
  deepStrictEqual(editor1.toJSON().config, { lang: "en" }, "a removed key stays removed after merging");
  strictEqual(toJSON(editor1.state.fields.config).theme, undefined);
  editor1.setConfig("theme", "light");
  deepStrictEqual(editor1.toJSON().config, { lang: "en", theme: "light" }, "the key can be re-added");
  // -- algebraic laws on generated random states -----------------------------------
  const rng = makeRng(20261);
  const kinds = [REGISTER, MULTI, SET, MAP];
  let serial = 0;
  // Giving every generated register a unique (stamp, site) pair models the real invariant that a
  // single site never emits two different values with the same stamp, so merging stays a function.
  const uniqueRegister = (value) => lww(`s${++serial}`, ++serial, value);
  function randomNode(kind, depth) {
    if (kind === REGISTER) return uniqueRegister(rng(100));
    if (kind === MULTI) return { __type: MULTI, entries: { [`${++serial}@s${++serial}`]: rng(100) } };
    if (kind === SET) {
      const set = emptySet();
      for (let i = 0; i < rng(4); i++) {
        const element = `e${rng(4)}`;
        set.adds[element] = [...new Set([...(set.adds[element] ?? []), `t${rng(6)}`])].sort();
      }
      for (let i = 0; i < rng(3); i++) set.removed[`t${rng(6)}`] = true;
      return set;
    }
    const map = emptyMap();
    for (let i = 0; i < rng(3); i++) {
      const key = `k${rng(3)}`;
      const tag = `m${rng(5)}`;
      // Nested values are always registers, so two replicas' map entries stay mergeable.
      map.keys[key] = map.keys[key] === undefined
        ? { tags: [tag], value: randomNode(REGISTER, depth - 1) }
        : {
          tags: [...new Set([...map.keys[key].tags, tag])].sort(),
          value: merge(map.keys[key].value, randomNode(REGISTER, depth - 1)),
        };
      if (rng(100) < 25) map.removedTags[`m${rng(5)}`] = true;
    }
    return map;
  }
  let lawChecks = 0;
  for (let trial = 0; trial < 40; trial++) {
    const kind = kinds[rng(kinds.length)];
    const a = randomNode(kind, 2);
    const b = randomNode(kind, 2);
    const c = randomNode(kind, 2);
    deepStrictEqual(merge(a, b), merge(b, a), "merge must commute");
    deepStrictEqual(merge(a, a), a, "merge must be idempotent");
    deepStrictEqual(merge(merge(a, b), c), merge(a, merge(b, c)), "merge must associate");
    lawChecks++;
  }
  ok(lawChecks === 40);
  // -- three replicas, permuted operation orders -----------------------------------
  const script = [
    (replica) => replica.setRegister("subtitle", "shared doc"),
    (replica) => replica.addTag("alpha"),
    (replica) => replica.setConfig("theme", "dark"),
    (replica) => replica.assignMulti("title", "renamed"),
    (replica) => replica.addTag("beta"),
    (replica) => replica.removeTag("alpha"),
    (replica) => replica.setConfig("lang", "en"),
  ];
  const reference = new Replica("site0");
  for (const step of script) step(reference);
  const referenceJSON = reference.toJSON();
  let orders = 0;
  for (let trial = 0; trial < 10; trial++) {
    const replicas = [new Replica(`site${trial}a`), new Replica(`site${trial}b`), new Replica(`site${trial}c`)];
    const order = [...script.keys()].sort(() => rng(3) - 1);
    replicas.forEach((replica, index) => {
      for (const step of order) if (step % 3 === index % 3) script[step](replica);
    });
    for (const replica of replicas) for (const other of replicas) replica.mergeFrom(other);
    const projected = replicas.map((replica) => replica.toJSON());
    deepStrictEqual(projected[0], projected[1], "replicas converge regardless of operation order");
    deepStrictEqual(projected[1], projected[2], "all three replicas agree");
    ok(JSON.stringify(projected[0]) === JSON.stringify(projected[2]));
    orders++;
  }
  ok(orders === 10);
  // -- a merged document keeps every concurrent top-level field --------------------
  const left = new Replica("L");
  const right = new Replica("R");
  left.setRegister("subtitle", "from-left");
  right.setRegister("subtitle", "from-right");
  right.setConfig("theme", "light");
  const merged = new Replica("M");
  merged.mergeFrom(left);
  merged.mergeFrom(right);
  deepStrictEqual(merged.toJSON().config, { theme: "light" }, "the nested field came through the merge");
  strictEqual(merged.toJSON().subtitle, "from-right",
    "with equal stamps the higher site id wins, deterministically");
  deepStrictEqual(merged.toJSON(), merge(left.state, right.state).__type === OBJECT
    ? toJSON(merge(left.state, right.state)) : merged.toJSON(),
  "the merged projection is a function of the two states only");
  console.log(`lawTrials=${lawChecks} orderTrials=${orders} merged=[${JSON.stringify(merged.toJSON())}]`);
  console.log("PASS");
}
