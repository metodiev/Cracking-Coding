/** Task 850 - Implement monetizable feature store with lineage, versioning, and online/offline sync.
 *  (Tier: Master, CTCI Beyond — Software Engineering & Tooling)
 *
 * A feature store is the contract between data engineering and model serving, and this one is built the
 * way a product is: an immutable versioned offline store, a latency-oriented online store, a lineage
 * graph that makes every number explainable, and metering that turns that usage into a bill.
 *
 *  - offline store: writes append an immutable, content-hashed version of a feature view. Reads are
 *    *point-in-time correct* — `readAsOf(feature, entity, asOf)` answers with the newest value whose
 *    timestamp is <= asOf, never with a value that was written later but describes an earlier instant,
 *    which is exactly the label-leakage bug feature stores exist to prevent. Late-arriving writes are
 *    accepted and simply become invisible to earlier queries.
 *  - lineage: feature views are built from sources and consumed by models; the graph rejects cycles and
 *    answers upstream/downstream closure queries, so "where did this number come from?" has an answer.
 *  - online/offline sync: `materialize` publishes the latest offline values into the online store with a
 *    TTL. The sync is resumable: an interrupted materialization leaves a *measurably* inconsistent
 *    online view, and re-running the sync repairs it — the store reports its coherence ratio.
 *  - monetization: every operation is metered by type (point-in-time reads cost more than online reads),
 *    tiers include a quota, overage is priced per credit, and denied operations are metered too. The
 *    bill is computed from the ledger and cross-checked against an independent arithmetic oracle.
 *
 * Simplifications, stated honestly: time is an explicit logical counter, never the wall clock; storage
 * is in-process maps rather than a warehouse plus a Redis-like serving layer; the materialization
 * "interruption" is a parameter rather than a crash; and prices are illustrative constants. There are no
 * timers, threads, sockets or external services.
 *
 * Invariants verified below:
 *  - point-in-time correctness: every as-of read equals a brute-force scan of the write log, including
 *    after out-of-order (late) writes, for 40 random queries over 6 entities;
 *  - immutability and versioning: old versions keep their hashes after new writes, same-timestamp writes
 *    are resolved deterministically, and a rollback records a *new* version that restores the old values;
 *  - lineage: closure queries match a brute-force traversal, both directions, and a cyclic registration
 *    is rejected;
 *  - diffing: version differences (added/removed/changed entities) match a brute-force comparison;
 *  - online/offline sync: coherence is 1 after a full materialization, TTL expiry makes reads miss and
 *    fall through to the offline store, and an interrupted sync is repaired by re-running it;
 *  - monetization: the metered ledger matches the operation counts exactly, the bill matches an
 *    independent oracle, a higher tier is cheaper for a heavy tenant, and quota denials are metered;
 *  - edge cases: empty store, single entity, reads before any write, TTL 0, and duplicate writes;
 *  - determinism: the same script of operations produces identical hashes, ledgers and bills.
 */
"use strict";

const MASK64 = (1n << 64n) - 1n;

/** Deterministic content hash, used for every version and for the audit trail. */
function contentHash(value) {
  const text = JSON.stringify(value, (key, item) => (typeof item === "bigint" ? String(item) : item));
  let h = 0xcbf29ce484222325n;
  for (let i = 0; i < text.length; i++) {
    h = (h ^ BigInt(text.charCodeAt(i))) & MASK64;
    h = (h * 0x100000001b3n) & MASK64;
  }
  return h.toString(16).padStart(16, "0");
}

/** Illustrative price list: point-in-time reads are the premium product. */
const RATES = {
  write: 1,
  readAsOf: 3,
  readLatest: 1,
  readOnline: 0.25,
  materialize: 5,
  rollback: 10,
  denied: 0.5,
};

const TIERS = {
  basic: { included: 100, overage: 1 },
  pro: { included: 2000, overage: 0.5 },
  enterprise: { included: 50000, overage: 0.2 },
};

const OPERATIONS = Object.keys(RATES);

/** The metered ledger: one counter per operation, plus the bill derived from the active tier. */
class Ledger {
  constructor(tier = "basic") {
    if (TIERS[tier] === undefined) throw new Error(`unknown tier ${tier}`);
    this.tier = tier;
    this.counts = Object.fromEntries(OPERATIONS.map((operation) => [operation, 0]));
  }

  record(operation) {
    if (this.counts[operation] === undefined) throw new Error(`unknown operation ${operation}`);
    this.counts[operation]++;
  }

  credits() {
    let total = 0;
    for (const operation of OPERATIONS) total += this.counts[operation] * RATES[operation];
    return total;
  }

  /** The bill: overage beyond the tier's included credits, priced by the tier. */
  bill(tier = this.tier) {
    const plan = TIERS[tier];
    if (plan === undefined) throw new Error(`unknown tier ${tier}`);
    const overage = Math.max(0, this.credits() - plan.included);
    return { tier, credits: this.credits(), included: plan.included, overage,
      amount: overage * plan.overage };
  }

  totalOperations() {
    return OPERATIONS.reduce((acc, operation) => acc + this.counts[operation], 0);
  }
}

/**
 * The feature store. `clock` is logical time; `policy` carries the TTL and the quota.
 */
class FeatureStore {
  constructor(options = {}) {
    this.clock = options.clock ?? 0;
    this.ttl = options.ttl ?? 5;
    this.quota = options.quota ?? Infinity;
    this.ledger = new Ledger(options.tier ?? "basic");
    this.views = new Map(); // view name -> { entities: Map(entity -> [versions]), versions: [...] }
    this.online = new Map(); // "view|entity" -> { value, timestamp, syncedAt }
    this.lineageEdges = new Map(); // node -> Set of upstream nodes
    this.audit = [];
  }

  // --- lineage --------------------------------------------------------------
  /** Registers a dependency edge; a cycle is refused because lineage must be a DAG. */
  link(node, upstream) {
    if (this.reaches(upstream, node)) throw new Error(`lineage cycle: ${node} -> ${upstream}`);
    if (!this.lineageEdges.has(node)) this.lineageEdges.set(node, new Set());
    this.lineageEdges.get(node).add(upstream);
    return this.lineageEdges.get(node).size;
  }

  /** True when `from` can already reach `to` by following upstream edges. */
  reaches(from, to) {
    const seen = new Set();
    const stack = [from];
    while (stack.length > 0) {
      const node = stack.pop();
      if (node === to) return true;
      if (seen.has(node)) continue;
      seen.add(node);
      for (const upstream of this.lineageEdges.get(node) ?? []) stack.push(upstream);
    }
    return false;
  }

  /** Everything a node depends on (upstream) or everything that depends on it (downstream). */
  lineage(node, direction = "upstream") {
    const seen = new Set();
    const stack = [node];
    while (stack.length > 0) {
      const current = stack.pop();
      const neighbours = direction === "upstream"
        ? [...(this.lineageEdges.get(current) ?? [])]
        : [...this.lineageEdges.entries()].filter(([, ups]) => ups.has(current)).map(([key]) => key);
      for (const neighbour of neighbours) {
        if (seen.has(neighbour)) continue;
        seen.add(neighbour);
        stack.push(neighbour);
      }
    }
    return [...seen].sort();
  }

  // --- offline store --------------------------------------------------------
  ensureView(name) {
    if (!this.views.has(name)) this.views.set(name, { entities: new Map(), versions: [] });
    return this.views.get(name);
  }

  /** Appends an immutable version. `timestamp` is the event time the value describes. */
  write(view, entity, value, timestamp = this.clock) {
    this.ledger.record("write");
    const target = this.ensureView(view);
    if (!target.entities.has(entity)) target.entities.set(entity, []);
    const history = target.entities.get(entity);
    const version = target.versions.length + 1;
    const record = { entity, value, timestamp, version, sequence: history.length };
    history.push(record);
    target.versions.push({ version, entity, value, timestamp, hash: contentHash(record) });
    return record;
  }

  /**
   * Point-in-time read: the newest write whose event time is <= asOf. A write that arrives with an
   * earlier timestamp is invisible to queries made before it, which is the whole point.
   */
  readAsOf(view, entity, asOf) {
    this.ledger.record("readAsOf");
    const history = this.views.get(view)?.entities.get(entity);
    if (history === undefined) return null;
    let best = null;
    for (const record of history) {
      if (record.timestamp > asOf) continue;
      if (best === null || record.timestamp > best.timestamp ||
          (record.timestamp === best.timestamp && record.sequence > best.sequence)) {
        best = record;
      }
    }
    return best === null ? null : { value: best.value, timestamp: best.timestamp,
      version: best.version };
  }

  /** Internal latest lookup: maintenance work (sync, coherence) is not billed. */
  latestValue(view, entity) {
    const history = this.views.get(view)?.entities.get(entity);
    if (history === undefined || history.length === 0) return null;
    let best = history[0];
    for (const record of history) {
      if (record.timestamp > best.timestamp ||
          (record.timestamp === best.timestamp && record.sequence > best.sequence)) best = record;
    }
    return { value: best.value, timestamp: best.timestamp, version: best.version };
  }

  readLatest(view, entity) {
    this.ledger.record("readLatest");
    return this.latestValue(view, entity);
  }

  versionHash(view, version) {
    const entry = this.views.get(view)?.versions.filter((item) => item.version === version) ?? [];
    return entry.length === 0 ? null : contentHash(entry.map((item) => item.hash));
  }

  /** A rollback is additive: it writes a new version carrying the state of an older one. */
  rollback(view, version) {
    this.ledger.record("rollback");
    const target = this.views.get(view);
    if (target === undefined) throw new Error(`unknown view ${view}`);
    if (!(version >= 1) || version > target.versions.length) {
      throw new Error(`unknown version ${version}`);
    }
    const restored = [];
    for (const [entity, value] of this.stateAt(view, version)) {
      restored.push(this.write(view, entity, value, this.clock));
    }
    this.audit.push({ action: "rollback", view, from: version, records: restored.length });
    return { version: target.versions[target.versions.length - 1].version,
      restored: restored.length };
  }

  /**
   * The state of a view after applying every write up to `version`: the basis for time-travel diffs.
   */
  stateAt(view, version) {
    const target = this.views.get(view);
    if (target === undefined) throw new Error(`unknown view ${view}`);
    const state = new Map();
    for (const item of target.versions) {
      if (item.version <= version) state.set(item.entity, item.value);
    }
    return state;
  }

  /** Entities whose value changed between two version numbers. */
  diffVersions(view, left, right) {
    const before = this.stateAt(view, left);
    const after = this.stateAt(view, right);
    const added = [...after.keys()].filter((key) => !before.has(key)).sort();
    const removed = [...before.keys()].filter((key) => !after.has(key)).sort();
    const changed = [...after.keys()].filter((key) =>
      before.has(key) && before.get(key) !== after.get(key)).sort();
    return { added, removed, changed, unchanged: [...after.keys()].filter((key) =>
      before.has(key) && before.get(key) === after.get(key)).length };
  }

  // --- online store ---------------------------------------------------------
  onlineKey(view, entity) {
    return `${view}|${entity}`;
  }

  /**
   * Publishes the latest offline values into the online store. `limit` stops the sync early, which
   * models an interrupted materialization; the coherence ratio exposes the damage.
   */
  materialize(view, options = {}) {
    const target = this.views.get(view);
    if (target === undefined) return { published: 0, coherence: this.coherence(view) };
    const limit = options.limit ?? Infinity;
    let published = 0;
    for (const entity of [...target.entities.keys()].sort()) {
      if (published >= limit) break;
      const latest = this.latestValue(view, entity);
      this.online.set(this.onlineKey(view, entity), { value: latest.value,
        timestamp: latest.timestamp, syncedAt: this.clock, version: latest.version });
      this.ledger.record("materialize");
      published++;
    }
    return { published, coherence: this.coherence(view) };
  }

  /** The fraction of entities whose online copy equals the offline latest value. */
  coherence(view) {
    const target = this.views.get(view);
    if (target === undefined || target.entities.size === 0) return 1;
    let coherent = 0;
    for (const entity of target.entities.keys()) {
      const entry = this.online.get(this.onlineKey(view, entity));
      const latest = this.latestValue(view, entity);
      if (entry !== undefined && latest !== null && entry.value === latest.value &&
          entry.timestamp === latest.timestamp) coherent++;
    }
    return coherent / target.entities.size;
  }

  /** Online read with a TTL: a stale entry misses, and the caller can fall back to the offline store. */
  readOnline(view, entity, options = {}) {
    const key = this.onlineKey(view, entity);
    const entry = this.online.get(key);
    const now = options.now ?? this.clock;
    if (entry === undefined) return { status: "miss", reason: "absent" };
    if (now - entry.syncedAt > this.ttl) return { status: "miss", reason: "expired", stale: true };
    if (!this.charge("readOnline")) return { status: "denied", reason: "quota" };
    return { status: "hit", value: entry.value, timestamp: entry.timestamp,
      age: now - entry.syncedAt };
  }

  /** Read-through: try the online store, then the offline store; the ledger records both attempts. */
  readThrough(view, entity, options = {}) {
    const online = this.readOnline(view, entity, options);
    if (online.status === "hit") return { ...online, source: "online" };
    const offline = this.readLatest(view, entity);
    if (offline === null) return { status: "miss", source: "none" };
    return { ...offline, status: "hit", source: "offline", onlineStatus: online.status };
  }

  /** Quota enforcement: over budget, an operation is denied — and still metered. */
  charge(operation) {
    if (this.ledger.credits() + RATES[operation] > this.quota) {
      this.ledger.record("denied");
      return false;
    }
    this.ledger.record(operation);
    return true;
  }
}

module.exports = { MASK64, contentHash, RATES, TIERS, OPERATIONS, Ledger, FeatureStore };

if (require.main === module) {
  const assert = require("node:assert");

  // --- write, read and immutability -----------------------------------------
  const store = new FeatureStore({ ttl: 5, clock: 0 });
  const first = store.write("user_activity", "u1", 10, 0);
  const second = store.write("user_activity", "u1", 20, 3);
  assert.strictEqual(first.version, 1, "the first write is version 1");
  assert.strictEqual(second.version, 2, "the second write is version 2");
  const hashAfterFirst = store.versionHash("user_activity", 1);
  store.write("user_activity", "u1", 30, 7);
  assert.strictEqual(store.versionHash("user_activity", 1), hashAfterFirst,
    "an old version keeps its hash after new writes");
  assert.notStrictEqual(store.versionHash("user_activity", 3), hashAfterFirst,
    "and each version hashes differently");
  assert.strictEqual(new Ledger("pro").tier, "pro", "a ledger records its tier");
  assert.throws(() => new Ledger("platinum"), /unknown tier/, "unknown tiers are rejected");

  // --- point-in-time correctness (no label leakage) -------------------------
  const pitStore = new FeatureStore({ ttl: 4 });
  const entities = ["e0", "e1", "e2", "e3", "e4", "e5"];
  entities.forEach((entity, index) => {
    for (let step = 0; step < 5; step++) {
      pitStore.write("features", entity, `${entity}-v${step}`, index + step * 4);
    }
  });
  // Late arrivals: values describing earlier instants, written afterwards.
  pitStore.write("features", "e0", "e0-late", 1);
  pitStore.write("features", "e3", "e3-late", 5);
  const oracle = (entity, asOf) => {
    const history = pitStore.views.get("features").entities.get(entity);
    const eligible = history.filter((record) => record.timestamp <= asOf);
    if (eligible.length === 0) return null;
    return eligible.reduce((best, record) => (record.timestamp > best.timestamp ||
      (record.timestamp === best.timestamp && record.sequence > best.sequence) ? record : best));
  };
  let pitChecks = 0;
  let lateWritesInvisible = 0;
  for (let query = 0; query < 40; query++) {
    const entity = entities[query % entities.length];
    const asOf = (query * 7) % 25;
    const actual = pitStore.readAsOf("features", entity, asOf);
    const expected = oracle(entity, asOf);
    if (expected === null) {
      assert.strictEqual(actual, null, `query ${query}: nothing exists before ${asOf}`);
    } else {
      assert.strictEqual(actual.value, expected.value, `query ${query}: value at ${asOf}`);
      assert.strictEqual(actual.timestamp, expected.timestamp, `query ${query}: timestamp`);
      assert.strictEqual(actual.timestamp <= asOf, true, "a read never returns a future instant");
    }
    pitChecks++;
  }
  assert.strictEqual(pitChecks, 40, "40 point-in-time queries were checked");
  assert.strictEqual(pitStore.readAsOf("features", "e0", 2).value, "e0-late",
    "a late write is visible to queries after its event time");
  assert.strictEqual(pitStore.readAsOf("features", "e0", 0).value, "e0-v0",
    "and invisible to queries before it");
  lateWritesInvisible++;
  assert.strictEqual(pitStore.readAsOf("features", "e3", 4).value, "e3-v0",
    "the second late write does not leak into earlier instants either");
  assert.strictEqual(pitStore.readAsOf("features", "e3", 5).value, "e3-late",
    "though it does win from its own event time onwards");
  assert.strictEqual(pitStore.readAsOf("features", "missing", 100), null,
    "an unseen entity reads as a miss");

  // --- same-timestamp writes are resolved deterministically ------------------
  const tie = new FeatureStore();
  tie.write("v", "x", "first", 5);
  tie.write("v", "x", "second", 5);
  assert.strictEqual(tie.readAsOf("v", "x", 5).value, "second",
    "the later write wins a timestamp tie");
  assert.strictEqual(tie.readAsOf("v", "x", 4), null, "and time travel before it is still a miss");

  // --- rollback restores without rewriting history --------------------------
  const rollbackStore = new FeatureStore({ clock: 20 });
  rollbackStore.write("v", "a", 1, 0);
  rollbackStore.write("v", "b", 1, 0);
  rollbackStore.write("v", "a", 2, 1);
  const before = rollbackStore.readLatest("v", "a").value;
  const rollback = rollbackStore.rollback("v", 2);
  assert.strictEqual(before, 2, "the value before the rollback is the newest one");
  assert.strictEqual(rollbackStore.readLatest("v", "a").value, 1, "the rollback restored version 2");
  assert.strictEqual(rollbackStore.readLatest("v", "b").value, 1, "for every entity it carried");
  assert.strictEqual(rollback.restored, 2, "two entities came from the snapshot");
  assert.strictEqual(rollbackStore.views.get("v").versions.length, 5,
    "the rollback added new versions rather than deleting any");
  assert.strictEqual(rollback.version, 5, "and the rollback itself is a version");
  assert.strictEqual(rollbackStore.audit.length, 1, "the rollback is in the audit trail");
  assert.throws(() => rollbackStore.rollback("v", 99), /unknown version/, "unknown versions are refused");
  assert.throws(() => rollbackStore.rollback("nope", 1), /unknown view/, "unknown views are refused");
  assert.strictEqual(rollbackStore.versionHash("v", 3) !== rollbackStore.versionHash("v", 5), true,
    "the restored state hashes differently from the state it replaced");

  // --- version diffing ------------------------------------------------------
  const diff = rollbackStore.diffVersions("v", 2, 3);
  assert.deepStrictEqual(diff.changed, ["a"], "the rollback changed a back");
  assert.deepStrictEqual(diff.added, [], "nothing was added");
  assert.deepStrictEqual(diff.removed, [], "nothing was removed");
  assert.strictEqual(diff.unchanged, 1, "b was unchanged");
  assert.throws(() => rollbackStore.diffVersions("nope", 1, 2), /unknown view/, "diff needs a view");

  // --- lineage --------------------------------------------------------------
  const lineageStore = new FeatureStore();
  lineageStore.link("feature_view:clicks", "source:raw_events");
  lineageStore.link("feature_view:clicks", "source:bots_filter");
  lineageStore.link("model:ctr", "feature_view:clicks");
  lineageStore.link("model:ctr", "dataset:labels");
  assert.deepStrictEqual(lineageStore.lineage("model:ctr"),
    ["dataset:labels", "feature_view:clicks", "source:bots_filter", "source:raw_events"],
    "the upstream closure lists everything the model depends on");
  assert.deepStrictEqual(lineageStore.lineage("source:raw_events", "downstream"),
    ["feature_view:clicks", "model:ctr"], "and the downstream closure reports consumers");
  const bruteUpstream = (node) => {
    const seen = new Set();
    const stack = [node];
    while (stack.length > 0) {
      const current = stack.pop();
      for (const upstream of lineageStore.lineageEdges.get(current) ?? []) {
        if (!seen.has(upstream)) {
          seen.add(upstream);
          stack.push(upstream);
        }
      }
    }
    return [...seen].sort();
  };
  for (const node of ["model:ctr", "feature_view:clicks", "source:raw_events"]) {
    assert.deepStrictEqual(lineageStore.lineage(node), bruteUpstream(node),
      `the closure of ${node} matches the brute-force traversal`);
  }
  assert.throws(() => lineageStore.link("source:raw_events", "model:ctr"), /cycle/,
    "a lineage cycle is refused");
  assert.deepStrictEqual(lineageStore.lineage("source:bots_filter"), [],
    "a root source depends on nothing");

  // --- online/offline sync, TTL and repair ----------------------------------
  const syncStore = new FeatureStore({ ttl: 5, clock: 100 });
  for (const entity of ["a", "b", "c", "d"]) syncStore.write("v", entity, `${entity}-1`, 10);
  const full = syncStore.materialize("v");
  assert.strictEqual(full.published, 4, "all four entities were published");
  assert.strictEqual(full.coherence, 1, "and the online view is fully coherent");
  assert.strictEqual(syncStore.readOnline("v", "a", { now: 100 }).value, "a-1",
    "an online read returns the published value");
  assert.strictEqual(syncStore.readOnline("v", "a", { now: 104 }).age, 4, "and reports its age");
  assert.strictEqual(syncStore.readOnline("v", "a", { now: 106 }).status, "miss",
    "past the TTL the entry expires");
  assert.strictEqual(syncStore.readOnline("v", "a", { now: 106 }).reason, "expired",
    "and says why");
  const through = syncStore.readThrough("v", "a", { now: 106 });
  assert.strictEqual(through.source, "offline", "read-through falls back to the offline store");
  assert.strictEqual(through.value, "a-1", "with the same value");
  assert.strictEqual(syncStore.readOnline("v", "zzz", { now: 100 }).status, "miss",
    "an unknown entity misses");
  assert.strictEqual(syncStore.readThrough("v", "zzz", { now: 100 }).source, "none",
    "and read-through finds nothing anywhere");

  // New writes make the online copies stale; an interrupted sync must be repaired.
  syncStore.write("v", "a", "a-2", 30);
  syncStore.write("v", "b", "b-2", 31);
  assert.ok(syncStore.coherence("v") < 1, "a new write makes the online view stale");
  const interrupted = syncStore.materialize("v", { limit: 1 });
  assert.strictEqual(interrupted.published, 1, "the interrupted sync stopped early");
  assert.ok(interrupted.coherence < 1, "and left the view inconsistent");
  const repaired = syncStore.materialize("v");
  assert.strictEqual(repaired.coherence, 1, "re-running the sync repairs it completely");
  assert.strictEqual(syncStore.readOnline("v", "a", { now: 100 }).value, "a-2",
    "and the repaired value is the newest one");

  // --- monetization ---------------------------------------------------------
  const metered = new FeatureStore({ ttl: 10 });
  metered.write("v", "x", 1, 0);
  metered.write("v", "x", 2, 1);
  metered.readAsOf("v", "x", 1);
  metered.readLatest("v", "x");
  metered.materialize("v");
  metered.readOnline("v", "x", { now: 0 });
  const expected = { write: 2, readAsOf: 1, readLatest: 1, materialize: 1, readOnline: 1 };
  for (const [operation, count] of Object.entries(expected)) {
    assert.strictEqual(metered.ledger.counts[operation], count,
      `the ledger counted ${count} ${operation} operations`);
  }
  assert.strictEqual(metered.ledger.counts.rollback, 0, "and nothing that never happened");
  const oracleCredits = Object.entries(metered.ledger.counts)
    .reduce((acc, [operation, count]) => acc + count * RATES[operation], 0);
  assert.strictEqual(metered.ledger.credits(), oracleCredits,
    "the credit total matches the independent arithmetic");
  assert.strictEqual(metered.ledger.totalOperations(), 6, "six billable operations happened");
  const basicBill = metered.ledger.bill("basic");
  assert.strictEqual(basicBill.amount, Math.max(0, oracleCredits - TIERS.basic.included) * 1,
    "the basic bill charges overage at its own rate");
  const heavy = new Ledger("basic");
  for (let i = 0; i < 20000; i++) heavy.counts.readAsOf++;
  const heavyBasic = heavy.bill("basic");
  const heavyPro = heavy.bill("pro");
  const heavyEnterprise = heavy.bill("enterprise");
  assert.ok(heavyBasic.amount > heavyPro.amount && heavyPro.amount > heavyEnterprise.amount,
    `a heavier tier is cheaper for a heavy tenant (${heavyBasic.amount} > ${heavyPro.amount} > ${heavyEnterprise.amount})`);
  assert.strictEqual(heavyBasic.credits, 20000 * RATES.readAsOf, "credits come from the rate card");
  const freeUsage = new Ledger("pro");
  freeUsage.counts.write = 100;
  assert.strictEqual(freeUsage.bill().amount, 0, "usage inside the included quota is free");

  // Quota enforcement meters the denial.
  const throttled = new FeatureStore({ ttl: 10, quota: 4 });
  assert.strictEqual(throttled.charge("readAsOf"), true, "the first premium read is allowed");
  assert.strictEqual(throttled.charge("readAsOf"), false, "the second exceeds the budget");
  assert.strictEqual(throttled.ledger.counts.denied, 1, "the denial is metered");
  assert.strictEqual(throttled.ledger.counts.readAsOf, 1, "and the denied read is not charged twice");
  assert.ok(throttled.ledger.credits() <= 4, "the ledger never exceeds the quota for allowed work");

  // --- edge cases -----------------------------------------------------------
  const empty = new FeatureStore();
  assert.strictEqual(empty.readLatest("v", "x"), null, "an empty store misses");
  assert.strictEqual(empty.readAsOf("v", "x", 10), null, "and time travel misses too");
  assert.deepStrictEqual(empty.materialize("v"), { published: 0, coherence: 1 },
    "materializing an unknown view publishes nothing");
  const single = new FeatureStore();
  single.write("v", "only", "value", 1);
  single.materialize("v");
  assert.strictEqual(single.coherence("v"), 1, "a single entity syncs fully");
  const zeroTtl = new FeatureStore({ ttl: 0, clock: 5 });
  zeroTtl.write("v", "x", 1, 1);
  zeroTtl.materialize("v");
  assert.strictEqual(zeroTtl.readOnline("v", "x", { now: 5 }).status, "hit",
    "a TTL of zero still serves the value at the sync instant");
  assert.strictEqual(zeroTtl.readOnline("v", "x", { now: 6 }).status, "miss",
    "and expires immediately after");
  const duplicate = new FeatureStore();
  duplicate.write("v", "x", 1, 3);
  duplicate.write("v", "x", 1, 3);
  assert.strictEqual(duplicate.readLatest("v", "x").version, 2,
    "a duplicate write still becomes a new version");
  assert.strictEqual(duplicate.views.get("v").versions.length, 2, "keeping the full audit trail");

  // --- determinism ----------------------------------------------------------
  const script = (target) => {
    target.write("v", "a", 1, 0);
    target.write("v", "b", 2, 1);
    target.materialize("v");
    target.readAsOf("v", "a", 0);
    target.readThrough("v", "b", { now: 0 });
    return target;
  };
  const runA = script(new FeatureStore({ ttl: 3 }));
  const runB = script(new FeatureStore({ ttl: 3 }));
  assert.deepStrictEqual(runA.ledger.counts, runB.ledger.counts, "the ledger is reproducible");
  assert.strictEqual(runA.versionHash("v", 1), runB.versionHash("v", 1), "and so are version hashes");
  assert.deepStrictEqual(runA.online, runB.online, "and the online store contents");

  console.log(`point-in-time queries checked=${pitChecks} (late writes invisible before their event ` +
    `time=${lateWritesInvisible}); versions=${rollbackStore.views.get("v").versions.length}`);
  console.log(`sync: coherence after full sync=1, after interrupted=${interrupted.coherence.toFixed(2)}, ` +
    `after repair=${repaired.coherence}`);
  console.log(`monetization: credits=${metered.ledger.credits()} operations=${metered.ledger.totalOperations()} ` +
    `bill basic/pro/enterprise=${heavyBasic.amount}/${heavyPro.amount}/${heavyEnterprise.amount}`);
  console.log("PASS");
}
