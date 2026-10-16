/** Task 925 - Implement hybrid transactional/analytical processing demo (HTAP) with maintenance.
 *  (Tier: Master, CTCI Ch 14 - Databases)
 *
 * One engine serving both workloads:
 *   row store (OLTP)    versioned rows; every mutation closes the live version with `to` and (for an
 *                       insert/update) opens a fresh one, so point reads work at any snapshot of the
 *                       recent past (MVCC time travel);
 *   column store (OLAP) group-by aggregates kept incrementally: count and sum move by +-1/+-amount,
 *                       a group disappears when its count reaches zero;
 *   refresh             a full recomputation from the row store. It is both the correctness oracle
 *                       for the incremental view and the cost baseline: a mutation touches O(1)
 *                       view cells while a refresh scans every row version.
 * Two independent read paths (replaying the mutation history vs. scanning the version chains) must
 * agree on every snapshot. Simplification: single-threaded, one table, integer measures, no WAL and
 * no recovery. `count` and `sum` are maintained exactly; `min`/`max` are monotone superset bounds
 * that only a refresh tightens, and the checks assert exactly that weaker invariant.
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17; x ^= x << 5;
    return (this.state = x >>> 0);
  }
  int(bound) { return this.next() % bound; }
}

const emptyAggregate = () => ({ count: 0, sum: 0, min: Infinity, max: -Infinity });

function addRow(aggregate, group, amount) {
  let cell = aggregate.get(group);
  if (!cell) { cell = emptyAggregate(); aggregate.set(group, cell); }
  cell.count++;
  cell.sum += amount;
  cell.min = Math.min(cell.min, amount);
  cell.max = Math.max(cell.max, amount);
}

const sameCell = (a, b) =>
  a.count === b.count && a.sum === b.sum && a.min === b.min && a.max === b.max;

/** Equality on the exactly maintained statistics (count, sum). */
function sameAggregates(a, b) {
  if (a.size !== b.size) return false;
  for (const [group, cell] of a) {
    const other = b.get(group);
    if (!other || cell.count !== other.count || cell.sum !== other.sum) return false;
  }
  return true;
}

function sameExtremes(a, b) {
  if (a.size !== b.size) return false;
  for (const [group, cell] of a) {
    const other = b.get(group);
    if (!other || cell.min !== other.min || cell.max !== other.max) return false;
  }
  return true;
}

class HtapStore {
  constructor() {
    this.rows = new Map();   // key -> version chain [{from, to|null, group, amount}]
    this.view = new Map();   // incrementally maintained group -> {count, sum, min, max}
    this.now = 0;
    this.mutations = 0;
    this.viewWork = 0;       // view cells touched by incremental maintenance
    this.refreshWork = 0;    // row versions scanned by the last full refresh
    this.history = [];       // (op, key, group, amount, at) log for snapshot replay
  }
  tick() { return ++this.now; }
  /** The live version of a key, or undefined when the key is absent or deleted. */
  current(key) {
    const chain = this.rows.get(key);
    if (!chain || chain.length === 0) return undefined;
    const last = chain[chain.length - 1];
    return last.to === null ? last : undefined;
  }
  /** Incremental maintenance of one cell; a group with a non-positive count is dropped. */
  applyToView(group, amount, sign) {
    let cell = this.view.get(group);
    if (!cell) {
      if (sign < 0) return; // nothing to remove
      cell = emptyAggregate();
      this.view.set(group, cell);
    }
    cell.count += sign;
    cell.sum += sign * amount;
    cell.min = Math.min(cell.min, amount);
    cell.max = Math.max(cell.max, amount);
    this.viewWork++;
    if (cell.count <= 0) this.view.delete(group);
  }
  insert(key, group, amount) {
    assert.ok(!this.current(key), `key ${key} already exists`);
    const at = this.tick();
    const chain = this.rows.get(key) ?? [];
    chain.push({ from: at, to: null, group, amount });
    this.rows.set(key, chain);
    this.applyToView(group, amount, 1);
    this.mutations++;
    this.history.push({ op: "insert", key, group, amount, at });
    return at;
  }
  remove(key) {
    const live = this.current(key);
    assert.ok(live, `cannot delete the unknown key ${key}`);
    const at = this.tick();
    live.to = at;
    this.applyToView(live.group, live.amount, -1);
    this.mutations++;
    this.history.push({ op: "remove", key, group: live.group, amount: live.amount, at });
    return at;
  }
  update(key, group, amount) {
    const live = this.current(key);
    assert.ok(live, `cannot update the unknown key ${key}`);
    const at = this.tick();
    live.to = at;
    this.applyToView(live.group, live.amount, -1);
    this.rows.get(key).push({ from: at, to: null, group, amount });
    this.applyToView(group, amount, 1);
    this.mutations++;
    this.history.push({ op: "update", key, group, amount, at });
    return at;
  }
  /** MVCC point read of the version alive at `asOf` (defaults to the newest snapshot). */
  pointRead(key, asOf = this.now) {
    for (const version of this.rows.get(key) ?? []) {
      if (version.from <= asOf && (version.to === null || asOf < version.to)) return version.amount;
    }
    return undefined;
  }
  /** Snapshot aggregate by scanning version chains (independent of the history replay below). */
  snapshotAggregate(asOf) {
    const result = new Map();
    for (const [, chain] of this.rows) {
      for (const version of chain) {
        if (version.from <= asOf && (version.to === null || asOf < version.to)) {
          addRow(result, version.group, version.amount);
        }
      }
    }
    return result;
  }
  /** Snapshot aggregate by replaying the mutation log up to `asOf` (second, independent path). */
  replayAggregate(asOf) {
    const live = new Map();
    for (const event of this.history) {
      if (event.at > asOf) break;
      if (event.op === "remove") live.delete(event.key);
      else live.set(event.key, { group: event.group, amount: event.amount });
    }
    const result = new Map();
    for (const [, row] of live) addRow(result, row.group, row.amount);
    return result;
  }
  /** Full recomputation from the row store: oracle for the view and the cost baseline. */
  refresh() {
    const fresh = new Map();
    this.refreshWork = 0;
    for (const [, chain] of this.rows) {
      for (const version of chain) {
        this.refreshWork++;
        if (version.to === null) addRow(fresh, version.group, version.amount);
      }
    }
    this.view = fresh;
    return fresh;
  }
  viewSnapshot() { return new Map([...this.view].map(([g, c]) => [g, { ...c }])); }
}

function main() {
  const store = new HtapStore();
  const prng = new Prng(20250925);

  // 1. Happy path: the maintenance invariant is re-checked after every mutation.
  const seeds = [["a", "north", 10], ["b", "north", 25], ["c", "south", 5], ["d", "east", 40]];
  for (const [key, group, amount] of seeds) {
    store.insert(key, group, amount);
    assert.ok(sameAggregates(store.viewSnapshot(), store.refresh()), "view drifted after insert");
  }
  assert.strictEqual(store.view.get("north").sum, 35);
  assert.strictEqual(store.view.get("north").count, 2);
  assert.strictEqual(store.view.get("east").max, 40);

  // 2. Updates and deletes are maintained by their delta; no stale cell survives.
  store.update("b", "south", 12);
  assert.strictEqual(store.view.get("north").sum, 10, "the old group kept the moved row");
  assert.strictEqual(store.view.get("north").count, 1);
  assert.strictEqual(store.view.get("south").sum, 17);
  assert.ok(sameAggregates(store.viewSnapshot(), store.refresh()));
  store.remove("c");
  assert.strictEqual(store.view.get("south").count, 1);
  assert.strictEqual(store.view.get("south").sum, 12);
  assert.ok(sameAggregates(store.viewSnapshot(), store.refresh()));

  // 3. Boundary: an emptied group vanishes; an empty table aggregates to nothing.
  store.remove("a");
  assert.strictEqual(store.view.has("north"), false, "an emptied group must vanish from the view");
  const empty = new HtapStore();
  assert.deepStrictEqual([...empty.refresh()], []);
  assert.strictEqual(empty.pointRead("missing"), undefined);
  const single = new HtapStore();
  single.insert("k", "g", -7);
  assert.strictEqual(single.view.get("g").sum, -7);
  assert.strictEqual(single.view.get("g").min, -7);
  assert.strictEqual(single.view.get("g").max, -7);

  // 4. MVCC: the two independent snapshot paths agree, which cross-checks the view itself.
  const marks = [];
  const mvcc = new HtapStore();
  for (let k = 0; k < 12; k++) {
    mvcc.insert(`row${k}`, ["north", "south", "east"][prng.int(3)], 1 + prng.int(50));
    marks.push(mvcc.now);
  }
  mvcc.update("row3", "south", 99);
  marks.push(mvcc.now);
  mvcc.remove("row5");
  marks.push(mvcc.now);
  for (const mark of marks) {
    const scanned = mvcc.snapshotAggregate(mark);
    const replayed = mvcc.replayAggregate(mark);
    assert.ok(sameAggregates(scanned, replayed), `snapshot paths disagree at ${mark}`);
    assert.ok(sameExtremes(scanned, replayed), `snapshot extremes disagree at ${mark}`);
  }
  assert.ok(sameAggregates(mvcc.viewSnapshot(), mvcc.refresh()),
    "the incremental view must agree with the latest full recompute");

  // 5. Time travel: an analytical read at an old snapshot never sees later mutations.
  const history = new HtapStore();
  history.insert("x", "g", 100);
  const early = history.now;
  history.update("x", "g", 250);
  assert.strictEqual(history.pointRead("x", early), 100, "the snapshot must not see the update");
  assert.strictEqual(history.pointRead("x"), 250);
  assert.strictEqual(history.snapshotAggregate(early).get("g").sum, 100);
  assert.strictEqual(history.view.get("g").sum, 250);
  assert.strictEqual(history.snapshotAggregate(early).get("g").count, 1);

  // 6. min/max are monotone superset bounds that only a refresh tightens.
  const bounds = new HtapStore();
  for (const amount of [5, 90, 40]) bounds.insert(`b${amount}`, "g", amount);
  bounds.remove("b90");
  const lagging = { ...bounds.view.get("g") };
  assert.ok(lagging.max >= 90, "the incremental max may lag behind a removal");
  const tightened = bounds.refresh().get("g");
  assert.strictEqual(tightened.max, 40, "a refresh must recompute the true maximum");
  assert.ok(lagging.max >= tightened.max && lagging.min <= tightened.min, "bounds must not tighten");
  assert.ok(lagging.count === tightened.count && lagging.sum === tightened.sum);
  assert.ok(!sameCell(lagging, tightened), "the extreme must actually have been stale");

  // 7. Maintenance cost: O(1) view work per mutation versus an O(versions) refresh.
  const big = new HtapStore();
  for (let k = 0; k < 200; k++) big.insert(`r${k}`, `g${k % 5}`, k);
  const workBefore = big.viewWork;
  big.update("r7", "g1", 7);
  big.remove("r8");
  assert.strictEqual(big.viewWork - workBefore, 3, "a two-row mutation touches three view cells");
  big.refresh();
  assert.strictEqual(big.refreshWork, 201, "a refresh must scan every stored version");
  const perMutation = big.viewWork / big.mutations;
  assert.ok(perMutation <= 2, `incremental maintenance touched ${perMutation} cells per mutation`);
  assert.ok(big.refreshWork > 50 * perMutation,
    `a refresh (${big.refreshWork}) must cost far more than maintenance (${perMutation})`);

  // 8. Invalid input is rejected instead of corrupting the view.
  const strict = new HtapStore();
  strict.insert("k", "g", 1);
  assert.throws(() => strict.insert("k", "g", 1), /already exists/);
  assert.throws(() => strict.remove("nope"), /unknown key/);
  assert.throws(() => strict.update("nope", "g", 1), /unknown key/);

  // 9. Determinism: the same mutation script rebuilds the same view and the same snapshots.
  const script = () => {
    const local = new HtapStore();
    const p = new Prng(31);
    for (let k = 0; k < 30; k++) local.insert(`k${k}`, `g${p.int(4)}`, p.int(100));
    for (let k = 0; k < 10; k++) local.update(`k${k}`, `g${p.int(4)}`, p.int(100));
    return { view: [...local.view], snapshots: [local.snapshotAggregate(20), local.snapshotAggregate(35)] };
  };
  assert.deepStrictEqual(script(), script());

  console.log(`htap: ${store.mutations} mutations, groups ${[...store.view.keys()].join(",")}; ` +
    `incremental work ${big.viewWork} vs refresh ${big.refreshWork} on ${big.rows.size} rows`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, HtapStore, emptyAggregate, sameAggregates, sameExtremes, addRow };
