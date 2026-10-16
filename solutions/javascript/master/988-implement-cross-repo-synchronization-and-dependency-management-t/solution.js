/** Task 988 - Implement cross-repo synchronization and dependency management tool for exercises.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A deterministic resolver for a set of exercise repositories. Each repository declares exercise
 * versions; a version declares the (name, semver-range) pairs it depends on, possibly naming an
 * exercise hosted by another repository. The tool unions those declarations into one graph, refuses
 * cyclic releases, resolves every name to the highest version satisfying *all* accumulated ranges,
 * emits a dependency-first install order and diffs the result against the local checkouts.
 * What is simplified: no network and no git -- repositories, versions and the lockfile are in-memory
 * objects and a "fetch" is a plan entry. Verified: cycle reporting, version selection under
 * intersected ranges, conflicts, topological validity against a naive O(n^2) oracle, dependency-first
 * ordering, lockfile stability on a second sync and order-independence of the plan.
 */
"use strict";
const assert = require("node:assert");
/** splitmix64 - the file's only randomness source, so fixtures are byte-reproducible. */
function splitmix64(seed) {
  let s = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    s = BigInt.asUintN(64, s + 0x9e3779b97f4a7c15n);
    let z = BigInt.asUintN(64, (s ^ (s >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}
function fnv1a(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i += 1) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return h >>> 0;
}
/** Standard semver ordering; releases here are always fully specified (major.minor.patch). */
function compareVersions(a, b) {
  const x = a.split(".").map(Number), y = b.split(".").map(Number);
  for (let i = 0; i < 3; i += 1) if (x[i] !== y[i]) return x[i] < y[i] ? -1 : 1;
  return 0;
}
/** Supported ranges: "*", "1.2.3", ">=1.2.3", "^1.2.3" (same major) and "~1.2.3" (same minor). */
function satisfies(version, range) {
  if (!range || range === "*") return true;
  if (range.startsWith(">=")) return compareVersions(version, range.slice(2)) >= 0;
  if (range.startsWith("^") || range.startsWith("~")) {
    const v = version.split(".").map(Number), r = range.slice(1).split(".").map(Number);
    const sameMinor = range.startsWith("~") ? v[1] === r[1] : true;
    return v[0] === r[0] && sameMinor && compareVersions(version, range.slice(1)) >= 0;
  }
  return compareVersions(version, range) === 0;
}
/** Flat index: exercise name -> releases sorted descending, each with its name-sorted dependency list. */
function buildIndex(repositories) {
  const index = new Map();
  for (const repo of repositories) for (const ex of repo.exercises) {
    const list = index.get(ex.name) || [];
    list.push({ repo: repo.name, version: ex.version,
      deps: (ex.deps || []).slice().sort((a, b) => (a.name < b.name ? -1 : 1)) });
    index.set(ex.name, list);
  }
  for (const list of index.values()) list.sort((a, b) => compareVersions(b.version, a.version));
  return index;
}
/** Depth-first cycle hunt over the union of declared edges; returns the cycle it closes on. */
function findCycle(index) {
  const state = new Map(), stack = [];
  let found = null;
  const visit = (name) => {
    state.set(name, 1);
    stack.push(name);
    for (const dep of index.get(name)[0].deps) {
      if (!index.has(dep.name) || found) continue;
      const s = state.get(dep.name) || 0;
      if (s === 1) { found = stack.slice(stack.indexOf(dep.name)).concat(dep.name); return; }
      if (s === 0) visit(dep.name);
    }
    if (found) return;
    stack.pop();
    state.set(name, 2);
  };
  for (const name of Array.from(index.keys()).sort()) if (!state.get(name) && !found) visit(name);
  return found;
}
/**
 * Resolve every requirement to one release. Ranges accumulate as releases are chosen, so the loop
 * repeats until a fixed point; the iteration budget makes termination provable for a DAG and turns a
 * pathological input into a reported failure instead of a hang.
 */
function resolve(index, requirements) {
  const ranges = new Map(), chosen = new Map(), missing = [];
  const fail = (reason, extra = {}) => ({ ok: false, reason, chosen, missing, ...extra });
  /** True only when a genuinely new range was recorded -- that is what drives the fixed point. */
  const add = (name, range) => {
    if (!index.has(name)) return false;
    const list = ranges.get(name) || [];
    if (list.indexOf(range) !== -1) return false;
    list.push(range);
    ranges.set(name, list);
    return true;
  };
  const ordered = requirements.slice().sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  for (const req of ordered) { if (index.has(req.name)) add(req.name, req.range); else missing.push(req.name); }
  if (missing.length > 0) return fail("unknown-exercise");
  const budget = Array.from(index.values()).reduce((a, l) => a + l.length, 0) + 1;
  for (let round = 0; round < budget; round += 1) {
    let changed = false;
    for (const name of Array.from(ranges.keys()).sort()) {
      const wanted = ranges.get(name).slice();
      const candidate = index.get(name).find((v) => wanted.every((r) => satisfies(v.version, r)));
      if (!candidate) return fail("conflict", { conflicting: name, ranges: wanted });
      if (chosen.get(name) !== candidate.version) { chosen.set(name, candidate.version); changed = true; }
      for (const dep of candidate.deps) {
        if (!index.has(dep.name)) { missing.push(dep.name); return fail("unknown-exercise"); }
        if (add(dep.name, dep.range)) changed = true;
      }
    }
    if (!changed) return { ok: true, reason: "resolved", chosen, missing: [] };
  }
  return fail("no-fixed-point");
}
/** Kahn's algorithm with a lexicographic ready-set, so the install order is a pure function. */
function installOrder(index, chosen) {
  const names = Array.from(chosen.keys()).sort();
  const pending = new Map(names.map((n) => [n, 0])), dependents = new Map(names.map((n) => [n, []]));
  for (const name of names) for (const dep of index.get(name).find((v) => v.version === chosen.get(name)).deps) {
    if (!pending.has(dep.name)) continue;
    pending.set(name, pending.get(name) + 1);
    dependents.get(dep.name).push(name);
  }
  const order = [], ready = names.filter((n) => pending.get(n) === 0);
  while (ready.length > 0) {
    const name = ready.shift();
    order.push(name);
    for (const d of dependents.get(name).sort()) {
      pending.set(d, pending.get(d) - 1);
      if (pending.get(d) === 0) { ready.push(d); ready.sort(); }
    }
  }
  return { order, complete: order.length === names.length };
}
/** One sync: resolve, order, then diff against the releases already present in the checkouts. */
function planSync(repositories, requirements, installed = {}) {
  const index = buildIndex(repositories);
  const cycle = findCycle(index);
  if (cycle) return { ok: false, reason: "cycle", cycle, actions: [], order: [] };
  const resolution = resolve(index, requirements);
  if (!resolution.ok) {
    return { ok: false, reason: resolution.reason, cycle: null, actions: [], order: [],
      conflicting: resolution.conflicting, missing: resolution.missing };
  }
  const { order, complete } = installOrder(index, resolution.chosen);
  if (!complete) return { ok: false, reason: "cycle", cycle: order.slice(), actions: [], order };
  const actions = order.map((name) => {
    const version = resolution.chosen.get(name), current = installed[name];
    return { name, version, from: current === undefined ? null : current,
      action: current === version ? "no-op" : current === undefined ? "install" : "update" };
  });
  // Actions are built in a fixed topological order, so JSON is already a stable encoding.
  return { ok: true, reason: "planned", cycle: null, actions, order, digest: fnv1a(JSON.stringify(actions)) };
}
const REPOS = [
  { name: "core", exercises: [
    { name: "arrays", version: "1.2.0", deps: [] }, { name: "arrays", version: "1.1.0", deps: [] },
    { name: "sorting", version: "2.0.0", deps: [{ name: "arrays", range: "^1.0.0" }] },
    { name: "sorting", version: "1.5.0", deps: [{ name: "arrays", range: "~1.1.0" }] }] },
  { name: "graphs", exercises: [
    { name: "bfs", version: "1.0.0", deps: [{ name: "sorting", range: "^2.0.0" }] },
    { name: "dijkstra", version: "0.9.0", deps: [{ name: "bfs", range: "^1.0.0" }] }] },
  { name: "extra", exercises: [{ name: "arrays", version: "1.2.1", deps: [] }] },
];
const ROOTS = [{ name: "dijkstra", range: "^0.9.0" }];
const CYCLIC = [{ name: "r", exercises: [{ name: "a", version: "1.0.0", deps: [{ name: "b", range: "*" }] },
  { name: "b", version: "1.0.0", deps: [{ name: "c", range: "*" }] },
  { name: "c", version: "1.0.0", deps: [{ name: "a", range: "*" }] }] }];
/** Dependencies only ever point at lower indices, so the seeded fixture is acyclic by construction. */
function seededRepos(seed, count) {
  const rng = splitmix64(seed);
  const names = Array.from({ length: count }, (_, i) => `ex-${String(i).padStart(2, "0")}`);
  return [{ name: "big", exercises: names.map((name, i) => ({ name, version: `1.${rng.int(5)}.${rng.int(3)}`,
    deps: names.slice(0, i).filter(() => rng.int(4) === 0).map((d) => ({ name: d, range: "*" })) })) }];
}
function main() {
  const plan = planSync(REPOS, ROOTS);
  const versions = Object.fromEntries(plan.actions.map((a) => [a.name, a.version]));
  // 1. Happy path: highest satisfying release per name, dependencies before their dependents.
  assert.deepStrictEqual([plan.ok, plan.order, versions, plan.actions.map((a) => a.action)],
    [true, ["arrays", "sorting", "bfs", "dijkstra"],
      { arrays: "1.2.1", sorting: "2.0.0", bfs: "1.0.0", dijkstra: "0.9.0" },
      ["install", "install", "install", "install"]], "dependency-first cross-repo plan");
  // 2. Boundary: a dependency-free exercise is a one-entry plan; no requirements means no work.
  assert.deepStrictEqual([planSync(REPOS, [{ name: "arrays", range: "1.1.0" }]).order,
    planSync(REPOS, []).actions, planSync(REPOS, []).reason], [["arrays"], [], "planned"],
    "dependency-free exercise and empty requirement list");
  // 3. Conflict: ranges sharing a release resolve; disjoint ranges are refused instead of guessed.
  const compatible = planSync(REPOS, [{ name: "arrays", range: "^1.1.0" }, { name: "arrays", range: "1.1.0" }]);
  const impossible = planSync(REPOS, [{ name: "arrays", range: "^1.2.0" }, { name: "arrays", range: "1.1.0" }]);
  assert.deepStrictEqual([compatible.ok, compatible.actions[0].version], [true, "1.1.0"],
    "intersected ranges pin the only shared release");
  assert.deepStrictEqual([impossible.ok, impossible.reason, impossible.conflicting,
    planSync(REPOS, [{ name: "nope", range: "*" }]).missing], [false, "conflict", "arrays", ["nope"]],
    "disjoint ranges conflict; unknown exercise names are reported");
  // 4. Cycle: a self-referential release chain is refused with the offending path and no plan.
  const cycle = planSync(CYCLIC, [{ name: "a", range: "*" }]);
  assert.deepStrictEqual([cycle.reason, cycle.cycle, cycle.actions], ["cycle", ["a", "b", "c", "a"], []],
    "cycle path returned in order with an empty plan");
  // 5. Oracle: every edge points backwards, a naive O(n^2) extraction yields the same order, and a
  // brute-force scan over candidate releases picks the same versions.
  const meta = buildIndex(REPOS);
  const chosen = resolve(meta, ROOTS).chosen;
  const depsOf = (n) => meta.get(n).find((v) => v.version === chosen.get(n)).deps;
  const demands = new Map(Array.from(chosen.keys()).map((n) => [n, []]));
  for (const other of chosen.keys()) for (const d of depsOf(other)) {
    if (demands.has(d.name)) demands.get(d.name).push(d.range);
  }
  for (const name of plan.order) for (const dep of depsOf(name)) {
    if (chosen.has(dep.name)) {
      assert.ok(plan.order.indexOf(dep.name) < plan.order.indexOf(name), `edge ${dep.name} precedes ${name}`);
    }
  }
  const naive = [], remaining = new Set(plan.order);
  while (remaining.size > 0) {
    const ready = Array.from(remaining).filter((n) => depsOf(n).every((d) => !remaining.has(d.name))).sort();
    assert.ok(ready.length > 0, "a DAG always offers a ready node");
    naive.push(ready[0]); remaining.delete(ready[0]);
  }  assert.deepStrictEqual(naive, plan.order, "Kahn order matches the naive oracle");
  for (const name of chosen.keys()) assert.strictEqual(meta.get(name)
    .find((v) => demands.get(name).every((r) => satisfies(v.version, r))).version, chosen.get(name),
    `brute force picks the same ${name} release`);
  // 6. Lockfile stability: re-syncing against the produced plan is a no-op run; a stale pin updates.
  const installed = Object.fromEntries(plan.actions.map((a) => [a.name, a.version]));
  assert.deepStrictEqual([planSync(REPOS, ROOTS, installed).actions.map((a) => a.action),
    planSync(REPOS, ROOTS, { ...installed, arrays: "1.1.0" }).actions.find((a) => a.name === "arrays").action],
    [["no-op", "no-op", "no-op", "no-op"], "update"], "second sync is a no-op; stale pin is updated");
  // 7. Order independence: shuffling the repositories cannot change the plan or its digest.
  const shuffled = planSync(REPOS.slice().reverse(), ROOTS);
  assert.deepStrictEqual([shuffled.actions, shuffled.digest], [plan.actions, plan.digest],
    "plan is input-order independent");
  // 8. Seeded fixture: 24 acyclic exercises; the plan must reproduce on a second pass and stay stable.
  const fixture = seededRepos(0x988, 24);
  const last = fixture[0].exercises[23].name;
  const big = planSync(fixture, [{ name: last, range: "*" }]);
  const lock = Object.fromEntries(big.actions.map((a) => [a.name, a.version]));
  assert.deepStrictEqual([big.ok, big.order.length >= 1 && big.order.length <= 24,
    planSync(fixture, [{ name: last, range: "*" }]).digest,
    planSync(fixture, [{ name: last, range: "*" }], lock).actions.every((a) => a.action === "no-op")],
    [true, true, big.digest, true], "fixture plan reproduces and its lockfile is stable");
  console.log(`resolved=${plan.actions.length} order=${plan.order.join(">")} digest=${plan.digest.toString(16)}`);
  console.log("PASS");
}
if (require.main === module) main();
module.exports = { splitmix64, fnv1a, compareVersions, satisfies, buildIndex, findCycle, resolve,
  installOrder, planSync, seededRepos };
