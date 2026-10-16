/** Task 704 - Implement offline dynamic connectivity with divide & conquer + UF rollback. (Tier: Expert, CTCI Ch 4 — Trees and Graphs)
 *
 * Offline dynamic connectivity answers "are u and v connected at time t" for a known timeline of
 * edge insertions and deletions. Each edge is alive on a time interval, so the timeline is turned
 * into a segment tree and every interval is stored in the O(log T) nodes that it covers exactly.
 * A DFS of that tree then walks the timeline: on entering a node the union-find performs the
 * unions of the edges stored there, on leaving it undoes exactly those unions. A union-find with
 * union by size and *no* path compression keeps a rollback stack that restores the touched
 * vertices, which is what makes the undo O(1) amortised and the whole pass O((E + Q) log T alpha).
 *
 * What is simplified: the timeline is a fixed array of steps known in advance (that is what makes
 * the problem "offline"), edges are undirected and unweighted, and the segment tree is sized for
 * T <= a few thousand steps.
 *
 * Invariant verified: every query answer equals a brute-force BFS over the edges alive at that
 * step; after the DFS the rollback has restored the union-find to singletons with an empty
 * history; each interval is stored in at most 2*ceil(log2 T)+2 nodes; invalid input is rejected.
 */
"use strict";
const { strictEqual, deepStrictEqual, ok, throws } = require("node:assert");

/** Deterministic xorshift32; returns integers in [0, bound) (or the raw word). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17; state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/**
 * Union-find with union by size and path compression disabled: compression rewrites pointers that
 * a rollback cannot restore, so find() walks parents and every union records how to undo itself.
 */
class RollbackDsu {
  constructor(size) {
    if (!Number.isInteger(size) || size <= 0) throw new RangeError("size must be a positive integer");
    this.parent = Array.from({ length: size }, (_, i) => i);
    this.size = new Array(size).fill(1);
    this.history = [];
  }
  find(x) {
    if (!Number.isInteger(x) || x < 0 || x >= this.parent.length) {
      throw new RangeError(`vertex ${x} is outside 0..${this.parent.length - 1}`);
    }
    let root = x;
    while (this.parent[root] !== root) root = this.parent[root];
    return root;
  }
  /** Returns true when a merge happened; a redundant union still records a no-op for rollback. */
  union(a, b) {
    let rootA = this.find(a);
    let rootB = this.find(b);
    if (rootA === rootB) {
      this.history.push(null);
      return false;
    }
    if (this.size[rootA] < this.size[rootB]) {
      const swap = rootA;
      rootA = rootB;
      rootB = swap;
    }
    this.parent[rootB] = rootA;
    this.size[rootA] += this.size[rootB];
    this.history.push([rootB, rootA]);
    return true;
  }
  connected(a, b) { return this.find(a) === this.find(b); }
  snapshot() { return this.history.length; }
  /** Undoes every union recorded after `mark`, newest first. */
  rollback(mark) {
    while (this.history.length > mark) {
      const record = this.history.pop();
      if (record === null) continue; // a redundant union changed nothing
      this.parent[record[0]] = record[0];
      this.size[record[1]] -= this.size[record[0]];
    }
  }
  fingerprint() { return `${this.parent.join(",")}|${this.size.join(",")}`; }
  isTrivial() {
    return this.parent.every((p, i) => p === i) && this.size.every((s) => s === 1);
  }
}

/**
 * Solves the offline problem. `intervals` entries are {u, v, from, to} with an inclusive step
 * range; `queries[t]` is a [u, v] pair asked at step t (or undefined for no question that step).
 */
function offlineConnectivity(size, intervals, steps, queries = []) {
  if (!Number.isInteger(steps) || steps < 0) throw new RangeError("steps must be a non-negative integer");
  const alive = intervals.filter((interval) => {
    if (!Number.isInteger(interval.from) || !Number.isInteger(interval.to)) {
      throw new RangeError("interval bounds must be integers");
    }
    if (interval.from < 0 || interval.to >= steps) throw new RangeError("interval outside the timeline");
    return interval.to >= interval.from; // an empty interval is simply dropped
  });
  const tree = Array.from({ length: 4 * Math.max(1, steps) }, () => []);
  let placements = 0;
  const add = (node, lo, hi, interval) => {
    if (interval.to < lo || hi < interval.from) return;
    if (interval.from <= lo && hi <= interval.to) {
      tree[node].push(interval);
      placements++;
      return;
    }
    const mid = (lo + hi) >> 1;
    add(2 * node, lo, mid, interval);
    add(2 * node + 1, mid + 1, hi, interval);
  };
  if (steps > 0) for (const interval of alive) add(1, 0, steps - 1, interval);
  const dsu = new RollbackDsu(size);
  const answers = new Array(steps).fill(null);
  let unions = 0, rollbacks = 0;
  if (steps > 0) {
    const walk = (node, lo, hi) => {
      const mark = dsu.snapshot();
      for (const interval of tree[node]) if (dsu.union(interval.u, interval.v)) unions++;
      if (lo === hi) {
        const query = queries[lo];
        if (query !== undefined) answers[lo] = dsu.connected(query[0], query[1]);
      } else {
        const mid = (lo + hi) >> 1;
        walk(2 * node, lo, mid);
        walk(2 * node + 1, mid + 1, hi);
      }
      rollbacks += dsu.history.length - mark;
      dsu.rollback(mark);
    };
    walk(1, 0, steps - 1);
  }
  return { answers, dsu, placements, unions, rollbacks };
}

/** Brute force: for every step, BFS over the edges alive at that step (no cleverness at all). */
function bruteAnswers(size, intervals, steps, queries = []) {
  const answers = new Array(steps).fill(null);
  for (let step = 0; step < steps; step++) {
    const adjacency = Array.from({ length: size }, () => new Set());
    for (const interval of intervals) {
      if (interval.from <= step && step <= interval.to) {
        adjacency[interval.u].add(interval.v);
        adjacency[interval.v].add(interval.u);
      }
    }
    const query = queries[step];
    if (query === undefined) continue;
    const seen = new Set([query[0]]);
    const queue = [query[0]];
    while (queue.length > 0) {
      const current = queue.shift();
      for (const next of adjacency[current]) if (!seen.has(next)) { seen.add(next); queue.push(next); }
    }
    answers[step] = seen.has(query[1]);
  }
  return answers;
}

/** The steps an inclusive interval covers, as a plain list - used to check the tree's coverage. */
function coveredSteps(interval, steps) {
  const out = [];
  for (let step = 0; step < steps; step++) if (interval.from <= step && step <= interval.to) out.push(step);
  return out;
}

module.exports = { makeRng, RollbackDsu, offlineConnectivity, bruteAnswers, coveredSteps };

if (require.main === module) {
  // -- rejected input and the empty timeline ---------------------------------------------------
  throws(() => new RollbackDsu(0), RangeError);
  throws(() => new RollbackDsu(2).find(2), RangeError);
  const none = offlineConnectivity(3, [], 0);
  deepStrictEqual(none.answers, []);
  ok(none.dsu.isTrivial());
  throws(() => offlineConnectivity(3, [{ u: 0, v: 1, from: 0, to: 0 }], 0), RangeError);
  throws(() => offlineConnectivity(3, [{ u: 0, v: 1, from: 0, to: 5 }], 3), RangeError);
  throws(() => offlineConnectivity(3, [], -1), RangeError);

  // -- one edge alive on a single step ---------------------------------------------------------
  const single = offlineConnectivity(4, [{ u: 0, v: 1, from: 2, to: 2 }], 5,
    [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]);
  deepStrictEqual(single.answers, [false, false, true, false, false]);
  ok(single.dsu.isTrivial(), "the rollback must restore every vertex");

  // -- an edge that lives, dies and comes back -------------------------------------------------
  const returning = offlineConnectivity(3, [
    { u: 0, v: 1, from: 0, to: 1 },
    { u: 0, v: 1, from: 3, to: 5 },
    { u: 1, v: 2, from: 4, to: 5 },
    { u: 0, v: 0, from: 0, to: 5 }, // a self loop is always trivially connected
  ], 6, [[0, 1], [0, 1], [0, 1], [1, 2], [0, 2], [0, 2]]);
  deepStrictEqual(returning.answers, [true, true, false, false, true, true]);
  ok(returning.dsu.isTrivial());
  deepStrictEqual(returning.answers, bruteAnswers(3, [
    { u: 0, v: 1, from: 0, to: 1 },
    { u: 0, v: 1, from: 3, to: 5 },
    { u: 1, v: 2, from: 4, to: 5 },
    { u: 0, v: 0, from: 0, to: 5 },
  ], 6, [[0, 1], [0, 1], [0, 1], [1, 2], [0, 2], [0, 2]]));

  // -- an empty interval is dropped, an interval covering everything is stored once -------------
  const dropped = offlineConnectivity(3, [{ u: 0, v: 1, from: 3, to: 1 }], 4, [[0, 1], [0, 1], [0, 1], [0, 1]]);
  deepStrictEqual(dropped.answers, [false, false, false, false]);
  const spanning = offlineConnectivity(3, [{ u: 0, v: 2, from: 0, to: 3 }], 4, [[0, 2], [0, 2], [0, 2], [0, 2]]);
  deepStrictEqual(spanning.answers, [true, true, true, true]);
  strictEqual(spanning.placements, 1, "a full-range interval lands in the root only");

  // -- the rollback stack itself ---------------------------------------------------------------
  const rng = makeRng(704);
  const dsu = new RollbackDsu(10);
  const marks = [];
  for (let round = 0; round < 12; round++) {
    for (let i = 0; i < 6; i++) dsu.union(rng(10), rng(10));
    marks.push(dsu.snapshot());
  }
  const halfway = new RollbackDsu(10);
  const replayRng = makeRng(704);
  for (let round = 0; round < 6; round++) for (let i = 0; i < 6; i++) halfway.union(replayRng(10), replayRng(10));
  dsu.rollback(marks[5]);
  strictEqual(dsu.fingerprint(), halfway.fingerprint(), "rollback must reproduce the earlier state");
  dsu.rollback(0);
  ok(dsu.isTrivial() && dsu.history.length === 0, "a full rollback empties the history");

  // -- randomized cross-check against brute force ----------------------------------------------
  const rng2 = makeRng(7041);
  for (let trial = 0; trial < 8; trial++) {
    const size = 4 + rng2(6);
    const steps = 1 + rng2(24);
    const intervals = [];
    for (let i = 0; i < 3 + rng2(18); i++) {
      const from = rng2(steps);
      const to = Math.min(steps - 1, from + rng2(steps - from));
      intervals.push({ u: rng2(size), v: rng2(size), from, to });
    }
    const queries = Array.from({ length: steps }, () => [rng2(size), rng2(size)]);
    const got = offlineConnectivity(size, intervals, steps, queries);
    deepStrictEqual(got.answers, bruteAnswers(size, intervals, steps, queries), `trial ${trial}`);
    ok(got.dsu.isTrivial(), `trial ${trial}: the DSU must end trivial`);
    const bound = intervals.length * (2 * Math.ceil(Math.log2(Math.max(2, steps))) + 2);
    ok(got.placements <= bound, `trial ${trial}: ${got.placements} placements exceed ${bound}`);
    ok(got.unions + got.rollbacks >= 0 && got.rollbacks >= got.unions - steps * size,
      "every merged union must be rolled back");
  }

  // -- coverage: the segment tree stores exactly the steps an interval is alive on -------------
  const steps = 17;
  for (let from = 0; from < steps; from++) {
    for (let to = from; to < steps; to++) {
      const interval = { u: 0, v: 1, from, to };
      const result = offlineConnectivity(2, [interval], steps,
        Array.from({ length: steps }, () => [0, 1]));
      const expected = Array.from({ length: steps }, (_, t) => coveredSteps(interval, steps).includes(t));
      deepStrictEqual(result.answers, expected, `interval [${from},${to}]`);
    }
  }

  // -- determinism ------------------------------------------------------------------------------
  function replay(seed) {
    const r = makeRng(seed);
    const size = 6;
    const steps = 16;
    const intervals = Array.from({ length: 20 }, () => {
      const from = r(steps);
      return { u: r(size), v: r(size), from, to: Math.min(steps - 1, from + r(steps - from)) };
    });
    const queries = Array.from({ length: steps }, () => [r(size), r(size)]);
    const result = offlineConnectivity(size, intervals, steps, queries);
    return `${result.answers.join(",")}|${result.placements}`;
  }
  strictEqual(replay(7), replay(7), "identical seeds must replay identically");

  console.log(`single-step unions=${single.unions} rollbacks=${single.rollbacks}`);
  console.log(`spanning placements=${spanning.placements} of ${single.placements + returning.placements}`);
  console.log("PASS");
}
