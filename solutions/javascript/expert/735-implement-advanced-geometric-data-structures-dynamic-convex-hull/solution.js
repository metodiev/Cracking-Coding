/** Task 735 - Implement advanced geometric data structures: dynamic convex hull maintenance.
 *  (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * The hull is kept as two x-monotone chains - the *upper* chain (vertices turning clockwise
 * from left to right) and the *lower* chain (counter-clockwise) - each stored in a treap
 * ordered by x. A treap answers strict predecessor/successor in O(log n) expected time and
 * takes its priorities from the deterministic splitmix64 hash of the key, so the tree shape
 * and the runtime are reproducible. Insertion is the classic incremental repair: the point is
 * spliced in, dropped again when it lies on or under the chord of its two chain neighbours,
 * and then dominated neighbours are popped left and right. A popped vertex never returns to
 * that chain, so repairs amortise to O(log n) on top of the O(log n) splice.
 *
 * What is simplified: deletion rebuilds both chains from the surviving point set in
 * O(n log n); a fully dynamic hull needs Overmars-van Leeuwen / hull-tree bridges. The active
 * set is a treap x -> sorted distinct y, so re-inserting a point is idempotent and the hull
 * only moves when a per-column minimum or maximum changes.
 *
 * Invariant verified after *every* mutation: the ring equals a from-scratch monotone chain on
 * the same points (same vertices, same cyclic order after canonical rotation, same doubled
 * area via BigInt shoelace); `contains` matches an independent O(h) closed-polygon test on
 * probe grids; collinear points never become vertices; interior deletions leave the hull
 * untouched.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual } = require("node:assert");

/** Deterministic xorshift32; returns integers in [0, 2^32), or [0, bound) when given. */
function makeRng(seed) {
  let state = seed >>> 0 || 0x2545f491;
  return function next(bound) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

const MASK64 = (1n << 64n) - 1n;

/** splitmix64: bijective 64-bit mixer used as the treap priority function. */
function splitmix64(value) {
  let z = (value + 0x9e3779b97f4a7c15n) & MASK64;
  z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
  z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
  return (z ^ (z >> 31n)) & MASK64;
}

/** Treap keyed by integer: set / delete / get / floorEntry / ceilEntry / minEntry / entries. */
class Treap {
  constructor() {
    this.root = null;
    this.size = 0;
  }

  set(key, value) {
    this.root = this.#set(this.root, key, value);
  }

  #set(node, key, value) {
    if (node === null) {
      this.size++;
      return { key, value, prio: splitmix64(BigInt(key)), left: null, right: null };
    }
    if (key === node.key) {
      node.value = value;
      return node;
    }
    if (key < node.key) {
      node.left = this.#set(node.left, key, value);
      if (node.left.prio < node.prio) {
        const left = node.left;
        node.left = left.right;
        left.right = node;
        node = left;
      }
    } else {
      node.right = this.#set(node.right, key, value);
      if (node.right.prio < node.prio) {
        const right = node.right;
        node.right = right.left;
        right.left = node;
        node = right;
      }
    }
    return node;
  }

  delete(key) {
    const before = this.size;
    this.root = this.#delete(this.root, key);
    return this.size < before;
  }

  #delete(node, key) {
    if (node === null) return null;
    if (key === node.key) {
      if (node.left === null) { this.size--; return node.right; }
      if (node.right === null) { this.size--; return node.left; }
      if (node.left.prio < node.right.prio) {
        const left = node.left;
        node.left = left.right;
        left.right = node;
        node = left;
        node.right = this.#delete(node.right, key);
      } else {
        const right = node.right;
        node.right = right.left;
        right.left = node;
        node = right;
        node.left = this.#delete(node.left, key);
      }
      return node;
    }
    if (key < node.key) node.left = this.#delete(node.left, key);
    else node.right = this.#delete(node.right, key);
    return node;
  }

  get(key) {
    let node = this.root;
    while (node !== null) {
      if (key === node.key) return node.value;
      node = key < node.key ? node.left : node.right;
    }
    return undefined;
  }

  /** Greatest entry with key <= bound (smallest with key >= bound), or null. */
  floorEntry(bound) {
    let node = this.root;
    let best = null;
    while (node !== null) {
      if (node.key <= bound) { best = node; node = node.right; } else node = node.left;
    }
    return best === null ? null : { key: best.key, value: best.value };
  }

  ceilEntry(bound) {
    let node = this.root;
    let best = null;
    while (node !== null) {
      if (node.key >= bound) { best = node; node = node.left; } else node = node.right;
    }
    return best === null ? null : { key: best.key, value: best.value };
  }

  minEntry() {
    let node = this.root;
    if (node === null) return null;
    while (node.left !== null) node = node.left;
    return { key: node.key, value: node.value };
  }

  entries() {    const out = [];
    const walk = (node) => {
      if (node === null) return;
      walk(node.left);
      out.push([node.key, node.value]);
      walk(node.right);
    };
    walk(this.root);
    return out;
  }
}

/** Twice the signed area of triangle (a, b, c); positive means counter-clockwise. */
function turn(a, b, c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

/** A chain vertex is redundant when it lies on the wrong side of its neighbours' chord. */
function isRedundant(a, b, c, upper) {
  const t = turn(a, b, c);
  return upper ? t >= 0 : t <= 0;
}

function asPoint(entry) {
  return entry === null ? null : { x: entry.key, y: entry.value };
}

/** Splices (x, y) into one monotone chain, then pops dominated neighbours left and right. */
function insertIntoChain(chain, x, y, upper) {
  const existing = chain.get(x);
  if (existing !== undefined) {
    if (upper ? y <= existing : y >= existing) return; // same column: only the extreme y counts
    chain.delete(x);
  }
  chain.set(x, y);
  const point = { x, y };
  const left = asPoint(chain.floorEntry(x - 1));
  const right = asPoint(chain.ceilEntry(x + 1));
  if (left !== null && right !== null && isRedundant(left, point, right, upper)) {
    chain.delete(x); // shadowed by its own neighbours
    return;
  }
  let prev = left;
  while (prev !== null) {
    const prevPrev = asPoint(chain.floorEntry(prev.x - 1));
    if (prevPrev === null) break;
    if (!isRedundant(prevPrev, prev, point, upper)) break;
    chain.delete(prev.x);
    prev = prevPrev;
  }
  let next = right;
  while (next !== null) {
    const nextNext = asPoint(chain.ceilEntry(next.x + 1));
    if (nextNext === null) break;
    if (!isRedundant(point, next, nextNext, upper)) break;
    chain.delete(next.x);
    next = nextNext;
  }
}

class DynamicConvexHull {
  constructor() {
    this.upper = new Treap();
    this.lower = new Treap();
    this.active = new Treap(); // x -> ascending array of distinct y
    this.insertions = 0;
    this.rebuilds = 0;
  }

  get pointCount() {
    let total = 0;
    for (const [, ys] of this.active.entries()) total += ys.length;
    return total;
  }

  insert(x, y) {
    if (!Number.isInteger(x) || !Number.isInteger(y)) {
      throw new TypeError("hull coordinates must be integers (exact arithmetic)");
    }
    this.insertions++;
    const ys = this.active.get(x);
    if (ys === undefined) {
      this.active.set(x, [y]);
      insertIntoChain(this.upper, x, y, true);
      insertIntoChain(this.lower, x, y, false);
      return true;
    }
    if (ys.includes(y)) return false; // duplicate point: idempotent
    const low = ys[0];
    const high = ys[ys.length - 1];
    ys.push(y);
    ys.sort((a, b) => a - b);
    if (ys[ys.length - 1] !== high) insertIntoChain(this.upper, x, ys[ys.length - 1], true);
    if (ys[0] !== low) insertIntoChain(this.lower, x, ys[0], false);
    return true;
  }

  /** Removes the point (x, y); false when it was not present. */
  remove(x, y) {
    const ys = this.active.get(x);
    if (ys === undefined) return false;
    const index = ys.indexOf(y);
    if (index < 0) return false;
    const low = ys[0];
    const high = ys[ys.length - 1];
    ys.splice(index, 1);
    if (ys.length === 0) this.active.delete(x);
    if (ys.length === 0 || ys[0] !== low || ys[ys.length - 1] !== high) this.#rebuild();
    return true;
  }

  /** Deletion support: refit both chains from the survivors - the documented simplification. */
  #rebuild() {
    this.rebuilds++;
    this.upper = new Treap();
    this.lower = new Treap();
    for (const [x, ys] of this.active.entries()) {
      insertIntoChain(this.upper, x, ys[ys.length - 1], true);
      insertIntoChain(this.lower, x, ys[0], false);
    }
  }

  /** Hull vertices counter-clockwise: lower chain left to right, then the upper chain back. */
  vertices() {
    const low = this.lower.entries();
    if (low.length === 0) return [];
    const up = this.upper.entries();
    const ring = low.map(([x, y]) => ({ x, y }));
    for (let i = up.length - 1; i >= 0; i--) {
      const p = { x: up[i][0], y: up[i][1] };
      const last = ring[ring.length - 1];
      if (i === up.length - 1 && p.x === last.x && p.y === last.y) continue;
      const first = ring[0];
      if (i === 0 && p.x === first.x && p.y === first.y) continue;
      ring.push(p);
    }
    return ring;
  }

  /** Twice the hull area, exact (always non-negative). */
  area2() {
    const ring = this.vertices();
    let sum = 0n;
    for (let i = 0; i < ring.length; i++) {
      const a = ring[i];
      const b = ring[(i + 1) % ring.length];
      sum += BigInt(a.x) * BigInt(b.y) - BigInt(b.x) * BigInt(a.y);
    }
    return sum < 0n ? -sum : sum;
  }

  /** Closed-hull membership in O(log h) by interpolating the two chains at x. */
  contains(x, y) {
    if (this.upper.size === 0) return false;
    if (this.upper.size === 1) { // single column: both chains are degenerate
      const only = this.upper.minEntry();
      return only.key === x && this.lower.get(x) <= y && y <= this.upper.get(x);
    }
    const above = this.#strictSide(this.upper, x, y, true);
    const below = this.#strictSide(this.lower, x, y, false);
    if (above === null || below === null) return false; // x outside the hull's x-span
    return !above && !below;
  }

  /** true: strictly outside that chain; false: on/inside it; null: x outside its span. */
  #strictSide(chain, x, y, upper) {
    const left = chain.floorEntry(x);
    if (left === null || left.key > x) return null;
    if (left.key === x) return upper ? y > left.value : y < left.value;
    const right = chain.ceilEntry(x);
    if (right === null) return null;
    const t = turn({ x: left.key, y: left.value }, { x: right.key, y: right.value }, { x, y });
    return upper ? t > 0 : t < 0;
  }
}

/** Independent from-scratch hull: monotone chain with strict turns, integer coordinates. */
function bruteHullRing(points) {
  const unique = new Map();
  for (const p of points) unique.set(`${p.x},${p.y}`, { x: p.x, y: p.y });
  const sorted = [...unique.values()].sort((a, b) => a.x - b.x || a.y - b.y);
  if (sorted.length <= 1) return sorted;
  const build = (sequence) => {
    const chain = [];
    for (const p of sequence) {
      while (chain.length >= 2 && turn(chain[chain.length - 2], chain[chain.length - 1], p) <= 0) {
        chain.pop();
      }
      chain.push(p);
    }
    return chain;
  };
  const lower = build(sorted);
  const upper = build(sorted.slice().reverse());
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

/** Brute-force closed membership: a counter-clockwise ring must keep every edge test >= 0. */
function bruteContains(ring, x, y) {
  if (ring.length === 0) return false;
  if (ring.length === 1) return ring[0].x === x && ring[0].y === y;
  const p = { x, y };
  for (let i = 0; i < ring.length; i++) {
    if (turn(ring[i], ring[(i + 1) % ring.length], p) < 0) return false;
  }
  return true;
}

/** Rotates a closed ring to start at its lexicographically smallest vertex. */
function canonicalRing(ring) {
  if (ring.length === 0) return [];
  let best = 0;
  for (let i = 1; i < ring.length; i++) {
    if (ring[i].x < ring[best].x || (ring[i].x === ring[best].x && ring[i].y < ring[best].y)) best = i;
  }
  return ring.slice(best).concat(ring.slice(0, best));
}

function checkHull(hull, points, label) {
  const mine = canonicalRing(hull.vertices());
  const theirs = canonicalRing(bruteHullRing(points));
  deepStrictEqual(mine, theirs, `${label}: hull ring differs from the monotone-chain hull`);
  let expected = 0n;
  for (let i = 0; i < theirs.length; i++) {
    const a = theirs[i];
    const b = theirs[(i + 1) % theirs.length];
    expected += BigInt(a.x) * BigInt(b.y) - BigInt(b.x) * BigInt(a.y);
  }
  strictEqual(hull.area2(), expected < 0n ? -expected : expected, `${label}: doubled area`);
}

module.exports = { DynamicConvexHull, Treap, makeRng, turn, bruteHullRing, bruteContains };

if (require.main === module) {
  const empty = new DynamicConvexHull();
  deepStrictEqual(empty.vertices(), []);
  strictEqual(empty.area2(), 0n);
  ok(!empty.contains(0, 0) && !empty.remove(1, 1));

  const one = new DynamicConvexHull();
  ok(one.insert(3, 4));
  deepStrictEqual(one.vertices(), [{ x: 3, y: 4 }]);
  ok(one.contains(3, 4) && !one.contains(3, 5));
  ok(!one.insert(3, 4), "duplicate insertion is a no-op");
  strictEqual(one.pointCount, 1);
  checkHull(one, [{ x: 3, y: 4 }], "single point");

  const collinear = new DynamicConvexHull();
  const line = [[0, 0], [1, 1], [2, 2], [3, 3]];
  for (const [x, y] of line) collinear.insert(x, y);
  deepStrictEqual(collinear.vertices(), [{ x: 0, y: 0 }, { x: 3, y: 3 }],
    "interior collinear points must never become vertices");
  strictEqual(collinear.area2(), 0n);
  ok(collinear.contains(1, 1) && !collinear.contains(1, 2), "closed hull covers the segment");
  checkHull(collinear, line.map(([x, y]) => ({ x, y })), "collinear");

  const vertical = new DynamicConvexHull();
  const column = [[7, -2], [7, 0], [7, 5], [7, 9]];
  for (const [x, y] of column) vertical.insert(x, y);
  deepStrictEqual(vertical.vertices(), [{ x: 7, y: -2 }, { x: 7, y: 9 }]);
  ok(vertical.contains(7, 0) && !vertical.contains(8, 0));
  checkHull(vertical, column.map(([x, y]) => ({ x, y })), "vertical column");

  const square = new DynamicConvexHull();
  for (const [x, y] of [[0, 0], [4, 0], [4, 4], [0, 4]]) square.insert(x, y);
  strictEqual(square.area2(), 32n);
  ok(square.insert(2, 2) && square.insert(2, 0), "interior and edge points are accepted");
  strictEqual(square.vertices().length, 4, "interior/boundary points must not extend the hull");
  ok(square.contains(2, 2) && square.contains(2, 0) && square.contains(4, 4));
  ok(!square.contains(5, 2) && !square.contains(2, 5) && !square.contains(-1, 2));
  const gridRng = makeRng(99);
  for (let t = 0; t < 40; t++) {
    const x = gridRng(11) - 3;
    const y = gridRng(11) - 3;
    strictEqual(square.contains(x, y), bruteContains(square.vertices(), x, y), `probe ${x},${y}`);
  }

  // randomized incremental run: the invariant is re-checked after every single insertion
  const rng = makeRng(20260735);
  const mutations = [];
  const hull = new DynamicConvexHull();
  let points = [];
  for (let step = 0; step < 220; step++) {
    const x = rng(17) - 8;
    const y = rng(17) - 8;
    mutations.push({ x, y });
    if (hull.insert(x, y)) points.push({ x, y });
    checkHull(hull, points, `insert #${step}`);
  }
  strictEqual(hull.pointCount, points.length, "active point count tracks the insertions");
  strictEqual(hull.rebuilds, 0, "pure insertions never rebuild");

  // deletions: the hull of the survivors must be right after every removal
  const removed = new Set();
  for (let step = 0; step < 120; step++) {
    const victim = mutations[rng(mutations.length)];
    const key = `${victim.x},${victim.y}`;
    if (removed.has(key)) continue;
    removed.add(key);
    strictEqual(hull.remove(victim.x, victim.y), true, `removing ${key} must succeed`);
    points = points.filter((p) => !(p.x === victim.x && p.y === victim.y));
    checkHull(hull, points, `remove #${step} (${key})`);
    strictEqual(hull.remove(victim.x, victim.y), false, "a second removal reports a no-op");
  }
  ok(hull.rebuilds > 0, "the deletion path exercised the rebuild");
  strictEqual(hull.pointCount, points.length);

  const stable = new DynamicConvexHull();
  for (const [x, y] of [[0, 0], [10, 0], [10, 10], [0, 10], [5, 3], [5, 4], [5, 5], [3, 3]]) {
    stable.insert(x, y);
  }
  const before = stable.vertices();
  ok(stable.remove(5, 4), "the middle point of an interior column is removable");
  deepStrictEqual(stable.vertices(), before, "a non-extreme interior point cannot move the hull");
  strictEqual(stable.rebuilds, 0, "no rebuild while the column extremes survive");
  ok(stable.remove(3, 3));
  deepStrictEqual(stable.vertices(), before, "losing a column still keeps the hull unchanged");
  strictEqual(stable.rebuilds, 1, "losing a column rebuilds the chains");
  ok(stable.remove(10, 10));
  checkHull(stable, [[0, 0], [10, 0], [0, 10]].map(([x, y]) => ({ x, y })), "hull vertex removed");

  // scale + probe agreement on a large hull
  const big = new DynamicConvexHull();
  const bigPoints = [];
  const clock = makeRng(4242);
  for (let i = 0; i < 4000; i++) {
    const x = clock(2001) - 1000;
    const y = clock(2001) - 1000;
    if (big.insert(x, y)) bigPoints.push({ x, y });
  }
  checkHull(big, bigPoints, "4000-point hull");
  const probeRng = makeRng(7);
  let hits = 0;
  for (let t = 0; t < 400; t++) {
    const x = probeRng(2001) - 1000;
    const y = probeRng(2001) - 1000;
    const expected = bruteContains(big.vertices(), x, y);
    strictEqual(big.contains(x, y), expected, `large-hull probe ${x},${y}`);
    if (expected) hits++;
  }
  ok(hits > 0, "some probes must actually land inside the large hull");

  console.log(`points=${bigPoints.length} hull_vertices=${big.vertices().length} insertions=${big.insertions}`);
  console.log(`random_points=${mutations.length} rebuilds=${hull.rebuilds} area2=${big.area2()}`);
  console.log("PASS");
}
