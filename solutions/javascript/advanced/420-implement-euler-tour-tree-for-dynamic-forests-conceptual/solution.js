/** Task 420 - Implement Euler tour tree for dynamic forests (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * An Euler tour tree stores one Euler circuit of each tree as a sequence of DIRECTED EDGE
 * events (u -> v): travelling the circuit enters u and leaves towards v, so the sequence
 * chains (each event's target is the next event's source, cyclically) and every directed
 * edge of the tree appears exactly once. A real ETT keeps these sequences in balanced
 * search trees and links/cuts by splitting/merging them in O(log n); this conceptual
 * implementation uses plain lists, so each operation costs O(size) — the operations are
 * identical, only the balance is simulated away (documented deviation).
 *
 *   * connect(u, v): rotate u's tour to start leaving u and v's tour to start leaving v,
 *     then splice:  tour(u) + [u->v] + tour(v) + [v->u]  — a valid circuit of the merged tree.
 *   * disconnect(u, v) (edge exists): rotate to start with [u->v], find the matching
 *     [v->u]; the events strictly between them form the v-side circuit, the rest (after
 *     [v->u]) forms the u-side circuit.
 *   * reroot(v): rotate v's circuit so it starts with an event leaving v.
 *   * connected / componentVertices: answered from the owner map (a list simulation of
 *     the tree lookup a real ETT answers via the roots of its balanced trees).
 */
"use strict";

class EulerTourForest {
  constructor(n) {
    this.n = n;
    this.tours = new Map();
    this.owner = new Array(n + 1).fill(0);
    for (let v = 1; v <= n; v++) {
      this.owner[v] = v;
      this.tours.set(v, []);
    }
    this.nextId = n + 1;
  }

  tourOf(x) {
    return this.tours.get(this.owner[x]);
  }

  /** Rotate so the sequence starts with an event leaving x (no-op if isolated). */
  static rotate(tour, x) {
    for (let i = 0; i < tour.length; i++) {
      if (tour[i][0] === x) return tour.slice(i).concat(tour.slice(0, i));
    }
    return tour.slice();
  }

  static endpoints(tour) {
    const verts = new Set();
    for (const [a, b] of tour) {
      verts.add(a);
      verts.add(b);
    }
    return verts;
  }

  #reassign(id, members) {
    for (const vert of members) this.owner[vert] = id;
  }

  /** Link u and v (must be in different components); returns true when linked. */
  connect(u, v) {
    if (this.owner[u] === this.owner[v]) return false;
    const merged = EulerTourForest.rotate(this.tourOf(u), u)
      .concat([[u, v]])
      .concat(EulerTourForest.rotate(this.tourOf(v), v))
      .concat([[v, u]]);
    const oldIds = [this.owner[u], this.owner[v]];
    const id = this.nextId++;
    for (const oldId of oldIds) this.tours.delete(oldId);
    this.tours.set(id, merged);
    const members = EulerTourForest.endpoints(merged);
    members.add(u);
    members.add(v);
    this.#reassign(id, members);
    return true;
  }

  /** Cut the edge {u, v} when it exists; returns true when cut. */
  disconnect(u, v) {
    if (this.owner[u] !== this.owner[v]) return false;
    const tour = this.tourOf(u);
    let forward = -1, reverse = -1;
    for (let i = 0; i < tour.length; i++) {
      const [a, b] = tour[i];
      if (a === u && b === v) forward = i;
      if (a === v && b === u) reverse = i;
    }
    if (forward < 0 || reverse < 0) return false;
    const rot = tour.slice(forward).concat(tour.slice(0, forward));   // starts with u -> v
    let q = -1;
    for (let i = 0; i < rot.length; i++) {
      const [a, b] = rot[i];
      if (a === v && b === u) { q = i; break; }
    }
    const sideV = rot.slice(1, q);           // closed walk at v
    const sideU = rot.slice(q + 1);          // closed walk at u
    const oldId = this.owner[u];
    const idU = this.nextId++;
    const idV = this.nextId++;
    this.tours.delete(oldId);
    this.tours.set(idU, sideU);
    this.tours.set(idV, sideV);
    const mu = EulerTourForest.endpoints(sideU);
    mu.add(u);
    this.#reassign(idU, mu);
    const mv = EulerTourForest.endpoints(sideV);
    mv.add(v);
    this.#reassign(idV, mv);
    return true;
  }

  reroot(x) {
    const id = this.owner[x];
    this.tours.set(id, EulerTourForest.rotate(this.tours.get(id), x));
  }

  connected(u, v) {
    return this.owner[u] === this.owner[v];
  }

  componentVertices(x) {
    const members = EulerTourForest.endpoints(this.tours.get(this.owner[x]));
    members.add(x);
    return [...members].sort((a, b) => a - b);
  }

  componentSize(x) {
    return this.componentVertices(x).length;
  }
}

/** Adjacency oracle with BFS components. */
class OracleForest {
  constructor(n) {
    this.adj = [];
    for (let i = 0; i <= n; i++) this.adj.push(new Set());
  }

  link(u, v) {
    this.adj[u].add(v);
    this.adj[v].add(u);
  }

  cut(u, v) {
    this.adj[u].delete(v);
    this.adj[v].delete(u);
  }

  hasEdge(u, v) {
    return this.adj[u].has(v);
  }

  component(x) {
    const seen = new Set([x]);
    const queue = [x];
    while (queue.length > 0) {
      const cur = queue.pop();
      for (const nxt of this.adj[cur]) {
        if (!seen.has(nxt)) {
          seen.add(nxt);
          queue.push(nxt);
        }
      }
    }
    return seen;
  }

  connected(u, v) {
    return this.component(u).has(v);
  }
}

let seed = 0;

function nextRand() {
  seed = (seed * 75 + 74) % 65537;
  return seed;
}

/** Every component's tour must be a valid Euler circuit of the oracle tree. */
function validate(forest, oracle, label) {
  const seen = new Set();
  for (let start = 1; start <= forest.n; start++) {
    if (seen.has(start)) continue;
    const members = oracle.component(start);
    for (const v of members) seen.add(v);
    const tour = forest.tourOf(start);
    if (members.size === 1) {
      if (tour.length !== 0) throw new Error(`singleton tour ${label} ${start}`);
      continue;
    }
    if (tour.length !== 2 * (members.size - 1)) throw new Error(`tour size ${label}`);
    for (let i = 0; i < tour.length; i++) {
      if (tour[i][1] !== tour[(i + 1) % tour.length][0]) throw new Error(`chaining ${label} ${i}`);
    }
    const seenEvents = new Set();
    const covered = new Set();
    for (const [a, b] of tour) {
      const key = `${a}->${b}`;
      if (seenEvents.has(key)) throw new Error(`duplicate event ${label}`);
      seenEvents.add(key);
      if (!oracle.hasEdge(a, b)) throw new Error(`bogus edge ${label}`);
      covered.add(a);
      covered.add(b);
    }
    if (covered.size !== members.size) throw new Error(`coverage ${label} ${start}`);
    for (const member of members) {
      if (!covered.has(member)) throw new Error(`missing vertex ${label}`);
    }
    const cv = forest.componentVertices(start);
    if (cv.length !== members.size) throw new Error(`vertices ${label}`);
    for (const v of cv) if (!members.has(v)) throw new Error(`stray vertex ${label}`);
    if (forest.componentSize(start) !== members.size) throw new Error(`size ${label}`);
  }
}

module.exports = { EulerTourForest, OracleForest, nextRand, validate };

if (require.main === module) {
  const assert = require("node:assert");

  // hand trace on three vertices
  let forest = new EulerTourForest(3);
  assert.strictEqual(forest.connected(1, 1), true);
  assert.strictEqual(forest.connected(1, 2), false);
  assert.deepStrictEqual(forest.componentVertices(1), [1]);
  assert.strictEqual(forest.connect(1, 2), true);
  assert.deepStrictEqual(forest.tourOf(1), [[1, 2], [2, 1]]);
  assert.strictEqual(forest.connect(2, 3), true);
  // tour of the walk 1 -> 2 -> 3 -> 2 -> 1, rotated to start leaving 2
  assert.deepStrictEqual(forest.tourOf(2), [[2, 1], [1, 2], [2, 3], [3, 2]]);
  assert.strictEqual(forest.connected(1, 3), true);
  assert.strictEqual(forest.componentSize(1), 3);
  assert.deepStrictEqual(forest.componentVertices(1), [1, 2, 3]);

  // reroot rotates the circuit so that it starts leaving the new root
  forest.reroot(3);
  assert.deepStrictEqual(forest.tourOf(3), [[3, 2], [2, 1], [1, 2], [2, 3]]);

  // cut edge 2-3: the v-side is the singleton {3}, the rest keeps {1, 2}
  assert.strictEqual(forest.disconnect(2, 3), true);
  assert.strictEqual(forest.connected(1, 3), false);
  assert.strictEqual(forest.connected(1, 2), true);
  assert.deepStrictEqual(forest.tourOf(3), []);
  const side = forest.tourOf(1);
  assert.strictEqual(side.length, 2);
  for (const [a, b] of side) {
    assert.ok((a === 1 && b === 2) || (a === 2 && b === 1));
  }
  assert.deepStrictEqual(forest.componentVertices(3), [3]);

  // cutting again without the edge is a no-op
  assert.strictEqual(forest.disconnect(2, 3), false);
  assert.strictEqual(forest.connected(1, 2), true);

  // a non-adjacent pair cannot be disconnected even when connected
  forest.connect(2, 3);
  assert.strictEqual(forest.connected(1, 3), true);
  assert.strictEqual(forest.disconnect(1, 3), false);   // 1-2-3 path, no 1-3 edge
  assert.strictEqual(forest.connected(1, 3), true);
  assert.strictEqual(forest.disconnect(1, 2), true);
  assert.strictEqual(forest.connected(1, 2), false);
  assert.strictEqual(forest.connected(2, 3), true);

  // scripted operations against the union-find + tree oracle
  const n = 25;
  seed = 4200;
  forest = new EulerTourForest(n);
  const oracle = new OracleForest(n);
  for (let step = 0; step < 320; step++) {
    const kind = nextRand() % 4;
    const u = 1 + (nextRand() % n);
    const v = 1 + (nextRand() % n);
    if (kind === 0) {
      if (u !== v && !oracle.connected(u, v)) {
        assert.strictEqual(forest.connect(u, v), true, `script connect ${step}`);
        oracle.link(u, v);
        assert.strictEqual(forest.connected(u, v), true, `script connect check ${step}`);
      }
    } else if (kind === 1) {
      if (u !== v && oracle.hasEdge(u, v)) {
        assert.strictEqual(forest.disconnect(u, v), true, `script cut ${step}`);
        oracle.cut(u, v);
        assert.strictEqual(forest.connected(u, v), false, `script cut check ${step}`);
      }
    } else if (kind === 2) {
      assert.strictEqual(forest.connected(u, v), oracle.connected(u, v), `script conn ${step}`);
      const cv = forest.componentVertices(u);
      const members = oracle.component(u);
      assert.strictEqual(cv.length, members.size, `script vertices ${step}`);
      for (const w of cv) assert.ok(members.has(w), `script member ${step}`);
      assert.strictEqual(forest.componentSize(u), members.size, `script size ${step}`);
    } else {
      forest.reroot(u);
      assert.strictEqual(forest.connected(u, v), oracle.connected(u, v), `script reroot ${step}`);
    }
    // structural validity after every single operation
    validate(forest, oracle, `step ${step}`);
    // incremental spot checks on fresh random pairs
    for (let q = 0; q < 3; q++) {
      const a = 1 + (nextRand() % n);
      const b = 1 + (nextRand() % n);
      assert.strictEqual(forest.connected(a, b), oracle.connected(a, b), `spot ${step}`);
    }
  }

  // final full audit: every pair, every component
  for (let u = 1; u <= n; u++) {
    for (let v = u + 1; v <= n; v++) {
      assert.strictEqual(forest.connected(u, v), oracle.connected(u, v), `final conn ${u} ${v}`);
    }
  }
  validate(forest, oracle, "final");
  // final shape: the deterministic script leaves one giant component plus a few
  // cut-off singletons; assert the giant exists (exact matching is oracle-checked)
  let largest = 0;
  for (let v = 1; v <= n; v++) largest = Math.max(largest, oracle.component(v).size);
  assert.ok(largest >= n - 5, "giant component");

  console.log("PASS");
}
