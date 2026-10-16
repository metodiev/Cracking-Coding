/** Task 772 - Implement dynamic planar graph algorithms (face updates) outline. (Tier: Expert, CTCI Ch 4 — Trees and Graphs)
 *
 * The graph is a *rotation system*: each vertex owns a cyclic list of the half-edges (darts)
 * leaving it, sorted counter-clockwise by the drawing. Faces are the orbits of
 *     next(u -> v) = the predecessor of (v -> u) in v's CCW rotation,
 * i.e. at v we turn clockwise from the incoming direction, which keeps the face left of the dart.
 * An edge update splices two rotation lists and only the orbits through the modified darts change,
 * so a *local* face update walks just those orbits, relabels them with fresh ids and drops the
 * faces that lost their darts — the "face update" outline: no global recomputation, no planarity
 * repair.
 *
 * What is simplified: the embedding comes from a planar straight-line drawing and updates that
 * would break planarity are rejected (proper crossings, collinear touching, duplicate edges,
 * self-loops) instead of being tested for planarity; face split/merge and Euler are real. What is
 * verified: after every update the maintained face partition equals a from-scratch walk over all
 * darts (the oracle) and V' - E + F == 2 * C for the non-isolated vertices; chord insertion splits
 * a face and deletion restores it; a dangling edge leaves F unchanged but rewrites a boundary;
 * joining components merges their outer orbits; grid updates stay local, and a seeded update
 * sequence replays identically.
 */
"use strict";

class PlanarGraph {
  constructor() {
    this.vertices = new Map();   // name -> { x, y }
    this.rotations = new Map();  // name -> dart ids leaving it, CCW by angle
    this.darts = [];             // dart id -> { from, to, twin, face, alive }
    this.faces = new Map();      // face id -> Set of dart ids
    this.nextFaceId = 0;
    this.epoch = 0;              // relabel generation, see #relabel
    this.localVisitCount = 0;    // darts touched by the last update (locality metric)
  }

  addVertex(name, x, y) {
    if (this.vertices.has(name)) return false;
    this.vertices.set(name, { x, y });
    this.rotations.set(name, []);
    return true;
  }

  get faceCount() { return this.faces.size; }

  #angle(from, to) {
    const a = this.vertices.get(from); const b = this.vertices.get(to);
    return Math.atan2(b.y - a.y, b.x - a.x);
  }

  hasEdge(u, v) { return this.#findDart(u, v) !== -1; }

  #findDart(u, v) {
    for (const id of this.rotations.get(u) ?? []) if (this.darts[id].to === v) return id;
    return -1;
  }

  /** Predecessor of `twin` in the CCW rotation of the head vertex. */
  nextDart(dartId) {
    const dart = this.darts[dartId];
    const rot = this.rotations.get(dart.to);
    const index = rot.indexOf(dart.twin);
    return rot[(index - 1 + rot.length) % rot.length];
  }

  #properIntersection(a, b, c, d) {
    const cross = (p, q, r) => (q.x - p.x) * (r.y - p.y) - (q.y - p.y) * (r.x - p.x);
    const p1 = this.vertices.get(a); const p2 = this.vertices.get(b);
    const p3 = this.vertices.get(c); const p4 = this.vertices.get(d);
    const d1 = cross(p3, p4, p1); const d2 = cross(p3, p4, p2);
    const d3 = cross(p1, p2, p3); const d4 = cross(p1, p2, p4);
    if (((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0))) return true;
    // collinear overlap or touching in the interior of either segment also breaks the embedding
    const onSegment = (p, q, r) => Math.min(p.x, r.x) <= q.x && q.x <= Math.max(p.x, r.x)
      && Math.min(p.y, r.y) <= q.y && q.y <= Math.max(p.y, r.y);
    if (d1 === 0 && onSegment(p3, p1, p4)) return true;
    if (d2 === 0 && onSegment(p3, p2, p4)) return true;
    if (d3 === 0 && onSegment(p1, p3, p2)) return true;
    return d4 === 0 && onSegment(p1, p4, p2);
  }

  /** True when segment u-v can be inserted without crossing the current drawing. */
  isEmbeddable(u, v) {
    for (const rot of this.rotations.values()) {
      for (const id of rot) {
        const dart = this.darts[id];
        if (!dart.alive || dart.from > dart.to) continue; // visit each edge once
        const { from: a, to: b } = dart;
        if (a === u || a === v || b === u || b === v) continue; // shared endpoints are fine
        if (this.#properIntersection(u, v, a, b)) return false;
      }
    }
    return true;
  }

  addEdge(u, v) {
    if (!this.vertices.has(u) || !this.vertices.has(v)) throw new RangeError("both endpoints must exist");
    if (u === v) throw new RangeError("self-loops are outside this outline");
    if (this.hasEdge(u, v)) throw new RangeError("duplicate edge");
    if (!this.isEmbeddable(u, v)) throw new RangeError("the edge would cross the existing drawing");
    const forward = this.darts.length;
    const backward = forward + 1;
    this.darts.push({ from: u, to: v, twin: backward, face: -1, faceEpoch: -1, alive: true });
    this.darts.push({ from: v, to: u, twin: forward, face: -1, faceEpoch: -1, alive: true });
    this.#spliceInto(u, forward);
    this.#spliceInto(v, backward);
    this.#relabel([forward, backward]);
  }

  removeEdge(u, v) {
    const dartId = this.#findDart(u, v);
    if (dartId === -1) throw new RangeError("edge not present");
    const twinId = this.darts[dartId].twin;
    const survivors = [];
    for (const id of [dartId, twinId]) {
      const dart = this.darts[id];
      const rot = this.rotations.get(dart.from);
      const index = rot.indexOf(id);
      survivors.push(rot[(index + 1) % rot.length], rot[(index - 1 + rot.length) % rot.length]);
      rot.splice(index, 1);
      dart.alive = false;
    }
    const face = this.faces.get(this.darts[dartId].face);
    if (face !== undefined) face.delete(dartId);
    const twinFace = this.faces.get(this.darts[twinId].face);
    if (twinFace !== undefined) twinFace.delete(twinId);
    this.#relabel(survivors.filter((id) => id !== dartId && id !== twinId));
  }

  /** Insert a dart into its rotation list, ordered counter-clockwise by angle. */
  #spliceInto(vertex, dartId) {
    const rot = this.rotations.get(vertex);
    const angle = this.#angle(vertex, this.darts[dartId].to);
    let at = rot.length;
    for (let i = 0; i < rot.length; i++) {
      if (this.#angle(vertex, this.darts[rot[i]].to) > angle) { at = i; break; }
    }
    rot.splice(at, 0, dartId);
  }

  /** Relabels the orbits through `candidates`; the only faces whose dart sets can change. */
  #relabel(candidates) {
    this.epoch++;
    let visits = 0;
    for (const start of candidates) {
      if (!this.darts[start]?.alive || this.darts[start].faceEpoch === this.epoch) continue;
      const orbit = [];
      let dartId = start;
      do {
        orbit.push(dartId);
        dartId = this.nextDart(dartId);
      } while (dartId !== start);
      visits += orbit.length;
      const newFace = this.nextFaceId++;
      for (const id of orbit) {
        this.faces.get(this.darts[id].face)?.delete(id); // leave the old face
        this.darts[id].face = newFace;
        this.darts[id].faceEpoch = this.epoch;
      }
      this.faces.set(newFace, new Set(orbit));
    }
    for (const [id, darts] of [...this.faces]) if (darts.size === 0) this.faces.delete(id);
    this.localVisitCount = visits;
  }

  /** From-scratch face walk: the independent oracle for the maintained partition. */
  recomputeFaces() {
    const seen = new Set();
    const out = [];
    for (let id = 0; id < this.darts.length; id++) {
      if (!this.darts[id].alive || seen.has(id)) continue;
      const orbit = [];
      let dartId = id;
      do {
        orbit.push(dartId);
        seen.add(dartId);
        dartId = this.nextDart(dartId);
      } while (dartId !== id);
      out.push(orbit);
    }
    return out;
  }

  /** Canonical form (sorted orbits of sorted dart ids) used to compare partitions. */
  static #canonical(orbits) {
    return orbits.map((f) => [...f].sort((a, b) => a - b)).sort((a, b) => a[0] - b[0]);
  }

  canonicalFaces() { return PlanarGraph.#canonical([...this.faces.values()]); }

  canonicalRecomputed() { return PlanarGraph.#canonical(this.recomputeFaces()); }

  /** Euler over the non-isolated part: V' - E + F == 2 * C (one outer orbit per component). */
  checkEuler() {
    const active = [...this.vertices.keys()].filter((name) => (this.rotations.get(name) ?? []).length > 0);
    const parent = new Map(active.map((name) => [name, name]));
    const find = (x) => (parent.get(x) === x ? x : (parent.set(x, find(parent.get(x))), parent.get(x)));
    let edges = 0;
    for (let id = 0; id < this.darts.length; id++) {
      const dart = this.darts[id];
      if (!dart.alive || dart.from > dart.to) continue;
      edges++;
      const a = find(dart.from);
      const b = find(dart.to);
      if (a !== b) parent.set(a, b);
    }
    const roots = new Set(active.map(find));
    return { v: active.length, e: edges, f: this.faces.size, c: roots.size,
      holds: active.length - edges + this.faces.size === 2 * roots.size };
  }
}

function gridGraph(cols, rows) {
  const g = new PlanarGraph();
  for (let y = 0; y <= rows; y++) for (let x = 0; x <= cols; x++) g.addVertex(`${x},${y}`, x, y);
  for (let y = 0; y <= rows; y++) for (let x = 0; x <= cols; x++) {
    if (x < cols) g.addEdge(`${x},${y}`, `${x + 1},${y}`);
    if (y < rows) g.addEdge(`${x},${y}`, `${x},${y + 1}`);
  }
  return g;
}

function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

module.exports = { PlanarGraph, gridGraph, makeRng };

if (require.main === module) {
  const assert = require("node:assert");
  const { strictEqual, ok, deepStrictEqual, throws } = assert;
  const rng = makeRng(20260772);
  let updatesVerified = 0;

  const verify = (g, label) => {
    deepStrictEqual(g.canonicalFaces(), g.canonicalRecomputed(), `${label}: maintained faces differ from the walk`);
    const euler = g.checkEuler();
    ok(euler.holds, `${label}: Euler violated (V=${euler.v} E=${euler.e} F=${euler.f} C=${euler.c})`);
    updatesVerified++;
    return euler;
  };
  const build = (coords, edges) => {
    const g = new PlanarGraph();
    for (const [name, x, y] of coords) g.addVertex(name, x, y);
    for (const [u, v] of edges) g.addEdge(u, v);
    return g;
  };
  const square = [["A", 0, 0], ["B", 4, 0], ["C", 4, 4], ["D", 0, 4]];
  const squareEdges = [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]];
  const eulerData = (e) => [e.v, e.e, e.f, e.c];
  const aliveDarts = (g) => g.darts.filter((d) => d.alive).length;

  // -- triangle and quadrilateral: face counts, the walk and Euler agree --------------
  const triangle = build([["A", 0, 0], ["B", 4, 0], ["C", 1, 3]], [["A", "B"], ["B", "C"], ["C", "A"]]);
  strictEqual(triangle.faceCount, 2, "a triangle has an inner and an outer face");
  deepStrictEqual(eulerData(verify(triangle, "triangle")), [3, 3, 2, 1], "triangle Euler data");
  const quad = build(square, squareEdges);
  strictEqual(quad.faceCount, 2, "a quadrilateral has two faces");
  verify(quad, "quadrilateral");
  const beforeDiagonal = quad.canonicalFaces().find((f) => f.length === 4);

  // -- splitting a face with a chord, then restoring it by deletion -------------------
  quad.addEdge("A", "C");
  strictEqual(quad.faceCount, 3, "a chord inside the quad splits its inner face");
  strictEqual(quad.canonicalFaces().filter((f) => f.length === 3).length, 2, "the split made two triangles");
  ok(quad.localVisitCount < aliveDarts(quad), "the update touches less than the whole dart set");
  verify(quad, "quad + diagonal");
  quad.removeEdge("A", "C");
  strictEqual(quad.faceCount, 2, "deleting the chord merges the faces again");
  deepStrictEqual(quad.canonicalFaces().find((f) => f.length === 4), beforeDiagonal, "deletion restores the face");
  verify(quad, "quad restored");
  throws(() => quad.removeEdge("B", "D"), RangeError, "removing a missing edge must fail");

  // -- a dangling edge cannot separate the plane, but rewrites one face boundary -----
  quad.addVertex("E", 8, 4);
  const facesBefore = quad.faceCount;
  quad.addEdge("C", "E");
  strictEqual(quad.faceCount, facesBefore, "an edge to a new degree-1 vertex leaves F unchanged");
  const withE = [...quad.faces.values()].filter((f) => [...f].some((d) => quad.darts[d].from === "E" || quad.darts[d].to === "E"));
  strictEqual(withE.length, 1, "both darts of a dangling edge bound the same face");
  strictEqual([...withE[0]].filter((d) => quad.darts[d].from === "E" || quad.darts[d].to === "E").length, 2,
    "the dangling edge is traversed in both directions inside that face");
  verify(quad, "quad + dangling edge");
  quad.removeEdge("C", "E");
  strictEqual(quad.faceCount, facesBefore, "removing the dangling edge keeps F");
  verify(quad, "dangling edge removed");

  // -- joining two components merges their outer orbits ------------------------------
  const paired = build([...square.slice(0, 3), ["D", 20, 0], ["E", 24, 0], ["F", 21, 3]],
    [["A", "B"], ["B", "C"], ["C", "A"], ["D", "E"], ["E", "F"], ["F", "D"]]);
  strictEqual(paired.faceCount, 4, "two disjoint triangles give two inner and two outer orbits");
  strictEqual(verify(paired, "two components").c, 2, "the triangles are separate components");
  paired.addEdge("C", "F");
  strictEqual(verify(paired, "joined components").c, 1, "the join merges the components");
  strictEqual(paired.faceCount, 3, "the join also merges the two outer orbits");
  paired.removeEdge("C", "F");
  strictEqual(paired.faceCount, 4, "deleting the join restores the two outer orbits");

  // -- rejected updates, and isolated vertices ignored by Euler ----------------------
  const crossing = build(square, squareEdges);
  crossing.addEdge("A", "C");
  throws(() => crossing.addEdge("B", "D"), RangeError, "a crossing chord must be rejected");
  throws(() => crossing.addEdge("A", "B"), RangeError, "duplicate edges must be rejected");
  throws(() => crossing.addEdge("A", "A"), RangeError, "self-loops are not supported");
  throws(() => crossing.addEdge("A", "Z"), RangeError, "unknown endpoints must be rejected");
  strictEqual(crossing.faceCount, 3, "rejected updates must not change the structure");
  verify(crossing, "after rejections");
  strictEqual(crossing.addVertex("A", 9, 9), false, "re-adding a vertex is a no-op");
  crossing.addVertex("ISO", 100, 100);
  ok(crossing.checkEuler().holds, "isolated vertices must not break Euler");

  // -- locality: a chord inside one grid cell touches a handful of darts -------------
  const grid = gridGraph(8, 8);
  deepStrictEqual(eulerData(verify(grid, "8x8 grid")), [81, 144, 65, 1], "grid Euler data");
  grid.addEdge("0,0", "1,1");
  const localVisits = grid.localVisitCount;
  const visitedByOracle = grid.recomputeFaces().reduce((sum, face) => sum + face.length, 0);
  ok(localVisits <= 12, `a unit-cell chord must stay local (visited ${localVisits} darts)`);
  strictEqual(visitedByOracle, aliveDarts(grid), "the oracle walk visits every dart");
  ok(localVisits * 10 < visitedByOracle, "the local update must be an order of magnitude cheaper");
  verify(grid, "grid + chord");
  grid.removeEdge("0,0", "1,1");
  verify(grid, "grid restored");
  strictEqual(grid.faceCount, 65, "the grid is back to 65 faces");
  // -- seeded insertion/deletion workload, verified after every step and replayed -----
  const diagonals = [];
  for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) diagonals.push([`${x},${y}`, `${x + 1},${y + 1}`]);
  let workloadUpdates = 0;
  const runWorkload = (trial) => {
    const g = gridGraph(4, 4);
    const log = [];
    const r = makeRng(987654);
    for (let step = 0; step < 12; step++) {
      const [u, v] = diagonals[r(diagonals.length)];
      if (g.hasEdge(u, v)) { g.removeEdge(u, v); log.push(`-${u}`); }
      else { g.addEdge(u, v); log.push(`+${u}`); }
      workloadUpdates++;
      verify(g, `workload trial ${trial} step ${step}`);
    }
    return { log, faces: g.faceCount, euler: g.checkEuler() };
  };
  deepStrictEqual(runWorkload(0), runWorkload(1), "the same seed must replay the same update sequence");
  ok(runWorkload(2).log.some((e) => e.startsWith("-")), "the workload must exercise deletions too");

  console.log(`updates verified=${updatesVerified + workloadUpdates} grid local=${localVisits} vs full=${visitedByOracle}`);
  console.log("PASS");
}
