/** Task 768 - Implement skyline queries in multi-criteria optimization (Pareto front). (Tier: Expert, CTCI Ch 8 — Recursion and Dynamic Programming)
 *
 * All dimensions are minimised, so a point p is in the skyline (Pareto front) iff no other
 * point q dominates it: q[i] <= p[i] for every i and q[i] < p[i] for at least one i. Four
 * implementations of the same predicate are cross-checked against each other:
 *
 *   bruteForceSkyline  O(n^2) pairwise dominance — the independent oracle.
 *   bnlSkyline         Block Nested Loop: bounded "window" of pairwise-incomparable
 *                      candidates prunes most points in pass 1; the survivors are rechecked
 *                      pairwise in pass 2 (the second pass is what makes BNL exact even when
 *                      a dominator has already been spilled).
 *   sfsSkyline         Sort-Filter-Skyline: sort on the first dimension, then keep a window
 *                      of incomparable candidates. Sorting is safe because a dominator can
 *                      never appear *after* the point it dominates.
 *   staircase2d        O(n log n) 2-D special case: sweep x ascending keeping the running
 *                      minimum of y; a point survives iff its y is at most that minimum.
 *
 * The recursive part of the chapter shows up in `skylineLayers`: repeatedly peeling the
 * skyline of the *remaining* points gives the k-skyline layers, and layer(p) is exactly the
 * dominance depth (longest chain of strict dominators) computed by DP.
 *
 * What is verified: all three general algorithms vs the brute-force oracle on random 2-D..5-D
 * integer clouds (duplicates included), the 2-D staircase vs the oracle, transitivity and
 * irreflexivity of `dominates`, skyline invariance under a strictly monotone rescaling of the
 * axes, layer sum == n with layer(p) == dominance depth, degenerate inputs (empty, single,
 * 1-D, 0-D, all-equal, fully dominated) and input validation.
 */
"use strict";

/** Strict Pareto dominance for equal-length numeric tuples (minimisation). */
function dominates(a, b) {
  if (a.length !== b.length) throw new RangeError("tuples must have the same dimension");
  let strictlySmaller = false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] > b[i]) return false;
    if (a[i] < b[i]) strictlySmaller = true;
  }
  return strictlySmaller;
}

function validate(points) {
  if (!Array.isArray(points)) throw new TypeError("points must be an array of tuples");
  const dim = points.length === 0 ? 0 : points[0].length;
  for (const p of points) {
    if (!Array.isArray(p) || p.length !== dim) throw new TypeError("ragged point cloud");
    for (const v of p) if (typeof v !== "number" || Number.isNaN(v)) throw new TypeError("coordinates must be numbers");
  }
  return dim;
}

function bruteForceSkyline(points) {
  const dim = validate(points);
  if (dim === 0) return points.slice();
  const out = [];
  for (const p of points) {
    let undominated = true;
    for (const q of points) {
      if (q === p) continue;
      if (dominates(q, p)) { undominated = false; break; }
    }
    if (undominated) out.push(p);
  }
  return out;
}

/** Block Nested Loop with a bounded window plus an exact second pass. */
function bnlSkyline(points, windowSize = 8) {
  validate(points);
  if (windowSize < 1) throw new RangeError("windowSize must be >= 1");
  const spilled = [];
  let window = [];
  for (const p of points) {
    let keep = true;
    for (const w of window) if (dominates(w, p)) { keep = false; break; }
    if (!keep) continue;
    // p survives the window, so it may retire window entries it dominates.
    window = window.filter((w) => !dominates(p, w));
    window.push(p);
    if (window.length >= windowSize) { spilled.push(...window); window = []; }
  }
  spilled.push(...window); // flush the tail; `spilled` now holds every pass-1 survivor
  const out = [];
  for (let i = 0; i < spilled.length; i++) {
    let undominated = true;
    for (let j = 0; j < spilled.length; j++) {
      if (i !== j && dominates(spilled[j], spilled[i])) { undominated = false; break; }
    }
    if (undominated) out.push(spilled[i]);
  }
  return out;
}

/** Sort-Filter-Skyline: lexicographic sort makes later points unable to dominate earlier ones. */
function sfsSkyline(points) {
  validate(points);
  const sorted = points.slice().sort(compareTuples);
  const window = [];
  for (const p of sorted) {
    let keep = true;
    for (const w of window) if (dominates(w, p)) { keep = false; break; }
    if (!keep) continue;
    for (let i = window.length - 1; i >= 0; i--) if (dominates(p, window[i])) window.splice(i, 1);
    window.push(p);
  }
  return window;
}

function compareTuples(a, b) {
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i];
  return 0;
}

/** O(n log n) 2-D staircase: group equal x, keep the group minimum while it improves on y. */
function staircase2d(points) {
  const dim = validate(points);
  if (dim !== 2) throw new RangeError("staircase2d expects 2-D points");
  const sorted = points.slice().sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const out = [];
  let bestY = Infinity;
  for (let i = 0; i < sorted.length;) {
    let j = i;
    while (j < sorted.length && sorted[j][0] === sorted[i][0]) j++;
    const minY = sorted[i][1]; // sorted by y inside the x-group
    if (minY < bestY) {
      for (let k = i; k < j && sorted[k][1] === minY; k++) out.push(sorted[k]);
      bestY = minY;
    }
    i = j;
  }
  return out;
}

/** Peels skyline layers; also returns the dominance depth of every point (longest dominator chain). */
function skylineLayers(points) {
  validate(points);
  const remaining = points.slice();
  const layers = [];
  while (remaining.length > 0) {
    const layer = bruteForceSkyline(remaining);
    const inLayer = new Set(layer);
    layers.push(layer);
    for (let i = remaining.length - 1; i >= 0; i--) if (inLayer.has(remaining[i])) remaining.splice(i, 1);
  }
  /**
   * dominanceDepth(p) = 0 when nothing dominates p, else 1 + max depth of a strict dominator.
   * Relaxing that recurrence to a fixed point over all pairs is a DAG DP: strict dominance is
   * acyclic, so the relaxation terminates after at most n rounds.
   */
  const depth = new Map();
  for (const p of points) depth.set(p, 0);
  let changed = true;
  while (changed) {
    changed = false;
    for (const q of points) for (const p of points) {
      if (dominates(q, p) && depth.get(p) < depth.get(q) + 1) {
        depth.set(p, depth.get(q) + 1);
        changed = true;
      }
    }
  }
  return { layers, depth };
}

/** Membership test: is p on the skyline of `points`? */
function isInSkyline(points, p) {
  for (const q of points) if (dominates(q, p)) return false;
  return true;
}

function makeRng(seed) {
  let state = seed >>> 0 || 0x2545f491;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

const asKeys = (pts) => pts.map((p) => p.join(",")).sort();

module.exports = {
  dominates, bruteForceSkyline, bnlSkyline, sfsSkyline, staircase2d, skylineLayers, isInSkyline,
};

if (require.main === module) {
  const assert = require("node:assert");
  const { deepStrictEqual, strictEqual, ok, throws } = assert;
  const rng = makeRng(20260768);
  let totalChecked = 0;

  // -- degenerate clouds ------------------------------------------------------------
  deepStrictEqual(bruteForceSkyline([]), [], "empty cloud has an empty skyline");
  deepStrictEqual(bnlSkyline([]), [], "BNL on empty input");
  deepStrictEqual(sfsSkyline([]), [], "SFS on empty input");
  deepStrictEqual(bruteForceSkyline([[5]]), [[5]], "single 1-D point survives");
  deepStrictEqual(bruteForceSkyline([[3], [1], [2]]), [[1]], "1-D skyline is the minimum");
  deepStrictEqual(bruteForceSkyline([[9, 9], [5, 5]]), [[5, 5]], "the dominated point is dropped");
  deepStrictEqual(bruteForceSkyline([[1, 1], [1, 1]]), [[1, 1], [1, 1]],
    "equal tuples do not strictly dominate each other, so both stay");
  deepStrictEqual(bruteForceSkyline([[1, 2], [2, 1], [2, 2]]), [[1, 2], [2, 1]],
    "incomparable pair survives, the trade-off point does not");
  const degenerate = [[4, 4], [4, 4], [3, 7], [7, 3]];
  deepStrictEqual(asKeys(bruteForceSkyline(degenerate)), asKeys(bnlSkyline(degenerate)), "dup handling BNL");
  deepStrictEqual(asKeys(bruteForceSkyline(degenerate)), asKeys(sfsSkyline(degenerate)), "dup handling SFS");
  deepStrictEqual(asKeys(bruteForceSkyline(degenerate)), asKeys(staircase2d(degenerate)), "dup handling staircase");

  // -- dominates predicate properties ------------------------------------------------
  ok(!dominates([1, 2], [1, 2]), "irreflexive for equal tuples");
  ok(dominates([1, 1], [1, 2]), "equal on one axis, better on the other strictly dominates");
  ok(!dominates([1, 3], [1, 2]), "worse on one axis cannot dominate");
  const a = [0, 1], b = [1, 2], c = [2, 3];
  ok(dominates(a, b) && dominates(b, c) && dominates(a, c), "dominance is transitive");
  ok(!dominates(c, a), "dominance is antisymmetric");
  throws(() => dominates([1], [1, 2]), RangeError, "dimension mismatch must throw");
  throws(() => bruteForceSkyline([[1, 2], [3]]), TypeError, "ragged cloud must throw");
  throws(() => bruteForceSkyline([[1, "x"]]), TypeError, "non-numeric coordinate must throw");
  throws(() => staircase2d([[1, 2, 3]]), RangeError, "staircase is 2-D only");
  throws(() => bnlSkyline([[1, 2]], 0), RangeError, "windowSize must be positive");

  // -- random clouds, 2-D..5-D, all four algorithms ----------------------------------
  for (let trial = 0; trial < 40; trial++) {
    const dim = 2 + (trial % 4);
    const n = 1 + rng(40);
    const spread = 1 + rng(6); // small spread -> many ties and many dominated points
    const pts = Array.from({ length: n }, () => Array.from({ length: dim }, () => rng(spread)));
    const oracle = bruteForceSkyline(pts);
    deepStrictEqual(asKeys(sfsSkyline(pts)), asKeys(oracle), `SFS mismatch on trial ${trial}`);
    deepStrictEqual(asKeys(bnlSkyline(pts, 1 + rng(5))), asKeys(oracle), `BNL mismatch on trial ${trial}`);
    for (const p of pts) strictEqual(isInSkyline(pts, p), oracle.includes(p), "membership test disagrees with oracle");
    if (dim === 2) {
      deepStrictEqual(asKeys(staircase2d(pts)), asKeys(oracle), `staircase mismatch on trial ${trial}`);
    }
    // every skyline member is undominated, and the skyline is an antichain
    for (const p of oracle) for (const q of oracle) if (p !== q) ok(!dominates(p, q), "skyline must be an antichain");
    totalChecked += pts.length;
  }

  // -- the empty tuple space (0-D) is a boundary: everything is mutually non-dominating ---
  deepStrictEqual(bruteForceSkyline([[], []]), [[], []], "0-D points cannot dominate each other");

  // -- invariance under a strictly increasing rescaling of each axis -----------------
  const base = Array.from({ length: 30 }, () => [rng(10), rng(10), rng(10)]);
  const scaled = base.map((p) => p.map((v, i) => (i + 1) * v * v + 3 * i)); // v -> v*v is strictly increasing on v >= 0
  strictEqual(bruteForceSkyline(base).length, bruteForceSkyline(scaled).length,
    "a strictly monotone per-axis map preserves the skyline size");
  for (let i = 0; i < base.length; i++) {
    strictEqual(isInSkyline(scaled, scaled[i]), isInSkyline(base, base[i]), "rescaling preserves membership");
  }

  // -- peptide layers: sum of layer sizes == n and layer == dominance depth ----------
  const { layers, depth } = skylineLayers(base);
  strictEqual(layers.reduce((s, l) => s + l.length, 0), base.length, "layers partition the cloud");
  deepStrictEqual(asKeys(layers[0]), asKeys(bruteForceSkyline(base)), "layer 0 is the skyline");
  for (let i = 0; i < base.length; i++) {
    const layerIndex = layers.findIndex((l) => l.includes(base[i]));
    strictEqual(layerIndex, depth.get(base[i]), `dominance depth of point ${i} must equal its layer`);
  }
  // peeling is monotone: removing layers cannot increase a point's depth
  const sub = base.filter((p) => !layers[0].includes(p));
  for (const p of sub) ok(skylineLayers(sub).depth.get(p) <= depth.get(p), "depth cannot grow after peeling");
  deepStrictEqual(skylineLayers([]), { layers: [], depth: new Map() }, "empty input has no layers");

  // -- 2-D worst case: a full antichain staircase of size n --------------------------
  const antichain = Array.from({ length: 60 }, (_, i) => [i, 60 - i]);
  strictEqual(bruteForceSkyline(antichain).length, 60, "a decreasing staircase is fully undominated");
  strictEqual(staircase2d(antichain).length, 60, "staircase agrees on the antichain");
  strictEqual(sfsSkyline(antichain).length, 60, "SFS agrees on the antichain");

  console.log(`random trials=40 points checked=${totalChecked} layers=${layers.length}`);
  console.log("PASS");
}
