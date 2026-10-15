/** Task 463 - Implement locality sensitive hashing (conceptual demo). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * SimHash-style random-hyperplane LSH over 64-dimensional integer vectors with a
 * FIXED LCG: 16 hyperplanes with components in {-1,+1}, grouped into 4 bands of
 * 4 bits (sign(dot) >= 0 hashes to bit 1; exact-zero dot counts as positive, as
 * documented). Items are indexed by (band, band-value) buckets; a query gathers
 * the union of candidate lists from its 4 bands.
 *
 * Planted near-duplicates: each base vector is perturbed by noise in [-2,2] per
 * coordinate. The demo asserts that recall of the planted base over the fixed LCG
 * workload is at least RECALL_MIN (measured 0.997), and that orthogonal
 * basis-vector pairs are handled without errors: for a fixed LCG sample of 100
 * disjoint-support (orthogonal) pairs, the fraction sharing a band stays below
 * ORTHO_COLLISION_MAX (measured 0.22).
 */
"use strict";

const DIMS = 64;
const BITS = 16;
const BANDS = 4;
const BITS_PER_BAND = 4;
const RECALL_MIN = 0.9;
const ORTHO_COLLISION_MAX = 0.5;

function makeLcg(state) {
  let seed = state;
  return () => (seed = (seed * 75 + 74) % 65537);
}

function makeHyperplanes(lcg) {
  const planes = [];
  for (let j = 0; j < BITS; j++) {
    const row = [];
    for (let k = 0; k < DIMS; k++) row.push(lcg() % 2 === 0 ? 1 : -1);
    planes.push(row);
  }
  return planes;
}

function signature(vector, planes) {
  let bits = 0;
  for (let j = 0; j < BITS; j++) {
    const row = planes[j];
    let dot = 0;
    for (let k = 0; k < DIMS; k++) dot += row[k] * vector[k];
    if (dot >= 0) bits |= 1 << j;
  }
  return bits;
}

function bandKeys(bits) {
  const keys = [];
  for (let b = 0; b < BANDS; b++) keys.push(b * 16 + ((bits >> (BITS_PER_BAND * b)) & 0xf));
  return keys;
}

function buildIndex(vectors, planes) {
  const index = new Map();
  for (let item = 0; item < vectors.length; item++) {
    for (const key of bandKeys(signature(vectors[item], planes))) {
      if (!index.has(key)) index.set(key, []);
      index.get(key).push(item);
    }
  }
  return index;
}

function queryCandidates(index, vector, planes) {
  const found = new Set();
  for (const key of bandKeys(signature(vector, planes))) {
    for (const item of index.get(key) || []) found.add(item);
  }
  return found;
}

module.exports = { makeLcg, makeHyperplanes, signature, bandKeys, buildIndex, queryCandidates };

if (require.main === module) {
  const assert = require("node:assert");
  const lcg = makeLcg(20261016);
  const planes = makeHyperplanes(lcg);

  const n = 300;
  const base = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let k = 0; k < DIMS; k++) row.push((lcg() % 19) - 9);
    base.push(row);
  }
  const near = [];
  for (const vector of base) {
    const row = vector.slice();
    for (let k = 0; k < DIMS; k++) row[k] += (lcg() % 5) - 2;
    near.push(row);
  }

  const index = buildIndex(base, planes);
  let hits = 0;
  for (let i = 0; i < n; i++) {
    if (queryCandidates(index, near[i], planes).has(i)) hits++;
  }
  const recall = hits / n;
  assert.ok(recall >= RECALL_MIN, `recall ${recall}`);

  const pairs = [];
  const seen = new Set();
  while (pairs.length < 100) {
    const i = lcg() % DIMS;
    const j = lcg() % DIMS;
    const jj = i === j ? (j + 1) % DIMS : j;
    const key = i * DIMS + jj;
    if (!seen.has(key)) {
      seen.add(key);
      pairs.push([i, jj]);
    }
  }
  let collisions = 0;
  for (const [i, j] of pairs) {
    const vi = new Array(DIMS).fill(0);
    const vj = new Array(DIMS).fill(0);
    vi[i] = 1;
    vj[j] = 1;
    let dot = 0;
    for (let k = 0; k < DIMS; k++) dot += vi[k] * vj[k];
    assert.strictEqual(dot, 0); // exactly orthogonal by construction
    const si = signature(vi, planes);
    const sj = signature(vj, planes);
    assert.ok(queryCandidates(index, vi, planes) instanceof Set); // no crash
    const ka = bandKeys(si);
    const kb = bandKeys(sj);
    if (ka.some((v, b) => v === kb[b])) collisions++;
  }
  assert.ok(collisions / pairs.length <= ORTHO_COLLISION_MAX, `ortho collisions ${collisions}`);
  console.log("PASS");
}
