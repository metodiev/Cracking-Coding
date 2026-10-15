/** Task 322 - Implement algorithm to find skyline (divide & conquer). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
"use strict";

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
  constructor(seed) {
    this.seed = seed;
  }

  next(modulus) {
    this.seed = (this.seed * 75 + 74) % 65537;
    return this.seed % modulus;
  }
}

/** buildings: [left, right, height]; returns critical points [x, height]. */
function skylineDivideConquer(buildings) {
  if (buildings.length === 0) return [];
  if (buildings.length === 1) {
    const [left, right, height] = buildings[0];
    return [[left, height], [right, 0]];
  }
  const mid = Math.floor(buildings.length / 2);
  return mergeSkylines(
    skylineDivideConquer(buildings.slice(0, mid)),
    skylineDivideConquer(buildings.slice(mid))
  );
}

/** Linear merge of two x-sorted skylines of [x, height] critical points. */
function mergeSkylines(a, b) {
  const merged = [];
  let i = 0;
  let j = 0;
  let heightA = 0;
  let heightB = 0;
  while (i < a.length || j < b.length) {
    let x;
    if (j >= b.length || (i < a.length && a[i][0] < b[j][0])) {
      [x, heightA] = a[i];
      i++;
    } else if (i >= a.length || b[j][0] < a[i][0]) {
      [x, heightB] = b[j];
      j++;
    } else {
      [x, heightA] = a[i];
      [, heightB] = b[j];
      i++;
      j++;
    }
    const height = Math.max(heightA, heightB);
    if (merged.length === 0 || merged[merged.length - 1][1] !== height) {
      merged.push([x, height]);
    }
  }
  return merged;
}

/** Independent oracle: sweep all boundaries keeping a count of active heights. */
function skylineSweep(buildings) {
  if (buildings.length === 0) return [];
  let maxHeight = 0;
  const starts = new Map();
  const ends = new Map();
  for (const [left, right, height] of buildings) {
    if (!starts.has(left)) starts.set(left, []);
    starts.get(left).push(height);
    if (!ends.has(right)) ends.set(right, []);
    ends.get(right).push(height);
    maxHeight = Math.max(maxHeight, height);
  }
  const counts = new Array(maxHeight + 1).fill(0);
  let current = 0;
  const result = [];
  const xs = [...new Set([...starts.keys(), ...ends.keys()])].sort((u, v) => u - v);
  for (const x of xs) {
    for (const height of ends.get(x) ?? []) counts[height]--; // active on [left, right)
    for (const height of starts.get(x) ?? []) {
      counts[height]++;
      if (height > current) current = height;
    }
    while (current > 0 && counts[current] === 0) current--;
    if (result.length === 0 || result[result.length - 1][1] !== current) {
      result.push([x, current]);
    }
  }
  return result;
}

function lcgBuildings(count, lcg) {
  const buildings = [];
  for (let i = 0; i < count; i++) {
    const left = lcg.next(40);
    const width = 1 + lcg.next(10);
    const height = 1 + lcg.next(20);
    buildings.push([left, left + width, height]);
  }
  return buildings;
}

const CANONICAL = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]];
const CANONICAL_SKYLINE = [[2, 10], [3, 15], [7, 12], [12, 0], [15, 10], [20, 8], [24, 0]];

module.exports = { skylineDivideConquer, mergeSkylines, skylineSweep, lcgBuildings, Lcg };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(skylineDivideConquer(CANONICAL), CANONICAL_SKYLINE);
  assert.deepStrictEqual(skylineSweep(CANONICAL), CANONICAL_SKYLINE);

  assert.deepStrictEqual(skylineDivideConquer([]), []);
  assert.deepStrictEqual(skylineSweep([]), []);
  assert.deepStrictEqual(skylineDivideConquer([[1, 3, 4]]), [[1, 4], [3, 0]]);
  assert.deepStrictEqual(skylineDivideConquer([[0, 2, 5], [2, 4, 7]]), [[0, 5], [2, 7], [4, 0]]);
  assert.deepStrictEqual(skylineDivideConquer([[0, 2, 5], [5, 8, 3]]), [[0, 5], [2, 0], [5, 3], [8, 0]]);
  assert.deepStrictEqual(skylineSweep([[0, 2, 5], [5, 8, 3]]), [[0, 5], [2, 0], [5, 3], [8, 0]]);

  const lcg = new Lcg(1123);
  for (let trial = 0; trial < 300; trial++) {
    const count = 1 + lcg.next(8);
    const buildings = lcgBuildings(count, lcg);
    assert.deepStrictEqual(skylineDivideConquer(buildings), skylineSweep(buildings), `trial ${trial}`);
  }
  console.log("PASS");
}
