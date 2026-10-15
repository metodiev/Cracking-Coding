/** Task 328 - Skyline problem (heap + sweep line). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
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

/** Sweep all critical x values; a max-heap keyed by (-height, right) gives the current height. */
function skylineHeap(buildings) {
  if (buildings.length === 0) return [];
  const starts = new Map();
  const xs = new Set();
  for (const [left, right, height] of buildings) {
    if (!starts.has(left)) starts.set(left, []);
    starts.get(left).push([right, height]);
    xs.add(left);
    xs.add(right);
  }
  const heap = []; // entries [-height, right], ordered by (height desc, right asc)
  const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);
  const push = (entry) => {
    heap.push(entry);
    let child = heap.length - 1;
    while (child > 0) {
      const parent = (child - 1) >> 1;
      if (!less(heap[child], heap[parent])) break;
      [heap[parent], heap[child]] = [heap[child], heap[parent]];
      child = parent;
    }
  };
  const pop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      let parent = 0;
      for (;;) {
        const left = parent * 2 + 1;
        if (left >= heap.length) break;
        const right = left + 1;
        let smallest = left;
        if (right < heap.length && less(heap[right], heap[left])) smallest = right;
        if (!less(heap[smallest], heap[parent])) break;
        [heap[parent], heap[smallest]] = [heap[smallest], heap[parent]];
        parent = smallest;
      }
    }
    return top;
  };
  const result = [];
  for (const x of [...xs].sort((u, v) => u - v)) {
    for (const [right, height] of starts.get(x) ?? []) push([-height, right]);
    while (heap.length > 0 && heap[0][1] <= x) pop(); // lazy deletion of finished buildings
    const height = heap.length > 0 ? -heap[0][0] : 0;
    if (result.length === 0 || result[result.length - 1][1] !== height) {
      result.push([x, height]);
    }
  }
  return result;
}

/** Cross-check implementation: recursive split with a linear skyline merge. */
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

function lcgBuildings(count, lcg) {
  const buildings = [];
  for (let i = 0; i < count; i++) {
    const left = lcg.next(40);
    buildings.push([left, left + 1 + lcg.next(10), 1 + lcg.next(20)]);
  }
  return buildings;
}

const CANONICAL = [[2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8]];
const CANONICAL_SKYLINE = [[2, 10], [3, 15], [7, 12], [12, 0], [15, 10], [20, 8], [24, 0]];

module.exports = { skylineHeap, skylineDivideConquer, mergeSkylines, lcgBuildings, Lcg };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(skylineHeap(CANONICAL), CANONICAL_SKYLINE);
  assert.deepStrictEqual(skylineDivideConquer(CANONICAL), CANONICAL_SKYLINE);

  assert.deepStrictEqual(skylineHeap([]), []);
  assert.deepStrictEqual(skylineDivideConquer([]), []);
  assert.deepStrictEqual(skylineHeap([[0, 2, 5], [2, 4, 7]]), [[0, 5], [2, 7], [4, 0]]);
  assert.deepStrictEqual(skylineHeap([[0, 2, 5], [5, 8, 3]]), [[0, 5], [2, 0], [5, 3], [8, 0]]);

  const lcg = new Lcg(24601);
  for (let trial = 0; trial < 300; trial++) {
    const count = 1 + lcg.next(8);
    const buildings = lcgBuildings(count, lcg);
    assert.deepStrictEqual(
      skylineHeap(buildings),
      skylineDivideConquer(buildings),
      `trial ${trial}`
    );
  }
  console.log("PASS");
}
