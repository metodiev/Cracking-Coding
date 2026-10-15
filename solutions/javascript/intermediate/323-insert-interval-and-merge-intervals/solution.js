/** Task 323 - Insert interval and merge intervals. (Tier: Intermediate, CTCI Ch 1) */
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

/** Sort by start and merge overlapping or touching intervals (start <= current end). */
function mergeIntervals(intervals) {
  if (intervals.length === 0) return [];
  const ordered = intervals.map((iv) => iv.slice()).sort((u, v) => u[0] - v[0] || u[1] - v[1]);
  const merged = [ordered[0]];
  for (let i = 1; i < ordered.length; i++) {
    const last = merged[merged.length - 1];
    if (ordered[i][0] <= last[1]) {
      last[1] = Math.max(last[1], ordered[i][1]);
    } else {
      merged.push(ordered[i]);
    }
  }
  return merged;
}

/** Insert into a sorted non-overlapping list, merging overlapping/touching parts. */
function insertInterval(intervals, newInterval) {
  let [start, end] = newInterval;
  const result = [];
  let i = 0;
  while (i < intervals.length && intervals[i][1] < start) {
    result.push(intervals[i].slice());
    i++;
  }
  while (i < intervals.length && intervals[i][0] <= end) {
    start = Math.min(start, intervals[i][0]);
    end = Math.max(end, intervals[i][1]);
    i++;
  }
  result.push([start, end]);
  while (i < intervals.length) {
    result.push(intervals[i].slice());
    i++;
  }
  return result;
}

/** Independent oracle: endpoint events; starts sort before ends at the same x. */
function mergeIntervalsSweep(intervals) {
  const events = [];
  for (const [start, end] of intervals) {
    events.push([start, 0]);
    events.push([end, 1]);
  }
  events.sort((u, v) => u[0] - v[0] || u[1] - v[1]);
  const merged = [];
  let depth = 0;
  for (const [x, kind] of events) {
    if (kind === 0) {
      if (depth === 0) merged.push([x, 0]);
      depth++;
    } else {
      depth--;
      if (depth === 0) merged[merged.length - 1][1] = x;
    }
  }
  return merged;
}

function lcgIntervals(count, lcg) {
  const intervals = [];
  for (let i = 0; i < count; i++) {
    const start = lcg.next(25);
    intervals.push([start, start + 1 + lcg.next(8)]);
  }
  return intervals;
}

function lcgDisjointSorted(count, lcg) {
  const intervals = [];
  let cursor = 0;
  for (let i = 0; i < count; i++) {
    cursor += 1 + lcg.next(4);
    const end = cursor + 1 + lcg.next(5);
    intervals.push([cursor, end]);
    cursor = end + 1;
  }
  return intervals;
}

module.exports = { mergeIntervals, insertInterval, mergeIntervalsSweep, lcgIntervals, lcgDisjointSorted, Lcg };

if (require.main === module) {
  const assert = require("node:assert");
  assert.deepStrictEqual(mergeIntervals([[1, 3], [2, 6], [8, 10], [15, 18]]), [[1, 6], [8, 10], [15, 18]]);
  assert.deepStrictEqual(mergeIntervals([[1, 2], [2, 3]]), [[1, 3]]); // touching-merge policy
  assert.deepStrictEqual(mergeIntervals([[1, 4], [4, 5], [7, 9]]), [[1, 5], [7, 9]]);
  assert.deepStrictEqual(mergeIntervals([[1, 10], [2, 3], [4, 5]]), [[1, 10]]);
  assert.deepStrictEqual(mergeIntervals([[5, 5]]), [[5, 5]]);
  assert.deepStrictEqual(mergeIntervals([]), []);
  assert.deepStrictEqual(mergeIntervalsSweep([[1, 2], [2, 3]]), [[1, 3]]);

  const base = [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]];
  assert.deepStrictEqual(insertInterval(base, [9, 10]), [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]);
  assert.deepStrictEqual(insertInterval(base, [0, 0]), [[0, 0], [1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]);
  assert.deepStrictEqual(insertInterval(base, [4, 4]), [[1, 2], [3, 5], [6, 7], [8, 10], [12, 16]]);
  assert.deepStrictEqual(insertInterval([[1, 2], [5, 7]], [2, 5]), [[1, 7]]); // touches both sides
  assert.deepStrictEqual(insertInterval([[3, 4]], [1, 2]), [[1, 2], [3, 4]]);
  assert.deepStrictEqual(insertInterval([], [4, 6]), [[4, 6]]);

  const lcg = new Lcg(20077);
  for (let trial = 0; trial < 200; trial++) {
    const intervals = lcgIntervals(6, lcg);
    assert.deepStrictEqual(mergeIntervals(intervals), mergeIntervalsSweep(intervals), `trial ${trial}`);
  }

  const insertLcg = new Lcg(4242);
  for (let trial = 0; trial < 200; trial++) {
    const baseSorted = lcgDisjointSorted(5, insertLcg);
    const newStart = insertLcg.next(30);
    const newInterval = [newStart, newStart + 1 + insertLcg.next(8)];
    assert.deepStrictEqual(
      insertInterval(baseSorted, newInterval),
      mergeIntervals(baseSorted.concat([newInterval])),
      `insert trial ${trial}`
    );
  }
  console.log("PASS");
}
