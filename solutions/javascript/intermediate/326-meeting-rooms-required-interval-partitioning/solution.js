/** Task 326 - Meeting rooms required (interval partitioning). (Tier: Intermediate, CTCI Ch 1) */
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

/** Maximum overlap via endpoint events; an end (-1) sorts before a start (+1). */
function minMeetingRooms(intervals) {
  const events = [];
  for (const [start, end] of intervals) {
    events.push([start, 1]);
    events.push([end, -1]);
  }
  events.sort((u, v) => u[0] - v[0] || u[1] - v[1]);
  let rooms = 0;
  let best = 0;
  for (const [, delta] of events) {
    rooms += delta;
    if (rooms > best) best = rooms;
  }
  return best;
}

/** Independent sweep with a min-heap of finish times. */
function minRoomsHeap(intervals) {
  const ordered = intervals.map((iv) => iv.slice()).sort((u, v) => u[0] - v[0] || u[1] - v[1]);
  const heap = [];
  const push = (value) => {
    heap.push(value);
    let child = heap.length - 1;
    while (child > 0) {
      const parent = (child - 1) >> 1;
      if (heap[parent] <= heap[child]) break;
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
        if (right < heap.length && heap[right] < heap[left]) smallest = right;
        if (heap[parent] <= heap[smallest]) break;
        [heap[parent], heap[smallest]] = [heap[smallest], heap[parent]];
        parent = smallest;
      }
    }
    return top;
  };
  for (const [start, end] of ordered) {
    if (heap.length > 0 && heap[0] <= start) pop();
    push(end);
  }
  return heap.length;
}

/** Independent oracle for small coordinates: count active meetings at each minute. */
function minRoomsPerMinute(intervals) {
  if (intervals.length === 0) return 0;
  const lo = Math.min(...intervals.map((iv) => iv[0]));
  const hi = Math.max(...intervals.map((iv) => iv[1]));
  let best = 0;
  for (let minute = lo; minute <= hi; minute++) {
    let active = 0;
    for (const [start, end] of intervals) {
      if (start <= minute && minute < end) active++;
    }
    if (active > best) best = active;
  }
  return best;
}

function lcgIntervals(count, lcg) {
  const intervals = [];
  for (let i = 0; i < count; i++) {
    const start = lcg.next(30);
    intervals.push([start, start + 1 + lcg.next(10)]);
  }
  return intervals;
}

module.exports = { minMeetingRooms, minRoomsHeap, minRoomsPerMinute, lcgIntervals, Lcg };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(minMeetingRooms([[0, 30], [5, 10], [15, 20]]), 2);
  assert.strictEqual(minMeetingRooms([[7, 10], [2, 4]]), 1);
  assert.strictEqual(minRoomsHeap([[0, 30], [5, 10], [15, 20]]), 2);
  assert.strictEqual(minRoomsHeap([[7, 10], [2, 4]]), 1);

  assert.strictEqual(minMeetingRooms([[1, 2], [2, 3], [3, 4]]), 1); // back-to-back shares a room
  assert.strictEqual(minMeetingRooms([[1, 5], [2, 6], [3, 7]]), 3);
  assert.strictEqual(minMeetingRooms([]), 0);
  assert.strictEqual(minRoomsHeap([]), 0);

  const lcg = new Lcg(2718);
  for (let trial = 0; trial < 300; trial++) {
    const count = 1 + lcg.next(10);
    const intervals = lcgIntervals(count, lcg);
    const expected = minRoomsPerMinute(intervals);
    assert.strictEqual(minMeetingRooms(intervals), expected, `event trial ${trial}`);
    assert.strictEqual(minRoomsHeap(intervals), expected, `heap trial ${trial}`);
  }
  console.log("PASS");
}
