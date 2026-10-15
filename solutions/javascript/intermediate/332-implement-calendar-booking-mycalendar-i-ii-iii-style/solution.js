/** Task 332 - Implement calendar booking (MyCalendar I/II/III style). (Tier: Intermediate, CTCI Ch 7) */
"use strict";

/**
 * Three interval-bookkeeping designs over half-open [start, end) bookings:
 *   MyCalendar  - accept only bookings overlapping nothing accepted;
 *   MyCalendarTwo - double booking allowed, reject only a would-be triple overlap
 *                   (singles + overlapping pairs);
 *   MyCalendarK - at most k simultaneous bookings (III generalisation), kept as a delta map
 *                 with a max-prefix sweep. All variants cross-checked against a sweep oracle.
 */

function overlaps(aStart, aEnd, other) {
  return aStart < other[1] && other[0] < aEnd;
}

/** Maximum number of intervals covering any single instant (sweep line). */
function sweepPeak(intervals) {
  const events = new Map();
  for (const [start, end] of intervals) {
    events.set(start, (events.get(start) || 0) + 1);
    events.set(end, (events.get(end) || 0) - 1);
  }
  let current = 0, peak = 0;
  for (const time of [...events.keys()].sort((a, b) => a - b)) {
    current += events.get(time);
    peak = Math.max(peak, current);
  }
  return peak;
}

class MyCalendar {
  constructor() {
    this.accepted = [];
  }

  book(start, end) {
    if (this.accepted.some((other) => overlaps(start, end, other))) return false;
    this.accepted.push([start, end]);
    return true;
  }
}

class MyCalendarTwo {
  constructor() {
    this.singles = [];
    this.pairs = [];
  }

  book(start, end) {
    if (this.pairs.some((other) => overlaps(start, end, other))) return false;
    for (const other of this.singles) {
      if (overlaps(start, end, other)) {
        this.pairs.push([Math.max(start, other[0]), Math.min(end, other[1])]);
      }
    }
    this.singles.push([start, end]);
    return true;
  }

  maxOverlap() {
    return sweepPeak(this.singles);
  }
}

class MyCalendarK {
  constructor(k) {
    if (k < 1) throw new Error("k must be >= 1");
    this.k = k;
    this.delta = new Map();
  }

  book(start, end) {
    this.delta.set(start, (this.delta.get(start) || 0) + 1);
    this.delta.set(end, (this.delta.get(end) || 0) - 1);
    if (this.peak() > this.k) {
      this.delta.set(start, this.delta.get(start) - 1);
      this.delta.set(end, this.delta.get(end) + 1);
      return false;
    }
    return true;
  }

  maxOverlap() {
    return this.peak();
  }

  peak() {
    let current = 0, peak = 0;
    for (const time of [...this.delta.keys()].sort((a, b) => a - b)) {
      current += this.delta.get(time);
      peak = Math.max(peak, current);
    }
    return peak;
  }
}

let seed = 1717;

function lcgBookings(count, span) {
  const bookings = [];
  for (let i = 0; i < count; i++) {
    seed = (seed * 75 + 74) % 65537;
    const start = seed % span;
    seed = (seed * 75 + 74) % 65537;
    bookings.push([start, start + 1 + (seed % 9)]);
  }
  return bookings;
}

module.exports = { MyCalendar, MyCalendarTwo, MyCalendarK, sweepPeak };

if (require.main === module) {
  const assert = require("node:assert");

  // MyCalendar I, hand-checked.
  const cal = new MyCalendar();
  assert.strictEqual(cal.book(10, 20), true);
  assert.strictEqual(cal.book(15, 25), false);
  assert.strictEqual(cal.book(20, 30), true);
  assert.strictEqual(cal.book(5, 10), true);
  assert.strictEqual(cal.book(19, 20), false);
  assert.strictEqual(cal.book(10, 15), false);
  assert.deepStrictEqual(cal.accepted, [[10, 20], [20, 30], [5, 10]]);

  // MyCalendar II, the classic LeetCode sequence.
  const two = new MyCalendarTwo();
  const classic = [[10, 20], [50, 60], [10, 40], [5, 15], [5, 10], [25, 55]];
  assert.deepStrictEqual(classic.map(([s, e]) => two.book(s, e)),
    [true, true, true, false, true, true]);
  assert.strictEqual(two.maxOverlap(), 2);
  assert.strictEqual(two.book(30, 45), false);       // would triple-cover [30, 40)
  assert.strictEqual(two.book(60, 70), true);

  // MyCalendar III with k = 3: the same sequence is fully accepted (peak is exactly 3).
  const three = new MyCalendarK(3);
  assert.deepStrictEqual(classic.map(([s, e]) => three.book(s, e)), [true, true, true, true, true, true]);
  assert.strictEqual(three.maxOverlap(), 3);
  assert.strictEqual(three.book(12, 13), false);     // fourth cover at t = 12

  // k = 2 variant: the tail never exceeds two deep, but (12, 13) would.
  const twoK = new MyCalendarK(2);
  assert.deepStrictEqual(classic.map(([s, e]) => twoK.book(s, e)),
    [true, true, true, false, true, true]);
  assert.strictEqual(twoK.maxOverlap(), 2);
  assert.strictEqual(twoK.book(12, 13), false);

  assert.strictEqual(sweepPeak([]), 0);
  assert.strictEqual(sweepPeak([[1, 2]]), 1);
  assert.strictEqual(sweepPeak([[1, 5], [2, 6], [3, 7]]), 3);
  assert.strictEqual(sweepPeak([[1, 2], [2, 3]]), 1);

  // LCG cross-check: book() must match the sweep oracle, and MyCalendar II must agree with
  // MyCalendarK(2) on the same input.
  for (const k of [1, 2, 3]) {
    seed = 1717;
    const bookings = lcgBookings(240, 40);
    const calendarK = new MyCalendarK(k);
    const twoForComparison = k === 2 ? new MyCalendarTwo() : null;
    const oracle = [];
    for (const [start, end] of bookings) {
      const expected = sweepPeak([...oracle, [start, end]]) <= k;
      const actual = calendarK.book(start, end);
      assert.strictEqual(actual, expected, `k=${k} [${start},${end}]`);
      if (actual) oracle.push([start, end]);
      if (twoForComparison) {
        assert.strictEqual(twoForComparison.book(start, end), actual, `II vs k=2 [${start},${end}]`);
      }
      assert.strictEqual(calendarK.maxOverlap(), sweepPeak(oracle));
      assert.ok(calendarK.maxOverlap() <= k);
    }
  }

  seed = 2626;
  const bookings = lcgBookings(200, 40);
  const one = new MyCalendar();
  const oneK = new MyCalendarK(1);
  const oracle = [];
  for (const [start, end] of bookings) {
    const expected = sweepPeak([...oracle, [start, end]]) <= 1;
    assert.strictEqual(one.book(start, end), expected, `I [${start},${end}]`);
    assert.strictEqual(oneK.book(start, end), expected, `I via k=1 [${start},${end}]`);
    if (expected) oracle.push([start, end]);
  }
  console.log("PASS");
}
