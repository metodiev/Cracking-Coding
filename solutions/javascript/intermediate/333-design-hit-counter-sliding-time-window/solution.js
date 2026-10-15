/** Task 333 - Design hit counter (sliding time window). (Tier: Intermediate, CTCI Ch 7) */
"use strict";

/**
 * Hits arrive with non-decreasing timestamps; getHits(t) counts the hits h recorded so far
 * with t - 299 <= h <= t, i.e. the classic 300-second sliding window. Keeps the timestamps
 * in one array and answers with two binary searches. Cross-checked against a naive scan.
 */

const WINDOW = 300;

class HitCounter {
  constructor() {
    this.times = [];
  }

  hit(timestamp) {
    if (this.times.length > 0) {
      if (timestamp < this.times[this.times.length - 1]) {
        throw new Error("timestamps must not go backwards");
      }
    }
    this.times.push(timestamp);
  }

  /** Number of hits h with timestamp - 299 <= h <= timestamp. */
  getHits(timestamp) {
    return this.firstGreater(timestamp) - this.firstAtLeast(timestamp - (WINDOW - 1));
  }

  firstAtLeast(timestamp) {
    let lo = 0, hi = this.times.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.times[mid] < timestamp) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  firstGreater(timestamp) {
    let lo = 0, hi = this.times.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.times[mid] <= timestamp) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
}

function naiveHits(times, timestamp) {
  return times.filter((h) => timestamp - (WINDOW - 1) <= h && h <= timestamp).length;
}

module.exports = { HitCounter, naiveHits };

if (require.main === module) {
  const assert = require("node:assert");

  // Hand-checked: hits at 1, 2, 3 -> get(4) = 3; windows slide away one at a time.
  const counter = new HitCounter();
  for (const t of [1, 2, 3]) counter.hit(t);
  assert.strictEqual(counter.getHits(1), 1);
  assert.strictEqual(counter.getHits(2), 2);
  assert.strictEqual(counter.getHits(3), 3);
  assert.strictEqual(counter.getHits(4), 3);
  assert.strictEqual(counter.getHits(299), 3);
  assert.strictEqual(counter.getHits(300), 3);   // 300 - 299 = 1 still counts hit 1
  assert.strictEqual(counter.getHits(301), 2);   // hit 1 left the window
  assert.strictEqual(counter.getHits(302), 1);
  assert.strictEqual(counter.getHits(303), 0);

  // A later hit at 300 joins the window: get(300) = 4, get(301) = 3, get(302) = 2.
  counter.hit(300);
  assert.strictEqual(counter.getHits(300), 4);
  assert.strictEqual(counter.getHits(301), 3);
  assert.strictEqual(counter.getHits(302), 2);
  assert.strictEqual(counter.getHits(599), 1);
  assert.strictEqual(counter.getHits(600), 0);

  // LeetCode-flavoured sequence: hits at 1, 2, 3 and (again) at 300.
  const leet = new HitCounter();
  for (const t of [1, 2, 3, 300]) leet.hit(t);
  assert.strictEqual(leet.getHits(3), 3);
  assert.strictEqual(leet.getHits(300), 4);
  assert.strictEqual(leet.getHits(301), 3);
  assert.strictEqual(leet.getHits(2), 2);        // query before the newest hit
  assert.strictEqual(leet.getHits(0), 0);

  // Same-timestamp bursts.
  const burst = new HitCounter();
  for (let i = 0; i < 5; i++) burst.hit(7);
  assert.strictEqual(burst.getHits(7), 5);
  assert.strictEqual(burst.getHits(306), 5);
  assert.strictEqual(burst.getHits(307), 0);

  // LCG event stream against the naive scan.
  let seed = 2027;
  let now = 0;
  const stream = new HitCounter();
  const history = [];
  for (let step = 0; step < 4000; step++) {
    seed = (seed * 75 + 74) % 65537;
    now += seed % 5;
    seed = (seed * 75 + 74) % 65537;
    if (seed % 4 === 0) {
      seed = (seed * 75 + 74) % 65537;
      const timestamp = now + (seed % 3);
      stream.hit(timestamp);
      history.push(timestamp);
      now = timestamp;
    } else {
      assert.strictEqual(stream.getHits(now), naiveHits(history, now), `random get ${step}`);
    }
    if (step % 97 === 0) {
      assert.strictEqual(stream.getHits(now), naiveHits(history, now), `periodic get ${step}`);
    }
  }
  assert.strictEqual(stream.getHits(now), naiveHits(history, now));
  console.log("PASS");
}
