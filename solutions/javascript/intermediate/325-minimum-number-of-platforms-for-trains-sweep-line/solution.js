/** Task 325 - Minimum number of platforms for trains (sweep-line). (Tier: Intermediate, CTCI Beyond — Software Engineering & Tooling) */
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

/** Two-pointer sweep; a train leaving at t frees its platform before any arrival at t. */
function minPlatforms(arrivals, departures) {
  const arr = arrivals.slice().sort((u, v) => u - v);
  const dep = departures.slice().sort((u, v) => u - v);
  let i = 0;
  let j = 0;
  let needed = 0;
  let best = 0;
  while (i < arr.length) {
    if (arr[i] < dep[j]) {
      needed++;
      if (needed > best) best = needed;
      i++;
    } else {
      needed--;
      j++;
    }
  }
  return best;
}

/** Independent oracle: count active trains at every minute of the schedule. */
function minPlatformsPerMinute(arrivals, departures) {
  if (arrivals.length === 0) return 0;
  const lo = Math.min(...arrivals);
  const hi = Math.max(...departures);
  let best = 0;
  for (let minute = lo; minute <= hi; minute++) {
    let active = 0;
    for (let k = 0; k < arrivals.length; k++) {
      if (arrivals[k] <= minute && minute < departures[k]) active++;
    }
    if (active > best) best = active;
  }
  return best;
}

function lcgSchedules(count, lcg) {
  const arrivals = [];
  const departures = [];
  for (let i = 0; i < count; i++) {
    const arrival = lcg.next(60);
    arrivals.push(arrival);
    departures.push(arrival + 1 + lcg.next(20));
  }
  return { arrivals, departures };
}

module.exports = { minPlatforms, minPlatformsPerMinute, lcgSchedules, Lcg };

if (require.main === module) {
  const assert = require("node:assert");
  const arrivals = [900, 940, 950, 1100, 1500, 1800];
  const departures = [910, 1200, 1120, 1130, 1900, 2000];
  assert.strictEqual(minPlatforms(arrivals, departures), 3);
  assert.strictEqual(minPlatformsPerMinute(arrivals, departures), 3);

  assert.strictEqual(minPlatforms([1, 2], [2, 3]), 1); // release before acquire at minute 2
  assert.strictEqual(minPlatformsPerMinute([1, 2], [2, 3]), 1);
  assert.strictEqual(minPlatforms([1, 1, 1], [2, 2, 2]), 3);
  assert.strictEqual(minPlatforms([5], [9]), 1);
  assert.strictEqual(minPlatforms([], []), 0);

  const lcg = new Lcg(31415);
  for (let trial = 0; trial < 300; trial++) {
    const count = 1 + lcg.next(12);
    const { arrivals: arr, departures: dep } = lcgSchedules(count, lcg);
    assert.strictEqual(
      minPlatforms(arr, dep),
      minPlatformsPerMinute(arr, dep),
      `trial ${trial}`
    );
  }
  console.log("PASS");
}
