/** Task 327 - Car pooling capacity check (difference array / sweep). (Tier: Intermediate, CTCI Ch 1) */
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

/** Difference-array sweep; true when the load never exceeds the capacity. */
function carPooling(trips, capacity) {
  const delta = new Map();
  for (const [passengers, start, end] of trips) {
    delta.set(start, (delta.get(start) ?? 0) + passengers);
    delta.set(end, (delta.get(end) ?? 0) - passengers);
  }
  const stops = [...delta.keys()].sort((u, v) => u - v);
  let load = 0;
  for (const stop of stops) {
    load += delta.get(stop);
    if (load > capacity) return false;
  }
  return true;
}

/** Independent oracle for small coordinates: materialise the load at every stop. */
function carPoolingBruteforce(trips, capacity) {
  if (trips.length === 0) return capacity >= 0;
  const lo = Math.min(...trips.map((trip) => trip[1]));
  const hi = Math.max(...trips.map((trip) => trip[2]));
  const loads = new Array(hi - lo).fill(0);
  for (const [passengers, start, end] of trips) {
    for (let stop = start; stop < end; stop++) loads[stop - lo] += passengers;
  }
  return loads.reduce((peak, load) => Math.max(peak, load), 0) <= capacity;
}

function lcgTrips(count, lcg) {
  const trips = [];
  for (let i = 0; i < count; i++) {
    const passengers = 1 + lcg.next(6);
    const start = lcg.next(25);
    trips.push([passengers, start, start + 1 + lcg.next(8)]);
  }
  return trips;
}

module.exports = { carPooling, carPoolingBruteforce, lcgTrips, Lcg };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(carPooling([[2, 1, 5], [3, 3, 7]], 4), false);
  assert.strictEqual(carPooling([[2, 1, 5], [3, 3, 7]], 5), true);
  assert.strictEqual(carPooling([[2, 1, 5], [3, 5, 7]], 3), true);
  assert.strictEqual(carPooling([[2, 1, 5], [3, 5, 7]], 2), false);
  assert.strictEqual(carPooling([], 0), true);
  assert.strictEqual(carPooling([[1, 0, 1]], 0), false);

  assert.strictEqual(carPoolingBruteforce([[2, 1, 5], [3, 3, 7]], 4), false);
  assert.strictEqual(carPoolingBruteforce([[2, 1, 5], [3, 5, 7]], 3), true);

  const lcg = new Lcg(1618);
  for (let trial = 0; trial < 300; trial++) {
    const count = 1 + lcg.next(8);
    const trips = lcgTrips(count, lcg);
    const capacity = lcg.next(21);
    assert.strictEqual(
      carPooling(trips, capacity),
      carPoolingBruteforce(trips, capacity),
      `trial ${trial}`
    );
  }
  console.log("PASS");
}
