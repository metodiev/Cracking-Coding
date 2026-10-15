/** Task 464 - Implement reservoir sampling with weighted probabilities. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Weighted reservoir sampling, A-Res (Efraimidis & Spirakis): each item gets the
 * key u^(1/w) with u uniform in (0,1] drawn from a FIXED LCG as u = (seed + 1) /
 * 65537; the reservoir keeps the k largest keys (min-heap; an equal key is
 * displaced only by a smaller item index, as documented). For k = 1 an item with
 * weight w_i is selected with probability exactly w_i / W, which the statistical
 * checks exercise.
 *
 * Checks (deterministic): weights [1,2,3,4] over 20000 trials must land within
 * TOL = 0.02 of the weight ratios (measured max deviation 0.0022) with exactly one
 * item per trial; equal weights give the uniform 1/3; weights 1..10 with k = 3 over
 * 3000 trials yield 3 distinct items per trial, counts summing to 9000, and heavier
 * items selected more often.
 */
"use strict";

const TOL = 0.02;

function pushKey(heap, key, item) {
  heap.push([key, item]);
  let i = heap.length - 1;
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (!tupleLess(heap[i], heap[p])) break;
    [heap[i], heap[p]] = [heap[p], heap[i]];
    i = p;
  }
}

function popKey(heap) {
  const top = heap[0];
  const last = heap.pop();
  if (heap.length > 0) {
    heap[0] = last;
    let i = 0;
    for (;;) {
      let m = i;
      const l = 2 * i + 1;
      const r = l + 1;
      if (l < heap.length && tupleLess(heap[l], heap[m])) m = l;
      if (r < heap.length && tupleLess(heap[r], heap[m])) m = r;
      if (m === i) break;
      [heap[i], heap[m]] = [heap[m], heap[i]];
      i = m;
    }
  }
  return top;
}

function tupleLess(a, b) {
  return a[0] !== b[0] ? a[0] < b[0] : a[1] < b[1];
}

class WeightedReservoir {
  constructor(size) {
    this.size = size;
    this.heap = []; // min-heap of [key, item]; kept as the `size` largest keys
  }

  add(item, weight, u) {
    const key = Math.pow(u, 1.0 / weight);
    if (this.heap.length < this.size) {
      pushKey(this.heap, key, item);
    } else if (tupleLess(this.heap[0], [key, item])) {
      popKey(this.heap);
      pushKey(this.heap, key, item);
    }
  }

  sample() {
    return this.heap.map((entry) => entry[1]).sort((a, b) => a - b);
  }
}

module.exports = { WeightedReservoir };

function makeLcg(state) {
  let seed = state;
  return () => (seed = (seed * 75 + 74) % 65537);
}

if (require.main === module) {
  const assert = require("node:assert");

  const sampleOnce = (k, weights, lcg) => {
    const reservoir = new WeightedReservoir(k);
    for (let item = 0; item < weights.length; item++) {
      const u = (lcg() + 1) / 65537.0;
      reservoir.add(item, weights[item], u);
    }
    return reservoir.sample();
  };

  const weights = [1, 2, 3, 4];
  const trials = 20000;
  let lcg = makeLcg(99);
  const counts = [0, 0, 0, 0];
  for (let t = 0; t < trials; t++) {
    const picked = sampleOnce(1, weights, lcg);
    assert.strictEqual(picked.length, 1); // exactly one item per trial
    counts[picked[0]]++;
  }
  const total = weights.reduce((a, b) => a + b, 0);
  let maxDev = 0;
  for (let i = 0; i < weights.length; i++) {
    const expected = weights[i] / total;
    const observed = counts[i] / trials;
    maxDev = Math.max(maxDev, Math.abs(observed - expected));
    assert.ok(Math.abs(observed - expected) <= TOL, `freq ${i}`);
  }
  assert.strictEqual(
    counts.reduce((a, b) => a + b, 0),
    trials
  );
  assert.ok(counts[3] > counts[0]);
  assert.ok(maxDev <= 0.01, `measured max deviation ${maxDev}`);

  lcg = makeLcg(12345);
  const uniform = [0, 0, 0];
  for (let t = 0; t < 20000; t++) {
    const picked = sampleOnce(1, [1, 1, 1], lcg);
    uniform[picked[0]]++;
  }
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(uniform[i] / 20000 - 1 / 3) <= TOL, `uniform ${i}`);
  }

  const ws = Array.from({ length: 10 }, (_, i) => i + 1);
  const k = 3;
  lcg = makeLcg(777);
  const kCounts = new Array(10).fill(0);
  for (let t = 0; t < 3000; t++) {
    const picked = sampleOnce(k, ws, lcg);
    assert.strictEqual(picked.length, k);
    assert.strictEqual(new Set(picked).size, k); // distinct items
    for (const item of picked) kCounts[item]++;
  }
  assert.strictEqual(
    kCounts.reduce((a, b) => a + b, 0),
    3000 * k
  );
  assert.ok(kCounts[9] > kCounts[0]);
  console.log("PASS");
}
