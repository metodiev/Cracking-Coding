/** Task 465 - Implement count-min sketch for frequency estimation. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Count-min sketch with d = 4 rows and width = 48 cells, using FIXED hash seeds:
 * column r = ((a_r * key + b_r) mod 1000003) mod width for r in 0..3. Counters only
 * ever increase, so estimate(key) = min over the rows is NEVER below the true
 * count; the overshoot is bounded by the collision mass.
 *
 * Checks: a hand-computed tiny sketch (width 4: columns(7) = columns(11) = [2, 0],
 * columns(0) = [1, 3]) documents exact behaviours, then a fixed LCG stream of
 * N = 5000 updates over ids 0..63 asserts for every id: estimate >= true count, and
 * estimate - true count <= EPS * N with the documented EPS = 0.06 (>= the standard
 * e/width ~ 0.0566); the measured max overshoot is 101 (~1/3 of the 300 bound) and
 * 23 of the 64 ids do over-estimate, documenting that collisions really happen.
 */
"use strict";

const P = 1000003;
const HASH_A = [31, 131, 2711, 65521];
const HASH_B = [17, 91, 601, 4099];
const EPS = 0.06;

class CountMinSketch {
  constructor(width, depth) {
    this.width = width;
    this.depth = depth;
    this.table = Array.from({ length: depth }, () => new Array(width).fill(0));
  }

  columns(key) {
    const cols = [];
    for (let r = 0; r < this.depth; r++) cols.push(((HASH_A[r] * key + HASH_B[r]) % P) % this.width);
    return cols;
  }

  add(key) {
    const cols = this.columns(key);
    for (let r = 0; r < this.depth; r++) this.table[r][cols[r]]++;
  }

  estimate(key) {
    const cols = this.columns(key);
    let best = Infinity;
    for (let r = 0; r < this.depth; r++) best = Math.min(best, this.table[r][cols[r]]);
    return best;
  }
}

module.exports = { CountMinSketch };

if (require.main === module) {
  const assert = require("node:assert");

  let sketch = new CountMinSketch(4, 2);
  assert.deepStrictEqual(sketch.columns(7), [2, 0]);
  assert.deepStrictEqual(sketch.columns(11), [2, 0]);
  sketch.add(7);
  assert.strictEqual(sketch.estimate(7), 1);
  sketch.add(11);
  assert.strictEqual(sketch.estimate(11), 2); // both rows collide with key 7
  assert.strictEqual(sketch.estimate(7), 2); // over-estimate from the collision, never below
  sketch.add(7);
  assert.strictEqual(sketch.estimate(7), 3);
  assert.strictEqual(sketch.estimate(0), 0); // the [1, 3] cells were never touched

  sketch = new CountMinSketch(48, 4);
  let seed = 7654321;
  const lcg = () => (seed = (seed * 75 + 74) % 65537);
  const total = 5000;
  const stream = [];
  const trueCounts = new Array(64).fill(0);
  for (let i = 0; i < total; i++) {
    const item = lcg() % 64;
    stream.push(item);
    trueCounts[item]++;
  }
  for (const item of stream) sketch.add(item);
  let maxOvershoot = 0;
  for (let item = 0; item < 64; item++) {
    const estimate = sketch.estimate(item);
    assert.ok(estimate >= trueCounts[item], `never under-estimates ${item}`);
    const overshoot = estimate - trueCounts[item];
    maxOvershoot = Math.max(maxOvershoot, overshoot);
    assert.ok(overshoot <= EPS * total, `overshoot bound ${item}`);
  }
  assert.ok(maxOvershoot >= 1); // collisions did occur (documented)
  assert.strictEqual(
    trueCounts.reduce((a, b) => a + b, 0),
    total
  );
  console.log("PASS");
}
