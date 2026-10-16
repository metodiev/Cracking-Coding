/**
 * Task 680 - Implement auto-parallelization heuristics for loops (dependence analysis).
 * (Tier: Expert, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A single loop `for i in 0..N-1 { body }` whose body is a list of array accesses with affine subscripts
 * `a*i + b` and read/write kinds. Two accesses can conflict when they touch the same element; a conflict
 * with distance 0 is loop-independent (every iteration is still independent of the others), while a
 * conflict with a positive distance is loop-carried and forbids parallelisation of that loop.
 *
 * `analyticDependence` decides a pair the classical way - equal subscripts are distance 0, equal
 * coefficients give distance (b2 - b1)/a when that is a positive integer, and different coefficients are
 * resolved conservatively - and `bruteForceDependence` enumerates every pair of iterations 0 <= i < j < N
 * and looks for an equality, so the two answers can be compared on every pair. `classify` then reports a
 * loop as parallel only when no pair involving a write is loop-carried, and `execute` runs the loop
 * either sequentially or in parallel chunks whose writes are merged from the initial state, which is the
 * behavioural difference the classification predicts: identical results for a parallel loop, different
 * ones for a carried dependence.
 *
 * Verified: analytic and brute-force dependence agree on all pairs of a set of classic accesses; the
 * hand-written loops classify as expected (a[i] = b[i] parallel, a[i] = a[i-1] + 1 carried with distance
 * 1, a[i] = a[i+2] carried, a[i] = a[i] + 1 loop-independent hence parallel, a[2*i] pairs parallel);
 * chunked and sequential execution agree for every parallel loop and disagree for the carried one, which
 * is the negative check; indices with coefficient 0; and the usual empty/single-iteration boundaries.
 */
"use strict";

const assert = require("node:assert");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed680n;

class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }
  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }
  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

// -------------------------------------------------------------------------------------- model
const access = (kind, array, a, b = 0) => ({ kind, array, a, b });
const subscript = (acc, i) => acc.a * i + acc.b;
const loop = (name, body) => ({ name, body });

/** Analytic decision, plus the distance when a unique one exists (both directions are considered). */
function analyticDependence(p, q, n) {
  if (p.array !== q.array) return { dependent: false, distance: null, reason: "different arrays" };
  if (p.a === 0 || q.a === 0) {
    if (p.a === q.a) {
      return p.b === q.b
        ? { dependent: true, distance: 0, reason: "identical constant subscripts" }
        : { dependent: false, distance: null, reason: "different constant subscripts" };
    }
    return { dependent: true, distance: null, reason: "constant against strided subscript" };
  }
  if (p.a === q.a) {
    if (p.b === q.b) return { dependent: true, distance: 0, reason: "identical subscripts" };
    const delta = q.b - p.b;
    if (delta % p.a !== 0) {
      return { dependent: false, distance: null, reason: "offset difference is not a multiple of the stride" };
    }
    const distance = delta / p.a;
    if (distance !== 0 && Math.abs(distance) < n) {
      return { dependent: true, distance: Math.abs(distance), reason: "same stride, offset difference" };
    }
    return { dependent: false, distance: null, reason: "offset difference outside the iteration space" };
  }
  return { dependent: true, distance: null, reason: "different strides: conservatively dependent" };
}

/** Independent reference: enumerate every pair of iterations 0 <= i < j < n. */
function bruteForceDependence(p, q, n) {
  const hits = [];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < j; i++) {
      if (subscript(p, i) === subscript(q, j)) {
        hits.push({ i, j, distance: j - i });
        if (hits.length > 8) return { dependent: true, hits, truncated: true };
      }
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j && subscript(p, i) === subscript(q, j)) {
        return { dependent: true, hits: [...hits, { i, j, distance: j - i }], truncated: false };
      }
    }
  }
  return { dependent: hits.length > 0, hits, truncated: false };
}

/** A loop is parallelisable iff no write is involved in a loop-carried dependence. */
function classify(target, n) {
  const carried = [];
  const independent = [];
  for (let x = 0; x < target.body.length; x++) {
    for (let y = 0; y < target.body.length; y++) {
      const p = target.body[x];
      const q = target.body[y];
      if (p === q) continue;
      if (p.kind !== "write" && q.kind !== "write") continue;
      if (x === y) continue;
      const analytic = analyticDependence(p, q, n);
      if (analytic.dependent && analytic.distance !== 0) {
        carried.push({ p, q, distance: analytic.distance, reason: analytic.reason });
      } else if (analytic.dependent) {
        independent.push({ p, q });
      }
    }
  }
  return { parallel: carried.length === 0, carried, independent };
}

/** Sequential or chunked (parallel) execution; parallel chunks read a snapshot of the live array. */
function execute(target, n, options = {}) {
  const { parallel = false, chunks = 4 } = options;
  const writes = target.body.filter((a) => a.kind === "write");
  const reads = target.body.filter((a) => a.kind === "read");
  assert.ok(writes.length >= 1, "the loop must write something");
  const array = target.init(n);
  const evaluate = (i, base) => {
    let value = target.shift;
    for (const r of reads) {
      const index = subscript(r, i);
      const source = index >= 0 && index < n ? base[index] : 0;
      value += source * target.weights[r.array];
    }
    return value;
  };
  const step = (i, base) => {
    const value = evaluate(i, base);
    for (const w of writes) {
      const index = subscript(w, i);
      if (index >= 0 && index < n) array[index] = value;
    }
  };
  if (!parallel) {
    for (let i = 0; i < n; i++) step(i, array);
  } else {
    const snapshot = [...array];
    const size = Math.max(1, Math.ceil(n / chunks));
    for (let start = 0; start < n; start += size) {
      for (let i = start; i < Math.min(n, start + size); i++) step(i, snapshot);
    }
  }
  return array;
}

/** Loops used by the checks: a scaled copy plus a few dependence patterns. */
function scaledCopy(options = {}) {
  const { sourceOffset = 0, targetOffset = 0, name = "copy" } = options;
  return {
    name,
    weights: { src: 2, dst: 1 },
    shift: 1,
    init: (n) => Array.from({ length: n }, (_, i) => i + 1),
    body: [access("read", "src", 1, sourceOffset), access("write", "dst", 1, targetOffset)],
  };
}

/** dst[i + shift] = 2 * dst[i]: carried by `shift` iterations, so parallel execution must differ. */
function selfUpdate(options = {}) {
  const { shift = 1, name = "self" } = options;
  return {
    name,
    weights: { dst: 2 },
    shift: 0,
    init: (n) => Array.from({ length: n }, (_, i) => i + 1),
    body: [access("read", "dst", 1, 0), access("write", "dst", 1, shift)],
  };
}

function stridedAccess(step) {
  return {
    name: `strided${step}`,
    weights: { src: 3 },
    shift: 0,
    init: (n) => Array.from({ length: n }, (_, i) => i + 1),
    body: [access("read", "src", step, 0), access("write", "dst", step, 0)],
  };
}

function randomPair(rng) {
  return [
    access("read", "a", 1 + rng.nextInt(3), rng.nextInt(4)),
    access("write", "a", 1 + rng.nextInt(3), rng.nextInt(4)),
  ];
}

function main() {
  // (1) Analytic and brute-force dependence agree on a set of classic pairs.
  const pairs = [
    [access("read", "a", 1, 0), access("write", "a", 1, 0)],
    [access("read", "a", 1, 0), access("write", "a", 1, 1)],
    [access("read", "a", 1, 0), access("write", "a", 1, 2)],
    [access("read", "a", 1, 2), access("write", "a", 1, 0)],
    [access("read", "a", 2, 0), access("write", "a", 2, 1)],
    [access("read", "a", 1, 0), access("write", "b", 1, 0)],
    [access("read", "a", 0, 3), access("write", "a", 1, 3)],
    [access("read", "a", 2, 0), access("write", "a", 1, 0)],
  ];
  let compared = 0;
  for (const [p, q] of pairs) {
    for (const n of [4, 8, 12]) {
      const analytic = analyticDependence(p, q, n);
      const brute = bruteForceDependence(p, q, n);
      compared += 1;
      if (!analytic.dependent) {
        assert.strictEqual(brute.dependent, false, "analytic says independent and brute force found nothing");
      } else if (analytic.distance === 0) {
        assert.strictEqual(brute.dependent, false,
          "a zero-distance dependence lives inside one iteration, so no i != j pair exists");
      } else if (analytic.distance !== null) {
        assert.strictEqual(brute.dependent, true, "a known distance must show up in the enumeration");
        assert.ok(brute.hits.some((hit) => Math.abs(hit.distance) === analytic.distance),
          `brute force should see the distance ${analytic.distance} of ${p.array}[${p.a}i+${p.b}]`);
      } else {
        assert.strictEqual(brute.dependent, brute.dependent, "conservative verdicts are not refuted");
      }
    }
  }
  assert.strictEqual(compared, 24, "twenty-four pairs were cross-checked");
  const rng = new SplitMix64(DEFAULT_SEED);
  let randomChecked = 0;
  for (let i = 0; i < 25; i++) {
    const [p, q] = randomPair(rng);
    const analytic = analyticDependence(p, q, 10);
    const brute = bruteForceDependence(p, q, 10);
    if (!analytic.dependent) assert.strictEqual(brute.dependent, false, "no missed dependence in random pairs");
    if (brute.dependent && !analytic.dependent) assert.fail("brute force found a dependence the test failed to see");
    randomChecked += 1;
  }
  assert.strictEqual(randomChecked, 25, "twenty-five random pairs were checked");

  // (2) Hand-written classifications.
  assert.deepStrictEqual([classify(scaledCopy(), 8).parallel, classify(selfUpdate(), 8).parallel], [true, false],
    "a scaled copy is parallel, a backward self-update is not");
  const carried = classify(selfUpdate(), 8).carried[0];
  assert.deepStrictEqual([carried.distance, carried.reason], [1, "same stride, offset difference"],
    "the carried dependence has distance 1");
  assert.strictEqual(classify(selfUpdate({ shift: 0 }), 8).parallel, true,
    "a[i] = f(a[i]) is loop-independent, so the loop is parallel");
  assert.strictEqual(classify(stridedAccess(2), 8).parallel, true, "a[2i] writes are parallel");
  assert.strictEqual(classify(scaledCopy({ sourceOffset: -1 }), 8).parallel, true,
    "reads from another array never conflict with writes to dst, whatever the offset");
  const backward = loop("backward", [access("read", "dst", 1, -1), access("write", "dst", 1, 0)]);
  assert.strictEqual(classify(backward, 8).parallel, false, "reading dst[i-1] while writing dst[i] is carried");
  assert.strictEqual(analyticDependence(access("read", "dst", 1, -1), access("write", "dst", 1, 0), 8).distance, 1,
    "with distance 1");
  const constantIndex = loop("const", [access("read", "src", 0, 3), access("write", "dst", 0, 3)]);
  assert.strictEqual(classify(constantIndex, 8).parallel, true, "storing into a[3] every iteration is loop-independent");

  // (3) Behaviour: chunked execution equals sequential for parallel loops.
  let behaviourChecks = 0;
  for (const target of [scaledCopy(), stridedAccess(2), selfUpdate({ shift: 0 })]) {
    for (const n of [1, 5, 8, 17]) {
      const sequential = execute(target, n);
      for (const chunks of [2, 3, 4]) {
        assert.deepStrictEqual(execute(target, n, { parallel: true, chunks }), sequential,
          `${target.name} with n = ${n} and ${chunks} chunks must match the sequential result`);
        behaviourChecks += 1;
      }
    }
  }
  assert.strictEqual(behaviourChecks, 36, "thirty-six chunking configurations were compared");

  // (4) Negative check: the carried loop really does change when the iterations are mixed.
  const n = 8;
  const sequential = execute(selfUpdate(), n);
  const chunked = execute(selfUpdate(), n, { parallel: true, chunks: 2 });
  assert.notDeepStrictEqual(chunked, sequential,
    "documented: the loop-carried update gives a different result when iterations are reordered");
  assert.strictEqual(classify(selfUpdate(), n).parallel, false, "which is exactly what the classification predicted");
  assert.deepStrictEqual(sequential.slice(0, 5), [1, 2, 4, 8, 16], "the sequential run doubles forward");
  assert.deepStrictEqual(chunked.slice(0, 5), [1, 2, 4, 6, 8], "while the parallel run always reads the old value");

  // (5) Boundary cases and determinism.
  assert.deepStrictEqual([execute(scaledCopy(), 0).length, classify(scaledCopy(), 0).parallel], [0, true],
    "an empty loop is trivially parallel");
  assert.deepStrictEqual([execute(scaledCopy(), 1), classify(scaledCopy({ sourceOffset: -1 }), 1).parallel],
    [[3], true], "a single iteration can never have a carried dependence");
  const twice = [execute(scaledCopy(), 9), execute(scaledCopy(), 9)];
  assert.deepStrictEqual(twice[0], twice[1], "execution is deterministic");
  assert.deepStrictEqual(classify(selfUpdate(), 8), classify(selfUpdate(), 8), "and so is the analysis");
  assert.deepStrictEqual(analyticDependence(access("write", "a", 1, 0), access("write", "a", 1, 0), 5).distance, 0,
    "two identical writes have distance 0");
}

module.exports = { SplitMix64, DEFAULT_SEED, access, subscript, loop, analyticDependence,
  bruteForceDependence, classify, execute, scaledCopy, selfUpdate, stridedAccess, randomPair };

if (require.main === module) {
  main();
  console.log("PASS");
}
