/** Task 386 - Implement elliptic curve point arithmetic (outline). (Tier: Intermediate, CTCI Beyond — Security & Privacy) */
"use strict";

function extendedGcd(a, b) {
  let oldR = a, r = b;
  let oldS = 1n, s = 0n;
  while (r !== 0n) {
    const q = oldR / r;
    [oldR, r] = [r, oldR - q * r];
    [oldS, s] = [s, oldS - q * s];
  }
  return [oldR, oldS];
}

function modInverse(a, modulus) {
  const [g, x] = extendedGcd(((a % modulus) + modulus) % modulus, modulus);
  if (g !== 1n) throw new Error("modular inverse does not exist");
  return ((x % modulus) + modulus) % modulus;
}

function modPower(base, exponent, modulus) {
  let result = 1n % modulus;
  let b = base % modulus;
  let e = exponent;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % modulus;
    b = (b * b) % modulus;
    e >>= 1n;
  }
  return result;
}

function isOnCurve(point, a, b, p) {
  if (point === null) return true;                  // O is on every curve by definition
  const [x, y] = point;
  return (((y * y - (x * x * x % p + a * x + b)) % p) + p) % p === 0n;
}

function samePoint(one, two) {
  if (one === null || two === null) return one === two;
  return one[0] === two[0] && one[1] === two[1];
}

/** Group addition on y^2 = x^3 + a*x + b over F_p; null is the identity O. */
function add(point1, point2, a, p) {
  if (point1 === null) return point2;
  if (point2 === null) return point1;
  const [x1, y1] = point1;
  const [x2, y2] = point2;
  if (x1 === x2 && (y1 + y2) % p === 0n) return null;   // P + (-P) = O (also y = 0 doubling)
  let slope;
  if (x1 === x2) {
    slope = ((3n * x1 * x1 + a) % p * modInverse(2n * y1, p)) % p;
  } else {
    slope = ((((y2 - y1) % p) + p) % p * modInverse(x2 - x1, p)) % p;
  }
  const x3 = (((slope * slope - x1 - x2) % p) + p) % p;
  const y3 = ((((slope * (x1 - x3) - y1) % p) + p) % p);
  return [x3, y3];
}

/** Double-and-add; k >= 0, k = 0 gives null (O). */
function scalarMultiply(k, point, a, p) {
  let result = null;
  let addend = point;
  let e = k;
  while (e > 0n) {
    if (e & 1n) result = add(result, addend, a, p);
    addend = add(addend, addend, a, p);
    e >>= 1n;
  }
  return result;
}

/** Repeated addition; reference oracle for small k. */
function scalarMultiplyNaive(k, point, a, p) {
  let result = null;
  for (let i = 0n; i < k; i++) result = add(result, point, a, p);
  return result;
}

/** All affine points in deterministic order. */
function enumeratePoints(a, b, p) {
  const points = [];
  for (let x = 0n; x < p; x++) {
    for (let y = 0n; y < p; y++) {
      if ((((y * y - (x * x * x % p + a * x + b)) % p) + p) % p === 0n) points.push([x, y]);
    }
  }
  return points;
}

/** #E(F_p) = 1 + sum_x (1 + chi(f(x))) with chi the Legendre symbol. */
function curveOrder(a, b, p) {
  let count = 1n;
  for (let x = 0n; x < p; x++) {
    const rhs = (x * x * x % p + a * x + b) % p;
    if (rhs === 0n) count += 1n;
    else if (modPower(rhs, (p - 1n) / 2n, p) === 1n) count += 2n;
  }
  return count;
}

module.exports = { modInverse, modPower, isOnCurve, samePoint, add, scalarMultiply,
  scalarMultiplyNaive, enumeratePoints, curveOrder };

if (require.main === module) {
  const assert = require("node:assert");

  const a = 2n, b = 3n, p = 97n;
  const point = [3n, 6n];
  assert.ok(isOnCurve(point, a, b, p));
  assert.strictEqual((6n * 6n - (3n * 3n * 3n + 2n * 3n + 3n)) % 97n, 0n);

  const points = enumeratePoints(a, b, p);
  assert.strictEqual(points.length, 99);
  for (const pt of points) assert.ok(isOnCurve(pt, a, b, p));
  assert.strictEqual(points.filter((pt) => pt[1] === 0n).length, 3);

  // independent group-order computation via Legendre symbols, and the Hasse bound
  const order = curveOrder(a, b, p);
  assert.strictEqual(order, 100n);
  assert.strictEqual(order, BigInt(points.length + 1));
  const delta = order - (p + 1n);
  assert.ok(delta * delta <= 4n * p, "Hasse bound");

  // identity handling
  assert.ok(samePoint(add(null, point, a, p), point));
  assert.ok(samePoint(add(point, null, a, p), point));
  assert.strictEqual(add(null, null, a, p), null);

  // hand-verified group law on P = (3, 6): 2P, 3P, 4P, 5P = O, 6P = P
  const p2 = add(point, point, a, p);
  assert.deepStrictEqual(p2, [80n, 10n]);
  const p3 = add(p2, point, a, p);
  assert.deepStrictEqual(p3, [80n, 87n]);
  const p4 = add(p2, p2, a, p);
  assert.deepStrictEqual(p4, [3n, 91n]);
  assert.strictEqual(add(point, [3n, 91n], a, p), null);   // -P = (3, -6) = (3, 91)
  assert.strictEqual(scalarMultiply(5n, point, a, p), null);
  assert.deepStrictEqual(scalarMultiply(6n, point, a, p), point);
  for (const derived of [p2, p3, p4]) assert.ok(isOnCurve(derived, a, b, p));

  // negation and an order-2 point (doubling must take the y1 + y2 = 0 branch)
  assert.strictEqual(add([0n, 10n], [0n, 87n], a, p), null);
  assert.strictEqual(add([30n, 0n], [30n, 0n], a, p), null);

  // commutativity and associativity spot checks
  for (let i = 0; i < 12; i += 3) {
    const q = points[i];
    const r = points[(i + 4) % points.length];
    assert.ok(samePoint(add(point, q, a, p), add(q, point, a, p)), `commutativity ${i}`);
    assert.ok(samePoint(add(add(point, q, a, p), r, a, p), add(point, add(q, r, a, p), a, p)),
      `associativity ${i}`);
  }

  // double-and-add equals naive repeated addition, and results stay on the curve
  for (let k = 0n; k <= 30n; k++) {
    assert.ok(samePoint(scalarMultiply(k, point, a, p), scalarMultiplyNaive(k, point, a, p)),
      `double-and-add P ${k}`);
    assert.ok(samePoint(scalarMultiply(k, [80n, 10n], a, p), scalarMultiplyNaive(k, [80n, 10n], a, p)),
      `double-and-add 2P ${k}`);
  }
  for (let i = 0; i < 6; i++) {
    const pt = points[i];
    for (let k = 0n; k <= 12n; k++) {
      const got = scalarMultiply(k, pt, a, p);
      assert.ok(samePoint(got, scalarMultiplyNaive(k, pt, a, p)), `pt ${i} k ${k}`);
      assert.ok(got === null || isOnCurve(got, a, b, p), `on curve ${i} ${k}`);
    }
  }

  // the group order is 100, so 100 * P = O for every point
  for (const pt of points) {
    assert.strictEqual(scalarMultiply(100n, pt, a, p), null, `100P = O ${pt}`);
  }

  // LCG-driven keys: double-and-add and naive must agree
  let seed = 3131;
  for (let trial = 0; trial < 40; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const k = BigInt(seed % 50);
    seed = (seed * 75 + 74) % 65537;
    const base = points[seed % points.length];
    const got = scalarMultiply(k, base, a, p);
    assert.ok(samePoint(got, scalarMultiplyNaive(k, base, a, p)), `LCG ${trial}`);
    assert.ok(got === null || isOnCurve(got, a, b, p), `LCG on curve ${trial}`);
  }

  console.log("PASS");
}
