/** Task 755 - Implement timing attack resistant comparison function and test. (Tier: Expert, CTCI Beyond — Security & Privacy)
 *
 * Two byte-array comparisons plus a timing model that makes the security claim falsifiable.
 * The model is the standard constant-time *operation count*: a `Counter` counts every byte
 * read (`probe`) and every data-dependent split (`branch`), and the arrays are wrapped in
 * `Memory` so a comparison cannot read a byte without the model noticing. `naiveEqual`
 * returns at the first differing byte, so its probe count is 2 * (matched prefix + 1) — the
 * classic oracle for a timing attack. `constantTimeEqual` accumulates the whole difference
 * with `diff |= x.read(i) ^ y.read(i)` over `min(len)` bytes, folds the length difference into
 * the same accumulator and returns a boolean, so its probe count is exactly 2 * min(len) and
 * its branch count is zero whatever the contents are. `constantTimeCompare` decides on the
 * *first* differing byte with `pick = nonzero & (decided ^ 1)` bookkeeping — still branch-free,
 * still a fixed number of reads.
 *
 * What is simplified: real timing is replaced by this deterministic operation-count model (no
 * caches, no noise, no speculation) and `Memory` exposes byte reads synchronously; the arrays'
 * lengths remain public in this model, which is why the claim is only "the count depends on the
 * lengths and on nothing else". What is verified: (a) all three comparisons agree with an
 * independent oracle (`Buffer.from(...).equals` and `Buffer.compare`) on every pair of a small
 * universe; (b) the constant-time probe count is 2 * min(len) and its branch count is 0 on every
 * input, while the naive counts grow with the matched prefix (a negative control proving the
 * model can see a leak); (c) the constant-time cost is unchanged across 32 chosen-prefix attack
 * vectors and across random contents; (d) a prefix-timing attack recovers the secret's first
 * n-1 bytes from `naiveEqual` (one discriminating cost per position; the final byte ties because
 * a full match and a last-byte mismatch read the same number of bytes) and recovers nothing from
 * `constantTimeEqual` (cost spread exactly zero at every position); (e) `ctSelect` is
 * branch-free and correct; (f) length boundaries and input validation.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

/** Probe/branch counter: the "clock" of the timing model. */
class Counter {
  constructor() { this.probes = 0; this.branches = 0; }
  probe() { this.probes++; }
  branch() { this.branches++; }
  snapshot() { return { probes: this.probes, branches: this.branches }; }
}

/** Byte array whose reads are visible to the counter. */
class Memory {
  constructor(bytes, counter) {
    if (!(bytes instanceof Uint8Array)) throw new TypeError("bytes must be a Uint8Array");
    if (!(counter instanceof Counter)) throw new TypeError("counter must be a Counter");
    this.bytes = bytes;
    this.counter = counter;
  }
  get length() { return this.bytes.length; }
  read(index) { this.counter.probe(); return this.bytes[index]; }
}

/** Early-exit comparison: the probe count leaks how long the matching prefix is. */
function naiveEqual(left, right, counter) {
  if (left.length !== right.length) { counter.branch(); return false; }
  for (let i = 0; i < left.length; i++) {
    const a = left.read(i);
    const b = right.read(i);
    if (a !== b) { counter.branch(); return false; }
  }
  return true;
}

/**
 * Constant-time equality: every byte pair of the shared prefix is read exactly once on every
 * input, and the verdict is the value of a single accumulator, so the cost cannot depend on
 * where the first difference is.
 */
function constantTimeEqual(left, right, counter) {
  const shared = Math.min(left.length, right.length);
  let diff = left.length ^ right.length;
  for (let i = 0; i < shared; i++) diff |= left.read(i) ^ right.read(i);
  return diff === 0;
}

/**
 * Sign-only constant-time comparison (-1/0/1). `decided` records whether an earlier byte
 * already settled the order, so only the first difference counts, exactly like a
 * lexicographic compare — without ever branching on the data.
 */
function constantTimeCompare(left, right, counter) {
  const shared = Math.min(left.length, right.length);
  let less = 0;
  let greater = 0;
  let decided = 0;
  for (let i = 0; i < shared; i++) {
    const difference = left.read(i) - right.read(i);
    const nonzero = ((difference | -difference) >> 31) & 1;
    const negative = (difference >> 31) & 1;
    const pick = nonzero & (decided ^ 1);
    less |= pick & negative;
    greater |= pick & (negative ^ 1);
    decided |= pick;
  }
  const lengthDifference = left.length - right.length;
  const lengthNegative = (lengthDifference >> 31) & 1;
  const pick = (((lengthDifference | -lengthDifference) >> 31) & 1) & (decided ^ 1);
  less |= pick & lengthNegative;
  greater |= pick & (lengthNegative ^ 1);
  return greater - less;
}

/** Branch-free select: `whenSet` for a 1 bit, `whenClear` for a 0 bit. */
function ctSelect(bit, whenSet, whenClear) {
  if (!(bit === 0 || bit === 1 || bit === 0n || bit === 1n)) throw new RangeError("bit must be 0 or 1");
  const mask = -Number(bit);
  return (whenSet & mask) | (whenClear & ~mask);
}

/** Runs `fn` under a fresh counter and returns the verdict plus the measured cost. */
function costOf(fn, left, right) {
  const counter = new Counter();
  const result = fn(new Memory(left, counter), new Memory(right, counter), counter);
  return { result, ...counter.snapshot() };
}

const oracleEqual = (a, b) => Buffer.from(a).equals(Buffer.from(b));
const oracleOrder = (a, b) => Math.sign(Buffer.compare(Buffer.from(a), Buffer.from(b)));

/** A byte array whose first `prefix` bytes match the secret (chosen-prefix timing probe). */
function attackVector(secret, prefix) {
  const candidate = Uint8Array.from(secret);
  if (prefix < candidate.length) candidate[prefix] = (candidate[prefix] + 1) & 0xff;
  return candidate;
}

/**
 * Prefix-timing attack: at every position try all 256 candidates and keep the one with the
 * maximal measured cost, since an early-exit comparison is slowest when the prefix matches.
 * `discriminating` counts the positions where the cost actually varied over the candidates —
 * zero means the timing channel carries no information about that position.
 */
function prefixAttack(compare, secret) {
  const guess = new Uint8Array(secret.length);
  let discriminating = 0;
  for (let position = 0; position < secret.length; position++) {
    const costs = new Set();
    let best = 0;
    let bestCost = -1;
    for (let candidate = 0; candidate < 256; candidate++) {
      const probe = Uint8Array.from(secret);
      probe.fill(0, position);
      probe[position] = candidate;
      const { probes } = costOf(compare, secret, probe);
      costs.add(probes);
      if (probes > bestCost) { bestCost = probes; best = candidate; }
    }
    if (costs.size > 1) discriminating++;
    guess[position] = best;
  }
  return { guess, discriminating };
}

/** Deterministic xorshift32 byte source. */
function makeByteSource(seed) {
  let state = seed >>> 0 || 0x755755;
  return () => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return state & 0xff;
  };
}

module.exports = {
  Counter, Memory, naiveEqual, constantTimeEqual, constantTimeCompare, ctSelect,
  costOf, oracleEqual, oracleOrder, attackVector, prefixAttack, makeByteSource,
};

if (require.main === module) {
  const SECRET = Uint8Array.from([0xde, 0xad, 0xbe, 0xef, 0x42, 0x17, 0x99, 0x00]);

  // 1. all three comparisons agree with the independent Buffer oracle on a small universe
  const universe = [
    Uint8Array.from([]), Uint8Array.from([0]), Uint8Array.from([1]), Uint8Array.from([0, 0]),
    Uint8Array.from([0, 1]), Uint8Array.from([1, 0]), Uint8Array.from([255, 255]),
    Uint8Array.from([0, 0, 0]), Uint8Array.from([0, 0, 1]), Uint8Array.from([1, 2, 3]), SECRET,
  ];
  let pairs = 0;
  for (const left of universe) {
    for (const right of universe) {
      pairs++;
      const expected = oracleEqual(left, right);
      strictEqual(costOf(constantTimeEqual, left, right).result, expected, `constantTimeEqual ${left}/${right}`);
      strictEqual(costOf(naiveEqual, left, right).result, expected, `naiveEqual ${left}/${right}`);
      strictEqual(costOf(constantTimeCompare, left, right).result, oracleOrder(left, right), `order ${left}/${right}`);
      const sign = costOf(constantTimeCompare, left, right).result;
      strictEqual(sign === 0, expected, "the order is zero exactly when the arrays are equal");
    }
  }

  // 2. the constant-time cost is a function of the lengths only; the naive cost leaks
  const constantCosts = new Set();
  const naiveCosts = [];
  for (let prefix = 0; prefix < SECRET.length; prefix++) {
    const candidate = attackVector(SECRET, prefix);
    const constant = costOf(constantTimeEqual, SECRET, candidate);
    constantCosts.add(constant.probes);
    strictEqual(constant.probes, 2 * SECRET.length, "constant-time probes are exactly 2 * min(len)");
    strictEqual(constant.branches, 0, "the constant-time path takes no data-dependent branch");
    strictEqual(constant.result, false, "the candidate really differs from the secret");
    naiveCosts.push(costOf(naiveEqual, SECRET, candidate).probes);
  }
  strictEqual(constantCosts.size, 1, "the constant-time cost must not depend on the matched prefix");
  deepStrictEqual(naiveCosts, Array.from({ length: SECRET.length }, (_, i) => 2 * (i + 1)),
    "negative control: the naive probe count grows with the matched prefix");

  // 3. random contents never change the constant-time cost, for every length pair
  const nextByte = makeByteSource(0x755755);
  for (let trial = 0; trial < 40; trial++) {
    const left = Uint8Array.from({ length: 8 }, nextByte);
    const right = Uint8Array.from({ length: 8 }, nextByte);
    const measured = costOf(constantTimeEqual, left, right);
    strictEqual(measured.probes, 16, "same length, same cost, whatever the bytes are");
    strictEqual(measured.branches, 0);
    strictEqual(measured.result, oracleEqual(left, right), "and the answer is still correct");
    const order = costOf(constantTimeCompare, left, right);
    strictEqual(order.probes, 16, "the sign-only comparison reads the shared prefix once");
    strictEqual(order.branches, 0);
    strictEqual(order.result, oracleOrder(left, right));
  }

  // 4. the prefix attack recovers a secret from the early-exit version and nothing from the other
  const naiveAttack = prefixAttack(naiveEqual, SECRET);
  deepStrictEqual([...naiveAttack.guess.slice(0, -1)], [...SECRET.slice(0, -1)], "the attack recovers the secret but for the final byte");
  strictEqual(naiveAttack.discriminating, SECRET.length - 1,
    "every position but the last is decided by timing: a full match and a last-byte mismatch cost the same");
  ok(costOf(naiveEqual, SECRET, SECRET).probes === costOf(naiveEqual, SECRET, attackVector(SECRET, SECRET.length - 1)).probes,
    "the tie at the final position is exactly why that byte is not recovered");
  const constantAttack = prefixAttack(constantTimeEqual, SECRET);
  strictEqual(constantAttack.discriminating, 0, "the constant-time comparison leaks no position");
  deepStrictEqual([...constantAttack.guess], new Array(SECRET.length).fill(0), "the attacker learns nothing");

  // 5. ctSelect is correct and branch-free for every mask/value combination
  for (const bit of [0, 1, 0n, 1n]) {
    strictEqual(ctSelect(bit, 0x5a, 0xa5), bit ? 0x5a : 0xa5, `ctSelect(${bit})`);
    for (let i = 0; i < 32; i++) strictEqual(ctSelect(bit, i, i ^ 0x7f), bit ? i : i ^ 0x7f, "ctSelect with a varying value");
  }
  throws(() => ctSelect(2, 1, 0), RangeError);
  throws(() => ctSelect(-1, 1, 0), RangeError);

  // 6. boundaries: empty/empty, empty/non-empty, single byte, prefix-only difference
  strictEqual(costOf(constantTimeEqual, Uint8Array.from([]), Uint8Array.from([])).result, true);
  const emptyVsOne = costOf(constantTimeEqual, Uint8Array.from([]), Uint8Array.from([0]));
  deepStrictEqual([emptyVsOne.result, emptyVsOne.probes], [false, 0], "nothing is read when nothing is shared");
  strictEqual(costOf(constantTimeEqual, Uint8Array.from([7]), Uint8Array.from([7])).result, true);
  strictEqual(costOf(constantTimeEqual, Uint8Array.from([7]), Uint8Array.from([8])).result, false);
  strictEqual(costOf(naiveEqual, Uint8Array.from([1, 2]), Uint8Array.from([1, 2, 3])).result, false);
  strictEqual(costOf(naiveEqual, Uint8Array.from([1, 2, 3]), Uint8Array.from([1, 2])).probes, 0, "a length mismatch reads nothing");
  const stretched = costOf(constantTimeEqual, Uint8Array.from([9, 9]), Uint8Array.from([9, 9, 9]));
  deepStrictEqual([stretched.result, stretched.probes], [false, 4], "only the shared prefix is read");
  strictEqual(oracleOrder(Uint8Array.from([9, 9]), Uint8Array.from([9, 9, 9])), -1, "the oracle agrees on the order");
  strictEqual(costOf(constantTimeCompare, Uint8Array.from([9, 9]), Uint8Array.from([9, 9, 9])).result, -1);

  // 7. validation and determinism
  throws(() => new Memory([1, 2], new Counter()), TypeError);
  throws(() => new Memory(new Uint8Array(1), undefined), TypeError);
  ok(typeof costOf(constantTimeEqual, SECRET, SECRET).probes === "number");
  deepStrictEqual(costOf(constantTimeEqual, SECRET, SECRET), costOf(constantTimeEqual, SECRET, SECRET), "same inputs, same cost");

  console.log(
    `pairs=${pairs} secretLen=${SECRET.length} ctProbes=${SECRET.length * 2} ` +
    `naiveSpread=${naiveCosts[naiveCosts.length - 1] - naiveCosts[0]} ctSpread=0 ` +
    `naiveLeak=${naiveAttack.discriminating}of${SECRET.length - 1} ctLeak=${constantAttack.discriminating}`
  );
  console.log("PASS");
}
