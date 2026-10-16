/** Task 660 - Implement quorum consensus read/write latency optimization tradeoffs. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * In a quorum-replicated store with N replicas, a read waits for R acks and a write waits for W.
 * The consistency guarantee is pure set algebra: R + W > N means every read quorum intersects
 * every write quorum, so a read always sees the latest write. The *performance* side is an
 * order statistic: with per-replica latencies L and no failures, a quorum of size k completes
 * when the k-th fastest replica answers, i.e. at the k-th order statistic of L.
 *
 * SKETCH — simplified on purpose: replica latencies are fixed per replica (no per-request
 * noise, no retries, no timeouts, no failures), a "hedged" request is not modelled, and the
 * write path is charged a single order statistic instead of the full anti-entropy pipeline.
 * What is genuinely verified: (1) the intersection property is checked exhaustively over all
 * read/write quorum *sets*, not just over R + W > N; (2) quorum latency is a real order
 * statistic, monotone in the acknowledgement count; (3) the optimiser's recommendation, which
 * only searches the Pareto frontier R + W = N + 1, is compared against a brute-force search over
 * every valid (R, W) pair — the frontier restriction is justified because both latencies are
 * non-decreasing in the quorum size, and that is exactly what the brute-force check confirms.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

function makeRng(seed) {
  let state = seed >>> 0 || 0x2545f491;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

/** Fixed per-replica latency profile: `base` plus jitter, one slow replica by default. */
function replicaProfile(n, seed = 7, options = {}) {
  if (!Number.isInteger(n) || n < 1) throw new RangeError("replica count must be a positive integer");
  const base = options.base ?? 10;
  const jitter = options.jitter ?? 6;
  const slowIndex = options.slowIndex ?? n - 1;
  const slowFactor = options.slowFactor ?? 1;
  const rng = makeRng(seed);
  return Array.from({ length: n }, (_, i) =>
    Math.round((base + rng(jitter)) * (i === slowIndex ? slowFactor : 1) * 100) / 100);
}

/** k-th smallest latency (k = 1..n): the time a quorum of k acknowledgements takes. */
function orderStatistic(latencies, k) {
  if (!Number.isInteger(k) || k < 1 || k > latencies.length) {
    throw new RangeError(`ack count ${k} is outside 1..${latencies.length}`);
  }
  const sorted = [...latencies].sort((a, b) => a - b);
  return sorted[k - 1];
}

const isValidQuorum = (n, r, w) =>
  Number.isInteger(r) && Number.isInteger(w) && r >= 1 && w >= 1 && r <= n && w <= n && r + w > n;

/** Every valid (R, W) pair for N replicas. */
function enumerateConfigurations(n) {
  const out = [];
  for (let r = 1; r <= n; r++) for (let w = 1; w <= n; w++) if (isValidQuorum(n, r, w)) out.push({ r, w });
  return out;
}

const subsetsOfSize = (items, size) => {
  if (size === 0) return [[]];
  if (size > items.length) return [];
  const [head, ...rest] = items;
  return [...subsetsOfSize(rest, size - 1).map((s) => [head, ...s]), ...subsetsOfSize(rest, size)];
};

/**
 * Brute force: is every R-subset intersecting every W-subset? This is the actual consistency
 * condition, checked against the arithmetic rule R + W > N in the tests.
 */
function intersectionHolds(n, r, w) {
  const replicas = Array.from({ length: n }, (_, i) => i);
  const reads = subsetsOfSize(replicas, r);
  const writes = subsetsOfSize(replicas, w);
  for (const read of reads) {
    for (const write of writes) {
      if (!read.some((replica) => write.includes(replica))) return false;
    }
  }
  return true;
}

/** Expected latency of one read / one write under a given configuration. */
function latenciesFor(profile, r, w) {
  return { read: orderStatistic(profile, r), write: orderStatistic(profile, w) };
}

/** Weighted cost of a configuration: readLatency * readShare + writeLatency * writeShare. */
function weightedCost(profile, r, w, readShare) {
  const { read, write } = latenciesFor(profile, r, w);
  return read * readShare + write * (1 - readShare);
}

/**
 * Optimiser: restrict to the Pareto frontier R + W = N + 1 (any larger quorum pair can be
 * shrunk by one on either side without breaking the intersection rule, and both latencies are
 * non-decreasing in the quorum size) and scan the N frontier points.
 */
function optimiseQuorum(profile, options = {}) {
  const n = profile.length;
  const readShare = options.readShare ?? 0.5;
  let best = null;
  for (let r = 1; r <= n; r++) {
    const w = n + 1 - r;
    if (!isValidQuorum(n, r, w)) continue;
    const cost = weightedCost(profile, r, w, readShare);
    if (best === null || cost < best.cost) best = { r, w, cost };
  }
  return { ...best, ...latenciesFor(profile, best.r, best.w) };
}

/** Independent reference: brute force over every valid (R, W) pair. */
function bruteForceOptimum(profile, options = {}) {
  const n = profile.length;
  const readShare = options.readShare ?? 0.5;
  let best = null;
  let evaluated = 0;
  for (const { r, w } of enumerateConfigurations(n)) {
    evaluated++;
    const cost = weightedCost(profile, r, w, readShare);
    if (best === null || cost < best.cost) best = { r, w, cost };
  }
  return { ...best, evaluated };
}

module.exports = {
  makeRng, replicaProfile, orderStatistic, isValidQuorum, enumerateConfigurations,
  intersectionHolds, latenciesFor, weightedCost, optimiseQuorum, bruteForceOptimum, subsetsOfSize,
};

if (require.main === module) {
  // -- order statistics -----------------------------------------------------------
  const probe = [30, 10, 50, 20, 40];
  strictEqual(orderStatistic(probe, 1), 10, "one ack is the fastest replica");
  strictEqual(orderStatistic(probe, 3), 30, "three acks is the median here");
  strictEqual(orderStatistic(probe, 5), 50, "all acks means waiting for the slowest");
  strictEqual(orderStatistic([7], 1), 7);
  throws(() => orderStatistic(probe, 0), RangeError, "an ack count of zero is invalid");
  throws(() => orderStatistic(probe, 6), RangeError, "more acks than replicas is invalid");
  for (let k = 2; k <= probe.length; k++) {
    ok(orderStatistic(probe, k) >= orderStatistic(probe, k - 1), "quorum latency is non-decreasing in k");
  }

  // -- the actual consistency rule, by brute force --------------------------------
  for (const n of [3, 4, 5]) {
    for (let r = 1; r <= n; r++) {
      for (let w = 1; w <= n; w++) {
        strictEqual(intersectionHolds(n, r, w), isValidQuorum(n, r, w),
          `N=${n} R=${r} W=${w}: intersection must be equivalent to R + W > N`);
      }
    }
  }
  strictEqual(intersectionHolds(4, 2, 2), false, "N=4 R=W=2 admits disjoint read/write quorums");
  strictEqual(intersectionHolds(4, 2, 3), true, "N=4 R=2 W=3 always intersects");
  strictEqual(enumerateConfigurations(3).length, 6, "N=3 has six valid configurations");
  deepStrictEqual(enumerateConfigurations(1), [{ r: 1, w: 1 }], "N=1 only admits R=W=1");
  ok(enumerateConfigurations(5).every(({ r, w }) => r + w > 5));

  // -- latency profile and monotonicity ------------------------------------------
  const profile = replicaProfile(5, 2025, { base: 10, jitter: 8, slowIndex: 4, slowFactor: 3 });
  strictEqual(profile.length, 5);
  deepStrictEqual(replicaProfile(5, 2025, { base: 10, jitter: 8, slowIndex: 4, slowFactor: 3 }), profile,
    "the profile must be reproducible");
  ok(profile[4] > profile[0], "the slow replica must actually be slow");
  throws(() => replicaProfile(0), RangeError, "zero replicas is invalid");
  for (let k = 2; k <= 5; k++) {
    ok(latenciesFor(profile, k, k).read >= latenciesFor(profile, k - 1, k - 1).read,
      "waiting for more acks cannot finish earlier");
  }
  ok(latenciesFor(profile, 3, 3).read < latenciesFor(profile, 5, 5).read,
    "a 3-of-5 read beats a 5-of-5 read when one replica is slow");

  // -- optimiser versus brute force ---------------------------------------------
  for (const readShare of [0.1, 0.5, 0.9]) {
    const recommendation = optimiseQuorum(profile, { readShare });
    const reference = bruteForceOptimum(profile, { readShare });
    strictEqual(recommendation.cost, reference.cost,
      `the frontier search must find the global optimum (readShare=${readShare})`);
    strictEqual(recommendation.r + recommendation.w, profile.length + 1,
      "the optimum always lies on the minimal frontier R + W = N + 1");
    strictEqual(reference.evaluated, enumerateConfigurations(profile.length).length,
      "the reference must evaluate every valid pair");
    ok(recommendation.r >= 1 && recommendation.w >= 1 && recommendation.r <= 5 && recommendation.w <= 5);
  }
  const readHeavy = optimiseQuorum(profile, { readShare: 0.9 });
  const writeHeavy = optimiseQuorum(profile, { readShare: 0.1 });
  ok(readHeavy.r <= writeHeavy.r,
    `a read-heavy workload must not use more read acks: ${readHeavy.r} vs ${writeHeavy.r}`);
  ok(readHeavy.w >= writeHeavy.w, "and it must pay with more write acks");

  // -- tradeoff curve -----------------------------------------------------------
  const curve = enumerateConfigurations(5)
    .map(({ r, w }) => ({ r, w, ...latenciesFor(profile, r, w) }))
    .sort((a, b) => (a.r - b.r) || (a.w - b.w));
  ok(curve.every((p) => p.read >= orderStatistic(profile, 1)), "no read can beat the fastest replica");
  ok(curve.every((p) => p.write <= orderStatistic(profile, 5)), "no write exceeds the slowest replica");
  const strictPairs = curve.filter((p) => p.r + p.w > 6);
  ok(strictPairs.length > 0 && strictPairs.every((p) => p.read + p.write > 0));
  const minRead = curve.reduce((best, p) => (p.read < best.read ? p : best));
  strictEqual(minRead.r, 1, "the cheapest read quorum is a single ack");
  const minWrite = curve.reduce((best, p) => (p.write < best.write ? p : best));
  strictEqual(minWrite.w, 1, "the cheapest write quorum is a single ack");

  // -- identical latencies and single replica ------------------------------------
  const flat = new Array(4).fill(25);
  const flatRecommendation = optimiseQuorum(flat, { readShare: 0.5 });
  strictEqual(flatRecommendation.cost, bruteForceOptimum(flat, { readShare: 0.5 }).cost,
    "with identical latencies the tie must still be the global optimum");
  strictEqual(flatRecommendation.r + flatRecommendation.w, 5);
  strictEqual(weightedCost(flat, 2, 3, 0.5), 25, "flat latencies make every quorum equally fast");
  strictEqual(optimiseQuorum([15]).cost, 15, "a single replica has exactly one configuration");
  ok(enumerateConfigurations(1).every(({ r, w }) => r === 1 && w === 1));

  // -- extra replicas cannot lower the k-th order statistic ----------------------
  const four = replicaProfile(4, 99, { base: 10, jitter: 5 });
  const five = [...four, 900];
  for (let k = 1; k <= 4; k++) {
    strictEqual(orderStatistic(five, k), orderStatistic(four, k),
      "adding a slow replica must not speed up a quorum that ignores it");
  }
  ok(orderStatistic(five, 5) > orderStatistic(four, 4), "but demanding every ack is now worse");
  ok(bruteForceOptimum(five, { readShare: 0.5 }).w <= 4,
    "with a 900ms replica the optimal write quorum avoids requiring every ack");

  console.log(`N=5 profile=[${profile.join(", ")}] frontier-optimal R=${readHeavy.r}/W=${readHeavy.w} ` +
    `(read-heavy, cost=${readHeavy.cost.toFixed(2)}ms), R=${writeHeavy.r}/W=${writeHeavy.w} ` +
    `(write-heavy, cost=${writeHeavy.cost.toFixed(2)}ms)`);
  console.log("PASS");
}
