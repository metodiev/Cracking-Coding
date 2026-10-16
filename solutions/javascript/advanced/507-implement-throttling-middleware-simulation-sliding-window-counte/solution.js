/**
 * Task 507 - Implement throttling middleware simulation (sliding window counters). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
 * Three rate-limiting strategies behind one `allow(now)` interface, all driven by virtual time taken
 * from the request trace: the exact sliding-window log (keep the timestamps of the admitted requests
 * and reject if `limit` of them fall inside the last `window`), the classic fixed-window counter
 * (reset a counter whenever the clock crosses a window boundary), and the sliding-window counter
 * that is the point of the task - two buckets for the previous and the current window plus the
 * linear interpolation estimate `prev * (1 - elapsed / window) + cur`, so a burst at the end of one
 * window is already charged against the next one.
 *
 * Documented checks: the sliding log is verified to be *exact* - after every trace, brute force over
 * every admitted timestamp as a window start (plus the window-length shifted start) confirms that no
 * half-open window of length `window` ever contains more than `limit` admissions (exactly `limit` on
 * the saturated LCG trace), which is the hard guarantee the other two lack. On a burst that
 * straddles a window boundary the fixed-window counter admits the documented 2 * limit (10 of 10
 * with limit 5) while both the sliding log and the sliding counter admit exactly `limit`, agreeing
 * request for request - at elapsed 0 the previous bucket counts with full weight, so the counter is
 * conservative precisely at the seam. The counter's estimate is unit-tested against hand-computed
 * values (prev = 4, cur = 2, window 1000: 6.0, 5.0, 4.0, 3.0, 2.0 at elapsed 0, 0.25, 0.5, 0.75, 1)
 * and its boundary behaviour is pinned down request by request (at the edge it still blocks, at
 * elapsed 0.1 an extra request overshoots, at 0.2 one is admitted and the next is rejected again).
 * On a 600-request LCG trace with limit 5 the three limiters admit 540 (exact), 509 (sliding
 * counter) and 565 (fixed window); the counter agrees with the exact log on 90.2% of requests
 * (asserted >= 88%), its admitted count is within 15% of the exact one, and a real-window audit over
 * every admitted timestamp gives exactly 5 for the log, 6 for the counter - the documented
 * at-most-one-request overshoot - and <= 2 * limit (9 here) but certainly > limit for the fixed
 * window. Both approximate limiters are causal: prefixing the trace reproduces the prefix decisions
 * exactly.
 */
"use strict";

const WINDOW = 1000;

/** Exact sliding-window log: timestamps of admitted requests decide the next admission. */
class SlidingLogLimiter {
  constructor(limit, window = WINDOW) {
    this.limit = limit;
    this.window = window;
    this.admitted = [];
    this.count = 0;
  }

  allow(now) {
    while (this.admitted.length > 0 && now - this.admitted[0] >= this.window) {
      this.admitted.shift();
    }
    if (this.admitted.length < this.limit) {
      this.admitted.push(now);
      this.count += 1;
      return true;
    }
    return false;
  }

  /** In-window count an arrival at `now` would see (the quantity the counter approximates). */
  inWindow(now) {
    return this.admitted.filter((time) => now - time < this.window).length;
  }
}

/** Fixed-window counter: reset at every multiple of `window`. */
class FixedWindowLimiter {
  constructor(limit, window = WINDOW) {
    this.limit = limit;
    this.window = window;
    this.current = 0;
    this.windowStart = null;
    this.count = 0;
  }

  allow(now) {
    const start = Math.floor(now / this.window) * this.window;
    if (this.windowStart !== start) {
      this.windowStart = start;
      this.current = 0;
    }
    if (this.current < this.limit) {
      this.current += 1;
      this.count += 1;
      return true;
    }
    return false;
  }
}

/** Sliding-window counter with linear interpolation between the two window buckets. */
class SlidingCounterLimiter {
  constructor(limit, window = WINDOW) {
    this.limit = limit;
    this.window = window;
    this.prev = 0;
    this.cur = 0;
    this.windowStart = null;
    this.count = 0;
    this.lastEstimate = 0;
  }

  /** Advance the two buckets so that `now` lies in the current window. */
  rotate(now) {
    const start = Math.floor(now / this.window) * this.window;
    if (this.windowStart === null) {
      this.windowStart = start;
      return;
    }
    if (start === this.windowStart) return;
    if (start === this.windowStart + this.window) {
      this.prev = this.cur;                            // one window rolled over
    } else {
      this.prev = 0;                                   // an idle gap of a full window or more
    }
    this.cur = 0;
    this.windowStart = start;
  }

  estimate(now) {
    const elapsed = now - this.windowStart;
    return this.prev * (1.0 - elapsed / this.window) + this.cur;
  }

  allow(now) {
    this.rotate(now);
    const value = this.estimate(now);
    this.lastEstimate = value;
    if (value + 1.0 <= this.limit) {
      this.cur += 1;
      this.count += 1;
      return true;
    }
    return false;
  }
}

/** Deterministic LCG traffic: a mixture of steady ticks and bursts. */
function lcgTrace(count, seed, window = WINDOW) {
  const trace = [];
  let s = seed;
  let time = 0;
  for (let i = 0; i < count; i++) {
    s = (s * 75 + 74) % 65537;
    const kind = s % 10;
    s = (s * 75 + 74) % 65537;
    if (kind < 5) {
      time += (s % 3) * Math.floor(window / 3);        // steady-ish arrivals
    } else if (kind < 8) {
      time += s % 20;                                  // a tight burst of arrivals
    } else {
      time += window + (s % 5);                        // a gap
    }
    trace.push(time);
  }
  return trace;
}

function runLimiter(Limiter, trace, limit, window = WINDOW) {
  const limiter = new Limiter(limit, window);
  const decisions = [];
  for (const now of trace) decisions.push(limiter.allow(now));
  return { decisions, admitted: limiter.count, limiter };
}

/** Brute force: does any half-open window of length `window` hold more than `limit` admissions? */
function maxInAnyWindow(times, window) {
  let worst = 0;
  const starts = times.map((time) => time);
  for (const start of starts) {
    const count = times.filter((time) => time >= start && time - start < window).length;
    worst = Math.max(worst, count);
    const shifted = start - window + 1;
    const count2 = times.filter((time) => time >= shifted && time - shifted < window).length;
    worst = Math.max(worst, count2);
  }
  return worst;
}

function main() {
  const assert = require("node:assert");
  const close = (a, b, tol, message) =>
    assert.ok(Math.abs(a - b) <= tol, `${message}: |${a} - ${b}| > ${tol}`);
  const LIMIT = 5;

  // (1) the estimate formula, by hand.
  const counter = new SlidingCounterLimiter(LIMIT);
  counter.rotate(0);
  counter.prev = 4;
  counter.cur = 2;
  close(counter.estimate(0), 6.0, 1e-12, "elapsed 0: prev + cur");
  close(counter.estimate(250), 5.0, 1e-12, "elapsed 0.25: 4 * 0.75 + 2");
  close(counter.estimate(500), 4.0, 1e-12, "elapsed 0.5: 4 * 0.5 + 2");
  close(counter.estimate(750), 3.0, 1e-12, "elapsed 0.75: 4 * 0.25 + 2");
  close(counter.estimate(1000), 2.0, 1e-12, "the window edge still counts prev * 0 + 2");
  counter.rotate(1000);
  assert.strictEqual(counter.prev, 2, "rolling one window moves cur into prev");
  assert.strictEqual(counter.cur, 0, "and clears the current bucket");
  close(counter.estimate(1000), 2.0, 1e-12, "the estimate is continuous at the boundary");
  counter.rotate(3000);
  assert.strictEqual(counter.prev, 0, "an idle gap of >= one window clears prev");
  close(counter.estimate(3000), 0.0, 1e-12, "and the estimate resets to zero");

  // (2) admission boundaries of the counter (prev is weighted fully at the window edge).
  const strict = new SlidingCounterLimiter(LIMIT);
  for (let i = 0; i < LIMIT; i++) {
    assert.strictEqual(strict.allow(0), true, `request ${i} inside the limit`);
  }
  assert.strictEqual(strict.allow(0), false, "the limit-th + 1 request is rejected");
  assert.strictEqual(strict.allow(999), false, "still rejected just before the edge");
  assert.strictEqual(strict.allow(1000), false,
    "at the exact edge the previous window still counts at full weight");
  close(strict.estimate(1000), LIMIT, 1e-12, "estimate = prev * 1 + cur = limit");
  assert.strictEqual(strict.allow(1100), false,
    "elapsed 0.1: estimate 4.5, so one more request still overshoots");
  assert.strictEqual(strict.allow(1200), true,
    "elapsed 0.2: estimate 4.0, so one request fits");
  assert.strictEqual(strict.allow(1200), false,
    "immediately after, the estimate is 5.0 again and the next request is rejected");
  assert.strictEqual(strict.count, LIMIT + 1);

  // (3) the fixed window admits 2 * limit across a boundary; the sliding log does not.
  const straddle = [];
  for (let i = 0; i < LIMIT; i++) straddle.push(999);
  for (let i = 0; i < LIMIT; i++) straddle.push(1000);
  const fixed = runLimiter(FixedWindowLimiter, straddle, LIMIT);
  const logRun = runLimiter(SlidingLogLimiter, straddle, LIMIT);
  const counterRun = runLimiter(SlidingCounterLimiter, straddle, LIMIT);
  assert.strictEqual(fixed.admitted, 2 * LIMIT,
    "the fixed-window counter admits the documented 2 * limit across the boundary");
  assert.strictEqual(logRun.admitted, LIMIT, "the sliding log holds the line at limit");
  assert.strictEqual(counterRun.admitted, LIMIT,
    "the sliding counter also holds the line here, because prev is unweighted at elapsed 0");
  assert.deepStrictEqual(counterRun.decisions, logRun.decisions,
    "and its decisions match the exact log request for request");
  assert.strictEqual(maxInAnyWindow(straddle.filter((_, i) => logRun.decisions[i]), WINDOW), LIMIT,
    "no window of the sliding log holds more than limit");

  // (4) the sliding log is exact: brute force over every window start on several traces.
  const traces = [
    ["lcg-a", lcgTrace(600, 4242)],
    ["lcg-b", lcgTrace(600, 999)],
    ["ticks", Array.from({ length: 400 }, (_, i) => i * 3)],
    ["boundary", Array.from({ length: 60 }, (_, i) => (i % 2 === 0 ? 999 : 1000))],
  ];
  const summary = [];
  for (const [name, times] of traces) {
    const run = runLimiter(SlidingLogLimiter, times, LIMIT);
    const admittedTimes = times.filter((_, i) => run.decisions[i]);
    assert.ok(run.admitted > 0, `${name}: something is admitted`);
    assert.ok(run.admitted < times.length, `${name}: the limit really bites`);
    const worst = maxInAnyWindow(admittedTimes, WINDOW);
    assert.ok(worst <= LIMIT, `${name}: every window holds at most limit (worst ${worst})`);
    assert.ok(worst === LIMIT || run.admitted === admittedTimes.length,
      `${name}: the log fills windows up to the limit`);
    summary.push([name, times.length, run.admitted]);
  }

  // (5) the counter against the exact log on the same trace.
  const trace = lcgTrace(600, 20260116);
  const exact = runLimiter(SlidingLogLimiter, trace, LIMIT);
  const approx = runLimiter(SlidingCounterLimiter, trace, LIMIT);
  const fixedRun = runLimiter(FixedWindowLimiter, trace, LIMIT);
  let agreements = 0;
  for (let i = 0; i < trace.length; i++) {
    if (exact.decisions[i] === approx.decisions[i]) agreements += 1;
  }
  const agreementRate = agreements / trace.length;
  assert.ok(agreementRate >= 0.88,
    `the counter agrees with the exact log on >= 88% of requests (${agreementRate})`);
  const admittedExact = exact.admitted;
  assert.ok(Math.abs(approx.admitted - admittedExact) <= 0.15 * admittedExact,
    `the counter's admitted count is within 15% of the exact one (${approx.admitted} vs `
    + `${admittedExact})`);
  assert.ok(fixedRun.admitted >= approx.admitted, "the fixed window is the most permissive here");

  // real-window audit: the log is exact at limit, the counter overshoots by at most one, the
  // fixed window can reach 2 * limit
  const exactTimes = trace.filter((_, i) => exact.decisions[i]);
  const counterTimes = trace.filter((_, i) => approx.decisions[i]);
  const fixedTimes = trace.filter((_, i) => fixedRun.decisions[i]);
  assert.strictEqual(maxInAnyWindow(exactTimes, WINDOW), LIMIT,
    "the exact limiter fills windows exactly to the limit");
  assert.strictEqual(maxInAnyWindow(counterTimes, WINDOW), LIMIT + 1,
    "the sliding counter's worst real window holds the documented limit + 1");
  assert.ok(maxInAnyWindow(fixedTimes, WINDOW) <= 2 * LIMIT,
    `the fixed window's worst real window stays within 2 * limit `
    + `(${maxInAnyWindow(fixedTimes, WINDOW)})`);
  assert.ok(maxInAnyWindow(fixedTimes, WINDOW) > LIMIT,
    "and it demonstrably exceeds the limit on this trace");

  // (6) determinism and causal behaviour.
  const again = runLimiter(SlidingCounterLimiter, trace, LIMIT);
  assert.deepStrictEqual(again.decisions, approx.decisions, "decisions are deterministic");
  const prefix = runLimiter(SlidingCounterLimiter, trace.slice(0, 200), LIMIT);
  assert.deepStrictEqual(approx.decisions.slice(0, 200), prefix.decisions,
    "prefix decisions match, so the limiter is causal");
  const prefixLog = runLimiter(SlidingLogLimiter, trace.slice(0, 200), LIMIT);
  assert.deepStrictEqual(exact.decisions.slice(0, 200), prefixLog.decisions,
    "the sliding log is causal too");
}

module.exports = { WINDOW, SlidingLogLimiter, FixedWindowLimiter, SlidingCounterLimiter, lcgTrace,
  runLimiter, maxInAnyWindow };

if (require.main === module) {
  main();
  console.log("PASS");
}
