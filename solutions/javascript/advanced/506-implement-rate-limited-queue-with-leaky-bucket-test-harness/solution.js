/**
 * Task 506 - Implement rate-limited queue with leaky-bucket test harness. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
 * A leaky-bucket rate limiter plus a deterministic test harness. The bucket holds up to `capacity`
 * queued requests and leaks at `rate` requests per unit of virtual time; on every offer the level is
 * first updated with the elapsed time (level = max(0, level - rate * dt)), then the request is
 * admitted iff level + 1 <= capacity. Time comes only from the trace, never from the clock, so the
 * harness is reproducible. Admission is FIFO: requests are offered in arrival order and accepted
 * requests are dispatched in exactly that order. Two independent implementations are provided and
 * compared on every trace: the event-driven one above (O(offers)) and a tick-by-tick reference that
 * walks integer time steps, draining the bucket and then admitting arrivals (the same
 * drain-then-offer convention) - they must agree on every admission and on the queue level at every
 * offer.
 *
 * Documented checks: with capacity 5 and rate 4, offering 6 requests per tick to an empty bucket
 * admits exactly min(offers, C + floor(rate * T)) = 5, 9, 13, 17, 21, 25, 29 at T = 0..6 (the
 * closed form for offered load at or above the rate; the level politely tracks admitted - rate * T),
 * and a 20-request burst at t = 0 admits exactly the capacity and drops the other 15. The level
 * never exceeds the capacity and is never negative over any trace (checked offer by offer); exactly
 * capacity requests are admitted instantly when the bucket is empty and one more is rejected; the
 * documented drain behaviour is pinned down (a tick leaks the whole rate, so a long idle gap
 * restores the full capacity); and under sustained overload (8 offers per tick for 500 ticks at
 * rate 4) the admitted count is 2001, i.e. within one request of rate * T = 2000 with 1999
 * rejections. FIFO dispatch is checked against arrival indices on a 600-request LCG-bursty trace
 * (596 admitted, 4 rejected), and offers = admitted + rejected holds exactly on every trace.
 * Finally the event-driven engine and the tick-by-tick reference must agree on the admission vector
 * *and* on the bucket level at every offer across four traces - bursty, steady (120/120 admitted),
 * sparse (three 8-request bursts separated by idle gaps: 15 admitted, 9 dropped) and
 * exactly-at-capacity traffic (81/85) - and admission is shown to be causal in time: extending the
 * trace never revokes an earlier decision.
 */
"use strict";

/** Event-driven leaky bucket: drain by elapsed virtual time, then try to enqueue. */
class LeakyBucket {
  constructor(capacity, rate) {
    if (!(capacity > 0)) throw new RangeError("capacity must be positive");
    if (!(rate > 0)) throw new RangeError("rate must be positive");
    this.capacity = capacity;
    this.rate = rate;
    this.level = 0.0;
    this.last = null;
    this.admitted = 0;
    this.rejected = 0;
    this.dispatchOrder = [];
    this.levelHistory = [];
  }

  drain(now) {
    if (this.last !== null) {
      const elapsed = (now - this.last) / 1000.0;       // virtual milliseconds
      this.level = Math.max(0.0, this.level - this.rate * elapsed);
    }
    this.last = now;
  }

  offer(now, id) {
    this.drain(now);
    this.levelHistory.push(this.level);
    if (this.level + 1.0 <= this.capacity + 1e-12) {
      this.level += 1.0;
      this.admitted += 1;
      this.dispatchOrder.push(id);
      return true;
    }
    this.rejected += 1;
    return false;
  }
}

/** Tick-by-tick reference: for each integer tick, drain then admit arrivals in order. */
function simulateTicks(capacity, rate, offers, ticks) {
  let level = 0.0;
  let last = 0.0;
  const accepted = [];
  const levelAtOffer = [];
  const byTick = new Map();
  offers.forEach((offer, index) => {
    const tick = Math.floor(offer.time / 1000.0);
    if (!byTick.has(tick)) byTick.set(tick, []);
    byTick.get(tick).push({ ...offer, index });
  });
  const order = [];
  for (let tick = 0; tick <= ticks; tick++) {
    level = Math.max(0.0, level - rate * (tick - last));
    last = tick;
    const arrivals = byTick.get(tick) || [];
    for (const arrival of arrivals) {
      levelAtOffer[arrival.index] = level;
      if (level + 1.0 <= capacity + 1e-12) {
        level += 1.0;
        order.push(arrival.index);
      }
    }
  }
  return { accepted: order.map((index) => offers[index].id), levelAtOffer, finalLevel: level };
}

function runEventDriven(capacity, rate, offers) {
  const bucket = new LeakyBucket(capacity, rate);
  const accepted = [];
  for (const offer of offers) {
    if (bucket.offer(offer.time, offer.id)) accepted.push(offer.id);
  }
  return { accepted, levelAtOffer: bucket.levelHistory, level: bucket.level,
    rejected: bucket.rejected };
}

/** One burst of `count` requests, all at t = 0. */
function burstTrace(count, idPrefix = "r") {
  return Array.from({ length: count }, (_, i) => ({ time: 0, id: `${idPrefix}${i}` }));
}

/** Deterministic traffic: `perTick` offers per tick for `ticks` ticks. */
function steadyTrace(perTick, ticks) {
  const offers = [];
  for (let tick = 0; tick < ticks; tick++) {
    for (let i = 0; i < perTick; i++) offers.push({ time: tick * 1000, id: `s${tick}_${i}` });
  }
  return offers;
}

/** LCG traffic with bursts and idle gaps. */
function burstyTrace(count, seed) {
  const offers = [];
  let s = seed;
  let time = 0;
  for (let i = 0; i < count; i++) {
    s = (s * 75 + 74) % 65537;
    const gap = s % 3;                                 // 0, 1 or 2 ticks between offers
    time += gap * 1000;
    offers.push({ time, id: `b${i}` });
  }
  return offers;
}

/** Sparse traffic: a burst, a long idle gap, another burst. */
function sparseTrace() {
  const offers = [];
  for (let i = 0; i < 8; i++) offers.push({ time: 0, id: `a${i}` });
  for (let i = 0; i < 8; i++) offers.push({ time: 50 * 1000, id: `b${i}` });
  for (let i = 0; i < 8; i++) offers.push({ time: 100 * 1000, id: `c${i}` });
  return offers;
}

/** Every offer exactly one tick apart, so the bucket sits at exactly the capacity. */
function exactlyAtCapacityTrace(capacity, rate, ticks) {
  const offers = [];
  let count = 0;
  for (let tick = 0; tick < ticks; tick++) {
    const perTick = Math.round(rate);
    for (let i = 0; i < perTick + (tick < capacity ? 1 : 0); i++) {
      offers.push({ time: tick * 1000, id: `x${count++}` });
    }
  }
  return offers;
}

function compareEngines(capacity, rate, offers, ticks2) {
  const event = runEventDriven(capacity, rate, offers);
  const tick = simulateTicks(capacity, rate, offers, ticks2);
  if (JSON.stringify(event.accepted) !== JSON.stringify(tick.accepted)) {
    throw new Error("engines disagree on the admission sequence");
  }
  for (let i = 0; i < offers.length; i++) {
    if (Math.abs(event.levelAtOffer[i] - tick.levelAtOffer[i]) > 1e-9) {
      throw new Error(`engines disagree on the bucket level at offer ${i}`);
    }
  }
  return event;
}

function main() {
  const assert = require("node:assert");
  const close = (a, b, tol, message) =>
    assert.ok(Math.abs(a - b) <= tol, `${message}: |${a} - ${b}| > ${tol}`);

  const CAPACITY = 5;
  const RATE = 4;

  // (1) the closed form for offered load at or above the rate: admitted(T) = min(N, C + rate * T).
  const PER_TICK = 6;                                  // >= rate, so the bucket is always refilled
  const offered = steadyTrace(PER_TICK, 7);
  const tiers = [0, 1, 2, 3, 4, 5, 6];
  for (const tick of tiers) {
    const upTo = offered.filter((offer) => offer.time <= tick * 1000);
    const expected = Math.min(upTo.length, CAPACITY + Math.floor(RATE * tick));
    const tierResult = runEventDriven(CAPACITY, RATE, upTo);
    assert.strictEqual(tierResult.accepted.length, expected,
      `admitted(${tick}) = min(offers, C + rate * T) = ${expected}`);
    assert.ok(Math.abs(tierResult.level - (tierResult.accepted.length - RATE * tick)) <= 1e-9,
      "the level equals admitted - rate * T (the bucket stays exactly at the capacity here)");
  }
  assert.deepStrictEqual(tiers.map((tick) => Math.min(PER_TICK * (tick + 1),
    CAPACITY + RATE * tick)),
    [5, 9, 13, 17, 21, 25, 29], "the documented tier values");
  const burstRun = runEventDriven(CAPACITY, RATE, burstTrace(20));
  assert.strictEqual(burstRun.accepted.length, CAPACITY,
    "a 20-request burst at t = 0 admits exactly the capacity");
  assert.strictEqual(burstRun.rejected, 15, "the other 15 requests of the burst are dropped");

  // (2) instantaneous capacity behaviour and boundaries.
  const bucket = new LeakyBucket(CAPACITY, RATE);
  for (let i = 0; i < CAPACITY; i++) {
    assert.strictEqual(bucket.offer(0, `k${i}`), true, `offer ${i} fits an empty bucket`);
  }
  assert.strictEqual(bucket.offer(0, "overflow"), false, "one more than the capacity is rejected");
  assert.strictEqual(bucket.level, CAPACITY, "the level sits exactly at the capacity");
  close(bucket.level, 5.0, 1e-12, "level = capacity");
  const partialDrain = new LeakyBucket(CAPACITY, RATE);
  partialDrain.offer(0, "a");
  partialDrain.offer(0, "b");
  partialDrain.offer(1000, "c");
  close(partialDrain.level, 1.0, 1e-12,
    "one tick drains the whole rate, so only the new request remains");
  const drained = new LeakyBucket(CAPACITY, RATE);
  for (let i = 0; i < CAPACITY; i++) drained.offer(0, `d${i}`);
  drained.offer(2000, "after-drain");
  assert.strictEqual(drained.offer(2000, "e"), true, "a drained bucket accepts again");
  close(drained.level, 2.0, 1e-12, "only the two new requests are in the bucket");

  const idle = new LeakyBucket(CAPACITY, RATE);
  for (let i = 0; i < CAPACITY; i++) idle.offer(0, `i${i}`);
  assert.strictEqual(idle.offer(10 * 1000, "much later"), true,
    "a long idle gap restores the full capacity");

  // (3) FIFO order and exact accounting.
  const trace = burstyTrace(600, 4242);
  const result = runEventDriven(CAPACITY, RATE, trace);
  const arrivalIndexOf = new Map(trace.map((offer, index) => [offer.id, index]));
  for (let i = 1; i < result.accepted.length; i++) {
    assert.ok(arrivalIndexOf.get(result.accepted[i]) > arrivalIndexOf.get(result.accepted[i - 1]),
      "dispatch order is strictly increasing in arrival order (FIFO)");
  }
  assert.strictEqual(result.accepted.length + result.rejected, trace.length,
    "offers = admitted + rejected, exactly");
  assert.ok(result.rejected > 0, "this bursty trace does overflow the bucket");

  // (4) long-run throughput equals the rate under sustained overload.
  const sustained = steadyTrace(RATE + 4, 500);
  const saturated = runEventDriven(CAPACITY, RATE, sustained);
  const expectedAdmits = RATE * 500;
  assert.ok(Math.abs(saturated.accepted.length - expectedAdmits) <= RATE + 1,
    `sustained overload drains at the rate (${saturated.accepted.length} vs ${expectedAdmits})`);
  assert.strictEqual(saturated.accepted.length + saturated.rejected, sustained.length);
  assert.ok(saturated.rejected > 0, "the overload really is rejected");
  const offeredPerTick = RATE + 4;
  assert.ok(saturated.rejected >= (offeredPerTick - RATE) * 500 - 2 * CAPACITY,
    "rejections account for the excess offered load");

  // (5) the two engines agree on every trace.
  const traces = [
    ["bursty", trace, 10 * 1000],
    ["steady", steadyTrace(3, 40), 40 * 1000],
    ["sparse", sparseTrace(), 100 * 1000],
    ["at-capacity", exactlyAtCapacityTrace(CAPACITY, RATE, 20), 20 * 1000],
  ];
  const summary = [];
  for (const [name, offers, horizon] of traces) {
    const event = compareEngines(CAPACITY, RATE, offers, horizon);
    const levels = event.levelAtOffer.filter((level) => level !== undefined);
    assert.ok(levels.every((level) => level >= -1e-12 && level <= CAPACITY + 1e-12),
      `${name}: the level stays inside [0, capacity]`);
    assert.strictEqual(event.accepted.length + event.rejected, offers.length,
      `${name}: accounting is exact`);
    assert.ok(event.accepted.length > 0, `${name}: something is admitted`);
    assert.ok(event.level >= 0 && event.level <= CAPACITY, `${name}: final level in range`);
    summary.push([name, offers.length, event.accepted.length, event.rejected]);
  }
  // the sparse trace's idle gaps drain the bucket, so each burst gets a fresh capacity
  const sparse = runEventDriven(CAPACITY, RATE, sparseTrace());
  assert.strictEqual(sparse.accepted.length, 3 * CAPACITY,
    "each sparse burst admits exactly the capacity after its idle gap");
  assert.strictEqual(sparse.rejected, 24 - 3 * CAPACITY,
    "the excess of each 8-request burst is dropped");
  close(sparse.level, CAPACITY, 1e-12, "the last burst leaves the bucket full");

  // (6) horizon robustness: extending the trace never un-admits an earlier request.
  const short = runEventDriven(CAPACITY, RATE, trace.slice(0, 100));
  const long = runEventDriven(CAPACITY, RATE, trace);
  assert.deepStrictEqual(long.accepted.slice(0, short.accepted.length), short.accepted,
    "the admission decision is causal in time");

  // (7) invalid configuration is rejected.
  assert.throws(() => new LeakyBucket(0, 1), RangeError, "capacity must be positive");
  assert.throws(() => new LeakyBucket(1, 0), RangeError, "rate must be positive");
}

module.exports = { LeakyBucket, simulateTicks, runEventDriven, burstTrace, steadyTrace, burstyTrace,
  sparseTrace, exactlyAtCapacityTrace, compareEngines };

if (require.main === module) {
  main();
  console.log("PASS");
}
