/** Task 689 - Implement congestion control algorithms simulator (TCP Reno/CUBIC). (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Model: a round-based (one RTT per round) congestion-control simulator for a single bulk flow over
 * one bottleneck. The bottleneck is a deterministic droptail link: of the `offered` segments in a
 * round, `capacity` get through and the rest are dropped. A loss is classified the way a sender can
 * observe it: if at least 3 segments were delivered behind the loss the sender sees duplicate ACKs
 * and performs a fast retransmit, otherwise the ACK clock starves and it takes a timeout. All window
 * arithmetic is in segments and every metric below is deterministic.
 *
 * Reno: slow start doubles cwnd per round until ssthresh, congestion avoidance adds one segment per
 * RTT, and a loss halves the window (ssthresh = max(floor(cwnd/2), 2), cwnd = ssthresh for a fast
 * retransmit, cwnd = 1 for a timeout).
 *
 * CUBIC: after the first loss the window no longer follows an AIMD ramp but the cubic function
 * W(t) = C (t - K)^3 + W_max with K = cbrt(W_max (1 - beta) / C), C = 0.4, beta = 0.7; each round
 * cwnd moves a fraction (W(t) - cwnd) / cwnd towards the target, which is exactly the per-ACK
 * increment of the real implementation. W_max is retained when a loss happens before the previous
 * maximum was reached. The function is concave up to K and convex beyond it, which is what lets
 * CUBIC probe quickly when it is far from the last loss point and stop crowding the link when it is
 * close to it.
 *
 * Simplified versus a real stack: no delayed ACKs, no fast-recovery inflation (cwnd jumps straight
 * to ssthresh), no SACK, no pacing, infinite receiver window, one flow, and the loss pattern comes
 * from the droptail model rather than a queue simulation. Verified: the per-round send/ack
 * accounting, cwnd >= 1 and the exact post-loss window for both algorithms, slow start doubling,
 * congestion-avoidance slope, the timeout path, the analytic properties of the CUBIC window
 * function (W(K) = W_max, W(0) = beta W_max, concavity then convexity, plateau at K), link
 * utilisation in a sane band, and that CUBIC's gentler decrease keeps a larger average window than
 * Reno on the same bottleneck.
 */
"use strict";

const MSS = 1;                                    // all windows are counted in segments
const MIN_SSHRESH = 2;                            // a window of 1 segment can never be halved further

/** Deterministic droptail bottleneck: everything above the capacity is lost. */
class Bottleneck {
  constructor(capacity) {
    if (!Number.isInteger(capacity) || capacity < 1) throw new RangeError("capacity must be >= 1");
    this.capacity = capacity;
  }

  offer(segments) {
    const delivered = Math.min(segments, this.capacity);
    return { delivered, dropped: segments - delivered };
  }
}

function classifyLoss(delivered) {
  return delivered >= 3 ? "fast-retransmit" : "timeout";
}

class Reno {
  constructor({ bottleneck, initialWindow = 1, initialSsthresh = 64, maxRounds = 120 }) {
    this.bottleneck = bottleneck;
    this.cwnd = initialWindow;
    this.ssthresh = initialSsthresh;
    this.maxRounds = maxRounds;
    this.history = [];
  }

  run(rounds = this.maxRounds) {
    this.history = [];
    for (let round = 1; round <= rounds; round++) {
      const offered = Math.floor(this.cwnd);
      const { delivered, dropped } = this.bottleneck.offer(offered);
      const before = this.cwnd;
      let event;
      if (dropped > 0) {
        event = classifyLoss(delivered);
        this.ssthresh = Math.max(Math.floor(this.cwnd / 2), MIN_SSHRESH);
        this.cwnd = event === "timeout" ? 1 : this.ssthresh;
      } else if (this.cwnd < this.ssthresh) {
        event = "slow-start";
        this.cwnd += delivered;                   // one ACK per segment: the window doubles
      } else {
        event = "congestion-avoidance";
        this.cwnd += 1;                           // one extra segment per RTT
      }
      this.history.push({ round, before, cwnd: this.cwnd, ssthresh: this.ssthresh, offered, delivered, dropped, event });
    }
    return this.history;
  }
}

class Cubic {
  constructor({ bottleneck, beta = 0.7, c = 0.4, initialWindow = 1, initialSsthresh = 64, maxRounds = 120 }) {
    if (!(beta > 0 && beta < 1)) throw new RangeError("beta must be in (0, 1)");
    if (!(c > 0)) throw new RangeError("C must be positive");
    this.bottleneck = bottleneck;
    this.beta = beta;
    this.c = c;
    this.cwnd = initialWindow;
    this.initialSsthresh = initialSsthresh;
    this.maxRounds = maxRounds;
    this.history = [];
  }

  /** W(t) = C (t - K)^3 + W_max, with K = cbrt(W_max (1 - beta) / C). */
  windowAt(t, wMax) {
    const k = Cubic.kFor(wMax, this.c, this.beta);
    return this.c * (t - k) ** 3 + wMax;
  }

  static kFor(wMax, c = 0.4, beta = 0.7) {
    return Math.cbrt((Math.max(wMax, 0) * (1 - beta)) / c);
  }

  run(rounds = this.maxRounds) {
    let wMax = 0;
    let epochStart = 0;
    let slowStart = true;
    this.history = [];
    for (let round = 1; round <= rounds; round++) {
      const offered = Math.floor(this.cwnd);
      const { delivered, dropped } = this.bottleneck.offer(offered);
      const before = this.cwnd;
      let event;
      if (dropped > 0) {
        event = classifyLoss(delivered);
        wMax = Math.max(wMax, this.cwnd);          // retained if the maximum was not reached again
        this.cwnd = Math.max(this.cwnd * this.beta / MSS, 1);
        epochStart = round;
        slowStart = event === "timeout";
      } else if (slowStart && this.cwnd < this.initialSsthresh) {
        event = "slow-start";
        this.cwnd += delivered;
      } else {
        event = "congestion-avoidance";
        const target = this.windowAt(round - epochStart, wMax);
        this.cwnd += (target - this.cwnd) / Math.max(this.cwnd, 1);
      }
      this.history.push({ round, before, cwnd: this.cwnd, wMax, offered, delivered, dropped, event });
    }
    return this.history;
  }
}

function summarize(history, capacity) {
  const total = history.length;
  const sumSent = history.reduce((acc, step) => acc + step.delivered, 0);
  const sumWindow = history.reduce((acc, step) => acc + step.before, 0);
  const losses = history.filter((step) => step.dropped > 0);
  return {
    rounds: total,
    utilization: sumSent / (total * capacity),
    meanWindow: sumWindow / total,
    minWindow: Math.min(...history.map((step) => step.before)),
    maxWindow: Math.max(...history.map((step) => step.before)),
    losses: losses.length,
    timeouts: losses.filter((step) => step.event === "timeout").length,
    fastRetransmits: losses.filter((step) => step.event === "fast-retransmit").length,
  };
}

module.exports = { MSS, MIN_SSHRESH, Bottleneck, Reno, Cubic, classifyLoss, summarize };

if (require.main === module) {
  const assert = require("node:assert");

  // --- the loss classifier -----------------------------------------------------------------
  assert.strictEqual(classifyLoss(0), "timeout");
  assert.strictEqual(classifyLoss(2), "timeout");
  assert.strictEqual(classifyLoss(3), "fast-retransmit", "three duplicate ACKs are enough");
  assert.strictEqual(classifyLoss(50), "fast-retransmit");
  assert.throws(() => new Bottleneck(0), RangeError);
  assert.deepStrictEqual(new Bottleneck(5).offer(3), { delivered: 3, dropped: 0 });
  assert.deepStrictEqual(new Bottleneck(5).offer(8), { delivered: 5, dropped: 3 });

  // --- Reno slow start doubling and the congestion-avoidance slope -------------------------
  const wide = new Reno({ bottleneck: new Bottleneck(1000), initialWindow: 1, initialSsthresh: 32 });
  const doubling = wide.run(6);
  assert.deepStrictEqual(doubling.map((step) => step.before), [1, 2, 4, 8, 16, 32], "cwnd doubles per RTT");
  assert.ok(doubling.slice(0, 5).every((step) => step.event === "slow-start"), "five doubling rounds");
  assert.strictEqual(doubling[5].event, "congestion-avoidance", "slow start ends when cwnd reaches ssthresh");
  const after = wide.run(3);
  assert.deepStrictEqual(after.map((step) => step.cwnd - step.before), [1, 1, 1], "+1 segment per RTT in CA");
  assert.strictEqual(after[0].event, "congestion-avoidance", "ssthresh is where slow start ends");
  assert.strictEqual(wide.ssthresh, 32, "no loss: ssthresh is untouched");

  // --- Reno on a real bottleneck ------------------------------------------------------------
  const reno = new Reno({ bottleneck: new Bottleneck(8), initialWindow: 1, initialSsthresh: 64 });
  const renoHistory = reno.run(60);
  for (const step of renoHistory) {
    assert.strictEqual(step.offered, Math.floor(step.before), "the sender offers its whole window");
    assert.strictEqual(step.delivered + step.dropped, step.offered, "every offered segment is accounted for");
    assert.ok(step.delivered <= 8 && step.delivered >= 0);
    assert.ok(step.cwnd >= 1, "cwnd never falls below one segment");
    if (step.dropped > 0) {
      assert.strictEqual(step.ssthresh, Math.max(Math.floor(step.before / 2), MIN_SSHRESH), "halved on loss");
      assert.strictEqual(step.cwnd, step.event === "timeout" ? 1 : step.ssthresh, "window after the loss");
    }
    if (step.event === "congestion-avoidance") assert.strictEqual(step.cwnd - step.before, 1);
  }
  const renoStats = summarize(renoHistory, 8);
  assert.ok(renoStats.losses >= 5, `Reno must keep hitting the ceiling, got ${renoStats.losses}`);
  assert.strictEqual(renoStats.timeouts, 0, "with 8 segments delivered there are always dup ACKs");
  assert.ok(renoStats.utilization > 0.6 && renoStats.utilization <= 1, `utilisation ${renoStats.utilization}`);
  assert.ok(renoStats.maxWindow <= 16, "the window never runs away");
  const firstLoss = renoHistory.find((step) => step.dropped > 0);
  assert.strictEqual(firstLoss.event, "fast-retransmit");
  assert.strictEqual(firstLoss.before, 16, "slow start overshoots to 16 before the first loss");
  assert.strictEqual(firstLoss.ssthresh, 8);
  assert.strictEqual(firstLoss.cwnd, 8);
  assert.strictEqual(renoHistory[1].event, "slow-start", "and it starts in slow start");

  // --- the timeout path --------------------------------------------------------------------
  const tiny = new Reno({ bottleneck: new Bottleneck(1), initialWindow: 1, initialSsthresh: 64 });
  const tinyHistory = tiny.run(12);
  const tinyTimeouts = tinyHistory.filter((step) => step.event === "timeout");
  assert.ok(tinyTimeouts.length >= 3, "a one-segment link starves the ACK clock");
  for (const step of tinyTimeouts) {
    assert.strictEqual(step.cwnd, 1, "a timeout restarts from one segment");
    assert.strictEqual(step.ssthresh, Math.max(Math.floor(step.before / 2), MIN_SSHRESH));
  }
  assert.strictEqual(tinyHistory[1].before, 2, "cwnd 1 -> delivered 1 -> cwnd 2 -> one segment lost");
  assert.strictEqual(tinyHistory[1].delivered, 1);

  // --- the CUBIC window function ------------------------------------------------------------
  const cubicFunction = new Cubic({ bottleneck: new Bottleneck(8), beta: 0.7, c: 0.4 });
  const wMax = 16;
  const k = Cubic.kFor(wMax, 0.4, 0.7);
  assert.ok(Math.abs(k - Math.cbrt((16 * 0.3) / 0.4)) < 1e-12);
  assert.ok(Math.abs(cubicFunction.windowAt(k, wMax) - wMax) < 1e-9, "W(K) = W_max exactly");
  assert.ok(Math.abs(cubicFunction.windowAt(0, wMax) - 0.7 * wMax) < 1e-9, "W(0) = beta * W_max");
  assert.ok(cubicFunction.windowAt(k + 3, wMax) > wMax, "past K the window probes above W_max");
  assert.ok(cubicFunction.windowAt(0, wMax) < cubicFunction.windowAt(k / 2, wMax), "rising before K");
  assert.ok(cubicFunction.windowAt(k / 2, wMax) < wMax, "but still below the previous maximum");
  const secondDifference = (t) => cubicFunction.windowAt(t + 1, wMax) - 2 * cubicFunction.windowAt(t, wMax)
    + cubicFunction.windowAt(t - 1, wMax);
  assert.ok(secondDifference(k / 2) < 0, "concave in the fast-recovery region");
  assert.ok(secondDifference(k + 2) > 0, "convex in the probing region");
  const plateau = Math.abs(cubicFunction.windowAt(k + 0.5, wMax) - cubicFunction.windowAt(k, wMax));
  const distant = Math.abs(cubicFunction.windowAt(k + 4, wMax) - cubicFunction.windowAt(k, wMax));
  assert.ok(plateau < distant, "the function flattens around K, which is why it is gentle there");
  assert.strictEqual(Cubic.kFor(0, 0.4, 0.7), 0);

  // --- CUBIC on the same bottleneck ---------------------------------------------------------
  const cubic = new Cubic({ bottleneck: new Bottleneck(8), initialWindow: 1, initialSsthresh: 64 });
  const cubicHistory = cubic.run(60);
  for (const step of cubicHistory) {
    assert.strictEqual(step.delivered + step.dropped, step.offered, "accounting");
    assert.ok(step.cwnd >= 1, "cwnd stays at least one segment");
    if (step.dropped > 0) {
      assert.ok(Math.abs(step.cwnd - Math.max(step.before * 0.7, 1)) < 1e-9, "beta = 0.7 decrease");
      assert.ok(step.wMax >= step.before, "W_max is retained or updated, never shrunk");
    }
  }
  const cubicStats = summarize(cubicHistory, 8);
  assert.ok(cubicStats.utilization > 0.6 && cubicStats.utilization <= 1, `utilisation ${cubicStats.utilization}`);
  assert.ok(cubicStats.meanWindow > renoStats.meanWindow,
    `CUBIC keeps a larger average window (${cubicStats.meanWindow} vs ${renoStats.meanWindow})`);
  const postLossFloor = (history) => {
    const first = history.findIndex((step) => step.dropped > 0);
    return Math.min(...history.slice(first).map((step) => step.before));
  };
  assert.ok(postLossFloor(cubicHistory) > postLossFloor(renoHistory),
    `the gentler decrease keeps the post-loss floor higher (${postLossFloor(cubicHistory)} vs ${postLossFloor(renoHistory)})`);
  assert.ok(cubicStats.utilization > renoStats.utilization, "and the link stays busier");
  assert.ok(cubicStats.losses > renoStats.losses, "at the price of more loss events in the same window of time");
  assert.ok(cubicStats.maxWindow <= 20, "though the ceiling is still bounded by the link");

  // --- determinism --------------------------------------------------------------------------
  const rerunReno = new Reno({ bottleneck: new Bottleneck(8), initialWindow: 1, initialSsthresh: 64 }).run(60);
  assert.deepStrictEqual(rerunReno, renoHistory, "identical history on a second run");
  const rerunCubic = new Cubic({ bottleneck: new Bottleneck(8), initialWindow: 1, initialSsthresh: 64 }).run(60);
  assert.deepStrictEqual(rerunCubic, cubicHistory, "identical CUBIC history");
  assert.throws(() => new Cubic({ bottleneck: new Bottleneck(8), beta: 1.5 }), RangeError);
  assert.throws(() => new Cubic({ bottleneck: new Bottleneck(8), c: 0 }), RangeError);

  console.log(`reno util=${renoStats.utilization.toFixed(3)} mean=${renoStats.meanWindow.toFixed(2)} losses=${renoStats.losses}`);
  console.log(`cubic util=${cubicStats.utilization.toFixed(3)} mean=${cubicStats.meanWindow.toFixed(2)} losses=${cubicStats.losses}`);
  console.log("PASS");
}
