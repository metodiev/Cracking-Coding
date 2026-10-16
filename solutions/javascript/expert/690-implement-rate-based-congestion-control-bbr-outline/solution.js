/** Task 690 - Implement rate-based congestion control (BBR outline). (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Model: a scaled-down but faithful BBR-style rate-based congestion controller. Where Reno/CUBIC
 * treat loss as the congestion signal, BBR continuously estimates the *bottleneck bandwidth* and
 * the *round-trip propagation delay* and paces packets at `pacingGain * maxBw` while never letting
 * more than `cwndGain * BDP` bytes be in flight, with BDP = maxBw * minRtt.
 *
 * The state machine follows the BBR draft:
 *   STARTUP    pacingGain 2.89, cwndGain 2   - ramp until the bandwidth estimate stops growing
 *                                              (three rounds without 25% growth);
 *   DRAIN      pacingGain 1/2.89             - drain the queue STARTUP built, until inflight <= BDP;
 *   PROBE_BW   pacing gain cycle             - [1.25, 0.75, 1, 1, 1, 1, 1, 1] to probe and yield;
 *   PROBE_RTT  cwnd = 4 packets              - periodically empty the pipe to re-measure minRtt.
 *
 * Network model: one droptail bottleneck link of `linkRate` packets per round, a queue of
 * `queueCapacity` packets, and a propagation delay of `propRtt` rounds. Accepted packets travel the
 * propagation delay in a delay line, then queue, then are served at the link rate; a packet that
 * finds neither queue space nor service capacity is dropped. The ACK rate of a round is exactly the
 * service rate of `propRtt` rounds earlier, so maxBw is measured from ACKs (entries never
 * inflate the estimate) and rtt = propRtt + queueing delay.
 *
 * Simplified versus the draft: round granularity with one measurement per round (no per-packet
 * pacing, no ACK aggregation), a windowed max/min filter of 10 rounds instead of the full 10-second
 * windowed filters, no BBRv2 loss/ECN response, one flow, infinite receiver window, and PROBE_RTT is
 * triggered on a fixed round interval instead of a timer. Verified: the transition order of the state
 * machine, the exact gain constants and the pacing / cwnd formulas, the PROBE_BW gain cycle, ACK rate
 * never exceeding the link rate, inflight never exceeding the cwnd target, the bandwidth estimate
 * converging to the true link rate and BDP to linkRate * propRtt, DRAIN shrinking the queue, no drops
 * once BBR is in steady state, PROBE_RTT forcing cwnd down to 4 packets, and determinism.
 */
"use strict";

const STARTUP_GAIN = 2.89;
const DRAIN_GAIN = 1 / STARTUP_GAIN;
const CWND_GAIN = 2;
const MIN_CWND = 4;                                   // packets, the PROBE_RTT window
const PROBE_BW_GAINS = [1.25, 0.75, 1, 1, 1, 1, 1, 1];
const BW_WINDOW = 10;                                 // rounds kept by the max-bandwidth filter
const RTT_WINDOW = 10;                                // rounds kept by the min-rtt filter
const STARTUP_GROWTH = 1.25;                          // 25% growth keeps STARTUP alive
const STARTUP_PATIENCE = 3;                           // rounds without growth before DRAIN

class BottleneckLink {
  constructor({ linkRate, queueCapacity, propRtt }) {
    if (!(linkRate >= 1) || !(queueCapacity >= 1) || !(propRtt >= 1)) {
      throw new RangeError("linkRate, queueCapacity and propRtt must all be >= 1");
    }
    this.linkRate = linkRate;
    this.queueCapacity = queueCapacity;
    this.propRtt = propRtt;
    this.queue = 0;
    this.slots = new Array(propRtt).fill(0);         // delay line: packets in flight
    this.cursor = 0;
  }

  /**
   * One round. Packets offered now occupy a propagation slot; the packets that filled the slot
   * `propRtt` rounds ago arrive at the droptail queue, where anything that does not fit is dropped.
   * The queue is then served at the link rate; the served packets are the ACKs of this round.
   */
  step(offered) {
    const arriving = this.slots[this.cursor];
    this.slots[this.cursor] = offered;
    this.cursor = (this.cursor + 1) % this.propRtt;

    const fits = Math.min(arriving, this.queueCapacity - this.queue);
    const dropped = arriving - fits;
    this.queue += fits;
    const acked = Math.min(this.queue, this.linkRate);
    this.queue -= acked;

    const inflight = this.queue + this.slots.reduce((acc, value) => acc + value, 0);
    return {
      dropped, acked, queue: this.queue, inflight,
      rtt: this.propRtt + this.queue / this.linkRate,
    };
  }
}

class Bbr {
  constructor({ link, initialWindow = 10, probeRttInterval = 60, maxRounds = 200 }) {
    this.link = link;
    this.initialWindow = initialWindow;
    this.probeRttInterval = probeRttInterval;
    this.maxRounds = maxRounds;
    this.state = "STARTUP";
    this.phase = 0;
    this.maxBw = initialWindow;                       // bootstrap: initial cwnd over one round
    this.minRtt = link.propRtt;
    this.lastProbeRtt = 0;
    this.probeRttDeadline = 0;
    this.history = [];
  }

  /** Windowed max over the ACK-rate history (the bandwidth filter). */
  bandwidthFilter(recentAcks) {
    return recentAcks.length === 0 ? this.initialWindow : Math.max(...recentAcks);
  }

  windowedMinRtt(recentRtts) {
    return Math.min(...recentRtts, this.minRtt);
  }

  run(rounds = this.maxRounds) {
    this.history = [];
    const recentAcks = [];
    const recentRtts = [];
    let growthStreak = 0;
    let lastMaxBw = 0;

    for (let round = 1; round <= rounds; round++) {
      const bdp = this.maxBw * this.minRtt;
      let gain;
      if (this.state === "STARTUP") gain = STARTUP_GAIN;
      else if (this.state === "DRAIN") gain = DRAIN_GAIN;
      else gain = PROBE_BW_GAINS[this.phase];

      const pacingRate = gain * this.maxBw;
      const cwndTarget = this.state === "PROBE_RTT" ? MIN_CWND : Math.max(CWND_GAIN * bdp, MIN_CWND);
      const inflight = this.link.queue + this.link.slots.reduce((acc, value) => acc + value, 0);
      const rateAllowance = Math.floor(pacingRate);
      const cwndAllowance = Math.max(Math.floor(cwndTarget) - inflight, 0);
      const sent = Math.max(Math.min(rateAllowance, cwndAllowance), cwndAllowance > 0 ? 1 : 0);

      const measured = this.link.step(sent);
      if (measured.acked > 0 || recentAcks.length > 0) {
        recentAcks.push(measured.acked);                // skip the warmup rounds of the delay line
        if (recentAcks.length > BW_WINDOW) recentAcks.shift();
      }
      recentRtts.push(measured.rtt);
      if (recentRtts.length > RTT_WINDOW) recentRtts.shift();

      this.maxBw = this.bandwidthFilter(recentAcks);
      const windowMinRtt = this.windowedMinRtt(recentRtts);
      if (windowMinRtt < this.minRtt) this.minRtt = windowMinRtt;

      const before = this.state;
      if (this.state === "STARTUP") {
        growthStreak = this.maxBw > lastMaxBw * STARTUP_GROWTH ? 0 : growthStreak + 1;
        lastMaxBw = this.maxBw;
        if (growthStreak >= STARTUP_PATIENCE) this.state = "DRAIN";
      } else if (this.state === "DRAIN") {
        if (measured.inflight <= this.maxBw * this.minRtt) this.state = "PROBE_BW";
      } else if (this.state === "PROBE_BW") {
        this.phase = (this.phase + 1) % PROBE_BW_GAINS.length;
        if (round - this.lastProbeRtt >= this.probeRttInterval) {
          this.state = "PROBE_RTT";
          this.lastProbeRtt = round;
          this.probeRttDeadline = round + 2 * this.link.propRtt;   // bounded drain, as in the draft
        }
      } else if (this.state === "PROBE_RTT") {
        const windowMin = this.windowedMinRtt(recentRtts);
        this.minRtt = Math.min(this.minRtt, windowMin, measured.rtt);
        if (measured.inflight <= MIN_CWND || round >= this.probeRttDeadline) {
          this.state = "PROBE_BW";
          this.phase = 0;
        }
      }

      this.history.push({
        round, state: before, nextState: this.state, gain, pacingRate, cwndTarget, sent,
        ...measured, bdp: this.maxBw * this.minRtt, maxBw: this.maxBw, minRtt: this.minRtt,
      });
    }
    return this.history;
  }
}

function bdpOf(history) {
  return history[history.length - 1].bdp;
}

module.exports = {
  STARTUP_GAIN, DRAIN_GAIN, CWND_GAIN, MIN_CWND, PROBE_BW_GAINS,
  BottleneckLink, Bbr, bdpOf,
};

if (require.main === module) {
  const assert = require("node:assert");

  const LINK = { linkRate: 10, queueCapacity: 40, propRtt: 5 };
  const trueBdp = LINK.linkRate * LINK.propRtt;

  // --- the bottleneck itself ----------------------------------------------------------------
  const probeLink = new BottleneckLink({ ...LINK });
  const first = probeLink.step(10);
  assert.strictEqual(first.dropped, 0);
  assert.strictEqual(first.acked, 0, "nothing can be served before the propagation delay");
  assert.strictEqual(first.inflight, 10);
  assert.strictEqual(first.rtt, LINK.propRtt, "an empty queue means the pure propagation delay");
  const flooded = new BottleneckLink({ ...LINK });
  let totalDropped = 0;
  for (let i = 0; i < 20; i++) totalDropped += flooded.step(30).dropped;
  assert.ok(totalDropped > 0, "the droptail queue overflows when the sender ignores the link rate");
  assert.ok(flooded.queue <= LINK.queueCapacity, "the queue never exceeds its capacity");
  const served = new BottleneckLink({ ...LINK });
  for (let i = 0; i < 20; i++) served.step(20);
  assert.ok(served.step(0).rtt > LINK.propRtt, "a full queue inflates the RTT");
  assert.throws(() => new BottleneckLink({ linkRate: 0, queueCapacity: 1, propRtt: 1 }), RangeError);
  assert.throws(() => new BottleneckLink({ linkRate: 1, queueCapacity: 1, propRtt: 0 }), RangeError);

  // --- a full BBR run ------------------------------------------------------------------------
  const bbr = new Bbr({ link: new BottleneckLink({ ...LINK }), initialWindow: 10, probeRttInterval: 60 });
  const history = bbr.run(240);
  assert.strictEqual(history.length, 240);

  const states = history.map((step) => step.state);
  const firstDrain = states.indexOf("DRAIN");
  const firstProbeBw = states.indexOf("PROBE_BW");
  const firstProbeRtt = states.indexOf("PROBE_RTT");
  assert.strictEqual(states[0], "STARTUP", "BBR starts in STARTUP");
  assert.ok(firstDrain > 0, "STARTUP ends once the bandwidth estimate stops growing");
  assert.ok(firstProbeBw > firstDrain, "DRAIN comes after STARTUP");
  assert.ok(firstProbeRtt > firstProbeBw, "PROBE_RTT comes after PROBE_BW");
  assert.deepStrictEqual([...new Set(states)].sort(),
    ["DRAIN", "PROBE_BW", "PROBE_RTT", "STARTUP"], "all four states are exercised");
  assert.strictEqual(states[1], "STARTUP", "STARTUP lasts more than a single round");

  // --- gains and the pacing / cwnd formulas --------------------------------------------------
  for (const step of history) {
    if (step.state === "STARTUP") {
      assert.strictEqual(step.gain, STARTUP_GAIN, "STARTUP paces at 2.89x");
      assert.strictEqual(step.cwndTarget, Math.max(CWND_GAIN * step.bdp, MIN_CWND));
    } else if (step.state === "DRAIN") {
      assert.ok(Math.abs(step.gain - DRAIN_GAIN) < 1e-12, "DRAIN paces at 1/2.89x");
    } else if (step.state === "PROBE_BW") {
      assert.ok(PROBE_BW_GAINS.includes(step.gain), "PROBE_BW uses the documented gain table");
    } else {
      assert.strictEqual(step.cwndTarget, MIN_CWND, "PROBE_RTT clamps the window to 4 packets");
    }
    assert.ok(Math.abs(step.pacingRate - step.gain * step.maxBw) < 1e-9, "pacingRate = gain * maxBw");
    assert.ok(Math.abs(step.bdp - step.maxBw * step.minRtt) < 1e-9, "BDP = maxBw * minRtt");
  }

  // the PROBE_BW gain cycle is walked in order
  const probeGains = history.filter((step) => step.state === "PROBE_BW").map((step) => step.gain).slice(0, 16);
  assert.deepStrictEqual(probeGains, [...PROBE_BW_GAINS, ...PROBE_BW_GAINS],
    "the 8-phase gain cycle repeats in order");

  // --- invariants ---------------------------------------------------------------------------
  for (const step of history) {
    assert.ok(step.acked <= LINK.linkRate, "the ACK rate can never exceed the link rate");
    assert.ok(step.dropped >= 0 && step.queue <= LINK.queueCapacity, "droptail accounting");
    assert.ok(step.rtt >= LINK.propRtt - 1e-9, "the RTT can never fall below the propagation delay");
    assert.ok(step.sent <= Math.max(Math.floor(step.cwndTarget), 1), "never send beyond the cwnd target");
    if (step.state !== "STARTUP") {
      assert.ok(step.inflight <= Math.max(CWND_GAIN * step.bdp, MIN_CWND) + 1,
        "inflight stays inside the cwnd target after STARTUP");
    }
  }

  // --- convergence of the estimates ----------------------------------------------------------
  const tail = history.slice(120);
  assert.strictEqual(tail[tail.length - 1].maxBw, LINK.linkRate, "maxBw converges to the link rate");
  for (const step of tail) assert.ok(step.maxBw <= LINK.linkRate, "and never overestimates it");
  assert.strictEqual(tail[tail.length - 1].minRtt, LINK.propRtt, "minRtt converges to the propagation delay");
  assert.ok(Math.abs(bdpOf(history) - trueBdp) <= 1, `BDP converges to ${trueBdp}, got ${bdpOf(history)}`);
  for (const step of tail) assert.ok(Math.abs(step.bdp - trueBdp) <= 5, "the BDP estimate is stable");
  assert.ok(Math.abs(tail[tail.length - 1].cwndTarget - Math.max(CWND_GAIN * trueBdp, MIN_CWND)) < 1e-9,
    "the steady-state window is the cwndGain multiple of the true BDP");

  // --- queue behaviour ----------------------------------------------------------------------
  const startupInflight = Math.max(...history.slice(0, firstDrain).map((step) => step.inflight));
  assert.ok(startupInflight > trueBdp, `STARTUP overshoots two BDPs in flight (${startupInflight})`);
  const overshootQueue = Math.max(...history.slice(0, firstProbeBw).map((step) => step.queue));
  assert.ok(overshootQueue > LINK.linkRate, "the overshoot leaves a standing queue behind");
  const drainEnd = history.findIndex((step) => step.state === "DRAIN" && step.nextState === "PROBE_BW");
  assert.ok(history[drainEnd].inflight <= history[drainEnd].bdp + 1, "DRAIN stops at one BDP in flight");
  assert.ok(history[drainEnd].inflight < startupInflight, "with far fewer packets outstanding");
  assert.ok(history.filter((step) => step.dropped > 0).length > 0, "the overshoot costs packets");
  assert.strictEqual(history.filter((step) => step.dropped > 0 && step.round > 20).length, 0,
    "and nothing is dropped once the loop is under control");

  // --- PROBE_RTT ----------------------------------------------------------------------------
  const probeEntries = history.filter((step, index) => step.state === "PROBE_RTT"
    && (index === 0 || history[index - 1].state !== "PROBE_RTT"));
  assert.strictEqual(probeEntries.length, 3, "PROBE_RTT recurs on the interval");
  assert.strictEqual(probeEntries[1].round - probeEntries[0].round, 60);
  assert.strictEqual(probeEntries[2].round - probeEntries[1].round, 60);
  const firstProbeIndex = history.findIndex((step) => step.state === "PROBE_RTT");
  const firstProbeSpan = [];
  for (let i = firstProbeIndex; i < history.length && history[i].state === "PROBE_RTT"; i++) {
    firstProbeSpan.push(history[i]);
  }
  assert.strictEqual(history[firstProbeIndex - 1].nextState, "PROBE_RTT", "PROBE_BW hands over to PROBE_RTT");
  assert.strictEqual(firstProbeSpan[0].cwndTarget, MIN_CWND, "the window drops to 4 packets");
  for (const step of firstProbeSpan) {
    assert.strictEqual(step.cwndTarget, MIN_CWND, "PROBE_RTT clamps the window to 4 packets");
    assert.ok(step.sent <= MIN_CWND, "and stops sending more than that");
    assert.ok(step.inflight <= history[firstProbeIndex - 1].inflight, "while the pipe drains");
  }
  const drained = firstProbeSpan[firstProbeSpan.length - 1];
  assert.ok(drained.inflight <= MIN_CWND, "and the pipe ends up empty enough for a clean minRtt sample");
  assert.strictEqual(history[drained.round].state, "PROBE_BW", "then PROBE_BW resumes");

  const steady = history.slice(drained.round + 2);
  assert.strictEqual(steady.reduce((acc, step) => acc + step.dropped, 0), 0,
    "BBR does not drop packets in steady state");
  const steadyQueue = Math.max(...steady.map((step) => step.queue));
  assert.ok(steadyQueue <= LINK.linkRate, `the standing queue stays around one round of service (${steadyQueue})`);
  assert.ok(steadyQueue < overshootQueue, "which is far less than the STARTUP overshoot");

  // --- the ESTIMATE really is measurement-driven --------------------------------------------
  const slow = new Bbr({ link: new BottleneckLink({ linkRate: 4, queueCapacity: 40, propRtt: 8 }), initialWindow: 10 });
  const slowHistory = slow.run(200);
  const slowTail = slowHistory[slowHistory.length - 1];
  assert.strictEqual(slowTail.maxBw, 4, "a different link rate produces a different estimate");
  assert.strictEqual(slowTail.minRtt, 8, "and a different propagation delay");
  assert.ok(Math.abs(slowTail.bdp - 32) <= 1, "BDP = 4 * 8");
  assert.notStrictEqual(slowTail.maxBw, LINK.linkRate);

  // --- determinism ---------------------------------------------------------------------------
  const rerun = new Bbr({ link: new BottleneckLink({ ...LINK }), initialWindow: 10, probeRttInterval: 60 });
  assert.deepStrictEqual(rerun.run(240), history, "identical trace on a second run");

  console.log(`states: startup<${firstDrain}, drain<${firstProbeBw}, probeBw, probeRtt=${firstProbeRtt}`);
  console.log(`maxBw=${history[239].maxBw} minRtt=${history[239].minRtt} bdp=${bdpOf(history)} queue=${history[239].queue}`);
  console.log("PASS");
}
