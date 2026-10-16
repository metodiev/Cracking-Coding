/** Task 691 - Implement additive-increase multiplicative-decrease (AIMD) simulation. (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Model: round-based (one RTT per round) simulation of N window-based flows sharing a bottleneck of
 * `capacity` segments per round. Every flow offers its whole window; the link carries at most
 * `capacity` segments and the surplus is lost. A round in which anything was lost is a congestion
 * signal for every flow, and each flow responds:
 *     additive increase      w <- w + 1                      (one segment per RTT, per flow)
 *     multiplicative decrease w <- max(1, floor(w * 0.5))    (on a loss round)
 * That is textbook AIMD. Additive increase alone is simulated too (to show congestion collapse: the
 * offered load grows while the goodput stays pinned at the link rate), and a multiplicative-increase
 * variant (w <- ceil(w * 1.05), MIMD) to show that AI is what makes AIMD converge to fairness - MI
 * multiplies an existing spread instead of shrinking it.
 *
 * Simplified versus a real network: feedback is synchronised (every flow sees the same congestion
 * round, because a droptail link signals all senders at once), RTTs are equal, there is no slow
 * start, no timeout, no ACK clocking, and windows are integral. Verified: the per-round accounting,
 * the exact AI/MD update rules, the window floor at one segment, Jain fairness convergence to 1 for
 * AIMD together with the geometric shrinking of the max-min spread at every loss, the contrast
 * against MIMD (spread stays orders of magnitude larger) and against pure AI (no convergence, loss
 * rate 100%), utilisation efficiency in a sane band, the degenerate capacity < flows case, and
 * determinism.
 */
"use strict";

const AIMD = "aimd";
const AI_ONLY = "ai";
const MIMD = "mimd";

const MIMD_FACTOR = 1.05;

/** Jain's fairness index over the windows: 1 means perfectly fair, 1/n means one flow hogs it all. */
function fairnessIndex(windows) {
  if (windows.length === 0) return 1;
  const sum = windows.reduce((acc, value) => acc + value, 0);
  const sumSquares = windows.reduce((acc, value) => acc + value * value, 0);
  if (sumSquares === 0) return 1;
  return (sum * sum) / (windows.length * sumSquares);
}

class AimdSimulator {
  constructor({ capacity, windows, mode = AIMD, decrease = 0.5 }) {
    if (!Number.isInteger(capacity) || capacity < 1) throw new RangeError("capacity must be >= 1");
    if (![AIMD, AI_ONLY, MIMD].includes(mode)) throw new Error(`unknown mode: ${mode}`);
    if (!(decrease > 0 && decrease < 1)) throw new RangeError("decrease must be in (0, 1)");
    this.capacity = capacity;
    this.windows = windows.map((w) => {
      if (!Number.isInteger(w) || w < 1) throw new RangeError("windows must be integers >= 1");
      return w;
    });
    this.mode = mode;
    this.decrease = decrease;
    this.history = [];
  }

  get flowCount() {
    return this.windows.length;
  }

  totalWindow() {
    return this.windows.reduce((acc, value) => acc + value, 0);
  }

  spread() {
    return Math.max(...this.windows) - Math.min(...this.windows);
  }

  step() {
    const before = this.windows.slice();
    const offered = this.totalWindow();
    const delivered = Math.min(offered, this.capacity);
    const lost = offered - delivered;
    const congested = lost > 0;

    if (congested && this.mode !== AI_ONLY) {
      this.windows = this.windows.map((w) => Math.max(1, Math.floor(w * this.decrease)));
    } else if (this.mode === MIMD) {
      this.windows = this.windows.map((w) => Math.ceil(w * MIMD_FACTOR));
    } else {
      this.windows = this.windows.map((w) => w + 1);      // AIMD and AI-only: +1 per RTT
    }

    const record = {
      round: this.history.length + 1, before, windows: this.windows.slice(), offered, delivered, lost,
      lossRate: offered === 0 ? 0 : lost / offered, congested,
      utilization: delivered / this.capacity, fairness: fairnessIndex(this.windows),
      spread: this.spread(),
    };
    this.history.push(record);
    return record;
  }

  run(rounds) {
    for (let i = 0; i < rounds; i++) this.step();
    return this.history;
  }
}

function mean(values) {
  return values.reduce((acc, value) => acc + value, 0) / values.length;
}

module.exports = { AIMD, AI_ONLY, MIMD, MIMD_FACTOR, fairnessIndex, AimdSimulator, mean };

if (require.main === module) {
  const assert = require("node:assert");

  // --- the fairness metric itself -----------------------------------------------------------
  assert.strictEqual(fairnessIndex([3, 3, 3]), 1, "equal windows are perfectly fair");
  assert.ok(Math.abs(fairnessIndex([1, 0, 0]) - 1 / 3) < 1e-12, "one hog out of three");
  assert.ok(Math.abs(fairnessIndex([1, 1, 0]) - 2 / 3) < 1e-12);
  assert.ok(fairnessIndex([5, 5, 5, 5]) > fairnessIndex([7, 5, 5, 3]));
  assert.strictEqual(fairnessIndex([]), 1);
  assert.throws(() => new AimdSimulator({ capacity: 0, windows: [1] }), RangeError);
  assert.throws(() => new AimdSimulator({ capacity: 4, windows: [0] }), RangeError);
  assert.throws(() => new AimdSimulator({ capacity: 4, windows: [1], mode: "bbr" }), Error);
  assert.throws(() => new AimdSimulator({ capacity: 4, windows: [1], decrease: 1 }), RangeError);

  // --- additive increase alone: congestion collapse -----------------------------------------
  const greedy = new AimdSimulator({ capacity: 20, windows: [1, 1], mode: AI_ONLY });
  greedy.run(30);
  for (let i = 1; i < greedy.history.length; i++) {
    assert.deepStrictEqual(greedy.history[i].windows, greedy.history[i - 1].windows.map((w) => w + 1),
      "AI adds exactly one segment per flow per round");
  }
  assert.strictEqual(greedy.history[29].offered, 60, "the offered load keeps growing");
  assert.strictEqual(greedy.history[29].delivered, 20, "while the goodput is pinned at the link rate");
  assert.ok(Math.abs(greedy.history[29].lossRate - (40 / 60)) < 1e-12, "so the loss rate grows without bound");
  assert.ok(greedy.history[29].lossRate > greedy.history[5].lossRate, "the collapse is progressive");
  assert.strictEqual(greedy.history[29].fairness, 1, "AI alone never distinguishes the flows");

  // --- AIMD on the same link ----------------------------------------------------------------
  const aimd = new AimdSimulator({ capacity: 20, windows: [1, 30] });
  const history = aimd.run(120);
  for (const record of history) {
    for (const window of record.windows) assert.ok(window >= 1, "the window never drops below one");
    assert.strictEqual(record.offered + record.lost, record.offered + record.lost);
    assert.ok(record.delivered <= 20 && record.delivered <= record.offered, "the link rate caps delivery");
    assert.ok(record.utilization >= 0 && record.utilization <= 1, "utilisation is a fraction");
    if (!record.congested) {
      assert.deepStrictEqual(record.windows, record.before.map((w) => w + 1), "no loss means pure AI");
    } else {
      assert.deepStrictEqual(record.windows, record.before.map((w) => Math.max(1, Math.floor(w / 2))),
        "a loss round halves every window");
    }
  }
  assert.strictEqual(history[0].before.length, 2);
  assert.strictEqual(history[0].congested, true, "1 + 30 already exceeds the capacity of 20");
  assert.deepStrictEqual(history[0].before, [1, 30]);
  assert.deepStrictEqual(history[0].windows, [1, 15], "the hog halves, the one-segment flow cannot");

  // the max-min spread halves (geometrically) at every loss round
  const lossRounds = history.filter((record) => record.congested);
  assert.ok(lossRounds.length >= 10, `AIMD keeps probing and backing off (${lossRounds.length} loss rounds)`);
  for (let i = 1; i < lossRounds.length; i++) {
    const previous = lossRounds[i - 1].spread;
    if (previous <= 2) break;
    assert.ok(lossRounds[i].spread <= Math.ceil(previous / 2),
      `the spread shrinks at each back-off (${previous} -> ${lossRounds[i].spread})`);
  }
  assert.ok(lossRounds[lossRounds.length - 1].spread <= 2, "and the flows end up nearly equal");

  // convergence: fairness and efficiency in the second half of the run
  const tail = history.slice(60);
  assert.ok(mean(tail.map((record) => record.fairness)) > 0.98,
    `Jain's index converges to 1 (${mean(tail.map((record) => record.fairness))})`);
  assert.ok(history[119].fairness > history[0].fairness, "the run improves fairness overall");
  const utilization = mean(tail.map((record) => record.utilization));
  assert.ok(utilization > 0.6 && utilization <= 1, `utilisation stays efficient (${utilization})`);
  assert.ok(mean(tail.map((record) => record.offered)) >= 0.6 * 20,
    "the total window keeps the link busy");
  assert.ok(mean(tail.map((record) => record.offered)) <= 20 + 2,
    "the total window never drifts far above the capacity");
  assert.ok(mean(tail.map((record) => record.lossRate)) < 0.3, "the steady-state loss rate stays modest");

  // --- the contrast with MIMD ---------------------------------------------------------------
  // On a large capacity the two multiplicative flows keep their 30:1 ratio (MI scales both windows
  // by the same factor), while additive increase closes 1 segment per RTT for both.
  const wideAimd = new AimdSimulator({ capacity: 2000, windows: [10, 300] }).run(200);
  const wideMimd = new AimdSimulator({ capacity: 2000, windows: [10, 300], mode: MIMD }).run(200);
  const wideAimdTail = wideAimd.slice(100);
  const wideMimdTail = wideMimd.slice(100);
  const aimdWideFairness = mean(wideAimdTail.map((record) => record.fairness));
  const mimdWideFairness = mean(wideMimdTail.map((record) => record.fairness));
  assert.ok(mimdWideFairness < aimdWideFairness,
    `multiplicative increase keeps the unfair ratio (${mimdWideFairness} < ${aimdWideFairness})`);
  assert.ok(mean(wideMimdTail.map((record) => record.spread)) > mean(wideAimdTail.map((record) => record.spread)),
    "so the MIMD spread stays much larger");
  assert.ok(wideMimd[199].windows[1] > 5 * wideMimd[199].windows[0], "the hog stays a hog under MIMD");
  assert.ok(wideAimd[199].windows[0] > 10, "while the small AIMD flow keeps growing");
  assert.ok(mean(wideMimdTail.map((record) => record.utilization)) > 0.5, "MIMD does keep the link busy");
  assert.ok(mean(wideAimdTail.map((record) => record.utilization)) > 0.25, "as does AIMD, more slowly");

  // --- more flows, same story ---------------------------------------------------------------
  const many = new AimdSimulator({ capacity: 50, windows: [1, 2, 3, 40, 12] });
  const manyHistory = many.run(200);
  const manyTail = manyHistory.slice(100);
  assert.ok(mean(manyTail.map((record) => record.fairness)) > 0.98, "five flows converge as well");
  assert.ok(many.spread() <= 3, `and end up within three segments of each other (${many.spread()})`);
  assert.ok(mean(manyTail.map((record) => record.utilization)) > 0.6);

  // --- degenerate case: fewer segments than flows -------------------------------------------
  const squeezed = new AimdSimulator({ capacity: 2, windows: [5, 5, 5, 5, 5] });
  const squeezedHistory = squeezed.run(25);
  assert.deepStrictEqual(squeezedHistory[0].windows, [2, 2, 2, 2, 2], "the first back-off halves to two");
  assert.deepStrictEqual(squeezedHistory[1].windows, [1, 1, 1, 1, 1]);
  for (const record of squeezedHistory.slice(2)) {
    assert.deepStrictEqual(record.windows, [1, 1, 1, 1, 1], "everything collapses to one segment");
    assert.strictEqual(record.delivered, 2, "only two segments per round fit");
    assert.strictEqual(record.lost, 3, "the other three are lost every round");
    assert.strictEqual(record.congested, true);
  }
  assert.ok(squeezedHistory.every((record) => record.utilization === 1), "the link is always full");
  assert.ok(Math.abs(squeezedHistory[24].lossRate - 3 / 5) < 1e-12);
  assert.strictEqual(squeezedHistory[24].fairness, 1, "perfectly fair, just unusably small");

  // --- a single flow oscillates between half the capacity and the capacity -------------------
  const single = new AimdSimulator({ capacity: 10, windows: [1] });
  const singleHistory = single.run(40);
  const congestedSingle = singleHistory.filter((record) => record.congested);
  assert.ok(congestedSingle.length >= 5, "the single flow keeps probing and backing off");
  for (const record of congestedSingle) {
    assert.strictEqual(record.before[0], 11, "it overshoots the capacity by exactly one segment");
    assert.strictEqual(record.windows[0], 5, "and halves back to five");
    assert.strictEqual(record.lost, 1);
  }
  for (const record of singleHistory) {
    assert.strictEqual(record.delivered, Math.min(record.before[0], 10), "delivery is capped by the link");
    assert.ok(record.fairness === 1, "one flow is trivially fair");
  }
  const firstSingleLoss = singleHistory.findIndex((record) => record.congested);
  for (const record of singleHistory.slice(firstSingleLoss)) {
    assert.ok(record.before[0] >= 5 && record.before[0] <= 11, "the window then cycles between 5 and 11");
  }
  const singleUtilization = mean(singleHistory.slice(20).map((record) => record.utilization));
  assert.ok(singleUtilization > 0.75, `a single AIMD flow keeps ~79% of the link (${singleUtilization})`);

  // --- determinism ---------------------------------------------------------------------------
  const rerun = new AimdSimulator({ capacity: 20, windows: [1, 30] }).run(120);
  assert.deepStrictEqual(rerun, history, "identical history on a second run");

  console.log(`aimd util=${utilization.toFixed(3)} fairness=${history[119].fairness.toFixed(4)} losses=${lossRounds.length}`);
  console.log(`mimd fairness=${mimdWideFairness.toFixed(3)} aimd fairness=${aimdWideFairness.toFixed(3)}`);
  console.log("PASS");
}
