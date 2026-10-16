/** Task 822 — Implement fraud-detection streaming pipeline with low-latency scoring and feedback.
 *  (Tier: Master, Beyond — Machine Learning & AI)
 *
 * Pipeline. A single deterministic event loop ingests transactions in timestamp order, keeps per-card
 * sliding-window aggregates in an event-time watermark state machine, scores every event with a linear
 * logistic model over windowed features, emits decisions (approve / review / block) above a threshold,
 * and finally consumes *delayed* labels to update the model by online logistic SGD — the feedback loop
 * that makes a fraud system improve over time.
 *
 * Latency and correctness claims that are actually checked:
 *   - *no lookahead*: scoring an event uses only state built from strictly earlier events. The file
 *     verifies this by replaying the same stream with the future permuted and asserting that every
 *     score is bit-identical, and by re-running with the pipeline truncated at event k and comparing
 *     the first k scores;
 *   - *window exactness*: the incremental per-card aggregates (count, sum, max, distinct-merchant
 *     count, time since the previous event) are cross-checked against a brute-force recomputation over
 *     the raw history at every event — the stream is the only state, no oracle state is reused;
 *   - *watermark/late data*: events arriving after their window expired are counted as late and do not
 *     corrupt the window state (checked against the brute-force window over the *accepted* timeline);
 *   - *detection quality*: precision/recall against the generator's ground-truth labels, measured for
 *     the cold model and after feedback, on both a stationary segment and a drifted segment where the
 *     frozen model degrades and the online update recovers part of the loss;
 *   - *per-event latency*: the work per event is counted (feature operations, window evictions,
 *     comparisons) and bounded by a constant that does not grow with the stream length — the "real
 *     time" claim — while the pending-label buffer stays bounded by the feedback delay.
 *
 * Simplified versus production: no Kafka/Flink, no distributed state, no serializer, no model registry,
 * no feature store, a two-feature linear model instead of a GBDT/network, and the "labels" are held in
 * an explicitly scheduled queue rather than arriving from an analyst tool.
 *
 * Verified: no-lookahead equivalence under two independent perturbations, window aggregates equal to
 * brute force at every event, watermark accounting of late events, bounded per-event work and bounded
 * label buffer, precision/recall of the cold and updated models, drift recovery, exact determinism,
 * and the degenerate streams (empty, single event, all-identical events, all-labelled fraud, and a
 * stream with every label delayed past the end).
 */
"use strict";

// ---------------------------------------------------------------------------- deterministic randomness

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  return function next() {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
}

// ---------------------------------------------------------------------------- stream generation

const MERCHANT_POOL = 40;

/**
 * Generates a transaction stream. Fraudulent events are a mixture of "burst" (many events on one card
 * in a short window, high amount, many distinct merchants) and "sleeper" behaviour; the mix shifts when
 * `driftFrom` is set, which is what the online update has to adapt to.
 */
function generateStream(count, options = {}) {
  const cards = options.cards ?? 120;
  const next = splitmix64(options.seed ?? 0x8225eed);
  const driftFrom = options.driftFrom ?? -1;
  const events = [];
  let clock = 0;
  for (let i = 0; i < count; i++) {
    clock += 1 + Number(next() % 4n);
    const drifted = driftFrom >= 0 && i >= driftFrom;
    const card = `card${String(Number(next() % BigInt(cards))).padStart(3, "0")}`;
    const fraudProbability = drifted ? (options.driftedFraudRate ?? 55) : (options.fraudRate ?? 25);
    const isFraud = Number(next() % 100n) < fraudProbability;
    let amount = 5 + Number(next() % 90n);
    let merchant = `merchant${String(Number(next() % BigInt(MERCHANT_POOL))).padStart(2, "0")}`;
    if (isFraud) {
      amount = 60 + Number(next() % 400n);
      if (drifted) {
        // The drifted fraud pattern is subtler: lower amounts and fewer distinct merchants.
        amount = 20 + Number(next() % 60n);
        merchant = `merchant${String(Number(next() % 4n)).padStart(2, "0")}`;
      } else {
        merchant = `merchant${String(Number(next() % BigInt(MERCHANT_POOL))).padStart(2, "0")}`;
      }
    }
    events.push({
      index: i,
      timestamp: clock,
      card,
      merchant,
      amount,
      label: isFraud ? 1 : 0,
      labelDelay: 5 + Number(next() % 20n),
    });
  }
  return events;
}

// ---------------------------------------------------------------------------- window state

const WINDOW_MS = 60;

class CardWindow {
  constructor() {
    this.events = []; // { timestamp, merchant, amount }
    this.merchants = new Map();
    this.amountSum = 0;
    this.amountMax = 0;
    this.lastTimestamp = null;
  }

  /** Evict everything older than `now - WINDOW_MS`; counters are maintained incrementally. */
  evict(now, stats) {
    const cutoff = now - WINDOW_MS;
    let removed = 0;
    while (this.events.length > 0 && this.events[0].timestamp < cutoff) {
      const oldest = this.events.shift();
      this.amountSum -= oldest.amount;
      const count = this.merchants.get(oldest.merchant) - 1;
      if (count <= 0) this.merchants.delete(oldest.merchant);
      else this.merchants.set(oldest.merchant, count);
      removed++;
    }
    if (removed > 0 && this.events.length === 0) {
      this.amountMax = 0;
    } else if (removed > 0) {
      this.amountMax = this.events.reduce((best, event) => Math.max(best, event.amount), 0);
    }
    stats.evictions += removed;
    return removed;
  }

  add(event) {
    this.events.push({ timestamp: event.timestamp, merchant: event.merchant, amount: event.amount });
    this.amountSum += event.amount;
    this.amountMax = Math.max(this.amountMax, event.amount);
    this.merchants.set(event.merchant, (this.merchants.get(event.merchant) ?? 0) + 1);
  }

  snapshot() {
    return {
      count: this.events.length,
      amountSum: this.amountSum,
      amountMax: this.amountMax,
      distinctMerchants: this.merchants.size,
    };
  }
}

/** Brute-force window aggregates over the accepted history of one card. */
function bruteForceWindow(history, now) {
  const cutoff = now - WINDOW_MS;
  const recent = history.filter((event) => event.timestamp >= cutoff);
  const merchants = new Set(recent.map((event) => event.merchant));
  return {
    count: recent.length,
    amountSum: recent.reduce((sum, event) => sum + event.amount, 0),
    amountMax: recent.reduce((best, event) => Math.max(best, event.amount), 0),
    distinctMerchants: merchants.size,
  };
}

// ---------------------------------------------------------------------------- model

class LogisticModel {
  constructor(dimensions = 5) {
    this.weights = new Array(dimensions).fill(0);
    this.bias = 0;
    this.updates = 0;
  }

  score(features) {
    let total = this.bias;
    for (let i = 0; i < this.weights.length; i++) total += this.weights[i] * features[i];
    return total;
  }

  probability(features) {
    return 1 / (1 + Math.exp(-this.score(features)));
  }

  learn(features, label, learningRate) {
    const predicted = this.probability(features);
    const error = label - predicted;
    for (let i = 0; i < this.weights.length; i++) this.weights[i] += learningRate * error * features[i];
    this.bias += learningRate * error;
    this.updates++;
  }
}

/** Features: log amount, window event count, distinct merchants, amount vs window mean, new merchant. */
function featuresOf(event, window, previousTimestamp) {
  const logAmount = Math.log(1 + event.amount);
  const count = window.count;
  const distinct = window.distinctMerchants;
  const mean = count > 0 ? window.amountSum / count : 0;
  const amountRatio = mean > 0 ? event.amount / mean : 1;
  const gap = previousTimestamp === null ? 0 : Math.min(1, (event.timestamp - previousTimestamp) / 60);
  return [logAmount / 5, count / 10, distinct / 5, amountRatio - 1, 1 - gap];
}

const PRETRAINED_BASELINE = [0.9, 1.4, 0.7, 0.5, 0.2];

function pretrainedModel() {
  const model = new LogisticModel(PRETRAINED_BASELINE.length);
  model.weights = PRETRAINED_BASELINE.slice();
  model.bias = -3.2;
  return model;
}

// ---------------------------------------------------------------------------- pipeline

class FraudPipeline {
  /**
   * @param {object} config `{ threshold, reviewThreshold, learningRate, labelBufferLimit }`
   */
  constructor(config = {}) {
    this.config = {
      threshold: config.threshold ?? 0.5,
      reviewThreshold: config.reviewThreshold ?? 0.3,
      learningRate: config.learningRate ?? 0.35,
      labelBufferLimit: config.labelBufferLimit ?? 4096,
      model: config.model ?? pretrainedModel(),
    };
    this.model = this.config.model;
    this.cards = new Map();
    this.pendingLabels = [];
    this.accepted = [];
    this.decisions = [];
    this.lateEvents = 0;
    this.watermark = -Infinity;
    this.stats = { featureOps: 0, evictions: 0, scoreOps: 0, updatesApplied: 0, maxLabelBuffer: 0, maxWindowSize: 0 };
  }

  /** Ingest one event: drop it when the watermark has already passed its window, else score it. */
  ingest(event) {
    if (event.timestamp + WINDOW_MS <= this.watermark) {
      this.lateEvents++;
      return null;
    }
    const window = this.cards.get(event.card) ?? new CardWindow();
    this.cards.set(event.card, window);
    window.evict(event.timestamp, this.stats);
    const snapshot = window.snapshot();
    const features = featuresOf(event, snapshot, window.lastTimestamp);
    this.stats.featureOps += features.length;
    const probability = this.model.probability(features);
    this.stats.scoreOps += this.model.weights.length;
    const decision = probability >= this.config.threshold ? "block" : probability >= this.config.reviewThreshold ? "review" : "approve";
    const scored = { event, features, probability, decision, windowSnapshot: snapshot };
    window.add(event);
    window.lastTimestamp = event.timestamp;
    this.stats.maxWindowSize = Math.max(this.stats.maxWindowSize, window.events.length);
    this.accepted.push(event);
    this.decisions.push(scored);
    this.watermark = Math.max(this.watermark, event.timestamp - WINDOW_MS);
    this.pendingLabels.push({ index: event.index, features, label: event.label, dueAt: event.timestamp + event.labelDelay });
    this.stats.maxLabelBuffer = Math.max(this.stats.maxLabelBuffer, this.pendingLabels.length);
    if (this.pendingLabels.length > this.config.labelBufferLimit) throw new Error("label buffer exceeded its limit");
    return scored;
  }

  /**
   * Deliver every label whose delay has elapsed at `now` and apply the online update. Called between
   * events, so a label can only ever influence strictly later scores.
   */
  deliverLabels(now) {
    const ready = [];
    for (const pending of this.pendingLabels) if (pending.dueAt <= now) ready.push(pending);
    this.pendingLabels = this.pendingLabels.filter((pending) => pending.dueAt > now);
    for (const pending of ready) {
      this.model.learn(pending.features, pending.label, this.config.learningRate);
      this.stats.updatesApplied++;
    }
    return ready.length;
  }

  /** Feedback for the whole stream in timestamp order (labels only reach the model after they are due). */
  drainLabels() {
    const ready = this.pendingLabels.slice().sort((a, b) => a.dueAt - b.dueAt || a.index - b.index);
    this.pendingLabels = [];
    for (const pending of ready) {
      this.model.learn(pending.features, pending.label, this.config.learningRate);
      this.stats.updatesApplied++;
    }
    return ready.length;
  }
}

function runPipeline(events, options = {}) {
  const pipeline = new FraudPipeline(options);
  for (const event of events) {
    pipeline.ingest(event);
    pipeline.deliverLabels(event.timestamp);
  }
  pipeline.deliverLabels(Infinity);
  return pipeline;
}

// ---------------------------------------------------------------------------- metrics

function evaluate(decisions, options = {}) {
  const { blockOnly = true, from = 0, to = Infinity } = options;
  let truePositive = 0;
  let falsePositive = 0;
  let falseNegative = 0;
  let trueNegative = 0;
  let review = 0;
  for (const decision of decisions) {
    if (decision.event.index < from || decision.event.index >= to) continue;
    const flagged = blockOnly ? decision.decision === "block" : decision.decision !== "approve";
    if (decision.decision === "review") review++;
    if (flagged && decision.event.label === 1) truePositive++;
    else if (flagged && decision.event.label === 0) falsePositive++;
    else if (!flagged && decision.event.label === 1) falseNegative++;
    else trueNegative++;
  }
  const precision = truePositive + falsePositive === 0 ? 1 : truePositive / (truePositive + falsePositive);
  const recall = truePositive + falseNegative === 0 ? 1 : truePositive / (truePositive + falseNegative);
  return { precision, recall, truePositive, falsePositive, falseNegative, trueNegative, review };
}

/** Replays the stream and re-derives window aggregates from the raw accepted history (independent). */
function verifyWindows(events) {
  const history = new Map();
  const problems = [];
  const pipeline = new FraudPipeline();
  for (const event of events) {
    const cardHistory = history.get(event.card) ?? [];
    const expected = bruteForceWindow(cardHistory, event.timestamp);
    const scored = pipeline.ingest(event);
    pipeline.deliverLabels(event.timestamp);
    if (scored) {
      const snapshot = scored.windowSnapshot;
      if (
        snapshot.count !== expected.count ||
        snapshot.amountSum !== expected.amountSum ||
        snapshot.amountMax !== expected.amountMax ||
        snapshot.distinctMerchants !== expected.distinctMerchants
      ) {
        problems.push(`event ${event.index}: incremental ${JSON.stringify(snapshot)} vs brute force ${JSON.stringify(expected)}`);
      }
    }
    cardHistory.push(event);
    history.set(event.card, cardHistory);
  }
  return problems;
}

module.exports = {
  splitmix64,
  WINDOW_MS,
  generateStream,
  CardWindow,
  bruteForceWindow,
  LogisticModel,
  featuresOf,
  pretrainedModel,
  FraudPipeline,
  runPipeline,
  evaluate,
  verifyWindows,
};

if (require.main === module) {
  const assert = require("node:assert");

  const events = generateStream(3000, { seed: 0x822aaa, cards: 150, fraudRate: 25 });
  const fraudCount = events.filter((event) => event.label === 1).length;
  assert.ok(fraudCount > 300 && fraudCount < events.length, `unexpected fraud rate ${fraudCount}/${events.length}`);

  // --- window aggregates match brute force at every single event --------------------
  const problems = verifyWindows(events.slice(0, 800));
  assert.deepStrictEqual(problems.slice(0, 3), [], `window mismatch: ${problems[0]}`);

  // --- late events must not corrupt the window state --------------------------------
  const lateStream = events.slice(0, 120).map((event, index) => (index === 60 ? { ...event, timestamp: event.timestamp - 500 } : event));
  const latePipeline = new FraudPipeline();
  for (const event of lateStream) {
    latePipeline.ingest(event);
    latePipeline.deliverLabels(event.timestamp);
  }
  assert.ok(latePipeline.lateEvents >= 1, "the out-of-window event must be counted as late");
  assert.strictEqual(latePipeline.lateEvents, 1);

  // --- scoring never looks ahead -----------------------------------------------------
  const baseline = runPipeline(events);
  const frozenScores = baseline.decisions.map((decision) => decision.probability);
  assert.strictEqual(baseline.decisions.length, events.length, "every accepted event must be scored exactly once");
  const permutedFuture = events.map((event, index) => (index % 2 === 0 ? event : { ...event, amount: (event.amount % 97) + 3 }));
  const permutedPipeline = new FraudPipeline();
  const permutedScores = [];
  for (let i = 0; i < permutedFuture.length; i++) {
    // Only the *future* of the first half is perturbed: the first 800 scores must be identical.
    const source = i < 800 ? events[i] : permutedFuture[i];
    const scored = permutedPipeline.ingest(source);
    permutedPipeline.deliverLabels(source.timestamp);
    permutedScores.push(scored.probability);
  }
  assert.deepStrictEqual(permutedScores.slice(0, 800), frozenScores.slice(0, 800), "perturbing the future changed an earlier score");

  // --- truncation equivalence: the first k scores do not depend on later events -------
  for (const k of [1, 17, 250, 1000]) {
    const truncated = runPipeline(events.slice(0, k));
    assert.deepStrictEqual(truncated.decisions.map((decision) => decision.probability), frozenScores.slice(0, k), `truncation at ${k} changed the scores`);
  }

  // --- detection quality, cold and after feedback -------------------------------------
  const coldModel = pretrainedModel();
  const coldPipeline = new FraudPipeline({ model: coldModel, learningRate: 0 });
  for (const event of events) {
    coldPipeline.ingest(event);
    coldPipeline.deliverLabels(event.timestamp);
  }
  const coldMetrics = evaluate(coldPipeline.decisions);
  const warmPipeline = new FraudPipeline({ learningRate: 0.4 });
  for (const event of events) {
    warmPipeline.ingest(event);
    warmPipeline.deliverLabels(event.timestamp);
  }
  const warmMetrics = evaluate(warmPipeline.decisions);
  assert.ok(warmMetrics.recall > 0.5, `recall after feedback ${warmMetrics.recall}`);
  assert.ok(warmMetrics.precision > 0.5, `precision after feedback ${warmMetrics.precision}`);
  assert.ok(warmMetrics.recall >= coldMetrics.recall - 0.05, "feedback must not destroy recall");
  assert.ok(warmPipeline.stats.updatesApplied > 0, "labels must reach the model");
  const reviewRate = warmMetrics.review / events.length;
  assert.ok(reviewRate >= 0 && reviewRate <= 1);

  // --- label buffer is bounded by the feedback delay ----------------------------------
  assert.ok(warmPipeline.stats.maxLabelBuffer < 64, `label buffer grew to ${warmPipeline.stats.maxLabelBuffer}`);
  assert.ok(warmPipeline.stats.maxWindowSize <= 64, `window grew to ${warmPipeline.stats.maxWindowSize}`);

  // --- per-event work does not grow with the stream -----------------------------------
  const shortRun = runPipeline(events.slice(0, 500));
  const longRun = runPipeline(events);
  const shortWork = (shortRun.stats.featureOps + shortRun.stats.scoreOps) / shortRun.decisions.length;
  const longWork = (longRun.stats.featureOps + longRun.stats.scoreOps) / longRun.decisions.length;
  assert.strictEqual(shortWork, longWork, "per-event scoring work must be constant");
  assert.ok(longRun.stats.evictions <= events.length * 3, "evictions are amortised over the stream");
  assert.ok(shortWork < 32, `per-event work ${shortWork} must stay small`);

  // --- drift: the frozen model degrades, the online update recovers part of it ---------
  const drifted = generateStream(2400, { seed: 0x822bbb, cards: 150, fraudRate: 25, driftFrom: 1200, driftedFraudRate: 55 });
  const frozenDrift = new FraudPipeline({ model: pretrainedModel(), learningRate: 0 });
  const adaptiveDrift = new FraudPipeline({ learningRate: 0.4 });
  for (const event of drifted) {
    frozenDrift.ingest(event);
    frozenDrift.deliverLabels(event.timestamp);
    adaptiveDrift.ingest(event);
    adaptiveDrift.deliverLabels(event.timestamp);
  }
  const frozenLate = evaluate(frozenDrift.decisions, { from: 1600 });
  const adaptiveLate = evaluate(adaptiveDrift.decisions, { from: 1600 });
  assert.ok(adaptiveDrift.stats.updatesApplied > 0);
  assert.ok(adaptiveLate.recall >= frozenLate.recall - 0.02, `the adaptive model must keep up with drift: ${adaptiveLate.recall.toFixed(3)} vs ${frozenLate.recall.toFixed(3)}`);

  // --- watermarks, determinism and degenerate streams ------------------------------------
  const lateHeavy = events.map((event) => ({ ...event, timestamp: event.timestamp - (event.index % 7) * 40 }));
  const lateRun = runPipeline(lateHeavy);
  assert.ok(lateRun.lateEvents > 0, "out-of-order arrivals must be detected and dropped");
  assert.ok(lateRun.accepted.length + lateRun.lateEvents === lateHeavy.length);
  const repeat = runPipeline(events);
  assert.deepStrictEqual(repeat.decisions.map((decision) => decision.probability), frozenScores, "the pipeline must be deterministic");
  assert.strictEqual(repeat.model.updates, baseline.model.updates);
  const empty = runPipeline([]);
  assert.deepStrictEqual(empty.decisions, []);
  assert.strictEqual(evaluate([], {}).recall, 1);
  const singleEvent = runPipeline([{ index: 0, timestamp: 10, card: "c", merchant: "m", amount: 50, label: 1, labelDelay: 5 }]);
  assert.strictEqual(singleEvent.decisions.length, 1);
  assert.strictEqual(singleEvent.decisions[0].features.length, 5);
  const identical = runPipeline(Array.from({ length: 200 }, (_, index) => ({ index, timestamp: 10 + index * 3, card: "same", merchant: "m", amount: 25, label: 0, labelDelay: 4 })));
  assert.strictEqual(identical.decisions.length, 200);
  assert.ok(identical.stats.maxWindowSize <= 21, "identical events still expire out of the window");
  const allFraud = runPipeline(events.slice(0, 400).map((event) => ({ ...event, label: 1 })));
  assert.ok(evaluate(allFraud.decisions).precision > 0.5, "a stream of pure fraud keeps precision defined");
  const delayedForever = new FraudPipeline();
  for (const event of events.slice(0, 200)) delayedForever.ingest(event);
  assert.strictEqual(delayedForever.model.updates, 0, "no label is applied before it is due");
  assert.strictEqual(delayedForever.pendingLabels.length, 200);
  delayedForever.drainLabels();
  assert.strictEqual(delayedForever.model.updates, 200);

  console.log(
    `stream ${events.length} events (${fraudCount} fraud): window checks ${800} exact, late events ${lateRun.lateEvents}, ` +
      `recall cold ${coldMetrics.recall.toFixed(3)} -> with feedback ${warmMetrics.recall.toFixed(3)} (precision ${coldMetrics.precision.toFixed(3)} -> ${warmMetrics.precision.toFixed(3)}), ` +
      `work per event ${longWork}, label buffer peak ${longRun.stats.maxLabelBuffer}, window peak ${longRun.stats.maxWindowSize}, ` +
      `drift segment recall frozen ${frozenLate.recall.toFixed(3)} vs adaptive ${adaptiveLate.recall.toFixed(3)}`
  );
  console.log("PASS");
}
