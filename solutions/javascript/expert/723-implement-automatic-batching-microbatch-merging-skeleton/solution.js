/** Task 723 - Implement automatic batching (microbatch merging) skeleton. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * An inference server turns a stream of single requests into microbatches, because one GPU
 * step over B sequences is far cheaper than B steps over one. The contract of a dynamic
 * batching scheduler (TensorFlow Serving's batching scheduler, Triton's dynamic batcher) is:
 *
 *   * requests wait in an arrival-ordered queue; a batch leaves as soon as it holds
 *     `maxBatchSize` *arrived* sequences or the oldest arrived request has waited
 *     `maxLatency`, whichever happens first -- so batching can never break the latency SLO;
 *   * `maxTokensPerBatch` caps the work per step, and a sequence longer than the per-request
 *     share is split into several microbatches (that is the "micro" in microbatching) while
 *     every step is padded to a bucket-shaped batch, which is the padding waste accounted
 *     for below.
 *
 * This file is the scheduling half of such a server: a deterministic discrete-event
 * simulator over an integer clock with an abstract step cost, plus the throughput, wait and
 * padding metrics the policy is judged by. Simplified: no real clock or threads, one model
 * replica, no preemption, no per-kernel shapes (a scalar token cost stands in for the
 * hardware model), and pre-submitted traces instead of live demand. What is verified: every
 * request is served exactly once and in order, the clock never runs backwards, a request
 * never starts before it arrives nor waits beyond `maxLatency`, batch size and token budget
 * bounds, agreement of the whole dispatch sequence with an independent reference
 * implementation of the same policy on randomized traces, additive token/waste accounting
 * recomputed from the batches, token-preserving sequence splits, the degenerate cases
 * (empty queue, single request, batch size 1, zero latency window, oversized sequence) and
 * deterministic replay.
 */
"use strict";

const { deepStrictEqual, ok, strictEqual } = require("node:assert");

/** Deterministic xorshift32; integers in [0, bound). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x9e3779b9;
  return function next(bound) {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

class Request {
  constructor(id, arrival, tokens) {
    this.id = id;
    this.arrival = arrival;
    this.tokens = tokens;
    this.start = -1;
    this.end = -1;
  }

  get latency() {
    return this.end - this.arrival;
  }
}

class BatchScheduler {
  constructor({ maxBatchSize, maxLatency, maxTokensPerBatch, stepCost = (tokens) => 1 + tokens }) {
    this.maxBatchSize = maxBatchSize;
    this.maxLatency = maxLatency;
    this.maxTokensPerBatch = maxTokensPerBatch;
    this.stepCost = stepCost;
    this.queue = [];                 // kept sorted by (arrival, submission order)
    this.batches = [];
    this.clock = 0;
    this.paddingWaste = 0;
    this.totalTokens = 0;
    this.waits = [];
  }

  /** Insert by arrival time (stable), the way an arrival-ordered scheduler queue behaves. */
  submit(request) {
    ok(this.queue.every((queued) => queued.id !== request.id), `duplicate request id ${request.id}`);
    let at = this.queue.length;
    while (at > 0 && this.queue[at - 1].arrival > request.arrival) at--;
    this.queue.splice(at, 0, request);
    return request;
  }

  arrived() {
    return this.queue.filter((request) => request.arrival <= this.clock).length;
  }

  /** The head leaves when a full batch of arrived work exists or the oldest request expired. */
  shouldDispatch() {
    if (this.queue.length === 0 || this.queue[0].arrival > this.clock) return false;
    if (this.arrived() >= this.maxBatchSize) return true;
    return this.clock - this.queue[0].arrival >= this.maxLatency;
  }

  /** The next moment at which the decision could change: an arrival or the deadline. */
  nextWake() {
    const deadline = this.queue[0].arrival + this.maxLatency;
    const follower = this.queue.find((request) => request.arrival > this.clock);
    return follower === undefined ? deadline : Math.min(deadline, follower.arrival);
  }

  /** Take up to maxBatchSize arrived sequences whose tokens fit the step budget. */
  formBatch() {
    const batch = [];
    let tokens = 0;
    while (this.queue.length > 0 && batch.length < this.maxBatchSize && this.queue[0].arrival <= this.clock) {
      const head = this.queue[0];
      if (batch.length > 0 && tokens + head.tokens > this.maxTokensPerBatch) break;
      this.queue.shift();
      batch.push(head);
      tokens += head.tokens;
    }
    return { requests: batch, tokens };
  }

  runStep() {
    const { requests, tokens } = this.formBatch();
    const widest = requests.reduce((acc, request) => Math.max(acc, request.tokens), 0);
    this.paddingWaste += widest * requests.length - tokens;
    this.totalTokens += tokens;
    const cost = this.stepCost(tokens);
    for (const request of requests) {
      this.waits.push(this.clock - request.arrival);
      request.start = this.clock;
      request.end = this.clock + cost;
    }
    this.batches.push({ requests, tokens, cost, start: this.clock, end: this.clock + cost });
    this.clock += cost;
    return this.batches[this.batches.length - 1];
  }

  /** Schedule everything that has been submitted until the queue is empty. */
  drain() {
    while (this.queue.length > 0) {
      if (this.queue[0].arrival > this.clock) this.clock = this.queue[0].arrival;   // let it arrive
      if (this.shouldDispatch()) {
        this.runStep();
        continue;
      }
      this.clock = Math.max(this.clock, this.nextWake());
    }
    return this.batches;
  }
}

/** Microbatch split of one sequence into chunks of at most `limit` tokens. */
function splitSequence(request, limit) {
  const chunks = [];
  for (let offset = 0; offset < request.tokens; offset += limit) {
    chunks.push({ id: request.id, from: offset, to: Math.min(request.tokens, offset + limit) });
  }
  return chunks;
}

/** Padding waste of one batch: slots * widest sequence - tokens actually needed. */
function paddingWasteOf(batch) {
  const widest = batch.requests.reduce((acc, request) => Math.max(acc, request.tokens), 0);
  const used = batch.requests.reduce((acc, request) => acc + request.tokens, 0);
  return widest * batch.requests.length - used;
}

/**
 * Independent reference scheduler for a pre-submitted trace. Same policy, different shape:
 * the waiting set is filtered out of the trace on every iteration instead of being a queue,
 * and the clock is advanced explicitly to the next decision point.
 */
function referenceSchedule(trace, { maxBatchSize, maxLatency, maxTokensPerBatch, stepCost }) {
  const waiting = [];
  const batches = [];
  const served = [];
  let clock = trace.length > 0 ? trace[0].arrival : 0;
  let cursor = 0;
  while (cursor < trace.length || waiting.length > 0) {
    while (cursor < trace.length && trace[cursor].arrival <= clock) waiting.push(trace[cursor++]);
    if (waiting.length === 0) {
      clock = trace[cursor].arrival;
      continue;
    }
    const expired = clock - waiting[0].arrival >= maxLatency;
    if (waiting.length < maxBatchSize && !expired) {
      const deadline = waiting[0].arrival + maxLatency;
      clock = cursor < trace.length ? Math.min(deadline, trace[cursor].arrival) : deadline;
      continue;
    }
    const batch = [];
    let tokens = 0;
    while (waiting.length > 0 && batch.length < maxBatchSize) {
      if (batch.length > 0 && tokens + waiting[0].tokens > maxTokensPerBatch) break;
      const request = waiting.shift();
      batch.push(request);
      tokens += request.tokens;
    }
    const cost = stepCost(tokens);
    batches.push({ ids: batch.map((request) => request.id), tokens, cost, start: clock, end: clock + cost });
    for (const request of batch) served.push(request.id);
    clock += cost;
  }
  return { batches, served };
}

/**
 * The SLO the scheduler really guarantees: the time a request spends waiting for a *free*
 * server is at most maxLatency -- longer waits are queueing behind earlier steps, not a
 * formation delay. Returns the formation delay of every request in the run.
 */
function serverBusyOverlap(batches, index, arrival) {
  let busy = 0;
  for (let j = 0; j < index; j++) {
    busy += Math.max(0, batches[j].end - Math.max(batches[j].start, arrival));
  }
  return busy;
}

function formationDelays(batches) {
  return batches.flatMap((batch, index) => batch.requests.map((request) =>
    request.start - request.arrival - serverBusyOverlap(batches, index, request.arrival)));
}

function requestOf(row) {
  return new Request(row.id, row.arrival, row.tokens);
}

module.exports = {
  makeRng, Request, BatchScheduler, splitSequence, referenceSchedule, paddingWasteOf, requestOf,
  formationDelays,
};

if (require.main === module) {
  const rng = makeRng(20260723);

  // -- a hand-checked trace ---------------------------------------------------------------
  const handTrace = [
    { id: 0, arrival: 0, tokens: 8 }, { id: 1, arrival: 1, tokens: 16 }, { id: 2, arrival: 2, tokens: 4 },
    { id: 3, arrival: 3, tokens: 32 }, { id: 4, arrival: 4, tokens: 8 }, { id: 5, arrival: 5, tokens: 16 },
  ];
  const hand = new BatchScheduler({ maxBatchSize: 4, maxLatency: 3, maxTokensPerBatch: 64 });
  handTrace.forEach((row) => hand.submit(requestOf(row)));
  const handBatches = hand.drain();
  deepStrictEqual(handBatches.map((b) => b.requests.map((r) => r.id)), [[0, 1, 2, 3], [4, 5]],
    "three arrivals fill nothing, the fourth fills the batch; the remaining two expire together");
  deepStrictEqual(handBatches.map((b) => b.tokens), [60, 24], "token accounting per batch");
  strictEqual(hand.waits[0], 3, "the first batch leaves exactly when the fourth request arrives");
  ok(handBatches.every((b) => b.requests.every((r) => r.start >= r.arrival)),
    "no request may start before it arrives");
  ok(formationDelays(handBatches).every((delay) => delay >= 0 && delay <= 3),
    "the wait for a free server is bounded by maxLatency, longer waits are queueing");
  const handReference = referenceSchedule(handTrace.map(requestOf),
    { maxBatchSize: 4, maxLatency: 3, maxTokensPerBatch: 64, stepCost: (t) => 1 + t });
  deepStrictEqual(handReference.batches.map((b) => b.ids), [[0, 1, 2, 3], [4, 5]],
    "the reference scheduler agrees on the hand-checked trace");

  // -- invariants on a bigger randomized trace --------------------------------------------
  const trace = [];
  const big = new BatchScheduler({ maxBatchSize: 6, maxLatency: 4, maxTokensPerBatch: 40 });
  let arrival = 0;
  for (let i = 0; i < 120; i++) {
    arrival += rng(3);
    const row = { id: i, arrival, tokens: 1 + rng(20) };
    trace.push(row);
    big.submit(requestOf(row));
  }
  const bigBatches = big.drain();
  deepStrictEqual(bigBatches.flatMap((b) => b.requests.map((r) => r.id)).sort((a, b) => a - b),
    Array.from({ length: 120 }, (_, i) => i), "every request must be served exactly once");
  ok(bigBatches.every((b) => b.requests.length >= 1 && b.requests.length <= 6),
    "batch sizes must stay within [1, maxBatchSize]");
  ok(bigBatches.every((b) => b.tokens <= 40 || b.requests.length === 1),
    "the token budget holds; a lone oversized sequence is the documented exception");
  ok(bigBatches.every((b) => b.requests.every((r) => r.end - r.arrival >= b.cost)),
    "latency is at least one step, since the step itself has to run");
  ok(formationDelays(bigBatches).every((delay) => delay >= 0 && delay <= 4),
    "the wait for a free server must respect the latency window");
  ok(bigBatches.every((b) => b.requests.length === 6 || b.start - b.requests[0].arrival >= 4),
    "a batch under maxBatchSize leaves only when the head's latency window expired");
  ok(bigBatches.every((b, i) => i === 0 || b.start >= bigBatches[i - 1].end), "the clock never runs backwards");
  strictEqual(big.paddingWaste, bigBatches.reduce((acc, b) => acc + paddingWasteOf(b), 0),
    "padding waste must match the independent recomputation");
  strictEqual(big.totalTokens, bigBatches.reduce((acc, b) => acc + b.tokens, 0), "token accounting is additive");
  strictEqual(big.waits.length, 120, "one wait sample per request");
  ok(big.waits.every((wait) => wait >= 0), "waits can never be negative");
  console.log(`120 requests -> ${bigBatches.length} batches, padding waste ${big.paddingWaste} of ${big.totalTokens} tokens`);

  // -- the whole dispatch sequence matches the independent reference ----------------------
  const compared = [];
  for (let trial = 0; trial < 8; trial++) {
    const size = 10 + rng(30);
    const parameters = {
      maxBatchSize: 2 + rng(5), maxLatency: 1 + rng(4),
      maxTokensPerBatch: 20 + rng(40), stepCost: (t) => 1 + t,
    };
    const program = [];
    let clock = 0;
    for (let i = 0; i < size; i++) {
      clock += rng(3);
      program.push({ id: i, arrival: clock, tokens: 1 + rng(12) });
    }
    const left = new BatchScheduler(parameters);
    program.forEach((row) => left.submit(requestOf(row)));
    const leftBatches = left.drain().map((b) => ({ ids: b.requests.map((r) => r.id), start: b.start, tokens: b.tokens }));
    const right = referenceSchedule(program.map(requestOf), parameters);
    deepStrictEqual(leftBatches, right.batches.map((b) => ({ ids: b.ids, start: b.start, tokens: b.tokens })),
      `trial ${trial}: the scheduler must agree with the reference dispatch sequence`);
    deepStrictEqual(right.served.slice().sort((a, b) => a - b), program.map((r) => r.id),
      `trial ${trial}: the reference must serve everyone exactly once`);
    ok(leftBatches.every((b, i) => i === 0 || b.start >= leftBatches[i - 1].start),
      `trial ${trial}: dispatch times are monotone`);
    compared.push(leftBatches.length);
  }
  console.log(`randomized agreement over ${compared.length} traces (${compared.join(",")} batches)`);

  // -- micro-batch splitting never loses a token -----------------------------------------
  for (const limit of [1, 4, 7, 100]) {
    const chunks = splitSequence(new Request(0, 0, 23), limit);
    strictEqual(chunks.length, Math.ceil(23 / limit), `limit ${limit}: chunk count must match ceil(T/limit)`);
    const covered = [];
    for (const chunk of chunks) for (let i = chunk.from; i < chunk.to; i++) covered.push(i);
    deepStrictEqual(covered, Array.from({ length: 23 }, (_, i) => i),
      `limit ${limit}: the split must cover every token exactly once`);
  }
  deepStrictEqual(splitSequence(new Request(1, 0, 0), 5), [], "a zero-token sequence needs no chunk");

  // -- degenerate configurations ----------------------------------------------------------
  const empty = new BatchScheduler({ maxBatchSize: 4, maxLatency: 2, maxTokensPerBatch: 32 });
  deepStrictEqual(empty.drain(), [], "an empty queue produces no batches");
  strictEqual(empty.paddingWaste, 0);
  strictEqual(empty.clock, 0, "an idle scheduler does not advance its clock");

  const solo = new BatchScheduler({ maxBatchSize: 8, maxLatency: 5, maxTokensPerBatch: 100 });
  solo.submit(new Request(0, 0, 7));
  const soloBatches = solo.drain();
  strictEqual(soloBatches.length, 1, "a single request forms one batch");
  strictEqual(soloBatches[0].tokens, 7);
  strictEqual(soloBatches[0].cost, 8, "the abstract step cost is 1 + tokens");
  strictEqual(paddingWasteOf(soloBatches[0]), 0, "a single-sequence batch wastes no padding");

  const serial = new BatchScheduler({ maxBatchSize: 1, maxLatency: 5, maxTokensPerBatch: 100 });
  for (let i = 0; i < 5; i++) serial.submit(new Request(i, i, 4));
  strictEqual(serial.drain().length, 5, "maxBatchSize = 1 degenerates to serial serving");
  ok(serial.batches.every((b) => b.requests.length === 1), "and no batch ever merges");

  const eager = new BatchScheduler({ maxBatchSize: 4, maxLatency: 0, maxTokensPerBatch: 100 });
  for (let i = 0; i < 4; i++) eager.submit(new Request(i, i, 3));
  const eagerBatches = eager.drain();
  strictEqual(eagerBatches[0].requests.length, 1,
    "a zero latency window dispatches immediately, sacrificing the merge for latency");
  deepStrictEqual(eagerBatches.flatMap((b) => b.requests.map((r) => r.id)), [0, 1, 2, 3],
    "even with no window every request is still served, in order");
  ok(eagerBatches.every((b) => b.requests[0].latency >= b.cost), "the step itself still has to run");
  ok(formationDelays(eagerBatches).every((delay) => delay === 0),
    "with a zero window every request leaves the moment a server is free");

  const oversized = new BatchScheduler({ maxBatchSize: 3, maxLatency: 1, maxTokensPerBatch: 4 });
  for (let i = 0; i < 3; i++) oversized.submit(new Request(i, 0, 9));
  const oversizedBatches = oversized.drain();
  ok(oversizedBatches.every((b) => b.requests.length === 1),
    "a sequence larger than the whole token budget travels alone");
  strictEqual(oversizedBatches.length, 3, "and nobody is dropped");

  // -- the token budget forces a split of a wide batch ------------------------------------
  const wide = new BatchScheduler({ maxBatchSize: 4, maxLatency: 1, maxTokensPerBatch: 10 });
  for (let i = 0; i < 3; i++) wide.submit(new Request(i, 0, 6));
  const wideBatches = wide.drain();
  ok(wideBatches.length >= 3, "three 6-token sequences cannot share a 10-token budget");
  ok(wideBatches.every((b) => b.tokens <= 10), "the token budget is respected in every batch");
  deepStrictEqual(wideBatches.flatMap((b) => b.requests.map((r) => r.id)).sort(), [0, 1, 2],
    "nobody is dropped when the batch has to be split");

  // -- determinism -------------------------------------------------------------------------
  const replay = () => {
    const local = makeRng(1234);
    const run = new BatchScheduler({ maxBatchSize: 4, maxLatency: 3, maxTokensPerBatch: 30 });
    let time = 0;
    for (let i = 0; i < 40; i++) {
      time += local(3);
      run.submit(new Request(i, time, 1 + local(9)));
    }
    return run.drain().map((b) => `${b.start}/${b.requests.map((r) => r.id).join("+")}`);
  };
  deepStrictEqual(replay(), replay(), "identical traces must schedule identically");
  console.log("PASS");
}
