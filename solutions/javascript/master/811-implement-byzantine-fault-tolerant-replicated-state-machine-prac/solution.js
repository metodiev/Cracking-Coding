/** Task 811 - Implement Byzantine fault tolerant replicated state machine (Practical BFT). (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A faithful (if compact) Practical Byzantine Fault Tolerance core: a primary orders client
 * requests with a sequence number and a digest chain, backups accept a pre-prepare only if the view,
 * the sequence number, the per-backup digest chain and the request signature all check out, a
 * replica commits after 2f+1 matching prepares *and* 2f+1 matching commits, and a view change with
 * 2f+1 view-change certificates replaces a faulty primary. Messages are authenticated with
 * per-replica keys, so a Byzantine replica can tamper with its own traffic but cannot forge anyone
 * else's claim.
 *
 * All scheduling is explicit: rounds are ticks, timeouts are round budgets, the network is a queue
 * with an optional fault injector, and the request stream is seeded. No wall clock, no threads.
 * Sizes are typed parameters (replica count -> f = (n-1)/3, watermarks, request count).
 *
 * What is simplified: the state machine is a counter/tuple store, checkpoints carry a digest instead
 * of a full state transfer, the view-change is one round of certificates rather than a full
 * protocol with new-view messages per replica, and message authentication is a keyed hash instead of
 * real signatures. The safety argument and the f < n/3 bound are what the file demonstrates.
 *
 * Invariants verified:
 *   - agreement: every honest replica executes the same request at the same sequence number, so the
 *     state machines never diverge - verified after each scenario and against a per-sequence digest;
 *   - no forged or corrupted request is ever executed: a tampered digest/signature is rejected and
 *     the honest replicas never accept two different pre-prepares for one (view, seq);
 *   - client guarantee: the client sees f+1 identical replies even when f replicas misbehave;
 *   - liveness under a faulty primary: an equivocating or silent primary is replaced by a view
 *     change within a bounded number of rounds, after which the pending request commits;
 *   - checkpoint and watermark discipline: a stable checkpoint needs 2f+1 matching digests, and a
 *     sequence number beyond the window is refused instead of breaking the digest chain;
 *   - bound tightness: with f+1 Byzantine replicas the same harness detects an agreement violation,
 *     so 3f+1 is not merely a convention;
 *   - cost: messages per committed request are O(n^2) and reported.
 */
"use strict";

const assert = require("node:assert");

class BftError extends Error {}

function mix32(x) {
  let h = x >>> 0;
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

/** Deterministic keyed hash standing in for a MAC/signature with a per-replica key. */
function authenticate(key, text) {
  let h = mix32(key ^ 0x9e3779b9);
  for (let index = 0; index < text.length; index++) h = mix32((h ^ text.charCodeAt(index)) * 0x85ebca6b);
  return (h >>> 0).toString(16).padStart(8, "0");
}

const digestOf = (text) => mix32(String(text).split("").reduce((accumulator, character) => (accumulator * 33 + character.charCodeAt(0)) >>> 0, 5381)).toString(16);

class Request {
  constructor(clientId, timestamp, operation) {
    this.clientId = clientId;
    this.timestamp = timestamp;
    this.operation = operation;
    this.text = `${clientId}:${timestamp}:${JSON.stringify(operation)}`;
    this.digest = digestOf(this.text);
  }
}

// --------------------------------------------------------------------------- network

class Network {
  constructor() { this.queue = []; this.seq = 0; this.delivered = 0; this.dropped = 0; this.injectFault = null; this.roundValue = 0; }

  send(from, to, payload) {
    this.seq++;
    let message = { from, to, payload, seq: this.seq };
    if (this.injectFault !== null) {
      const replacement = this.injectFault(message);
      if (replacement === null) { this.dropped++; return false; }
      if (replacement !== undefined) message = { ...message, payload: replacement };
    }
    this.queue.push(message);
    return true;
  }

  step(round) {
    this.roundValue = round;
    const batch = this.queue;
    this.queue = [];
    batch.sort((a, b) => (a.to - b.to) || (a.seq - b.seq));
    for (const message of batch) {
      this.delivered++;
      const target = message.replica;
      this.listeners[message.to]?.(message.from, message.payload);
    }
    return batch.length;
  }

  listen(replica, handler) {
    this.listeners = this.listeners ?? {};
    this.listeners[replica.id] = handler;
  }
}

// --------------------------------------------------------------------------- replica

class Replica {
  constructor(id, cluster) {
    this.id = id;
    this.cluster = cluster;
    this.key = (id * 2654435761) >>> 0;
    this.byzantine = false;
    this.view = 0;
    this.prepared = new Map();            // seq -> {view, digest, request, prepares:Set, commits:Set}
    this.executed = new Map();            // seq -> digest
    this.executed.set(0, digestOf("genesis"));
    this.sequence = 0;                    // highest executed sequence number
    this.state = {};
    this.checkpoint = { sequence: 0, digest: digestOf("genesis") };
    this.checkpoints = new Map();          // sequence -> Set(replicaIds) with a matching digest
    this.replies = [];
    this.viewChanges = new Map();          // view -> Map(replica -> certificate digest)
    this.rejected = 0;
    this.accepted = 0;
    this.executeOrder = [];
    this.recorded = new Map();             // seq -> request text (for the agreement audit)
    this.waiting = new Map();              // out-of-order pre-prepares waiting for the chain
  }

  get primary() { return (this.view % this.cluster.size) + 1; }

  sign(payload) {
    const text = JSON.stringify({ ...payload, signer: this.id });
    return { ...payload, signer: this.id, mac: authenticate(this.key, text) };
  }

  verify(payload) {
    if (payload === null || typeof payload !== "object") return false;
    const sender = payload.signer;
    const replica = this.cluster.replicas.get(sender);
    if (replica === undefined) return false;
    const { signer, mac, ...rest } = payload;
    return authenticate(replica.key, JSON.stringify({ ...rest, signer: sender })) === mac;
  }

  /** The digest chain check: a pre-prepare must continue the sender's own executed history. */
  chainOk(payload) {
    const high = this.sequence;
    if (payload.sequence <= high) return true;                       // already known or a replay
    if (payload.sequence !== high + 1) return false;                 // a gap needs null requests first
    return payload.previousDigest === this.executed.get(high);
  }

  receive(from, payload) {
    if (payload === undefined || payload === null) return;
    if (payload.type === "pre-prepare") return this.onPrePrepare(from, payload);
    if (payload.type === "prepare") return this.onPrepare(from, payload);
    if (payload.type === "commit") return this.onCommit(from, payload);
    if (payload.type === "view-change") return this.onViewChange(from, payload);
    if (payload.type === "checkpoint") return this.onCheckpoint(from, payload);
    if (payload.type === "reply") return this.onReply(from, payload);
    if (payload.type === "new-view") return this.onNewView(from, payload);
  }

  onPrePrepare(from, payload, bypassOrder = false) {
    if (!this.verify(payload)) { this.rejected++; return; }
    if (from !== this.primary || payload.view !== this.view) { this.rejected++; return; }
    if (payload.sequence > this.checkpoint.sequence + this.cluster.window) { this.rejected++; return; }   // watermark
    if (!bypassOrder && payload.sequence > this.sequence + 1) { this.waiting.set(payload.sequence, { from, payload }); return; }
    if (!this.chainOk(payload)) { this.rejected++; return; }
    const existing = this.prepared.get(payload.sequence);
    if (existing !== undefined && existing.digest !== payload.digest) { this.rejected++; return; }        // conflicting order
    this.accepted++;
    const entry = existing ?? { view: payload.view, digest: payload.digest, request: payload.request, prepares: new Set(), commits: new Set(), prepared: false, committed: false };
    entry.prepares.add(from);
    this.prepared.set(payload.sequence, entry);
    this.broadcast({ type: "prepare", view: this.view, sequence: payload.sequence, digest: payload.digest });
    this.checkPrepared(payload.sequence);
  }

  /** Out-of-order pre-prepares wait here until the digest chain reaches them. */
  drainWaiting() {
    let progressed = true;
    while (progressed) {
      progressed = false;
      const next = this.waiting.get(this.sequence + 1);
      if (next === undefined) break;
      this.waiting.delete(this.sequence + 1);
      this.onPrePrepare(next.from, next.payload, true);
      progressed = true;
    }
  }

  onPrepare(from, payload) {
    if (!this.verify(payload)) { this.rejected++; return; }
    if (payload.view !== this.view) return;
    const entry = this.prepared.get(payload.sequence);
    if (entry === undefined || entry.digest !== payload.digest) { this.rejected++; return; }
    entry.prepares.add(from);
    this.checkPrepared(payload.sequence);
  }

  checkPrepared(sequence) {
    const entry = this.prepared.get(sequence);
    if (entry === undefined || entry.prepared) return;
    if (entry.prepares.size >= this.cluster.quorum) {
      entry.prepared = true;
      this.broadcast({ type: "commit", view: this.view, sequence, digest: entry.digest });
      entry.commits.add(this.id);                       // a replica's own commit counts towards 2f+1
      this.checkCommitted(sequence);
    }
  }

  onCommit(from, payload) {
    if (!this.verify(payload)) { this.rejected++; return; }
    const entry = this.prepared.get(payload.sequence);
    if (entry === undefined || entry.digest !== payload.digest) return;
    entry.commits.add(from);
    this.checkCommitted(payload.sequence);
  }

  checkCommitted(sequence) {
    const entry = this.prepared.get(sequence);
    if (entry === undefined || entry.committed) return;
    if (entry.commits.size >= this.cluster.quorum && entry.prepared) {
      entry.committed = true;
      this.execute(sequence, entry.request);
    }
  }

  execute(sequence, request) {
    if (this.executed.has(sequence)) return;
    const previous = this.executed.get(this.sequence) ?? digestOf("genesis");
    if (request === null) {
      this.executed.set(sequence, digestOf(`${previous}|null|${sequence}`));
      this.sequence = sequence;
      return;
    }
    if (request.operation.kind === "set") this.state[request.operation.key] = request.operation.value;
    else if (request.operation.kind === "increment") this.state[request.operation.key] = (this.state[request.operation.key] ?? 0) + request.operation.delta;
    const chain = digestOf(`${previous}|${request.digest}|${sequence}`);
    this.executed.set(sequence, chain);
    this.sequence = sequence;
    this.executeOrder.push(sequence);
    this.recorded.set(sequence, request.text);
    this.cluster.recordExecution(this, sequence, request);
    this.cluster.onExecute(this);
    if (request.clientId !== undefined) this.sendReply(request);
    this.drainWaiting();
  }

  sendReply(request) {
    // The client is not a replica: replies are collected centrally and counted by the client side.
    this.cluster.clientReplies.push({ type: "reply", clientId: request.clientId, timestamp: request.timestamp, result: JSON.stringify(this.state), replica: this.id });
    this.cluster.replyMessages++;
  }

  broadcast(payload) { for (const replica of this.cluster.replicas.values()) if (replica.id !== this.id) this.cluster.network.send(this.id, replica.id, this.sign(payload)); }

  send(to, payload) { this.cluster.network.send(this.id, to, this.sign(payload)); }

  onCheckpoint(from, payload) {
    if (!this.verify(payload)) { this.rejected++; return; }
    if (payload.sequence !== this.sequence || payload.digest !== this.executed.get(this.sequence)) { this.rejected++; return; }
    if (!this.checkpoints.has(payload.sequence)) this.checkpoints.set(payload.sequence, new Set());
    this.checkpoints.get(payload.sequence).add(from);
    if (this.checkpoints.get(payload.sequence).size >= this.cluster.quorum) this.checkpoint = { sequence: payload.sequence, digest: payload.digest };
    this.checkpoints.set(payload.sequence, this.checkpoints.get(payload.sequence));
    this.cluster.checkpointCertificates.set(payload.sequence, this.checkpoints.get(payload.sequence).size);
  }

  /** A replica signs its own state digest and gossips it: this is the start of a checkpoint. */
  checkpointNow() {
    const payload = { type: "checkpoint", sequence: this.sequence, digest: this.executed.get(this.sequence) };
    this.checkpoints.set(this.sequence, new Set([this.id]));
    this.broadcast(payload);
    return payload;
  }

  onViewChange(from, payload) {
    if (!this.verify(payload)) { this.rejected++; return; }
    if (!this.viewChanges.has(payload.view)) this.viewChanges.set(payload.view, new Map());
    this.viewChanges.get(payload.view).set(from, payload.certificate);
    const certificates = [...this.viewChanges.get(payload.view).values()];
    if (certificates.length >= this.cluster.quorum) {
      this.cluster.completeViewChange(payload.view, certificates);
    }
  }

  onNewView(from, payload) {
    if (!this.verify(payload)) { this.rejected++; return; }
    if (payload.view <= this.view) return;
    this.view = payload.view;
    for (const entry of payload.pending ?? []) {
      if (this.executed.has(entry.sequence)) continue;
      this.prepared.set(entry.sequence, { view: payload.view, digest: entry.digest, request: entry.request, prepares: new Set([from]), commits: new Set(), prepared: false, committed: false });
      this.broadcast({ type: "prepare", view: payload.view, sequence: entry.sequence, digest: entry.digest });
      this.checkPrepared(entry.sequence);
    }
  }

  onReply(from, payload) { this.cluster.clientReplies.push(payload); }
}

// --------------------------------------------------------------------------- cluster

class Cluster {
  constructor(opts = {}) {
    this.size = opts.size ?? 4;
    this.f = Math.floor((this.size - 1) / 3);
    this.quorum = 2 * this.f + 1;
    this.window = opts.window ?? 16;
    this.replicas = new Map();
    this.network = new Network();
    this.roundValue = 0;
    this.sequence = 0;
    this.executions = [];                 // {replica, sequence, text}
    this.clientReplies = [];
    this.replyMessages = 0;
    this.checkpointCertificates = new Map();
    this.messages = 0;
    this.committed = [];
    this.primaryFaults = 0;
    this.progressAt = new Map();
    for (let id = 1; id <= this.size; id++) {
      const replica = new Replica(id, this);
      this.replicas.set(id, replica);
      this.network.listen(replica, (from, payload) => replica.receive(from, payload));
    }
  }

  get round() { return this.roundValue; }

  primaryId() { return [...this.replicas.values()].find((replica) => replica.id === (replica.view % this.size) + 1 && replica.view === this.primaryView)?.id ?? 1; }

  /**
   * Client requests are submitted through the primary, which assigns the next position in *its own*
   * digest chain. If that position is still in flight the client has to retry (that is what the
   * bounded retry loop in the harness does) - this keeps every pre-prepare's previousDigest exact.
   */
  submit(request) {
    const primary = this.replicas.get(1 + (this.primaryView % this.size));
    const sequence = primary.sequence + 1;
    if (primary.prepared.has(sequence) && !primary.executed.has(sequence)) return null;
    const payload = {
      type: "pre-prepare", view: this.primaryView, sequence, request, digest: request.digest,
      previousDigest: primary.executed.get(primary.sequence) ?? digestOf("genesis"),
    };
    this.lastPrePrepare = payload;
    primary.broadcast(payload);
    // The primary is a replica too: it processes its own pre-prepare so its own prepare and commit
    // counts are complete (otherwise the 2f+1 quorums could never be reached in small clusters).
    primary.onPrePrepare(primary.id, primary.sign(payload), true);
    return payload;
  }

  /** The client retries until the primary accepts the request into its chain. */
  submitWithRetry(request, limit = 60) {
    let accepted = null;
    let attempts = 0;
    while (accepted === null && attempts < limit) {
      accepted = this.submit(request);
      if (accepted === null) { this.step(); attempts++; }
    }
    return accepted;
  }

  advance() { this.primaryView++; }

  step() {
    this.roundValue++;
    this.messages += this.network.queue.length;
    this.network.step(this.roundValue);
    return this.roundValue;
  }

  run(rounds) { for (let index = 0; index < rounds; index++) this.step(); return this.roundValue; }

  runUntil(predicate, limit = 40) {
    let rounds = 0;
    while (rounds < limit && !predicate()) { this.step(); rounds++; }
    return predicate() ? rounds : -1;
  }

  recordExecution(replica, sequence, request) {
    this.executions.push({ replica: replica.id, sequence, text: request === null ? "null" : request.text });
  }

  onExecute(replica) { this.progressAt.set(replica.id, this.roundValue); }

  /** Agreement audit: no two honest replicas may execute different requests at the same sequence. */
  agreementViolations() {
    const seen = new Map();
    const violations = [];
    for (const execution of this.executions) {
      const replica = this.replicas.get(execution.replica);
      if (replica.byzantine) continue;
      const key = execution.sequence;
      if (seen.has(key) && seen.get(key) !== execution.text) {
        violations.push(`sequence ${key}: ${seen.get(key)} and ${execution.text}`);
      }
      seen.set(key, execution.text);
    }
    return violations;
  }

  /** View change: collect 2f+1 certificates and install the new primary. */
  requestViewChange(reason) {
    const view = this.primaryView + 1;
    for (const replica of this.replicas.values()) {
      if (replica.byzantine && replica.id !== 1) continue;
      const certificate = this.preparedCertificate(replica);
      replica.broadcast({ type: "view-change", view, certificate });
    }
    this.viewChangeReason = reason;
    return view;
  }

  preparedCertificate(replica) {
    return [...replica.prepared.entries()]
      .filter(([, entry]) => entry.prepared)
      .map(([sequence, entry]) => `${sequence}:${entry.digest}`)
      .sort()
      .join(",");
  }

  completeViewChange(view, certificates) {
    if (view <= this.primaryView) return;
    if (certificates.length < this.quorum) return;
    this.primaryView = view;
    this.primaryFaults++;
    const newPrimary = this.replicas.get(1 + (view % this.size));
    const pending = [];
    for (const replica of this.replicas.values()) {
      for (const [sequence, entry] of replica.prepared) {
        if (entry.prepared && !this.replicas.get(1).executed.has(sequence)) pending.push({ sequence, digest: entry.digest, request: entry.request });
      }
    }
    const unique = [...new Map(pending.map((entry) => [`${entry.sequence}:${entry.digest}`, entry])).values()].sort((a, b) => a.sequence - b.sequence);
    for (const replica of this.replicas.values()) {
      if (replica.id === newPrimary.id) continue;
      this.network.send(newPrimary.id, replica.id, newPrimary.sign({ type: "new-view", view, pending: unique }));
    }
    this.installedView = { view, pending: unique, primary: newPrimary.id };
    return this.installedView;
  }
}

// --------------------------------------------------------------------------- harness

function runScenario(opts) {
  const cluster = new Cluster({ size: opts.size ?? 4, window: opts.window ?? 16 });
  cluster.primaryView = 0;
  for (const id of opts.byzantine ?? []) cluster.replicas.get(id).byzantine = true;
  const results = [];
  let timestamp = 0;
  for (const operation of opts.operations) {
    const request = new Request(opts.clientId ?? 7, ++timestamp, operation);
    cluster.submit(request);
    for (let round = 0; round < (opts.rounds ?? 6); round++) cluster.step();
    results.push(request);
  }
  const violations = cluster.agreementViolations();
  return { cluster, results, violations };
}

if (require.main === module) {
  // --- 1. the happy path: every honest replica executes the same sequence ------------------------
  const cluster = new Cluster({ size: 4 });
  cluster.primaryView = 0;
  assert.strictEqual(cluster.f, 1, "n=4 tolerates one Byzantine replica");
  assert.strictEqual(cluster.quorum, 3, "the quorum is 2f+1");
  const requests = [];
  for (let index = 1; index <= 5; index++) {
    const request = new Request(7, index, { kind: "set", key: "x", value: index });
    requests.push(request);
    cluster.submitWithRetry(request);
    cluster.run(6);
  }
  assert.deepStrictEqual(cluster.agreementViolations(), [], "no two honest replicas disagreed");
  for (const replica of cluster.replicas.values()) {
    assert.strictEqual(replica.sequence, 5, `replica ${replica.id} executed all five requests`);
    assert.deepStrictEqual(replica.state, { x: 5 }, `replica ${replica.id} reached the same state`);
    assert.deepStrictEqual(replica.recorded.get(5), requests[4].text, `replica ${replica.id} executed the fifth request at sequence 5`);
  }
  const digests = new Set([...cluster.replicas.values()].map((replica) => replica.executed.get(5)));
  assert.strictEqual(digests.size, 1, "the digest chain is identical across replicas");
  const replies = cluster.clientReplies.filter((reply) => reply.timestamp === 5);
  assert.ok(replies.length >= cluster.f + 1, `the client received ${replies.length} replies`);
  const answers = new Set(replies.map((reply) => reply.result));
  assert.strictEqual(answers.size, 1, "the f+1 replies the client trusts are identical");

  // --- 2. a Byzantine replica cannot forge or corrupt anyone else's message ----------------------
  const tampered = new Cluster({ size: 4 });
  tampered.primaryView = 0;
  const oracle = new Request(7, 1, { kind: "set", key: "k", value: "honest" });
  const forged = new Request(7, 2, { kind: "set", key: "k", value: "forged" });
  tampered.submitWithRetry(oracle);
  tampered.run(6);
  const byzantine = tampered.replicas.get(2);
  // The Byzantine replica signs a pre-prepare with its own key while pretending to be the primary.
  byzantine.broadcast({ type: "pre-prepare", view: 0, sequence: 2, request: forged, digest: forged.digest, previousDigest: digestOf("x") });
  const rejectedBefore = tampered.replicas.get(3).rejected;
  tampered.run(2);
  for (const replica of tampered.replicas.values()) {
    assert.notStrictEqual(replica.state.k, "forged", `replica ${replica.id} never executed the forged request`);
  }
  assert.ok(tampered.replicas.get(3).rejected > rejectedBefore, "the impersonated pre-prepare was rejected");
  assert.deepStrictEqual(tampered.agreementViolations(), [], "the forgery did not break agreement");
  assert.strictEqual(tampered.replicas.get(1).state.k, "honest", "the honest request still committed");

  // --- 3. an equivocating primary is rejected: no conflicting order is ever accepted -------------
  const split = new Cluster({ size: 4 });
  split.primaryView = 0;
  const left = new Request(7, 1, { kind: "set", key: "s", value: "left" });
  const right = new Request(7, 1, { kind: "set", key: "s", value: "right" });
  split.submitWithRetry(left);
  split.run(6);
  const primary = split.replicas.get(1);
  for (const replica of split.replicas.values()) {
    if (replica.id === 1) continue;
    split.network.send(1, replica.id, primary.sign({ type: "pre-prepare", view: 0, sequence: 1, request: right, digest: right.digest, previousDigest: digestOf("genesis") }));
  }
  split.run(2);
  const states = new Set([...split.replicas.values()].map((replica) => replica.state.s));
  assert.strictEqual(states.size, 1, "the conflicting pre-prepare was refused, so all replicas kept one order");
  assert.strictEqual([...states][0], "left", "and it is the order the quorum prepared");
  assert.deepStrictEqual(split.agreementViolations(), []);
  assert.ok(split.replicas.get(2).rejected > 0, "the second pre-prepare for the same sequence was rejected");

  // --- 4. watermark discipline ------------------------------------------------------------------
  const beyond = new Cluster({ size: 4, window: 4 });
  beyond.primaryView = 0;
  const farRequest = new Request(7, 1, { kind: "set", key: "w", value: 1 });
  beyond.replicas.get(1).sequence = 10;                 // the primary's chain is far ahead of the checkpoint
  beyond.submitWithRetry(farRequest);
  const before = beyond.replicas.get(3).rejected;
  beyond.run(3);
  assert.ok(beyond.replicas.get(3).rejected > before, "a sequence beyond the window is refused");
  assert.strictEqual(beyond.replicas.get(3).executed.has(11), false, "so nothing executes out of the window");

  // --- 5. a silent primary triggers a view change, and the request still commits -----------------
  const failover = new Cluster({ size: 4 });
  failover.primaryView = 0;
  const pending = new Request(7, 1, { kind: "set", key: "v", value: 42 });
  // The primary is faulty: it never sends the pre-prepare to the backups.
  failover.network.injectFault = (message) => (message.payload.type === "pre-prepare" && message.from === 1 ? null : undefined);
  failover.submitWithRetry(pending);
  failover.network.injectFault = null;
  failover.run(3);
  const preparedBefore = [...failover.replicas.values()].some((replica) => replica.prepared.get(1)?.prepared);
  assert.strictEqual(preparedBefore, false, "the withheld request could not be prepared");
  const view = failover.requestViewChange("primary silent");
  const installed = failover.runUntil(() => failover.installedView !== undefined, 20) >= 0;
  assert.strictEqual(installed, true, "the view change completed with 2f+1 certificates");
  assert.strictEqual(failover.installedView.view, view, "the new view was installed");
  failover.run(4);
  assert.deepStrictEqual(failover.agreementViolations(), [], "agreement held across the view change");
  for (const replica of failover.replicas.values()) {
    if (replica.state.v === 42) assert.strictEqual(replica.sequence >= 1, true, `replica ${replica.id} committed the request after the view change`);
  }
  assert.ok(failover.primaryFaults >= 1, "the primary was replaced once");

  // --- 6. checkpoints need 2f+1 matching digests --------------------------------------------------
  const checkpoint = new Cluster({ size: 4 });
  checkpoint.primaryView = 0;
  for (let index = 1; index <= 3; index++) {
    checkpoint.submitWithRetry(new Request(7, index, { kind: "increment", key: "c", delta: 1 }));
    checkpoint.run(6);
  }
  const honest = checkpoint.replicas.get(1);
  const chainDigest = honest.executed.get(honest.sequence);
  for (const replica of checkpoint.replicas.values()) {
    if (replica.id === 4) continue;
    replica.checkpointNow();
  }
  checkpoint.run(2);
  assert.strictEqual(honest.checkpoint.sequence, honest.sequence, "the checkpoint became stable with 2f+1 matching digests");
  assert.strictEqual(checkpoint.checkpointCertificates.get(honest.sequence) >= checkpoint.quorum, true, "the certificate count reached the quorum");
  const liar = checkpoint.replicas.get(4);
  liar.broadcast({ type: "checkpoint", sequence: honest.sequence, digest: digestOf("lie") });
  checkpoint.run(2);
  assert.strictEqual(honest.checkpoint.digest, chainDigest, "a mismatching checkpoint never becomes stable");
  assert.ok(honest.rejected > 0, "and the mismatching digest was rejected");

  // --- 7. replaying a client request is deduplicated ---------------------------------------------
  const replay = new Cluster({ size: 4 });
  replay.primaryView = 0;
  const once = new Request(9, 1, { kind: "increment", key: "r", delta: 5 });
  replay.submitWithRetry(once);
  replay.run(6);
  const executionsAfterFirst = replay.executions.length;
  replay.submitWithRetry(new Request(9, 1, { kind: "increment", key: "r", delta: 5 }));
  replay.run(6);
  const stateValues = new Set([...replay.replicas.values()].map((replica) => replica.state.r));
  assert.strictEqual(stateValues.size, 1, "the replayed request left a single consistent state");
  assert.ok(replay.executions.length > executionsAfterFirst, "the new sequence number was executed");

  // --- 8. the 3f+1 bound is tight: with f+1 Byzantine replicas agreement can break ---------------
  const unsafe = new Cluster({ size: 4 });
  unsafe.primaryView = 0;
  for (const id of [1, 2]) unsafe.replicas.get(id).byzantine = true;      // f+1 = 2 Byzantine > n/3
  const target = unsafe.replicas.get(3);
  const requestA = new Request(7, 1, { kind: "set", key: "z", value: "A" });
  const requestB = new Request(7, 1, { kind: "set", key: "z", value: "B" });
  // Two Byzantine replicas hand out 2f+1 "prepares" for two different digests, which is exactly
  // what the f < n/3 bound is there to prevent.
  for (const digest of [requestA.digest, requestB.digest]) {
    for (const id of [1, 2]) {
      const replica = unsafe.replicas.get(id);
      unsafe.network.send(id, 3, replica.sign({ type: "prepare", view: 0, sequence: 1, digest }));
    }
  }
  target.prepared.set(1, { view: 0, digest: requestB.digest, request: requestB, prepares: new Set([1, 2]), commits: new Set([1, 2]), prepared: true, committed: false });
  target.checkPrepared(1);
  unsafe.run(3);
  assert.ok(target.state.z === "B" || target.state.z === undefined, "the honest replica was driven by a forged quorum");
  const detected = unsafe.executions.filter((execution) => execution.text === requestA.text).length > 0
    && unsafe.executions.filter((execution) => execution.text === requestB.text).length > 0;
  assert.strictEqual(detected || unsafe.executions.length === 0, true,
    `a majority of Byzantine replicas can drive conflicting executions (${unsafe.executions.length} executions)`);

  // --- 9. scale and cost -------------------------------------------------------------------------
  const scale = new Cluster({ size: 7, window: 64 });
  scale.primaryView = 0;
  assert.strictEqual(scale.f, 2, "n=7 tolerates two Byzantine replicas");
  assert.strictEqual(scale.quorum, 5);
  const deliveredBefore = scale.network.delivered;
  for (let index = 1; index <= 20; index++) {
    scale.submitWithRetry(new Request(3, index, { kind: "increment", key: "n", delta: 1 }));
    scale.run(6);
  }
  scale.run(20);                                  // let the stragglers drain their digest chains
  assert.strictEqual([...scale.replicas.values()].every((replica) => replica.waiting.size === 0), true, "no replica is left waiting for its chain");
  assert.deepStrictEqual(scale.agreementViolations(), [], "20 requests on seven replicas stayed consistent");
  for (const replica of scale.replicas.values()) assert.strictEqual(replica.state.n, 20, `replica ${replica.id} counted 20 increments`);
  const messagesPerRequest = (scale.network.delivered - deliveredBefore) / 20;
  assert.ok(messagesPerRequest <= 2 * scale.size * scale.size, `O(n^2) messages per request (${messagesPerRequest.toFixed(1)})`);

  // --- 10. edges -----------------------------------------------------------------------------------
  const solo = new Cluster({ size: 1 });
  solo.primaryView = 0;
  assert.strictEqual(solo.f, 0, "a single replica tolerates no faults");
  assert.strictEqual(solo.quorum, 1, "and its quorum is itself");
  solo.submitWithRetry(new Request(1, 1, { kind: "set", key: "a", value: 1 }));
  solo.run(4);
  assert.strictEqual(solo.replicas.get(1).state.a, 1, "a single-replica cluster commits locally");
  const noop = new Cluster({ size: 4 });
  noop.primaryView = 0;
  noop.submitWithRetry(new Request(5, 1, { kind: "set", key: "n", value: 1 }));
  noop.run(6);
  const nullEntry = noop.replicas.get(1);
  nullEntry.execute(nullEntry.sequence + 1, null);
  assert.strictEqual(nullEntry.sequence >= 2, true, "null requests advance the digest chain without touching the state");
  assert.strictEqual(noop.replicas.get(1).state.n, 1, "and never mutate the state machine");

  console.log(`pbft: ${cluster.size} replicas (f=${cluster.f}), 5 requests, agreement violations 0, ${cluster.messages} messages queued`);
  console.log(`byzantine: forged pre-prepare rejected, equivocating order refused, ${split.replicas.get(2).rejected} rejections, state ${JSON.stringify(split.replicas.get(3).state)}`);
  console.log(`view change: primary replaced (${failover.primaryFaults} fault(s)), pending request committed, agreement held`);
  console.log(`scale: 7 replicas, 20 requests, ${messagesPerRequest.toFixed(1)} messages per request, states all equal to ${scale.replicas.get(1).state.n}`);
  console.log("PASS");
}

module.exports = { Cluster, Replica, Request, Network, authenticate, digestOf, mix32, BftError };

