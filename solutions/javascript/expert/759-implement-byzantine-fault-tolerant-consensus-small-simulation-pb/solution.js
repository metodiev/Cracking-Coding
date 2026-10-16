/** Task 759 - Implement Byzantine fault tolerant consensus small simulation (PBFT). (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
 *
 * A deterministic in-process PBFT: n replicas of which f may be faulty and a message queue as the
 * network (FIFO delivery, no loss, no reordering), so every run is reproducible. A client request
 * reaches every replica; the primary (view mod n) assigns it the next sequence number and sends
 * PRE-PREPARE(view, seq, digest). A replica accepts a pre-prepare only if the view is current, the
 * request is known and its digest matches, and it has not accepted another digest for that
 * (view, seq); it then broadcasts PREPARE. With 2f+1 matching prepares it becomes prepared and
 * broadcasts COMMIT, and with 2f+1 matching commits it executes the request and replies; the client
 * decides only on f+1 identical replies. Timeouts are modelled explicitly: `forceViewChange()`
 * moves to view+1, the new primary re-proposes every not-yet-decided request at a fresh sequence
 * number, and the reply cache keeps each replica from answering twice.
 * Safety is a consequence of the quorum arithmetic: with n=4, f=1 the quorum is 3, so two different
 * digests would need 6 votes out of 4 replicas.
 *
 * What is simplified: no message authentication or signatures, the digest is a 32-bit toy mix
 * (nothing cryptographic is claimed), there is no checkpointing or garbage collection, a view
 * change re-proposes the outstanding requests instead of replaying per-sequence prepared
 * certificates, the reply cache is the only anti-replay mechanism, delivery is FIFO, and timeouts
 * are scripted rather than timed. What is verified: with no faults all four replicas commit
 * identical digests per sequence number and the client decides after f+1 replies; fewer than 2f+1
 * prepares never commit; one silent replica does not stop the other three; a forged prepare is
 * rejected and does not count; an equivocating primary that proposes two different requests for the
 * same slot commits at most one digest, keeps safety, and a view change restores liveness; with
 * f+1 = 2 faulty replicas liveness is (correctly) lost while safety still holds; duplicate requests
 * execute once per replica; malformed messages are rejected; and runs replay identically.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

/** Toy 32-bit digest binding a request header to its payload: claims nothing cryptographic. */
function digestOf(request) {
  const text = `${request.client}|${request.requestId}|${request.op}`;
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = (hash * 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

function assertConfiguration(n, f) {
  if (!Number.isInteger(n) || !Number.isInteger(f) || f < 0 || n < 1) {
    throw new RangeError("n and f must be non-negative integers with n >= 1");
  }
  if (n < 3 * f + 1) throw new RangeError(`PBFT needs n >= 3f + 1 (got n=${n}, f=${f})`);
  return { n, f, quorum: 2 * f + 1 };
}

const slotKey = (view, seq) => `${view}#${seq}`;
const voteKey = (view, seq, digest) => `${view}#${seq}#${digest}`;

class Replica {
  constructor(id, config, faulty = "none") {
    this.id = id;
    this.n = config.n;
    this.f = config.f;
    this.quorum = config.quorum;
    this.faulty = faulty;
    this.view = 0;
    this.requests = new Map();      // requestId -> request payload
    this.slots = new Map();         // view#seq -> {digest, requestId, request}
    this.prepares = new Map();      // view#seq#digest -> Set(voter)
    this.commits = new Map();
    this.prepared = new Set();
    this.executed = new Map();      // requestId -> digest, the reply cache
    this.committed = [];            // {view, seq, digest} in commit order
    this.rejected = [];
  }

  get isPrimary() { return this.id === this.view % this.n; }

  enterView(view) { this.view = view; }

  reject(reason) { this.rejected.push(reason); return []; }

  vote(table, tableKey, voter) {
    if (!table.has(tableKey)) table.set(tableKey, new Set());
    table.get(tableKey).add(voter);
    return table.get(tableKey).size;
  }

  /** Handles one delivered message and returns the messages this replica wants to send. */
  handle(message) {
    if (this.faulty === "silent") return [];
    if (message.type === "request") {
      this.requests.set(message.requestId, message);
      return [];
    }
    if (message.type === "pre-prepare") {
      const request = this.requests.get(message.requestId);
      if (message.view !== this.view) return this.reject("pre-prepare from a stale view");
      if (request === undefined) return this.reject("pre-prepare for an unknown request");
      if (digestOf(request) !== message.digest) return this.reject("pre-prepare digest does not match the payload");
      const slot = slotKey(message.view, message.seq);
      if (this.slots.has(slot) && this.slots.get(slot).digest !== message.digest) {
        return this.reject("conflicting pre-prepare for the same slot");
      }
      this.slots.set(slot, { digest: message.digest, requestId: message.requestId, request });
      return [{ type: "prepare", view: message.view, seq: message.seq, digest: message.digest, from: this.id }];
    }
    if (message.type === "prepare" || message.type === "commit") {
      const slot = slotKey(message.view, message.seq);
      if (message.view !== this.view) return this.reject(`${message.type} from a stale view`);
      const accepted = this.slots.get(slot);
      if (accepted === undefined || accepted.digest !== message.digest) {
        return this.reject(`${message.type} for an unaccepted digest`);
      }
      if (message.type === "prepare") {
        if (this.prepared.has(slot)) return [];
        if (this.vote(this.prepares, voteKey(message.view, message.seq, message.digest), message.from) < this.quorum) return [];
        this.prepared.add(slot);
        return [{ type: "commit", view: message.view, seq: message.seq, digest: message.digest, from: this.id }];
      }
      if (this.vote(this.commits, voteKey(message.view, message.seq, message.digest), message.from) < this.quorum) return [];
      const digest = accepted.digest;
      if (!this.committed.some((entry) => entry.view === message.view && entry.seq === message.seq)) {
        this.committed.push({ view: message.view, seq: message.seq, digest });
      }
      if (this.executed.has(accepted.requestId)) return [];   // the reply cache keeps replies unique
      this.executed.set(accepted.requestId, digest);
      return [{ type: "reply", from: this.id, client: accepted.request.client, requestId: accepted.requestId, digest }];
    }
    return this.reject(`unknown message type '${message.type}'`);
  }
}
class Engine {
  constructor({ n, f, faults = {} }) {
    this.config = assertConfiguration(n, f);
    this.replicas = Array.from({ length: n }, (_, id) => new Replica(id, this.config, faults[id] ?? "none"));
    this.queue = [];
    this.view = 0;
    this.nextSeq = 1;
    this.requests = new Map();
    this.proposed = new Map();
    this.replies = new Map();       // requestId -> Map(digest -> Set(replica))
    this.decisions = new Map();
    this.conflicts = [];
    this.delivered = 0;
  }
  get primary() { return this.view % this.config.n; }
  enqueue(message) { this.queue.push(message); }
  sendToAll(message) { for (const replica of this.replicas) this.enqueue({ ...message, to: replica.id }); }
  /** Delivers a client request to every replica and, unless suppressed, lets the primary propose. */
  submit(client, requestId, op, { autoPropose = true } = {}) {
    const request = { client, requestId, op };
    this.requests.set(requestId, request);
    this.sendToAll({ type: "request", ...request });
    if (autoPropose) this.propose(request);
  }
  propose(request, seq = null) {
    if (this.proposed.has(request.requestId)) return null;
    const assigned = seq === null ? this.nextSeq++ : seq;
    this.proposed.set(request.requestId, assigned);
    this.sendToAll({
      type: "pre-prepare", view: this.view, seq: assigned, digest: digestOf(request),
      requestId: request.requestId, from: this.primary,
    });
    return assigned;
  }
  recordReply(message) {
    if (!this.replies.has(message.requestId)) this.replies.set(message.requestId, new Map());
    const byDigest = this.replies.get(message.requestId);
    if (!byDigest.has(message.digest)) byDigest.set(message.digest, new Set());
    byDigest.get(message.digest).add(message.from);
    if (byDigest.get(message.digest).size >= this.config.f + 1) {
      const decided = this.decisions.get(message.requestId);
      if (decided !== undefined && decided !== message.digest) {
        this.conflicts.push({ requestId: message.requestId, was: decided, now: message.digest });
      }
      this.decisions.set(message.requestId, message.digest);
    }
  }

  pump(maxSteps = 20000) {
    let steps = 0;
    while (this.queue.length > 0 && steps < maxSteps) {
      const { to, ...message } = this.queue.shift();
      steps++;
      this.delivered++;
      for (const outgoing of this.replicas[to].handle(message)) {
        if (outgoing.type === "reply") this.recordReply(outgoing);
        else this.sendToAll(outgoing);
      }
    }
    return steps;
  }

  /** Scripted timeout: the next view's primary re-proposes every request that is not decided. */
  forceViewChange() {
    this.view++;
    for (const replica of this.replicas) replica.enterView(this.view);
    // sequence numbers are never reused across views, so start above everything seen so far
    const used = Math.max(0, ...this.replicas.flatMap((replica) =>
      [...replica.slots.keys()].map((slot) => Number(slot.split("#")[1]))));
    this.nextSeq = Math.max(this.nextSeq, used + 1);
    for (const [requestId, request] of this.requests) {
      if (this.decisions.has(requestId)) continue;
      this.proposed.delete(requestId);
      this.propose(request);
    }
    return this.view;
  }

  /** Safety: no sequence number may be committed with two different digests. */
  safety() {
    const bySeq = new Map();
    for (const replica of this.replicas) {
      for (const entry of replica.committed) {
        if (!bySeq.has(entry.seq)) bySeq.set(entry.seq, new Set());
        bySeq.get(entry.seq).add(entry.digest);
      }
    }
    const violations = [...bySeq].filter(([, digests]) => digests.size > 1);
    return { ok: violations.length === 0, violations, bySeq };
  }

  transcript() {
    return this.replicas.map((replica) => `${replica.id}:${replica.committed.map((entry) => `${entry.seq}=${entry.digest}`).join(",")}`).join("|");
  }
}

function scenario(overrides = {}) {
  return new Engine({ n: 4, f: 1, ...overrides });
}

module.exports = { digestOf, assertConfiguration, Replica, Engine, scenario };

if (require.main === module) {
  // 1. configuration validation
  deepStrictEqual(assertConfiguration(4, 1), { n: 4, f: 1, quorum: 3 });
  deepStrictEqual(assertConfiguration(1, 0), { n: 1, f: 0, quorum: 1 });
  throws(() => assertConfiguration(3, 1), RangeError);
  throws(() => assertConfiguration(4, -1), RangeError);

  // 2. fault-free run: every replica commits the same digest per sequence number
  const faultFree = scenario();
  for (let request = 0; request < 3; request++) faultFree.submit("client1", `r${request}`, `op${request}`);
  faultFree.pump();
  strictEqual(faultFree.safety().ok, true, "no conflicting commits");
  deepStrictEqual([...faultFree.safety().bySeq.keys()], [1, 2, 3], "one sequence number per request");
  for (const [seq, digests] of faultFree.safety().bySeq) strictEqual(digests.size, 1, `sequence ${seq} has a single digest`);
  for (const replica of faultFree.replicas) strictEqual(replica.committed.length, 3, `replica ${replica.id} committed everything`);
  deepStrictEqual([...faultFree.decisions.keys()], ["r0", "r1", "r2"], "the client decided all three requests");
  deepStrictEqual(faultFree.conflicts, []);

  // 3. below the prepare quorum nothing commits
  const lonely = scenario();
  lonely.submit("client1", "r0", "op0", { autoPropose: false });
  const lonelyRequest = lonely.requests.get("r0");
  for (const id of [1, 2]) lonely.enqueue({ type: "pre-prepare", view: 0, seq: 1, digest: digestOf(lonelyRequest), requestId: "r0", from: 0, to: id });
  lonely.pump();
  ok(lonely.replicas.every((replica) => replica.committed.length === 0), "2 of 3 votes is not a quorum");
  strictEqual(lonely.decisions.size, 0, "the client must not decide");
  strictEqual(lonely.safety().ok, true);

  // 4. one silent replica: the remaining three still commit and the client decides
  const silent = scenario({ faults: { 3: "silent" } });
  silent.submit("client1", "r0", "transfer");
  silent.pump();
  strictEqual(silent.decisions.get("r0"), digestOf({ client: "client1", requestId: "r0", op: "transfer" }));
  deepStrictEqual(silent.replicas.slice(0, 3).map((replica) => replica.committed.length), [1, 1, 1]);
  strictEqual(silent.replicas[3].committed.length, 0, "the silent replica commits nothing");
  strictEqual(silent.safety().ok, true);

  // 5. forged messages are rejected and never counted towards a quorum
  const forged = scenario();
  forged.submit("client1", "r0", "op0", { autoPropose: false });
  const request = forged.requests.get("r0");
  forged.enqueue({ type: "pre-prepare", view: 0, seq: 1, digest: "deadbeef", requestId: "r0", from: 0, to: 1 });
  forged.enqueue({ type: "prepare", view: 0, seq: 1, digest: "deadbeef", from: 2, to: 0 });
  forged.enqueue({ type: "prepare", view: 9, seq: 1, digest: digestOf(request), from: 2, to: 0 });
  forged.propose(request, 1);
  forged.pump();
  strictEqual(forged.decisions.get("r0"), digestOf(request), "the honest proposal still commits");
  const rejections = forged.replicas.reduce((total, replica) => total + replica.rejected.length, 0);
  ok(rejections >= 3, `forged and stale messages must be rejected (${rejections})`);
  strictEqual(forged.safety().bySeq.get(1).size, 1, "only one digest was committed for sequence 1");

  // 6. equivocating primary: two different requests offered for the same slot
  const equivocating = scenario();
  equivocating.submit("client1", "a", "opA", { autoPropose: false });
  equivocating.submit("client1", "b", "opB", { autoPropose: false });
  const requestA = equivocating.requests.get("a");
  const requestB = equivocating.requests.get("b");
  for (const id of [0, 1, 2]) equivocating.enqueue({ type: "pre-prepare", view: 0, seq: 1, digest: digestOf(requestA), requestId: "a", from: 0, to: id });
  equivocating.enqueue({ type: "pre-prepare", view: 0, seq: 1, digest: digestOf(requestB), requestId: "b", from: 0, to: 3 });
  equivocating.pump();
  strictEqual(equivocating.decisions.get("a"), digestOf(requestA), "the three-replica quorum decides A");
  strictEqual(equivocating.decisions.has("b"), false, "B never gathers a quorum");
  strictEqual(equivocating.safety().bySeq.get(1).size, 1, "sequence 1 carries exactly one digest");
  strictEqual(equivocating.replicas[3].committed.filter((entry) => entry.seq === 1).length, 0, "replica 3 commits nothing at sequence 1");
  deepStrictEqual(equivocating.conflicts, []);
  // 6b. the view change restores liveness for the request that had no quorum
  strictEqual(equivocating.forceViewChange(), 1);
  equivocating.pump();
  strictEqual(equivocating.decisions.get("b"), digestOf(requestB), "the new primary commits B in view 1");
  const afterViewChange = equivocating.safety();
  ok(afterViewChange.ok, "safety holds across the view change");
  deepStrictEqual([...afterViewChange.bySeq.get(2)], [digestOf(requestB)]);
  ok(equivocating.replicas.slice(0, 3).every((replica) => replica.executed.size === 2), "each correct replica executed both requests once");

  // 7. f+1 faulty replicas: liveness is lost, safety is not
  const tooManyFaults = scenario({ faults: { 2: "silent", 3: "silent" } });
  tooManyFaults.submit("client1", "r0", "op0");
  tooManyFaults.pump();
  strictEqual(tooManyFaults.decisions.size, 0, "with 2 faulty replicas of 4 there is no quorum");
  ok(tooManyFaults.replicas.every((replica) => replica.committed.length === 0));
  strictEqual(tooManyFaults.safety().ok, true, "safety is never given up");
  tooManyFaults.forceViewChange();
  tooManyFaults.pump();
  strictEqual(tooManyFaults.decisions.size, 0, "a view change cannot invent replicas");
  strictEqual(tooManyFaults.safety().ok, true);

  // 8. a duplicate request is executed once and answered once per replica
  const duplicate = scenario();
  duplicate.submit("client1", "r0", "op0");
  duplicate.submit("client1", "r0", "op0");
  duplicate.pump();
  strictEqual(duplicate.decisions.get("r0"), digestOf({ client: "client1", requestId: "r0", op: "op0" }));
  deepStrictEqual([...duplicate.safety().bySeq.keys()], [1], "the duplicate did not get a second sequence number");
  for (const replica of duplicate.replicas) strictEqual(replica.executed.size, 1, `replica ${replica.id} answered once`);

  // 9. determinism: the same scripted run replays identically
  const replay = () => {
    const engine = scenario({ faults: { 3: "silent" } });
    engine.submit("client1", "r0", "op0");
    engine.submit("client1", "r1", "op1");
    engine.pump();
    engine.forceViewChange();
    engine.pump();
    return engine.transcript();
  };
  strictEqual(replay(), replay(), "identical scripts must replay identically");

  // 10. boundary: a single replica reaches every quorum by itself
  const single = new Engine({ n: 1, f: 0 });
  single.submit("client1", "r0", "op0");
  single.pump();
  strictEqual(single.decisions.get("r0"), digestOf({ client: "client1", requestId: "r0", op: "op0" }));
  strictEqual(single.replicas[0].executed.size, 1);

  console.log(
    `quorum=${faultFree.config.quorum} delivered=${faultFree.delivered + equivocating.delivered + silent.delivered} ` +
    `decisions=${faultFree.decisions.size + equivocating.decisions.size} rejections=${rejections} safe=true`
  );
  console.log("PASS");
}
