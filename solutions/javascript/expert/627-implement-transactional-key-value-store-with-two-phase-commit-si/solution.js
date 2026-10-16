/** Task 627 - Implement transactional key-value store with two-phase commit simulation. (Tier: Expert, CTCI Beyond — Distributed Systems)
 *
 * Two-phase commit makes a write set that spans several participants atomic: the coordinator asks
 * every participant to *prepare* (validate, take locks, write a prepare record to stable storage),
 * and only if all of them vote yes does it record and broadcast the commit decision. Each
 * participant keeps a versioned store, so a transaction reads the versions visible at its snapshot
 * and a prepare is refused when someone else wrote one of the keys after that snapshot
 * (write-write conflict). Recovery is log-driven: the coordinator's decision log is the single
 * source of truth, so a phase-2 message that never arrived is retried until every participant has
 * applied the decision exactly once.
 *
 * What is simplified: there are no real processes, sockets or clocks. Messages are delivered by
 * explicit calls and "lost" messages are named by the test; a crash is a flag that stops phase 2
 * midway. The checks therefore verify the *protocol logic* - atomicity, vote rules, idempotent
 * recovery, snapshot reads - not real durability. The blocking property of 2PC is modelled, not
 * solved: a participant that voted yes and hears nothing stays prepared (it cannot decide alone),
 * which is exactly why real systems add a termination protocol or three-phase commit.
 *
 * Verified invariants: (1) atomicity - after every run and every recovery either all participants
 * applied the transaction or none did; (2) a NO vote means nobody applies anything; (3) prepared
 * data is invisible to readers and a prepared participant refuses a second transaction; (4) recovery
 * from the decision log is idempotent - replaying it never applies a transaction twice; (5) the
 * deliberately buggy coordinator (commit if *any* participant prepared) breaks atomicity on the very
 * same scenarios, so the atomicity check has teeth.
 */
"use strict";

const IDLE = "idle";
const PREPARED = "prepared";

/** One participant: a versioned key/value store, a decision log and a staged write set. */
class Participant {
  constructor(name, entries = {}) {
    this.name = name;
    this.history = new Map(); // key -> [{ version, value }]
    for (const [key, value] of Object.entries(entries)) this.history.set(key, [{ version: 0, value }]);
    this.version = 0;
    this.log = []; // { txId, vote } | { txId, decision }
    this.applied = [];
    this.staged = null;
    this.state = IDLE;
  }

  /** Snapshot read: the newest version at or below `snapshot`. */
  read(key, snapshot = this.version) {
    const records = this.history.get(key) || [];
    let found;
    for (const record of records) if (record.version <= snapshot) found = record;
    return found === undefined ? undefined : found.value;
  }

  /** Phase 1. A write-write conflict or an outstanding transaction forces a NO vote. */
  prepare(txId, writes, snapshot) {
    if (this.state === PREPARED) {
      this.log.push({ txId, vote: "no", reason: "busy" });
      return { vote: "no", reason: "participant is already prepared" };
    }
    for (const key of writes.keys()) {
      const records = this.history.get(key) || [];
      if (records.length > 0 && records[records.length - 1].version > snapshot) {
        this.log.push({ txId, vote: "no", reason: "conflict" });
        return { vote: "no", reason: `write-write conflict on ${key}` };
      }
    }
    this.staged = { txId, writes, snapshot };
    this.state = PREPARED;
    this.log.push({ txId, vote: "yes" });
    return { vote: "yes" };
  }

  /** Phase 2. Applying the same decision twice is a no-op, which is what makes recovery safe. */
  decide(txId, decision) {
    if (this.log.some((entry) => entry.txId === txId && entry.decision === decision)) return "duplicate";
    if (decision === "abort") {
      if (this.state === PREPARED && this.staged.txId === txId) {
        this.staged = null;
        this.state = IDLE;
      }
      this.log.push({ txId, decision });
      return "aborted";
    }
    if (this.state !== PREPARED || this.staged.txId !== txId) throw new Error(`${this.name}: commit without prepare`);
    this.version++;
    for (const [key, value] of this.staged.writes) {
      if (!this.history.has(key)) this.history.set(key, []);
      this.history.get(key).push({ version: this.version, value });
    }
    this.applied.push(txId);
    this.staged = null;
    this.state = IDLE;
    this.log.push({ txId, decision });
    return "committed";
  }

  decideOrBlock(txId) {
    if (this.state !== PREPARED) return "idle";
    return `blocked-on-${this.staged.txId}`;
  }
}

class Coordinator {
  constructor(name, participants, { buggy = false } = {}) {
    this.name = name;
    this.participants = participants;
    this.buggy = buggy;
    this.decisions = new Map(); // txId -> "commit" | "abort"
    this.trace = [];
    this.crashed = false;
    this.atomicityViolations = 0;
    this.inFlight = null; // { txId, prepared, pending }
  }

  /** Phase 1 and the decision. `drop` names participants whose prepare message never arrives;
   *  `crash` = "before-decision" or "after-decision" stops phase 2 the way a real crash would. */
  commit(tx, { drop = new Set(), crash = null } = {}) {
    const txId = tx.id;
    const prepared = [];
    let unanimous = true;
    this.trace.push(`${txId}: begin`);
    for (const [participant, writes] of tx.writes) {
      if (drop.has(participant.name)) {
        this.trace.push(`${txId}: prepare to ${participant.name} lost`);
        unanimous = false;
        continue;
      }
      const vote = participant.prepare(txId, writes, tx.snapshot.get(participant.name));
      this.trace.push(`${txId}: ${participant.name} voted ${vote.vote}`);
      if (vote.vote === "yes") prepared.push(participant);
      else unanimous = false;
    }
    if (crash === "before-decision") {
      this.crashed = true;
      this.inFlight = { txId, prepared, pending: [] };
      this.trace.push(`${txId}: crashed before deciding`);
      return { txId, decision: null, blocked: prepared.map((p) => p.name) };
    }
    const decision = this.buggy ? (prepared.length > 0 ? "commit" : "abort") : (unanimous ? "commit" : "abort");
    this.decisions.set(txId, decision);
    this.trace.push(`${txId}: decided ${decision}`);
    if (crash === "after-decision") {
      this.crashed = true;
      this.inFlight = { txId, prepared, pending: prepared.map((p) => p.name) };
      this.trace.push(`${txId}: crashed before phase 2`);
      return { txId, decision, blocked: prepared.map((p) => p.name) };
    }
    for (const participant of prepared) participant.decide(txId, decision);
    this.inFlight = null;
    return { txId, decision, blocked: [] };
  }

  /** Restart: replay the decision log; with no decision recorded the transaction aborts (a
   *  presumed-abort recovery is safe only because a decision is always logged before phase 2). */
  recover() {
    this.crashed = false;
    const inFlight = this.inFlight;
    if (inFlight === null) return { txId: null, decision: null, recovered: 0 };
    let decision = this.decisions.get(inFlight.txId);
    if (decision === undefined) {
      decision = "abort";
      this.decisions.set(inFlight.txId, decision);
      this.trace.push(`${inFlight.txId}: recovery assumed abort (no decision was logged)`);
    }
    let recovered = 0;
    for (const participant of inFlight.prepared) {
      const result = participant.decide(inFlight.txId, decision);
      if (result !== "duplicate") recovered++;
    }
    this.trace.push(`${inFlight.txId}: recovery resent ${decision}`);
    this.inFlight = null;
    return { txId: inFlight.txId, decision, recovered };
  }

  /** Atomicity: every participant named in the transaction applied it, or none did. */
  audit(tx) {
    const applied = tx.writes.filter(([participant]) => participant.applied.includes(tx.id)).length;
    const ok = applied === 0 || applied === tx.writes.length;
    if (!ok) this.atomicityViolations++;
    return ok;
  }
}

function newTransaction(id, participants, writesByParticipant) {
  return {
    id,
    snapshot: new Map(participants.map((participant) => [participant.name, participant.version])),
    writes: writesByParticipant,
  };
}

module.exports = { IDLE, PREPARED, Participant, Coordinator, newTransaction };

if (require.main === module) {
  const assert = require("node:assert");

  const build = () => [
    new Participant("A", { balance: 100 }),
    new Participant("B", { balance: 50 }),
    new Participant("C", { balance: 0 }),
  ];
  const write = (value) => new Map([["balance", value]]);

  // --- happy path: everyone prepares, everyone commits -----------------------------------
  let participants = build();
  let coordinator = new Coordinator("coord", participants);
  let tx = newTransaction("t1", participants, [[participants[0], write(99)], [participants[1], write(51)]]);
  let outcome = coordinator.commit(tx);
  assert.strictEqual(outcome.decision, "commit");
  assert.ok(coordinator.audit(tx), "atomicity after a successful commit");
  assert.strictEqual(participants[0].read("balance"), 99);
  assert.strictEqual(participants[1].read("balance"), 51);
  assert.strictEqual(participants[2].read("balance"), 0, "an untouched participant is untouched");
  assert.deepStrictEqual(participants[0].applied, ["t1"]);
  assert.ok(participants.every((participant) => participant.state === IDLE), "no locks survive");
  assert.strictEqual(participants[0].read("balance", 0), 100, "the old version is still readable");

  // --- a conflict votes NO and nobody applies anything ------------------------------------
  participants = build();
  coordinator = new Coordinator("coord", participants);
  const firstTx = newTransaction("t2", participants, [[participants[0], write(1)], [participants[1], write(2)]]);
  assert.strictEqual(coordinator.commit(firstTx).decision, "commit");
  const conflicting = newTransaction("t4", participants, [[participants[0], write(7)]]);
  conflicting.snapshot.set("A", 0); // this transaction started before t2 committed: A must refuse
  const refused = coordinator.commit(conflicting);
  assert.strictEqual(refused.decision, "abort", "a write-write conflict forces a NO vote");
  assert.ok(coordinator.audit(conflicting), "atomicity after an abort");
  assert.strictEqual(participants[0].read("balance"), 1, "the aborted transaction changed nothing");
  assert.deepStrictEqual(participants[0].applied, ["t2"]);
  assert.ok(participants.every((participant) => participant.state === IDLE), "the NO vote held no lock");
  const retry = newTransaction("t10", participants, [[participants[0], write(8)]]);
  assert.strictEqual(coordinator.commit(retry).decision, "commit", "a fresh snapshot prepares again");
  assert.strictEqual(participants[0].read("balance"), 8);
  assert.strictEqual(participants[0].read("balance", 1), 1, "older versions stay readable");

  // --- a dropped prepare with a crash before the decision blocks the prepared ones --------
  participants = build();
  coordinator = new Coordinator("coord", participants);
  const blockedTx = newTransaction("t5", participants, [
    [participants[0], write(1)], [participants[1], write(2)], [participants[2], write(3)],
  ]);
  const crashed = coordinator.commit(blockedTx, { drop: new Set(["C"]), crash: "before-decision" });
  assert.deepStrictEqual(crashed.blocked, ["A", "B"]);
  assert.strictEqual(participants[0].state, PREPARED, "A holds its locks and waits");
  assert.strictEqual(participants[0].read("balance"), 100, "prepared data is invisible");
  assert.strictEqual(participants[0].decideOrBlock("t5"), "blocked-on-t5", "a prepared participant is blocked");
  assert.throws(() => participants[0].decide("other", "commit"), Error, "a decision for another transaction is rejected");
  assert.deepStrictEqual(participants[0].applied, [], "and nothing was applied while blocked");
  assert.strictEqual(participants[0].prepare("t9", write(9), participants[0].version).vote, "no",
    "a prepared participant refuses a second transaction");
  assert.strictEqual(participants[1].state, PREPARED);
  assert.strictEqual(participants[2].state, IDLE, "the participant that never got the prepare is free");
  const recovery = coordinator.recover();
  assert.strictEqual(recovery.decision, "abort", "with no logged decision the presumed abort applies");
  assert.strictEqual(recovery.recovered, 2);
  assert.ok(participants.every((participant) => participant.state === IDLE), "the locks are released");
  assert.ok(coordinator.audit(blockedTx), "atomicity after recovery to abort");
  assert.deepStrictEqual(participants[0].applied, []);

  // --- a crash after the commit decision still commits everywhere -------------------------
  participants = build();
  coordinator = new Coordinator("coord", participants);
  const durableTx = newTransaction("t6", participants, [
    [participants[0], write(11)], [participants[1], write(12)], [participants[2], write(13)],
  ]);
  const late = coordinator.commit(durableTx, { crash: "after-decision" });
  assert.strictEqual(late.decision, "commit", "the decision was logged before the crash");
  assert.deepStrictEqual(late.blocked, ["A", "B", "C"]);
  assert.strictEqual(participants[0].read("balance"), 100, "nothing is visible while blocked");
  const replay = coordinator.recover();
  assert.strictEqual(replay.decision, "commit");
  assert.strictEqual(replay.recovered, 3, "every prepared participant is told about the commit");
  assert.ok(coordinator.audit(durableTx), "atomicity after recovery to commit");
  assert.strictEqual(participants[0].read("balance"), 11);
  assert.strictEqual(participants[1].read("balance"), 12);
  assert.strictEqual(participants[2].read("balance"), 13, "even the participant whose reply was late");
  assert.ok(participants.every((participant) => participant.state === IDLE));
  const auditBefore = coordinator.atomicityViolations;
  coordinator.recover();
  assert.strictEqual(coordinator.atomicityViolations, auditBefore, "a second recovery changes nothing");
  assert.deepStrictEqual(participants[0].applied, ["t6"], "recovery is idempotent");
  assert.strictEqual(participants[0].version, 1, "the commit was applied exactly once");

  // --- the buggy coordinator breaks atomicity on the very same scenarios -----------------
  const buggyParticipants = build();
  const buggy = new Coordinator("buggy", buggyParticipants, { buggy: true });
  const buggyTx = newTransaction("t7", buggyParticipants, [
    [buggyParticipants[0], write(1)], [buggyParticipants[1], write(2)],
  ]);
  buggyParticipants[1].history.get("balance").push({ version: 5, value: 777 });
  buggyParticipants[1].version = 5;
  buggyTx.snapshot.set("B", 0); // B must vote NO
  const buggyOutcome = buggy.commit(buggyTx);
  assert.strictEqual(buggyOutcome.decision, "commit", "the buggy coordinator ignores the NO vote");
  assert.strictEqual(buggy.audit(buggyTx), false, "and that breaks atomicity");
  assert.ok(buggy.atomicityViolations > 0, "the violation is detected");
  assert.strictEqual(buggyParticipants[0].read("balance"), 1);
  assert.strictEqual(buggyParticipants[1].read("balance"), 777, "B never applied the transaction");
  const strictParticipants = build();
  const strict = new Coordinator("strict", strictParticipants);
  const strictTx = newTransaction("t8", strictParticipants, [
    [strictParticipants[0], write(1)], [strictParticipants[1], write(2)],
  ]);
  strict.commit(strictTx, { drop: new Set(["B"]) });
  assert.strictEqual(strict.decisions.get("t8"), "abort", "the strict coordinator aborts on a lost prepare");
  assert.ok(strict.audit(strictTx), "and therefore keeps atomicity");
  assert.strictEqual(strict.atomicityViolations, 0);
  assert.strictEqual(strictParticipants[0].state, IDLE, "A prepared and then aborted cleanly");
  assert.deepStrictEqual(strictParticipants[0].applied, []);

  console.log(`2pc transactions=${coordinator.decisions.size} trace-lines=${coordinator.trace.length}`);
  console.log(`buggy-atomicity-violations=${buggy.atomicityViolations}`);
  console.log("PASS");
}
