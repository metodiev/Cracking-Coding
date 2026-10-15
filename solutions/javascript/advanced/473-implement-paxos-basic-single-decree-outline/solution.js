/** Task 473 - Implement Paxos basic single-decree outline. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability) */
"use strict";

const MAJORITY = 2; // of 3 acceptors

class Acceptor {
  constructor(id) {
    this.id = id;
    this.promised = -1;
    this.acceptedRound = -1;
    this.acceptedValue = null;
    this.history = []; // [round, value]
  }
}

class Proposer {
  constructor(id) {
    this.id = id;
    this.phase = "idle"; // idle | P1 | P2 | done
    this.roundNo = -1;
    this.ownValue = null;
    this.decidedValue = null;
    this.promises = []; // [acceptorId, acceptedRound, acceptedValue]
    this.accepts = new Set();
    this.chosen = null; // [round, value]
  }
}

class PaxosSim {
  constructor() {
    this.acceptors = [new Acceptor(0), new Acceptor(1), new Acceptor(2)];
    this.proposers = [new Proposer(0), new Proposer(1)];
    this.queue = []; // {destKind, destId, msg}
    this.chosenEvents = []; // [round, value]
    this.oracleChosen = new Set();
    this.lastPromised = [-1, -1, -1];
  }

  propose(p, roundNo, value) {
    const pr = this.proposers[p];
    pr.phase = "P1";
    pr.roundNo = roundNo;
    pr.ownValue = value;
    pr.decidedValue = null;
    pr.promises = [];
    pr.accepts = new Set();
    for (let a = 0; a < 3; a++) this.queue.push({ destKind: "A", destId: a, msg: { kind: "prepare", round: roundNo, proposer: p } });
  }

  deliverOne() {
    const item = this.queue.shift();
    const msg = item.msg;
    if (item.destKind === "A") {
      const a = this.acceptors[item.destId];
      if (msg.kind === "prepare") {
        if (msg.round > a.promised) {
          a.promised = msg.round;
          this.queue.push({
            destKind: "P",
            destId: msg.proposer,
            msg: { kind: "promise", round: msg.round, acc: a.id, accRound: a.acceptedRound, accValue: a.acceptedValue },
          });
        } else {
          this.queue.push({ destKind: "P", destId: msg.proposer, msg: { kind: "nack", round: msg.round, acc: a.id } });
        }
      } else {
        // accept
        if (msg.round >= a.promised) {
          a.promised = msg.round;
          a.acceptedRound = msg.round;
          a.acceptedValue = msg.value;
          a.history.push([msg.round, msg.value]);
          this.queue.push({ destKind: "P", destId: msg.proposer, msg: { kind: "accepted", round: msg.round, acc: a.id, value: msg.value } });
        } else {
          this.queue.push({ destKind: "P", destId: msg.proposer, msg: { kind: "nack", round: msg.round, acc: a.id } });
        }
      }
    } else {
      const pr = this.proposers[item.destId];
      if (msg.kind === "promise") {
        if (pr.phase !== "P1" || msg.round !== pr.roundNo) return;
        pr.promises.push([msg.acc, msg.accRound, msg.accValue]);
        if (pr.promises.length >= MAJORITY) {
          let bestRound = -1;
          let bestValue = pr.ownValue;
          for (const promise of pr.promises) {
            if (promise[2] !== null && promise[1] > bestRound) {
              bestRound = promise[1];
              bestValue = promise[2];
            }
          }
          pr.decidedValue = bestValue;
          pr.phase = "P2";
          for (let a = 0; a < 3; a++) this.queue.push({ destKind: "A", destId: a, msg: { kind: "accept", round: msg.round, value: bestValue, proposer: pr.id } });
        }
      } else if (msg.kind === "accepted") {
        if (pr.phase !== "P2" || msg.round !== pr.roundNo) return;
        pr.accepts.add(msg.acc);
        if (pr.accepts.size >= MAJORITY) {
          pr.chosen = [msg.round, msg.value];
          pr.phase = "done";
          this.chosenEvents.push([msg.round, msg.value]);
        }
      }
    }
  }

  deliver(k) {
    for (let i = 0; i < k && this.queue.length > 0; i++) this.deliverOne();
  }

  drain() {
    while (this.queue.length > 0) this.deliverOne();
  }

  drop(k) {
    for (let i = 0; i < k && this.queue.length > 0; i++) this.queue.shift();
  }

  check() {
    const oracle = new Set();
    for (const a of this.acceptors) {
      for (const [roundNo, value] of a.history) {
        let count = 0;
        for (const b of this.acceptors) {
          for (const [r2, v2] of b.history) if (r2 === roundNo && v2 === value) count += 1;
        }
        if (count >= MAJORITY) oracle.add(value);
      }
    }
    for (const v of oracle) this.oracleChosen.add(v);
    const chosenValues = new Set(this.oracleChosen);
    for (const [roundNo, value] of this.chosenEvents) chosenValues.add(value);
    if (chosenValues.size > 1) throw new Error(`at most one value may be chosen: ${[...chosenValues]}`);
    for (const a of this.acceptors) {
      if (a.promised < this.lastPromised[a.id]) throw new Error("promise regression");
      this.lastPromised[a.id] = a.promised;
      for (let i = 1; i < a.history.length; i++) {
        if (a.history[i][0] < a.history[i - 1][0]) throw new Error("accept rounds not monotone");
      }
    }
  }

  run(actions) {
    for (const action of actions) {
      const kind = action[0];
      if (kind === "propose") this.propose(action[1], action[2], action[3]);
      else if (kind === "deliver") this.deliver(action[1]);
      else if (kind === "drain") this.drain();
      else if (kind === "drop") this.drop(action[1]);
      this.check();
    }
  }
}

function lcg(seed) {
  return (seed * 75 + 74) % 65537;
}

function randomActions(count, seed, startRound) {
  const actions = [];
  let rnd = startRound;
  for (let i = 0; i < count; i++) {
    seed = lcg(seed);
    const a = seed % 8;
    seed = lcg(seed);
    const p = seed % 2;
    seed = lcg(seed);
    const value = seed % 20;
    seed = lcg(seed);
    const k = seed % 3;
    if (a < 3) {
      actions.push(["propose", p, rnd, value]);
      rnd += 1;
    } else if (a === 3) actions.push(["deliver", 1 + k]);
    else if (a < 6) actions.push(["drain"]);
    else if (a === 6) actions.push(["drop", k]);
    else actions.push(["deliver", 1]);
  }
  return actions;
}

module.exports = { PaxosSim, randomActions, lcg, MAJORITY };

if (require.main === module) {
  const assert = require("node:assert");
  const sim = new PaxosSim();

  // P0's value 7 is accepted by a majority (A0, A1) but all replies are dropped.
  sim.run([
    ["propose", 0, 1000, 7], ["deliver", 3], ["deliver", 2], ["deliver", 3], ["drop", 3],
  ]);
  assert.deepStrictEqual(sim.chosenEvents, []);
  assert.deepStrictEqual([...sim.oracleChosen], [7]);
  assert.deepStrictEqual(sim.acceptors[0].history, [[1000, 7]]);
  assert.deepStrictEqual(sim.acceptors[1].history, [[1000, 7]]);
  assert.deepStrictEqual(sim.acceptors[2].history, []);

  // P1 proposes 9 with a higher round; the promises carry (1000, 7), so it must re-propose 7.
  sim.run([["propose", 1, 1001, 9], ["drain"]]);
  assert.strictEqual(sim.proposers[1].decidedValue, 7, "P1 must adopt the chosen value, not its own 9");
  assert.deepStrictEqual(sim.proposers[1].chosen, [1001, 7]);
  assert.deepStrictEqual(sim.chosenEvents, [[1001, 7]]);
  assert.deepStrictEqual([...sim.oracleChosen], [7]);

  // P0 retries with round 1002 and its own value 8; it must also re-propose 7.
  sim.run([["propose", 0, 1002, 8], ["drain"]]);
  assert.strictEqual(sim.proposers[0].decidedValue, 7, "P0 retry must also re-propose 7");
  assert.deepStrictEqual(sim.chosenEvents, [[1001, 7], [1002, 7]]);
  assert.deepStrictEqual(sim.acceptors[0].history, [[1000, 7], [1001, 7], [1002, 7]]);
  assert.deepStrictEqual(sim.acceptors[1].history, [[1000, 7], [1001, 7], [1002, 7]]);
  assert.deepStrictEqual(sim.acceptors[2].history, [[1001, 7], [1002, 7]]);
  const acceptedValues = new Set();
  for (const a of sim.acceptors) for (const [, value] of a.history) acceptedValues.add(value);
  assert.deepStrictEqual([...acceptedValues], [7], "values 9 and 8 must never be accepted anywhere");

  // Randomized churn with losses: safety (at most one chosen value) must hold throughout.
  sim.run(randomActions(30, 13579, 2000));

  // Fair finale: fresh proposals with all messages delivered; liveness must return.
  for (let p = 0; p < 2; p++) {
    sim.run([["propose", p, 3000 + p, 5 + p], ["drain"]]);
  }
  assert.ok(sim.chosenEvents.length > 0, "fair run must choose a value");
  const chosenValues = new Set(sim.chosenEvents.map((event) => event[1]));
  assert.deepStrictEqual([...chosenValues], [7]);
  assert.deepStrictEqual([...sim.oracleChosen], [7]);
  console.log("PASS");
}
