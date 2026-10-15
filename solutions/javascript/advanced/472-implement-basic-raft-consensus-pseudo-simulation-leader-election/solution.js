/** Task 472 - Implement basic Raft consensus pseudo-simulation (leader election outline). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability) */
"use strict";

const N = 5;
const MAJORITY = 3;

class Node {
  constructor(id) {
    this.id = id;
    this.term = 0;
    this.votedFor = -1;
    this.state = "F"; // F follower, C candidate, L leader
    this.log = []; // {term, cmd}
    this.commit = 0;
    this.voters = new Set();
    this.nextIndex = new Array(N).fill(0);
    this.matchIndex = new Array(N).fill(-1);
  }
}

function sameEntry(a, b) {
  return a !== null && a !== undefined && b !== null && b !== undefined && a.term === b.term && a.cmd === b.cmd;
}

class RaftSim {
  constructor() {
    this.nodes = [];
    for (let i = 0; i < N; i++) this.nodes.push(new Node(i));
    this.alive = new Array(N).fill(true);
    this.partition = null;
    this.queue = [];
    this.leadersByTerm = new Map();
    this.committed = new Map();
    this.committedUpTo = 0;
    this.lastTerm = new Array(N).fill(0);
    this.lastCommit = new Array(N).fill(0);
  }

  connected(a, b) {
    if (this.partition === null) return true;
    return this.partition.some((g) => g.includes(a) && g.includes(b));
  }

  send(frm, to, msg) {
    if (!this.alive[frm] || !this.alive[to]) return;
    if (!this.connected(frm, to)) return;
    this.queue.push([frm, to, msg]);
  }

  drain() {
    while (this.queue.length > 0) {
      const [frm, to, msg] = this.queue.shift();
      if (!this.alive[frm] || !this.alive[to] || !this.connected(frm, to)) continue;
      this.deliver(frm, to, msg);
    }
  }

  crash(i) {
    this.alive[i] = false;
  }

  recover(i) {
    this.alive[i] = true;
    this.nodes[i].state = "F";
  }

  partitionOff(groups) {
    this.partition = groups.map((g) => g.slice());
  }

  heal() {
    this.partition = null;
  }

  timeout(i) {
    if (!this.alive[i]) return;
    const n = this.nodes[i];
    n.term += 1;
    n.votedFor = i;
    n.state = "C";
    n.voters = new Set([i]);
    const lastIdx = n.log.length - 1;
    const lastLogTerm = n.log.length === 0 ? 0 : n.log[n.log.length - 1].term;
    for (let j = 0; j < N; j++) if (j !== i) this.send(i, j, { kind: "RV", term: n.term, cand: i, lastIdx, lastTerm: lastLogTerm });
  }

  client(cmd) {
    let best = null;
    for (const n of this.nodes) {
      if (n.state === "L" && this.alive[n.id] && (best === null || n.term > best.term)) best = n;
    }
    if (best === null) return false;
    best.log.push({ term: best.term, cmd });
    return true;
  }

  replicate() {
    for (const n of this.nodes) {
      if (n.state !== "L" || !this.alive[n.id]) continue;
      for (let f = 0; f < N; f++) {
        if (f === n.id) continue;
        const nn = Math.min(n.nextIndex[f], n.log.length);
        const prevIdx = nn - 1;
        const prevTerm = prevIdx >= 0 ? n.log[prevIdx].term : 0;
        this.send(n.id, f, { kind: "AE", term: n.term, leader: n.id, prevIdx, prevTerm, entries: n.log.slice(nn), commit: n.commit });
      }
    }
  }

  deliver(frm, to, msg) {
    const n = this.nodes[to];
    if (msg.kind === "RV") {
      if (msg.term < n.term) {
        this.send(to, frm, { kind: "RVResp", term: n.term, voter: to, granted: false });
        return;
      }
      if (msg.term > n.term) {
        n.term = msg.term;
        n.votedFor = -1;
        if (n.state !== "F") n.state = "F";
      }
      const ownIdx = n.log.length - 1;
      const ownTerm = n.log.length === 0 ? 0 : n.log[n.log.length - 1].term;
      const upToDate = msg.lastTerm > ownTerm || (msg.lastTerm === ownTerm && msg.lastIdx >= ownIdx);
      const granted = (n.votedFor === -1 || n.votedFor === msg.cand) && upToDate;
      if (granted) n.votedFor = msg.cand;
      this.send(to, frm, { kind: "RVResp", term: n.term, voter: to, granted });
    } else if (msg.kind === "RVResp") {
      if (msg.term > n.term) {
        n.term = msg.term;
        n.votedFor = -1;
        n.state = "F";
        return;
      }
      if (n.state === "C" && msg.term === n.term && msg.granted && !n.voters.has(msg.voter)) {
        n.voters.add(msg.voter);
        if (n.voters.size >= MAJORITY) this.becomeLeader(n);
      }
    } else if (msg.kind === "AE") {
      if (msg.term < n.term) {
        this.send(to, frm, { kind: "AEResp", term: n.term, follower: to, success: false, hint: n.log.length });
        return;
      }
      if (msg.term > n.term) {
        n.term = msg.term;
        n.votedFor = -1;
      }
      n.state = "F";
      if (msg.prevIdx >= n.log.length || (msg.prevIdx >= 0 && n.log[msg.prevIdx].term !== msg.prevTerm)) {
        this.send(to, frm, { kind: "AEResp", term: n.term, follower: to, success: false, hint: n.log.length });
        return;
      }
      let idx = msg.prevIdx + 1;
      for (const entry of msg.entries) {
        if (idx < n.log.length) {
          if (n.log[idx].term !== entry.term) {
            n.log.length = idx;
            n.log.push(entry);
          }
        } else {
          n.log.push(entry);
        }
        idx += 1;
      }
      const lastNew = msg.prevIdx + msg.entries.length + 1;
      if (msg.commit > n.commit) {
        const newCommit = Math.min(msg.commit, lastNew);
        for (let i = n.commit; i < newCommit; i++) this.recordCommit(i, n.log[i]);
        n.commit = Math.max(n.commit, newCommit);
      }
      this.send(to, frm, { kind: "AEResp", term: n.term, follower: to, success: true, hint: lastNew });
    } else {
      if (msg.term > n.term) {
        n.term = msg.term;
        n.votedFor = -1;
        n.state = "F";
        return;
      }
      if (n.state !== "L" || msg.term !== n.term) return;
      if (msg.success) {
        if (msg.hint > n.matchIndex[msg.follower]) {
          n.matchIndex[msg.follower] = msg.hint;
          n.nextIndex[msg.follower] = msg.hint + 1;
        }
        let newCommit = n.commit;
        for (let cand = n.log.length; cand > n.commit; cand--) {
          if (n.log[cand - 1].term !== n.term) continue;
          let votes = 0;
          for (let j = 0; j < N; j++) {
            if ((j === n.id && n.log.length >= cand) || n.matchIndex[j] >= cand) votes += 1;
          }
          if (votes >= MAJORITY) {
            newCommit = cand;
            break;
          }
        }
        if (newCommit > n.commit) {
          for (let i = n.commit; i < newCommit; i++) this.recordCommit(i, n.log[i]);
          n.commit = newCommit;
        }
      } else {
        n.nextIndex[msg.follower] = Math.max(0, Math.min(n.nextIndex[msg.follower] - 1, msg.hint));
      }
    }
  }

  becomeLeader(n) {
    n.state = "L";
    const prev = this.leadersByTerm.get(n.term);
    if (!(prev === undefined || prev === n.id)) throw new Error("two leaders in same term");
    this.leadersByTerm.set(n.term, n.id);
    n.nextIndex = new Array(N).fill(n.log.length);
    n.matchIndex = new Array(N).fill(-1);
    for (let i = 0; i < this.committedUpTo; i++) {
      if (!(i < n.log.length && sameEntry(n.log[i], this.committed.get(i)))) {
        throw new Error("elected leader lacks committed entry " + i);
      }
    }
  }

  recordCommit(i, entry) {
    const prev = this.committed.get(i);
    if (prev !== undefined && !sameEntry(prev, entry)) throw new Error("committed entry changed at " + i);
    this.committed.set(i, entry);
    while (this.committed.has(this.committedUpTo)) this.committedUpTo += 1;
  }

  checkInvariants() {
    for (let a = 0; a < N; a++) {
      for (let b = a + 1; b < N; b++) {
        const la = this.nodes[a].log;
        const lb = this.nodes[b].log;
        const m = Math.min(la.length, lb.length);
        for (let i = 0; i < m; i++) {
          if (la[i].term === lb[i].term) {
            for (let k = 0; k <= i; k++) {
              if (!sameEntry(la[k], lb[k])) throw new Error(`log matching violated ${a},${b}@${i}`);
            }
          }
        }
      }
    }
    for (let i = 0; i < N; i++) {
      const n = this.nodes[i];
      if (n.commit > n.log.length) throw new Error("commit beyond log");
      if (n.term < this.lastTerm[i]) throw new Error("term regression");
      if (n.commit < this.lastCommit[i]) throw new Error("commit regression");
      this.lastTerm[i] = n.term;
      this.lastCommit[i] = n.commit;
    }
  }

  run(actions) {
    for (const action of actions) {
      const kind = action[0];
      if (kind === "timeout") this.timeout(action[1]);
      else if (kind === "client") this.client(action[1]);
      else if (kind === "replicate") this.replicate();
      else if (kind === "drain") this.drain();
      else if (kind === "partition") this.partitionOff(action[1]);
      else if (kind === "heal") this.heal();
      else if (kind === "crash") this.crash(action[1]);
      else if (kind === "recover") this.recover(action[1]);
      this.checkInvariants();
    }
  }
}

function lcg(seed) {
  return (seed * 75 + 74) % 65537;
}

function randomActions(count, seed) {
  const actions = [];
  let counter = 0;
  for (let k = 0; k < count; k++) {
    seed = lcg(seed);
    const a = seed % 8;
    seed = lcg(seed);
    const node = seed % N;
    if (a === 0) actions.push(["timeout", node]);
    else if (a === 1) {
      counter += 1;
      actions.push(["client", `c${counter}`]);
    } else if (a === 2) actions.push(["replicate"]);
    else if (a === 3) actions.push(["drain"]);
    else if (a === 4) {
      seed = lcg(seed);
      const mask = seed;
      const g0 = [];
      const g1 = [];
      for (let i = 0; i < N; i++) {
        if ((mask >> i) & 1) g0.push(i);
        else g1.push(i);
      }
      actions.push(["partition", [g0, g1]]);
    } else if (a === 5) actions.push(["heal"]);
    else if (a === 6) actions.push(["crash", node]);
    else actions.push(["recover", node]);
    if (a !== 3) actions.push(["drain"]);
  }
  return actions;
}

module.exports = { RaftSim, randomActions, lcg, N };

if (require.main === module) {
  const assert = require("node:assert");
  const sim = new RaftSim();
  sim.run([["timeout", 0], ["drain"]]);
  assert.deepStrictEqual([...sim.leadersByTerm.entries()], [[1, 0]]);
  sim.run([["client", "x=1"], ["replicate"], ["drain"], ["replicate"], ["drain"]]);
  assert.strictEqual(sim.committedUpTo, 1);
  assert.ok(sameEntry(sim.committed.get(0), { term: 1, cmd: "x=1" }));

  sim.run([["crash", 0], ["timeout", 1], ["drain"]]);
  assert.strictEqual(sim.leadersByTerm.get(2), 1);
  sim.run([["client", "y=2"], ["replicate"], ["drain"], ["replicate"], ["drain"]]);
  assert.strictEqual(sim.committedUpTo, 2);
  assert.ok(sameEntry(sim.committed.get(1), { term: 2, cmd: "y=2" }));

  sim.run([
    ["partition", [[0, 1, 2], [3, 4]]],
    ["client", "z=3"], ["replicate"], ["drain"], ["replicate"], ["drain"],
  ]);
  assert.strictEqual(sim.committedUpTo, 2, "z=3 must not commit without a quorum");
  assert.ok(sameEntry(sim.nodes[1].log[sim.nodes[1].log.length - 1], { term: 2, cmd: "z=3" }));
  assert.ok(sameEntry(sim.nodes[2].log[sim.nodes[2].log.length - 1], { term: 2, cmd: "z=3" }));

  sim.run([["timeout", 3], ["drain"]]);
  assert.ok(!sim.leadersByTerm.has(3), "minority must not elect a leader");
  sim.run([["heal"], ["drain"]]);
  sim.run([["timeout", 1], ["drain"]]);
  assert.ok(!sim.leadersByTerm.has(3));
  sim.run([["timeout", 1], ["drain"]]);
  assert.strictEqual(sim.leadersByTerm.get(4), 1);

  sim.run([
    ["client", "w=4"], ["replicate"], ["drain"], ["replicate"], ["drain"], ["replicate"], ["drain"],
  ]);
  assert.strictEqual(sim.committedUpTo, 4);
  assert.ok(sameEntry(sim.committed.get(2), { term: 2, cmd: "z=3" }));
  assert.ok(sameEntry(sim.committed.get(3), { term: 4, cmd: "w=4" }));
  const expected = [[1, "x=1"], [2, "y=2"], [2, "z=3"], [4, "w=4"]];
  for (const i of [1, 2, 3, 4]) {
    assert.deepStrictEqual(sim.nodes[i].log.map((e) => [e.term, e.cmd]), expected, `log mismatch node ${i}`);
  }

  sim.run(randomActions(64, 24680));

  const finale = [["heal"]];
  for (let i = 0; i < N; i++) finale.push(["recover", i]);
  finale.push(["drain"], ["drain"]);
  sim.run(finale);
  let best = 0;
  for (let i = 1; i < N; i++) {
    const key = (j) => {
      const log = sim.nodes[j].log;
      return [log.length === 0 ? 0 : log[log.length - 1].term, log.length];
    };
    const [ta, la] = key(i);
    const [tb, lb] = key(best);
    if (ta > tb || (ta === tb && la > lb)) best = i;
  }
  for (let attempt = 0; attempt < 40; attempt++) {
    sim.run([["timeout", best], ["drain"]]);
    if (sim.nodes[best].state === "L") break;
  }
  assert.strictEqual(sim.nodes[best].state, "L", "most up-to-date node must win");
  sim.run([["client", "end"]]);
  for (let attempt = 0; attempt < 12; attempt++) {
    sim.run([["replicate"], ["drain"], ["replicate"], ["drain"]]);
    if (sim.committedUpTo === sim.nodes[best].log.length) break;
  }
  assert.strictEqual(sim.committedUpTo, sim.nodes[best].log.length);
  const sawEnd = [...sim.committed.values()].some((e) => sameEntry(e, { term: sim.nodes[best].term, cmd: "end" }));
  assert.ok(sawEnd, "end command committed");
  console.log("PASS");
}
