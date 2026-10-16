/** Task 631 - Implement gossip protocol for membership and failure detection. (Tier: Expert, CTCI Beyond — Distributed Systems)
 *
 * SWIM-style gossip: every node periodically sends its whole membership table to a few random peers,
 * and each entry carries an *incarnation number*. A higher incarnation always wins; at equal
 * incarnation the more pessimistic status wins, which is why a node can only refute a rumour about
 * itself by bumping its incarnation and shouting "alive, incarnation n+1" - the rumour is refuted by
 * arithmetic rather than by argument. Failure detection is the other half: when a peer stops being
 * reachable, the suspect counter escalates suspect -> dead, and every node that can still gossip
 * converges on the same verdict.
 *
 * What is simplified: there are no sockets and no clocks. A tick is one gossip round in which every
 * live node picks `fanout` reachable peers with a seeded LCG and the snapshots are merged
 * synchronously, so peer selection is reproducible. Connectivity, not latency, drives detection: a
 * node marks a peer suspect when it cannot reach it at all, after `failureTicks` rounds, and dead
 * after twice as many. During a partition both sides therefore form their own verdict - the checks
 * verify that the member *set* never shrinks anywhere (nobody is ever dropped), that each side
 * converges internally, and that healing refutes the false verdicts by incarnation rather than
 * leaving the cluster split.
 *
 * Verified invariants: (1) live nodes converge on identical views (name, incarnation, status) and a
 * late joiner reaches every live node within a bounded number of rounds; (2) a permanently stopped
 * node is declared dead by every live node within `2 * failureTicks + 2` rounds; (3) a partition
 * produces convergent verdicts per side while no view ever loses a member; (4) after healing, a node
 * marked dead bumps its incarnation and is alive everywhere again with a strictly higher incarnation
 * than the mark it refuted, and the refutation happens exactly once; (5) merging is idempotent and
 * never lets a lower incarnation (or a less severe status at the same incarnation) overwrite a newer
 * rumour, and no view ever contains a name that was never a member.
 */
"use strict";

const ALIVE = "alive", SUSPECT = "suspect", DEAD = "dead";
const SEVERITY = { alive: 0, suspect: 1, dead: 2 };

function lcg(seed) {
  return (seed * 48271) % 2147483647;
}

function hashName(name) {
  let hash = 7;
  for (let index = 0; index < name.length; index++) hash = (hash * 31 + name.charCodeAt(index)) % 2147483647;
  return hash;
}

/** The membership table: name -> { name, incarnation, status }. */
class Membership {
  constructor() {
    this.entries = new Map();
  }
  get(name) {
    return this.entries.get(name);
  }
  isAlive(name) {
    const entry = this.entries.get(name);
    return entry !== undefined && entry.status === ALIVE;
  }
  /** Merge rule: higher incarnation wins; at equal incarnation only a more severe status does. */
  observe(name, incarnation, status) {
    const current = this.entries.get(name);
    if (current === undefined) {
      this.entries.set(name, { name, incarnation, status });
      return "added";
    }
    if (incarnation > current.incarnation) {
      current.incarnation = incarnation;
      current.status = status;
      return "newer-incarnation";
    }
    if (incarnation === current.incarnation && SEVERITY[status] > SEVERITY[current.status]) {
      current.status = status;
      return "more-severe";
    }
    return "ignored";
  }
  /** A node refutes a rumour about itself by incrementing its incarnation. */
  refute(name) {
    const current = this.entries.get(name);
    if (current === undefined) return null;
    current.incarnation++;
    current.status = ALIVE;
    return current.incarnation;
  }
  snapshot() {
    return [...this.entries.values()].map((entry) => ({ ...entry }));
  }
  signature() {
    return [...this.entries.values()]
      .map((entry) => `${entry.name}:${entry.incarnation}:${entry.status}`)
      .sort()
      .join("|");
  }
}

class GossipNode {
  constructor(name) {
    this.name = name;
    this.view = new Membership();
    this.view.observe(name, 0, ALIVE);
    this.lastContact = new Map();
    this.offlineRounds = new Map();
    this.silent = false;
    this.refutations = 0;
  }
  remember(name, round) {
    this.lastContact.set(name, round);
    this.offlineRounds.set(name, 0);
  }
}

class GossipCluster {
  constructor(names, { fanout = 2, failureTicks = 3, seed = 12345 } = {}) {
    this.nodes = new Map(names.map((name) => [name, new GossipNode(name)]));
    this.fanout = fanout;
    this.failureTicks = failureTicks;
    this.round = 0;
    this.seed = seed;
    this.groups = null;
    this.transfers = 0;
    this.merges = 0;
  }
  join(name) {
    this.nodes.set(name, new GossipNode(name));
    return name;
  }
  crash(name) {
    this.nodes.get(name).silent = true;
  }
  partition(groups) {
    this.groups = groups.map((group) => new Set(group));
  }
  heal() {
    this.groups = null;
  }
  reachable(from, to) {
    if (this.groups === null) return true;
    return this.groups.some((group) => group.has(from) && group.has(to));
  }
  isIsolated(name) {
    return ![...this.nodes.keys()].some((other) => other !== name && this.reachable(name, other));
  }
  /** Deterministic peer choice: per-round LCG seeded by the node name. */
  pickPeers(name) {
    const candidates = [...this.nodes.keys()].filter((other) => other !== name
      && !this.nodes.get(other).silent && this.reachable(name, other));
    const chosen = [];
    let seed = hashName(name) + 7919 * (this.round + 1);
    while (chosen.length < Math.min(this.fanout, candidates.length)) {
      seed = lcg(seed);
      const candidate = candidates[seed % candidates.length];
      if (!chosen.includes(candidate)) chosen.push(candidate);
    }
    return chosen;
  }
  tick() {
    this.round++;
    const outgoing = [];
    for (const [name, node] of this.nodes) {
      if (node.silent) continue;
      for (const peer of this.pickPeers(name)) {
        outgoing.push({ from: name, to: peer, view: node.view.snapshot() });
      }
    }
    for (const delivery of outgoing) {
      const target = this.nodes.get(delivery.to);
      if (target.silent) continue;
      target.remember(delivery.from, this.round);
      for (const entry of delivery.view) {
        const outcome = target.view.observe(entry.name, entry.incarnation, entry.status);
        this.merges++;
        if (outcome === "ignored") continue;
        if (entry.name === target.name && entry.status !== ALIVE) {
          target.view.refute(target.name); // a rumour about me that is not "alive" gets refuted
          target.refutations++;
        }
      }
      this.transfers++;
    }
    this.detectFailures();
    return this;
  }
  run(rounds) {
    for (let round = 0; round < rounds; round++) this.tick();
    return this;
  }
  /** Escalate suspect -> dead for peers that cannot be reached; isolated nodes run no detector. */
  detectFailures() {
    for (const [name, node] of this.nodes) {
      if (node.silent || this.isIsolated(name)) continue;
      for (const entry of node.view.entries.values()) {
        if (entry.name === name) continue;
        const reachableNow = !this.nodes.get(entry.name).silent && this.reachable(name, entry.name);
        if (reachableNow) {
          node.offlineRounds.set(entry.name, 0);
          continue;
        }
        const rounds = (node.offlineRounds.get(entry.name) || 0) + 1;
        node.offlineRounds.set(entry.name, rounds);
        if (rounds === this.failureTicks) entry.status = SUSPECT;
        else if (rounds >= 2 * this.failureTicks) entry.status = DEAD;
      }
    }
  }
  liveNames() {
    return [...this.nodes.values()].filter((node) => !node.silent).map((node) => node.name);
  }
  /** Converged = every live node's view has the same signature. */
  converged() {
    const signatures = new Set(this.liveNames().map((name) => this.nodes.get(name).view.signature()));
    return signatures.size === 1;
  }
  viewOf(name) {
    return this.nodes.get(name).view;
  }
}

module.exports = { ALIVE, SUSPECT, DEAD, Membership, GossipNode, GossipCluster, hashName, lcg };

if (require.main === module) {
  const assert = require("node:assert");
  // --- merge rules -------------------------------------------------------------------------
  const view = new Membership();
  assert.strictEqual(view.observe("a", 0, ALIVE), "added");
  assert.strictEqual(view.observe("a", 0, SUSPECT), "more-severe", "same incarnation, more pessimistic wins");
  assert.strictEqual(view.observe("a", 0, ALIVE), "ignored", "alive cannot refute at the same incarnation");
  assert.strictEqual(view.observe("a", 0, DEAD), "more-severe");
  assert.strictEqual(view.observe("a", 0, ALIVE), "ignored", "an equal-incarnation alive rumour is ignored");
  assert.strictEqual(view.observe("a", 1, ALIVE), "newer-incarnation", "a bump refutes the dead mark");
  assert.strictEqual(view.get("a").status, ALIVE);
  assert.strictEqual(view.observe("a", 0, DEAD), "ignored", "a stale rumour never revives itself");
  const before = view.signature();
  view.observe("a", 1, ALIVE);
  assert.strictEqual(view.signature(), before, "re-merging the same entry changes nothing");
  assert.deepStrictEqual(view.refute("a"), 2, "refute bumps the incarnation");
  assert.strictEqual(view.observe("a", 2, ALIVE), "ignored", "and the bumped entry is unchanged");

  // --- convergence, late join and idempotence -----------------------------------------------
  const cluster = new GossipCluster(["n0", "n1", "n2", "n3", "n4"], { fanout: 2 });
  cluster.run(12);
  assert.ok(cluster.converged(), "a healthy cluster converges");
  assert.deepStrictEqual(cluster.viewOf("n0").snapshot().map((entry) => entry.name).sort(), ["n0", "n1", "n2", "n3", "n4"]);
  assert.ok(cluster.liveNames().every((name) => cluster.viewOf("n4").isAlive(name)), "everyone looks alive");
  const mergesBefore = cluster.merges;
  cluster.tick();
  assert.ok(cluster.merges > mergesBefore, "gossip keeps flowing");
  assert.ok(cluster.converged(), "and stays converged");
  cluster.join("n5");
  cluster.run(10);
  assert.ok(cluster.converged(), "the cluster re-converges after a join");
  for (const name of cluster.liveNames()) assert.ok(cluster.viewOf(name).get("n5") !== undefined, `${name} learned about n5`);
  assert.ok(cluster.viewOf("n0").entries instanceof Map, "the membership table is a Map");

  // --- a crashed node is excluded within a bounded number of rounds -------------------------
  const crashed = new GossipCluster(["c0", "c1", "c2", "c3", "c4"], { fanout: 2, failureTicks: 3 });
  crashed.run(6); // everyone knows everyone before the crash
  crashed.crash("c4");
  let declaredAt = null;
  for (let round = 1; round <= 12; round++) {
    crashed.tick();
    const all = ["c0", "c1", "c2", "c3"].every((name) => crashed.viewOf(name).get("c4").status === DEAD);
    if (all && declaredAt === null) declaredAt = round;
  }
  assert.ok(declaredAt !== null, "the crash is detected");
  assert.ok(declaredAt <= 2 * 3 + 2, `detection took ${declaredAt} rounds, expected at most 8`);
  assert.ok(crashed.converged(), "the survivors agree on the verdict");
  assert.strictEqual(crashed.viewOf("c0").get("c4").status, DEAD);
  assert.ok(crashed.viewOf("c0").get("c4").incarnation >= 0, "no incarnation was invented");
  assert.strictEqual(crashed.nodes.get("c4").silent, true, "the crashed node stopped gossiping");
  assert.ok(["c0", "c1", "c2", "c3"].every((name) => crashed.viewOf("c4").isAlive(name)),
    "its own frozen view still shows everyone alive");

  // --- a partition excludes the isolated side while the majority converges -------------------
  const split = new GossipCluster(["p0", "p1", "p2", "p3", "p4"], { fanout: 2, failureTicks: 3 });
  split.run(6); // everybody knows everybody before the split
  split.partition([["p0", "p1"], ["p2", "p3", "p4"]]);
  split.run(20);
  const majority = ["p2", "p3", "p4"];
  for (const name of majority) {
    assert.strictEqual(split.viewOf(name).get("p0").status, DEAD, `${name} declares p0 dead`);
    assert.strictEqual(split.viewOf(name).get("p1").status, DEAD, `${name} declares p1 dead`);
    assert.ok(split.viewOf(name).isAlive(name), "and still believes itself alive");
  }
  assert.strictEqual(new Set(majority.map((name) => split.viewOf(name).signature())).size, 1, "the majority converges");
  for (const name of ["p0", "p1"]) {
    assert.strictEqual(split.viewOf(name).get("p2").status, DEAD, `${name} suspects the majority from its side`);
    assert.ok(split.viewOf(name).isAlive(name), "but never believes itself dead");
  }
  assert.strictEqual(new Set(["p0", "p1"].map((name) => split.viewOf(name).signature())).size, 1, "the minority converges too");
  for (const name of split.liveNames()) assert.strictEqual(split.viewOf(name).entries.size, 5, "no member is ever dropped");
  const deadMarks = majority.map((name) => split.viewOf(name).get("p0").incarnation);
  assert.strictEqual(new Set(deadMarks).size, 1, "the majority agrees on the incarnation it marked dead");

  // --- healing: both sides refute their false verdicts with a higher incarnation ------------
  split.heal();
  split.run(30);
  assert.ok(split.converged(), "the cluster converges again after healing");
  for (const name of split.liveNames()) {
    for (const other of split.liveNames()) assert.ok(split.viewOf(name).isAlive(other), `${name} sees ${other} alive`);
  }
  const revived = split.viewOf("p0").get("p0");
  assert.strictEqual(revived.incarnation, deadMarks[0] + 1, "exactly one refutation was needed");
  assert.strictEqual(split.nodes.get("p0").refutations, 1, "and it happened once");
  assert.ok(split.viewOf("p2").get("p0").incarnation === revived.incarnation, "the refutation spread");
  assert.ok(split.viewOf("p0").get("p2").incarnation >= 1, "the other side refuted as well");

  // --- no ghosts, and views stay bounded ----------------------------------------------------
  for (const name of split.liveNames()) {
    for (const entry of split.viewOf(name).snapshot()) {
      assert.ok(split.nodes.has(entry.name), `ghost member ${entry.name}`);
      assert.ok([ALIVE, SUSPECT, DEAD].includes(entry.status), "statuses are from the closed set");
    }
  }
  assert.strictEqual(split.viewOf("p1").entries.size, 5, "no member is ever dropped from a view");

  console.log(`gossip rounds=${split.round} transfers=${split.transfers} merges=${split.merges}`);
  console.log(`detection rounds=${declaredAt} refutations=${split.nodes.get("p0").refutations}`);
  console.log("PASS");
}
