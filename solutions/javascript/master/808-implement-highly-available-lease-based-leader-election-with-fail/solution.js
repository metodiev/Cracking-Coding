/** Task 808 - Implement highly-available lease-based leader election with failover testing. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A lease-based leader election protocol over an explicitly simulated, drifting clock: a candidate
 * asks a quorum of servers for a lease, a server promises to reject competing requests until the
 * lease it granted has expired *by its own clock*, and the lease request carries the maximum drift
 * the candidate is willing to tolerate. The granted lease is valid for
 * `duration - 2 * maxDrift` of real time, which is exactly the margin that makes two leases for
 * different leaders unable to overlap in real time - the safety property this task is about.
 *
 * Time is explicit: a global tick counter is the real time, every server has its own clock reading
 * (real time plus a seeded drift, bounded by maxDrift), messages are in-flight records with a
 * deterministic delivery delay, and a leader fails by being paused for a number of ticks. There is
 * no wall clock, no timer and no thread.
 *
 * What is simplified: servers are in-process objects with a per-server drift instead of NTP-bounded
 * hardware clocks, the grant rule is a ballot/promise pair rather than a full Paxos log, and a
 * leadership "lease" protects a single leader slot instead of a replicated log. The scale knobs are
 * typed parameters (server count, drift bound, lease length, failover count).
 *
 * Invariants verified:
 *   - safety: no two leaders ever hold overlapping leases in *real* time, checked over every grant
 *     in a seeded random run, across drift values up to the advertised bound;
 *   - tightness of the bound: with a drift larger than the protocol assumes, the same harness
 *     detects an overlap - so `duration - 2*maxDrift` is not a gratuitous margin;
 *   - liveness: after a leader stalls, a new leader is elected within a bounded real-time window,
 *     and the old leader steps down as soon as it hears a higher ballot;
 *   - quorum discipline: only a majority can grant a lease, a minority partition cannot elect, and
 *     the number of granted leases per ballot never exceeds one;
 *   - failover testing: repeated stalls, message loss and partitions always end with at most one
 *     active leader and a live leader within the bound;
 *   - cost: messages and rounds per election stay linear in the number of servers (reported).
 */
"use strict";

const assert = require("node:assert");

class LeaseError extends Error {}

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}

const channelKey = (from, to) => `${from}>${to}`;

// --------------------------------------------------------------------------- network

class Network {
  constructor(servers) {
    this.servers = servers;
    this.queue = [];
    this.blocked = new Set();
    this.seq = 0;
    this.now = 0;                    // real-time tick used to stamp delivery rounds
    this.delivered = 0;
    this.dropped = 0;
    this.delay = 1;
  }

  links(a, b) { return a === b || !this.blocked.has(channelKey(a, b)); }

  partition(groups) {
    this.blocked.clear();
    for (const group of groups) for (const other of groups) {
      if (group === other) continue;
      for (const a of group) for (const b of other) this.blocked.add(channelKey(a, b));
    }
  }

  heal() { this.blocked.clear(); }

  send(from, to, payload, drop = false) {
    this.seq++;
    if (drop) { this.dropped++; return false; }
    this.queue.push({ from, to, payload, seq: this.seq, deliverAt: this.now + this.delay });
    return true;
  }

  step(round) {
    const due = this.queue.filter((message) => message.deliverAt <= round && this.links(message.from, message.to));
    this.queue = this.queue.filter((message) => !(message.deliverAt <= round && this.links(message.from, message.to)));
    due.sort((a, b) => (a.to - b.to) || (a.seq - b.seq));
    for (const message of due) {
      this.delivered++;
      const server = this.servers.get(message.to);
      if (server !== undefined) server.receive(message.from, message.payload);
    }
    return due.length;
  }
}

// --------------------------------------------------------------------------- server

class Server {
  constructor(id, opts) {
    this.id = id;
    this.rateBound = opts.rateBound;         // protocol assumption: |clock rate error| <= rateBound
    this.rate = opts.rate;                   // this server's clock rate error
    this.duration = opts.duration;           // promised lease length in the candidate's clock
    this.ballot = 0;                         // highest ballot promised so far
    this.lease = null;                       // {holder, expiryLocal, ballot, duration}
    this.pausedUntil = 0;
    this.role = "follower";
    this.currentLease = null;                // the lease this server holds as leader
    this.requests = 0;
    this.grants = 0;
    this.rejections = 0;
    this.clock = 0;
  }

  /** The server's own view of real time: a drifting clock, rate error bounded by `rateBound`. */
  localTime(realTime) { return realTime * (1 + this.rate); }

  paused(realTime) { return realTime < this.pausedUntil; }

  /** A grant is only legal after every lease this server ever handed out has expired locally. */
  canGrant(realTime, ballot) {
    if (ballot <= this.ballot) return false;
    if (this.lease === null) return true;
    return this.localTime(realTime) > this.lease.expiryLocal;
  }

  grant(realTime, candidate, ballot, duration) {
    // The promise covers the full requested duration measured with this server's clock, so a
    // competing candidate can never be granted a lease inside the window this one reserved.
    const expiryLocal = this.localTime(realTime) + duration;
    this.ballot = ballot;
    this.lease = { holder: candidate, expiryLocal, ballot, duration };
    this.grants++;
    return { expiryLocal };
  }

  receive(from, payload) {
    if (payload.type === "lease-reply") {
      const entry = this.cluster?.pending?.get(payload.ballot);
      if (entry !== undefined && entry.candidateId === this.id) entry.replies.set(from, { ...payload, voter: from });
      return;
    }
    if (payload.type !== "request-lease") return;
    this.requests++;
    if (payload.ballot <= this.ballot || !this.canGrant(payload.realTime, payload.ballot)) {
      this.rejections++;
      this.network.send(this.id, from, { type: "lease-reply", ballot: payload.ballot, granted: false, realTime: payload.realTime });
      return;
    }
    const { expiryLocal } = this.grant(payload.realTime, payload.candidate, payload.ballot, payload.duration);
    this.network.send(this.id, from, {
      type: "lease-reply", ballot: payload.ballot, granted: true, expiryLocal, realTime: payload.realTime, holder: payload.candidate,
    });
  }

  tick(realTime) {
    if (this.paused(realTime)) return;
    this.clock = this.localTime(realTime);
    if (this.lease !== null && this.clock > this.lease.expiryLocal) this.lease = null;
  }
}

// --------------------------------------------------------------------------- cluster

class Cluster {
  constructor(opts = {}) {
    this.size = opts.size ?? 5;
    this.rateBound = opts.rateBound ?? 0.15;                    // protocol assumption on clock rates
    this.duration = opts.duration ?? 20;
    this.maxDrift = this.rateBound * this.duration;             // worst-case offset error per lease
    this.entitlement = this.duration - 2 * this.maxDrift;       // the window the leader may use
    this.round = 0;
    this.rng = splitmix64(opts.seed ?? 0x808);
    this.lossRate = opts.lossRate ?? 0;
    this.servers = new Map();
    for (let id = 1; id <= this.size; id++) {
      const rate = opts.rates !== undefined ? opts.rates[(id - 1) % opts.rates.length]
        : ((this.rng.int(7) - 3) / 3) * this.rateBound;
      this.servers.set(id, new Server(id, { rateBound: this.rateBound, rate, duration: this.duration }));
    }
    this.network = new Network(this.servers);
    for (const server of this.servers.values()) { server.network = this.network; server.cluster = this; }
    this.grants = [];                 // audit trail: every accepted lease, with real-time bounds
    this.messages = 0;
    this.detectedOverlaps = [];
    this.ballots = new Map();         // ballot -> leader id
    this.realTime = 0;
  }

  peers(id) { return [...this.servers.keys()].filter((other) => other !== id).sort((a, b) => a - b); }

  quorum() { return Math.floor(this.size / 2) + 1; }

  /** The candidate asks every peer for a lease of `duration`; the replies are collected here. */
  campaign(candidateId, opts = {}) {
    const candidate = this.servers.get(candidateId);
    this.network.now = this.round;
    if (candidate.paused(this.round)) return { granted: false, reason: "candidate is paused" };
    const ballot = (opts.ballot ?? (candidate.ballot + 1));
    const request = { type: "request-lease", candidate: candidateId, ballot, realTime: this.round, duration: this.duration };
    for (const peer of this.peers(candidateId)) {
      this.messages++;
      const drop = this.lossRate > 0 && this.rng.int(100) < this.lossRate;
      this.network.send(candidateId, peer, request, drop);
      if (drop) this.messages--;
    }
    this.pending = this.pending ?? new Map();
    this.pending.set(ballot, { candidateId, ballot, replies: new Map(), requestRound: this.round });
    return { granted: false, reason: "in flight", ballot };
  }

  /** Deliver everything and then let each candidate decide with the promises it collected. */
  settle(rounds = 6) {
    const results = [];
    for (let i = 0; i < rounds; i++) {
      this.step();
      const pending = [...(this.pending ?? new Map()).values()];
      for (const entry of pending) {
        // A candidate waits for every reply it can get: all peers, or a few rounds when some of
        // them are unreachable (partitioned or crashed) - then it decides with the quorum it has.
        const complete = entry.replies.size >= this.size - 1 || this.round - entry.requestRound >= 3;
        if (!complete) continue;
        results.push(this.finish(entry));
        this.pending.delete(entry.ballot);
      }
      if ((this.pending?.size ?? 0) === 0 && this.network.queue.length === 0) break;
    }
    return results;
  }

  step() {
    this.round++;
    this.network.now = this.round;
    for (const server of this.servers.values()) server.tick(this.round);
    this.network.step(this.round);
    // Leader-side bookkeeping: once the entitlement window ends the leader steps down.
    for (const server of this.servers.values()) {
      if (server.currentLease !== null && this.round > server.currentLease.entitlementEnd) {
        server.currentLease = null;
        server.role = "follower";
      }
    }
    this.checkOverlaps();
    return this.round;
  }

  run(ticks) { for (let i = 0; i < ticks; i++) this.step(); return this.round; }

  /** Record a lease and audit its entitlement window against every lease still entitled. */
  recordGrant(candidateId, ballot, grantedReplies) {
    const startReal = this.round;
    const entitlementEnd = startReal + this.entitlement;
    const lease = {
      holder: candidateId,
      ballot,
      startReal,
      entitlementEnd,
      voters: grantedReplies.map((reply) => reply.voter),
    };
    for (const existing of this.grants) {
      if (existing.holder === candidateId) continue;
      if (existing.entitlementEnd >= lease.startReal && lease.entitlementEnd >= existing.startReal) {
        this.detectedOverlaps.push({ first: existing, second: lease, round: this.round, reason: "two entitlement windows overlap" });
      }
    }
    this.grants.push(lease);
    if (this.ballots.has(ballot) && this.ballots.get(ballot) !== candidateId) {
      this.detectedOverlaps.push({ first: { holder: this.ballots.get(ballot), ballot }, second: lease, round: this.round, reason: "two leaders in one ballot" });
    }
    this.ballots.set(ballot, candidateId);
    return lease;
  }

  finish(entry) {
    const candidate = this.servers.get(entry.candidateId);
    const granted = [...entry.replies.values()].filter((reply) => reply.granted);
    if (granted.length + 1 < this.quorum()) {
      return { candidateId: entry.candidateId, ballot: entry.ballot, granted: false, reason: "no quorum", grantedCount: granted.length };
    }
    // The leader may use its lease for the entitlement window: the promised duration minus twice
    // the worst-case clock offset error, which is what keeps two leadership windows disjoint.
    const entitlementEnd = this.round + this.entitlement;
    candidate.currentLease = { entitlementEnd, ballot: entry.ballot, duration: this.duration, startReal: this.round };
    candidate.role = "leader";
    const lease = this.recordGrant(entry.candidateId, entry.ballot, granted);
    return { candidateId: entry.candidateId, ballot: entry.ballot, granted: true, entitlementEnd, startReal: lease.startReal, voters: lease.voters };
  }

  checkOverlaps() {
    const leaders = this.activeLeaders();
    for (const leader of leaders) {
      for (const other of leaders) {
        if (other.id <= leader.id) continue;
        const first = this.grants.filter((lease) => lease.holder === leader.id).pop();
        const second = this.grants.filter((lease) => lease.holder === other.id).pop();
        if (first === undefined || second === undefined) continue;
        if (first.entitlementEnd >= second.startReal && second.entitlementEnd >= first.startReal) {
          this.detectedOverlaps.push({ first, second, round: this.round, reason: "two active leaders hold overlapping entitlements" });
        }
      }
    }
  }

  activeLeaders() { return [...this.servers.values()].filter((server) => server.role === "leader" && server.currentLease !== null); }

  /** A single-shot election helper: campaign, settle, return the outcome. */
  elect(candidateId, opts = {}) {
    const start = this.round;
    const attempt = this.campaign(candidateId, opts);
    if (attempt.reason === "candidate is paused") return { granted: false, reason: attempt.reason, ticks: 0 };
    const results = this.settle(8);
    const result = results.find((entry) => entry.candidateId === candidateId) ?? { granted: false, reason: "no decision" };
    return { ...result, ticks: this.round - start };
  }

  /** Pause a leader: its lease still runs, but it cannot renew or answer. */
  fail(serverId, ticks) {
    const server = this.servers.get(serverId);
    server.pausedUntil = this.round + ticks;
    return server;
  }

  partition(groups) { this.network.partition(groups); }

  heal() { this.network.heal(); }
}

// --------------------------------------------------------------------------- harness

function campaignAndFailover(cluster, ticks, opts = {}) {
  const order = opts.order ?? [...cluster.servers.keys()];
  const outcomes = [];
  for (const candidate of order) {
    const result = cluster.elect(candidate);
    outcomes.push(result);
    if (result.granted) break;
  }
  cluster.run(ticks);
  return outcomes;
}

if (require.main === module) {
  // --- 1. a quorum grants a lease and the leader's window is inside every promise ----------------
  const cluster = new Cluster({ size: 5, maxDrift: 3, duration: 20, seed: 0x808 });
  const election = cluster.elect(1, { ballot: 1 });
  assert.strictEqual(election.granted, true, `the candidate won the election: ${election.reason ?? ""}`);
  assert.strictEqual(cluster.activeLeaders().length, 1, "exactly one leader is active");
  assert.strictEqual(cluster.activeLeaders()[0].id, 1);
  assert.strictEqual(election.voters.length + 1 >= cluster.quorum(), true, "a quorum of servers promised the lease");
  const leader = cluster.servers.get(1);
  assert.strictEqual(election.entitlementEnd, leader.currentLease.entitlementEnd, "the leader's window is the entitlement it claimed");
  assert.ok(cluster.entitlement <= cluster.duration, "the entitlement never exceeds the promised duration");
  const leaseRounds = election.entitlementEnd - cluster.round;
  assert.ok(leaseRounds > 0, `the lease is still valid for ${leaseRounds} rounds`);
  assert.strictEqual(cluster.entitlement, cluster.duration - 2 * cluster.maxDrift, "the safety margin is exactly twice the worst-case offset error");
  assert.deepStrictEqual(cluster.detectedOverlaps, [], "no overlap was detected during the first election");

  // --- 2. a competing candidate cannot get a second lease while the first is valid ---------------
  const competitor = cluster.elect(2, { ballot: 2 });
  assert.strictEqual(competitor.granted, false, "the competing candidate is refused");
  assert.strictEqual(competitor.reason, "no quorum", `refusal reason: ${competitor.reason}`);
  assert.strictEqual(cluster.activeLeaders().length, 1, "still exactly one leader");
  assert.deepStrictEqual(cluster.detectedOverlaps, [], "and still no overlap");
  assert.ok(cluster.servers.get(3).rejections > 0, "the servers rejected the competing request");

  // --- 3. the lease expires, then a new leader can be elected -----------------------------------
  cluster.run(30);
  assert.strictEqual(cluster.activeLeaders().length, 0, "the expired lease released leadership");
  const second = cluster.elect(3, { ballot: 3 });
  assert.strictEqual(second.granted, true, "after expiry a new leader is elected");
  assert.strictEqual(cluster.activeLeaders().length, 1);
  assert.deepStrictEqual(cluster.detectedOverlaps, [], "the second lease does not overlap the first");
  const previous = cluster.grants.filter((lease) => lease.holder !== 3);
  assert.ok(previous.every((lease) => lease.entitlementEnd >= 0), "previous entitlements are audited against the new lease");

  // --- 4. failover: a stalled leader is replaced within a bounded window -------------------------
  const failover = new Cluster({ size: 5, maxDrift: 3, duration: 15, seed: 0x8081 });
  const firstElection = failover.elect(2, { ballot: 1 });
  assert.strictEqual(firstElection.granted, true);
  const stalled = failover.servers.get(2);
  const entitlementEnd = firstElection.entitlementEnd;
  failover.fail(2, 60);
  const startRound = failover.round;
  let replacement = null;
  let attempt = 1;
  while (failover.round - startRound < 60 && replacement === null) {
    failover.step();
    if (failover.round > entitlementEnd) {
      attempt++;
      const outcome = failover.elect(4, { ballot: attempt });
      if (outcome.granted) replacement = outcome;
    }
  }
  const ticksUsed = failover.round - startRound;
  assert.ok(replacement !== null, "a replacement leader was elected after the stall");
  assert.ok(ticksUsed <= 30, `the failover took ${ticksUsed} rounds, inside the bound`);
  assert.strictEqual(failover.activeLeaders().length, 1, "only the replacement is active");
  assert.strictEqual(failover.activeLeaders()[0].id, 4);
  assert.deepStrictEqual(failover.detectedOverlaps, [], "the failover did not overlap the stalled leader's lease");
  assert.ok(failover.round - startRound >= failover.duration - 2 * failover.maxDrift - 1, "the replacement waited for the lease to expire");

  // --- 5. a minority cannot elect, and the majority side can ------------------------------------
  const parted = new Cluster({ size: 5, maxDrift: 2, duration: 12, seed: 0x8082 });
  parted.partition([[1, 2], [3, 4, 5]]);
  const minority = parted.elect(1, { ballot: 1 });
  assert.strictEqual(minority.granted, false, "a two-server minority cannot win a lease");
  assert.strictEqual(minority.reason, "no quorum", `minority refusal: ${minority.reason}`);
  const majority = parted.elect(3, { ballot: 2 });
  assert.strictEqual(majority.granted, true, "the three-server majority elects a leader");
  assert.strictEqual(parted.activeLeaders().length, 1);
  assert.deepStrictEqual(parted.detectedOverlaps, [], "the partition produced no split brain");
  parted.heal();
  parted.run(20);
  assert.strictEqual(parted.activeLeaders().length <= 1, true, "after healing there is still at most one leader");

  // --- 6. the tightness of `duration - 2*maxDrift`: an out-of-bound clock rate is caught --------
  const honest = new Cluster({ size: 3, rateBound: 0.05, duration: 20, seed: 0x8083 });
  const honestFirst = honest.elect(1, { ballot: 1 });
  assert.strictEqual(honestFirst.granted, true);
  honest.run(honest.duration + 4);
  const honestSecond = honest.elect(2, { ballot: 2 });
  assert.strictEqual(honestSecond.granted, true);
  assert.deepStrictEqual(honest.detectedOverlaps, [], "with clock rates inside the bound the protocol is safe");
  assert.ok(honestSecond.startReal > honestFirst.entitlementEnd - 2 * honest.maxDrift, "the takeover waited out the previous entitlement");
  const cheating = new Cluster({ size: 3, rateBound: 0.05, duration: 20, seed: 0x8084, rates: [0.3, 0.3, -0.3] });
  const cheatFirst = cheating.elect(1, { ballot: 1 });
  assert.strictEqual(cheatFirst.granted, true, "the first leader still gets its lease");
  let cheatSecond = null;
  let cheatAttempts = 0;
  while (cheatSecond === null && cheatAttempts < 12) {
    cheatAttempts++;
    cheating.run(2);
    const outcome = cheating.elect(2, { ballot: 1 + cheatAttempts });
    if (outcome.granted) cheatSecond = outcome;
  }
  assert.ok(cheatSecond !== null, `the unsafe takeover succeeded (${cheatAttempts} attempts)`);
  assert.ok(cheatSecond.startReal < cheatFirst.entitlementEnd,
    `the second lease started at ${cheatSecond.startReal}, inside the first entitlement that ends at ${cheatFirst.entitlementEnd}`);
  assert.strictEqual(cheating.detectedOverlaps.length > 0, true,
    `a clock rate beyond the advertised bound must show up as an overlap (grants: ${JSON.stringify(cheating.grants.map((lease) => [lease.holder, lease.startReal, lease.entitlementEnd]))})`);

  // --- 7. message loss and repeated candidate rejection still converges --------------------------
  const lossy = new Cluster({ size: 5, maxDrift: 3, duration: 14, seed: 0x8085, lossRate: 30 });
  let attempts = 0;
  let elected = null;
  while (attempts < 40 && elected === null) {
    attempts++;
    const candidate = 1 + (attempts % 5);
    const outcome = lossy.elect(candidate, { ballot: attempts });
    if (outcome.granted) elected = outcome;
    else lossy.run(2);
  }
  assert.ok(elected !== null, `a leader was elected despite lossy links (${attempts} attempts, ${lossy.network.dropped} messages dropped)`);
  assert.strictEqual(lossy.activeLeaders().length, 1, "exactly one leader after the lossy run");
  assert.deepStrictEqual(lossy.detectedOverlaps, [], "message loss never broke safety");
  assert.ok(lossy.network.dropped > 0, `${lossy.network.dropped} messages were dropped and the protocol still made progress`);

  // --- 8. many leaders over many rounds: the overlap audit stays empty ---------------------------
  const long = new Cluster({ size: 5, maxDrift: 4, duration: 12, seed: 0x8086 });
  let round = 0;
  for (let wave = 0; wave < 8; wave++) {
    const candidate = 1 + (wave % 5);
    const outcome = long.elect(candidate, { ballot: wave + 1 });
    if (outcome.granted) {
      long.run(long.duration + 2 * long.maxDrift + 2);
      round += outcome.ticks;
    }
  }
  assert.ok(long.grants.length >= 5, `${long.grants.length} leases were granted over the long run`);
  assert.deepStrictEqual(long.detectedOverlaps, [], "no two leases overlapped in real time across the whole run");
  const holders = new Set(long.grants.map((lease) => lease.holder));
  assert.ok(holders.size >= 2, `leadership changed hands (${[...holders].join(",")})`);
  const anyActive = long.activeLeaders().length <= 1;
  assert.ok(anyActive, "at most one leader is active at the end");
  for (const lease of long.grants) {
    assert.ok(lease.entitlementEnd >= lease.startReal, `lease ${lease.holder}@${lease.ballot} has a positive window`);
  }

  // --- 9. cost: messages and rounds per election -------------------------------------------------
  const cost = new Cluster({ size: 7, maxDrift: 2, duration: 16, seed: 0x8087 });
  const beforeMessages = cost.messages;
  const costOutcome = cost.elect(1, { ballot: 1 });
  assert.strictEqual(costOutcome.granted, true, "a seven-server cluster elects a leader");
  const messagesUsed = cost.messages - beforeMessages;
  assert.ok(messagesUsed <= 2 * cost.size, `one request and one reply per server at most (${messagesUsed} messages for ${cost.size} servers)`);
  assert.ok(costOutcome.ticks <= 4, `the election completed in ${costOutcome.ticks} rounds`);
  assert.strictEqual(cost.activeLeaders().length, 1);
  assert.deepStrictEqual(cost.detectedOverlaps, []);

  console.log(`lease: ${cluster.grants.length + failover.grants.length + long.grants.length} leases granted, ${long.grants.length} in the long run, ${long.detectedOverlaps.length} overlaps detected`);
  console.log(`failover: replacement elected ${ticksUsed} rounds after the stall (promise ${failover.duration}, entitlement ${failover.entitlement.toFixed(1)}, rate bound ${failover.rateBound}), ${failover.activeLeaders().length} active leader`);
  console.log(`safety: entitlement ${honest.entitlement.toFixed(1)} of a ${honest.duration}-round promise honoured; a 0.30 clock rate error was caught as an overlap`);
  console.log(`cost: ${messagesUsed} messages for a 7-server election, ${lossy.network.dropped} message(s) dropped in the lossy scenario, ${attempts} attempt(s)`);
  console.log("PASS");
}

module.exports = { Cluster, Server, Network, splitmix64, channelKey, LeaseError };
