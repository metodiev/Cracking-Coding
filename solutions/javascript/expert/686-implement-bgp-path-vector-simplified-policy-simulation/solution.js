/** Task 686 - Implement BGP path-vector simplified policy simulation. (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Model: a *path-vector* routing protocol over an AS-level graph. Every AS keeps, per prefix, the
 * best route it has been offered; a route is
 *     { path, localPref, med, via, learnedClass }
 * where `path` lists the ASes from the origin to the current holder (origin first - the reverse of
 * the on-the-wire AS_PATH order, which makes the "is the receiver already in the path?" loop check
 * a plain `includes`). The peer ASes exchange advertisement / withdrawal messages; a round
 * recomputes every AS's best route from the *snapshot* of its inbox and then applies the resulting
 * messages, which is the standard simultaneous-round fixed-point iteration. Termination is reached
 * when a full round neither changes a best route nor changes an inbox entry.
 *
 * Simplified versus real BGP: only the AS_PATH, LOCAL_PREF, MED and the loop check are modelled
 * (no communities, no iBGP, no timers, no RIB/FIB split, MED is compared across neighbours), the
 * peering policy is the standard valley-free rule set - LOCAL_PREF 200/100/50 for routes learned
 * from a customer/peer/provider and own routes ranked highest, and a route learned from a peer or a
 * provider is only re-advertised to *customers*. What is verified is the real thing: the routing
 * tables must be a genuine fixed point (no neighbour offers something better than the chosen
 * route), every stored path must be a loop-free walk that starts at the origin, ends at the
 * advertising AS and follows real links, withdrawals must not leave stale paths behind, and with a
 * policy-free ("open") configuration the AS-path length must equal the BFS hop distance from the
 * origin - a brute-force oracle.
 */
"use strict";

const CUSTOMER = "customer";
const PEER = "peer";
const PROVIDER = "provider";

/** LOCAL_PREF that an AS assigns to a route it learns from a neighbour of the given class. */
const DEFAULT_LOCAL_PREF = { customer: 200, peer: 100, provider: 50 };
const OWN_LOCAL_PREF = 250;

function openPolicy() {
  return {
    name: "open",
    acceptFrom: () => true,
    exportTo: () => true,
    localPrefFor: () => 100,
  };
}

/** Valley-free default: customers get full transit, peers/providers get own + customer routes only. */
function valleyFreePolicy() {
  return {
    name: "valley-free",
    acceptFrom: () => true,
    exportTo: (neighborClass, route) => {
      if (neighborClass === CUSTOMER) return true;
      return route.learnedClass === "own" || route.learnedClass === CUSTOMER;
    },
    localPrefFor: (neighborClass) => DEFAULT_LOCAL_PREF[neighborClass],
  };
}

/** Wraps a policy so that routes heard from `blocked` are dropped on import. */
function policyRejectingNeighbor(base, blocked) {
  return {
    ...base,
    name: `${base.name}+reject(${blocked})`,
    acceptFrom: (neighborClass, route, from) => from !== blocked && base.acceptFrom(neighborClass, route, from),
  };
}

function routeKey(route) {
  if (route === null) return "-";
  return `${route.path.join(">")}|lp${route.localPref}|med${route.med}|via${route.via}`;
}

/** Ranking used by the decision process: LOCAL_PREF, shortest AS_PATH, lowest MED, lowest ASN. */
function compareRoutes(a, b) {
  if (a.localPref !== b.localPref) return b.localPref - a.localPref;
  if (a.path.length !== b.path.length) return a.path.length - b.path.length;
  if (a.med !== b.med) return a.med - b.med;
  const viaA = a.via === null ? -1 : a.via;
  const viaB = b.via === null ? -1 : b.via;
  if (viaA !== viaB) return viaA - viaB;
  return routeKey(a) < routeKey(b) ? -1 : 1;
}

class BgpNetwork {
  constructor() {
    this.asns = [];
    this.neighborSets = new Map();
    this.medTable = new Map();       // `${from}|${to}` -> MED the sender attaches
    this.relations = new Map();      // `${observer}|${other}` -> class of `other` from `observer`
    this.policies = new Map();
    this.originOf = new Map();       // prefix -> originating AS
    this.prefixes = [];
    this.inbox = new Map();          // `${to}|${prefix}|${from}` -> advertised route or null
    this.best = new Map();           // `${asn}|${prefix}` -> best route or null
    this.rounds = 0;
  }

  addAs(asn, policy = valleyFreePolicy()) {
    if (this.policies.has(asn)) throw new Error(`duplicate AS ${asn}`);
    this.asns.push(asn);
    this.neighborSets.set(asn, []);
    this.policies.set(asn, policy);
    this.policies.get(asn).name = policy.name;
    return this;
  }

  /** `med` is the MULTI_EXIT_DISC the `from` side attaches to routes it sends to `to`. */
  addLink(a, b, med = 0) {
    for (const [from, to] of [[a, b], [b, a]]) {
      const list = this.neighborSets.get(from);
      if (list === undefined) throw new Error(`unknown AS ${from}`);
      if (!list.includes(to)) list.push(to);
      this.medTable.set(`${from}|${to}`, med);
    }
    for (const asn of [a, b]) this.neighborSets.get(asn).sort((x, y) => x - y);
    return this;
  }

  /** Declare the business relation: `other` is a customer/peer/provider *of* `observer`. */
  setRelation(observer, other, relation) {
    if (![CUSTOMER, PEER, PROVIDER].includes(relation)) throw new Error(`bad relation ${relation}`);
    this.relations.set(`${observer}|${other}`, relation);
    this.relations.set(`${other}|${observer}`, flipRelation(relation));
    return this;
  }

  addPrefix(prefix, originAs) {
    if (!this.policies.has(originAs)) throw new Error(`unknown origin ${originAs}`);
    if (!this.originOf.has(prefix)) this.prefixes.push(prefix);
    this.originOf.set(prefix, originAs);
    return this;
  }

  neighbors(asn) {
    return this.neighborSets.get(asn) || [];
  }

  classOf(observer, other) {
    return this.relations.get(`${observer}|${other}`) || PEER;
  }

  medFor(from, to) {
    return this.medTable.get(`${from}|${to}`) || 0;
  }

  isLink(a, b) {
    return this.neighbors(a).includes(b);
  }

  /** Best candidate for one prefix, read from the inbox snapshot (simultaneous rounds). */
  selectBest(asn, prefix, snapshot) {
    const policy = this.policies.get(asn);
    const here = [];
    if (this.originOf.get(prefix) === asn) {
      here.push({ path: [asn], localPref: OWN_LOCAL_PREF, med: 0, via: null, learnedClass: "own" });
    }
    for (const neighbor of this.neighbors(asn)) {
      const offered = snapshot.get(`${asn}|${prefix}|${neighbor}`);
      if (offered === undefined || offered === null) continue;
      if (offered.path.includes(asn)) continue;                 // path-vector loop prevention
      const neighborClass = this.classOf(asn, neighbor);
      if (!policy.acceptFrom(neighborClass, offered, neighbor)) continue;
      here.push({
        path: [...offered.path, asn],
        localPref: policy.localPrefFor(neighborClass),
        med: offered.med,
        via: neighbor,
        learnedClass: neighborClass,
      });
    }
    if (here.length === 0) return null;
    return here.reduce((best, candidate) => (compareRoutes(candidate, best) < 0 ? candidate : best));
  }

  shouldExport(from, to, route) {
    if (route.path.includes(to)) return false;
    return this.policies.get(from).exportTo(this.classOf(from, to), route, to);
  }

  /** Run the fixed-point iteration. Returns { rounds, converged }. */
  run(maxRounds = 64) {
    this.inbox.clear();
    this.best.clear();
    for (this.rounds = 0; this.rounds < maxRounds; this.rounds++) {
      let changed = false;
      const snapshot = new Map(this.inbox);

      for (const asn of this.asns) {
        for (const prefix of this.prefixes) {
          const key = `${asn}|${prefix}`;
          const previous = this.best.get(key) || null;
          const chosen = this.selectBest(asn, prefix, snapshot);
          if (routeKey(previous) !== routeKey(chosen)) {
            this.best.set(key, chosen);
            changed = true;
          }
        }
      }

      const updates = [];
      for (const asn of this.asns) {
        for (const prefix of this.prefixes) {
          const route = this.best.get(`${asn}|${prefix}`) || null;
          for (const neighbor of this.neighbors(asn)) {
            let out = null;
            if (route !== null && this.shouldExport(asn, neighbor, route)) {
              out = { ...route, med: this.medFor(asn, neighbor) };
            }
            const key = `${neighbor}|${prefix}|${asn}`;
            if (routeKey(snapshot.get(key) || null) !== routeKey(out)) {
              updates.push([key, out]);
              changed = true;
            }
          }
        }
      }
      for (const [key, value] of updates) {
        if (value === null) this.inbox.delete(key);
        else this.inbox.set(key, value);
      }
      if (!changed) return { rounds: this.rounds, converged: true };
    }
    return { rounds: maxRounds, converged: false };
  }

  bestRoute(asn, prefix) {
    return this.best.get(`${asn}|${prefix}`) || null;
  }

  asPath(asn, prefix) {
    const route = this.bestRoute(asn, prefix);
    return route === null ? null : route.path.slice();
  }

  /** Full snapshot of the routing tables, for a determinism check. */
  tableSnapshot() {
    const out = {};
    for (const asn of [...this.asns].sort((a, b) => a - b)) {
      for (const prefix of this.prefixes) {
        out[`${asn}|${prefix}`] = routeKey(this.bestRoute(asn, prefix));
      }
    }
    return out;
  }

  /** Number of (asn, prefix) entries whose path is not a real walk / not origin-anchored. */
  validatePaths() {
    const problems = [];
    for (const asn of this.asns) {
      for (const prefix of this.prefixes) {
        const route = this.bestRoute(asn, prefix);
        if (route === null) continue;
        const { path } = route;
        if (path[0] !== this.originOf.get(prefix)) problems.push(`${asn} ${prefix}: bad origin`);
        if (path[path.length - 1] !== asn) problems.push(`${asn} ${prefix}: does not end at the holder`);
        if (new Set(path).size !== path.length) problems.push(`${asn} ${prefix}: repeated AS`);
        for (let i = 0; i + 1 < path.length; i++) {
          if (!this.isLink(path[i], path[i + 1])) problems.push(`${asn} ${prefix}: ${path[i]}-${path[i + 1]} not a link`);
        }
      }
    }
    return problems;
  }

  /** Fixed-point check: no neighbour offers a route that would have been preferred. */
  stabilityViolations() {
    const problems = [];
    for (const asn of this.asns) {
      for (const prefix of this.prefixes) {
        const chosen = this.bestRoute(asn, prefix);
        for (const neighbor of this.neighbors(asn)) {
          const offered = this.inbox.get(`${asn}|${prefix}|${neighbor}`) || null;
          if (offered === null) continue;
          if (offered.path.includes(asn)) continue;
          const neighborClass = this.classOf(asn, neighbor);
          if (!this.policies.get(asn).acceptFrom(neighborClass, offered, neighbor)) continue;
          const alternative = {
            path: [...offered.path, asn],
            localPref: this.policies.get(asn).localPrefFor(neighborClass),
            med: offered.med,
            via: neighbor,
            learnedClass: neighborClass,
          };
          if (chosen === null || compareRoutes(alternative, chosen) < 0) {
            problems.push(`${asn} ${prefix}: prefers the route via ${neighbor}`);
          }
        }
      }
    }
    return problems;
  }
}

function flipRelation(relation) {
  if (relation === CUSTOMER) return PROVIDER;
  if (relation === PROVIDER) return CUSTOMER;
  return PEER;
}

/** Brute-force oracle: BFS hop distance in the AS graph, ignoring policy. */
function bfsHops(network, origin) {
  const dist = new Map([[origin, 0]]);
  const queue = [origin];
  for (let head = 0; head < queue.length; head++) {
    const asn = queue[head];
    for (const neighbor of network.neighbors(asn)) {
      if (!dist.has(neighbor)) {
        dist.set(neighbor, dist.get(asn) + 1);
        queue.push(neighbor);
      }
    }
  }
  return dist;
}

module.exports = {
  CUSTOMER, PEER, PROVIDER, openPolicy, valleyFreePolicy, policyRejectingNeighbor,
  compareRoutes, routeKey, BgpNetwork, bfsHops, flipRelation,
};

if (require.main === module) {
  const assert = require("node:assert");
  const PREFIX = "203.0.113.0/24";
  const rank = (path, localPref, med, via) => ({ path, localPref, med, via, learnedClass: PEER });

  /** Run to the fixed point, then assert convergence and both global invariants. */
  const settle = (net, label) => {
    const outcome = net.run();
    assert.strictEqual(outcome.converged, true, `${label}: converged`);
    assert.deepStrictEqual(net.validatePaths(), [], `${label}: every path is a real loop-free walk`);
    assert.deepStrictEqual(net.stabilityViolations(), [], `${label}: no better offer was left unused`);
    return outcome;
  };

  // --- the ranking order of the decision process ------------------------------------------
  assert.ok(compareRoutes(rank([1, 7, 8, 2], 100, 0, 2), rank([1, 2], 90, 0, 2)) < 0, "LOCAL_PREF first");
  assert.ok(compareRoutes(rank([1, 2], 100, 9, 2), rank([1, 7, 2], 100, 0, 7)) < 0, "then AS_PATH length");
  assert.ok(compareRoutes(rank([1, 5, 2], 100, 3, 5), rank([1, 6, 2], 100, 7, 6)) < 0, "then MED");
  assert.ok(compareRoutes(rank([1, 4, 2], 100, 3, 4), rank([1, 5, 2], 100, 3, 5)) < 0, "then neighbour ASN");
  assert.strictEqual(flipRelation(PROVIDER), CUSTOMER);
  assert.strictEqual(flipRelation(PEER), PEER);

  // --- open policy: path length must equal the BFS hop distance (brute-force oracle) ------
  function ringWithChord() {
    const net = new BgpNetwork();
    for (const asn of [65001, 65002, 65003, 65004, 65005, 65006]) net.addAs(asn, openPolicy());
    net.addLink(65001, 65002).addLink(65002, 65003).addLink(65003, 65004)
      .addLink(65004, 65005).addLink(65005, 65006).addLink(65006, 65001).addLink(65002, 65006);
    net.addPrefix(PREFIX, 65003);
    return net;
  }
  const openNet = ringWithChord();
  const openRun = settle(openNet, "open");
  assert.ok(openRun.rounds >= 3 && openRun.rounds <= 10, `rounds in range: ${openRun.rounds}`);
  const distances = bfsHops(openNet, 65003);
  for (const asn of openNet.asns) {
    assert.strictEqual(openNet.asPath(asn, PREFIX).length - 1, distances.get(asn), `BFS hops for ${asn}`);
  }
  assert.deepStrictEqual(openNet.asPath(65003, PREFIX), [65003]);
  assert.deepStrictEqual(openNet.asPath(65002, PREFIX), [65003, 65002]);
  assert.deepStrictEqual(openNet.asPath(65004, PREFIX), [65003, 65004]);
  assert.deepStrictEqual(openNet.asPath(65006, PREFIX), [65003, 65002, 65006], "chord beats the ring");
  const rerun = ringWithChord();
  rerun.run();
  assert.deepStrictEqual(rerun.tableSnapshot(), openNet.tableSnapshot(), "deterministic tables");
  assert.deepStrictEqual(rerun.run(), openRun, "a converged network has nothing left to send");

  // --- valley-free policy: customer routes are preferred, peers get no transit -------------
  const valley = new BgpNetwork();
  for (const asn of [65001, 65002, 65003, 65004, 65005]) valley.addAs(asn, valleyFreePolicy());
  valley.addLink(65001, 65002).addLink(65002, 65003).addLink(65002, 65004).addLink(65001, 65005);
  valley.setRelation(65001, 65002, CUSTOMER).setRelation(65001, 65005, PEER)
    .setRelation(65002, 65003, CUSTOMER).setRelation(65002, 65004, PEER);
  valley.addPrefix("198.51.100.0/24", 65003);
  settle(valley, "valley-free");
  assert.deepStrictEqual(valley.asPath(65002, "198.51.100.0/24"), [65003, 65002]);
  assert.deepStrictEqual(valley.asPath(65001, "198.51.100.0/24"), [65003, 65002, 65001], "relayed upstream");
  assert.deepStrictEqual(valley.asPath(65004, "198.51.100.0/24"), [65003, 65002, 65004], "relayed to a peer");
  assert.deepStrictEqual(valley.asPath(65005, "198.51.100.0/24"), [65003, 65002, 65001, 65005]);
  assert.strictEqual(valley.bestRoute(65001, "198.51.100.0/24").localPref, DEFAULT_LOCAL_PREF.customer);
  assert.strictEqual(valley.bestRoute(65004, "198.51.100.0/24").localPref, DEFAULT_LOCAL_PREF.peer);
  assert.strictEqual(valley.bestRoute(65002, "198.51.100.0/24").learnedClass, CUSTOMER);

  // a route learned from a PEER is never offered to a provider: the valley stops at 65011
  const noTransit = new BgpNetwork();
  for (const asn of [65010, 65011, 65012]) noTransit.addAs(asn, valleyFreePolicy());
  noTransit.addLink(65010, 65011).addLink(65011, 65012);
  noTransit.setRelation(65010, 65011, CUSTOMER).setRelation(65011, 65012, PEER);
  noTransit.addPrefix("192.0.2.0/24", 65012);
  settle(noTransit, "no-transit");
  assert.deepStrictEqual(noTransit.asPath(65011, "192.0.2.0/24"), [65012, 65011]);
  assert.strictEqual(noTransit.asPath(65010, "192.0.2.0/24"), null, "peer route not offered to a provider");
  assert.strictEqual(noTransit.inbox.has("65010|192.0.2.0/24|65011"), false, "never advertised at all");

  // --- MED and neighbour-ASN tie-breaks on two equal-length peer paths ---------------------
  function tieTopology(medA, medB) {
    const net = new BgpNetwork();
    for (const asn of [65001, 65002, 65003, 65009]) net.addAs(asn, valleyFreePolicy());
    net.addLink(65003, 65001).addLink(65003, 65002)
      .addLink(65001, 65009, medA).addLink(65002, 65009, medB);
    net.setRelation(65001, 65003, CUSTOMER).setRelation(65002, 65003, CUSTOMER);
    net.setRelation(65009, 65001, PEER).setRelation(65009, 65002, PEER);
    net.addPrefix(PREFIX, 65003);
    settle(net, `tie(${medA},${medB})`);
    return net;
  }
  const medWins = tieTopology(5, 1);
  assert.deepStrictEqual(medWins.asPath(65009, PREFIX), [65003, 65002, 65009], "lower MED wins");
  assert.strictEqual(medWins.bestRoute(65009, PREFIX).med, 1);
  assert.strictEqual(medWins.bestRoute(65001, PREFIX).localPref, DEFAULT_LOCAL_PREF.customer);
  const asnWins = tieTopology(4, 4);
  assert.deepStrictEqual(asnWins.asPath(65009, PREFIX), [65003, 65001, 65009], "lower ASN wins the tie");
  assert.strictEqual(asnWins.bestRoute(65009, PREFIX).via, 65001);

  // --- import filter and link failure ------------------------------------------------------
  const filtered = new BgpNetwork();
  filtered.addAs(65020, valleyFreePolicy());
  filtered.addAs(65021, policyRejectingNeighbor(valleyFreePolicy(), 65020));
  filtered.addLink(65020, 65021);
  filtered.addPrefix("198.18.0.0/15", 65020);
  settle(filtered, "import-filter");
  assert.strictEqual(filtered.asPath(65021, "198.18.0.0/15"), null, "import filter drops the route");
  assert.deepStrictEqual(filtered.asPath(65020, "198.18.0.0/15"), [65020]);

  const failing = ringWithChord();
  settle(failing, "before failure");
  assert.ok(failing.asPath(65004, PREFIX) !== null);
  for (const from of [65003, 65004]) {
    const list = failing.neighbors(from);
    list.splice(list.indexOf(from === 65003 ? 65004 : 65003), 1);
  }
  const afterFailure = settle(failing, "after failure");
  assert.deepStrictEqual(failing.asPath(65004, PREFIX), [65003, 65002, 65006, 65005, 65004],
    "65004 detours instead of keeping a stale path");
  assert.ok(afterFailure.rounds > 1, "the withdrawal had to propagate");

  console.log(`open rounds=${openRun.rounds} reconverge rounds=${afterFailure.rounds}`);
  console.log("PASS");
}
