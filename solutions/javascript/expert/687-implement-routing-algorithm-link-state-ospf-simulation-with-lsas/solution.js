/** Task 687 - Implement routing algorithm link-state (OSPF) simulation with LSAs. (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Model: link-state routing. Each router originates a Router-LSA { routerId, seq, links } listing
 * its interfaces with their costs. Every router owns an LSDB and floods LSAs hop by hop: on
 * accepting a *newer* LSA it re-floods it to all neighbours except the one it arrived from, at most
 * once per (routerId, seq) - which is why a converged flood costs exactly one transmission per
 * router-side link except the reverse edges, i.e. 2E - (V - 1) sends on a connected graph. The LSDB
 * keeps only the highest sequence number per router and drops stale or duplicate copies. Dijkstra's
 * SPF then runs on the graph rebuilt from a router's own LSDB, producing { cost, nextHop,
 * predecessor } with a deterministic tie-break: on equal cost the path through the smallest
 * forwarding id wins. The metric is cost, not hop count, so the chosen route may have more hops than
 * a rejected candidate.
 *
 * Simplified versus real OSPF: no areas, no DR/BDR election, no LSA aging / MaxAge / DoNotAge, no
 * retransmission timers (flooding is instantaneous and deduplicated), no equal-cost multipath.
 * Verified: sequence-number monotonicity, the flooding invariant above plus LSDB convergence across
 * all routers, agreement of every routing table with a Floyd-Warshall all-pairs oracle, cost sum and
 * loop-freedom of the paths recovered from the predecessor tree, the forwarding-table/path
 * consistency, the cost-beats-hops property, equal-cost tie-breaks, and re-convergence after a cost
 * change, a link failure and a partition.
 */
"use strict";

const INFINITY_COST = Number.POSITIVE_INFINITY;

class Topology {
  constructor() {
    this.links = new Map();          // routerId -> Map(neighbourId -> cost)
  }

  addRouter(id) {
    if (!this.links.has(id)) this.links.set(id, new Map());
    return this;
  }

  addLink(a, b, cost) {
    if (!(cost > 0)) throw new RangeError(`link cost must be positive: ${cost}`);
    this.addRouter(a).addRouter(b);
    this.links.get(a).set(b, cost);
    this.links.get(b).set(a, cost);
    return this;
  }

  removeLink(a, b) {
    this.links.get(a).delete(b);
    this.links.get(b).delete(a);
    return this;
  }

  routers() {
    return [...this.links.keys()].sort((x, y) => x - y);
  }

  neighbours(id) {
    return [...this.links.get(id).entries()].sort((p, q) => p[0] - q[0]);
  }

  degree(id) {
    return this.links.get(id).size;
  }

  edgeCount() {
    let total = 0;
    for (const [, map] of this.links) total += map.size;
    return total / 2;
  }

  /** The Router-LSA this router originates for the current topology. */
  lsaOf(id, seq) {
    return { routerId: id, seq, links: this.neighbours(id).map(([neighbour, cost]) => ({ neighbour, cost })) };
  }
}

/** Per-router Link State Database: the highest sequence number per router wins. */
class Lsdb {
  constructor() {
    this.entries = new Map();
  }

  install(lsa) {
    const current = this.entries.get(lsa.routerId);
    if (current === undefined || lsa.seq > current.seq) {
      this.entries.set(lsa.routerId, { ...lsa, links: lsa.links.map((link) => ({ ...link })) });
      return "newer";
    }
    return lsa.seq === current.seq ? "same" : "stale";
  }

  get(id) {
    return this.entries.get(id) || null;
  }

  sequenceOf(id) {
    const entry = this.get(id);
    return entry === null ? -1 : entry.seq;
  }

  /** Undirected adjacency from the installed LSAs; a link is usable only if both sides agree. */
  adjacency() {
    const adj = new Map();
    for (const id of this.entries.keys()) adj.set(id, new Map());
    for (const [id, lsa] of this.entries) {
      for (const { neighbour, cost } of lsa.links) {
        if (!adj.has(neighbour)) continue;                       // neighbour has no LSA here
        const back = this.get(neighbour);
        if (back === null) continue;
        const reverse = back.links.find((link) => link.neighbour === id);
        if (reverse === undefined) continue;
        adj.get(id).set(neighbour, Math.min(cost, reverse.cost));
      }
    }
    return adj;
  }

  signature() {
    return [...this.entries.entries()]
      .sort((p, q) => p[0] - q[0])
      .map(([id, lsa]) => `${id}:${lsa.seq}:[${lsa.links.map((l) => `${l.neighbour}@${l.cost}`).join(",")}]`)
      .join(";");
  }
}

function createLsdbs(topology) {
  const lsdbs = new Map();
  for (const id of topology.routers()) lsdbs.set(id, new Lsdb());
  return lsdbs;
}

/** Reliable flooding of one LSA into per-router databases. Returns transmission statistics. */
function floodLsa(topology, lsdbs, lsa, origin) {
  const transmissions = [];
  const flooded = new Set([`${origin}|${lsa.routerId}|${lsa.seq}`]);
  lsdbs.get(origin).install(lsa);
  const queue = [[origin, null]];

  while (queue.length > 0) {
    const [router, cameFrom] = queue.shift();
    for (const [neighbour] of topology.neighbours(router)) {
      if (neighbour === cameFrom) continue;                       // never bounce back to the sender
      transmissions.push([router, neighbour]);
      const key = `${neighbour}|${lsa.routerId}|${lsa.seq}`;
      if (flooded.has(key)) continue;                             // this router already handled it
      flooded.add(key);
      if (lsdbs.get(neighbour).install(lsa) === "newer") queue.push([neighbour, router]);
    }
  }

  const transmissionsBy = new Map();
  for (const [from] of transmissions) transmissionsBy.set(from, (transmissionsBy.get(from) || 0) + 1);
  return { transmissions: transmissions.length, transmissionsBy, holders: [...lsdbs.keys()].sort((x, y) => x - y) };
}

/** Flood every router's own LSA once; the result is the converged network state. */
function syncLsdbs(topology, seq = 1) {
  const lsdbs = createLsdbs(topology);
  for (const id of topology.routers()) floodLsa(topology, lsdbs, topology.lsaOf(id, seq), id);
  return lsdbs;
}

function allLsdbsAgree(lsdbs) {
  const signatures = new Set([...lsdbs.values()].map((db) => db.signature()));
  return signatures.size === 1;
}

/** Dijkstra SPF with lowest-id tie-break on equal cost; nextHop is the first hop from `source`. */
function spf(adjacency, source) {
  const ids = [...adjacency.keys()].sort((x, y) => x - y);
  const dist = new Map([[source, 0]]);
  const nextHop = new Map();
  const predecessor = new Map();
  const visited = new Set();

  for (;;) {
    let current = null;
    for (const id of ids) {
      if (visited.has(id) || !dist.has(id)) continue;
      if (current === null) { current = id; continue; }
      const candidate = dist.get(id);
      const best = dist.get(current);
      if (candidate < best || (candidate === best && id < current)) current = id;
    }
    if (current === null) break;
    visited.add(current);
    for (const [neighbour, cost] of [...adjacency.get(current).entries()].sort((p, q) => p[0] - q[0])) {
      const candidateCost = dist.get(current) + cost;
      const candidateHop = current === source ? neighbour : nextHop.get(current);
      const known = dist.get(neighbour);
      const better = known === undefined
        || candidateCost < known
        || (candidateCost === known && candidateHop < nextHop.get(neighbour));
      if (better) {
        dist.set(neighbour, candidateCost);
        nextHop.set(neighbour, candidateHop);
        predecessor.set(neighbour, current);
      }
    }
  }
  return { source, dist, nextHop, predecessor };
}

/** Recover the path by walking the predecessor tree back to `source`; null when unreachable. */
function routePath(result, destination) {
  if (!result.dist.has(destination)) return null;
  const path = [destination];
  const seen = new Set(path);
  let current = destination;
  while (current !== result.source) {
    const previous = result.predecessor.get(current);
    if (previous === undefined || seen.has(previous)) return null;
    path.push(previous);
    seen.add(previous);
    current = previous;
  }
  return path.reverse();
}

/** Oracle: Floyd-Warshall over the same adjacency. */
function allPairsShortest(adjacency) {
  const ids = [...adjacency.keys()].sort((x, y) => x - y);
  const dist = new Map(ids.map((id) => [id, new Map(ids.map((other) => [other, id === other ? 0 : INFINITY_COST]))]));
  for (const id of ids) {
    for (const [neighbour, cost] of adjacency.get(id)) dist.get(id).set(neighbour, Math.min(dist.get(id).get(neighbour), cost));
  }
  for (const k of ids) {
    for (const i of ids) {
      for (const j of ids) {
        const through = dist.get(i).get(k) + dist.get(k).get(j);
        if (through < dist.get(i).get(j)) dist.get(i).set(j, through);
      }
    }
  }
  return dist;
}

function pathCost(adjacency, path) {
  if (path === null) return INFINITY_COST;
  let total = 0;
  for (let i = 0; i + 1 < path.length; i++) {
    const cost = adjacency.get(path[i]).get(path[i + 1]);
    if (cost === undefined) return INFINITY_COST;
    total += cost;
  }
  return total;
}

class Lcg {
  constructor(seed) {
    this.state = seed >>> 0;
  }

  next() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state;
  }

  int(bound) {
    return this.next() % bound;
  }
}

module.exports = {
  INFINITY_COST, Topology, Lsdb, createLsdbs, floodLsa, syncLsdbs, allLsdbsAgree,
  spf, routePath, allPairsShortest, pathCost, Lcg,
};

if (require.main === module) {
  const assert = require("node:assert");

  /** R1 -4- R2, R1 -2- R3, R3 -1- R2, R2 -5- R4, R3 -8- R4, R4 -3- R5, R2 -6- R5. */
  function referenceTopology() {
    return new Topology()
      .addLink(1, 2, 4).addLink(1, 3, 2).addLink(3, 2, 1)
      .addLink(2, 4, 5).addLink(3, 4, 8).addLink(4, 5, 3).addLink(2, 5, 6);
  }

  // --- hand-derived SPF ---------------------------------------------------------------------
  const topology = referenceTopology();
  const lsdbs = syncLsdbs(topology);
  assert.ok(allLsdbsAgree(lsdbs), "every router's LSDB converged to the same content");
  assert.strictEqual(lsdbs.get(1).adjacency().get(1).get(2), 4);
  assert.strictEqual(lsdbs.get(5).adjacency().get(2).get(1), 4, "costs are symmetric");

  const fromOne = spf(lsdbs.get(1).adjacency(), 1);
  assert.deepStrictEqual([...fromOne.dist.entries()].sort((a, b) => a[0] - b[0]),
    [[1, 0], [2, 3], [3, 2], [4, 8], [5, 9]], "hand-computed costs");
  assert.strictEqual(fromOne.nextHop.get(2), 3, "R1 reaches R2 through R3, not directly");
  assert.strictEqual(fromOne.nextHop.get(3), 3);
  assert.strictEqual(fromOne.nextHop.get(4), 3, "forwarding entry for R4 points at R3");
  assert.strictEqual(fromOne.nextHop.get(5), 3);
  assert.deepStrictEqual(routePath(fromOne, 5), [1, 3, 2, 5], "cost metric picks the 4-AS path");
  assert.deepStrictEqual(routePath(fromOne, 4), [1, 3, 2, 4]);
  assert.strictEqual(routePath(fromOne, 5)[1], fromOne.nextHop.get(5), "first hop matches the path");
  assert.strictEqual(pathCost(lsdbs.get(1).adjacency(), routePath(fromOne, 5)), 9);
  assert.deepStrictEqual(routePath(fromOne, 1), [1], "route to self is the empty path");
  assert.strictEqual(routePath(fromOne, 999), null, "unknown destination");
  assert.strictEqual(spf(lsdbs.get(1).adjacency(), 1).dist.get(1), 0);

  // --- cost beats hop count -----------------------------------------------------------------
  const adjacency = lsdbs.get(1).adjacency();
  const cheapButLong = routePath(fromOne, 5);
  const fewerHops = [1, 2, 5];
  assert.strictEqual(fewerHops.length - 1, 2, "the direct path has fewer hops");
  assert.strictEqual(cheapButLong.length - 1, 3, "the chosen path has more hops");
  assert.ok(pathCost(adjacency, fewerHops) > pathCost(adjacency, cheapButLong), "fewer hops but pricier loses");

  // --- all routing tables against the Floyd-Warshall oracle ---------------------------------
  const oracle = allPairsShortest(adjacency);
  let tableEntries = 0;
  for (const source of topology.routers()) {
    const result = spf(lsdbs.get(source).adjacency(), source);
    for (const destination of topology.routers()) {
      tableEntries++;
      assert.strictEqual(result.dist.get(destination), oracle.get(source).get(destination),
        `cost ${source}->${destination}`);
      const path = routePath(result, destination);
      assert.ok(path !== null, `path ${source}->${destination}`);
      assert.strictEqual(pathCost(adjacency, path), oracle.get(source).get(destination),
        `path cost ${source}->${destination}`);
      assert.strictEqual(new Set(path).size, path.length, `loop-free ${source}->${destination}`);
      if (destination !== source) {
        assert.strictEqual(result.nextHop.get(destination), path[1], `forwarding entry ${source}->${destination}`);
        assert.strictEqual(result.dist.get(path[1]), adjacency.get(source).get(path[1]), "first hop is a neighbour");
      }
    }
  }
  assert.strictEqual(tableEntries, 25);
  for (const a of topology.routers()) {
    for (const b of topology.routers()) {
      for (const c of topology.routers()) {
        assert.ok(oracle.get(a).get(c) <= oracle.get(a).get(b) + oracle.get(b).get(c), `triangle ${a},${b},${c}`);
      }
    }
  }

  // --- equal-cost tie-break picks the lowest forwarding id ----------------------------------
  const diamond = new Topology().addLink(1, 2, 1).addLink(1, 3, 1).addLink(2, 4, 1).addLink(3, 4, 1);
  const diamondSpf = spf(syncLsdbs(diamond).get(1).adjacency(), 1);
  assert.strictEqual(diamondSpf.dist.get(4), 2);
  assert.strictEqual(diamondSpf.nextHop.get(4), 2, "lower router id wins the tie");
  assert.deepStrictEqual(routePath(diamondSpf, 4), [1, 2, 4]);

  // --- LSA sequence numbers -----------------------------------------------------------------
  const sequenceDb = new Lsdb();
  assert.strictEqual(sequenceDb.install({ routerId: 9, seq: 5, links: [] }), "newer");
  assert.strictEqual(sequenceDb.install({ routerId: 9, seq: 4, links: [] }), "stale");
  assert.strictEqual(sequenceDb.install({ routerId: 9, seq: 5, links: [] }), "same");
  assert.strictEqual(sequenceDb.sequenceOf(9), 5);
  assert.strictEqual(sequenceDb.install({ routerId: 9, seq: 6, links: [{ neighbour: 1, cost: 3 }] }), "newer");
  assert.strictEqual(sequenceDb.sequenceOf(9), 6, "sequence numbers stay monotone");
  assert.strictEqual(sequenceDb.install({ routerId: 9, seq: 6, links: [{ neighbour: 2, cost: 9 }] }), "same");
  assert.strictEqual(sequenceDb.get(9).links[0].neighbour, 1, "duplicate does not overwrite the entry");

  // --- reliable flooding invariant ----------------------------------------------------------
  const floodTopology = referenceTopology();
  const floodDbs = syncLsdbs(floodTopology);
  const refire = floodLsa(floodTopology, floodDbs, floodTopology.lsaOf(1, 2), 1);
  assert.deepStrictEqual(refire.holders, [1, 2, 3, 4, 5], "every router holds the interface");
  assert.strictEqual(refire.transmissions, 2 * floodTopology.edgeCount() - (floodTopology.routers().length - 1),
    "one re-flood per router: 2E - (V-1) transmissions");
  assert.strictEqual(refire.transmissions, 10);
  assert.strictEqual(refire.transmissionsBy.get(1), floodTopology.degree(1), "origin floods to all neighbours");
  assert.strictEqual(refire.transmissionsBy.get(5), floodTopology.degree(5) - 1, "leaf floods to all but the sender");
  assert.strictEqual(refire.transmissionsBy.get(2), floodTopology.degree(2) - 1, "R2 re-floods only once");
  assert.ok(allLsdbsAgree(floodDbs), "LSDBs agree again after flooding");
  for (const id of floodTopology.routers()) assert.strictEqual(floodDbs.get(id).sequenceOf(1), 2, `seq at ${id}`);
  const duplicate = floodLsa(floodTopology, floodDbs, floodTopology.lsaOf(1, 2), 1);
  assert.strictEqual(duplicate.transmissions, floodTopology.degree(1), "a duplicate stops immediately");
  assert.strictEqual(floodDbs.get(4).sequenceOf(1), 2);

  // --- cost change -------------------------------------------------------------------------
  const cheaper = referenceTopology();
  const cheaperDbs = syncLsdbs(cheaper);
  assert.strictEqual(spf(cheaperDbs.get(1).adjacency(), 1).dist.get(4), 8);
  cheaper.links.get(1).set(3, 1);              // R1-R3 drops from 2 to 1
  cheaper.links.get(3).set(1, 1);
  floodLsa(cheaper, cheaperDbs, cheaper.lsaOf(1, 2), 1);
  floodLsa(cheaper, cheaperDbs, cheaper.lsaOf(3, 2), 3);
  const cheaperSpf = spf(cheaperDbs.get(1).adjacency(), 1);
  assert.ok(allLsdbsAgree(cheaperDbs), "the new LSAs reached everyone");
  assert.strictEqual(cheaperSpf.dist.get(4), 7, "cheaper R1-R3 shortens the route to R4");
  assert.deepStrictEqual(routePath(cheaperSpf, 4), [1, 3, 2, 4]);
  assert.strictEqual(cheaperSpf.dist.get(5), 8, "and the route to R5");

  // --- link failure ------------------------------------------------------------------------
  const broken = referenceTopology();
  const brokenDbs = syncLsdbs(broken);
  assert.strictEqual(spf(brokenDbs.get(1).adjacency(), 1).dist.get(5), 9);
  broken.removeLink(2, 5);
  floodLsa(broken, brokenDbs, broken.lsaOf(2, 2), 2);
  floodLsa(broken, brokenDbs, broken.lsaOf(5, 2), 5);
  const brokenSpf = spf(brokenDbs.get(1).adjacency(), 1);
  assert.strictEqual(brokenDbs.get(3).sequenceOf(2), 2, "R3 learned the new LSA of R2");
  assert.strictEqual(brokenSpf.dist.get(5), 11, "long way round after the failure");
  assert.deepStrictEqual(routePath(brokenSpf, 5), [1, 3, 2, 4, 5]);
  assert.strictEqual(pathCost(brokenDbs.get(1).adjacency(), routePath(brokenSpf, 5)), 11);
  assert.strictEqual(spf(brokenDbs.get(5).adjacency(), 5).dist.get(2), 8, "R5 agrees on the reverse cost");

  // --- partition ---------------------------------------------------------------------------
  const split = referenceTopology().removeLink(4, 5).removeLink(2, 5);   // {1,2,3,4} | {5}
  const splitDbs = syncLsdbs(split);
  const splitSpf = spf(splitDbs.get(1).adjacency(), 1);
  assert.strictEqual(splitSpf.dist.get(5), undefined, "unreachable destination stays out of the table");
  assert.strictEqual(routePath(splitSpf, 5), null);
  assert.strictEqual(allPairsShortest(splitDbs.get(1).adjacency()).get(1).get(5), undefined,
    "R1's LSDB never receives an LSA from the island, so the oracle does not know R5 either");
  assert.strictEqual(splitSpf.dist.get(4), 8, "the rest of the network is still reachable");
  assert.strictEqual(spf(splitDbs.get(5).adjacency(), 5).dist.get(1), undefined, "the island routes nowhere");
  assert.strictEqual(spf(splitDbs.get(5).adjacency(), 5).dist.get(5), 0);

  // --- deterministic random graphs against the oracle ---------------------------------------
  const lcg = new Lcg(687);
  for (let trial = 0; trial < 8; trial++) {
    const size = 5 + lcg.int(4);
    const graph = new Topology();
    for (let i = 1; i <= size; i++) graph.addRouter(i);
    for (let i = 1; i < size; i++) graph.addLink(i, i + 1, 1 + lcg.int(9));
    const extra = lcg.int(size);
    for (let i = 0; i < extra; i++) {
      const a = 1 + lcg.int(size);
      const b = 1 + lcg.int(size);
      if (a !== b) graph.addLink(a, b, 1 + lcg.int(9));
    }
    const randomDbs = syncLsdbs(graph);
    assert.ok(allLsdbsAgree(randomDbs), `random LSDB agreement ${trial}`);
    const randomAdjacency = randomDbs.get(1).adjacency();
    const randomOracle = allPairsShortest(randomAdjacency);
    for (const source of graph.routers()) {
      const result = spf(randomAdjacency, source);
      for (const destination of graph.routers()) {
        assert.strictEqual(result.dist.get(destination), randomOracle.get(source).get(destination),
          `random ${trial} ${source}->${destination}`);
        const path = routePath(result, destination);
        assert.strictEqual(pathCost(randomAdjacency, path), randomOracle.get(source).get(destination),
          `random path ${trial} ${source}->${destination}`);
        assert.strictEqual(new Set(path).size, path.length, `random loop-free ${trial}`);
        if (destination !== source) assert.strictEqual(result.nextHop.get(destination), path[1]);
      }
    }
  }

  console.log(`routers=${topology.routers().length} edges=${topology.edgeCount()} flood=${refire.transmissions}`);
  console.log("PASS");
}
