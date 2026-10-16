/** Task 688 - Implement distance-vector routing (RIP) with split horizon. (Tier: Expert, CTCI Beyond — Networking, OS & Real-Time)
 *
 * Model: a distance-vector protocol with RIP semantics. Every router keeps a table
 * dest -> { cost, nextHop }; a router knows itself at cost 0 and its directly attached networks at
 * the link metric. Rounds are synchronous: all advertisements are built from the tables at the start
 * of the round, then every router replaces its table with the Bellman-Ford minimum over the
 * advertisements it received. Metric 16 means infinity (unreachable), so a metric of 15 is the
 * largest usable distance - a 16-router chain is routable end to end, a 17-router chain is not.
 *
 * Three advertisement strategies are implemented, and the difference between them is what the
 * checks measure:
 *   "plain"             - advertise the whole table: a failed link makes the metric creep upwards
 *                         one per round (the classic count-to-infinity) until it finally hits 16;
 *   "split-horizon"     - never advertise a destination back to the neighbour it was learned from;
 *   "poisoned-reverse"  - the same, but advertised as an explicit cost of 16 so the neighbour drops
 *                         the stale route immediately.
 *
 * Verified: hand-derived converged tables, agreement with a Floyd-Warshall oracle everywhere the
 * oracle distance is <= 15, unreachability exactly where RIP's diameter limit is exceeded,
 * first-hop validity and loop-freedom of every route, the adapter rules of all three strategies,
 * round counts (synchronous Bellman-Ford converges in about one round per hop), the count-to-
 * infinity metric ramp of "plain" versus the fast invalidation of the two split-horizon variants,
 * and determinism.
 */
"use strict";

const RIP_INFINITY = 16;
const RIP_MAX_METRIC = 15;

class Network {
  constructor() {
    this.links = new Map();
  }

  addRouter(id) {
    if (!this.links.has(id)) this.links.set(id, new Map());
    return this;
  }

  addLink(a, b, metric = 1) {
    if (!Number.isInteger(metric) || metric < 1 || metric > RIP_MAX_METRIC) {
      throw new RangeError(`metric must be an integer in 1..${RIP_MAX_METRIC}: ${metric}`);
    }
    this.addRouter(a).addRouter(b);
    this.links.get(a).set(b, metric);
    this.links.get(b).set(a, metric);
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

  metricOf(a, b) {
    return this.links.get(a).get(b);
  }
}

class DistanceVector {
  constructor(network, strategy = "poisoned-reverse") {
    if (!["plain", "split-horizon", "poisoned-reverse"].includes(strategy)) {
      throw new Error(`unknown advertisement strategy: ${strategy}`);
    }
    this.network = network;
    this.strategy = strategy;
    this.tables = new Map();
    for (const id of network.routers()) {
      const table = new Map([[id, { cost: 0, nextHop: null }]]);
      for (const [neighbour, metric] of network.neighbours(id)) {
        table.set(neighbour, { cost: metric, nextHop: neighbour });
      }
      this.tables.set(id, table);
    }
    this.rounds = 0;
    this.lastChanged = true;
  }

  tableOf(id) {
    return this.tables.get(id);
  }

  costTo(id, dest) {
    const entry = this.tables.get(id).get(dest);
    return entry === undefined ? RIP_INFINITY : entry.cost;
  }

  nextHopTo(id, dest) {
    const entry = this.tables.get(id).get(dest);
    return entry === undefined ? null : entry.nextHop;
  }

  /** What `from` tells `to` this round, given the strategy. */
  advertisement(from, to) {
    const out = new Map();
    for (const [dest, entry] of this.tables.get(from)) {
      if (entry.cost >= RIP_INFINITY) continue;
      if (entry.nextHop === to) {
        if (this.strategy === "split-horizon") continue;
        if (this.strategy === "poisoned-reverse") {
          out.set(dest, RIP_INFINITY);
          continue;
        }
      }
      out.set(dest, entry.cost);
    }
    return out;
  }

  /** One synchronous round of Bellman-Ford over the advertisements built before any update. */
  step() {
    const advertisements = new Map();
    for (const id of this.network.routers()) {
      for (const [neighbour] of this.network.neighbours(id)) {
        advertisements.set(`${id}->${neighbour}`, this.advertisement(id, neighbour));
      }
    }

    let changed = false;
    const next = new Map();
    for (const id of this.network.routers()) {
      const table = new Map([[id, { cost: 0, nextHop: null }]]);
      for (const [neighbour, metric] of this.network.neighbours(id)) {
        for (const [dest, advertised] of advertisements.get(`${neighbour}->${id}`)) {
          const total = advertised + metric;
          if (total >= RIP_INFINITY) continue;                 // 16 = unreachable
          const current = table.get(dest);
          if (current === undefined || total < current.cost
            || (total === current.cost && neighbour < current.nextHop)) {
            table.set(dest, { cost: total, nextHop: neighbour });
          }
        }
      }
      next.set(id, table);
      if (!tablesEqual(table, this.tables.get(id))) changed = true;
    }
    this.tables = next;
    this.rounds++;
    this.lastChanged = changed;
    return changed;
  }

  settled() {
    return !this.lastChanged;
  }

  run(maxRounds = 64) {
    for (let round = 0; round < maxRounds; round++) {
      if (!this.step()) return { rounds: this.rounds, converged: true };
    }
    return { rounds: this.rounds, converged: false };
  }

  /** Follow the first hops; null when the destination is unreachable or a loop appears. */
  pathTo(id, dest) {
    if (id === dest) return [id];
    const path = [id];
    const seen = new Set(path);
    let current = id;
    while (current !== dest) {
      const hop = this.nextHopTo(current, dest);
      if (hop === null || seen.has(hop)) return null;
      path.push(hop);
      seen.add(hop);
      current = hop;
    }
    return path;
  }
}

function tablesEqual(a, b) {
  if (a.size !== b.size) return false;
  for (const [dest, entry] of a) {
    const other = b.get(dest);
    if (other === undefined || other.cost !== entry.cost || other.nextHop !== entry.nextHop) return false;
  }
  return true;
}

/** Oracle: Floyd-Warshall with no infinity cap (metric 16 stays a real number here). */
function allPairsMetric(network) {
  const ids = network.routers();
  const dist = new Map(ids.map((id) => [id, new Map(ids.map((other) => [other, id === other ? 0 : Infinity]))]));
  for (const id of ids) {
    for (const [neighbour, metric] of network.neighbours(id)) dist.get(id).set(neighbour, metric);
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

function chain(count) {
  const network = new Network();
  for (let i = 1; i < count; i++) network.addLink(i, i + 1, 1);
  network.addRouter(count);
  return network;
}

module.exports = { RIP_INFINITY, RIP_MAX_METRIC, Network, DistanceVector, allPairsMetric, chain };

if (require.main === module) {
  const assert = require("node:assert");

  // --- hand-derived convergence ------------------------------------------------------------
  const small = new Network().addLink(1, 2, 1).addLink(2, 3, 1).addLink(3, 4, 1).addLink(1, 4, 5);
  assert.throws(() => new Network().addLink(1, 2, 16), RangeError, "metric 16 is reserved");
  assert.throws(() => new Network().addLink(1, 2, 0), RangeError);
  const dv = new DistanceVector(small, "poisoned-reverse");
  const outcome = dv.run();
  assert.strictEqual(outcome.converged, true, "converged");
  assert.strictEqual(dv.costTo(1, 4), 3, "1-2-3-4 beats the direct metric-5 link");
  assert.strictEqual(dv.costTo(1, 3), 2);
  assert.strictEqual(dv.costTo(4, 1), 3, "routes are symmetric");
  assert.deepStrictEqual(dv.pathTo(1, 4), [1, 2, 3, 4]);
  assert.deepStrictEqual(dv.pathTo(4, 1), [4, 3, 2, 1]);
  assert.deepStrictEqual(dv.pathTo(2, 2), [2], "route to self");
  assert.strictEqual(dv.costTo(1, 1), 0);
  assert.ok(outcome.rounds <= dv.network.routers().length, `rounds bounded by diameter: ${outcome.rounds}`);
  assert.strictEqual(dv.nextHopTo(1, 4), 2, "first hop on the shortest path");

  // --- oracle agreement on every routable pair ---------------------------------------------
  const oracle = allPairsMetric(small);
  for (const source of small.routers()) {
    for (const destination of small.routers()) {
      const expected = oracle.get(source).get(destination);
      assert.ok(expected <= RIP_MAX_METRIC, "this topology is inside the RIP diameter");
      assert.strictEqual(dv.costTo(source, destination), expected, `oracle ${source}->${destination}`);
      const path = dv.pathTo(source, destination);
      assert.ok(path !== null && path.length - 1 === expected, `hop count equals the metric ${source}->${destination}`);
      assert.strictEqual(new Set(path).size, path.length, "loop-free");
      if (destination !== source) {
        assert.strictEqual(small.metricOf(source, path[1]) + dv.costTo(path[1], destination), expected,
          "first hop lies on a shortest path");
      }
      assert.strictEqual(source === destination ? 0 : 1, source === destination ? 0 : 1);
    }
  }

  // --- RIP's diameter limit: 16 routers are routable, 17 are not ---------------------------
  const chain16 = new DistanceVector(chain(16), "poisoned-reverse");
  chain16.run();
  assert.strictEqual(chain16.costTo(1, 16), 15, "metric 15 is still usable");
  const chain17 = new DistanceVector(chain(17), "poisoned-reverse");
  const chain17Outcome = chain17.run();
  assert.strictEqual(chain17Outcome.converged, true);
  assert.strictEqual(chain17.costTo(1, 17), RIP_INFINITY, "metric 16 means unreachable");
  assert.strictEqual(chain17.pathTo(1, 17), null);
  assert.strictEqual(allPairsMetric(chain(17)).get(1).get(17), 16, "the oracle measures 16 hops");
  assert.strictEqual(chain17.costTo(1, 16), 15, "everything closer still converges");
  assert.strictEqual(chain17.costTo(2, 17), 15);
  assert.strictEqual(chain17.costTo(3, 17), 14, "closer to 17 the metric is inside the limit again");
  const chain18 = new DistanceVector(chain(18), "poisoned-reverse");
  chain18.run();
  assert.strictEqual(chain18.costTo(1, 18), RIP_INFINITY, "16 hops is already unreachable");
  assert.strictEqual(chain18.costTo(3, 18), 15, "15 hops is the last reachable distance");
  assert.strictEqual(chain18.costTo(2, 18), RIP_INFINITY);
  assert.strictEqual(chain18.costTo(4, 18), 14);

  // --- advertisement rules ------------------------------------------------------------------
  const pair = new Network().addLink(1, 2, 1).addLink(2, 3, 1).addLink(3, 4, 1);
  const plain = new DistanceVector(pair, "plain");
  const withSplit = new DistanceVector(pair, "split-horizon");
  const poisoned = new DistanceVector(pair, "poisoned-reverse");
  for (const dvInstance of [plain, withSplit, poisoned]) dvInstance.run();
  assert.strictEqual(plain.advertisement(3, 2).get(2), 1, "plain advertises the route back");
  assert.strictEqual(withSplit.advertisement(3, 2).has(2), false, "split horizon omits it");
  assert.strictEqual(poisoned.advertisement(3, 2).get(2), RIP_INFINITY, "poisoned reverse makes it 16");
  assert.strictEqual(withSplit.advertisement(3, 2).get(4), 1, "unrelated destinations still travel");
  assert.strictEqual(withSplit.advertisement(3, 2).get(3), 0, "own address is always advertised");
  assert.strictEqual(poisoned.advertisement(2, 1).get(1), RIP_INFINITY, "2 poisons the route home");
  assert.strictEqual(withSplit.advertisement(3, 2).has(1), false, "the route learned from 2 stays hidden too");

  // --- count to infinity without split horizon ---------------------------------------------
  const trace = (strategy) => {
    const network = new Network().addLink(1, 2, 1).addLink(2, 3, 1);
    const instance = new DistanceVector(network, strategy);
    instance.run();                                        // converges on the intact chain A-B-C
    assert.strictEqual(instance.costTo(3, 1), 2, "C starts out two hops from A");
    network.removeLink(1, 2);                              // the A-B link fails
    const history = [];
    for (;;) {
      instance.step();
      history.push(instance.costTo(2, 1));
      if (instance.settled() || history.length >= 40) break;
    }
    return { history, rounds: instance.rounds, finalCost: instance.costTo(2, 1) };
  };
  const plainTrace = trace("plain");
  const splitTrace = trace("split-horizon");
  const poisonedTrace = trace("poisoned-reverse");
  for (const fast of [splitTrace, poisonedTrace]) {
    assert.ok(fast.history.every((value) => value === RIP_INFINITY), "invalidated in the very first round");
    assert.ok(fast.history.length <= 4, `settles within a few rounds, got ${fast.history.length}`);
    assert.strictEqual(fast.finalCost, RIP_INFINITY);
  }
  assert.ok(plainTrace.history.length >= 10,
    `count to infinity needs many rounds, got ${plainTrace.history.length}`);
  for (let i = 1; i < plainTrace.history.length; i++) {
    assert.ok(plainTrace.history[i] >= plainTrace.history[i - 1], "the metric never decreases while counting up");
  }
  assert.ok(plainTrace.history[0] >= 3 && plainTrace.history[0] < RIP_INFINITY, "B first believes C's stale route");
  assert.ok(plainTrace.history.includes(RIP_MAX_METRIC), "the metric reaches 15 before RIP gives up");
  assert.strictEqual(plainTrace.history[plainTrace.history.length - 1], RIP_INFINITY, "and creeps all the way to 16");
  assert.strictEqual(plainTrace.rounds > 3 * splitTrace.rounds, true, "split horizon reconverges far faster");

  // --- a link failure inside a two-path topology --------------------------------------------
  const resilient = new Network().addLink(1, 2, 1).addLink(1, 3, 1).addLink(2, 4, 1).addLink(3, 4, 1);
  const resilientDv = new DistanceVector(resilient, "poisoned-reverse");
  resilientDv.run();
  assert.strictEqual(resilientDv.costTo(1, 4), 2);
  assert.strictEqual(resilientDv.nextHopTo(1, 4), 2, "lowest neighbour id wins the tie");
  resilient.removeLink(1, 2);
  const afterFailure = resilientDv.run();
  assert.strictEqual(afterFailure.converged, true);
  assert.strictEqual(resilientDv.costTo(1, 4), 2, "still two hops, now through 3");
  assert.strictEqual(resilientDv.nextHopTo(1, 4), 3);
  assert.deepStrictEqual(resilientDv.pathTo(1, 4), [1, 3, 4]);
  assert.strictEqual(resilientDv.costTo(4, 1), 2, "reverse direction agrees");
  assert.strictEqual(resilientDv.pathTo(1, 4).length - 1, allPairsMetric(resilient).get(1).get(4));
  resilient.removeLink(3, 4);
  resilientDv.run();
  assert.strictEqual(resilientDv.costTo(1, 4), RIP_INFINITY, "no path left");
  assert.strictEqual(resilientDv.pathTo(4, 1), null);

  // --- determinism and round-bound ---------------------------------------------------------
  const rerunA = new DistanceVector(chain(9), "poisoned-reverse");
  const rerunB = new DistanceVector(chain(9), "poisoned-reverse");
  const firstOutcome = rerunA.run();
  const secondOutcome = rerunB.run();
  assert.deepStrictEqual(secondOutcome, firstOutcome, "same rounds every time");
  for (const id of rerunA.network.routers()) {
    assert.deepStrictEqual([...rerunA.tableOf(id)].sort((p, q) => p[0] - q[0]),
      [...rerunB.tableOf(id)].sort((p, q) => p[0] - q[0]), `identical table at ${id}`);
  }
  assert.strictEqual(firstOutcome.rounds, 8, "a 9-router chain converges in 8 rounds");

  // a router can never advertise a destination at a lower cost than it holds
  for (const [from, to] of [[1, 2], [3, 2], [4, 3]]) {
    for (const [dest, advertised] of rerunA.advertisement(from, to)) {
      assert.ok(advertised === RIP_INFINITY || advertised >= rerunA.costTo(from, dest),
        `monotone advertisement ${from}->${to} for ${dest}`);
    }
  }

  console.log(`rounds chain9=${firstOutcome.rounds} countToInfinity rounds=${plainTrace.rounds}`);
  console.log("PASS");
}
