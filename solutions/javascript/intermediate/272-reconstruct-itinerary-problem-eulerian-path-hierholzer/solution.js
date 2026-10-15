/** Task 272 - Reconstruct itinerary (Eulerian path, Hierholzer). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

function findItinerary(tickets, start = "JFK") {
  const adjacency = new Map();
  for (const [origin, destination] of tickets) {
    if (!adjacency.has(origin)) adjacency.set(origin, []);
    adjacency.get(origin).push(destination);
  }
  for (const list of adjacency.values()) list.sort();

  const route = [];
  const stack = [start];
  while (stack.length) {
    const airport = stack[stack.length - 1];
    const outgoing = adjacency.get(airport);
    if (outgoing && outgoing.length) stack.push(outgoing.shift());
    else route.push(stack.pop());
  }
  route.reverse();
  return route.length === tickets.length + 1 ? route : null;
}

function usesAllTickets(route, tickets) {
  if (!route) return false;
  const key = (a, b) => `${a}->${b}`;
  const used = new Map();
  for (let i = 0; i + 1 < route.length; i++) {
    const k = key(route[i], route[i + 1]);
    used.set(k, (used.get(k) || 0) + 1);
  }
  const wanted = new Map();
  for (const [a, b] of tickets) {
    const k = key(a, b);
    wanted.set(k, (wanted.get(k) || 0) + 1);
  }
  if (used.size !== wanted.size) return false;
  for (const [k, v] of wanted) if (used.get(k) !== v) return false;
  return true;
}

module.exports = { findItinerary, usesAllTickets };

if (require.main === module) {
  const assert = require("node:assert");
  const first = [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]];
  let route = findItinerary(first);
  assert.deepStrictEqual(route, ["JFK", "MUC", "LHR", "SFO", "SJC"]);
  assert.ok(usesAllTickets(route, first));

  const second = [["JFK", "SFO"], ["JFK", "ATL"], ["SFO", "ATL"], ["ATL", "JFK"], ["ATL", "SFO"]];
  route = findItinerary(second);
  assert.deepStrictEqual(route, ["JFK", "ATL", "JFK", "SFO", "ATL", "SFO"]);
  assert.ok(usesAllTickets(route, second));

  assert.deepStrictEqual(findItinerary([["A", "B"]], "A"), ["A", "B"]);
  assert.deepStrictEqual(findItinerary([], "JFK"), ["JFK"]);
  assert.strictEqual(findItinerary([["A", "B"], ["C", "D"]], "A"), null);
  assert.strictEqual(findItinerary([["A", "B"]], "C"), null);
  assert.deepStrictEqual(findItinerary([["JFK", "B"], ["JFK", "A"], ["A", "JFK"]]), ["JFK", "A", "JFK", "B"]);
  console.log("PASS");
}
