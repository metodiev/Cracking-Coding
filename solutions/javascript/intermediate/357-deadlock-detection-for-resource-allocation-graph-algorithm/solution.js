/**
 * Task 357 - Deadlock detection for resource allocation graph (algorithm). (Tier: Intermediate, CTCI Ch 15)
 *
 * Deadlock detection by cycle finding: the resource allocation graph (processes hold and
 * request resources, a resource has at most one holder) collapses into the wait-for graph -
 * each request (p, r) with r held by q becomes the edge p -> q; requests for free resources
 * contribute no edge. A deadlock exists exactly when the wait-for graph has a cycle, so
 * findCycle() colour-DFSes (white/gray/black) and returns the first cycle it closes, rotated
 * so its smallest vertex comes first (a process requesting a resource it already holds
 * yields the self-loop [p]). The self-test checks hand-built deadlocked, acyclic and
 * self-loop graphs, then cross-checks cycle existence against an independent Kosaraju SCC
 * check on 300 LCG-generated resource allocation graphs.
 */
"use strict";

const SCRIPT_TRIALS = 300;

function canonicalCycle(cycle) {
  let minIndex = 0;
  for (let i = 1; i < cycle.length; i++) if (cycle[i] < cycle[minIndex]) minIndex = i;
  const result = [];
  for (let i = 0; i < cycle.length; i++) result.push(cycle[(minIndex + i) % cycle.length]);
  return result;
}

/**
 * Collapse the resource allocation graph into the wait-for graph: a request (p, r)
 * with r held by q becomes the edge p -> q; a request for a free resource adds nothing.
 */
function buildWaitFor(processes, resources, holds, requests) {
  const holder = new Array(resources).fill(-1);
  for (const [process, resource] of holds) {
    if (holder[resource] !== -1 && holder[resource] !== process) {
      throw new Error(`resource held by more than one process: ${resource}`);
    }
    holder[resource] = process;
  }
  const adjacency = Array.from({ length: processes }, () => []);
  for (const [process, resource] of requests) {
    const currentHolder = holder[resource];
    if (currentHolder !== -1) adjacency[process].push(currentHolder);
  }
  return adjacency;
}

/** Returns a canonical cycle or null. 0 = white, 1 = gray (on the DFS path), 2 = black. */
function findCycle(processes, adjacency) {
  const color = new Array(processes).fill(0);
  const path = [];

  function dfs(node) {
    color[node] = 1;
    path.push(node);
    for (const next of adjacency[node]) {
      if (color[next] === 1) {                       // back edge: the path slice is a cycle
        return path.slice(path.indexOf(next));
      }
      if (color[next] === 0) {
        const cycle = dfs(next);
        if (cycle !== null) return cycle;
      }
    }
    path.pop();
    color[node] = 2;
    return null;
  }

  for (let start = 0; start < processes; start++) {
    if (color[start] === 0) {
      const cycle = dfs(start);
      if (cycle !== null) return canonicalCycle(cycle);
    }
  }
  return null;
}

/** Independent check: a cycle exists iff some Kosaraju SCC has size > 1 or a self-loop exists. */
function hasCycleViaScc(processes, adjacency) {
  const reverse = Array.from({ length: processes }, () => []);
  for (let u = 0; u < processes; u++) {
    for (const v of adjacency[u]) reverse[v].push(u);
  }

  const visited = new Array(processes).fill(false);
  const order = [];

  function dfsFinish(u) {
    visited[u] = true;
    for (const v of adjacency[u]) if (!visited[v]) dfsFinish(v);
    order.push(u);
  }

  for (let u = 0; u < processes; u++) if (!visited[u]) dfsFinish(u);

  const component = new Array(processes).fill(-1);
  const sizes = [];
  for (let i = order.length - 1; i >= 0; i--) {
    const u = order[i];
    if (component[u] !== -1) continue;
    const groupIndex = sizes.length;
    component[u] = groupIndex;
    let size = 0;
    const stack = [u];
    while (stack.length > 0) {
      const x = stack.pop();
      size += 1;
      for (const y of reverse[x]) {
        if (component[y] === -1) {
          component[y] = groupIndex;
          stack.push(y);
        }
      }
    }
    sizes.push(size);
  }

  if (sizes.some((size) => size > 1)) return true;
  for (let u = 0; u < processes; u++) if (adjacency[u].includes(u)) return true;
  return false;
}

function assertValidCycle(adjacency, cycle) {
  if (JSON.stringify(canonicalCycle(cycle)) !== JSON.stringify(cycle)) {
    throw new Error("cycle returned canonicalised");
  }
  if (new Set(cycle).size !== cycle.length) throw new Error("cycle vertices are distinct");
  for (let i = 0; i < cycle.length; i++) {
    const next = cycle[(i + 1) % cycle.length];
    if (!adjacency[cycle[i]].includes(next)) throw new Error("cycle edges exist");
  }
}

let seed = 13579;

/** LCG RAG: every resource gets one holder or stays free; processes issue 0..2 requests. */
function randomRag(processes, resources) {
  const holder = new Array(resources).fill(-1);
  const holds = [];
  for (let resource = 0; resource < resources; resource++) {
    seed = (seed * 75 + 74) % 65537;
    if (seed % 3 !== 0) {
      const process = seed % processes;
      holder[resource] = process;
      holds.push([process, resource]);
    }
  }
  const requests = [];
  for (let process = 0; process < processes; process++) {
    seed = (seed * 75 + 74) % 65537;
    const count = seed % 3;
    for (let i = 0; i < count; i++) {
      seed = (seed * 75 + 74) % 65537;
      requests.push([process, seed % resources]);
    }
  }
  return { holds, requests };
}

function main() {
  const assert = require("node:assert");

  // Hand-built deadlock: two processes wait for each other; canonical cycle [0, 1].
  let adjacency = buildWaitFor(2, 2, [[0, 0], [1, 1]], [[0, 1], [1, 0]]);
  assert.deepStrictEqual(adjacency, [[1], [0]], "wait-for edges");
  assert.deepStrictEqual(findCycle(2, adjacency), [0, 1], "classic two-process deadlock");

  // Acyclic: process 0 waits for process 1, which waits for nothing.
  adjacency = buildWaitFor(2, 2, [[0, 0], [1, 1]], [[0, 1]]);
  assert.deepStrictEqual(adjacency, [[1], []], "one-way wait edge");
  assert.strictEqual(findCycle(2, adjacency), null, "no cycle without a wait cycle");

  // Self-loop: a process requests a resource it already holds.
  adjacency = buildWaitFor(1, 1, [[0, 0]], [[0, 0]]);
  assert.deepStrictEqual(adjacency, [[0]], "self-loop edge");
  assert.deepStrictEqual(findCycle(1, adjacency), [0], "self-loop is detected");

  // Three-process cycle; the returned cycle is the hand-checked canonical rotation.
  adjacency = buildWaitFor(3, 3, [[1, 0], [2, 1], [0, 2]], [[0, 0], [1, 1], [2, 2]]);
  assert.deepStrictEqual(adjacency, [[1], [2], [0]], "three-cycle edges");
  assert.deepStrictEqual(findCycle(3, adjacency), [0, 1, 2], "three-process deadlock");

  // Requests for free resources create no edges.
  adjacency = buildWaitFor(2, 2, [[0, 0]], [[1, 1]]);
  assert.deepStrictEqual(adjacency, [[], []], "free resource: no edge");
  assert.strictEqual(findCycle(2, adjacency), null, "waiting for a free resource is not a deadlock");

  // All processes idle, and an extra process outside the cycle.
  adjacency = buildWaitFor(3, 3, [[0, 0], [1, 1], [2, 2]], []);
  assert.strictEqual(findCycle(3, adjacency), null, "idle processes never deadlock");
  adjacency = buildWaitFor(3, 3, [[0, 0], [1, 1], [2, 2]], [[0, 1], [1, 0]]);
  assert.deepStrictEqual(findCycle(3, adjacency), [0, 1], "cycle independent of an idle process");

  // A resource may have only one holder.
  assert.throws(() => buildWaitFor(2, 1, [[0, 0], [1, 0]], []), /more than one process/,
    "a double-held resource must be rejected");

  // Cross-check cycle existence against the independent SCC check on LCG RAGs.
  seed = 13579;
  let cyclesSeen = 0;
  for (let trial = 0; trial < SCRIPT_TRIALS; trial++) {
    seed = (seed * 75 + 74) % 65537;
    const processes = 2 + (seed % 6);
    seed = (seed * 75 + 74) % 65537;
    const resources = 1 + (seed % 6);
    const { holds, requests } = randomRag(processes, resources);
    const waitFor = buildWaitFor(processes, resources, holds, requests);
    const cycle = findCycle(processes, waitFor);
    const sccSaysCycle = hasCycleViaScc(processes, waitFor);
    assert.strictEqual(cycle !== null, sccSaysCycle, `SCC cross-check at trial ${trial}`);
    if (cycle !== null) {
      cyclesSeen += 1;
      assertValidCycle(waitFor, cycle);
    }
  }
  assert.ok(cyclesSeen >= 20, `LCG graphs should contain deadlocks (saw ${cyclesSeen})`);
}

module.exports = { buildWaitFor, findCycle, hasCycleViaScc, canonicalCycle };

if (require.main === module) {
  try {
    main();
    console.log("PASS");
  } catch (err) {
    console.error(err);
    process.exit(1);
  }
}
