/**
 * Task 509 - Implement dependency graph scheduling (topo + priorities). (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
 * Dependency-graph scheduling over a small task DAG: each task has a priority and the graph is
 * scheduled with a priority-aware Kahn algorithm - repeatedly pick, among the tasks whose
 * dependencies are all satisfied, the one with the smallest (priority, id) key - which yields the
 * lexicographically smallest valid topological order under that comparator. The module also computes
 * the level of every task (longest path from a source, by DP over a topological order, with levels
 * forming the execution waves), the critical-path length, a second independent topological order
 * (DFS post-order reversed), the full set of topological orderings for tiny graphs, and cycle
 * detection (both a self-loop and a longer cycle must be rejected with a RangeError).
 *
 * Documented checks: the priority order respects every edge and, on all graphs with at most 6 nodes,
 * is compared against a brute-force enumeration of all n! permutations: it must equal the
 * lexicographically smallest valid order by (priority, id), and the DFS order must be *some* valid
 * order (the two orders are only required to agree when there is a unique topological order, which
 * the test asserts on the hand-checked graph). The level assignment is verified to be exactly the
 * longest-path distance from the sources (compared against a memoised brute-force longest-path
 * search) and every edge is asserted to go from a lower level to a higher one, so groups by level are
 * valid execution waves; the wave count equals the critical-path length + 1 (critical path in nodes),
 * and the number of topological orderings of a small DAG computed by subset DP is cross-checked with
 * the brute-force permutation count. Cycle detection is asserted on a 3-cycle, a longer cycle and a
 * self-loop, and the empty and single-node graphs are handled (one ordering, one wave).
 */
"use strict";

class TaskGraph {
  constructor(ids) {
    this.tasks = new Map();
    for (const id of ids) {
      this.tasks.set(id, { id, priority: 0, deps: new Set() });
    }
    this.order = ids.slice();
  }

  setPriority(id, priority) {
    this.tasks.get(id).priority = priority;
    return this;
  }

  addEdge(from, to) {
    if (!this.tasks.has(from) || !this.tasks.has(to)) throw new RangeError("unknown task id");
    this.tasks.get(to).deps.add(from);
    return this;
  }

  get size() {
    return this.tasks.size;
  }

  successors(id) {
    const out = [];
    for (const [other, task] of this.tasks) {
      if (task.deps.has(id)) out.push(other);
    }
    return out;
  }
}

/** Simple binary min-heap over (priority, id) keys. */
class ReadyHeap {
  constructor() {
    this.items = [];
  }

  get size() {
    return this.items.length;
  }

  push(id, priority) {
    this.items.push([priority, id]);
    let index = this.items.length - 1;
    while (index > 0) {
      const parent = (index - 1) >> 1;
      if (ReadyHeap.less(this.items[index], this.items[parent])) {
        [this.items[parent], this.items[index]] = [this.items[index], this.items[parent]];
        index = parent;
      } else break;
    }
  }

  pop() {
    const top = this.items[0];
    const last = this.items.pop();
    if (this.items.length > 0) {
      this.items[0] = last;
      let index = 0;
      for (;;) {
        const left = 2 * index + 1;
        const right = left + 1;
        let best = index;
        if (left < this.items.length && ReadyHeap.less(this.items[left], this.items[best])) {
          best = left;
        }
        if (right < this.items.length && ReadyHeap.less(this.items[right], this.items[best])) {
          best = right;
        }
        if (best === index) break;
        [this.items[best], this.items[index]] = [this.items[index], this.items[best]];
        index = best;
      }
    }
    return top;
  }

  static less(a, b) {
    return a[0] !== b[0] ? a[0] < b[0] : a[1] < b[1];
  }
}

/** Priority-aware Kahn schedule; throws RangeError when the graph has a cycle. */
function prioritySchedule(graph, heapOnly = true) {
  const remaining = new Map();
  for (const [id, task] of graph.tasks) remaining.set(id, new Set(task.deps));
  const ready = [];
  for (const [id, deps] of remaining) {
    if (deps.size === 0) ready.push(id);
  }
  const order = [];
  const heap = new ReadyHeap();
  const readyIds = new Set(ready);
  if (heapOnly) {
    for (const id of ready) heap.push(id, graph.tasks.get(id).priority);
  } else {
    ready.sort((a, b) => (graph.tasks.get(a).priority !== graph.tasks.get(b).priority
      ? graph.tasks.get(a).priority - graph.tasks.get(b).priority
      : (a < b ? -1 : 1)));
  }
  const pop = () => {
    if (heapOnly) return heap.pop()[1];
    return ready.shift();
  };
  const push = (id) => {
    if (heapOnly) heap.push(id, graph.tasks.get(id).priority);
    else {
      ready.push(id);
      ready.sort((a, b) => (graph.tasks.get(a).priority !== graph.tasks.get(b).priority
        ? graph.tasks.get(a).priority - graph.tasks.get(b).priority
        : (a < b ? -1 : 1)));
    }
  };
  const sizeOf = () => (heapOnly ? heap.size : ready.length);
  while (sizeOf() > 0) {
    const id = pop();
    readyIds.delete(id);
    order.push(id);
    for (const next of graph.successors(id)) {
      remaining.get(next).delete(id);
      if (remaining.get(next).size === 0 && !readyIds.has(next) && !order.includes(next)) {
        readyIds.add(next);
        push(next);
      }
    }
  }
  if (order.length !== graph.size) throw new RangeError("dependency graph contains a cycle");
  return order;
}

/** Independent topological order: DFS post-order reversed (iterative, deterministic). */
function dfsTopologicalOrder(graph) {
  const state = new Map();                             // 0 unseen, 1 on stack, 2 done
  for (const id of graph.tasks.keys()) state.set(id, 0);
  const post = [];
  for (const root of graph.tasks.keys()) {
    if (state.get(root) !== 0) continue;
    const stack = [{ id: root, successors: graph.successors(root), index: 0 }];
    state.set(root, 1);
    while (stack.length > 0) {
      const frame = stack[stack.length - 1];
      if (frame.index < frame.successors.length) {
        const next = frame.successors[frame.index];
        frame.index += 1;
        if (state.get(next) === 1) throw new RangeError("dependency graph contains a cycle");
        if (state.get(next) === 0) {
          state.set(next, 1);
          stack.push({ id: next, successors: graph.successors(next), index: 0 });
        }
      } else {
        stack.pop();
        state.set(frame.id, 2);
        post.push(frame.id);
      }
    }
  }
  return post.reverse();
}

function respectsEdges(graph, order) {
  const position = new Map(order.map((id, index) => [id, index]));
  for (const [id, task] of graph.tasks) {
    for (const dep of task.deps) {
      if (position.get(dep) >= position.get(id)) return false;
    }
  }
  return order.length === graph.size;
}

/** Levels = longest path from a source, by DP over a topological order. */
function computeLevels(graph, order = prioritySchedule(graph)) {
  const levels = new Map(order.map((id) => [id, 0]));
  for (const id of order) {
    for (const next of graph.successors(id)) {
      levels.set(next, Math.max(levels.get(next), levels.get(id) + 1));
    }
  }
  return levels;
}

/** Memoised brute-force longest path from the sources, as a cross-check of the levels. */
function longestPathFromSources(graph, id, memo = new Map()) {
  if (memo.has(id)) return memo.get(id);
  let best = 0;
  for (const dep of graph.tasks.get(id).deps) {
    best = Math.max(best, longestPathFromSources(graph, dep, memo) + 1);
  }
  memo.set(id, best);
  return best;
}

function waves(graph, levels) {
  const grouped = new Map();
  for (const [id, level] of levels) {
    if (!grouped.has(level)) grouped.set(level, []);
    grouped.get(level).push(id);
  }
  return [...grouped.entries()].sort((a, b) => a[0] - b[0])
    .map(([, ids]) => ids.slice().sort());
}

function allPermutations(items) {
  if (items.length <= 1) return [items.slice()];
  const out = [];
  for (let i = 0; i < items.length; i++) {
    const rest = items.slice(0, i).concat(items.slice(i + 1));
    for (const permutation of allPermutations(rest)) out.push([items[i], ...permutation]);
  }
  return out;
}

/** Brute force: the lexicographically smallest valid order by (priority, id). */
function bruteForceBestOrder(graph) {
  const ids = [...graph.tasks.keys()];
  let best = null;
  for (const permutation of allPermutations(ids)) {
    if (!respectsEdges(graph, permutation)) continue;
    if (best === null) {
      best = permutation;
      continue;
    }
    for (let i = 0; i < permutation.length; i++) {
      const a = graph.tasks.get(permutation[i]);
      const b = graph.tasks.get(best[i]);
      if (a.priority !== b.priority) {
        if (a.priority < b.priority) best = permutation;
        break;
      }
      if (a.id !== b.id) {
        if (a.id < b.id) best = permutation;
        break;
      }
    }
  }
  return best;
}

function countTopologicalOrderings(graph) {
  const ids = [...graph.tasks.keys()];
  const index = new Map(ids.map((id, i) => [id, i]));
  const depMask = ids.map((id) => {
    let mask = 0;
    for (const dep of graph.tasks.get(id).deps) mask |= 1 << index.get(dep);
    return mask;
  });
  const dp = new Array(1 << ids.length).fill(0);
  dp[0] = 1;
  for (let mask = 1; mask < (1 << ids.length); mask++) {
    for (let i = 0; i < ids.length; i++) {
      const bit = 1 << i;
      if ((mask & bit) === 0) continue;
      const previous = mask ^ bit;
      if ((depMask[i] & previous) === depMask[i]) dp[mask] += dp[previous];
    }
  }
  return dp[(1 << ids.length) - 1];
}

function lcgGraph(nodes, edges, seed) {
  const ids = Array.from({ length: nodes }, (_, i) => `t${String(i).padStart(2, "0")}`);
  const graph = new TaskGraph(ids);
  let s = seed;
  for (const id of ids) {
    s = (s * 75 + 74) % 65537;
    graph.setPriority(id, s % 5);
  }
  let added = 0;
  let attempts = 0;
  while (added < edges && attempts < 200 * edges) {
    attempts += 1;
    s = (s * 75 + 74) % 65537;
    const from = s % nodes;
    s = (s * 75 + 74) % 65537;
    const to = s % nodes;
    if (from >= to) continue;                          // keep it a DAG by index order
    if (graph.tasks.get(ids[to]).deps.has(ids[from])) continue;
    graph.addEdge(ids[from], ids[to]);
    added += 1;
  }
  return graph;
}

function main() {
  const assert = require("node:assert");
  const close = (a, b, tol, message) =>
    assert.ok(Math.abs(a - b) <= tol, `${message}: |${a} - ${b}| > ${tol}`);

  // (1) a hand-checked DAG with a unique topological order.
  const chain = new TaskGraph(["a", "b", "c", "d"]);
  chain.setPriority("a", 3).setPriority("b", 0).setPriority("c", 1).setPriority("d", 2);
  chain.addEdge("a", "b").addEdge("b", "c").addEdge("a", "c").addEdge("c", "d");
  const order = prioritySchedule(chain);
  assert.deepStrictEqual(order, ["a", "b", "c", "d"], "the chain has one order only");
  assert.deepStrictEqual(dfsTopologicalOrder(chain), order,
    "with a unique topological order both methods agree");
  assert.strictEqual(respectsEdges(chain, order), true);
  assert.strictEqual(countTopologicalOrderings(chain), 1, "exactly one ordering");
  const levels = computeLevels(chain, order);
  assert.deepStrictEqual([...levels.entries()].sort(),
    [["a", 0], ["b", 1], ["c", 2], ["d", 3]], "levels are the longest path from a");
  assert.deepStrictEqual(waves(chain, levels), [["a"], ["b"], ["c"], ["d"]],
    "a chain is four waves");

  // (2) priorities genuinely reorder independent tasks.
  const flat = new TaskGraph(["x", "y", "z", "w"]);
  flat.setPriority("x", 4).setPriority("y", 1).setPriority("z", 3).setPriority("w", 1);
  assert.deepStrictEqual(prioritySchedule(flat), ["w", "y", "z", "x"],
    "priority first, then id, and every order is valid with no edges");
  const dfsFlat = dfsTopologicalOrder(flat);
  assert.deepStrictEqual(dfsFlat.slice().sort(), ["w", "x", "y", "z"],
    "the DFS order is a permutation of the tasks");
  assert.notDeepStrictEqual(dfsFlat, prioritySchedule(flat),
    "and a different (still valid) order than the priority one");
  assert.strictEqual(respectsEdges(flat, dfsTopologicalOrder(flat)), true);
  assert.strictEqual(countTopologicalOrderings(flat), 24, "4! orderings for an empty DAG");

  // (3) brute force over permutations for small graphs.
  const small = [lcgGraph(5, 5, 11), lcgGraph(6, 8, 22), lcgGraph(6, 3, 33), lcgGraph(4, 4, 44)];
  for (let i = 0; i < small.length; i++) {
    const graph = small[i];
    const scheduled = prioritySchedule(graph);
    const brute = bruteForceBestOrder(graph);
    assert.deepStrictEqual(scheduled, brute,
      `graph ${i}: priority Kahn gives the lexicographically smallest valid order`);
    assert.strictEqual(respectsEdges(graph, scheduled), true, `graph ${i}: edges respected`);
    assert.strictEqual(respectsEdges(graph, dfsTopologicalOrder(graph)), true,
      `graph ${i}: the DFS order is valid too`);
    assert.strictEqual(countTopologicalOrderings(graph), allPermutations([...graph.tasks.keys()])
      .filter((permutation) => respectsEdges(graph, permutation)).length,
      `graph ${i}: the subset-DP count equals the brute-force count`);
  }

  // (4) levels equal the memoised longest path and edges go strictly up a level.
  const graphs = [lcgGraph(12, 18, 55), lcgGraph(20, 30, 66), lcgGraph(9, 0, 77),
    lcgGraph(15, 40, 88)];
  const summary = [];
  for (const graph of graphs) {
    const scheduled = prioritySchedule(graph);
    const lv = computeLevels(graph, scheduled);
    for (const id of graph.tasks.keys()) {
      assert.strictEqual(lv.get(id), longestPathFromSources(graph, id),
        "level = longest path from the sources");
    }
    for (const [id, task] of graph.tasks) {
      for (const dep of task.deps) {
        assert.ok(lv.get(id) >= lv.get(dep) + 1, "every edge goes to a higher level");
      }
    }
    const grouped = waves(graph, lv);
    const maxLevel = Math.max(...lv.values());
    assert.strictEqual(grouped.length, maxLevel + 1, "wave count = longest path + 1");
    for (const wave of grouped) {
      for (const a of wave) {
        for (const b of wave) {
          if (a === b) continue;
          assert.ok(!graph.tasks.get(b).deps.has(a), "tasks in a wave are independent");
        }
      }
    }
    const positions = new Map(scheduled.map((id, index) => [id, index]));
    for (const [id, task] of graph.tasks) {
      for (const dep of task.deps) {
        assert.ok(positions.get(dep) < positions.get(id), "the schedule respects every edge");
      }
    }
    summary.push([graph.size, grouped.length, maxLevel]);
  }
  assert.ok(summary.every(([, count, max]) => count === max + 1));
  assert.strictEqual(summary[2][1], 1, "an edgeless graph is one wave");

  // (5) the heap and the linear-scan implementation of the same priority rule agree.
  for (const graph of graphs) {
    assert.deepStrictEqual(prioritySchedule(graph, true), prioritySchedule(graph, false),
      "heap and sorted-array ready sets produce the same schedule");
  }

  // (6) determinism.
  const again = prioritySchedule(graphs[0]);
  assert.deepStrictEqual(again, prioritySchedule(graphs[0]), "scheduling is deterministic");

  // (7) cycles are rejected, from every entry point.
  const cyclic = new TaskGraph(["a", "b", "c"]);
  cyclic.addEdge("a", "b").addEdge("b", "c").addEdge("c", "a");
  assert.throws(() => prioritySchedule(cyclic), RangeError, "a 3-cycle is rejected");
  assert.throws(() => dfsTopologicalOrder(cyclic), RangeError, "the DFS also detects the cycle");
  const longCycle = new TaskGraph(["a", "b", "c", "d", "e"]);
  longCycle.addEdge("a", "b").addEdge("b", "c").addEdge("c", "d").addEdge("d", "e")
    .addEdge("e", "b");
  assert.throws(() => prioritySchedule(longCycle), RangeError, "a longer cycle is rejected");
  const selfLoop = new TaskGraph(["a"]);
  selfLoop.addEdge("a", "a");
  assert.throws(() => prioritySchedule(selfLoop), RangeError, "a self-loop is rejected");
  assert.throws(() => selfLoop.addEdge("a", "nope"), RangeError, "unknown ids are rejected");

  // (8) degenerate graphs.
  const empty = new TaskGraph([]);
  assert.deepStrictEqual(prioritySchedule(empty), [], "an empty graph schedules nothing");
  assert.deepStrictEqual(waves(empty, computeLevels(empty, [])), [], "and has no waves");
  assert.strictEqual(countTopologicalOrderings(empty), 1, "the empty product is 1");
  const single = new TaskGraph(["only"]);
  assert.deepStrictEqual(prioritySchedule(single), ["only"]);
  close(single.tasks.get("only").priority, 0.0, 0.0, "default priority is 0");
}

module.exports = { TaskGraph, ReadyHeap, prioritySchedule, dfsTopologicalOrder, respectsEdges,
  computeLevels, longestPathFromSources, waves, allPermutations, bruteForceBestOrder,
  countTopologicalOrderings, lcgGraph };

if (require.main === module) {
  main();
  console.log("PASS");
}
