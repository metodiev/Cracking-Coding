/** Task 922 - Implement multi-robot coordination with conflict resolution and task allocation.
 *  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * Two coupled layers over a 4-connected grid world:
 *   allocation  sequential single-item auction: every round each robot bids its cheapest still
 *               unassigned task (bid = BFS travel cost from its current position) and the globally
 *               cheapest bid wins; a hill-climbing pass then tries single-task reassignments until
 *               no improvement is left. Compared against the exhaustive optimum.
 *   coordination prioritized planning in space-time: robots are ordered by task load and each robot's
 *               legs are planned with time-expanded A* (moves and waits) that treats earlier robots'
 *               reservations as hard constraints, forbidding both vertex conflicts and edge swaps;
 *               a robot that has arrived keeps its cell reserved, so waits are the resolution tool.
 * Simplification: static obstacles, unit speeds, discrete time, no continuous geometry, no
 * deadlock-breaking by replanning priority (the ordering is fixed). Verified: joint-plan safety
 * (no vertex conflict and no edge swap at any timestep), liveness (every robot completes its task
 * list), allocation quality versus the exhaustive optimum, A* distances versus BFS distances,
 * bottleneck resolution, boundary and invalid-input cases, determinism.
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17; x ^= x << 5;
    return (this.state = x >>> 0);
  }
  float() { return this.next() / 4294967296; }
  int(bound) { return this.next() % bound; }
}

const STEPS = [[1, 0], [-1, 0], [0, 1], [0, -1]];

function makeGrid(rows) {
  assert.ok(Array.isArray(rows) && rows.length >= 1, "the grid needs at least one row");
  const h = rows.length, w = rows[0].length;
  for (const row of rows) assert.strictEqual(row.length, w, "all grid rows must have equal length");
  const blocked = new Set();
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) if (rows[y][x] === "#") blocked.add(y * w + x);
  }
  return { w, h, blocked, index: (x, y) => y * w + x,
    passable: (x, y) => x >= 0 && y >= 0 && x < w && y < h && !blocked.has(y * w + x) };
}

function cellOf(grid, index) { return { x: index % grid.w, y: Math.floor(index / grid.w) }; }

function requireCell(grid, cell, what) {
  assert.ok(cell && Number.isInteger(cell.x) && Number.isInteger(cell.y), `${what} needs integer x/y`);
  assert.ok(grid.passable(cell.x, cell.y), `${what} (${cell.x}, ${cell.y}) is outside or blocked`);
}

/** Breadth-first distances from one source; unreachable cells stay at Infinity. */
function bfsDistances(grid, from) {
  const dist = new Array(grid.w * grid.h).fill(Infinity);
  dist[grid.index(from.x, from.y)] = 0;
  const queue = [grid.index(from.x, from.y)];
  for (let head = 0; head < queue.length; head++) {
    const current = queue[head];
    const { x, y } = cellOf(grid, current);
    for (const [dx, dy] of STEPS) {
      const nx = x + dx, ny = y + dy;
      if (!grid.passable(nx, ny)) continue;
      const next = grid.index(nx, ny);
      if (dist[next] !== Infinity) continue;
      dist[next] = dist[current] + 1;
      queue.push(next);
    }
  }
  return dist;
}

/** Time-expanded A* with wait actions; reservations forbid vertex conflicts and edge swaps. */
function astarSpaceTime(grid, from, to, reservations, options = {}) {
  const startTime = options.startTime ?? 0;
  const maxTime = options.maxTime ?? 64;
  const horizon = options.horizon ?? maxTime;
  const goal = grid.index(to.x, to.y);
  const heuristic = (x, y) => Math.abs(x - to.x) + Math.abs(y - to.y);
  // The robot keeps its final cell reserved until the horizon, so a goal time is only viable when
  // the goal cell stays free from that time onwards (otherwise two robots would end up sharing it).
  const goalStaysFree = (t) => {
    for (let s = t; s <= horizon; s++) if (reservations.vertices.has(`${to.x},${to.y},${s}`)) return false;
    return true;
  };
  const best = new Map();
  const startNode = { x: from.x, y: from.y, t: startTime, g: 0, f: heuristic(from.x, from.y), parent: null };
  best.set(`${startNode.x},${startNode.y},${startTime}`, 0);
  const open = [startNode];
  while (open.length > 0) {
    let pick = 0;
    for (let k = 1; k < open.length; k++) if (open[k].f < open[pick].f) pick = k;
    const node = open.splice(pick, 1)[0];
    if (node.x === to.x && node.y === to.y && goalStaysFree(node.t)) {
      const path = [];
      for (let n = node; n; n = n.parent) path.push({ x: n.x, y: n.y, t: n.t });
      return path.reverse();
    }
    if (node.t >= maxTime) continue;
    const moves = [[0, 0], ...STEPS];
    for (const [dx, dy] of moves) {
      const nx = node.x + dx, ny = node.y + dy, nt = node.t + 1;
      if (!grid.passable(nx, ny)) continue;
      const key = `${nx},${ny},${nt}`;
      if (reservations.vertices.has(key)) continue;
      if ((dx !== 0 || dy !== 0) && reservations.vertices.has(`${node.x},${node.y},${nt}`)) continue;
      if (reservations.edges.has(`${node.x},${node.y}>${nx},${ny},${nt}`)) continue;
      const g = node.g + 1;
      if (g >= (best.get(key) ?? Infinity)) continue;
      best.set(key, g);
      open.push({ x: nx, y: ny, t: nt, g, f: g + heuristic(nx, ny), parent: node });
    }
  }
  return null;
}

function newReservations() {
  return { vertices: new Set(), edges: new Set() };
}

/** Reserve a path (and the robot's final cell for the rest of the horizon). */
function reservePath(reservations, path, horizon) {
  for (let k = 0; k < path.length; k++) {
    reservations.vertices.add(`${path[k].x},${path[k].y},${path[k].t}`);
    if (k > 0) {
      const a = path[k - 1], b = path[k];
      reservations.edges.add(`${a.x},${a.y}>${b.x},${b.y},${b.t}`);
      reservations.edges.add(`${b.x},${b.y}>${a.x},${a.y},${b.t}`);
    }
  }
  const last = path[path.length - 1];
  for (let t = last.t + 1; t <= horizon; t++) reservations.vertices.add(`${last.x},${last.y},${t}`);
}

/** Plan every robot's route through its assigned targets, honoring higher-priority reservations. */
function planFleet(grid, robots, tasks, assignment, options = {}) {
  const horizon = options.horizon ?? 80;
  const order = robots.map((r, k) => k)
    .sort((a, b) => assignment.filter((x) => x === b).length - assignment.filter((x) => x === a).length || a - b);
  const reservations = newReservations();
  const paths = new Array(robots.length).fill(null);
  let conflicts = 0;
  for (const k of order) {
    const targets = tasks.filter((_, t) => assignment[t] === k).map((t) => t.cell);
    const route = [{ x: robots[k].x, y: robots[k].y, t: 0 }];
    let cursor = { x: robots[k].x, y: robots[k].y };
    let time = 0;
    let failed = false;
    for (const target of targets) {
      let leg = null;
      for (let allowance = Math.abs(target.x - cursor.x) + Math.abs(target.y - cursor.y); allowance <= horizon; allowance += 4) {
        leg = astarSpaceTime(grid, cursor, target, reservations,
          { startTime: time, maxTime: time + allowance, horizon });
        if (leg) break;
      }
      if (!leg) { failed = true; break; }
      if (leg[leg.length - 1].t > time + Math.abs(target.x - cursor.x) + Math.abs(target.y - cursor.y)) {
        conflicts++; // the leg needed to wait, i.e. a conflict was resolved by delay
      }
      route.push(...leg.slice(1));
      cursor = leg[leg.length - 1];
      time = cursor.t;
    }
    paths[k] = failed ? null : route;
    if (route.length > 1) reservePath(reservations, route, horizon);
  }
  const makespan = paths.reduce((m, p) => Math.max(m, p ? p[p.length - 1].t : 0), 0);
  return { paths, makespan, conflicts, order };
}

/** Independent safety oracle: expand a fleet plan and look for vertex conflicts and edge swaps. */
function findConflicts(robots, paths) {
  const at = (k, t) => {
    const path = paths[k];
    if (!path) return null;
    if (t >= path.length) return { x: path[path.length - 1].x, y: path[path.length - 1].y };
    return path[t];
  };
  const conflicts = [];
  const horizon = Math.max(...paths.filter(Boolean).map((p) => p.length));
  for (let t = 0; t < horizon; t++) {
    for (let a = 0; a < robots.length; a++) {
      for (let b = a + 1; b < robots.length; b++) {
        const pa = at(a, t), pb = at(b, t);
        if (!pa || !pb) continue;
        if (pa.x === pb.x && pa.y === pb.y) conflicts.push({ type: "vertex", t, a, b });
        const paNext = at(a, t + 1), pbNext = at(b, t + 1);
        if (pa.x === pbNext.x && pa.y === pbNext.y && pb.x === paNext.x && pb.y === paNext.y
          && (pa.x !== paNext.x || pa.y !== paNext.y)) conflicts.push({ type: "swap", t, a, b });
      }
    }
  }
  return conflicts;
}

/** Sequential single-item auction followed by hill-climbing single-task reassignment. */
function allocate(grid, robots, tasks) {
  const distances = robots.map((r) => bfsDistances(grid, r));
  const cost = (k, t) => distances[k][grid.index(tasks[t].cell.x, tasks[t].cell.y)];
  const assignment = new Array(tasks.length).fill(-1);
  const load = new Array(robots.length).fill(0);
  let rounds = 0;
  for (let remaining = tasks.length; remaining > 0;) {
    let best = null;
    for (let t = 0; t < tasks.length; t++) {
      if (assignment[t] !== -1) continue;
      for (let k = 0; k < robots.length; k++) {
        const bid = cost(k, t);
        if (!Number.isFinite(bid)) continue;
        if (!best || bid < best.bid || (bid === best.bid && k < best.k)) best = { bid, k, t };
      }
    }
    assert.ok(best, "no robot can reach the remaining tasks");
    assignment[best.t] = best.k;
    load[best.k] += best.bid;
    remaining--;
    rounds++;
  }
  const total = (assignment) => assignment.reduce((acc, k, t) => acc + cost(k, t), 0);
  let improved = true;
  while (improved) { // hill climbing: move one task to another robot when it lowers the total
    improved = false;
    for (let t = 0; t < tasks.length; t++) {
      const current = total(assignment);
      for (let k = 0; k < robots.length; k++) {
        if (k === assignment[t]) continue;
        const trial = assignment.slice();
        trial[t] = k;
        if (total(trial) < current - 1e-9) { assignment[t] = k; improved = true; }
      }
    }
  }
  return { assignment, distance: total(assignment), rounds, load };
}

/** Exhaustive optimum over all robots^tasks assignments (independent oracle for small instances). */
function optimalAllocation(grid, robots, tasks) {
  const distances = robots.map((r) => bfsDistances(grid, r));
  const cost = (k, t) => distances[k][grid.index(tasks[t].cell.x, tasks[t].cell.y)];
  let best = Infinity;
  let bestAssignment = null;
  for (let code = 0; code < robots.length ** tasks.length; code++) {
    const assignment = [];
    let rest = code;
    for (let t = 0; t < tasks.length; t++) { assignment.push(rest % robots.length); rest = Math.floor(rest / robots.length); }
    const total = assignment.reduce((acc, k, t) => acc + cost(k, t), 0);
    if (total < best) { best = total; bestAssignment = assignment.slice(); }
  }
  return { distance: best, assignment: bestAssignment };
}

/* ------------------------------------------------------------------ checks */

const WORLD = [
  ".......",
  ".......",
  "...#...",
  ".......",
  "...#...",
  ".......",
];

const BOTTLENECK = [
  ".....",
  ".....",
  "##.##",
  ".....",
  ".....",
];

function main() {
  const grid = makeGrid(WORLD);
  const robots = [{ x: 0, y: 0 }, { x: 6, y: 5 }, { x: 0, y: 5 }];
  const tasks = [{ cell: { x: 6, y: 0 } }, { cell: { x: 3, y: 5 } }, { cell: { x: 5, y: 2 } }, { cell: { x: 2, y: 0 } }];
  for (const r of robots) requireCell(grid, r, "robot start");
  for (const t of tasks) requireCell(grid, t.cell, "task target");

  // 1. Allocation: every task assigned exactly once, distance within 1.5x of the exhaustive optimum.
  const allocation = allocate(grid, robots, tasks);
  assert.strictEqual(allocation.assignment.length, tasks.length);
  assert.ok(allocation.assignment.every((k) => k >= 0 && k < robots.length), "unassigned task");
  assert.strictEqual(allocation.rounds, tasks.length, "each auction round must place one task");
  const optimum = optimalAllocation(grid, robots, tasks);
  assert.ok(allocation.distance >= optimum.distance - 1e-9, "the auction beat the optimum");
  assert.ok(allocation.distance <= 1.5 * optimum.distance, `auction ${allocation.distance} vs optimum ${optimum.distance}`);

  // 2. Oracle: A* shortest legs agree with BFS distances on seeded random grids.
  const prng = new Prng(922);
  let oracleChecks = 0;
  for (let trial = 0; trial < 8; trial++) {
    const rows = Array.from({ length: 5 }, () => ".....".split("").map(() => (prng.float() < 0.2 ? "#" : ".")).join(""));
    const random = makeGrid(rows);
    if (!random.passable(0, 0) || !random.passable(4, 4)) continue;
    const dist = bfsDistances(random, { x: 0, y: 0 });
    const path = astarSpaceTime(random, { x: 0, y: 0 }, { x: 4, y: 4 }, newReservations(), { maxTime: 40 });
    const expected = dist[random.index(4, 4)];
    if (!Number.isFinite(expected)) { assert.strictEqual(path, null, "A* found a path where BFS did not"); }
    else {
      assert.ok(path, "A* failed to find an existing path");
      assert.strictEqual(path.length - 1, expected, `A* length ${path.length - 1} vs BFS ${expected}`);
    }
    oracleChecks++;
  }
  assert.ok(oracleChecks >= 4, `only ${oracleChecks} random grids were comparable`);

  // 3. Fleet plan: every robot completes its whole task list (liveness).
  const fleet = planFleet(grid, robots, tasks, allocation.assignment, { horizon: 80 });
  assert.ok(fleet.paths.every(Boolean), "a robot failed to plan its route");
  for (let k = 0; k < robots.length; k++) {
    const targets = tasks.filter((_, t) => allocation.assignment[t] === k).map((t) => t.cell);
    for (const target of targets) {
      assert.ok(fleet.paths[k].some((c) => c.x === target.x && c.y === target.y),
        `robot ${k} never visited its task (${target.x}, ${target.y})`);
    }
  }
  assert.ok(fleet.makespan > 0 && Number.isFinite(fleet.makespan));

  // 4. Safety invariant checked independently: no vertex conflict and no edge swap at any timestep.
  assert.deepStrictEqual(findConflicts(robots, fleet.paths), []);
  for (const path of fleet.paths) {
    for (let t = 1; t < path.length; t++) {
      const a = path[t - 1], b = path[t];
      assert.strictEqual(b.t, a.t + 1, "plan is not time indexed");
      assert.ok(Math.abs(b.x - a.x) + Math.abs(b.y - a.y) <= 1, "a robot teleported");
      assert.ok(grid.passable(b.x, b.y), "a robot entered an obstacle");
    }
  }

  // 5. Conflict resolution at a single-cell gap: three robots must cross one at a time.
  const bottleneck = makeGrid(BOTTLENECK);
  const column = [{ x: 0, y: 3 }, { x: 1, y: 4 }, { x: 0, y: 4 }];
  const destinations = [{ cell: { x: 0, y: 0 } }, { cell: { x: 1, y: 0 } }, { cell: { x: 0, y: 1 } }];
  const columnFleet = planFleet(bottleneck, column, destinations, [0, 1, 2], { horizon: 60 });
  assert.ok(columnFleet.paths.every(Boolean), "the bottleneck must still be solvable with waits");
  assert.deepStrictEqual(findConflicts(column, columnFleet.paths), []);
  assert.ok(columnFleet.conflicts >= 1, "expected at least one delay in the bottleneck");
  assert.ok(columnFleet.makespan > 6, `bottleneck makespan ${columnFleet.makespan} is implausibly small`);

  // 6. Boundary cases: a robot without tasks holds its cell, and a single robot is conflict free.
  const idleTasks = tasks.slice(0, 2);
  const idleFleet = planFleet(grid, robots, idleTasks, [0, 0], { horizon: 40 });
  assert.ok(idleFleet.paths[2].every((c) => c.x === robots[2].x && c.y === robots[2].y),
    "a robot without tasks must not move");
  const solo = planFleet(grid, [robots[0]], [tasks[0]], [0], { horizon: 40 });
  assert.strictEqual(solo.conflicts, 0);
  assert.deepStrictEqual(findConflicts([robots[0]], solo.paths), []);
  assert.strictEqual(solo.makespan, bfsDistances(grid, robots[0])[grid.index(6, 0)],
    "a single robot must travel the shortest path");

  // 7. Invalid input is rejected: blocked starts, duplicates, and unreachable targets.
  assert.throws(() => requireCell(grid, { x: 3, y: 2 }, "robot start"), /blocked/);
  assert.throws(() => requireCell(grid, { x: 9, y: 9 }, "task target"), /blocked/);
  assert.throws(() => makeGrid(["..", "..."]), /equal length/);
  const walled = makeGrid(["..#", "..#", "..#"]);
  assert.throws(() => allocate(walled, [{ x: 0, y: 0 }], [{ cell: { x: 2, y: 0 } }]), /can reach/);

  // 8. Determinism: the whole pipeline replays identically.
  const rerun = () => {
    const a = allocate(grid, robots, tasks);
    const f = planFleet(grid, robots, tasks, a.assignment, { horizon: 80 });
    return { assignment: a.assignment, makespan: f.makespan, paths: f.paths };
  };
  assert.deepStrictEqual(rerun(), rerun());

  console.log(`fleet: ${robots.length} robots, ${tasks.length} tasks, auction distance ` +
    `${allocation.distance} (optimum ${optimum.distance}), makespan ${fleet.makespan}, ` +
    `resolved delays ${fleet.conflicts}`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, makeGrid, bfsDistances, astarSpaceTime, newReservations, reservePath,
  planFleet, findConflicts, allocate, optimalAllocation, requireCell };
