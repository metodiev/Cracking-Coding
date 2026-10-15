/** Task 270 - Implement A* algorithm for grid pathfinding. (Tier: Intermediate, CTCI Ch 4) */
"use strict";

/** Small binary min-heap driven by a comparator. */
class MinHeap {
  constructor(cmp) {
    this.cmp = cmp;
    this.data = [];
  }

  push(value) {
    this.data.push(value);
    this.siftUp(this.data.length - 1);
  }

  pop() {
    if (this.data.length === 0) throw new Error("heap empty");
    const top = this.data[0];
    const last = this.data.pop();
    if (this.data.length > 0) {
      this.data[0] = last;
      this.siftDown(0);
    }
    return top;
  }

  siftUp(i) {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.cmp(this.data[parent], this.data[i]) <= 0) break;
      [this.data[i], this.data[parent]] = [this.data[parent], this.data[i]];
      i = parent;
    }
  }

  siftDown(i) {
    const n = this.data.length;
    for (;;) {
      let smallest = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.cmp(this.data[l], this.data[smallest]) < 0) smallest = l;
      if (r < n && this.cmp(this.data[r], this.data[smallest]) < 0) smallest = r;
      if (smallest === i) break;
      [this.data[i], this.data[smallest]] = [this.data[smallest], this.data[i]];
      i = smallest;
    }
  }

  get size() { return this.data.length; }
}

function manhattan(a, b) {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
}

/**
 * A* on a 4-neighbour grid with unit move cost and the Manhattan heuristic, which is
 * admissible (one move changes the distance to the goal by at most 1) and consistent.
 * grid[r][c] === 1 is a wall. Returns the minimum-move path including both endpoints
 * as [r, c] pairs, or null when no path exists (also when start or goal lies on a wall).
 * Deterministic tie-break: the open set is ordered by (f, h, row-major cell id), so with
 * equal f the cell closer to the goal and then the smaller index expands first.
 */
function astar(grid, start, goal) {
  const rows = grid.length, cols = rows > 0 ? grid[0].length : 0;
  const free = (r, c) => r >= 0 && r < rows && c >= 0 && c < cols && grid[r][c] === 0;
  if (!free(start[0], start[1]) || !free(goal[0], goal[1])) return null;
  const [sr, sc] = start, [gr, gc] = goal;
  if (sr === gr && sc === gc) return [[sr, sc]];
  const key = (r, c) => r * cols + c;
  const heap = new MinHeap((a, b) => (a[0] - b[0]) || (a[1] - b[1]) || (a[2] - b[2]));
  const gScore = new Map([[key(sr, sc), 0]]);
  const parent = new Map();
  const closed = new Set();
  const dirs = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  const h0 = Math.abs(sr - gr) + Math.abs(sc - gc);
  heap.push([h0, h0, key(sr, sc), sr, sc]);         // (f, h, id, r, c)
  while (heap.size > 0) {
    const [, , , r, c] = heap.pop();
    const k = key(r, c);
    if (closed.has(k)) continue;
    closed.add(k);
    if (r === gr && c === gc) {
      const path = [[r, c]];
      let cur = k;
      while (cur !== key(sr, sc)) {
        const p = parent.get(cur);
        path.push([Math.floor(p / cols), p % cols]);
        cur = p;
      }
      path.reverse();
      return path;
    }
    for (const [dr, dc] of dirs) {
      const nr = r + dr, nc = c + dc;
      if (!free(nr, nc)) continue;
      const nk = key(nr, nc);
      if (closed.has(nk)) continue;
      const tentative = gScore.get(k) + 1;
      if (tentative < (gScore.has(nk) ? gScore.get(nk) : Infinity)) {
        gScore.set(nk, tentative);
        parent.set(nk, k);
        const hn = Math.abs(nr - gr) + Math.abs(nc - gc);
        heap.push([tentative + hn, hn, nk, nr, nc]);
      }
    }
  }
  return null;
}

function astarCost(grid, start, goal) {
  const path = astar(grid, start, goal);
  return path === null ? null : path.length - 1;
}

/** Exact distance from every free cell to goal (-1 when unreachable/walled). */
function bfsDistancesFrom(grid, goal) {
  const rows = grid.length, cols = rows > 0 ? grid[0].length : 0;
  const dist = Array.from({ length: rows }, () => new Array(cols).fill(-1));
  const free = (r, c) => r >= 0 && r < rows && c >= 0 && c < cols && grid[r][c] === 0;
  if (!free(goal[0], goal[1])) return dist;
  dist[goal[0]][goal[1]] = 0;
  let queue = [[goal[0], goal[1]]];
  while (queue.length > 0) {
    const next = [];
    for (const [r, c] of queue) {
      for (const [dr, dc] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nr = r + dr, nc = c + dc;
        if (free(nr, nc) && dist[nr][nc] === -1) {
          dist[nr][nc] = dist[r][c] + 1;
          next.push([nr, nc]);
        }
      }
    }
    queue = next;
  }
  return dist;
}

function bfsCost(grid, start, goal) {
  const dist = bfsDistancesFrom(grid, goal);
  if (dist.length === 0 || dist[0].length === 0) return -1;
  if (start[0] < 0 || start[0] >= dist.length || start[1] < 0 || start[1] >= dist[0].length) return -1;
  return dist[start[0]][start[1]];
}

function validPath(grid, path, start, goal) {
  if (path === null) return false;
  if (path[0][0] !== start[0] || path[0][1] !== start[1]) return false;
  const last = path[path.length - 1];
  if (last[0] !== goal[0] || last[1] !== goal[1]) return false;
  for (const [r, c] of path)
    if (!(r >= 0 && r < grid.length && c >= 0 && c < grid[0].length && grid[r][c] === 0)) return false;
  for (let i = 1; i < path.length; i++)
    if (Math.abs(path[i][0] - path[i - 1][0]) + Math.abs(path[i][1] - path[i - 1][1]) !== 1) return false;
  return true;
}

module.exports = { MinHeap, manhattan, astar, astarCost, bfsDistancesFrom, bfsCost, validPath };

if (require.main === module) {
  const assert = require("node:assert");
  const serpentine = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
  ];
  const detour = [
    [0, 0, 0, 0],
    [1, 1, 1, 0],
    [0, 0, 0, 0],
  ];
  const blocked = [
    [0, 1, 0],
    [0, 1, 0],
    [1, 1, 0],
  ];
  const openGrid = Array.from({ length: 4 }, () => new Array(6).fill(0));

  assert.strictEqual(astarCost(serpentine, [0, 0], [4, 4]), 8);
  assert.strictEqual(bfsCost(serpentine, [0, 0], [4, 4]), 8);
  assert.strictEqual(astarCost(detour, [0, 0], [2, 0]), 8);       // vs Manhattan 2
  assert.strictEqual(bfsCost(detour, [0, 0], [2, 0]), 8);
  assert.strictEqual(astarCost(detour, [2, 0], [0, 0]), 8);
  assert.strictEqual(astarCost(openGrid, [0, 0], [3, 5]), 8);     // == Manhattan on an open field
  assert.strictEqual(astarCost(blocked, [0, 0], [0, 2]), null);   // no path
  assert.strictEqual(bfsCost(blocked, [0, 0], [0, 2]), -1);

  assert.deepStrictEqual(astar(serpentine, [2, 2], [2, 2]), [[2, 2]]);   // start == goal
  assert.strictEqual(astar(serpentine, [1, 1], [0, 0]), null);           // start on a wall
  assert.strictEqual(astar(serpentine, [0, 0], [1, 1]), null);           // goal on a wall

  const cases = [
    [serpentine, [0, 0], [4, 4]],
    [detour, [0, 0], [2, 0]],
    [detour, [2, 0], [0, 0]],
    [openGrid, [0, 0], [3, 5]],
    [serpentine, [2, 2], [2, 2]],
  ];
  for (const [grid, start, goal] of cases) {
    const path = astar(grid, start, goal);
    assert.ok(validPath(grid, path, start, goal), `valid path ${start} -> ${goal}`);
    if (grid === openGrid) {                    // on an open grid Manhattan is exact
      const dist = bfsDistancesFrom(openGrid, goal);
      for (let r = 0; r < openGrid.length; r++)
        for (let c = 0; c < openGrid[0].length; c++)
          assert.strictEqual(Math.abs(r - goal[0]) + Math.abs(c - goal[1]), dist[r][c]);
    }
  }

  for (const [grid, goal] of [[serpentine, [4, 4]], [detour, [2, 0]], [blocked, [0, 2]]]) {
    const dist = bfsDistancesFrom(grid, goal);
    for (let r = 0; r < grid.length; r++)       // admissibility on every reachable cell
      for (let c = 0; c < grid[0].length; c++)
        if (dist[r][c] >= 0)
          assert.ok(Math.abs(r - goal[0]) + Math.abs(c - goal[1]) <= dist[r][c], "admissible");
  }

  // deterministic LCG so the cross-check is reproducible
  let seed = 270;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  const randInt = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1));
  for (let test = 0; test < 300; test++) {      // A* cost == BFS cost + valid path
    const rows = randInt(2, 6), cols = randInt(2, 6);
    const grid = Array.from({ length: rows }, () => Array.from({ length: cols }, () => (rand() < 0.3 ? 1 : 0)));
    const start = [randInt(0, rows - 1), randInt(0, cols - 1)];
    const goal = [randInt(0, rows - 1), randInt(0, cols - 1)];
    const got = astarCost(grid, start, goal);
    const want = bfsCost(grid, start, goal);
    assert.strictEqual(got === null ? -1 : got, want, `random cost #${test}`);
    if (got !== null) {
      assert.ok(validPath(grid, astar(grid, start, goal), start, goal), `random path #${test}`);
      if (want > 0) {
        const dist = bfsDistancesFrom(grid, goal);
        assert.ok(Math.abs(start[0] - goal[0]) + Math.abs(start[1] - goal[1]) <= dist[start[0]][start[1]],
          "random admissibility");
      }
    }
  }
  console.log("PASS");
}
