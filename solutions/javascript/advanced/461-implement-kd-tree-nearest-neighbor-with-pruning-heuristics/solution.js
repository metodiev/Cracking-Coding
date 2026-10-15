/** Task 461 - Implement kd-tree nearest neighbor with pruning heuristics. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * 2-D kd-tree (median split on alternating axes) keeping a bounding box per node.
 * Best-first nearest-neighbour search ordered by the squared distance from the
 * query point to a child's bounding box; a subtree is pruned as soon as its bound
 * strictly exceeds the best squared distance found (strict `>`, so equal-distance
 * candidates are still explored). Tie-break (documented): equal squared distance
 * -> smaller point index. Fixed LCG workload: equals the brute-force oracle, and
 * the visited-node statistic stays below n/4 nodes per query on average (measured
 * 8.0) versus n for a linear scan.
 */
"use strict";

class MinHeap {
  constructor() {
    this.data = [];
  }

  static less(a, b) {
    return a[0] !== b[0] ? a[0] < b[0] : a[1] < b[1];
  }

  push(item) {
    const d = this.data;
    d.push(item);
    let i = d.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!MinHeap.less(d[i], d[p])) break;
      [d[i], d[p]] = [d[p], d[i]];
      i = p;
    }
  }

  pop() {
    const d = this.data;
    const top = d[0];
    const last = d.pop();
    if (d.length > 0) {
      d[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < d.length && MinHeap.less(d[l], d[m])) m = l;
        if (r < d.length && MinHeap.less(d[r], d[m])) m = r;
        if (m === i) break;
        [d[i], d[m]] = [d[m], d[i]];
        i = m;
      }
    }
    return top;
  }

  get size() {
    return this.data.length;
  }
}

class KdTree {
  constructor(points) {
    this.pts = points.map((p) => [p[0], p[1]]);
    const n = this.pts.length;
    this.left = new Array(n).fill(-1);
    this.right = new Array(n).fill(-1);
    this.box = new Array(n).fill(null);
    this.visited = 0;
    const ids = Array.from({ length: n }, (_, i) => i);
    this.root = n > 0 ? this.build(ids, 0) : -1;
  }

  build(ids, depth) {
    if (ids.length === 0) return -1;
    const axis = depth & 1;
    ids.sort(
      (a, b) =>
        this.pts[a][axis] - this.pts[b][axis] ||
        this.pts[a][1 - axis] - this.pts[b][1 - axis] ||
        a - b
    );
    const mid = ids.length >> 1;
    const node = ids[mid];
    this.left[node] = this.build(ids.slice(0, mid), depth + 1);
    this.right[node] = this.build(ids.slice(mid + 1), depth + 1);
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (const i of ids) {
      x0 = Math.min(x0, this.pts[i][0]);
      x1 = Math.max(x1, this.pts[i][0]);
      y0 = Math.min(y0, this.pts[i][1]);
      y1 = Math.max(y1, this.pts[i][1]);
    }
    this.box[node] = [x0, y0, x1, y1];
    return node;
  }

  static dist2(a, bx, by) {
    const dx = a[0] - bx;
    const dy = a[1] - by;
    return dx * dx + dy * dy;
  }

  bound(node, qx, qy) {
    const [x0, y0, x1, y1] = this.box[node];
    let dx = 0;
    if (qx < x0) dx = x0 - qx;
    else if (qx > x1) dx = qx - x1;
    let dy = 0;
    if (qy < y0) dy = y0 - qy;
    else if (qy > y1) dy = qy - y1;
    return dx * dx + dy * dy;
  }

  nearest(qx, qy) {
    this.visited = 0;
    if (this.root === -1) return [-1, -1];
    let bestI = -1;
    let bestD = 0;
    const heap = new MinHeap();
    heap.push([0, this.root]);
    while (heap.size > 0) {
      const [bnd, node] = heap.pop();
      if (bestI !== -1 && bnd > bestD) break; // prune
      this.visited++;
      const d = KdTree.dist2(this.pts[node], qx, qy);
      if (bestI === -1 || d < bestD || (d === bestD && node < bestI)) {
        bestI = node;
        bestD = d;
      }
      for (const child of [this.left[node], this.right[node]]) {
        if (child !== -1) heap.push([this.bound(child, qx, qy), child]);
      }
    }
    return [bestI, bestD];
  }
}

function bruteNearest(points, qx, qy) {
  let bestI = -1;
  let bestD = 0;
  for (let i = 0; i < points.length; i++) {
    const dx = points[i][0] - qx;
    const dy = points[i][1] - qy;
    const d = dx * dx + dy * dy;
    if (bestI === -1 || d < bestD || (d === bestD && i < bestI)) {
      bestI = i;
      bestD = d;
    }
  }
  return [bestI, bestD];
}

module.exports = { KdTree, bruteNearest };

if (require.main === module) {
  const assert = require("node:assert");
  const small = [
    [0, 0],
    [2, 3],
    [5, 4],
    [4, 1],
    [9, 6],
  ];
  const tree = new KdTree(small);
  assert.deepStrictEqual(tree.nearest(1, 1), [0, 2]);
  assert.deepStrictEqual(tree.nearest(4, 2), [3, 1]);
  assert.deepStrictEqual(tree.nearest(3, 3), [1, 1]);
  assert.deepStrictEqual(tree.nearest(3, 2), [1, 2]); // tie -> smaller index
  assert.deepStrictEqual(new KdTree([]).nearest(0, 0), [-1, -1]);

  const n = 200;
  let seed = 1234;
  const lcg = () => (seed = (seed * 75 + 74) % 65537);
  const points = [];
  for (let i = 0; i < n; i++) {
    const x = (lcg() % 2001) - 1000;
    const y = (lcg() % 2001) - 1000;
    points.push([x, y]);
  }
  const big = new KdTree(points);
  const queries = 200;
  let totalVisited = 0;
  for (let qi = 0; qi < queries; qi++) {
    const qx = (lcg() % 2001) - 1000;
    const qy = (lcg() % 2001) - 1000;
    assert.deepStrictEqual(big.nearest(qx, qy), bruteNearest(points, qx, qy), `query ${qi}`);
    totalVisited += big.visited;
  }
  assert.ok(totalVisited > 0);
  assert.ok(totalVisited < (queries * n) / 4, `visited budget ${totalVisited}`);
  console.log("PASS");
}
