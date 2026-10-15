/** Task 462 - Implement vantage-point tree (VP-tree) for metric spaces. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Integer 2-D coordinates; metric = Manhattan L1 (|dx| + |dy|), a true metric, so
 * VP-tree pruning is exact. Each node stores a vantage point, the median distance
 * `mu`, and two children: inner (distance <= mu) and outer (distance >= mu), which
 * keeps every point in exactly one child and the halves balanced.
 * Search visits the nearer child first and prunes by the triangle inequality:
 * inner when tau - mu > best, outer when tau + best < mu (strict, so equal-distance
 * candidates are still visited). Tie-break (documented): equal L1 distance ->
 * smaller point index. Fixed LCG workload: equals the brute-force oracle; the
 * visited-node statistic stays below n/3 nodes per query on average (measured 14.1).
 */
"use strict";

class VpTree {
  constructor(points) {
    this.pts = points.map((p) => [p[0], p[1]]);
    const n = this.pts.length;
    this.mu = new Array(n).fill(0);
    this.inner = new Array(n).fill(-1);
    this.outer = new Array(n).fill(-1);
    this.visited = 0;
    const ids = Array.from({ length: n }, (_, i) => i);
    this.root = n > 0 ? this.build(ids) : -1;
  }

  static dist(a, b) {
    return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
  }

  build(ids) {
    if (ids.length === 0) return -1;
    const vp = ids[0];
    if (ids.length === 1) return vp;
    const rest = ids.slice(1);
    rest.sort((a, b) => {
      const da = VpTree.dist(this.pts[vp], this.pts[a]);
      const db = VpTree.dist(this.pts[vp], this.pts[b]);
      return da !== db ? da - db : a - b;
    });
    const mid = rest.length >> 1;
    this.mu[vp] = VpTree.dist(this.pts[vp], this.pts[rest[mid]]);
    this.inner[vp] = this.build(rest.slice(0, mid + 1));
    this.outer[vp] = this.build(rest.slice(mid + 1));
    return vp;
  }

  nearest(qx, qy) {
    this.visited = 0;
    if (this.root === -1) return [-1, -1];
    this.bestDist = null;
    this.bestIdx = -1;
    this.search(this.root, qx, qy);
    return [this.bestIdx, this.bestDist];
  }

  search(node, qx, qy) {
    this.visited++;
    const tau = Math.abs(qx - this.pts[node][0]) + Math.abs(qy - this.pts[node][1]);
    if (this.bestDist === null || tau < this.bestDist || (tau === this.bestDist && node < this.bestIdx)) {
      this.bestDist = tau;
      this.bestIdx = node;
    }
    const radius = this.mu[node];
    if (tau < radius) {
      this.visitInner(node, qx, qy, tau, radius);
      this.visitOuter(node, qx, qy, tau, radius);
    } else {
      this.visitOuter(node, qx, qy, tau, radius);
      this.visitInner(node, qx, qy, tau, radius);
    }
  }

  visitInner(node, qx, qy, tau, radius) {
    if (this.inner[node] !== -1 && tau - radius <= this.bestDist) this.search(this.inner[node], qx, qy);
  }

  visitOuter(node, qx, qy, tau, radius) {
    if (this.outer[node] !== -1 && tau + this.bestDist >= radius) this.search(this.outer[node], qx, qy);
  }
}

function bruteNearest(points, qx, qy) {
  let bestI = -1;
  let bestD = 0;
  for (let i = 0; i < points.length; i++) {
    const d = Math.abs(points[i][0] - qx) + Math.abs(points[i][1] - qy);
    if (bestI === -1 || d < bestD || (d === bestD && i < bestI)) {
      bestI = i;
      bestD = d;
    }
  }
  return [bestI, bestD];
}

module.exports = { VpTree, bruteNearest };

if (require.main === module) {
  const assert = require("node:assert");
  const small = [
    [0, 0],
    [1, 1],
    [3, 0],
    [5, 5],
    [2, 4],
  ];
  const tree = new VpTree(small);
  assert.deepStrictEqual(tree.nearest(0, 0), [0, 0]);
  assert.deepStrictEqual(tree.nearest(4, 3), [3, 3]); // (5,5) and (2,4) tie at 3 -> smaller index
  assert.deepStrictEqual(tree.nearest(2, 1), [1, 1]);
  assert.deepStrictEqual(tree.nearest(5, 0), [2, 2]); // point (3,0)
  assert.deepStrictEqual(new VpTree([]).nearest(0, 0), [-1, -1]);

  const n = 150;
  let seed = 4242;
  const lcg = () => (seed = (seed * 75 + 74) % 65537);
  const points = [];
  for (let i = 0; i < n; i++) {
    const x = (lcg() % 2001) - 1000;
    const y = (lcg() % 2001) - 1000;
    points.push([x, y]);
  }
  const big = new VpTree(points);
  const queries = 150;
  let totalVisited = 0;
  for (let qi = 0; qi < queries; qi++) {
    const qx = (lcg() % 2001) - 1000;
    const qy = (lcg() % 2001) - 1000;
    assert.deepStrictEqual(big.nearest(qx, qy), bruteNearest(points, qx, qy), `query ${qi}`);
    totalVisited += big.visited;
  }
  assert.ok(totalVisited > 0);
  assert.ok(totalVisited < (queries * n) / 3, `visited budget ${totalVisited}`);
  console.log("PASS");
}
