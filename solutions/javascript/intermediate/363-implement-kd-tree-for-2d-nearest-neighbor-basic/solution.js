/** Task 363 - Implement kd-tree for 2D nearest neighbor (basic). (Tier: Intermediate, CTCI Ch 4) */
"use strict";

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
  constructor(seed) {
    this.seed = seed;
  }

  next(modulus) {
    this.seed = (this.seed * 75 + 74) % 65537;
    return this.seed % modulus;
  }
}

/**
 * Axis alternates with depth and each node takes the median of its points sorted by that
 * axis (input index breaks ties), so the tree is balanced and fully deterministic.
 */
class KdTree {
  constructor(points) {
    this.points = points.map((p) => p.slice());
    const ids = this.points.map((_, i) => i);
    this.root = this._build(ids, 0);
  }

  _build(ids, depth) {
    if (ids.length === 0) return null;
    const axis = depth % 2;
    const sorted = ids.slice().sort((i, j) => {
      if (this.points[i][axis] !== this.points[j][axis]) {
        return this.points[i][axis] - this.points[j][axis];
      }
      return i - j;
    });
    const mid = sorted.length >> 1;
    const index = sorted[mid];
    return {
      point: this.points[index],
      index,
      axis,
      left: this._build(sorted.slice(0, mid), depth + 1),
      right: this._build(sorted.slice(mid + 1), depth + 1),
    };
  }

  /** Returns { distance, index } of the nearest stored point (squared distance). */
  nearest(query) {
    if (this.root === null) throw new Error("empty kd-tree");
    let best = null;
    const stack = [this.root];
    while (stack.length > 0) {
      const node = stack.pop();
      if (node === null) continue;
      const dx = query[0] - node.point[0];
      const dy = query[1] - node.point[1];
      const distance = dx * dx + dy * dy;
      if (best === null || distance < best.distance ||
          (distance === best.distance && node.index < best.index)) {
        best = { distance, index: node.index };
      }
      const offset = query[node.axis] - node.point[node.axis];
      const near = offset <= 0 ? node.left : node.right;
      const far = offset <= 0 ? node.right : node.left;
      // `<=` also explores the far side on ties so an equal-distance point with a
      // smaller input index can still win, exactly matching brute force.
      if (offset * offset <= best.distance) stack.push(far);
      stack.push(near);
    }
    return best;
  }
}

function bruteNearest(points, query) {
  let best = null;
  points.forEach((p, index) => {
    const dx = query[0] - p[0];
    const dy = query[1] - p[1];
    const distance = dx * dx + dy * dy;
    if (best === null || distance < best.distance ||
        (distance === best.distance && index < best.index)) {
      best = { distance, index };
    }
  });
  return best;
}

module.exports = { KdTree, bruteNearest, Lcg };

if (require.main === module) {
  const assert = require("node:assert");

  const sameResult = (tree, points, query, label) => {
    const got = tree.nearest(query);
    const want = bruteNearest(points, query);
    assert.strictEqual(got.distance, want.distance, `${label} distance`);
    assert.strictEqual(got.index, want.index, `${label} index`);
  };

  const hand = [[0, 0], [3, 4], [1, 1], [-5, -5]];
  const tree = new KdTree(hand);
  assert.deepStrictEqual(tree.nearest([3, 3]), { distance: 1, index: 1 });
  assert.deepStrictEqual(tree.nearest([1, 1]), { distance: 0, index: 2 });
  assert.deepStrictEqual(tree.nearest([0, 0]), { distance: 0, index: 0 });
  assert.deepStrictEqual(tree.nearest([-4, -6]), { distance: 2, index: 3 });
  assert.deepStrictEqual(tree.nearest([7, 7]), { distance: 25, index: 1 });
  for (const query of [[3, 3], [1, 1], [-4, -6], [7, 7], [-9, 4]]) {
    sameResult(tree, hand, query, "hand");
  }

  // Deterministic tie-break: four points at squared distance 2 from (1,1); index 0 wins.
  const tie = [[0, 0], [2, 0], [0, 2], [2, 2]];
  const tieTree = new KdTree(tie);
  assert.deepStrictEqual(tieTree.nearest([1, 1]), { distance: 2, index: 0 });
  sameResult(tieTree, tie, [1, 1], "tie set");

  // Duplicate points: the earliest input index among equal points must win.
  const dup = [[5, 5], [5, 5], [1, 1]];
  const dupTree = new KdTree(dup);
  assert.deepStrictEqual(dupTree.nearest([4, 4]), { distance: 2, index: 0 });
  assert.deepStrictEqual(dupTree.nearest([1, 1]), { distance: 0, index: 2 });

  const single = new KdTree([[-2, 7]]);
  assert.deepStrictEqual(single.nearest([0, 0]), { distance: 53, index: 0 });

  // LCG cross-check: many sets, queries both random and drawn from the set itself.
  const lcg = new Lcg(60601);
  for (let trial = 0; trial < 40; trial++) {
    const size = 1 + lcg.next(40);
    const points = [];
    for (let i = 0; i < size; i++) {
      points.push([lcg.next(101) - 50, lcg.next(101) - 50]);
    }
    const randomTree = new KdTree(points);
    for (const point of points) sameResult(randomTree, points, point, `in-set ${trial}`);
    for (let q = 0; q < 12; q++) {
      const query = [lcg.next(121) - 60, lcg.next(121) - 60];
      sameResult(randomTree, points, query, `random ${trial}`);
    }
  }

  // Duplicate-heavy sets (coordinates in a tiny window) agree with brute force too.
  for (let trial = 0; trial < 10; trial++) {
    const size = 2 + lcg.next(30);
    const points = [];
    for (let i = 0; i < size; i++) points.push([lcg.next(4), lcg.next(4)]);
    const randomTree = new KdTree(points);
    for (const query of [[0, 0], [1, 1], [3, 3], [2, 1], [1, 2]]) {
      sameResult(randomTree, points, query, `dense ${trial}`);
    }
  }
  console.log("PASS");
}
