/** Task 112 - Prim's MST using heap. (Tier: Beginner, CTCI Ch 4) */
"use strict";

function prim(n, adj) {
  const visited = new Array(n).fill(false);
  let total = 0;
  const chosen = [];
  // simple array-backed min-heap of [weight, node, parent]
  const heap = [[0, 0, -1]];
  const push = (item) => {
    heap.push(item);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (heap[p][0] <= heap[i][0]) break;
      [heap[p], heap[i]] = [heap[i], heap[p]];
      i = p;
    }
  };
  const pop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      while (true) {
        let s = i;
        const l = 2 * i + 1, r = 2 * i + 2;
        if (l < heap.length && heap[l][0] < heap[s][0]) s = l;
        if (r < heap.length && heap[r][0] < heap[s][0]) s = r;
        if (s === i) break;
        [heap[i], heap[s]] = [heap[s], heap[i]];
        i = s;
      }
    }
    return top;
  };

  while (heap.length && chosen.length < n - 1) {
    const [weight, u, parent] = pop();
    if (visited[u]) continue;
    visited[u] = true;
    total += weight;
    if (parent !== -1) chosen.push([parent, u, weight]);
    for (const [v, w] of adj[u]) if (!visited[v]) push([w, v, u]);
  }
  return chosen.length === Math.max(0, n - 1) ? { total, chosen } : null;
}

module.exports = { prim };

if (require.main === module) {
  const assert = require("node:assert");
  const adj = [
    [[1, 4], [2, 3]],
    [[0, 4], [2, 1], [3, 2]],
    [[0, 3], [1, 1], [3, 4], [4, 5]],
    [[1, 2], [2, 4], [4, 2]],
    [[3, 2], [2, 5]],
  ];
  const result = prim(5, adj);
  assert.strictEqual(result.total, 8);
  assert.strictEqual(result.chosen.length, 4);
  assert.deepStrictEqual(prim(1, [[]]), { total: 0, chosen: [] });
  assert.strictEqual(prim(3, [[[1, 1]], [[0, 1]], []]), null);
  console.log("PASS");
}
