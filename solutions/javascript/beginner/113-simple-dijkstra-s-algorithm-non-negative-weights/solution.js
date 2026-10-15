/** Task 113 - Simple Dijkstra's algorithm (non-negative weights). (Tier: Beginner, CTCI Ch 4) */
"use strict";

function dijkstra(n, adj, src) {
  const INF = Infinity;
  const dist = new Array(n).fill(INF);
  dist[src] = 0;
  const heap = [[0, src]];
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

  while (heap.length) {
    const [d, u] = pop();
    if (d > dist[u]) continue;
    for (const [v, w] of adj[u]) {
      const nd = d + w;
      if (nd < dist[v]) {
        dist[v] = nd;
        push([nd, v]);
      }
    }
  }
  return dist;
}

module.exports = { dijkstra };

if (require.main === module) {
  const assert = require("node:assert");
  const adj = [
    [[1, 1], [2, 4]],
    [[2, 2], [3, 6]],
    [[3, 3]],
    [],
  ];
  assert.deepStrictEqual(dijkstra(4, adj, 0), [0, 1, 3, 6]);

  const adj2 = [[[1, 5]], [[0, 5]], []];
  const dist2 = dijkstra(3, adj2, 0);
  assert.strictEqual(dist2[2], Infinity);
  console.log("PASS");
}
