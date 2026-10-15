/** Task 176 - Check matrix connectivity (graph view). (Tier: Beginner, CTCI Ch 4) */
"use strict";

function isConnected(adj) {
  const n = adj.length;
  if (n <= 1) return true;
  const seen = new Array(n).fill(false);
  seen[0] = true;
  const q = [0];
  let reached = 1;
  while (q.length) {
    const u = q.shift();
    for (let v = 0; v < n; v++) {
      if (adj[u][v] === 1 && !seen[v]) {
        seen[v] = true;
        reached++;
        q.push(v);
      }
    }
  }
  return reached === n;
}

function connectedComponents(adj) {
  const n = adj.length;
  const seen = new Array(n).fill(false);
  let components = 0;
  for (let start = 0; start < n; start++) {
    if (seen[start]) continue;
    components++;
    seen[start] = true;
    const q = [start];
    while (q.length) {
      const u = q.shift();
      for (let v = 0; v < n; v++) {
        if (adj[u][v] === 1 && !seen[v]) {
          seen[v] = true;
          q.push(v);
        }
      }
    }
  }
  return components;
}

module.exports = { isConnected, connectedComponents };

if (require.main === module) {
  const assert = require("node:assert");
  const path = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]];
  assert.strictEqual(isConnected(path), true);
  assert.strictEqual(connectedComponents(path), 1);

  const split = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]];
  assert.strictEqual(isConnected(split), false);
  assert.strictEqual(connectedComponents(split), 2);

  const lonely = [[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]];
  assert.strictEqual(isConnected(lonely), false);
  assert.strictEqual(connectedComponents(lonely), 2);

  assert.strictEqual(isConnected([]), true);
  assert.strictEqual(isConnected([[0]]), true);
  console.log("PASS");
}
