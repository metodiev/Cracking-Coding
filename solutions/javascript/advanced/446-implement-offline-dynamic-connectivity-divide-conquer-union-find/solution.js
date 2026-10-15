/** Task 446 - Implement offline dynamic connectivity (divide & conquer + union find). (Tier: Advanced, CTCI Ch 4) */
"use strict";

class RollbackDSU {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.size = new Array(n).fill(1);
    this.history = [];
  }

  find(x) {
    while (this.parent[x] !== x) x = this.parent[x];
    return x;
  }

  union(a, b) {
    let ra = this.find(a);
    let rb = this.find(b);
    if (ra === rb) return false;
    if (this.size[ra] > this.size[rb]) {
      const swap = ra;
      ra = rb;
      rb = swap;
    }
    this.parent[ra] = rb;
    this.size[rb] += this.size[ra];
    this.history.push([ra, rb]);
    return true;
  }

  same(a, b) {
    return this.find(a) === this.find(b);
  }

  snapshot() {
    return this.history.length;
  }

  rollback(snapshot) {
    while (this.history.length > snapshot) {
      const [ra, rb] = this.history.pop();
      this.parent[ra] = ra;
      this.size[rb] -= this.size[ra];
    }
  }
}

function offlineDynamicConnectivity(n, ops) {
  const q = ops.length;
  if (q === 0) return [];
  const counts = new Map();
  const open = new Map();
  const intervals = [];
  for (let t = 0; t < q; t++) {
    const kind = ops[t][0];
    const key = `${Math.min(ops[t][1], ops[t][2])},${Math.max(ops[t][1], ops[t][2])}`;
    if (kind === 0) {
      const c = (counts.get(key) || 0) + 1;
      counts.set(key, c);
      if (c === 1) open.set(key, [ops[t][1], ops[t][2], t]);
    } else if (kind === 1) {
      const c = counts.get(key) || 0;
      if (c > 0) {
        counts.set(key, c - 1);
        if (c - 1 === 0) {
          const info = open.get(key);
          open.delete(key);
          if (info[2] <= t - 1) intervals.push([info[0], info[1], info[2], t - 1]);
        }
      }
    }
  }
  for (const info of open.values()) intervals.push([info[0], info[1], info[2], q - 1]);

  const dsu = new RollbackDSU(n);
  const answers = new Array(q).fill(null);

  function solve(lo, hi, active) {
    const snapshot = dsu.snapshot();
    const rest = [];
    for (const e of active) {
      if (e[2] <= lo && hi <= e[3]) dsu.union(e[0], e[1]);
      else rest.push(e);
    }
    if (lo === hi) {
      if (ops[lo][0] === 2) answers[lo] = dsu.same(ops[lo][1], ops[lo][2]);
    } else {
      const mid = (lo + hi) >> 1;
      solve(lo, mid, rest.filter((e) => e[2] <= mid && e[3] >= lo));
      solve(mid + 1, hi, rest.filter((e) => e[2] <= hi && e[3] >= mid + 1));
    }
    dsu.rollback(snapshot);
  }

  solve(0, q - 1, intervals);
  return answers;
}

function bruteAnswers(n, ops) {
  const counts = new Map();
  const answers = new Array(ops.length).fill(null);
  for (let t = 0; t < ops.length; t++) {
    const kind = ops[t][0];
    const key = `${Math.min(ops[t][1], ops[t][2])},${Math.max(ops[t][1], ops[t][2])}`;
    if (kind === 0) {
      counts.set(key, (counts.get(key) || 0) + 1);
    } else if (kind === 1) {
      const c = counts.get(key) - 1;
      if (c === 0) counts.delete(key);
      else counts.set(key, c);
    } else {
      const adjacency = Array.from({ length: n }, () => []);
      for (const edge of counts.keys()) {
        const [a, b] = edge.split(",").map(Number);
        adjacency[a].push(b);
        adjacency[b].push(a);
      }
      const seen = new Array(n).fill(false);
      seen[ops[t][1]] = true;
      const stack = [ops[t][1]];
      while (stack.length > 0) {
        const x = stack.pop();
        for (const y of adjacency[x]) {
          if (!seen[y]) {
            seen[y] = true;
            stack.push(y);
          }
        }
      }
      answers[t] = seen[ops[t][2]];
    }
  }
  return answers;
}

function lcgOpScripts(trials, seed, n, q) {
  const scripts = [];
  for (let t = 0; t < trials; t++) {
    const counts = new Map();
    const ops = [];
    for (let step = 0; step < q; step++) {
      seed = (seed * 75 + 74) % 65537;
      let kind = seed % 3;
      seed = (seed * 75 + 74) % 65537;
      let u = seed % n;
      seed = (seed * 75 + 74) % 65537;
      let v = seed % n;
      if (u === v) v = (v + 1) % n;
      const key = `${Math.min(u, v)},${Math.max(u, v)}`;
      if (kind === 1 && counts.size === 0) kind = 0;
      if (kind === 0) {
        ops.push([0, u, v]);
        counts.set(key, (counts.get(key) || 0) + 1);
      } else if (kind === 1) {
        const keys = Array.from(counts.keys()).sort();
        seed = (seed * 75 + 74) % 65537;
        const pick = keys[seed % keys.length];
        const [a, b] = pick.split(",").map(Number);
        ops.push([1, a, b]);
        const c = counts.get(pick) - 1;
        if (c === 0) counts.delete(pick);
        else counts.set(pick, c);
      } else {
        ops.push([2, u, v]);
      }
    }
    scripts.push(ops);
  }
  return scripts;
}

module.exports = { RollbackDSU, offlineDynamicConnectivity, bruteAnswers, lcgOpScripts };

if (require.main === module) {
  const assert = require("node:assert");

  let ops = [[0, 0, 1], [2, 0, 1], [0, 1, 2], [2, 0, 2], [1, 0, 1], [2, 0, 2], [2, 0, 0]];
  assert.deepStrictEqual(offlineDynamicConnectivity(3, ops), [null, true, null, true, null, false, true]);
  ops = [[2, 0, 1], [0, 0, 1], [2, 0, 1], [1, 0, 1], [2, 0, 1], [2, 1, 1]];
  assert.deepStrictEqual(offlineDynamicConnectivity(2, ops), [false, null, true, null, false, true]);

  let queries = 0;
  for (const script of lcgOpScripts(30, 446, 8, 40)) {
    const expected = bruteAnswers(8, script);
    queries += expected.filter((answer) => answer !== null).length;
    assert.deepStrictEqual(offlineDynamicConnectivity(8, script), expected);
  }
  assert.ok(queries > 100);

  for (const script of lcgOpScripts(10, 777, 12, 60)) {
    assert.deepStrictEqual(offlineDynamicConnectivity(12, script), bruteAnswers(12, script));
  }
  console.log("PASS");
}
