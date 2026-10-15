/** Task 471 - Implement vector clock for distributed event ordering. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability) */
"use strict";

const NODES = 3;

function lcg(seed) {
  return (seed * 75 + 74) % 65537;
}

function classify(a, b) {
  let le = true;
  let ge = true;
  for (let i = 0; i < a.length; i++) {
    if (a[i] > b[i]) le = false;
    if (a[i] < b[i]) ge = false;
  }
  if (le && ge) return "equal";
  if (le) return "before";
  if (ge) return "after";
  return "concurrent";
}

function featuredEvents() {
  const c = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const events = [];
  const emit = (node, kind, msg) => events.push({ node, kind, msg: msg === undefined ? -1 : msg, clock: c[node].slice() });
  c[0][0] += 1;
  emit(0, "L");                     // 0: A has 1 local event
  c[0][0] += 1;
  emit(0, "L");                     // 1: A has 2
  c[0][0] += 1;
  emit(0, "S", 0);                  // 2: A sends {A:3} to B
  c[1][1] += 1;
  emit(1, "L");                     // 3: B local tick, unaware of A
  c[1][0] = Math.max(c[1][0], 3);
  c[1][1] += 1;
  emit(1, "R", 0);                  // 4: {A:3, B:2}
  c[1][1] += 1;
  emit(1, "S", 1);                  // 5: B sends {A:3, B:3} to C
  c[2][0] = Math.max(c[2][0], 3);
  c[2][1] = Math.max(c[2][1], 3);
  c[2][2] += 1;
  emit(2, "R", 1);                  // 6: C merges -> {3,3,1}
  return events;
}

function genScript(count, seed) {
  const events = [];
  const pending = [[], [], []];
  const clocks = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  let nextMsg = 0;
  for (let k = 0; k < count; k++) {
    seed = lcg(seed);
    const node = seed % NODES;
    seed = lcg(seed);
    let kind = seed % 3;
    if (kind === 2 && pending[node].length === 0) kind = 0;
    if (kind === 1) {
      seed = lcg(seed);
      const target = (node + 1 + (seed % (NODES - 1))) % NODES;
      clocks[node][node] += 1;
      pending[target].push({ id: nextMsg, clock: clocks[node].slice() });
      events.push({ node, kind: "S", msg: nextMsg, clock: clocks[node].slice() });
      nextMsg += 1;
    } else if (kind === 2) {
      const m = pending[node].shift();
      for (let i = 0; i < NODES; i++) clocks[node][i] = Math.max(clocks[node][i], m.clock[i]);
      clocks[node][node] += 1;
      events.push({ node, kind: "R", msg: m.id, clock: clocks[node].slice() });
    } else {
      clocks[node][node] += 1;
      events.push({ node, kind: "L", msg: -1, clock: clocks[node].slice() });
    }
  }
  return events;
}

function causalReachability(events) {
  const n = events.length;
  const reach = Array.from({ length: n }, () => new Array(n).fill(false));
  const last = [-1, -1, -1];
  const sentAt = new Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    const e = events[i];
    if (last[e.node] !== -1) reach[last[e.node]][i] = true;
    last[e.node] = i;
    if (e.kind === "S") sentAt[e.msg] = i;
    else if (e.kind === "R") reach[sentAt[e.msg]][i] = true;
  }
  for (let k = 0; k < n; k++)
    for (let i = 0; i < n; i++)
      if (reach[i][k])
        for (let j = 0; j < n; j++)
          if (reach[k][j]) reach[i][j] = true;
  return reach;
}

module.exports = { classify, featuredEvents, genScript, causalReachability, lcg };

if (require.main === module) {
  const assert = require("node:assert");
  const fe = featuredEvents();
  const cl = fe.map((e) => e.clock);
  assert.deepStrictEqual(cl[0], [1, 0, 0]);
  assert.deepStrictEqual(cl[1], [2, 0, 0]);
  assert.deepStrictEqual(cl[2], [3, 0, 0]);
  assert.deepStrictEqual(cl[3], [0, 1, 0]);
  assert.deepStrictEqual(cl[4], [3, 2, 0]);
  assert.deepStrictEqual(cl[5], [3, 3, 0]);
  assert.deepStrictEqual(cl[6], [3, 3, 1]);
  assert.strictEqual(classify(cl[0], cl[1]), "before");
  assert.strictEqual(classify(cl[1], cl[2]), "before");
  assert.strictEqual(classify(cl[0], cl[2]), "before");
  assert.strictEqual(classify(cl[3], cl[0]), "concurrent");
  assert.strictEqual(classify(cl[3], cl[1]), "concurrent");
  assert.strictEqual(classify(cl[3], cl[2]), "concurrent");
  assert.strictEqual(classify(cl[3], cl[4]), "before");
  assert.strictEqual(classify(cl[2], cl[4]), "before");
  assert.strictEqual(classify(cl[4], cl[5]), "before");
  assert.strictEqual(classify(cl[5], cl[6]), "before");
  assert.strictEqual(classify(cl[1], cl[6]), "before");
  assert.strictEqual(classify(cl[6], cl[4]), "after");
  assert.strictEqual(classify(cl[6], cl[0]), "after");
  for (let i = 0; i < fe.length; i++)
    for (let j = i + 1; j < fe.length; j++) assert.notStrictEqual(classify(cl[i], cl[j]), "equal");

  const events = genScript(26, 1234);
  const kinds = { L: 0, S: 0, R: 0 };
  for (const e of events) kinds[e.kind] += 1;
  assert.deepStrictEqual(kinds, { L: 20, S: 4, R: 2 });
  const reach = causalReachability(events);
  let order = 0;
  let concurrent = 0;
  for (let i = 0; i < events.length; i++) {
    for (let j = i + 1; j < events.length; j++) {
      const got = classify(events[i].clock, events[j].clock);
      const expected = reach[i][j] ? "before" : reach[j][i] ? "after" : "concurrent";
      assert.strictEqual(got, expected, `pair ${i},${j}`);
      if (expected === "concurrent") concurrent += 1;
      else order += 1;
    }
  }
  assert.strictEqual(order, 140);
  assert.strictEqual(concurrent, 185);
  console.log("PASS");
}
