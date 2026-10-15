/** Task 480 - Implement write-ahead logging (WAL) + crash recovery simulation. (Tier: Advanced, CTCI Ch 14 — Databases) */
"use strict";

class WALEngine {
  constructor() {
    this.state = new Map();
    this.wal = []; // [lsn, op, key, value]
    this.lsn = 0;
    this.durableLsn = 0;
    this.checkpointState = null;
    this.checkpointLsn = 0;
    this.checkpointCount = 0;
    this.crashCount = 0;
    this.lastReplayCount = 0;
  }

  set(key, value) {
    this.lsn += 1;
    this.wal.push([this.lsn, "set", key, value]);
    this.state.set(key, value);
  }

  delete(key) {
    this.lsn += 1;
    this.wal.push([this.lsn, "del", key, null]);
    this.state.delete(key);
  }

  flush() {
    this.durableLsn = this.lsn;
  }

  checkpointNow() {
    this.flush();
    this.checkpointState = new Map(this.state);
    this.checkpointLsn = this.lsn;
    this.checkpointCount += 1;
  }

  walText(tornTail) {
    let text = "";
    for (const [recordLsn, op, key, value] of this.wal) {
      text += `${recordLsn}|${op}|${key}|${value === null ? "" : value}\n`;
    }
    if (tornTail) text += tornTail;
    return text;
  }

  crash(tornTail) {
    const onDisk = this.walText(tornTail);
    const durable = this.durableLsn;
    const result = WALEngine.recover(onDisk, durable, this.checkpointState, this.checkpointLsn);
    this.state = result.state;
    this.lastReplayCount = result.replayed;
    this.lsn = durable;
    this.wal = this.wal.slice(0, Math.min(durable, this.wal.length)); // truncate past watermark
    this.durableLsn = durable;
    this.crashCount += 1;
    return this.state;
  }

  static recover(walText, durableLsn, checkpointState, checkpointLsn) {
    const state = checkpointState === null ? new Map() : new Map(checkpointState);
    const snapshotLsn = checkpointState === null ? 0 : checkpointLsn;
    let replayed = 0;
    for (const line of walText.split("\n")) {
      if (line === "") continue;
      const parts = line.split("|");
      if (parts.length !== 4) break; // torn record: stop, drop it
      const recordLsn = Number(parts[0]);
      if (recordLsn <= snapshotLsn) continue;
      if (recordLsn > durableLsn) break; // never made durable: lost
      if (parts[1] === "set") state.set(parts[2], Number(parts[3]));
      else state.delete(parts[2]);
      replayed += 1;
    }
    return { state, replayed };
  }
}

function lcg(seed) {
  return (seed * 75 + 74) % 65537;
}

function mapsEqual(a, b) {
  if (a.size !== b.size) return false;
  for (const [k, v] of a) if (b.get(k) !== v) return false;
  return true;
}

function runRandom() {
  const assert = require("node:assert");
  const eng = new WALEngine();
  let oracleLive = new Map();
  let oracleDurable = new Map();
  let crashes = 0;
  let seed = 8642;
  for (let i = 0; i < 60; i++) {
    seed = lcg(seed);
    const action = seed % 10;
    seed = lcg(seed);
    const key = `k${seed % 12}`;
    seed = lcg(seed);
    const value = seed % 1000;
    if (action < 5) {
      eng.set(key, value);
      oracleLive.set(key, value);
    } else if (action < 7) {
      eng.delete(key);
      oracleLive.delete(key);
    } else if (action === 7) {
      eng.flush();
      oracleDurable = new Map(oracleLive);
    } else if (action === 8) {
      eng.checkpointNow();
      oracleDurable = new Map(oracleLive);
    } else {
      crashes += 1;
      eng.crash();
      assert.ok(mapsEqual(eng.state, oracleDurable), `random crash ${i} recovered the durable prefix`);
      oracleLive = new Map(oracleDurable);
    }
  }
  assert.ok(crashes >= 1);
  assert.ok(eng.checkpointCount >= 1);
  eng.checkpointNow();
  oracleDurable = new Map(oracleLive);
  assert.ok(mapsEqual(eng.state, oracleLive) && mapsEqual(eng.state, oracleDurable));
  eng.crash();
  assert.ok(mapsEqual(eng.state, oracleLive), "a crash with nothing in flight is idempotent");
  return [crashes, eng.checkpointCount];
}

module.exports = { WALEngine, lcg };

if (require.main === module) {
  const assert = require("node:assert");
  const text = "1|set|a|1\n2|set|b|2\n3|del|a|\n4|set|c|3\n5|set|d";
  let r = WALEngine.recover(text, 4, null, 0);
  assert.ok(mapsEqual(r.state, new Map([["b", 2], ["c", 3]])) && r.replayed === 4, "parser watermark");
  r = WALEngine.recover(text, 2, null, 0);
  assert.ok(mapsEqual(r.state, new Map([["a", 1], ["b", 2]])) && r.replayed === 2, "earlier watermark");
  r = WALEngine.recover(text, 5, new Map([["z", 9]]), 3);
  assert.ok(mapsEqual(r.state, new Map([["z", 9], ["c", 3]])) && r.replayed === 1, "snapshot skips LSNs <= 3");

  const eng = new WALEngine();
  for (let i = 1; i <= 5; i++) eng.set(`k${i}`, i);
  assert.strictEqual(eng.lsn, 5);
  assert.strictEqual(eng.durableLsn, 0);
  eng.checkpointNow();
  assert.strictEqual(eng.durableLsn, 5);
  assert.strictEqual(eng.checkpointCount, 1);
  for (let i = 6; i <= 10; i++) eng.set(`k${i}`, i);
  eng.flush();
  eng.set("k11", 11);
  eng.delete("k3");
  eng.set("k12", 12);
  assert.strictEqual(eng.lsn, 13);
  eng.crash("14|set|k13");
  const expected = new Map();
  for (let i = 1; i <= 10; i++) expected.set(`k${i}`, i);
  assert.ok(mapsEqual(eng.state, expected), "recovered = committed prefix k1..k10");
  assert.strictEqual(eng.lastReplayCount, 5, "only the five post-checkpoint durable records replay");
  assert.strictEqual(eng.lsn, 10);
  assert.strictEqual(eng.durableLsn, 10);

  eng.set("k100", 100);
  assert.strictEqual(eng.lsn, 11);
  eng.flush();
  eng.crash();
  expected.set("k100", 100);
  assert.ok(mapsEqual(eng.state, expected), "second recovery");
  assert.strictEqual(eng.lastReplayCount, 6, "six replayed records");

  const eng2 = new WALEngine();
  eng2.set("x", 1);
  eng2.flush();
  eng2.set("x", 2);
  eng2.crash();
  assert.ok(mapsEqual(eng2.state, new Map([["x", 1]])), "unflushed write lost");

  const eng3 = new WALEngine();
  for (let i = 0; i < 4; i++) eng3.set(`a${i}`, i);
  eng3.flush();
  eng3.crash();
  assert.ok(mapsEqual(eng3.state, new Map([["a0", 0], ["a1", 1], ["a2", 2], ["a3", 3]])), "bare WAL replay");
  assert.strictEqual(eng3.lastReplayCount, 4);

  const [crashes, checkpoints] = runRandom();
  assert.strictEqual(crashes, 9, "a==9 rolls at 5,9,15,24,27,32,47,51,55");
  assert.strictEqual(checkpoints, 11, "ten a==8 checkpoints plus the final settle checkpoint");
  console.log("PASS");
}
