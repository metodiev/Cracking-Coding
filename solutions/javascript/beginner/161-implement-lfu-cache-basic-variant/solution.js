/** Task 161 - Implement LFU cache (basic variant). (Tier: Beginner, CTCI Ch 7) */
"use strict";

class LFUCache {
  constructor(capacity) {
    if (capacity <= 0) throw new Error("capacity must be positive");
    this.capacity = capacity;
    this.data = new Map();          // key -> { value, freq, seq }
    this.tick = 0;
  }

  get(key) {
    const entry = this.data.get(key);
    if (!entry) return -1;
    entry.freq++;
    entry.seq = this.tick++;
    return entry.value;
  }

  put(key, value) {
    const entry = this.data.get(key);
    if (entry) {
      entry.value = value;
      entry.freq++;
      entry.seq = this.tick++;
      return;
    }
    if (this.data.size >= this.capacity) {
      let victim = null, bestFreq = Infinity, bestSeq = Infinity;
      for (const [k, e] of this.data) {
        if (e.freq < bestFreq || (e.freq === bestFreq && e.seq < bestSeq)) {
          bestFreq = e.freq;
          bestSeq = e.seq;
          victim = k;
        }
      }
      this.data.delete(victim);
    }
    this.data.set(key, { value, freq: 1, seq: this.tick++ });
  }
}

module.exports = { LFUCache };

if (require.main === module) {
  const assert = require("node:assert");
  const cache = new LFUCache(2);
  cache.put(1, 1);
  cache.put(2, 2);
  assert.strictEqual(cache.get(1), 1);
  cache.put(3, 3);
  assert.strictEqual(cache.get(2), -1);
  assert.strictEqual(cache.get(3), 3);
  cache.put(4, 4);
  assert.strictEqual(cache.get(1), -1);
  assert.strictEqual(cache.get(3), 3);
  assert.strictEqual(cache.get(4), 4);
  console.log("PASS");
}
