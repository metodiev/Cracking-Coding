/** Task 162 - Implement rate limiter (token bucket simple). (Tier: Beginner, CTCI Ch 7) */
"use strict";

class TokenBucket {
  constructor(capacity, refillPerSecond) {
    if (capacity <= 0 || refillPerSecond <= 0) {
      throw new Error("capacity and refill rate must be positive");
    }
    this.capacity = capacity;
    this.rate = refillPerSecond;
    this.tokens = capacity;
    this.lastMs = null;
  }

  refill(nowMs) {
    if (this.lastMs === null) {
      this.lastMs = nowMs;
      return;
    }
    const elapsed = Math.max(0, nowMs - this.lastMs);
    this.tokens = Math.min(this.capacity, this.tokens + (elapsed / 1000) * this.rate);
    this.lastMs = nowMs;
  }

  allow(nowMs, cost = 1) {
    this.refill(nowMs);
    if (this.tokens >= cost) {
      this.tokens -= cost;
      return true;
    }
    return false;
  }
}

module.exports = { TokenBucket };

if (require.main === module) {
  const assert = require("node:assert");
  const bucket = new TokenBucket(3, 1.0);
  assert.ok(bucket.allow(0) && bucket.allow(0) && bucket.allow(0));
  assert.strictEqual(bucket.allow(0), false);
  assert.strictEqual(bucket.allow(1000), true);
  assert.strictEqual(bucket.allow(1000, 2), false);
  assert.strictEqual(bucket.allow(3000, 2), true);
  assert.strictEqual(bucket.allow(100000), true);
  assert.strictEqual(bucket.allow(100000, 4), false);
  console.log("PASS");
}
