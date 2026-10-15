/** Task 172 - Implement stack that supports getMin in O(1). (Tier: Beginner, CTCI Ch 3) */
"use strict";

class MinStack {
  constructor() {
    this.main = [];
    this.mins = [];
  }

  push(value) {
    this.main.push(value);
    this.mins.push(this.mins.length === 0 ? value : Math.min(value, this.mins[this.mins.length - 1]));
  }

  pop() {
    if (this.main.length === 0) throw new Error("stack empty");
    this.mins.pop();
    return this.main.pop();
  }

  peek() {
    if (this.main.length === 0) throw new Error("stack empty");
    return this.main[this.main.length - 1];
  }

  min() {
    if (this.mins.length === 0) throw new Error("stack empty");
    return this.mins[this.mins.length - 1];
  }

  get isEmpty() { return this.main.length === 0; }
  get size() { return this.main.length; }
}

module.exports = { MinStack };

if (require.main === module) {
  const assert = require("node:assert");
  const s = new MinStack();
  for (const v of [5, 3, 7, 3, 9]) s.push(v);
  assert.strictEqual(s.min(), 3);
  assert.strictEqual(s.peek(), 9);
  assert.strictEqual(s.size, 5);
  assert.strictEqual(s.pop(), 9);
  s.push(1);
  assert.strictEqual(s.min(), 1);
  assert.strictEqual(s.pop(), 1);
  assert.strictEqual(s.min(), 3);
  assert.strictEqual(s.pop(), 3);
  assert.strictEqual(s.min(), 3);
  assert.strictEqual(s.pop(), 7);
  assert.strictEqual(s.min(), 3);
  assert.strictEqual(s.pop(), 3);
  assert.strictEqual(s.min(), 5);
  assert.strictEqual(s.pop(), 5);
  assert.strictEqual(s.isEmpty, true);
  console.log("PASS");
}
