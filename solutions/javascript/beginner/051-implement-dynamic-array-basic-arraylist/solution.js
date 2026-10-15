/** Task 051 - Implement dynamic array (basic ArrayList). (Tier: Beginner, CTCI Ch 1) */
class DynamicArray {
  constructor() {
    this.data = new Array(2).fill(0);
    this.count = 0;
  }

  pushBack(value) {
    if (this.count === this.data.length) this.grow();
    this.data[this.count++] = value;
  }

  grow() {
    const bigger = new Array(this.data.length * 2).fill(0);
    for (let i = 0; i < this.count; i++) bigger[i] = this.data[i];
    this.data = bigger;
  }

  get(index) { this.check(index); return this.data[index]; }
  set(index, value) { this.check(index); this.data[index] = value; }
  popBack() {
    if (this.count === 0) throw new Error("empty");
    return this.data[--this.count];
  }
  check(index) {
    if (index < 0 || index >= this.count) throw new RangeError("index out of range");
  }
  size() { return this.count; }
  capacity() { return this.data.length; }
}

module.exports = { DynamicArray };

if (require.main === module) {
  const assert = require("node:assert");
  const a = new DynamicArray();
  for (let i = 0; i < 5; i++) a.pushBack(i * 10);
  assert.strictEqual(a.size(), 5);
  assert.strictEqual(a.capacity(), 8);
  assert.strictEqual(a.get(3), 30);
  a.set(0, 99);
  assert.strictEqual(a.get(0), 99);
  assert.strictEqual(a.popBack(), 40);
  console.log("PASS");
}
