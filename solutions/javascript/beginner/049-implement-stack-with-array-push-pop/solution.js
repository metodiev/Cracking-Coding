/** Task 049 - Implement stack with array (push/pop). (Tier: Beginner, CTCI Ch 3) */
class StackArray {
  constructor(capacity) {
    this.data = new Array(capacity);
    this.top = -1;
  }

  push(value) {
    if (this.top + 1 === this.data.length) throw new Error("stack full");
    this.data[++this.top] = value;
  }

  pop() {
    if (this.isEmpty()) throw new Error("stack empty");
    return this.data[this.top--];
  }

  peek() {
    if (this.isEmpty()) throw new Error("stack empty");
    return this.data[this.top];
  }

  isEmpty() { return this.top < 0; }
  size() { return this.top + 1; }
}

module.exports = { StackArray };

if (require.main === module) {
  const assert = require("node:assert");
  const s = new StackArray(3);
  assert.strictEqual(s.isEmpty(), true);
  s.push(1); s.push(2); s.push(3);
  assert.strictEqual(s.peek(), 3);
  assert.strictEqual(s.size(), 3);
  assert.deepStrictEqual([s.pop(), s.pop(), s.pop()], [3, 2, 1]);
  assert.strictEqual(s.isEmpty(), true);
  console.log("PASS");
}
