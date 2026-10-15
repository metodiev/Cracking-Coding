/** Task 080 - Implement string builder/concatenation efficient. (Tier: Beginner, CTCI Ch 1) */
"use strict";

class StringBuilder {
  constructor() {
    this.parts = [];
  }

  append(text) {
    this.parts.push(String(text));
    return this;
  }

  build() {
    return this.parts.join("");
  }

  get length() {
    return this.parts.reduce((n, p) => n + p.length, 0);
  }
}

module.exports = { StringBuilder };

if (require.main === module) {
  const assert = require("node:assert");
  const sb = new StringBuilder();
  sb.append("Hello").append(", ").append("world").append("!").append(42);
  assert.strictEqual(sb.build(), "Hello, world!42");
  assert.strictEqual(sb.length, 15);
  assert.strictEqual(new StringBuilder().build(), "");
  console.log("PASS");
}
