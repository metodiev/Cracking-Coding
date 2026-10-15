/** Task 196 - Implement file line reader that yields lines lazily. (Tier: Beginner, CTCI Ch 7) */
"use strict";

const fs = require("node:fs");

const CHUNK_SIZE = 64;

// A generator: the fd is opened on the first next(), and lines are emitted chunk by chunk.
function* readLines(filePath) {
  const fd = fs.openSync(filePath, "r");
  try {
    const buffer = Buffer.alloc(CHUNK_SIZE);
    let carry = "";
    while (true) {
      const bytesRead = fs.readSync(fd, buffer, 0, buffer.length, null);
      if (bytesRead === 0) break;
      carry += buffer.toString("utf8", 0, bytesRead);
      let newline;
      while ((newline = carry.indexOf("\n")) >= 0) {
        yield carry.slice(0, newline);
        carry = carry.slice(newline + 1);
      }
    }
    if (carry.length > 0) yield carry;                 // final line without a newline
  } finally {
    fs.closeSync(fd);
  }
}

module.exports = { readLines };

if (require.main === module) {
  const assert = require("node:assert");
  const os = require("node:os");
  const path = require("node:path");

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "line-reader-"));
  const file = path.join(dir, "sample.txt");
  try {
    fs.writeFileSync(file, "alpha\nbeta\n\ngamma\ndelta");   // 5 lines, last one unterminated

    // Laziness: nothing is read (the fd is opened) until the generator is advanced.
    const lazy = readLines(file);
    fs.unlinkSync(file);
    assert.throws(() => lazy.next(), /ENOENT/);
    lazy.return();

    fs.writeFileSync(file, "alpha\nbeta\n\ngamma\ndelta");

    const iterator = readLines(file);
    assert.strictEqual(iterator.next().value, "alpha");
    assert.deepStrictEqual([...iterator], ["beta", "", "gamma", "delta"]);
    assert.deepStrictEqual([...readLines(file)], ["alpha", "beta", "", "gamma", "delta"]);

    fs.writeFileSync(file, "x\ny\n");
    assert.deepStrictEqual([...readLines(file)], ["x", "y"]);   // trailing newline, no empty tail
  } finally {
    if (fs.existsSync(file)) fs.unlinkSync(file);
    fs.rmdirSync(dir);
  }
  assert.ok(!fs.existsSync(file) && !fs.existsSync(dir), "no artifacts left behind");
  console.log("PASS");
}
