/** Task 168 - Implement basic checksum (sum bytes). (Tier: Beginner, CTCI Ch 5) */
"use strict";

function toBuffer(data) {
  return Buffer.isBuffer(data) ? data : Buffer.from(String(data), "utf8");
}

function checksum8(data) {
  const buf = toBuffer(data);
  let sum = 0;
  for (const b of buf) sum += b;
  return sum % 256;
}

function checksum16(data) {
  const buf = toBuffer(data);
  let sum = 0;
  for (const b of buf) sum += b;
  return sum % 65536;
}

module.exports = { checksum8, checksum16 };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(checksum8(""), 0);
  assert.strictEqual(checksum8("abc"), 38);
  assert.strictEqual(checksum16("abc"), 294);
  assert.strictEqual(checksum8(Buffer.from([0xff, 0x01])), 0);
  assert.strictEqual(checksum16(Buffer.from([0xff, 0xff])), 510);

  const original = Buffer.from(Array.from({ length: 50 }, (_, i) => i));
  const corrupt = Buffer.from(original);
  corrupt[10] ^= 0x01;
  assert.notStrictEqual(checksum8(original), checksum8(corrupt));
  console.log("PASS");
}
