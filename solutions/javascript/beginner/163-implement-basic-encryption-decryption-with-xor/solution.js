/** Task 163 - Implement basic encryption/decryption with XOR. (Tier: Beginner, CTCI Ch 5) */
"use strict";

function xorBytes(data, keyBuffer) {
  if (keyBuffer.length === 0) throw new Error("key must not be empty");
  const out = Buffer.alloc(data.length);
  for (let i = 0; i < data.length; i++) out[i] = data[i] ^ keyBuffer[i % keyBuffer.length];
  return out;
}

function xorText(text, key) {
  return xorBytes(Buffer.from(text, "utf8"), Buffer.from(key, "utf8")).toString("hex");
}

function decryptText(hexCipher, key) {
  const cipher = Buffer.from(hexCipher, "hex");
  return xorBytes(cipher, Buffer.from(key, "utf8")).toString("utf8");
}

module.exports = { xorBytes, xorText, decryptText };

if (require.main === module) {
  const assert = require("node:assert");
  const message = "attack at dawn";
  const key = "s3cret";
  const cipher = xorText(message, key);
  assert.notStrictEqual(cipher, message);
  assert.match(cipher, /^[0-9a-f]+$/);
  assert.strictEqual(decryptText(cipher, key), message);
  assert.strictEqual(xorText("", key), "");
  assert.strictEqual(decryptText("", key), "");
  assert.notStrictEqual(decryptText(cipher, "other"), message);
  console.log("PASS");
}
