/** Task 095 - Validate IPv4 address (basic). (Tier: Beginner, CTCI Ch 1)

 * Leading zeros are rejected ("1.2.3.04" invalid), except the value "0" itself.
 */
"use strict";

function isValidIpv4(s) {
  const parts = s.split(".");
  if (parts.length !== 4) return false;
  for (const part of parts) {
    if (part.length === 0 || part.length > 3) return false;
    for (let i = 0; i < part.length; i++) {
      const c = part[i];
      if (c < "0" || c > "9") return false;
    }
    if (part.length > 1 && part[0] === "0") return false;
    if (Number(part) > 255) return false;
  }
  return true;
}

module.exports = { isValidIpv4 };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(isValidIpv4("192.168.1.1"), true);
  assert.strictEqual(isValidIpv4("0.0.0.0"), true);
  assert.strictEqual(isValidIpv4("255.255.255.255"), true);
  assert.strictEqual(isValidIpv4("10.0.0.255"), true);
  assert.strictEqual(isValidIpv4("1.2.3.0"), true);
  assert.strictEqual(isValidIpv4("256.1.1.1"), false);
  assert.strictEqual(isValidIpv4("1.2.3.256"), false);
  assert.strictEqual(isValidIpv4("1.2.3"), false);
  assert.strictEqual(isValidIpv4("1.2.3.4.5"), false);
  assert.strictEqual(isValidIpv4(""), false);
  assert.strictEqual(isValidIpv4(".1.2.3"), false);
  assert.strictEqual(isValidIpv4("1.2.3."), false);
  assert.strictEqual(isValidIpv4("1..2.3"), false);
  assert.strictEqual(isValidIpv4("1.2.3.a"), false);
  assert.strictEqual(isValidIpv4("1.2.3.4x"), false);
  assert.strictEqual(isValidIpv4("01.2.3.4"), false);
  assert.strictEqual(isValidIpv4("1.2.3.00"), false);
  assert.strictEqual(isValidIpv4("1.2.3.1234"), false);
  assert.strictEqual(isValidIpv4("-1.2.3.4"), false);
  assert.strictEqual(isValidIpv4("1.2.3.4 "), false);
  console.log("PASS");
}
