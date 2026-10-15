/** Task 175 - Multiply large integers as strings. (Tier: Beginner, CTCI Ch 6) */
"use strict";

function multiplyStrings(a, b) {
  if (a === "0" || b === "0") return "0";
  const negative = (a[0] === "-") !== (b[0] === "-");
  const da = a.replace(/^-/, "");
  const db = b.replace(/^-/, "");
  const result = new Array(da.length + db.length).fill(0);
  for (let i = da.length - 1; i >= 0; i--) {
    for (let j = db.length - 1; j >= 0; j--) {
      result[i + j + 1] += (da.charCodeAt(i) - 48) * (db.charCodeAt(j) - 48);
    }
  }
  for (let k = result.length - 1; k > 0; k--) {
    result[k - 1] += Math.floor(result[k] / 10);
    result[k] %= 10;
  }
  const digits = result.join("").replace(/^0+/, "") || "0";
  if (digits === "0") return "0";
  return negative ? "-" + digits : digits;
}

module.exports = { multiplyStrings };

if (require.main === module) {
  const assert = require("node:assert");
  assert.strictEqual(multiplyStrings("123", "456"), "56088");
  assert.strictEqual(multiplyStrings("0", "999"), "0");
  assert.strictEqual(multiplyStrings("-12", "12"), "-144");
  assert.strictEqual(multiplyStrings("-12", "-12"), "144");
  assert.strictEqual(multiplyStrings("99999", "99999"), "9999800001");

  // verify against BigInt on random large pairs
  let seed = 11;
  const rng = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x80000000;
  };
  for (let t = 0; t < 30; t++) {
    let x = 0n, y = 0n;
    const digitsX = 1 + Math.floor(rng() * 40);
    const digitsY = 1 + Math.floor(rng() * 40);
    for (let i = 0; i < digitsX; i++) x = x * 10n + BigInt(Math.floor(rng() * 10));
    for (let i = 0; i < digitsY; i++) y = y * 10n + BigInt(Math.floor(rng() * 10));
    assert.strictEqual(multiplyStrings(x.toString(), y.toString()), (x * y).toString(), `random ${t}`);
  }

  const nines = "9".repeat(100);
  assert.strictEqual(multiplyStrings(nines, nines), (BigInt(nines) * BigInt(nines)).toString());
  console.log("PASS");
}
