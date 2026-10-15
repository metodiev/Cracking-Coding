/** Task 198 - Serialize object to JSON string (manual). (Tier: Beginner, CTCI Ch 7) */
"use strict";

const ESCAPES = {
  "\"": "\\\"",
  "\\": "\\\\",
  "\n": "\\n",
  "\t": "\\t",
  "\r": "\\r",
  "\b": "\\b",
  "\f": "\\f",
};

function serializeString(value) {
  let out = '"';
  for (const ch of value) {
    if (Object.prototype.hasOwnProperty.call(ESCAPES, ch)) out += ESCAPES[ch];
    else if (ch.charCodeAt(0) < 0x20) out += "\\u" + ch.charCodeAt(0).toString(16).padStart(4, "0");
    else out += ch;
  }
  return out + '"';
}

function isPlainObject(value) {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function toJson(value) {
  if (value === null) return "null";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "number") {
    if (!Number.isInteger(value)) throw new TypeError("only integers are supported");
    return String(value);
  }
  if (typeof value === "string") return serializeString(value);
  if (Array.isArray(value)) return "[" + value.map(toJson).join(",") + "]";
  if (isPlainObject(value)) {
    const keys = Object.keys(value);
    for (const key of keys) {
      if (typeof key !== "string") throw new TypeError("object keys must be strings");
    }
    keys.sort();
    return "{" + keys.map((key) => serializeString(key) + ":" + toJson(value[key])).join(",") + "}";
  }
  throw new TypeError("unsupported type: " + typeof value);
}

module.exports = { toJson };

if (require.main === module) {
  const assert = require("node:assert");

  assert.strictEqual(toJson(null), "null");
  assert.strictEqual(toJson(true), "true");
  assert.strictEqual(toJson(false), "false");
  assert.strictEqual(toJson(42), "42");
  assert.strictEqual(toJson(-7), "-7");
  assert.strictEqual(toJson(""), '""');
  assert.strictEqual(toJson([]), "[]");
  assert.strictEqual(toJson({}), "{}");
  assert.strictEqual(toJson("a\"b\\c\nd\te\rf"), '"a\\"b\\\\c\\nd\\te\\rf"');
  assert.strictEqual(toJson("x\u0001y"), '"x\\u0001y"');
  assert.strictEqual(toJson({ b: 1, a: [true, null, "x"] }), '{"a":[true,null,"x"],"b":1}');
  assert.strictEqual(toJson({ a: { b: { c: [1, [2, [3, null]]] } } }), '{"a":{"b":{"c":[1,[2,[3,null]]]}}}');

  const doc = {
    users: [{ id: 2, name: "Bob" }, { id: 1, name: "Ann" }],
    count: 2,
    ok: true,
    tags: [],
  };
  const expected = '{"count":2,"ok":true,"tags":[],"users":[{"id":2,"name":"Bob"},{"id":1,"name":"Ann"}]}';
  assert.strictEqual(toJson(doc), expected);
  assert.deepStrictEqual(JSON.parse(toJson(doc)), doc);       // stdlib oracle round-trip

  assert.throws(() => toJson(() => 1), /unsupported type/);
  assert.throws(() => toJson(3.5), /only integers/);
  console.log("PASS");
}
