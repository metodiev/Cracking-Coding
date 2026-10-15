/**
 * Task 344 - URL shortener: deterministic hash, base62 codes, linear probing. (Tier: Intermediate, CTCI Ch 7)
 *
 * code(url) = base62 of (sum of the URL's code points) % 62^6, zero-padded to six characters
 * (MSB first).  The deliberately simple hash lets the tests engineer collisions ("…/ab" vs
 * "…/ba"); linear probing (index+1 mod 62^6) finds the first free slot, so distinct URLs always
 * receive distinct codes.  shorten() is idempotent and expand() returns null for unknown codes.
 * 62^6 is below 2^53, so plain Numbers stay exact.
 */
"use strict";

const ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const BASE = 62;
const CODE_WIDTH = 6;
const SPACE = 62 ** 6;                 // 56_800_235_584

function hashUrl(url) {
  let total = 0;
  for (const ch of url) total += ch.codePointAt(0);
  return total % SPACE;
}

function encodeBase62(valueIn) {
  let value = valueIn % SPACE;
  let digits = "";
  for (let i = 0; i < CODE_WIDTH; i++) {
    digits = ALPHABET[value % BASE] + digits;
    value = Math.floor(value / BASE);
  }
  return digits;
}

/** Independent inverse; used by the tests to verify the probing order. */
function decodeBase62(code) {
  let value = 0;
  for (const ch of code) {
    const index = ALPHABET.indexOf(ch);
    if (index < 0) throw new Error(`invalid base62 character '${ch}'`);
    value = value * BASE + index;
  }
  return value;
}

class UrlShortener {
  constructor() {
    this.urlToCode = new Map();
    this.codeToUrl = new Map();
    this.probes = 0;
  }

  shorten(url) {
    if (this.urlToCode.has(url)) return this.urlToCode.get(url);   // idempotent
    let index = hashUrl(url);
    for (;;) {
      const code = encodeBase62(index);
      if (!this.codeToUrl.has(code)) {
        this.urlToCode.set(url, code);
        this.codeToUrl.set(code, url);
        return code;
      }
      index = (index + 1) % SPACE;                                 // linear probing
      this.probes++;
    }
  }

  expand(code) {
    return this.codeToUrl.has(code) ? this.codeToUrl.get(code) : null;   // null when unknown
  }

  size() {
    return this.urlToCode.size;
  }
}

module.exports = { ALPHABET, CODE_WIDTH, SPACE, hashUrl, encodeBase62, decodeBase62, UrlShortener };

if (require.main === module) {
  const assert = require("node:assert");

  const shortener = new UrlShortener();
  const url = "http://example.com/articles/42";
  const code = shortener.shorten(url);
  assert.strictEqual(code.length, CODE_WIDTH);
  for (const ch of code) assert.ok(ALPHABET.includes(ch), "code alphabet");
  assert.strictEqual(code, encodeBase62(hashUrl(url)), "initial slot free, no probing");
  assert.strictEqual(shortener.expand(code), url, "expand round trip");
  assert.strictEqual(shortener.shorten(url), code, "idempotent");
  assert.strictEqual(shortener.size(), 1);
  assert.strictEqual(shortener.probes, 0);
  assert.strictEqual(shortener.expand("zzzzzz"), null, "unknown code");
  assert.strictEqual(shortener.expand(""), null, "empty code");
  assert.strictEqual(shortener.expand("!!!!!!"), null, "invalid code unknown");

  assert.strictEqual(encodeBase62(0), "000000");
  assert.strictEqual(encodeBase62(SPACE - 1), "ZZZZZZ");
  let seed = 9137;
  for (let i = 0; i < 200; i++) {
    seed = (seed * 75 + 74) % 65537;
    const value = (seed * 12345) % SPACE;
    assert.strictEqual(decodeBase62(encodeBase62(value)), value, `base62 round trip ${value}`);
  }

  // engineered collision: anagram paths hash to the same slot
  const u1 = "http://x.com/pages/ab";
  const u2 = "http://x.com/pages/ba";
  assert.strictEqual(hashUrl(u1), hashUrl(u2), "engineered collision");
  const collision = new UrlShortener();
  const c1 = collision.shorten(u1);
  const c2 = collision.shorten(u2);
  assert.notStrictEqual(c1, c2, "distinct codes despite collision");
  assert.strictEqual(decodeBase62(c1), hashUrl(u1), "c1 sits at the initial slot");
  assert.strictEqual(decodeBase62(c2), (decodeBase62(c1) + 1) % SPACE, "exactly one probe step");
  assert.strictEqual(collision.probes, 1);
  assert.strictEqual(collision.expand(c1), u1);
  assert.strictEqual(collision.expand(c2), u2);

  // many URLs, every one in a known-colliding pair
  const codes = [];
  const batch = new UrlShortener();
  for (let k = 0; k < 250; k++) {
    const ab = `http://bench/${k}/ab`;
    const ba = `http://bench/${k}/ba`;
    assert.strictEqual(hashUrl(ab), hashUrl(ba), `pair collision ${k}`);
    for (const u of [ab, ba]) {
      const c = batch.shorten(u);
      assert.strictEqual(batch.shorten(u), c, `idempotent ${u}`);
      assert.strictEqual(batch.expand(c), u, `round trip ${u}`);
      assert.strictEqual(c.length, CODE_WIDTH);
      codes.push(c);
    }
  }
  assert.strictEqual(codes.length, 500);
  assert.strictEqual(new Set(codes).size, 500, "all codes distinct despite collisions");
  assert.ok(batch.probes > 0, "collisions were exercised");
  assert.strictEqual(batch.size(), 500);

  const again = new UrlShortener();
  const reproduced = [];
  for (let k = 0; k < 250; k++) {
    reproduced.push(again.shorten(`http://bench/${k}/ab`));
    reproduced.push(again.shorten(`http://bench/${k}/ba`));
  }
  assert.deepStrictEqual(reproduced, codes, "deterministic replay");
  console.log("PASS");
}
