/** Task 199 - Deserialize JSON with arrays & objects (basic). (Tier: Beginner, CTCI Ch 7) */
"use strict";

class JsonParseError extends Error {}

class JsonParser {
  constructor(text) {
    this.text = text;
    this.pos = 0;
  }

  parse() {
    const value = this._parseValue();
    this._skipWs();
    if (this.pos !== this.text.length) this._fail("trailing characters after JSON value");
    return value;
  }

  _fail(message) {
    throw new JsonParseError(`${message} at index ${this.pos}`);
  }

  _skipWs() {
    while (this.pos < this.text.length && " \t\n\r".includes(this.text[this.pos])) this.pos++;
  }

  _peek() {
    if (this.pos >= this.text.length) this._fail("unexpected end of input");
    return this.text[this.pos];
  }

  _expect(ch) {
    if (this._peek() !== ch) this._fail(`expected '${ch}'`);
    this.pos++;
  }

  _parseValue() {
    this._skipWs();
    const ch = this._peek();
    if (ch === "{") return this._parseObject();
    if (ch === "[") return this._parseArray();
    if (ch === '"') return this._parseString();
    if (ch === "t" || ch === "f" || ch === "n") return this._parseLiteral();
    if (ch === "-" || /[0-9]/.test(ch)) return this._parseNumber();
    this._fail(`unexpected character '${ch}'`);
  }

  _parseLiteral() {
    for (const [word, value] of [["true", true], ["false", false], ["null", null]]) {
      if (this.text.startsWith(word, this.pos)) {
        this.pos += word.length;
        return value;
      }
    }
    this._fail("invalid literal");
  }

  _parseNumber() {
    const start = this.pos;
    if (this._peek() === "-") this.pos++;
    if (this.pos >= this.text.length || !/[0-9]/.test(this.text[this.pos])) this._fail("invalid number");
    if (this.text[this.pos] === "0") {
      this.pos++;
      if (this.pos < this.text.length && /[0-9]/.test(this.text[this.pos])) this._fail("leading zeros are not allowed");
    } else {
      while (this.pos < this.text.length && /[0-9]/.test(this.text[this.pos])) this.pos++;
    }
    if (this.pos < this.text.length && ".eE".includes(this.text[this.pos])) {
      this._fail("only integer numbers are supported");
    }
    return Number.parseInt(this.text.slice(start, this.pos), 10);
  }

  _parseString() {
    this._expect('"');
    let out = "";
    while (true) {
      if (this.pos >= this.text.length) this._fail("unterminated string");
      const ch = this.text[this.pos++];
      if (ch === '"') return out;
      if (ch === "\\") out += this._parseEscape();
      else if (ch.charCodeAt(0) < 0x20) this._fail("unescaped control character in string");
      else out += ch;
    }
  }

  _parseEscape() {
    if (this.pos >= this.text.length) this._fail("unterminated escape");
    const esc = this.text[this.pos++];
    const simple = { '"': '"', "\\": "\\", "/": "/", b: "\b", f: "\f", n: "\n", r: "\r", t: "\t" };
    if (Object.prototype.hasOwnProperty.call(simple, esc)) return simple[esc];
    if (esc === "u") {
      if (this.pos + 4 > this.text.length) this._fail("incomplete \\u escape");
      const digits = this.text.slice(this.pos, this.pos + 4);
      if (!/^[0-9a-fA-F]{4}$/.test(digits)) this._fail("invalid \\u escape");
      this.pos += 4;
      return String.fromCharCode(Number.parseInt(digits, 16));
    }
    this._fail(`invalid escape '\\${esc}'`);
  }

  _parseObject() {
    this._expect("{");
    const result = {};
    this._skipWs();
    if (this._peek() === "}") {
      this.pos++;
      return result;
    }
    while (true) {
      this._skipWs();
      const key = this._parseString();
      this._skipWs();
      this._expect(":");
      result[key] = this._parseValue();
      this._skipWs();
      const ch = this._peek();
      if (ch === ",") {
        this.pos++;
        this._skipWs();
        if (this._peek() === "}") this._fail("trailing comma in object");
        continue;
      }
      if (ch === "}") {
        this.pos++;
        return result;
      }
      this._fail("expected ',' or '}'");
    }
  }

  _parseArray() {
    this._expect("[");
    const result = [];
    this._skipWs();
    if (this._peek() === "]") {
      this.pos++;
      return result;
    }
    while (true) {
      result.push(this._parseValue());
      this._skipWs();
      const ch = this._peek();
      if (ch === ",") {
        this.pos++;
        this._skipWs();
        if (this._peek() === "]") this._fail("trailing comma in array");
        continue;
      }
      if (ch === "]") {
        this.pos++;
        return result;
      }
      this._fail("expected ',' or ']'");
    }
  }
}

function parseJson(text) {
  return new JsonParser(text).parse();
}

module.exports = { parseJson, JsonParser, JsonParseError };

if (require.main === module) {
  const assert = require("node:assert");

  const documents = [
    "{}",
    "[]",
    "null",
    "true",
    '{"a":1,"b":[true,false,null],"c":"hi\\n","d":{"e":[]}}',
    '{"nested":[1,[2,[3]]],"u":"\\u0041","esc":"a\\"b\\\\c"}',
    "  {  \"x\" : [ 1 , 2 ]  } ",
    "-42",
    "0",
  ];
  for (const text of documents) {
    assert.deepStrictEqual(parseJson(text), JSON.parse(text));   // stdlib oracle
  }

  const bad = ['{"a":1,}', "[1,]", '"abc', '{"a" 1}', "tru", "@", "", "1 2", "01", "1.5",
               '{"a":}', "[", "]", "[1 2]", '"\\x"', '"\\u12"'];
  for (const text of bad) {
    assert.throws(() => parseJson(text), JsonParseError, `expected error for ${text}`);
  }
  console.log("PASS");
}
