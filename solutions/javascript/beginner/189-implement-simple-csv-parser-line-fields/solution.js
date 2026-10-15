/** Task 189 - Implement simple CSV parser (line -> fields). (Tier: Beginner, CTCI Ch 7) */
"use strict";

/**
 * RFC-4180 basics for a single line:
 * - ',' separates fields outside quotes;
 * - a field starting with '"' is quoted, and "" inside a quoted field is one literal quote;
 * - only ',' or the end of the line may follow a closing quote — anything else is an error;
 * - an empty line yields one empty field and a trailing comma yields a final empty field.
 *
 * Throws Error for an unterminated quoted field, junk after a closing quote, or a quote
 * inside an unquoted field.
 */
function parseCsvLine(line) {
  const fields = [];
  let i = 0;
  for (;;) {
    let chunk = "";
    if (i < line.length && line[i] === '"') {
      i++;
      for (;;) {
        if (i >= line.length) throw new Error("unterminated quoted field");
        if (line[i] === '"') {
          if (i + 1 < line.length && line[i + 1] === '"') {
            chunk += '"';
            i += 2;
            continue;
          }
          i++; // closing quote
          break;
        }
        chunk += line[i];
        i++;
      }
      if (i < line.length && line[i] !== ",") throw new Error("unexpected character after closing quote");
    } else {
      while (i < line.length && line[i] !== ",") {
        if (line[i] === '"') throw new Error("quote inside unquoted field");
        chunk += line[i];
        i++;
      }
    }
    fields.push(chunk);
    if (i >= line.length) return fields;
    i++; // skip the separating comma
  }
}

module.exports = { parseCsvLine };

if (require.main === module) {
  const assert = require("node:assert");

  assert.deepStrictEqual(parseCsvLine("a,b,c"), ["a", "b", "c"]);
  assert.deepStrictEqual(parseCsvLine('a,"b,c",d'), ["a", "b,c", "d"]);
  assert.deepStrictEqual(parseCsvLine('"a""b",c'), ['a"b', "c"]);
  assert.deepStrictEqual(parseCsvLine(""), [""]);
  assert.deepStrictEqual(parseCsvLine("a,,c"), ["a", "", "c"]);
  assert.deepStrictEqual(parseCsvLine("a,b,"), ["a", "b", ""]);
  assert.deepStrictEqual(parseCsvLine(",,"), ["", "", ""]);
  assert.deepStrictEqual(parseCsvLine('"",x'), ["", "x"]);
  assert.deepStrictEqual(parseCsvLine('"line""break"'), ['line"break']);
  assert.deepStrictEqual(parseCsvLine('"a,b","c,d"'), ["a,b", "c,d"]);
  assert.deepStrictEqual(parseCsvLine(" , "), [" ", " "]);
  assert.deepStrictEqual(parseCsvLine('"  spaced  ",tail'), ["  spaced  ", "tail"]);
  assert.deepStrictEqual(parseCsvLine("single"), ["single"]);

  assert.throws(() => parseCsvLine('a,"b,c'), Error, "unterminated quote");
  assert.throws(() => parseCsvLine('"'), Error, "lone quote");
  assert.throws(() => parseCsvLine('a,"b"c,d'), Error, "junk after closing quote");
  assert.throws(() => parseCsvLine('a"b'), Error, "quote inside unquoted field");
  assert.throws(() => parseCsvLine('"unterminated'), Error, "unterminated at end of line");
  console.log("PASS");
}
