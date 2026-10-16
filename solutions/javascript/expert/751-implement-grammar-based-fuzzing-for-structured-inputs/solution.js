/** Task 751 - Implement grammar-based fuzzing for structured inputs. (Tier: Expert, CTCI Ch 11 — Testing)
 *
 * A context-free grammar maps a nonterminal to a list of alternatives; an alternative is a
 * sequence of symbols, each either a terminal (one character) or another nonterminal. The
 * constructor computes two fixed points that make generation safe:
 *   - minLength[nt]: length of the shortest string derivable from nt (Infinity when the
 *     nonterminal is unproductive). Any nonterminal that cannot terminate is rejected, so a
 *     generator can never be trapped in infinite recursion.
 *   - nullable iff minLength === 0.
 * Generation then picks, at every step, uniformly among the alternatives that *fit* in the
 * remaining character budget, and reserves minLength of the tail before expanding a piece.
 * That budget discipline is the whole trick: it makes every generated string a real
 * derivation (structurally valid by construction) while still exploring the language.
 * `enumerate(maxLength)` solves the same grammar by chaotic iteration over (nonterminal,
 * length) buckets, which gives an independent, exhaustive view of the language.
 *
 * What is simplified: the grammar format is the classic BNF subset (no attributes, no
 * repetition operators, no semantic predicates) and a terminal is a single character, so
 * "structured inputs" means character-level CFG inputs; token-level lexing and coverage
 * instrumentation of the target program are out of scope, and the fuzzing loop drives a real
 * recognizer (a Dyck-language parser) instead of an instrumented binary.
 * What is verified: (a) generation and enumeration agree exactly with an independent
 * brute-force recognizer over the whole language up to length 7 — generation only produces
 * strings in the language and enumeration finds all of them; (b) every production is covered
 * by a sample corpus (production coverage); (c) 100% of generated inputs are accepted while
 * character-level mutations of the same corpus are overwhelmingly rejected, the measurable
 * advantage of grammar-based fuzzing; (d) replay determinism for a fixed seed; (e) rejection
 * of unproductive / ill-formed grammars.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

/** A terminal symbol of the grammar: exactly one character. */
function T(text) {
  if (typeof text !== "string" || text.length !== 1) {
    throw new TypeError(`a terminal is exactly one character (got ${JSON.stringify(text)})`);
  }
  return { terminal: true, text };
}

const isTerminal = (symbol) => typeof symbol === "object" && symbol !== null && symbol.terminal === true;

/** Deterministic xorshift32; `next(bound)` returns integers in [0, bound). */
function makeRng(seed) {
  let state = seed >>> 0 || 0x2545f491;
  return function next(bound) {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return bound === undefined ? state : state % bound;
  };
}

class Grammar {
  constructor(start, rules) {
    if (typeof start !== "string" || !(start in rules)) {
      throw new TypeError(`start symbol '${start}' has no production`);
    }
    this.start = start;
    this.rules = rules;
    this.minLengths = this.#analyse();
    this.productions = [];
    for (const [nonterminal, alternatives] of Object.entries(rules)) {
      alternatives.forEach((_, index) => this.productions.push(`${nonterminal}#${index}`));
    }
  }

  /** Validates the grammar and returns the shortest-derivation-length fix point. */
  #analyse() {
    const rules = this.rules;
    const min = new Map(Object.keys(rules).map((nt) => [nt, Infinity]));
    for (const [nonterminal, alternatives] of Object.entries(rules)) {
      if (!Array.isArray(alternatives) || alternatives.length === 0) {
        throw new TypeError(`nonterminal '${nonterminal}' needs at least one alternative`);
      }
      for (const alternative of alternatives) {
        if (!Array.isArray(alternative)) throw new TypeError(`alternative of '${nonterminal}' must be an array`);
        for (const symbol of alternative) {
          if (isTerminal(symbol)) {
            T(symbol.text);            // re-validates the terminal shape
          } else if (typeof symbol === "string") {
            if (!(symbol in rules)) throw new TypeError(`undefined nonterminal '${symbol}' in '${nonterminal}'`);
          } else {
            throw new TypeError(`symbol of '${nonterminal}' must be a nonterminal name or T(char)`);
          }
        }
      }
    }
    for (let round = 0; round < 4 * Object.keys(rules).length * 4 + 8; round++) {
      let changed = false;
      for (const [nonterminal, alternatives] of Object.entries(rules)) {
        for (const alternative of alternatives) {
          let total = 0;
          let finite = true;
          for (const symbol of alternative) {
            const piece = isTerminal(symbol) ? 1 : min.get(symbol);
            if (piece === Infinity) { finite = false; break; }
            total += piece;
          }
          if (finite && total < min.get(nonterminal)) { min.set(nonterminal, total); changed = true; }
        }
      }
      if (!changed) {
        for (const [nonterminal, value] of min) {
          if (value === Infinity) throw new TypeError(`nonterminal '${nonterminal}' is unproductive (no finite derivation)`);
        }
        return min;
      }
    }
    throw new Error("grammar analysis did not reach a fix point");
  }

  minLength(nonterminal) { return this.minLengths.get(nonterminal); }
  isNullable(nonterminal) { return this.minLengths.get(nonterminal) === 0; }
  get productionCount() { return this.productions.length; }

  /** Length of the shortest string an alternative can produce. */
  #alternativeMin(alternative) {
    let total = 0;
    for (const symbol of alternative) total += isTerminal(symbol) ? 1 : this.minLengths.get(symbol);
    return total;
  }

  /** Expands `nonterminal` inside `budget` characters, recording the productions used. */
  #expand(nonterminal, budget, rng, trace) {
    const alternatives = this.rules[nonterminal];
    const feasible = [];
    alternatives.forEach((alternative, index) => {
      if (this.#alternativeMin(alternative) <= budget) feasible.push(index);
    });
    if (feasible.length === 0) throw new RangeError(`no alternative of '${nonterminal}' fits in ${budget} characters`);
    const chosen = feasible[rng(feasible.length)];
    trace.push(`${nonterminal}#${chosen}`);
    const alternative = alternatives[chosen];
    let remaining = budget;
    let text = "";
    for (let i = 0; i < alternative.length; i++) {
      const symbol = alternative[i];
      if (isTerminal(symbol)) {
        text += symbol.text;
        remaining -= 1;
        continue;
      }
      let tail = 0;
      for (let j = i + 1; j < alternative.length; j++) {
        const rest = alternative[j];
        tail += isTerminal(rest) ? 1 : this.minLengths.get(rest);
      }
      const piece = this.#expand(symbol, remaining - tail, rng, trace);
      text += piece;
      remaining -= piece.length;
    }
    return text;
  }

  /** One fuzz case: the generated text plus the derivation (production ids) that produced it. */
  generateDerivation(rng, maxLength) {
    if (maxLength < this.minLength(this.start)) {
      throw new RangeError(`budget ${maxLength} is below the shortest derivation of '${this.start}' (${this.minLength(this.start)})`);
    }
    const trace = [];
    const text = this.#expand(this.start, maxLength, rng, trace);
    return { text, trace };
  }

  generate(rng, maxLength) { return this.generateDerivation(rng, maxLength).text; }

  /**
   * All strings of length <= maxLength derivable from the start symbol, by chaotic
   * iteration over (nonterminal, length) buckets: a monotone map on a finite lattice, so it
   * reaches its least fix point. Because every production is applied to shorter or equally
   * long pieces and the universe is finite, the loop terminates.
   */
  enumerate(maxLength) {
    const nonterminals = Object.keys(this.rules);
    const buckets = new Map(nonterminals.map((nt) => [nt, Array.from({ length: maxLength + 1 }, () => new Set())]));
    const limit = 8 * (maxLength + 1) * (nonterminals.length + 1);
    for (let round = 0; ; round++) {
      if (round > limit) throw new Error("enumeration did not reach a fix point");
      let changed = false;
      for (const nonterminal of nonterminals) {
        for (const alternative of this.rules[nonterminal]) {
          let partial = Array.from({ length: maxLength + 1 }, () => new Set());
          partial[0].add("");
          for (const symbol of alternative) {
            const next = Array.from({ length: maxLength + 1 }, () => new Set());
            for (let len = 0; len <= maxLength; len++) {
              if (partial[len].size === 0) continue;
              if (isTerminal(symbol)) {
                if (len + 1 <= maxLength) for (const prefix of partial[len]) next[len + 1].add(prefix + symbol.text);
              } else {
                for (let extra = 0; len + extra <= maxLength; extra++) {
                  for (const suffix of buckets.get(symbol)[extra]) {
                    for (const prefix of partial[len]) next[len + extra].add(prefix + suffix);
                  }
                }
              }
            }
            partial = next;
          }
          for (let len = 0; len <= maxLength; len++) {
            for (const text of partial[len]) {
              if (!buckets.get(nonterminal)[len].has(text)) { buckets.get(nonterminal)[len].add(text); changed = true; }
            }
          }
        }
      }
      if (!changed) break;
    }
    const language = new Set();
    for (let len = 0; len <= maxLength; len++) for (const text of buckets.get(this.start)[len]) language.add(text);
    return language;
  }
}

/** Independent recognizer for the bracket language: stack machine, no grammar involved. */
function parseDyck(text) {
  const stack = [];
  const closing = { ")": "(", "]": "[" };
  for (const character of text) {
    if (character === "(" || character === "[") {
      stack.push(character);
    } else if (character in closing) {
      if (stack.pop() !== closing[character]) return false;
    } else {
      return false;      // character outside the alphabet
    }
  }
  return stack.length === 0;
}

/** Brute-force enumeration of every string over `alphabet` with length <= maxLength. */
function* allStrings(alphabet, maxLength) {
  const pending = [""];
  while (pending.length > 0) {
    const text = pending.pop();
    yield text;
    if (text.length < maxLength) for (const character of alphabet) pending.push(text + character);
  }
}

/** Character-level mutation, the classic non-grammar-aware fuzzer primitive. */
function mutate(text, rng) {
  const alphabet = "()[]x";
  const operation = rng(3);
  const cut = text.length === 0 ? 0 : rng(text.length);
  if (operation === 0) return text.slice(0, cut) + alphabet[rng(alphabet.length)] + text.slice(cut);
  if (operation === 1) return text.length === 0 ? alphabet[rng(alphabet.length)] : text.slice(0, cut) + text.slice(cut + 1);
  return text.length === 0
    ? alphabet[rng(alphabet.length)]
    : text.slice(0, cut) + alphabet[rng(alphabet.length)] + text.slice(cut + 1);
}

/** DYCK = { balanced strings over ( ) [ ] }: S := eps | '(' S ')' S | '[' S ']' S. */
function dyckGrammar() {
  return new Grammar("S", {
    S: [
      [],
      [T("("), "S", T(")"), "S"],
      [T("["), "S", T("]"), "S"],
    ],
  });
}

/** S := 'a' S 'b' | 'ab': shortest string is "ab", so a budget of 1 is impossible. */
function strictGrammar() {
  return new Grammar("S", { S: [[T("a"), "S", T("b")], [T("a"), T("b")]] });
}

module.exports = { Grammar, T, makeRng, parseDyck, allStrings, mutate, dyckGrammar, strictGrammar };

if (require.main === module) {
  const grammar = dyckGrammar();
  strictEqual(grammar.productionCount, 3, "three productions");
  ok(grammar.isNullable("S"), "S derives the empty string");
  strictEqual(grammar.minLength("S"), 0, "shortest derivation is empty");

  // -- (a) generation and enumeration agree with an independent recognizer ------------
  const ALPHABET = "()[]";
  for (let maxLength = 0; maxLength <= 7; maxLength++) {
    const enumerated = grammar.enumerate(maxLength);
    const bruteForce = new Set();
    for (const candidate of allStrings(ALPHABET, maxLength)) if (parseDyck(candidate)) bruteForce.add(candidate);
    deepStrictEqual([...enumerated].sort(), [...bruteForce].sort(), `language up to length ${maxLength}`);
    for (const text of enumerated) ok(parseDyck(text), `enumerated '${text}' must be balanced`);
  }
  strictEqual(grammar.enumerate(0).size, 1, "only the empty word fits in 0 characters");
  strictEqual(grammar.enumerate(2).size, 3, "the empty word plus the two balanced words of length 2");
  deepStrictEqual([...grammar.enumerate(2)].filter((t) => t.length === 2).sort(), ["()", "[]"], "balanced words of length 2");

  // -- (b) fuzzing: every generated input is structurally valid, and deterministic -----
  const rng = makeRng(0x751751);
  const corpus = [];
  const covered = new Set();
  for (let i = 0; i < 60; i++) {
    const sample = grammar.generateDerivation(rng, 12);
    ok(parseDyck(sample.text), `generated '${sample.text}' must be accepted by the recognizer`);
    ok(sample.text.length <= 12, "generated input must respect the character budget");
    corpus.push(sample.text);
    for (const production of sample.trace) covered.add(production);
  }
  deepStrictEqual([...covered].sort(), grammar.productions.slice().sort(), "every production must be covered");
  ok(corpus.some((text) => text.length >= 6), "the corpus must reach non-trivial inputs");
  ok(corpus.some((text) => text.includes("[")), "the corpus must use both bracket kinds");

  function replay(seed) {
    const local = makeRng(seed);
    return Array.from({ length: 25 }, () => grammar.generate(local, 10)).join("|");
  }
  strictEqual(replay(99), replay(99), "a fixed seed must replay the same corpus");
  ok(replay(99) !== replay(100), "different seeds must explore differently");

  // -- (c) grammar-based generation beats character mutation at producing valid inputs
  const mutationRng = makeRng(0xbeef);
  let mutations = 0;
  let mutationsAccepted = 0;
  for (const text of corpus) {
    for (let i = 0; i < 10; i++) {
      mutations++;
      if (parseDyck(mutate(text, mutationRng))) mutationsAccepted++;
    }
  }
  const grammarAcceptRate = corpus.filter(parseDyck).length / corpus.length;
  const mutationAcceptRate = mutationsAccepted / mutations;
  strictEqual(grammarAcceptRate, 1, "grammar-based inputs are valid by construction");
  ok(mutationAcceptRate < 0.5, `character mutations should mostly be rejected (got ${mutationAcceptRate})`);

  // -- (e) ill-formed grammars are rejected before any fuzzing happens ----------------
  throws(() => new Grammar("S", { S: [["S"]] }), TypeError);            // unproductive: no finite derivation
  throws(() => new Grammar("S", { S: [[T("a"), "Missing"]] }), TypeError);
  throws(() => new Grammar("S", { T: [[T("a")]] }), TypeError);         // start symbol has no production
  throws(() => new Grammar("S", { S: [] }), TypeError);                 // an empty alternative list is not a grammar
  throws(() => new Grammar("S", { S: [[T("ab")]] }), TypeError);        // terminals are exactly one character
  ok(grammar.generate(makeRng(1), 13).length <= 13, "a legal budget must produce a short input");
  // -- boundary: budget below the shortest derivation must be refused -----------------
  const strict = strictGrammar();
  strictEqual(strict.minLength("S"), 2, "shortest string of the strict grammar is 'ab'");
  deepStrictEqual([...strict.enumerate(4)].sort(), ["aabb", "ab"], "strict grammar language");
  throws(() => strict.generate(makeRng(3), 1), RangeError);
  strictEqual(strict.generate(makeRng(3), 2), "ab", "the only derivation of length 2");
  strictEqual(strict.generate(makeRng(3), 6).length % 2, 0, "'a'*n 'b'*n has even length");
  strictEqual(grammar.generate(makeRng(7), 0), "", "budget 0 can only produce the empty word");
  const generatedLengths = corpus.map((text) => text.length);
  console.log(`productions=${grammar.productionCount} covered=${covered.size} samples=${corpus.length} ` +
    `maxLen=${Math.max(...generatedLengths)} grammarAccept=${grammarAcceptRate.toFixed(2)} ` +
    `mutationAccept=${mutationAcceptRate.toFixed(3)}`);
  console.log("PASS");
}
