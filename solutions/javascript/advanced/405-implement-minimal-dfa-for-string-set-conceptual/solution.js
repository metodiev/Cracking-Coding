/** Task 405 - Implement a minimal DFA for a fixed string set (conceptual). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Scope: the DFA recognizes exactly the finite set of words it was built from. Only the
 * *prefixes* of the set's words matter: the builder's states are exactly the distinct
 * non-empty prefixes plus the empty root, and any character no prefix uses leads to an
 * explicit dead state. The alphabet is the sorted union of the words' characters; characters
 * outside it are documented to reject.
 *
 * Minimization uses the classic table-filling (Moore) algorithm: mark every pair that differs
 * in acceptance, then repeatedly mark a pair when some character leads to an already marked
 * pair; inseparable pairs are merged with union-find. For {"he", "she", "his"} the three
 * accepting states merge into one class, so the complete minimal DFA has 7 states: {""},
 * {"h"}, {"s"}, {"sh"}, {"hi"}, {"he","she","his"} and one dead state. Note "h" does NOT
 * merge with "sh": "h" still accepts the continuation "is" ("his") while "sh" only accepts
 * "e". Dropping the dead state leaves the 6-state partial DFA that tracks exactly the
 * prefixes.
 *
 * Checks: the enumerated language up to length 4 equals the key set exactly, near-miss
 * strings are rejected, every state is reachable, and the minimized automaton has no pair of
 * equivalent states (product-BFS equivalence test), i.e. it is minimal.
 */
"use strict";

function buildExactMatchDfa(words, alphabet) {
  const states = [""];
  const index = new Map([["", 0]]);
  const accept = [false];
  const trans = [new Array(alphabet.length).fill(-1)];
  for (const word of words) {
    let node = 0;
    let prefix = "";
    for (const ch of word) {
      const nxt = prefix + ch;
      if (!index.has(nxt)) {
        index.set(nxt, states.length);
        states.push(nxt);
        accept.push(false);
        trans.push(new Array(alphabet.length).fill(-1));
      }
      const col = alphabet.indexOf(ch);
      if (trans[node][col] === -1) trans[node][col] = index.get(nxt);
      node = index.get(nxt);
      prefix = nxt;
    }
    accept[node] = true;
  }
  const dead = states.length;
  states.push("(dead)");
  accept.push(false);
  trans.push(new Array(alphabet.length).fill(dead));
  for (const row of trans) {
    for (let c = 0; c < alphabet.length; c++) if (row[c] === -1) row[c] = dead;
  }
  return { states, index, trans, accept, dead, alphabet };
}

function minimizeDfa(trans, accept, alphabetSize) {
  const n = trans.length;
  const marked = Array.from({ length: n }, () => new Array(n).fill(false));
  for (let p = 0; p < n; p++) {
    for (let q = p + 1; q < n; q++) if (accept[p] !== accept[q]) marked[p][q] = true;
  }
  let changed = true;
  while (changed) {
    changed = false;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (marked[p][q]) continue;
        for (let c = 0; c < alphabetSize; c++) {
          const a = trans[p][c];
          const b = trans[q][c];
          if (a === b) continue;
          const lo = Math.min(a, b);
          const hi = Math.max(a, b);
          if (marked[lo][hi]) {
            marked[p][q] = true;
            changed = true;
            break;
          }
        }
      }
    }
  }
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  for (let p = 0; p < n; p++) {
    for (let q = p + 1; q < n; q++) {
      if (!marked[p][q]) {
        const rp = find(p);
        const rq = find(q);
        if (rp !== rq) parent[Math.max(rp, rq)] = Math.min(rp, rq);
      }
    }
  }
  const reps = [...new Set(Array.from({ length: n }, (_, i) => find(i)))].sort((a, b) => a - b);
  const remap = new Map(reps.map((rep, i) => [rep, i]));
  const classOf = Array.from({ length: n }, (_, i) => remap.get(find(i)));
  const newTrans = reps.map((rep) => trans[rep].map((nxt) => classOf[nxt]));
  const newAccept = reps.map((rep) => accept[rep]);
  return { classOf, trans: newTrans, accept: newAccept };
}

function dfaAccepts(trans, accept, word, alphabet) {
  let state = 0;
  for (const ch of word) {
    const col = alphabet.indexOf(ch);
    if (col === -1) return false;           // documented: outside the alphabet, reject
    state = trans[state][col];
  }
  return accept[state];
}

function reachableStates(trans) {
  const seen = new Set([0]);
  const stack = [0];
  while (stack.length) {
    const state = stack.pop();
    for (const nxt of trans[state]) {
      if (!seen.has(nxt)) {
        seen.add(nxt);
        stack.push(nxt);
      }
    }
  }
  return seen;
}

function allPairsDistinguishable(trans, accept, alphabetSize) {
  const n = trans.length;
  for (let p = 0; p < n; p++) {
    for (let q = p + 1; q < n; q++) {
      const seen = new Set([`${p},${q}`]);
      const stack = [[p, q]];
      let equivalent = true;
      while (stack.length && equivalent) {
        const [a, b] = stack.pop();
        if (accept[a] !== accept[b]) {
          equivalent = false;
          break;
        }
        for (let c = 0; c < alphabetSize; c++) {
          const na = trans[a][c];
          const nb = trans[b][c];
          const key = `${Math.min(na, nb)},${Math.max(na, nb)}`;
          if (!seen.has(key)) {
            seen.add(key);
            stack.push([na, nb]);
          }
        }
      }
      if (equivalent) return false;
    }
  }
  return true;
}

function languageOfDfa(trans, accept, alphabet, maxLen) {
  const accepted = [];
  let frontier = [""];
  for (let depth = 0; depth <= maxLen; depth++) {
    const nextFrontier = [];
    for (const word of frontier) {
      if (dfaAccepts(trans, accept, word, alphabet)) accepted.push(word);
      if (word.length < maxLen) {
        for (const ch of alphabet) nextFrontier.push(word + ch);
      }
    }
    frontier = nextFrontier;
  }
  return accepted.sort();
}

function deadStates(trans, accept) {
  const dead = [];
  for (let state = 0; state < trans.length; state++) {
    if (accept[state]) continue;
    if (trans[state].every((nxt) => nxt === state)) dead.push(state);
  }
  return dead;
}

module.exports = { buildExactMatchDfa, minimizeDfa, dfaAccepts, reachableStates, allPairsDistinguishable, languageOfDfa, deadStates };

if (require.main === module) {
  const assert = require("node:assert");
  const alphabet = "ehis";
  const words = ["he", "she", "his"];
  const builder = buildExactMatchDfa(words, alphabet);
  const expectReject = ["", "h", "s", "e", "i", "heh", "hehe", "sh", "shi", "shee", "shes",
    "hiss", "hishe", "hhi", "hh", "sheh", "hise"];

  const prefixes = new Set([""]);
  for (const word of words) {
    for (let k = 1; k <= word.length; k++) prefixes.add(word.slice(0, k));
  }
  assert.deepStrictEqual([...builder.states.slice(0, -1)].sort(), [...prefixes].sort());
  assert.strictEqual(builder.states.length, prefixes.size + 1);
  assert.strictEqual(builder.dead, builder.states.length - 1);
  assert.strictEqual(reachableStates(builder.trans).size, builder.trans.length);

  for (const word of words) assert.ok(dfaAccepts(builder.trans, builder.accept, word, alphabet), word);
  for (const word of expectReject) assert.ok(!dfaAccepts(builder.trans, builder.accept, word, alphabet), word);
  for (const word of ["hey", "his!", "she?"]) {
    assert.ok(!dfaAccepts(builder.trans, builder.accept, word, alphabet), word);
  }
  assert.deepStrictEqual(languageOfDfa(builder.trans, builder.accept, alphabet, 4), [...words].sort());

  const minimal = minimizeDfa(builder.trans, builder.accept, alphabet.length);
  assert.strictEqual(minimal.trans.length, 7);
  assert.strictEqual(minimal.accept.filter(Boolean).length, 1);
  assert.strictEqual(reachableStates(minimal.trans).size, minimal.trans.length);
  assert.ok(allPairsDistinguishable(minimal.trans, minimal.accept, alphabet.length));

  assert.deepStrictEqual(languageOfDfa(minimal.trans, minimal.accept, alphabet, 4), [...words].sort());
  for (const word of words) assert.ok(dfaAccepts(minimal.trans, minimal.accept, word, alphabet), word);
  for (const word of expectReject) assert.ok(!dfaAccepts(minimal.trans, minimal.accept, word, alphabet), word);

  assert.strictEqual(minimal.classOf[builder.index.get("he")], minimal.classOf[builder.index.get("she")]);
  assert.strictEqual(minimal.classOf[builder.index.get("she")], minimal.classOf[builder.index.get("his")]);
  assert.notStrictEqual(minimal.classOf[builder.index.get("h")], minimal.classOf[builder.index.get("sh")]);
  assert.notStrictEqual(minimal.classOf[builder.index.get("sh")], minimal.classOf[builder.index.get("hi")]);
  assert.notStrictEqual(minimal.classOf[builder.index.get("h")], minimal.classOf[builder.index.get("s")]);
  assert.notStrictEqual(minimal.classOf[builder.index.get("")], minimal.classOf[builder.dead]);
  assert.strictEqual(new Set(minimal.classOf).size, 7);
  assert.strictEqual(deadStates(minimal.trans, minimal.accept).length, 1);
  assert.strictEqual(minimal.trans.length - 1, 6);

  const alphabet2 = "ab";
  const words2 = ["ab", "b", "ba"];
  const builder2 = buildExactMatchDfa(words2, alphabet2);
  for (const word of words2) assert.ok(dfaAccepts(builder2.trans, builder2.accept, word, alphabet2), word);
  for (const word of ["", "a", "bb", "aba", "abb", "bab", "baa"]) {
    assert.ok(!dfaAccepts(builder2.trans, builder2.accept, word, alphabet2), word);
  }
  assert.deepStrictEqual(languageOfDfa(builder2.trans, builder2.accept, alphabet2, 4), [...words2].sort());
  const minimal2 = minimizeDfa(builder2.trans, builder2.accept, alphabet2.length);
  assert.strictEqual(minimal2.trans.length, 5);
  assert.ok(allPairsDistinguishable(minimal2.trans, minimal2.accept, alphabet2.length));
  assert.deepStrictEqual(languageOfDfa(minimal2.trans, minimal2.accept, alphabet2, 4), [...words2].sort());
  assert.strictEqual(minimal2.classOf[builder2.index.get("ab")], minimal2.classOf[builder2.index.get("ba")]);
  assert.notStrictEqual(minimal2.classOf[builder2.index.get("ab")], minimal2.classOf[builder2.index.get("b")]);
  assert.notStrictEqual(minimal2.classOf[builder2.index.get("a")], minimal2.classOf[builder2.dead]);
  console.log("PASS");
}
