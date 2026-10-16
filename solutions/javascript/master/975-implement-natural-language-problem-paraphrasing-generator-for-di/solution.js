/** Task 975 - Implement natural-language problem paraphrasing generator for diverse practice.
 * (Tier: Master, CTCI chapter Beyond - Software Engineering & Tooling)
 *
 * A problem is a small structured spec (required slots plus the numbers the statement must keep),
 * and each frame is an English sentence plan with `{slot}` placeholders and `%concept%` markers.
 * The generator first validates every frame against the spec (no missing or unknown slots), then
 * draws frames and synonyms from a hand-written lexicon with a seeded PRNG, deduplicates the
 * renderings and enforces a slot-preservation invariant on every single variant it emits.
 * Simplified: frames and synonyms are curated data (no seq2seq model) and the slot check is a
 * substring/number scan rather than a semantic parse. Verified: slot and number preservation on
 * every variant, an independent regex extraction oracle, frame and lexicon diversity, the k=0 and
 * combination-exhaustion boundaries, invalid-frame rejection and determinism per seed.
 */
"use strict";
const assert = require("node:assert");

const SPEC = { entities: ["nums"], numbers: [9],
  slots: { array: "nums", target: "9" } };

const FRAMES = [
  "Given {array} and a target {target}, %find% a %pair% that adds up to {target} and %report% their indices.",
  "In {array}, %find% a %pair% adding up to {target}.",
  "%find% a %pair% inside {array} whose sum equals {target} and %report% the positions.",
];
const LEXICON = {
  find: ["find", "locate", "identify", "determine"],
  pair: ["pair", "two entries", "two values"],
  report: ["report", "return", "produce"],
};

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, unit: () => Number(next() >> 11n) / 2 ** 53, int: (n) => Number(next() % BigInt(n)) };
}

const slotsIn = (frame) => [...frame.matchAll(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g)].map((m) => m[1]);
const conceptsIn = (frame) => [...new Set([...frame.matchAll(/%([A-Za-z_][A-Za-z0-9_]*)%/g)].map((m) => m[1]))];

/** A frame must cover every required slot and may only mention slots the spec knows. */
function validateFrame(frame, spec, lexicon = LEXICON) {
  const known = new Set(Object.keys(spec.slots));
  for (const slot of slotsIn(frame)) {
    if (!known.has(slot)) throw new Error(`frame references unknown slot {${slot}}`);
  }
  for (const slot of known) {
    if (!slotsIn(frame).includes(slot)) throw new Error(`frame is missing required slot {${slot}}`);
  }
  for (const concept of conceptsIn(frame)) {
    if (!(concept in lexicon)) throw new Error(`frame references unknown concept %${concept}%`);
  }
  return true;
}

/** Render one frame: synonyms come from the lexicon, slots from the spec. */
function renderFrame(frame, spec, lexicon, rng) {
  return frame
    .replace(/%([A-Za-z_][A-Za-z0-9_]*)%/g, (_, concept) => lexicon[concept][rng.int(lexicon[concept].length)])
    .replace(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (_, slot) => spec.slots[slot]);
}

/** Independent oracle: the numbers and entity names that survived into a rendering. */
function inspect(text) {
  return { numbers: [...new Set((text.match(/\d+/g) || []).map(Number))],
    entities: ["nums"].filter((name) => text.split(/\W+/).includes(name)) };
}

/** Numbers a variant may legitimately contain: the spec's own, plus those inside slot values. */
const allowedNumbers = (spec) => new Set([...spec.numbers,
  ...Object.values(spec.slots).flatMap((value) => (value.match(/\d+/g) || []).map(Number))]);

const preserves = (text, spec) => spec.entities.every((name) => inspect(text).entities.includes(name))
  && spec.numbers.every((value) => inspect(text).numbers.includes(value))
  && inspect(text).numbers.every((value) => allowedNumbers(spec).has(value));

/** Distinct (frame, synonym combination) pairs the space can possibly produce. */
const frameSpace = (frames, lexicon) => frames.reduce((sum, frame) => sum
  + conceptsIn(frame).reduce((product, concept) => product * lexicon[concept].length, 1), 0);

/** Draw `count` distinct paraphrases; stop early once the combination space is exhausted. */
function generate({ spec = SPEC, frames = FRAMES, lexicon = LEXICON, count = 6, seed = 975 } = {}) {
  frames.forEach((frame) => validateFrame(frame, spec, lexicon));
  const rng = splitmix64(seed);
  const space = frameSpace(frames, lexicon);
  const variants = [];
  const seen = new Set();
  for (let attempt = 0; attempt < Math.max(40 * count, 4 * space) && variants.length < count; attempt += 1) {
    const index = rng.int(frames.length);
    const text = renderFrame(frames[index], spec, lexicon, rng);
    if (seen.has(text)) continue;
    assert(preserves(text, spec), `variant dropped a slot: ${text}`);
    seen.add(text);
    variants.push({ text, frame: index, numbers: inspect(text).numbers, concepts: conceptsIn(frames[index]) });
  }
  return variants;
}

function main() {
  // 1. Happy path: every paraphrase keeps the slot values and stays distinct.
  const variants = generate({ count: 6, seed: 975 });
  assert.strictEqual(variants.length, 6, `six paraphrases: ${variants.length}`);
  assert.strictEqual(new Set(variants.map((v) => v.text)).size, 6, "paraphrases are unique");
  assert(variants.every((v) => v.text.includes("nums") && v.text.includes("9")), "slots survive rendering");
  assert(variants.every((v) => !v.text.includes("%") && !v.text.includes("{")), "no markers are left behind");
  assert(variants.every((v) => v.frame >= 0 && v.frame < FRAMES.length), "frame indices are in range");
  assert(variants.every((v) => v.concepts.every((concept) => LEXICON[concept]
    .some((word) => v.text.toLowerCase().includes(word)))), "every variant uses its frame's concepts");

  // 2. Independent oracle: numbers and entities re-extracted from the text match the spec.
  for (const variant of variants) {
    const found = inspect(variant.text);
    assert.deepStrictEqual(found.numbers, SPEC.numbers, `numbers preserved in "${variant.text}"`);
    assert.deepStrictEqual(found.entities, ["nums"], "the array name is preserved");
    assert(preserves(variant.text, SPEC), "the preservation predicate agrees with the scan");
  }

  // 3. Diversity: several frames and several synonyms of each group are exercised.
  assert(new Set(variants.map((v) => v.frame)).size >= 2, "more than one frame is used");
  for (const [concept, list] of Object.entries(LEXICON)) {
    const used = new Set(list.filter((word) => variants.some((v) => v.text.includes(word))
      || variants.some((v) => v.text.includes(` ${word} `))));
    assert(used.size >= 2, `concept ${concept} varies: ${[...used]}`);
  }
  assert(variants.every((v) => v.concepts.length > 0), "every variant stems from a concept frame");

  // 4. Boundary: zero variants, one-combination frames and an exhausted combination space.
  assert.deepStrictEqual(generate({ count: 0 }), [], "asking for nothing yields nothing");
  const single = generate({ frames: [FRAMES[1]], lexicon: { find: ["find"], pair: ["pair"] }, count: 1, seed: 4 });
  assert.strictEqual(single.length, 1, "one frame with one synonym can only produce one paraphrase");
  assert.strictEqual(single[0].text, "In nums, find a pair adding up to 9.", "the single combination renders");
  const space = frameSpace(FRAMES, LEXICON);
  const exhausted = generate({ count: 1000, seed: 975 });
  assert.strictEqual(exhausted.length, space, `the whole combination space is ${space}`);
  assert.strictEqual(new Set(exhausted.map((v) => v.text)).size, space, "no duplicates at exhaustion");

  // 5. A different seed explores a different corner of the space.
  const other = generate({ count: 6, seed: 1975 });
  assert.notDeepStrictEqual(other.map((v) => v.text), variants.map((v) => v.text), "another seed paraphrases differently");

  // 6. Negative: frames that lose or invent slots are rejected before any rendering happens.
  assert.throws(() => validateFrame("In {array}, find a pair.", SPEC), /missing required slot \{target\}/,
    "a frame without the target slot is rejected");
  assert.throws(() => validateFrame("In {array}, find two values summing to {target} using {ghost}.", SPEC),
    /unknown slot \{ghost\}/, "an invented slot is rejected");
  assert.throws(() => validateFrame("In {array}, %sprint% to {target}.", SPEC), /unknown concept %sprint%/,
    "an unknown concept marker is rejected");
  assert.throws(() => generate({ frames: ["In {array}, find a pair."] }), /missing required slot/, "the gate fires");
  assert.throws(() => generate({ frames: [FRAMES[1]], lexicon: { ...LEXICON, pair: ["2 entries"] }, count: 1 }),
    /dropped a slot/, "a synonym that invents a number is caught by the preservation invariant");

  // 7. Boundary spec: a statement that must keep no numbers still renders and validates.
  const bare = { entities: ["nums"], numbers: [], slots: { array: "nums" } };
  const bareVariants = generate({ spec: bare, frames: ["In {array}, %find% a %pair%."],
    lexicon: { find: ["find", "locate"], pair: ["pair"] }, count: 2, seed: 7 });
  assert.strictEqual(bareVariants.length, 2, "a number-free spec still paraphrases");
  assert(bareVariants.every((v) => preserves(v.text, bare)), "the preservation predicate is vacuous but holds");
  assert(bareVariants.every((v) => v.numbers.length === 0), "no numbers are invented");

  // 8. Determinism: the same seed always yields the same six sentences.
  assert(JSON.stringify(generate({ count: 6, seed: 975 })) === JSON.stringify(variants), "generation is stable");
  assert(JSON.stringify(generate({ count: 6, seed: 975, frames: [...FRAMES].reverse() }))
    !== JSON.stringify(variants), "frame order is part of the seed's mapping");

  const synonymUse = new Set(variants.flatMap((v) => v.text.split(/\W+/))).size;
  console.log(`paraphrase: ${variants.length} variants from ${FRAMES.length} frames, ` +
    `${space} combinations in the space, ${synonymUse} distinct words, numbers kept ${SPEC.numbers.join(",")}`);  console.log("PASS");
}
module.exports = { SPEC, FRAMES, LEXICON, splitmix64, slotsIn, conceptsIn, frameSpace, validateFrame,
  renderFrame, inspect, allowedNumbers, preserves, generate };
if (require.main === module) main();
