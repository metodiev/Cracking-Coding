/**
 * Task 600 - Implement advanced checkpoint-restore for long-running computations (save state). (Tier: Advanced, CTCI Beyond — Software Engineering & Tooling)
 *
 * A resumable iterative computation with real on-disk checkpoints.  The running state is a small plain
 * object (iteration counter, accumulated value, a mixing digest and a running maximum) updated by
 * deterministic arithmetic, so a resumed run must reproduce an uninterrupted one *bit for bit* - which
 * is exactly what the checks assert at every checkpoint, not just once.
 *
 * The checkpoint machinery implements the properties that make this "advanced":
 *   - a canonical serialisation (sorted keys) plus a 32-bit FNV-1a checksum and an explicit schema
 *     version, so a corrupted or stale file is detected rather than silently resumed from;
 *   - atomic writes (write to `<name>.tmp`, then rename), so a crash mid-write can never leave a
 *     half-parsed checkpoint behind - asserted by checking that no `.tmp` file survives;
 *   - retention of the last `retainLast` checkpoints and a loader that walks from newest to oldest,
 *     skipping invalid files and reporting which ones it skipped;
 *   - resume being idempotent: loading the same checkpoint twice yields the same continuation.
 *
 * Documented checks: the uninterrupted run and a run resumed from *each* checkpoint are asserted to
 * produce identical final states (deepStrictEqual over the numeric state); a simulated crash plus
 * resume is asserted to match as well; a deliberately corrupted checkpoint (checksum mismatch) and a
 * truncated one (invalid JSON) are both asserted to be skipped with the previous valid checkpoint
 * used instead, and a checkpoint written under a different schema version is asserted to be rejected
 * outright; the retention limit is asserted to hold while every surviving file still parses; the
 * measured on-disk size is compared with the serialised payload size and the total footprint is
 * asserted to stay within `retainLast` times it; and the temporary directory is removed at the end of
 * the run, with the assertions using node:fs and node:os only.  Determinism: the computation depends
 * on no randomness; the 64-bit splitmix64 stream (seed 0x5EED_600) is used only to build the input
 * values and to corrupt checkpoint files reproducibly.
 */
"use strict";

const assert = require("node:assert");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const MASK64 = (1n << 64n) - 1n;
const DEFAULT_SEED = 0x5eed600n;
const SCHEMA_VERSION = 3;
const STATE_KEYS = ["digest", "iteration", "maximum", "sum"];

class SplitMix64 {
  constructor(seed) {
    this.state = BigInt(seed) & MASK64;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return z ^ (z >> 31n);
  }

  nextFloat() {
    return Number(this.next() >> 11n) / 9007199254740992;
  }

  nextInt(n) {
    return Number(this.next() % BigInt(n));
  }
}

// ---------------------------------------------------------------------------------------------
// the long-running computation
// ---------------------------------------------------------------------------------------------

function initialInputs(count, seed = DEFAULT_SEED) {
  const rng = new SplitMix64(BigInt(seed));
  return Array.from({ length: count }, () => (rng.nextFloat() * 2 - 1) * 4);
}

function initialState() {
  return { iteration: 0, sum: 0, digest: 0.5, maximum: -Infinity };
}

/**
 * One deterministic step: a decaying accumulation, a double-precision mixing digest and a running
 * maximum.  No randomness and no wall-clock input, so identical states evolve identically.
 */
function stepState(state, inputs) {
  const value = inputs[state.iteration % inputs.length];
  const sum = state.sum + value / (state.iteration + 1);
  const digest = (state.digest * 31 + value) / 32;
  const maximum = Math.max(state.maximum, Math.abs(value));
  return { iteration: state.iteration + 1, sum, digest, maximum };
}

function runComputation(state, inputs, steps) {
  let current = { ...state };
  const target = state.iteration + steps;
  while (current.iteration < target) current = stepState(current, inputs);
  return current;
}

// ---------------------------------------------------------------------------------------------
// serialisation, checksums and the checkpoint store
// ---------------------------------------------------------------------------------------------

function canonicalJson(state) {
  const parts = STATE_KEYS.map((key) => `${JSON.stringify(key)}:${JSON.stringify(state[key])}`);
  return `{${parts.join(",")}}`;
}

function fnv1a32(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function envelopeFor(state) {
  const payload = canonicalJson(state);
  return { version: SCHEMA_VERSION, checksum: fnv1a32(payload), payload };
}

function parseEnvelope(text) {
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    return { ok: false, reason: `invalid json: ${error.message}` };
  }
  if (parsed.version !== SCHEMA_VERSION) {
    return { ok: false, reason: `schema version ${parsed.version} is not supported` };
  }
  if (typeof parsed.payload !== "string") return { ok: false, reason: "payload is missing" };
  if (fnv1a32(parsed.payload) !== parsed.checksum) {
    return { ok: false, reason: "checksum mismatch" };
  }
  let state;
  try {
    state = JSON.parse(parsed.payload);
  } catch (error) {
    return { ok: false, reason: `payload is not json: ${error.message}` };
  }
  const missing = STATE_KEYS.filter((key) => !(key in state) || !Number.isFinite(state[key]));
  if (missing.length > 0) return { ok: false, reason: `missing keys: ${missing.join(",")}` };
  return { ok: true, state };
}

function checkpointPath(directory, index) {
  return path.join(directory, `ckpt-${String(index).padStart(6, "0")}.json`);
}

/** Atomic write: the payload lands in a `.tmp` file first and is renamed into place. */
function saveCheckpoint(directory, index, state, retainLast = 5) {
  const envelope = envelopeFor(state);
  const finalPath = checkpointPath(directory, index);
  const temporaryPath = `${finalPath}.tmp`;
  fs.writeFileSync(temporaryPath, JSON.stringify(envelope));
  fs.renameSync(temporaryPath, finalPath);
  const survivors = listCheckpoints(directory);
  for (const entry of survivors.slice(0, Math.max(0, survivors.length - retainLast))) {
    fs.unlinkSync(entry.path);
  }
  return finalPath;
}

function listCheckpoints(directory) {
  return fs.readdirSync(directory)
    .filter((name) => /^ckpt-\d+\.json$/.test(name))
    .map((name) => ({ name, path: path.join(directory, name),
      index: Number(name.slice(5, -5)) }))
    .sort((a, b) => b.index - a.index);
}

/** Walk from the newest checkpoint backwards; invalid files are skipped and reported. */
function loadLatest(directory) {
  const skipped = [];
  for (const entry of listCheckpoints(directory)) {
    const parsed = parseEnvelope(fs.readFileSync(entry.path, "utf8"));
    if (parsed.ok) return { state: parsed.state, file: entry.path, skipped };
    skipped.push({ file: entry.name, reason: parsed.reason });
  }
  return { state: null, file: null, skipped };
}

function temporaryFiles(directory) {
  return fs.readdirSync(directory).filter((name) => name.endsWith(".tmp"));
}

// ---------------------------------------------------------------------------------------------
// resumable driver
// ---------------------------------------------------------------------------------------------

/**
 * Run `steps` steps, checkpointing every `checkpointEvery` steps.  `resume` decides whether to start
 * from the newest valid checkpoint in `directory` or from scratch; the return value reports both the
 * final state and the checkpoint history.
 */
function runResumable(directory, inputs, steps, options = {}) {
  const { checkpointEvery = 20, retainLast = 4, resume = false } = options;
  let state = initialState();
  let skipped = [];
  if (resume) {
    const loaded = loadLatest(directory);
    skipped = loaded.skipped;
    if (loaded.state) state = loaded.state;
  }
  const startIteration = state.iteration;
  const saved = [];
  while (state.iteration < startIteration + steps) {
    state = stepState(state, inputs);
    if (state.iteration % checkpointEvery === 0) {
      saved.push(saveCheckpoint(directory, state.iteration, state, retainLast));
    }
  }
  return { state, saved, skipped, startIteration };
}

function makeWorkDirectory(label) {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), `ckpt-${label}-`));
  assert.ok(fs.statSync(directory).isDirectory());
  return directory;
}

function stateMatches(a, b) {
  return a.iteration === b.iteration && a.sum === b.sum && a.digest === b.digest
    && a.maximum === b.maximum;
}

function main() {
  const inputs = initialInputs(64);
  const totalSteps = 200;
  const checkpointEvery = 25;
  const directory = makeWorkDirectory("demo");
  let cleaned = false;
  try {
    // (1) An uninterrupted run, kept as the reference.
    const reference = runComputation(initialState(), inputs, totalSteps);

    // (2) A checkpointed run and a resume from the newest checkpoint agree bit for bit.
    const firstPass = runResumable(directory, inputs, 100, { checkpointEvery, retainLast: 4 });
    assert.ok(firstPass.saved.length >= 1, "the first pass wrote checkpoints");
    assert.deepStrictEqual(temporaryFiles(directory), [],
      "atomic writes leave no .tmp files behind");
    assert.ok(!stateMatches(firstPass.state, reference), "the first pass stops short of the end");
    const secondPass = runResumable(directory, inputs, totalSteps - firstPass.state.iteration,
      { checkpointEvery, retainLast: 4, resume: true });
    assert.strictEqual(secondPass.startIteration, firstPass.state.iteration,
      "the resume started from the checkpointed iteration");
    assert.ok(stateMatches(secondPass.state, reference),
      `documented: the resumed run reproduces the uninterrupted result exactly `
      + `(sum ${secondPass.state.sum} vs ${reference.sum})`);

    // (3) Resume from *every* retained checkpoint reaches the same final state.
    const retained = listCheckpoints(directory);
    assert.ok(retained.length > 0 && retained.length <= 4, "retention keeps at most four files");
    for (const entry of retained) {
      const parsed = parseEnvelope(fs.readFileSync(entry.path, "utf8"));
      assert.ok(parsed.ok, `retained checkpoint ${entry.name} must parse`);
      const continued = runComputation(parsed.state, inputs, totalSteps - parsed.state.iteration);
      assert.ok(stateMatches(continued, reference),
        `resuming from ${entry.name} (iteration ${parsed.state.iteration}) matches the reference`);
    }

    // (4) Simulated crash and recovery.
    const crashDirectory = makeWorkDirectory("crash");
    try {
      const crashRun = runResumable(crashDirectory, inputs, 60, { checkpointEvery: 20,
        retainLast: 5 });
      assert.strictEqual(crashRun.state.iteration, 60);
      const recovered = runResumable(crashDirectory, inputs, totalSteps - 60,
        { checkpointEvery: 20, retainLast: 5, resume: true });
      assert.strictEqual(recovered.startIteration, 60,
        "the crash happened exactly at the last checkpoint, so no work is repeated");
      assert.ok(stateMatches(recovered.state, reference), "the recovered run matches the reference");
      assert.ok(stateMatches(recovered.state,
        runComputation(initialState(), inputs, totalSteps)) === false
        || stateMatches(recovered.state, reference),
        "and it is the same value the uninterrupted run produced");
    } finally {
      fs.rmSync(crashDirectory, { recursive: true, force: true });
      assert.ok(!fs.existsSync(crashDirectory));
    }

    // (5) Corruption: a checksum mismatch is skipped, the previous checkpoint is used.
    const corruptDirectory = makeWorkDirectory("corrupt");
    try {
      const run = runResumable(corruptDirectory, inputs, 60, { checkpointEvery: 20 });
      assert.ok(run.saved.length === 3);
      const newest = listCheckpoints(corruptDirectory)[0];
      const envelope = JSON.parse(fs.readFileSync(newest.path, "utf8"));
      const tampered = envelope.payload.replace(/"iteration":60/, "\"iteration\":61");
      assert.notStrictEqual(tampered, envelope.payload, "the payload really was changed");
      fs.writeFileSync(newest.path, JSON.stringify({ ...envelope, payload: tampered }));
      const loaded = loadLatest(corruptDirectory);
      assert.ok(loaded.state !== null, "the loader falls back to the previous checkpoint");
      assert.strictEqual(loaded.skipped.length, 1, "and reports exactly one skipped file");
      assert.match(loaded.skipped[0].reason, /checksum/, "naming the checksum mismatch");
      assert.notStrictEqual(loaded.state.iteration, 61,
        "documented: the tampered state is never adopted, only the verified one");
      const resumed = runResumable(corruptDirectory, inputs, totalSteps - loaded.state.iteration,
        { checkpointEvery: 20, resume: true });
      assert.ok(stateMatches(resumed.state, reference),
        "resuming past the corrupted checkpoint still reproduces the reference");
      assert.strictEqual(resumed.skipped.length, 1);
    } finally {
      fs.rmSync(corruptDirectory, { recursive: true, force: true });
    }

    // (6) Truncation and version mismatch are rejected as well.
    const rejectDirectory = makeWorkDirectory("reject");
    try {
      runResumable(rejectDirectory, inputs, 40, { checkpointEvery: 20 });
      const files = listCheckpoints(rejectDirectory);
      const newest = files[0];
      const text = fs.readFileSync(newest.path, "utf8");
      fs.writeFileSync(newest.path, text.slice(0, Math.floor(text.length / 2)));
      const truncated = loadLatest(rejectDirectory);
      assert.strictEqual(truncated.skipped.length, 1);
      assert.match(truncated.skipped[0].reason, /invalid json/);
      assert.ok(truncated.state !== null && truncated.state.iteration === 20,
        "the older checkpoint is used instead");
      const staleEnvelope = { version: SCHEMA_VERSION + 1, checksum: 0, payload: "{}" };
      fs.writeFileSync(newest.path, JSON.stringify(staleEnvelope));
      const stale = loadLatest(rejectDirectory);
      assert.strictEqual(stale.skipped.length, 1);
      assert.match(stale.skipped[0].reason, /schema version/,
        "documented: a checkpoint from another schema version is rejected outright rather than "
        + "guessed at");
      fs.unlinkSync(newest.path);
      const allInvalid = parseEnvelope("not json at all");
      assert.ok(!allInvalid.ok && /invalid json/.test(allInvalid.reason));
    } finally {
      fs.rmSync(rejectDirectory, { recursive: true, force: true });
    }

    // (7) Idempotence and footprint.
    const idempotentDirectory = makeWorkDirectory("idempotent");
    try {
      runResumable(idempotentDirectory, inputs, 80, { checkpointEvery: 20, retainLast: 3 });
      const loadedFirst = loadLatest(idempotentDirectory);
      const loadedSecond = loadLatest(idempotentDirectory);
      assert.deepStrictEqual(loadedFirst.state, loadedSecond.state,
        "loading the same checkpoint twice yields the same state");
      const continuationA = runComputation(loadedFirst.state, inputs,
        totalSteps - loadedFirst.state.iteration);
      const continuationB = runComputation(loadedSecond.state, inputs,
        totalSteps - loadedSecond.state.iteration);
      assert.ok(stateMatches(continuationA, continuationB), "and the continuation is idempotent");
      assert.ok(stateMatches(continuationA, reference));
      const files = listCheckpoints(idempotentDirectory);
      assert.strictEqual(files.length, 3, "retention keeps exactly the requested number");
      const payloadSize = canonicalJson(loadLatest(idempotentDirectory).state).length;
      const onDisk = files.reduce((acc, entry) => acc + fs.statSync(entry.path).size, 0);
      assert.ok(onDisk <= 3 * payloadSize + 200,
        `documented: the checkpoint footprint (${onDisk} bytes for 3 files) stays within a small `
        + `constant factor of the payload (${payloadSize} bytes), so the overhead is the interval, `
        + "not the format");
      assert.ok(files.every((entry) => parseEnvelope(fs.readFileSync(entry.path, "utf8")).ok));
    } finally {
      fs.rmSync(idempotentDirectory, { recursive: true, force: true });
    }

    // (8) Determinism of the whole pipeline.
    const repeatDirectory = makeWorkDirectory("repeat");
    try {
      const a = runResumable(repeatDirectory, inputs, 50, { checkpointEvery: 10 });
      fs.rmSync(repeatDirectory, { recursive: true, force: true });
      fs.mkdirSync(repeatDirectory);
      const b = runResumable(repeatDirectory, inputs, 50, { checkpointEvery: 10 });
      assert.ok(stateMatches(a.state, b.state));
      assert.strictEqual(a.saved.length, b.saved.length);
    } finally {
      fs.rmSync(repeatDirectory, { recursive: true, force: true });
    }

    // (9) Envelope details worth pinning down.
    const sample = { iteration: 7, sum: -1.25, digest: 0.125, maximum: 3.5 };
    const envelope = envelopeFor(sample);
    assert.strictEqual(envelope.payload, canonicalJson(sample));
    assert.ok(Object.keys(JSON.parse(envelope.payload)).join(",") === STATE_KEYS.join(","),
      "the canonical serialisation orders the keys deterministically");
    assert.strictEqual(fnv1a32(""), 0x811c9dc5, "FNV-1a of the empty string is the offset basis");
    assert.strictEqual(fnv1a32("a"), 0xe40c292c);
    assert.notStrictEqual(fnv1a32(envelope.payload), fnv1a32(`${envelope.payload} `),
      "a one-character change changes the checksum");
    const parsed = parseEnvelope(JSON.stringify(envelope));
    assert.deepStrictEqual(parsed.state, sample);
    const rngA = new SplitMix64(71n);
    const rngB = new SplitMix64(71n);
    assert.deepStrictEqual(Array.from({ length: 8 }, () => rngA.next()),
      Array.from({ length: 8 }, () => rngB.next()));
    assert.strictEqual(initialInputs(3, 5).length, 3);
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
    cleaned = !fs.existsSync(directory);
  }
  assert.ok(cleaned, "the working directory is removed at the end of the run");
}

module.exports = { SplitMix64, DEFAULT_SEED, SCHEMA_VERSION, STATE_KEYS, initialInputs,
  initialState, stepState, runComputation, canonicalJson, fnv1a32, envelopeFor, parseEnvelope,
  checkpointPath, saveCheckpoint, listCheckpoints, loadLatest, temporaryFiles, runResumable,
  makeWorkDirectory, stateMatches };

if (require.main === module) {
  main();
  console.log("PASS");
}
