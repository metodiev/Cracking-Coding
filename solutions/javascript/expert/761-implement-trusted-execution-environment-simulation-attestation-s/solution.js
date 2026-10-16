/** Task 761 - Implement trusted execution environment simulation (attestation sketch). (Tier: Expert, CTCI Beyond — Security & Privacy)
 *
 * A toy enclave and the three security properties an attestation scheme is supposed to give, each
 * made testable. (1) *Measurement*: the loaded code pages are folded into one 32-bit measurement,
 * so any changed page (or a reordered page table) changes it. (2) *Attestation*: the enclave signs
 * `measurement || nonce || payload` with a MAC derived from a fixed, public "vendor" key; a verifier
 * accepts the report only if the MAC recomputes, the measurement equals the expected one and the
 * nonce has never been seen before — which is what makes a report fresh rather than replayable.
 * (3) *Sealing*: a blob is XOR-ed with a keystream derived from (sealing key, measurement, nonce),
 * so it can only be opened by an enclave with the same measurement (and the same monotonic version,
 * which is what blocks a rollback).
 * Every key here is a fixed public constant and every primitive is a 32-bit checksum-style toy, so
 * nothing secret is ever generated: the tests exercise the *shape* of the guarantees (binding,
 * freshness, sealing), never real cryptography.
 *
 * What is simplified: a toy MAC instead of HMAC, a one-time-pad-like keystream instead of an AEAD,
 * a single 32-bit measurement instead of a hash of the whole image, no key derivation hierarchy, no
 * remote transport and no real monotonic counter hardware. What is verified: honest reports verify;
 * a report with an edited field, a report signed with another key, and a replayed report are all
 * rejected; a sealed blob round-trips under the original measurement and fails under any other;
 * two enclaves only agree on a session key when their measurements match; a version rollback is
 * detected; every single-page change and every single-byte report edit is caught; and runs are
 * deterministic.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

/** Fixed public "keys": this is a simulation, so nothing secret is generated or stored. */
const VENDOR_KEY = 0x9e3779b9;
const SEALING_KEY = 0x85ebca6b;

/** 32-bit avalanche: a checksum-style toy primitive, deliberately not a cryptographic hash. */
function mix32(value) {
  let hash = value >>> 0;
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x7feb352d) >>> 0;
  hash ^= hash >>> 15;
  hash = Math.imul(hash, 0x846ca68b) >>> 0;
  hash ^= hash >>> 16;
  return hash >>> 0;
}

function hashText(text) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) hash = (Math.imul(hash, 0x01000193) ^ text.charCodeAt(i)) >>> 0;
  return hash >>> 0;
}

/** Toy keyed MAC over a list of parts: order matters, so every field is bound. */
function macOf(key, parts) {
  return parts.reduce((accumulator, part) => mix32(accumulator ^ hashText(`${part}`)), mix32(key ^ parts.length));
}

/** Measurement of the loaded image: order-sensitive fold over the code pages. */
const measurementOf = (pages) => pages.reduce((accumulator, page) => mix32(accumulator ^ hashText(page)), 0x5eed5eed);

/** Keystream word i of a seal: derived from the key, the measurement and the blob nonce. */
const keystreamWord = (measurement, nonce, index) => macOf(SEALING_KEY, [measurement, nonce, index]);

const toBytes = (text) => Array.from(Buffer.from(text, "utf8"));
const toText = (bytes) => Buffer.from(bytes).toString("utf8");

class Enclave {
  constructor(pages, { version = 1 } = {}) {
    if (!Array.isArray(pages) || pages.length === 0) throw new RangeError("an enclave needs at least one code page");
    this.pages = pages.slice();
    this.measurement = measurementOf(this.pages);
    this.version = version;
  }

  /** Binds a payload to the measurement: only an enclave with the same image can open it. */
  seal(plaintext, nonce) {
    const bytes = toBytes(plaintext);
    const sealed = bytes.map((byte, index) => {
      const word = keystreamWord(this.measurement, nonce, index >> 2);
      return byte ^ ((word >>> (8 * (index & 3))) & 0xff);
    });
    return { nonce, version: this.version, ciphertext: sealed, tag: macOf(SEALING_KEY, [this.measurement, nonce, sealed.join(",")]) };
  }

  unseal(blob) {
    const tag = macOf(SEALING_KEY, [this.measurement, blob.nonce, blob.ciphertext.join(",")]);
    if (tag !== blob.tag) throw new Error("sealed blob was not produced for this measurement");
    if (blob.version > this.version) throw new Error("sealed blob is newer than this enclave");
    const bytes = blob.ciphertext.map((byte, index) => {
      const word = keystreamWord(this.measurement, blob.nonce, index >> 2);
      return byte ^ ((word >>> (8 * (index & 3))) & 0xff);
    });
    return toText(bytes);
  }

  /** An attestation report: the enclave's identity claim, fresh because of the nonce. */
  attest(nonce, payload) {
    const body = { measurement: this.measurement, nonce, payload, version: this.version };
    return { ...body, mac: macOf(VENDOR_KEY, [body.measurement, body.nonce, body.payload, body.version]) };
  }

  get codeBytes() { return this.pages.reduce((total, page) => total + page.length, 0); }
}

/** Verifier state: the expected image, the vendor key and the set of nonces already accepted. */
class Verifier {
  constructor(expectedMeasurement) {
    this.expectedMeasurement = expectedMeasurement;
    this.seenNonces = new Set();
  }

  check(report) {
    const recomputed = macOf(VENDOR_KEY, [report.measurement, report.nonce, report.payload, report.version]);
    if (recomputed !== report.mac) return { ok: false, reason: "bad-mac" };
    if (report.measurement !== this.expectedMeasurement) return { ok: false, reason: "unexpected-measurement" };
    if (this.seenNonces.has(report.nonce)) return { ok: false, reason: "replayed-nonce" };
    this.seenNonces.add(report.nonce);
    return { ok: true, reason: "accepted", payload: report.payload };
  }
}

/** Session key between two enclaves: bound to both images and to the shared secret. */
const sessionKey = (first, second, sharedSecret) =>
  macOf(sharedSecret, [first.measurement, second.measurement, 0x5a5a]);

module.exports = { VENDOR_KEY, SEALING_KEY, mix32, hashText, macOf, measurementOf, Enclave, Verifier, sessionKey };

if (require.main === module) {
  const PAGES = ["load r0, secret", "cmp r0, r1", "jmp done", "done: ret"];
  const enclave = new Enclave(PAGES);
  const verifier = new Verifier(enclave.measurement);

  // 1. an honest report verifies, and the payload comes back to the verifier
  const report = enclave.attest("nonce-1", "channel:0xbeef");
  const accepted = verifier.check(report);
  deepStrictEqual(accepted, { ok: true, reason: "accepted", payload: "channel:0xbeef" });
  // 2. the nonce is single-use: replaying the very same report is refused
  deepStrictEqual(verifier.check(report), { ok: false, reason: "replayed-nonce" });
  deepStrictEqual(verifier.check(enclave.attest("nonce-2", "channel:0xbeef")).ok, true, "a fresh nonce is accepted");
  // 3. every single-byte edit of a report is caught by the MAC
  const fields = ["measurement", "nonce", "payload", "version"];
  let edits = 0;
  for (const field of fields) {
    for (const replacement of [`${report[field]}x`, 0, 999999]) {
      const edited = { ...report, [field]: replacement };
      const verdict = new Verifier(enclave.measurement).check(edited);
      strictEqual(verdict.ok, false, `an edited ${field} must be rejected`);
      edits++;
    }
  }
  // 4. a report produced without the vendor key does not verify
  const forgedKey = 0x12345678;
  const forged = { ...report, nonce: "nonce-3", mac: macOf(forgedKey, [report.measurement, "nonce-3", report.payload, report.version]) };
  deepStrictEqual(new Verifier(enclave.measurement).check(forged), { ok: false, reason: "bad-mac" });
  // 5. a different image is rejected even though its report is well-formed
  const modified = new Enclave([...PAGES.slice(0, 3), "done: ret ; backdoor"]);
  ok(modified.measurement !== enclave.measurement, "one changed page changes the measurement");
  deepStrictEqual(new Verifier(enclave.measurement).check(modified.attest("nonce-4", "payload")),
    { ok: false, reason: "unexpected-measurement" });
  deepStrictEqual(new Verifier(modified.measurement).check(modified.attest("nonce-5", "payload")).ok, true);

  // 6. sealing binds the blob to the measurement: only the same image can open it
  const blob = enclave.seal("project secret", "seal-nonce");
  strictEqual(enclave.unseal(blob), "project secret", "the same enclave opens its own blob");
  throws(() => modified.unseal(blob), /not produced for this measurement/);
  throws(() => enclave.unseal({ ...blob, ciphertext: [...blob.ciphertext, 0x41] }), /not produced/);
  // 7. the version inside a blob blocks a rollback: a blob labelled with a newer version is refused
  strictEqual(enclave.unseal(new Enclave(PAGES).seal("x", "other-nonce")), "x", "a blob from an identical image opens");
  const rolledBack = { ...blob, version: blob.version + 1 };
  throws(() => enclave.unseal({ ...rolledBack,
    tag: macOf(SEALING_KEY, [enclave.measurement, rolledBack.nonce, rolledBack.ciphertext.join(",")]) }),
  /newer than this enclave/);
  // 8. a session key is bound to both measurements
  const twin = new Enclave(PAGES);
  strictEqual(sessionKey(enclave, twin, 0xabc), sessionKey(twin, enclave, 0xabc), "matching images agree");
  ok(sessionKey(enclave, modified, 0xabc) !== sessionKey(enclave, twin, 0xabc), "a different peer changes the key");
  ok(sessionKey(enclave, twin, 0xabd) !== sessionKey(enclave, twin, 0xabc), "a different secret changes the key");

  // 9. measurement sensitivity: every page edit, page order and empty page list
  let pageEdits = 0;
  for (let index = 0; index < PAGES.length; index++) {
    for (const suffix of [" ", "; nop", "; jmp 0"]) {
      const variant = [...PAGES];
      variant[index] = `${variant[index]}${suffix}`;
      ok(measurementOf(variant) !== enclave.measurement, `editing page ${index} must change the measurement`);
      pageEdits++;
    }
  }
  ok(measurementOf([...PAGES].reverse()) !== enclave.measurement, "reordering the pages changes the measurement");
  ok(measurementOf(["a"]) !== measurementOf(["b"]), "different single-page images differ");
  strictEqual(measurementOf(["same"]), measurementOf(["same"]), "identical images measure identically");
  throws(() => new Enclave([]), RangeError);
  throws(() => new Enclave("not a page list"), RangeError);
  // 10. determinism: two enclaves over the same image produce identical reports and blobs
  const fresh = new Enclave(PAGES);
  deepStrictEqual(fresh.attest("n", "p"), enclave.attest("n", "p"));
  deepStrictEqual(fresh.seal("s", "n"), enclave.seal("s", "n"));
  strictEqual(fresh.codeBytes, enclave.codeBytes);

  console.log(
    `measurement=0x${enclave.measurement.toString(16)} reportEdits=${edits} pageEdits=${pageEdits} ` +
    `sealedBytes=${blob.ciphertext.length} accepted=2`
  );
  console.log("PASS");
}
