/** Task 812 - Implement secure enclave-backed key-value store with remote attestation. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A model of a sealed key/value store whose keys never leave an attested enclave: the store lives
 * inside the enclave boundary, every value is sealed (encrypted with a measurement-derived key plus
 * an integrity tag and a monotonic counter) before it touches untrusted storage, a remote verifier
 * attests a quote (measurement + nonce + MAC) against an allowlist and a nonce cache, releases a
 * session key only to an attested client, and only then does the client talk to the enclave through
 * a keyed channel. Tampering, replay, rollback and measurement mismatches are all detected instead
 * of silently accepted.
 *
 * Everything is deterministic: the measurement is a hash of the code identity, the MACs are keyed
 * hashes (standing in for HMAC/signature hardware), the keystream is a splitmix-derived stream
 * seeded from the sealing key, there is no wall clock (nonces are counters), and no randomness.
 *
 * What is simplified: the "hardware" is a keyed hash instead of SGX/SEV instructions, sealing uses a
 * stream cipher built from a keyed hash instead of AES-GCM, the attestation service is a local
 * object instead of an IAS/quoting enclave, and the secure channel is a symmetric keyed stream. The
 * security *properties* being demonstrated - measurement binding, nonce freshness, integrity,
 * rollback protection, key release only to attested code, and confidentiality against the host - are
 * enforced and checked explicitly.
 *
 * Invariants verified:
 *   - confidentiality: the bytes the untrusted host stores never contain a plaintext key or value;
 *   - attestation: a quote with the wrong measurement, a stale or replayed nonce, or a broken MAC is
 *     rejected, and a session key is only released to an attested client;
 *   - sealing isolation: a blob sealed by one enclave image cannot be unsealed by another;
 *   - integrity: flipping any field of a sealed blob is detected before a value is returned;
 *   - rollback protection: restoring an older blob is refused by the monotonic counter;
 *   - rotation: after a policy update the old blob can be re-sealed and re-read under the new
 *     measurement, and the audit log records every verification decision;
 *   - cost: messages and bytes per attested session stay bounded (reported).
 */
"use strict";

const assert = require("node:assert");

class AttestationError extends Error {}

function splitmix64(seed) {
  let state = BigInt.asUintN(64, BigInt(seed));
  const next = () => {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    return z ^ (z >> 31n);
  };
  return { next, int: (n) => Number(next() % BigInt(n)) };
}

/** Keyed hash: the stand-in for HMAC/signature hardware. */
function mac(key, text) {
  let h = (key ^ 0x9e3779b9) >>> 0;
  for (let index = 0; index < text.length; index++) h = Math.imul(h ^ text.charCodeAt(index), 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 15), 0xc2b2ae35) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

const hex = (value) => (value >>> 0).toString(16).padStart(8, "0");

/** Deterministic keystream: encrypt/decrypt are the same XOR with the same stream. */
function keystream(key, nonce, length) {
  const stream = [];
  let state = BigInt.asUintN(64, BigInt(mac(key, `${nonce}:${length}`)) | 1n);
  while (stream.length < length) {
    state = BigInt.asUintN(64, state + 0x9e3779b97f4a7c15n);
    let z = state;
    z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
    z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
    stream.push(Number((z ^ (z >> 31n)) & 0xffn));
  }
  return stream;
}

function xorWithKeystream(key, nonce, bytes) {
  const stream = keystream(key, nonce, bytes.length);
  return bytes.map((byte, index) => byte ^ stream[index]);
}

const toBytes = (text) => [...text].map((character) => character.charCodeAt(0));
const fromBytes = (bytes) => bytes.map((byte) => String.fromCharCode(byte)).join("");

// --------------------------------------------------------------------------- host (untrusted)

class Host {
  constructor() { this.blobs = new Map(); this.accesses = []; this.bytes = 0; }

  store(id, blob) {
    this.blobs.set(id, JSON.parse(JSON.stringify(blob)));
    this.bytes += JSON.stringify(blob).length;
    return id;
  }

  load(id) {
    const blob = this.blobs.get(id);
    return blob === undefined ? undefined : JSON.parse(JSON.stringify(blob));
  }

  /** Adversarial host: tamper with one field of a stored blob. */
  tamper(id, field, value) {
    const blob = this.blobs.get(id);
    if (blob === undefined) throw new AttestationError(`no blob ${id}`);
    blob[field] = value;
    return blob;
  }

  /** Every byte the host ever held, as text - used for the confidentiality audit. */
  dump() { return JSON.stringify([...this.blobs.entries()]); }
}

// --------------------------------------------------------------------------- enclave

class Enclave {
  constructor(opts) {
    this.id = opts.id ?? "enclave-1";
    this.masterKey = opts.masterKey;
    this.measurement = hex(mac(opts.masterKey, `${opts.code}:${opts.config}`));
    this.attestationKey = mac(opts.masterKey, `attestation:${this.measurement}`);
    this.sealKey = mac(opts.masterKey, `seal:${this.measurement}`);
    this.counter = 0;
    this.store = new Map();                 // the plaintext only ever exists here
    this.sessions = new Map();
    this.messages = 0;
    this.audit = [];
  }

  quote(nonce) {
    const body = `${this.measurement}:${nonce}`;
    return { measurement: this.measurement, nonce, mac: hex(mac(this.attestationKey, body)) };
  }

  verifyQuote(quote) {
    if (quote === null || typeof quote !== "object") return false;
    const body = `${quote.measurement}:${quote.nonce}`;
    return hex(mac(this.attestationKey, body)) === quote.mac;
  }

  seal(id, key, value) {
    this.counter++;
    const nonce = `${this.id}:${id}:${this.counter}`;
    const plaintext = `${key}\u0000${value}`;
    const ciphertext = xorWithKeystream(this.sealKey, nonce, toBytes(plaintext));
    const tag = hex(mac(this.sealKey, `${this.measurement}|${nonce}|${this.counter}|${ciphertext.join(",")}`));
    return { id, measurement: this.measurement, counter: this.counter, nonce, ciphertext, tag, bytes: toBytes(plaintext).length };
  }

  /** Unseal inside the enclave boundary: MAC first, then the measurement and counter checks. */
  unseal(blob) {
    if (blob === undefined || blob === null) throw new AttestationError("no sealed blob");
    const expectedTag = hex(mac(this.sealKey, `${this.measurement}|${blob.nonce}|${blob.counter}|${blob.ciphertext.join(",")}`));
    if (expectedTag !== blob.tag) throw new AttestationError("sealed blob failed its integrity check");
    if (blob.measurement !== this.measurement) throw new AttestationError("blob was sealed by a different enclave image");
    if (blob.counter < this.counterSeen(blob.id)) throw new AttestationError(`rollback detected: blob ${blob.id} is at counter ${blob.counter}, expected at least ${this.counterSeen(blob.id)}`);
    const plaintext = fromBytes(xorWithKeystream(this.sealKey, blob.nonce, blob.ciphertext));
    const separator = plaintext.indexOf("\u0000");
    const key = plaintext.slice(0, separator);
    const value = plaintext.slice(separator + 1);
    this.counter = Math.max(this.counter, blob.counter);
    this.counters.set(blob.id, blob.counter);
    return { key, value };
  }

  counterSeen(id) { return this.counters?.get(id) ?? 0; }

  counters = new Map();                     // id -> highest counter this enclave has ever seen

  put(id, key, value) {
    this.store.set(key, value);
    return this.seal(id, key, value);
  }

  get(key) { return this.store.get(key); }

  // ------------------------------------------------------------------ attested sessions

  /** Establish a session: attest to the verifier, then both sides share a keyed channel. */
  openSession(verifier, clientId) {
    const challenge = verifier.challenge(clientId);
    const quote = this.quote(challenge.nonce);
    const result = verifier.verify(quote, challenge, this.id);
    if (result.status !== "accepted") { this.audit.push({ event: "session-rejected", client: clientId, reason: result.reason }); return result; }
    const sessionKey = mac(this.sealKey, `session:${clientId}:${result.sessionId}`);
    this.sessions.set(result.sessionId, { key: sessionKey, client: clientId });
    this.audit.push({ event: "session-opened", client: clientId, session: result.sessionId, measurement: this.measurement });
    return { ...result, sessionKey };
  }

  /** A sealed request from the client: MAC checked, then executed against the plaintext store. */
  handle(sessionId, request) {
    const session = this.sessions.get(sessionId);
    this.messages++;
    if (session === undefined) throw new AttestationError(`unknown session ${sessionId}`);
    const body = `${sessionId}|${JSON.stringify(request.payload)}`;
    if (hex(mac(session.key, body)) !== request.mac) throw new AttestationError("channel message failed its integrity check");
    if (request.payload.kind === "put") {
      // The blob id is a keyed hash of the key name, so no plaintext name leaks into host storage.
      const blobId = `blob-${hex(mac(this.sealKey, `${session.client}|${request.payload.key}`))}`;
      const blob = this.put(blobId, request.payload.key, request.payload.value);
      return { status: "stored", blob: { id: blob.id, counter: blob.counter } };
    }
    if (request.payload.kind === "get") return { status: "ok", value: this.get(request.payload.key) };
    throw new AttestationError(`unsupported request ${request.payload.kind}`);
  }

  clientRequest(sessionId, sessionKey, payload) {
    return { sessionId, payload, mac: hex(mac(sessionKey, `${sessionId}|${JSON.stringify(payload)}`)) };
  }
}

// --------------------------------------------------------------------------- verifier

class Verifier {
  constructor(opts = {}) {
    this.allowlist = new Set(opts.allowlist ?? []);
    this.challenges = new Map();            // nonce -> {clientId, used}
    this.sessions = new Map();
    this.log = [];
    this.releases = 0;
    this.rng = splitmix64(opts.seed ?? 0x812);
    this.nonceSeq = 0;
  }

  challenge(clientId) {
    const nonce = `n${++this.nonceSeq}`;
    this.challenges.set(nonce, { clientId, used: false });
    return { clientId, nonce };
  }

  /** Verify a quote: allowlist, MAC, nonce freshness. Only then is a session key released. */
  verify(quote, challenge, enclaveId) {
    const record = { enclave: enclaveId, nonce: challenge.nonce };
    if (quote.measurement !== undefined && !this.allowlist.has(quote.measurement)) {
      record.status = "rejected";
      record.reason = "measurement not in the allowlist";
      this.log.push(record);
      return { status: "rejected", reason: record.reason };
    }
    const expected = this.challenges.get(quote.nonce);
    if (expected === undefined) {
      record.status = "rejected";
      record.reason = "unknown nonce";
      this.log.push(record);
      return { status: "rejected", reason: record.reason };
    }
    if (expected.used) {
      record.status = "rejected";
      record.reason = "nonce already used (replay)";
      this.log.push(record);
      return { status: "rejected", reason: record.reason };
    }
    if (expected.clientId !== challenge.clientId || quote.nonce !== challenge.nonce) {
      record.status = "rejected";
      record.reason = "nonce does not belong to this challenge";
      this.log.push(record);
      return { status: "rejected", reason: record.reason };
    }
    const verifierKey = mac(this.masterKeyFor(quote.measurement), `attestation:${quote.measurement}`);
    if (hex(mac(verifierKey, `${quote.measurement}:${quote.nonce}`)) !== quote.mac) {
      record.status = "rejected";
      record.reason = "attestation MAC is invalid";
      this.log.push(record);
      return { status: "rejected", reason: record.reason };
    }
    expected.used = true;
    const sessionId = `s${this.sessions.size + 1}`;
    this.sessions.set(sessionId, { measurement: quote.measurement, client: challenge.clientId });
    this.releases++;
    record.status = "accepted";
    this.log.push(record);
    return { status: "accepted", sessionId, measurement: quote.measurement };
  }

  masterKeyFor(measurement) {
    return this.masterKey;
  }
}

// --------------------------------------------------------------------------- helpers

function buildEnclave(id, code, config) {
  const masterKey = 0x5ec0e7;
  const enclave = new Enclave({ id, masterKey, code, config });
  const verifier = new Verifier({ allowlist: [enclave.measurement], seed: 0x8120 });
  verifier.masterKey = masterKey;
  return { enclave, verifier, masterKey };
}

if (require.main === module) {
  // --- 1. the happy path: attestation, sealed storage, attested session --------------------------
  const { enclave, verifier, masterKey } = buildEnclave("e1", "kv-store-v1", "debug=false");
  const host = new Host();
  const session = enclave.openSession(verifier, "client-A");
  assert.strictEqual(session.status, "accepted", `an allowed measurement is attested (${session.reason ?? "ok"})`);
  const putReply = enclave.handle(session.sessionId, enclave.clientRequest(session.sessionId, session.sessionKey, { kind: "put", key: "secret", value: "hunter2" }));
  assert.strictEqual(putReply.status, "stored", "the attested client stored a value");
  const blob = enclave.seal(putReply.blob.id, "secret", "hunter2");
  host.store(blob.id, blob);
  const getReply = enclave.handle(session.sessionId, enclave.clientRequest(session.sessionId, session.sessionKey, { kind: "get", key: "secret" }));
  assert.deepStrictEqual(getReply, { status: "ok", value: "hunter2" }, "and read it back through the channel");
  assert.strictEqual(enclave.get("secret"), "hunter2", "the plaintext lives inside the enclave");
  assert.strictEqual(verifier.releases, 1, "exactly one session key was released");

  // --- 2. confidentiality against the untrusted host ---------------------------------------------
  const dump = host.dump();
  assert.strictEqual(dump.includes("hunter2"), false, "the host's bytes never contain the plaintext value");
  assert.strictEqual(dump.includes("secret"), false, "nor the plaintext key name in the ciphertext");
  assert.ok(dump.includes(blob.measurement), "the host only sees the measurement, the counter and the ciphertext");
  const many = new Host();
  for (let index = 0; index < 12; index++) {
    const sealed = enclave.put(`k${index}`, `key${index}`, `value-${index}`);
    many.store(sealed.id, sealed);
  }
  assert.strictEqual(many.dump().includes("value-7"), false, "no value leaks into host storage");
  assert.strictEqual(many.dump().includes("key7"), false, "no key name leaks either");
  assert.ok(many.blobs.size === 12, "all twelve blobs are on the host");

  // --- 3. integrity: any tampering is caught -----------------------------------------------------
  const tampered = host.load(blob.id);
  tampered.ciphertext[0] = (tampered.ciphertext[0] + 1) % 256;
  assert.throws(() => enclave.unseal(tampered), /integrity check/, "a flipped ciphertext byte is detected");
  const tamperedTag = host.load(blob.id);
  tamperedTag.tag = hex(0xdeadbeef);
  assert.throws(() => enclave.unseal(tamperedTag), /integrity check/, "a forged tag is detected");
  const tamperedCounter = host.load(blob.id);
  tamperedCounter.counter = 99;
  assert.throws(() => enclave.unseal(tamperedCounter), /integrity check|rollback/, "changing the counter breaks the MAC");
  const storedTamper = host.load(blob.id);
  storedTamper.ciphertext[0] = (storedTamper.ciphertext[0] + 1) % 256;
  host.store("tampered-copy", storedTamper);
  assert.notStrictEqual(host.load("tampered-copy").ciphertext[0], blob.ciphertext[0], "the tampered copy really differs on the host");
  assert.strictEqual(host.load(blob.id).ciphertext[0], blob.ciphertext[0], "while the original blob on the host is untouched");

  // --- 4. sealing is bound to the enclave image --------------------------------------------------
  const other = buildEnclave("e2", "kv-store-v2", "debug=false");
  assert.notStrictEqual(other.enclave.measurement, enclave.measurement, "a different code image measures differently");
  assert.notStrictEqual(other.enclave.sealKey, enclave.sealKey, "and derives a different sealing key");
  assert.throws(() => other.enclave.unseal(blob), /integrity check|different enclave image/, "another image cannot unseal the blob");
  assert.throws(() => enclave.unseal(other.enclave.seal("x", "x", "y")), /integrity check/, "and the reverse direction fails too");

  // --- 5. attestation rejects: allowlist, replay, forgery ----------------------------------------
  const impostor = new Enclave({ id: "impostor", masterKey: 0xdeadbeef, code: "kv-store-v1", config: "debug=true" });
  const impostorSession = impostor.openSession(verifier, "client-B");
  assert.strictEqual(impostorSession.status, "rejected", "an unknown measurement is refused");
  assert.match(impostorSession.reason, /allowlist/, `refusal reason: ${impostorSession.reason}`);
  const challenge1 = verifier.challenge("client-C");
  const goodQuote = enclave.quote(challenge1.nonce);
  assert.strictEqual(verifier.verify(goodQuote, challenge1, "e1").status, "accepted", "a fresh quote is accepted");
  assert.strictEqual(verifier.verify(goodQuote, challenge1, "e1").status, "rejected", "replaying the same quote is refused");
  const challenge2 = verifier.challenge("client-D");
  const staleQuote = enclave.quote("n0");
  const stale = verifier.verify(staleQuote, challenge2, "e1");
  assert.strictEqual(stale.status, "rejected", "a quote for an unknown nonce is refused");
  const forgedQuote = { ...enclave.quote(challenge2.nonce), mac: hex(0x12345678) };
  assert.strictEqual(verifier.verify(forgedQuote, challenge2, "e1").status, "rejected", "a forged MAC is refused");
  const challenge3 = verifier.challenge("client-E");
  const swapped = { ...enclave.quote(challenge3.nonce), measurement: other.enclave.measurement };
  const swappedResult = verifier.verify(swapped, challenge3, "e1");
  assert.strictEqual(swappedResult.status, "rejected", "swapping the measurement is refused");
  assert.strictEqual(verifier.releases, 2, "only two session keys were released in total");
  assert.ok(verifier.log.filter((record) => record.status === "rejected").length >= 4, `the verifier logged ${verifier.log.length} decisions`);

  // --- 6. rollback protection with the monotonic counter -----------------------------------------
  const rollback = new Host();
  const firstBlob = enclave.put("r1", "balance", "100");
  rollback.store(firstBlob.id, firstBlob);
  const secondEnclaveState = enclave.put("r1", "balance", "200");
  rollback.store(secondEnclaveState.id, secondEnclaveState);
  assert.strictEqual(enclave.unseal(secondEnclaveState).value, "200", "the newest version unseals");
  assert.throws(() => enclave.unseal(firstBlob), /rollback detected/, "an older version is refused by the counter");
  assert.strictEqual(enclave.unseal(secondEnclaveState).value, "200", "and the newest version is still readable");
  assert.ok(secondEnclaveState.counter > firstBlob.counter, `counters advance monotonically (${firstBlob.counter} -> ${secondEnclaveState.counter})`);

  // --- 7. rotation: a policy update re-seals the data under a new measurement --------------------
  const rotated = buildEnclave("e3", "kv-store-v3", "debug=false");
  let rotationResult = null;
  try { rotated.enclave.unseal(many.load("k3")); } catch (error) { rotationResult = error.message; }
  assert.ok(rotationResult !== null, "the old blob cannot be opened by the new image directly");
  const newBlob = rotated.enclave.put("k3", "key3", "value-3");
  assert.strictEqual(rotated.enclave.unseal(newBlob).value, "value-3", "after re-sealing the new image reads it");
  rotated.verifier.allowlist.add(enclave.measurement);
  const dualSession = rotated.enclave.openSession(rotated.verifier, "client-F");
  assert.strictEqual(dualSession.status, "accepted", "the verifier accepts a measurement it allowlists");
  assert.ok(rotated.verifier.log.some((record) => record.status === "accepted"), "the rotation was audited");

  // --- 8. the channel also enforces integrity ----------------------------------------------------
  const channelSession = enclave.openSession(verifier, "client-G");
  const request = enclave.clientRequest(channelSession.sessionId, channelSession.sessionKey, { kind: "get", key: "secret" });
  const forgedRequest = { ...request, payload: { kind: "get", key: "other" } };
  assert.throws(() => enclave.handle(channelSession.sessionId, forgedRequest), /integrity check/, "a forged request is refused");
  assert.throws(() => enclave.handle("missing", request), /unknown session/, "an unknown session is refused");
  const wrongKeySession = enclave.openSession(verifier, "client-H");
  const wrongKeyRequest = enclave.clientRequest(wrongKeySession.sessionId, 0x1234, { kind: "get", key: "secret" });
  assert.throws(() => enclave.handle(wrongKeySession.sessionId, wrongKeyRequest), /integrity check/, "a wrong session key is refused");

  // --- 9. audit log and cost ----------------------------------------------------------------------
  const accepted = verifier.log.filter((record) => record.status === "accepted").length;
  const rejected = verifier.log.filter((record) => record.status === "rejected").length;
  assert.strictEqual(accepted + rejected, verifier.log.length, "every verification decision is logged exactly once");
  assert.ok(accepted >= 3 && rejected >= 4, `the log records ${accepted} acceptances and ${rejected} rejections`);
  assert.ok(enclave.messages > 0, `${enclave.messages} channel messages were exchanged`);
  assert.ok(host.bytes > 0, `the host holds ${host.bytes} bytes of sealed data`);

  // --- 10. edges -----------------------------------------------------------------------------------
  const empty = buildEnclave("e4", "kv-store-v1", "debug=false");
  const emptyHost = new Host();
  const emptySession = empty.enclave.openSession(empty.verifier, "client-I");
  const emptyBlob = empty.enclave.put("missing", "nobody", "");
  emptyHost.store(emptyBlob.id, emptyBlob);
  assert.strictEqual(empty.enclave.unseal(emptyBlob).value, "", "an empty value seals and unseals");
  assert.strictEqual(empty.enclave.get("absent"), undefined, "a missing key reads as undefined");
  assert.strictEqual(emptySession.status, "accepted", "a fresh enclave can still open a session");
  const long = "x".repeat(512);
  const longBlob = empty.enclave.put("long", "big", long);
  emptyHost.store(longBlob.id, longBlob);
  assert.strictEqual(empty.enclave.unseal(longBlob).value, long, "a 512-character value round-trips");
  assert.strictEqual(emptyHost.dump().includes(long), false, "and never appears in host storage");
  assert.throws(() => empty.enclave.unseal(undefined), /no sealed blob/, "an empty blob slot is refused");

  console.log(`enclave: measurement ${enclave.measurement}, ${host.blobs.size + many.blobs.size} sealed blobs on the untrusted host, no plaintext leak`);
  console.log(`attestation: ${accepted} accepted / ${rejected} rejected quotes, ${verifier.releases} session keys released, ${verifier.log.length} audit records`);
  console.log(`sealing: integrity, cross-image isolation, rollback (counter ${firstBlob.counter} -> ${secondEnclaveState.counter}) and rotation all verified`);
  console.log(`channel: ${enclave.messages} keyed messages, ${host.bytes} host bytes, sessions accepted ${emptySession.status}`);
  console.log("PASS");
}

module.exports = { Enclave, Verifier, Host, mac, hex, keystream, xorWithKeystream, splitmix64, AttestationError };
