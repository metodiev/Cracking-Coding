/** Task 841 - Implement secure enclaves chain-of-trust verification demo.
 *  (Tier: Master, CTCI Ch 9 — System Design and Scalability)
 *
 * Remote attestation for secure enclaves: a verifier must decide whether to trust a quote produced
 * inside an enclave, and it must be able to say *why* it refuses. The demo builds the whole
 * chain-of-trust path and checks every link:
 *
 *   pinned root anchor -> platform/intermediate CA(s) -> enclave signing key -> quote
 *
 *  - every certificate is signed by its parent's key over a *canonical* serialisation of the body
 *    (sorted keys, big integers in decimal) and carries a role, a key-usage set, an explicit logical
 *    validity window, an SVN and — for enclave certificates — the measurement (MRENCLAVE) it attests;
 *  - the quote is signed by the enclave key and carries MRENCLAVE, MRSIGNER, SVN, the debug attribute
 *    and report data that binds the verifier's nonce *and* the session key, so a captured quote can
 *    never be replayed into another session;
 *  - the verifier runs an ordered check list and returns a verdict with machine-readable reason codes:
 *    anchor pinning, link signatures, issuer names, parent key usage, validity windows, revocation
 *    list, leaf role/usage, quote signature, quote/certificate identity binding, measurement
 *    allowlist, TCB (SVN) level, debug attribute and freshness/replay.
 *
 * The tests are mutation tests: every single-field mutation of an otherwise valid report must be
 * rejected with *exactly* the expected reason set, and a second, independently written flat oracle
 * (plain booleans, no shared control flow) must produce the identical reason set.
 *
 * Simplifications, stated honestly: RSA is real RSA — modexp, PKCS#1 v1.5 style padding, and the
 * verifier recovers the padded block and validates its structure — but only 384 bits and generated
 * from fixed seeds, so it is factorable and gives no real security. SHA-256 comes from node:crypto.
 * There is no hardware, no CPU TCB database and no X.509 parser: SVN thresholds and revocation lists
 * are explicit policy inputs. Time is an explicit logical integer (never the wall clock), randomness
 * is seeded splitmix64, and there are no timers, threads or sockets.
 *
 * Invariants verified below:
 *  - safety: a 3-level and a 6-level chain are accepted, and each of 17 single-field mutations is
 *    rejected with exactly its own reason code (tampered measurement, stale SVN, out-of-date TCB with
 *    strict policy, revoked CA key, expired and not-yet-valid certificates, wrong anchor, non-root
 *    anchor, wrong issuer name, parent that may not sign, debug enclave, broken quote signature,
 *    broken link, key-id mismatch, tampered report data, disallowed measurement, truncated chain);
 *  - liveness/binding: the quote only verifies for the session key it names, nonce replay is rejected
 *    while a fresh nonce is accepted;
 *  - unforgeability measured: 200 random blocks and 200 signatures lifted from another key are all
 *    rejected (0 accepted);
 *  - cost: signature verifications per verdict and modular multiplications per verification counted;
 *  - determinism: the same scenario replays to the identical verdict.
 */
"use strict";

const crypto = require("node:crypto");

const SHA256_BYTES = 32;
const RSA_BITS = 384; // toy size: factorable, see the header
const BLOCK_BYTES = RSA_BITS / 8;
const FILL_BYTES = BLOCK_BYTES - SHA256_BYTES - 2; // 0x01 || 0xFF * FILL || 0x00 || digest
const MASK64 = (1n << 64n) - 1n;
const MILLER_RABIN_BASES = [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n];
let MULTIPLICATIONS = 0; // latency proxy: modular multiplications inside the exponentiations

/** Deterministic splitmix64: the only source of pseudo-randomness. */
class SplitMix64 {
  constructor(seed) {
    this.state = (BigInt(seed) & MASK64) || 1n;
  }

  next() {
    this.state = (this.state + 0x9e3779b97f4a7c15n) & MASK64;
    let z = this.state;
    z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
    z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
    return (z ^ (z >> 31n)) & MASK64;
  }

  /** Uniform integer in [0, 2^width). */
  bits(width) {
    const w = BigInt(width);
    let value = 0n;
    for (let filled = 0n; filled < w; filled += 64n) value = (value << 64n) | this.next();
    return value & ((1n << w) - 1n);
  }
}

function powMod(base, exponent, modulus) {
  let acc = 1n;
  let b = base % modulus;
  let e = exponent;
  while (e > 0n) {
    if ((e & 1n) === 1n) {
      acc = (acc * b) % modulus;
      MULTIPLICATIONS++;
    }
    b = (b * b) % modulus;
    MULTIPLICATIONS++;
    e >>= 1n;
  }
  return acc;
}

/** Modular inverse via Fermat; the modulus must be prime (used for field arithmetic in tests). */
const modInverse = (a, m) => powMod(((a % m) + m) % m, m - 2n, m);

/** Modular inverse via the extended Euclidean algorithm; works for any coprime pair (RSA's phi). */
function modInverseEuclid(a, m) {
  let [oldR, r] = [((a % m) + m) % m, m];
  let [oldS, s] = [1n, 0n];
  while (r !== 0n) {
    const quotient = oldR / r;
    [oldR, r] = [r, oldR - quotient * r];
    [oldS, s] = [s, oldS - quotient * s];
  }
  if (oldR !== 1n) throw new Error("not invertible");
  return ((oldS % m) + m) % m;
}

/** Miller-Rabin with the fixed base set; deterministic well past the 192-bit primes used here. */
function isProbablePrime(candidate) {
  if (candidate < 2n) return false;
  for (const base of MILLER_RABIN_BASES) {
    if (candidate === base) return true;
    if (candidate % base === 0n) return false;
  }
  let d = candidate - 1n;
  let s = 0;
  while (d % 2n === 0n) {
    d /= 2n;
    s++;
  }
  for (const base of MILLER_RABIN_BASES) {
    let x = powMod(base, d, candidate);
    if (x === 1n || x === candidate - 1n) continue;
    let witness = true;
    for (let i = 1; i < s; i++) {
      x = (x * x) % candidate;
      if (x === candidate - 1n) {
        witness = false;
        break;
      }
    }
    if (witness) return false;
  }
  return true;
}

/** Random prime with the top two bits set, so the modulus reaches the full bit length. */
function generatePrime(bits, rng) {
  const half = BigInt(bits / 2);
  const high = (1n << (half - 1n)) | (1n << (half - 2n));
  for (let attempt = 0; attempt < 20000; attempt++) {
    const candidate = high | (rng.bits(bits / 2) | 1n);
    if (isProbablePrime(candidate)) return candidate;
  }
  throw new Error("no prime found");
}

const hex = (value) => value.toString(16);
const sha256Hex = (text) => crypto.createHash("sha256").update(text).digest("hex");

/** Deterministic serialisation used for signing: sorted keys, big integers in decimal. */
function canonical(value) {
  if (value === null || value === undefined) return "null";
  if (typeof value === "bigint") return value.toString(10);
  if (typeof value === "number" || typeof value === "boolean") return JSON.stringify(value);
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  const keys = Object.keys(value).sort();
  return `{${keys.map((k) => `${JSON.stringify(k)}:${canonical(value[k])}`).join(",")}}`;
}

/** Toy RSA key pair with real structure: e = 65537 and d = e^-1 mod phi(n). */
class RsaKey {
  constructor(seed, bits = RSA_BITS) {
    const rng = new SplitMix64(seed);
    for (;;) {
      const p = generatePrime(bits, rng);
      const q = generatePrime(bits, rng);
      if (p === q) continue;
      const phi = (p - 1n) * (q - 1n);
      if (phi % 65537n === 0n) continue;
      this.n = p * q;
      this.e = 65537n;
      this.d = modInverseEuclid(this.e, phi);
      this.bits = bits;
      this.keyId = sha256Hex(hex(this.n)).slice(0, 16);
      return;
    }
  }

  /** PKCS#1 v1.5 style padded block: 0x01 || 0xFF.. || 0x00 || SHA-256(message). */
  paddedBlock(message) {
    const block = `01${"ff".repeat(FILL_BYTES)}00${sha256Hex(message)}`;
    if (block.length !== BLOCK_BYTES * 2) throw new Error("padding size mismatch");
    return BigInt(`0x${block}`);
  }

  sign(message) {
    const block = this.paddedBlock(message);
    if (block >= this.n) throw new Error("block exceeds the modulus");
    return powMod(block, this.d, this.n);
  }

  verify(message, signature) {
    return verifyPublic(this.publicKey(), message, signature);
  }

  publicKey() {
    return { n: hex(this.n), e: hex(this.e), keyId: this.keyId };
  }
}

const publicFrom = (key) => ({ n: key.n, e: key.e, keyId: key.keyId });

/** Public keys are carried as hex strings; tolerate BigInt too so both call styles are safe. */
const asBigInt = (value) => (typeof value === "bigint" ? value : BigInt(`0x${value}`));

/** Verification with public material only: recover the padded block and validate its structure. */
function verifyPublic(publicKey, message, signature) {
  const n = asBigInt(publicKey.n);
  const e = asBigInt(publicKey.e);
  if (typeof signature !== "bigint" || signature <= 0n || signature >= n) return false;
  const recovered = powMod(signature, e, n);
  const text = hex(recovered).padStart(BLOCK_BYTES * 2, "0");
  return text === `01${"ff".repeat(FILL_BYTES)}00${sha256Hex(message)}`;
}

/** The verdict reason codes; the verifier and the flat oracle both emit only these. */
const REASONS = {
  chainTooShort: "chain-too-short",
  anchorNotRoot: "anchor-not-root",
  anchorMismatch: "anchor-mismatch",
  linkSignature: (i) => `chain-signature-invalid@${i}`,
  issuerMismatch: (i) => `issuer-mismatch@${i}`,
  parentCannotSign: (i) => `parent-cannot-sign@${i}`,
  notYetValid: (i) => `not-yet-valid@${i}`,
  expired: (i) => `expired@${i}`,
  revoked: (i) => `revoked@${i}`,
  leafNotEnclave: "leaf-not-enclave",
  leafCannotQuote: "leaf-cannot-quote",
  quoteKeyMismatch: "quote-key-mismatch",
  quoteSignature: "quote-signature-invalid",
  measurementNotBound: "measurement-not-bound",
  measurementNotAllowed: "measurement-not-allowed",
  svnBelowMinimum: "svn-below-minimum",
  debugEnclave: "debug-enclave",
  reportData: "report-data-mismatch",
  nonceReplay: "nonce-replay",
  tcbOutOfDate: "tcb-out-of-date",
};

/** Builds a chain: root -> (1 + levels) platform CAs -> enclave signing key. */
function buildChain(options = {}) {
  const seed = options.seed ?? 841;
  const now = options.now ?? 1000;
  const levels = options.levels ?? 0;
  const root = new RsaKey(seed + 1);
  const rootBody = {
    subject: "toy-root-ca",
    issuer: null,
    role: "root",
    publicKey: root.publicKey(),
    usage: ["certSign"],
    notBefore: 0,
    notAfter: 10000,
    svn: 0,
  };
  const chain = [{ body: rootBody, signature: root.sign(canonical(rootBody)) }];
  const keys = [root];
  let signer = root;
  let signerSubject = rootBody.subject;
  for (let level = 0; level < 1 + levels; level++) {
    const intermediate = new RsaKey(seed + 10 + level);
    const body = {
      subject: `toy-platform-ca-${level}`,
      issuer: signerSubject,
      role: "intermediate",
      publicKey: intermediate.publicKey(),
      usage: ["certSign"],
      notBefore: 10 + level,
      notAfter: 9000 - level,
      svn: 0,
    };
    chain.push({ body, signature: signer.sign(canonical(body)) });
    keys.push(intermediate);
    signer = intermediate;
    signerSubject = body.subject;
  }
  const enclave = new RsaKey(seed + 100);
  const enclaveBody = {
    subject: "toy-enclave-signing-key",
    issuer: signerSubject,
    role: "enclave",
    publicKey: enclave.publicKey(),
    usage: ["quoteSign"],
    notBefore: 20,
    notAfter: 8000,
    svn: options.svn ?? 7,
    measurement: options.measurement ?? "aa".repeat(32),
  };
  chain.push({ body: enclaveBody, signature: signer.sign(canonical(enclaveBody)) });
  keys.push(enclave);
  return { chain, keys, now, root, enclave, enclaveBody, rootKeyId: options.rootKeyId ?? root.keyId };
}

/** Creates the report a verifier receives: the chain, the quote, and the session's nonce/key. */
function createReport({ built, nonce, sessionKey, svn = null, debug = null }) {
  const { chain, enclave, enclaveBody, now } = built;
  const body = {
    leafKeyId: enclaveBody.publicKey.keyId,
    measurement: enclaveBody.measurement,
    signerMeasurement: sha256Hex("toy-enclave-signer"),
    reportData: sha256Hex(`${nonce}|${sessionKey}`),
    svn: svn === null ? enclaveBody.svn : svn,
    debug: debug === null ? false : debug,
    timestamp: now,
  };
  return { chain, now, nonce, sessionKey, quote: { body, signature: enclave.sign(canonical(body)) } };
}

const DEFAULT_POLICY = {
  minSvn: 5,
  tcbUpToDateSvn: 6,
  allowedMeasurements: null, // null = whatever the certificate binds
  revokedKeyIds: [],
  allowDebug: false,
  allowOutOfDate: true,
};

/**
 * The verifier. Every failed stage appends one reason code; only an empty reason list is accepted,
 * and an accepted nonce is burned so the quote cannot be replayed.
 */
class Verifier {
  constructor(anchorKeyId, policy = {}) {
    this.anchorKeyId = anchorKeyId;
    this.policy = { ...DEFAULT_POLICY, ...policy };
    this.usedNonces = new Set();
    this.verifications = 0;
    this.verdicts = 0;
  }

  verify(report) {
    const reasons = [];
    this.verdicts++;
    const chain = report.chain ?? [];
    if (chain.length < 2) {
      return { accepted: false, status: "rejected", reasons: [REASONS.chainTooShort],
        tcbStatus: "untrusted" };
    }
    const root = chain[0];
    if (root.body.role !== "root") reasons.push(REASONS.anchorNotRoot);
    if (root.body.publicKey.keyId !== this.anchorKeyId) reasons.push(REASONS.anchorMismatch);

    for (let i = 1; i < chain.length; i++) {
      const parent = publicFrom(chain[i - 1].body.publicKey);
      this.verifications++;
      if (!verifyPublic(parent, canonical(chain[i].body), chain[i].signature)) {
        reasons.push(REASONS.linkSignature(i));
      }
      if (chain[i].body.issuer !== chain[i - 1].body.subject) reasons.push(REASONS.issuerMismatch(i));
      if (!(chain[i - 1].body.usage ?? []).includes("certSign")) {
        reasons.push(REASONS.parentCannotSign(i));
      }
    }
    for (let i = 0; i < chain.length; i++) {
      const body = chain[i].body;
      if (report.now < body.notBefore) reasons.push(REASONS.notYetValid(i));
      else if (report.now > body.notAfter) reasons.push(REASONS.expired(i));
      if ((this.policy.revokedKeyIds ?? []).includes(body.publicKey.keyId)) {
        reasons.push(REASONS.revoked(i));
      }
    }

    const leaf = chain[chain.length - 1];
    if (leaf.body.role !== "enclave") reasons.push(REASONS.leafNotEnclave);
    if (!(leaf.body.usage ?? []).includes("quoteSign")) reasons.push(REASONS.leafCannotQuote);

    const quote = report.quote?.body;
    if (!quote) return { accepted: false, status: "rejected",
      reasons: reasons.concat(["missing-quote"]), tcbStatus: "untrusted" };
    if (quote.leafKeyId !== leaf.body.publicKey.keyId) reasons.push(REASONS.quoteKeyMismatch);
    this.verifications++;
    if (!verifyPublic(publicFrom(leaf.body.publicKey), canonical(report.quote.body),
      report.quote.signature)) {
      reasons.push(REASONS.quoteSignature);
    }
    if (quote.measurement !== leaf.body.measurement) reasons.push(REASONS.measurementNotBound);
    if (this.policy.allowedMeasurements &&
        !this.policy.allowedMeasurements.includes(quote.measurement)) {
      reasons.push(REASONS.measurementNotAllowed);
    }
    if (quote.svn < this.policy.minSvn) reasons.push(REASONS.svnBelowMinimum);
    if (quote.debug && !this.policy.allowDebug) reasons.push(REASONS.debugEnclave);
    if (quote.reportData !== sha256Hex(`${report.nonce}|${report.sessionKey}`)) {
      reasons.push(REASONS.reportData);
    }
    if (this.usedNonces.has(report.nonce)) reasons.push(REASONS.nonceReplay);

    let tcbStatus = "up-to-date";
    if (quote.svn < this.policy.tcbUpToDateSvn) {
      tcbStatus = "out-of-date";
      if (!this.policy.allowOutOfDate) reasons.push(REASONS.tcbOutOfDate);
    }
    const accepted = reasons.length === 0;
    if (accepted) this.usedNonces.add(report.nonce);
    return { accepted, status: accepted ? "accepted" : "rejected", reasons,
      tcbStatus: accepted ? tcbStatus : "untrusted" };
  }
}

/**
 * Independently written flat oracle: plain booleans, no shared control flow with the verifier, and
 * the identical reason-code vocabulary. Any disagreement is a bug in one of the two.
 */
function oracleCheck(report, anchorKeyId, policy = {}, usedNonces = new Set()) {
  const p = { ...DEFAULT_POLICY, ...policy };
  const failures = [];
  const chain = report.chain ?? [];
  if (chain.length < 2) return [REASONS.chainTooShort];
  if (chain[0].body.role !== "root") failures.push(REASONS.anchorNotRoot);
  if (chain[0].body.publicKey.keyId !== anchorKeyId) failures.push(REASONS.anchorMismatch);
  for (let i = 1; i < chain.length; i++) {
    const good = verifyPublic(publicFrom(chain[i - 1].body.publicKey), canonical(chain[i].body),
      chain[i].signature);
    if (!good) failures.push(REASONS.linkSignature(i));
    const sameName = chain[i].body.issuer === chain[i - 1].body.subject;
    if (!sameName) failures.push(REASONS.issuerMismatch(i));
    const maySign = (chain[i - 1].body.usage ?? []).includes("certSign");
    if (!maySign) failures.push(REASONS.parentCannotSign(i));
  }
  for (let i = 0; i < chain.length; i++) {
    const body = chain[i].body;
    if (report.now < body.notBefore) failures.push(REASONS.notYetValid(i));
    else if (report.now > body.notAfter) failures.push(REASONS.expired(i));
    if ((p.revokedKeyIds ?? []).includes(body.publicKey.keyId)) failures.push(REASONS.revoked(i));
  }
  const leaf = chain[chain.length - 1];
  if (leaf.body.role !== "enclave") failures.push(REASONS.leafNotEnclave);
  if (!(leaf.body.usage ?? []).includes("quoteSign")) failures.push(REASONS.leafCannotQuote);
  const quote = report.quote?.body;
  if (!quote) return failures.concat(["missing-quote"]);
  if (quote.leafKeyId !== leaf.body.publicKey.keyId) failures.push(REASONS.quoteKeyMismatch);
  const quoteOk = verifyPublic(publicFrom(leaf.body.publicKey), canonical(report.quote.body),
    report.quote.signature);
  if (!quoteOk) failures.push(REASONS.quoteSignature);
  if (quote.measurement !== leaf.body.measurement) failures.push(REASONS.measurementNotBound);
  if (p.allowedMeasurements && !p.allowedMeasurements.includes(quote.measurement)) {
    failures.push(REASONS.measurementNotAllowed);
  }
  if (quote.svn < p.minSvn) failures.push(REASONS.svnBelowMinimum);
  if (quote.debug && !p.allowDebug) failures.push(REASONS.debugEnclave);
  if (quote.reportData !== sha256Hex(`${report.nonce}|${report.sessionKey}`)) {
    failures.push(REASONS.reportData);
  }
  if (usedNonces.has(report.nonce)) failures.push(REASONS.nonceReplay);
  if (quote.svn < p.tcbUpToDateSvn && !p.allowOutOfDate) failures.push(REASONS.tcbOutOfDate);
  return failures;
}

module.exports = { SHA256_BYTES, RSA_BITS, BLOCK_BYTES, FILL_BYTES, SplitMix64, powMod, modInverse,
  modInverseEuclid, isProbablePrime, generatePrime, hex, sha256Hex, canonical, RsaKey, publicFrom,
  verifyPublic, REASONS, buildChain, createReport, Verifier, oracleCheck, DEFAULT_POLICY,
  counter: () => MULTIPLICATIONS, resetCounter: () => { MULTIPLICATIONS = 0; } };

if (require.main === module) {
  const assert = require("node:assert");
  const V = module.exports;
  const reasonSet = (reasons) => [...new Set(reasons)].sort();

  // --- toy RSA structure -----------------------------------------------------
  const key = new V.RsaKey(8411);
  assert.ok(V.hex(key.n).length >= V.RSA_BITS / 4 - 1, `modulus is near ${V.RSA_BITS} bits`);
  assert.strictEqual(key.e, 65537n, "the public exponent is 65537");
  assert.ok(V.modInverse(7n, 101n) * 7n % 101n === 1n, "modular inverse is exact");
  assert.ok(V.isProbablePrime(97n) && !V.isProbablePrime(91n), "Miller-Rabin spot checks");
  const message = "attest me";
  const signature = key.sign(message);
  assert.strictEqual(key.verify(message, signature), true, "a fresh signature verifies");
  assert.strictEqual(key.verify(`${message} `, signature), false, "a tampered message fails");
  assert.strictEqual(key.sign(message), signature, "PKCS#1 v1.5 signing is deterministic");
  const blockHex = V.hex(key.paddedBlock(message)).padStart(V.BLOCK_BYTES * 2, "0");
  assert.strictEqual(blockHex.slice(0, 2), "01", "padding starts with 0x01");
  assert.strictEqual(blockHex.slice(2, 2 + 2 * V.FILL_BYTES), "ff".repeat(V.FILL_BYTES),
    "padding is 0xFF filled");
  assert.strictEqual(blockHex.slice(2 + 2 * V.FILL_BYTES, 4 + 2 * V.FILL_BYTES), "00",
    "a 0x00 separator precedes the digest");
  assert.strictEqual(blockHex.slice(4 + 2 * V.FILL_BYTES), V.sha256Hex(message),
    "the digest is carried verbatim");
  const otherKey = new V.RsaKey(999);
  assert.strictEqual(key.verify(message, otherKey.sign(message)), false,
    "a signature from another key does not verify");
  assert.strictEqual(V.verifyPublic(key.publicKey(), message, signature), true,
    "public-key-only verification agrees");
  assert.strictEqual(new V.RsaKey(8411).n, key.n, "key generation is deterministic");
  assert.notStrictEqual(new V.RsaKey(8412).n, key.n, "a different seed yields a different key");

  // --- unforgeability, measured ---------------------------------------------
  const forgeRng = new V.SplitMix64(4242);
  let forgedAccepted = 0;
  for (let i = 0; i < 200; i++) {
    if (key.verify(message, forgeRng.bits(V.RSA_BITS))) forgedAccepted++;
    if (key.verify(message, otherKey.sign(`other ${i}`))) forgedAccepted++;
  }
  assert.strictEqual(forgedAccepted, 0, "400 forged signatures never verify");

  // --- the honest chain is accepted in both shapes ---------------------------
  const base = V.buildChain({ seed: 841 });
  const report = V.createReport({ built: base, nonce: "nonce-a", sessionKey: "sess-1" });
  const verifier = new V.Verifier(base.rootKeyId);
  const good = verifier.verify(report);
  assert.deepStrictEqual(good.reasons, [], "the honest report is accepted");
  assert.strictEqual(good.status, "accepted");
  assert.strictEqual(good.tcbStatus, "up-to-date", "svn 7 is up to date at threshold 6");
  assert.strictEqual(base.chain.length, 3, "root -> platform CA -> enclave");
  assert.strictEqual(verifier.usedNonces.size, 1, "the accepted nonce is remembered");
  assert.deepStrictEqual(reasonSet(V.oracleCheck(report, base.rootKeyId)), [],
    "the flat oracle accepts it too");

  const deepChain = V.buildChain({ seed: 845, levels: 3 });
  const deepReport = V.createReport({ built: deepChain, nonce: "nonce-deep", sessionKey: "sess-d" });
  assert.strictEqual(deepChain.chain.length, 6, "a six-level chain is built too");
  const deepVerdict = new V.Verifier(deepChain.rootKeyId).verify(deepReport);
  assert.deepStrictEqual(deepVerdict.reasons, [], "the six-level chain is accepted");
  assert.deepStrictEqual(reasonSet(V.oracleCheck(deepReport, deepChain.rootKeyId)), [],
    "the oracle agrees on the deep chain");
  const oldTcbChain = V.buildChain({ seed: 850, svn: 5 });
  const oldTcbReport = V.createReport({ built: oldTcbChain, nonce: "n-old", sessionKey: "s" });
  const oldTcbVerdict = new V.Verifier(oldTcbChain.rootKeyId).verify(oldTcbReport);
  assert.strictEqual(oldTcbVerdict.accepted, true, "an out-of-date TCB is tolerated by default");
  assert.strictEqual(oldTcbVerdict.tcbStatus, "out-of-date", "but the verdict says so");

  // --- single-field mutations: exactly the expected reason set ---------------
  const reseal = (subject, built) => {
    for (let i = 1; i < subject.chain.length; i++) {
      subject.chain[i].signature = built.keys[i - 1].sign(V.canonical(subject.chain[i].body));
    }
    subject.quote.signature =
      built.keys[built.keys.length - 1].sign(V.canonical(subject.quote.body));
  };
  const leafIndex = () => 2;
  const mutations = [
    { label: "non-root anchor", expected: V.REASONS.anchorNotRoot,
      mutate: (r) => { r.chain[0].body.role = "intermediate"; } },
    { label: "unpinned anchor", expected: V.REASONS.anchorMismatch,
      mutate: (r) => { r.chain[0].body.publicKey.keyId = "deadbeefdeadbeef"; } },
    { label: "tampered measurement", expected: V.REASONS.measurementNotBound,
      mutate: (r) => { r.quote.body.measurement = "bb".repeat(32); } },
    { label: "disallowed measurement", policy: { allowedMeasurements: ["cc".repeat(32)] },
      expected: V.REASONS.measurementNotAllowed, mutate: () => {} },
    { label: "stale svn", expected: V.REASONS.svnBelowMinimum,
      mutate: (r) => { r.chain[leafIndex()].body.svn = 2; r.quote.body.svn = 2; } },
    { label: "strict TCB policy", policy: { allowOutOfDate: false },
      expected: V.REASONS.tcbOutOfDate,
      mutate: (r) => { r.chain[leafIndex()].body.svn = 5; r.quote.body.svn = 5; } },
    { label: "debug enclave", expected: V.REASONS.debugEnclave,
      mutate: (r) => { r.quote.body.debug = true; } },
    { label: "expired leaf certificate", expected: V.REASONS.expired(2),
      mutate: (r) => { r.now = 9000; } },
    { label: "not yet valid leaf certificate", expected: V.REASONS.notYetValid(2),
      mutate: (r) => { r.now = 15; } },
    { label: "revoked platform key",
      expected: V.REASONS.revoked(1),
      mutate: (r, built) => { r.policy = { revokedKeyIds: [built.chain[1].body.publicKey.keyId] }; } },
    { label: "wrong issuer name", expected: V.REASONS.issuerMismatch(2),
      mutate: (r) => { r.chain[leafIndex()].body.issuer = "somebody-else"; } },
    { label: "parent may not sign", expected: V.REASONS.parentCannotSign(2),
      mutate: (r) => { r.chain[1].body.usage = []; } },
    { label: "leaf is not an enclave key", expected: V.REASONS.leafNotEnclave,
      mutate: (r) => { r.chain[leafIndex()].body.role = "intermediate"; } },
    { label: "leaf may not quote", expected: V.REASONS.leafCannotQuote,
      mutate: (r) => { r.chain[leafIndex()].body.usage = ["certSign"]; } },
    { label: "quote bound to another key id", expected: V.REASONS.quoteKeyMismatch,
      mutate: (r) => { r.quote.body.leafKeyId = "0123456789abcdef"; } },
    { label: "tampered report data", expected: V.REASONS.reportData,
      mutate: (r) => { r.quote.body.reportData = "00".repeat(32); } },
    { label: "session key swapped", expected: V.REASONS.reportData,
      mutate: (r) => { r.sessionKey = "sess-2"; } },
    { label: "broken quote signature", expected: V.REASONS.quoteSignature, reseal: false,
      mutate: (r) => { r.quote.signature = r.quote.signature + 1n; } },
    { label: "broken chain link", expected: V.REASONS.linkSignature(2), reseal: false,
      mutate: (r) => { r.chain[2].signature = r.chain[2].signature + 1n; } },
    { label: "truncated chain", expected: V.REASONS.chainTooShort,
      mutate: (r) => { r.chain = [r.chain[0]]; } },
  ];
  let mutationCases = 0;
  for (const spec of mutations) {
    const built = V.buildChain({ seed: 841 });
    const subject = V.createReport({ built, nonce: `nonce-${spec.label}`, sessionKey: "sess-1" });
    spec.mutate(subject, built);
    if (spec.reseal !== false) reseal(subject, built);
    const policy = { ...(subject.policy ?? {}), ...(spec.policy ?? {}) };
    const verdict = new V.Verifier(built.rootKeyId, policy).verify(subject);
    assert.strictEqual(verdict.accepted, false, `${spec.label}: rejected`);
    assert.deepStrictEqual(reasonSet(verdict.reasons), [spec.expected],
      `${spec.label}: exactly ${spec.expected} (got ${verdict.reasons.join("|")})`);
    assert.deepStrictEqual(reasonSet(V.oracleCheck(subject, built.rootKeyId, policy)),
      [spec.expected], `${spec.label}: the flat oracle agrees`);
    mutationCases++;
  }
  assert.strictEqual(mutationCases, 20, "all 20 single-field mutations were exercised");

  // --- nonce replay and session binding -------------------------------------
  const replayBase = V.buildChain({ seed: 900 });
  const replayVerifier = new V.Verifier(replayBase.rootKeyId);
  const firstUse = V.createReport({ built: replayBase, nonce: "once", sessionKey: "sess-x" });
  assert.strictEqual(replayVerifier.verify(firstUse).accepted, true, "first use is accepted");
  const replayAttempt = V.createReport({ built: replayBase, nonce: "once", sessionKey: "sess-x" });
  replayAttempt.quote = firstUse.quote; // the attacker replays the captured quote verbatim
  const replayVerdict = replayVerifier.verify(replayAttempt);
  assert.strictEqual(replayVerdict.accepted, false, "the replayed quote is rejected");
  assert.deepStrictEqual(reasonSet(replayVerdict.reasons), [V.REASONS.nonceReplay],
    "and the only reason is the replay");
  assert.deepStrictEqual(reasonSet(V.oracleCheck(replayAttempt, replayBase.rootKeyId, {},
    replayVerifier.usedNonces)), [V.REASONS.nonceReplay], "the oracle agrees on the replay");
  const freshUse = V.createReport({ built: replayBase, nonce: "twice", sessionKey: "sess-x" });
  assert.strictEqual(replayVerifier.verify(freshUse).accepted, true, "a fresh nonce is accepted");
  const crossedSession = V.createReport({ built: replayBase, nonce: "twice", sessionKey: "sess-y" });
  crossedSession.quote = freshUse.quote;
  assert.deepStrictEqual(
    reasonSet(new V.Verifier(replayBase.rootKeyId).verify(crossedSession).reasons),
    [V.REASONS.reportData], "the same quote cannot be re-bound to another session key");

  // --- splicing a foreign root into the chain -------------------------------
  const hostChain = V.buildChain({ seed: 841 });
  const foreignChain = V.buildChain({ seed: 700 });
  const spliced = V.createReport({ built: hostChain, nonce: "splice", sessionKey: "s" });
  spliced.chain = [foreignChain.chain[0], hostChain.chain[1], hostChain.chain[2]];
  const spliceVerdict = new V.Verifier(hostChain.rootKeyId).verify(spliced);
  assert.strictEqual(spliceVerdict.accepted, false, "a spliced chain is rejected");
  assert.deepStrictEqual(reasonSet(spliceVerdict.reasons),
    reasonSet([V.REASONS.anchorMismatch, V.REASONS.linkSignature(1)]),
    "rejection names the foreign anchor and the broken link");
  assert.deepStrictEqual(reasonSet(V.oracleCheck(spliced, hostChain.rootKeyId)),
    reasonSet(spliceVerdict.reasons), "the oracle agrees on the splice");

  // --- cost ------------------------------------------------------------------
  V.resetCounter();
  const costVerifier = new V.Verifier(base.rootKeyId);
  costVerifier.verify(V.createReport({ built: base, nonce: "cost", sessionKey: "s" }));
  const verificationsPerVerdict = costVerifier.verifications;
  const verificationCost = V.counter();
  assert.strictEqual(verificationsPerVerdict, 3, "one signature check per link plus the quote");
  assert.ok(verificationCost < 5000,
    `verification stays cheap (${verificationCost} modular multiplications)`);
  V.resetCounter();
  key.sign("measure a signature");
  const signingCost = V.counter();
  assert.ok(signingCost > verificationCost / 3,
    `signing costs more than verifying (${signingCost} vs ${verificationCost})`);

  // --- determinism -----------------------------------------------------------
  const replayChain = V.buildChain({ seed: 841 });
  const replayReport = V.createReport({ built: replayChain, nonce: "nonce-a", sessionKey: "sess-1" });
  const replayGood = new V.Verifier(replayChain.rootKeyId).verify(replayReport);
  assert.deepStrictEqual(replayGood, good, "the identical scenario yields the identical verdict");
  assert.strictEqual(replayChain.enclave.n, base.enclave.n, "keys are seeded, not sampled");

  console.log(`rsa=${V.RSA_BITS} bits (toy), padding block=${V.BLOCK_BYTES} bytes, sha256 digest=${V.SHA256_BYTES} bytes`);
  console.log(`chain depths accepted= 3 and 6 levels; mutations rejected with exact reason= ${mutationCases}; forgeries accepted= ${forgedAccepted}`);
  console.log(`cost per verdict= ${verificationsPerVerdict} signature verifications, ${verificationCost} modular multiplications (signing ${signingCost})`);
  console.log("PASS");
}
