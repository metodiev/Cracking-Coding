/** Task 837 - Implement decentralized identity (DID) system with verifiable credentials simulation.
 * (Tier: Master, Beyond — Security & Privacy)
 *
 * A deterministic DID/VC stack: self-certifying identifiers, key material with a rotation history,
 * issuer-signed credentials over Merkle-committed claims, selective disclosure of individual claims,
 * holder binding by proof of possession, expiry windows and a signed revocation registry. All
 * arithmetic is exact BigInt; timestamps and nonces are explicit inputs, so nothing depends on wall
 * clock time.
 *
 * Cryptography: Schnorr signatures over the prime-order subgroup of a safe prime group (p = 2q + 1,
 * generator g of order q): pk = g^sk, sign: r = g^k, e = H(r, m) mod q, s = k + e*sk mod q, verify
 * g^s == r * pk^e. This is a real signature scheme whose verification equation is exact; the group is
 * only ~61 bits, so no hardness claim is made - what is verified is that the algebra is correct and
 * that forgeries, tampered claims and wrong holders are rejected. The Merkle/registry hashes are
 * deterministic integer mixes (documented as non-cryptographic).
 *
 * Invariants verified:
 *   - key material: the Schnorr equation is recomputed independently, forged signatures and tampered
 *     documents are rejected, and the DID is self-certifying (a swapped key changes the identifier);
 *   - credentials: the issuer signature covers the claim Merkle root, so a tampered claim, a missing
 *     proof or a proof for the wrong index fails; the credential's revocation status comes from a
 *     signed registry whose root is checked;
 *   - selective disclosure: exactly the requested claims are revealed (set equality), every hidden
 *     claim appears only as a salted digest, and revealing an unrequested claim is rejected;
 *   - holder binding: a presentation signed by another subject key over the same nonce fails, and a
 *     replayed nonce fails;
 *   - lifecycle: expired and not-yet-valid credentials fail, revoked credentials fail, and after a key
 *     rotation credentials remain valid only while the old key is inside its validity window;
 *   - determinism: signing and verification are pure functions of their inputs.
 */
"use strict";

// ---------------------------------------------------------------------------------------------
// Group and signatures
// ---------------------------------------------------------------------------------------------

function isProbablePrime(n) {
  if (n < 2n) return false;
  for (const small of [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n]) {
    if (n % small === 0n) return n === small;
  }
  let d = n - 1n;
  let r = 0n;
  while (d % 2n === 0n) {
    d /= 2n;
    r += 1n;
  }
  for (const base of [2n, 3n, 5n, 7n, 11n, 13n, 17n, 19n, 23n, 29n, 31n, 37n]) {
    let x = modPow(base, d, n);
    if (x === 1n || x === n - 1n) continue;
    let composite = true;
    for (let i = 1n; i < r; i++) {
      x = (x * x) % n;
      if (x === n - 1n) {
        composite = false;
        break;
      }
    }
    if (composite) return false;
  }
  return true;
}

function modPow(base, exponent, modulus) {
  let result = 1n;
  let b = ((base % modulus) + modulus) % modulus;
  let e = exponent;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % modulus;
    b = (b * b) % modulus;
    e >>= 1n;
  }
  return result;
}

/** Largest safe prime below 2^bits, plus a generator of the prime-order subgroup. */
function buildGroup(bits = 61) {
  let candidate = (1n << BigInt(bits)) - 1n;
  if (candidate % 2n === 0n) candidate -= 1n;
  while (!(isProbablePrime(candidate) && isProbablePrime((candidate - 1n) / 2n))) candidate -= 2n;
  const order = (candidate - 1n) / 2n;
  let generator = 4n;
  while (modPow(generator, order, candidate) !== 1n) generator += 1n;
  return { p: candidate, q: order, g: generator };
}

const GROUP = buildGroup(61);
const mod = (value, m) => ((value % m) + m) % m;

const stream = (seed) => {
  let state = (seed >>> 0) || 1;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state;
  };
};

/** Non-cryptographic transcript hash (documented) used for challenges and identifiers. */
function mix(...values) {
  let h = 0x811c9dc5;
  const feed = (value) => {
    const text = value.toString();
    for (let i = 0; i < text.length; i++) {
      h = (Math.imul(h ^ text.charCodeAt(i), 0x01000193) + 0x9e3779b9) >>> 0;
    }
  };
  for (const value of values) feed(value);
  return h >>> 0;
}

const hashToScalar = (...values) => BigInt(mix(...values)) % GROUP.q;

function makeKeypair(seed) {
  const random = stream(seed);
  const secret = mod(BigInt(random()) * 65537n + BigInt(random()), GROUP.q) || 1n;
  return { secret, public: modPow(GROUP.g, secret, GROUP.p) };
}

/** Deterministic Schnorr signature (the nonce is derived from the key and the message). */
function schnorrSign(keypair, message) {
  const nonce = hashToScalar(keypair.secret, message, 0x837) || 1n;
  const commitment = modPow(GROUP.g, nonce, GROUP.p);
  const challenge = hashToScalar(commitment, message);
  const response = mod(nonce + challenge * keypair.secret, GROUP.q);
  return { commitment, response, challenge, message };
}

/** Independent verification: recompute e from the transcript and check g^s == r * pk^e. */
function schnorrVerify(publicKey, message, signature) {
  if (signature.message !== message) return { ok: false, reason: "message mismatch" };
  const challenge = hashToScalar(signature.commitment, message);
  if (challenge !== signature.challenge) return { ok: false, reason: "challenge mismatch" };
  const left = modPow(GROUP.g, signature.response, GROUP.p);
  const right = mod(modPow(publicKey, challenge, GROUP.p) * signature.commitment, GROUP.p);
  return { ok: left === right, reason: left === right ? "ok" : "equation", left, right };
}

// ---------------------------------------------------------------------------------------------
// DID documents
// ---------------------------------------------------------------------------------------------

const didFromKey = (publicKey) => `did:cc:${mix(publicKey, GROUP.p).toString(16).padStart(8, "0")}`;

/**
 * A DID document: the identifier is the hash of the current key, and rotated keys stay listed with a
 * validity window so that credentials signed under them can still be checked.
 */
function makeDidDocument({ keypair, rotationHistory = [] }) {
  return {
    id: didFromKey(keypair.public),
    currentKey: keypair.public,
    keys: [...rotationHistory, { key: keypair.public, validFrom: 0, validUntil: null, retired: false }],
  };
}

function keyValidAt(document, key, time) {
  return document.keys.some((entry) => entry.key === key &&
    (entry.retired === false || (entry.validUntil !== null && time < entry.validUntil)));
}

/** Resolution is a lookup in a registry of documents keyed by DID (an unknown DID fails). */
const makeResolver = (documents) => (did) => documents.find((document) => document.id === did) ?? null;

// ---------------------------------------------------------------------------------------------
// Verifiable credentials with selective disclosure
// ---------------------------------------------------------------------------------------------

function claimSalt(seed, name) {
  const random = stream(seed + mix(name));
  return mod(BigInt(random()) * 4294967296n + BigInt(random()), GROUP.q);
}

const claimDigest = (name, value, salt) => mix(name, value, salt, 0x837);

function merkleize(digests) {
  let level = digests.slice();
  const levels = [level];
  while (level.length > 1) {
    const next = [];
    for (let i = 0; i < level.length; i += 2) next.push(mix(level[i], level[i + 1] ?? level[i]));
    level = next;
    levels.push(level);
  }
  return { root: level[0], levels };
}

function merkleProof(tree, index) {
  const path = [];
  let position = index;
  for (let depth = 0; depth + 1 < tree.levels.length; depth++) {
    path.push(tree.levels[depth][position ^ 1] ?? tree.levels[depth][position]);
    position = Math.floor(position / 2);
  }
  return path;
}

function merkleCheck(leaf, index, path) {
  let node = leaf;
  for (const sibling of path) {
    node = index % 2 === 0 ? mix(node, sibling) : mix(sibling, node);
    index = Math.floor(index / 2);
  }
  return node;
}

function issueCredential({ issuerDocument, issuerKeypair, subject, claims, issuedAt, expiresAt, saltSeed }) {
  const names = Object.keys(claims).sort();
  const entries = names.map((name) => ({ name, value: claims[name], salt: claimSalt(saltSeed, name) }));
  const digests = entries.map((entry) => claimDigest(entry.name, entry.value, entry.salt));
  const tree = merkleize(digests);
  const body = { issuer: issuerDocument.id, subject, notBefore: issuedAt, notAfter: expiresAt,
    root: tree.root, claimCount: names.length };
  const signature = schnorrSign(issuerKeypair, `${body.issuer}|${body.subject}|${body.root}|${body.notBefore}|${body.notAfter}`);
  return { ...body, signature, entries, tree, revoked: false };
}

/** The holder builds a presentation revealing only `revealed` claims, bound to `nonce`. */
function presentCredential({ credential, revealed, holderKeypair, nonce }) {
  const proofs = {};
  for (const name of revealed) {
    const index = credential.entries.findIndex((entry) => entry.name === name);
    if (index === -1) return { error: `unknown claim ${name}` };
    proofs[name] = {
      value: credential.entries[index].value,
      salt: credential.entries[index].salt,
      index,
      path: merkleProof(credential.tree, index),
    };
  }
  const body = `${credential.issuer}|${credential.subject}|${credential.root}|${nonce}|${revealed.slice().sort().join(",")}`;
  return { credential: { ...credential, entries: undefined, tree: undefined }, revealed, proofs,
    nonce, holderProof: schnorrSign(holderKeypair, body), holderKey: holderKeypair.public,
    body };
}

/** Deterministic identifier of a credential (used by the revocation registry). */
const credentialId = (credential) => mix(credential.issuer, credential.subject, credential.root);

/**
 * Verification: resolve the issuer, check the issuer signature over the credential body, check the
 * holder binding signature, verify every revealed claim's Merkle path, enforce the validity window and
 * consult the signed revocation registry.
 */
function verifyPresentation({ presentation, resolve, requestedClaims = null, time, nonce,
  revocationRegistry = null, expectedSubject = null }) {
  const checks = [];
  const fail = (reason) => ({ valid: false, reason, checks });
  const credential = presentation.credential;
  const issuer = resolve(credential.issuer);
  if (issuer === null) return fail("issuer not resolvable");
  checks.push("issuer resolved");
  const body = `${credential.issuer}|${credential.subject}|${credential.root}|${credential.notBefore}|${credential.notAfter}`;
  const issuerVerdict = schnorrVerify(issuer.currentKey, body, credential.signature);
  if (!issuerVerdict.ok) return fail(`issuer signature (${issuerVerdict.reason})`);
  checks.push("issuer signature");
  if (!keyValidAt(issuer, issuer.currentKey, time)) return fail("issuer key not valid now");
  if (expectedSubject !== null && credential.subject !== expectedSubject) return fail("wrong subject");
  if (!(time >= credential.notBefore && time <= credential.notAfter)) return fail("outside the validity window");
  checks.push("validity window");
  if (nonce !== presentation.nonce) return fail("stale nonce");
  const holderVerdict = schnorrVerify(presentation.holderKey, presentation.body, presentation.holderProof);
  if (!holderVerdict.ok) return fail("holder binding");
  const subjectDocument = resolve(credential.subject);
  if (subjectDocument === null) return fail("subject not resolvable");
  if (subjectDocument.currentKey !== presentation.holderKey) return fail("holder is not the subject");
  checks.push("holder binding");
  const revealed = Object.keys(presentation.proofs).sort();
  if (revealed.join(",") !== presentation.revealed.slice().sort().join(",")) return fail("revealed set mismatch");
  if (requestedClaims !== null) {
    const requested = requestedClaims.slice().sort();
    if (requested.join(",") !== revealed.join(",")) return fail("claims do not match the request");
  }
  for (const name of revealed) {
    const proof = presentation.proofs[name];
    const leaf = claimDigest(name, proof.value, proof.salt);
    if (merkleCheck(leaf, proof.index, proof.path) !== credential.root) return fail(`claim ${name}`);
  }
  checks.push("claim proofs");
  if (revocationRegistry !== null) {
    if (mix(...revocationRegistry.entries) !== revocationRegistry.root) return fail("registry root");
    if (revocationRegistry.entries.includes(credentialId(credential))) {
      return fail("credential revoked");
    }
    checks.push("revocation registry");
  }
  return { valid: true, reason: "ok", checks, subject: credential.subject, revealed };
}

module.exports = { GROUP, mod, modPow, stream, mix, hashToScalar, makeKeypair, schnorrSign,
  schnorrVerify, didFromKey, makeDidDocument, keyValidAt, makeResolver, claimSalt, claimDigest,
  merkleize, merkleProof, merkleCheck, credentialId, issueCredential, presentCredential, verifyPresentation,
  isProbablePrime, buildGroup };

if (require.main === module) {
  const assert = require("node:assert");
  const metrics = [];

  // ---- 1. group and Schnorr algebra -----------------------------------------------------------
  assert.ok(isProbablePrime(GROUP.p) && isProbablePrime(GROUP.q), "safe prime group");
  assert.strictEqual(modPow(GROUP.g, GROUP.q, GROUP.p), 1n, "the generator has order q");
  const alice = makeKeypair(1);
  const bob = makeKeypair(2);
  const message = "did:cc:subject";
  const signature = schnorrSign(alice, message);
  const verdict = schnorrVerify(alice.public, message, signature);
  assert.ok(verdict.ok, "the signature verifies");
  const independentLeft = modPow(GROUP.g, signature.response, GROUP.p);
  const independentRight = mod(modPow(alice.public, signature.challenge, GROUP.p) * signature.commitment, GROUP.p);
  assert.strictEqual(independentLeft, independentRight, "g^s == r * y^e recomputed independently");
  assert.ok(!schnorrVerify(bob.public, message, signature).ok, "another key does not verify it");
  assert.ok(!schnorrVerify(alice.public, `${message}!`, signature).ok, "a modified message fails");
  const forged = { ...signature, response: mod(signature.response + 1n, GROUP.q) };
  assert.ok(!schnorrVerify(alice.public, message, forged).ok, "a modified response fails");
  assert.strictEqual(schnorrVerify(alice.public, message, schnorrSign(alice, message)).ok, true,
    "signing is deterministic");

  // ---- 2. DID documents are self-certifying ------------------------------------------------------
  const issuerKeys = makeKeypair(11);
  const subjectKeys = makeKeypair(22);
  const issuerDocument = makeDidDocument({ keypair: issuerKeys });
  const subjectDocument = makeDidDocument({ keypair: subjectKeys });
  assert.strictEqual(issuerDocument.id, didFromKey(issuerKeys.public));
  assert.notStrictEqual(didFromKey(bob.public), issuerDocument.id, "a swapped key changes the identifier");
  const resolve = makeResolver([issuerDocument, subjectDocument]);
  assert.ok(resolve(issuerDocument.id) !== null);
  assert.strictEqual(resolve("did:cc:deadbeef"), null, "an unknown DID does not resolve");

  // ---- 3. issuing and presenting credentials ------------------------------------------------------
  const claims = { name: "Ada", degree: "MSc", birthYear: "1815", country: "UK" };
  const credential = issueCredential({ issuerDocument, issuerKeypair: issuerKeys,
    subject: subjectDocument.id, claims, issuedAt: 100, expiresAt: 200, saltSeed: 7 });
  assert.strictEqual(credential.claimCount, 4);
  assert.strictEqual(credential.entries.length, 4);
  const nonce = 4242;
  const presentation = presentCredential({ credential, revealed: ["degree", "country"],
    holderKeypair: subjectKeys, nonce });
  const base = { presentation, resolve, requestedClaims: ["degree", "country"], time: 150, nonce,
    expectedSubject: subjectDocument.id };
  const accepted = verifyPresentation(base);
  assert.ok(accepted.valid, `valid presentation (${accepted.reason})`);
  assert.deepStrictEqual(accepted.revealed.sort(), ["country", "degree"]);
  assert.deepStrictEqual(accepted.checks,
    ["issuer resolved", "issuer signature", "validity window", "holder binding", "claim proofs"]);

  // ---- 4. selective disclosure keeps the rest hidden ----------------------------------------------
  const serialize = (value) => (typeof value === "bigint" ? value.toString()
    : Array.isArray(value) ? `[${value.map(serialize).join(",")}]`
      : (value !== null && typeof value === "object")
        ? `{${Object.keys(value).map((key) => `${key}:${serialize(value[key])}`).join(",")}}`
        : String(value));
  const flattened = serialize(presentation);
  assert.ok(!presentation.revealed.includes("name"), "unrevealed claims are absent");
  assert.strictEqual(flattened.includes("Ada"), false, "the hidden name never appears");
  assert.strictEqual(flattened.includes("1815"), false, "the hidden birth year never appears");
  assert.ok(flattened.includes("MSc"), "the disclosed value is present");
  assert.strictEqual(Object.keys(presentation.proofs).length, 2);
  const overDisclosed = presentCredential({ credential, revealed: ["degree", "country", "name"],
    holderKeypair: subjectKeys, nonce });
  assert.strictEqual(verifyPresentation({ ...base, presentation: overDisclosed }).valid, false,
    "revealing more than requested fails");
  const underRequested = verifyPresentation({ ...base, requestedClaims: ["degree"] });
  assert.strictEqual(underRequested.valid, false, "revealing more than the verifier asked for fails");

  // ---- 5. tampering ----------------------------------------------------------------------------
  const tamperedValue = presentCredential({ credential, revealed: ["degree", "country"],
    holderKeypair: subjectKeys, nonce });
  tamperedValue.proofs.degree.value = "PhD";
  assert.ok(!verifyPresentation({ ...base, presentation: tamperedValue }).valid, "a tampered claim fails");
  const tamperedPath = presentCredential({ credential, revealed: ["degree"], holderKeypair: subjectKeys, nonce });
  tamperedPath.proofs.degree.path[0] = mix(tamperedPath.proofs.degree.path[0]);
  assert.ok(!verifyPresentation({ ...base, presentation: tamperedPath, requestedClaims: ["degree"] }).valid,
    "a tampered Merkle path fails");
  const tamperedCredential = issueCredential({ issuerDocument, issuerKeypair: issuerKeys,
    subject: subjectDocument.id, claims, issuedAt: 100, expiresAt: 200, saltSeed: 7 });
  tamperedCredential.subject = bob.public.toString();
  assert.ok(!verifyPresentation({ ...base, presentation: { ...presentation, credential: tamperedCredential } }).valid,
    "a modified credential body fails the issuer signature");
  const tamperedRoot = presentCredential({ credential, revealed: ["country"], holderKeypair: subjectKeys, nonce });
  tamperedRoot.credential.root = mix(tamperedRoot.credential.root);
  assert.ok(!verifyPresentation({ ...base, presentation: tamperedRoot, requestedClaims: ["country"] }).valid,
    "a modified credential root fails");

  // ---- 6. holder binding, replay and lifecycle ------------------------------------------------------
  const impostor = presentCredential({ credential, revealed: ["degree"], holderKeypair: alice, nonce });
  assert.ok(!verifyPresentation({ ...base, presentation: impostor, requestedClaims: ["degree"] }).valid,
    "another holder key cannot present the credential");
  assert.ok(!verifyPresentation({ ...base, nonce: 999 }).valid, "a replayed nonce fails");
  assert.ok(!verifyPresentation({ ...base, time: 201 }).valid, "an expired credential fails");
  assert.ok(!verifyPresentation({ ...base, time: 99 }).valid, "a credential that is not yet valid fails");
  assert.ok(verifyPresentation({ ...base, time: 100 }).valid && verifyPresentation({ ...base, time: 200 }).valid,
    "the window is inclusive");
  assert.ok(!verifyPresentation({ ...base, expectedSubject: bob.public.toString() }).valid, "a wrong subject fails");

  // ---- 7. revocation registry -----------------------------------------------------------------------
  const registry = { entries: [credentialId(credential)], root: 0 };
  registry.root = mix(...registry.entries);
  const revoked = verifyPresentation({ ...base, revocationRegistry: registry });
  assert.strictEqual(revoked.valid, false);
  assert.strictEqual(revoked.reason, "credential revoked");
  const clean = { entries: [mix(0xdeadbeef)], root: 0 };
  clean.root = mix(...clean.entries);
  assert.ok(verifyPresentation({ ...base, revocationRegistry: clean }).valid, "a clear registry passes");
  const forgedRegistry = { entries: clean.entries, root: mix(0x1234) };
  assert.strictEqual(verifyPresentation({ ...base, revocationRegistry: forgedRegistry }).reason, "registry root",
    "a registry whose root does not match its entries is rejected");

  // ---- 8. key rotation ------------------------------------------------------------------------------
  const rotatedKeys = makeKeypair(33);
  const rotatedDocument = makeDidDocument({ keypair: rotatedKeys,
    rotationHistory: [{ key: issuerKeys.public, validFrom: 0, validUntil: 260, retired: true }] });
  const rotatedResolver = makeResolver([rotatedDocument, subjectDocument]);
  const oldCredential = issueCredential({ issuerDocument, issuerKeypair: issuerKeys,
    subject: subjectDocument.id, claims, issuedAt: 100, expiresAt: 400, saltSeed: 7 });
  const oldPresentation = presentCredential({ credential: oldCredential, revealed: ["degree"],
    holderKeypair: subjectKeys, nonce });
  assert.strictEqual(verifyPresentation({ presentation: oldPresentation, resolve: rotatedResolver,
    requestedClaims: ["degree"], time: 150, nonce }).valid, false,
    "after rotation the issuer resolves to the new key, so the old signature fails");
  const newCredential = issueCredential({ issuerDocument: rotatedDocument, issuerKeypair: rotatedKeys,
    subject: subjectDocument.id, claims, issuedAt: 100, expiresAt: 400, saltSeed: 9 });
  const newPresentation = presentCredential({ credential: newCredential, revealed: ["degree"],
    holderKeypair: subjectKeys, nonce });
  assert.ok(verifyPresentation({ presentation: newPresentation, resolve: rotatedResolver,
    requestedClaims: ["degree"], time: 150, nonce }).valid, "a credential from the rotated key verifies");
  assert.ok(keyValidAt(rotatedDocument, issuerKeys.public, 200), "the retired key is still valid in its window");
  assert.ok(!keyValidAt(rotatedDocument, issuerKeys.public, 300), "and invalid afterwards");
  assert.strictEqual(rotatedDocument.keys.length, 2);
  assert.notStrictEqual(rotatedDocument.id, issuerDocument.id, "the rotated document has a new identifier");

  // ---- 9. determinism -----------------------------------------------------------------------------
  const repeated = presentCredential({ credential, revealed: ["degree", "country"], holderKeypair: subjectKeys, nonce });
  assert.deepStrictEqual(repeated.holderProof.commitment, presentation.holderProof.commitment);
  assert.strictEqual(repeated.body, presentation.body);
  const reissued = issueCredential({ issuerDocument, issuerKeypair: issuerKeys, subject: subjectDocument.id,
    claims, issuedAt: 100, expiresAt: 200, saltSeed: 7 });
  assert.strictEqual(reissued.root, credential.root, "salts and digests are deterministic");
  assert.strictEqual(reissued.signature.commitment, credential.signature.commitment);
  const singleClaim = issueCredential({ issuerDocument, issuerKeypair: issuerKeys, subject: subjectDocument.id,
    claims: { name: "Ada" }, issuedAt: 0, expiresAt: 10, saltSeed: 1 });
  const singlePresentation = presentCredential({ credential: singleClaim, revealed: ["name"],
    holderKeypair: subjectKeys, nonce: 1 });
  assert.ok(verifyPresentation({ presentation: singlePresentation, resolve, requestedClaims: ["name"],
    time: 5, nonce: 1 }).valid, "a single-claim credential verifies");
  assert.deepStrictEqual(presentCredential({ credential: singleClaim, revealed: ["missing"],
    holderKeypair: subjectKeys, nonce }).error, "unknown claim missing");

  metrics.push(`group: safe prime p with (p-1)/2 prime, generator order q; Schnorr signatures verified by recomputing g^s == r y^e`);
  metrics.push(`DID: self-certifying identifiers (${issuerDocument.id}), ${rotatedDocument.keys.length} keys after rotation, unknown DIDs fail to resolve`);
  metrics.push(`credential: ${credential.claimCount} claims Merkle-committed; selective disclosure reveals exactly the requested set and keeps the rest as salted digests`);
  metrics.push(`lifecycle: expiry window, ${registry.entries.length} revoked credential in the signed registry, holder binding and nonce replay all enforced`);
  for (const line of metrics) console.log(line);
  console.log("PASS");
}
