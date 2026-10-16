/** Task 923 - Implement distributed ledger with privacy-preserving transaction aggregation.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * A toy account ledger plus additive secret sharing so that an aggregator can publish the total
 * transfer volume without ever seeing an individual amount.
 *   ledger      account balances with signed transfers. Signatures are a keyed FNV-1a MAC over
 *               (from, to, amount, nonce); a monotone nonce per sender makes replays invalid, and
 *               the balance check makes overspending invalid. Total supply is conserved.
 *   shares      Shamir additive sharing over the prime field F_p: the dealer picks a random degree
 *               t-1 polynomial with f(0) = amount and hands f(1)..f(n) to n participants.
 *               t shares reconstruct the amount by Lagrange interpolation at 0, t-1 shares are
 *               information-theoretically useless.
 *   aggregation each participant sends its share when asked for a "batch total"; the aggregator adds
 *               shares coordinate-wise (x-coordinates stay fixed) and reconstructs the sum, which is
 *               exactly the sum of the plaintext amounts.
 * Simplification: a single honest dealer, no verifiable secret sharing, no network or Byzantine
 * faults; the ledger is in-memory and single-process. Verified: supply conservation, replay and
 * forgery rejection, exact homomorphic aggregation, the t-of-n threshold (all subsets), the
 * uniformity of a single share over F_101, boundary cases, invalid input and determinism.
 */
"use strict";
const assert = require("node:assert");

class Prng {
  constructor(seed) { this.state = (seed >>> 0) || 0x9e3779b9; }
  next() {
    let x = this.state;
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17; x ^= x << 5;
    return (this.state = x >>> 0);
  }
  float() { return this.next() / 4294967296; }
  int(bound) { return this.next() % bound; }
}

/* ------------------------------------------------------------------ signature and ledger */

function fnv1a(text) {
  let hash = 0x811c9dc5;
  for (let k = 0; k < text.length; k++) {
    hash ^= text.charCodeAt(k);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash >>> 0;
}

function sign(key, tx) {
  return fnv1a(`${key}|${tx.from}|${tx.to}|${tx.amount}|${tx.nonce}`);
}

function makeLedger(seed, options = {}) {
  const key = `k${seed}`;
  const supply = options.supply ?? 1000;
  const accounts = new Map(options.accounts ?? [["alice", supply], ["bob", 0], ["carol", 0]]);
  const nonces = new Map();
  const applied = [];
  return {
    key,
    accounts,
    nonces,
    applied,
    totalSupply: () => [...accounts.values()].reduce((a, v) => a + v, 0),
    /** Validates a signed transfer and, when accepted, moves the money. Returns a reason otherwise. */
    submit(tx) {
      if (!Number.isInteger(tx.amount) || tx.amount <= 0) return "invalid amount";
      if (tx.from === tx.to) return "self transfer";
      if (!accounts.has(tx.from) || !accounts.has(tx.to)) return "unknown account";
      if (sign(key, tx) !== tx.mac) return "bad signature";
      if (tx.nonce !== (nonces.get(tx.from) ?? 0) + 1) return "stale nonce (replay)";
      if (accounts.get(tx.from) < tx.amount) return "insufficient funds";
      accounts.set(tx.from, accounts.get(tx.from) - tx.amount);
      accounts.set(tx.to, accounts.get(tx.to) + tx.amount);
      nonces.set(tx.from, tx.nonce);
      applied.push(tx);
      return null;
    },
  };
}

function transfer(ledger, from, to, amount) {
  const nonce = (ledger.nonces.get(from) ?? 0) + 1;
  const tx = { from, to, amount, nonce };
  tx.mac = sign(ledger.key, tx);
  return { tx, reason: ledger.submit(tx) };
}

/* ------------------------------------------------------------------ Shamir additive sharing */

const DEFAULT_PRIME = 1000003;

function mod(value, p) { return ((value % p) + p) % p; }

function modPow(base, exponent, p) {
  let result = 1, b = mod(base, p), e = exponent;
  while (e > 0) {
    if (e & 1) result = (result * b) % p;
    b = (b * b) % p;
    e >>= 1;
  }
  return result;
}

function modInverse(value, p) {
  assert.notStrictEqual(mod(value, p), 0, "zero has no inverse in the field");
  return modPow(value, p - 2, p); // p is prime: Fermat
}

/** Split a secret into n shares with threshold t; shares are {x, y} points of a degree t-1 polynomial. */
function share(secret, n, t, prng, p = DEFAULT_PRIME) {
  assert.ok(Number.isInteger(t) && t >= 1, "the threshold must be at least 1");
  assert.ok(Number.isInteger(n) && n >= t, "the number of shares must be at least the threshold");
  assert.ok(n < p, "there cannot be p or more distinct non-zero evaluation points");
  assert.ok(secret >= 0 && secret < p, "the secret must live in the field");
  const coefficients = [secret];
  for (let k = 1; k < t; k++) coefficients.push(prng ? prng.int(p) : Math.floor(Math.random() * p));
  return Array.from({ length: n }, (_, index) => ({ x: index + 1, y: evaluate(coefficients, index + 1, p) }));
}

function evaluate(coefficients, x, p) {
  let acc = 0;
  for (let k = coefficients.length - 1; k >= 0; k--) acc = mod(acc * x + coefficients[k], p);
  return acc;
}

/** Lagrange interpolation at x = 0; fewer than t points simply produce some other field element. */
function reconstruct(shares, p = DEFAULT_PRIME) {
  assert.ok(shares.length >= 1, "reconstruction needs at least one share");
  const seen = new Set();
  for (const s of shares) {
    assert.ok(s.x >= 1 && s.x < p && s.y >= 0 && s.y < p, "share outside the field");
    assert.ok(!seen.has(s.x), `duplicate share index ${s.x}`);
    seen.add(s.x);
  }
  let total = 0;
  for (let i = 0; i < shares.length; i++) {
    let numerator = 1, denominator = 1;
    for (let j = 0; j < shares.length; j++) {
      if (i === j) continue;
      numerator = mod(numerator * (0 - shares[j].x), p);
      denominator = mod(denominator * (shares[i].x - shares[j].x), p);
    }
    total = mod(total + shares[i].y * numerator % p * modInverse(denominator, p), p);
  }
  return total;
}

/** Homomorphic addition: shares of the same x-coordinate add coordinate-wise. */
function addShares(lists, p = DEFAULT_PRIME) {
  assert.ok(lists.length >= 1, "aggregation needs at least one share list");
  const width = lists[0].length;
  for (const list of lists) assert.strictEqual(list.length, width, "share lists must have equal length");
  return Array.from({ length: width }, (_, index) => {
    const x = lists[0][index].x;
    for (const list of lists) assert.strictEqual(list[index].x, x, "share x-coordinates must align");
    const y = lists.reduce((acc, list) => mod(acc + list[index].y, p), 0);
    return { x, y };
  });
}

/* ------------------------------------------------------------------ checks */

function shareHistogram(secret, samples, prng, p) {
  const buckets = new Array(p).fill(0);
  for (let k = 0; k < samples; k++) {
    const [only] = share(secret, 2, 2, prng, p);
    buckets[only.y]++;
  }
  return buckets;
}

function main() {
  const prng = new Prng(20250923);

  // 1. Ledger happy path: signed transfers move value and the total supply is conserved.
  const ledger = makeLedger(923, { supply: 1000 });
  assert.strictEqual(transfer(ledger, "alice", "bob", 200).reason, null);
  assert.strictEqual(transfer(ledger, "alice", "carol", 150).reason, null);
  assert.strictEqual(transfer(ledger, "bob", "carol", 50).reason, null);
  assert.strictEqual(ledger.accounts.get("alice"), 650);
  assert.strictEqual(ledger.accounts.get("bob"), 150);
  assert.strictEqual(ledger.accounts.get("carol"), 200);
  assert.strictEqual(ledger.totalSupply(), 1000, "supply must be conserved");

  // 2. Forgery, replay and overspending are rejected without touching the balances.
  const replay = ledger.applied[0];
  assert.match(ledger.submit(replay), /replay/, "a replayed transaction must be refused");
  const forged = { ...ledger.applied[1], amount: 999 };
  assert.match(ledger.submit(forged), /signature/, "a tampered amount must break the MAC");
  const overspend = transfer(ledger, "bob", "carol", 100000);
  assert.match(overspend.reason, /insufficient/, "overspending must be refused");
  assert.strictEqual(ledger.totalSupply(), 1000);
  assert.strictEqual(ledger.applied.length, 3, "only the valid transfers were applied");

  // 3. Threshold sharing: any t of n shares reconstruct, any t-1 shares do not.
  const secret = 424242;
  const shares = share(secret, 5, 3, prng);
  assert.strictEqual(reconstruct(shares.slice(0, 3)), secret);
  assert.strictEqual(reconstruct(shares.slice(2, 5)), secret);
  for (let a = 0; a < 5; a++) {
    for (let b = a + 1; b < 5; b++) {
      assert.notStrictEqual(reconstruct([shares[a], shares[b]]), secret,
        `shares ${a},${b} must not reveal the secret`);
    }
  }
  assert.deepStrictEqual(new Set(shares.map((s) => s.x)), new Set([1, 2, 3, 4, 5]));

  // 4. Boundary cases: t = 1 hands out the secret itself, t = n needs every share, 0 and p-1 work.
  const trivial = share(77, 4, 1, prng);
  assert.ok(trivial.every((s) => s.y === 77), "with t = 1 every share equals the secret");
  assert.strictEqual(reconstruct([trivial[3]]), 77);
  const full = share(12345, 4, 4, prng);
  assert.strictEqual(reconstruct(full), 12345);
  assert.notStrictEqual(reconstruct(full.slice(0, 3)), 12345, "t-1 = 3 shares must not suffice");
  assert.strictEqual(reconstruct(share(0, 3, 2, prng)), 0);
  const top = DEFAULT_PRIME - 1;
  assert.strictEqual(reconstruct(share(top, 3, 2, prng)), top);

  // 5. Homomorphic aggregation equals the plaintext total, and the ledger volume matches.
  const amounts = [120, 999, 5000, 77, 31337];
  const dealers = amounts.map((amount) => share(amount, 4, 3, prng));
  const combined = addShares(dealers);
  const volume = reconstruct(combined);
  assert.strictEqual(volume, amounts.reduce((a, b) => a + b, 0));
  assert.strictEqual(volume, 37533);
  const ledgerVolume = ledger.applied.reduce((acc, tx) => acc + tx.amount, 0);
  assert.strictEqual(ledgerVolume, 400, "the ledger volume is the sum of the applied amounts");
  assert.strictEqual(reconstruct(addShares([share(ledger.applied[0].amount, 3, 2, prng),
    share(ledgerVolume - ledger.applied[0].amount, 3, 2, prng)])), ledgerVolume);

  // 6. Privacy oracle: over F_101 a single share is uniform and independent of the secret.
  {
    const p = 101, samples = 101 * 80;
    const histA = shareHistogram(13, samples, new Prng(5), p);
    const histB = shareHistogram(88, samples, new Prng(5), p);
    const expected = samples / p;
    assert.ok(histA.every((count) => count > 0), "every field element must be reachable");
    const maxDeviation = Math.max(...histA.map((count) => Math.abs(count - expected)));
    assert.ok(maxDeviation < 4 * Math.sqrt(expected), `histogram deviation ${maxDeviation} too large`);
    const l1 = histA.reduce((acc, count, k) => acc + Math.abs(count - histB[k]), 0) / (2 * samples);
    assert.ok(l1 < 0.08, `share distributions for two secrets differ by ${l1}`);
    assert.ok(histA.some((count, k) => count !== histB[k]), "the test must not be a tautology");
  }

  // 7. Invalid input is rejected.
  assert.throws(() => share(5, 3, 4, prng), /at least the threshold/);
  assert.throws(() => share(5, 4, 0, prng), /at least 1/);
  assert.throws(() => share(DEFAULT_PRIME, 3, 2, prng), /live in the field/);
  assert.throws(() => reconstruct([{ x: 1, y: 2 }, { x: 1, y: 3 }]), /duplicate share/);
  assert.throws(() => reconstruct([]), /at least one share/);
  assert.throws(() => addShares([[{ x: 1, y: 2 }], [{ x: 2, y: 3 }]]), /must align/);

  // 8. Determinism: identical seeds rebuild identical shares and identical ledger state.
  const rerun = () => {
    const local = makeLedger(923, { supply: 1000 });
    transfer(local, "alice", "bob", 200);
    transfer(local, "alice", "carol", 150);
    return { shares: share(424242, 5, 3, new Prng(20250923)), balances: [...local.accounts] };
  };
  assert.deepStrictEqual(rerun(), rerun());
  const other = makeLedger(924, { supply: 1000 });
  assert.notStrictEqual(transfer(other, "alice", "bob", 200).tx.mac, ledger.applied[0].mac,
    "different keys must produce different signatures");

  console.log(`ledger: ${ledger.applied.length} transfers, supply ${ledger.totalSupply()}, ` +
    `aggregated volume ${volume} (plaintext total 37533), 5-of-3 sharing verified`);
  console.log("PASS");
}

if (require.main === module) main();
module.exports = { Prng, fnv1a, sign, makeLedger, transfer, share, reconstruct, addShares, evaluate,
  modInverse, DEFAULT_PRIME };
