/** Task 756 - Implement side-channel resistant constant-time modular exponentiation. (Tier: Expert, CTCI Beyond — Security & Privacy)
 *
 * Toy-scale modular arithmetic on BigInt with two exponentiation algorithms and a cost model
 * that turns "constant time" into something a test can refute. `naiveModPow` is the textbook
 * left-to-right square-and-multiply: it multiplies only for set exponent bits, so its cost is
 * `bitlen + popcount(exponent)` — the classic leakage that lets an attacker read the Hamming
 * weight of a secret exponent out of a timing trace. `ladderModPow` is the Montgomery ladder:
 * for every exponent bit it conditionally swaps the two accumulators with a branch-free
 * `ctSwap`, computes *both* the multiply and the square, and picks between them with
 * `ctSelect`. The cost is therefore exactly `2 * bitlen` multiplies and `4 * bitlen` selects,
 * independent of the exponent's value and of the base.
 * Toy scale throughout: the moduli are fixed, public constants (a small prime and the 61-bit
 * Mersenne prime 2^61-1) and the exponents are small — this is a *model* of constant-time
 * exponentiation, never key material.
 *
 * What is simplified: no Montgomery multiplication, no fixed-width exponent blinding, no
 * windowing and no attempt at true constant-time behaviour of BigInt itself — the model
 * assumes every BigInt operation costs a fixed amount, which is exactly what a real
 * implementation cannot assume. What is verified: (a) correctness of the ladder against the
 * naive implementation, against BigInt's own `**` for small exponents, against Fermat's little
 * theorem (b^(p-1) = 1 mod p) and against the homomorphism b^(e1+e2) = b^e1 * b^e2; (b) the
 * ladder's cost is identical for every exponent of a given bit length (all 512 ten-bit
 * exponents) and for every base, while the naive cost equals bitlen + popcount and takes
 * several distinct values over the same exponents (negative control); (c) leak quantification:
 * over a candidate set with distinct Hamming weights the naive version identifies the secret
 * exactly (3 bits leaked out of 3) while the ladder leaks exactly 0 bits; (d) exponent/modulus
 * boundaries (0, 1, modulus 1, base 0, base >= modulus) and input validation.
 */
"use strict";

const { deepStrictEqual, strictEqual, ok, throws } = require("node:assert");

/** Fixed, public toy moduli: 97 and the 61-bit Mersenne prime 2^61 - 1. */
const SMALL_PRIME = 97n;
const MERSENNE_61 = 2305843009213693951n;

/** Cost model: a BigInt multiply-modulo dominates a branch-free select. */
const MULMOD_COST = 10;
const SELECT_COST = 1;

class Cost {
  constructor() { this.mulmods = 0; this.selects = 0; }
  mulmod() { this.mulmods++; }
  select() { this.selects++; }
  total() { return this.mulmods * MULMOD_COST + this.selects * SELECT_COST; }
}

const mulmod = (a, b, modulus, cost) => { cost.mulmod(); return (a * b) % modulus; };

/** Branch-free select on BigInt: `bit` is 0n or 1n, the mask is 0n or -1n. */
function ctSelect(bit, whenSet, whenClear) {
  if (bit !== 0n && bit !== 1n) throw new RangeError("the selector must be 0n or 1n");
  const mask = -bit;
  return (whenSet & mask) | (whenClear & ~mask);
}

const normalise = (value, modulus) => ((value % modulus) + modulus) % modulus;

/** Textbook square-and-multiply: skips the multiply for zero bits, so it leaks them. */
function naiveModPow(base, exponent, modulus, cost = new Cost()) {
  if (modulus <= 0n) throw new RangeError("the modulus must be positive");
  if (exponent < 0n) throw new RangeError("the exponent must be non-negative");
  let result = 1n % modulus;
  let square = normalise(base, modulus);
  for (let remaining = exponent; remaining > 0n; remaining >>= 1n) {
    if ((remaining & 1n) === 1n) result = mulmod(result, square, modulus, cost);
    square = mulmod(square, square, modulus, cost);
  }
  return result;
}

/**
 * Montgomery ladder. Invariant for the prefix processed so far: (low, high) = (b^prefix,
 * b^(prefix+1)). One bit costs two multiply-modulos and four selects whatever the bit is.
 */
function ladderModPow(base, exponent, modulus, cost = new Cost()) {
  if (modulus <= 0n) throw new RangeError("the modulus must be positive");
  if (exponent < 0n) throw new RangeError("the exponent must be non-negative");
  const root = normalise(base, modulus);
  let low = 1n % modulus;
  let high = root;
  const bits = exponent.toString(2);
  for (const character of bits) {
    const bit = character === "1" ? 1n : 0n;
    const left = ctSelect(bit, high, low);        // low after the conditional swap
    const right = ctSelect(bit, low, high);       // high after the conditional swap
    const product = mulmod(left, right, modulus, cost);
    const square = mulmod(left, left, modulus, cost);
    low = ctSelect(bit, product, square);
    high = ctSelect(bit, square, product);
    cost.selects += 4;
  }
  return low;
}

/** Independent oracle for tiny exponents: BigInt's own exponentiation, then reduce. */
const repeatedMultiplication = (base, exponent, modulus) => (base ** exponent) % modulus;

/** Cost of one exponentiation, with the result thrown away. */
function costOf(exponentiate, base, exponent, modulus) {
  const cost = new Cost();
  const result = exponentiate(base, exponent, modulus, cost);
  return { result, mulmods: cost.mulmods, selects: cost.selects, total: cost.total() };
}

/**
 * Leakage of one exponentiation: with the candidate set known to the attacker, how many
 * candidates are indistinguishable from the secret? 0 bits leaked means every candidate looks
 * the same; log2(|candidates|) bits means the secret is identified exactly.
 */
function leakAnalysis(exponentiate, base, modulus, candidates, secret) {
  const secretCost = costOf(exponentiate, base, secret, modulus).total;
  const indistinguishable = candidates.filter((candidate) => costOf(exponentiate, base, candidate, modulus).total === secretCost).length;
  return { indistinguishable, bits: Math.log2(candidates.length / indistinguishable), secretCost };
}

/** Every exponent of an exact bit width, ascending. */
function exponentsOfWidth(width) {
  const low = 1n << BigInt(width - 1);
  const high = (1n << BigInt(width)) - 1n;
  const found = [];
  for (let value = low; value <= high; value++) found.push(value);
  return found;
}

module.exports = {
  SMALL_PRIME, MERSENNE_61, Cost, mulmod, ctSelect, naiveModPow, ladderModPow,
  repeatedMultiplication, costOf, leakAnalysis, exponentsOfWidth,
};

if (require.main === module) {
  // 1. the ladder is correct for every small base/exponent/modulus combination
  for (let base = 0n; base <= 20n; base++) {
    for (let exponent = 0n; exponent <= 12n; exponent++) {
      const expected = repeatedMultiplication(base, exponent, SMALL_PRIME);
      strictEqual(ladderModPow(base, exponent, SMALL_PRIME), expected, `ladder ${base}^${exponent} mod 97`);
      strictEqual(naiveModPow(base, exponent, SMALL_PRIME), expected, `naive ${base}^${exponent} mod 97`);
    }
  }
  // 2. the ladder agrees with the naive implementation on large, real-looking exponents
  for (let trial = 0; trial < 24; trial++) {
    const exponent = (1n << 40n) + BigInt(trial * 7919 + 13);
    const base = BigInt(trial + 2);
    strictEqual(ladderModPow(base, exponent, MERSENNE_61), naiveModPow(base, exponent, MERSENNE_61), `trial ${trial}`);
  }
  // 3. Fermat's little theorem is an independent property check (no reference implementation)
  for (let base = 1n; base <= 12n; base++) {
    strictEqual(ladderModPow(base, SMALL_PRIME - 1n, SMALL_PRIME), 1n, `Fermat for base ${base}`);
    strictEqual(ladderModPow(base, MERSENNE_61 - 1n, MERSENNE_61), 1n, `Fermat for base ${base} mod 2^61-1`);
  }
  // 4. the exponential homomorphism b^(e1+e2) = b^e1 * b^e2 mod m
  for (const [first, second] of [[5n, 7n], [0n, 11n], [13n, 100n], [1n, 1n]]) {
    const left = ladderModPow(3n, first + second, SMALL_PRIME);
    const right = (ladderModPow(3n, first, SMALL_PRIME) * ladderModPow(3n, second, SMALL_PRIME)) % SMALL_PRIME;
    strictEqual(left, right, `3^${first} * 3^${second} mod 97`);
  }

  // 5. the ladder's cost is a function of the bit length alone, over all 512 ten-bit exponents
  const tenBit = exponentsOfWidth(10);
  strictEqual(tenBit.length, 512, "ten-bit exponents");
  const ladderCosts = new Set(tenBit.map((exponent) => costOf(ladderModPow, 5n, exponent, SMALL_PRIME).total));
  deepStrictEqual([...ladderCosts], [MULMOD_COST * 2 * 10 + SELECT_COST * 4 * 10], "one cost for every ten-bit exponent");
  const ladderMulmods = new Set(tenBit.map((exponent) => costOf(ladderModPow, 5n, exponent, SMALL_PRIME).mulmods));
  deepStrictEqual([...ladderMulmods], [20], "two multiply-modulos per bit, always");
  // 6. negative control: the naive cost equals bitlen + popcount and therefore varies
  const naiveCosts = tenBit.map((exponent) => costOf(naiveModPow, 5n, exponent, SMALL_PRIME).mulmods);
  strictEqual(new Set(naiveCosts).size, 10, "the naive multiply count takes ten distinct values");
  for (const exponent of tenBit) {
    let ones = 0;
    for (const character of exponent.toString(2)) if (character === "1") ones++;
    strictEqual(costOf(naiveModPow, 5n, exponent, SMALL_PRIME).mulmods, 10 + ones, `bitlen + popcount for ${exponent}`);
  }
  // 7. the cost never depends on the base either
  const baseCosts = new Set(Array.from({ length: 12 }, (_, index) => costOf(ladderModPow, BigInt(index), 1000003n, SMALL_PRIME).total));
  strictEqual(baseCosts.size, 1, "the ladder cost is base-independent");

  // 8. leak quantification over a candidate set with pairwise distinct Hamming weights
  const candidates = [0b1000000000n, 0b1100000000n, 0b1110000000n, 0b1111000000n,
    0b1111100000n, 0b1111110000n, 0b1111111000n, 0b1111111100n];
  for (const secret of candidates) {
    const naiveLeak = leakAnalysis(naiveModPow, 5n, SMALL_PRIME, candidates, secret);
    strictEqual(naiveLeak.indistinguishable, 1, "the naive version singles the secret out");
    strictEqual(naiveLeak.bits, 3, "all three bits of the eight-candidate set leak");
    const ladderLeak = leakAnalysis(ladderModPow, 5n, SMALL_PRIME, candidates, secret);
    strictEqual(ladderLeak.indistinguishable, candidates.length, "the ladder cannot separate any two candidates");
    strictEqual(ladderLeak.bits, 0, "no bits leak");
  }
  // ...and the same holds for a much larger candidate set over ten-bit exponents
  const bigSet = tenBit.filter((exponent) => exponent % 37n !== 0n);
  const bigLeak = leakAnalysis(ladderModPow, 5n, SMALL_PRIME, bigSet, bigSet[5]);
  strictEqual(bigLeak.indistinguishable, bigSet.length, `no candidate of the ${bigSet.length}-exponent set is distinguished`);
  strictEqual(bigLeak.bits, 0);

  // 9. boundaries: exponent 0 and 1, modulus 1, base 0, base above the modulus
  strictEqual(ladderModPow(5n, 0n, SMALL_PRIME), 1n, "anything to the power 0 is 1");
  strictEqual(ladderModPow(5n, 1n, SMALL_PRIME), 5n, "to the power 1 is the base");
  strictEqual(ladderModPow(0n, 0n, SMALL_PRIME), 1n, "0^0 is 1 by convention");
  strictEqual(ladderModPow(0n, 5n, SMALL_PRIME), 0n, "0^5 is 0");
  strictEqual(ladderModPow(5n, 3n, 1n), 0n, "modulus 1 collapses everything to 0");
  strictEqual(ladderModPow(100n, 3n, SMALL_PRIME), repeatedMultiplication(100n % SMALL_PRIME, 3n, SMALL_PRIME), "the base is reduced");
  strictEqual(costOf(ladderModPow, 5n, 1n, SMALL_PRIME).mulmods, 2, "a one-bit exponent costs one ladder step");
  throws(() => ladderModPow(5n, -1n, SMALL_PRIME), RangeError);
  throws(() => ladderModPow(5n, 3n, 0n), RangeError);
  throws(() => ladderModPow(5n, 3n, -7n), RangeError);
  throws(() => naiveModPow(5n, -1n, SMALL_PRIME), RangeError);
  throws(() => ctSelect(2n, 1n, 0n), RangeError);
  // 10. determinism
  deepStrictEqual(costOf(ladderModPow, 7n, 123456789n, MERSENNE_61), costOf(ladderModPow, 7n, 123456789n, MERSENNE_61));

  const sample = costOf(ladderModPow, 5n, 123456789n, MERSENNE_61);
  console.log(
    `tenBitExponents=${tenBit.length} ladderMulmods=${sample.mulmods} naiveMulmodsRange=${Math.min(...naiveCosts)}..${Math.max(...naiveCosts)} ` +
    `naiveLeakBits=3 ladderLeakBits=0`
  );
  console.log("PASS");
}
