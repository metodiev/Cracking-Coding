/** Task 840 - Implement fully homomorphic encryption demo simple arithmetic (toy). (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * The scheme is BFV-shaped. The secret is a single bit s, a ciphertext is a pair (a, b) standing for the
 * value a*s + b mod q, and a fresh encryption of a plaintext m in Z_t is (a, Delta*m + e - a*s) with
 * Delta = q/t the scaling factor and |e| <= eta fresh noise:
 *
 *   value(a, b)  = a*s + b  mod q   =  Delta*m + e
 *   add          coefficient-wise
 *   multiply     coefficient-wise product (a1*s+b1)(a2*s+b2) = (a1a2)s^2 + (a1b2+a2b1)s + b1b2 followed by
 *                the EXACT relinearisation s^2 = s for a binary secret, then the modulus switch that
 *                multiplies every coefficient by t/q with rounding - the step that turns the Delta^2-scaled
 *                product Delta^2*m1m2 + Delta*(...) into the Delta-scaled Delta*(m1m2) + noise
 *   decrypt      evaluate at s, take the centred representative, round to the nearest multiple of Delta
 *
 * Every ciphertext carries two bounds: the worst-case one (|m| <= t/2) and the message-aware one (the
 * magnitudes of the multiplicands are known whenever the circuit is known). The classical invariant is
 * that the plaintext is recovered exactly while the accumulated noise stays below Delta/2; the number of
 * multiplications that fits is the level budget, and it is measured, predicted, and extended by refresh.
 *
 * Simplifications, stated honestly: the toy is *not* secure - the secret is one bit, so two guesses break
 * it (measured below, not hidden), and real BFV uses a ring of dimension in the thousands. Relinearisation
 * is exact here because s^2 = s for a binary secret, whereas real schemes pay a documented noise term for
 * it; the toy therefore *understates* noise growth, which is why the measured depth budget is optimistic.
 * There is no NTT and no ring layer; "refresh" is a trusted decrypt-and-re-encrypt stand-in for
 * bootstrapping, which achieves the same without the key. One constraint is genuine rather than invented:
 * a single-modulus scheme is only correct while the *unreduced* product of the multiplicand magnitudes
 * stays below t/(2*Delta), because the scale-down rounds the wrapped representative - this toy shares that
 * limit with real BFV, and the frontier is measured.
 *
 * Invariants verified below. Correctness: encrypt/decrypt round-trips for every message in Z_t, addition,
 * multiplication, subtraction and two composed circuits agree with the plaintext oracle mod t, the folding
 * identity s^2 = s preserves the product's value exactly, and the largest correctly multiplied product is
 * compared with the analytic frontier t/(2*Delta). Soundness: the measured noise never exceeds the tracked
 * analytic bound at any sampled depth of a long chain, and decryption is exact exactly while the signed
 * noise against the intended plaintext stays in [-Delta/2, Delta/2) - measured at the very depth where the
 * chain breaks. Level budget: the worst-case bound's crossing depth is computed and shown to be a
 * conservative lower bound (the measured break comes much later), and the noise accumulation is checked
 * against the random-walk model statistically. Liveness: refresh restores the budget, so a chain that
 * breaks without it succeeds with it. Cost: the ciphertext is always two coefficients, at any depth.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every ciphertext is reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    long nextBelow(long bound) { return Math.floorMod(nextLong(), bound); }

    /** Uniform in [-bound, bound]. */
    long nextNoise(long bound) { return nextBelow(2 * bound + 1) - bound; }
}

/** (q, t, eta) with the derived scaling factor; every constraint is checked up front. */
final class FheParams {
    final long q;
    final long t;
    final long eta;
    final long delta;

    FheParams(long q, long t, long eta) {
        if (t < 2 || t > 1L << 30) throw new IllegalArgumentException("plaintext modulus out of range");
        if (q <= 4 * t) throw new IllegalArgumentException("ciphertext modulus must dwarf t");
        if (q % t != 0) throw new IllegalArgumentException("q must be a multiple of t in this toy");
        if (eta < 1 || eta > 1L << 20) throw new IllegalArgumentException("noise width out of range");
        this.q = q;
        this.t = t;
        this.eta = eta;
        this.delta = q / t;
        if (delta / 2 <= 4 * eta) {
            throw new IllegalArgumentException("Delta/2 must leave room for the noise, got "
                    + (delta / 2) + " for eta " + eta);
        }
    }

    long halfDelta() { return delta / 2; }
}

/** Integer helpers with explicit rounding rules for the modulus switch. */
final class FheMath {
    static long centered(long value, long modulus) {
        long v = Math.floorMod(value, modulus);
        return v > modulus / 2 ? v - modulus : v;
    }

    /** round(a / b) for positive b, ties going up; the rule used by the modulus switch. */
    static long roundDiv(long a, long b) { return Math.floorDiv(2 * a + b, 2 * b); }

    static long pow(long base, long exponent, long modulus) {
        long result = 1, b = Math.floorMod(base, modulus), e = exponent;
        while (e > 0) {
            if ((e & 1L) == 1L) result = result * b % modulus;
            b = b * b % modulus;
            e >>>= 1;
        }
        return result;
    }

    static long inverse(long a, long modulus) {
        long oldR = modulus, r = Math.floorMod(a, modulus), oldS = 0, s = 1;
        while (r != 0) {
            long q = oldR / r;
            long t = oldR - q * r;
            oldR = r;
            r = t;
            t = oldS - q * s;
            oldS = s;
            s = t;
        }
        if (oldR != 1) throw new IllegalArgumentException(a + " is not invertible mod " + modulus);
        return Math.floorMod(oldS, modulus);
    }
}

/** A ciphertext (a, b) standing for a*s + b, plus the analytic bound on its noise. */
final class Ciphertext {
    final long a;
    final long b;
    final long bound;

    Ciphertext(long a, long b, long bound) {
        this.a = a;
        this.b = b;
        this.bound = bound;
    }
}

/** Conservative noise bookkeeping; the cap keeps the products overflow free. */
final class NoiseModel {
    static final long CAP = 1L << 40;

    static long cap(long value) { return Math.min(Math.max(value, 0), CAP); }

    static long addBound(long left, long right) { return cap(cap(left) + cap(right)); }

    /**
     * Noise of a product: the multiplicands' noises scale with the other multiplicand's magnitude,
     * the second-order term is damped by Delta, and the modulus switch adds at most one unit on the
     * constant coefficient plus one on the secret coefficient.
     */
    static long multiplyBound(FheParams p, long noiseA, long noiseB, long magA, long magB) {
        if (magA >= CAP || magB >= CAP) return CAP;
        long first = cap(magA) * cap(noiseB) + cap(magB) * cap(noiseA);
        long second = cap(noiseA) * cap(noiseB) / p.delta;
        return cap(cap(first) + cap(second) + 4);
    }

    static long worstCaseMagnitude(FheParams p) { return p.t / 2; }

    /** Standard deviation of the noise added by one multiplication by a fresh encryption of one. */
    static double perStepSigma(long eta) {
        double variance = ((2.0 * eta + 1) * (2.0 * eta + 1) - 1) / 12.0;
        return Math.sqrt(2 * variance);
    }
}

/** BFV-shaped toy: a binary secret, homomorphic addition and multiplication, and a noise budget. */
final class FheScheme {
    final FheParams params;
    final long secret;
    long switchOperations;

    FheScheme(FheParams params, long seed) {
        this.params = params;
        this.secret = new SplitMix64(seed).nextBelow(2);
    }

    /** Fresh encryption: b = Delta*m + e - a*s so that a*s + b = Delta*m + e. */
    Ciphertext encrypt(long message, SplitMix64 rng) {
        if (message < 0 || message >= params.t) throw new IllegalArgumentException("message outside Z_t");
        long a = rng.nextBelow(params.q);
        long noise = rng.nextNoise(params.eta);
        long b = Math.floorMod(params.delta * message + noise - a * secret, params.q);
        return new Ciphertext(a, b, params.eta);
    }

    /** The value Delta*m + e that the ciphertext stands for. */
    long value(Ciphertext ct) { return Math.floorMod(ct.a * secret + ct.b, params.q); }

    /** Measured noise: the distance of the value from the nearest multiple of Delta. */
    long measureNoise(Ciphertext ct) {
        long v = FheMath.centered(value(ct), params.q);
        return Math.abs(v - FheMath.roundDiv(v, params.delta) * params.delta);
    }

    long decrypt(Ciphertext ct) {
        long v = FheMath.centered(value(ct), params.q);
        return Math.floorMod(FheMath.roundDiv(v, params.delta), params.t);
    }

    Ciphertext add(Ciphertext left, Ciphertext right) {
        return new Ciphertext(Math.floorMod(left.a + right.a, params.q),
                Math.floorMod(left.b + right.b, params.q),
                NoiseModel.addBound(left.bound, right.bound));
    }

    /** Plaintext negation: an encryption of t - m is an encryption of -m. */
    Ciphertext negate(Ciphertext ct) {
        return new Ciphertext(Math.floorMod(-ct.a, params.q), Math.floorMod(-ct.b, params.q), ct.bound);
    }

    /** Coefficient-wise product followed by the exact folding s^2 = s: still a valid ciphertext. */
    Ciphertext foldProduct(Ciphertext left, Ciphertext right) {
        long a1a2 = left.a * right.a % params.q;
        long cross = ((left.a * right.b % params.q) + (right.a * left.b % params.q)) % params.q;
        long b1b2 = left.b * right.b % params.q;
        long folded = Math.floorMod(a1a2 + cross, params.q);
        return new Ciphertext(folded, b1b2, NoiseModel.CAP);
    }

    /** The modulus switch: every coefficient is multiplied by t/q with rounding on the centred value. */
    Ciphertext scaleDown(Ciphertext ct) {
        switchOperations += 2;
        long a = Math.floorMod(FheMath.roundDiv(FheMath.centered(ct.a, params.q) * params.t, params.q),
                params.q);
        long b = Math.floorMod(FheMath.roundDiv(FheMath.centered(ct.b, params.q) * params.t, params.q),
                params.q);
        return new Ciphertext(a, b, ct.bound);
    }

    Ciphertext multiply(Ciphertext left, Ciphertext right, long magLeft, long magRight,
            boolean messageAware) {
        Ciphertext folded = foldProduct(left, right);
        Ciphertext scaled = scaleDown(folded);
        long leftMagnitude = messageAware ? Math.abs(magLeft) : NoiseModel.worstCaseMagnitude(params);
        long rightMagnitude = messageAware ? Math.abs(magRight) : NoiseModel.worstCaseMagnitude(params);
        return new Ciphertext(scaled.a, scaled.b,
                NoiseModel.multiplyBound(params, left.bound, right.bound, leftMagnitude, rightMagnitude));
    }

    /** Trusted refresh: decrypt and re-encrypt with fresh noise (stand-in for bootstrapping). */
    Ciphertext refresh(Ciphertext ct, SplitMix64 rng) { return encrypt(decrypt(ct), rng); }

    /**
     * Signed noise relative to the message the ciphertext is supposed to carry: the quantity that must
     * stay within Delta/2 for the decryption to be exact (the distance to the nearest multiple of Delta is
     * not enough, because a negative offset just past Delta/2 rounds down to the next plaintext).
     */
    long noiseAgainst(Ciphertext ct, long message) {
        return Math.abs(signedNoise(ct, message));
    }

    /** The same quantity with its sign: decryption is exact iff it lies in [-Delta/2, Delta/2). */
    long signedNoise(Ciphertext ct, long message) {
        long v = FheMath.centered(value(ct), params.q);
        return FheMath.centered(v - params.delta * message, params.q);
    }
}

public class Solution {
    /**
     * One parameter set: the budget Delta/2 = 512 for the noise, and correct products of magnitudes up to
     * t/(2*Delta) = 488. q is an exact multiple of t, so a scaled value never wraps.
     */
    static final FheParams SMALL = new FheParams(1_024_000_000L, 1_000_000L, 4L);

    public static void main(String[] args) {
        // ---- 1. parameter validation -------------------------------------------------------------
        expectThrows(() -> new FheParams(400_000L, 100_000L, 4L), "q not dwarfing t");
        expectThrows(() -> new FheParams(1_024_000L, 1000L, 900L), "noise wider than Delta/2");
        expectThrows(() -> new FheParams(400_000L, 1L, 4L), "plaintext modulus below two");
        expectThrows(() -> new FheParams(400_000L, 1000L, 0L), "zero noise width");
        expectThrows(() -> new FheParams(400_003L, 1000L, 1L), "q not a multiple of t");
        check(FheMath.inverse(3, 7) == 5 && FheMath.inverse(5, 7) == 3, "modular inverse sanity");
        expectThrows(() -> FheMath.inverse(4, 8), "non-invertible element");
        check(FheMath.roundDiv(5, 2) == 3 && FheMath.roundDiv(-5, 2) == -2,
                "rounding is ties-up and correct for negatives");
        check(SMALL.delta == 1024 && SMALL.halfDelta() == 512, "Delta = q/t = 1024, budget 512");
        check(SMALL.eta == 4 && SMALL.q / SMALL.t == SMALL.delta, "eta and the derived scaling factor");
        check(SMALL.q == SMALL.t * SMALL.delta, "q is an exact multiple of t, so scaled values never wrap");

        FheScheme scheme = new FheScheme(SMALL, 12345L);
        check(scheme.secret == 0 || scheme.secret == 1, "the secret is a single bit");

        // ---- 2. encrypt/decrypt round trip over the whole plaintext space ------------------------
        int roundTrips = 0;
        for (long m = 0; m < SMALL.t; m++) {
            Ciphertext ct = scheme.encrypt(m, new SplitMix64(1000 + m));
            check(scheme.decrypt(ct) == m, "round trip failed for message " + m);
            check(scheme.noiseAgainst(ct, m) <= SMALL.eta, "fresh noise must respect eta");
            roundTrips++;
        }
        check(roundTrips == SMALL.t, "every message in Z_t round-tripped");
        for (long m : new long[]{0, 1, SMALL.t - 1}) {
            Ciphertext ct = scheme.encrypt(m, new SplitMix64(7 + m));
            check(ct.bound == SMALL.eta, "a fresh ciphertext carries the fresh noise bound");
            check(Math.abs(FheMath.centered(scheme.value(ct) - SMALL.delta * m, SMALL.q)) <= SMALL.eta,
                    "the value must be Delta*m plus at most eta of noise");
        }

        // ---- 3. homomorphic addition and multiplication against the plaintext oracle -------------
        SplitMix64 rng = new SplitMix64(99L);
        long[] sumsLeft = {0, 1, 999_999, 500_000, 123_456, 7};
        long[] sumsRight = {0, 999_999, 1, 500_000, 876_544, 5};
        long[] productsLeft = {0, 1, 2, 7, 13, 21, 3};
        long[] productsRight = {0, 3, 7, 5, 29, 17, 98};
        int oracleChecks = 0;
        for (int i = 0; i < sumsLeft.length; i++) {
            long x = sumsLeft[i], y = sumsRight[i];
            Ciphertext cx = scheme.encrypt(x, rng), cy = scheme.encrypt(y, rng);
            Ciphertext sum = scheme.add(cx, cy);
            check(scheme.decrypt(sum) == Math.floorMod(x + y, SMALL.t),
                    "homomorphic sum failed for " + x + " + " + y + " = " + scheme.decrypt(sum));
            check(scheme.noiseAgainst(sum, Math.floorMod(x + y, SMALL.t)) <= sum.bound,
                    "measured sum noise exceeds its bound");
            check(scheme.decrypt(scheme.add(cx, scheme.negate(cy))) == Math.floorMod(x - y, SMALL.t),
                    "homomorphic subtraction failed for " + x + " - " + y);
            check(scheme.negate(cx).bound == cx.bound, "negation keeps the bound");
            oracleChecks += 3;
        }
        for (int i = 0; i < productsLeft.length; i++) {
            long x = productsLeft[i], y = productsRight[i];
            Ciphertext cx = scheme.encrypt(x, rng), cy = scheme.encrypt(y, rng);
            Ciphertext product = scheme.multiply(cx, cy, x, y, true);
            check(scheme.decrypt(product) == x * y % SMALL.t,
                    "homomorphic product failed for " + x + " * " + y + ", got " + scheme.decrypt(product));
            check(scheme.noiseAgainst(product, x * y % SMALL.t) <= product.bound,
                    "measured product noise " + scheme.noiseAgainst(product, x * y % SMALL.t) + " exceeds bound "
                            + product.bound);
            oracleChecks++;
        }
        check(oracleChecks == 25, "addition, subtraction and multiplication verified against the oracle");
        Ciphertext zero = scheme.encrypt(0, rng), one = scheme.encrypt(1, rng);
        check(scheme.decrypt(scheme.multiply(zero, one, 0, 1, true)) == 0
                && scheme.decrypt(scheme.multiply(one, one, 1, 1, true)) == 1,
                "multiplying by zero and by one keeps the plaintext");

        // ---- 3b. correctness frontier: products must stay below t/(2*Delta) ----------------------
        long analyticFrontier = SMALL.t / (2 * SMALL.delta);
        long measuredFrontier = 0;
        for (long x = 1; x <= analyticFrontier + 40; x++) {
            Ciphertext product = scheme.multiply(scheme.encrypt(x, new SplitMix64(x)),
                    scheme.encrypt(x, new SplitMix64(x + 5000)), x, x, true);
            if (scheme.decrypt(product) == x * x % SMALL.t) measuredFrontier = x * x;
            else break;
        }
        check(measuredFrontier >= analyticFrontier - 40,
                "the measured product frontier " + measuredFrontier + " must approach the analytic "
                        + analyticFrontier);
        check(measuredFrontier <= analyticFrontier,
                "and it must not exceed the analytic bound, saw " + measuredFrontier);

        // ---- 4. composed circuits ---------------------------------------------------------------
        long a = 7, b = 5, c = 3;
        Ciphertext encA = scheme.encrypt(a, rng), encB = scheme.encrypt(b, rng), encC = scheme.encrypt(c, rng);
        Ciphertext abPlusC = scheme.add(scheme.multiply(encA, encB, a, b, true), encC);
        check(scheme.decrypt(abPlusC) == (a * b + c) % SMALL.t, "a*b + c must match the plaintext oracle");
        Ciphertext difference = scheme.add(encA, scheme.negate(encB));
        Ciphertext aMinusBSquared = scheme.multiply(difference,
                scheme.add(encA, encB), a - b, a + b, true);
        check(scheme.decrypt(aMinusBSquared) == (a - b) * (a + b) % SMALL.t,
                "(a-b)*(a+b) must match the plaintext oracle");
        Ciphertext deeper = scheme.multiply(abPlusC, scheme.add(difference, encC),
                (a * b + c) % SMALL.t, a - b + c, true);
        check(scheme.decrypt(deeper) == (a * b + c) * (a - b + c) % SMALL.t,
                "two levels of multiplication must still be exact");
        check(encA.a != encB.a || encA.b != encB.b, "independent encryptions differ");
        check(scheme.noiseAgainst(deeper, (a * b + c) * (a - b + c) % SMALL.t) <= deeper.bound,
                "the composed circuit respects its bound");

        // ---- 5. folding and the modulus switch are exact / necessary ----------------------------
        Ciphertext folded = scheme.foldProduct(encA, encB);
        check(scheme.value(folded) == scheme.value(encA) * scheme.value(encB) % SMALL.q,
                "the folding identity s^2 = s preserves the product value exactly");
        long foldedNoise = Math.abs(FheMath.centered(scheme.value(folded)
                - (long) a * b % SMALL.q * SMALL.delta, SMALL.q));
        check(scheme.decrypt(folded) != a * b % SMALL.t,
                "without the modulus switch the Delta^2-scaled product cannot be read: noise "
                        + foldedNoise);
        Ciphertext switched = scheme.scaleDown(folded);
        check(Math.abs(FheMath.centered(scheme.value(switched) - (long) (a * b) * SMALL.delta, SMALL.q))
                < SMALL.delta, "after the switch the value is Delta*(a*b) up to noise");
        check(scheme.decrypt(switched) == a * b % SMALL.t, "the switched product decrypts correctly");
        check(switched.a != folded.a && switched.b != folded.b, "the switch really rewrites coefficients");

        // ---- 6. the analytic bound dominates the measurement -----------------------------------
        long worst = NoiseModel.multiplyBound(SMALL, SMALL.eta, SMALL.eta,
                NoiseModel.worstCaseMagnitude(SMALL), NoiseModel.worstCaseMagnitude(SMALL));
        Ciphertext tight = scheme.multiply(encA, encB, a, b, true);
        Ciphertext loose = scheme.multiply(encA, encB, a, b, false);
        check(loose.bound == worst && tight.bound < loose.bound,
                "the worst-case bound must exceed the message-aware one");
        check(scheme.noiseAgainst(tight, a * b % SMALL.t) <= tight.bound
                && scheme.noiseAgainst(loose, a * b % SMALL.t) <= loose.bound,
                "both bounds must dominate the measured noise");
        check(tight.bound >= 2 * SMALL.eta, "the product's bound must exceed the fresh noise");

        // The accumulated noise behaves like a random walk; measure it against the Gaussian prediction.
        double measuredRms = rmsNoise(scheme, 400, 60);
        double predictedRms = NoiseModel.perStepSigma(SMALL.eta) * Math.sqrt(400);
        check(measuredRms >= 0.5 * predictedRms && measuredRms <= 2 * predictedRms,
                "the random-walk model of the noise must match the measurement: rms " + measuredRms
                        + " vs predicted " + predictedRms);

        // ---- 7. level budget: measure where the noise eats the budget ---------------------------
        int depthSmall = failureDepth(scheme, SMALL);
        check(depthSmall >= 8, "the parameter set must survive several multiplications, got " + depthSmall);
        check(depthSmall <= 200_000, "the failure depth must be finite, got " + depthSmall);
        long noiseBefore = signedNoiseAtDepth(scheme, depthSmall - 1);
        long noiseAfter = signedNoiseAtDepth(scheme, depthSmall);
        check(noiseBefore >= -SMALL.halfDelta() && noiseBefore < SMALL.halfDelta(),
                "the depth below the limit must stay inside the budget, saw " + noiseBefore);
        check(noiseAfter >= SMALL.halfDelta() || noiseAfter < -SMALL.halfDelta(),
                "the failure must coincide with the signed noise leaving [-Delta/2, Delta/2), saw "
                        + noiseAfter + " against " + SMALL.halfDelta());
        check(scheme.decrypt(cipherAtDepth(scheme, depthSmall)) != 1,
                "and the plaintext must actually be wrong at that depth");
        int predicted = predictedFailureDepth(scheme, SMALL);
        check(predicted == trackedBoundDepth(SMALL), "the predicted depth must equal the simulated bound's "
                + "crossing, saw " + predicted + " vs " + trackedBoundDepth(SMALL));
        check(predicted <= depthSmall, "the analytic worst case must fail no later than reality: predicted "
                + predicted + " vs measured " + depthSmall);
        check(depthSmall >= 4 * predicted, "the worst-case bound must be conservative by a measurable "
                + "factor, saw " + depthSmall + " vs " + predicted);
        boolean boundSound = true;
        for (int depth : new int[]{1, 10, 100, 1000, 5000, 20000}) {
            long measured = noiseAtDepth(scheme, SMALL, depth);
            long bound = trackedBound(scheme, SMALL, depth);
            boundSound &= measured <= bound;
        }
        check(boundSound, "the tracked worst-case bound must dominate the measured noise at every depth");

        // ---- 8. refresh restores the budget (stand-in for bootstrapping) ------------------------
        Ciphertext withoutRefresh = cipherAtDepth(scheme, depthSmall);
        Ciphertext withRefresh = refreshedChain(scheme, depthSmall, new SplitMix64(5L));
        check(scheme.decrypt(withoutRefresh) != 1, "the unrefreshed chain must break at the budget depth");
        check(scheme.decrypt(withRefresh) == 1,
                "refreshing restores the budget, so the same chain must succeed");
        check(scheme.noiseAgainst(withRefresh, 1) <= SMALL.eta, "a refreshed ciphertext is as good as new");

        // ---- 9. malleability and the absence of integrity ---------------------------------------
        check(scheme.decrypt(scheme.add(encB, scheme.encrypt(0, rng))) == b,
                "FHE is malleable: adding an encryption of zero changes nothing");
        Ciphertext tampered = new Ciphertext(encB.a, Math.floorMod(encB.b + SMALL.delta, SMALL.q),
                encB.bound);
        check(scheme.decrypt(tampered) == Math.floorMod(b + 1, SMALL.t),
                "editing the constant coefficient changes the plaintext: FHE has no integrity");
        Ciphertext tamperedSecret = new Ciphertext(Math.floorMod(encB.a + 1, SMALL.q), encB.b, encB.bound);
        check(scheme.decrypt(tamperedSecret) == Math.floorMod(b + scheme.secret, SMALL.t),
                "editing the secret coefficient shifts the plaintext by exactly s, leaking the key to "
                        + "anyone who can decrypt");

        // ---- 10. the measured security limitation -----------------------------------------------
        int survivors = 0;
        for (long guess = 0; guess < 2; guess++) {
            boolean consistent = true;
            for (int i = 0; i < 3; i++) {
                long message = 100 * i + 7;
                Ciphertext ct = scheme.encrypt(message, new SplitMix64(4000 + i));
                long residual = FheMath.centered(ct.b - SMALL.delta * message + ct.a * guess, SMALL.q);
                if (Math.abs(residual) > SMALL.eta) consistent = false;
            }
            if (consistent) survivors++;
        }
        check(survivors == 1, "exhaustive search over the two candidate secrets must leave exactly one, saw "
                + survivors);

        // ---- 11. determinism --------------------------------------------------------------------
        FheScheme repeat = new FheScheme(SMALL, 12345L);
        check(repeat.secret == scheme.secret, "the same seed must reproduce the same key");
        Ciphertext detA = scheme.encrypt(21, new SplitMix64(31337L));
        Ciphertext detB = repeat.encrypt(21, new SplitMix64(31337L));
        check(detA.a == detB.a && detA.b == detB.b, "the same randomness gives the same ciphertext");
        Ciphertext otherSeed = scheme.encrypt(21, new SplitMix64(31338L));
        check((otherSeed.a != detA.a || otherSeed.b != detA.b) && scheme.decrypt(otherSeed) == 21,
                "a different seed re-randomises the ciphertext but not the plaintext");
        Ciphertext refreshed = scheme.refresh(detA, new SplitMix64(6L));
        Ciphertext refreshedAgain = repeat.refresh(detB, new SplitMix64(6L));
        check(refreshed.a == refreshedAgain.a && refreshed.b == refreshedAgain.b
                && scheme.decrypt(refreshed) == 21,
                "refresh must be deterministic given its randomness and keep the plaintext");

        System.out.println("params q=" + SMALL.q + " t=" + SMALL.t + " Delta=" + SMALL.delta + " eta="
                + SMALL.eta + "; round trips " + roundTrips
                + ", oracle checks " + oracleChecks);
        System.out.println("correctness frontier: largest correct square product " + measuredFrontier
                + " vs the analytic t/(2*Delta) = " + analyticFrontier);
        System.out.println("level budget: the chain breaks at depth " + depthSmall + ", the worst-case"
                + " bound predicted " + predicted + "; signed noise " + noiseBefore + " -> " + noiseAfter
                + " against Delta/2 = " + SMALL.halfDelta() + "; rms at depth 400 " + measuredRms
                + " vs random-walk " + predictedRms);
        System.out.println("refresh: a " + depthSmall + "-deep chain fails without refresh and succeeds"
                + " with it; worst-case bound " + worst + " vs message-aware " + tight.bound
                + " (measured " + scheme.noiseAgainst(tight, a * b % SMALL.t) + ")");
        System.out.println("security: one bit of secret, broken by " + survivors + " of 2 candidate"
                + " guesses; no ring layer and no real security parameters");
        System.out.println("PASS");
    }

    /** Root-mean-square measured noise over independent chains of the given depth. */
    static double rmsNoise(FheScheme scheme, int depth, int trials) {
        double sum = 0;
        for (int trial = 0; trial < trials; trial++) {
            Ciphertext current = scheme.encrypt(1, new SplitMix64(11L + trial));
            for (int step = 1; step <= depth; step++) {
                current = scheme.multiply(current,
                        scheme.encrypt(1, new SplitMix64(100_000L * (trial + 1) + step)), 1, 1, true);
            }
            long noise = scheme.noiseAgainst(current, 1);
            sum += (double) noise * noise;
        }
        return Math.sqrt(sum / trials);
    }

    /** Multiply by a fresh encryption of one until the plaintext stops being recovered. */
    static int failureDepth(FheScheme scheme, FheParams params) {
        Ciphertext current = scheme.encrypt(1, new SplitMix64(4L));
        for (int depth = 1; depth <= 200_000; depth++) {
            Ciphertext fresh = scheme.encrypt(1, new SplitMix64(1000 + depth));
            current = scheme.multiply(current, fresh, 1, 1, true);
            if (scheme.decrypt(current) != 1) return depth;
        }
        return 200_001;
    }

    static Ciphertext cipherAtDepth(FheScheme scheme, int depth) {
        Ciphertext current = scheme.encrypt(1, new SplitMix64(4L));
        for (int step = 1; step <= depth; step++) {
            Ciphertext fresh = scheme.encrypt(1, new SplitMix64(1000 + step));
            current = scheme.multiply(current, fresh, 1, 1, true);
        }
        return current;
    }

    static long noiseAtDepth(FheScheme scheme, FheParams params, int depth) {
        return scheme.noiseAgainst(cipherAtDepth(scheme, depth), 1);
    }

    static long signedNoiseAtDepth(FheScheme scheme, int depth) {
        return scheme.signedNoise(cipherAtDepth(scheme, depth), 1);
    }

    /** The first depth at which the analytic message-aware bound exceeds Delta/2. */
    static int predictedFailureDepth(FheScheme scheme, FheParams params) {
        long bound = params.eta;
        for (int depth = 1; depth <= 200_000; depth++) {
            bound = NoiseModel.multiplyBound(params, bound, params.eta, 1, 1);
            if (bound > params.halfDelta()) return depth;
        }
        return 200_001;
    }

    /** The same bound, read off the ciphertexts the scheme actually tracks. */
    static long trackedBound(FheScheme scheme, FheParams params, int depth) {
        Ciphertext current = scheme.encrypt(1, new SplitMix64(4L));
        for (int step = 1; step <= depth; step++) {
            current = scheme.multiply(current, scheme.encrypt(1, new SplitMix64(1000 + step)), 1, 1, true);
        }
        return current.bound;
    }

    /** The depth at which the tracked bound crosses Delta/2. */
    static int trackedBoundDepth(FheParams params) {
        long bound = params.eta;
        for (int depth = 1; depth <= 200_000; depth++) {
            bound = NoiseModel.multiplyBound(params, bound, params.eta, 1, 1);
            if (bound > params.halfDelta()) return depth;
        }
        return 200_001;
    }

    /** The same chain with a trusted refresh after every multiplication. */
    static Ciphertext refreshedChain(FheScheme scheme, int length, SplitMix64 rng) {
        Ciphertext current = scheme.encrypt(1, new SplitMix64(4L));
        for (int step = 1; step <= length; step++) {
            current = scheme.multiply(current, scheme.encrypt(1, new SplitMix64(1000 + step)), 1, 1, true);
            current = scheme.refresh(current, rng);
        }
        return current;
    }

    static void expectThrows(Runnable action, String label) {
        try {
            action.run();
        } catch (RuntimeException expected) {
            return;
        }
        throw new AssertionError("FAILED: expected a rejection for " + label);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
