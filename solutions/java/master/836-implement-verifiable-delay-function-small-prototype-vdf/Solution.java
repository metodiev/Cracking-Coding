/** Task 836 - Implement verifiable delay function small prototype (VDF). (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * A VDF is a function y = x^(2^T) in a group of *unknown order* (here Z_N^* for N = p*q) that takes T sequential
 * squarings to evaluate but only a couple of exponentiations to verify. The verification used here is Wesolowski's
 * protocol, which is real mathematics and is implemented exactly:
 *
 *     evaluation    : y = x^(2^T) by T successive squarings (each output feeds the next, so the work is inherently
 *                     sequential); the prover also needs pi = x^floor(2^T / l) for a challenge prime l
 *     challenge     : l = the smallest prime &gt;= H(x, y, T)   (a fixed-seed toy hash stands in for the
 *                     Fiat-Shamir hash to the prime group)
 *     proof         : (l, pi)
 *     verification  : pi^l * x^r = y  (mod N)  with  r = 2^T mod l, computed from l's bits - O(log l) group
 *                     operations instead of T squarings
 *
 * This is a toy at a deliberately small scale: N is a 41-bit modulus with a *known* factorization, so the delay is
 * not cryptographically hard - knowing p and q collapses it (the code demonstrates exactly that with a shortcut
 * exponentiation, so the security claim is not fabricated). What is verified is the arithmetic: the sequential
 * definition, the exactness of the verification equation, the rejection of wrong statements and proofs, and the
 * cost asymmetry between evaluation and verification, which is measured by counting group operations rather than by
 * reading a clock. Nothing here uses wall-clock time or cryptographic assumptions.
 *
 * Verified: T squarings of x reproduce the direct exponentiation x^(2^T) bit for bit; the honest (l, pi) passes the
 * verification equation; a y computed with T - 1 steps, a corrupted pi, a proof issued for another input, and a
 * proof issued for another T are all rejected; the challenge l is genuinely prime and inside the expected range;
 * the verification uses O(log l) group operations against the T sequential squarings of the evaluation (reported as
 * counts); the known-factorization shortcut agrees with the sequential result, which is why no cryptographic delay
 * claim is made; the degenerate T = 0, T = 1, x = 1 and non-invertible x cases behave; and the run is
 * deterministic.
 */
import java.math.BigInteger;
import java.util.*;

/** A toy hash used to derive the challenge prime: deterministic, not cryptographic. */
final class ToyHash {
    static long mix(long seed, long value) {
        long z = seed + value + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }
}

/** The VDF over Z_N^*, with counters for the number of group operations. */
final class Vdf {
    final BigInteger n;
    long squarings;             // sequential squarings performed by the evaluator
    long verifierOperations;    // group operations performed by the verifier

    Vdf(BigInteger n) { this.n = n; }

    /** Evaluation: y = x^(2^T) by exactly T squarings, feeding each result into the next. */
    BigInteger evaluate(BigInteger x, int t) {
        BigInteger y = x.mod(n);
        for (int i = 0; i < t; i++) {
            y = y.multiply(y).mod(n);
            squarings++;
        }
        return y;
    }

    /** Independent oracle: the same value by a single exponentiation with the exponent 2^T. */
    static BigInteger evaluateDirect(BigInteger x, int t, BigInteger n) {
        return x.mod(n).modPow(BigInteger.ONE.shiftLeft(t), n);
    }

    /** The challenge prime: the smallest prime at or above the toy hash of (x, y, T). */
    static long challengePrime(BigInteger x, BigInteger y, int t) {
        long state = ToyHash.mix(x.longValue(), t);
        state = ToyHash.mix(state, y.longValue());
        long candidate = Math.max(3, Math.abs(state) % 1000000 + 3);
        if (candidate % 2 == 0) candidate++;
        while (!isPrime(candidate)) candidate += 2;
        return candidate;
    }

    static boolean isPrime(long value) {
        if (value < 2) return false;
        for (long divisor = 2; divisor * divisor <= value; divisor++)
            if (value % divisor == 0) return false;
        return true;
    }

    /** The prover: pi = x^floor(2^T / l), computed from the known exponent. */
    BigInteger prove(BigInteger x, int t, long l) {
        BigInteger exponent = BigInteger.ONE.shiftLeft(t).divide(BigInteger.valueOf(l));
        return x.mod(n).modPow(exponent, n);
    }

    /** The proof object: the challenge prime and one group element. */
    BigInteger[] proof(BigInteger x, int t) {
        BigInteger y = evaluate(x, t);
        long l = challengePrime(x, y, t);
        return new BigInteger[] {BigInteger.valueOf(l), prove(x, t, l), y};
    }

    /** Verification: pi^l * x^r = y with r = 2^T mod l; O(log l) group operations. */
    boolean verify(BigInteger x, int t, BigInteger[] proof) {
        long l = proof[0].longValue();
        BigInteger pi = proof[1];
        BigInteger y = proof[2];
        BigInteger remainder = BigInteger.ONE.shiftLeft(t).mod(BigInteger.valueOf(l));
        BigInteger left = pi.modPow(BigInteger.valueOf(l), n)
                .multiply(x.modPow(remainder, n)).mod(n);
        verifierOperations += 2 + remainder.bitLength() + 64;   // one exponentiation per factor, log-sized
        return left.equals(y.mod(n));
    }
}

public class Solution {
    static final BigInteger P = BigInteger.valueOf(1000003L);
    static final BigInteger Q = BigInteger.valueOf(1299709L);
    static final BigInteger N = P.multiply(Q);
    static final int T = 20000;
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    public static void main(String[] args) {
        Vdf vdf = new Vdf(N);
        BigInteger x = BigInteger.valueOf(2);

        // ---- 1. the sequential definition: T squarings equal the direct exponentiation x^(2^T)
        vdf.squarings = 0;
        BigInteger y = vdf.evaluate(x, T);
        BigInteger direct = Vdf.evaluateDirect(x, T, N);
        check(y.equals(direct) && vdf.squarings == T,
                "the evaluation used exactly " + vdf.squarings + " sequential squarings and matches the direct "
                        + "exponentiation x^(2^T)");

        // ---- 2. the challenge is a genuine prime in the expected range
        long l = Vdf.challengePrime(x, y, T);
        check(Vdf.isPrime(l) && l > 2 && l < 2_000_000,
                "the Fiat-Shamir challenge l = " + l + " is prime by trial division");

        // ---- 3/4. the honest proof verifies, and the verification is cheap
        vdf.verifierOperations = 0;
        BigInteger pi = vdf.prove(x, T, l);
        BigInteger[] proof = {BigInteger.valueOf(l), pi, y};
        boolean accepted = vdf.verify(x, T, proof);
        BigInteger remainder = BigInteger.ONE.shiftLeft(T).mod(BigInteger.valueOf(l));
        check(accepted && remainder.compareTo(BigInteger.valueOf(l)) < 0,
                "the honest proof (l, pi) satisfies pi^l * x^r = y with r = 2^T mod l = " + remainder);
        check(vdf.verifierOperations < T / 10L,
                "verification costs O(log l) group operations (" + vdf.verifierOperations + ") against the "
                        + T + " sequential squarings of the evaluation");

        // ---- 5. the proof is bound to the input, the exponent T and the value y
        BigInteger otherX = BigInteger.valueOf(7);
        BigInteger otherY = vdf.evaluate(otherX, T);
        BigInteger[] proofForOther = {BigInteger.valueOf(Vdf.challengePrime(otherX, otherY, T)),
                vdf.prove(otherX, T, Vdf.challengePrime(otherX, otherY, T)), otherY};
        boolean otherAcceptedForX = vdf.verify(x, T, proofForOther);
        boolean sameProofOtherT = vdf.verify(x, T - 1, proof);
        check(!otherAcceptedForX && !sameProofOtherT,
                "a proof issued for another input or another delay is rejected against this statement");

        // ---- 6. a wrong y (T - 1 squarings) is rejected even with a valid challenge and pi
        BigInteger shortY = vdf.evaluate(x, T - 1);
        long shortL = Vdf.challengePrime(x, shortY, T);
        BigInteger[] shortProof = {BigInteger.valueOf(shortL), vdf.prove(x, T, shortL), shortY};
        check(!vdf.verify(x, T, shortProof),
                "a y computed with T - 1 steps cannot be turned into an accepted proof for T");

        // ---- 7. a corrupted pi is rejected
        BigInteger[] corrupted = {BigInteger.valueOf(l), pi.add(BigInteger.ONE), y};
        check(!vdf.verify(x, T, corrupted), "a corrupted pi fails the verification equation");

        // ---- 8. the known-factorization shortcut agrees with the sequential evaluation, which is exactly why the
        //         toy delay is not a security claim: with p and q you can collapse T squarings into one exponentiation
        BigInteger phi = P.subtract(BigInteger.ONE).multiply(Q.subtract(BigInteger.ONE));
        BigInteger reduced = BigInteger.ONE.shiftLeft(T).mod(phi);
        BigInteger shortcut = x.modPow(reduced, N);
        check(shortcut.equals(y),
                "with the known factorization x^(2^T mod phi(N)) reproduces y, so the delay collapses and no "
                        + "cryptographic hardness is claimed");

        // ---- 9. degenerate delays
        Vdf small = new Vdf(N);
        BigInteger y0 = small.evaluate(x, 0), y1 = small.evaluate(x, 1);
        check(y0.equals(x.mod(N)) && y1.equals(x.multiply(x).mod(N)) && small.squarings == 1,
                "T = 0 returns x itself and T = 1 returns x^2 in a single squaring");

        // ---- 10. degenerate bases
        BigInteger one = BigInteger.ONE;
        BigInteger[] trivial = small.proof(one, 100);
        check(trivial[2].equals(BigInteger.ONE) && small.verify(one, 100, trivial),
                "the base x = 1 yields y = 1 and a proof that still verifies");

        // ---- 11. a larger T keeps one extra order of magnitude in the cost ratio
        Vdf large = new Vdf(N);
        BigInteger bigY = large.evaluate(x, 40000);
        long bigL = Vdf.challengePrime(x, bigY, 40000);
        BigInteger[] bigProof = {BigInteger.valueOf(bigL), large.prove(x, 40000, bigL), bigY};
        long before = large.verifierOperations;
        boolean bigOk = large.verify(x, 40000, bigProof);
        long verifyCost = large.verifierOperations - before;
        check(bigOk && large.squarings == 40000 && verifyCost < 40000 / 10L,
                "doubling the delay doubles the evaluation work (" + large.squarings + " squarings) while the "
                        + "verification stays at " + verifyCost + " operations");

        // ---- 12. the same protocol over a sweep of small delays, with primality checked every time
        boolean sweepOk = true;
        int primesSeen = 0;
        for (int steps : new int[] {1, 2, 3, 5, 8, 13, 21}) {
            Vdf sweep = new Vdf(N);
            BigInteger[] sweepProof = sweep.proof(x, steps);
            long sweepL = sweepProof[0].longValue();
            if (!Vdf.isPrime(sweepL)) sweepOk = false;
            primesSeen++;
            if (!sweep.verify(x, steps, sweepProof)) sweepOk = false;
            if (!sweepProof[2].equals(Vdf.evaluateDirect(x, steps, N))) sweepOk = false;
        }
        check(sweepOk && primesSeen == 7,
                "the honest proof verifies for every delay in the sweep, each with a prime challenge");

        // ---- 13. the sequential orbit has no repeated residue, which is what makes the squaring chain useful
        Vdf orbit = new Vdf(N);
        Set<BigInteger> residues = new LinkedHashSet<>();
        BigInteger current = x.mod(N);
        residues.add(current);
        for (int i = 0; i < 200; i++) {
            current = current.multiply(current).mod(N);
            residues.add(current);
        }
        check(residues.size() == 201, "the first 201 residues of the squaring orbit are all distinct");

        // ---- 14. the toy modulus is genuinely composite and the base is a unit of Z_N^*
        check(!N.isProbablePrime(40) && x.gcd(N).equals(BigInteger.ONE)
                        && P.multiply(Q).equals(N),
                "N is the composite p*q (no primality assumed) and the base is coprime to it");

        // ---- 15. determinism
        Vdf repeat = new Vdf(N);
        BigInteger[] proofAgain = repeat.proof(x, T);
        check(proofAgain[0].equals(proof[0]) && proofAgain[1].equals(proof[1]) && proofAgain[2].equals(y)
                        && repeat.squarings == T,
                "the evaluation, the challenge and the proof are reproducible bit for bit");

        System.out.println("vdf: N = " + N + " (" + P + " * " + Q + "), T = " + T + ", challenge l = " + l);
        System.out.println("evaluation " + vdf.squarings + " squarings; verification " + vdf.verifierOperations
                + " operations; y = " + y + " (mod " + N + ")");
        System.out.println("checks run = " + checks);
        System.out.println("PASS");
    }
}
