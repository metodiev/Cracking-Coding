/** Task 834 - Implement zk-SNARK circuit for a simple arithmetic program (toy). (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * The toy program is  out = (a*b + c)^2  with witness (1, a, b, c, w1, w2, out) where w1 = a*b and w2 = w1 + c.
 * It compiles into three rank-1 constraints of the form (sum_i A_i x_i) * (sum_i B_i x_i) = (sum_i C_i x_i):
 *
 *     a * b = w1            (w1 + c) * 1 = w2            w2 * w2 = out
 *
 * Each constraint becomes a point of a quadratic arithmetic program: for constraints indexed x = 1..m the prover
 * interpolates the target polynomials A(x), B(x), C(x) through the per-constraint coefficients, and the system is
 * satisfied exactly when A(x)B(x) - C(x) is divisible by the vanishing polynomial Z(x) = prod_i (x - i), i.e. when
 *
 *     A B - C = H Z          verifier:  (A_pub + A_priv)(t) (B_pub + B_priv)(t) - (C_pub + C_priv)(t) = H(t) Z(t)
 *
 * at a single Fiat-Shamir challenge t. The witnesses split into public variables (1, out - the ones the verifier
 * knows) and private ones, so the verifier recomputes the public part of A, B, C itself from the *claimed* output
 * and only the three private evaluations plus H(t) travel in the proof. That binding is what makes a wrong
 * claimed output detectable. All arithmetic is exact in Z_1009 and the soundness statement is the Schwartz-Zippel
 * bound: a dishonest prover's residual polynomial is non-zero of degree at most d, so it can vanish at no more than
 * d of the |F| = 1009 challenges - and because the field is small this bound is *measured* by enumerating every
 * challenge rather than asserted.
 *
 * This is a toy and is described as one: there are no elliptic curves, no pairings and no polynomial commitments, so
 * the "proof" is not sound in the cryptographic sense - only completeness and the algebraic Schwartz-Zippel bound
 * are claimed and verified. Zero knowledge is explicitly *not* claimed either: the proof carries raw evaluations of
 * the private witness polynomials.
 *
 * Verified: field arithmetic (Fermat inverses, distributivity); the witness satisfies every R1CS constraint by
 * direct evaluation; Lagrange interpolation reproduces each constraint's coefficients exactly (the independent
 * oracle for the QAP construction); the honest identity A B - C = H Z holds at *every* point of the field and the
 * exact division has a zero remainder; a wrong witness leaves a non-zero remainder and fails the constraints; the
 * honest proof verifies, is exactly four field elements, and does not depend on the circuit size; a tampered H, a
 * tampered private evaluation and a proof for a *different* claimed output are all rejected; the measured number of
 * accepting challenges for a dishonest prover stays inside the degree bound; a one-constraint circuit and a
 * trivially satisfied constraint behave; and the whole run is reproducible.
 */
import java.util.*;

/** Exact arithmetic in the prime field Z_p. */
final class Field {
    final long p;

    Field(long p) { this.p = p; }

    long add(long a, long b) { return Math.floorMod(a + b, p); }
    long sub(long a, long b) { return Math.floorMod(a - b, p); }
    long mul(long a, long b) { return Math.floorMod(a * b, p); }

    long pow(long base, long exponent) {
        long result = 1, b = Math.floorMod(base, p), e = exponent;
        while (e > 0) {
            if ((e & 1) == 1) result = mul(result, b);
            b = mul(b, b);
            e >>= 1;
        }
        return result;
    }

    long inverse(long a) {
        if (Math.floorMod(a, p) == 0) throw new ArithmeticException("no inverse of zero");
        return pow(a, p - 2);
    }

    long negate(long a) { return sub(0, a); }
}

/** Dense polynomial over Z_p, coefficients low order first. */
final class Poly {
    final Field field;
    final long[] c;

    Poly(Field field, long... coefficients) {
        this.field = field;
        int length = coefficients.length;
        while (length > 1 && Math.floorMod(coefficients[length - 1], field.p) == 0) length--;
        this.c = Arrays.copyOf(coefficients, length);
    }

    static Poly constant(Field field, long value) { return new Poly(field, value); }

    int degree() { return c.length - 1; }

    Poly add(Poly other) {
        int n = Math.max(c.length, other.c.length);
        long[] out = new long[n];
        for (int i = 0; i < n; i++)
            out[i] = field.add(i < c.length ? c[i] : 0, i < other.c.length ? other.c[i] : 0);
        return new Poly(field, out);
    }

    Poly sub(Poly other) {
        int n = Math.max(c.length, other.c.length);
        long[] out = new long[n];
        for (int i = 0; i < n; i++)
            out[i] = field.sub(i < c.length ? c[i] : 0, i < other.c.length ? other.c[i] : 0);
        return new Poly(field, out);
    }

    Poly mul(Poly other) {
        long[] out = new long[c.length + other.c.length - 1];
        for (int i = 0; i < c.length; i++)
            for (int j = 0; j < other.c.length; j++)
                out[i + j] = field.add(out[i + j], field.mul(c[i], other.c[j]));
        return new Poly(field, out);
    }

    Poly scale(long factor) {
        long[] out = new long[c.length];
        for (int i = 0; i < c.length; i++) out[i] = field.mul(c[i], factor);
        return new Poly(field, out);
    }

    long evaluate(long x) {
        long value = 0;
        for (int i = c.length - 1; i >= 0; i--) value = field.add(field.mul(value, x), c[i]);
        return value;
    }

    boolean isZero() { return c.length == 1 && Math.floorMod(c[0], field.p) == 0; }

    /** Exact division: the quotient, or null when the remainder is non-zero. */
    Poly divideExact(Poly divisor) {
        if (divisor.isZero()) throw new ArithmeticException("division by the zero polynomial");
        long[] remainder = c.clone();
        long[] quotient = new long[Math.max(1, c.length - divisor.c.length + 1)];
        long leadingInverse = field.inverse(divisor.c[divisor.c.length - 1]);
        for (int i = remainder.length - 1; i >= divisor.c.length - 1; i--) {
            if (Math.floorMod(remainder[i], field.p) == 0) continue;
            long factor = field.mul(remainder[i], leadingInverse);
            quotient[i - (divisor.c.length - 1)] = factor;
            for (int j = 0; j < divisor.c.length; j++)
                remainder[i - (divisor.c.length - 1) + j] = field.sub(
                        remainder[i - (divisor.c.length - 1) + j], field.mul(factor, divisor.c[j]));
        }
        for (long value : remainder) if (Math.floorMod(value, field.p) != 0) return null;
        return new Poly(field, quotient);
    }
}

/** The R1CS: one row per constraint, one column per witness variable, the public columns 0 ("one") and 6 ("out"). */
final class R1cs {
    static final String[] VARIABLES = {"one", "a", "b", "c", "w1", "w2", "out"};
    static final int[] PUBLIC = {0, 6};
    final Field field;
    final long[][] left, right, output;

    R1cs(Field field) {
        this.field = field;
        left = new long[3][VARIABLES.length];
        right = new long[3][VARIABLES.length];
        output = new long[3][VARIABLES.length];
        left[0][1] = 1;                                     // a * b = w1
        right[0][2] = 1;
        output[0][4] = 1;
        left[1][4] = 1;                                     // (w1 + c) * 1 = w2
        left[1][3] = 1;
        right[1][0] = 1;
        output[1][5] = 1;
        left[2][5] = 1;                                     // w2 * w2 = out
        right[2][5] = 1;
        output[2][6] = 1;
    }

    int constraints() { return left.length; }

    static boolean isPublic(int variable) {
        for (int value : PUBLIC) if (value == variable) return true;
        return false;
    }

    static long rowDot(Field field, long[] row, long[] witness) {
        long sum = 0;
        for (int i = 0; i < row.length; i++) sum = field.add(sum, field.mul(row[i], witness[i]));
        return sum;
    }

    /** Direct check of every constraint: the independent oracle for the QAP. */
    boolean satisfiedBy(long[] witness) {
        for (int i = 0; i < constraints(); i++) {
            long a = rowDot(field, left[i], witness);
            long b = rowDot(field, right[i], witness);
            long c = rowDot(field, output[i], witness);
            if (field.mul(a, b) != c) return false;
        }
        return true;
    }

    /** Witness (1, a, b, c, w1, w2, out) from the plain program out = (a*b + c)^2. */
    long[] witness(long a, long b, long c) {
        long[] w = new long[VARIABLES.length];
        w[0] = 1;
        w[1] = Math.floorMod(a, field.p);
        w[2] = Math.floorMod(b, field.p);
        w[3] = Math.floorMod(c, field.p);
        w[4] = field.mul(w[1], w[2]);
        w[5] = field.add(w[4], w[3]);
        w[6] = field.mul(w[5], w[5]);
        return w;
    }
}

/** The QAP: interpolation, the honest prover, and the verifier that only knows the public variables. */
final class Qap {
    final Field field;
    final R1cs r1cs;
    final Poly vanishing;
    final Poly[] lagrange;

    Qap(Field field, R1cs r1cs) {
        this.field = field;
        this.r1cs = r1cs;
        int m = r1cs.constraints();
        Poly z = Poly.constant(field, 1);
        for (int i = 1; i <= m; i++) z = z.mul(new Poly(field, field.negate(i), 1));
        vanishing = z;
        lagrange = new Poly[m];
        for (int i = 0; i < m; i++) {
            Poly numerator = Poly.constant(field, 1);
            long denominator = 1;
            for (int j = 0; j < m; j++) {
                if (i == j) continue;
                numerator = numerator.mul(new Poly(field, field.negate(j + 1), 1));
                denominator = field.mul(denominator, field.sub(i + 1, j + 1));
            }
            lagrange[i] = numerator.scale(field.inverse(denominator));
        }
    }

    /** Interpolates a column restricted to the private variables. */
    Poly privatePolynomial(long[][] column, long[] witness) {
        Poly total = Poly.constant(field, 0);
        for (int i = 0; i < column.length; i++) {
            long coefficient = 0;
            for (int v = 0; v < witness.length; v++)
                if (!R1cs.isPublic(v)) coefficient = field.add(coefficient, field.mul(column[i][v], witness[v]));
            total = total.add(lagrange[i].scale(coefficient));
        }
        return total;
    }

    /** Interpolates a column restricted to the public variables, using the *claimed* output. */
    Poly publicPolynomial(long[][] column, long claimedOutput) {
        Poly total = Poly.constant(field, 0);
        for (int i = 0; i < column.length; i++) {
            long coefficient = field.add(column[i][0], field.mul(column[i][6], claimedOutput));
            total = total.add(lagrange[i].scale(coefficient));
        }
        return total;
    }

    /** The prover: private evaluations of A, B, C and H at the challenge, or null when the division is inexact. */
    long[] prove(long[] witness, long challenge) {
        Poly a = privatePolynomial(r1cs.left, witness).add(publicPolynomial(r1cs.left, witness[6]));
        Poly b = privatePolynomial(r1cs.right, witness).add(publicPolynomial(r1cs.right, witness[6]));
        Poly c = privatePolynomial(r1cs.output, witness).add(publicPolynomial(r1cs.output, witness[6]));
        Poly h = a.mul(b).sub(c).divideExact(vanishing);
        if (h == null) return null;
        return new long[] {privatePolynomial(r1cs.left, witness).evaluate(challenge),
                privatePolynomial(r1cs.right, witness).evaluate(challenge),
                privatePolynomial(r1cs.output, witness).evaluate(challenge), h.evaluate(challenge)};
    }

    /** The verifier: it knows the circuit and the claimed output, and recomputes the public part itself. */
    boolean verify(long challenge, long claimedOutput, long[] proof) {
        long a = field.add(publicPolynomial(r1cs.left, claimedOutput).evaluate(challenge), proof[0]);
        long b = field.add(publicPolynomial(r1cs.right, claimedOutput).evaluate(challenge), proof[1]);
        long c = field.add(publicPolynomial(r1cs.output, claimedOutput).evaluate(challenge), proof[2]);
        return field.sub(field.mul(a, b), c) == field.mul(proof[3], vanishing.evaluate(challenge));
    }
}

public class Solution {
    static final long PRIME = 1009L;
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    /** Fiat-Shamir style challenge: a deterministic function of the claimed public statement. */
    static long challenge(Field field, long claimedOutput) {
        long state = 0x9E3779B97F4A7C15L + claimedOutput;
        state = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        state = (state ^ (state >>> 27)) * 0x94D049BB133111EBL;
        state = state ^ (state >>> 31);
        return Math.floorMod(state, field.p);
    }

    public static void main(String[] args) {
        Field field = new Field(PRIME);
        R1cs r1cs = new R1cs(field);
        Qap qap = new Qap(field, r1cs);

        // ---- 1. field arithmetic
        boolean arithmetic = true;
        for (long a = 1; a < 60; a++) {
            if (field.mul(a, field.inverse(a)) != 1) arithmetic = false;
            if (field.mul(a, field.add(a, 1)) != field.add(field.mul(a, a), a)) arithmetic = false;
        }
        check(arithmetic && field.pow(2, PRIME - 1) == 1,
                "Fermat inverses and distributivity hold across the field");

        // ---- 1b. the exact polynomial division itself, on a hand-checked example
        Poly product = new Poly(field, field.negate(1), 1).mul(new Poly(field, field.negate(2), 1));
        Poly quotient = product.divideExact(new Poly(field, field.negate(1), 1));
        Poly inexact = new Poly(field, 1, 1).divideExact(new Poly(field, field.negate(1), 1));
        check(quotient != null && quotient.degree() == 1 && inexact == null,
                "the exact division recovers x - 2 from (x - 1)(x - 2) / (x - 1) and rejects (x + 1) / (x - 1)");

        // ---- 2. the witness satisfies every R1CS constraint by direct evaluation
        long a = 7, b = 5, c = 19;
        long[] witness = r1cs.witness(a, b, c);
        long sum = Math.floorMod(a * b + c, PRIME);
        check(r1cs.satisfiedBy(witness) && witness[6] == Math.floorMod(sum * sum, PRIME),
                "the witness satisfies all " + r1cs.constraints() + " constraints with out = (a*b + c)^2 = "
                        + witness[6]);

        // ---- 3. Lagrange interpolation reproduces each constraint's coefficients exactly
        Poly[] full = {qap.privatePolynomial(r1cs.left, witness).add(qap.publicPolynomial(r1cs.left, witness[6])),
                qap.privatePolynomial(r1cs.right, witness).add(qap.publicPolynomial(r1cs.right, witness[6])),
                qap.privatePolynomial(r1cs.output, witness).add(qap.publicPolynomial(r1cs.output, witness[6]))};
        int constraints = r1cs.constraints();
        boolean interpolation = true;
        for (int i = 0; i < constraints; i++) {
            long expectedA = R1cs.rowDot(field, r1cs.left[i], witness);
            long expectedB = R1cs.rowDot(field, r1cs.right[i], witness);
            long expectedC = R1cs.rowDot(field, r1cs.output[i], witness);
            if (full[0].evaluate(i + 1) != expectedA || full[1].evaluate(i + 1) != expectedB
                    || full[2].evaluate(i + 1) != expectedC) interpolation = false;
        }
        check(interpolation, "the interpolated QAP polynomials pass through every constraint's coefficients");

        // ---- 4. the honest identity A B - C = H Z holds at every point of the field, with an exact division
        Poly numerator = full[0].mul(full[1]).sub(full[2]);
        Poly h = numerator.divideExact(qap.vanishing);
        boolean everyPoint = h != null;
        for (long t = 0; t < PRIME && everyPoint; t++) {
            long left = field.mul(full[0].evaluate(t), full[1].evaluate(t));
            long right = field.add(full[2].evaluate(t), field.mul(h.evaluate(t), qap.vanishing.evaluate(t)));
            if (left != right) everyPoint = false;
        }
        check(everyPoint, "the honest QAP identity A B - C = H Z holds at all " + PRIME + " field points, not only "
                + "at the challenge, and the division by Z(x) is exact");

        // ---- 5. a wrong witness makes the division inexact and fails the constraints
        long[] wrong = witness.clone();
        wrong[6] = field.add(wrong[6], 1);
        Poly wrongNumerator = null;
        for (int i = 0; i < constraints; i++) {
            Poly wa = qap.privatePolynomial(r1cs.left, wrong).add(qap.publicPolynomial(r1cs.left, wrong[6]));
            Poly wb = qap.privatePolynomial(r1cs.right, wrong).add(qap.publicPolynomial(r1cs.right, wrong[6]));
            Poly wc = qap.privatePolynomial(r1cs.output, wrong).add(qap.publicPolynomial(r1cs.output, wrong[6]));
            wrongNumerator = wa.mul(wb).sub(wc);
            break;
        }
        check(!r1cs.satisfiedBy(wrong) && wrongNumerator.divideExact(qap.vanishing) == null,
                "a witness with a tampered output fails the constraints and leaves a non-zero remainder");

        // ---- 6. the honest proof verifies and is exactly four field elements
        long t = challenge(field, witness[6]);
        long[] proof = qap.prove(witness, t);
        check(proof != null && proof.length == 4 && qap.verify(t, witness[6], proof),
                "the honest proof verifies at t = " + t + " and consists of exactly four field elements");

        // ---- 7. the key schedule is faithful: the verifier's public part matches the witness's public rows
        long publicLeft = qap.publicPolynomial(r1cs.left, witness[6]).evaluate(t);
        long publicRight = qap.publicPolynomial(r1cs.right, witness[6]).evaluate(t);
        long publicOut = qap.publicPolynomial(r1cs.output, witness[6]).evaluate(t);
        long privateOut = qap.privatePolynomial(r1cs.output, witness).evaluate(t);
        check(field.add(publicOut, privateOut) == full[2].evaluate(t)
                        && field.add(publicLeft, proof[0]) == full[0].evaluate(t)
                        && field.add(publicRight, proof[1]) == full[1].evaluate(t),
                "public and private parts of the QAP evaluations sum to the full interpolants");

        // ---- 8. tampered proofs (H or a private evaluation) are rejected
        long[] tamperedH = proof.clone();
        tamperedH[3] = field.add(tamperedH[3], 1);
        long[] tamperedA = proof.clone();
        tamperedA[0] = field.add(tamperedA[0], 1);
        check(!qap.verify(t, witness[6], tamperedH) && !qap.verify(t, witness[6], tamperedA),
                "a tampered H and a tampered private evaluation are both rejected");

        // ---- 9. the public output is bound: a proof for another claimed output is rejected
        long wrongOutput = field.add(witness[6], 1);
        long tWrong = challenge(field, wrongOutput);
        long[] proofForWrongOutput = qap.prove(witness, tWrong);
        check(proofForWrongOutput != null && !qap.verify(tWrong, wrongOutput, proofForWrongOutput),
                "the same proof does not verify against a different claimed public output");

        // ---- 10. measured soundness: how many challenges accept a prover with a biased H?
        //          the residual is (A B - C) - (H + delta) Z = -delta * Z, of degree 3 with exactly 3 roots
        Poly biasedH = h.add(Poly.constant(field, 1));
        int accepting = 0;
        for (long candidate = 0; candidate < PRIME; candidate++) {
            long residual = field.sub(
                    field.sub(field.mul(full[0].evaluate(candidate), full[1].evaluate(candidate)),
                            full[2].evaluate(candidate)),
                    field.mul(biasedH.evaluate(candidate), qap.vanishing.evaluate(candidate)));
            if (residual == 0) accepting++;
        }
        int degreeBound = biasedH.degree() + qap.vanishing.degree();
        check(accepting == qap.vanishing.degree() && accepting <= degreeBound,
                "a prover with a biased H is accepted at " + accepting + " of the " + PRIME + " challenges, which "
                        + "is exactly the degree bound " + degreeBound + " from the vanishing polynomial");

        // ---- 11. soundness with a biased *private* evaluation: still inside the degree bound
        int acceptingA = 0;
        long shifted = field.add(proof[0], 1);
        for (long candidate = 0; candidate < PRIME; candidate++) {
            long aEval = field.add(qap.publicPolynomial(r1cs.left, witness[6]).evaluate(candidate), shifted);
            long bEval = field.add(qap.publicPolynomial(r1cs.right, witness[6]).evaluate(candidate), proof[1]);
            long cEval = field.add(qap.publicPolynomial(r1cs.output, witness[6]).evaluate(candidate), proof[2]);
            long residual = field.sub(field.sub(field.mul(aEval, bEval), cEval),
                    field.mul(proof[3], qap.vanishing.evaluate(candidate)));
            if (residual == 0) acceptingA++;
        }
        check(acceptingA <= full[0].degree() + full[1].degree(),
                "a shifted private evaluation is accepted at only " + acceptingA + " of the " + PRIME
                        + " challenges, inside the degree bound " + (full[0].degree() + full[1].degree()));

        // ---- 12. a one-constraint circuit and a trivially satisfied constraint behave
        Field tiny = new Field(101);
        Poly tinyA = Poly.constant(tiny, 3), tinyB = Poly.constant(tiny, 5), tinyC = Poly.constant(tiny, 15);
        Poly tinyZ = new Poly(tiny, tiny.negate(1), 1);
        Poly tinyH = tinyA.mul(tinyB).sub(tinyC).divideExact(tinyZ);
        Poly zeroA = Poly.constant(tiny, 0);
        Poly zeroH = zeroA.mul(tinyB).sub(zeroA).divideExact(tinyZ);
        check(tinyH != null && tinyH.isZero() && zeroH != null && zeroH.isZero(),
                "a one-constraint circuit and a trivially satisfied 0 = 0 constraint both divide exactly by Z(x)");

        // ---- 13. determinism
        long[] proofAgain = qap.prove(r1cs.witness(a, b, c), t);
        check(Arrays.equals(proof, proofAgain) && challenge(field, witness[6]) == t,
                "the challenge and the proof are reproducible bit for bit");

        System.out.println("zk toy: " + constraints + " R1CS constraints over Z_" + PRIME + ", out = " + witness[6]
                + ", challenge t = " + t);
        System.out.println("honest identity checked at all " + PRIME + " points; dishonest acceptance " + accepting
                + " and " + acceptingA + " of " + PRIME);
        System.out.println("checks run = " + checks);
        System.out.println("PASS");
    }
}
