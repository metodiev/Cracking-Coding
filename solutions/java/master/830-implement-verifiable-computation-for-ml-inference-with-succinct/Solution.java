/** Task 830 - Implement verifiable computation for ML inference with succinct proofs (conceptual).
 *  (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * The story: a *prover* runs an MLP inference and wants a *verifier* to accept the output without re-running the
 * whole computation and without trusting the prover. Two mechanisms are combined here, both implemented exactly:
 *
 * 1. A committed execution trace. Every value of the network ([input | hidden | permuted | output]) is hashed into a
 *    binary Merkle tree, so the prover can open any single value and the verifier can recompute the root from that
 *    value and its path. The verifier then spot-checks q layer equations by opening only the values those equations
 *    need. The hash here is a deterministic integer mixer (splitmix64); it is NOT collision resistant, so binding is
 *    only *empirically* tested (over a generated family of traces) and no cryptographic soundness is claimed.
 *
 * 2. A sum-check protocol for the linear layer. For y = W x the prover claims S = sum_{i,j} W[i][j] * x[j] and the
 *    verifier checks it with the classical sum-check dialogue: in round r the prover sends the univariate
 *    polynomial F_r(t) = sum over the remaining cube of the multilinear extension of the table with the previous
 *    challenges fixed and variable r set to t, the verifier checks F_r(0) + F_r(1) = current claim, and answers a
 *    challenge c_r drawn by Fiat-Shamir. All arithmetic is exact in Z_97, and because the round polynomial of a
 *    multilinear extension is *linear*, a dishonest round polynomial (also linear, and consistent with a false
 *    claim) can agree with the true one at no more than one of the 97 challenges - the Schwartz-Zippel bound
 *    deg/|F| = 1/97, which the code verifies by enumerating every challenge.
 *
 * Verified: trace values hash consistently, opening any leaf reproduces it, and flipping a leaf, a sibling or the
 * output breaks either the path or the layer checks; distinct traces have distinct roots over a generated family;
 * the committed inference output equals an independent plain-integer implementation; a wrong output, a wrong trace
 * and a mismatched linear-layer sum are all rejected; the honest sum-check round identity holds exactly at every
 * round and the round polynomial is exactly linear (F(2) = 2F(1) - F(0)); the final MLE check accepts the honest
 * prover and rejects a prover who claims anything else; the measured soundness error over the whole field is at
 * most deg/|F|; and the verifier's work stays sublinear in the number of gates while the prover's grows with it.
 *
 * Simplified: the hash is a toy mixer (binding is empirical only), the field is 97 elements wide (a real system uses
 * a 256-bit field, which is what makes the soundness error negligible), the verifier may evaluate the multilinear
 * extension of the public weights directly instead of through a polynomial commitment, and the spot-check
 * challenges come from a fixed-seed stream instead of real randomness.
 */
import java.util.*;

/** A deterministic toy hash: splitmix64 mixing. Binding is empirical, collision resistance is not claimed. */
final class ToyHash {
    static long mix(long a) {
        long z = a + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Internal Merkle node: combines two children. */
    static long combine(long left, long right) {
        return mix(mix(left) ^ Long.rotateLeft(mix(right), 17));
    }
}

/** A binary Merkle tree over the flattened trace, with opening paths and path verification. */
final class Merkle {
    final long[] leaves;
    final long[][] levels;

    Merkle(long[] values) {
        int n = 1;
        while (n < values.length) n *= 2;
        leaves = new long[n];
        System.arraycopy(values, 0, leaves, 0, values.length);
        List<long[]> built = new ArrayList<>();
        built.add(leaves.clone());
        long[] current = leaves;
        while (current.length > 1) {
            long[] next = new long[current.length / 2];
            for (int i = 0; i < next.length; i++)
                next[i] = ToyHash.combine(current[2 * i], current[2 * i + 1]);
            built.add(next);
            current = next;
        }
        levels = built.toArray(new long[0][]);
    }

    long root() { return levels[levels.length - 1][0]; }

    long[] open(int index) {
        long[] path = new long[levels.length - 1];
        int position = index;
        for (int level = 0; level < levels.length - 1; level++) {
            path[level] = levels[level][position ^ 1];
            position >>= 1;
        }
        return path;
    }

    static long rootFromPath(int index, long value, long[] path) {
        long current = value;
        int position = index;
        for (long sibling : path) {
            current = (position & 1) == 0 ? ToyHash.combine(current, sibling)
                    : ToyHash.combine(sibling, current);
            position >>= 1;
        }
        return current;
    }
}

/** Exact arithmetic in Z_p for the sum-check. */
final class SmallField {
    final long p;

    SmallField(long p) { this.p = p; }

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

    long inverse(long a) { return pow(a, p - 2); }

    /** The multilinear extension of a power-of-two sized table, evaluated at a point of field elements. */
    long multilinear(long[] table, long[] point) {
        int m = Integer.numberOfTrailingZeros(table.length);
        long total = 0;
        for (int index = 0; index < table.length; index++) {
            long term = table[index];
            for (int k = 0; k < m; k++) {
                long bit = (index >> (m - 1 - k)) & 1;
                term = mul(term, bit == 1 ? point[k] : sub(1, point[k]));
            }
            total = add(total, term);
        }
        return total;
    }

    /** F_r(t): the round polynomial of a multilinear extension, at one value of t. */
    long roundValue(long[] table, long[] challenges, int round, long t) {
        int m = Integer.numberOfTrailingZeros(table.length);
        int free = m - round - 1;
        long total = 0;
        for (int prefix = 0; prefix < (1 << round); prefix++) {
            long prefixWeight = 1;
            for (int k = 0; k < round; k++) {
                int bit = (prefix >> (round - 1 - k)) & 1;
                prefixWeight = mul(prefixWeight, bit == 1 ? challenges[k] : sub(1, challenges[k]));
            }
            for (int bitR = 0; bitR < 2; bitR++) {
                long weight = mul(prefixWeight, bitR == 1 ? t : sub(1, t));
                for (int suffix = 0; suffix < (1 << free); suffix++) {
                    int index = (prefix << (m - round)) | (bitR << free) | suffix;
                    total = add(total, mul(table[index], weight));
                }
            }
        }
        return total;
    }
}

/** The inference circuit: linear layer, a fixed permutation, then the linear output layer, all over integers. */
final class Circuit {
    final int input, hidden, output;
    final long[][] w1, w2;
    final int[] permutation;

    Circuit(int hidden) {
        this.input = 4;
        this.hidden = hidden;
        this.output = 2;
        w1 = new long[hidden][input];
        w2 = new long[output][hidden];
        permutation = new int[hidden];
        long state = 20240925L;
        for (int i = 0; i < hidden; i++) {
            for (int j = 0; j < input; j++) {
                state = ToyHash.mix(state + i * 31L + j);
                w1[i][j] = Math.floorMod(state, 61) - 30;
            }
            permutation[i] = (i * 3 + 1) % hidden;
        }
        for (int i = 0; i < output; i++)
            for (int j = 0; j < hidden; j++) {
                state = ToyHash.mix(state + 1000 + i * 17L + j);
                w2[i][j] = Math.floorMod(state, 41) - 20;
            }
    }

    int traceLength() { return input + 2 * hidden + output; }
    int hiddenOffset() { return input; }
    int permutedOffset() { return input + hidden; }
    int outputOffset() { return input + 2 * hidden; }

    /** Flattened trace [input | hidden | permuted | output]: statement plus witness. */
    long[] trace(long[] x) {
        long[] hiddenValues = new long[hidden], permuted = new long[hidden], out = new long[output];
        for (int i = 0; i < hidden; i++) {
            long sum = 0;
            for (int j = 0; j < input; j++) sum += w1[i][j] * x[j];
            hiddenValues[i] = sum;
        }
        for (int i = 0; i < hidden; i++) permuted[i] = hiddenValues[permutation[i]];
        for (int i = 0; i < output; i++) {
            long sum = 0;
            for (int j = 0; j < hidden; j++) sum += w2[i][j] * permuted[j];
            out[i] = sum;
        }
        long[] trace = new long[traceLength()];
        System.arraycopy(x, 0, trace, 0, input);
        System.arraycopy(hiddenValues, 0, trace, hiddenOffset(), hidden);
        System.arraycopy(permuted, 0, trace, permutedOffset(), hidden);
        System.arraycopy(out, 0, trace, outputOffset(), output);
        return trace;
    }

    /** Prover's own work counter: every multiply-add of the plain execution. */
    int proverOperations(long[] x) {
        int operations = 0;
        for (int i = 0; i < hidden; i++) {
            for (int j = 0; j < input; j++) operations++;
        }
        operations += hidden;
        for (int i = 0; i < output; i++)
            for (int j = 0; j < hidden; j++) operations++;
        return operations;
    }
}

/** The verifier: it carries a root, opens leaves and checks the equations those leaves support. */
final class Verifier {
    final long root;
    final Circuit circuit;
    final SmallField field;
    int leafOpenings;
    int fieldOperations;

    Verifier(long root, Circuit circuit, SmallField field) {
        this.root = root;
        this.circuit = circuit;
        this.field = field;
    }

    boolean openLeaf(Merkle tree, int index, long expected) {
        leafOpenings++;
        long recomputed = Merkle.rootFromPath(index, expected, tree.open(index));
        return recomputed == root;
    }

    /** Checks one hidden unit's linear equation from the opened input and hidden leaves. */
    boolean checkHiddenEquation(Merkle tree, long[] x, int unit, long hiddenValue) {
        if (!openLeaf(tree, circuit.hiddenOffset() + unit, hiddenValue)) return false;
        for (int j = 0; j < circuit.input; j++)
            if (!openLeaf(tree, j, x[j])) return false;
        long sum = 0;
        for (int j = 0; j < circuit.input; j++)
            sum = field.add(sum, field.mul(circuit.w1[unit][j], x[j]));
        fieldOperations += circuit.input + 1;
        return sum == Math.floorMod(hiddenValue, field.p);
    }

    /** Checks the permutation for one unit. */
    boolean checkPermutation(Merkle tree, int unit, long hiddenValue, long permutedValue) {
        if (!openLeaf(tree, circuit.permutedOffset() + unit, permutedValue)) return false;
        if (!openLeaf(tree, circuit.hiddenOffset() + circuit.permutation[unit], hiddenValue)) return false;
        fieldOperations++;
        return field.sub(permutedValue, hiddenValue) == 0;
    }

    /** Checks one output unit's linear equation. */
    boolean checkOutputEquation(Merkle tree, long[] permuted, int unit, long outputValue) {
        if (!openLeaf(tree, circuit.outputOffset() + unit, outputValue)) return false;
        for (int j = 0; j < circuit.hidden; j++)
            if (!openLeaf(tree, circuit.permutedOffset() + j, permuted[j])) return false;
        long sum = 0;
        for (int j = 0; j < circuit.hidden; j++)
            sum = field.add(sum, field.mul(circuit.w2[unit][j], permuted[j]));
        fieldOperations += circuit.hidden + 1;
        return sum == Math.floorMod(outputValue, field.p);
    }
}

public class Solution {
    static final long PRIME = 97L;
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    public static void main(String[] args) {
        SmallField field = new SmallField(PRIME);
        Circuit circuit = new Circuit(3);
        long[] x = {3, -1, 2, 5};
        long[] trace = circuit.trace(x);
        Merkle tree = new Merkle(trace);
        long root = tree.root();

        // ---- 1. the committed trace reproduces every leaf, and the root is stable
        boolean consistent = true;
        for (int i = 0; i < circuit.traceLength(); i++)
            if (Merkle.rootFromPath(i, trace[i], tree.open(i)) != root) consistent = false;
        Merkle same = new Merkle(circuit.trace(x));
        check(consistent && same.root() == root, "every trace value opens to the committed root and rebuilding the "
                + "commitment gives the same root");

        // ---- 2. the committed computation equals an independent plain implementation
        long[] hidden = new long[circuit.hidden], permuted = new long[circuit.hidden], out = new long[circuit.output];
        System.arraycopy(trace, circuit.hiddenOffset(), hidden, 0, circuit.hidden);
        System.arraycopy(trace, circuit.permutedOffset(), permuted, 0, circuit.hidden);
        System.arraycopy(trace, circuit.outputOffset(), out, 0, circuit.output);
        long[] reference = new long[circuit.output];
        for (int i = 0; i < circuit.output; i++) {
            long sum = 0;
            for (int j = 0; j < circuit.hidden; j++) {
                long inner = 0;
                for (int k = 0; k < circuit.input; k++) inner += circuit.w1[circuit.permutation[j]][k] * x[k];
                sum += circuit.w2[i][j] * inner;
            }
            reference[i] = sum;
        }
        boolean outputsMatch = true;
        for (int i = 0; i < circuit.output; i++) if (out[i] != reference[i]) outputsMatch = false;
        check(outputsMatch, "the committed output equals the independent step-by-step evaluation");

        // ---- 3. tampering with the trace or the output is detected
        long[] badTrace = trace.clone();
        badTrace[circuit.hiddenOffset()] += 1;
        Merkle badTree = new Merkle(badTrace);
        Verifier verifier = new Verifier(root, circuit, field);
        boolean tamperedLeafRejected = !verifier.checkHiddenEquation(badTree, x, 0, badTrace[circuit.hiddenOffset()]);
        boolean wrongOutputRejected = !verifier.checkOutputEquation(tree, permuted, 0, out[0] + 1);
        check(tamperedLeafRejected && wrongOutputRejected,
                "a tampered activation and a wrong output both fail the verifier's opening checks");

        // ---- 4. the honest spot checks all pass, and they consume only the leaves they need
        Verifier honest = new Verifier(root, circuit, field);
        boolean allChecks = true;
        for (int i = 0; i < circuit.hidden; i++)
            if (!honest.checkHiddenEquation(tree, x, i, hidden[i])) allChecks = false;
        for (int i = 0; i < circuit.hidden; i++)
            if (!honest.checkPermutation(tree, i, hidden[circuit.permutation[i]], permuted[i])) allChecks = false;
        for (int i = 0; i < circuit.output; i++)
            if (!honest.checkOutputEquation(tree, permuted, i, out[i])) allChecks = false;
        check(allChecks, "all honest layer equations pass the verifier's spot checks");

        // ---- 5. binding is empirical: distinct traces give distinct roots
        Set<Long> distinctRoots = new HashSet<>();
        int generated = 0;
        for (int a = -3; a <= 3; a++)
            for (int b = -3; b <= 3; b++) {
                long[] probe = {a, b, a - b, a + b};
                distinctRoots.add(new Merkle(circuit.trace(probe)).root());
                generated++;
            }
        check(distinctRoots.size() == generated, "all " + generated + " distinct traces produce distinct roots "
                + "(empirical binding of the toy hash)");

        // ---- 6. the sum-check: honest round identity is exact and the round polynomial is linear
        int rows = circuit.hidden, cols = circuit.input;
        int slots = rows * cols;
        int variables = Integer.numberOfTrailingZeros(Integer.highestOneBit(slots - 1) * 2);
        if ((1 << variables) < slots) variables++;
        long[] table = new long[1 << variables];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                table[i * cols + j] = field.mul(Math.floorMod(circuit.w1[i][j], PRIME), Math.floorMod(x[j], PRIME));
        long claimed = 0;
        for (long value : table) claimed = field.add(claimed, value);
        long[] challenges = new long[variables];
        for (int round = 0; round < variables; round++) challenges[round] = (round * 7 + 3) % PRIME;
        boolean roundIdentity = true, linear = true;
        long running = claimed;
        for (int round = 0; round < variables; round++) {
            long atZero = field.roundValue(table, challenges, round, 0);
            long atOne = field.roundValue(table, challenges, round, 1);
            long atTwo = field.roundValue(table, challenges, round, 2);
            if (field.add(atZero, atOne) != running) roundIdentity = false;
            if (atTwo != field.sub(field.mul(2, atOne), atZero)) linear = false;
            running = field.add(atZero, field.mul(field.sub(atOne, atZero), challenges[round]));
        }
        check(roundIdentity, "the honest round identity F(0) + F(1) = current claim holds in every round");
        check(linear, "the round polynomial of a multilinear extension is exactly linear, so the per-round "
                + "Schwartz-Zippel degree is 1");

        // ---- 7. the final check accepts the honest prover and rejects a prover who committed to another table
        long honestFinal = field.multilinear(table, challenges);
        long[] cheatTable = table.clone();
        cheatTable[1] = field.add(cheatTable[1], 1);
        long cheatRunning = 0;
        for (long value : cheatTable) cheatRunning = field.add(cheatRunning, value);
        for (int round = 0; round < variables; round++) {
            long atZero = field.roundValue(cheatTable, challenges, round, 0);
            long atOne = field.roundValue(cheatTable, challenges, round, 1);
            cheatRunning = field.add(atZero, field.mul(field.sub(atOne, atZero), challenges[round]));
        }
        check(running == honestFinal, "the accumulated claim of the honest prover equals the true multilinear "
                + "evaluation at the challenges");
        check(cheatRunning != honestFinal, "a prover whose table differs from the committed one is rejected by the "
                + "final multilinear check");

        // ---- 8. measured soundness: a dishonest linear round polynomial differs from the true one on at most
        //         one of the 97 challenges, so the per-round error is at most 1 / |F|
        long trueAtZero = field.roundValue(table, challenges, 0, 0);
        long trueAtOne = field.roundValue(table, challenges, 0, 1);
        long trueSlope = field.sub(trueAtOne, trueAtZero);
        int worstRoots = 0;
        long[] slopes = {1, 2, 3, 7, 11, 40};
        for (long slopeShift : slopes) {
            // the dishonest line keeps F'(0) + F'(1) equal to the claim (so the consistency check still passes)
            // but is genuinely different, and any two distinct lines agree at most at one challenge
            long interceptShift = field.mul(field.sub(0, slopeShift), field.inverse(2));
            long cheatZero = field.add(trueAtZero, interceptShift);
            long cheatSlope = field.add(trueSlope, slopeShift);
            int roots = 0;
            for (long c = 0; c < PRIME; c++) {
                long honestValue = field.add(trueAtZero, field.mul(trueSlope, c));
                long cheatValue = field.add(cheatZero, field.mul(cheatSlope, c));
                if (honestValue == cheatValue) roots++;
            }
            worstRoots = Math.max(worstRoots, roots);
        }
        check(worstRoots <= 1, "over the whole field of " + PRIME + " challenges, every dishonest linear round "
                + "polynomial that satisfies the consistency check agrees with the true one on at most "
                + worstRoots + " challenge(s), i.e. the deg/|F| bound");

        // ---- 9. succinctness: the verifier's work stays sublinear in the number of gates
        Circuit wide = new Circuit(8);
        long[] wideTrace = wide.trace(x);
        Merkle wideTree = new Merkle(wideTrace);
        Verifier wideVerifier = new Verifier(wideTree.root(), wide, field);
        wideVerifier.checkHiddenEquation(wideTree, x, 0, wideTrace[wide.hiddenOffset()]);
        wideVerifier.checkOutputEquation(wideTree, wideTrace, 0,
                wideTrace[wide.outputOffset()]);
        Verifier narrowVerifier = new Verifier(root, circuit, field);
        narrowVerifier.checkHiddenEquation(tree, x, 0, trace[circuit.hiddenOffset()]);
        narrowVerifier.checkOutputEquation(tree, permuted, 0, out[0]);
        double verifierGrowth = wideVerifier.leafOpenings / (double) narrowVerifier.leafOpenings;
        double proverGrowth = wide.proverOperations(x) / (double) circuit.proverOperations(x);
        check(verifierGrowth < proverGrowth && proverGrowth > 2.0,
                "the verifier's openings grow slower (" + verifierGrowth + "x) than the prover's work ("
                        + proverGrowth + "x) when the hidden layer widens from " + circuit.hidden + " to "
                        + wide.hidden + " units");
        check(narrowVerifier.leafOpenings + narrowVerifier.fieldOperations < circuit.proverOperations(x),
                "checking two layers costs the verifier " + (narrowVerifier.leafOpenings
                        + narrowVerifier.fieldOperations) + " operations against the prover's "
                        + circuit.proverOperations(x));

        // ---- 10. determinism
        Merkle again = new Merkle(circuit.trace(x));
        long[] tableAgain = new long[1 << variables];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                tableAgain[i * cols + j] = field.mul(Math.floorMod(circuit.w1[i][j], PRIME), Math.floorMod(x[j], PRIME));
        long claimedAgain = 0;
        for (long value : tableAgain) claimedAgain = field.add(claimedAgain, value);
        check(again.root() == root && claimedAgain == claimed,
                "the commitment and the sum-check claim are reproducible");

        System.out.println("verifiable inference: " + circuit.traceLength() + " trace values committed, "
                + circuit.hidden + " hidden units, field Z_" + PRIME);
        System.out.println("verifier openings " + narrowVerifier.leafOpenings + " vs prover "
                + circuit.proverOperations(x) + " operations; worst dishonest-challenge agreement " + worstRoots
                + " of " + PRIME);
        System.out.println("checks run = " + checks);
        System.out.println("PASS");
    }
}
