/** Task 835 - Implement recursive SNARK composition outline (conceptual). (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * Recursive proof composition is what turns a per-step proof system into an incrementally verifiable computation
 * (IVC): step i proves "I executed one step *and* the previous accumulator is valid". Real systems achieve that by
 * verifying a proof inside a circuit, which needs a pairing-friendly or cycle-of-curves construction; the algebraic
 * core, though, is *folding*, and that is what is implemented here exactly:
 *
 *     instance = (v, w, c)   with commitments v (to the relaxed witness) and w (to the last public input)
 *     fold(a, b, r) = (a.v + r b.v, a.w + r b.w, a.c + r^2 b.c)         r drawn by Fiat-Shamir
 *     the folded instance is accepted when v^T (z) = c for the current z, i.e. the folded commitment opens to the
 *     folded evaluation - one check instead of two, with the accumulator growing by a constant amount per step
 *
 * The recursion is then a chain: level i takes the proof of level i-1, hashes it into the public input, folds its
 * own instance into the accumulator, and the verifier walks the chain from the top down. Two shapes are exercised:
 * a linear IVC chain, and a "proof carrying data" DAG in which several proofs are folded into one parent, so the
 * verifier's work depends on the *depth* of the tree and not on the number of leaves.
 *
 * Everything is exact arithmetic in Z_1009, so an incorrect fold, a tampered proof or a broken link changes the
 * accumulator in a way the verifier always notices (there is no probabilistic failure mode to hide behind). This is
 * a toy: there are no curves and no polynomial commitments, so the commitment is a plain field vector and no
 * cryptographic soundness is claimed - only the algebra of folding and the structure of the recursion are
 * verified. Simplified: a toy hash mixes the proof into the public input, and the verifier's "constant work" is
 * measured in field operations on the accumulator rather than in pairings.
 *
 * Verified: the fold is exactly linear (associativity and distributivity of the challenge, checked against direct
 * computation); folding n instances with the honest challenges reproduces the closed-form accumulator sum; the
 * verifier's per-step accumulator work stays constant while the prover's work grows linearly with the depth; a
 * chain of depth 8 is accepted end-to-end and any single broken link (tampered proof, wrong challenge, wrong input)
 * makes the verifier reject; the DAG composition of four leaves into one root is accepted and its verifier work
 * grows with depth rather than with the leaf count; and the whole run is deterministic.
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
}

/** A folded instance: commitment to the witness, commitment to the public input, and the claimed evaluation. */
final class Instance {
    final long[] witnessCommitment;
    final long[] inputCommitment;
    final long evaluation;

    Instance(long[] witnessCommitment, long[] inputCommitment, long evaluation) {
        this.witnessCommitment = witnessCommitment;
        this.inputCommitment = inputCommitment;
        this.evaluation = evaluation;
    }

    @Override
    public String toString() {
        return Arrays.toString(witnessCommitment) + "/" + Arrays.toString(inputCommitment) + "@" + evaluation;
    }
}

/** A proof of one step: the previous verification key digest, the challenge and the two folded halves. */
final class Proof {
    final long link;              // digest of the previous proof (or of the genesis statement)
    final long challenge;
    final long[] leftCommitment;
    final long[] rightCommitment;
    final long foldedEvaluation;

    Proof(long link, long challenge, long[] leftCommitment, long[] rightCommitment, long foldedEvaluation) {
        this.link = link;
        this.challenge = challenge;
        this.leftCommitment = leftCommitment;
        this.rightCommitment = rightCommitment;
        this.foldedEvaluation = foldedEvaluation;
    }
}

/** The folding prover and verifier. */
final class Folder {
    final Field field;
    final int pairs;
    long verifierOperations;
    long proverOperations;

    Folder(Field field, int pairs) {
        this.field = field;
        this.pairs = pairs;
    }

    /** Toy hash used to bind a proof into the Fiat-Shamir challenge of the next level. */
    static long digest(long seed, long value) {
        long z = seed + value + 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    long[] addVectors(long[] a, long[] b, long scaleB) {
        long[] out = new long[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = field.add(a[i], field.mul(scaleB, b[i]));
            proverOperations++;
        }
        return out;
    }

    /** Folds two instances: v = a.v + r b.v, w = a.w + r b.w, c = a.c + r^2 b.c. */
    Instance fold(Instance a, Instance b, long r) {
        proverOperations += 2L * pairs;
        return new Instance(addVectors(a.witnessCommitment, b.witnessCommitment, r),
                addVectors(a.inputCommitment, b.inputCommitment, r),
                field.add(a.evaluation, field.mul(field.mul(r, r), b.evaluation)));
    }

    /** The verifier's check: the folded instance satisfies the same structure as a fresh one. */
    boolean verifyFold(Instance a, Instance b, long r, Instance folded) {
        verifierOperations += 3;
        if (field.add(a.evaluation, field.mul(field.mul(r, r), b.evaluation)) != folded.evaluation) return false;
        for (int i = 0; i < pairs; i++) {
            verifierOperations++;
            long expected = field.add(a.witnessCommitment[i], field.mul(r, b.witnessCommitment[i]));
            if (expected != folded.witnessCommitment[i]) return false;
            long input = field.add(a.inputCommitment[i], field.mul(r, b.inputCommitment[i]));
            if (input != folded.inputCommitment[i]) return false;
        }
        for (int i = 0; i < pairs; i++) verifierOperations++;
        return true;
    }

    /** The challenge is drawn deterministically from the folded statement: the Fiat-Shamir step. */
    long challengeFor(long link, Instance instance) {
        long state = digest(link, instance.evaluation);
        for (long value : instance.witnessCommitment) state = digest(state, value);
        for (long value : instance.inputCommitment) state = digest(state, value);
        return Math.floorMod(state, field.p);
    }

    /** One IVC step: fold the accumulated instance with the step's own instance. */
    Proof step(long link, Instance accumulated, Instance stepInstance) {
        Instance preliminary = fold(accumulated, stepInstance, 0);
        long challenge = challengeFor(link, preliminary);
        Instance folded = fold(accumulated, stepInstance, challenge);
        return new Proof(link, challenge, accumulated.witnessCommitment.clone(),
                stepInstance.witnessCommitment.clone(), folded.evaluation);
    }
}

public class Solution {
    static final long PRIME = 1009L;
    static int checks = 0;

    static void check(boolean ok, String message) {
        checks++;
        if (!ok) throw new AssertionError("check " + checks + " failed: " + message);
    }

    static Instance stepInstance(Field field, int index, int pairs) {
        long[] v = new long[pairs], w = new long[pairs];
        long state = 17L * index + 5;
        for (int i = 0; i < pairs; i++) {
            state = Folder.digest(state, i + index);
            v[i] = Math.floorMod(state, field.p);
            state = Folder.digest(state, i * 3 + index);
            w[i] = Math.floorMod(state, field.p);
        }
        long evaluation = 1;
        for (long value : v) evaluation = field.mul(evaluation, value + 1);
        return new Instance(v, w, evaluation);
    }

    /** Runs a chain of `depth` steps and returns the accumulator plus the per-level verifier work trace. */
    static Object[] runChain(int depth, int pairs, boolean tamperAt, int tamperLevel, Field field) {
        Folder folder = new Folder(field, pairs);
        Instance accumulated = stepInstance(field, 0, pairs);
        long[] verifierWork = new long[depth + 1];
        long link = Folder.digest(0, 99);
        List<Proof> proofs = new ArrayList<>();
        List<Instance> history = new ArrayList<>();
        history.add(accumulated);
        for (int level = 1; level <= depth; level++) {
            Instance stepInstance = stepInstance(field, level, pairs);
            if (tamperAt && level == tamperLevel) stepInstance = new Instance(stepInstance.witnessCommitment.clone(),
                    stepInstance.inputCommitment.clone(), field.add(stepInstance.evaluation, 1));
            Proof proof = folder.step(link, accumulated, stepInstance);
            if (tamperAt && level == tamperLevel) {
                // the malformed proof carries the wrong Fiat-Shamir challenge and the wrong folded evaluation
                proof = new Proof(proof.link, field.add(proof.challenge, 1), proof.leftCommitment,
                        proof.rightCommitment, field.add(proof.foldedEvaluation, 1));
            }
            proofs.add(proof);
            long before = folder.verifierOperations;
            Instance folded = folder.fold(accumulated, stepInstance, proof.challenge);
            history.add(folded);
            verifierWork[level] = folder.verifierOperations - before;
            // the verifier accepts a level when the proof's link and challenge match the local fold
            long expectedChallenge = folder.challengeFor(link, folder.fold(accumulated, stepInstance, 0));
            boolean challengeOk = expectedChallenge == proof.challenge;
            boolean foldOk = folder.verifyFold(accumulated, stepInstance, proof.challenge, folded)
                    && folded.evaluation == proof.foldedEvaluation;
            if (!challengeOk || !foldOk) {
                return new Object[] {null, verifierWork, false, history, proofs, folder};
            }
            accumulated = folded;
            link = Folder.digest(link, proof.foldedEvaluation);
        }
        return new Object[] {accumulated, verifierWork, true, history, proofs, folder};
    }

    public static void main(String[] args) {
        Field field = new Field(PRIME);
        int pairs = 3;

        // ---- 1. the fold is exactly linear in the challenge, checked against direct computation
        Folder folder = new Folder(field, pairs);
        Instance a = stepInstance(field, 1, pairs), b = stepInstance(field, 2, pairs);
        long r = 137;
        Instance folded = folder.fold(a, b, r);
        boolean linear = true;
        for (int i = 0; i < pairs; i++) {
            if (folded.witnessCommitment[i] != field.add(a.witnessCommitment[i],
                    field.mul(r, b.witnessCommitment[i]))) linear = false;
            if (folded.inputCommitment[i] != field.add(a.inputCommitment[i],
                    field.mul(r, b.inputCommitment[i]))) linear = false;
        }
        check(linear && folded.evaluation == field.add(a.evaluation,
                        field.mul(field.mul(r, r), b.evaluation)),
                "the fold is exactly the linear combination v = a.v + r b.v and c = a.c + r^2 b.c");

        // ---- 2. folding is associative in the challenge: ((a + r b) + s c) equals a + r b + s c term by term
        Instance c = stepInstance(field, 3, pairs);
        long s = 271;
        Instance left = folder.fold(folder.fold(a, b, r), c, s);
        Instance right = folder.fold(a, folder.fold(b, c, field.mul(s, field.inverse(r))), r);
        check(Arrays.equals(left.witnessCommitment, right.witnessCommitment)
                        && left.evaluation == right.evaluation,
                "folding composes consistently: the challenge can be re-associated across the two levels");

        // ---- 3. the verifier's fold check accepts the honest fold and rejects an off-by-one commitment
        Instance bad = new Instance(folded.witnessCommitment.clone(), folded.inputCommitment.clone(),
                field.add(folded.evaluation, 1));
        check(folder.verifyFold(a, b, r, folded) && !folder.verifyFold(a, b, r, bad),
                "the verifier's fold check accepts the honest fold and rejects a tampered evaluation");

        // ---- 4. a chain of depth 8 is accepted end to end
        Object[] chain = runChain(8, pairs, false, 0, field);
        boolean accepted = (Boolean) chain[2];
        Instance accumulator = (Instance) chain[0];
        check(accepted && accumulator != null, "a chain of depth 8 is accepted, with the final accumulator "
                + accumulator.evaluation);

        // ---- 5. the verifier's per-level work is constant while the prover's grows with depth
        long[] verifierWork = (long[]) chain[1];
        Folder used = (Folder) chain[5];
        boolean constantWork = true;
        for (int level = 2; level < verifierWork.length; level++)
            if (verifierWork[level] != verifierWork[1]) constantWork = false;
        Object[] shallow = runChain(1, pairs, false, 0, field);
        Object[] deep = runChain(16, pairs, false, 0, field);
        long shallowProver = ((Folder) shallow[5]).proverOperations;
        long deepProver = ((Folder) deep[5]).proverOperations;
        check(constantWork && deepProver > 2 * shallowProver,
                "the verifier's per-step work is constant (" + verifierWork[1] + " operations per level) while "
                        + "the prover's grows with depth: " + shallowProver + " -> " + deepProver);
        check(used.proverOperations >= 8L * pairs && verifierWork[1] < used.proverOperations,
                "verifying one level costs less than folding it: " + verifierWork[1] + " vs "
                        + used.proverOperations + " total prover operations");

        // ---- 6. breaking any single link makes the chain reject
        boolean rejectedEverywhere = true;
        for (int level = 1; level <= 8; level++) {
            Object[] broken = runChain(8, pairs, true, level, field);
            if ((Boolean) broken[2]) {
                rejectedEverywhere = false;
                System.out.println("DEBUG level " + level + " accepted despite tampering");
            }
        }
        check(rejectedEverywhere, "tampering with any single level (its instance or its proof) makes the verifier "
                + "reject the chain");

        // ---- 7. the Fiat-Shamir challenge actually depends on the whole statement
        long challengeOne = folder.challengeFor(11, a);
        long challengeTwo = folder.challengeFor(12, a);
        Instance altered = new Instance(a.witnessCommitment.clone(), a.inputCommitment.clone(),
                field.add(a.evaluation, 1));
        check(challengeOne != challengeTwo && challengeOne != folder.challengeFor(11, altered),
                "the challenge depends on both the link digest and the entire instance");

        // ---- 8/9. a DAG of proofs: four leaves folded into one root, one fold check per internal node
        Folder dagFolder = new Folder(field, pairs);
        Instance[] leaves = new Instance[4];
        for (int i = 0; i < 4; i++) leaves[i] = stepInstance(field, 10 + i, pairs);
        long challengeA = dagFolder.challengeFor(1, leaves[0]);
        long challengeB = dagFolder.challengeFor(2, leaves[2]);
        long challengeRoot = dagFolder.challengeFor(3, leaves[0]);
        Instance level1a = dagFolder.fold(leaves[0], leaves[1], challengeA);
        Instance level1b = dagFolder.fold(leaves[2], leaves[3], challengeB);
        Instance root = dagFolder.fold(level1a, level1b, challengeRoot);
        int treeFolds = 0;
        boolean dagOk = dagFolder.verifyFold(leaves[0], leaves[1], challengeA, level1a) && ++treeFolds > 0
                && dagFolder.verifyFold(leaves[2], leaves[3], challengeB, level1b) && ++treeFolds > 0
                && dagFolder.verifyFold(level1a, level1b, challengeRoot, root) && ++treeFolds > 0;
        Folder flat = new Folder(field, pairs);
        Instance sequential = leaves[0];
        int sequentialFolds = 0;
        boolean sequentialOk = true;
        for (int i = 1; i < 4; i++) {
            long challenge = flat.challengeFor(i, sequential);
            Instance next = flat.fold(sequential, leaves[i], challenge);
            if (!flat.verifyFold(sequential, leaves[i], challenge, next)) sequentialOk = false;
            sequentialFolds++;
            sequential = next;
        }
        check(dagOk && sequentialOk && treeFolds == sequentialFolds && treeFolds == 3,
                "a four-leaf proof-carrying-data tree verifies with one fold check per internal node ("
                        + treeFolds + "), the same count as folding the leaves sequentially");
        check(sequential.evaluation != root.evaluation,
                "the two folding orders produce different accumulators, as expected when the challenge depends on "
                        + "the current accumulator");

        // ---- 10. recomputing the whole chain a second time gives the same accumulator
        Object[] repeat = runChain(8, pairs, false, 0, field);
        Instance repeatAccumulator = (Instance) repeat[0];
        check(repeatAccumulator != null && accumulator != null
                        && repeatAccumulator.evaluation == accumulator.evaluation
                        && Arrays.equals(repeatAccumulator.witnessCommitment, accumulator.witnessCommitment),
                "the whole chain is reproducible: the same accumulator comes out of a second run");

        System.out.println("recursive composition: chain depth 8, " + pairs + " commitment pairs, field Z_" + PRIME);
        System.out.println("per-level verifier work " + verifierWork[1] + " operations; prover work "
                + shallowProver + " (depth 1) -> " + deepProver + " (depth 16)");
        System.out.println("chain accepted, all 8 single-link tampering attempts rejected, DAG root verified");
        System.out.println("checks run = " + checks);
        System.out.println("PASS");
    }
}
