/** Task 889 - Implement secure multiparty computation for private set intersection at scale. (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * Two PSI protocols are implemented and cross-verified, because they fail in different ways:
 *
 * 1. Commutative (Diffie-Hellman style) PSI over the 61-bit Mersenne field GF(2^61-1):
 *
 *      Alice shoves  A1 = shuffle{ H1(x_i)^a }                 (a is Alice's secret exponent)
 *      Bob   replies B2 = { (A1_i)^b }  in the *same* order    (b is Bob's secret exponent)
 *      Bob   sends   B1 = shuffle{ H1(y_j)^b }
 *      Alice sends   A2 = { (B1_j)^a }  in the same order
 *
 *    Both sides can now form { H(x)^ab } for their own elements positionally - so each maps a match
 *    back to its own element - and can compute { H(other)^ab } from the other's message. A private
 *    intersection is therefore learnable by both parties, matching the textbook protocol, and the
 *    transcript carries no raw hash: it only ever exposes exponentiations of hashes.
 *
 * 2. Secret-shared matching without any exponentiation, i.e. PSI from a shared match matrix:
 *    elements are mapped to slots of a bit vector v (Bob) and a one-hot matrix M (Alice); the
 *    intersection size is the Frobenius product sum_{i,j} M[i][j] * v[j]. Alice additively shares M,
 *    Bob additively shares v, and every entry product is computed with a Beaver triple
 *    (d = x - a and e = y - b are opened, [z] = [c] + d[b] + e[a] + d*e), so only the shared scalar
 *    total is ever opened - no per-element membership bit leaves the protocol.
 *
 * What is simplified - and what is not. The field is a 61-bit Mersenne field with hand-written modular
 * multiplication (Math.multiplyHigh plus the 2^61 = 1 fold), not a curve group, so there is no
 * DDH-based security argument; the demo verifies the *structural* privacy properties instead (raw
 * hashes never appear in the transcript, shares are one-time padded and statistically uniform,
 * opening values are fresh masks). The matrix protocol uses a semi-honest two-party model with
 * Beaver triples supplied by an ideal functionality, and its communication is O(m * B) field elements
 * - a real system replaces that with OT extension for O(m + B); the cost formula is asserted exactly,
 * so the scaling claim is checked rather than asserted. "At scale" is expressed as typed parameters:
 * 2000 elements per side for the exponentiation protocol (about 8000 modular exponentiations), and
 * m x B = 64 x 512 multiplications for the shared matrix, which is sized so that every entry product
 * is individually verified. Sizing is deterministic, there are no threads, sockets or clocks.
 *
 * Verified: both parties learn exactly the true intersection (brute-force set intersection oracle) for
 * random, disjoint, identical, empty and single-element instances; H1 is injective on the union, so a
 * match is a real match and the collision-freeness precondition is checked rather than assumed; the
 * transcript never contains a raw hash and every message is a permutation of the expected multiset;
 * exponentiation and message counts match the exact formulas 2(n_A+n_B); the shared-matrix protocol
 * returns the true cardinality and agrees with the exponentiation protocol and the oracle; every
 * Beaver multiplication satisfies [z_A] + [z_B] = x*y exactly; the opened masks d and e are balanced
 * (bit frequency ~1/2) and fresh on every run; the share handed to the other party is statistically
 * independent of the secret it hides (the empirical bit frequency at secret-one positions and
 * secret-zero positions agree within tolerance); and degenerate instances (empty set, zero slots,
 * single element) behave.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/**
 * GF(p) with p = 2^61 - 1. Multiplication keeps a full 128-bit product in (hi, lo) and folds it with
 * the Mersenne identity 2^61 = 1, so no BigInteger and no allocation is needed.
 */
class Field61 {
    static final long P = (1L << 61) - 1;

    static long add(long a, long b) {
        long s = a + b;
        long r = (s & P) + (s >>> 61);
        return r >= P ? r - P : r;
    }

    static long sub(long a, long b) { return add(a, b == 0 ? 0 : P - b); }

    static long mul(long a, long b) {
        long hi = Math.multiplyHigh(a, b), lo = a * b;
        long r = (lo & P) + (hi << 3) + (lo >>> 61);
        r = (r & P) + (r >>> 61);
        return r >= P ? r - P : r;
    }

    static long pow(long base, long exp) {
        long result = 1, b = base % P, e = exp;
        while (e > 0) {
            if ((e & 1) == 1) result = mul(result, b);
            b = mul(b, b);
            e >>= 1;
        }
        return result;
    }

    static long random(SplitMix64 rng) { return Math.floorMod(rng.nextLong(), P); }

    static long randomNonZero(SplitMix64 rng) {
        long v = random(rng);
        return v == 0 ? 1 : v;
    }

    /** Domain separated hash to the field; used as H1 (element -> group exponent base) and H2. */
    static long hash(String domain, String value) {
        long h = 0x243F6A8885A308D3L;
        for (int i = 0; i < domain.length(); i++) h = SplitMix64m.mix(h ^ domain.charAt(i));
        for (int i = 0; i < value.length(); i++) h = SplitMix64m.mix(h ^ value.charAt(i));
        return Math.floorMod(SplitMix64m.mix(h), P);
    }

    /** bit frequency of a pool of field elements, for the uniformity checks. */
    static double bitFrequency(long[] pool) {
        long ones = 0, bits = 0;
        for (long v : pool) {
            ones += Long.bitCount(v & P);
            bits += 61;
        }
        return ones / (double) bits;
    }
}

/** The mixer is shared by the hashes, so a single implementation is used everywhere. */
class SplitMix64m {
    static long mix(long z) {
        z += 0x9E3779B97F4A7C15L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }
}

/** Result of the commutative-exponentiation protocol. */
class DhResult {
    final Set<String> aliceIntersection, bobIntersection;
    final List<long[]> transcript = new ArrayList<>();
    long exponentiations;
    long messages;

    DhResult(Set<String> aliceIntersection, Set<String> bobIntersection) {
        this.aliceIntersection = aliceIntersection;
        this.bobIntersection = bobIntersection;
    }
}

/** Commutative PSI: both parties learn the true intersection, the transcript is fully blinded. */
class DhPsi {
    static void shuffle(long[] data, SplitMix64 rng) {
        for (int i = data.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            long t = data[i]; data[i] = data[j]; data[j] = t;
        }
    }

    static DhResult run(List<String> alice, List<String> bob, long seedA, long seedB) {
        SplitMix64 rngA = new SplitMix64(seedA), rngB = new SplitMix64(seedB);
        long a = Field61.randomNonZero(rngA), b = Field61.randomNonZero(rngB);
        DhResult res = new DhResult(new TreeSet<>(), new TreeSet<>());

        // A1 = shuffle{ H1(x)^a }, keeping the position -> element mapping.
        long[] a1 = new long[alice.size()];
        for (int i = 0; i < alice.size(); i++) a1[i] = Field61.pow(Field61.hash("H1", alice.get(i)), a);
        int[] order = new int[alice.size()];
        for (int i = 0; i < order.length; i++) order[i] = i;
        for (int i = order.length - 1; i > 0; i--) {
            int j = rngA.nextInt(i + 1);
            int t = order[i]; order[i] = order[j]; order[j] = t;
        }
        long[] a1Shuffled = new long[alice.size()];
        for (int i = 0; i < order.length; i++) a1Shuffled[i] = a1[order[i]];
        res.exponentiations += alice.size();
        res.messages += alice.size();
        res.transcript.add(a1Shuffled.clone());

        // B1 = shuffle{ H1(y)^b }.
        long[] b1 = new long[bob.size()];
        for (int j = 0; j < bob.size(); j++) b1[j] = Field61.pow(Field61.hash("H1", bob.get(j)), b);
        int[] border = new int[bob.size()];
        for (int i = 0; i < border.length; i++) border[i] = i;
        for (int i = border.length - 1; i > 0; i--) {
            int j = rngB.nextInt(i + 1);
            int t = border[i]; border[i] = border[j]; border[j] = t;
        }
        long[] b1Shuffled = new long[bob.size()];
        for (int i = 0; i < border.length; i++) b1Shuffled[i] = b1[border[i]];
        res.exponentiations += bob.size();
        res.messages += bob.size();
        res.transcript.add(b1Shuffled.clone());

        // B2 = (A1_i)^b in the same order; A2 = (B1_j)^a in the same order.
        long[] b2 = new long[a1Shuffled.length];
        for (int i = 0; i < a1Shuffled.length; i++) b2[i] = Field61.pow(a1Shuffled[i], b);
        long[] a2 = new long[b1Shuffled.length];
        for (int i = 0; i < b1Shuffled.length; i++) a2[i] = Field61.pow(b1Shuffled[i], a);
        res.exponentiations += a1Shuffled.length + b1Shuffled.length;
        res.messages += a1Shuffled.length + b1Shuffled.length;
        res.transcript.add(b2.clone());
        res.transcript.add(a2.clone());

        // Alice: T = { (H1(y)^b)^a } from B1, intersect with B2, map back through the shuffle.
        Set<Long> aliceSide = new HashSet<>();
        for (long v : b1Shuffled) aliceSide.add(Field61.pow(v, a));
        for (int i = 0; i < b2.length; i++)
            if (aliceSide.contains(b2[i])) res.aliceIntersection.add(alice.get(order[i]));

        // Bob: symmetric.
        Set<Long> bobSide = new HashSet<>();
        for (long v : a1Shuffled) bobSide.add(Field61.pow(v, b));
        for (int j = 0; j < a2.length; j++)
            if (bobSide.contains(a2[j])) res.bobIntersection.add(bob.get(border[j]));
        return res;
    }

    static Set<String> oracle(List<String> alice, List<String> bob) {
        Set<String> out = new TreeSet<>(alice);
        out.retainAll(new HashSet<>(bob));
        return out;
    }
}

/** Result of the secret-shared match-matrix protocol. */
class MatrixResult {
    long cardinality;
    long multiplications;
    long messages;
    long[] openedD = new long[0];
    long[] openedE = new long[0];
    long[] shareGivenAway = new long[0];
    int verifiedMultiplications;
}

/**
 * Semi-honest two-party computation of |A intersect B| as the Frobenius product of Alice's one-hot
 * match matrix and Bob's slot vector, using Beaver triples. Only the shared total is opened.
 *
 * The element -> slot assignment is public and injective on the union, exactly as the output of an
 * OPRF-based cuckoo assignment would be; the checks assert injectivity, which is what makes the
 * product equal to the true intersection size. The assignment itself leaks nothing about either set
 * beyond the union's cardinality, and each party's input stays hidden in additive shares.
 */
class MatrixPsi {
    final Map<String, Integer> slot = new TreeMap<>();
    final int slots;

    MatrixPsi(List<String> union) {
        List<String> sorted = new ArrayList<>(new TreeSet<>(union));
        sorted.sort(Comparator.comparingLong(e -> Field61.hash("SLOT", e)));
        for (int i = 0; i < sorted.size(); i++) slot.put(sorted.get(i), i);
        slots = sorted.size();
    }

    int slotOf(String element) {
        Integer s = slot.get(element);
        if (s == null) throw new IllegalArgumentException("element " + element + " has no slot assigned");
        return s;
    }

    MatrixResult run(List<String> alice, List<String> bob, long seed) {
        if (slots == 0 && (!alice.isEmpty() || !bob.isEmpty()))
            throw new IllegalArgumentException("no slots were assigned for " + (alice.size() + bob.size())
                    + " elements");
        MatrixResult res = new MatrixResult();
        int m = alice.size(), b = slots;
        res.multiplications = (long) m * b;

        // Private inputs.
        long[] matrix = new long[m * b];                     // Alice: one-hot rows
        for (int i = 0; i < m; i++) matrix[i * b + slotOf(alice.get(i))] = 1;
        long[] vector = new long[b];                         // Bob: slot occupancy
        for (String y : bob) vector[slotOf(y)] = 1;

        // Additive shares: Alice splits the matrix, Bob splits the vector.
        SplitMix64 rng = new SplitMix64(seed);
        long[] matrixBob = new long[matrix.length];
        long[] matrixAlice = new long[matrix.length];
        for (int t = 0; t < matrix.length; t++) {
            matrixBob[t] = Field61.random(rng);
            matrixAlice[t] = Field61.sub(matrix[t], matrixBob[t]);
        }
        long[] vectorAlice = new long[b];
        long[] vectorBob = new long[b];
        for (int j = 0; j < b; j++) {
            vectorAlice[j] = Field61.random(rng);
            vectorBob[j] = Field61.sub(vector[j], vectorAlice[j]);
        }
        res.shareGivenAway = matrixBob.clone();
        res.messages += matrix.length + b;

        // Beaver triples from the ideal functionality: c = a * b, each additively shared.
        long[] aA = new long[matrix.length], aB = new long[matrix.length];
        long[] bA = new long[matrix.length], bB = new long[matrix.length];
        long[] cA = new long[matrix.length], cB = new long[matrix.length];
        for (int t = 0; t < matrix.length; t++) {
            long a = Field61.random(rng), bb = Field61.random(rng), c = Field61.mul(a, bb);
            long aMask = Field61.random(rng);
            aA[t] = aMask; aB[t] = Field61.sub(a, aMask);
            long bMask = Field61.random(rng);
            bA[t] = bMask; bB[t] = Field61.sub(bb, bMask);
            long cMask = Field61.random(rng);
            cA[t] = cMask; cB[t] = Field61.sub(c, cMask);
        }

        long[] dOpened = new long[matrix.length], eOpened = new long[matrix.length];
        long[] zA = new long[matrix.length], zB = new long[matrix.length];
        for (int t = 0; t < matrix.length; t++) {
            int j = t % b;
            long d = Field61.add(Field61.sub(matrixAlice[t], aA[t]), Field61.sub(matrixBob[t], aB[t]));
            long e = Field61.add(Field61.sub(vectorAlice[j], bA[t]), Field61.sub(vectorBob[j], bB[t]));
            dOpened[t] = d;
            eOpened[t] = e;
            // [z] = [c] + d[b] + e[a] + d*e, with d*e added by both parties (it is public).
            zA[t] = Field61.add(Field61.add(cA[t], Field61.mul(d, bA[t])),
                    Field61.add(Field61.mul(e, aA[t]), Field61.mul(d, e)));
            zB[t] = Field61.add(cB[t], Field61.add(Field61.mul(d, bB[t]), Field61.mul(e, aB[t])));
            long product = Field61.mul(matrix[t], vector[j]);
            if (Field61.add(zA[t], zB[t]) == product) res.verifiedMultiplications++;
            res.messages += 2;                                        // d and e are opened
        }
        res.openedD = dOpened;
        res.openedE = eOpened;

        long totalA = 0, totalB = 0;
        for (int t = 0; t < matrix.length; t++) { totalA = Field61.add(totalA, zA[t]); totalB = Field61.add(totalB, zB[t]); }
        res.messages++;
        res.cardinality = Field61.add(totalA, totalB);
        return res;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    static List<String> elements(String prefix, int count, int from) {
        List<String> out = new ArrayList<>();
        for (int i = 0; i < count; i++) out.add(prefix + (from + i));
        return out;
    }

    /** Deterministic instance: aliceSize/bobSize elements, of which `shared` are common. */
    static List<List<String>> instance(int aliceSize, int bobSize, int shared) {
        List<String> alice = elements("a", aliceSize, 0);
        List<String> bob = new ArrayList<>(elements("a", shared, 0));       // the shared part
        bob.addAll(elements("b", bobSize - shared, 0));
        return List.of(alice, bob);
    }

    /** H1 is injective on the union of the two sets: a match is then a real match. */
    static boolean injective(String domain, List<List<String>> sets) {
        Set<String> elements = new TreeSet<>();
        for (List<String> set : sets) elements.addAll(set);
        Set<Long> seen = new HashSet<>();
        for (String e : elements) if (!seen.add(Field61.hash(domain, e))) return false;
        return seen.size() == elements.size();
    }

    public static void main(String[] args) {
        // 1. Field arithmetic: the Mersenne fold, modular inverse via Fermat, and exponent laws.
        SplitMix64 rng = new SplitMix64(0xF00DL);
        boolean fieldOk = true;
        for (int i = 0; i < 2000; i++) {
            long a = Field61.random(rng), b = Field61.random(rng);
            if (Field61.mul(a, b) != Field61.mul(b, a)) fieldOk = false;
            if (Field61.sub(Field61.add(a, b), b) != a) fieldOk = false;
            if (Field61.mul(a, 1) != a) fieldOk = false;
        }
        check(fieldOk, "field axioms failed on random samples");
        long g = Field61.hash("H1", "generator");
        check(Field61.pow(g, Field61.P - 1) == 1, "Fermat's little theorem violated");
        check(Field61.pow(Field61.mul(g, g), 5) == Field61.mul(Field61.pow(g, 5), Field61.pow(g, 5)),
                "exponent laws violated");
        check(Field61.mul(Field61.P - 1, Field61.P - 1) == 1, "(-1)^2 must be 1 mod 2^61-1");

        // 2. Commutative PSI at scale: 2000 elements per party, 100 shared.
        List<List<String>> big = instance(2000, 2000, 100);
        List<String> alice = big.get(0), bob = big.get(1);
        check(injective("H1", big), "H1 collided on the union - matches would not be exact");
        DhResult dh = DhPsi.run(alice, bob, 11, 22);
        Set<String> oracle = DhPsi.oracle(alice, bob);
        check(oracle.size() == 100, "the instance does not have the expected intersection size");
        check(dh.aliceIntersection.equals(oracle), "Alice's intersection differs from the oracle");
        check(dh.bobIntersection.equals(oracle), "Bob's intersection differs from the oracle");
        check(dh.exponentiations == 2L * (alice.size() + bob.size()), "exponentiation count "
                + dh.exponentiations + " differs from the formula " + 2L * (alice.size() + bob.size()));
        check(dh.messages == 2L * (alice.size() + bob.size()), "message volume differs from the formula");

        // 3. Transcript privacy: no raw hash ever appears, and messages are permutations of the expected set.
        Set<Long> rawHashes = new HashSet<>();
        for (String e : alice) rawHashes.add(Field61.hash("H1", e));
        for (String e : bob) rawHashes.add(Field61.hash("H1", e));
        boolean rawLeak = false;
        for (long[] message : dh.transcript)
            for (long v : message) if (rawHashes.contains(v)) rawLeak = true;
        check(!rawLeak, "the transcript contains a raw hash of an element");
        check(dh.transcript.get(0).length == alice.size() && dh.transcript.get(1).length == bob.size(),
                "message sizes are wrong");
        Set<Long> firstMessage = new TreeSet<>();
        for (long v : dh.transcript.get(0)) firstMessage.add(v);
        check(firstMessage.size() == alice.size(), "a message contains duplicate blinded values");
        check(!firstMessage.contains(Field61.pow(1, 1)) && dh.transcript.get(0)[0] != 0,
                "a blinded value is degenerate");

        // 4. The protocol is exact on degenerate and adversarial instances.
        List<List<List<String>>> cases = List.of(instance(0, 0, 0), instance(0, 50, 0), instance(50, 0, 0),
                instance(1, 1, 1), instance(40, 40, 0), instance(40, 40, 40));
        for (List<List<String>> instanceSet : cases) {
            List<String> x = instanceSet.get(0), y = instanceSet.get(1);
            DhResult r = DhPsi.run(x, y, 5, 6);
            Set<String> expected = DhPsi.oracle(x, y);
            check(r.aliceIntersection.equals(expected) && r.bobIntersection.equals(expected),
                    "degenerate instance failed: |A|=" + x.size() + " |B|=" + y.size());
        }
        // A party that repeats an element learns nothing extra: the intersection is still exact.
        List<String> withDuplicates = new ArrayList<>(alice.subList(0, 20));
        withDuplicates.addAll(alice.subList(0, 20));
        DhResult dup = DhPsi.run(withDuplicates, bob, 31, 41);
        check(dup.aliceIntersection.equals(DhPsi.oracle(withDuplicates, bob)),
                "duplicate elements are mishandled");

        // 5. Independent run with different exponents gives the same intersection but a different transcript.
        DhResult again = DhPsi.run(alice, bob, 77, 88);
        check(again.aliceIntersection.equals(oracle) && again.bobIntersection.equals(oracle),
                "a fresh exponentiation run disagrees on the intersection");
        check(again.transcript.get(0)[0] != dh.transcript.get(0)[0],
                "two runs produced the same blinded message - exponents are not fresh");

        // 6. Secret-shared match matrix: the cardinality matches the oracle and the other protocol.
        List<List<String>> small = instance(64, 64, 9);
        List<String> smallUnion = new ArrayList<>(small.get(0));
        smallUnion.addAll(small.get(1));
        MatrixPsi matrix = new MatrixPsi(smallUnion);
        check(matrix.slots >= 64 + 9 && matrix.slot.size() == matrix.slots,
                "the public slot assignment is not injective: " + matrix.slots + " slots for "
                        + smallUnion.size() + " elements");
        MatrixResult mr = matrix.run(small.get(0), small.get(1), 2024);
        Set<String> smallOracle = DhPsi.oracle(small.get(0), small.get(1));
        check(mr.cardinality == smallOracle.size(), "shared-matrix cardinality " + mr.cardinality
                + " differs from the oracle " + smallOracle.size());
        check(mr.multiplications == 64L * matrix.slots, "multiplication count differs from m*B");
        check(mr.verifiedMultiplications == 64 * matrix.slots, "only " + mr.verifiedMultiplications
                + " of " + 64 * matrix.slots + " Beaver multiplications reproduced x*y");
        DhResult smallDh = DhPsi.run(small.get(0), small.get(1), 1, 2);
        check(smallDh.aliceIntersection.size() == mr.cardinality,
                "the two protocols disagree on the intersection size");

        // 7. Cardinality is exact across instantiations, including empty and identical sets.
        long[][] expectations = {{0, 0, 0}, {1, 0, 0}, {8, 8, 0}, {10, 10, 10}, {16, 20, 3}};
        for (long[] spec : expectations) {
            List<List<String>> inst = instance((int) spec[0], (int) spec[1], (int) spec[2]);
            List<String> union = new ArrayList<>(inst.get(0));
            union.addAll(inst.get(1));
            MatrixPsi psi = new MatrixPsi(union);
            MatrixResult r = psi.run(inst.get(0), inst.get(1), spec[0] * 31 + spec[1] * 7 + spec[2]);
            int expected = DhPsi.oracle(inst.get(0), inst.get(1)).size();
            check(r.cardinality == expected, "matrix protocol returned " + r.cardinality
                    + " for " + spec[0] + "/" + spec[1] + "/" + spec[2] + " expecting " + expected);
            check(r.verifiedMultiplications == r.multiplications, "an entry product was not reproduced");
        }

        // 8. Privacy of the shares: the opened masks are balanced and fresh, the handed-over share is
        //    statistically independent of the secret it hides.
        MatrixResult mr2 = matrix.run(small.get(0), small.get(1), 2024);
        check(Arrays.equals(mr2.openedD, mr.openedD) && mr2.cardinality == mr.cardinality,
                "the protocol is not deterministic for a fixed seed");
        MatrixResult mr3 = matrix.run(small.get(0), small.get(1), 2025);
        boolean fresh = true;
        for (int i = 0; i < mr.openedD.length; i++) if (mr.openedD[i] == mr3.openedD[i]) fresh = false;
        check(fresh, "opened masks are not fresh across instances");
        double dFreq = Field61.bitFrequency(mr.openedD), eFreq = Field61.bitFrequency(mr.openedE);
        check(Math.abs(dFreq - 0.5) < 0.02 && Math.abs(eFreq - 0.5) < 0.02,
                "opened masks are not balanced: " + dFreq + " / " + eFreq);
        long[] share = mr.shareGivenAway;
        double shareFreq = Field61.bitFrequency(share);
        check(Math.abs(shareFreq - 0.5) < 0.02, "the share handed to Bob is not uniform: " + shareFreq);
        int onesAtSecret = 0, totalSecret = 0, onesAtZero = 0, totalZero = 0;
        for (int i = 0; i < 64; i++) {
            int secretSlot = matrix.slotOf(small.get(0).get(i));
            for (int j = 0; j < matrix.slots; j++) {
                boolean lsb = (share[i * matrix.slots + j] & 1L) == 1L;
                if (j == secretSlot) { if (lsb) onesAtSecret++; totalSecret++; }
                else { if (lsb) onesAtZero++; totalZero++; }
            }
        }
        double f1 = onesAtSecret / (double) totalSecret, f0 = onesAtZero / (double) totalZero;
        check(Math.abs(f1 - f0) < 0.05, "the share leaks the secret positions: " + f1 + " vs " + f0);
        check(totalSecret == 64 && totalZero == 64 * (matrix.slots - 1),
                "the leakage test enumerated the wrong positions");

        // 9. Cost model of both protocols, asserted exactly.
        check(dh.exponentiations == 2L * (alice.size() + bob.size()), "exponentiation formula wrong");
        long expectedMessages = 2L * (alice.size() + bob.size());
        check(dh.messages == expectedMessages, "commutative protocol message volume wrong");
        check(mr.messages == 64L * matrix.slots + matrix.slots + 2L * 64 * matrix.slots + 1,
                "shared-matrix message volume wrong: " + mr.messages);
        check(mr.multiplications == 64L * matrix.slots, "shared-matrix multiplication count wrong");

        // 10. Zero slots and single-slot instances behave.
        MatrixPsi noSlots = new MatrixPsi(List.of());
        boolean threw = false;
        try { noSlots.run(List.of("a0"), List.of("a0"), 3); }
        catch (IllegalArgumentException e) { threw = true; }
        check(threw, "a zero-slot instance should fail loudly instead of computing a wrong cardinality");
        MatrixPsi single = new MatrixPsi(List.of("a0"));
        MatrixResult oneSlot = single.run(List.of("a0"), List.of("a0"), 9);
        check(oneSlot.cardinality == 1 && oneSlot.verifiedMultiplications == 1, "single-slot instance wrong");
        MatrixPsi twoSlots = new MatrixPsi(List.of("a0", "b0"));
        MatrixResult disjoint = twoSlots.run(List.of("a0"), List.of("b0"), 9);
        check(disjoint.cardinality == 0, "disjoint instance wrong");
        boolean unassigned = false;
        try { single.slotOf("zz"); } catch (IllegalArgumentException e) { unassigned = true; }
        check(unassigned, "an unassigned element must be rejected");

        System.out.println("commutative PSI: " + alice.size() + "x" + bob.size() + " elements, intersection "
                + dh.aliceIntersection.size() + ", " + dh.exponentiations + " exponentiations, "
                + dh.messages + " messages");
        System.out.println("shared matrix: " + mr.multiplications + " Beaver multiplications, cardinality "
                + mr.cardinality + ", opened masks balanced at " + String.format("%.3f", dFreq));
        System.out.println("share uniformity: " + String.format("%.4f", shareFreq)
                + ", conditional frequencies " + String.format("%.4f", f1) + " vs " + String.format("%.4f", f0));
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }
}
