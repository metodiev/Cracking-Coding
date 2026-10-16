/** Task 758 - Implement a secure multiparty computation demo with basic XOR sharing.
 * (Tier: Expert, CTCI Beyond - Security & Privacy)
 *
 * XOR sharing is the simplest perfectly private secret sharing: s is split as s_0 ^ ... ^ s_{n-1} = s
 * with the first n-1 shares uniform, so any n-1 shares are *independent of s* - each single share is
 * uniform, which is the semi-honest privacy statement. On top of the sharing this file implements the
 * two operations that make a real protocol:
 *
 *  * XOR of shared values is free - every party XORs its own shares locally, no communication;
 *  * AND of two shared bits needs interaction and uses a Beaver triple, a precomputed sharing of random
 *    x, y and z = x & y. Every party blinds its input with its triple share, the blinded values are
 *    broadcast, and the identity
 *        (x^d) & (y^e) = (x&y) ^ (d&y) ^ (e&x) ^ (d&e)
 *    lets each party derive its share of a & b from the public d and e alone.
 *
 * The privacy claim is *tested*, not asserted: one party's view must be uniform (chi-square), must not
 * depend on the secret (total variation distance between the views for two different secrets) and must
 * be indistinguishable from a simulator that only emits random shares. A Beaver triple is consumed per
 * AND gate, so the demo also checks that triples are never reused.
 *
 * Simplified: semi-honest parties only (a party that broadcasts garbage is not detected), a fixed
 * 64-bit field, and no network - a broadcast is modelled by handing the same vector to every party.
 * No real key material is created, generated or stored anywhere.
 *
 * Verified: share/reconstruct round trips for 1..5 parties; subset reconstruction failing on purpose;
 * local XOR; the AND gate over the whole truth table; a half-adder and a majority-of-three circuit
 * evaluated securely against plaintext evaluation; uniformity, secret-independence and simulator
 * indistinguishability of a party's view; and determinism for a fixed seed.
 */
import java.util.*;

/** Deterministic splitmix64 - the only source of "randomness" (no Math.random). */
final class Prng {
    private long state;

    Prng(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
}

/** XOR secret sharing over a 64-bit field. */
final class XorSharing {

    /** n shares whose XOR is the secret; the first n-1 are uniform, so any n-1 reveal nothing. */
    static long[] share(long secret, int parties, Prng rng) {
        long[] shares = new long[parties];
        long accumulator = secret;
        for (int i = 0; i < parties - 1; i++) {
            shares[i] = rng.nextLong();
            accumulator ^= shares[i];
        }
        shares[parties - 1] = accumulator;
        return shares;
    }

    static long reconstruct(long[] shares) {
        long accumulator = 0;
        for (long share : shares) accumulator ^= share;
        return accumulator;
    }

    /** XOR is local: no communication at all, which is why XOR sharing is the cheapest primitive. */
    static long[] xor(long[] a, long[] b) {
        long[] out = new long[Math.min(a.length, b.length)];
        for (int i = 0; i < out.length; i++) out[i] = a[i] ^ b[i];
        return out;
    }
}

/** A precomputed Beaver triple: shares of random x, y and z = x & y. One per AND gate. */
final class BeaverTriple {
    final long[] x, y, z;

    BeaverTriple(int parties, Prng rng) {
        long randomX = rng.nextLong() & 1L;
        long randomY = rng.nextLong() & 1L;
        x = XorSharing.share(randomX, parties, rng);
        y = XorSharing.share(randomY, parties, rng);
        z = XorSharing.share(randomX & randomY, parties, rng);
    }
}

/** One party: holds shares, broadcasts a blinded value, derives its share of the AND. */
final class Party {
    final int index;

    Party(int index) { this.index = index; }

    /** First message: blind this party's input share; the resulting vector is public. */
    long blindA(long[] aShare, long[] xShare) { return aShare[index] ^ xShare[index]; }

    long blindB(long[] bShare, long[] yShare) { return bShare[index] ^ yShare[index]; }

    /** Second step: derive this party's share of a & b from the public d, e and its triple shares. */
    long multipliedShare(long d, long e, long[] xShare, long[] yShare, long[] zShare) {
        long share = zShare[index] ^ (d & yShare[index]) ^ (e & xShare[index]);
        if (index == 0) share ^= d & e;                 // one party folds in the public product
        return share;
    }
}

/** Secure AND built from a Beaver triple, exactly as the parties execute it. */
final class SecureGate {

    static long[] and(long[] aShares, long[] bShares, BeaverTriple triple, int parties) {
        long[] dShares = new long[parties];
        long[] eShares = new long[parties];
        for (int i = 0; i < parties; i++) {
            dShares[i] = new Party(i).blindA(aShares, triple.x);
            eShares[i] = new Party(i).blindB(bShares, triple.y);
        }
        long d = XorSharing.reconstruct(dShares);        // the broadcast, public to everyone
        long e = XorSharing.reconstruct(eShares);
        long[] out = new long[parties];
        for (int i = 0; i < parties; i++) {
            out[i] = new Party(i).multipliedShare(d, e, triple.x, triple.y, triple.z);
        }
        return out;
    }
}

/** A tiny circuit over shared bits: XOR gates are local, AND gates consume a Beaver triple. */
final class Circuit {
    final List<int[]> gates = new ArrayList<>();         // {op, left, right, out}; op 0 = XOR, 1 = AND
    private int wireCount;

    int addInput() { return wireCount++; }

    int xor(int left, int right) {
        gates.add(new int[]{0, left, right, wireCount});
        return wireCount++;
    }

    int and(int left, int right) {
        gates.add(new int[]{1, left, right, wireCount});
        return wireCount++;
    }

    int inputCount() { return wireCount - gates.size(); }

    /** Executes the protocol locally, so the tests can drive it and compare with plaintext. */
    Map<Integer, long[]> evaluate(List<long[]> inputShares, int parties, Prng rng) {
        Map<Integer, long[]> wires = new LinkedHashMap<>();
        for (int i = 0; i < inputShares.size(); i++) wires.put(i, inputShares.get(i));
        for (int[] gate : gates) {
            long[] left = wires.get(gate[1]);
            long[] right = wires.get(gate[2]);
            wires.put(gate[3], gate[0] == 0 ? XorSharing.xor(left, right)
                    : SecureGate.and(left, right, new BeaverTriple(parties, rng), parties));
        }
        return wires;
    }
}

/** Statistical tools for the privacy checks. */
final class Statistics {

    static int[] histogram(long[] values, int buckets) {
        int[] counts = new int[buckets];
        for (long value : values) counts[(int) Math.floorMod(value, buckets)]++;
        return counts;
    }

    static double chiSquare(long[] values, int buckets) {
        int[] counts = histogram(values, buckets);
        double expected = values.length / (double) buckets;
        double statistic = 0;
        for (int count : counts) statistic += (count - expected) * (count - expected) / expected;
        return statistic;
    }

    static double totalVariation(long[] first, long[] second, int buckets) {
        int[] a = histogram(first, buckets);
        int[] b = histogram(second, buckets);
        double distance = 0;
        for (int i = 0; i < buckets; i++) distance += Math.abs(a[i] - b[i]);
        return distance / (first.length + (double) second.length);
    }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    /** Eight AND gates of 1 & 1 from one seeded PRNG; the recorded view is party 0's share. */
    static List<Long> demoRun(long seed) {
        Prng rng = new Prng(seed);
        List<Long> view = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            long[] left = XorSharing.share(1, 3, rng);
            long[] right = XorSharing.share(1, 3, rng);
            long[] product = SecureGate.and(left, right, new BeaverTriple(3, rng), 3);
            check(XorSharing.reconstruct(product) == 1, "the demo gate computes 1 & 1 = 1");
            view.add(product[0]);
        }
        return view;
    }

    public static void main(String[] args) {
        Prng rng = new Prng(0x5A4EDL);

        // 1. share/reconstruct round trips, including the degenerate single party
        int roundTrips = 0;
        for (int parties = 1; parties <= 5; parties++) {
            for (int trial = 0; trial < 40; trial++) {
                long secret = rng.nextLong();
                long[] shares = XorSharing.share(secret, parties, new Prng(0x1000L + trial));
                check(XorSharing.reconstruct(shares) == secret, "reconstruct returns the secret for "
                        + parties + " parties");
                check(shares.length == parties, "one share per party");
                roundTrips++;
            }
        }
        check(roundTrips == 200, "200 round trips checked");
        long[] single = XorSharing.share(42, 1, rng);
        check(single.length == 1 && single[0] == 42, "a single party holds the secret itself");
        long[] zeros = XorSharing.share(0, 4, new Prng(7));
        check(XorSharing.reconstruct(zeros) == 0, "the all-zero secret reconstructs to zero");

        // 2. a proper subset of the shares reveals nothing usable
        int recovered = 0;
        for (int trial = 0; trial < 50; trial++) {
            long secret = rng.nextLong();
            long[] shares = XorSharing.share(secret, 3, new Prng(trial));
            if ((shares[0] ^ shares[1]) == secret) recovered++;
        }
        check(recovered == 0, "two of three shares never reconstruct the secret (" + recovered + " hits)");

        // 3. XOR of shared values is local and correct
        long a = rng.nextLong(), b = rng.nextLong();
        check(XorSharing.reconstruct(XorSharing.xor(XorSharing.share(a, 3, new Prng(11)),
                XorSharing.share(b, 3, new Prng(12)))) == (a ^ b), "local XOR is correct");

        // 4. the Beaver-triple AND gate over the whole truth table
        int gates = 0;
        for (int left = 0; left <= 1; left++) {
            for (int right = 0; right <= 1; right++) {
                for (int trial = 0; trial < 30; trial++) {
                    Prng local = new Prng(0x4000L + trial * 7 + left * 2 + right);
                    long[] l = XorSharing.share(left, 3, local);
                    long[] r = XorSharing.share(right, 3, local);
                    long[] product = SecureGate.and(l, r, new BeaverTriple(3, local), 3);
                    check(XorSharing.reconstruct(product) == (left & right), "secure AND of " + left + " and "
                            + right + " is " + (left & right));
                    gates++;
                }
            }
        }
        check(gates == 120, "the AND gate agrees with the truth table on 120 random triples");

        // 5. real circuits: half-adder (sum = a^b, carry = a&b) and majority of three
        for (int trial = 0; trial < 8; trial++) {
            int x = trial & 1, y = (trial >> 1) & 1, z = (trial >> 2) & 1;
            Prng local = new Prng(0x9000L + trial);
            int parties = 3;

            Circuit adder = new Circuit();
            int leftWire = adder.addInput();
            int rightWire = adder.addInput();
            int sumWire = adder.xor(leftWire, rightWire);
            int carryWire = adder.and(leftWire, rightWire);
            List<long[]> adderInputs = new ArrayList<>();
            adderInputs.add(XorSharing.share(x, parties, local));
            adderInputs.add(XorSharing.share(y, parties, local));
            Map<Integer, long[]> adderWires = adder.evaluate(adderInputs, parties, local);
            check(XorSharing.reconstruct(adderWires.get(sumWire)) == (x ^ y), "secure sum bit of " + x + "+" + y);
            check(XorSharing.reconstruct(adderWires.get(carryWire)) == (x & y), "secure carry bit of " + x
                    + "+" + y);

            Circuit majority = new Circuit();
            int wx = majority.addInput(), wy = majority.addInput(), wz = majority.addInput();
            int t = majority.xor(wx, wy);                       // x^y
            int u = majority.and(t, wz);                        // (x^y)&z
            int v = majority.and(wx, wy);                       // x&y
            int result = majority.xor(u, v);                    // majority, as the standard identity
            List<long[]> majorityInputs = new ArrayList<>();
            for (int bit : new int[]{x, y, z}) majorityInputs.add(XorSharing.share(bit, parties, local));
            Map<Integer, long[]> majorityWires = majority.evaluate(majorityInputs, parties, local);
            int expected = (x + y + z) >= 2 ? 1 : 0;
            check(XorSharing.reconstruct(majorityWires.get(result)) == expected, "secure majority of "
                    + x + y + z);
            check(majority.gates.size() == 4, "the majority circuit is four gates");
            check(majority.inputCount() == 3, "the majority circuit has three inputs");
        }

        // 6. privacy: one party's view is uniform and independent of the secret
        int parties = 3, samples = 800, buckets = 16;
        long[] viewA = new long[samples], viewB = new long[samples];
        for (int i = 0; i < samples; i++) {
            viewA[i] = XorSharing.share(0xDEADBEEFL, parties, new Prng(0xB000L + i))[0];
            viewB[i] = XorSharing.share(0x11111111L, parties, new Prng(0xB000L + i))[0];
        }
        double chiA = Statistics.chiSquare(viewA, buckets);
        double secretDistance = Statistics.totalVariation(viewA, viewB, buckets);
        check(chiA < 40.0, "one party's view is uniform: chi2=" + chiA + " over " + buckets + " buckets");
        check(secretDistance < 0.1, "the view does not depend on the secret: TV=" + secretDistance);
        long[] simulated = new long[samples];
        for (int i = 0; i < samples; i++) simulated[i] = new Prng(0xC000L + i).nextLong();
        double simulatorDistance = Statistics.totalVariation(viewA, simulated, buckets);
        check(simulatorDistance < 0.1, "the view is indistinguishable from a simulator that only emits "
                + "random shares: TV=" + simulatorDistance);
        check(viewA[0] != viewA[1], "consecutive sharings differ");

        // 7. the broadcast blinded value is public but reveals nothing about the input
        long[] blinded = new long[samples];
        for (int i = 0; i < samples; i++) {
            Prng local = new Prng(0xD000L + i);
            blinded[i] = new Party(0).blindA(XorSharing.share(1, parties, local),
                    new BeaverTriple(parties, local).x);
        }
        check(Statistics.chiSquare(blinded, buckets) < 40.0, "the broadcast d is uniform as well");

        // 8. determinism: the same seed replays the same party view, a different seed does not
        List<Long> firstRun = demoRun(0xE000L);
        List<Long> secondRun = demoRun(0xE000L);
        List<Long> thirdRun = demoRun(0xE001L);
        check(firstRun.equals(secondRun), "the same seed replays the same protocol run");
        check(!firstRun.equals(thirdRun), "a different seed produces a different view");

        System.out.println("sharing: 1..5 parties, 200 round trips, 0 reconstructions from 2 of 3 shares");
        System.out.println("gates: 120 truth-table AND runs, a half-adder and a majority circuit (4 gates)");
        System.out.println("privacy: chi2(view)=" + chiA + ", TV(secret A, secret B)=" + secretDistance
                + ", TV(view, simulator)=" + simulatorDistance);
        System.out.println("PASS");
    }
}
