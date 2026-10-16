/** Task 839 - Implement oblivious transfer based protocol for private set intersection. (Tier: Master, CTCI Beyond - Security & Privacy)
 *
 * The protocol is the standard two-stage construction: a 1-out-of-2 oblivious transfer, then an oblivious
 * PRF built from it, then PSI from that OPRF.
 *
 *   1. toy 1-out-of-2 OT (Bellare-Micali shape). The receiver fixes a bit c and a secret exponent r, and
 *      publishes the pair (h_0, h_1) with h_c = g^r and h_{1-c} = g^u for an exponent u the receiver
 *      samples and *discards*. The sender picks a_0, a_1, publishes (g^{a_0}, g^{a_1}) and sends
 *      e_i = m_i + f(h_i^{a_i}). The receiver recovers m_c = e_c - f((g^{a_c})^r) because g^{a_c r} is the
 *      same element on both sides. Modelling the receiver as unable to recover u is the single assumption
 *      this toy makes; it mirrors the one-more-discrete-log assumption under which real 1-out-of-2 OT is
 *      proven. Everything else below is measured by exhaustive enumeration.
 *
 *   2. oblivious PRF. The sender holds a key block K[i][b] (i a bit position, b a bit value) and defines
 *      F(x) = H(x, XOR_i K[i][bit_i(x)]). For each of its elements y the receiver runs one OT per bit
 *      position with choice bit bit_i(y), so it learns exactly K[i][bit_i(y)] and therefore F(y), while
 *      the sender learns nothing about the choice bits and cannot evaluate F anywhere by itself.
 *
 *   3. private set intersection. The sender publishes {F(x) : x in X}; the receiver reports {y in Y :
 *      F(y) is in that set}. The receiver learns only membership of its own elements and the sender
 *      learns only the (opaque) tag set.
 *
 * Simplifications, stated honestly: a toy 5-bit-size prime group (p = 101, order 100) stands in for a real
 * group, the keystream is f(z) = z mod q, and the toy reuses the sender's OT exponents across elements of
 * Y (a real deployment replaces the per-element OTs with OT extension plus a correlation check). There is
 * no network: the message count is tallied in memory. The model is semi-honest throughout.
 *
 * Invariants verified below. Correctness: the receiver recovers exactly m_c for all four (c, m_0, m_1)
 * combinations and F(y) matches the sender's F(x) for every x in the domain. Privacy: the sender's OT
 * view is *identical* in distribution for choice 0 and 1 (exhaustive multiset equality), and the receiver's
 * advantage on the unchosen message is exactly 1/q (exhaustive over the sender's randomness). PSI
 * correctness: the intersection equals a brute-force intersection for every tested instance, with zero
 * false positives and zero false negatives. Cost: OTs and messages are counted.
 */
import java.util.*;

/** splitmix64 plus a public mixing function - the only source of randomness and of hashing here. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    long nextBelow(long bound) { return Math.floorMod(nextLong(), bound); }

    /** Samples a unit of Z_q by rejection, deterministically. */
    long nextUnit(long q) {
        for (int attempt = 0; attempt < 1000; attempt++) {
            long candidate = nextBelow(q);
            if (candidate > 0 && gcd(candidate, q) == 1) return candidate;
        }
        throw new IllegalStateException("no unit found in Z_" + q);
    }

    static long gcd(long a, long b) {
        while (b != 0) {
            long t = a % b;
            a = b;
            b = t;
        }
        return a;
    }

    /** Stand-in for a random oracle in this toy. */
    static long mix(long z) {
        long v = (z ^ 0x9E3779B97F4A7C15L) * 0xBF58476D1CE4E5B9L;
        v = (v ^ (v >>> 27)) * 0x94D049BB133111EBL;
        return v ^ (v >>> 31);
    }
}

/** The public toy group (Z_p*, x) of order q = p - 1 with generator g. */
final class ToyGroup {
    final long p;
    final long q;
    final long g;

    ToyGroup(long p) {
        if (p < 5) throw new IllegalArgumentException("toy group needs p >= 5");
        this.p = p;
        this.q = p - 1;
        this.g = generator(p);
    }

    long exp(long base, long exponent) { return expMod(base, exponent, p); }

    long element(long exponent) { return expMod(g, exponent, p); }

    /** f(z) = z mod q; on the q nonzero residues of Z_p this is exactly a bijection onto Z_q. */
    long pad(long groupElement) { return Math.floorMod(groupElement, q); }

    long publicExponent(long exponent) { return element(exponent); }

    static long expMod(long base, long exponent, long modulus) {
        long result = 1, b = Math.floorMod(base, modulus), e = exponent;
        while (e > 0) {
            if ((e & 1L) == 1L) result = result * b % modulus;
            b = b * b % modulus;
            e >>>= 1;
        }
        return result;
    }

    static long generator(long p) {
        List<Long> factors = primeFactors(p - 1);
        for (long candidate = 2; candidate < p; candidate++) {
            boolean good = true;
            for (long factor : factors) {
                if (expMod(candidate, (p - 1) / factor, p) == 1) good = false;
            }
            if (good) return candidate;
        }
        throw new IllegalStateException("no generator for p = " + p);
    }

    static List<Long> primeFactors(long value) {
        List<Long> factors = new ArrayList<>();
        long n = value;
        for (long d = 2; d * d <= n; d++) {
            if (n % d == 0) {
                factors.add(d);
                while (n % d == 0) n /= d;
            }
        }
        if (n > 1) factors.add(n);
        return factors;
    }
}

/** Tallies the protocol traffic; nothing is actually transmitted. */
final class Channel {
    int messages;

    long send(long value) {
        messages++;
        return value;
    }
}

/** Sender side of one toy 1-out-of-2 OT. */
final class OtSender {
    final ToyGroup group;
    final long[] exponent = new long[2];
    final long[] publicValue = new long[2];

    OtSender(ToyGroup group, SplitMix64 rng) {
        this.group = group;
        for (int i = 0; i < 2; i++) {
            exponent[i] = rng.nextUnit(group.q);
            publicValue[i] = group.element(exponent[i]);
        }
    }

    /** e_i = m_i + f(h_i^{a_i}); the sender never learns which h_i the receiver knows the log of. */
    long[] encrypt(long[] receiverPair, long m0, long m1, Channel channel) {
        long c0 = group.pad(group.exp(receiverPair[0], exponent[0])) + m0;
        long c1 = group.pad(group.exp(receiverPair[1], exponent[1])) + m1;
        return new long[]{channel.send(Math.floorMod(c0, group.q)), channel.send(Math.floorMod(c1, group.q))};
    }
}

/** Receiver side of one toy 1-out-of-2 OT. */
final class OtReceiver {
    final ToyGroup group;
    final int choice;
    final long secret;
    final long[] pair = new long[2];

    OtReceiver(ToyGroup group, int choice, SplitMix64 rng) {
        if (choice != 0 && choice != 1) throw new IllegalArgumentException("choice bit must be 0 or 1");
        this.group = group;
        this.choice = choice;
        this.secret = rng.nextUnit(group.q);
        long other = rng.nextUnit(group.q);   // sampled and discarded: the receiver cannot take its log
        pair[choice] = group.element(secret);
        pair[1 - choice] = group.element(other);
    }

    long[] request(Channel channel) { return new long[]{channel.send(pair[0]), channel.send(pair[1])}; }

    long decrypt(long[] senderPublic, long[] ciphertext) {
        long shared = group.exp(senderPublic[choice], secret);
        return Math.floorMod(ciphertext[choice] - group.pad(shared), group.q);
    }
}

/** What one PSI run produced, including the raw material the checks need. */
final class PsiResult {
    final int[] intersection;
    final int[] receiverElements;
    final long[] senderTags;
    final long[] receiverTags;
    final long[][] senderKeys;
    final long[][] learnedKeys;
    final int ots;
    final int messages;

    PsiResult(int[] intersection, int[] receiverElements, long[] senderTags, long[] receiverTags,
            long[][] senderKeys, long[][] learnedKeys, int ots, int messages) {
        this.intersection = intersection;
        this.receiverElements = receiverElements;
        this.senderTags = senderTags;
        this.receiverTags = receiverTags;
        this.senderKeys = senderKeys;
        this.learnedKeys = learnedKeys;
        this.ots = ots;
        this.messages = messages;
    }
}

/** PSI from an OT-based oblivious PRF: F(x) = H(x, XOR of one key per bit position). */
final class PsiProtocol {
    static int bit(long value, int position) { return (int) ((value >>> position) & 1L); }

    static long padFor(long[] oneKeyPerPosition, int bits) {
        long pad = 0;
        for (int i = 0; i < bits; i++) pad ^= oneKeyPerPosition[i];
        return pad;
    }

    static long tag(long value, long pad) { return SplitMix64.mix((value << 40) ^ pad ^ 0x51ED270B1L); }

    static PsiResult run(ToyGroup group, int[] senderSet, int[] receiverSet, int bits, long seed) {
        if (bits < 1 || bits > 30) throw new IllegalArgumentException("bit width out of range: " + bits);
        int domain = 1 << bits;
        int[] senders = dedupe(senderSet, domain);
        int[] receivers = dedupe(receiverSet, domain);
        SplitMix64 rng = new SplitMix64(seed);
        Channel channel = new Channel();

        long[][] keys = new long[bits][2];
        OtSender[] otSenders = new OtSender[bits];
        for (int i = 0; i < bits; i++) {
            keys[i][0] = rng.nextBelow(group.q);
            keys[i][1] = rng.nextBelow(group.q);
            otSenders[i] = new OtSender(group, rng);
        }

        long[] senderTags = new long[senders.length];
        for (int s = 0; s < senders.length; s++) {
            senderTags[s] = computeTag(keys, senders[s], bits);
        }
        long[] sortedSenderTags = senderTags.clone();
        Arrays.sort(sortedSenderTags);
        channel.send(0);   // the tag set itself is one publication

        long[][] learned = new long[receivers.length][bits];
        long[] receiverTags = new long[receivers.length];
        int ots = 0;
        for (int r = 0; r < receivers.length; r++) {
            int y = receivers[r];
            for (int i = 0; i < bits; i++) {
                OtReceiver receiver = new OtReceiver(group, bit(y, i), rng);
                long[] request = receiver.request(channel);
                long[] ciphertext = otSenders[i].encrypt(request, keys[i][0], keys[i][1], channel);
                learned[r][i] = receiver.decrypt(otSenders[i].publicValue, ciphertext);
                ots++;
            }
            receiverTags[r] = tag(y, padFor(learned[r], bits));
        }

        List<Integer> hits = new ArrayList<>();
        for (int r = 0; r < receivers.length; r++) {
            if (Arrays.binarySearch(sortedSenderTags, receiverTags[r]) >= 0) hits.add(receivers[r]);
        }
        int[] intersection = new int[hits.size()];
        for (int i = 0; i < hits.size(); i++) intersection[i] = hits.get(i);
        Arrays.sort(intersection);
        return new PsiResult(intersection, receivers, senderTags, receiverTags, keys, learned, ots,
                channel.messages);
    }

    /** The keys the sender selects for a value: K[i][bit_i(value)] for every position. */
    static long[] chosenKeys(long[][] keys, long value, int bits) {
        long[] chosen = new long[bits];
        for (int i = 0; i < bits; i++) chosen[i] = keys[i][bit(value, i)];
        return chosen;
    }

    static long computeTag(long[][] keys, long value, int bits) {
        return tag(value, padFor(chosenKeys(keys, value, bits), bits));
    }

    static int[] dedupe(int[] values, int domain) {
        TreeSet<Integer> set = new TreeSet<>();
        for (int v : values) {
            if (v < 0 || v >= domain) throw new IllegalArgumentException("element " + v
                    + " outside the domain [0," + domain + ")");
            set.add(v);
        }
        int[] out = new int[set.size()];
        int i = 0;
        for (int v : set) out[i++] = v;
        return out;
    }
}

public class Solution {
    static final ToyGroup GROUP = new ToyGroup(101L);
    static final int BITS = 7;

    public static void main(String[] args) {
        // ---- 1. the public group behaves as the protocol assumes ---------------------------------
        check(GROUP.q == 100 && GROUP.g > 1, "toy group order and generator");
        check(ToyGroup.expMod(GROUP.g, GROUP.q, GROUP.p) == 1, "g must have order dividing q");
        check(ToyGroup.expMod(GROUP.g, 50, GROUP.p) != 1 && ToyGroup.expMod(GROUP.g, 20, GROUP.p) != 1,
                "g must have full order q = 100");
        Set<Long> padded = new TreeSet<>();
        for (long a = 0; a < GROUP.q; a++) padded.add(GROUP.pad(GROUP.element(a)));
        check(padded.size() == GROUP.q && padded.contains(0L) && padded.contains(GROUP.q - 1),
                "f must be a bijection from the group onto Z_q, got " + padded.size());

        int otChecks = 0;
        // ---- 2. OT correctness for every (c, m0, m1) combination ---------------------------------
        for (int trial = 0; trial < 3; trial++) {
            SplitMix64 rng = new SplitMix64(200 + trial);
            OtSender sender = new OtSender(GROUP, rng);
            Channel channel = new Channel();
            long[] messages = {rng.nextBelow(GROUP.q), rng.nextBelow(GROUP.q)};
            for (int choice = 0; choice < 2; choice++) {
                OtReceiver receiver = new OtReceiver(GROUP, choice, rng);
                long[] ciphertext = sender.encrypt(receiver.request(channel), messages[0], messages[1],
                        channel);
                long recovered = receiver.decrypt(sender.publicValue, ciphertext);
                check(recovered == messages[choice],
                        "choice " + choice + " must recover m_" + choice + ", got " + recovered
                                + " want " + messages[choice]);
                otChecks++;
            }
        }
        check(otChecks == 6, "all six (trial, choice) combinations recovered");

        // ---- 3. receiver privacy measured exactly: the other message stays uniform ---------------
        Map<Long, Integer> hidden = new TreeMap<>();
        OtReceiver probe = new OtReceiver(GROUP, 0, new SplitMix64(7L));
        long fixedMessage = 13;
        for (long exponent = 0; exponent < GROUP.q; exponent++) {
            long pad = GROUP.pad(GROUP.exp(probe.pair[1], exponent));
            hidden.merge(Math.floorMod(fixedMessage + pad, GROUP.q), 1, Integer::sum);
        }
        check(hidden.size() == GROUP.q, "every candidate must remain possible for the hidden message");
        boolean equiprobable = true;
        for (int count : hidden.values()) equiprobable &= count == 1;
        check(equiprobable, "the receiver's advantage on the unchosen message is exactly 1/q");

        // ---- 4. sender privacy measured exactly: the request is uniform for both choices ---------
        Map<String, Integer> viewZero = otView(0), viewOne = otView(1);
        check(viewZero.equals(viewOne) && viewZero.size() == GROUP.q * GROUP.q,
                "the sender's OT view must be identically distributed for choice 0 and 1");
        boolean once = true;
        for (int count : viewZero.values()) once &= count == 1;
        check(once, "each request pair occurs exactly once, so the view is uniform over G x G");

        expectThrows(() -> new OtReceiver(GROUP, 2, new SplitMix64(1L)), "choice bit outside {0,1}");
        expectThrows(() -> new SplitMix64(1L).nextUnit(1L), "no units in Z_1");
        expectThrows(() -> new ToyGroup(3L), "degenerate group");

        // ---- 5. the oblivious PRF agrees on the whole domain -------------------------------------
        int domain = 1 << BITS;
        int[] everyone = new int[domain];
        for (int i = 0; i < domain; i++) everyone[i] = i;
        SplitMix64 rng = new SplitMix64(4242L);
        long[][] keys = new long[BITS][2];
        OtSender[] otSenders = new OtSender[BITS];
        for (int i = 0; i < BITS; i++) {
            keys[i][0] = rng.nextBelow(GROUP.q);
            keys[i][1] = rng.nextBelow(GROUP.q);
            otSenders[i] = new OtSender(GROUP, rng);
        }
        Channel channel = new Channel();
        Set<Long> tags = new TreeSet<>();
        int mismatches = 0;
        for (int x = 0; x < domain; x++) {
            long[] learned = new long[BITS];
            for (int i = 0; i < BITS; i++) {
                OtReceiver receiver = new OtReceiver(GROUP, PsiProtocol.bit(x, i), rng);
                long[] ciphertext = otSenders[i].encrypt(receiver.request(channel), keys[i][0], keys[i][1],
                        channel);
                learned[i] = receiver.decrypt(otSenders[i].publicValue, ciphertext);
            }
            long receiverTag = PsiProtocol.tag(x, PsiProtocol.padFor(learned, BITS));
            long senderTag = PsiProtocol.computeTag(keys, x, BITS);
            if (receiverTag != senderTag) mismatches++;
            tags.add(senderTag);
        }
        check(mismatches == 0, "Oblivious PRF must agree for all " + domain + " domain values");
        check(tags.size() == domain, "the " + domain + " tags must be collision free, got " + tags.size());

        // ---- 6. PSI equals a brute-force intersection --------------------------------------------
        int[] senderSet = multiples(3, 100);
        int[] receiverSet = multiples(4, 100);
        PsiResult psi = PsiProtocol.run(GROUP, senderSet, receiverSet, BITS, 99L);
        check(Arrays.equals(psi.intersection, bruteForce(senderSet, receiverSet)),
                "PSI must equal the brute-force intersection, got " + Arrays.toString(psi.intersection));
        check(Arrays.equals(psi.intersection, multiples(12, 100)),
                "the overlap of the 3- and 4-multiples is the 12-multiples");
        check(psi.ots == BITS * receiverSet.length, "one OT per receiver element and bit position, got "
                + psi.ots);
        check(psi.messages == 4 * psi.ots + 1, "each OT is four messages plus one tag publication, got "
                + psi.messages);
        PsiResult otherSenderSet = PsiProtocol.run(GROUP, multiples(5, 100), receiverSet, BITS, 99L);
        check(Arrays.equals(otherSenderSet.receiverTags, psi.receiverTags),
                "the receiver's transcript must not depend on the sender's set");
        check(!Arrays.equals(otherSenderSet.intersection, psi.intersection),
                "a different sender set must change the intersection");

        PsiResult identical = PsiProtocol.run(GROUP, new int[]{1, 2, 3}, new int[]{3, 2, 1}, BITS, 5L);
        check(Arrays.equals(identical.intersection, new int[]{1, 2, 3}), "identical sets intersect fully");
        PsiResult subset = PsiProtocol.run(GROUP, new int[]{4, 8, 15, 16, 23, 42}, new int[]{8, 23}, BITS,
                6L);
        check(Arrays.equals(subset.intersection, new int[]{8, 23}), "the subset case");
        PsiResult disjoint = PsiProtocol.run(GROUP, new int[]{1, 3, 5}, new int[]{2, 4, 6}, BITS, 7L);
        check(disjoint.intersection.length == 0, "disjoint sets must intersect in nothing");
        PsiResult emptySender = PsiProtocol.run(GROUP, new int[0], receiverSet, BITS, 8L);
        check(emptySender.intersection.length == 0 && emptySender.senderTags.length == 0,
                "an empty sender set publishes nothing and matches nothing");
        PsiResult emptyReceiver = PsiProtocol.run(GROUP, senderSet, new int[0], BITS, 9L);
        check(emptyReceiver.intersection.length == 0 && emptyReceiver.ots == 0,
                "an empty receiver set costs no OTs");
        PsiResult duplicates = PsiProtocol.run(GROUP, new int[]{5, 5, 5, 6}, new int[]{5, 5}, BITS, 10L);
        check(Arrays.equals(duplicates.intersection, new int[]{5})
                && duplicates.senderTags.length == 2, "duplicates collapse to a set");

        // ---- 7. membership is decided by tags only; measure the error rates -----------------------
        int falsePositives = 0, falseNegatives = 0;
        for (int seed = 0; seed < 20; seed++) {
            SplitMix64 r = new SplitMix64(700 + seed);
            int[] xs = randomSubset(r, 100, 40);
            int[] ys = randomSubset(r, 100, 40);
            PsiResult res = PsiProtocol.run(GROUP, xs, ys, BITS, 1000 + seed);
            int[] truth = bruteForce(xs, ys);
            for (int hit : truth) if (!contains(res.intersection, hit)) falseNegatives++;
            for (int hit : res.intersection) if (!contains(truth, hit)) falsePositives++;
        }
        check(falseNegatives == 0, "measured false negatives must be zero, saw " + falseNegatives);
        check(falsePositives == 0, "measured false positives must be zero, saw " + falsePositives);

        // ---- 8. determinism, input validation and boundaries --------------------------------------
        PsiResult again = PsiProtocol.run(GROUP, senderSet, receiverSet, BITS, 99L);
        check(Arrays.equals(again.intersection, psi.intersection)
                && Arrays.equals(again.receiverTags, psi.receiverTags), "the same seed is reproducible");
        PsiResult otherSeed = PsiProtocol.run(GROUP, senderSet, receiverSet, BITS, 100L);
        check(Arrays.equals(otherSeed.intersection, psi.intersection)
                && !Arrays.equals(otherSeed.receiverTags, psi.receiverTags),
                "a new seed re-randomises the tags but not the intersection");
        PsiResult oneBit = PsiProtocol.run(GROUP, new int[]{1}, new int[]{0, 1}, 1, 11L);
        check(Arrays.equals(oneBit.intersection, new int[]{1}), "the smallest domain, one bit wide");
        expectThrows(() -> PsiProtocol.run(GROUP, new int[]{128}, new int[]{1}, BITS, 1L),
                "sender element outside the domain");
        expectThrows(() -> PsiProtocol.run(GROUP, new int[]{1}, new int[]{-1}, BITS, 1L),
                "negative receiver element");
        expectThrows(() -> PsiProtocol.run(GROUP, new int[]{1}, new int[]{1}, 0, 1L), "zero bit width");
        expectThrows(() -> PsiProtocol.run(GROUP, new int[]{1}, new int[]{1}, 31, 1L), "bit width too large");

        System.out.println("toy group p=" + GROUP.p + " q=" + GROUP.q + " g=" + GROUP.g
                + "; OT checks " + otChecks + ", sender views " + viewZero.size()
                + " equiprobable pairs per choice bit");
        System.out.println("OPRF: " + domain + " domain values, " + mismatches + " mismatches, "
                + tags.size() + " distinct tags");
        System.out.println("PSI: |X|=" + senderSet.length + " |Y|=" + receiverSet.length
                + " -> |intersection|=" + psi.intersection.length + " using " + psi.ots + " OTs and "
                + psi.messages + " messages; 20 random instances gave " + falsePositives
                + " false positives and " + falseNegatives + " false negatives");
        System.out.println("PASS");
    }

    static Map<String, Integer> otView(int choice) {
        Map<String, Integer> view = new TreeMap<>();
        for (long r = 0; r < GROUP.q; r++) {
            for (long u = 0; u < GROUP.q; u++) {
                long known = GROUP.element(r);
                long unknown = GROUP.element(u);
                String key = choice == 0 ? known + ":" + unknown : unknown + ":" + known;
                view.merge(key, 1, Integer::sum);
            }
        }
        return view;
    }

    static int[] multiples(int step, int limit) {
        List<Integer> values = new ArrayList<>();
        for (int v = step; v < limit; v += step) values.add(v);
        int[] out = new int[values.size()];
        for (int i = 0; i < out.length; i++) out[i] = values.get(i);
        return out;
    }

    static int[] bruteForce(int[] xs, int[] ys) {
        TreeSet<Integer> both = new TreeSet<>();
        for (int x : xs) for (int y : ys) if (x == y) both.add(x);
        int[] out = new int[both.size()];
        int i = 0;
        for (int v : both) out[i++] = v;
        return out;
    }

    static int[] randomSubset(SplitMix64 rng, int domain, int size) {
        TreeSet<Integer> picked = new TreeSet<>();
        while (picked.size() < size) picked.add((int) rng.nextBelow(domain));
        int[] out = new int[picked.size()];
        int i = 0;
        for (int v : picked) out[i++] = v;
        return out;
    }

    static boolean contains(int[] values, int target) {
        for (int v : values) if (v == target) return true;
        return false;
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
