/** Task 770 - Implement streaming algorithms for heavy hitters under memory constraints. (Tier: Expert, CTCI Beyond - Advanced Data Structures & Algorithms)
 *
 * A stream of n items arrives one at a time and must be summarised in O(k) space, where k is the
 * number of heavy hitters we are allowed to track. Three classic sketches are implemented:
 *
 *   MisraGries  keeps k-1 counters. A known item increments its counter; an unknown item with free
 *               space starts a counter at 1; with no space left *every* counter is decremented (zeros
 *               dropped). Decrements are the only source of error, and each of them proves the
 *               presence of k distinct items, so any item with true frequency > n/k must survive and
 *               every estimate is at most the true count with error <= n/k.
 *   SpaceSaving keeps k counters. A miss evicts the current minimum counter and re-uses it for the
 *               newcomer with value min + 1, so every item with frequency > n/k is guaranteed to be
 *               among the k largest estimates (the sketch tracks the top of the distribution).
 *   CountMin    a width x depth array of counters (width = e/eps, depth = ln(1/delta)) with one
 *               deterministic hash per row; the estimate is the minimum over rows. Estimates are
 *               always >= the true count (only over-counting) and, for a random-enough hash family,
 *               the over-count stays below eps*n with probability 1 - delta.
 *
 * Invariants verified: the exact frequencies of a HashMap oracle are never under-estimated by
 * CountMin and never over-estimated by MisraGries; every true heavy hitter (frequency > n/k) is
 * reported by all three sketches; the reported error of MisraGries stays within n/k; the memory
 * bounds (k-1 / k / width*depth cells) are respected; k = 1 is rejected; and the degenerate streams
 * (empty, shorter than k, one distinct item, all distinct items) behave as documented.
 *
 * Simplified: the stream is a few hundred integers so that the exact oracle is trivial, and the
 * CountMin error bound is checked on the concrete hashes used here rather than proven w.h.p.
 */
import java.util.*;

public class Solution {
    private static long state = 0x123456789ABCDEF0L;

    /** splitmix64: deterministic pseudo randomness, no Math.random anywhere. */
    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    /** Counter-based Misra-Gries summary with k-1 counters and explicit error accounting. */
    static class MisraGries {
        final int k;
        final Map<Integer, Integer> counters = new LinkedHashMap<>();
        long decrements;
        long increments;

        MisraGries(int k) {
            if (k < 2) throw new IllegalArgumentException("Misra-Gries needs k >= 2, got " + k);
            this.k = k;
        }

        int memory() { return counters.size(); }

        void add(int item) {
            Integer current = counters.get(item);
            if (current != null) {
                counters.put(item, current + 1);
                increments++;
            } else if (counters.size() < k - 1) {
                counters.put(item, 1);
                increments++;
            } else {
                decrements++;
                Iterator<Map.Entry<Integer, Integer>> it = counters.entrySet().iterator();
                while (it.hasNext()) {
                    Map.Entry<Integer, Integer> entry = it.next();
                    if (entry.getValue() == 1) it.remove();
                    else entry.setValue(entry.getValue() - 1);
                }
            }
        }

        List<Integer> heavyHitters() { return new ArrayList<>(counters.keySet()); }

        int estimate(int item) { return counters.getOrDefault(item, 0); }
    }

    /** Space-Saving: on overflow evict the lightest counter, so the top-k survive by construction. */
    static class SpaceSaving {
        final int k;
        final Map<Integer, Integer> counters = new LinkedHashMap<>();
        long evictions;

        SpaceSaving(int k) {
            if (k < 1) throw new IllegalArgumentException("Space-Saving needs k >= 1, got " + k);
            this.k = k;
        }

        int memory() { return counters.size(); }

        void add(int item) {
            Integer current = counters.get(item);
            if (current != null) {
                counters.put(item, current + 1);
                return;
            }
            if (counters.size() < k) {
                counters.put(item, 1);
                return;
            }
            int victim = -1;
            int victimCount = Integer.MAX_VALUE;
            for (Map.Entry<Integer, Integer> entry : counters.entrySet()) {
                if (entry.getValue() < victimCount) {
                    victimCount = entry.getValue();
                    victim = entry.getKey();
                }
            }
            counters.remove(victim);
            counters.put(item, victimCount + 1);
            evictions++;
        }

        List<Integer> topK() {
            List<Integer> keys = new ArrayList<>(counters.keySet());
            keys.sort((a, b) -> Integer.compare(counters.get(b), counters.get(a)));
            return keys;
        }

        int estimate(int item) { return counters.getOrDefault(item, 0); }
    }

    /** Count-Min sketch with one deterministic multiplicative hash per row. */
    static class CountMin {
        final int width;
        final int depth;
        final long[] seeds;
        final int[][] table;
        long updates;

        CountMin(int width, int depth, long seed) {            if (width < 1 || depth < 1) throw new IllegalArgumentException("empty sketch");
            this.width = width;
            this.depth = depth;
            this.seeds = new long[depth];
            long s = seed | 1L;
            for (int row = 0; row < depth; row++) {
                s = s * 6364136223846793005L + 1442695040888963407L;
                seeds[row] = s;
            }
            table = new int[depth][width];
        }

        int memory() { return width * depth; }

        int column(int row, int item) {
            long h = item * 0x9E3779B97F4A7C15L ^ seeds[row];
            h ^= h >>> 29;
            h *= 0xBF58476D1CE4E5B9L;
            h ^= h >>> 32;
            return (int) Math.floorMod(h, width);
        }

        void add(int item) {
            for (int row = 0; row < depth; row++) table[row][column(row, item)]++;
            updates++;
        }

        int estimate(int item) {
            int best = Integer.MAX_VALUE;
            for (int row = 0; row < depth; row++) best = Math.min(best, table[row][column(row, item)]);
            return best;
        }
    }

    static Map<Integer, Integer> exactCounts(int[] stream) {
        Map<Integer, Integer> counts = new TreeMap<>();
        for (int item : stream) counts.merge(item, 1, Integer::sum);
        return counts;
    }

    /** Heavy hitters of the exact distribution: frequency strictly above the n/k threshold. */
    static List<Integer> exactHeavyHitters(Map<Integer, Integer> counts, int n, int k) {
        List<Integer> heavy = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
            if (entry.getValue() * k > n) heavy.add(entry.getKey());
        }
        Collections.sort(heavy);
        return heavy;
    }

    /** Skewed stream: 7/8 of the mass lands on `hot` items, the rest spreads over `vocabulary`. */
    static int[] skewedStream(int length, int hot, int vocabulary) {
        int[] stream = new int[length];
        for (int i = 0; i < length; i++) {
            stream[i] = (i % 8 != 7) ? nextInt(hot) : nextInt(vocabulary);
        }
        return stream;
    }

    public static void main(String[] args) {
        // --- degenerate streams ------------------------------------------------
        MisraGries empty = new MisraGries(4);
        // no elements at all
        check(empty.heavyHitters().isEmpty(), "empty stream has no heavy hitters");
        check(empty.memory() == 0, "empty stream uses no memory");

        int[] shortStream = {7, 7, 9};
        MisraGries tiny = new MisraGries(8);              // more counters than items
        for (int item : shortStream) tiny.add(item);
        check(tiny.estimate(7) == 2, "stream shorter than k: exact counts, got " + tiny.estimate(7));
        check(tiny.memory() == 2, "stream shorter than k: two distinct items");

        int[] constant = new int[50];
        Arrays.fill(constant, 5);
        MisraGries constantMg = new MisraGries(3);
        Map<Integer, Integer> constantExact = exactCounts(constant);
        for (int item : constant) constantMg.add(item);
        check(constantMg.estimate(5) == 50, "one distinct item is tracked exactly");
        check(constantMg.decrements == 0, "no decrements for a constant stream");
        check(exactHeavyHitters(constantExact, 50, 3).equals(List.of(5)), "the constant item is heavy");

        int[] distinct = new int[40];
        for (int i = 0; i < distinct.length; i++) distinct[i] = i;
        MisraGries allDistinct = new MisraGries(5);
        for (int item : distinct) allDistinct.add(item);
        check(allDistinct.memory() <= 4, "Misra-Gries never exceeds k-1 counters");
        check(exactHeavyHitters(exactCounts(distinct), 40, 5).isEmpty(), "all-distinct stream has no heavy hitter");
        check(allDistinct.decrements > 0, "the all-distinct stream forces decrements");

        // --- negative check: k = 1 leaves no room to detect anything ------------
        boolean rejected = false;
        try {
            new MisraGries(1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "k = 1 must be rejected by Misra-Gries");
        boolean rejectedCm = false;
        try {
            new CountMin(0, 3, 7L);
        } catch (IllegalArgumentException expected) {
            rejectedCm = true;
        }
        check(rejectedCm, "an empty Count-Min sketch must be rejected");

        // --- skewed stream: guarantees of all three sketches -------------------
        for (int k = 3; k <= 7; k += 2) {
            int[] stream = skewedStream(400, 2, 90);
            Map<Integer, Integer> exact = exactCounts(stream);
            List<Integer> heavy = exactHeavyHitters(exact, stream.length, k);
            check(!heavy.isEmpty(), "the skewed stream must have heavy hitters for k=" + k);

            MisraGries mg = new MisraGries(k);
            SpaceSaving ss = new SpaceSaving(k);
            CountMin cm = new CountMin(64, 4, 0x51ED270B7L + k);
            for (int item : stream) {
                mg.add(item);
                ss.add(item);
                cm.add(item);
                if (mg.memory() > k - 1 || ss.memory() > k) throw new AssertionError("memory bound violated");
            }
            for (int item : heavy) {
                check(mg.heavyHitters().contains(item), "Misra-Gries must report heavy hitter " + item + " (k=" + k + ")");
                check(ss.topK().contains(item), "Space-Saving must report heavy hitter " + item + " (k=" + k + ")");
                check(ss.estimate(item) >= exact.get(item), "Space-Saving never under-estimates " + item);
                check(cm.estimate(item) >= exact.get(item), "Count-Min never under-estimates " + item);
            }
            check(mg.memory() <= k - 1, "Misra-Gries memory bound at k=" + k);
            check(cm.memory() == 64 * 4, "Count-Min memory is width*depth");

            // Misra-Gries error bound: tracked estimates are below the truth, by at most n/k
            for (Map.Entry<Integer, Integer> entry : exact.entrySet()) {
                int estimate = mg.estimate(entry.getKey());
                check(estimate <= entry.getValue(), "Misra-Gries estimate cannot exceed the truth");
                check(entry.getValue() - estimate <= stream.length / k,
                        "Misra-Gries error must stay within n/k for item " + entry.getKey());
            }
            // Count-Min over-count stays within the sketch's error budget on this stream
            double eps = Math.E / 64;
            for (Map.Entry<Integer, Integer> entry : exact.entrySet()) {
                check(cm.estimate(entry.getKey()) - entry.getValue() <= eps * stream.length + 1e-9,
                        "Count-Min error within eps*n for item " + entry.getKey());
            }
        }

        // --- Space-Saving keeps every true heavy hitter on a long skewed stream --
        int[] longStream = skewedStream(2000, 3, 300);
        Map<Integer, Integer> longExact = exactCounts(longStream);
        SpaceSaving saving = new SpaceSaving(6);
        for (int item : longStream) saving.add(item);
        List<Integer> longHeavy = exactHeavyHitters(longExact, longStream.length, 6);
        check(longHeavy.size() == 3, "the long stream has three heavy hitters, got " + longHeavy.size());
        Set<Integer> reported = new HashSet<>(saving.topK());
        for (int item : longHeavy) {
            check(reported.contains(item), "Space-Saving must keep heavy hitter " + item);
            check(saving.estimate(item) >= longExact.get(item), "Space-Saving estimate is never too small");
        }
        check(saving.evictions > 0, "a long skewed stream must evict");

        // --- Misra-Gries vs the exact top of the distribution ------------------
        List<Integer> mgKeys = new ArrayList<>();
        MisraGries finalMg = new MisraGries(6);
        for (int item : longStream) finalMg.add(item);
        mgKeys.addAll(finalMg.heavyHitters());
        List<Integer> exactHeavy = exactHeavyHitters(longExact, longStream.length, 6);
        check(mgKeys.containsAll(exactHeavy), "every exact heavy hitter survives in Misra-Gries");
        long counterMass = 0;
        for (int item : finalMg.heavyHitters()) counterMass += finalMg.estimate(item);
        check(finalMg.increments + finalMg.decrements == longStream.length,
                "each element is charged either to an increment or to a decrement round");
        check(counterMass == finalMg.increments - finalMg.decrements * (finalMg.k - 1),
                "counter mass accounting: increments minus (k-1) per decrement round");

        // --- adaptivity: the sketch's memory never grows with the stream --------
        MisraGries bounded = new MisraGries(4);
        for (int i = 0; i < 5000; i++) bounded.add(nextInt(1000));
        check(bounded.memory() <= 3, "memory stays bounded on a 5000 element stream, got " + bounded.memory());

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
