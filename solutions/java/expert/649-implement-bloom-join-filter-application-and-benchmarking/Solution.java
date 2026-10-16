/** Task 649 - Implement bloom join filter application and benchmarking. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * A Bloom filter over the join key of the build (smaller) side, exactly the "Bloom join" of a
 * distributed database: the filter is shipped to the probe side, probe rows that cannot match are never
 * sent into the join, and the survivors go through a normal hash join. The filter is a plain bit vector
 * of m bits with k independent hash positions derived from a deterministic integer mixer, sized from
 * the textbook formulas
 *      m = -n ln p / (ln 2)^2        k = round((m/n) ln 2)
 * for n = build rows and a target false-positive rate p.
 *
 * The two properties that make this safe and useful are checked separately and quantitatively: a Bloom
 * filter can never produce a false negative (so the filtered join must return exactly the same rows as
 * the unfiltered join, and every row the filter rejected must be re-checked to really have no match),
 * and its false-positive rate must stay near the theoretical value (1 - e^{-kn/m})^k, which is what the
 * bandwidth saving is bought with.
 *
 * Simplified (deliberately): the filter is kept in memory and passed by reference instead of being
 * serialised (no network bytes are counted, the "benchmark" counts rows and hash lookups), the joining
 * is a single int column, and the counter is not tracked through the probe (bit-setting is not
 * refilled or resized).
 *
 * Verified: no false negatives over every build key including duplicate-heavy builds; the measured
 * false-positive rate stays inside the design bound and close to theory; the filtered join equals the
 * unfiltered join and each rejected row is confirmed to have no match; a deliberately mis-built filter
 * (half the keys) does drop matching rows, which shows the equality check has teeth; the chosen k beats
 * bad choices of k; the filter rejects everything when the build side is empty, still never a false
 * negative with m = 1, and the degenerate arguments (k = 0, negative sizes) are rejected.
 */
import java.util.*;

class BloomFilter {

    final int bits;
    final int hashes;
    final long[] words;
    int inserted;

    BloomFilter(int bits, int hashes) {
        if (bits <= 0) throw new IllegalArgumentException("bit count must be positive");
        if (hashes <= 0) throw new IllegalArgumentException("hash count must be positive");
        this.bits = bits;
        this.hashes = hashes;
        this.words = new long[(bits + 63) >>> 6];
    }

    /** Textbook sizing for n rows and a target false-positive probability p. */
    static BloomFilter sizedFor(int rows, double targetFalsePositive) {
        if (rows <= 0) throw new IllegalArgumentException("row count must be positive");
        if (targetFalsePositive <= 0 || targetFalsePositive >= 1) {
            throw new IllegalArgumentException("target false-positive rate must be in (0,1)");
        }
        int m = (int) Math.ceil(-rows * Math.log(targetFalsePositive) / (Math.log(2) * Math.log(2)));
        int k = Math.max(1, (int) Math.round((m / (double) rows) * Math.log(2)));
        return new BloomFilter(Math.max(m, 1), k);
    }

    int position(int key, int index) {
        long z = key * 0x9E3779B97F4A7C15L + index * 0xBF58476D1CE4E5B9L + 0x94D049BB133111EBL;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z ^= z >>> 31;
        return (int) Math.floorMod(z, bits);
    }

    void add(int key) {
        for (int i = 0; i < hashes; i++) {
            int pos = position(key, i);
            words[pos >>> 6] |= 1L << (pos & 63);
        }
        inserted++;
    }

    boolean mightContain(int key) {
        for (int i = 0; i < hashes; i++) {
            int pos = position(key, i);
            if ((words[pos >>> 6] & (1L << (pos & 63))) == 0) return false;
        }
        return true;
    }

    int setBits() {
        int count = 0;
        for (long word : words) count += Long.bitCount(word);
        return count;
    }

    double loadFactor() { return setBits() / (double) bits; }

    /** Load factor predicted by the textbook occupancy formula 1 - (1 - 1/m)^(kn). */
    static double predictedLoad(int bits, int hashes, int rows) {
        return 1.0 - Math.pow(1.0 - 1.0 / bits, (double) hashes * rows);
    }

    /** Predicted false-positive rate (1 - e^{-kn/m})^k. */
    static double predictedFalsePositive(int bits, int hashes, int rows) {
        return Math.pow(1.0 - Math.exp(-(double) hashes * rows / bits), hashes);
    }
}

class BloomJoin {

    static final class Counters {
        long probeRows, rejected, passed, matches, hashLookups, falsePositives;

        @Override
        public String toString() {
            return "probe=" + probeRows + " rejected=" + rejected + " passed=" + passed + " matches="
                    + matches + " lookups=" + hashLookups;
        }
    }

    /** Full join with no filter and no bloom lookups. */
    static List<int[]> hashJoin(List<int[]> build, List<int[]> probe) {
        Map<Integer, List<int[]>> buckets = new HashMap<>();
        for (int[] row : build) buckets.computeIfAbsent(row[0], key -> new ArrayList<>()).add(row);
        List<int[]> out = new ArrayList<>();
        for (int[] row : probe) {
            List<int[]> bucket = buckets.get(row[0]);
            if (bucket == null) continue;
            for (int[] match : bucket) out.add(new int[]{row[0], row[1], match[1]});
        }
        return out;
    }

    /** Bloom join: rejected probe rows never touch the hash table. */
    static List<int[]> bloomJoin(List<int[]> build, List<int[]> probe, BloomFilter filter, Counters counters) {
        Map<Integer, List<int[]>> buckets = new HashMap<>();
        for (int[] row : build) buckets.computeIfAbsent(row[0], key -> new ArrayList<>()).add(row);
        List<int[]> out = new ArrayList<>();
        for (int[] row : probe) {
            counters.probeRows++;
            counters.hashLookups++;
            if (!filter.mightContain(row[0])) {
                counters.rejected++;
                continue;
            }
            counters.passed++;
            List<int[]> bucket = buckets.get(row[0]);
            if (bucket == null) {
                counters.falsePositives++;
                continue;
            }
            for (int[] match : bucket) {
                out.add(new int[]{row[0], row[1], match[1]});
                counters.matches++;
            }
        }
        return out;
    }

    static List<int[]> rows(int[] keys, int[] payloads) {
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < keys.length; i++) out.add(new int[]{keys[i], payloads[i]});
        return out;
    }

    static List<String> canonical(List<int[]> rows) {
        List<String> out = new ArrayList<>();
        for (int[] row : rows) out.add(Arrays.toString(row));
        Collections.sort(out);
        return out;
    }
}

class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }
}

public class Solution {

    static int[] keys(SplitMix64 rng, int rows, int domain) {
        int[] out = new int[rows];
        for (int i = 0; i < rows; i++) out[i] = rng.nextInt(domain);
        return out;
    }

    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(4242L);
        int domain = 20000;
        int[] buildKeys = keys(rng, 400, domain);
        int[] buildPayloads = new int[buildKeys.length];
        for (int i = 0; i < buildPayloads.length; i++) buildPayloads[i] = 1000 + i;
        int[] probeKeys = keys(rng, 1000, domain);
        int[] probePayloads = new int[probeKeys.length];
        for (int i = 0; i < probePayloads.length; i++) probePayloads[i] = i;

        List<int[]> build = BloomJoin.rows(buildKeys, buildPayloads);
        List<int[]> probe = BloomJoin.rows(probeKeys, probePayloads);

        BloomFilter filter = BloomFilter.sizedFor(build.size(), 0.01);
        for (int[] row : build) filter.add(row[0]);

        int noFalseNegatives = 0;
        for (int[] row : build) if (!filter.mightContain(row[0])) noFalseNegatives++;
        check(noFalseNegatives == 0, "no build key is ever rejected: " + noFalseNegatives);

        Set<Integer> distinct = new TreeSet<>();
        for (int key : buildKeys) distinct.add(key);
        int distinctMissed = 0;
        for (int key : distinct) if (!filter.mightContain(key)) distinctMissed++;
        check(distinctMissed == 0, "no distinct build key is rejected either");

        double predictedLoad = BloomFilter.predictedLoad(filter.bits, filter.hashes, build.size());
        check(Math.abs(filter.loadFactor() - predictedLoad) < 0.03,
                "load factor " + filter.loadFactor() + " tracks the formula " + predictedLoad);
        check(filter.hashes >= 6 && filter.hashes <= 8, "the sizing formula picks k near the optimum: "
                + filter.hashes + " hashes for " + filter.bits + " bits");

        int nonMembers = 40000, falsePositivesSeen = 0;
        for (int i = 0; i < nonMembers; i++) {
            int candidate = domain + i;
            if (filter.mightContain(candidate)) falsePositivesSeen++;
        }
        double measured = falsePositivesSeen / (double) nonMembers;
        double predicted = BloomFilter.predictedFalsePositive(filter.bits, filter.hashes, build.size());
        check(measured < 0.02, "measured false-positive rate stays near the 1% target: " + measured);
        check(measured < predicted * 2 + 0.005, "measured rate " + measured + " vs theory " + predicted);

        BloomJoin.Counters counters = new BloomJoin.Counters();
        List<int[]> bloomResult = BloomJoin.bloomJoin(build, probe, filter, counters);
        List<int[]> fullResult = BloomJoin.hashJoin(build, probe);
        check(BloomJoin.canonical(bloomResult).equals(BloomJoin.canonical(fullResult)),
                "the bloom join returns exactly the unfiltered join rows");
        check(counters.rejected + counters.passed == counters.probeRows,
                "every probe row is either rejected or passed: " + counters);
        check(counters.matches == fullResult.size(), "the same number of matching rows was produced");
        check(counters.rejected > 900, "the filter rejects almost every probe row: " + counters.rejected
                + " of " + counters.probeRows);
        check(counters.passed < 200, "only a small filtered stream reaches the join: " + counters.passed);
        check(counters.passed >= fullResult.size(), "at least every matching row survives the filter");
        check(counters.hashLookups == counters.probeRows, "each probe row costs one filter lookup: "
                + counters.hashLookups);

        Map<Integer, List<int[]>> buildBuckets = new HashMap<>();
        for (int[] row : build) buildBuckets.computeIfAbsent(row[0], key -> new ArrayList<>()).add(row);
        int rejectedWithMatch = 0;
        for (int[] row : probe) {
            if (filter.mightContain(row[0])) continue;
            if (buildBuckets.containsKey(row[0])) rejectedWithMatch++;
        }
        check(rejectedWithMatch == 0, "every rejected row really has no match in the build side: "
                + rejectedWithMatch);

        BloomFilter half = new BloomFilter(filter.bits, filter.hashes);
        for (int i = 0; i < build.size() / 2; i++) half.add(build.get(i)[0]);
        BloomJoin.Counters brokenCounters = new BloomJoin.Counters();
        List<int[]> brokenResult = BloomJoin.bloomJoin(build, probe, half, brokenCounters);
        check(!BloomJoin.canonical(brokenResult).equals(BloomJoin.canonical(fullResult)),
                "a mis-built filter really does drop matching rows, so the equality check has teeth");
        check(brokenCounters.rejected > 0, "the mis-built filter rejects rows: " + brokenCounters.rejected);

        double bestRate = 1;
        int bestK = 0;
        StringBuilder benchmark = new StringBuilder();
        for (int k : new int[]{1, 2, 4, filter.hashes, 12}) {
            BloomFilter candidate = new BloomFilter(filter.bits, k);
            for (int[] row : build) candidate.add(row[0]);
            int seen = 0;
            for (int i = 0; i < 20000; i++) if (candidate.mightContain(domain + i)) seen++;
            double rate = seen / 20000.0;
            benchmark.append("k=").append(k).append(" fp=").append(String.format("%.3f", rate)).append("  ");
            if (rate < bestRate) {
                bestRate = rate;
                bestK = k;
            }
        }
        check(bestK == filter.hashes, "the formula's k is the best of the tested values: " + bestK
                + " (rates: " + benchmark.toString().trim() + ")");
        check(bestRate < 0.02, "the best measured rate is within the design target: " + bestRate);
        BloomFilter singleHash = new BloomFilter(filter.bits, 1);
        for (int[] row : build) singleHash.add(row[0]);
        int singleSeen = 0;
        for (int i = 0; i < 20000; i++) if (singleHash.mightContain(domain + i)) singleSeen++;
        check(bestRate < singleSeen / 20000.0 / 2, "the tuned filter is twice as good as k=1: " + bestRate
                + " vs " + (singleSeen / 20000.0));

        List<int[]> emptyBuild = List.of();
        BloomFilter emptyFilter = BloomFilter.sizedFor(1, 0.01);
        BloomJoin.Counters emptyCounters = new BloomJoin.Counters();
        check(BloomJoin.bloomJoin(emptyBuild, probe, emptyFilter, emptyCounters).isEmpty(),
                "an empty build side joins to nothing");
        check(emptyCounters.rejected == probe.size(), "and every probe row is rejected: "
                + emptyCounters.rejected);
        check(BloomJoin.bloomJoin(build, List.of(), filter, new BloomJoin.Counters()).isEmpty(),
                "an empty probe side yields nothing");

        BloomFilter oneBit = new BloomFilter(1, 3);
        for (int[] row : build) oneBit.add(row[0]);
        int oneBitMisses = 0;
        for (int[] row : build) if (!oneBit.mightContain(row[0])) oneBitMisses++;
        check(oneBitMisses == 0, "even a one-bit filter never produces a false negative");
        check(oneBit.setBits() == 1, "a one-bit filter can only hold one bit set");
        BloomJoin.Counters oneBitCounters = new BloomJoin.Counters();
        check(BloomJoin.canonical(BloomJoin.bloomJoin(build, probe, oneBit, oneBitCounters))
                        .equals(BloomJoin.canonical(fullResult)),
                "a degenerate filter is slower but still correct: " + oneBitCounters.passed + " rows passed");

        List<int[]> duplicateBuild = new ArrayList<>();
        for (int i = 0; i < 50; i++) duplicateBuild.add(new int[]{7, i});
        BloomFilter duplicateFilter = BloomFilter.sizedFor(50, 0.05);
        for (int[] row : duplicateBuild) duplicateFilter.add(row[0]);
        List<int[]> duplicateProbe = List.of(new int[]{7, 99}, new int[]{8, 100});
        BloomJoin.Counters duplicateCounters = new BloomJoin.Counters();
        List<int[]> duplicateResult = BloomJoin.bloomJoin(duplicateBuild, duplicateProbe, duplicateFilter,
                duplicateCounters);
        check(duplicateResult.size() == 50, "duplicate build keys still produce every match: "
                + duplicateResult.size());
        check(BloomJoin.canonical(duplicateResult).equals(
                        BloomJoin.canonical(BloomJoin.hashJoin(duplicateBuild, duplicateProbe))),
                "duplicate-key joins agree between the two paths");

        boolean rejectedBadHashes = false;
        try {
            new BloomFilter(1024, 0);
        } catch (IllegalArgumentException expected2) {
            rejectedBadHashes = true;
        }
        check(rejectedBadHashes, "a filter with zero hashes is rejected");
        boolean rejectedBadBits = false;
        try {
            new BloomFilter(0, 3);
        } catch (IllegalArgumentException expected2) {
            rejectedBadBits = true;
        }
        check(rejectedBadBits, "a filter with zero bits is rejected");
        boolean rejectedBadTarget = false;
        try {
            BloomFilter.sizedFor(100, 1.5);
        } catch (IllegalArgumentException expected2) {
            rejectedBadTarget = true;
        }
        check(rejectedBadTarget, "a target false-positive rate outside (0,1) is rejected");

        System.out.println("filter: m=" + filter.bits + " bits, k=" + filter.hashes + ", load "
                + String.format("%.3f", filter.loadFactor()) + " (formula "
                + String.format("%.3f", predictedLoad) + ")");
        System.out.println("false positives measured " + String.format("%.4f", measured) + " vs predicted "
                + String.format("%.4f", predicted) + " on " + nonMembers + " non-member keys");
        System.out.println("bloom join: " + counters + " vs unfiltered join rows " + fullResult.size());
        System.out.println("k sweep: " + benchmark.toString().trim());
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
