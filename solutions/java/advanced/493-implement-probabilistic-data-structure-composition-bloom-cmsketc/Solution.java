/** Task 493 - Implement probabilistic data structure composition (bloom + cmsketch). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A SketchPipeline composes two sketches over the same item stream: a Bloom filter (relaxed
 * membership) and a Count-Min sketch (frequency estimation with d rows and w counters).
 * membership(item) = bloom.maybeContains(item) AND cms.estimate(item) >= 1 - so a positive
 * answer always implies an estimated frequency of at least 1, and a Bloom false positive whose
 * every CMS counter is still empty is rejected (the pipeline tightens membership without ever
 * introducing a false negative). Hashing is the stable task-254 scheme: h1 = fmix32(fnv1a(item)),
 * h2 = fmix32(fnv1a(item + '#')) | 1 for the Bloom probes, and per-row fmix32(fnv1a(item + '|r'))
 * for the CMS; all arithmetic is 32-bit integer work (Java ints wrap), identical in all four
 * languages. Documented guarantees: CMS estimates never underestimate the true count and
 * overestimate by at most epsilon*N with epsilon = e/w and delta = e^(-d) (w = 256, d = 4,
 * N = 754 updates); on 5000 absent probes the Bloom rate (m = 1024, k = 7, n = 250 items,
 * theory ~25.6%) stays under the documented 28% tolerance, and the pipeline rejects most of
 * those Bloom hits because their CMS counters are still empty - with the same Bloom filter and
 * a width-1024 CMS almost all are rejected. The exact popcounts and false-positive counts are
 * asserted as cross-language constants.
 */
import java.util.*;

class BloomFilter {
    final int m;
    final int k;
    final byte[] words;

    BloomFilter(int bits, int hashes) {
        m = bits;
        k = hashes;
        words = new byte[bits];
    }

    int[] positions(String item) {
        long h1 = Solution.fmix32(Solution.fnv1a(item)) & 0xFFFFFFFFL;
        long h2 = (Solution.fmix32(Solution.fnv1a(item + "#")) & 0xFFFFFFFFL) | 1L;
        int[] out = new int[k];
        for (int i = 0; i < k; i++) out[i] = (int) ((h1 + i * h2) % m);
        return out;
    }

    void add(String item) {
        for (int p : positions(item)) words[p] = 1;
    }

    boolean maybeContains(String item) {
        for (int p : positions(item)) {
            if (words[p] == 0) return false;
        }
        return true;
    }

    int popcount() {
        int total = 0;
        for (byte word : words) total += word;
        return total;
    }
}

class CountMinSketch {
    final int w;
    final int d;
    final int[][] counters;

    CountMinSketch(int width, int rows) {
        w = width;
        d = rows;
        counters = new int[rows][width];
    }

    int column(int row, String item) {
        return (int) ((Solution.fmix32(Solution.fnv1a(item + "|" + row)) & 0xFFFFFFFFL) % w);
    }

    void add(String item, int delta) {
        for (int row = 0; row < d; row++) counters[row][column(row, item)] += delta;
    }

    int estimate(String item) {
        int best = Integer.MAX_VALUE;
        for (int row = 0; row < d; row++) best = Math.min(best, counters[row][column(row, item)]);
        return best;
    }

    int total() {
        int total = 0;
        for (int[] row : counters) {
            for (int value : row) total += value;
        }
        return total;
    }
}

class SketchPipeline {
    final BloomFilter bloom;
    final CountMinSketch cms;

    SketchPipeline() {
        this(1024, 7, 256, 4);
    }

    SketchPipeline(int bloomBits, int bloomHashes, int cmsWidth, int cmsRows) {
        bloom = new BloomFilter(bloomBits, bloomHashes);
        cms = new CountMinSketch(cmsWidth, cmsRows);
    }

    void add(String item, int count) {
        bloom.add(item);
        cms.add(item, count);
    }

    int estimate(String item) {
        return cms.estimate(item);
    }

    /** True only when the Bloom filter accepts AND the CMS sees a nonzero count. */
    boolean membership(String item) {
        return bloom.maybeContains(item) && cms.estimate(item) >= 1;
    }
}

public class Solution {
    static int fnv1a(String text) {
        int h = 0x811C9DC5;
        for (int i = 0; i < text.length(); i++) {
            h ^= text.charAt(i);
            h *= 0x01000193;
        }
        return h;
    }

    /** MurmurHash3 avalanche: raw FNV-1a low bits are poorly mixed. */
    static int fmix32(int x) {
        x ^= x >>> 16;
        x *= 0x85EBCA6B;
        x ^= x >>> 13;
        x *= 0xC2B2AE35;
        x ^= x >>> 16;
        return x;
    }

    /** Deterministic insertion stream: item key-<j> gets 1 + seed % 5 copies. */
    static LinkedHashMap<String, Integer> lcgCounts(int itemCount, int seed) {
        LinkedHashMap<String, Integer> counts = new LinkedHashMap<>();
        for (int j = 0; j < itemCount; j++) {
            seed = (seed * 75 + 74) % 65537;
            counts.put("key-" + j, 1 + seed % 5);
        }
        return counts;
    }

    public static void main(String[] args) {
        LinkedHashMap<String, Integer> counts = lcgCounts(250, 123);
        int totalUpdates = 0;
        for (int count : counts.values()) totalUpdates += count;
        SketchPipeline pipeline = new SketchPipeline();
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            pipeline.add(entry.getKey(), entry.getValue());
        }

        double slack = (Math.E / pipeline.cms.w) * totalUpdates;
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            String item = entry.getKey();
            int trueCount = entry.getValue();
            check(pipeline.membership(item), "no false negatives");
            int estimate = pipeline.estimate(item);
            check(estimate >= trueCount, "CMS must never underestimate");
            check(estimate <= trueCount + slack, "CMS additive error bound");
            check(pipeline.bloom.maybeContains(item), "bloom member");
        }
        for (String item : counts.keySet()) {
            if (pipeline.membership(item)) check(pipeline.estimate(item) >= 1, "membership >= 1");
        }

        check(pipeline.estimate("never-seen") == 0, "cold estimate");
        check(!pipeline.membership("never-seen"), "cold membership");

        List<String> probes = new ArrayList<>();
        for (int j = 0; j < 5000; j++) probes.add("probe-" + j);
        int bloomFp = 0;
        int pipelineFp = 0;
        for (String item : probes) {
            if (pipeline.bloom.maybeContains(item)) bloomFp++;
            if (pipeline.membership(item)) pipelineFp++;
        }
        check(bloomFp <= 1400, "documented Bloom tolerance");
        check(pipelineFp <= bloomFp - 1000, "the pipeline must reject most Bloom false hits");
        check(pipelineFp <= 200, "documented pipeline tolerance");

        SketchPipeline tight = new SketchPipeline(1024, 7, 1024, 4);
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            tight.add(entry.getKey(), entry.getValue());
        }
        int tightBloomFp = 0;
        int tightPipelineFp = 0;
        for (String item : probes) {
            if (tight.bloom.maybeContains(item)) tightBloomFp++;
            if (tight.membership(item)) tightPipelineFp++;
        }
        check(tightPipelineFp < tightBloomFp, "cold counters must reject some Bloom hits");
        check(tightPipelineFp <= tightBloomFp - 1000, "tightening factor");
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            check(tight.membership(entry.getKey()), "tight member");
            check(tight.estimate(entry.getKey()) >= entry.getValue(), "tight lower bound");
            check(tight.estimate(entry.getKey()) <= entry.getValue() + 1, "wider sketch error");
        }

        check(totalUpdates == 754, "total updates " + totalUpdates);
        check(pipeline.bloom.popcount() == 842, "popcount " + pipeline.bloom.popcount());
        check(tight.bloom.popcount() == 842, "tight popcount");
        check(bloomFp == 1296, "bloom fps " + bloomFp);
        check(pipelineFp == 194, "pipeline fps " + pipelineFp);
        check(tightBloomFp == 1296, "tight bloom fps");
        check(tightPipelineFp == 5, "tight pipeline fps " + tightPipelineFp);

        SketchPipeline solo = new SketchPipeline();
        solo.add("solo", 3);
        check(solo.membership("solo"), "solo member");
        check(solo.estimate("solo") == 3, "solo estimate");
        for (String probe : new String[]{"solo#", "solo1", "sol", "proxy-7"}) {
            if (solo.membership(probe)) check(solo.estimate(probe) >= 1, "solo consistency");
        }
        check(solo.cms.total() == 12, "solo cms total");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
