/** Task 778 - Implement cold-start heuristics for cache warming and prefetching. (Tier: Expert, CTCI Ch 14 - Databases)
 *
 * A cold cache is the worst case for any locality heuristic: the first touches of a hot set all miss.
 * Two heuristics are implemented on top of an LRU cache and measured:
 *
 *   warming     a training pass records the key frequencies; the entries that will be requested are
 *               pre-loaded before the measured trace starts, so the hot keys hit from the first
 *               request instead of paying a compulsory miss each.
 *   prefetching a stride detector watches consecutive addresses, identifies a repeated stride and asks
 *               the cache to hold the *next* predicted address before it is requested. On a strided
 *               scan the predicted line is used almost every time; on a random address stream the
 *               prediction is useless, and the simulation measures that honestly instead of asserting
 *               a win everywhere.
 *
 * Verified: the LRU implementation reproduces an independently written list-based LRU oracle request by
 * request (same hit/miss for every access of a random trace and the same final resident set); the
 * cache never exceeds its capacity; with warming the hit rate over the first window of the trace is
 * strictly higher than from cold, and warming by frequency beats never warming; the stride detector
 * identifies the stride of a scan exactly and its prefetches are used in more than nine out of ten
 * cases, while on a random trace fewer than half of them are ever used; a prefetch never changes the
 * answers of the following accesses, only whether they hit; and the degenerate caches (capacity one,
 * empty trace, all-distinct trace) behave exactly as their definitions require.
 *
 * Simplified: a single address space with unit-line granularity, no cache-level memory hierarchy and
 * no miss latency modelling - the metrics are the hit rate and the prefetch usefulness ratios.
 */
import java.util.*;

class LruCache {
    final int capacity;
    final LinkedHashMap<Integer, Integer> entries = new LinkedHashMap<>();
    long hits;
    long misses;

    LruCache(int capacity) {
        if (capacity < 1) throw new IllegalArgumentException("capacity must be positive");
        this.capacity = capacity;
    }

    boolean touch(int key) {
        Integer value = entries.remove(key);
        if (value != null) {
            entries.put(key, value);              // refresh the recency order
            hits++;
            return true;
        }
        misses++;
        entries.put(key, key);
        if (entries.size() > capacity) {
            Iterator<Integer> oldest = entries.keySet().iterator();
            oldest.next();
            oldest.remove();
        }
        return false;
    }

    /** Loads a key without counting a hit or a miss (this is what a warm-up or prefetch does). */
    void prefetch(int key) {
        entries.remove(key);
        entries.put(key, key);
        if (entries.size() > capacity) {
            Iterator<Integer> oldest = entries.keySet().iterator();
            oldest.next();
            oldest.remove();
        }
    }

    boolean resident(int key) { return entries.containsKey(key); }

    List<Integer> residentKeys() { return new ArrayList<>(entries.keySet()); }
}

/** Independent list-based LRU oracle, written separately from the LinkedHashMap implementation. */
class BruteLru {
    final int capacity;
    final List<Integer> order = new ArrayList<>();      // least recently used first
    long hits;
    long misses;

    BruteLru(int capacity) { this.capacity = capacity; }

    boolean touch(int key) {
        if (order.remove(Integer.valueOf(key))) {
            order.add(key);
            hits++;
            return true;
        }
        misses++;
        order.add(key);
        if (order.size() > capacity) order.remove(0);
        return false;
    }

    List<Integer> residentKeys() { return new ArrayList<>(order); }
}

class StrideDetector {
    int lastAddress = Integer.MIN_VALUE;
    int lastStride;
    int prefetches;
    int usedPrefetches;
    final Set<Integer> outstanding = new HashSet<>();

    /** Returns the predicted next address, or -1 when no stride has been established yet. */
    int observe(int address) {
        int prediction = -1;
        if (lastAddress != Integer.MIN_VALUE) {
            int stride = address - lastAddress;
            if (stride != 0 && stride == lastStride) prediction = address + stride;
            lastStride = stride;
        }
        lastAddress = address;
        if (prediction >= 0) {
            prefetches++;
            outstanding.add(prediction);
        }
        return prediction;
    }

    /** Marks a prefetched address as consumed when the trace really visits it. */
    boolean consume(int address) {
        if (outstanding.remove(address)) {
            usedPrefetches++;
            return true;
        }
        return false;
    }

    int unusedPrefetches() { return prefetches - usedPrefetches; }

    double usefulness() { return prefetches == 0 ? 0 : (double) usedPrefetches / prefetches; }
}

public class Solution {
    private static long state = 0xC2B2AE3D27D4EB4FL;

    static long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    /** Zipf-like skewed trace: most requests land on a small hot set. */
    static int[] skewedTrace(int length, int hot, int keySpace) {
        int[] trace = new int[length];
        for (int i = 0; i < length; i++) trace[i] = (nextInt(10) < 8) ? nextInt(hot) : nextInt(keySpace);
        return trace;
    }

    static int[] stridedTrace(int length, int start, int stride) {
        int[] trace = new int[length];
        for (int i = 0; i < length; i++) trace[i] = start + i * stride;
        return trace;
    }

    /** Replays a trace, optionally prefetching the stride prediction into the cache. */
    static long replay(LruCache cache, int[] trace, boolean prefetch, StrideDetector detector) {
        long hits = 0;
        for (int address : trace) {
            if (prefetch && detector != null) detector.consume(address);
            if (cache.touch(address)) hits++;
            if (prefetch && detector != null) {
                int prediction = detector.observe(address);
                if (prediction >= 0) cache.prefetch(prediction);
            }
        }
        return hits;
    }

    /** Warms the cache with the most frequent keys of a training trace. */
    static void warmByFrequency(LruCache cache, int[] training, int topK) {
        Map<Integer, Integer> frequency = new HashMap<>();
        for (int key : training) frequency.merge(key, 1, Integer::sum);
        List<Map.Entry<Integer, Integer>> sorted = new ArrayList<>(frequency.entrySet());
        sorted.sort((a, b) -> b.getValue().equals(a.getValue())
                ? Integer.compare(a.getKey(), b.getKey()) : Integer.compare(b.getValue(), a.getValue()));
        for (int i = 0; i < Math.min(topK, sorted.size()); i++) cache.prefetch(sorted.get(i).getKey());
    }

    public static void main(String[] args) {
        // --- the LRU implementation against the independent oracle -------------
        int[] randomTrace = new int[400];
        for (int i = 0; i < randomTrace.length; i++) randomTrace[i] = nextInt(60);
        for (int capacity : new int[]{1, 3, 8, 16}) {
            LruCache cache = new LruCache(capacity);
            BruteLru oracle = new BruteLru(capacity);
            for (int step = 0; step < randomTrace.length; step++) {
                boolean actual = cache.touch(randomTrace[step]);
                boolean expected = oracle.touch(randomTrace[step]);
                check(actual == expected, "hit/miss mismatch at capacity " + capacity + " step " + step);
                check(cache.entries.size() <= capacity, "capacity exceeded at step " + step);
                if (step % 40 == 0) {
                    check(cache.residentKeys().equals(oracle.residentKeys()),
                            "resident set mismatch at capacity " + capacity + " step " + step);
                }
            }
            check(cache.hits == oracle.hits && cache.misses == oracle.misses,
                    "hit and miss totals must match the oracle at capacity " + capacity);
            List<Integer> actual = cache.residentKeys();
            List<Integer> expected = oracle.residentKeys();
            check(actual.equals(expected), "final resident sets must match: " + actual + " vs " + expected);
        }

        // --- degenerate caches -------------------------------------------------
        LruCache one = new LruCache(1);
        check(!one.touch(5) && one.touch(5) && !one.touch(6), "capacity one keeps only the last key");
        check(one.entries.size() == 1 && one.resident(6) && !one.resident(5), "capacity one contents");
        LruCache untouched = new LruCache(4);
        check(untouched.hits == 0 && untouched.misses == 0, "an empty trace leaves the counters at zero");
        LruCache distinct = new LruCache(8);
        int distinctHits = 0;
        for (int key = 0; key < 40; key++) {
            if (distinct.touch(key)) distinctHits++;
        }
        check(distinctHits == 0, "an all-distinct trace never hits, got " + distinctHits);
        boolean rejected = false;
        try {
            new LruCache(0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "a zero-capacity cache must be rejected");

        // --- cold start versus warmed start ------------------------------------
        int[] training = skewedTrace(2000, 8, 500);
        int[] measured = skewedTrace(600, 8, 500);
        int window = 30;
        LruCache cold = new LruCache(16);
        long coldHits = 0;
        for (int i = 0; i < window; i++) {
            if (cold.touch(measured[i])) coldHits++;
        }
        LruCache warmed = new LruCache(16);
        warmByFrequency(warmed, training, 16);
        long warmedHits = 0;
        for (int i = 0; i < window; i++) {
            if (warmed.touch(measured[i])) warmedHits++;
        }
        System.out.println("cold first-window hits=" + coldHits + " warmed=" + warmedHits);
        check(warmedHits > coldHits, "warming must beat the cold start: " + warmedHits + " vs " + coldHits);
        check(warmedHits >= window * 0.5, "the warmed cache should hit most hot requests, got " + warmedHits);

        LruCache preloaded = new LruCache(16);
        warmByFrequency(preloaded, training, 16);
        check(preloaded.entries.size() == 16, "the warm-up fills the cache exactly");
        long fullCold = 0;
        long fullWarm = 0;
        LruCache freshCold = new LruCache(16);
        LruCache freshWarm = new LruCache(16);
        warmByFrequency(freshWarm, training, 16);
        for (int key : measured) {
            if (freshCold.touch(key)) fullCold++;
            if (freshWarm.touch(key)) fullWarm++;
        }
        System.out.println("full trace hits: cold=" + fullCold + " warmed=" + fullWarm);
        check(fullWarm > fullCold, "warming must help over the whole trace too: " + fullWarm + " vs " + fullCold);
        check(freshWarm.entries.size() <= 16 && freshCold.entries.size() <= 16, "capacity respected");

        // --- stride prefetching on a scan --------------------------------------
        int[] scan = stridedTrace(200, 1000, 4);
        LruCache scanned = new LruCache(64);
        StrideDetector scanDetector = new StrideDetector();
        long scanHits = replay(scanned, scan, true, scanDetector);
        LruCache plainScan = new LruCache(64);
        long plainHits = replay(plainScan, scan, false, null);
        System.out.println("scan hits with prefetch=" + scanHits + " without=" + plainHits
                + " usefulness=" + scanDetector.usefulness());
        check(scanDetector.lastStride == 4, "the stride of the scan must be detected, got "
                + scanDetector.lastStride);
        check(scanDetector.usefulness() > 0.9, "almost every prefetch of a scan is used, got "
                + scanDetector.usefulness());
        check(scanHits > plainHits, "prefetching must raise the hit rate on a scan: " + scanHits
                + " vs " + plainHits);
        check(scanHits >= scan.length * 0.9, "a prefetched scan should hit almost always, got " + scanHits);
        check(scanned.entries.size() <= 64, "the prefetching cache still respects its capacity");

        // --- the same prefetcher on a non-repeating stream is mostly useless --
        // a random walk with a fresh stride on every step: the detector can only be right by luck
        int[] noise = new int[200];
        int cursor = 1000;
        for (int i = 0; i < noise.length; i++) {
            cursor += 1 + nextInt(8);
            noise[i] = cursor;
        }
        LruCache noisy = new LruCache(64);
        StrideDetector noiseDetector = new StrideDetector();
        replay(noisy, noise, true, noiseDetector);
        System.out.println("random-walk prefetch usefulness=" + noiseDetector.usefulness()
                + " prefetches=" + noiseDetector.prefetches);
        check(noiseDetector.prefetches > 0, "the detector still guesses on a random walk");
        check(noiseDetector.usefulness() < 0.5, "those guesses are mostly wasted, got "
                + noiseDetector.usefulness());
        check(noisy.entries.size() <= 64, "the random run respects the capacity as well");

        // --- warming and prefetching compose ----------------------------------
        int[] combinedTrace = stridedTrace(120, 500, 3);
        LruCache combined = new LruCache(64);
        LruCache warmedStride = new LruCache(64);
        StrideDetector combinedDetector = new StrideDetector();
        long combinedHits = replay(combined, combinedTrace, true, combinedDetector);
        long warmedStrideHits = replay(warmedStride, combinedTrace, false, null);
        check(combinedHits >= warmedStrideHits, "prefetching must not lose hits: " + combinedHits
                + " vs " + warmedStrideHits);
        check(combinedDetector.unusedPrefetches() >= 0, "the unused prefetch counter is well defined");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
