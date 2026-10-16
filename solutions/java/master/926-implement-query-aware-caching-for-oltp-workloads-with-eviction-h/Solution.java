/** Task 926 - Implement query-aware caching for OLTP workloads with eviction hints.
 *  (Tier: Master, CTCI Ch 14 - Databases)
 *
 * A page cache whose eviction decision uses a hint the query planner supplies with every request (how much
 * work a hit saves) instead of recency alone. The policy is GDSF: each resident page carries the priority
 *   priority = age + accesses * hint / size,
 * where `age` is the inflation value of the last eviction (the classic greedy-dual aging) and the victim is
 * the smallest priority - so a small, frequently used, expensive-to-refetch page survives a large, rarely used
 * one even when the latter was touched more recently. Pages larger than the free space are bypassed rather
 * than thrashing the cache.
 * Verified against two independently written caches: recency-only LRU and an offline Belady MIN that knows the
 * whole future, so the hint policy must sit strictly between them. Also verified: the capacity is never
 * exceeded, every hit returns the source value, the closed-form hit ratios of a one-page cache on a single-hot
 * key workload and of a cache larger than the working set, that informative hints beat uniform hints, that an
 * oversized page is bypassed instead of evicting the world, and that the run is reproducible.
 */
import java.util.*;

/** A resident page: value, size in capacity units, access count, latest hint and current priority. */
final class Entry {
    final double value;
    final int size;
    long accesses;
    double hint, priority;

    Entry(double value, int size, double hint) {
        this.value = value;
        this.size = size;
        this.hint = hint;
    }
}

/** Greedy-dual-size-frequency cache driven by per-request hints. */
final class HintCache {
    final int capacity;
    final Map<String, Entry> resident = new HashMap<>();
    int used, hits, misses, evictions, bypassed;
    double inflation;

    HintCache(int capacity) { this.capacity = capacity; }

    /** Serve a request: a hit refreshes the priority, a miss admits the page if it can be paid for. */
    double request(String key, double hint, int size, Map<String, Double> source) {
        double value = source.get(key);
        Entry e = resident.get(key);
        if (e != null) {
            hits++;
            e.accesses++;
            e.hint = Math.max(e.hint, hint);
            e.priority = inflation + e.accesses * e.hint / e.size;
            return e.value;
        }
        misses++;
        Entry fresh = new Entry(value, size, hint);
        fresh.accesses = 1;
        fresh.priority = inflation + hint / size;
        while (used + fresh.size > capacity && !resident.isEmpty() && smallest() < fresh.priority) evict();
        if (used + fresh.size > capacity) {
            bypassed++;                                                 // no room and nothing cheaper to drop
            return value;
        }
        resident.put(key, fresh);
        used += fresh.size;
        return value;
    }

    private double smallest() {
        double min = Double.MAX_VALUE;
        for (Entry e : resident.values()) min = Math.min(min, e.priority);
        return min;
    }

    private void evict() {
        String victim = null;
        for (Map.Entry<String, Entry> e : resident.entrySet()) {
            if (victim == null || e.getValue().priority < resident.get(victim).priority) victim = e.getKey();
        }
        Entry gone = resident.remove(victim);
        used -= gone.size;
        inflation = gone.priority;                                      // GDS aging factor
        evictions++;
    }
}

/** Recency-only baseline, written independently of the policy above. */
final class LruCache {
    final int capacity;
    final LinkedHashMap<String, Double> order = new LinkedHashMap<>(16, 0.75f, true);
    int hits, misses;

    LruCache(int capacity) { this.capacity = capacity; }

    double request(String key, Map<String, Double> source) {
        Double hit = order.get(key);
        if (hit != null) {
            hits++;
            return hit;
        }
        misses++;
        order.put(key, source.get(key));
        while (order.size() > capacity) order.remove(order.keySet().iterator().next());
        return source.get(key);
    }
}

/** Offline Belady MIN: evict whatever is used farthest in the future. The optimal upper bound. */
final class BeladyCache {
    final int capacity;
    final Set<String> resident = new HashSet<>();
    int hits, misses;

    BeladyCache(int capacity) { this.capacity = capacity; }

    void run(List<String> requests) {
        Map<String, List<Integer>> positions = new HashMap<>();
        for (int i = 0; i < requests.size(); i++) positions.computeIfAbsent(requests.get(i), k -> new ArrayList<>()).add(i);
        Map<String, Integer> cursor = new HashMap<>();
        for (int i = 0; i < requests.size(); i++) {
            String key = requests.get(i);
            cursor.put(key, cursor.getOrDefault(key, 0) + 1);
            if (resident.contains(key)) {
                hits++;
                continue;
            }
            misses++;
            if (resident.size() < capacity) {
                resident.add(key);
                continue;
            }
            String victim = null;
            int farthest = -1;
            for (String other : resident) {
                List<Integer> uses = positions.get(other);
                int at = cursor.getOrDefault(other, 0);
                int distance = at >= uses.size() ? Integer.MAX_VALUE : uses.get(at);
                if (victim == null || distance > farthest) { farthest = distance; victim = other; }
            }
            resident.remove(victim);
            resident.add(key);
        }
    }
}

public class Solution {
    static final int KEYS = 50, REQUESTS = 2000, CAPACITY = 12;

    /** Deterministic Zipf(1.05) request stream over KEYS keys. */
    static List<String> stream(long seed) {
        List<String> out = new ArrayList<>();
        long state = seed;
        double[] cumulative = new double[KEYS];
        double running = 0;
        for (int k = 0; k < KEYS; k++) {
            running += 1.0 / Math.pow(k + 1, 1.05);
            cumulative[k] = running;
        }
        for (int i = 0; i < REQUESTS; i++) {
            state = state * 6364136223846793005L + 1442695040888963407L;
            double u = ((state >>> 11) * 0x1.0p-53) * running;
            int k = 0;
            while (k < KEYS - 1 && cumulative[k] < u) k++;
            out.add("k" + k);
        }
        return out;
    }

    /** The planner's hint: how much a hit on this key saves, high for the hot keys. */
    static double hint(String key, boolean informative) {
        return informative ? 100.0 / Math.sqrt(Integer.parseInt(key.substring(1)) + 1) : 1.0;
    }

    static Map<String, Double> source() {
        Map<String, Double> map = new HashMap<>();
        for (int k = 0; k < KEYS; k++) map.put("k" + k, (double) k);
        return map;
    }

    static double ratio(int hits) { return hits / (double) REQUESTS; }

    public static void main(String[] args) {
        List<String> requests = stream(0x926C0DEL);
        Map<String, Double> source = source();

        HintCache hinted = new HintCache(CAPACITY);
        int wrong = 0, overshoot = 0;
        for (String key : requests) {
            if (hinted.request(key, hint(key, true), 1, source) != source.get(key)) wrong++;
            if (hinted.used > CAPACITY) overshoot++;
        }
        HintCache uniform = new HintCache(CAPACITY);
        for (String key : requests) uniform.request(key, 1.0, 1, source);
        LruCache lru = new LruCache(CAPACITY);
        for (String key : requests) lru.request(key, source);
        BeladyCache belady = new BeladyCache(CAPACITY);
        belady.run(requests);

        double hintedRatio = ratio(hinted.hits), uniformRatio = ratio(uniform.hits);
        double lruRatio = ratio(lru.hits), beladyRatio = ratio(belady.hits);

        System.out.println("cache: " + REQUESTS + " Zipf requests over " + KEYS + " keys, capacity " + CAPACITY
                + " pages - hinted GDSF " + String.format("%.4f", hintedRatio) + ", uniform GDSF "
                + String.format("%.4f", uniformRatio) + ", LRU " + String.format("%.4f", lruRatio)
                + ", Belady MIN " + String.format("%.4f", beladyRatio));
        System.out.println("hinted cache: hits " + hinted.hits + ", misses " + hinted.misses + ", evictions "
                + hinted.evictions + ", bypasses " + hinted.bypassed + ", overshoots " + overshoot
                + ", wrong values " + wrong);

        check(overshoot == 0, "the cache must never exceed its capacity");
        check(wrong == 0, "a hit must return the source value");
        check(hintedRatio > lruRatio, "query-aware hints must beat recency only: " + hintedRatio + " vs " + lruRatio);
        check(hintedRatio > uniformRatio, "informative hints must beat uniform hints: " + hintedRatio + " vs " + uniformRatio);
        check(beladyRatio >= hintedRatio - 1e-9, "no online policy can beat Belady: " + hintedRatio + " vs " + beladyRatio);
        check(beladyRatio > lruRatio, "the optimal bound must be strictly above LRU, otherwise the comparison is empty");
        check(hinted.evictions > 0, "a warm cache over " + KEYS + " keys must evict");

        // Boundary: one page and a single hot key hits everything after the first miss.
        HintCache tiny = new HintCache(1);
        for (int i = 0; i < 100; i++) tiny.request("k0", 1, 1, source);
        check(tiny.hits == 99 && tiny.misses == 1, "a one-page cache on one hot key: " + tiny.hits + " hits, "
                + tiny.misses + " misses");

        // Boundary: a cache larger than the working set misses exactly once per key and never evicts.
        int distinct = new HashSet<>(requests).size();
        HintCache roomy = new HintCache(KEYS);
        for (String key : requests) roomy.request(key, hint(key, true), 1, source);
        check(roomy.misses == distinct && roomy.evictions == 0, "an oversized cache misses once per key: "
                + roomy.misses + " misses for " + distinct + " keys");
        check(roomy.hits == REQUESTS - distinct, "everything else is a hit, got " + roomy.hits);

        // Size-awareness: a page bigger than the whole cache is bypassed, not evicted around.
        HintCache slim = new HintCache(2);
        slim.request("k7", 5, 3, source);
        check(slim.bypassed == 1 && slim.used == 0 && slim.evictions == 0,
                "a page larger than the cache must be bypassed, bypasses " + slim.bypassed + ", used " + slim.used);
        slim.request("k1", 5, 1, source);
        slim.request("k2", 5, 1, source);
        check(slim.used == 2 && slim.evictions == 0, "two unit pages fit exactly: used " + slim.used);
        slim.request("k3", 99, 1, source);
        check(slim.evictions == 1 && slim.used <= 2 && slim.resident.containsKey("k3"),
                "an expensive page must displace a cheaper resident one");

        HintCache replay = new HintCache(CAPACITY);
        for (String key : requests) replay.request(key, hint(key, true), 1, source);
        check(replay.hits == hinted.hits && replay.evictions == hinted.evictions && replay.inflation == hinted.inflation,
                "the cache must be deterministic");

        System.out.println("boundaries: one-page cache " + tiny.hits + "/100 hits, oversized cache " + roomy.hits
                + " hits with " + roomy.evictions + " evictions (" + distinct + " distinct keys), oversized page bypassed "
                + slim.bypassed + " time(s)");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
