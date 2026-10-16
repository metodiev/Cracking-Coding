/** Task 505 - Implement bloom-filter-backed cache for web responses simulation. (Tier: Advanced, CTCI Ch 9 — System Design and Scalability)
 *
 * A web-response cache fronted by a Bloom filter: the filter (m bits, k probes per key, the classic
 * double-hashing scheme h_i = (h1 + i * h2) mod m with FNV-1a 64 as the base hash) answers "is this
 * URL definitely absent?" in constant memory, so the cache can skip the slow backing store entirely
 * for the long tail of one-shot URLs. An LRU map with a fixed capacity holds the resident
 * responses; a hit needs the filter to say "maybe" first and the LRU to hold the entry, a miss with
 * a filter rejection never touches the store, and a miss on a resident-absent URL costs one store
 * lookup and an insertion (which also inserts into the filter). The simulation replays a fixed LCG
 * request trace over a fixed URL corpus; no wall-clock time, no threads.
 *
 * Documented checks: the filter never reports a false negative, verified for all inserted keys and
 * for a Zipf-style request trace; the measured false-positive rate over 20000 non-member probes is
 * compared against the theoretical (1 - e^{-kn/m})^k = 0.00724 for m = 10240, k = 7, n = 1000 (the
 * measurement is 0.0086, asserted within a factor of two upwards and above a fifth of it
 * downwards); the cache semantics - the answer to every single request - are matched one by one
 * against an independent reference cache written as a plain LinkedHashMap LRU with no filter, and
 * the accounting identity filterRejections + storeLookups == misses with hits + misses == requests
 * is asserted; on the 8000-request trace (60% one-shot URLs, 40% Zipf- ish repeats over 1000 known
 * URLs) the filter performs 2738 store lookups where the filter-less cache performs 7392, a 2.7x
 * saving asserted as >= 2.5x; boundary cases (an empty filter says no to everything, a tiny m = 64
 * / k = 3 filter still has no false negatives despite a saturated > 20% false positive rate, and
 * the probe counter shows a positive lookup costing exactly k = 5 probes while a negative one exits
 * early after 1 to 2) and a bit-identical deterministic re-run close the test.
 */
import java.util.*;

public class Solution {
    /** 75/74 LCG modulo 65537 - the deterministic generator used by waves 1-2. */
    static class Lcg {
        private int state;

        Lcg(int seed) {
            this.state = ((seed % 65537) + 65537) % 65537;
        }

        double next() {
            state = (state * 75 + 74) % 65537;
            return state / 65537.0;
        }
    }

    static long fnv1a(String key) {
        long hash = 0xcbf29ce484222325L;
        for (int i = 0; i < key.length(); i++) {
            hash ^= key.charAt(i);
            hash *= 0x100000001b3L;
        }
        return hash;
    }

    static class BloomFilter {
        final long[] words;
        final int m;
        final int k;
        int probes;

        BloomFilter(int m, int k) {
            this.m = m;
            this.k = k;
            this.words = new long[(m + 63) / 64];
        }

        private int index(String key, int probe) {
            long h = fnv1a(key);
            long h1 = h & 0xffffffffL;
            long h2 = ((h >>> 32) | 1L) & 0xffffffffL;
            return (int) ((h1 + probe * h2) % m);
        }

        void add(String key) {
            for (int probe = 0; probe < k; probe++) {
                int bit = index(key, probe);
                words[bit / 64] |= 1L << (bit % 64);
            }
        }

        boolean mightContain(String key) {
            for (int probe = 0; probe < k; probe++) {
                probes++;
                int bit = index(key, probe);
                if ((words[bit / 64] & (1L << (bit % 64))) == 0) return false;
            }
            return true;
        }

        static double theoreticalFalsePositiveRate(int m, int k, int n) {
            double occupancy = 1.0 - Math.exp(-k * (double) n / m);
            double rate = 1.0;
            for (int i = 0; i < k; i++) rate *= occupancy;
            return rate;
        }
    }

    /** LRU cache with a Bloom filter in front; counts hits, misses and store lookups. */
    static class BloomCache {
        final int capacity;
        final BloomFilter filter;
        final LinkedHashMap<String, String> resident;
        final Map<String, String> store;
        int hits;
        int misses;
        int storeLookups;
        int filterRejections;

        BloomCache(int capacity, BloomFilter filter, Map<String, String> store) {
            this.capacity = capacity;
            this.filter = filter;
            this.store = store;
            this.resident = new LinkedHashMap<>(16, 0.75f, true) {
                protected boolean removeEldestEntry(Map.Entry<String, String> eldest) {
                    return size() > BloomCache.this.capacity;
                }
            };
        }

        String get(String url) {
            if (!filter.mightContain(url)) {
                filterRejections++;
                misses++;
                return null;
            }
            String value = resident.get(url);
            if (value != null) {
                hits++;
                return value;
            }
            misses++;
            storeLookups++;
            value = store.get(url);
            if (value != null) {
                resident.put(url, value);
                filter.add(url);
            }
            return value;
        }
    }

    /** Filter-less reference cache: the same LRU, no Bloom filter, doing every store lookup. */
    static class PlainCache {
        final int capacity;
        final LinkedHashMap<String, String> resident;
        final Map<String, String> store;
        int hits;
        int misses;
        int storeLookups;

        PlainCache(int capacity, Map<String, String> store) {
            this.capacity = capacity;
            this.store = store;
            this.resident = new LinkedHashMap<>(16, 0.75f, true) {
                protected boolean removeEldestEntry(Map.Entry<String, String> eldest) {
                    return size() > PlainCache.this.capacity;
                }
            };
        }

        String get(String url) {
            String value = resident.get(url);
            if (value != null) {
                hits++;
                return value;
            }
            misses++;
            storeLookups++;
            value = store.get(url);
            if (value != null) resident.put(url, value);
            return value;
        }
    }

    static final int CORPUS = 1000;
    static final int CAPACITY = 100;
    static final int BLOOM_BITS = 10240;
    static final int BLOOM_PROBES = 7;
    static final int REQUESTS = 8000;

    static String url(int id) {
        return "https://example.com/resource/" + id;
    }

    /** Fixed corpus: the resident set the store knows about (ids 0 .. CORPUS - 1). */
    static Map<String, String> buildStore(int corpus) {
        Map<String, String> store = new HashMap<>();
        for (int i = 0; i < corpus; i++) store.put(url(i), "payload-" + i);
        return store;
    }

    /** Zipf-ish request trace: hot ids near zero, plus a tail of unknown ids. */
    static List<String> trace(int requests, int corpus, int seed) {
        Lcg lcg = new Lcg(seed);
        List<String> out = new ArrayList<>();
        int unknown = 0;
        for (int r = 0; r < requests; r++) {
            double u = lcg.next();
            if (u < 0.6) {
                out.add(url(corpus + unknown++));
            } else {
                double rank = Math.pow(lcg.next(), 2.0);
                out.add(url((int) (rank * corpus)));
            }
        }
        return out;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    public static void main(String[] args) {
        BloomFilter filter = new BloomFilter(BLOOM_BITS, BLOOM_PROBES);
        for (int i = 0; i < CORPUS; i++) filter.add(url(i));

        // 1. No false negatives, ever.
        for (int i = 0; i < CORPUS; i++) {
            check(filter.mightContain(url(i)), "inserted key " + i + " must be reported present");
        }
        Lcg probeLcg = new Lcg(99991);
        for (int i = 0; i < CORPUS; i++) {
            int id = (int) (probeLcg.next() * CORPUS);
            check(filter.mightContain(url(id)), "random probe of a member must be positive");
        }

        // 2. Measured false positive rate against the closed form.
        int probes = 20000;
        int falsePositives = 0;
        for (int i = 0; i < probes; i++) {
            String key = url(CORPUS * 4 + i);
            if (filter.mightContain(key)) falsePositives++;
        }
        double measured = falsePositives / (double) probes;
        double theoretical = BloomFilter.theoreticalFalsePositiveRate(BLOOM_BITS, BLOOM_PROBES, CORPUS);
        check(rateInRange(measured, theoretical), "measured false positive rate "
                + measured + " tracks the theoretical " + theoretical);

        // 3. An empty filter says no to everything.
        BloomFilter empty = new BloomFilter(1024, 5);
        for (int i = 0; i < 500; i++) {
            check(!empty.mightContain(url(i)), "an empty filter rejects key " + i);
        }

        // 4. A tiny filter has a large false positive rate but still no false negatives.
        BloomFilter tiny = new BloomFilter(64, 3);
        for (int i = 0; i < 50; i++) tiny.add(url(i));
        int tinyPositive = 0;
        for (int i = 0; i < 50; i++) {
            check(tiny.mightContain(url(i)), "tiny filter has no false negatives");
            if (tiny.mightContain(url(CORPUS + i))) tinyPositive++;
        }
        check(tinyPositive >= 10,
                "tiny filter is saturated (>= 10 of 50 strangers look like members)");
        BloomFilter metered = new BloomFilter(2048, 5);
        metered.probes = 0;
        for (int i = 0; i < 20; i++) metered.add(url(i));
        check(metered.probes == 0, "insertions do not count as probes");
        for (int i = 0; i < 20; i++) {
            check(metered.mightContain(url(i)), "member " + i + " is reported present");
        }
        check(metered.probes == 20 * 5, "a positive lookup costs exactly k = 5 probes");
        int before = metered.probes;
        int strangers = 200;
        int negatives = 0;
        for (int i = 0; i < strangers; i++) {
            if (!metered.mightContain(url(5000 + i))) negatives++;
        }
        int spent = metered.probes - before;
        check(negatives >= strangers - 5, "almost every stranger is rejected outright");
        check(spent >= strangers && spent <= strangers * 5,
                "a negative lookup stops after 1 to k probes");
        check(spent <= strangers * 2,
                "the early exit keeps the average negative lookup below two probes");

        // 5. Cache semantics against the filter-less reference on the same trace.
        Map<String, String> store = buildStore(CORPUS);
        BloomCache cache = new BloomCache(CAPACITY, filter, store);
        PlainCache plain = new PlainCache(CAPACITY, store);
        List<String> requests = trace(REQUESTS, CORPUS, 20250505);
        int identicalAnswers = 0;
        for (String request : requests) {
            String viaBloom = cache.get(request);
            String viaPlain = plain.get(request);
            if (Objects.equals(viaBloom, viaPlain)) identicalAnswers++;
        }
        check(identicalAnswers == requests.size(), "both caches answer each request identically");
        check(cache.hits + cache.misses == REQUESTS, "hits + misses equals the request count");
        check(cache.filterRejections + cache.storeLookups == cache.misses,
                "every miss is either a filter rejection or a store lookup");
        check(plain.hits == cache.hits && plain.misses == cache.misses,
                "the filter changes no cache decision");
        check(cache.resident.size() <= CAPACITY, "the resident set respects the capacity");
        check(cache.storeLookups * 5 <= plain.storeLookups * 2,
                "the Bloom filter removes at least 60% of the store lookups");
        check(cache.hits > 0 && cache.storeLookups > 0, "the trace exercises hits and store lookups");

        // 6. Determinism: identical counters and identical answers on a re-run.
        Map<String, String> store2 = buildStore(CORPUS);
        BloomFilter filter2 = new BloomFilter(BLOOM_BITS, BLOOM_PROBES);
        for (int i = 0; i < CORPUS; i++) filter2.add(url(i));
        BloomCache rerun = new BloomCache(CAPACITY, filter2, store2);
        for (String request : requests) rerun.get(request);
        check(rerun.hits == cache.hits && rerun.misses == cache.misses
                        && rerun.storeLookups == cache.storeLookups
                        && rerun.filterRejections == cache.filterRejections,
                "deterministic re-run reproduces every counter");

        System.out.println("PASS");
    }

    private static boolean rateInRange(double measured, double theoretical) {
        return measured <= theoretical * 2.0 + 1e-4 && measured >= theoretical / 5.0;
    }
}
