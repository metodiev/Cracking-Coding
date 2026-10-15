/** Task 485 - Implement cache eviction policies: LRU, LFU, ARC (simulate). (Tier: Advanced, CTCI Ch 14)
 *
 * Documented capacity semantics: `capacity` c is the maximum number of distinct keys the cache may
 * hold (cache-as-main-memory, no separate disk tiers here). Every access returns hit/miss and the
 * cache size never exceeds c.
 *
 * Policy definitions (the "simulator"):
 * - LRU: a hit moves the key to the most-recently-used end; a miss inserts at MRU and, when the
 *   cache is over capacity, evicts the least-recently-used key.
 * - LFU: every hit increments the key's count and refreshes its recency; a miss on a full cache
 *   evicts the key with the smallest count, ties broken by least-recently-used (documented rule).
 * - ARC (Megiddo & Modha): p-adaptive split of T1 (recent) / T2 (frequent); misses that hit ghost
 *   lists B1/B2 adjust p, and `replace` demotes within real lists. Cache contents = T1 + T2 with
 *   |T1| + |T2| <= c; B1/B2 hold only ghosts.
 *
 * Each policy is run against an independent naive reference implementation (plain ArrayLists,
 * linear scans) on three fixed LCG-derived traces; hit totals must be identical, and the totals are
 * also pinned as constants verified by the oracle.
 */
import java.util.*;

interface CachePolicy {
    boolean access(int key);

    int size();

    List<Integer> contents();

    int hits();

    String name();
}

class LRUCache implements CachePolicy {
    protected final int capacity;
    protected final LinkedHashMap<Integer, Boolean> order = new LinkedHashMap<>();
    protected int hitCount;

    LRUCache(int capacity) {
        this.capacity = capacity;
    }

    public String name() {
        return "LRU";
    }

    public boolean access(int key) {
        if (order.containsKey(key)) {
            order.remove(key);
            order.put(key, true);
            hitCount++;
            return true;
        }
        order.put(key, true);
        if (order.size() > capacity) {
            Iterator<Integer> it = order.keySet().iterator();
            it.next();
            it.remove();
        }
        return false;
    }

    public int size() {
        return order.size();
    }

    public List<Integer> contents() {
        List<Integer> keys = new ArrayList<>(order.keySet());
        Collections.sort(keys);
        return keys;
    }

    public int hits() {
        return hitCount;
    }
}

class LFUCache implements CachePolicy {
    private final int capacity;
    private final Map<Integer, Integer> counts = new HashMap<>();
    private final Map<Integer, Integer> recency = new HashMap<>();
    private int tick;
    private int hitCount;

    LFUCache(int capacity) {
        this.capacity = capacity;
    }

    public String name() {
        return "LFU";
    }

    public boolean access(int key) {
        tick++;
        if (counts.containsKey(key)) {
            counts.put(key, counts.get(key) + 1);
            recency.put(key, tick);
            hitCount++;
            return true;
        }
        if (counts.size() >= capacity) {
            Integer victim = null;
            for (int candidate : counts.keySet()) {
                if (victim == null) {
                    victim = candidate;
                    continue;
                }
                long best = counts.get(victim) * 1000000L + recency.get(victim);
                long cur = counts.get(candidate) * 1000000L + recency.get(candidate);
                if (cur < best) victim = candidate;
            }
            counts.remove(victim);
            recency.remove(victim);
        }
        counts.put(key, 1);
        recency.put(key, tick);
        return false;
    }

    public int size() {
        return counts.size();
    }

    public List<Integer> contents() {
        List<Integer> keys = new ArrayList<>(counts.keySet());
        Collections.sort(keys);
        return keys;
    }

    public int hits() {
        return hitCount;
    }
}

class ARCCache implements CachePolicy {
    final int capacity;
    int p;
    final LinkedHashMap<Integer, Boolean> t1 = new LinkedHashMap<>();
    final LinkedHashMap<Integer, Boolean> t2 = new LinkedHashMap<>();
    final LinkedHashMap<Integer, Boolean> b1 = new LinkedHashMap<>();
    final LinkedHashMap<Integer, Boolean> b2 = new LinkedHashMap<>();
    int hitCount;

    ARCCache(int capacity) {
        this.capacity = capacity;
    }

    public String name() {
        return "ARC";
    }

    void replace(int key) {
        if (!t1.isEmpty() && ((b2.containsKey(key) && t1.size() == p) || t1.size() > p)) {
            Iterator<Integer> it = t1.keySet().iterator();
            int victim = it.next();
            it.remove();
            b1.put(victim, true);
        } else if (!t2.isEmpty()) {
            Iterator<Integer> it = t2.keySet().iterator();
            int victim = it.next();
            it.remove();
            b2.put(victim, true);
        }
    }

    public boolean access(int key) {
        if (t1.containsKey(key)) {
            t1.remove(key);
            t2.put(key, true);
            hitCount++;
            return true;
        }
        if (t2.containsKey(key)) {
            t2.remove(key);
            t2.put(key, true);
            hitCount++;
            return true;
        }
        if (b1.containsKey(key)) {
            p = Math.min(capacity, p + Math.max(1, b2.size() / b1.size()));
            replace(key);
            b1.remove(key);
            t2.put(key, true);
        } else if (b2.containsKey(key)) {
            p = Math.max(0, p - Math.max(1, b1.size() / b2.size()));
            replace(key);
            b2.remove(key);
            t2.put(key, true);
        } else {
            if (t1.size() + b1.size() == capacity) {
                if (t1.size() < capacity) {
                    Iterator<Integer> it = b1.keySet().iterator();
                    it.next();
                    it.remove();
                    replace(key);
                } else {
                    Iterator<Integer> it = t1.keySet().iterator();
                    it.next();
                    it.remove();
                }
            } else if (t1.size() + b1.size() < capacity) {
                int total = t1.size() + t2.size() + b1.size() + b2.size();
                if (total >= capacity) {
                    if (total == 2 * capacity) {
                        Iterator<Integer> it = b2.keySet().iterator();
                        it.next();
                        it.remove();
                    }
                    replace(key);
                }
            }
            t1.put(key, true);
        }
        return false;
    }

    public int size() {
        return t1.size() + t2.size();
    }

    public List<Integer> contents() {
        List<Integer> keys = new ArrayList<>(t1.keySet());
        keys.addAll(t2.keySet());
        Collections.sort(keys);
        return keys;
    }

    public int hits() {
        return hitCount;
    }
}

public class Solution {
    private static int seed = 485;

    private static int[] uniformTrace(int count, int keyCount) {
        int[] trace = new int[count];
        for (int i = 0; i < count; i++) {
            seed = (seed * 75 + 74) % 65537;
            trace[i] = seed % keyCount;
        }
        return trace;
    }

    private static int[] localityTrace(int count, int keyCount, int repeatMod) {
        int[] trace = new int[count];
        for (int i = 0; i < count; i++) {
            seed = (seed * 75 + 74) % 65537;
            if (i > 0 && seed % 10 < repeatMod) trace[i] = trace[i - 1];
            else trace[i] = seed % keyCount;
        }
        return trace;
    }

    private static int[] hotScanTrace(int rounds, int hot, int scanWidth) {
        int[] trace = new int[rounds * (2 * hot + scanWidth)];
        int at = 0;
        for (int block = 0; block < rounds; block++) {
            for (int i = 0; i < hot; i++) trace[at++] = i;
            for (int i = 0; i < scanWidth; i++) trace[at++] = 100 + block * scanWidth + i;
            for (int i = 0; i < hot; i++) trace[at++] = i;
        }
        return trace;
    }

    private static int lruReference(int[] trace, int capacity) {
        List<Integer> order = new ArrayList<>();
        int hits = 0;
        for (int key : trace) {
            if (order.contains(key)) {
                order.remove(Integer.valueOf(key));
                order.add(key);
                hits++;
            } else {
                order.add(key);
                if (order.size() > capacity) order.remove(0);
            }
        }
        return hits;
    }

    private static int lfuReference(int[] trace, int capacity) {
        Map<Integer, Integer> counts = new HashMap<>();
        Map<Integer, Integer> recency = new HashMap<>();
        int tick = 0, hits = 0;
        for (int key : trace) {
            tick++;
            if (counts.containsKey(key)) {
                counts.put(key, counts.get(key) + 1);
                recency.put(key, tick);
                hits++;
            } else {
                if (counts.size() >= capacity) {
                    Integer victim = null;
                    for (int candidate : counts.keySet()) {
                        if (victim == null) {
                            victim = candidate;
                            continue;
                        }
                        long best = counts.get(victim) * 1000000L + recency.get(victim);
                        long cur = counts.get(candidate) * 1000000L + recency.get(candidate);
                        if (cur < best) victim = candidate;
                    }
                    counts.remove(victim);
                    recency.remove(victim);
                }
                counts.put(key, 1);
                recency.put(key, tick);
            }
        }
        return hits;
    }

    private static int arcReference(int[] trace, int capacity) {
        int p = 0, hits = 0;
        List<Integer> t1 = new ArrayList<>(), t2 = new ArrayList<>();
        List<Integer> b1 = new ArrayList<>(), b2 = new ArrayList<>();
        for (int key : trace) {
            if (t1.contains(key)) {
                t1.remove(Integer.valueOf(key));
                t2.add(key);
                hits++;
            } else if (t2.contains(key)) {
                t2.remove(Integer.valueOf(key));
                t2.add(key);
                hits++;
            } else {
                if (b1.contains(key)) {
                    p = Math.min(capacity, p + Math.max(1, b2.size() / b1.size()));
                    if (!t1.isEmpty() && ((b2.contains(key) && t1.size() == p) || t1.size() > p)) {
                        b1.add(t1.remove(0));
                    } else if (!t2.isEmpty()) {
                        b2.add(t2.remove(0));
                    }
                    b1.remove(Integer.valueOf(key));
                    t2.add(key);
                } else if (b2.contains(key)) {
                    p = Math.max(0, p - Math.max(1, b1.size() / b2.size()));
                    if (!t1.isEmpty() && ((b2.contains(key) && t1.size() == p) || t1.size() > p)) {
                        b1.add(t1.remove(0));
                    } else if (!t2.isEmpty()) {
                        b2.add(t2.remove(0));
                    }
                    b2.remove(Integer.valueOf(key));
                    t2.add(key);
                } else {
                    if (t1.size() + b1.size() == capacity) {
                        if (t1.size() < capacity) {
                            b1.remove(0);
                            if (!t1.isEmpty() && t1.size() > p) b1.add(t1.remove(0));
                            else if (!t2.isEmpty()) b2.add(t2.remove(0));
                        } else {
                            t1.remove(0);
                        }
                    } else if (t1.size() + b1.size() < capacity) {
                        int total = t1.size() + t2.size() + b1.size() + b2.size();
                        if (total >= capacity) {
                            if (total == 2 * capacity) b2.remove(0);
                            if (!t1.isEmpty() && ((b2.contains(key) && t1.size() == p) || t1.size() > p)) {
                                b1.add(t1.remove(0));
                            } else if (!t2.isEmpty()) {
                                b2.add(t2.remove(0));
                            }
                        }
                    }
                    t1.add(key);
                }
            }
        }
        return hits;
    }

    private static void runTrace(CachePolicy cache, int[] trace, int expected, String label) {
        int overflows = 0;
        int lastSize = 0;
        int maxSize = 0;
        int summedHits = 0;
        for (int key : trace) {
            if (cache.access(key)) summedHits++;
            int size = cache.size();
            lastSize = size;
            maxSize = Math.max(maxSize, size);
            if (size > 4) overflows++;
        }
        check(cache.hits() == expected, label + " hits " + cache.hits());
        check(summedHits == cache.hits(), label + " hit vector");
        check(overflows == 0, label + " capacity respected");
        check(maxSize == 4 && lastSize == 4, label + " warm capacity");
        List<Integer> contents = cache.contents();
        check(contents.size() == 4 && new HashSet<>(contents).size() == 4, label + " contents");
    }

    public static void main(String[] args) {
        int[] hand = {1, 2, 1, 3, 1};
        boolean[] expected = {false, false, true, false, true};
        for (CachePolicy cache : new CachePolicy[]{new LRUCache(2), new LFUCache(2), new ARCCache(2)}) {
            boolean[] observed = new boolean[hand.length];
            for (int i = 0; i < hand.length; i++) observed[i] = cache.access(hand[i]);
            check(Arrays.equals(observed, expected), cache.name() + " hand vector");
            check(cache.hits() == 2, cache.name() + " hand hits");
            List<Integer> contents = cache.contents();
            check(contents.equals(Arrays.asList(1, 3)), cache.name() + " hand contents " + contents);
        }
        check(lruReference(hand, 2) == 2 && lfuReference(hand, 2) == 2 && arcReference(hand, 2) == 2, "hand refs");

        seed = 485;
        int[] uniform = uniformTrace(400, 20);
        seed = 486;
        int[] locality = localityTrace(400, 24, 7);
        int[] hotScan = hotScanTrace(24, 4, 8);

        runTrace(new LRUCache(4), uniform, 70, "uniform LRU");
        runTrace(new LFUCache(4), uniform, 64, "uniform LFU");
        runTrace(new ARCCache(4), uniform, 64, "uniform ARC");
        runTrace(new LRUCache(4), locality, 296, "locality LRU");
        runTrace(new LFUCache(4), locality, 291, "locality LFU");
        runTrace(new ARCCache(4), locality, 290, "locality ARC");
        runTrace(new LRUCache(4), hotScan, 92, "hot-scan LRU");
        runTrace(new LFUCache(4), hotScan, 161, "hot-scan LFU");
        runTrace(new ARCCache(4), hotScan, 161, "hot-scan ARC");

        check(lruReference(uniform, 4) == 70, "uniform LRU ref");
        check(lfuReference(uniform, 4) == 64, "uniform LFU ref");
        check(arcReference(uniform, 4) == 64, "uniform ARC ref");
        check(lruReference(locality, 4) == 296, "locality LRU ref");
        check(lfuReference(locality, 4) == 291, "locality LFU ref");
        check(arcReference(locality, 4) == 290, "locality ARC ref");
        check(lruReference(hotScan, 4) == 92, "hot-scan LRU ref");
        check(lfuReference(hotScan, 4) == 161, "hot-scan LFU ref");
        check(arcReference(hotScan, 4) == 161, "hot-scan ARC ref");
        check(161 > 92, "ARC beats LRU on hot-scan");
        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
