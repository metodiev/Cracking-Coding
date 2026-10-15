/** Task 486 - Implement adaptive replacement cache (ARC) outline & test. (Tier: Advanced, CTCI Ch 14)
 *
 * ARC outline (Megiddo & Modha 2003), capacity c:
 * - Real lists: T1 (keys seen once recently), T2 (keys seen at least twice). Cache contents = T1+T2,
 *   with |T1| + |T2| <= c.
 * - Ghost lists: B1 (recently evicted from T1), B2 (evicted from T2); |T1| + |B1| <= c and
 *   |T1|+|T2|+|B1|+|B2| <= 2c.
 * - A request found in T1 is promoted to T2; in T2 it is refreshed (MRU). A request found in a
 *   ghost list adapts the target size p for T1: p grows on a B1 hit, shrinks on a B2 hit, always
 *   clamped to [0, c]; the hit key is admitted to T2. A brand-new request is admitted to T1, first
 *   discarding a ghost or demoting the LRU of T1/T2 via `replace` (T1 when |T1| > p, else T2).
 *
 * Adaptivity test:
 * - Trace S ("scan-resistance", 20 rounds of [hot 0..3, 8-key scan, hot 0..3], scan keys cycling
 *   mod 8): LRU thrashes on the scans; pinned ARC 133 vs LRU 76 hits.
 * - Trace A ("adaptivity", same rounds but 4 rotating scan keys): ghost hits drive p up to c = 4
 *   and back down; pinned p_max = 4, 4 B1-hits, 141 B2-hits, ARC 79 vs LRU 76.
 *
 * Correctness: on every access the requested key is in the cache, the four lists are pairwise
 * disjoint with the documented size bounds honoured, and per-step cache contents/hits match an
 * independent list-based ARC reference implementation.
 */
import java.util.*;

class ARCCache {
    final int c;
    int p;
    final LinkedHashMap<Integer, Boolean> t1 = new LinkedHashMap<>();
    final LinkedHashMap<Integer, Boolean> t2 = new LinkedHashMap<>();
    final LinkedHashMap<Integer, Boolean> b1 = new LinkedHashMap<>();
    final LinkedHashMap<Integer, Boolean> b2 = new LinkedHashMap<>();
    int hits;
    int pMax;
    int pMin;
    int b1Hits;
    int b2Hits;

    ARCCache(int capacity) {
        this.c = capacity;
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

    boolean access(int key) {
        if (t1.containsKey(key)) {
            t1.remove(key);
            t2.put(key, true);
            hits++;
            return true;
        }
        if (t2.containsKey(key)) {
            t2.remove(key);
            t2.put(key, true);
            hits++;
            return true;
        }
        if (b1.containsKey(key)) {
            p = Math.min(c, p + Math.max(1, b2.size() / b1.size()));
            b1Hits++;
            replace(key);
            b1.remove(key);
            t2.put(key, true);
        } else if (b2.containsKey(key)) {
            p = Math.max(0, p - Math.max(1, b1.size() / b2.size()));
            b2Hits++;
            replace(key);
            b2.remove(key);
            t2.put(key, true);
        } else {
            if (t1.size() + b1.size() == c) {
                if (t1.size() < c) {
                    Iterator<Integer> it = b1.keySet().iterator();
                    it.next();
                    it.remove();
                    replace(key);
                } else {
                    Iterator<Integer> it = t1.keySet().iterator();
                    it.next();
                    it.remove();
                }
            } else if (t1.size() + b1.size() < c) {
                int total = t1.size() + t2.size() + b1.size() + b2.size();
                if (total >= c) {
                    if (total == 2 * c) {
                        Iterator<Integer> it = b2.keySet().iterator();
                        it.next();
                        it.remove();
                    }
                    replace(key);
                }
            }
            t1.put(key, true);
        }
        pMax = Math.max(pMax, p);
        pMin = Math.min(pMin, p);
        return false;
    }

    List<Integer> contents() {
        List<Integer> keys = new ArrayList<>(t1.keySet());
        keys.addAll(t2.keySet());
        Collections.sort(keys);
        return keys;
    }

    void checkInvariants(int key, int step) {
        if (!t1.containsKey(key) && !t2.containsKey(key)) throw new AssertionError("FAILED: key absent " + step);
        if (t1.size() + t2.size() > c) throw new AssertionError("FAILED: cache bound " + step);
        if (t1.size() + b1.size() > c) throw new AssertionError("FAILED: t1+b1 bound " + step);
        if (t1.size() + t2.size() + b1.size() + b2.size() > 2 * c) throw new AssertionError("FAILED: total bound " + step);
        if (p < 0 || p > c) throw new AssertionError("FAILED: p range " + step);
        for (int i = 0; i < 4; i++) {
            for (int j = i + 1; j < 4; j++) {
                Set<Integer> a = new HashSet<>(listAt(i));
                Set<Integer> b = new HashSet<>(listAt(j));
                a.retainAll(b);
                if (!a.isEmpty()) throw new AssertionError("FAILED: lists overlap " + step);
            }
        }
    }

    private List<Integer> listAt(int index) {
        if (index == 0) return new ArrayList<>(t1.keySet());
        if (index == 1) return new ArrayList<>(t2.keySet());
        if (index == 2) return new ArrayList<>(b1.keySet());
        return new ArrayList<>(b2.keySet());
    }
}

class ArcReferenceResult {
    final int hits;
    final List<List<Integer>> contents;

    ArcReferenceResult(int hits, List<List<Integer>> contents) {
        this.hits = hits;
        this.contents = contents;
    }
}

public class Solution {
    static ArcReferenceResult arcReference(int[] trace, int capacity) {
        int p = 0, hits = 0;
        List<Integer> t1 = new ArrayList<>(), t2 = new ArrayList<>();
        List<Integer> b1 = new ArrayList<>(), b2 = new ArrayList<>();
        List<List<Integer>> contents = new ArrayList<>();
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
            List<Integer> snapshot = new ArrayList<>(t1);
            snapshot.addAll(t2);
            Collections.sort(snapshot);
            contents.add(snapshot);
        }
        return new ArcReferenceResult(hits, contents);
    }

    static int lruHits(int[] trace, int capacity) {
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

    static int[] scanTrace(int rounds, int hot, int scanWidth, int cold) {
        int[] trace = new int[rounds * (2 * hot + scanWidth)];
        int at = 0;
        for (int r = 0; r < rounds; r++) {
            for (int i = 0; i < hot; i++) trace[at++] = i;
            for (int j = 0; j < scanWidth; j++) trace[at++] = 100 + (r * scanWidth + j) % cold;
            for (int i = 0; i < hot; i++) trace[at++] = i;
        }
        return trace;
    }

    static ARCCache verifyTrace(int[] trace, int capacity, int expectedHits, int expectedLru, String label) {
        ARCCache cache = new ARCCache(capacity);
        List<List<Integer>> contentsVector = new ArrayList<>();
        int summedHits = 0;
        for (int step = 0; step < trace.length; step++) {
            if (cache.access(trace[step])) summedHits++;
            cache.checkInvariants(trace[step], step);
            contentsVector.add(cache.contents());
        }
        ArcReferenceResult reference = arcReference(trace, capacity);
        if (cache.hits != reference.hits || cache.hits != expectedHits) {
            throw new AssertionError("FAILED: " + label + " hits " + cache.hits + " ref " + reference.hits);
        }
        if (summedHits != cache.hits) throw new AssertionError("FAILED: " + label + " hit vector");
        if (!contentsVector.equals(reference.contents)) throw new AssertionError("FAILED: " + label + " contents vector");
        int lru = lruHits(trace, capacity);
        if (lru != expectedLru) throw new AssertionError("FAILED: " + label + " lru " + lru);
        if (cache.hits < lru) throw new AssertionError("FAILED: " + label + " ARC < LRU");
        return cache;
    }

    public static void main(String[] args) {
        ARCCache cache = new ARCCache(2);
        int[] hand = {1, 2, 1, 3, 1};
        boolean[] expected = {false, false, true, false, true};
        boolean[] observed = new boolean[hand.length];
        for (int i = 0; i < hand.length; i++) observed[i] = cache.access(hand[i]);
        if (!Arrays.equals(observed, expected)) throw new AssertionError("FAILED: hand vector");
        if (cache.hits != 2 || !cache.contents().equals(Arrays.asList(1, 3))) throw new AssertionError("FAILED: hand state");

        int[] scan = scanTrace(20, 4, 8, 8);
        if (scan.length != 320 || scan[0] != 0 || scan[1] != 1 || scan[2] != 2 || scan[3] != 3) {
            throw new AssertionError("FAILED: scan trace");
        }
        ARCCache scanCache = verifyTrace(scan, 4, 133, 76, "scan-resistance");
        if (!scanCache.contents().equals(Arrays.asList(0, 1, 2, 3))) throw new AssertionError("FAILED: scan final cache");

        int[] adaptive = scanTrace(20, 4, 4, 4);
        if (adaptive.length != 240) throw new AssertionError("FAILED: adaptive length");
        ARCCache adaptiveCache = verifyTrace(adaptive, 4, 79, 76, "adaptivity");
        if (adaptiveCache.pMax != 4 || adaptiveCache.pMin != 0 || adaptiveCache.p != 0) {
            throw new AssertionError("FAILED: p trajectory " + adaptiveCache.pMax + " " + adaptiveCache.pMin + " " + adaptiveCache.p);
        }
        if (adaptiveCache.b1Hits != 4 || adaptiveCache.b2Hits != 141) {
            throw new AssertionError("FAILED: ghost hits " + adaptiveCache.b1Hits + " " + adaptiveCache.b2Hits);
        }
        if (!adaptiveCache.contents().equals(Arrays.asList(0, 1, 2, 3))) throw new AssertionError("FAILED: adaptive final cache");
        System.out.println("PASS");
    }
}
