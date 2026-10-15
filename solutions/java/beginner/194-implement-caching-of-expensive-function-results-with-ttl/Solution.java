/** Task 194 - Implement caching of expensive function results with TTL. (Tier: Beginner, CTCI Ch 7) */
import java.util.*;
import java.util.function.Function;

public class Solution {
    static class TTLCache<V> {
        private static final class Entry<V> {
            final V value;
            final long expiresAtMs;

            Entry(V value, long expiresAtMs) {
                this.value = value;
                this.expiresAtMs = expiresAtMs;
            }
        }

        private final Map<String, Entry<V>> entries = new HashMap<>();
        int hits = 0;
        int misses = 0;
        int evictions = 0;

        void put(String key, V value, long nowMs, long ttlMs) {
            if (ttlMs <= 0) throw new IllegalArgumentException("ttl_ms must be positive");
            entries.put(key, new Entry<>(value, nowMs + ttlMs));
        }

        V get(String key, long nowMs) {
            Entry<V> entry = entries.get(key);
            if (entry == null) {
                misses++;
                return null;
            }
            if (nowMs >= entry.expiresAtMs) {            // exactly at TTL -> expired
                entries.remove(key);
                evictions++;
                misses++;
                return null;
            }
            hits++;
            return entry.value;
        }

        V memoize(Function<String, V> fn, String key, long nowMs, long ttlMs) {
            V cached = get(key, nowMs);
            if (cached != null) return cached;
            V value = fn.apply(key);
            put(key, value, nowMs, ttlMs);
            return value;
        }

        int size() {
            return entries.size();
        }
    }

    public static void main(String[] args) {
        TTLCache<Integer> cache = new TTLCache<>();
        cache.put("a", 10, 1000, 100);                   // valid while now < 1100
        check(cache.get("a", 1099) == 10, "valid before TTL boundary");
        check(cache.get("a", 1100) == null, "exactly at TTL -> expired");
        check(cache.hits == 1 && cache.misses == 1 && cache.evictions == 1, "stats after boundary access");
        check(cache.size() == 0, "expired entry removed lazily");

        cache.put("b", 20, 0, 50);
        check(cache.get("b", 25) == 20, "fresh entry");
        cache.put("b", 21, 30, 50);                      // refresh -> expires at 80
        check(cache.get("b", 79) == 21, "refreshed value");
        check(cache.get("b", 80) == null, "refresh moved the expiry");

        TTLCache<Integer> missing = new TTLCache<>();
        check(missing.get("nope", 0) == null, "missing key");
        check(missing.hits == 0 && missing.misses == 1, "missing key counted as a miss");
        checkThrows(() -> missing.put("x", 1, 0, 0), "non-positive ttl rejected");

        List<String> calls = new ArrayList<>();
        Function<String, Integer> slow = n -> {
            calls.add(n);
            return 49;
        };
        TTLCache<Integer> memo = new TTLCache<>();
        check(memo.memoize(slow, "7", 0, 100) == 49, "first call computes");
        check(memo.memoize(slow, "7", 99, 100) == 49, "second call hits the cache");
        check(calls.size() == 1, "expensive function called once");
        check(memo.memoize(slow, "7", 100, 100) == 49, "expired value recomputed");
        check(calls.size() == 2, "expensive function called again after expiry");
        check(memo.hits == 1 && memo.misses == 2 && memo.evictions == 1, "memoize stats");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static void checkThrows(Runnable action, String label) {
        try {
            action.run();
        } catch (IllegalArgumentException expected) {
            return;
        }
        throw new AssertionError("FAILED: expected error - " + label);
    }
}
