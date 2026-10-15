/** Task 161 - Implement LFU cache (basic variant). (Tier: Beginner, CTCI Ch 7) */
import java.util.*;

public class Solution {
    static class LFUCache {
        private final int capacity;
        private final Map<Integer, long[]> data = new HashMap<>();  // key -> {value, freq, seq}
        private long tick = 0;

        LFUCache(int capacity) {
            if (capacity <= 0) throw new IllegalArgumentException("capacity must be positive");
            this.capacity = capacity;
        }

        int get(int key) {
            long[] entry = data.get(key);
            if (entry == null) return -1;
            entry[1]++;
            entry[2] = tick++;
            return (int) entry[0];
        }

        void put(int key, int value) {
            long[] entry = data.get(key);
            if (entry != null) {
                entry[0] = value;
                entry[1]++;
                entry[2] = tick++;
                return;
            }
            if (data.size() >= capacity) {
                int victim = -1;
                long bestFreq = Long.MAX_VALUE, bestSeq = Long.MAX_VALUE;
                for (Map.Entry<Integer, long[]> e : data.entrySet()) {
                    long freq = e.getValue()[1], seq = e.getValue()[2];
                    if (freq < bestFreq || (freq == bestFreq && seq < bestSeq)) {
                        bestFreq = freq;
                        bestSeq = seq;
                        victim = e.getKey();
                    }
                }
                data.remove(victim);
            }
            data.put(key, new long[]{value, 1, tick++});
        }
    }

    public static void main(String[] args) {
        LFUCache cache = new LFUCache(2);
        cache.put(1, 1);
        cache.put(2, 2);
        check(cache.get(1) == 1, "get 1");
        cache.put(3, 3);
        check(cache.get(2) == -1, "2 evicted (lowest freq)");
        check(cache.get(3) == 3, "get 3");
        cache.put(4, 4);
        check(cache.get(1) == -1, "tie on freq -> LRU key 1 evicted");
        check(cache.get(3) == 3 && cache.get(4) == 4, "3 and 4 remain");

        LFUCache c2 = new LFUCache(1);
        c2.put(9, 1);
        c2.put(9, 2);
        check(c2.get(9) == 2, "update key");
        c2.put(8, 8);
        check(c2.get(9) == -1 && c2.get(8) == 8, "single-slot replacement");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
