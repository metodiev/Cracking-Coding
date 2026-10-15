/** Task 159 - Implement function memoizer decorator (or equivalent). (Tier: Beginner, CTCI Ch 8) */
import java.util.*;
import java.util.function.IntUnaryOperator;

/**
 * Memoizer for single-argument int functions, distinct from the task 128 wrapper:
 * records hits / misses, exposes the cache size, and supports an optional capacity
 * with FIFO eviction - when full, the OLDEST inserted key leaves first and a hit
 * never changes insertion order (FIFO, deliberately not LRU; see task 160).
 */
class Memoizer {
    private final IntUnaryOperator fn;
    private final Integer capacity;
    private final Map<Integer, Integer> cache = new HashMap<>();
    private final Deque<Integer> order = new ArrayDeque<>();
    int hits;
    int misses;

    Memoizer(IntUnaryOperator fn) {
        this(fn, null);
    }

    Memoizer(IntUnaryOperator fn, Integer capacity) {
        if (capacity != null && capacity <= 0) throw new IllegalArgumentException("capacity must be >= 1");
        this.fn = fn;
        this.capacity = capacity;
    }

    int apply(int n) {
        Integer cached = cache.get(n);
        if (cached != null) {
            hits++;
            return cached;
        }
        misses++;
        int value = fn.applyAsInt(n);
        if (capacity != null && cache.size() >= capacity) {
            Integer evicted = order.pollFirst();     // FIFO: oldest insertion goes first
            cache.remove(evicted);
        }
        cache.put(n, value);
        order.addLast(n);
        return value;
    }

    int size() {
        return cache.size();
    }

    /** True iff n is currently cached (used to assert eviction). */
    boolean cached(int n) {
        return cache.containsKey(n);
    }
}

public class Solution {
    /** Steps for n to reach 1; collatzSteps(27) == 111. */
    static int collatzSteps(int n) {
        if (n < 1) throw new IllegalArgumentException("n must be >= 1");
        int steps = 0;
        while (n != 1) {
            n = (n % 2 == 0) ? n / 2 : 3 * n + 1;
            steps++;
        }
        return steps;
    }

    public static void main(String[] args) {
        // 1) no capacity: everything stays cached
        Memoizer memo = new Memoizer(Solution::collatzSteps);
        check(memo.apply(27) == 111, "collatz 27");
        check(memo.apply(6) == 8 && memo.apply(7) == 16, "more values");
        check(memo.hits == 0 && memo.misses == 3 && memo.size() == 3, "stats after three misses");
        check(memo.apply(27) == 111 && memo.apply(6) == 8, "cached repeats");
        check(memo.hits == 2 && memo.misses == 3, "two hits");

        // 2) capacity 3 with FIFO eviction
        Memoizer small = new Memoizer(Solution::collatzSteps, 3);
        small.apply(27);
        small.apply(6);
        small.apply(7);
        check(small.size() == 3 && small.hits == 0 && small.misses == 3, "filled to capacity");
        check(small.apply(27) == 111 && small.cached(27), "hit keeps FIFO order");
        check(small.apply(9) == 19, "collatz 9");
        check(small.size() == 3 && !small.cached(27) && small.cached(9), "oldest evicted");
        check(small.apply(6) == 8, "6 still cached");
        check(small.hits == 2 && small.misses == 4, "small stats");

        // 3) capacity 1 behaves like a one-slot cache
        Memoizer tiny = new Memoizer(x -> x * x, 1);
        check(tiny.apply(3) == 9 && tiny.apply(4) == 16, "capacity 1");
        check(tiny.size() == 1 && !tiny.cached(3) && tiny.cached(4), "single slot");
        check(tiny.apply(4) == 16 && tiny.hits == 1 && tiny.misses == 2, "tiny stats");

        // capacity must be positive
        boolean threw = false;
        try { new Memoizer(x -> x, 0); } catch (IllegalArgumentException e) { threw = true; }
        check(threw, "capacity 0 throws");

        // stats are per-instance, not shared
        Memoizer other = new Memoizer(x -> x + 1);
        check(other.apply(1) == 2 && other.hits == 0 && other.misses == 1, "other instance");
        check(memo.hits == 2 && memo.misses == 3, "original stats untouched");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
