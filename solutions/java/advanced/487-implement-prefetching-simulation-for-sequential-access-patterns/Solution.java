/** Task 487 - Implement prefetching simulation for sequential access patterns. (Tier: Advanced, CTCI Ch 14)
 *
 * Simulated memory with documented semantics:
 * - Address space is [0, bound); every access and every prefetched address must be inside it - the
 *   prefetcher clips at bound-1 so it can never read past the array.
 * - Sequential pattern detector: the first access of a run is only recorded; every following access
 *   with addr == last + 1 continues the run and issues speculative prefetches for
 *   addr + 1 .. min(addr + depth, bound - 1). Any non-consecutive access restarts the detector.
 * - A prefetch "hit" is an access to an address sitting in the prefetched set (one entry per
 *   address); a hit consumes the entry. Documented lower bound: a cold run of length L >= 2 yields
 *   at least L - 2 prefetch hits (the run's first access and its immediate successor cannot be
 *   covered), so a fresh run of length L scores (L-2)/L (0.9375 at L = 32).
 *
 * The test pins hand cases, then asserts on sequential traces that the per-segment prefetch hit
 * ratio is >= 0.9 (documented threshold) and that no simulated read ever leaves [0, bound).
 */
import java.util.*;

class Prefetcher {
    final int bound;
    final int depth;
    int last = -1;
    boolean started;
    final Set<Integer> pending = new HashSet<>();
    int hits;
    int accesses;
    int issued;
    int maxPrefetch = -1;

    Prefetcher(int bound, int depth) {
        this.bound = bound;
        this.depth = depth;
    }

    boolean access(int addr) {
        if (addr < 0 || addr >= bound) throw new AssertionError("FAILED: access out of bounds " + addr);
        accesses++;
        boolean hit = pending.remove(addr);
        if (hit) hits++;
        boolean continues = started && addr == last + 1;
        last = addr;
        started = true;
        if (continues) {
            for (int nxt = addr + 1; nxt <= Math.min(addr + depth, bound - 1); nxt++) {
                pending.add(nxt);
                issued++;
                maxPrefetch = Math.max(maxPrefetch, nxt);
            }
        }
        return hit;
    }
}

public class Solution {
    private static int seed = 487;

    private static int[] lcgValues(int count, int limit) {
        int[] values = new int[count];
        for (int i = 0; i < count; i++) {
            seed = (seed * 75 + 74) % 65537;
            values[i] = seed % limit;
        }
        return values;
    }

    public static void main(String[] args) {
        Prefetcher prefetcher = new Prefetcher(10, 3);
        boolean[] observed = new boolean[7];
        for (int addr = 0; addr < 7; addr++) observed[addr] = prefetcher.access(addr);
        check(Arrays.equals(observed, new boolean[]{false, false, true, true, true, true, true}), "hand vector");
        check(prefetcher.hits == 5, "hand hits " + prefetcher.hits);
        check(prefetcher.pending.equals(new HashSet<>(Arrays.asList(7, 8, 9))), "hand pending " + prefetcher.pending);
        check(prefetcher.maxPrefetch == 9, "hand max");
        check(prefetcher.issued == 18, "hand issued " + prefetcher.issued);
        check(prefetcher.access(9), "stale hit");
        check(prefetcher.hits == 6, "stale hits");
        check(prefetcher.pending.equals(new HashSet<>(Arrays.asList(7, 8))), "pending after 9");
        check(!prefetcher.access(4), "backwards jump misses");
        check(prefetcher.hits == 6 && prefetcher.issued == 18, "backwards leaves counters");

        prefetcher = new Prefetcher(10, 3);
        observed = new boolean[10];
        for (int addr = 0; addr < 10; addr++) observed[addr] = prefetcher.access(addr);
        boolean[] expected = {false, false, true, true, true, true, true, true, true, true};
        check(Arrays.equals(observed, expected), "clip vector");
        check(prefetcher.hits == 8, "clip hits");
        check(prefetcher.maxPrefetch == 9, "clip max");
        check(prefetcher.pending.isEmpty(), "clip pending empty");

        prefetcher = new Prefetcher(64, 4);
        for (int addr = 0; addr < 40; addr++) prefetcher.access(addr);
        check(prefetcher.hits == 38 && prefetcher.accesses == 40, "fresh run " + prefetcher.hits);
        check(prefetcher.hits * 10 >= 9 * prefetcher.accesses, "fresh ratio");
        check(prefetcher.maxPrefetch == 43, "fresh max " + prefetcher.maxPrefetch);

        prefetcher = new Prefetcher(128, 4);
        int[] jumps = lcgValues(4, 80);
        int totalSegmentHits = 0;
        for (int jump : jumps) {
            int before = prefetcher.hits;
            for (int offset = 0; offset < 32; offset++) prefetcher.access(jump + offset);
            int got = prefetcher.hits - before;
            check(got >= 30, "segment hits " + jump + " " + got);
            check(got * 10 >= 9 * 32, "segment ratio " + jump);
            totalSegmentHits += got;
        }
        check(totalSegmentHits == 120, "segment total " + totalSegmentHits);
        check(prefetcher.maxPrefetch < 128, "mixed max");
        check(prefetcher.hits <= prefetcher.accesses, "sane hits");
        check(prefetcher.hits * 10 >= 9 * prefetcher.accesses, "mixed ratio");

        prefetcher = new Prefetcher(64, 5);
        seed = 488;
        List<Integer> trace = new ArrayList<>();
        for (int value : lcgValues(30, 64)) {
            trace.add(value);
            if (value + 2 < 64) {
                trace.add(value + 1);
                trace.add(value + 2);
            }
        }
        for (int addr : trace) {
            check(addr >= 0 && addr < 64, "trace bounds");
            prefetcher.access(addr);
        }
        check(prefetcher.maxPrefetch <= 63, "interleaved max");
        check(prefetcher.hits <= prefetcher.accesses && prefetcher.accesses == trace.size(), "interleaved counters");

        prefetcher = new Prefetcher(128, 4);
        seed = 487;
        for (int jump : lcgValues(4, 80)) {
            for (int offset = 0; offset < 32; offset++) prefetcher.access(jump + offset);
        }
        check(prefetcher.accesses == 128 && prefetcher.issued == 496, "pins " + prefetcher.accesses + " " + prefetcher.issued);
        check(prefetcher.hits == 120, "pinned hits " + prefetcher.hits);
        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
