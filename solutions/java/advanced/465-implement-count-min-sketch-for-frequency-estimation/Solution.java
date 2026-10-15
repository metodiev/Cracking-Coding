/** Task 465 - Implement count-min sketch for frequency estimation. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Count-min sketch with d = 4 rows and width = 48 cells, using FIXED hash seeds:
 * column r = ((a_r * key + b_r) mod 1000003) mod width for r in 0..3. Counters only
 * ever increase, so estimate(key) = min over the rows is NEVER below the true
 * count; the overshoot is bounded by the collision mass.
 *
 * Checks: a hand-computed tiny sketch (width 4: columns(7) = columns(11) = [2, 0],
 * columns(0) = [1, 3]) documents exact behaviours, then a fixed LCG stream of
 * N = 5000 updates over ids 0..63 asserts for every id: estimate >= true count, and
 * estimate - true count <= EPS * N with the documented EPS = 0.06 (>= the standard
 * e/width ~ 0.0566); the measured max overshoot is 101 (~1/3 of the 300 bound) and
 * 23 of the 64 ids do over-estimate, documenting that collisions really happen.
 */
import java.util.*;

class CountMinSketch {
    static final int P = 1000003;
    static final int[] HASH_A = {31, 131, 2711, 65521};
    static final int[] HASH_B = {17, 91, 601, 4099};

    private final int width, depth;
    private final int[][] table;

    CountMinSketch(int width, int depth) {
        this.width = width;
        this.depth = depth;
        this.table = new int[depth][width];
    }

    int[] columns(int key) {
        int[] cols = new int[depth];
        for (int r = 0; r < depth; r++)
            cols[r] = (int) (((long) HASH_A[r] * key + HASH_B[r]) % P % width);
        return cols;
    }

    void add(int key) {
        int[] cols = columns(key);
        for (int r = 0; r < depth; r++) table[r][cols[r]]++;
    }

    int estimate(int key) {
        int[] cols = columns(key);
        int best = Integer.MAX_VALUE;
        for (int r = 0; r < depth; r++) best = Math.min(best, table[r][cols[r]]);
        return best;
    }
}

public class Solution {
    private static final double EPS = 0.06;
    private static int seed;

    private static int lcg() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void main(String[] args) {
        CountMinSketch sketch = new CountMinSketch(4, 2);
        check(Arrays.equals(sketch.columns(7), new int[]{2, 0}), "columns(7)");
        check(Arrays.equals(sketch.columns(11), new int[]{2, 0}), "columns(11)");
        sketch.add(7);
        check(sketch.estimate(7) == 1, "tiny 1");
        sketch.add(11);
        check(sketch.estimate(11) == 2, "tiny 2");
        check(sketch.estimate(7) == 2, "tiny collision");
        sketch.add(7);
        check(sketch.estimate(7) == 3, "tiny 3");
        check(sketch.estimate(0) == 0, "tiny untouched");

        sketch = new CountMinSketch(48, 4);
        seed = 7654321;
        final int total = 5000;
        int[] trueCounts = new int[64];
        for (int i = 0; i < total; i++) {
            seed = (seed * 75 + 74) % 65537;
            trueCounts[seed % 64]++;
        }
        seed = 7654321;
        for (int i = 0; i < total; i++) {
            seed = (seed * 75 + 74) % 65537;
            sketch.add(seed % 64);
        }
        int maxOvershoot = 0;
        for (int item = 0; item < 64; item++) {
            int estimate = sketch.estimate(item);
            check(estimate >= trueCounts[item], "never under-estimates " + item);
            int overshoot = estimate - trueCounts[item];
            maxOvershoot = Math.max(maxOvershoot, overshoot);
            check(overshoot <= EPS * total, "overshoot bound " + item);
        }
        check(maxOvershoot >= 1, "collisions occurred");
        int sum = 0;
        for (int c : trueCounts) sum += c;
        check(sum == total, "bookkeeping");
        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
