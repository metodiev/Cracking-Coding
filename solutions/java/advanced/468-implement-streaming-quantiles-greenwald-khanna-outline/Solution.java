/** Task 468 - Implement streaming quantiles (Greenwald-Khanna outline). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Greenwald-Khanna ε-approximate quantile summary (SIGMOD 2001), batch merge-compress
 * variant: inserted values are buffered and every int(1/eps)+1 arrivals are merged
 * into the summary entries {value, g, delta} with band floor(2*eps*(n-1)); a fresh
 * value landing before an existing entry gets delta = entry.g + entry.delta - 1,
 * otherwise neighbours are absorbed while g_left + g_right + delta_right <= band.
 * QUANTILE(φ) accumulates g and returns the last entry with g_sum + delta within
 * int(φ*(n-1)+1) + int(eps*(n-1)), falling back to the tracked minimum.
 *
 * Checks against the exact sorted oracle: for every returned value v,
 *   rl = #{x < v} <= φ*N + ε*N   and   ru = #{x <= v} >= φ*N − ε*N,
 * with eps = 0.05, N = 2000, φ in {0, 0.05, ..., 1.0} over six stream patterns
 * (random, increasing, decreasing, constant, heavy duplicates, sawtooth); query(0)
 * returns the true minimum and the value is always a stream member. Documented
 * summary size bound (1/(2ε)+1)*(log2(2εn)+2) = 106 tuples, measured maximum 82.
 */
import java.util.*;

class GreenwaldKhanna {
    private final double eps;
    private final List<int[]> entries = new ArrayList<>(); // {value, g, delta}
    private final List<Integer> pending = new ArrayList<>();
    private int count;
    private int minimum;

    GreenwaldKhanna(double eps) {
        this.eps = eps;
    }

    int size() {
        return entries.size();
    }

    void insert(int value) {
        pending.add(value);
        count++;
        if (count == 1 || value < minimum) minimum = value;
        if (count % ((int) (1.0 / eps) + 1) == 0) compress();
    }

    void compress() {
        int threshold = (int) (2.0 * eps * (count - 1));
        Collections.sort(pending);
        List<int[]> fresh = new ArrayList<>();
        for (int value : pending) fresh.add(new int[]{value, 1, 0});
        List<int[]> old = entries;
        List<int[]> out = new ArrayList<>();
        int i = 0, j = 0;
        int nf = fresh.size(), no = old.size();
        while (i < nf || j < no) {
            if (i == nf) {
                int[] t = old.get(j);
                j++;
                if (j < no) {
                    int[] next = old.get(j);
                    if (t[1] + next[1] + next[2] <= threshold) {
                        next[1] += t[1];
                        continue;
                    }
                }
                out.add(t);
            } else if (j == no) {
                int[] t = fresh.get(i);
                i++;
                if (i < nf) {
                    int[] next = fresh.get(i);
                    if (t[1] + next[1] + next[2] <= threshold) {
                        next[1] += t[1];
                        continue;
                    }
                }
                out.add(t);
            } else if (fresh.get(i)[0] < old.get(j)[0]) {
                int[] t = fresh.get(i);
                int[] u = old.get(j);
                if (t[1] + u[1] + u[2] <= threshold) {
                    u[1] += t[1];
                } else {
                    t[2] = u[1] + u[2] - t[1];
                    out.add(t);
                }
                i++;
            } else {
                int[] t = old.get(j);
                j++;
                if (j < no) {
                    int[] next = old.get(j);
                    if (t[1] + next[1] + next[2] <= threshold) {
                        next[1] += t[1];
                        continue;
                    }
                }
                out.add(t);
            }
        }
        entries.clear();
        entries.addAll(out);
        pending.clear();
    }

    Integer query(double phi) {
        if (count == 0) return null;
        if (!pending.isEmpty()) compress();
        int rank = (int) (phi * (count - 1) + 1);
        int spread = (int) (eps * (count - 1));
        int total = 0;
        int i = 0;
        int n = entries.size();
        while (i < n) {
            total += entries.get(i)[1];
            if (total + entries.get(i)[2] > rank + spread) break;
            i++;
        }
        if (i == 0) return minimum;
        return entries.get(i - 1)[0];
    }
}

public class Solution {
    private static final double EPS = 0.05;
    private static int seed;

    private static int lcg() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    private static List<int[]> streams() {
        seed = 20260116;
        int[] randomStream = new int[2000];
        for (int i = 0; i < 2000; i++) randomStream[i] = lcg() % 1000;
        int[] dupes = new int[2000];
        for (int i = 0; i < 2000; i++) dupes[i] = lcg() % 5;
        int[] increasing = new int[2000];
        int[] decreasing = new int[2000];
        int[] constant = new int[2000];
        int[] sawtooth = new int[2000];
        for (int i = 0; i < 2000; i++) {
            increasing[i] = i;
            decreasing[i] = 1999 - i;
            constant[i] = 42;
            sawtooth[i] = i % 50;
        }
        List<int[]> out = new ArrayList<>();
        out.add(randomStream);
        out.add(increasing);
        out.add(decreasing);
        out.add(constant);
        out.add(dupes);
        out.add(sawtooth);
        return out;
    }

    private static int lowerBound(int[] sorted, int value) {
        int lo = 0, hi = sorted.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (sorted[mid] < value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    private static int upperBound(int[] sorted, int value) {
        int lo = 0, hi = sorted.length;
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            if (sorted[mid] <= value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    public static void main(String[] args) {
        List<int[]> data = streams();
        String[] names = {"random", "increasing", "decreasing", "constant", "dupes", "sawtooth"};
        int maxSize = 0;
        for (int s = 0; s < data.size(); s++) {
            int[] stream = data.get(s);
            int n = stream.length;
            GreenwaldKhanna sketch = new GreenwaldKhanna(EPS);
            for (int value : stream) sketch.insert(value);
            int[] exactly = stream.clone();
            Arrays.sort(exactly);
            for (int q = 0; q <= 20; q++) {
                double phi = q / 20.0;
                int value = sketch.query(phi);
                int lo = lowerBound(exactly, value);
                int hi = upperBound(exactly, value);
                double target = phi * n;
                double slack = EPS * n;
                check(lo <= target + slack + 1e-9, names[s] + " upper rank " + phi);
                check(hi >= target - slack - 1e-9, names[s] + " lower rank " + phi);
            }
            check(sketch.query(0.0) == exactly[0], names[s] + " min");
            maxSize = Math.max(maxSize, sketch.size()); // after queries force compression
        }
        double bound = (1.0 / (2 * EPS) + 1) * (Math.log(2 * EPS * 2000) / Math.log(2) + 2);
        check(maxSize <= bound, "size bound " + maxSize);
        check(maxSize == 82, "measured max size " + maxSize);

        GreenwaldKhanna empty = new GreenwaldKhanna(0.01);
        check(empty.query(0.5) == null, "empty");
        GreenwaldKhanna tiny = new GreenwaldKhanna(0.01);
        for (int value : new int[]{5, 1, 9, 3, 7}) tiny.insert(value);
        check(tiny.query(0.0) == 1, "tiny min");
        check(tiny.query(1.0) == 9, "tiny max");
        Integer median = tiny.query(0.5);
        check(median == 3 || median == 5, "tiny median");
        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
