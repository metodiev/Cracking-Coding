/** Task 464 - Implement reservoir sampling with weighted probabilities. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Weighted reservoir sampling, A-Res (Efraimidis & Spirakis): each item gets the
 * key u^(1/w) with u uniform in (0,1] drawn from a FIXED LCG as u = (seed + 1) /
 * 65537; the reservoir keeps the k largest keys (min-heap; an equal key is
 * displaced only by a smaller item index, as documented). For k = 1 an item with
 * weight w_i is selected with probability exactly w_i / W, which the statistical
 * checks exercise.
 *
 * Checks (deterministic): weights [1,2,3,4] over 20000 trials must land within
 * TOL = 0.02 of the weight ratios (measured max deviation 0.0022) with exactly one
 * item per trial; equal weights give the uniform 1/3; weights 1..10 with k = 3 over
 * 3000 trials yield 3 distinct items per trial, counts summing to 9000, and heavier
 * items selected more often.
 */
import java.util.*;

class WeightedReservoir {
    private final int size;
    private final PriorityQueue<double[]> heap;

    WeightedReservoir(int size) {
        this.size = size;
        this.heap = new PriorityQueue<>((a, b) -> a[0] != b[0] ? Double.compare(a[0], b[0]) : Double.compare(a[1], b[1]));
    }

    void add(int item, double weight, double u) {
        double key = Math.pow(u, 1.0 / weight);
        if (heap.size() < size) {
            heap.add(new double[]{key, item});
        } else {
            double[] top = heap.peek();
            if (key > top[0] || (key == top[0] && item < top[1])) {
                heap.poll();
                heap.add(new double[]{key, item});
            }
        }
    }

    int[] sample() {
        int[] items = new int[heap.size()];
        int i = 0;
        for (double[] entry : heap) items[i++] = (int) entry[1];
        Arrays.sort(items);
        return items;
    }
}

public class Solution {
    private static final double TOL = 0.02;
    private static int seed;

    private static double nextU() {
        seed = (seed * 75 + 74) % 65537;
        return (seed + 1) / 65537.0;
    }

    private static int[] sampleOnce(int k, double[] weights) {
        WeightedReservoir reservoir = new WeightedReservoir(k);
        for (int item = 0; item < weights.length; item++) {
            double u = nextU();
            reservoir.add(item, weights[item], u);
        }
        return reservoir.sample();
    }

    public static void main(String[] args) {
        double[] weights = {1, 2, 3, 4};
        int trials = 20000;
        seed = 99;
        int[] counts = new int[weights.length];
        for (int t = 0; t < trials; t++) {
            int[] picked = sampleOnce(1, weights);
            check(picked.length == 1, "one item per trial");
            counts[picked[0]]++;
        }
        double total = 0;
        for (double w : weights) total += w;
        double maxDev = 0;
        for (int i = 0; i < weights.length; i++) {
            double expected = weights[i] / total;
            double observed = counts[i] / (double) trials;
            maxDev = Math.max(maxDev, Math.abs(observed - expected));
            check(Math.abs(observed - expected) <= TOL, "freq " + i);
        }
        int sum = 0;
        for (int c : counts) sum += c;
        check(sum == trials, "count bookkeeping");
        check(counts[3] > counts[0], "heavier selected more");
        check(maxDev <= 0.01, "measured max deviation " + maxDev);

        seed = 12345;
        counts = new int[3];
        for (int t = 0; t < 20000; t++) {
            int[] picked = sampleOnce(1, new double[]{1, 1, 1});
            counts[picked[0]]++;
        }
        for (int i = 0; i < 3; i++) check(Math.abs(counts[i] / 20000.0 - 1.0 / 3) <= TOL, "uniform " + i);

        double[] ws = new double[10];
        for (int i = 0; i < 10; i++) ws[i] = i + 1;
        int k = 3;
        trials = 3000;
        seed = 777;
        counts = new int[10];
        for (int t = 0; t < trials; t++) {
            int[] picked = sampleOnce(k, ws);
            check(picked.length == k, "sample size");
            Set<Integer> distinct = new HashSet<>();
            for (int item : picked) distinct.add(item);
            check(distinct.size() == k, "distinct items");
            for (int item : picked) counts[item]++;
        }
        sum = 0;
        for (int c : counts) sum += c;
        check(sum == trials * k, "k bookkeeping");
        check(counts[9] > counts[0], "monotonicity");
        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
