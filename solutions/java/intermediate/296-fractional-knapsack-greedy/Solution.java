/** Task 296 - Fractional knapsack (greedy). (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    /** Greedy by ratio. Result: {total, index0, fraction0, index1, fraction1, ...}. */
    static double[] fractionalKnapsack(int[] weights, int[] values, int capacity) {
        int n = weights.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        Arrays.sort(order, (a, b) -> {
            double left = (double) values[a] / weights[a];
            double right = (double) values[b] / weights[b];
            if (left != right) return Double.compare(right, left);
            return Integer.compare(a, b);
        });
        double remaining = capacity;
        double total = 0.0;
        List<Double> picks = new ArrayList<>();
        for (int index : order) {
            if (remaining <= 0) break;
            double fraction = Math.min(1.0, remaining / weights[index]);
            total += fraction * values[index];
            remaining -= fraction * weights[index];
            picks.add((double) index);
            picks.add(fraction);
        }
        double[] result = new double[picks.size() + 1];
        result[0] = total;
        for (int i = 0; i < picks.size(); i++) result[i + 1] = picks.get(i);
        return result;
    }

    /**
     * Oracle: any optimum takes whole items plus at most one fraction; enumerate both.
     * The fraction is capped at 1 because the remaining item exists only once -- omitting
     * the cap turns the oracle into the unbounded (fluid-supply) variant by mistake.
     */
    static double bruteFractional(int[] weights, int[] values, int capacity) {
        int n = weights.length;
        double best = 0.0;
        for (int mask = 0; mask < (1 << n); mask++) {
            int weight = 0;
            int value = 0;
            for (int i = 0; i < n; i++) {
                if (((mask >> i) & 1) != 0) {
                    weight += weights[i];
                    value += values[i];
                }
            }
            if (weight > capacity) continue;
            best = Math.max(best, value);
            for (int j = 0; j < n; j++) {
                if (((mask >> j) & 1) == 0) {
                    double fraction = Math.min(1.0, (capacity - weight) / (double) weights[j]);
                    best = Math.max(best, value + fraction * values[j]);
                }
            }
        }
        return best;
    }

    private static int seed = 424242;

    private static int[][] lcgInstance(int n) {
        int[] weights = new int[n];
        int[] values = new int[n];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            weights[i] = 1 + seed % 9;
            seed = (seed * 75 + 74) % 65537;
            values[i] = 1 + seed % 9;
        }
        return new int[][]{weights, values};
    }

    public static void main(String[] args) {
        int[] weights = new int[]{10, 20, 30};
        int[] values = new int[]{60, 100, 120};
        double[] result = fractionalKnapsack(weights, values, 50);
        check(Math.abs(result[0] - 240.0) < 1e-9, "classic example value");
        check(result.length == 7 && result[1] == 0.0 && result[3] == 1.0 && result[5] == 2.0, "picked item indexes");
        check(Math.abs(result[2] - 1.0) < 1e-9 && Math.abs(result[4] - 1.0) < 1e-9, "whole items");
        check(Math.abs(result[6] - 2.0 / 3.0) < 1e-9, "two thirds of item 2");

        result = fractionalKnapsack(weights, values, 60);
        check(Math.abs(result[0] - 280.0) < 1e-9 && result.length == 7, "capacity covers everything");

        result = fractionalKnapsack(weights, values, 0);
        check(result[0] == 0.0 && result.length == 1, "capacity 0");

        result = fractionalKnapsack(new int[]{5}, new int[]{3}, 2);
        check(Math.abs(result[0] - 1.2) < 1e-9 && Math.abs(result[2] - 0.4) < 1e-9, "single fractional item");

        for (int trial = 0; trial < 120; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int n = 1 + seed % 8;
            int[][] instance = lcgInstance(n);
            int[] w = instance[0];
            int[] v = instance[1];
            seed = (seed * 75 + 74) % 65537;
            int capacity = seed % 50;
            double[] answer = fractionalKnapsack(w, v, capacity);
            check(Math.abs(answer[0] - bruteFractional(w, v, capacity)) < 1e-9, "random trial " + trial);
            double weight = 0;
            double value = 0;
            for (int i = 1; i + 1 < answer.length; i += 2) {
                weight += answer[i + 1] * w[(int) answer[i]];
                value += answer[i + 1] * v[(int) answer[i]];
            }
            check(weight <= capacity + 1e-9 && Math.abs(value - answer[0]) < 1e-9, "random picks " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
