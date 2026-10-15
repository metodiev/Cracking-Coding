/** Task 165 - Reservoir sampling for k items from stream. (Tier: Beginner, CTCI Ch 6) */
import java.util.*;

public class Solution {
    /** Algorithm R with an injected Random, so tests are reproducible. */
    static List<Integer> reservoirSample(int[] stream, int k, Random rng) {
        List<Integer> sample = new ArrayList<>();
        for (int i = 0; i < stream.length; i++) {
            if (i < k) {
                sample.add(stream[i]);
            } else {
                int j = rng.nextInt(i + 1);
                if (j < k) sample.set(j, stream[i]);
            }
        }
        return sample;
    }

    public static void main(String[] args) {
        int[] stream = new int[100];
        for (int i = 0; i < 100; i++) stream[i] = i;

        List<Integer> sample = reservoirSample(stream, 5, new Random(7));
        check(sample.size() == 5, "sample size");
        check(new HashSet<>(sample).size() == 5, "distinct picks");
        check(sample.equals(reservoirSample(stream, 5, new Random(7))), "reproducible with same seed");
        check(!sample.equals(reservoirSample(stream, 5, new Random(8))), "differs with another seed");

        check(reservoirSample(new int[]{1, 2, 3}, 5, new Random(1)).toString().equals("[1, 2, 3]"), "k >= n keeps all");
        check(reservoirSample(new int[]{}, 3, new Random(1)).isEmpty(), "empty stream");

        int trials = 4000, hits = 0;
        Random rng = new Random(123);
        for (int t = 0; t < trials; t++)
            if (reservoirSample(stream, 5, rng).contains(0)) hits++;
        double ratio = hits / (double) trials;
        check(ratio > 0.03 && ratio < 0.08, "inclusion ratio ~0.05, got " + ratio);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
