/** Task 089 - Implement simple frequency histogram (integers). (Tier: Beginner, CTCI Ch 1)
 *
 * histogramCounts -> TreeMap value -> count, built in one pass.
 * renderHistogram -> one text line per distinct value, sorted ascending,
 * rendered as "value: " followed by one '#' per occurrence, e.g. "3: ###".
 */
import java.util.*;

public class Solution {
    static TreeMap<Integer, Integer> histogramCounts(int[] values) {
        TreeMap<Integer, Integer> counts = new TreeMap<>();
        for (int value : values) {
            counts.merge(value, 1, Integer::sum);
        }
        return counts;
    }

    static List<String> renderHistogram(TreeMap<Integer, Integer> counts) {
        List<String> lines = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
            StringBuilder bar = new StringBuilder();
            for (int i = 0; i < entry.getValue(); i++) bar.append('#');
            lines.add(entry.getKey() + ": " + bar);
        }
        return lines;
    }

    public static void main(String[] args) {
        TreeMap<Integer, Integer> counts = histogramCounts(new int[]{1, 2, 2, 3, 3, 3});
        check(counts.size() == 3 && counts.get(1) == 1 && counts.get(2) == 2 && counts.get(3) == 3, "counts");
        List<String> lines = renderHistogram(counts);
        check(lines.size() == 3, "one line per distinct value");
        check(lines.equals(Arrays.asList("1: #", "2: ##", "3: ###")), "rendering");

        TreeMap<Integer, Integer> empty = histogramCounts(new int[]{});
        check(empty.isEmpty() && renderHistogram(empty).isEmpty(), "empty");

        TreeMap<Integer, Integer> negatives = histogramCounts(new int[]{-1, -1, 0, 2, 2, 2});
        check(negatives.size() == 3 && negatives.get(-1) == 2 && negatives.get(0) == 1 && negatives.get(2) == 3,
                "negatives counts");
        check(renderHistogram(negatives).equals(Arrays.asList("-1: ##", "0: #", "2: ###")), "negatives rendering");

        TreeMap<Integer, Integer> single = histogramCounts(new int[]{7});
        check(single.size() == 1 && single.get(7) == 1, "single");
        check(renderHistogram(single).equals(Arrays.asList("7: #")), "single rendering");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
