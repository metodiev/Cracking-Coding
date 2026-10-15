/** Task 469 - Implement distributed aggregations map-reduce toy. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Deterministic word-count MapReduce with explicit phases:
 *   split   - three input splits of LCG-generated lines over a 40-word vocabulary;
 *   map     - each split emits (word, 1) pairs;
 *   partition - partition_of(word) = polynomial hash (h = (h*31 + ord(ch)) mod
 *             1000003) mod R with R = 4 reducers (documented rule);
 *   shuffle - pairs grouped by (partition, word), partitions and words in sorted order;
 *   reduce  - each (partition, word) group summed to one (word, count) pair.
 *
 * Checks: reduced counts equal a plain HashMap count of all words; every word's pairs
 * land in exactly one partition equal to its hash; per-partition pair totals are
 * conserved; every word is reduced exactly once; the pipeline is deterministic; and
 * the hand-computed partition of "w00" is 3 (h("w00") = 115895, mod 4 = 3).
 */
import java.util.*;

public class Solution {
    static final int REDUCERS = 4;

    private static int seed;

    private static int lcg() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    static String[][] makeSplits(int startSeed) {
        seed = startSeed;
        String[][] splits = new String[3][];
        for (int s = 0; s < 3; s++) {
            List<String> words = new ArrayList<>();
            for (int line = 0; line < 20; line++) {
                int length = lcg() % 8 + 1;
                for (int k = 0; k < length; k++) words.add(String.format("w%02d", lcg() % 40));
            }
            splits[s] = words.toArray(new String[0]);
        }
        return splits;
    }

    static int partitionOf(String word) {
        int h = 0;
        for (int i = 0; i < word.length(); i++) h = (h * 31 + word.charAt(i)) % 1000003;
        return h % REDUCERS;
    }

    /** map: list of {splitId, word} pairs. */
    static List<String[]> mapPhase(String[][] splits) {
        List<String[]> pairs = new ArrayList<>();
        for (int splitId = 0; splitId < splits.length; splitId++) {
            for (String word : splits[splitId]) pairs.add(new String[]{Integer.toString(splitId), word});
        }
        return pairs;
    }

    /** partition + shuffle: partition -> word -> counts (partitions/words in sorted order). */
    static TreeMap<Integer, TreeMap<String, List<Integer>>> shufflePhase(List<String[]> pairs) {
        TreeMap<Integer, TreeMap<String, List<Integer>>> groups = new TreeMap<>();
        for (String[] pair : pairs) {
            String word = pair[1];
            groups.computeIfAbsent(partitionOf(word), unused -> new TreeMap<>())
                  .computeIfAbsent(word, unused -> new ArrayList<>())
                  .add(1);
        }
        return groups;
    }

    /** reduce: partition -> word -> total count. */
    static TreeMap<Integer, TreeMap<String, Integer>> reducePhase(TreeMap<Integer, TreeMap<String, List<Integer>>> groups) {
        TreeMap<Integer, TreeMap<String, Integer>> reduced = new TreeMap<>();
        for (Map.Entry<Integer, TreeMap<String, List<Integer>>> partEntry : groups.entrySet()) {
            TreeMap<String, Integer> partOut = new TreeMap<>();
            for (Map.Entry<String, List<Integer>> wordEntry : partEntry.getValue().entrySet()) {
                int total = 0;
                for (int count : wordEntry.getValue()) total += count;
                partOut.put(wordEntry.getKey(), total);
            }
            reduced.put(partEntry.getKey(), partOut);
        }
        return reduced;
    }

    static TreeMap<Integer, TreeMap<String, Integer>> runPipeline(int startSeed) {
        return reducePhase(shufflePhase(mapPhase(makeSplits(startSeed))));
    }

    public static void main(String[] args) {
        check(partitionOf("w00") == 3, "hand partition w00"); // h("w00") = 115895
        check(partitionOf("w") == 3, "hand partition w"); // h("w") = 119

        String[][] splits = makeSplits(20260116);
        List<String[]> pairs = mapPhase(splits);
        TreeMap<Integer, TreeMap<String, Integer>> reduced = runPipeline(20260116);

        // Plain count oracle over all words.
        Map<String, Integer> counts = new HashMap<>();
        for (String[] split : splits) for (String word : split) counts.merge(word, 1, Integer::sum);
        Map<String, Integer> flat = new HashMap<>();
        for (TreeMap<String, Integer> part : reduced.values()) {
            for (Map.Entry<String, Integer> entry : part.entrySet()) {
                check(!flat.containsKey(entry.getKey()), "word reduced once");
                flat.put(entry.getKey(), entry.getValue());
            }
        }
        check(flat.equals(counts), "counter oracle");

        // Routing: every word sits in the partition of its hash.
        for (Map.Entry<Integer, TreeMap<String, Integer>> part : reduced.entrySet()) {
            for (String word : part.getValue().keySet()) check(part.getKey() == partitionOf(word), "routing");
        }

        // Conservation of pair totals per partition.
        int[] perPartitionIn = new int[REDUCERS];
        for (String[] pair : pairs) perPartitionIn[partitionOf(pair[1])]++;
        int[] perPartitionOut = new int[REDUCERS];
        for (Map.Entry<Integer, TreeMap<String, Integer>> part : reduced.entrySet()) {
            int total = 0;
            for (int count : part.getValue().values()) total += count;
            perPartitionOut[part.getKey()] = total;
        }
        check(Arrays.equals(perPartitionIn, perPartitionOut), "partition conservation");
        int pairTotal = 0;
        for (int value : perPartitionIn) pairTotal += value;
        check(pairTotal == pairs.size(), "pair total");

        // Determinism.
        check(runPipeline(20260116).equals(reduced), "deterministic");
        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
