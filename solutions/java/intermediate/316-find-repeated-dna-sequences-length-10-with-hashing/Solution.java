/** Task 316 - Repeated DNA sequences of length 10 (hashing). (Tier: Intermediate, CTCI Ch 1) */
import java.util.*;

public class Solution {
    static final int DNA_LENGTH = 10;
    static final String BASES = "ACGT";

    /**
     * All k-letter substrings appearing more than once, sorted; empty list if none. Each base is
     * packed into 2 bits, so the rolling key in [0, 4^k) is collision-free and O(1) per step.
     */
    static List<String> repeatedDnaSequences(String s, int k) {
        List<String> result = new ArrayList<>();
        if (k <= 0 || s.length() < k) return result;
        int mask = (1 << (2 * k)) - 1;                       // keeps only the last k bases
        Set<Integer> seen = new HashSet<>();
        Set<String> repeated = new TreeSet<>();
        int key = 0;
        for (int i = 0; i < s.length(); i++) {
            key = ((key << 2) | BASES.indexOf(s.charAt(i))) & mask;
            if (i >= k - 1) {
                if (!seen.add(key)) repeated.add(s.substring(i - k + 1, i + 1));
            }
        }
        result.addAll(repeated);
        return result;
    }

    /** Counting map over every substring; the independent oracle. */
    static List<String> repeatedDnaNaive(String s, int k) {
        List<String> result = new ArrayList<>();
        if (k <= 0 || s.length() < k) return result;
        Map<String, Integer> counts = new TreeMap<>();
        for (int i = 0; i + k <= s.length(); i++) {
            counts.merge(s.substring(i, i + k), 1, Integer::sum);
        }
        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
            if (entry.getValue() > 1) result.add(entry.getKey());
        }
        return result;
    }

    /** Occurrences counted with overlaps (non-overlapping counting would undercount repeats). */
    static int overlappingCount(String s, String word) {
        int count = 0;
        for (int i = 0; i + word.length() <= s.length(); i++) {
            if (s.startsWith(word, i)) count++;
        }
        return count;
    }

    private static int seed = 3160;

    private static String lcgDna(int length) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(BASES.charAt(seed % 4));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        String classic = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT";
        check(repeatedDnaSequences(classic, DNA_LENGTH).equals(Arrays.asList("AAAAACCCCC", "CCCCCAAAAA")), "classic");
        check(repeatedDnaNaive(classic, DNA_LENGTH).equals(Arrays.asList("AAAAACCCCC", "CCCCCAAAAA")), "classic oracle");

        check(repeatedDnaSequences("ACGTACGTAC", DNA_LENGTH).isEmpty(), "no repeat");
        check(repeatedDnaSequences("AAAAAAAAAAA", DNA_LENGTH).equals(Arrays.asList("AAAAAAAAAA")), "eleven A");
        check(repeatedDnaSequences("AAAAAAAAAA", DNA_LENGTH).isEmpty(), "ten A");
        check(repeatedDnaSequences("ACGT", DNA_LENGTH).isEmpty(), "short");
        check(repeatedDnaSequences("", DNA_LENGTH).isEmpty(), "empty");
        check(repeatedDnaSequences("ACGTACGTAC", 4).equals(Arrays.asList("ACGT", "CGTA", "GTAC")), "k=4");
        check(repeatedDnaSequences("ACGTACGTAC", 0).isEmpty(), "k=0");
        check(repeatedDnaSequences("ACGTACGTAC", 20).isEmpty(), "k too large");

        List<String> samples = new ArrayList<>(Arrays.asList(classic, "AAAAAAAAAAAA", "ACGTACGTACGTACGTACGT"));
        for (int i = 0; i < 40; i++) samples.add(lcgDna(40));
        for (int i = 0; i < 20; i++) samples.add(lcgDna(20));

        for (String s : samples) {
            List<String> found = repeatedDnaSequences(s, DNA_LENGTH);
            check(found.equals(repeatedDnaNaive(s, DNA_LENGTH)), "cross-check " + s);
            List<String> sorted = new ArrayList<>(found);
            Collections.sort(sorted);
            check(found.equals(sorted), "sorted " + s);
            check(new HashSet<>(found).size() == found.size(), "unique " + s);
            for (String word : found) {                     // every report really repeats
                check(word.length() == DNA_LENGTH && overlappingCount(s, word) > 1, "repeat " + s + " " + word);
            }
        }

        // The result is exactly the set of 10-mers occurring at least twice
        for (int i = 0; i < 20; i++) {
            String s = lcgDna(30);
            List<String> expect = new ArrayList<>();
            for (int start = 0; start + DNA_LENGTH <= s.length(); start++) {
                String word = s.substring(start, start + DNA_LENGTH);
                if (overlappingCount(s, word) > 1 && !expect.contains(word)) expect.add(word);
            }
            Collections.sort(expect);
            check(repeatedDnaSequences(s, DNA_LENGTH).equals(expect), "brute " + s);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
