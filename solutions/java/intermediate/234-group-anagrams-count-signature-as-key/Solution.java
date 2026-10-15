/** Task 234 - Group anagrams (count signature as key). (Tier: Intermediate, CTCI Ch 1) */
import java.util.*;

public class Solution {
    /**
     * Key each word by its 26-entry count signature (how many 'a's, 'b's, ... it
     * contains) instead of the sorted word: O(n * L) time, no sorting. Inputs are
     * assumed lowercase. Ordering is identical to task 233 (first-seen key order,
     * input order within a group), so both approaches return equal output.
     */
    static List<List<String>> groupAnagramsSignature(String[] words) {
        Map<String, List<String>> groups = new LinkedHashMap<>();
        for (String word : words) {
            int[] counts = new int[26];
            for (int i = 0; i < word.length(); i++) counts[word.charAt(i) - 'a']++;
            StringBuilder sb = new StringBuilder();
            for (int c : counts) sb.append(c).append(',');
            groups.computeIfAbsent(sb.toString(), k -> new ArrayList<>()).add(word);
        }
        return new ArrayList<>(groups.values());
    }

    /** Task 233's sorted-key variant, kept here only for the agreement checks. */
    static List<List<String>> groupAnagramsSorted(String[] words) {
        Map<String, List<String>> groups = new LinkedHashMap<>();
        for (String word : words) {
            char[] chars = word.toCharArray();
            Arrays.sort(chars);
            groups.computeIfAbsent(new String(chars), k -> new ArrayList<>()).add(word);
        }
        return new ArrayList<>(groups.values());
    }

    public static void main(String[] args) {
        String[][] cases = {
            {"eat", "tea", "tan", "ate", "nat", "bat"},
            {"listen", "silent", "enlist", "google", "banana", "banana"},
            {"", "a", "", "a", "ab", "ba"},
            {"abc", "abc"},
            {"z"},
            {},
        };
        for (String[] words : cases) {              // both approaches must agree
            check(groupAnagramsSignature(words).equals(groupAnagramsSorted(words)), "approaches agree");
        }

        check(groupAnagramsSignature(new String[]{"eat", "tea", "tan", "ate", "nat", "bat"}).equals(
                Arrays.asList(
                        Arrays.asList("eat", "tea", "ate"),
                        Arrays.asList("tan", "nat"),
                        Arrays.asList("bat"))),
                "classic input");
        check(groupAnagramsSignature(new String[]{}).isEmpty(), "empty input");
        check(groupAnagramsSignature(new String[]{""}).equals(Arrays.asList(Arrays.asList(""))), "empty word");
        check(groupAnagramsSignature(new String[]{"a", "b", "a"}).equals(
                Arrays.asList(Arrays.asList("a", "a"), Arrays.asList("b"))),
                "repeated words split by key");
        check(groupAnagramsSignature(new String[]{"ab", "ba", "abc", "cab", "bca"}).equals(
                Arrays.asList(
                        Arrays.asList("ab", "ba"),
                        Arrays.asList("abc", "cab", "bca"))),
                "different lengths");

        java.util.Random rng = new java.util.Random(234);
        for (int test = 0; test < 200; test++) {    // random agreement check
            int count = rng.nextInt(11);
            String[] words = new String[count];
            for (int i = 0; i < count; i++) {
                StringBuilder sb = new StringBuilder();
                int len = rng.nextInt(7);
                for (int j = 0; j < len; j++) sb.append((char) ('a' + rng.nextInt(3)));
                words[i] = sb.toString();
            }
            check(groupAnagramsSignature(words).equals(groupAnagramsSorted(words)), "random agreement");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
