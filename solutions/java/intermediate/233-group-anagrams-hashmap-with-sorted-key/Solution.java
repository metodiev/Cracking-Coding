/** Task 233 - Group anagrams (hashmap with sorted key). (Tier: Intermediate, CTCI Ch 10) */
import java.util.*;

public class Solution {
    /**
     * Key each word by its characters sorted ("eat"/"tea"/"ate" -> "aet") and collect
     * the words in a LinkedHashMap: O(n * L log L) for n words of length L.
     * Deterministic order: groups appear in first-seen key order, words keep input order.
     */
    static List<List<String>> groupAnagrams(String[] words) {
        Map<String, List<String>> groups = new LinkedHashMap<>();
        for (String word : words) {
            char[] chars = word.toCharArray();
            Arrays.sort(chars);
            groups.computeIfAbsent(new String(chars), k -> new ArrayList<>()).add(word);
        }
        return new ArrayList<>(groups.values());
    }

    public static void main(String[] args) {
        check(groupAnagrams(new String[]{"eat", "tea", "tan", "ate", "nat", "bat"}).equals(
                Arrays.asList(
                        Arrays.asList("eat", "tea", "ate"),
                        Arrays.asList("tan", "nat"),
                        Arrays.asList("bat"))),
                "classic input");
        check(groupAnagrams(new String[]{}).isEmpty(), "empty input");
        check(groupAnagrams(new String[]{""}).equals(Arrays.asList(Arrays.asList(""))), "single empty word");
        check(groupAnagrams(new String[]{"", ""}).equals(Arrays.asList(Arrays.asList("", ""))), "two empty words");
        check(groupAnagrams(new String[]{"a"}).equals(Arrays.asList(Arrays.asList("a"))), "single word");
        check(groupAnagrams(new String[]{"ab", "ba", "abc", "cab", "bca"}).equals(
                Arrays.asList(
                        Arrays.asList("ab", "ba"),
                        Arrays.asList("abc", "cab", "bca"))),
                "different lengths");
        check(groupAnagrams(new String[]{"a", "b", "a"}).equals(
                Arrays.asList(Arrays.asList("a", "a"), Arrays.asList("b"))),
                "repeated words split by key");
        check(groupAnagrams(new String[]{"abc", "abc"}).equals(
                Arrays.asList(Arrays.asList("abc", "abc"))),
                "same word twice");
        check(groupAnagrams(new String[]{"listen", "silent", "enlist", "google"}).equals(
                Arrays.asList(
                        Arrays.asList("listen", "silent", "enlist"),
                        Arrays.asList("google"))),
                "listen/silent/enlist");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
