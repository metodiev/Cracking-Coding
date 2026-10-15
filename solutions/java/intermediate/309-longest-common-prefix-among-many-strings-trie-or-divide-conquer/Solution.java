/** Task 309 - Longest common prefix among many strings. (Tier: Intermediate, CTCI Ch 1) */
import java.util.*;

public class Solution {
    /** min/max bound: the common prefix is the prefix of the two extremes. */
    static String longestCommonPrefixSorted(String[] strs) {
        if (strs.length == 0) return "";
        String smallest = strs[0], largest = strs[0];
        for (String s : strs) {
            if (s.compareTo(smallest) < 0) smallest = s;
            if (s.compareTo(largest) > 0) largest = s;
        }
        int i = 0;
        while (i < smallest.length() && i < largest.length() && smallest.charAt(i) == largest.charAt(i)) i++;
        return smallest.substring(0, i);
    }

    /** Column-by-column scan; the independent oracle. */
    static String longestCommonPrefixVertical(String[] strs) {
        if (strs.length == 0) return "";
        for (int column = 0; column < strs[0].length(); column++) {
            char ch = strs[0].charAt(column);
            for (int k = 1; k < strs.length; k++) {
                if (column >= strs[k].length() || strs[k].charAt(column) != ch) return strs[0].substring(0, column);
            }
        }
        return strs[0];
    }

    /** Trie walk: keep descending while a single child is present and no word ends here. */
    static String longestCommonPrefixTrie(String[] strs) {
        Map<Character, Object> root = new HashMap<>();
        for (String word : strs) {
            Map<Character, Object> node = root;
            for (int i = 0; i < word.length(); i++) {
                char ch = word.charAt(i);
                Object next = node.get(ch);
                if (!(next instanceof Map)) {
                    Map<Character, Object> fresh = new HashMap<>();
                    node.put(ch, fresh);
                    next = fresh;
                }
                @SuppressWarnings("unchecked")
                Map<Character, Object> typed = (Map<Character, Object>) next;
                node = typed;
            }
            node.put('$', Boolean.TRUE);
        }
        StringBuilder prefix = new StringBuilder();
        Map<Character, Object> current = root;
        while (current.size() == 1 && !current.containsKey('$')) {
            char ch = current.keySet().iterator().next();
            prefix.append(ch);
            @SuppressWarnings("unchecked")
            Map<Character, Object> typed = (Map<Character, Object>) current.get(ch);
            current = typed;
        }
        return prefix.toString();
    }

    private static int seed = 1592;

    private static String[][] lcgLists(int count) {
        List<String[]> lists = new ArrayList<>();
        for (int k = 0; k < count; k++) {
            int size = 1 + seed % 3;
            String[] words = new String[size];
            for (int w = 0; w < size; w++) {
                int length = 1 + seed % 4;
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < length; i++) {
                    seed = (seed * 75 + 74) % 65537;
                    sb.append("abc".charAt(seed % 3));
                }
                words[w] = sb.toString();
            }
            seed = (seed * 75 + 74) % 65537;
            lists.add(words);
        }
        return lists.toArray(new String[0][]);
    }

    public static void main(String[] args) {
        check(longestCommonPrefixSorted(new String[]{"flower","flow","flight"}).equals("fl"), "flower group");
        check(longestCommonPrefixSorted(new String[]{"dog","racecar","car"}).isEmpty(), "no common prefix");
        check(longestCommonPrefixSorted(new String[]{"abc","abc"}).equals("abc"), "identical");
        check(longestCommonPrefixSorted(new String[]{"a"}).equals("a"), "single");
        check(longestCommonPrefixSorted(new String[]{}).isEmpty(), "empty list");
        check(longestCommonPrefixSorted(new String[]{"", "b"}).isEmpty(), "empty string");
        check(longestCommonPrefixSorted(new String[]{"abab","aba","ababbb"}).equals("aba"), "nested words");

        String[][] samples = {{"flower","flow","flight"},{"dog","racecar","car"},{"abc","abc"},{"","b"}};
        for (String[] strs : samples) {
            String expected = longestCommonPrefixSorted(strs);
            check(longestCommonPrefixVertical(strs).equals(expected), "vertical " + Arrays.toString(strs));
            check(longestCommonPrefixTrie(strs).equals(expected), "trie " + Arrays.toString(strs));
        }

        for (String[] strs : lcgLists(80)) {
            String expected = longestCommonPrefixSorted(strs);
            check(longestCommonPrefixVertical(strs).equals(expected), "random vertical " + Arrays.toString(strs));
            check(longestCommonPrefixTrie(strs).equals(expected), "random trie " + Arrays.toString(strs));
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
