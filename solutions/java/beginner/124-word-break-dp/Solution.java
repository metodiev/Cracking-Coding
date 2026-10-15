/** Task 124 - Word break (DP). (Tier: Beginner, CTCI Ch 8) */
import java.util.*;

public class Solution {
    static boolean wordBreak(String s, List<String> dictionary) {
        Set<String> words = new HashSet<>(dictionary);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++)
            for (int j = 0; j < i; j++)
                if (dp[j] && words.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
        return dp[s.length()];
    }

    static List<String> wordBreakSegmentation(String s, List<String> dictionary) {
        Set<String> words = new HashSet<>(dictionary);
        boolean[] dp = new boolean[s.length() + 1];
        int[] prev = new int[s.length() + 1];
        Arrays.fill(prev, -1);
        dp[0] = true;
        for (int i = 1; i <= s.length(); i++)
            for (int j = 0; j < i; j++)
                if (dp[j] && words.contains(s.substring(j, i))) {
                    dp[i] = true;
                    prev[i] = j;
                    break;
                }
        if (!dp[s.length()]) return null;
        LinkedList<String> parts = new LinkedList<>();
        for (int i = s.length(); i > 0; i = prev[i]) parts.addFirst(s.substring(prev[i], i));
        return parts;
    }

    public static void main(String[] args) {
        check(wordBreak("leetcode", Arrays.asList("leet", "code")), "leetcode");
        check(!wordBreak("catsandog", Arrays.asList("cats", "dog", "sand", "and", "cat")), "catsandog");
        check(wordBreak("applepenapple", Arrays.asList("apple", "pen")), "applepenapple");
        check(!wordBreak("a", List.of()), "empty dictionary");
        check(wordBreakSegmentation("leetcode", Arrays.asList("leet", "code")).toString().equals("[leet, code]"), "segmentation");
        check(wordBreakSegmentation("catsandog", Arrays.asList("cats", "dog")) == null, "no segmentation");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
