/** Task 170 - Simple palindrome partitioning backtracking (list partitions). (Tier: Beginner, CTCI Ch 8) */
import java.util.*;

public class Solution {
    static boolean isPalindrome(String s, int lo, int hi) {
        while (lo < hi) {
            if (s.charAt(lo) != s.charAt(hi)) return false;
            lo++;
            hi--;
        }
        return true;
    }

    static List<List<String>> palindromePartitions(String s) {
        List<List<String>> out = new ArrayList<>();
        backtrack(s, 0, new ArrayList<>(), out);
        return out;
    }

    private static void backtrack(String s, int start, List<String> current, List<List<String>> out) {
        if (start == s.length()) {
            out.add(new ArrayList<>(current));
            return;
        }
        for (int end = start + 1; end <= s.length(); end++) {
            if (isPalindrome(s, start, end - 1)) {
                current.add(s.substring(start, end));
                backtrack(s, end, current, out);
                current.remove(current.size() - 1);
            }
        }
    }

    public static void main(String[] args) {
        check(palindromePartitions("aab").toString().equals("[[a, a, b], [aa, b]]"), "aab");
        check(palindromePartitions("a").toString().equals("[[a]]"), "single");
        check(palindromePartitions("").toString().equals("[[]]"), "empty -> one empty partition");
        check(palindromePartitions("aaa").size() == 4, "aaa has 4 partitions");
        check(palindromePartitions("ab").toString().equals("[[a, b]]"), "ab");
        for (List<String> partition : palindromePartitions("aabb")) {
            check(String.join("", partition).equals("aabb"), "concatenation round trip");
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
