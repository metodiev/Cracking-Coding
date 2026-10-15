/** Task 290 - Palindrome partitioning with backtracking. (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    private static String text = "";
    private static List<String> current = new ArrayList<>();
    private static List<List<String>> results = new ArrayList<>();

    /**
     * All ways to cut a string into palindromic parts. Pieces are extended shortest-first,
     * so partition lists come out in lexicographic order and results are deterministic.
     */
    static List<List<String>> palindromePartitioning(String s) {
        text = s;
        current = new ArrayList<>();
        results = new ArrayList<>();
        backtrack(0);
        return results;
    }

    private static void backtrack(int start) {
        if (start == text.length()) {
            results.add(new ArrayList<>(current));
            return;
        }
        for (int end = start + 1; end <= text.length(); end++) {
            String piece = text.substring(start, end);
            if (!isPalindrome(piece)) continue;
            current.add(piece);
            backtrack(end);
            current.remove(current.size() - 1);
        }
    }

    /** Independent checker: two pointers moving towards each other. */
    static boolean isPalindrome(String piece) {
        int left = 0, right = piece.length() - 1;
        while (left < right) {
            if (piece.charAt(left) != piece.charAt(right)) return false;
            left++;
            right--;
        }
        return true;
    }

    /** Oracle: enumerate all 2^(n-1) cut sets and keep the palindrome-only partitions. */
    static List<List<String>> bruteForcePartitions(String s) {
        int n = s.length();
        List<List<String>> found = new ArrayList<>();
        if (n == 0) {
            found.add(new ArrayList<>());
            return found;
        }
        for (int mask = 0; mask < (1 << (n - 1)); mask++) {
            List<String> parts = new ArrayList<>();
            int start = 0;
            for (int i = 0; i < n - 1; i++) {
                if (((mask >> i) & 1) == 1) {
                    parts.add(s.substring(start, i + 1));
                    start = i + 1;
                }
            }
            parts.add(s.substring(start));
            boolean ok = true;
            for (String part : parts) if (!isPalindrome(part)) { ok = false; break; }
            if (ok) found.add(parts);
        }
        found.sort((a, b) -> keyOf(a).compareTo(keyOf(b)));
        return found;
    }

    private static String keyOf(List<String> parts) {
        return String.join(",", parts);
    }

    static List<String> canonical(List<List<String>> partitions) {
        List<String> keys = new ArrayList<>();
        for (List<String> parts : partitions) keys.add(keyOf(parts));
        Collections.sort(keys);
        return keys;
    }

    private static int seed = 13579;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void main(String[] args) {
        check(palindromePartitioning("aab").equals(List.of(List.of("a", "a", "b"), List.of("aa", "b"))), "aab");
        check(palindromePartitioning("a").equals(List.of(List.of("a"))), "single char");
        check(palindromePartitioning("aba").equals(List.of(List.of("a", "b", "a"), List.of("aba"))), "aba");
        check(palindromePartitioning("aaa")
                .equals(List.of(List.of("a", "a", "a"), List.of("a", "aa"), List.of("aa", "a"), List.of("aaa"))), "aaa");
        check(palindromePartitioning("aaaa").equals(bruteForcePartitions("aaaa")), "aaaa");
        check(palindromePartitioning("aaaa").size() == 8, "aaaa count");
        check(palindromePartitioning("abba").size() == 3, "abba count");

        List<List<String>> empty = palindromePartitioning("");
        check(empty.size() == 1 && empty.get(0).isEmpty(), "empty input");

        for (int trial = 0; trial < 200; trial++) {
            int length = 1 + nextSeed() % 7;
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < length; i++) builder.append("abc".charAt(nextSeed() % 3));
            String s = builder.toString();
            List<List<String>> partitions = palindromePartitioning(s);
            check(canonical(partitions).equals(canonical(bruteForcePartitions(s))), "oracle " + s);
            check(canonical(partitions).equals(keyListSorted(partitions)), "lexicographic " + s);
            Set<String> seen = new HashSet<>();
            for (List<String> parts : partitions) {
                StringBuilder joined = new StringBuilder();
                for (String piece : parts) {
                    check(!piece.isEmpty() && isPalindrome(piece), "palindrome " + s);
                    joined.append(piece);
                }
                check(joined.toString().equals(s), "concatenation " + s);
                check(seen.add(keyOf(parts)), "duplicate " + s);
            }
            check(partitions.size() == seen.size(), "count " + s);
        }

        System.out.println("PASS");
    }

    /** Canonical form of the partitions in their own order (checks lexicographic output). */
    private static List<String> keyListSorted(List<List<String>> partitions) {
        List<String> keys = new ArrayList<>();
        for (List<String> parts : partitions) keys.add(keyOf(parts));
        return keys;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
