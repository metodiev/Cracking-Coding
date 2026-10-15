/** Task 015 - Check if two strings are anagrams. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static boolean areAnagrams(String a, String b) {
        if (a.length() != b.length()) return false;
        int[] count = new int[128];
        for (int i = 0; i < a.length(); i++) {
            count[a.charAt(i)]++;
            count[b.charAt(i)]--;
        }
        for (int c : count) if (c != 0) return false;
        return true;
    }

    public static void main(String[] args) {
        check(areAnagrams("listen", "silent"), "listen/silent");
        check(areAnagrams("", ""), "empty");
        check(!areAnagrams("hello", "world"), "hello/world");
        check(!areAnagrams("abc", "ab"), "length mismatch");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
