/** Task 094 - Find longest common prefix in array of strings. (Tier: Beginner, CTCI Ch 8) */
public class Solution {
    static String longestCommonPrefix(String[] strs) {
        if (strs.length == 0) return "";
        String first = strs[0];
        for (int i = 0; i < first.length(); i++) {
            char ch = first.charAt(i);
            for (int k = 1; k < strs.length; k++) {
                String s = strs[k];
                if (i >= s.length() || s.charAt(i) != ch) return first.substring(0, i);
            }
        }
        return first;
    }

    public static void main(String[] args) {
        check(longestCommonPrefix(new String[]{"flower", "flow", "flight"}).equals("fl"), "fl");
        check(longestCommonPrefix(new String[]{"dog", "racecar", "car"}).equals(""), "none");
        check(longestCommonPrefix(new String[]{"apple"}).equals("apple"), "single");
        check(longestCommonPrefix(new String[]{}).equals(""), "empty array");
        check(longestCommonPrefix(new String[]{""}).equals(""), "empty string entry");
        check(longestCommonPrefix(new String[]{"", "a"}).equals(""), "empty first");
        check(longestCommonPrefix(new String[]{"a", "ab"}).equals("a"), "prefix is whole first");
        check(longestCommonPrefix(new String[]{"abc", "abc", "abc"}).equals("abc"), "identical");
        check(longestCommonPrefix(new String[]{"prefix", "pref"}).equals("pref"), "shorter second");
        check(longestCommonPrefix(new String[]{"ab", "ac"}).equals("a"), "shared single char");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
