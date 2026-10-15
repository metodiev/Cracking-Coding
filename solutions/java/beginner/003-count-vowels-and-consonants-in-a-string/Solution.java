/** Task 003 - Count vowels and consonants in a string. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static int[] countVowelsAndConsonants(String s) {
        int vowels = 0, consonants = 0;
        for (char c : s.toLowerCase().toCharArray()) {
            if (!Character.isLetter(c)) continue;
            if ("aeiou".indexOf(c) >= 0) vowels++; else consonants++;
        }
        return new int[]{vowels, consonants};
    }

    public static void main(String[] args) {
        check(java.util.Arrays.equals(countVowelsAndConsonants("hello"), new int[]{2, 3}), "hello");
        check(java.util.Arrays.equals(countVowelsAndConsonants(""), new int[]{0, 0}), "empty");
        check(java.util.Arrays.equals(countVowelsAndConsonants("AEIOU"), new int[]{5, 0}), "AEIOU");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
