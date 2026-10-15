/** Task 043 - Count words in a string. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static int countWords(String s) {
        int count = 0;
        boolean inWord = false;
        for (char c : s.toCharArray()) {
            if (Character.isWhitespace(c)) inWord = false;
            else if (!inWord) { inWord = true; count++; }
        }
        return count;
    }

    public static void main(String[] args) {
        check(countWords("the sky is blue") == 4, "sentence");
        check(countWords("   ") == 0, "spaces only");
        check(countWords("") == 0, "empty");
        check(countWords("one") == 1, "single");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
