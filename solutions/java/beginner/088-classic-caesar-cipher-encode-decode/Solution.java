/** Task 088 - Classic Caesar cipher encode/decode. (Tier: Beginner, CTCI Ch 1)
 *
 * Shifts lowercase a-z with wrap-around; every shift is reduced mod 26 so large
 * and negative shifts work. Non-lowercase characters (including uppercase
 * letters, digits, spaces, and punctuation) are left unchanged, so
 * caesarDecode(caesarEncode(text, k), k) == text.
 */
public class Solution {
    static String caesarEncode(String text, int shift) {
        int normalized = ((shift % 26) + 26) % 26;
        StringBuilder out = new StringBuilder(text.length());
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (ch >= 'a' && ch <= 'z') {
                out.append((char) ('a' + (ch - 'a' + normalized) % 26));
            } else {
                out.append(ch);
            }
        }
        return out.toString();
    }

    static String caesarDecode(String text, int shift) {
        return caesarEncode(text, -shift);
    }

    public static void main(String[] args) {
        check(caesarEncode("abc", 3).equals("def"), "abc +3");
        check(caesarEncode("xyz", 3).equals("abc"), "wrap around");
        check(caesarEncode("hello, world!", 3).equals("khoor, zruog!"), "phrase");
        check(caesarEncode("abc", -3).equals("xyz"), "negative shift");
        check(caesarEncode("abc", 29).equals("def"), "shift > 26");
        check(caesarEncode("abc", 26).equals("abc"), "full rotation");
        check(caesarEncode("abc", -29).equals("xyz"), "large negative shift");
        check(caesarEncode("", 5).equals(""), "empty");

        check(caesarDecode("khoor", 3).equals("hello"), "decode");
        check(caesarDecode("def", 3).equals("abc"), "decode back");
        check(caesarDecode(caesarEncode("attack at dawn!", 7), 7).equals("attack at dawn!"), "round trip 7");
        check(caesarDecode(caesarEncode("the quick brown fox 123", 13), 13)
                .equals("the quick brown fox 123"), "round trip 13");

        // non-letters are unchanged; uppercase is not lowercase, so it is left alone
        check(caesarEncode("a1 B2 c!", 1).equals("b1 B2 d!"), "non-letters unchanged");
        check(caesarEncode("Hello", 1).equals("Hfmmp"), "uppercase unchanged");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
