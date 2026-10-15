/** Task 045 - Replace spaces with %20 (URLify). (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static char[] urlify(char[] chars, int length) {
        int write = chars.length - 1;
        for (int read = length - 1; read >= 0; read--) {
            if (chars[read] == ' ') {
                chars[write] = '0';
                chars[write - 1] = '2';
                chars[write - 2] = '%';
                write -= 3;
            } else {
                chars[write--] = chars[read];
            }
        }
        return chars;
    }

    public static void main(String[] args) {
        check(new String(urlify("a b c    ".toCharArray(), 5)).equals("a%20b%20c"), "urlify");
        check(new String(urlify("abc".toCharArray(), 3)).equals("abc"), "no spaces");
        System.out.println("PASS");
    }
    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
