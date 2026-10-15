/** Task 341 - Longest valid parentheses substring (stack of indices). (Tier: Intermediate, CTCI Ch 8) */
public class Solution {
    static int longestValidSubstringStack(String s) {
        int[] stack = new int[s.length() + 1];
        int top = 0;
        stack[top] = -1;                                   // base: last unmatched ')'
        int best = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack[++top] = i;
            } else {
                top--;                                     // pop
                if (top < 0) {
                    stack[++top] = i;                      // this ')' becomes the new base
                } else {
                    best = Math.max(best, i - stack[top]);
                }
            }
        }
        return best;
    }

    /** dp[i] = length of the longest valid substring ending exactly at i. */
    static int longestValidSubstringDp(String s) {
        int n = s.length();
        int[] dp = new int[n];
        int best = 0;
        for (int i = 0; i < n; i++) {
            if (s.charAt(i) != ')') continue;
            if (i >= 1 && s.charAt(i - 1) == '(') {
                dp[i] = (i >= 2 ? dp[i - 2] : 0) + 2;
            } else if (i >= 1 && dp[i - 1] > 0) {
                int j = i - dp[i - 1] - 1;
                if (j >= 0 && s.charAt(j) == '(') {
                    dp[i] = dp[i - 1] + 2 + (j >= 1 ? dp[j - 1] : 0);
                }
            }
            best = Math.max(best, dp[i]);
        }
        return best;
    }

    /** Exhaustive oracle: every substring that stays balanced and returns to 0. */
    static int longestValidSubstringBrute(String s) {
        int best = 0;
        for (int i = 0; i < s.length(); i++) {
            int depth = 0;
            for (int j = i; j < s.length(); j++) {
                depth += s.charAt(j) == '(' ? 1 : -1;
                if (depth < 0) break;
                if (depth == 0) best = Math.max(best, j - i + 1);
            }
        }
        return best;
    }

    private static int seed = 34142;

    private static String lcgString(int length) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append("()".charAt(seed % 2));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(longestValidSubstringStack("(()") == 2, "(() -> 2");
        check(longestValidSubstringStack(")()())") == 4, ")()()) -> 4");
        check(longestValidSubstringStack("") == 0, "empty -> 0");
        check(longestValidSubstringStack("()(())") == 6, "()(()) -> 6");
        check(longestValidSubstringStack("()(()") == 2, "()(() -> 2");
        check(longestValidSubstringStack("((())") == 4, "((()) -> 4");
        check(longestValidSubstringStack("())") == 2, "()) -> 2");
        check(longestValidSubstringStack(")(") == 0, ")(");
        check(longestValidSubstringStack("(((") == 0, "(((");
        check(longestValidSubstringStack("()") == 2, "() -> 2");
        check(longestValidSubstringStack("()()") == 4, "()() -> 4");
        check(longestValidSubstringStack(")()(") == 2, ")()( -> 2");

        String[] hand = {"(()", ")()())", "", "()(())", "()(()", "((())", "())", ")(", "(((", "()", "()()", ")()("};
        for (String s : hand) {
            check(longestValidSubstringDp(s) == longestValidSubstringStack(s), "dp vs stack " + s);
            check(longestValidSubstringBrute(s) == longestValidSubstringStack(s), "brute vs stack " + s);
        }

        for (int trial = 0; trial < 150; trial++) {
            seed = (seed * 75 + 74) % 65537;
            String s = lcgString(seed % 25);
            int expected = longestValidSubstringBrute(s);
            check(longestValidSubstringStack(s) == expected, "stack random " + s);
            check(longestValidSubstringDp(s) == expected, "dp random " + s);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
