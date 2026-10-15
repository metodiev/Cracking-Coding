/** Task 288 - Generate all balanced parentheses strings. (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    /**
     * Open-first DFS: at each step add "(" if some pair is still unopened, then ")" if the
     * closing count is below the opening count. This emits strings in lexicographic order
     * ("(" < ")"), so results are deterministic.
     */
    static List<String> generateParentheses(int n) {
        List<String> results = new ArrayList<>();
        if (n < 0) return results;
        build(new StringBuilder(), n, 0, 0, results);
        return results;
    }

    private static void build(StringBuilder current, int n, int opens, int closes, List<String> results) {
        if (current.length() == 2 * n) {
            results.add(current.toString());
            return;
        }
        if (opens < n) {
            current.append('(');
            build(current, n, opens + 1, closes, results);
            current.setLength(current.length() - 1);
        }
        if (closes < opens) {
            current.append(')');
            build(current, n, opens, closes + 1, results);
            current.setLength(current.length() - 1);
        }
    }

    /** Independent checker: only parentheses, no negative prefix, ends at depth 0. */
    static boolean isBalanced(String text) {
        int depth = 0;
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            if (ch == '(') {
                depth++;
            } else if (ch == ')') {
                depth--;
                if (depth < 0) return false;
            } else {
                return false;
            }
        }
        return depth == 0;
    }

    /** C(0)=1, C(n+1) = C(n) * 2 * (2n + 1) / (n + 2). */
    static int catalan(int n) {
        int value = 1;
        for (int i = 0; i < n; i++) value = value * 2 * (2 * i + 1) / (i + 2);
        return value;
    }

    /** Enumerates all 2^(2n) parenthesis strings and keeps the balanced ones. */
    static List<String> bruteForceBalanced(int n) {
        List<String> results = new ArrayList<>();
        for (int mask = 0; mask < (1 << (2 * n)); mask++) {
            StringBuilder text = new StringBuilder();
            for (int bit = 0; bit < 2 * n; bit++) text.append(((mask >> bit) & 1) == 1 ? '(' : ')');
            if (isBalanced(text.toString())) results.add(text.toString());
        }
        Collections.sort(results);
        return results;
    }

    public static void main(String[] args) {
        check(generateParentheses(0).equals(List.of("")), "n=0");
        check(generateParentheses(1).equals(List.of("()")), "n=1");
        check(generateParentheses(2).equals(List.of("(())", "()()")), "n=2");
        check(generateParentheses(3).equals(List.of("((()))", "(()())", "(())()", "()(())", "()()()")), "n=3");
        check(generateParentheses(-1).isEmpty(), "negative n");

        for (int n = 0; n <= 10; n++) {
            List<String> strings = generateParentheses(n);
            check(strings.size() == catalan(n), "catalan " + n);
            for (int i = 1; i < strings.size(); i++) {
                check(strings.get(i - 1).compareTo(strings.get(i)) < 0, "lexicographic " + n);
            }
            for (String text : strings) {
                check(text.length() == 2 * n, "length " + n);
                check(isBalanced(text), "balanced " + n + " " + text);
                int opens = 0, closes = 0;
                for (int i = 0; i < text.length(); i++) {
                    if (text.charAt(i) == '(') opens++; else closes++;
                }
                check(opens == n && closes == n, "pairs " + n);
            }
            check(generateParentheses(n).equals(strings), "deterministic " + n);
        }

        for (int n = 0; n <= 7; n++) {
            check(generateParentheses(n).equals(bruteForceBalanced(n)), "oracle " + n);
        }

        check(catalan(0) == 1 && catalan(1) == 1 && catalan(2) == 2 && catalan(3) == 5, "catalan small");
        check(catalan(4) == 14 && catalan(5) == 42 && catalan(8) == 1430 && catalan(10) == 16796, "catalan known");

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
