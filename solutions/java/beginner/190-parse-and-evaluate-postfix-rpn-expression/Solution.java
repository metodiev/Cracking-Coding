/** Task 190 - Parse and evaluate postfix (RPN) expression. (Tier: Beginner, CTCI Ch 3) */
import java.util.*;

public class Solution {
    /**
     * Tokens are separated by whitespace: integer operands and the operators + - * /.
     * Evaluation is a stack sweep — push operands, pop two for each operator. '/' is
     * integer division truncating toward zero (native Java behaviour), and division by
     * zero, unknown tokens, missing operands or leftover values throw
     * IllegalArgumentException.
     */
    static List<String> tokenizeRpn(String expression) {
        String trimmed = expression.trim();
        if (trimmed.isEmpty()) return new ArrayList<>();
        return new ArrayList<>(Arrays.asList(trimmed.split("\\s+")));
    }

    static int evalRpn(String expression) {
        Deque<Integer> stack = new ArrayDeque<>();
        for (String token : tokenizeRpn(expression)) {
            if (token.equals("+") || token.equals("-") || token.equals("*") || token.equals("/")) {
                if (stack.size() < 2) throw new IllegalArgumentException("not enough operands for " + token);
                int right = stack.pop();
                int left = stack.pop();
                switch (token) {
                    case "+":
                        stack.push(left + right);
                        break;
                    case "-":
                        stack.push(left - right);
                        break;
                    case "*":
                        stack.push(left * right);
                        break;
                    default:
                        if (right == 0) throw new IllegalArgumentException("division by zero");
                        stack.push(left / right);
                }
            } else {
                if (!token.matches("-?\\d+")) throw new IllegalArgumentException("invalid token: " + token);
                stack.push(Integer.parseInt(token));
            }
        }
        if (stack.size() != 1) {
            throw new IllegalArgumentException("expression leaves " + stack.size() + " values on the stack");
        }
        return stack.pop();
    }

    public static void main(String[] args) {
        check(tokenizeRpn("5 1 2 + 4 * + 3 -").equals(Arrays.asList("5", "1", "2", "+", "4", "*", "+", "3", "-")),
                "tokenize");
        check(tokenizeRpn("").isEmpty(), "tokenize empty");
        check(tokenizeRpn("  7   2 / ").equals(Arrays.asList("7", "2", "/")), "tokenize extra whitespace");

        check(evalRpn("3 4 +") == 7, "3 4 +");
        check(evalRpn("5 1 2 + 4 * + 3 -") == 14, "long example");
        check(evalRpn("7 2 /") == 3, "7 2 /");
        check(evalRpn("42") == 42, "single value");
        check(evalRpn("  42  ") == 42, "padded single value");
        check(evalRpn("2 3 4 * +") == 14, "2 3 4 * +");
        check(evalRpn("1 2 + 3 *") == 9, "1 2 + 3 *");
        check(evalRpn("10 3 - 4 2 / *") == 14, "mixed");
        check(evalRpn("5 1 - 1 -") == 3, "chained subtraction");
        check(evalRpn("0 5 /") == 0, "zero numerator");

        // integer division truncates toward zero
        check(evalRpn("-7 2 /") == -3, "-7 2 /");
        check(evalRpn("7 -2 /") == -3, "7 -2 /");
        check(evalRpn("-7 -2 /") == 3, "-7 -2 /");
        check(evalRpn("-1 3 /") == 0, "-1 3 /");

        for (String bad : new String[] {"1 0 /", "0 0 /", "3 4", "1 2 + 3", "+", "", "   ", "abc", "1 2 &", "1.5 2 +", "+5 1 +", "3 -"}) {
            final String expression = bad;
            checkThrows(() -> evalRpn(expression), "rejects " + expression);
        }
        System.out.println("PASS");
    }

    static void checkThrows(Runnable action, String label) {
        try {
            action.run();
        } catch (IllegalArgumentException e) {
            return;
        }
        throw new AssertionError("FAILED: expected IllegalArgumentException: " + label);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
