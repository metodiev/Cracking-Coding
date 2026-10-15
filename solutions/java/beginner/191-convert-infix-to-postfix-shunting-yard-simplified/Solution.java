/** Task 191 - Convert infix to postfix (shunting-yard simplified). (Tier: Beginner, CTCI Ch 3) */
import java.util.*;

public class Solution {
    public static void main(String[] args) {
        check(infixToPostfix("3 + 4 * 2").equals("3 4 2 * +"), "precedence");
        check(infixToPostfix("( 1 + 2 ) * 3").equals("1 2 + 3 *"), "parentheses");
        check(infixToPostfix("1 + 2 - 3").equals("1 2 + 3 -"), "left-associative + -");
        check(infixToPostfix("8 / 4 / 2").equals("8 4 / 2 /"), "left-associative /");
        check(infixToPostfix("2 * 3 + 4 * 5").equals("2 3 * 4 5 * +"), "mixed precedence");
        check(infixToPostfix("( 1 + 2 ) * ( 3 - 4 )").equals("1 2 + 3 4 - *"), "two groups");
        check(infixToPostfix("( ( 12 ) )").equals("12"), "nested redundant parentheses");
        check(infixToPostfix("").equals(""), "empty input");
        checkThrows("1 + 2 )", "unmatched ')'");
        checkThrows("( 1 + 2", "unmatched '('");
        checkThrows("( ( 1 ) + 2 ) )", "extra ')'");
        checkThrows("1 + a", "unknown token");
        checkThrows("1 ^ 2", "unsupported operator");
        System.out.println("PASS");
    }

    static String infixToPostfix(String expr) {
        String trimmed = expr.trim();
        String[] tokens = trimmed.isEmpty() ? new String[0] : trimmed.split("\\s+");
        List<String> output = new ArrayList<>();
        Deque<String> stack = new ArrayDeque<>();
        for (String token : tokens) {
            if (token.matches("\\d+")) {
                output.add(token);
            } else if (isOperator(token)) {
                while (!stack.isEmpty() && isOperator(stack.peek())
                        && precedence(stack.peek()) >= precedence(token)) {
                    output.add(stack.pop());
                }
                stack.push(token);
            } else if (token.equals("(")) {
                stack.push(token);
            } else if (token.equals(")")) {
                while (!stack.isEmpty() && !stack.peek().equals("(")) {
                    output.add(stack.pop());
                }
                if (stack.isEmpty()) throw new IllegalArgumentException("unmatched ')' in expression");
                stack.pop();
            } else {
                throw new IllegalArgumentException("unknown token: " + token);
            }
        }
        while (!stack.isEmpty()) {
            if (stack.peek().equals("(")) throw new IllegalArgumentException("unmatched '(' in expression");
            output.add(stack.pop());
        }
        return String.join(" ", output);
    }

    static boolean isOperator(String token) {
        return token.equals("+") || token.equals("-") || token.equals("*") || token.equals("/");
    }

    static int precedence(String op) {
        return (op.equals("*") || op.equals("/")) ? 2 : 1;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static void checkThrows(String expr, String label) {
        try {
            infixToPostfix(expr);
        } catch (IllegalArgumentException expected) {
            return;
        }
        throw new AssertionError("FAILED: expected error - " + label);
    }
}
