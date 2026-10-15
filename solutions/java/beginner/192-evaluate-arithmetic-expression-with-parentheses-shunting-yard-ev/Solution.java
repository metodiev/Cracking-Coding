/** Task 192 - Evaluate arithmetic expression with +,-,*,/, parentheses (shunting-yard + eval). (Tier: Beginner, CTCI Ch 3) */
import java.util.*;

public class Solution {
    public static void main(String[] args) {
        check(evaluate("3 + 4 * 2") == 11, "precedence");
        check(evaluate("( 1 + 2 ) * 3") == 9, "parentheses");
        check(evaluate("10 / 3") == 3, "integer division truncates");
        check(evaluate("2 * ( 3 + 4 ) - 5") == 9, "mixed expression");
        check(evaluate("( 0 - 7 ) / 2") == -3, "truncation toward zero");
        check(evaluate("100 / 7 / 2") == 7, "left-associative division");
        check(evaluate("7") == 7, "single number");
        check(evaluate("( ( 2 + 3 ) * ( 4 - 1 ) )") == 15, "nested groupings");
        checkThrows("10 / 0", "division by zero");
        checkThrows("( 1 + 2", "unmatched '('");
        checkThrows("1 + 2 )", "unmatched ')'");
        checkThrows("( 1 + 2 ) )", "extra ')'");
        checkThrows("1 +", "operator without operand");
        checkThrows("", "empty input");
        checkThrows("3 @ 4", "unknown token");
        System.out.println("PASS");
    }

    static int evaluate(String expr) {
        return evaluatePostfix(infixToPostfix(tokenize(expr)));
    }

    static List<String> tokenize(String expr) {
        String trimmed = expr.trim();
        List<String> tokens = trimmed.isEmpty() ? new ArrayList<>() : new ArrayList<>(Arrays.asList(trimmed.split("\\s+")));
        for (String token : tokens) {
            if (!token.matches("\\d+") && !isOperator(token) && !token.equals("(") && !token.equals(")")) {
                throw new IllegalArgumentException("unknown token: " + token);
            }
        }
        return tokens;
    }

    static List<String> infixToPostfix(List<String> tokens) {
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
                while (!stack.isEmpty() && !stack.peek().equals("(")) output.add(stack.pop());
                if (stack.isEmpty()) throw new IllegalArgumentException("unmatched ')' in expression");
                stack.pop();
            }
        }
        while (!stack.isEmpty()) {
            if (stack.peek().equals("(")) throw new IllegalArgumentException("unmatched '(' in expression");
            output.add(stack.pop());
        }
        return output;
    }

    static int evaluatePostfix(List<String> tokens) {
        Deque<Integer> stack = new ArrayDeque<>();
        for (String token : tokens) {
            if (isOperator(token)) {
                if (stack.size() < 2) throw new IllegalArgumentException("malformed expression: operator without operands");
                int right = stack.pop();
                int left = stack.pop();
                stack.push(apply(token, left, right));
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        if (stack.size() != 1) throw new IllegalArgumentException("malformed expression: stray operands");
        return stack.pop();
    }

    static int apply(String op, int left, int right) {
        switch (op) {
            case "+": return left + right;
            case "-": return left - right;
            case "*": return left * right;
            default:
                if (right == 0) throw new IllegalArgumentException("division by zero");
                return left / right;                       // Java int division truncates toward zero
        }
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
            evaluate(expr);
        } catch (IllegalArgumentException expected) {
            return;
        }
        throw new AssertionError("FAILED: expected error - " + label);
    }
}
