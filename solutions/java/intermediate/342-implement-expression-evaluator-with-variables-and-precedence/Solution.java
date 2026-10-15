/** Task 342 - Expression evaluator with variables and precedence. (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

class Token {
    static final int NUM = 0, VAR = 1, OP = 2;
    final int kind;
    final double num;
    final String text;

    private Token(int kind, double num, String text) {
        this.kind = kind;
        this.num = num;
        this.text = text;
    }

    static Token num(double value) {
        return new Token(NUM, value, null);
    }

    static Token var(String name) {
        return new Token(VAR, 0.0, name);
    }

    static Token op(String symbol) {
        return new Token(OP, 0.0, symbol);
    }
}

public class Solution {
    /**
     * Shunting-yard evaluator: + - * /, parentheses, unary minus ('~' in the RPN) and
     * variables from a map.  All arithmetic is double (no integer truncation), so
     * 100/8 == 12.5; malformed input raises IllegalArgumentException.  The brief's
     * example 3+4*2/(1-5) is 1 (3 + 8/(-4)), not 2.
     */
    static List<Token> tokenize(String expr) {
        List<Token> tokens = new ArrayList<>();
        int i = 0;
        while (i < expr.length()) {
            char ch = expr.charAt(i);
            if (ch == ' ') {
                i++;
            } else if (Character.isDigit(ch)) {
                int j = i;
                while (j < expr.length() && Character.isDigit(expr.charAt(j))) j++;
                tokens.add(Token.num(Double.parseDouble(expr.substring(i, j))));
                i = j;
            } else if (Character.isLetter(ch)) {
                int j = i;
                while (j < expr.length()
                        && (Character.isLetterOrDigit(expr.charAt(j)) || expr.charAt(j) == '_')) j++;
                tokens.add(Token.var(expr.substring(i, j)));
                i = j;
            } else if ("+-*/()".indexOf(ch) >= 0) {
                tokens.add(Token.op(String.valueOf(ch)));
                i++;
            } else {
                throw new IllegalArgumentException("unexpected character '" + ch + "'");
            }
        }
        return tokens;
    }

    static int precedence(String op) {
        if (op.equals("~")) return 4;                      // unary minus, right-associative
        if (op.equals("*") || op.equals("/")) return 2;
        return 1;
    }

    static List<Token> toRpn(List<Token> tokens) {
        List<Token> output = new ArrayList<>();
        Deque<Token> ops = new ArrayDeque<>();
        boolean expectOperand = true;
        for (Token token : tokens) {
            if (token.kind == Token.NUM) {
                if (!expectOperand) throw new IllegalArgumentException("expected an operator before " + token.num);
                output.add(token);
                expectOperand = false;
            } else if (token.kind == Token.VAR) {
                if (!expectOperand) throw new IllegalArgumentException("expected an operator before " + token.text);
                output.add(token);
                expectOperand = false;
            } else if (token.text.equals("(")) {
                if (!expectOperand) throw new IllegalArgumentException("implicit multiplication is not supported");
                ops.push(token);
            } else if (token.text.equals(")")) {
                if (expectOperand) throw new IllegalArgumentException("empty or dangling parenthesis group");
                while (!ops.isEmpty() && !ops.peek().text.equals("(")) output.add(ops.pop());
                if (ops.isEmpty()) throw new IllegalArgumentException("unbalanced ')'");
                ops.pop();
                expectOperand = false;
            } else if (expectOperand) {
                if (token.text.equals("-")) {
                    ops.push(Token.op("~"));
                } else if (!token.text.equals("+")) {      // unary plus is a no-op
                    throw new IllegalArgumentException("operator '" + token.text + "' has no left operand");
                }
            } else {
                int prec = precedence(token.text);
                while (!ops.isEmpty()) {
                    Token top = ops.peek();
                    if (top.text.equals("(")) break;
                    int topPrec = precedence(top.text);
                    if (topPrec > prec || (topPrec == prec && !top.text.equals("~"))) output.add(ops.pop());
                    else break;
                }
                ops.push(token);
                expectOperand = true;
            }
        }
        if (expectOperand && !tokens.isEmpty()) {
            throw new IllegalArgumentException("dangling operator at end of expression");
        }
        while (!ops.isEmpty()) {
            Token top = ops.pop();
            if (top.text.equals("(")) throw new IllegalArgumentException("unbalanced '('");
            output.add(top);
        }
        return output;
    }

    static double evaluateRpn(List<Token> rpn, Map<String, Double> variables) {
        Deque<Double> stack = new ArrayDeque<>();
        for (Token token : rpn) {
            if (token.kind == Token.NUM) {
                stack.push(token.num);
            } else if (token.kind == Token.VAR) {
                Double value = variables.get(token.text);
                if (value == null) throw new IllegalArgumentException("unknown variable '" + token.text + "'");
                stack.push(value);
            } else if (token.text.equals("~")) {
                if (stack.isEmpty()) throw new IllegalArgumentException("malformed expression");
                stack.push(-stack.pop());
            } else {
                if (stack.size() < 2) throw new IllegalArgumentException("malformed expression");
                double b = stack.pop();
                double a = stack.pop();
                if (token.text.equals("+")) stack.push(a + b);
                else if (token.text.equals("-")) stack.push(a - b);
                else if (token.text.equals("*")) stack.push(a * b);
                else {
                    if (b == 0.0) throw new IllegalArgumentException("division by zero");
                    stack.push(a / b);
                }
            }
        }
        if (stack.size() != 1) throw new IllegalArgumentException("malformed expression");
        return stack.pop();
    }

    static double evaluate(String expr, Map<String, Double> variables) {
        List<Token> tokens = tokenize(expr);
        if (tokens.isEmpty()) throw new IllegalArgumentException("empty expression");
        return evaluateRpn(toRpn(tokens), variables);
    }

    static Map<String, Double> map(Object... pairs) {
        Map<String, Double> out = new HashMap<>();
        for (int i = 0; i < pairs.length; i += 2) out.put((String) pairs[i], (Double) pairs[i + 1]);
        return out;
    }

    static final Map<String, Double> NO_VARS = new HashMap<>();

    static void expectIllegalArgument(String expr, Map<String, Double> variables) {
        try {
            evaluate(expr, variables);
        } catch (IllegalArgumentException expected) {
            return;
        }
        throw new AssertionError("expected IllegalArgumentException for " + expr);
    }

    public static void main(String[] args) {
        // brief examples - the first one is 1, not 2
        check(evaluate("3+4*2/(1-5)", NO_VARS) == 1.0, "3+4*2/(1-5) = 1");
        check(evaluate("x+y*2", map("x", 3.0, "y", 4.0)) == 11.0, "x+y*2 = 11");
        check(evaluate("(1+2)*(3+4)", NO_VARS) == 21.0, "(1+2)*(3+4) = 21");

        // values cross-checked against CPython's expression parser
        check(evaluate("1+2*3-4/8", NO_VARS) == 6.5, "6.5");
        check(evaluate("2*(3+4)-5", NO_VARS) == 9.0, "9");
        check(evaluate("100/8", NO_VARS) == 12.5, "12.5");
        check(evaluate("-(3+4)*2", NO_VARS) == -14.0, "-14");
        check(evaluate("3*-2", NO_VARS) == -6.0, "-6");
        check(evaluate("2*-3+4", NO_VARS) == -2.0, "-2");
        check(evaluate("10-4-3", NO_VARS) == 3.0, "left associative");
        check(evaluate("16/4/2", NO_VARS) == 2.0, "(16/4)/2");
        check(evaluate("2*(1+2*(3-1))", NO_VARS) == 10.0, "10");
        check(evaluate("((7))", NO_VARS) == 7.0, "7");
        check(evaluate("8/(2+2)", NO_VARS) == 2.0, "2");
        check(evaluate("--5", NO_VARS) == 5.0, "chained unary minus");
        check(evaluate("-(-5)", NO_VARS) == 5.0, "5");
        check(evaluate("3--2", NO_VARS) == 5.0, "binary minus then unary minus");
        check(evaluate("2+3*4-6/3", NO_VARS) == 12.0, "12");
        check(evaluate("((1+2)*(3+4))/3", NO_VARS) == 7.0, "7");
        check(evaluate("1+((2+3)*4)", NO_VARS) == 21.0, "21");
        check(evaluate("7/2", NO_VARS) == 3.5, "3.5");
        check(evaluate("(8-3)*(2+1)/5", NO_VARS) == 3.0, "3");
        check(evaluate(" 1 + 2 ", NO_VARS) == 3.0, "whitespace ignored");
        check(evaluate("2*-5+20/4", NO_VARS) == -5.0, "-5");
        check(evaluate("x*y+z", map("x", 3.0, "y", 4.0, "z", 5.0)) == 17.0, "17");
        check(evaluate("-x*2", map("x", 3.0)) == -6.0, "-6");
        check(evaluate("+x", map("x", 7.0)) == 7.0, "unary plus");
        check(evaluate("price*quantity", map("price", 2.5, "quantity", 4.0)) == 10.0, "10");

        String[] bad = {"", "1+", "*3", "1**2", "(1+2", "1+2)", "()", "( )", "1 2", "3x",
                        "1(2)", "&", "1&2", "4/0", "0/0", "(1+)"};
        for (String expr : bad) expectIllegalArgument(expr, NO_VARS);
        expectIllegalArgument("x+1", NO_VARS);               // unknown variable
        expectIllegalArgument("sum", map("total", 1.0));
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
