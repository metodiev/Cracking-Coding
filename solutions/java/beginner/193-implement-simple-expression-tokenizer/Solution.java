/** Task 193 - Implement simple expression tokenizer. (Tier: Beginner, CTCI Ch 3) */
import java.util.*;

public class Solution {
    static final class Token {
        final String type;
        final Object value;

        Token(String type, Object value) {
            this.type = type;
            this.value = value;
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof Token)) return false;
            Token token = (Token) other;
            return type.equals(token.type) && value.equals(token.value);
        }

        @Override
        public int hashCode() {
            return type.hashCode() * 31 + value.hashCode();
        }

        @Override
        public String toString() {
            return type + "(" + value + ")";
        }
    }

    public static void main(String[] args) {
        List<Token> tokens = tokenize("12 + (3*4)");
        check(tokens.equals(Arrays.asList(
                new Token("INT", 12), new Token("OP", "+"), new Token("LPAREN", "("),
                new Token("INT", 3), new Token("OP", "*"), new Token("INT", 4), new Token("RPAREN", ")"))),
                "exact token stream for '12 + (3*4)'");
        check(tokens.size() == 7, "token count");
        check(tokens.get(0).type.equals("INT") && tokens.get(0).value.equals(12), "INT token carries int value");

        check(tokenize("1+2").equals(Arrays.asList(
                new Token("INT", 1), new Token("OP", "+"), new Token("INT", 2))), "no whitespace");
        check(tokenize("  7  ").equals(Collections.singletonList(new Token("INT", 7))), "surrounding whitespace");
        check(tokenize("").isEmpty(), "empty input");
        check(tokenize(" \t\n ").isEmpty(), "blank input");
        check(tokenize("10/2-3").equals(Arrays.asList(
                new Token("INT", 10), new Token("OP", "/"), new Token("INT", 2),
                new Token("OP", "-"), new Token("INT", 3))), "multi-digit numbers");

        checkThrows("12.5", "'.'", "index 2");
        checkThrows("a+b", "'a'", "index 0");
        checkThrows("1 $ 2", "'$'", "index 2");
        checkThrows("#", "'#'", "index 0");
        System.out.println("PASS");
    }

    static List<Token> tokenize(String expr) {
        List<Token> tokens = new ArrayList<>();
        int index = 0;
        while (index < expr.length()) {
            char ch = expr.charAt(index);
            if (Character.isWhitespace(ch)) {
                index++;
            } else if (Character.isDigit(ch)) {
                int end = index;
                while (end < expr.length() && Character.isDigit(expr.charAt(end))) end++;
                tokens.add(new Token("INT", Integer.parseInt(expr.substring(index, end))));
                index = end;
            } else if ("+-*/".indexOf(ch) >= 0) {
                tokens.add(new Token("OP", String.valueOf(ch)));
                index++;
            } else if (ch == '(') {
                tokens.add(new Token("LPAREN", "("));
                index++;
            } else if (ch == ')') {
                tokens.add(new Token("RPAREN", ")"));
                index++;
            } else {
                throw new IllegalArgumentException("unknown character '" + ch + "' at index " + index);
            }
        }
        return tokens;
    }

    static void checkThrows(String expr, String charText, String indexText) {
        try {
            tokenize(expr);
        } catch (IllegalArgumentException exc) {
            if (!exc.getMessage().contains(charText) || !exc.getMessage().contains(indexText)) {
                throw new AssertionError("FAILED: bad message '" + exc.getMessage() + "' for " + expr);
            }
            return;
        }
        throw new AssertionError("FAILED: expected error for " + expr);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
