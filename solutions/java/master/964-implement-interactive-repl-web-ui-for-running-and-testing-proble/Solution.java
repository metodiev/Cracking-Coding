/** Task 964 - Implement interactive REPL/web UI for running and testing problems in-browser (sandboxed).
 *  (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 *  In-memory REPL for a tiny integer expression language: literals, variables, + - * / %, parentheses and
 *  the commands :help, :vars, :reset, :quit, plus `let name = expr` bindings. Every evaluation runs under
 *  a sandbox: a step budget bounds parsing and evaluation work, a transcript quota bounds how much output
 *  the session may produce, and divisions by zero or undefined names raise typed errors instead of
 *  crashing the session.
 *  Simplified: the "web UI" is the session/transcript protocol rendered as text instead of a browser, and
 *  the sandbox is a step counter rather than a worker thread; the protocol, quotas and error taxonomy are
 *  the real ones.
 *  Verified: results match an independently written shunting-yard evaluator on 30 generated expressions;
 *  precedence, integer division and modulo are correct; bindings, :vars and :reset behave; undefined names
 *  and division by zero produce errors without corrupting state; the step budget and transcript quota are
 *  enforced and leave the session usable; replaying a script reproduces the transcript digest byte for
 *  byte; empty input is ignored; the transcript size matches the number of accepted inputs.
 */
import java.util.*;

public class Solution {
    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }
    static final long FNV_OFFSET = 0xcbf29ce484222325L, FNV_PRIME = 0x100000001b3L;
    static long fnvText(long h, String s) {
        for (int i = 0; i < s.length(); i++) { h ^= s.charAt(i); h *= FNV_PRIME; }
        return h;
    }
    static String hex(long v) { return String.format("%016x", v); }
    static final class SandboxError extends RuntimeException {
        SandboxError(String message) { super(message); }
    }
    static List<String> tokenize(String source) {
        List<String> tokens = new ArrayList<>(); int i = 0;
        while (i < source.length()) {
            char c = source.charAt(i);
            if (Character.isWhitespace(c)) { i++; continue; }
            if (Character.isDigit(c)) {
                int start = i;
                while (i < source.length() && Character.isDigit(source.charAt(i))) i++;
                tokens.add(source.substring(start, i)); continue;
            }
            if (Character.isLetter(c) || c == '_') {
                int start = i;
                while (i < source.length() && (Character.isLetterOrDigit(source.charAt(i)) || source.charAt(i) == '_')) i++;
                tokens.add(source.substring(start, i)); continue;
            }
            if ("+-*/%()=".indexOf(c) < 0) throw new SandboxError("syntax error: unexpected character '" + c + "'");
            tokens.add(String.valueOf(c)); i++;
        }
        return tokens;
    }
    /** Recursive descent over the token list, charging one step per token consumed. */
    static final class Evaluator {
        final List<String> tokens; final Map<String, Long> variables; final int budget; int position, steps;
        Evaluator(List<String> tokens, Map<String, Long> variables, int budget) {
            this.tokens = tokens; this.variables = variables; this.budget = budget;
        }
        void charge() {
            steps++;
            if (steps > budget) throw new SandboxError("step budget exceeded");
        }
        String peek() { return position < tokens.size() ? tokens.get(position) : null; }
        String next() {
            if (position >= tokens.size()) throw new SandboxError("syntax error: unexpected end of input");
            return tokens.get(position++);
        }
        long evaluate() {
            long value = expression();
            if (position != tokens.size()) throw new SandboxError("syntax error: trailing token " + tokens.get(position));
            return value;
        }
        long expression() {
            long value = term();
            while (true) {
                String token = peek();
                if ("+".equals(token) || "-".equals(token)) {
                    charge(); next(); long right = term();
                    value = token.equals("+") ? value + right : value - right;
                } else return value;
            }
        }
        long term() {
            long value = factor();
            while (true) {
                String token = peek();
                if ("*".equals(token) || "/".equals(token) || "%".equals(token)) {
                    charge(); next(); long right = factor();
                    if ((token.equals("/") || token.equals("%")) && right == 0)
                        throw new SandboxError("runtime error: division by zero");
                    value = token.equals("*") ? value * right : token.equals("/") ? value / right : value % right;
                } else return value;
            }
        }
        long factor() {
            charge(); String token = next();
            if (token.equals("(")) {
                long value = expression();
                if (!")".equals(next())) throw new SandboxError("syntax error: missing closing parenthesis");
                return value;
            }
            if (token.equals("-")) return -factor();
            if (Character.isDigit(token.charAt(0))) return Long.parseLong(token);
            Long bound = variables.get(token);
            if (bound == null) throw new SandboxError("runtime error: undefined variable " + token);
            return bound;
        }
    }
    /** Independent evaluator: shunting-yard to RPN, then a stack machine. Literals only. */
    static long rpnEvaluate(List<String> tokens) {
        List<String> output = new ArrayList<>(); Deque<String> operators = new ArrayDeque<>();
        Map<String, Integer> precedence = Map.of("+", 1, "-", 1, "*", 2, "/", 2, "%", 2);
        for (String token : tokens) {
            if (Character.isDigit(token.charAt(0))) output.add(token);
            else if (token.equals("(")) operators.push(token);
            else if (token.equals(")")) {
                while (!operators.isEmpty() && !operators.peek().equals("(")) output.add(operators.pop());
                operators.pop();
            } else {
                while (!operators.isEmpty() && !operators.peek().equals("(") && precedence.get(operators.peek()) >= precedence.get(token)) output.add(operators.pop());
                operators.push(token);
            }
        }
        while (!operators.isEmpty()) output.add(operators.pop());
        Deque<Long> stack = new ArrayDeque<>();
        for (String token : output) {
            if (Character.isDigit(token.charAt(0))) { stack.push(Long.parseLong(token)); continue; }
            long right = stack.pop(), left = stack.pop();
            stack.push(token.equals("+") ? left + right : token.equals("-") ? left - right : token.equals("*") ? left * right : token.equals("/") ? left / right : left % right);
        }
        return stack.pop();
    }
    static final class Session {
        static final int STEP_BUDGET = 300, TRANSCRIPT_QUOTA = 8;
        final Map<String, Long> variables = new TreeMap<>(); final List<String> transcript = new ArrayList<>();
        boolean quit;
        String submit(String line) {
            String trimmed = line.trim();
            if (trimmed.isEmpty()) return null;                       // ignored, leaves no trace
            if (transcript.size() >= TRANSCRIPT_QUOTA) return "error: transcript quota exceeded";
            if (quit) return "error: session closed";
            String response;
            if (trimmed.startsWith(":")) {
                switch (trimmed) {
                    case ":help": response = "commands: :help :vars :reset :quit, else let name = expr"; break;
                    case ":vars": response = "vars: " + variables; break;
                    case ":reset": variables.clear(); response = "reset"; break;
                    case ":quit": quit = true; response = "bye"; break;
                    default: response = "error: unknown command " + trimmed;
                }
            } else if (trimmed.startsWith("let ")) {
                String rest = trimmed.substring(4); int equals = rest.indexOf('=');
                if (equals <= 0 || !rest.substring(0, equals).trim().matches("[a-z_][a-z0-9_]*"))
                    response = "error: syntax error in let binding";
                else {
                    String name = rest.substring(0, equals).trim();
                    try {
                        long value = evaluate(rest.substring(equals + 1), variables); variables.put(name, value);
                        response = name + " = " + value;
                    } catch (SandboxError error) {
                        response = "error: " + error.getMessage();
                    }
                }
            } else {
                try {
                    response = String.valueOf(evaluate(trimmed, variables));
                } catch (SandboxError error) {
                    response = "error: " + error.getMessage();
                }
            }
            transcript.add(trimmed + " -> " + response); return response;
        }
        String digest() {
            long h = fnvText(FNV_OFFSET, "session");
            for (String line : transcript) h = fnvText(h, line + "|");
            for (Map.Entry<String, Long> entry : variables.entrySet()) h = fnvText(h, entry.getKey() + "=" + entry.getValue());
            return hex(h);
        }
    }
    static long evaluate(String source, Map<String, Long> variables) {
        return new Evaluator(tokenize(source), variables, Session.STEP_BUDGET).evaluate();
    }
    static List<String> script() {
        return List.of("2 + 3 * 4", "let x = 10", "x % 3", ":vars", "x - 15", "7 / 2", "(2 + 3) * 4", ":unknown", "", "   ", "999999", ":reset", ":vars");
    }
    public static void main(String[] args) {
        Session session = new Session();
        for (String line : script()) session.submit(line);
        // 1. precedence, associativity, integer division and modulo
        Map<String, Long> empty = new TreeMap<>();
        check(evaluate("2 + 3 * 4", empty) == 14 && evaluate("(2 + 3) * 4", empty) == 20 && evaluate("7 / 2", empty) == 3 && evaluate("10 % 3", empty) == 1 && evaluate("10 - 3 - 2", empty) == 5 && evaluate("-4 + 1", empty) == -3, "precedence, associativity, integer division and modulo are correct");
        // 2. independent shunting-yard evaluator agrees on 30 generated expressions
        boolean oracleOk = true; StringBuilder source = new StringBuilder();
        for (int i = 0; i < 30; i++) {
            String expression = i % 2 == 0 ? (i % 6) + " + " + (i % 5) + " * (" + (i % 4) + " + " + (i % 3) + ")"
                    : "(" + (i % 9) + " - " + (i % 4) + ") * " + (i % 3 + 1) + " % 7";
            source.setLength(0); source.append(expression);
            oracleOk &= evaluate(expression, empty) == rpnEvaluate(tokenize(expression));
        }
        check(oracleOk, "30 generated expressions match the shunting-yard RPN oracle");
        // 3. bindings, :vars, :reset and the undefined-name error
        Session bindings = new Session();
        check(bindings.submit("let a = 5").equals("a = 5") && bindings.submit("a * 3").equals("15") && bindings.submit(":vars").equals("vars: {a=5}") && bindings.submit("b + 1").equals("error: runtime error: undefined variable b") && bindings.submit(":reset").equals("reset") && bindings.submit(":vars").equals("vars: {}"), "let bindings, :vars, :reset and undefined-variable errors behave as specified");
        // 4. the sandbox step budget kills runaway input but leaves the session usable
        Session budget = new Session(); StringBuilder nested = new StringBuilder();
        for (int i = 0; i < 400; i++) nested.append('('); nested.append(1); for (int i = 0; i < 400; i++) nested.append(')');
        String killed = budget.submit(nested.toString());
        check(killed.equals("error: step budget exceeded") && budget.submit("1 + 1").equals("2"), "deeply nested input hits the step budget and the next command still works");
        // 5. the transcript quota stops runaway output
        Session quota = new Session(); String last = "";
        for (int i = 0; i < 9; i++) last = quota.submit(String.valueOf(i));
        check(last.equals("error: transcript quota exceeded") && quota.transcript.size() == Session.TRANSCRIPT_QUOTA && quota.submit("5").equals("error: transcript quota exceeded"), "the transcript quota caps the session at " + Session.TRANSCRIPT_QUOTA + " entries");
        // 6. replaying the same script reproduces the transcript digest
        Session replay = new Session();
        for (String line : script()) replay.submit(line);
        check(replay.digest().equals(session.digest()) && replay.transcript.equals(session.transcript), "replaying the script reproduces the transcript and its digest");
        // 7. boundaries: empty and whitespace input are ignored, division by zero is an error
        Session boundary = new Session();
        check(boundary.submit("") == null && boundary.submit("   ") == null && boundary.transcript.isEmpty() && boundary.submit("1 / 0").equals("error: runtime error: division by zero") && boundary.submit("let z = 1 / 0").equals("error: runtime error: division by zero") && !boundary.variables.containsKey("z") && boundary.submit("8").equals("8"), "empty input is ignored, division by zero errors without binding anything");
        // 8. accounting oracle: transcript entries equal accepted inputs, variables match an independent map
        Map<String, Long> expected = new TreeMap<>(); int accepted = 0;
        for (String line : script()) {
            String trimmed = line.trim();
            if (trimmed.isEmpty() || accepted >= Session.TRANSCRIPT_QUOTA) continue;
            accepted++;
            if (trimmed.equals("let x = 10")) expected.put("x", 10L);
            if (trimmed.equals(":reset")) expected.clear();
        }
        check(session.transcript.size() == accepted && session.variables.equals(expected) && session.transcript.get(0).equals("2 + 3 * 4 -> 14"), "transcript accounting matches the accepted inputs and the final variable map");
        System.out.println("transcript entries: " + session.transcript.size() + ", digest: " + session.digest());
        System.out.println("first three: " + session.transcript.subList(0, Math.min(3, session.transcript.size())));
        System.out.println("quota session entries: " + quota.transcript.size() + ", budget reply: " + killed);
        System.out.println("PASS");
    }
}
