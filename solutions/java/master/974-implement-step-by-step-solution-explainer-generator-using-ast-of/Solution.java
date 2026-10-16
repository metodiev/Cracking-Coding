/** Task 974 - Implement step-by-step solution explainer generator using AST of canonical solution.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  The canonical solution is represented as an AST of statements (assign, loop, if, return) over
 *  expressions (literal, variable, array element, binary op, equality compare). Evaluating the AST
 *  produces a numbered explanation trace: every statement appends one line rendering the node with the
 *  concrete values of the moment (`if a[2] % 2 == 0 -> true`, `assign sum = 6`), so the trace reads as a
 *  step-by-step explanation of the algorithm. A runtime failure (index out of range) aborts the trace with
 *  an `error:` line instead of emitting a bogus result.
 *  Simplified: expression sub-nodes are not traced individually and the language covers only what this
 *  canonical program needs (no functions, no nesting of loops); statement-level tracing, the replayed
 *  state and the abort-on-fault behaviour are the real ones.
 *  Verified: the explained result against an independent flat reference interpreter, the trace length
 *  against a closed-form count, concrete trace lines, a replay of the printed assignments, loop
 *  unrolling counts, the empty/all-odd boundaries, the out-of-range fault path and determinism.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final class Env {
        final Map<String, Integer> vars = new TreeMap<>();
        final Map<String, int[]> arrays = new TreeMap<>();
        int value(String name) {
            if (!vars.containsKey(name)) throw new IllegalStateException("undefined variable " + name);
            return vars.get(name);
        }
    }

    static final class Trace {
        final List<String> lines = new ArrayList<>();
        void add(String text) { lines.add((lines.size() + 1) + ": " + text); }
    }

    static abstract class Expr {
        abstract int value(Env env);
        abstract String render(Env env);
    }
    static final class Lit extends Expr {
        final int v; Lit(int v) { this.v = v; }
        int value(Env env) { return v; }
        String render(Env env) { return Integer.toString(v); }
    }
    static final class Var extends Expr {
        final String name; Var(String name) { this.name = name; }
        int value(Env env) { return env.value(name); }
        String render(Env env) { return Integer.toString(env.value(name)); }   // concrete value, not the name
    }
    static final class ArrAt extends Expr {
        final String name; final Expr index;
        ArrAt(String name, Expr index) { this.name = name; this.index = index; }
        int value(Env env) {
            int i = index.value(env);
            int[] a = env.arrays.get(name);
            if (i < 0 || i >= a.length) throw new IllegalStateException("index " + i + " out of range for " + name
                    + " of length " + a.length);
            return a[i];
        }
        String render(Env env) { return name + "[" + index.render(env) + "]"; }
    }
    static final class Bin extends Expr {
        final char op; final Expr left, right;
        Bin(char op, Expr left, Expr right) { this.op = op; this.left = left; this.right = right; }
        int value(Env env) { return op == '+' ? left.value(env) + right.value(env) : left.value(env) % right.value(env); }
        String render(Env env) { return left.render(env) + " " + op + " " + right.render(env); }
    }
    static final class Cmp extends Expr {
        final Expr left, right;
        Cmp(Expr left, Expr right) { this.left = left; this.right = right; }
        int value(Env env) { return left.value(env) == right.value(env) ? 1 : 0; }
        String render(Env env) { return left.render(env) + " == " + right.render(env); }
    }

    static abstract class Stmt { abstract void run(Env env, Trace trace); }
    static final class Assign extends Stmt {
        final String name; final Expr value;
        Assign(String name, Expr value) { this.name = name; this.value = value; }
        void run(Env env, Trace trace) {
            int v = value.value(env);
            env.vars.put(name, v);
            trace.add("assign " + name + " = " + v);
        }
    }
    static final class Loop extends Stmt {
        final String var; final int lo, hi; final List<Stmt> body;
        Loop(String var, int lo, int hi, List<Stmt> body) { this.var = var; this.lo = lo; this.hi = hi; this.body = body; }
        void run(Env env, Trace trace) {
            for (int i = lo; i < hi; i++) {
                env.vars.put(var, i);
                trace.add("iteration " + var + "=" + i);
                for (Stmt s : body) s.run(env, trace);
            }
        }
    }
    static final class If extends Stmt {
        final Expr condition; final List<Stmt> body;
        If(Expr condition, List<Stmt> body) { this.condition = condition; this.body = body; }
        void run(Env env, Trace trace) {
            boolean taken = condition.value(env) != 0;
            trace.add("if " + condition.render(env) + " -> " + taken);
            if (taken) for (Stmt s : body) s.run(env, trace);
        }
    }
    static final class Return extends Stmt {
        final Expr value; Return(Expr value) { this.value = value; }
        void run(Env env, Trace trace) { trace.add("return " + value.value(env)); }
    }

    /** Runs the program; a fault aborts the trace with an `error:` line and a null result. */
    static Env explain(List<Stmt> program, Env env, Trace trace) {
        try {
            for (Stmt s : program) s.run(env, trace);
            return env;
        } catch (RuntimeException e) {
            trace.add("error: " + e.getMessage());
            return null;
        }
    }

    static Expr even(Expr element) { return new Cmp(new Bin('%', element, new Lit(2)), new Lit(0)); }

    /** sum and count of the even elements, written as an AST over `a`. */
    static List<Stmt> evenSumProgram(int length) {
        ArrAt element = new ArrAt("a", new Var("i"));
        return List.of(
                new Assign("sum", new Lit(0)),
                new Assign("count", new Lit(0)),
                new Loop("i", 0, length, List.of(
                        new If(even(element), List.of(
                                new Assign("sum", new Bin('+', new Var("sum"), element)),
                                new Assign("count", new Bin('+', new Var("count"), new Lit(1))))))),
                new Return(new Var("sum")));
    }

    static Env envOf(int[] values) {
        Env env = new Env();
        env.arrays.put("a", values);
        return env;
    }

    /** Independent reference interpreter: the same algorithm written as plain hand-rolled Java. */
    static int[] reference(int[] a) {
        int sum = 0, count = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] % 2 == 0) { sum += a[i]; count += 1; }
        }
        return new int[]{sum, count};
    }

    /** Closed-form trace length: 2 setup assigns + iteration and if lines per element + 2 lines per even. */
    static int expectedLines(int[] a, int length) {
        int evens = 0;
        for (int i = 0; i < length; i++) if (a[i] % 2 == 0) evens++;
        return 2 + 2 * length + 2 * evens + 1;
    }

    static int countMatches(List<String> lines, String marker) {
        int n = 0;
        for (String line : lines) if (line.contains(marker)) n++;
        return n;
    }

    static Trace traceOf(int[] a) {
        Trace trace = new Trace();
        explain(evenSumProgram(a.length), envOf(a), trace);
        return trace;
    }

    public static void main(String[] args) {
        int[] a = {4, 7, 2, 9, 6, 1};
        Trace trace = traceOf(a);
        Env env = explain(evenSumProgram(a.length), envOf(a), new Trace());
        int[] expected = reference(a);
        check(env.vars.get("sum") == expected[0] && expected[0] == 12 && env.vars.get("count") == expected[1]
                        && expected[1] == 3,
                "the explained result must match the independent reference interpreter: " + env.vars.get("sum")
                        + "/" + env.vars.get("count") + " vs " + Arrays.toString(expected));

        check(trace.lines.size() == 21 && trace.lines.size() == expectedLines(a, a.length)
                        && trace.lines.get(trace.lines.size() - 1).equals("21: return 12"),
                "the trace length must match the closed-form count and end with the return: " + trace.lines.size());

        check(trace.lines.get(0).equals("1: assign sum = 0") && trace.lines.get(2).equals("3: iteration i=0")
                        && trace.lines.get(3).equals("4: if a[0] % 2 == 0 -> true")
                        && trace.lines.get(4).equals("5: assign sum = 4")
                        && trace.lines.get(7).equals("8: if a[1] % 2 == 0 -> false"),
                "explanation lines must render the concrete values of that step: " + trace.lines.get(3));

        Map<String, Integer> replayed = new TreeMap<>();
        for (String line : trace.lines) {
            String[] parts = line.split(" ");
            if (parts.length == 5 && parts[1].equals("assign")) replayed.put(parts[2], Integer.parseInt(parts[4]));
        }
        check(replayed.get("sum") == 12 && replayed.get("count") == 3 && replayed.size() == 2,
                "replaying the printed assignments must rebuild the final state: " + replayed);

        check(countMatches(trace.lines, "iteration") == a.length && countMatches(trace.lines, "if a[") == a.length
                        && countMatches(trace.lines, "assign") == 2 + 2 * 3,
                "the loop must unroll into one iteration and one condition line per element: "
                        + countMatches(trace.lines, "iteration"));

        int[] oddOnly = {1, 3, 5};
        Trace oddTrace = traceOf(oddOnly);
        check(explain(evenSumProgram(oddOnly.length), envOf(oddOnly), new Trace()).vars.get("sum") == 0
                        && oddTrace.lines.size() == expectedLines(oddOnly, oddOnly.length)
                        && countMatches(oddTrace.lines, "-> true") == 0,
                "an all-odd input must explain a zero result without any taken branch");

        int[] empty = {};
        Trace emptyTrace = traceOf(empty);
        check(traceOf(empty).lines.size() == 3
                        && explain(evenSumProgram(0), envOf(empty), new Trace()).vars.get("sum") == 0
                        && Arrays.equals(reference(empty), new int[]{0, 0})
                        && emptyTrace.lines.get(2).equals("3: return 0"),
                "the empty-array boundary must explain setup then return 0: " + emptyTrace.lines);

        Trace fault = new Trace();
        List<Stmt> tooFar = List.of(new Loop("i", 0, a.length + 1,
                List.of(new Assign("x", new ArrAt("a", new Var("i"))))));
        Env faulted = explain(tooFar, envOf(a), fault);
        check(faulted == null && fault.lines.get(fault.lines.size() - 1)
                        .equals("14: error: index 6 out of range for a of length 6")
                        && countMatches(fault.lines, "return") == 0,
                "an out-of-range access must abort the explanation with an error line: "
                        + fault.lines.get(fault.lines.size() - 1));

        int[] other = {2, 4, 6};
        Env otherEnv = explain(evenSumProgram(other.length), envOf(other), new Trace());
        check(otherEnv.vars.get("sum") == reference(other)[0] && otherEnv.vars.get("count") == 3
                        && otherEnv.vars.get("sum") == 12,
                "a second input must be explained consistently with the reference: " + otherEnv.vars);

        check(traceOf(a).lines.equals(trace.lines) && traceOf(a).lines.equals(trace.lines),
                "explaining the same program twice must produce the identical trace");

        System.out.println("input=" + Arrays.toString(a) + " steps=" + trace.lines.size() + " sum=" + env.vars.get("sum"));
        System.out.println("sample=" + trace.lines.subList(2, 6));
        System.out.println("PASS");
    }
}
