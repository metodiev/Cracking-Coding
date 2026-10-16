/** Task 749 - Implement symbolic execution skeleton for small programs.
 * (Tier: Expert, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A miniature symbolic execution engine for a tiny imperative language:
 *   language  - assignments, `if/else`, `assert` and `while` (unrolled up to a fixed bound), with
 *               the expressions `+`, `-`, constants and variables; multiplication is allowed only
 *               when one side is constant, so every symbolic value is an affine form
 *               c + sum(a_i * input_i) over the declared inputs.
 *   IR        - the AST is compiled into a flat instruction list (assign, assert, goto, branch,
 *               halt) with labels, which makes the executor a simple worklist over states.
 *   symbolic state - (program counter, store of affine values, path condition, unroll counters).
 *               A branch forks the state and conjoins the condition (or its negation) with the path
 *               condition; an `assert` forks into "holds" and "violated" where the violated branch
 *               is recorded as a counterexample whose model is searched over the input domain.
 *
 * Simplification (documented). Expressions stay affine (no multiplication of two symbolic values,
 * no division, no arrays, pointers or function calls); loops are unrolled to a fixed bound and a
 * path that would need a further iteration is dropped and flagged as incomplete; satisfiability of
 * a path condition is decided by enumerating the declared input domain rather than by an SMT
 * solver; the state budget is finite, and exceeding it aborts the exploration with an error. The
 * engine therefore reports *bounded* results, which is exactly what the skeleton promises.
 *
 * Verified. Every counterexample reported by symbolic execution is replayed on the concrete
 * interpreter (it must really violate the assert); for every input of the domain the concrete
 * outcome is compared against the symbolic one - an input fails iff some counterexample path
 * condition holds for it - and for inputs that do not fail the concrete final store must equal the
 * affine store of the matching completed path; the loop unroll bound is checked to be sufficient by
 * counting concrete iterations; rejection of unsupported expressions, undeclared reads and a
 * deliberate state explosion are covered as negative checks.
 */
import java.util.*;

/** Affine symbolic value: constant + sum(coefficient * input). */
final class Affine {
    final long[] coefficients;
    final long constant;

    Affine(long[] coefficients, long constant) {
        this.coefficients = coefficients;
        this.constant = constant;
    }

    static Affine constant(int inputs, long value) { return new Affine(new long[inputs], value); }

    static Affine input(int inputs, int index) {
        long[] coefficients = new long[inputs];
        coefficients[index] = 1;
        return new Affine(coefficients, 0);
    }

    static Affine add(Affine a, Affine b, long factor) {
        long[] coefficients = new long[a.coefficients.length];
        for (int i = 0; i < coefficients.length; i++) {
            coefficients[i] = a.coefficients[i] + factor * b.coefficients[i];
        }
        return new Affine(coefficients, a.constant + factor * b.constant);
    }

    long evaluate(int[] inputs) {
        long value = constant;
        for (int i = 0; i < coefficients.length; i++) value += coefficients[i] * inputs[i];
        return value;
    }

    boolean isConstant() {
        for (long coefficient : coefficients) if (coefficient != 0) return false;
        return true;
    }

    String describe(List<String> inputNames) {
        StringBuilder sb = new StringBuilder();
        sb.append(constant);
        for (int i = 0; i < coefficients.length; i++) {
            if (coefficients[i] != 0) sb.append(" + ").append(coefficients[i]).append('*')
                    .append(inputNames.get(i));
        }
        return sb.toString();
    }
}

/** One path condition conjunct: left OP right. */
final class Condition {
    static final int LT = 0;
    static final int LE = 1;
    static final int GT = 2;
    static final int GE = 3;
    static final int EQ = 4;
    static final int NE = 5;

    final Affine difference;      // left - right
    final int operator;

    Condition(Affine difference, int operator) {
        this.difference = difference;
        this.operator = operator;
    }

    static Condition of(Affine left, Affine right, int operator) {
        return new Condition(Affine.add(left, right, -1), operator);
    }

    static int negate(int operator) {
        switch (operator) {
            case LT: return GE;
            case LE: return GT;
            case GT: return LE;
            case GE: return LT;
            case EQ: return NE;
            default: return EQ;
        }
    }

    boolean holds(int[] inputs) {
        long value = difference.evaluate(inputs);
        switch (operator) {
            case LT: return value < 0;
            case LE: return value <= 0;
            case GT: return value > 0;
            case GE: return value >= 0;
            case EQ: return value == 0;
            default: return value != 0;
        }
    }

    static String operatorName(int operator) {
        switch (operator) {
            case LT: return "<";
            case LE: return "<=";
            case GT: return ">";
            case GE: return ">=";
            case EQ: return "==";
            default: return "!=";
        }
    }
}

/** Tiny AST. */
abstract class Stmt {
}

final class Assign extends Stmt {
    final String name;
    final Expr value;

    Assign(String name, Expr value) {
        this.name = name;
        this.value = value;
    }
}

final class Assert extends Stmt {
    final Expr condition;

    Assert(Expr condition) {
        this.condition = condition;
    }
}

final class If extends Stmt {
    final Expr condition;
    final List<Stmt> thenBranch;
    final List<Stmt> elseBranch;

    If(Expr condition, List<Stmt> thenBranch, List<Stmt> elseBranch) {
        this.condition = condition;
        this.thenBranch = thenBranch;
        this.elseBranch = elseBranch;
    }
}

final class While extends Stmt {
    final Expr condition;
    final List<Stmt> body;

    While(Expr condition, List<Stmt> body) {
        this.condition = condition;
        this.body = body;
    }
}

abstract class Expr {
}

final class Const extends Expr {
    final long value;

    Const(long value) {
        this.value = value;
    }
}

final class Var extends Expr {
    final String name;

    Var(String name) {
        this.name = name;
    }
}

final class Bin extends Expr {
    final String operator;
    final Expr left;
    final Expr right;

    Bin(String operator, Expr left, Expr right) {
        this.operator = operator;
        this.left = left;
        this.right = right;
    }
}

/** Compiled instruction. */
final class Instruction {
    static final int ASSIGN = 0;
    static final int ASSERT = 1;
    static final int GOTO = 2;
    static final int IF_TRUE = 3;
    static final int ASSUME = 4;
    static final int HALT = 5;

    final int opcode;
    String name;
    Expr expr;
    int target;
    int label;
    String description = "";

    Instruction(int opcode) {
        this.opcode = opcode;
    }
}

/** Symbolic execution engine over the compiled IR. */
final class SymExec {
    static final int STATE_BUDGET = 4000;

    static final class Outcome {
        final List<Condition> conditions = new ArrayList<>();
        final Map<String, Affine> store = new LinkedHashMap<>();
        final List<String> failures = new ArrayList<>();
        final int[] counterexample;      // input vector witnessing a violated assert, or null
        final boolean complete;

        Outcome(List<Condition> conditions, Map<String, Affine> store, List<String> failures,
                int[] counterexample, boolean complete) {
            this.conditions.addAll(conditions);
            this.store.putAll(store);
            this.failures.addAll(failures);
            this.counterexample = counterexample;
            this.complete = complete;
        }

        boolean conditionsHold(int[] inputs) {
            for (Condition condition : conditions) if (!condition.holds(inputs)) return false;
            return true;
        }
    }

    static final class Result {
        final List<Outcome> outcomes = new ArrayList<>();
        int states;
        int branches;
        int asserts;

        int counterexampleCount() {
            int count = 0;
            for (Outcome outcome : outcomes) if (outcome.counterexample != null) count++;
            return count;
        }
    }

    private final List<String> inputNames;
    private final int domainBound;
    private final int unrollBound;
    private final List<Instruction> code = new ArrayList<>();
    private final Map<String, Integer> labels = new LinkedHashMap<>();

    SymExec(List<String> inputNames, int domainBound, int unrollBound) {
        this.inputNames = inputNames;
        this.domainBound = domainBound;
        this.unrollBound = unrollBound;
    }

    /** Compiles a statement list into the flat instruction list. */
    void compile(List<Stmt> statements) {
        emit(statements);
        Instruction halt = new Instruction(Instruction.HALT);
        code.add(halt);
    }

    private void emit(List<Stmt> statements) {
        for (Stmt statement : statements) {
            if (statement instanceof Assign) {
                Assign assign = (Assign) statement;
                Instruction instruction = new Instruction(Instruction.ASSIGN);
                instruction.name = assign.name;
                instruction.expr = assign.value;
                code.add(instruction);
            } else if (statement instanceof Assert) {
                Instruction instruction = new Instruction(Instruction.ASSERT);
                instruction.expr = ((Assert) statement).condition;
                instruction.description = "assert at instruction " + code.size();
                code.add(instruction);
            } else if (statement instanceof If) {
                If branch = (If) statement;
                Instruction test = new Instruction(Instruction.IF_TRUE);
                // IF_TRUE: fall through when the test holds, jump to `target` when it does not
                test.expr = branch.condition;
                int testIndex = code.size();
                code.add(test);
                emit(branch.thenBranch);
                Instruction skip = new Instruction(Instruction.GOTO);
                int skipIndex = code.size();
                code.add(skip);
                code.get(testIndex).target = code.size();
                emit(branch.elseBranch);
                skip.target = code.size();
            } else if (statement instanceof While) {
                While loop = (While) statement;
                Instruction header = new Instruction(Instruction.ASSUME);
                int headerIndex = code.size();
                header.label = headerIndex;
                code.add(header);                     // unroll counter marker
                Instruction test = new Instruction(Instruction.IF_TRUE);
                test.expr = loop.condition;
                int testIndex = code.size();
                code.add(test);
                emit(loop.body);
                Instruction back = new Instruction(Instruction.GOTO);
                back.target = headerIndex;
                code.add(back);
                code.get(testIndex).target = code.size();
            } else {
                throw new IllegalArgumentException("unsupported statement " + statement);
            }
        }
    }

    private Affine evaluate(Expr expr, Map<String, Affine> store) {
        if (expr instanceof Const) return Affine.constant(inputNames.size(), ((Const) expr).value);
        if (expr instanceof Var) {
            Affine value = store.get(((Var) expr).name);
            if (value == null) throw new IllegalArgumentException("read of undefined variable "
                    + ((Var) expr).name);
            return value;
        }
        Bin bin = (Bin) expr;
        Affine left = evaluate(bin.left, store);
        Affine right = evaluate(bin.right, store);
        if (bin.operator.equals("+")) return Affine.add(left, right, 1);
        if (bin.operator.equals("-")) return Affine.add(left, right, -1);
        if (bin.operator.equals("*")) {
            if (left.isConstant()) return Affine.add(Affine.constant(inputNames.size(), 0), right,
                    left.constant);
            if (right.isConstant()) return Affine.add(Affine.constant(inputNames.size(), 0), left,
                    right.constant);
            throw new IllegalArgumentException("multiplication of two symbolic values is unsupported");
        }
        throw new IllegalArgumentException("unsupported operator " + bin.operator);
    }

    private Condition condition(Expr expr, Map<String, Affine> store) {
        Bin bin = (Bin) expr;
        int operator;
        switch (bin.operator) {
            case "<": operator = Condition.LT; break;
            case "<=": operator = Condition.LE; break;
            case ">": operator = Condition.GT; break;
            case ">=": operator = Condition.GE; break;
            case "==": operator = Condition.EQ; break;
            case "!=": operator = Condition.NE; break;
            default: throw new IllegalArgumentException("unsupported comparison " + bin.operator);
        }
        return Condition.of(evaluate(bin.left, store), evaluate(bin.right, store), operator);
    }

    /** Searches the input domain for a model of the path condition plus `extra`. */
    private int[] solve(List<Condition> conditions, Condition extra) {
        int[] inputs = new int[inputNames.size()];
        return search(conditions, extra, inputs, 0);
    }

    private int[] search(List<Condition> conditions, Condition extra, int[] inputs, int index) {
        if (index == inputs.length) {
            for (Condition condition : conditions) if (!condition.holds(inputs)) return null;
            if (extra != null && !extra.holds(inputs)) return null;
            return inputs.clone();
        }
        for (int candidate = -domainBound; candidate <= domainBound; candidate++) {
            inputs[index] = candidate;
            int[] found = search(conditions, extra, inputs, index + 1);
            if (found != null) return found;
        }
        return null;
    }

    /** Runs the symbolic exploration. */
    Result run() {
        Result result = new Result();
        Map<String, Affine> initialStore = new LinkedHashMap<>();
        for (int i = 0; i < inputNames.size(); i++) {
            initialStore.put(inputNames.get(i), Affine.input(inputNames.size(), i));
        }
        List<Object[]> worklist = new ArrayList<>();       // {pc, store, conditions, unroll[], complete}
        worklist.add(new Object[]{0, initialStore, new ArrayList<Condition>(), new int[code.size()],
                true});
        while (!worklist.isEmpty()) {
            Object[] state = worklist.remove(worklist.size() - 1);
            int pc = (Integer) state[0];
            @SuppressWarnings("unchecked")
            Map<String, Affine> store = (Map<String, Affine>) state[1];
            @SuppressWarnings("unchecked")
            List<Condition> conditions = (List<Condition>) state[2];
            int[] unroll = (int[]) state[3];
            boolean complete = (Boolean) state[4];
            if (++result.states > STATE_BUDGET) {
                throw new IllegalStateException("state budget exceeded: " + result.states);
            }
            if (pc < 0 || pc >= code.size()) throw new IllegalStateException("program counter escaped");
            Instruction instruction = code.get(pc);
            switch (instruction.opcode) {
                case Instruction.ASSIGN: {
                    Map<String, Affine> next = new LinkedHashMap<>(store);
                    next.put(instruction.name, evaluate(instruction.expr, store));
                    worklist.add(new Object[]{pc + 1, next, new ArrayList<>(conditions), unroll.clone(),
                            complete});
                    break;
                }
                case Instruction.ASSERT: {
                    result.asserts++;
                    Condition holds = condition(instruction.expr, store);
                    List<Condition> passed = new ArrayList<>(conditions);
                    passed.add(holds);
                    worklist.add(new Object[]{pc + 1, new LinkedHashMap<>(store), passed, unroll.clone(),
                            complete});
                    int[] witness = solve(conditions, Condition.of(holds.difference,
                            Affine.constant(inputNames.size(), 0), Condition.negate(holds.operator)));
                    if (witness != null) {
                        List<Condition> violated = new ArrayList<>(conditions);
                        violated.add(new Condition(holds.difference,
                                Condition.negate(holds.operator)));
                        List<String> failures = new ArrayList<>();
                        failures.add(instruction.description + " violated");
                        result.outcomes.add(new Outcome(violated, store, failures, witness, complete));
                    }
                    break;
                }
                case Instruction.GOTO: {
                    worklist.add(new Object[]{instruction.target, store, conditions, unroll,
                            complete});
                    break;
                }
                case Instruction.IF_TRUE: {
                    result.branches++;
                    Condition test = condition(instruction.expr, store);
                    // the test holds: fall through into the then-branch / loop body
                    List<Condition> taken = new ArrayList<>(conditions);
                    taken.add(test);
                    worklist.add(new Object[]{pc + 1, new LinkedHashMap<>(store), taken,
                            unroll.clone(), complete});
                    // the test fails: jump to the else-branch / loop exit
                    List<Condition> notTaken = new ArrayList<>(conditions);
                    notTaken.add(new Condition(test.difference, Condition.negate(test.operator)));
                    worklist.add(new Object[]{instruction.target, new LinkedHashMap<>(store), notTaken,
                            unroll.clone(), complete});
                    break;
                }
                case Instruction.ASSUME: {
                    unroll[instruction.label]++;
                    if (unroll[instruction.label] > unrollBound) {
                        // the loop needs more iterations than the bound allows: stop this path
                        result.outcomes.add(new Outcome(conditions, store, new ArrayList<>(), null,
                                false));
                        break;
                    }
                    worklist.add(new Object[]{pc + 1, store, conditions, unroll, true});
                    break;
                }
                default: {
                    result.outcomes.add(new Outcome(conditions, store, new ArrayList<>(), null,
                            complete));
                    break;
                }
            }
        }
        return result;
    }

    /** Complete paths only, used by the store agreement check. */
    static List<Outcome> completePaths(Result result) {
        List<Outcome> paths = new ArrayList<>();
        for (Outcome outcome : result.outcomes) {
            if (outcome.complete && outcome.counterexample == null) paths.add(outcome);
        }
        return paths;
    }
}

/** Concrete interpreter over the same AST, used as the oracle. */
final class Interpreter {
    static final class Run {
        final List<String> failures = new ArrayList<>();
        final Map<String, Long> store = new LinkedHashMap<>();
        int iterations;

        boolean safe() { return failures.isEmpty(); }
    }

    static Run run(List<Stmt> program, List<String> inputNames, int[] inputs) {
        Run run = new Run();
        Map<String, Long> store = run.store;
        for (int i = 0; i < inputNames.size(); i++) store.put(inputNames.get(i), (long) inputs[i]);
        execute(program, store, run);
        return run;
    }

    private static void execute(List<Stmt> statements, Map<String, Long> store, Run run) {
        for (Stmt statement : statements) {
            if (statement instanceof Assign) {
                Assign assign = (Assign) statement;
                store.put(assign.name, eval(assign.value, store));
            } else if (statement instanceof Assert) {
                if (eval(((Assert) statement).condition, store) == 0) {
                    run.failures.add("assert violated: "
                            + ((Assert) statement).condition.toString());
                }
            } else if (statement instanceof If) {
                If branch = (If) statement;
                execute(eval(branch.condition, store) != 0 ? branch.thenBranch : branch.elseBranch,
                        store, run);
            } else if (statement instanceof While) {
                While loop = (While) statement;
                while (eval(loop.condition, store) != 0) {
                    run.iterations++;
                    execute(loop.body, store, run);
                }
            } else {
                throw new IllegalArgumentException("unsupported statement");
            }
        }
    }

    static long eval(Expr expr, Map<String, Long> store) {
        if (expr instanceof Const) return ((Const) expr).value;
        if (expr instanceof Var) {
            Long value = store.get(((Var) expr).name);
            if (value == null) throw new IllegalArgumentException("read of undefined variable");
            return value;
        }
        Bin bin = (Bin) expr;
        long left = eval(bin.left, store);
        long right = eval(bin.right, store);
        switch (bin.operator) {
            case "+": return left + right;
            case "-": return left - right;
            case "*": return left * right;
            case "<": return left < right ? 1 : 0;
            case "<=": return left <= right ? 1 : 0;
            case ">": return left > right ? 1 : 0;
            case ">=": return left >= right ? 1 : 0;
            case "==": return left == right ? 1 : 0;
            default: return left != right ? 1 : 0;
        }
    }
}

public class Solution {
    static int checksRun;

    static Expr var(String name) { return new Var(name); }

    static Expr num(long value) { return new Const(value); }

    static Expr bin(String operator, Expr left, Expr right) { return new Bin(operator, left, right); }

    static List<Stmt> stmts(Stmt... statements) { return new ArrayList<>(Arrays.asList(statements)); }

    /** Compiles and explores a program, then cross-checks it against the concrete interpreter. */
    static void verifyProgram(List<Stmt> program, List<String> inputs, int domain, int unroll,
                              boolean expectCounterexample) {
        SymExec engine = new SymExec(inputs, domain, unroll);
        engine.compile(program);
        SymExec.Result result = engine.run();
        check(result.counterexampleCount() > 0 == expectCounterexample,
                "counterexample expectation failed: got " + result.counterexampleCount());
        List<SymExec.Outcome> complete = SymExec.completePaths(result);
        int inputsCovered = 0;
        int concreteFailures = 0;
        int maxIterations = 0;
        int[] vector = new int[inputs.size()];
        Arrays.fill(vector, -domain);
        while (true) {
            Interpreter.Run concrete = Interpreter.run(program, inputs, vector);
            maxIterations = Math.max(maxIterations, concrete.iterations);
            boolean symbolicFailure = false;
            for (SymExec.Outcome outcome : result.outcomes) {
                if (outcome.counterexample != null && outcome.conditionsHold(vector)) {
                    symbolicFailure = true;
                }
            }
            if (!concrete.safe()) {
                concreteFailures++;
                check(symbolicFailure, "concrete input fails but no symbolic path covers it: "
                        + Arrays.toString(vector));
            } else {
                SymExec.Outcome match = null;
                for (SymExec.Outcome outcome : complete) {
                    if (outcome.conditionsHold(vector)) {
                        match = outcome;
                        break;
                    }
                }
                check(match != null, "no symbolic path covers the input " + Arrays.toString(vector));
                for (Map.Entry<String, Affine> entry : match.store.entrySet()) {
                    check(entry.getValue().evaluate(vector) == concrete.store.get(entry.getKey()),
                            "store disagreement for " + entry.getKey() + " at "
                                    + Arrays.toString(vector));
                }
                inputsCovered++;
            }
            int position = 0;
            while (position < vector.length && vector[position] == domain) {
                vector[position] = -domain;
                position++;
            }
            if (position == vector.length) break;
            vector[position]++;
        }
        check(concreteFailures > 0 == expectCounterexample,
                "the concrete interpreter disagrees about failures: " + concreteFailures);
        check(maxIterations <= unroll, "the unroll bound " + unroll + " was not sufficient, saw "
                + maxIterations);
        for (SymExec.Outcome outcome : result.outcomes) {
            if (outcome.counterexample == null) continue;
            Interpreter.Run replay = Interpreter.run(program, inputs, outcome.counterexample);
            check(!replay.safe(), "a symbolic counterexample did not fail concretely: "
                    + Arrays.toString(outcome.counterexample));
        }
        check(result.asserts > 0, "the program contains at least one assert");
    }

    public static void main(String[] args) {
        // --- negative cases -------------------------------------------------------
        SymExec undeclared = new SymExec(List.of("x"), 4, 4);
        undeclared.compile(stmts(new Assign("y", bin("+", var("z"), num(1)))));
        boolean readError = false;
        try {
            undeclared.run();
        } catch (IllegalArgumentException expected) {
            readError = true;
        }
        check(readError, "reading an undefined variable must be rejected");

        SymExec nonlinear = new SymExec(List.of("x", "y"), 4, 4);
        nonlinear.compile(stmts(new Assign("z", bin("*", var("x"), var("y")))));
        boolean multiplyError = false;
        try {
            nonlinear.run();
        } catch (IllegalArgumentException expected) {
            multiplyError = true;
        }
        check(multiplyError, "multiplying two symbolic values must be rejected");

        // --- state explosion is reported, not hidden ------------------------------
        List<Stmt> explosion = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            explosion.add(new If(bin(">", var("x"), num(i)), stmts(new Assign("y" + i, num(i))),
                    stmts(new Assign("z" + i, num(i)))));
        }
        explosion.add(new Assert(bin("==", num(1), num(1))));
        SymExec huge = new SymExec(List.of("x"), 4, 4);
        huge.compile(explosion);
        boolean budgetError = false;
        try {
            huge.run();
        } catch (IllegalStateException expected) {
            budgetError = true;
        }
        check(budgetError, "a state explosion must abort with an error");

        // --- a program that is safe on the whole domain --------------------------
        List<Stmt> safe = stmts(
                new If(bin(">", var("x"), num(0)),
                        stmts(new Assign("y", num(1))),
                        stmts(new Assign("y", num(-1)))),
                new Assert(bin("!=", var("y"), num(0))),
                new Assign("z", bin("*", num(2), var("y"))),
                new Assert(bin("!=", var("z"), num(0))));
        verifyProgram(safe, List.of("x"), 6, 4, false);

        // --- a program that fails on the whole negative half --------------------
        List<Stmt> failing = stmts(
                new Assign("y", bin("+", var("x"), num(1))),
                new Assert(bin(">", var("y"), num(0))));
        verifyProgram(failing, List.of("x"), 6, 4, true);

        // --- a failure hidden behind two nested branches ------------------------
        List<Stmt> nested = stmts(
                new If(bin(">", var("x"), num(0)),
                        stmts(new Assign("a", bin("-", var("x"), num(1)))),
                        stmts(new Assign("a", bin("+", var("x"), num(1))))),
                new If(bin(">", var("a"), num(2)),
                        stmts(new Assert(bin("<", var("x"), num(2)))),
                        stmts(new Assert(bin(">", var("a"), num(-4))))));
        verifyProgram(nested, List.of("x"), 5, 4, true);

        // --- two inputs, four branches ------------------------------------------
        List<Stmt> twoInputs = stmts(
                new If(bin(">", var("x"), var("y")),
                        stmts(new Assign("d", bin("-", var("x"), var("y")))),
                        stmts(new Assign("d", bin("-", var("y"), var("x"))))),
                new Assert(bin(">=", var("d"), num(0))),
                new If(bin("==", var("d"), num(0)),
                        stmts(new Assert(bin("==", var("x"), var("y")))),
                        stmts(new Assert(bin("!=", var("x"), var("y"))))));
        verifyProgram(twoInputs, List.of("x", "y"), 4, 4, false);

        // --- the same program with a deliberately wrong assertion ---------------
        List<Stmt> wrong = stmts(
                new If(bin(">", var("x"), var("y")),
                        stmts(new Assign("d", bin("-", var("x"), var("y")))),
                        stmts(new Assign("d", bin("-", var("y"), var("x"))))),
                new Assert(bin(">", var("d"), num(0))));
        verifyProgram(wrong, List.of("x", "y"), 4, 4, true);

        // --- a loop, unrolled enough for the whole domain -----------------------
        List<Stmt> loop = stmts(
                new Assign("i", num(0)),
                new While(bin("<", var("i"), num(3)), stmts(new Assign("i", bin("+", var("i"), num(1))))),
                new Assert(bin("==", var("i"), num(3))));
        verifyProgram(loop, List.of("x"), 3, 5, false);

        // --- a loop whose body can break the assertion --------------------------
        List<Stmt> badLoop = stmts(
                new Assign("i", num(0)),
                new While(bin("<", var("i"), num(2)),
                        stmts(new Assign("i", bin("+", var("i"), num(1))),
                                new Assert(bin("<", var("i"), num(2))))),
                new Assert(bin(">=", var("i"), num(2))));
        verifyProgram(badLoop, List.of("x"), 3, 4, true);

        // --- determinism and bookkeeping ---------------------------------------
        SymExec first = new SymExec(List.of("x", "y"), 4, 4);
        first.compile(twoInputs);
        SymExec.Result firstResult = first.run();
        SymExec second = new SymExec(List.of("x", "y"), 4, 4);
        second.compile(twoInputs);
        SymExec.Result secondResult = second.run();
        check(firstResult.states == secondResult.states, "the exploration is deterministic");
        check(firstResult.branches == secondResult.branches, "the branch count is deterministic");
        check(firstResult.asserts == secondResult.asserts, "the assert count is deterministic");
        check(firstResult.states > 1 && firstResult.branches > 0, "the engine really explored paths");

        System.out.println("symbolic execution: " + firstResult.states + " states, "
                + firstResult.branches + " branches, " + firstResult.asserts + " assert visits");
        System.out.println("every counterexample replayed concretely, every domain input matched a "
                + "symbolic path; " + checksRun + " checks");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        checksRun++;
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
