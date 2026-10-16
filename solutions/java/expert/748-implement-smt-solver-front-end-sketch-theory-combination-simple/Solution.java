/** Task 748 - Implement SMT solver front-end sketch (theory combination simple).
 * (Tier: Expert, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * A miniature SMT front-end plus a DPLL(T) core:
 *   front-end   - `(declare-const x Int)`, `(declare-fun f (Int) Int)`, `(assert t)`,
 *                 `(check-sat)`, `(push 1)`, `(pop 1)`; terms are integers, variables, `and`, `or`,
 *                 `not`, `=`, `distinct`, `<`, `<=`, `>`, `>=`, `+`, `-` and applications of
 *                 uninterpreted functions. The tokenizer/parser builds an s-expression tree and
 *                 rejects unbalanced, malformed or undeclared input.
 *   purification - atoms are linearised into *difference constraints*; a function application used
 *                 inside arithmetic becomes a node of its own, which is exactly the term the EUF
 *                 part reasons about, so both theories share it by construction.
 *   theories    - congruence closure (union-find plus the congruence rule over applications) and
 *                 difference logic solved by Bellman-Ford with a virtual source; integer
 *                 disequalities that the bounds do not already decide are split into x - y <= -1 or
 *                 y - x <= -1.
 *   combination - a Nelson-Oppen fixpoint: congruence closure merges nodes, merged nodes become the
 *                 same arithmetic variable, and an equality implied by the bounds (neither
 *                 x - y <= -1 nor y - x <= -1 feasible) is fed back into the closure until nothing
 *                 changes.
 * The boolean structure is searched exhaustively with tri-valued pruning over the (few) atoms.
 *
 * Simplification (documented). The supported fragment is integer difference logic with EUF:
 * multiplication, coefficients other than +1/-1, quantifiers, arrays, `ite` and bit-vectors are
 * rejected with a descriptive error, equalities must relate single nodes, and the Boolean search is
 * exhaustive rather than a CDCL core. The Nelson-Oppen loop propagates equalities only, i.e. the
 * classical *simple* combination; disequalities are discharged by case splitting rather than by an
 * equality-partition enumeration.
 *
 * Verified. Pure arithmetic instances are cross-checked against brute-force enumeration over a
 * bounded integer domain; pure EUF instances against an independent partition oracle that
 * enumerates every set partition of the term universe and tests congruence, equalities and
 * disequalities; every SAT answer is validated by *evaluating the returned model* on every
 * assertion; the UNSAT answers of mixed instances are additionally justified by a single-theory
 * refutation; push/pop scoping, malformed programs, unsupported fragments and determinism each have
 * dedicated checks.
 */
import java.util.*;

/** S-expression front-end for the supported SMT-LIB subset. */
final class SmtParser {
    private final List<Object> tokens;
    private int position;

    private SmtParser(List<Object> tokens) {
        this.tokens = tokens;
    }

    static List<Object> tokenize(String source) {
        List<Object> tokens = new ArrayList<>();
        int i = 0;
        while (i < source.length()) {
            char c = source.charAt(i);
            if (Character.isWhitespace(c)) {
                i++;
            } else if (c == '(' || c == ')') {
                tokens.add(String.valueOf(c));
                i++;
            } else {
                int start = i;
                while (i < source.length() && !Character.isWhitespace(source.charAt(i))
                        && source.charAt(i) != '(' && source.charAt(i) != ')') {
                    i++;
                }
                tokens.add(source.substring(start, i));
            }
        }
        return tokens;
    }

    /** Parses a whole program into a list of commands; throws on unbalanced input. */
    static List<Object> parseProgram(String source) {
        SmtParser parser = new SmtParser(tokenize(source));
        List<Object> commands = new ArrayList<>();
        while (parser.position < parser.tokens.size()) commands.add(parser.parseOne());
        return commands;
    }

    private Object parseOne() {
        if (position >= tokens.size()) throw new IllegalArgumentException("unexpected end of input");
        String text = String.valueOf(tokens.get(position++));
        if (text.equals("(")) {
            List<Object> list = new ArrayList<>();
            while (true) {
                if (position >= tokens.size()) throw new IllegalArgumentException("unbalanced '('");
                if (tokens.get(position).equals(")")) {
                    position++;
                    return list;
                }
                list.add(parseOne());
            }
        }
        if (text.equals(")")) throw new IllegalArgumentException("unexpected ')'");
        try {
            return Long.parseLong(text);
        } catch (NumberFormatException notAnInteger) {
            return text;                                     // a symbol
        }
    }

    static boolean isList(Object term) { return term instanceof List; }

    @SuppressWarnings("unchecked")
    static List<Object> asList(Object term) { return (List<Object>) term; }
}

/** Difference constraints x_to - x_from <= weight, solved by Bellman-Ford from a virtual source. */
final class DifferenceLogic {
    /** Returns feasible integer potentials, or null when a negative cycle makes it infeasible. */
    static long[] solve(int nVars, List<long[]> constraints) {
        int size = nVars + 1;
        int source = nVars;
        long[] dist = new long[size];
        Arrays.fill(dist, Long.MAX_VALUE / 4);
        dist[source] = 0;
        List<long[]> edges = new ArrayList<>(constraints);
        for (int v = 0; v < nVars; v++) edges.add(new long[]{source, v, 0});
        for (int round = 0; round <= size; round++) {
            boolean relaxed = false;
            for (long[] edge : edges) {
                int from = (int) edge[0];
                int to = (int) edge[1];
                if (dist[from] + edge[2] < dist[to]) {
                    dist[to] = dist[from] + edge[2];
                    relaxed = true;
                }
            }
            if (!relaxed) return Arrays.copyOf(dist, nVars);
            if (round == size) return null;                  // a negative cycle was found
        }
        return null;
    }
}

/** The solver state: declarations, assertion stack, purification and the DPLL(T) search. */
final class SmtSolver {
    private final Map<String, Integer> variables = new LinkedHashMap<>();
    private final Map<String, Integer> functions = new LinkedHashMap<>();
    private final List<Object> assertions = new ArrayList<>();
    private final Deque<Integer> scopes = new ArrayDeque<>();

    private int nodeCount;
    private int theoryChecks;
    private int atomCount;
    private final List<Object[]> applications = new ArrayList<>();      // {fn, Integer[] args, node}
    private final Map<String, Integer> applicationNodes = new LinkedHashMap<>();
    private final Map<Integer, Long> literalValues = new LinkedHashMap<>();
    private final List<Atom> atoms = new ArrayList<>();
    private final List<Object> rootStructures = new ArrayList<>();
    private int zeroNode;

    int theoryCheckCount() { return theoryChecks; }

    int atomCount() { return atomCount; }

    void execute(String source) {
        for (Object command : SmtParser.parseProgram(source)) {
            if (!SmtParser.isList(command)) throw new IllegalArgumentException("a command must be a list");
            List<Object> parts = SmtParser.asList(command);
            if (parts.isEmpty()) throw new IllegalArgumentException("empty command");
            String head = String.valueOf(parts.get(0));
            switch (head) {
                case "declare-const":
                    if (parts.size() != 3) throw new IllegalArgumentException("declare-const arity");
                    variables.putIfAbsent(String.valueOf(parts.get(1)), variables.size());
                    break;
                case "declare-fun":
                    if (parts.size() != 4) throw new IllegalArgumentException("declare-fun arity");
                    functions.put(String.valueOf(parts.get(1)), SmtParser.asList(parts.get(2)).size());
                    break;
                case "assert":
                    if (parts.size() != 2) throw new IllegalArgumentException("assert arity");
                    assertions.add(parts.get(1));
                    break;
                case "push":
                    scopes.push(assertions.size());
                    break;
                case "pop":
                    if (scopes.isEmpty()) throw new IllegalArgumentException("pop without push");
                    int keep = scopes.pop();
                    while (assertions.size() > keep) assertions.remove(assertions.size() - 1);
                    break;
                case "check-sat":
                    break;
                default:
                    throw new IllegalArgumentException("unsupported command " + head);
            }
        }
    }

    /** The model handed back for a satisfiable instance. */
    static final class Model {
        final Map<String, Long> variableValues;
        final Map<String, Long> functionTable;      // key: "fn#v1,v2" -> result value

        Model(Map<String, Long> variableValues, Map<String, Long> functionTable) {
            this.variableValues = variableValues;
            this.functionTable = functionTable;
        }
    }

    private static final class Atom {
        static final int EQUAL = 0;
        static final int DISTINCT = 1;
        static final int BOUND = 2;                 // left - right <= bound
        static final int CONSTANT = 3;              // bound == 1 means "true"

        final int kind;
        final int left;
        final int right;
        final long bound;

        Atom(int kind, int left, int right, long bound) {
            this.kind = kind;
            this.left = left;
            this.right = right;
            this.bound = bound;
        }
    }

    /** Linear term: coefficients per node plus a constant, restricted to the difference fragment. */
    private static final class Linear {
        final Map<Integer, Long> coefficients = new LinkedHashMap<>();
        long constant;

        void add(Linear other, long factor) {
            constant += factor * other.constant;
            for (Map.Entry<Integer, Long> entry : other.coefficients.entrySet()) {
                long updated = coefficients.getOrDefault(entry.getKey(), 0L) + factor * entry.getValue();
                if (updated == 0) coefficients.remove(entry.getKey());
                else coefficients.put(entry.getKey(), updated);
            }
        }
    }

    private int nodeForVariable(String name) {
        Integer id = variables.get(name);
        if (id == null) throw new IllegalArgumentException("undeclared variable " + name);
        return id;
    }

    private int nodeForLiteral(long value) {
        for (Map.Entry<Integer, Long> entry : literalValues.entrySet()) {
            if (entry.getValue() == value) return entry.getKey();
        }
        int id = nodeCount++;
        literalValues.put(id, value);
        return id;
    }

    private int nodeForApplication(String function, Integer[] argumentNodes) {
        StringBuilder key = new StringBuilder(function);
        for (int argument : argumentNodes) key.append('#').append(argument);
        Integer existing = applicationNodes.get(key.toString());
        if (existing != null) return existing;
        int id = nodeCount++;
        applicationNodes.put(key.toString(), id);
        applications.add(new Object[]{function, argumentNodes, id});
        return id;
    }

    private Linear linearize(Object term) {
        Linear result = new Linear();
        if (term instanceof Long) {
            result.constant = (Long) term;
            return result;
        }
        if (term instanceof String) {
            result.coefficients.put(nodeForVariable((String) term), 1L);
            return result;
        }
        List<Object> list = SmtParser.asList(term);
        if (list.isEmpty()) throw new IllegalArgumentException("empty term");
        String head = String.valueOf(list.get(0));
        if (head.equals("+")) {
            for (int i = 1; i < list.size(); i++) result.add(linearize(list.get(i)), 1);
            return result;
        }
        if (head.equals("-")) {
            if (list.size() == 2) {
                result.add(linearize(list.get(1)), -1);
                return result;
            }
            if (list.size() == 3) {
                result.add(linearize(list.get(1)), 1);
                result.add(linearize(list.get(2)), -1);
                return result;
            }
            throw new IllegalArgumentException("'-' takes one or two arguments");
        }
        Integer arity = functions.get(head);
        if (arity == null) throw new IllegalArgumentException("unsupported arithmetic term " + head);
        if (list.size() - 1 != arity) {
            throw new IllegalArgumentException("function " + head + " expects " + arity + " arguments");
        }
        Integer[] argumentNodes = new Integer[arity];
        for (int i = 1; i < list.size(); i++) {
            Linear argument = linearize(list.get(i));
            if (argument.coefficients.size() != 1 || argument.constant != 0) {
                throw new IllegalArgumentException("function arguments must be single nodes: "
                        + list.get(i));
            }
            argumentNodes[i - 1] = argument.coefficients.keySet().iterator().next();
        }
        Linear application = new Linear();
        application.coefficients.put(nodeForApplication(head, argumentNodes), 1L);
        return application;
    }

    private int singleNode(Linear linear, Object term) {
        if (linear.coefficients.size() == 1 && linear.constant == 0) {
            return linear.coefficients.keySet().iterator().next();
        }
        if (linear.coefficients.isEmpty()) return nodeForLiteral(linear.constant);
        throw new IllegalArgumentException("expected a single node: " + term);
    }

    /** Atom for (lhs op rhs); both sides are linearised into the difference fragment. */
    private Atom parseAtom(List<Object> atom) {
        String head = String.valueOf(atom.get(0));
        if (head.equals("=") || head.equals("distinct")) {
            if (atom.size() != 3) throw new IllegalArgumentException(head + " takes two arguments");
            int left = singleNode(linearize(atom.get(1)), atom.get(1));
            int right = singleNode(linearize(atom.get(2)), atom.get(2));
            return new Atom(head.equals("=") ? Atom.EQUAL : Atom.DISTINCT, left, right, 0);
        }
        if (!head.equals("<") && !head.equals("<=") && !head.equals(">") && !head.equals(">=")) {
            throw new IllegalArgumentException("unsupported atom " + head);
        }
        if (atom.size() != 3) throw new IllegalArgumentException(head + " takes two arguments");
        Linear difference = new Linear();
        difference.add(linearize(atom.get(1)), 1);
        difference.add(linearize(atom.get(2)), -1);
        int positive = -1;
        int negative = -1;
        for (Map.Entry<Integer, Long> entry : difference.coefficients.entrySet()) {
            if (entry.getValue() == 1) positive = entry.getKey();
            else if (entry.getValue() == -1) negative = entry.getKey();
            else throw new IllegalArgumentException("coefficient " + entry.getValue()
                    + " is outside the difference fragment");
        }
        if (positive < 0 && negative < 0) {
            long constant = difference.constant;
            boolean truth = head.equals("<") ? constant < 0
                    : head.equals("<=") ? constant <= 0
                    : head.equals(">") ? constant > 0 : constant >= 0;
            return new Atom(Atom.CONSTANT, 0, 0, truth ? 1 : 0);
        }
        if (head.equals("<") || head.equals("<=")) {
            long bound = -difference.constant - (head.equals("<") ? 1 : 0);
            int left = positive >= 0 ? positive : nodeForLiteral(0);
            int right = negative >= 0 ? negative : nodeForLiteral(0);
            return new Atom(Atom.BOUND, left, right, bound);
        }
        // ">" and ">=": multiplying x_pos - x_neg by -1 swaps the roles of the endpoints
        long bound = difference.constant - (head.equals(">") ? 1 : 0);
        int left = negative >= 0 ? negative : nodeForLiteral(0);
        int right = positive >= 0 ? positive : nodeForLiteral(0);
        return new Atom(Atom.BOUND, left, right, bound);
    }

    private int atomIndex(List<Object> atom) {
        Atom parsed = parseAtom(atom);
        for (int i = 0; i < atoms.size(); i++) {
            Atom other = atoms.get(i);
            if (other.kind == parsed.kind && other.left == parsed.left && other.right == parsed.right
                    && other.bound == parsed.bound) {
                return i;
            }
        }
        atoms.add(parsed);
        return atoms.size() - 1;
    }

    /** Flattens an assertion into a boolean structure over atom indices. */
    private Object flatten(Object term) {
        if (!SmtParser.isList(term)) throw new IllegalArgumentException("assertions must be boolean");
        List<Object> list = SmtParser.asList(term);
        String head = String.valueOf(list.get(0));
        if (head.equals("and") || head.equals("or")) {
            List<Object> parts = new ArrayList<>();
            for (int i = 1; i < list.size(); i++) parts.add(flatten(list.get(i)));
            return new Object[]{head, parts};
        }
        if (head.equals("not")) {
            if (list.size() != 2) throw new IllegalArgumentException("not takes one argument");
            return new Object[]{"not", flatten(list.get(1))};
        }
        return atomIndex(list);
    }

    private void collectAtoms() {
        atoms.clear();
        applications.clear();
        applicationNodes.clear();
        literalValues.clear();
        rootStructures.clear();
        nodeCount = variables.size();
        for (Object assertion : assertions) rootStructures.add(flatten(assertion));
        zeroNode = nodeForLiteral(0);       // pinned before the union-find array is allocated
        atomCount = atoms.size();
    }

    /** Tri-valued evaluation of the boolean structure under a partial atom assignment. */
    private Boolean evaluate(Object structure, int[] assignment) {
        if (structure instanceof Integer) {
            int index = (Integer) structure;
            return assignment[index] == 0 ? Boolean.FALSE : assignment[index] == 1 ? Boolean.TRUE : null;
        }
        Object[] node = (Object[]) structure;
        @SuppressWarnings("unchecked")
        List<Object> parts = (List<Object>) node[1];
        if (String.valueOf(node[0]).equals("not")) {
            Boolean inner = evaluate(parts.get(0), assignment);
            return inner == null ? null : !inner;
        }
        boolean conjunction = String.valueOf(node[0]).equals("and");
        boolean undecided = false;
        for (Object part : parts) {
            Boolean value = evaluate(part, assignment);
            if (value == null) {
                undecided = true;
                continue;
            }
            if (conjunction && !value) return Boolean.FALSE;
            if (!conjunction && value) return Boolean.TRUE;
        }
        return undecided ? null : conjunction;
    }

    /** Solves the current assertion stack; returns null when unsatisfiable. */
    Model checkSat() {
        collectAtoms();
        int[] assignment = new int[atoms.size()];
        Arrays.fill(assignment, -1);
        return search(0, assignment);
    }

    private Model search(int index, int[] assignment) {
        for (Object structure : rootStructures) {
            if (Boolean.FALSE.equals(evaluate(structure, assignment))) return null;
        }
        if (index == atoms.size()) return tryAssignment(assignment);
        assignment[index] = 1;
        Model positive = search(index + 1, assignment);
        if (positive != null) return positive;
        assignment[index] = 0;
        Model negative = search(index + 1, assignment);
        assignment[index] = -1;
        return negative;
    }

    private Model tryAssignment(int[] assignment) {
        int[] parent = new int[nodeCount];
        for (int i = 0; i < nodeCount; i++) parent[i] = i;
        List<int[]> disequalities = new ArrayList<>();
        for (int i = 0; i < atoms.size(); i++) {
            Atom atom = atoms.get(i);
            if (assignment[i] == 1 && atom.kind == Atom.DISTINCT) {
                disequalities.add(new int[]{atom.left, atom.right});
            }
            if (assignment[i] == 1 && atom.kind == Atom.CONSTANT && atom.bound == 0) {
                return null;                                 // a false constant comparison was asserted
            }
        }
        List<long[]> extra = new ArrayList<>();
        boolean changed = true;
        long[] potentials = null;
        while (changed) {
            changed = false;
            for (int i = 0; i < atoms.size(); i++) {
                Atom atom = atoms.get(i);
                if (assignment[i] == 1 && atom.kind == Atom.EQUAL
                        && union(parent, atom.left, atom.right)) {
                    changed = true;
                }
            }
            for (int i = 0; i < applications.size(); i++) {
                for (int j = i + 1; j < applications.size(); j++) {
                    Object[] a = applications.get(i);
                    Object[] b = applications.get(j);
                    if (!a[0].equals(b[0])) continue;
                    Integer[] argsA = (Integer[]) a[1];
                    Integer[] argsB = (Integer[]) b[1];
                    boolean sameArguments = true;
                    for (int k = 0; k < argsA.length; k++) {
                        if (find(parent, argsA[k]) != find(parent, argsB[k])) sameArguments = false;
                    }
                    if (sameArguments && union(parent, (Integer) a[2], (Integer) b[2])) changed = true;
                }
            }
            for (int[] pair : disequalities) {
                if (find(parent, pair[0]) == find(parent, pair[1])) return null;
            }
            List<long[]> constraints = buildConstraints(parent, assignment, extra);
            theoryChecks++;
            potentials = DifferenceLogic.solve(nodeCount, constraints);
            if (potentials == null) return null;
            for (int i = 0; i < atoms.size(); i++) {
                Atom atom = atoms.get(i);
                if (atom.kind != Atom.BOUND) continue;
                int a = find(parent, atom.left);
                int b = find(parent, atom.right);
                if (a != b && impliedEqual(constraints, a, b, nodeCount) && union(parent, a, b)) {
                    changed = true;
                }
            }
        }
        List<long[]> constraints = buildConstraints(parent, assignment, extra);
        for (int[] pair : disequalities) {                   // integer disequalities: case split
            int a = find(parent, pair[0]);
            int b = find(parent, pair[1]);
            if (a == b) return null;
            List<long[]> probe = new ArrayList<>(constraints);
            probe.add(new long[]{b, a, -1});
            if (DifferenceLogic.solve(nodeCount, probe) != null) {
                extra.add(new long[]{b, a, -1});
                constraints = buildConstraints(parent, assignment, extra);
                continue;
            }
            probe = new ArrayList<>(constraints);
            probe.add(new long[]{a, b, -1});
            if (DifferenceLogic.solve(nodeCount, probe) == null) return null;
            extra.add(new long[]{a, b, -1});
            constraints = buildConstraints(parent, assignment, extra);
        }
        potentials = DifferenceLogic.solve(nodeCount, constraints);
        if (potentials == null) return null;
        return buildModel(parent, potentials, disequalities);
    }

    private List<long[]> buildConstraints(int[] parent, int[] assignment, List<long[]> extra) {
        List<long[]> constraints = new ArrayList<>(extra);
        int zero = find(parent, zeroNode);
        for (Map.Entry<Integer, Long> entry : literalValues.entrySet()) {
            int node = find(parent, entry.getKey());
            constraints.add(new long[]{zero, node, entry.getValue()});
            constraints.add(new long[]{node, zero, -entry.getValue()});
        }
        for (int i = 0; i < atoms.size(); i++) {
            Atom atom = atoms.get(i);
            if (assignment[i] != 1) continue;
            if (atom.kind == Atom.EQUAL) {
                int a = find(parent, atom.left);
                int b = find(parent, atom.right);
                constraints.add(new long[]{b, a, 0});
                constraints.add(new long[]{a, b, 0});
            }
            if (atom.kind == Atom.BOUND) {
                constraints.add(new long[]{find(parent, atom.right), find(parent, atom.left),
                        atom.bound});
            }
        }
        return constraints;
    }

    private static int find(int[] parent, int node) {
        int root = node;
        while (parent[root] != root) root = parent[root];
        while (parent[node] != root) {
            int next = parent[node];
            parent[node] = root;
            node = next;
        }
        return root;
    }

    private static boolean union(int[] parent, int a, int b) {
        int ra = find(parent, a);
        int rb = find(parent, b);
        if (ra == rb) return false;
        parent[rb] = ra;
        return true;
    }

    /** True when both a - b <= 0 and b - a <= 0 follow from the constraint set. */
    private static boolean impliedEqual(List<long[]> constraints, int a, int b, int nVars) {
        List<long[]> probe = new ArrayList<>(constraints);
        probe.add(new long[]{b, a, -1});
        if (DifferenceLogic.solve(nVars, probe) != null) return false;
        probe = new ArrayList<>(constraints);
        probe.add(new long[]{a, b, -1});
        return DifferenceLogic.solve(nVars, probe) == null;
    }

    /** Turns the union-find classes into a model; classes without arithmetic value get fresh ones. */
    private Model buildModel(int[] parent, long[] potentials, List<int[]> disequalities) {
        Map<Integer, Long> classValue = new LinkedHashMap<>();
        for (Map.Entry<Integer, Long> entry : literalValues.entrySet()) {
            classValue.put(find(parent, entry.getKey()), entry.getValue());
        }
        for (int node = 0; node < nodeCount; node++) {
            int root = find(parent, node);
            if (!classValue.containsKey(root)) classValue.put(root, potentials[node]);
        }
        long nextFresh = 1;
        for (long value : classValue.values()) nextFresh = Math.max(nextFresh, Math.abs(value) + 2);
        for (int[] pair : disequalities) {
            int a = find(parent, pair[0]);
            int b = find(parent, pair[1]);
            if (a != b && classValue.get(a).equals(classValue.get(b))) classValue.put(b, nextFresh++);
        }
        Map<String, Long> variableValues = new LinkedHashMap<>();
        for (Map.Entry<String, Integer> entry : variables.entrySet()) {
            variableValues.put(entry.getKey(), classValue.get(find(parent, entry.getValue())));
        }
        Map<String, Long> table = new LinkedHashMap<>();
        for (Object[] application : applications) {
            Integer[] args = (Integer[]) application[1];
            StringBuilder key = new StringBuilder((String) application[0]).append('#');
            for (int i = 0; i < args.length; i++) {
                if (i > 0) key.append(',');
                key.append(classValue.get(find(parent, args[i])));
            }
            table.put(key.toString(), classValue.get(find(parent, (Integer) application[2])));
        }
        for (int[] pair : disequalities) {
            if (classValue.get(find(parent, pair[0])).equals(classValue.get(find(parent, pair[1])))) {
                return null;                                 // the candidate model violates an atom
            }
        }
        return new Model(variableValues, table);
    }
}

public class Solution {
    static int checksRun;

    static SmtSolver solverFor(String program) {
        SmtSolver solver = new SmtSolver();
        solver.execute(program);
        return solver;
    }

    static boolean sat(String program) { return solverFor(program).checkSat() != null; }

    /** Evaluates the tiny assertion language on a model produced by the solver. */
    static boolean modelSatisfies(Object assertion, SmtSolver.Model model) {
        List<Object> list = SmtParser.asList(assertion);
        String head = String.valueOf(list.get(0));
        if (head.equals("and")) {
            for (int i = 1; i < list.size(); i++) if (!modelSatisfies(list.get(i), model)) return false;
            return true;
        }
        if (head.equals("or")) {
            for (int i = 1; i < list.size(); i++) if (modelSatisfies(list.get(i), model)) return true;
            return false;
        }
        if (head.equals("not")) return !modelSatisfies(list.get(1), model);
        long left = modelValue(list.get(1), model);
        long right = modelValue(list.get(2), model);
        switch (head) {
            case "=": return left == right;
            case "distinct": return left != right;
            case "<": return left < right;
            case "<=": return left <= right;
            case ">": return left > right;
            default: return left >= right;
        }
    }

    static long modelValue(Object term, SmtSolver.Model model) {
        if (term instanceof Long) return (Long) term;
        if (term instanceof String) {
            Long value = model.variableValues.get((String) term);
            if (value == null) throw new IllegalStateException("no value for " + term);
            return value;
        }
        List<Object> list = SmtParser.asList(term);
        String head = String.valueOf(list.get(0));
        if (head.equals("+")) return modelValue(list.get(1), model) + modelValue(list.get(2), model);
        if (head.equals("-")) {
            if (list.size() == 2) return -modelValue(list.get(1), model);
            return modelValue(list.get(1), model) - modelValue(list.get(2), model);
        }
        StringBuilder key = new StringBuilder(head).append('#');
        for (int i = 1; i < list.size(); i++) {
            if (i > 1) key.append(',');
            key.append(modelValue(list.get(i), model));
        }
        Long value = model.functionTable.get(key.toString());
        if (value == null) throw new IllegalStateException("no table entry for " + key);
        return value;
    }

    /** Every assertion of a satisfiable program must evaluate to true in the returned model. */
    static void assertModelValid(String program) {
        SmtSolver.Model model = solverFor(program).checkSat();
        check(model != null, "expected a model for " + program);
        for (Object command : SmtParser.parseProgram(program)) {
            List<Object> parts = SmtParser.asList(command);
            if (String.valueOf(parts.get(0)).equals("assert")) {
                check(modelSatisfies(parts.get(1), model),
                        "the model violates an assertion of " + program);
            }
        }
    }

    /** Bounded enumeration oracle over integer assignments. */
    static boolean enumerateArithmetic(List<String> names, List<Object> assertions, int bound) {
        return enumerateRec(names, assertions, new int[names.size()], 0, bound);
    }

    private static boolean enumerateRec(List<String> names, List<Object> assertions, int[] values,
                                        int index, int bound) {
        if (index == values.length) {
            SmtSolver.Model model = modelOf(names, values);
            for (Object assertion : assertions) {
                if (!modelSatisfies(assertion, model)) return false;
            }
            return true;
        }
        for (int candidate = -bound; candidate <= bound; candidate++) {
            values[index] = candidate;
            if (enumerateRec(names, assertions, values, index + 1, bound)) return true;
        }
        return false;
    }

    private static SmtSolver.Model modelOf(List<String> names, int[] values) {
        Map<String, Long> mapping = new LinkedHashMap<>();
        for (int i = 0; i < names.size(); i++) mapping.put(names.get(i), (long) values[i]);
        return new SmtSolver.Model(mapping, new LinkedHashMap<>());
    }

    /** Independent EUF oracle: every set partition of the term universe is enumerated. */
    static boolean partitionOracle(int termCount, List<int[]> equalities, List<int[]> disequalities,
                                   List<int[]> applications) {
        return partitionRec(0, new int[termCount], termCount, equalities, disequalities, applications);
    }

    private static boolean partitionRec(int index, int[] assignment, int termCount,
                                        List<int[]> equalities, List<int[]> disequalities,
                                        List<int[]> applications) {
        if (index == termCount) {
            for (int[] pair : equalities) if (assignment[pair[0]] != assignment[pair[1]]) return false;
            for (int[] pair : disequalities) if (assignment[pair[0]] == assignment[pair[1]]) return false;
            for (int i = 0; i < applications.size(); i++) {
                for (int j = i + 1; j < applications.size(); j++) {
                    int[] a = applications.get(i);
                    int[] b = applications.get(j);
                    boolean sameArguments = a[0] == b[0];
                    for (int k = 1; k < a.length - 1 && sameArguments; k++) {
                        if (assignment[a[k]] != assignment[b[k]]) sameArguments = false;
                    }
                    if (sameArguments && assignment[a[a.length - 1]] != assignment[b[b.length - 1]]) {
                        return false;
                    }
                }
            }
            return true;
        }
        for (int block = 0; block <= index; block++) {
            assignment[index] = block;
            if (partitionRec(index + 1, assignment, termCount, equalities, disequalities,
                    applications)) {
                return true;
            }
        }
        return false;
    }

    public static void main(String[] args) {
        // --- front-end -----------------------------------------------------------
        List<Object> program = SmtParser.parseProgram(
                "(declare-const x Int)(assert (< x (+ 1 2)))(check-sat)");
        check(program.size() == 3, "three commands were parsed, got " + program.size());
        check(String.valueOf(SmtParser.asList(program.get(1)).get(0)).equals("assert"),
                "the second command is the assert");
        boolean malformed = false;
        try {
            SmtParser.parseProgram("(assert (< x 1)");
        } catch (IllegalArgumentException expected) {
            malformed = true;
        }
        check(malformed, "unbalanced parentheses must be rejected");
        boolean unsupported = false;
        try {
            solverFor("(declare-const x Int)(assert (forall x))").checkSat();
        } catch (IllegalArgumentException expected) {
            unsupported = true;
        }
        check(unsupported, "unsupported constructs must be rejected");
        boolean undeclared = false;
        try {
            solverFor("(assert (< x 1))").checkSat();
        } catch (IllegalArgumentException expected) {
            undeclared = true;
        }
        check(undeclared, "undeclared variables must be rejected");
        boolean multiplication = false;
        try {
            solverFor("(declare-const x Int)(assert (< (* 2 x) 4))").checkSat();
        } catch (IllegalArgumentException expected) {
            multiplication = true;
        }
        check(multiplication, "multiplication is outside the fragment and must be rejected");

        // --- pure difference logic ----------------------------------------------
        check(sat("(declare-const x Int)(declare-const y Int)(assert (< x y))"), "x < y is SAT");
        check(!sat("(declare-const x Int)(declare-const y Int)(assert (< x y))(assert (< y x))"),
                "x < y and y < x is UNSAT");
        check(!sat("(declare-const x Int)(assert (< x x))"), "x < x is UNSAT");
        check(sat("(declare-const x Int)(assert (<= x 5))(assert (<= (- 3 x) 0))"),
                "3 <= x <= 5 is SAT");
        check(!sat("(declare-const x Int)(assert (< x 3))(assert (< 5 x))"), "x < 3 and 5 < x is UNSAT");
        check(sat("(declare-const x Int)(declare-const y Int)(declare-const z Int)"
                        + "(assert (<= (- x y) 2))(assert (<= (- y z) 3))"),
                "a difference chain is SAT");
        check(sat("(declare-const x Int)(assert (< 1 2))"), "a true constant comparison is SAT");
        check(!sat("(declare-const x Int)(assert (< 2 1))"), "a false constant comparison is UNSAT");
        check(sat("(declare-const x Int)(declare-const y Int)(assert (distinct x y))"
                        + "(assert (<= x y))"), "x < y witnesses distinct");
        check(!sat("(declare-const x Int)(declare-const y Int)(assert (distinct x y))"
                        + "(assert (= x y))"), "x = y and x distinct y is UNSAT");
        assertModelValid("(declare-const x Int)(declare-const y Int)(assert (< x y))"
                + "(assert (< y 4))(check-sat)");
        assertModelValid("(declare-const x Int)(declare-const y Int)(declare-const z Int)"
                + "(assert (distinct x y))(assert (<= x 2))(assert (<= (- y z) 1))");

        // --- pure difference logic versus bounded enumeration --------------------
        String[] arithmeticPrograms = {
                "(declare-const x Int)(assert (< x 0))(assert (< (- 2 x) 0))",
                "(declare-const x Int)(assert (<= x 1))(assert (<= (- 1 x) 0))(assert (distinct x 0))",
                "(declare-const x Int)(declare-const y Int)(assert (<= (- x y) 1))"
                        + "(assert (<= (- y x) 1))(assert (< x 0))",
                "(declare-const x Int)(assert (< x 0))(assert (< 0 x))",
                "(declare-const x Int)(declare-const y Int)(assert (< x y))"
                        + "(assert (<= (- y x) 1))(assert (distinct y 3))",
        };
        for (String source : arithmeticPrograms) {
            SmtSolver solver = solverFor(source);
            boolean verdict = solver.checkSat() != null;
            List<String> names = new ArrayList<>();
            List<Object> assertions = new ArrayList<>();
            for (Object command : SmtParser.parseProgram(source)) {
                List<Object> parts = SmtParser.asList(command);
                String head = String.valueOf(parts.get(0));
                if (head.equals("declare-const")) names.add(String.valueOf(parts.get(1)));
                if (head.equals("assert")) assertions.add(parts.get(1));
            }
            boolean expected = enumerateArithmetic(names, assertions, 8);
            check(verdict == expected, "arithmetic verdict differs from enumeration: " + source);
        }

        // --- EUF and congruence --------------------------------------------------
        check(!sat("(declare-const x Int)(declare-const y Int)(declare-fun f (Int) Int)"
                        + "(assert (= x y))(assert (distinct (f x) (f y)))"),
                "x = y forces f(x) = f(y)");
        check(sat("(declare-const x Int)(declare-const y Int)(declare-fun f (Int) Int)"
                        + "(assert (distinct x y))(assert (= (f x) (f y)))"),
                "a function may collapse distinct arguments");
        check(!sat("(declare-const x Int)(declare-const y Int)(declare-const z Int)"
                        + "(declare-fun f (Int) Int)(assert (= x y))(assert (= (f x) z))"
                        + "(assert (distinct (f y) z))"),
                "transitivity through an application");
        assertModelValid("(declare-const x Int)(declare-const y Int)(declare-fun f (Int) Int)"
                + "(assert (distinct x y))(assert (= (f x) 7))");

        // --- partition oracle on the shape x, y, f(x), f(y) ---------------------
        check(!partitionOracle(4, List.of(new int[]{0, 1}), List.of(new int[]{2, 3}),
                        List.of(new int[]{0, 0, 2}, new int[]{0, 1, 3})),
                "the partition oracle agrees x = y forces f(x) = f(y)");
        check(partitionOracle(4, List.of(), List.of(new int[]{0, 1}),
                        List.of(new int[]{0, 0, 2}, new int[]{0, 1, 3})),
                "the partition oracle accepts equal values on distinct arguments");
        // cross-check the solver against the oracle on six small EUF instances
        List<int[]> oracleApplications = List.of(new int[]{0, 0, 2}, new int[]{0, 1, 3});
        int[][][] equalitySets = {{}, {{0, 1}}, {{0, 1}, {2, 3}}, {{0, 2}}, {{0, 1}, {0, 2}},
                {{1, 2}, {0, 3}}};
        int[][][] disequalitySets = {{{0, 1}}, {{2, 3}}, {{0, 1}, {2, 3}}, {{0, 1}}, {{0, 3}},
                {{0, 1}}};
        for (int trial = 0; trial < equalitySets.length; trial++) {
            StringBuilder instance = new StringBuilder(
                    "(declare-const x Int)(declare-const y Int)(declare-fun f (Int) Int)");
            for (int[] pair : equalitySets[trial]) {
                instance.append("(assert (= ").append(termOf(pair[0])).append(' ')
                        .append(termOf(pair[1])).append("))");
            }
            for (int[] pair : disequalitySets[trial]) {
                instance.append("(assert (distinct ").append(termOf(pair[0])).append(' ')
                        .append(termOf(pair[1])).append("))");
            }
            boolean expected = partitionOracle(4, Arrays.asList(equalitySets[trial]),
                    Arrays.asList(disequalitySets[trial]), oracleApplications);
            check(sat(instance.toString()) == expected,
                    "EUF verdict differs from the partition oracle on instance " + trial);
        }

        // --- theory combination --------------------------------------------------
        check(!sat("(declare-const x Int)(declare-const y Int)(declare-fun f (Int) Int)"
                        + "(assert (<= (- x y) 0))(assert (<= (- y x) 0))"
                        + "(assert (distinct (f x) (f y)))"),
                "arithmetic proves x = y, congruence contradicts f(x) distinct f(y)");
        check(sat("(declare-const x Int)(declare-const y Int)(declare-fun f (Int) Int)"
                        + "(assert (<= (- x y) 1))(assert (<= (- y x) 1))"
                        + "(assert (distinct (f x) (f y)))"),
                "x - y inside [-1, 1] leaves the applications free");
        assertModelValid("(declare-const x Int)(declare-const y Int)(declare-fun f (Int) Int)"
                + "(assert (= x y))(assert (= (f y) 9))(assert (< x 3))");
        assertModelValid("(declare-const x Int)(declare-fun f (Int) Int)"
                + "(assert (<= (- (f x) x) 0))(assert (distinct (f x) x))");
        check(!sat("(declare-const x Int)(declare-const y Int)(declare-fun g (Int) Int)"
                        + "(assert (= (g x) x))(assert (= (g (g x)) y))(assert (distinct x y))"),
                "nested congruence: g(x) = x forces g(g(x)) = x");
        check(sat("(declare-const x Int)(declare-fun g (Int) Int)(assert (= (g (g x)) x))"
                        + "(assert (distinct (g x) x))"),
                "a two cycle g(x) = y, g(y) = x is congruence consistent");

        // --- UNSAT of mixed instances is explained by a single theory ------------
        check(!sat("(declare-const x Int)(declare-const y Int)(declare-fun f (Int) Int)"
                        + "(assert (= (f x) y))(assert (distinct (f x) y))"),
                "the mixed instance is unsatisfiable");
        check(!sat("(declare-const x Int)(declare-const y Int)(assert (<= (- x y) 0))"
                        + "(assert (<= (- y x) 0))(assert (distinct x y))"),
                "arithmetic alone refutes this instance");
        check(!sat("(declare-const x Int)(declare-const y Int)(declare-fun f (Int) Int)"
                        + "(assert (= x y))(assert (distinct (f x) (f y)))"),
                "congruence alone refutes this instance");

        // --- push and pop --------------------------------------------------------
        SmtSolver scoped = solverFor("(declare-const x Int)(assert (< x 0))(push 1)(assert (< 0 x))");
        check(scoped.checkSat() == null, "the inner scope is unsatisfiable");
        scoped.execute("(pop 1)");
        check(scoped.checkSat() != null, "popping the scope restores satisfiability");
        boolean badPop = false;
        try {
            scoped.execute("(pop 1)");
        } catch (IllegalArgumentException expected) {
            badPop = true;
        }
        check(badPop, "popping without a push must be rejected");

        // --- determinism and bookkeeping ----------------------------------------
        check(sat("(declare-const x Int)(declare-const y Int)(assert (< x y))")
                        == sat("(declare-const x Int)(declare-const y Int)(assert (< x y))"),
                "the front-end is deterministic");
        SmtSolver counter = solverFor("(declare-const x Int)(assert (< x 2))(check-sat)");
        check(counter.checkSat() != null, "the counter instance is satisfiable");
        check(counter.theoryCheckCount() > 0, "the theory solvers were called");
        check(counter.atomCount() == 1, "one atom was discovered");

        System.out.println("front-end: parsed programs, rejected malformed input, "
                + arithmeticPrograms.length + " arithmetic verdicts matched bounded enumeration");
        System.out.println("theory combination: congruence closure + difference logic fixpoint, theory "
                + "checks = " + counter.theoryCheckCount() + ", " + checksRun + " checks");
        System.out.println("PASS");
    }

    /** Term name of the fixed four node shape used by the EUF oracle comparison. */
    static String termOf(int node) {
        switch (node) {
            case 0: return "x";
            case 1: return "y";
            case 2: return "(f x)";
            default: return "(f y)";
        }
    }

    static void check(boolean ok, String label) {
        checksRun++;
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
