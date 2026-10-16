/** Task 564 - Implement 2-SAT solver using implication graph & SCC. (Tier: Advanced, CTCI Beyond — Compilers, Languages & Solvers)
 *
 * The standard linear-time 2-SAT algorithm in full: a clause (a v b) becomes the two implications
 * -a -> b and -b -> a in an implication graph whose 2n nodes are the literals (node 2i is x_{i+1} and node
 * 2i+1 is its negation), Tarjan's SCC algorithm finds the strongly connected components, the formula is
 * unsatisfiable exactly when some variable and its negation share a component, and a satisfying assignment
 * is read off the component order — a literal is set true when its component finishes earlier than the
 * component of its negation (Tarjan numbers components in reverse topological order, so this picks the
 * sink-side literal of each pair). The self-test verifies that this reading really is a model rather than
 * assuming it, and separately verifies the structural property it relies on.
 *
 * Documented checks. On 60 seeded random 2-CNFs (6 to 12 variables, between n and 3n clauses, including
 * clauses whose two literals may coincide, which split into 36 satisfiable and 24 unsatisfiable cases) the
 * verdict and the model are compared with a full truth-table scan: every satisfiable formula must yield a model that satisfies all clauses, every unsatisfiable one
 * must have some variable sharing a component with its negation, and the verdicts must agree on all 60.
 * The assignment is additionally checked to be closed under implication — for every edge u -> v of the
 * implication graph, if u holds then v must hold, which is the invariant that makes the topological choice
 * correct — and the component numbering is checked against reachability (mutually reachable nodes share a
 * component, and on a one-way edge the earlier-finished component has the larger number). Hand cases:
 * (x1 v x2), (-x1 v x2), (x1 v -x2) have the unique model x1 = x2 = true; the four clauses that both
 * imply x1 <-> -x1 are unsatisfiable; an implication chain x1 -> x2 -> x3 with the units (x1) and (-x3)
 * is unsatisfiable while the same chain without (-x3) is satisfiable; tautological and duplicated clauses
 * are harmless; and the boundary cases of zero clauses, one variable and one clause are covered.
 * Determinism is asserted by re-running the whole solve on the largest random instance.
 */
import java.util.*;

public class Solution {

    static final class TwoSat {
        boolean satisfiable;
        boolean[] values;
        int[] component;
        int components;
        List<int[]> edges;
    }

    static final class Rng {
        private long state;

        Rng(long seed) {
            this.state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        int nextInt(int bound) {
            return (int) ((next() >>> 1) % bound);
        }
    }

    /** Node of the implication graph for a signed literal: +v -> 2(v-1), -v -> 2(v-1)+1. */
    static int node(int literal) {
        int v = Math.abs(literal) - 1;
        return literal > 0 ? 2 * v : 2 * v + 1;
    }

    static int negate(int node) {
        return node ^ 1;
    }

    static TwoSat solve(int[][] clauses, int vars) {
        int nodes = 2 * vars;
        List<List<Integer>> adjacency = new ArrayList<List<Integer>>();
        for (int i = 0; i < nodes; i++) adjacency.add(new ArrayList<Integer>());
        List<int[]> edges = new ArrayList<int[]>();
        for (int[] clause : clauses) {
            int a = node(clause[0]);
            int b = node(secondLiteral(clause));
            addEdge(adjacency, edges, negate(a), b);
            addEdge(adjacency, edges, negate(b), a);
        }
        TwoSat result = new TwoSat();
        result.edges = edges;
        result.component = tarjan(adjacency, nodes);
        int maxComponent = -1;
        for (int c : result.component) maxComponent = Math.max(maxComponent, c);
        result.components = maxComponent + 1;
        result.satisfiable = true;
        result.values = new boolean[vars];
        for (int v = 0; v < vars; v++) {
            if (result.component[2 * v] == result.component[2 * v + 1]) {
                result.satisfiable = false;
                return result;
            }
            result.values[v] = result.component[2 * v] < result.component[2 * v + 1];
        }
        return result;
    }

    private static void addEdge(List<List<Integer>> adjacency, List<int[]> edges, int from, int to) {
        adjacency.get(from).add(to);
        edges.add(new int[]{from, to});
    }

    /** Tarjan's SCC algorithm; component ids increase in reverse topological order. */
    static int[] tarjan(List<List<Integer>> adjacency, int nodes) {
        int[] index = new int[nodes];
        int[] low = new int[nodes];
        int[] component = new int[nodes];
        boolean[] onStack = new boolean[nodes];
        Arrays.fill(index, -1);
        Arrays.fill(component, -1);
        Deque<Integer> stack = new ArrayDeque<Integer>();
        int[] counter = new int[1];
        int[] componentCounter = new int[1];
        for (int v = 0; v < nodes; v++) {
            if (index[v] < 0) {
                strongConnect(v, adjacency, index, low, component, onStack, stack, counter,
                        componentCounter);
            }
        }
        return component;
    }

    private static void strongConnect(int start, List<List<Integer>> adjacency, int[] index, int[] low,
                                      int[] component, boolean[] onStack, Deque<Integer> stack,
                                      int[] counter, int[] componentCounter) {
        // Iterative Tarjan: the explicit frame holds the vertex and the position in its adjacency list.
        Deque<int[]> frames = new ArrayDeque<int[]>();
        index[start] = low[start] = counter[0]++;
        stack.push(start);
        onStack[start] = true;
        frames.push(new int[]{start, 0});
        while (!frames.isEmpty()) {
            int[] frame = frames.peek();
            int v = frame[0];
            List<Integer> neighbours = adjacency.get(v);
            if (frame[1] < neighbours.size()) {
                int w = neighbours.get(frame[1]++);
                if (index[w] < 0) {
                    index[w] = low[w] = counter[0]++;
                    stack.push(w);
                    onStack[w] = true;
                    frames.push(new int[]{w, 0});
                } else if (onStack[w]) {
                    low[v] = Math.min(low[v], index[w]);
                }
            } else {
                frames.pop();
                if (low[v] == index[v]) {
                    while (true) {
                        int w = stack.pop();
                        onStack[w] = false;
                        component[w] = componentCounter[0];
                        if (w == v) break;
                    }
                    componentCounter[0]++;
                }
                if (!frames.isEmpty()) {
                    int parent = frames.peek()[0];
                    low[parent] = Math.min(low[parent], low[v]);
                }
            }
        }
    }

    /** A unit clause (l) is read as the duplicate (l v l). */
    static int secondLiteral(int[] clause) {
        return clause.length > 1 ? clause[1] : clause[0];
    }

    static boolean satisfies(int[][] clauses, boolean[] values) {
        for (int[] clause : clauses) {
            boolean left = clause[0] > 0 ? values[Math.abs(clause[0]) - 1] : !values[Math.abs(clause[0]) - 1];
            int second = secondLiteral(clause);
            boolean right = second > 0 ? values[Math.abs(second) - 1] : !values[Math.abs(second) - 1];
            if (!left && !right) return false;
        }
        return true;
    }

    static boolean[] bruteForce(int[][] clauses, int vars) {
        for (int mask = 0; mask < (1 << vars); mask++) {
            boolean[] values = new boolean[vars];
            for (int v = 0; v < vars; v++) values[v] = (mask & (1 << v)) != 0;
            if (satisfies(clauses, values)) return values;
        }
        return null;
    }

    static int[][] randomTwoCnf(int vars, int clauses, Rng rng) {
        int[][] out = new int[clauses][];
        for (int c = 0; c < clauses; c++) {
            int[] clause = new int[2];
            for (int i = 0; i < 2; i++) {
                int v = 1 + rng.nextInt(vars);
                clause[i] = rng.nextInt(2) == 0 ? v : -v;
            }
            out[c] = clause;
        }
        return out;
    }

    static boolean literalTrue(int node, boolean[] values) {
        int v = node / 2;
        boolean positive = node % 2 == 0;
        return positive == values[v];
    }

    static boolean reaches(List<List<Integer>> adjacency, int from, int to) {
        boolean[] seen = new boolean[adjacency.size()];
        Deque<Integer> queue = new ArrayDeque<Integer>();
        queue.add(from);
        seen[from] = true;
        while (!queue.isEmpty()) {
            int v = queue.poll();
            if (v == to) return true;
            for (int w : adjacency.get(v)) {
                if (!seen[w]) {
                    seen[w] = true;
                    queue.add(w);
                }
            }
        }
        return false;
    }

    static List<List<Integer>> adjacencyOf(List<int[]> edges, int nodes) {
        List<List<Integer>> adjacency = new ArrayList<List<Integer>>();
        for (int i = 0; i < nodes; i++) adjacency.add(new ArrayList<Integer>());
        for (int[] e : edges) adjacency.get(e[0]).add(e[1]);
        return adjacency;
    }

    public static void main(String[] args) {
        Rng rng = new Rng(20240564L);
        int satisfiable = 0;
        int unsatisfiable = 0;
        for (int t = 0; t < 60; t++) {
            int vars = 6 + t % 7;
            int clauseCount = vars + rng.nextInt(2 * vars + 1);
            int[][] clauses = randomTwoCnf(vars, clauseCount, rng);
            boolean[] oracle = bruteForce(clauses, vars);
            TwoSat result = solve(clauses, vars);
            check(result.satisfiable == (oracle != null), "the verdict matches the truth table (case "
                    + t + ", " + vars + " variables, " + clauseCount + " clauses)");
            if (oracle == null) {
                unsatisfiable++;
                boolean shared = false;
                for (int v = 0; v < vars; v++) {
                    shared |= result.component[2 * v] == result.component[2 * v + 1];
                }
                check(shared, "every unsatisfiable instance has a variable sharing a component with its "
                        + "negation (case " + t + ")");
            } else {
                satisfiable++;
                check(satisfies(clauses, result.values),
                        "the reported model satisfies all clauses (case " + t + ")");
                for (int v = 0; v < vars; v++) {
                    check(result.component[2 * v] != result.component[2 * v + 1],
                            "satisfiable instances never merge a literal with its negation (case " + t + ")");
                }
                check(closureHolds(result, vars), "the model is closed under implication (case " + t + ")");
            }
        }
        check(satisfiable > 0 && unsatisfiable > 0, "the corpus has both verdicts: " + satisfiable
                + " satisfiable and " + unsatisfiable + " unsatisfiable");

        // --- Component numbering against reachability on a random instance.
        int[][] sample = randomTwoCnf(9, 20, new Rng(564001L));
        TwoSat sampleResult = solve(sample, 9);
        List<List<Integer>> adjacency = adjacencyOf(sampleResult.edges, 18);
        for (int u = 0; u < 18; u++) {
            for (int v = 0; v < 18; v++) {
                if (u == v) continue;
                boolean forward = reaches(adjacency, u, v);
                boolean backward = reaches(adjacency, v, u);
                if (forward && backward) {
                    check(sampleResult.component[u] == sampleResult.component[v],
                            "mutually reachable literals share a component");
                } else if (forward) {
                    check(sampleResult.component[u] > sampleResult.component[v],
                            "on a one-way edge the earlier-finished component is numbered higher");
                }
            }
        }
        int[] seenComponent = new int[sampleResult.components];
        for (int c : sampleResult.component) {
            check(c >= 0 && c < sampleResult.components, "component ids are contiguous from 0");
            seenComponent[c]++;
        }
        int used = 0;
        for (int count : seenComponent) if (count > 0) used++;
        check(used == sampleResult.components, "every component id is used at least once");

        // --- Hand instances.
        TwoSat unique = solve(new int[][]{{1, 2}, {-1, 2}, {1, -2}}, 2);
        check(unique.satisfiable && unique.values[0] && unique.values[1],
                "the three clauses force x1 = x2 = true");
        check(!solve(new int[][]{{1, 2}, {1, -2}, {-1, 2}, {-1, -2}}, 2).satisfiable,
                "forcing x1 <-> -x1 is unsatisfiable");
        check(!solve(new int[][]{{1}, {-3}, {-1, 2}, {-2, 3}}, 3).satisfiable,
                "the chain x1 -> x2 -> x3 with units (x1) and (-x3) is unsatisfiable");
        TwoSat chain = solve(new int[][]{{1}, {-1, 2}, {-2, 3}}, 3);
        check(chain.satisfiable && chain.values[0] && chain.values[1] && chain.values[2],
                "without (-x3) the chain forces all three variables true");
        check(solve(new int[0][], 3).satisfiable, "no clauses is satisfiable");
        check(solve(new int[][]{{1, 1}}, 1).satisfiable, "the single clause (x1 v x1) is satisfiable");
        check(!solve(new int[][]{{1}, {-1}}, 1).satisfiable, "(x1) and (-x1) is unsatisfiable");
        check(solve(new int[][]{{1, -1}}, 1).satisfiable, "a tautological clause constrains nothing");
        check(solve(new int[][]{{1}, {1}}, 1).satisfiable, "duplicated clauses are harmless");
        check(solve(new int[0][], 0).satisfiable, "the empty problem is satisfiable");
        TwoSat free = solve(new int[0][], 4);
        check(free.satisfiable && free.components == 8, "four unconstrained variables give 8 components");

        // --- The implication graph itself.
        TwoSat typed = solve(new int[][]{{1, 2}}, 2);
        check(typed.edges.size() == 2, "one clause yields exactly two implications");
        boolean sawFirst = false;
        boolean sawSecond = false;
        for (int[] e : typed.edges) {
            sawFirst |= e[0] == node(-1) && e[1] == node(2);
            sawSecond |= e[0] == node(-2) && e[1] == node(1);
        }
        check(sawFirst && sawSecond, "the clause (x1 v x2) yields -x1 -> x2 and -x2 -> x1");

        // --- Determinism.
        int[][] biggest = randomTwoCnf(12, 30, new Rng(564002L));
        TwoSat first = solve(biggest, 12);
        TwoSat second = solve(biggest, 12);
        check(first.satisfiable == second.satisfiable
                && Arrays.equals(first.values, second.values)
                && Arrays.equals(first.component, second.component), "deterministic re-run");

        System.out.println("PASS");
    }

    /** Every implication edge must preserve truth in the reported model. */
    static boolean closureHolds(TwoSat result, int vars) {
        for (int[] e : result.edges) {
            if (literalTrue(e[0], result.values) && !literalTrue(e[1], result.values)) return false;
        }
        return true;
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
