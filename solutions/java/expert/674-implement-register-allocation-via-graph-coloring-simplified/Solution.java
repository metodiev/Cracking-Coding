/** Task 674 - Implement register allocation via graph coloring (simplified). (Tier: Expert, CTCI Beyond — Compilers & Program Analysis)
 *
 * Simplification (stated explicitly): the input is straight-line three-address code rather than a CFG,
 * so liveness needs no fixpoint and no coalescing or precoloured registers are modelled. Everything else
 * is the real Chaitin-style algorithm.
 *
 * Points of the code are the gaps between statements. A variable is live at a point when some later
 * statement reads it before writing it, and two variables interfere when they are live at a common
 * point. Allocation is optimistic graph colouring: repeatedly remove a vertex of degree < K, and when
 * every remaining vertex has degree >= K remove the highest-degree one as a spill candidate anyway;
 * then pop the stack, colour with the smallest register unused by the coloured neighbours and mark a
 * vertex as spilled when all K registers are taken. Because a single spill candidate ordering can waste
 * registers on a K-colourable graph, the simplify/select pair is retried with a few deterministic node
 * orders and the run with the fewest spills wins.
 *
 * Verified: liveness against an independent forward scan ("read later before any write"), the
 * interference graph against the set of pairs simultaneously live at a point, properness of every
 * colouring (no two interfering variables share a register), the register bound of K, spill-free
 * allocation whenever the brute-force chromatic number fits in K, and - the strongest check - a machine
 * model with K registers plus memory runs the allocated code (reloading and spilling as dictated) and
 * must reproduce the naive interpreter's result for every program and every K, which fails if even one
 * interference edge is missing.
 */
import java.util.*;

final class Statement {
    static final int CONST = 0, COPY = 1, ADD = 2, SUB = 3, MUL = 4;
    final String dst, a, b;
    final int op, imm;

    Statement(int op, String dst, String a, String b, int imm) {
        this.op = op; this.dst = dst; this.a = a; this.b = b; this.imm = imm;
    }

    static Statement cst(String d, int k) { return new Statement(CONST, d, null, null, k); }
    static Statement cpy(String d, String s) { return new Statement(COPY, d, s, null, 0); }
    static Statement bin(int op, String d, String x, String y) { return new Statement(op, d, x, y, 0); }

    Set<String> uses() {
        Set<String> u = new LinkedHashSet<>();
        if (a != null) u.add(a);
        if (b != null) u.add(b);
        return u;
    }
}

final class RegProg {
    final List<Statement> stmts;
    final String result;

    RegProg(List<Statement> stmts, String result) { this.stmts = stmts; this.result = result; }

    Set<String> variables() {
        Set<String> vars = new LinkedHashSet<>();
        for (Statement s : stmts) {
            if (s.dst != null) vars.add(s.dst);
            vars.addAll(s.uses());
        }
        if (result != null) vars.add(result);
        return vars;
    }

    /** Points 0..n where point i is the gap before statement i and point n is the end. */
    List<Set<String>> liveAtPoints() {
        int n = stmts.size();
        List<Set<String>> live = new ArrayList<>();
        for (int i = 0; i <= n; i++) live.add(new LinkedHashSet<>());
        if (n > 0) {
            if (result != null) live.get(n).add(result);
            for (int i = n - 1; i >= 0; i--) {
                Set<String> set = new LinkedHashSet<>(live.get(i + 1));
                if (stmts.get(i).dst != null) set.remove(stmts.get(i).dst);
                set.addAll(stmts.get(i).uses());
                live.set(i, set);
            }
        }
        return live;
    }

    /** Oracle: v is live at point i when some statement j >= i reads v with no write of v before it. */
    boolean liveOracle(String v, int point) {
        for (int j = point; j < stmts.size(); j++) {
            if (stmts.get(j).uses().contains(v)) return true;
            if (v.equals(stmts.get(j).dst)) return false;
        }
        return result != null && result.equals(v) && point >= stmts.size();
    }

    /** Interference: variables live at a common point (the exact definition, not the def/use shortcut). */
    Map<String, Set<String>> interference() {
        Map<String, Set<String>> adj = new LinkedHashMap<>();
        for (String v : variables()) adj.put(v, new LinkedHashSet<>());
        for (Set<String> live : liveAtPoints()) {
            List<String> vars = new ArrayList<>(live);
            for (int i = 0; i < vars.size(); i++)
                for (int j = i + 1; j < vars.size(); j++) {
                    adj.get(vars.get(i)).add(vars.get(j));
                    adj.get(vars.get(j)).add(vars.get(i));
                }
        }
        return adj;
    }

    int runNaive(int[] ins) {
        Map<String, Integer> env = new LinkedHashMap<>();
        env.put("in0", ins[0]);
        env.put("in1", ins[1]);
        for (Statement s : stmts) {
            int value = switch (s.op) {
                case Statement.CONST -> s.imm;
                case Statement.COPY -> env.get(s.a);
                case Statement.ADD -> env.get(s.a) + env.get(s.b);
                case Statement.SUB -> env.get(s.a) - env.get(s.b);
                default -> env.get(s.a) * env.get(s.b);
            };
            env.put(s.dst, value);
        }
        return result == null ? 0 : env.get(result);
    }
}

/** Optimistic colouring with a spill fallback, plus the chromatic number by exhaustive search. */
final class Allocator {
    final Map<String, Integer> color = new LinkedHashMap<>();     // -1 marks a spilled variable
    final Set<String> spilled = new LinkedHashSet<>();
    int colorsUsed, restarts, simplifySpills;

    static Allocator allocate(RegProg prog, int k, int variant) {
        Set<String> vars = prog.variables();
        Map<String, Set<String>> adj = prog.interference();
        Allocator best = null;
        for (int restart = 0; restart < 6; restart++) {
            Allocator trial = new Allocator().colour(vars, adj, k, variant + restart);
            trial.restarts = restart + 1;
            if (best == null || trial.spilled.size() < best.spilled.size()) best = trial;
            if (best.spilled.isEmpty()) break;
        }
        return best;
    }

    Allocator colour(Set<String> vars, Map<String, Set<String>> adj, int k, int variant) {
        List<String> order = new ArrayList<>(vars);
        for (int i = order.size() - 1; i > 0; i--) {              // deterministic shuffle of the tie-breaks
            int j = (int) ((0x9E3779B9L * (variant + 1) + 7L * i) % (i + 1));
            String t = order.get(i); order.set(i, order.get(j)); order.set(j, t);
        }
        Map<String, Integer> degree = new LinkedHashMap<>();
        Set<String> remaining = new LinkedHashSet<>(vars);
        for (String v : vars) degree.put(v, adj.get(v).size());
        List<String> stack = new ArrayList<>();
        while (!remaining.isEmpty()) {
            String pick = null;
            for (String v : order) if (remaining.contains(v) && degree.get(v) < k) { pick = v; break; }
            if (pick == null) {
                for (String v : order) if (remaining.contains(v) && (pick == null || degree.get(v) > degree.get(pick))) pick = v;
                simplifySpills++;
            }
            stack.add(pick);
            remaining.remove(pick);
            for (String n : adj.get(pick)) if (remaining.contains(n)) degree.merge(n, -1, Integer::sum);
        }
        for (int i = stack.size() - 1; i >= 0; i--) {
            String v = stack.get(i);
            Set<Integer> taken = new LinkedHashSet<>();
            for (String n : adj.get(v)) {
                Integer c = color.get(n);
                if (c != null && c >= 0) taken.add(c);
            }
            int chosen = -1;
            for (int c = 0; c < k; c++) if (!taken.contains(c)) { chosen = c; break; }
            color.put(v, chosen);
            if (chosen < 0) spilled.add(v);
            else colorsUsed = Math.max(colorsUsed, chosen + 1);
        }
        return this;
    }

    /** Chromatic number by exhaustive backtracking (only used on the small test programs). */
    static int chromatic(RegProg prog) {
        List<String> list = new ArrayList<>(prog.variables());
        Map<String, Set<String>> adj = prog.interference();
        for (int k = 1; k <= list.size(); k++) if (colourWith(list, adj, k, 0, new HashMap<>())) return k;
        return list.size();
    }

    static boolean colourWith(List<String> vars, Map<String, Set<String>> adj, int k, int idx, Map<String, Integer> assign) {
        if (idx == vars.size()) return true;
        String v = vars.get(idx);
        for (int c = 0; c < k; c++) {
            boolean ok = true;
            for (String n : adj.get(v)) if (assign.getOrDefault(n, -1) == c) ok = false;
            if (!ok) continue;
            assign.put(v, c);
            if (colourWith(vars, adj, k, idx + 1, assign)) return true;
            assign.remove(v);
        }
        return false;
    }

    /** Machine model: K registers plus a memory cell per spilled variable. */
    static int runMachine(RegProg prog, Allocator alloc, int k, int[] ins) {
        int[] reg = new int[Math.max(1, k)];
        Map<String, Integer> mem = new LinkedHashMap<>();
        Set<String> vars = prog.variables();
        if (vars.contains("in0")) write("in0", ins[0], alloc, reg, mem);     // parameters live where they were put
        if (vars.contains("in1")) write("in1", ins[1], alloc, reg, mem);
        for (Statement s : prog.stmts) {
            int value = switch (s.op) {
                case Statement.CONST -> s.imm;
                case Statement.COPY -> read(s.a, alloc, reg, mem);
                case Statement.ADD -> read(s.a, alloc, reg, mem) + read(s.b, alloc, reg, mem);
                case Statement.SUB -> read(s.a, alloc, reg, mem) - read(s.b, alloc, reg, mem);
                default -> read(s.a, alloc, reg, mem) * read(s.b, alloc, reg, mem);
            };
            write(s.dst, value, alloc, reg, mem);
        }
        return prog.result == null ? 0 : read(prog.result, alloc, reg, mem);
    }

    static int read(String v, Allocator alloc, int[] reg, Map<String, Integer> mem) {
        Integer c = alloc.color.get(v);
        return c == null || c < 0 ? mem.getOrDefault(v, 0) : reg[c];
    }

    static void write(String v, int value, Allocator alloc, int[] reg, Map<String, Integer> mem) {
        Integer c = alloc.color.get(v);
        if (c == null || c < 0) mem.put(v, value);
        else reg[c] = value;
    }
}

public class Solution {
    static long state = 0x4EED5EEDL;

    static long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static RegProg simple() {
        List<Statement> s = new ArrayList<>();
        s.add(Statement.cpy("a", "in0"));
        s.add(Statement.cpy("b", "in1"));
        s.add(Statement.bin(Statement.ADD, "c", "a", "b"));
        s.add(Statement.bin(Statement.MUL, "d", "c", "a"));
        return new RegProg(s, "d");
    }

    static RegProg pressure() {
        List<Statement> s = new ArrayList<>();
        s.add(Statement.cpy("a", "in0"));
        s.add(Statement.cpy("b", "in1"));
        s.add(Statement.bin(Statement.ADD, "c", "a", "b"));
        s.add(Statement.bin(Statement.SUB, "d", "a", "b"));
        s.add(Statement.bin(Statement.MUL, "e", "c", "d"));
        s.add(Statement.bin(Statement.ADD, "f", "c", "d"));
        s.add(Statement.bin(Statement.ADD, "g", "e", "f"));
        return new RegProg(s, "g");
    }

    /** Short live ranges follow long ones, so registers can be reused after a variable dies. */
    static RegProg reuse() {
        List<Statement> s = new ArrayList<>();
        s.add(Statement.cpy("long1", "in0"));
        s.add(Statement.cpy("t1", "in1"));
        s.add(Statement.bin(Statement.ADD, "t2", "t1", "t1"));
        s.add(Statement.cpy("t3", "in0"));
        s.add(Statement.bin(Statement.MUL, "t4", "t3", "t2"));
        s.add(Statement.bin(Statement.ADD, "out", "long1", "t4"));
        return new RegProg(s, "out");
    }

    static RegProg longRange() {
        List<Statement> s = new ArrayList<>();
        s.add(Statement.cpy("keep", "in0"));
        s.add(Statement.cpy("x", "in1"));
        s.add(Statement.bin(Statement.ADD, "x", "x", "in0"));
        s.add(Statement.bin(Statement.MUL, "x", "x", "x"));
        s.add(Statement.bin(Statement.SUB, "x", "x", "keep"));
        return new RegProg(s, "x");
    }

    static RegProg single() {
        List<Statement> s = new ArrayList<>();
        s.add(Statement.cpy("only", "in0"));
        return new RegProg(s, "only");
    }

    static RegProg chain(int length) {
        List<Statement> s = new ArrayList<>();
        s.add(Statement.cpy("v0", "in0"));
        for (int i = 1; i <= length; i++)
            s.add(Statement.bin(i % 2 == 0 ? Statement.ADD : Statement.MUL, "v" + i, "v" + (i - 1), "in1"));
        return new RegProg(s, "v" + length);
    }

    static RegProg empty() { return new RegProg(new ArrayList<>(), null); }

    static List<RegProg> programs() {
        return List.of(simple(), pressure(), reuse(), longRange(), single(), chain(6), empty());
    }

    static String[] names() { return new String[]{"simple", "pressure", "reuse", "longRange", "single", "chain6", "empty"}; }

    public static void main(String[] args) {
        RegProg p = simple();
        List<Set<String>> live = p.liveAtPoints();
        check(live.get(0).equals(new TreeSet<>(List.of("in0", "in1")))
                        && live.get(1).equals(new TreeSet<>(List.of("a", "in1")))
                        && live.get(2).equals(new TreeSet<>(List.of("a", "b"))),
                "live sets of the simple program, got " + live);
        check(live.get(3).equals(new TreeSet<>(List.of("a", "c"))) && live.get(4).equals(new TreeSet<>(List.of("d"))),
                "a and c are live before the last multiply and only d afterwards");
        Map<String, Set<String>> adj = p.interference();
        check(adj.get("a").equals(new TreeSet<>(List.of("b", "c", "in1"))) && adj.get("b").equals(new TreeSet<>(List.of("a"))),
                "a interferes with b, c and in1 while b interferes only with a, got " + adj);
        check(!adj.get("c").contains("d") && !adj.get("d").contains("a"),
                "d is defined last, so it shares no register requirement with a");

        int livePairs = 0, oracleMismatch = 0;
        for (RegProg prog : programs()) {
            List<Set<String>> points = prog.liveAtPoints();
            for (String v : prog.variables())
                for (int i = 0; i <= prog.stmts.size(); i++) {
                    livePairs++;
                    if (points.get(i).contains(v) != prog.liveOracle(v, i)) oracleMismatch++;
                }
            Map<String, Set<String>> graph = prog.interference();
            Set<String> brute = new LinkedHashSet<>();
            for (Set<String> set : points) {
                List<String> vars = new ArrayList<>(set);
                for (int i = 0; i < vars.size(); i++)
                    for (int j = i + 1; j < vars.size(); j++) brute.add(vars.get(i) + "|" + vars.get(j));
            }
            for (Map.Entry<String, Set<String>> e : graph.entrySet()) {
                check(!e.getValue().contains(e.getKey()), "no self interference for " + e.getKey());
                for (String n : e.getValue()) check(graph.get(n).contains(e.getKey()), "interference is symmetric");
            }
            for (String a : prog.variables())
                for (String b : prog.variables()) {
                    boolean edge = graph.get(a).contains(b);
                    boolean expected = brute.contains(a + "|" + b) || brute.contains(b + "|" + a);
                    if (edge != expected) throw new AssertionError("FAILED: edge " + a + "-" + b + " disagreement");
                }
        }
        check(oracleMismatch == 0, "liveness matches the forward-scan oracle on " + livePairs + " (variable, point) pairs");

        int[] ins = {3, 5};
        int programsWithSpills = 0, executions = 0, spillFreeWhenFeasible = 0, feasible = 0, totalSpills = 0;
        int[] ks = {1, 2, 3, 4};
        for (RegProg prog : programs()) {
            int chi = Allocator.chromatic(prog);
            for (int k : ks) {
                Allocator alloc = Allocator.allocate(prog, k, 0);
                executions++;
                totalSpills += alloc.spilled.size();
                if (!alloc.spilled.isEmpty()) programsWithSpills++;
                if (chi <= k) {
                    feasible++;
                    if (alloc.spilled.isEmpty()) spillFreeWhenFeasible++;
                }
                for (String v : prog.variables()) {
                    Integer c = alloc.color.get(v);
                    check(c != null && (c < 0 || c < k), "every variable is placed in one of " + k + " registers");
                    if (c != null && c >= 0)
                        for (String n : prog.interference().get(v))
                            check(!c.equals(alloc.color.get(n)), "register clash between " + v + " and " + n + " at K=" + k);
                }
                check(alloc.colorsUsed <= k, "the allocator never uses more than " + k + " registers");
                check(Allocator.runMachine(prog, alloc, k, ins) == prog.runNaive(ins),
                        "machine model reproduces the interpreter for " + prog.result + " at K=" + k
                                + ": " + Allocator.runMachine(prog, alloc, k, ins) + " vs " + prog.runNaive(ins));
            }
        }
        check(spillFreeWhenFeasible == feasible,
                "no spill whenever the chromatic number fits K (" + spillFreeWhenFeasible + "/" + feasible + ")");
        check(programsWithSpills > 0 && totalSpills > 0,
                "tighter register budgets really do force spills (" + totalSpills + " spilling placements)");
        check(executions == programs().size() * ks.length, "every program was allocated at four register counts");

        RegProg busy = pressure();
        check(Allocator.chromatic(busy) == 3, "the pressure program needs 3 colours, got " + Allocator.chromatic(busy));
        Allocator tight = Allocator.allocate(busy, 2, 0);
        check(!tight.spilled.isEmpty(), "with 2 registers the triangle must spill something: " + tight.spilled);
        check(Allocator.allocate(busy, 3, 0).spilled.isEmpty(), "with 3 registers nothing spills");

        RegProg shared = reuse();
        Allocator two = Allocator.allocate(shared, 2, 0);
        int placed = two.color.size() - two.spilled.size();
        check(two.colorsUsed <= 2 && placed > two.colorsUsed,
                "registers are reused across dead variables: " + placed + " variables in " + two.colorsUsed + " registers");
        Allocator again = Allocator.allocate(shared, 2, 0);
        check(again.color.equals(two.color) && again.spilled.equals(two.spilled), "allocation is deterministic");

        RegProg none = empty();
        Allocator noneAlloc = Allocator.allocate(none, 1, 0);
        check(noneAlloc.color.isEmpty() && noneAlloc.spilled.isEmpty() && noneAlloc.colorsUsed == 0
                        && Allocator.chromatic(none) == 0,
                "an empty program needs no registers");
        RegProg one = single();
        Allocator oneAlloc = Allocator.allocate(one, 1, 0);
        check(oneAlloc.spilled.isEmpty() && oneAlloc.colorsUsed == 1 && oneAlloc.color.get("only") == 0,
                "one variable fits in one register");

        int randomChecks = 0, randomMismatch = 0;
        for (int trial = 0; trial < 40; trial++) {
            RegProg r = chain(2 + (int) (nextLong() >>> 40) % 5);
            for (int k = 1; k <= 3; k++) {
                Allocator alloc = Allocator.allocate(r, k, trial);
                for (int t = 0; t < 6; t++) {
                    int[] in = {(int) ((nextLong() >>> 41) % 11) - 5, (int) ((nextLong() >>> 42) % 7) - 3};
                    randomChecks++;
                    if (Allocator.runMachine(r, alloc, k, in) != r.runNaive(in)) randomMismatch++;
                }
            }
        }
        check(randomMismatch == 0, "spill code is correct on " + randomChecks + " random runs of chained programs");

        System.out.println("interference : " + livePairs + " (variable, point) liveness facts agree with the oracle");
        System.out.println("allocation   : " + executions + " allocations, " + totalSpills + " spilled variables, "
                + spillFreeWhenFeasible + "/" + feasible + " spill-free when K >= chi");
        System.out.println("machine model: " + (executions + randomChecks) + " end-to-end runs match the interpreter");
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
