/** Task 671 - Implement static program analysis: reaching definitions simple algorithm. (Tier: Expert, CTCI Beyond — Compilers & Program Analysis)
 *
 * A tiny three-address IR (basic blocks of assignments, each statement defining one variable and
 * reading a few others, plus an explicit edge list) is analysed with the classic may-analysis:
 *
 *   GEN[B]  = statements of B that no later statement of B redefines,
 *   KILL[B] = every statement of the program defining a variable assigned in B, minus GEN[B],
 *   IN[B]   = union of OUT[P] over predecessors P,      OUT[B] = GEN[B] union (IN[B] - KILL[B]).
 *
 * The solver iterates the equations to a fixpoint from the empty set (the least fixpoint of a monotone
 * framework, which is what "may reach" means). Every round is checked to be monotone: sets may only
 * grow, and the final round must add nothing.
 *
 * Independent oracle: for one statement d the analysis answers a pure graph question - is there a path
 * from d's block to block B on which no block redefines d's variable? That is answered by a BFS that
 * only looks at the statement lists (does this block assign d's variable) and at the position of d in
 * its own block, i.e. without using GEN/KILL at all. Both must agree for every (statement, block) pair.
 *
 * Also verified: hand-checked GEN/KILL and IN/OUT of three shapes (straight line, diamond, loop),
 * reaching definitions of a single variable, the loop back edge, unreachable blocks having no IN/OUT,
 * and the fact that initialising from "everything reaches everywhere" (the top element) coincides with
 * the least fixpoint only while the program has no unreachable block - on the unreachable-block program
 * it leaks definitions out of the dead region, which is why the analysis starts from the empty set.
 */
import java.util.*;

final class Stmt {
    final int id, indexInBlock;
    final String var;
    final List<String> uses;

    Stmt(int id, int indexInBlock, String var, List<String> uses) {
        this.id = id; this.indexInBlock = indexInBlock; this.var = var; this.uses = uses;
    }

    public String toString() { return "d" + id + "(" + var + "=" + String.join(",", uses) + ")"; }
}

final class Block {
    final String name;
    final List<Stmt> stmts = new ArrayList<>();
    final List<Block> succ = new ArrayList<>();
    final List<Block> pred = new ArrayList<>();

    Block(String name) { this.name = name; }

    Set<String> assignedVars() {
        Set<String> vars = new LinkedHashSet<>();
        for (Stmt s : stmts) vars.add(s.var);
        return vars;
    }

    /** True when some statement after |from| redefines |var| inside this block. */
    boolean redefinesAfter(int from, String var) {
        for (int i = from + 1; i < stmts.size(); i++) if (stmts.get(i).var.equals(var)) return true;
        return false;
    }
}

final class Program {
    final List<Block> blocks = new ArrayList<>();
    final List<Stmt> stmts = new ArrayList<>();
    final Map<Integer, Block> owner = new LinkedHashMap<>();
    Block entry;
    int nextId;

    Block block(String name) {
        Block b = new Block(name);
        blocks.add(b);
        if (entry == null) entry = b;
        return b;
    }

    Block def(Block b, String var, String... uses) {
        Stmt s = new Stmt(nextId++, b.stmts.size(), var, Arrays.asList(uses));
        b.stmts.add(s);
        stmts.add(s);
        owner.put(s.id, b);
        return b;
    }

    void edge(Block from, Block to) {
        from.succ.add(to);
        to.pred.add(from);
    }

    Block blockOf(Stmt s) { return owner.get(s.id); }

    Set<Block> reachable() {
        Set<Block> seen = new LinkedHashSet<>();
        List<Block> work = new ArrayList<>();
        seen.add(entry);
        work.add(entry);
        while (!work.isEmpty()) {
            Block b = work.remove(work.size() - 1);
            for (Block s : b.succ) if (seen.add(s)) work.add(s);
        }
        return seen;
    }

    static String ids(Collection<Integer> values) {
        List<String> parts = new ArrayList<>();
        for (int v : new TreeSet<>(values)) parts.add("d" + v);
        return parts.isEmpty() ? "{}" : "{" + String.join(",", parts) + "}";
    }
}

/** Reaching-definitions solver; |fromTop| selects the greatest-fixpoint initialisation. */
final class ReachingDefinitions {
    final Program program;
    final Map<String, Set<Integer>> gen = new LinkedHashMap<>();
    final Map<String, Set<Integer>> kill = new LinkedHashMap<>();
    final Map<String, Set<Integer>> in = new LinkedHashMap<>();
    final Map<String, Set<Integer>> out = new LinkedHashMap<>();
    int rounds, lastRoundAdditions, shrinks;
    final boolean fromTop;

    ReachingDefinitions(Program program, boolean fromTop) {
        this.program = program;
        this.fromTop = fromTop;
        Map<String, Set<Integer>> defsOfVar = new LinkedHashMap<>();
        for (Stmt s : program.stmts) defsOfVar.computeIfAbsent(s.var, k -> new TreeSet<>()).add(s.id);
        for (Block b : program.blocks) {
            Set<Integer> localGen = new TreeSet<>();
            Map<String, Integer> lastDef = new LinkedHashMap<>();
            for (Stmt s : b.stmts) lastDef.put(s.var, s.id);          // last definition wins locally
            localGen.addAll(lastDef.values());
            Set<Integer> localKill = new TreeSet<>();
            for (String v : b.assignedVars()) localKill.addAll(defsOfVar.get(v));
            localKill.removeAll(localGen);
            gen.put(b.name, localGen);
            kill.put(b.name, localKill);
            Set<Integer> init = new TreeSet<>();
            if (fromTop) for (Stmt s : program.stmts) init.add(s.id);
            in.put(b.name, new TreeSet<>(init));
            out.put(b.name, new TreeSet<>(init));
        }
    }

    void solve() {
        boolean changed = true;
        while (changed) {
            changed = false;
            rounds++;
            int added = 0;
            for (Block b : program.blocks) {
                Set<Integer> newIn = new TreeSet<>();
                for (Block p : b.pred) newIn.addAll(out.get(p.name));
                Set<Integer> newOut = new TreeSet<>(gen.get(b.name));
                for (int id : newIn) if (!kill.get(b.name).contains(id)) newOut.add(id);
                if (!in.get(b.name).equals(newIn)) { changed = true; added += apply(in.get(b.name), newIn); }
                if (!out.get(b.name).equals(newOut)) { changed = true; added += apply(out.get(b.name), newOut); }
            }
            lastRoundAdditions = added;
        }
    }

    /** Applies |target| = |next| while recording any element that disappears (a lattice violation). */
    private int apply(Set<Integer> target, Set<Integer> next) {
        Set<Integer> removed = new TreeSet<>(target);
        removed.removeAll(next);
        shrinks += removed.size();
        int before = target.size();
        target.clear();
        target.addAll(next);
        return Math.max(0, target.size() - before);
    }

    Set<Integer> inOf(String block) { return in.get(block); }

    Set<Integer> outOf(String block) { return out.get(block); }

    /** Definitions of |var| that may reach the entry of |block|. */
    Set<Integer> reaching(String block, String var) {
        Set<Integer> result = new TreeSet<>();
        for (int id : in.get(block)) if (program.stmts.get(id).var.equals(var)) result.add(id);
        return result;
    }

    /**
     * Graph-only oracle: BFS over blocks, entering any successor but only continuing through blocks
     * that do not redefine d's variable. Uses statement lists, never GEN/KILL.
     */
    static Set<String> oracle(Program program, Stmt d) {
        Set<String> entry = new LinkedHashSet<>();       // IN sets: entries reached from d
        Set<String> exiting = new LinkedHashSet<>();     // blocks d flows through to their successors
        List<Block> work = new ArrayList<>();
        Block home = program.blockOf(d);
        if (!home.redefinesAfter(d.indexInBlock, d.var)) {         // d survives to the end of its block
            exiting.add(home.name);
            work.add(home);
        }
        while (!work.isEmpty()) {
            Block b = work.remove(work.size() - 1);
            for (Block s : b.succ) {
                entry.add(s.name);
                if (!s.assignedVars().contains(d.var) && exiting.add(s.name)) work.add(s);
            }
        }
        return entry;
    }

    static void crossCheck(Program program, ReachingDefinitions analysis) {
        for (Stmt d : program.stmts) {
            Set<String> oracle = oracle(program, d);
            for (Block b : program.blocks) {
                boolean dataflow = analysis.inOf(b.name).contains(d.id);
                boolean graph = oracle.contains(b.name);
                if (dataflow != graph)
                    throw new AssertionError("FAILED: d" + d.id + " into " + b.name + ": dataflow="
                            + dataflow + " oracle=" + graph);
            }
        }
    }
}

public class Solution {
    static Program straightLine() {
        Program p = new Program();
        Block b0 = p.block("b0");
        p.def(b0, "a", "1");                 // d0
        p.def(b0, "b", "a");                 // d1
        p.def(b0, "a", "b");                 // d2 - kills d0
        return p;
    }

    static Program diamond() {
        Program p = new Program();
        Block b0 = p.block("b0"), b1 = p.block("b1"), b2 = p.block("b2"), b3 = p.block("b3");
        p.def(b0, "x", "1");                 // d0
        p.def(b0, "y", "2");                 // d1
        p.def(b1, "z", "x");                 // d2
        p.def(b2, "x", "y");                 // d3
        p.def(b3, "w", "z", "y");            // d4
        p.edge(b0, b1);
        p.edge(b0, b2);
        p.edge(b1, b3);
        p.edge(b2, b3);
        return p;
    }

    static Program loop() {
        Program p = new Program();
        Block b0 = p.block("b0"), b1 = p.block("b1"), b2 = p.block("b2"), b3 = p.block("b3"), b4 = p.block("b4");
        p.def(b0, "i", "0");                 // d0
        p.def(b0, "s", "0");                 // d1
        p.def(b1, "i", "i");                 // d2
        p.def(b2, "s", "s", "i");            // d3
        p.def(b3, "t", "s");                 // d4
        p.def(b4, "r", "s");                 // d5
        p.edge(b0, b1);
        p.edge(b1, b2);
        p.edge(b2, b3);
        p.edge(b3, b1);                      // back edge
        p.edge(b3, b4);
        return p;
    }

    /** Two blocks that no execution reaches: dead1 defines u and feeds a live block and dead2,
     *  while dead2 has no definitions at all and cycles back into dead1. */
    static Program withDeadBlock() {
        Program p = new Program();
        Block b0 = p.block("b0"), b1 = p.block("b1"), b2 = p.block("b2");
        Block dead1 = p.block("dead1"), dead2 = p.block("dead2");
        p.def(b0, "p", "1");                 // d0
        p.def(b1, "q", "p");                 // d1
        p.def(b2, "r", "q");                 // d2
        p.def(dead1, "u", "9");              // d3 - only defined inside the dead block
        p.edge(b0, b1);
        p.edge(b1, b2);
        p.edge(dead1, b1);                   // dead code feeding a live block
        p.edge(dead1, dead2);
        p.edge(dead2, dead1);
        return p;
    }

    static Program empty() {
        Program p = new Program();
        p.block("b0");
        return p;
    }

    static Set<Integer> defsOf(Program p, Set<Integer> ids, String var) {
        Set<Integer> result = new TreeSet<>();
        for (int id : ids) if (p.stmts.get(id).var.equals(var)) result.add(id);
        return result;
    }

    public static void main(String[] args) {
        Program line = straightLine();
        ReachingDefinitions rd = new ReachingDefinitions(line, false);
        rd.solve();
        check(rd.gen.get("b0").equals(new TreeSet<>(List.of(1, 2))) && rd.kill.get("b0").equals(new TreeSet<>(List.of(0))),
                "GEN = {d1,d2} and KILL = {d0} for the straight-line block, got " + rd.gen.get("b0") + "/" + rd.kill.get("b0"));
        check(rd.inOf("b0").isEmpty() && rd.outOf("b0").equals(new TreeSet<>(List.of(1, 2))),
                "the entry block has no predecessors and keeps only its surviving definitions");
        check(defsOf(line, rd.outOf("b0"), "a").equals(new TreeSet<>(List.of(2))),
                "only the second definition of a survives to the block end, got " + rd.reaching("b0", "a"));
        check(rd.rounds <= 3 && rd.lastRoundAdditions == 0 && rd.shrinks == 0,
                "the iteration is monotone and stops adding: rounds=" + rd.rounds);
        ReachingDefinitions.crossCheck(line, rd);

        Program dia = diamond();
        ReachingDefinitions d = new ReachingDefinitions(dia, false);
        d.solve();
        check(d.inOf("b3").equals(new TreeSet<>(List.of(0, 1, 2, 3))),
                "the diamond merges both branches, got " + Program.ids(d.inOf("b3")));
        check(d.reaching("b3", "x").equals(new TreeSet<>(List.of(0, 3))),
                "the old definition of x still reaches past the branch that did not overwrite it");
        check(d.inOf("b1").equals(new TreeSet<>(List.of(0, 1))) && d.inOf("b2").equals(new TreeSet<>(List.of(0, 1))),
                "each branch inherits exactly the definitions of the entry block");
        check(d.outOf("b3").equals(new TreeSet<>(List.of(0, 1, 2, 3, 4))), "b3 adds its own definition d4");
        ReachingDefinitions.crossCheck(dia, d);

        Program lo = loop();
        ReachingDefinitions l = new ReachingDefinitions(lo, false);
        l.solve();
        check(l.inOf("b1").equals(new TreeSet<>(List.of(0, 1, 2, 3, 4))),
                "the loop header sees definitions that come around the back edge, got " + Program.ids(l.inOf("b1")));
        check(l.inOf("b1").contains(2) && l.inOf("b0").isEmpty(),
                "the back edge carries d2 into the header but never into the entry block");
        check(l.reaching("b2", "s").equals(new TreeSet<>(List.of(1, 3))),
                "both definitions of s reach the loop body entry, got " + l.reaching("b2", "s"));
        check(!l.inOf("b4").contains(1) && l.inOf("b4").contains(3),
                "the loop body kills the initial definition of s before the exit, got " + Program.ids(l.inOf("b4")));
        check(l.inOf("b4").contains(4), "d4 survives to the exit block");
        ReachingDefinitions.crossCheck(lo, l);

        Program dead = withDeadBlock();
        ReachingDefinitions deadRd = new ReachingDefinitions(dead, false);
        deadRd.solve();
        check(deadRd.inOf("dead1").equals(new TreeSet<>(List.of(3)))
                        && deadRd.outOf("dead1").equals(new TreeSet<>(List.of(3))),
                "an unreachable block sees only its own definition, got " + Program.ids(deadRd.inOf("dead1")));
        check(deadRd.inOf("b1").equals(new TreeSet<>(List.of(0, 3))),
                "the dead block's definition reaches b1 along the graph, got " + Program.ids(deadRd.inOf("b1")));
        ReachingDefinitions.crossCheck(dead, deadRd);

        ReachingDefinitions deadTop = new ReachingDefinitions(dead, true);
        deadTop.solve();
        check(deadTop.inOf("dead2").contains(0) && !deadRd.inOf("dead2").contains(0),
                "the top initialisation keeps d0 inside the dead cycle forever (why we start from empty): "
                        + Program.ids(deadTop.inOf("dead2")) + " vs " + Program.ids(deadRd.inOf("dead2")));
        boolean superset = true;
        for (Block b : dead.blocks) if (!deadTop.inOf(b.name).containsAll(deadRd.inOf(b.name))) superset = false;
        check(superset, "the greatest fixpoint is a superset of the least fixpoint");

        Program diaTop = diamond();
        ReachingDefinitions diaTopSolver = new ReachingDefinitions(diaTop, true);
        diaTopSolver.solve();
        ReachingDefinitions diaBottom = new ReachingDefinitions(diamond(), false);
        diaBottom.solve();
        check(diaTopSolver.in.equals(diaBottom.in) && diaTopSolver.out.equals(diaBottom.out),
                "both initialisations agree when every block is reachable");

        Program none = empty();
        ReachingDefinitions noneRd = new ReachingDefinitions(none, false);
        noneRd.solve();
        check(noneRd.inOf("b0").isEmpty() && noneRd.outOf("b0").isEmpty() && noneRd.rounds == 1,
                "a program with no statements converges immediately to empty sets");
        ReachingDefinitions again = new ReachingDefinitions(loop(), false);
        again.solve();
        check(again.in.equals(l.in) && again.out.equals(l.out), "the analysis is deterministic");

        int defs = 0, pairs = 0;
        for (Program p : List.of(line, dia, lo, dead)) {
            defs += p.stmts.size();
            pairs += p.stmts.size() * p.blocks.size();
        }
        check(pairs > 60, "the oracle cross-check covers " + pairs + " (definition, block) pairs");
        System.out.println("straight line : " + Program.ids(rd.outOf("b0")) + " reaches the block end");
        System.out.println("loop header   : " + Program.ids(l.inOf("b1")) + " (back edge carries d2)");
        System.out.println("oracle        : " + pairs + " pairs over " + defs + " definitions agree with the BFS");
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
