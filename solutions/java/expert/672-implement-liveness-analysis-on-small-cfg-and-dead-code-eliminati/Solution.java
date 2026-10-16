/** Task 672 - Implement liveness analysis on small CFG and dead code elimination. (Tier: Expert, CTCI Beyond — Compilers & Program Analysis)
 *
 * A tiny three-address IR (basic blocks of `v = const|copy|a+b|a-b|a*b`, `v = (a > k)`, `store(a)` side
 * effects and `brif(c)` branches, plus an explicit edge list) is analysed with backward may-liveness:
 *
 *   USE[B] = variables read before being written in B,   DEF[B] = variables written in B,
 *   LIVE_OUT[B] = union of LIVE_IN[S] over successors S,  LIVE_IN[B] = USE[B] union (LIVE_OUT[B] - DEF[B]).
 *
 * Iterating this backwards to a fixpoint from the empty sets gives the least fixpoint, which is checked
 * to be monotone (sets only grow) and to stabilise. An independent oracle reinterprets the equations as
 * a graph question - from a block, is there a path to a read of v that crosses no write of v? - answered
 * by a DFS that scans statement lists only and never touches the computed USE/DEF/LIVE sets.
 *
 * Dead code elimination then repeatedly removes pure assignments whose variable is not live after the
 * statement (stores and branches are never removed) until a sweep removes nothing, and the fixpoint is
 * checked to contain no further removable statement. Correctness is not argued but measured: an
 * interpreter runs the program before and after elimination on deterministic inputs and the emitted
 * store trace must match exactly, which is also asserted to differ across inputs so the test has teeth.
 */
import java.util.*;

final class Stmt {
    static final int CONST = 0, COPY = 1, ADD = 2, SUB = 3, MUL = 4, GT = 5, STORE = 6, BRIF = 7,
            ADDI = 8, SUBI = 9, MULI = 10;
    final int id, kind, imm;
    final String var;
    final List<String> uses;

    Stmt(int id, int kind, String var, int imm, List<String> uses) {
        this.id = id; this.kind = kind; this.var = var; this.imm = imm; this.uses = uses;
    }

    boolean pure() { return kind != STORE && kind != BRIF; }

    public String toString() {
        return switch (kind) {
            case CONST -> var + "=" + imm;
            case COPY -> var + "=" + uses.get(0);
            case ADD -> var + "=" + uses.get(0) + "+" + uses.get(1);
            case SUB -> var + "=" + uses.get(0) + "-" + uses.get(1);
            case MUL -> var + "=" + uses.get(0) + "*" + uses.get(1);
            case ADDI -> var + "=" + uses.get(0) + "+" + imm;
            case SUBI -> var + "=" + uses.get(0) + "-" + imm;
            case MULI -> var + "=" + uses.get(0) + "*" + imm;
            case GT -> var + "=(" + uses.get(0) + ">" + imm + ")";
            case STORE -> "store(" + uses.get(0) + ")";
            default -> "brif(" + uses.get(0) + ")";
        };
    }
}

final class Block {
    final String name;
    final List<Stmt> stmts = new ArrayList<>();
    final List<Block> succ = new ArrayList<>();

    Block(String name) { this.name = name; }
}

final class Program {
    final List<Block> blocks = new ArrayList<>();
    final List<Stmt> stmts = new ArrayList<>();
    Block entry;
    int nextId;

    Block block(String name) {
        Block b = new Block(name);
        blocks.add(b);
        if (entry == null) entry = b;
        return b;
    }

    void edge(Block from, Block to) { from.succ.add(to); }

    Block constVal(Block b, String v, int k) { return add(b, new Stmt(nextId++, Stmt.CONST, v, k, List.of())); }
    Block copy(Block b, String v, String src) { return add(b, new Stmt(nextId++, Stmt.COPY, v, 0, List.of(src))); }
    Block bin(Block b, int op, String v, String a, String c) { return add(b, new Stmt(nextId++, op, v, 0, List.of(a, c))); }
    Block binImm(Block b, int op, String v, String a, int k) { return add(b, new Stmt(nextId++, op, v, k, List.of(a))); }
    Block gt(Block b, String v, String a, int k) { return add(b, new Stmt(nextId++, Stmt.GT, v, k, List.of(a))); }
    Block store(Block b, String a) { return add(b, new Stmt(nextId++, Stmt.STORE, null, 0, List.of(a))); }
    Block brif(Block b, String c) { return add(b, new Stmt(nextId++, Stmt.BRIF, null, 0, List.of(c))); }

    private Block add(Block b, Stmt s) {
        b.stmts.add(s);
        stmts.add(s);
        return b;
    }

    Set<String> variables() {
        Set<String> vars = new TreeSet<>();
        for (Stmt s : stmts) {
            if (s.var != null) vars.add(s.var);
            vars.addAll(s.uses);
        }
        return vars;
    }
}

final class Liveness {
    final Program program;
    final Map<String, Set<String>> use = new LinkedHashMap<>(), def = new LinkedHashMap<>();
    final Map<String, Set<String>> liveIn = new LinkedHashMap<>(), liveOut = new LinkedHashMap<>();
    int rounds, lastRoundAdditions, shrinks;

    Liveness(Program program) {
        this.program = program;
        for (Block b : program.blocks) {
            Set<String> used = new TreeSet<>(), written = new TreeSet<>();
            for (Stmt s : b.stmts) {
                for (String u : s.uses) if (!written.contains(u)) used.add(u);
                if (s.var != null) written.add(s.var);
            }
            use.put(b.name, used);
            def.put(b.name, written);
            liveIn.put(b.name, new TreeSet<>());
            liveOut.put(b.name, new TreeSet<>());
        }
    }

    void solve() {
        boolean changed = true;
        while (changed) {
            changed = false;
            rounds++;
            int added = 0;
            for (int i = program.blocks.size() - 1; i >= 0; i--) {
                Block b = program.blocks.get(i);
                Set<String> newOut = new TreeSet<>();
                for (Block s : b.succ) newOut.addAll(liveIn.get(s.name));
                Set<String> newIn = new TreeSet<>(use.get(b.name));
                for (String v : newOut) if (!def.get(b.name).contains(v)) newIn.add(v);
                if (!liveOut.get(b.name).equals(newOut)) { changed = true; added += apply(liveOut.get(b.name), newOut); }
                if (!liveIn.get(b.name).equals(newIn)) { changed = true; added += apply(liveIn.get(b.name), newIn); }
            }
            lastRoundAdditions = added;
        }
    }

    private int apply(Set<String> target, Set<String> next) {
        Set<String> removed = new TreeSet<>(target);
        removed.removeAll(next);
        shrinks += removed.size();
        int before = target.size();
        target.clear();
        target.addAll(next);
        return Math.max(0, target.size() - before);
    }

    Set<String> liveInOf(String block) { return liveIn.get(block); }

    Set<String> liveOutOf(String block) { return liveOut.get(block); }

    /** Reads |v| before writing it, walking the statements in order. */
    static boolean readsBeforeWrite(Block b, String v) {
        for (Stmt s : b.stmts) {
            if (s.uses.contains(v)) return true;
            if (v.equals(s.var)) return false;
        }
        return false;
    }

    /** Null when the block neither reads nor writes v. */
    static Boolean scan(Block b, String v) {
        for (Stmt s : b.stmts) {
            if (s.uses.contains(v)) return Boolean.TRUE;
            if (v.equals(s.var)) return Boolean.FALSE;
        }
        return null;
    }

    static boolean aliveOutOracle(Program p, Block b, String v) {
        Deque<Block> work = new ArrayDeque<>(b.succ);
        Set<Block> seen = new HashSet<>();
        while (!work.isEmpty()) {
            Block s = work.poll();
            if (!seen.add(s)) continue;
            Boolean verdict = scan(s, v);
            if (verdict == Boolean.TRUE) return true;
            if (verdict == null) for (Block t : s.succ) work.add(t);
        }
        return false;
    }

    static boolean aliveInOracle(Program p, Block b, String v) {
        if (readsBeforeWrite(b, v)) return true;              // read before any write in this block
        for (Stmt s : b.stmts) if (v.equals(s.var)) return false;   // written before any read
        return aliveOutOracle(p, b, v);
    }

    /** Every (block, variable) pair must agree between the equations and the graph oracle. */
    static void crossCheck(Program p, Liveness lv) {
        for (Block b : p.blocks) {
            for (String v : p.variables()) {
                boolean in = lv.liveInOf(b.name).contains(v);
                boolean out = lv.liveOutOf(b.name).contains(v);
                if (in != aliveInOracle(p, b, v) || out != aliveOutOracle(p, b, v))
                    throw new AssertionError("FAILED: liveness of " + v + " at " + b.name + ": in=" + in
                            + " out=" + out + " oracle=" + aliveInOracle(p, b, v) + "/" + aliveOutOracle(p, b, v));
            }
        }
    }
}

final class DeadCode {
    /** Removes dead pure assignments until a sweep changes nothing; returns the number removed. */
    static int eliminate(Program p) {
        int removed = 0, sweeps = 0;
        while (true) {
            sweeps++;
            Liveness lv = new Liveness(p);
            lv.solve();
            int thisSweep = 0;
            for (Block b : p.blocks) {
                Set<String> live = new TreeSet<>(lv.liveOutOf(b.name));
                for (int i = b.stmts.size() - 1; i >= 0; i--) {
                    Stmt s = b.stmts.get(i);
                    if (s.pure() && !live.contains(s.var)) {
                        b.stmts.remove(i);
                        p.stmts.remove(s);
                        thisSweep++;
                    } else {
                        if (s.var != null) live.remove(s.var);
                        live.addAll(s.uses);
                    }
                }
            }
            removed += thisSweep;
            if (thisSweep == 0) return removed;
        }
    }

    /** Statements a further sweep would remove: must be zero at the fixpoint. */
    static int removable(Program p) {
        Liveness lv = new Liveness(p);
        lv.solve();
        int count = 0;
        for (Block b : p.blocks) {
            Set<String> live = new TreeSet<>(lv.liveOutOf(b.name));
            for (int i = b.stmts.size() - 1; i >= 0; i--) {
                Stmt s = b.stmts.get(i);
                if (s.pure() && !live.contains(s.var)) count++;
                else {
                    if (s.var != null) live.remove(s.var);
                    live.addAll(s.uses);
                }
            }
        }
        return count;
    }
}

final class Interp {
    static final String[] INPUTS = {"in0", "in1", "in2", "in3"};

    /** Observable behaviour of the IR: the exact sequence of stored values. */
    static String run(Program p, int[] ins) {
        Map<String, Integer> env = new TreeMap<>();
        for (String v : p.variables()) env.put(v, 0);
        for (int i = 0; i < INPUTS.length; i++) if (env.containsKey(INPUTS[i])) env.put(INPUTS[i], ins[i]);
        StringBuilder log = new StringBuilder();
        Block b = p.entry;
        int steps = 0;
        while (b != null && steps++ < 5000) {
            Block next = null;
            for (Stmt s : b.stmts) {
                switch (s.kind) {
                    case Stmt.CONST -> env.put(s.var, s.imm);
                    case Stmt.COPY -> env.put(s.var, env.get(s.uses.get(0)));
                    case Stmt.ADD -> env.put(s.var, env.get(s.uses.get(0)) + env.get(s.uses.get(1)));
                    case Stmt.SUB -> env.put(s.var, env.get(s.uses.get(0)) - env.get(s.uses.get(1)));
                    case Stmt.MUL -> env.put(s.var, env.get(s.uses.get(0)) * env.get(s.uses.get(1)));
                    case Stmt.ADDI -> env.put(s.var, env.get(s.uses.get(0)) + s.imm);
                    case Stmt.SUBI -> env.put(s.var, env.get(s.uses.get(0)) - s.imm);
                    case Stmt.MULI -> env.put(s.var, env.get(s.uses.get(0)) * s.imm);
                    case Stmt.GT -> env.put(s.var, env.get(s.uses.get(0)) > s.imm ? 1 : 0);
                    case Stmt.STORE -> log.append(env.get(s.uses.get(0))).append(' ');
                    default -> next = env.get(s.uses.get(0)) != 0 ? b.succ.get(0) : b.succ.get(1);
                }
                if (next != null) break;
            }
            if (next == null) next = b.succ.isEmpty() ? null : b.succ.get(0);
            b = next;
        }
        return log.toString().trim();
    }
}

public class Solution {
    static long state = 0x1CEB00DA5L;

    static long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static Program straight() {
        Program p = new Program();
        Block b0 = p.block("b0"), exit = p.block("exit");
        p.constVal(b0, "a", 2);
        p.bin(b0, Stmt.MUL, "b", "a", "a");            // never read again
        p.binImm(b0, Stmt.ADDI, "c", "a", 1);
        p.store(b0, "c");
        p.edge(b0, exit);
        return p;
    }

    static Program chain() {
        Program p = new Program();
        Block b0 = p.block("b0"), exit = p.block("exit");
        p.copy(b0, "a", "in0");
        p.bin(b0, Stmt.MUL, "u", "a", "a");            // dead
        p.binImm(b0, Stmt.ADDI, "v", "u", 1);          // dead, but keeps u alive until it goes
        p.store(b0, "a");
        p.edge(b0, exit);
        return p;
    }

    static Program diamond() {
        Program p = new Program();
        Block b0 = p.block("b0"), b1 = p.block("b1"), b2 = p.block("b2"), b3 = p.block("b3"), exit = p.block("exit");
        p.binImm(b0, Stmt.SUBI, "cond", "in0", 3);
        p.brif(b0, "cond");
        p.binImm(b1, Stmt.ADDI, "p", "in0", 1);
        p.bin(b1, Stmt.MUL, "d", "p", "p");            // dead inside the taken branch
        p.store(b1, "p");
        p.bin(b2, Stmt.MUL, "p", "in0", "in0");
        p.store(b2, "p");
        p.binImm(b3, Stmt.ADDI, "q", "p", 1);
        p.store(b3, "q");
        p.edge(b0, b1);
        p.edge(b0, b2);
        p.edge(b1, b3);
        p.edge(b2, b3);
        p.edge(b3, exit);
        return p;
    }

    /** acc = sum of the positive input, with a dead multiply inside the loop body. */
    static Program loop() {
        Program p = new Program();
        Block b0 = p.block("b0"), header = p.block("header"), body = p.block("body"), exit = p.block("exit");
        p.copy(b0, "i", "in0");
        p.constVal(b0, "acc", 0);
        p.gt(header, "cond", "i", 0);
        p.brif(header, "cond");
        p.bin(body, Stmt.ADD, "acc", "acc", "i");
        p.binImm(body, Stmt.SUBI, "i", "i", 1);
        p.bin(body, Stmt.MUL, "junk", "i", "i");       // dead, and only live through the back edge
        p.edge(b0, header);
        p.edge(header, body);
        p.edge(header, exit);
        p.edge(body, header);
        p.store(exit, "acc");
        return p;
    }

    static Program sideEffects() {
        Program p = new Program();
        Block b0 = p.block("b0"), exit = p.block("exit");
        p.copy(b0, "x", "in1");
        p.bin(b0, Stmt.MUL, "y", "x", "x");
        p.store(b0, "y");
        p.constVal(b0, "w", 9);
        p.store(b0, "w");                              // w is only read by the store: must survive
        p.bin(b0, Stmt.ADD, "d", "w", "w");            // dead
        p.edge(b0, exit);
        return p;
    }

    static Program unreachable() {
        Program p = new Program();
        Block b0 = p.block("b0"), exit = p.block("exit"), dead = p.block("dead");
        p.copy(b0, "a", "in2");
        p.store(b0, "a");
        p.constVal(dead, "u", 5);
        p.bin(dead, Stmt.MUL, "v", "u", "u");          // dead inside an unreachable block
        p.store(dead, "u");
        p.edge(b0, exit);
        p.edge(dead, exit);
        return p;
    }

    static List<Program> programs() {
        return List.of(straight(), chain(), diamond(), loop(), sideEffects(), unreachable());
    }

    static String[] names() { return new String[]{"straight", "chain", "diamond", "loop", "sideEffects", "unreachable"}; }

    static int[][] inputs() {
        int[][] all = new int[24][4];
        for (int i = 0; i < all.length; i++)
            for (int j = 0; j < 4; j++) all[i][j] = (int) ((nextLong() >>> 40) % 9) - 4;
        return all;
    }

    public static void main(String[] args) {
        Program line = straight();
        Liveness lv = new Liveness(line);
        lv.solve();
        check(lv.use.get("b0").isEmpty() && lv.def.get("b0").equals(new TreeSet<>(List.of("a", "b", "c"))),
                "USE = {} and DEF = {a,b,c} for the straight-line block, got " + lv.use.get("b0") + "/" + lv.def.get("b0"));
        check(lv.liveOutOf("b0").isEmpty() && lv.liveInOf("b0").isEmpty(),
                "nothing is live on entry or after the straight-line block, got in="
                        + lv.liveInOf("b0") + " out=" + lv.liveOutOf("b0"));
        check(lv.liveInOf("b0").isEmpty() && lv.shrinks == 0 && lv.lastRoundAdditions == 0,
                "nothing is live on entry and the backward iteration is monotone");
        Liveness.crossCheck(line, lv);

        Program dia = diamond();
        Liveness dl = new Liveness(dia);
        dl.solve();
        check(dl.liveInOf("b0").equals(new TreeSet<>(List.of("in0"))) && dl.liveOutOf("b0").equals(new TreeSet<>(List.of("in0"))),
                "only in0 flows out of the branch block (p is written inside each arm), got "
                        + dl.liveInOf("b0") + "/" + dl.liveOutOf("b0"));
        check(dl.liveOutOf("b1").equals(new TreeSet<>(List.of("p"))) && dl.liveInOf("b1").equals(new TreeSet<>(List.of("in0"))),
                "the use of p inside b1 is preceded by its definition, so it is not live on entry");
        check(dl.liveInOf("b3").equals(new TreeSet<>(List.of("p"))) && dl.liveOutOf("b3").isEmpty(),
                "p is live on entry to b3 and nothing survives its exit, got "
                        + dl.liveInOf("b3") + "/" + dl.liveOutOf("b3"));
        Liveness.crossCheck(dia, dl);

        Program lo = loop();
        Liveness ll = new Liveness(lo);
        ll.solve();
        check(ll.liveInOf("body").equals(new TreeSet<>(List.of("i", "acc")))
                        && ll.liveOutOf("body").equals(new TreeSet<>(List.of("i", "acc"))),
                "acc and i are live around the loop body, got " + ll.liveInOf("body") + "/" + ll.liveOutOf("body"));
        check(ll.liveInOf("exit").equals(new TreeSet<>(List.of("acc"))) && !ll.liveInOf("exit").contains("i"),
                "i dies when the loop ends while acc is still needed by the store");
        check(ll.liveInOf("header").contains("i") && ll.liveOutOf("header").contains("i"),
                "the back edge keeps i live at the header");
        Liveness.crossCheck(lo, ll);

        Program side = sideEffects();
        Liveness sl = new Liveness(side);
        sl.solve();
        check(sl.liveInOf("b0").equals(new TreeSet<>(List.of("in1"))),
                "only the input is live on entry when every value is stored, got " + sl.liveInOf("b0"));
        check(sl.liveOutOf("b0").isEmpty() && !Liveness.readsBeforeWrite(side.blocks.get(0), "x"),
                "nothing is live at the end and x is written before it is read");
        Liveness.crossCheck(side, sl);

        Program dead = unreachable();
        Liveness ddl = new Liveness(dead);
        ddl.solve();
        check(ddl.liveInOf("dead").isEmpty() && ddl.liveOutOf("dead").isEmpty(),
                "an unreachable block that writes u before storing it keeps nothing live, got "
                        + ddl.liveInOf("dead"));
        check(!ddl.liveInOf("exit").contains("u"), "definitions in the dead block do not become live at the exit");
        Liveness.crossCheck(dead, ddl);

        int[][] ins = inputs();
        List<Program> originals = programs();
        int totalRemoved = 0, totalRemaining = 0, varying = 0, pairs = 0;
        for (int idx = 0; idx < originals.size(); idx++) {
            Program original = originals.get(idx);
            List<String> before = new ArrayList<>();
            for (int[] in : ins) before.add(Interp.run(original, in));
            pairs += original.blocks.size() * original.variables().size();

            Program copy = rebuild(idx);
            int removed = DeadCode.eliminate(copy);
            totalRemoved += removed;
            totalRemaining += DeadCode.removable(copy);
            Liveness after = new Liveness(copy);
            after.solve();
            Liveness.crossCheck(copy, after);
            for (int t = 0; t < ins.length; t++) {
                String got = Interp.run(copy, ins[t]);
                check(got.equals(before.get(t)), names()[idx] + " changed behaviour on input " + t
                        + ": " + before.get(t) + " -> " + got);
            }
            Set<String> distinct = new TreeSet<>(before);
            if (distinct.size() > 1) varying++;
        }
        check(totalRemoved >= 7, "the six programs gave up " + totalRemoved + " dead assignments");
        check(totalRemaining == 0, "no removable statement survives the fixpoint, found " + totalRemaining);
        check(varying >= 3, "at least three programs behave differently across the 24 inputs, got " + varying);
        check(pairs > 40, "the liveness oracle was cross-checked on " + pairs + " (block, variable) pairs");

        Program chainCopy = rebuild(1);
        check(DeadCode.eliminate(chainCopy) == 2
                        && chainCopy.blocks.get(0).stmts.size() == 2,
                "the two-statement dead chain collapses in one sweep, left "
                        + chainCopy.blocks.get(0).stmts.size() + " statements");
        Program sideCopy = rebuild(4);
        DeadCode.eliminate(sideCopy);
        boolean storeKept = false, deadGone = true;
        for (Stmt s : sideCopy.stmts) {
            if (s.kind == Stmt.STORE && s.uses.contains("w")) storeKept = true;
            if ("d".equals(s.var)) deadGone = false;
        }
        check(storeKept && deadGone, "side effects are kept while the unused arithmetic disappears");
        Program deadCopy = rebuild(5);
        DeadCode.eliminate(deadCopy);
        check(DeadCode.eliminate(deadCopy) == 0, "elimination is idempotent at the fixpoint");

        Program empty = new Program();
        empty.block("b0");
        Liveness emptyLv = new Liveness(empty);
        emptyLv.solve();
        check(emptyLv.liveInOf("b0").isEmpty() && emptyLv.liveOutOf("b0").isEmpty()
                        && DeadCode.eliminate(empty) == 0,
                "an empty block has empty live sets and nothing to remove");

        check(Interp.run(straight(), new int[]{0, 0, 0, 0}).equals("3"), "the straight-line program stores 3");
        check(Interp.run(sideEffects(), new int[]{0, 3, 0, 0}).equals("9 9"),
                "the side-effect program stores y then the constant 9");

        System.out.println("liveness    : oracle agrees on " + pairs + " (block, variable) pairs");
        System.out.println("elimination : " + totalRemoved + " dead assignments removed, 0 left, traces unchanged");
        System.out.println("inputs      : 24 deterministic runs, " + varying + " programs behave input-dependently");
        System.out.println("PASS");
    }

    static Program rebuild(int idx) {
        return switch (idx) {
            case 0 -> straight();
            case 1 -> chain();
            case 2 -> diamond();
            case 3 -> loop();
            case 4 -> sideEffects();
            default -> unreachable();
        };
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
