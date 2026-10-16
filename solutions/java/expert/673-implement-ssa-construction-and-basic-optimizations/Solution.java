/** Task 673 - Implement SSA construction and basic optimizations. (Tier: Expert, CTCI Beyond — Compilers & Program Analysis)
 *
 * A small three-address CFG (`v = const|copy|a+b|a-b|a*b|a<b`, terminators `br`, `brif`, `ret`) is put
 * into minimal SSA with the classic Cytron et al. pipeline: iterative dominators, immediate dominators
 * from the dominator sets, dominance frontiers by the runner walk, phi placement in the iterated
 * Dominance frontiers followed by phi placement in their iterated closure, and renaming by a
 * dominator-tree DFS with one stack of current names per variable. A variable that has no definition on
 * an incoming edge yet (the entry edge of a loop-header phi computed inside the header) is denoted by
 * the pseudo-name |undef|: that simplification is checked at run time - the interpreter counts every
 * read of an undefined value and the tests require the count to stay zero.
 *
 * Basic optimizations then iterate to a fixpoint: constant folding (a sparse propagation over the SSA
 * graph; a phi folds only when every operand carries the same value), copy propagation (aliases are
 * resolved transitively and every use is rewritten), trivial-phi elimination (all operands the same
 * name), and dead-code elimination for pure instructions whose result has no use.
 *
 * Verified: dominators against a brute-force "delete the node and see whether n is still reachable"
 * oracle, dominance frontiers against their definition, the SSA invariants themselves (every name has
 * one definition, every use is dominated by its definition - phi operands by the matching predecessor -
 * and every phi has exactly one operand per CFG predecessor), no phi in a straight-line block, and
 * behaviour preservation: an interpreter runs the original program, its SSA form and the optimized SSA
 * form on deterministic inputs and all three must return the same value, while the statistics confirm
 * that folding, copy propagation and elimination actually fired.
 */
import java.util.*;

final class Ins {
    static final int CONST = 0, COPY = 1, ADD = 2, SUB = 3, MUL = 4, LT = 5;
    final int op;
    final String dst, a, b;
    final int imm;

    Ins(int op, String dst, String a, String b, int imm) {
        this.op = op; this.dst = dst; this.a = a; this.b = b; this.imm = imm;
    }

    static Ins cst(String d, int k) { return new Ins(CONST, d, null, null, k); }
    static Ins cpy(String d, String s) { return new Ins(COPY, d, s, null, 0); }
    static Ins bin(int op, String d, String x, String y) { return new Ins(op, d, x, y, 0); }
}

final class Blk {
    static final int BR = 0, BRIF = 1, RET = 2;
    final String name;
    final List<Ins> ins = new ArrayList<>();
    final List<String> pred = new ArrayList<>();
    int term = BR;
    String target, alt, cond, value;

    Blk(String name) { this.name = name; }
}

final class Cfg {
    final List<Blk> blocks = new ArrayList<>();
    final Map<String, Blk> byName = new LinkedHashMap<>();

    Blk block(String name) {
        Blk b = new Blk(name);
        blocks.add(b);
        byName.put(name, b);
        return b;
    }

    Blk entry() { return blocks.get(0); }

    void link() {
        for (Blk b : blocks) b.pred.clear();
        for (Blk b : blocks) {
            if (b.term == Blk.BR) byName.get(b.target).pred.add(b.name);
            else if (b.term == Blk.BRIF) {
                byName.get(b.target).pred.add(b.name);
                byName.get(b.alt).pred.add(b.name);
            }
        }
    }

    List<String> successors(Blk b) {
        if (b.term == Blk.BR) return List.of(b.target);
        if (b.term == Blk.BRIF) return List.of(b.target, b.alt);
        return List.of();
    }

    /** Straight-line code: two foldable operations, a dead one, and a copy. */
    static Cfg straight() {
        Cfg c = new Cfg();
        Blk b0 = c.block("b0");
        b0.ins.add(Ins.cst("x", 2));
        b0.ins.add(Ins.cst("y", 3));
        b0.ins.add(Ins.bin(Ins.MUL, "z", "x", "y"));       // folds to 6
        b0.ins.add(Ins.bin(Ins.ADD, "w", "z", "in1"));     // stays symbolic
        b0.ins.add(Ins.bin(Ins.SUB, "dead", "z", "z"));    // folds to 0 and is then unused
        b0.ins.add(Ins.cpy("alias", "w"));                 // copy propagation material
        b0.term = Blk.RET;
        b0.value = "alias";
        c.link();
        return c;
    }

    static Cfg diamond() {
        Cfg c = new Cfg();
        Blk b0 = c.block("b0"), bt = c.block("bt"), bf = c.block("bf"), bj = c.block("bj");
        b0.ins.add(Ins.cst("one", 1));
        b0.ins.add(Ins.cst("zero", 0));
        b0.ins.add(Ins.cpy("a", "in0"));
        b0.ins.add(Ins.bin(Ins.ADD, "b", "a", "one"));
        b0.ins.add(Ins.bin(Ins.LT, "c", "a", "zero"));
        b0.term = Blk.BRIF;
        b0.cond = "c"; b0.target = "bt"; b0.alt = "bf";
        bt.ins.add(Ins.bin(Ins.ADD, "d", "a", "b"));
        bt.term = Blk.BR; bt.target = "bj";
        bf.ins.add(Ins.bin(Ins.MUL, "d", "a", "b"));
        bf.term = Blk.BR; bf.target = "bj";
        bj.ins.add(Ins.bin(Ins.ADD, "e", "d", "b"));
        bj.term = Blk.RET; bj.value = "e";
        c.link();
        return c;
    }

    /** acc = sum of the non-negative input, counting down to zero. */
    static Cfg loop() {
        Cfg c = new Cfg();
        Blk b0 = c.block("b0"), head = c.block("head"), body = c.block("body"), exit = c.block("exit");
        b0.ins.add(Ins.cpy("i", "in0"));
        b0.ins.add(Ins.cst("acc", 0));
        b0.term = Blk.BR; b0.target = "head";
        head.ins.add(Ins.cst("one", 1));
        head.ins.add(Ins.bin(Ins.LT, "c", "i", "one"));
        head.term = Blk.BRIF; head.cond = "c"; head.target = "exit"; head.alt = "body";
        body.ins.add(Ins.bin(Ins.ADD, "acc", "acc", "i"));
        body.ins.add(Ins.bin(Ins.SUB, "i", "i", "one"));
        body.ins.add(Ins.bin(Ins.MUL, "junk", "i", "i"));   // dead through the back edge
        body.term = Blk.BR; body.target = "head";
        exit.term = Blk.RET; exit.value = "acc";
        c.link();
        return c;
    }

    /** A ladder of forward branches: v[j % 3] += 1 under a data-dependent condition. */
    static Cfg ladder(int branches) {
        Cfg c = new Cfg();
        Blk start = c.block("b0");
        String[] vars = {"v0", "v1", "v2"};
        start.ins.add(Ins.cpy("v0", "in0"));
        start.ins.add(Ins.cst("v1", 1));
        start.ins.add(Ins.cst("v2", 3));
        for (int j = 1; j <= branches; j++) c.block("b" + j);
        Blk join = c.block("join");
        start.term = Blk.BR; start.target = "b1";
        for (int j = 1; j <= branches; j++) {
            Blk b = c.byName.get("b" + j);
            String v = vars[j % 3];
            b.ins.add(Ins.bin(Ins.ADD, v, v, "v1"));
            b.ins.add(Ins.bin(Ins.LT, "c" + j, "v0", "v2"));
            b.term = Blk.BRIF; b.cond = "c" + j;
            b.target = j < branches ? "b" + (j + 1) : "join";
            b.alt = "join";
        }
        join.ins.add(Ins.bin(Ins.ADD, "s1", "v0", "v1"));
        join.ins.add(Ins.bin(Ins.ADD, "s2", "s1", "v2"));
        join.term = Blk.RET; join.value = "s2";
        c.link();
        return c;
    }
}

final class SsaIns {
    static final int PHI = 6;
    int op;
    final String dst;
    String a, b;
    int imm;
    final String phiVar;
    final List<String> phiPred = new ArrayList<>(), phiVal = new ArrayList<>();

    SsaIns(int op, String dst, String phiVar) { this.op = op; this.dst = dst; this.phiVar = phiVar; }

    SsaIns(int op, String dst) { this(op, dst, null); }
}

final class SsaBlk {
    final String name;
    final List<SsaIns> ins = new ArrayList<>();
    String target, alt, cond, ret;
    boolean isRet;
    boolean isBranch;

    SsaBlk(String name) { this.name = name; }
}

final class SsaProgram {
    final List<SsaBlk> blocks = new ArrayList<>();
    final Map<String, SsaBlk> byName = new LinkedHashMap<>();
}

final class SsaBuilder {
    static final List<String> PARAMS = List.of("in0", "in1");
    /** Name standing for "not defined on this edge"; it is never read on any executed path. */
    static final String UNDEF = "undef";
    static int undefinedOperands = 0;

    static Map<String, Set<String>> dominators(Cfg cfg) {
        Set<String> all = new LinkedHashSet<>(cfg.byName.keySet());
        Map<String, Set<String>> dom = new LinkedHashMap<>();
        for (Blk b : cfg.blocks)
            dom.put(b.name, b == cfg.entry() ? new LinkedHashSet<>(List.of(b.name)) : new LinkedHashSet<>(all));
        boolean changed = true;
        while (changed) {
            changed = false;
            for (Blk b : cfg.blocks) {
                if (b == cfg.entry()) continue;
                Set<String> next = new LinkedHashSet<>(all);
                for (String p : b.pred) next.retainAll(dom.get(p));
                next.add(b.name);
                if (!next.equals(dom.get(b.name))) { dom.put(b.name, next); changed = true; }
            }
        }
        return dom;
    }

    static Map<String, String> immediateDominators(Cfg cfg, Map<String, Set<String>> dom) {
        Map<String, String> idom = new LinkedHashMap<>();
        for (Blk b : cfg.blocks) {
            String best = null;
            for (String d : dom.get(b.name)) {
                if (d.equals(b.name)) continue;
                if (best == null || dom.get(d).size() > dom.get(best).size()) best = d;
            }
            idom.put(b.name, best);
        }
        return idom;
    }

    static Map<String, Set<String>> frontiers(Cfg cfg, Map<String, String> idom) {
        Map<String, Set<String>> df = new LinkedHashMap<>();
        for (Blk b : cfg.blocks) df.put(b.name, new LinkedHashSet<>());
        for (Blk b : cfg.blocks) {
            if (b.pred.size() < 2) continue;
            for (String p : b.pred) {
                String runner = p;
                while (runner != null && !runner.equals(idom.get(b.name))) {
                    df.get(runner).add(b.name);
                    runner = idom.get(runner);
                }
            }
        }
        return df;
    }

    static SsaProgram build(Cfg cfg) {
        Map<String, Set<String>> dom = dominators(cfg);
        Map<String, String> idom = immediateDominators(cfg, dom);
        Map<String, Set<String>> df = frontiers(cfg, idom);
        SsaProgram ssa = new SsaProgram();
        for (Blk b : cfg.blocks) {
            SsaBlk sb = new SsaBlk(b.name);
            sb.isBranch = b.term != Blk.RET;
            sb.isRet = b.term == Blk.RET;
            sb.target = b.target;
            sb.alt = b.alt;
            ssa.blocks.add(sb);
            ssa.byName.put(b.name, sb);
        }
        // phi placement: iterated dominance frontier of every definition site
        Map<String, Set<String>> defSites = new LinkedHashMap<>();
        for (Blk b : cfg.blocks)
            for (Ins i : b.ins) defSites.computeIfAbsent(i.dst, k -> new LinkedHashSet<>()).add(b.name);
        int phiCounter = 0;
        for (Map.Entry<String, Set<String>> e : defSites.entrySet()) {
            Deque<String> work = new ArrayDeque<>(e.getValue());
            Set<String> placed = new LinkedHashSet<>();
            while (!work.isEmpty()) {
                String b = work.poll();
                for (String f : df.get(b)) {
                    if (!placed.add(f)) continue;
                    ssa.byName.get(f).ins.add(new SsaIns(SsaIns.PHI, e.getKey() + "#" + (++phiCounter), e.getKey()));
                    if (!e.getValue().contains(f)) work.add(f);
                }
            }
        }
        // renaming over the dominator tree
        Map<String, List<String>> children = new LinkedHashMap<>();
        Map<String, Deque<String>> stacks = new LinkedHashMap<>();
        for (Blk b : cfg.blocks) {
            children.put(b.name, new ArrayList<>());
            if (b != cfg.entry()) children.get(idom.get(b.name)).add(b.name);
        }
        for (String v : PARAMS) stacks.computeIfAbsent(v, k -> new ArrayDeque<>()).push(v);
        Map<String, Integer> counters = new LinkedHashMap<>();
        rename(cfg, cfg.entry(), children, stacks, counters, ssa);
        return ssa;
    }

    static String fresh(Map<String, Integer> counters, String v) {
        int n = counters.merge(v, 1, Integer::sum);
        return v + "." + n;
    }

    /** Current name of v, or the pseudo-name |undef| when some incoming edge has no definition yet. */
    static String current(Map<String, Deque<String>> stacks, String v) {
        Deque<String> stack = stacks.get(v);
        if (stack == null || stack.isEmpty()) {
            undefinedOperands++;                     // e.g. the entry edge of a loop-header phi
            return UNDEF;
        }
        return stack.peek();
    }

    static void rename(Cfg cfg, Blk b, Map<String, List<String>> children,
                       Map<String, Deque<String>> stacks, Map<String, Integer> counters, SsaProgram ssa) {
        SsaBlk sb = ssa.byName.get(b.name);
        List<String> pushed = new ArrayList<>();
        for (SsaIns phi : sb.ins) {                       // phi results are defined on block entry
            stacks.computeIfAbsent(phi.phiVar, k -> new ArrayDeque<>()).push(phi.dst);
            pushed.add(phi.phiVar);
        }
        for (Ins i : b.ins) {
            SsaIns si = new SsaIns(i.op, fresh(counters, i.dst));
            if (i.op == Ins.CONST) si.imm = i.imm;
            else {
                si.a = current(stacks, i.a);
                si.b = i.b == null ? null : current(stacks, i.b);
            }
            sb.ins.add(si);
            stacks.computeIfAbsent(i.dst, k -> new ArrayDeque<>()).push(si.dst);
            pushed.add(i.dst);
        }
        if (b.term == Blk.BRIF) sb.cond = current(stacks, b.cond);
        if (b.term == Blk.RET) sb.ret = current(stacks, b.value);
        for (String s : cfg.successors(b)) {
            for (SsaIns phi : ssa.byName.get(s).ins) {
                if (phi.op != SsaIns.PHI) continue;
                phi.phiPred.add(b.name);
                phi.phiVal.add(current(stacks, phi.phiVar));
            }
        }
        for (String child : children.get(b.name)) rename(cfg, cfg.byName.get(child), children, stacks, counters, ssa);
        for (String v : pushed) stacks.get(v).pop();
    }
}

final class Interp {
    static int undefReads = 0;

    static Map<String, Integer> inputs(int a, int b) {
        Map<String, Integer> in = new LinkedHashMap<>();
        in.put("in0", a);
        in.put("in1", b);
        return in;
    }

    static int run(Cfg cfg, Map<String, Integer> in) {
        Map<String, Integer> env = new HashMap<>(in);
        Blk b = cfg.entry();
        for (int steps = 0; steps < 20000; steps++) {
            for (Ins i : b.ins) env.put(i.dst, eval(i.op, i.a == null ? 0 : env.get(i.a), i.b == null ? 0 : env.get(i.b), i.imm));
            if (b.term == Blk.RET) return env.get(b.value);
            if (b.term == Blk.BR) b = cfg.byName.get(b.target);
            else b = cfg.byName.get(env.get(b.cond) != 0 ? b.target : b.alt);
        }
        throw new IllegalStateException("step limit");
    }

    static int runSsa(SsaProgram p, Cfg cfg, Map<String, Integer> in) {
        Map<String, Integer> env = new HashMap<>(in);
        env.put(SsaBuilder.UNDEF, 0);
        Set<String> unknown = new HashSet<>();
        SsaBlk b = p.byName.get(cfg.entry().name);
        String cameFrom = null;
        for (int steps = 0; steps < 20000; steps++) {
            for (SsaIns i : b.ins) {
                if (i.op == SsaIns.PHI) {
                    int idx = i.phiPred.indexOf(cameFrom);
                    if (idx < 0) throw new IllegalStateException("phi in " + b.name + " has no operand for " + cameFrom);
                    String src = i.phiVal.get(idx);
                    env.put(i.dst, env.get(src));
                    if (isUnknown(unknown, src)) unknown.add(i.dst);
                } else {
                    if (isUnknown(unknown, i.a) || isUnknown(unknown, i.b)) { undefReads++; unknown.add(i.dst); }
                    env.put(i.dst, eval(i.op, i.a == null ? 0 : env.get(i.a), i.b == null ? 0 : env.get(i.b), i.imm));
                }
            }
            if (b.isRet) {
                if (isUnknown(unknown, b.ret)) undefReads++;
                return env.get(b.ret);
            } else if (b.alt == null) { cameFrom = b.name; b = p.byName.get(b.target); }
            else {
                if (isUnknown(unknown, b.cond)) undefReads++;
                cameFrom = b.name;
                b = p.byName.get(env.get(b.cond) != 0 ? b.target : b.alt);
            }
        }
        throw new IllegalStateException("step limit");
    }

    static boolean isUnknown(Set<String> unknown, String name) {
        return name != null && (unknown.contains(name) || name.equals(SsaBuilder.UNDEF));
    }

    static int eval(int op, int x, int y, int imm) {
        return switch (op) {
            case Ins.CONST -> imm;
            case Ins.COPY -> x;
            case Ins.ADD -> x + y;
            case Ins.SUB -> x - y;
            case Ins.MUL -> x * y;
            default -> x < y ? 1 : 0;
        };
    }
}

final class Optimizer {
    int folds, copies, phis, removed;
    final Map<String, Integer> known = new HashMap<>();

    void optimize(SsaProgram p) {
        for (int round = 0; round < 40; round++) {
            int before = size(p);
            constantFold(p);
            copyPropagate(p);
            eliminateTrivialPhis(p);
            deadCode(p);
            if (size(p) == before) return;
        }
    }

    static int size(SsaProgram p) {
        int n = 0;
        for (SsaBlk b : p.blocks) n += b.ins.size();
        return n;
    }

    /** Sparse constant propagation over the SSA graph, then rewriting of the folded definitions. */
    void constantFold(SsaProgram p) {
        known.clear();
        boolean changed = true;
        while (changed) {
            changed = false;
            for (SsaBlk b : p.blocks)
                for (SsaIns i : b.ins) {
                    if (known.containsKey(i.dst)) continue;
                    Integer value = null;
                    if (i.op == Ins.CONST) value = i.imm;
                    else if (i.op == SsaIns.PHI) {
                        boolean ok = true;
                        Integer common = null;
                        for (String v : i.phiVal) {
                            if (v.equals(i.dst)) continue;             // self reference on the back edge
                            Integer c = known.get(v);
                            if (c == null || (common != null && !common.equals(c))) { ok = false; break; }
                            common = c;
                        }
                        if (ok) value = common;
                    } else {
                        Integer x = i.a == null ? null : known.get(i.a);
                        Integer y = i.b == null ? null : known.get(i.b);
                        if (x != null && (i.op == Ins.COPY || y != null))
                            value = Interp.eval(i.op, x, y == null ? 0 : y, i.imm);
                    }
                    if (value != null) { known.put(i.dst, value); changed = true; }
                }
        }
        for (SsaBlk b : p.blocks)
            for (SsaIns i : b.ins) {
                Integer c = known.get(i.dst);
                if (c == null || (i.op == Ins.CONST && i.imm == c)) continue;
                i.op = Ins.CONST;
                i.a = null;
                i.b = null;
                i.imm = c;
                i.phiVal.clear();
                i.phiPred.clear();
                folds++;
            }
    }

    void copyPropagate(SsaProgram p) {
        Map<String, String> alias = new LinkedHashMap<>();
        for (SsaBlk b : p.blocks)
            for (SsaIns i : b.ins)
                if (i.op == Ins.COPY && i.a != null && !i.a.equals(i.dst)) alias.put(i.dst, resolve(alias, i.a));
        if (alias.isEmpty()) return;
        applyAliases(p, alias);
        copies += alias.size();
    }

    static String resolve(Map<String, String> alias, String name) {
        String cur = name;
        while (alias.containsKey(cur)) cur = alias.get(cur);
        return cur;
    }

    static void applyAliases(SsaProgram p, Map<String, String> alias) {
        for (SsaBlk b : p.blocks) {
            for (SsaIns i : b.ins) {
                if (i.a != null) i.a = resolve(alias, i.a);
                if (i.b != null) i.b = resolve(alias, i.b);
                for (int k = 0; k < i.phiVal.size(); k++) i.phiVal.set(k, resolve(alias, i.phiVal.get(k)));
            }
            if (b.cond != null) b.cond = resolve(alias, b.cond);
            if (b.ret != null) b.ret = resolve(alias, b.ret);
        }
    }

    void eliminateTrivialPhis(SsaProgram p) {
        Map<String, String> alias = new LinkedHashMap<>();
        for (SsaBlk b : p.blocks)
            for (SsaIns i : b.ins) {
                if (i.op != SsaIns.PHI) continue;
                String only = null;
                boolean trivial = true;
                for (String v : i.phiVal) {
                    if (v.equals(i.dst)) continue;
                    if (only == null) only = v;
                    else if (!only.equals(v)) { trivial = false; break; }
                }
                if (trivial && only != null) alias.put(i.dst, only);
            }
        if (alias.isEmpty()) return;
        applyAliases(p, alias);
        phis += alias.size();
    }

    void deadCode(SsaProgram p) {
        boolean changed = true;
        while (changed) {
            changed = false;
            Map<String, Integer> uses = new HashMap<>();
            for (SsaBlk b : p.blocks) {
                for (SsaIns i : b.ins) {
                    if (i.a != null) uses.merge(i.a, 1, Integer::sum);
                    if (i.b != null) uses.merge(i.b, 1, Integer::sum);
                    for (String v : i.phiVal) uses.merge(v, 1, Integer::sum);
                }
                if (b.cond != null) uses.merge(b.cond, 1, Integer::sum);
                if (b.ret != null) uses.merge(b.ret, 1, Integer::sum);
            }
            for (SsaBlk b : p.blocks) {
                for (int k = b.ins.size() - 1; k >= 0; k--) {
                    SsaIns i = b.ins.get(k);
                    if (uses.getOrDefault(i.dst, 0) == 0) { b.ins.remove(k); removed++; changed = true; }
                }
            }
        }
    }
}

public class Solution {
    static long state = 0x51A5A123L;

    static long nextLong() {
        long z = (state += 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Brute force: d dominates n when n becomes unreachable once d is removed (n != entry). */
    static boolean dominatesByDeletion(Cfg cfg, String d, String n) {
        Set<String> seen = new LinkedHashSet<>();
        Deque<String> work = new ArrayDeque<>();
        if (cfg.entry().name.equals(d)) return true;
        seen.add(cfg.entry().name);
        work.add(cfg.entry().name);
        while (!work.isEmpty()) {
            String cur = work.poll();
            if (cur.equals(n)) return false;
            for (String s : cfg.successors(cfg.byName.get(cur)))
                if (!s.equals(d) && seen.add(s)) work.add(s);
        }
        return true;
    }

    static String definitions(SsaProgram p, String name) {
        for (SsaBlk b : p.blocks)
            for (SsaIns i : b.ins) if (i.dst.equals(name)) return b.name;
        return null;
    }

    static void checkSsa(Cfg cfg, SsaProgram p) {
        Map<String, Set<String>> dom = SsaBuilder.dominators(cfg);
        Map<String, Integer> defs = new LinkedHashMap<>();
        for (SsaBlk b : p.blocks)
            for (SsaIns i : b.ins) defs.merge(i.dst, 1, Integer::sum);
        for (Map.Entry<String, Integer> e : defs.entrySet())
            if (e.getValue() != 1) throw new AssertionError("FAILED: " + e.getKey() + " is defined " + e.getValue() + " times");
        for (SsaBlk b : p.blocks) {
            for (SsaIns i : b.ins) {
                if (i.op == SsaIns.PHI) {
                    if (i.phiPred.size() != cfg.byName.get(b.name).pred.size())
                        throw new AssertionError("FAILED: phi in " + b.name + " has " + i.phiPred.size() + " operands");
                    for (String pred : cfg.byName.get(b.name).pred)
                        if (!i.phiPred.contains(pred))
                            throw new AssertionError("FAILED: phi in " + b.name + " misses edge from " + pred);
                    for (int k = 0; k < i.phiVal.size(); k++) {
                        if (i.phiVal.get(k).equals(SsaBuilder.UNDEF)) continue;    // value not defined yet
                        String def = definitions(p, i.phiVal.get(k));
                        if (def == null) continue;                          // parameter: always available
                        if (!dom.get(i.phiPred.get(k)).contains(def))
                            throw new AssertionError("FAILED: operand " + i.phiVal.get(k) + " does not dominate edge "
                                    + i.phiPred.get(k) + " -> " + b.name);
                    }
                } else {
                    for (String use : new String[]{i.a, i.b}) {
                        if (use == null) continue;
                        String def = definitions(p, use);
                        if (def == null) continue;                       // parameter
                        if (!dom.get(b.name).contains(def))
                            throw new AssertionError("FAILED: " + use + " used in " + b.name
                                    + " but defined in " + def);
                    }
                }
            }
            String[] blockUses = {b.cond, b.ret};
            for (String use : blockUses) {
                if (use == null) continue;
                String def = definitions(p, use);
                if (def != null && !dom.get(b.name).contains(def))
                    throw new AssertionError("FAILED: " + use + " used by " + b.name + " but defined in " + def);
            }
        }
    }

    static boolean hasPhi(SsaProgram p) {
        for (SsaBlk b : p.blocks) for (SsaIns i : b.ins) if (i.op == SsaIns.PHI) return true;
        return false;
    }

    public static void main(String[] args) {
        Cfg line = Cfg.straight();
        Map<String, Set<String>> dom = SsaBuilder.dominators(line);
        int pairs = 0, mismatches = 0;
        for (Blk d : line.blocks)
            for (Blk n : line.blocks) {
                pairs++;
                if (dom.get(n.name).contains(d.name) != dominatesByDeletion(line, d.name, n.name)) mismatches++;
            }
        check(mismatches == 0 && pairs == 1, "dominators match the deletion oracle on the straight-line CFG");

        int totalPairs = 0, totalMismatch = 0;
        for (Cfg cfg : List.of(Cfg.diamond(), Cfg.loop(), Cfg.ladder(4))) {
            Map<String, Set<String>> doms = SsaBuilder.dominators(cfg);
            for (Blk d : cfg.blocks)
                for (Blk n : cfg.blocks) {
                    totalPairs++;
                    if (doms.get(n.name).contains(d.name) != dominatesByDeletion(cfg, d.name, n.name)) totalMismatch++;
                }
        }
        check(totalMismatch == 0, "dominators match the deletion oracle on " + totalPairs + " (d, n) pairs");

        for (Cfg cfg : List.of(Cfg.straight(), Cfg.diamond(), Cfg.loop(), Cfg.ladder(2), Cfg.ladder(5), Cfg.ladder(7))) {
            Map<String, Set<String>> doms = SsaBuilder.dominators(cfg);
            Map<String, String> idom = SsaBuilder.immediateDominators(cfg, doms);
            Map<String, Set<String>> df = SsaBuilder.frontiers(cfg, idom);
            for (Blk d : cfg.blocks)
                for (Blk b : cfg.blocks) {
                    boolean expected = false;
                    for (String p : b.pred)
                        if (doms.get(p).contains(d.name) && !strictlyDominated(doms, d.name, b.name)) expected = true;
                    if (df.get(d.name).contains(b.name) != expected)
                        throw new AssertionError("FAILED: dominance frontier of " + d.name + " at " + b.name
                                + ": " + df.get(d.name).contains(b.name) + " vs " + expected);
                }
        }

        SsaProgram lineSsa = SsaBuilder.build(line);
        check(!hasPhi(lineSsa), "a straight-line program needs no phi node");
        checkSsa(line, lineSsa);
        check(Interp.run(line, Interp.inputs(0, 4)) == 10 && Interp.runSsa(lineSsa, line, Interp.inputs(0, 4)) == 10,
                "the SSA form of the straight-line program evaluates identically (6 + in1)");

        Optimizer lineOpt = new Optimizer();
        lineOpt.optimize(lineSsa);
        check(lineOpt.folds >= 2 && lineOpt.copies >= 1 && lineOpt.removed >= 2 && lineOpt.phis == 0,
                "folding, copy propagation and elimination all fired: folds=" + lineOpt.folds + " copies="
                        + lineOpt.copies + " removed=" + lineOpt.removed);
        check(Interp.runSsa(lineSsa, line, Interp.inputs(0, 4)) == 10,
                "z folds to 6 and the copy disappears, so the result stays 6 + in1 = 10, got "
                        + Interp.runSsa(lineSsa, line, Interp.inputs(0, 4)));
        checkSsa(line, lineSsa);
        int instructionsAfter = Optimizer.size(lineSsa);
        new Optimizer().optimize(lineSsa);
        check(Optimizer.size(lineSsa) == instructionsAfter, "the optimizer is idempotent at its fixpoint");

        Cfg dia = Cfg.diamond();
        SsaProgram diaSsa = SsaBuilder.build(dia);
        check(hasPhi(diaSsa), "the diamond needs a phi at the join block");
        checkSsa(dia, diaSsa);
        Cfg lo = Cfg.loop();
        SsaProgram loSsa = SsaBuilder.build(lo);
        check(hasPhi(loSsa), "the loop needs phis at its header");
        checkSsa(lo, loSsa);

        int inputs = 0, worst = 0;
        for (Cfg cfg : List.of(dia, lo, Cfg.ladder(2), Cfg.ladder(4), Cfg.ladder(6))) {
            for (int trial = 0; trial < 12; trial++) {
                Map<String, Integer> in = Interp.inputs((int) ((nextLong() >>> 40) % 6), (int) ((nextLong() >>> 41) % 7) - 3);
                SsaProgram ssa = SsaBuilder.build(cfg);
                checkSsa(cfg, ssa);
                int plain = Interp.run(cfg, in);
                int ssaResult = Interp.runSsa(ssa, cfg, in);
                if (plain != ssaResult) throw new AssertionError("FAILED: SSA result " + ssaResult + " != " + plain);
                Optimizer opt = new Optimizer();
                opt.optimize(ssa);
                checkSsa(cfg, ssa);
                int optimized = Interp.runSsa(ssa, cfg, in);
                if (optimized != plain) throw new AssertionError("FAILED: optimized " + optimized + " != " + plain);
                worst += opt.removed;
                inputs++;
            }
        }
        check(worst > 0 && inputs == 60, "optimization preserved behaviour on " + inputs + " runs and removed "
                + worst + " instructions");

        SsaProgram loSsa2 = SsaBuilder.build(lo);
        Optimizer loOpt = new Optimizer();
        loOpt.optimize(loSsa2);
        boolean junkGone = true;
        for (SsaBlk b : loSsa2.blocks) for (SsaIns i : b.ins) if (i.dst.startsWith("junk")) junkGone = false;
        check(junkGone && loOpt.removed > 0, "the dead multiply inside the loop is eliminated after "
                + loOpt.removed + " removals");
        check(Interp.undefReads == 0 && SsaBuilder.undefinedOperands > 0,
                "no undefined value is ever read (" + Interp.undefReads + " reads) although "
                        + SsaBuilder.undefinedOperands + " phi operands start undefined");
        boolean noCopy = true;
        for (SsaBlk b : loSsa2.blocks) for (SsaIns i : b.ins) if (i.op == Ins.COPY) noCopy = false;
        check(noCopy, "no copy instruction survives the optimizer");
        check(Interp.runSsa(loSsa2, lo, Interp.inputs(4, 0)) == 10 && Interp.run(lo, Interp.inputs(4, 0)) == 10,
                "the optimized loop still sums 4+3+2+1 = 10");

        System.out.println("dominators : " + (pairs + totalPairs) + " (dominator, block) pairs agree with deletion");
        System.out.println("ssa        : " + (1 + 1 + 1) + " hand shapes + 60 random runs, all names single-def and dominated");
        System.out.println("optimizer  : " + lineOpt.folds + " folds, " + lineOpt.copies + " copies, "
                + worst + " dead instructions removed on random programs");
        System.out.println("PASS");
    }

    static boolean strictlyDominated(Map<String, Set<String>> dom, String d, String n) {
        return !n.equals(d) && dom.get(n).contains(d);
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
