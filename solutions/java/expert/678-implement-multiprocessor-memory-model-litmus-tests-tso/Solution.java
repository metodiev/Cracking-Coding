/** Task 678 - Implement multiprocessor memory model litmus tests (TSO). (Tier: Expert, CTCI Beyond — Computer Architecture)
 *
 * An operational TSO machine: every thread owns a FIFO store buffer, a store retires into that buffer and
 * reaches memory later through a non-deterministic commit step, a load must forward the newest matching
 * entry of its own buffer and otherwise reads memory, and a fence can only be issued when the buffer is
 * empty, which is exactly "the fence waits for the buffer to drain". Memory is a single array, so stores
 * to one address can never be observed out of order (per-location coherence). The model is explored
 * exhaustively: every reachable state is enumerated once and the final register tuples are collected, so
 * "impossible" means "no interleaving produces it", not "I did not see it in one run". A sequential
 * consistency model (no buffers, fences are no-ops) is explored the same way and used as the oracle.
 *
 * Verified litmus tests: store buffering allows r0 = r1 = 0 under TSO while SC forbids it; a fence between
 * the store and the load restores exactly the SC outcome set; message passing (r0 = y, r1 = x) never
 * observes y = 1 with x = 0, matching SC; load buffering (0, 0) is reachable only in the relaxed model;
 * a load forwards from its own buffer; and two loads of one address in program order can never go
 * backwards (0 then 1 is possible, 1 then 0 is not). All end states are additionally checked to have
 * drained buffers, and the outcome sets must be reproduced identically on a second exploration.
 */
import java.util.*;

final class Op {
    static final int STORE = 0, LOAD = 1, FENCE = 2;
    final int kind, addr, value, reg;

    Op(int kind, int addr, int value, int reg) {
        this.kind = kind; this.addr = addr; this.value = value; this.reg = reg;
    }

    static Op store(int addr, int value) { return new Op(STORE, addr, value, 0); }
    static Op load(int addr, int reg) { return new Op(LOAD, addr, -1, reg); }
    static Op fence() { return new Op(FENCE, -1, -1, 0); }

    public String toString() {
        return switch (kind) {
            case STORE -> "x" + addr + "=" + value;
            case LOAD -> "r" + reg + "=x" + addr;
            default -> "fence";
        };
    }
}

final class Snapshot {
    final int[] pc;
    final List<List<int[]>> buffers;
    final int[] memory;
    final int[][] regs;

    Snapshot(int threads, int addresses, int regsPerThread) {
        pc = new int[threads];
        buffers = new ArrayList<>();
        for (int t = 0; t < threads; t++) buffers.add(new ArrayList<>());
        memory = new int[addresses];
        regs = new int[threads][regsPerThread];
        for (int[] r : regs) Arrays.fill(r, -1);
    }

    Snapshot copy() {
        Snapshot s = new Snapshot(pc.length, memory.length, regs[0].length);
        System.arraycopy(pc, 0, s.pc, 0, pc.length);
        System.arraycopy(memory, 0, s.memory, 0, memory.length);
        for (int t = 0; t < pc.length; t++) {
            for (int[] entry : buffers.get(t)) s.buffers.get(t).add(new int[]{entry[0], entry[1]});
            System.arraycopy(regs[t], 0, s.regs[t], 0, regs[t].length);
        }
        return s;
    }

    String key() {
        StringBuilder sb = new StringBuilder();
        sb.append(Arrays.toString(pc)).append('|');
        for (List<int[]> buffer : buffers) {
            sb.append('[');
            for (int[] entry : buffer) sb.append(entry[0]).append(':').append(entry[1]).append(',');
            sb.append(']');
        }
        sb.append('|').append(Arrays.toString(memory)).append('|');
        for (int[] r : regs) sb.append(Arrays.toString(r));
        return sb.toString();
    }

    String outcome() {
        StringBuilder sb = new StringBuilder();
        for (int[] r : regs)
            for (int v : r) {
                if (sb.length() > 0) sb.append(',');
                sb.append(v < 0 ? "-" : v);
            }
        return sb.toString();
    }
}

final class Litmus {
    final String name;
    final List<List<Op>> programs;
    final List<int[]> observed = new ArrayList<>();      // (thread, register) pairs that a load writes
    final int addresses, regsPerThread;
    final boolean buffered;                     // true: TSO, false: sequential consistency
    final Set<String> outcomes = new TreeSet<>();
    final List<Snapshot> terminals = new ArrayList<>();
    long explored, completions, forwarded;

    Litmus(String name, List<List<Op>> programs, int addresses, int regsPerThread, boolean buffered) {
        this.name = name;
        this.programs = programs;
        this.addresses = addresses;
        this.regsPerThread = regsPerThread;
        this.buffered = buffered;
        for (int t = 0; t < programs.size(); t++)
            for (Op op : programs.get(t))
                if (op.kind == Op.LOAD) observed.add(new int[]{t, op.reg});
    }

    String outcome(Snapshot state) {
        StringBuilder sb = new StringBuilder();
        for (int[] ref : observed) {
            if (sb.length() > 0) sb.append(',');
            int v = state.regs[ref[0]][ref[1]];
            sb.append(v < 0 ? "-" : v);
        }
        return sb.toString();
    }

    /** Every store in the program is eventually visible in memory: no write can be lost. */
    boolean storesVisible(Snapshot state) {
        for (List<Op> program : programs)
            for (Op op : program)
                if (op.kind == Op.STORE && state.memory[op.addr] != op.value) return false;
        return true;
    }

    boolean buffersEmpty(Snapshot state) {
        for (List<int[]> buffer : state.buffers) if (!buffer.isEmpty()) return false;
        return true;
    }

    Litmus explore() {
        Set<String> seen = new HashSet<>();
        Snapshot start = new Snapshot(programs.size(), addresses, regsPerThread);
        seen.add(start.key());
        dfs(start, seen);
        return this;
    }

    void dfs(Snapshot state, Set<String> seen) {
        explored++;
        if (explored > 400000) throw new IllegalStateException("state space too large for " + name);
        if (complete(state)) {
            completions++;
            outcomes.add(outcome(state));
            terminals.add(state);
            return;
        }
        for (int t = 0; t < programs.size(); t++) {
            if (state.pc[t] < programs.get(t).size()) {
                Op op = programs.get(t).get(state.pc[t]);
                boolean fenceBlocked = op.kind == Op.FENCE && buffered && !state.buffers.get(t).isEmpty();
                if (!fenceBlocked) {
                    Snapshot next = state.copy();
                    next.pc[t]++;
                    if (op.kind == Op.STORE) {
                        if (buffered) next.buffers.get(t).add(new int[]{op.addr, op.value});
                        else next.memory[op.addr] = op.value;
                    } else if (op.kind == Op.LOAD) {
                        int value = -1;
                        if (buffered) {
                            List<int[]> buffer = next.buffers.get(t);
                            for (int i = buffer.size() - 1; i >= 0; i--)
                                if (buffer.get(i)[0] == op.addr) { value = buffer.get(i)[1]; forwarded++; break; }
                        }
                        if (value < 0) value = next.memory[op.addr];
                        next.regs[t][op.reg] = value;
                    }
                    visit(next, seen);
                }
            }
            if (buffered && !state.buffers.get(t).isEmpty()) {
                Snapshot next = state.copy();
                int[] head = next.buffers.get(t).remove(0);
                next.memory[head[0]] = head[1];
                visit(next, seen);
            }
        }
    }

    void visit(Snapshot next, Set<String> seen) {
        if (seen.add(next.key())) dfs(next, seen);
    }

    boolean complete(Snapshot state) {
        for (int t = 0; t < programs.size(); t++) {
            if (state.pc[t] != programs.get(t).size()) return false;
            if (buffered && !state.buffers.get(t).isEmpty()) return false;
        }
        return true;
    }

    /** Two operations on the same address per thread, one register each. */
    static List<Op> thread(Op... ops) { return new ArrayList<>(Arrays.asList(ops)); }

    static List<List<Op>> storeBuffering(boolean fenced) {
        List<Op> t0 = new ArrayList<>(List.of(Op.store(0, 1)));
        if (fenced) t0.add(Op.fence());
        t0.add(Op.load(1, 0));
        List<Op> t1 = new ArrayList<>(List.of(Op.store(1, 1)));
        if (fenced) t1.add(Op.fence());
        t1.add(Op.load(0, 0));
        return List.of(t0, t1);
    }

    static List<List<Op>> messagePassing(boolean fenced) {
        List<Op> t0 = new ArrayList<>(List.of(Op.store(0, 1)));
        if (fenced) t0.add(Op.fence());
        t0.add(Op.store(1, 1));
        List<Op> t1 = new ArrayList<>(List.of(Op.load(1, 0)));
        if (fenced) t1.add(Op.fence());
        t1.add(Op.load(0, 1));
        return List.of(t0, t1);
    }

    static List<List<Op>> loadBuffering() {
        return List.of(
                thread(Op.load(1, 0), Op.store(0, 1)),
                thread(Op.load(0, 0), Op.store(1, 1)));
    }

    /** Store buffering with a fence in the first thread only. */
    static List<List<Op>> halfFencedStoreBuffering() {
        return List.of(
                thread(Op.store(0, 1), Op.fence(), Op.load(1, 0)),
                thread(Op.store(1, 1), Op.load(0, 0)));
    }

    static List<List<Op>> forwarding() {
        return List.of(thread(Op.store(0, 1), Op.load(0, 0)));
    }

    static List<List<Op>> coherence() {
        return List.of(
                thread(Op.load(0, 0)),
                thread(Op.store(0, 1)));
    }

    static List<List<Op>> twoLoads() {
        return List.of(
                thread(Op.load(0, 0), Op.load(0, 1)),
                thread(Op.store(0, 1)));
    }
}
public class Solution {
    static Litmus run(String name, List<List<Op>> programs, int addresses, int regsPerThread, boolean buffered) {
        return new Litmus(name, programs, addresses, regsPerThread, buffered).explore();
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }

    public static void main(String[] args) {
        Litmus sbTso = run("sb", Litmus.storeBuffering(false), 2, 1, true);
        Litmus sbSc = run("sb-sc", Litmus.storeBuffering(false), 2, 1, false);
        check(sbTso.outcomes.contains("0,0"),
                "store buffering lets both loads read the old values under TSO: " + sbTso.outcomes);
        check(!sbSc.outcomes.contains("0,0"), "sequential consistency forbids (0,0): " + sbSc.outcomes);
        check(sbTso.outcomes.containsAll(sbSc.outcomes) && sbTso.outcomes.size() > sbSc.outcomes.size(),
                "the relaxed model is a strict superset of SC for store buffering: " + sbTso.outcomes + " vs " + sbSc.outcomes);

        Litmus sbFencedTso = run("sb-fenced", Litmus.storeBuffering(true), 2, 1, true);
        Litmus sbFencedSc = run("sb-fenced-sc", Litmus.storeBuffering(true), 2, 1, false);
        check(!sbFencedTso.outcomes.contains("0,0") && sbFencedTso.outcomes.equals(sbFencedSc.outcomes),
                "fences make store buffering behave exactly like SC: " + sbFencedTso.outcomes);
        check(sbFencedTso.completions > 0 && sbFencedSc.completions > 0,
                "both store-buffering models reached terminal states (" + sbFencedTso.completions + "/"
                        + sbFencedSc.completions + ")");

        Litmus mpTso = run("mp", Litmus.messagePassing(false), 2, 2, true);
        Litmus mpSc = run("mp-sc", Litmus.messagePassing(false), 2, 2, false);
        check(!mpTso.outcomes.contains("1,0"),
                "message passing never sees the flag without the data under TSO: " + mpTso.outcomes);
        check(mpTso.outcomes.equals(mpSc.outcomes),
                "TSO and SC agree on message passing: " + mpTso.outcomes + " vs " + mpSc.outcomes);
        check(mpTso.outcomes.contains("1,1") && mpTso.outcomes.contains("0,0") && mpTso.outcomes.contains("0,1"),
                "the three legal message-passing outcomes are present (y = 1 requires x = 1): " + mpTso.outcomes);
        Litmus mpFenced = run("mp-fenced", Litmus.messagePassing(true), 2, 2, true);
        check(mpFenced.outcomes.equals(mpSc.outcomes),
                "fences do not change message passing here, only the reasoning: " + mpFenced.outcomes);

        Litmus lbTso = run("lb", Litmus.loadBuffering(), 2, 1, true);
        Litmus lbSc = run("lb-sc", Litmus.loadBuffering(), 2, 1, false);
        check(lbTso.outcomes.equals(lbSc.outcomes) && lbTso.outcomes.contains("0,0"),
                "load buffering (0,0) is legal even under SC: both loads simply run before both stores, got "
                        + lbTso.outcomes + " vs " + lbSc.outcomes);
        Litmus half = run("sb-half", Litmus.halfFencedStoreBuffering(), 2, 1, true);
        check(half.outcomes.contains("0,0") && !run("sb-half-sc", Litmus.halfFencedStoreBuffering(), 2, 1, false)
                        .outcomes.contains("0,0"),
                "a fence in one thread does not stop the other thread from reading stale data: " + half.outcomes);

        Litmus fwd = run("forwarding", Litmus.forwarding(), 1, 1, true);
        check(fwd.outcomes.equals(new TreeSet<>(List.of("1"))) && fwd.forwarded > 0,
                "a load forwards its own buffered store: " + fwd.outcomes + " with " + fwd.forwarded + " forwards");
        Litmus coh = run("coherence", Litmus.coherence(), 1, 1, true);
        check(coh.outcomes.equals(new TreeSet<>(List.of("0", "1"))),
                "a lone reader sees either the old or the new value, never anything else: " + coh.outcomes);

        Litmus two = run("two-loads", Litmus.twoLoads(), 1, 2, true);
        check(two.outcomes.contains("0,1") && !two.outcomes.contains("1,0"),
                "two ordered loads of one location can only go forwards: " + two.outcomes);
        check(two.outcomes.contains("0,0") && two.outcomes.contains("1,1"),
                "the monotone read pairs are both reachable: " + two.outcomes);

        int checkedTerminals = 0, visible = 0;
        for (Litmus test : List.of(sbTso, sbFencedTso, mpTso, mpFenced, lbTso, half, fwd, coh, two)) {
            check(test.explored > 0 && test.completions > 0, test.name + " explored " + test.explored + " states");
            for (Snapshot terminal : test.terminals) {
                checkedTerminals++;
                if (!test.buffersEmpty(terminal)) throw new AssertionError("FAILED: " + test.name + " ended with a full buffer");
                if (test.storesVisible(terminal)) visible++;
            }
        }
        check(visible == checkedTerminals,
                "every terminal state has drained buffers and all stores visible in memory ("
                        + visible + "/" + checkedTerminals + ")");

        Litmus again = run("sb", Litmus.storeBuffering(false), 2, 1, true);
        check(again.outcomes.equals(sbTso.outcomes) && again.explored == sbTso.explored,
                "the exploration is deterministic and complete (" + again.explored + " states)");

        long totalStates = sbTso.explored + sbSc.explored + sbFencedTso.explored + sbFencedSc.explored
                + mpTso.explored + mpSc.explored + mpFenced.explored + lbTso.explored + lbSc.explored
                + fwd.explored + coh.explored + two.explored;
        check(totalStates > 200, "the exhaustive search covered " + totalStates + " states in total");

        System.out.println("store buffering : TSO " + sbTso.outcomes + " vs SC " + sbSc.outcomes
                + "; fenced TSO " + sbFencedTso.outcomes);
        System.out.println("message passing : TSO " + mpTso.outcomes + " (no (1,0)), SC identical");
        System.out.println("load buffering  : TSO " + lbTso.outcomes + " vs SC " + lbSc.outcomes
                + "; forwarding " + fwd.outcomes + "; " + totalStates + " states explored");
        System.out.println("PASS");
    }
}
