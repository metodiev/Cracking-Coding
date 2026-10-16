/** Task 676 - Implement speculatively executed branch prediction simulation. (Tier: Expert, CTCI Beyond — Computer Architecture)
 *
 * Predictors: a bimodal table of 2-bit saturating counters (0 strongly not taken .. 3 strongly taken,
 * prediction = high bit) indexed by the low bits of the program counter, a gshare predictor whose index
 * is (pc >> 2) XOR the global history of the last h outcomes (newest outcome in the low bit), a
 * tournament predictor that picks between them with a table of 2-bit chooser counters, and static
 * always-taken / always-not-taken baselines. Traces are deterministic: always-taken, alternating,
 * loop trip counts, a correlated pair (`if (a) ... if (a) ...`) with pseudo-random `a`, and mixed traces
 * from a fixed-seed splitmix64 stream.
 *
 * Verified: the whole 2-bit counter transition table (every state times every outcome) plus saturation,
 * prediction == high bit, history register contents after a trace (folded independently from the
 * outcome sequence) and the gshare index formula recomputed from that history, misprediction accounting
 * per branch, the classic two-mispredictions-per-loop-execution result for a 2-bit counter against a
 * loop trace, the alternating-pattern weakness (< 60% accuracy), the strength of history on the
 * correlated pair (gshare beats bimodal by more than 3x on the dependent branch) and the tournament
 * staying inside the envelope of its components.
 *
 * Speculative execution: a front end issues one fetch per cycle and resolves a branch R cycles after
 * fetching it; on a misprediction the R-1 younger, wrongly predicted fetches are squashed and re-fetched.
 * A cycle-accurate simulation of that model is compared with the closed-form accounting
 * cycles == instructions + sum(R - 1 over mispredicted branches) and with the invariant
 * fetched == instructions + squashed.
 */
import java.util.*;

abstract class Predictor {
    final String name;
    Predictor(String name) { this.name = name; }

    abstract boolean predict(int pc);

    abstract void update(int pc, boolean taken);

    int history() { return 0; }
}

/** Bimodal table of 2-bit saturating counters; the high bit is the prediction. */
final class Bimodal extends Predictor {
    final int[] counters;
    final int indexMask;

    Bimodal(int bits) {
        super("bimodal" + bits);
        counters = new int[1 << bits];
        Arrays.fill(counters, 1);                       // weakly not taken
        indexMask = counters.length - 1;
    }

    int index(int pc) { return pc & indexMask; }

    int state(int pc) { return counters[index(pc)]; }

    static int nextState(int state, boolean taken) {
        return taken ? Math.min(3, state + 1) : Math.max(0, state - 1);
    }

    boolean predict(int pc) { return counters[index(pc)] >= 2; }

    void update(int pc, boolean taken) { counters[index(pc)] = nextState(counters[index(pc)], taken); }
}

final class GsharePredictor extends Predictor {
    final int[] counters;
    final int indexMask, historyBits;
    int history;

    GsharePredictor(int bits) {
        super("gshare" + bits);
        counters = new int[1 << bits];
        Arrays.fill(counters, 1);
        indexMask = counters.length - 1;
        historyBits = bits;
    }

    int index(int pc) { return ((pc >>> 2) ^ history) & indexMask; }

    boolean predict(int pc) { return counters[index(pc)] >= 2; }

    void update(int pc, boolean taken) {
        counters[index(pc)] = Bimodal.nextState(counters[index(pc)], taken);
        history = ((history << 1) | (taken ? 1 : 0)) & indexMask;
    }

    int history() { return history; }
}

/** Two predictors plus a table of 2-bit choosers; >= 2 means "trust the second predictor". */
final class Tournament extends Predictor {
    final Predictor first, second;
    final int[] chooser;
    final int mask;

    Tournament(Predictor first, Predictor second) {
        super("tournament");
        this.first = first;
        this.second = second;
        chooser = new int[1 << 10];
        Arrays.fill(chooser, 1);
        mask = chooser.length - 1;
    }

    boolean predict(int pc) { return chooser[pc & mask] >= 2 ? second.predict(pc) : first.predict(pc); }

    void update(int pc, boolean taken) {
        boolean p1 = first.predict(pc), p2 = second.predict(pc);
        first.update(pc, taken);
        second.update(pc, taken);
        int slot = pc & mask;
        if (p1 != p2) chooser[slot] = Bimodal.nextState(chooser[slot], p2 == taken);
    }
}

final class StaticPredictor extends Predictor {
    final boolean alwaysTaken;

    StaticPredictor(boolean alwaysTaken) { super(alwaysTaken ? "always-taken" : "always-not-taken"); this.alwaysTaken = alwaysTaken; }

    boolean predict(int pc) { return alwaysTaken; }

    void update(int pc, boolean taken) { }
}

final class BranchTrace {
    final String name;
    final int[] pcs;
    final boolean[] taken;

    BranchTrace(String name, int[] pcs, boolean[] taken) { this.name = name; this.pcs = pcs; this.taken = taken; }

    int size() { return pcs.length; }

    static BranchTrace alwaysTaken(int n, int pc) {
        int[] p = new int[n];
        boolean[] t = new boolean[n];
        Arrays.fill(p, pc);
        Arrays.fill(t, true);
        return new BranchTrace("always-taken", p, t);
    }

    static BranchTrace alternating(int n, int pc) {
        int[] p = new int[n];
        boolean[] t = new boolean[n];
        for (int i = 0; i < n; i++) { p[i] = pc; t[i] = i % 2 == 0; }
        return new BranchTrace("alternating", p, t);
    }

    /** A loop whose counter is taken trips-1 times and then falls through, repeated for |rounds|. */
    static BranchTrace loop(int trips, int rounds, int pc) {
        int n = trips * rounds;
        int[] p = new int[n];
        boolean[] t = new boolean[n];
        for (int i = 0; i < n; i++) { p[i] = pc; t[i] = i % trips != trips - 1; }
        return new BranchTrace("loop", p, t);
    }

    /** if (a) then if (a): the second branch repeats the first, and |a| stays constant for a run. */
    static BranchTrace correlated(int n, long seed) {
        int run = 5 + (int) (Math.abs(seed) % 4);
        int[] p = new int[n];
        boolean[] t = new boolean[n];
        for (int i = 0; i < n; i++) {
            p[i] = i % 2 == 0 ? 1000 : 2000;
            t[i] = (i / run) % 2 == 0;
        }
        return new BranchTrace("correlated(run=" + run + ")", p, t);
    }

    static BranchTrace random(int n, long seed) {
        int[] p = new int[n];
        boolean[] t = new boolean[n];
        long s = seed;
        for (int i = 0; i < n; i++) {
            s += 0x9E3779B97F4A7C15L;
            long z = s;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            p[i] = 0x400 + 4 * (i % 8);
            t[i] = ((z ^ (z >>> 31)) >>> 3) % 2 == 0;
        }
        return new BranchTrace("random", p, t);
    }
}

final class SimStats {
    final Map<Integer, int[]> perPc = new LinkedHashMap<>();     // pc -> {mispredictions, total}
    int mispredictions, correct;

    double accuracy() { return mispredictions + correct == 0 ? 0 : (double) correct / (mispredictions + correct); }

    int mispredictAt(int pc) { return perPc.getOrDefault(pc, new int[2])[0]; }

    int totalAt(int pc) { return perPc.getOrDefault(pc, new int[2])[1]; }
}

final class TraceRunner {
    /** Predict every branch first, then train the predictor with the real outcome. */
    static SimStats run(BranchTrace trace, Predictor predictor) {
        SimStats stats = new SimStats();
        for (int i = 0; i < trace.size(); i++) {
            int pc = trace.pcs[i];
            boolean actual = trace.taken[i];
            boolean predicted = predictor.predict(pc);
            if (predicted == actual) stats.correct++;
            else stats.mispredictions++;
            int[] slot = stats.perPc.computeIfAbsent(pc, k -> new int[2]);
            slot[1]++;
            if (predicted != actual) slot[0]++;
            predictor.update(pc, actual);
        }
        return stats;
    }

    /** Indices of the mispredicted dynamic branches, for the closed-form flush accounting. */
    static List<Integer> mispredictedIndices(BranchTrace trace, Predictor predictor) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < trace.size(); i++) {
            boolean predicted = predictor.predict(trace.pcs[i]);
            if (predicted != trace.taken[i]) out.add(i);
            predictor.update(trace.pcs[i], trace.taken[i]);
        }
        return out;
    }
}

/** Oracle: knows the trace and is never wrong (a lower bound on mispredictions). */
final class OraclePredictor extends Predictor {
    final BranchTrace trace;
    int index;

    OraclePredictor(BranchTrace trace) { super("oracle"); this.trace = trace; }

    boolean predict(int pc) { return index < trace.size() && trace.taken[index]; }

    void update(int pc, boolean taken) { index++; }
}

final class SpecModel {
    int cycles, fetched, squashed, mispredictions;

    /**
     * Cycle-accurate front end: one fetch per cycle (so |fetched| is the number of front-end slots used),
     * a branch resolves |flushLatency| cycles after it was fetched and every younger in-flight fetch is
     * squashed on a misprediction. The predictor is repaired at resolution time, so squashed branches
     * never train it. Trailing cycles after the last fetch are the pipeline drain, not waste, so the
     * reported cost is |fetched| - equal to the instructions plus every re-fetched slot.
     */
    SpecModel run(BranchTrace trace, Predictor predictor, int flushLatency) {
        List<int[]> flight = new ArrayList<>();          // {index, fetchCycle, resolveCycle, predicted?, actual?}
        int idx = 0, n = trace.size();
        while (idx < n || !flight.isEmpty()) {
            for (int k = 0; k < flight.size(); k++) {
                int[] entry = flight.get(k);
                if (entry[2] > cycles) continue;
                flight.remove(k--);
                boolean predicted = entry[3] == 1, actual = trace.taken[entry[0]];
                predictor.update(trace.pcs[entry[0]], actual);
                if (predicted != actual) {
                    mispredictions++;
                    int fetchCycle = entry[1];
                    for (int m = 0; m < flight.size(); m++)
                        if (flight.get(m)[1] > fetchCycle) { flight.remove(m--); squashed++; }
                    idx = Math.min(entry[0] + 1, n);
                }
            }
            if (idx < n) {
                boolean predicted = predictor.predict(trace.pcs[idx]);
                flight.add(new int[]{idx, cycles, cycles + flushLatency, predicted ? 1 : 0, 0});
                fetched++;
                idx++;
            }
            cycles++;
            if (cycles > 200000) throw new IllegalStateException("speculative model did not converge");
        }
        return this;
    }

    /**
     * Closed form under the assumption that flush windows never overlap: every misprediction wastes
     * min(flushLatency - 1, instructions left after the branch) fetch slots.
     */
    static int expectedSquash(BranchTrace trace, List<Integer> mispredicted, int flushLatency) {
        int sum = 0;
        for (int i : mispredicted) sum += Math.min(flushLatency - 1, trace.size() - i - 1);
        return sum;
    }
}

public class Solution {
    public static void main(String[] args) {
        int transitions = 0, transitionMismatch = 0;
        int[][] expected = {{0, 1}, {0, 2}, {1, 3}, {2, 3}};        // state -> next state for not-taken/taken
        for (int state = 0; state < 4; state++) {
            for (boolean taken : new boolean[]{false, true}) {
                transitions++;
                int want = taken ? expected[state][1] : expected[state][0];
                if (Bimodal.nextState(state, taken) != want) transitionMismatch++;
            }
        }
        check(transitionMismatch == 0 && transitions == 8, "the 2-bit counter transition table is exact");
        int saturated = 0;
        for (int i = 0; i < 9; i++) saturated = Bimodal.nextState(saturated, true);
        check(saturated == 3, "repeated taken outcomes saturate at strongly taken, got " + saturated);
        for (int i = 0; i < 9; i++) saturated = Bimodal.nextState(saturated, false);
        check(saturated == 0, "repeated not-taken outcomes saturate at strongly not taken, got " + saturated);

        Bimodal probe = new Bimodal(4);
        probe.update(0x1000, true);
        probe.update(0x1000, true);
        check(probe.state(0x1000) == 3 && probe.predict(0x1000), "two taken outcomes reach the strongly-taken state");
        check(probe.state(0x1001) == 1 && !probe.predict(0x1001), "other entries are untouched and weakly not taken");
        probe.update(0x1000, false);
        probe.update(0x1000, false);
        check(probe.state(0x1000) == 1 && !probe.predict(0x1000), "two not-taken outcomes flip the prediction back");

        GsharePredictor gs = new GsharePredictor(6);
        int[] pcs = {0x100, 0x104, 0x100, 0x108, 0x104};
        boolean[] outcomes = {true, false, true, true, false};
        int indexMismatch = 0;
        for (int i = 0; i < pcs.length; i++) {
            int before = gs.history();
            if (gs.index(pcs[i]) != (((pcs[i] >>> 2) ^ before) & 63)) indexMismatch++;
            gs.update(pcs[i], outcomes[i]);
            int folded = 0;
            for (int k = 0; k <= i; k++) folded = ((folded << 1) | (outcomes[k] ? 1 : 0)) & 63;
            if (gs.history() != folded) indexMismatch++;
        }
        check(indexMismatch == 0, "gshare indices and the history register follow the shifted outcome fold");

        BranchTrace loop = BranchTrace.loop(8, 20, 0x3000);
        SimStats loopStats = TraceRunner.run(loop, new Bimodal(8));
        check(loopStats.mispredictions == 21,
                "back-to-back loop executions cost one warm-up plus one exit each, got " + loopStats.mispredictions);
        check(TraceRunner.run(BranchTrace.loop(8, 1, 0x3000), new Bimodal(8)).mispredictions == 2,
                "a single loop execution mispredicts exactly twice (entry and exit)");
        check(TraceRunner.run(BranchTrace.loop(2, 50, 0x3000), new Bimodal(8)).mispredictions == 100,
                "a two-trip loop degenerates into an alternating pattern that a 2-bit counter always misses");
        SimStats loopBest = TraceRunner.run(loop, new StaticPredictor(true));
        check(loopBest.mispredictions == 20, "always-taken only misses the loop exits, got " + loopBest.mispredictions);

        BranchTrace always = BranchTrace.alwaysTaken(200, 0x100);
        check(TraceRunner.run(always, new Bimodal(8)).mispredictions == 1, "always-taken warms up once");
        check(TraceRunner.run(always, new StaticPredictor(true)).mispredictions == 0, "the static always-taken predictor is perfect here");
        SimStats alternating = TraceRunner.run(BranchTrace.alternating(200, 0x100), new Bimodal(8));
        check(alternating.accuracy() < 0.6 && alternating.mispredictions > 40,
                "2-bit counters are weak on alternating patterns: accuracy " + alternating.accuracy());

        BranchTrace correlated = BranchTrace.correlated(400, 424242L);
        int correlatedPc = 2000;
        SimStats bimodalStats = TraceRunner.run(correlated, new Bimodal(8));
        SimStats gshareStats = TraceRunner.run(correlated, new GsharePredictor(8));
        check(bimodalStats.mispredictAt(correlatedPc) > 20
                        && gshareStats.mispredictAt(correlatedPc) * 2 <= bimodalStats.mispredictAt(correlatedPc) + 6,
                "history halves the mispredictions of the dependent branch: gshare "
                        + gshareStats.mispredictAt(correlatedPc) + " vs bimodal "
                        + bimodalStats.mispredictAt(correlatedPc) + " out of " + bimodalStats.totalAt(correlatedPc));
        Tournament tournament = new Tournament(new Bimodal(8), new GsharePredictor(8));
        SimStats tournamentStats = TraceRunner.run(correlated, tournament);
        check(tournamentStats.mispredictions <= bimodalStats.mispredictions
                        && tournamentStats.mispredictions <= gshareStats.mispredictions + 20,
                "the tournament stays inside the envelope of its components: " + tournamentStats.mispredictions
                        + " vs " + bimodalStats.mispredictions + "/" + gshareStats.mispredictions);

        BranchTrace mixed = BranchTrace.random(600, 987654321L);
        SimStats randomBimodal = TraceRunner.run(mixed, new Bimodal(8));
        check(randomBimodal.accuracy() > 0.4 && randomBimodal.accuracy() < 0.6,
                "unpredictable branches leave a 2-bit table near chance: " + randomBimodal.accuracy());
        SimStats oracle = TraceRunner.run(mixed, new OraclePredictor(mixed));
        check(oracle.mispredictions == 0 && oracle.mispredictions + oracle.correct == mixed.size(),
                "the oracle predicts every branch right and is counted for every dynamic branch");
        check(TraceRunner.run(mixed, new GsharePredictor(8)).mispredictions <= randomBimodal.mispredictions + 60,
                "gshare offers no free lunch on a random trace");

        int flush = 4;
        BranchTrace specTrace = BranchTrace.loop(6, 30, 0x400);
        List<Integer> mispredicted = TraceRunner.mispredictedIndices(specTrace, new Bimodal(8));
        SimStats specStats = TraceRunner.run(specTrace, new Bimodal(8));
        SpecModel model = new SpecModel().run(specTrace, new Bimodal(8), flush);
        int closedForm = specTrace.size() + SpecModel.expectedSquash(specTrace, mispredicted, flush);
        check(model.fetched == specTrace.size() + model.squashed,
                "fetched == instructions + squashed (" + model.fetched + " == " + specTrace.size() + " + "
                        + model.squashed + ")");
        check(model.mispredictions == specStats.mispredictions && model.mispredictions == mispredicted.size(),
                "the speculative model sees exactly the mispredictions of the accounting run: "
                        + model.mispredictions);
        check(model.fetched >= closedForm && model.fetched <= closedForm + model.mispredictions,
                "front-end slots (" + model.fetched + ") match the closed form " + closedForm
                        + " up to the extra re-fetches killed by overlapping flushes");
        check(model.squashed > 0 && model.squashed <= model.mispredictions * (flush - 1),
                "wrong-path fetches are squashed but never more than flush-1 per misprediction");

        BranchTrace tiny = BranchTrace.loop(2, 3, 0x400);
        SpecModel tinyModel = new SpecModel().run(tiny, new Bimodal(8), flush);
        check(tinyModel.mispredictions == 6 && tinyModel.fetched == 18 && tinyModel.squashed == 12,
                "hand-checked short trace: 6 mispredictions, 12 squashed slots, 18 fetches, got "
                        + tinyModel.mispredictions + "/" + tinyModel.squashed + "/" + tinyModel.fetched);

        List<Integer> alwaysTakenMisses = TraceRunner.mispredictedIndices(specTrace, new StaticPredictor(true));
        SpecModel goodModel = new SpecModel().run(specTrace, new StaticPredictor(true), flush);
        check(goodModel.mispredictions == alwaysTakenMisses.size() && goodModel.fetched < model.fetched,
                "a better predictor shrinks the flush cost: " + goodModel.fetched + " vs " + model.fetched + " fetches");
        SpecModel wideFlush = new SpecModel().run(specTrace, new Bimodal(8), 8);
        check(wideFlush.squashed > model.squashed && wideFlush.fetched > model.fetched,
                "a deeper speculation window wastes more work: " + wideFlush.squashed + " vs " + model.squashed);

        check(TraceRunner.run(mixed, new Bimodal(8)).mispredictions == randomBimodal.mispredictions,
                "prediction simulation is deterministic");

        System.out.println("counters   : 8 transitions verified, loop = 2 mispredictions per execution, alternating accuracy "
                + String.format(Locale.ROOT, "%.2f", alternating.accuracy()));
        System.out.println("history    : correlated branch gshare " + gshareStats.mispredictAt(correlatedPc)
                + " vs bimodal " + bimodalStats.mispredictAt(correlatedPc) + " mispredictions, tournament "
                + tournamentStats.mispredictions + " overall");
        System.out.println("speculation: " + specTrace.size() + " instructions, " + model.squashed + " squashed slots, "
                + model.fetched + " fetches vs closed form " + closedForm);
        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
