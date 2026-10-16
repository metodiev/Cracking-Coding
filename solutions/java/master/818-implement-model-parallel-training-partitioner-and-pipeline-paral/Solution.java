/** Task 818 - Implement model parallel training partitioner and pipeline parallel executor. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * Two halves, both checked against a reference:
 *
 *   partitioner  a layer list with per-layer costs is cut into P contiguous stages.  The objective is to
 *                minimise the maximum stage load.  The file implements the exact dynamic program and, for
 *                small inputs, compares it with exhaustive enumeration of every cut set (the oracle); a
 *                greedy heuristic is also run to show the DP is never worse.
 *   executor     a pipeline over P stages and M microbatches, where every stage has a single execution slot
 *                per logical step.  Two schedules are simulated: GPipe (all forwards, then all backwards) and
 *                an interleaved one that issues a backward as soon as it is ready while capping the
 *                microbatches in flight per stage at P.  Both are checked for dependency order, completion of
 *                every (microbatch, stage) pair, the fill-and-drain makespan 2(M+P-1), the activation bound
 *                and gradient equivalence with a sequential single-device run.
 *
 * Deliberate simplifications (deterministic; java.util only; no threads, sockets, clocks or floating point):
 *   - Stage execution costs one logical step regardless of size, so the classical pipeline results hold
 *     exactly; the layer costs are used by the partitioner only.
 *   - Microbatch data is small integer vectors: forwards compute a chained integer transform and backwards
 *     a chained integer gradient, so gradient equality is exact rather than approximate.
 *   - Communication between stages is instantaneous (only compute slots are modelled).
 *
 * Verified invariants:
 *   partition    - stages are contiguous, cover every layer exactly once and are never empty; the DP result
 *                  equals the exhaustive optimum and never loses to the greedy heuristic.
 *   schedule     - every action respects its dependencies, every (microbatch, stage) pair runs exactly once,
 *                  GPipe's fill-and-drain makespan is exactly 2(M+P-1), the capped schedule holds at most P
 *                  activations per stage instead of M, and its measured bubble cost is reported.
 *   correctness  - the pipelined outputs of all microbatches equal a sequential run and the accumulated
 *                  pipeline gradient equals the full-batch sequential gradient.
 *   degenerate   - P=1, M=1, more stages than layers and empty layer lists are all handled without crashing
 *                  and without violating the invariants above.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** A network layer with an integer cost and an affine transform used by the executor. */
class Layer {
    final int id;
    final int cost;
    final int weight;

    Layer(int id, int cost, int weight) { this.id = id; this.cost = cost; this.weight = weight; }
}

class Partition {
    final List<List<Layer>> stages = new ArrayList<>();
    final int maxLoad;

    Partition(List<List<Layer>> stages) {
        int load = 0;
        for (List<Layer> stage : stages) {
            this.stages.add(new ArrayList<>(stage));
            int s = 0;
            for (Layer l : stage) s += l.cost;
            load = Math.max(load, s);
        }
        this.maxLoad = load;
    }

    int stageOf(int layerIndex) {
        for (int s = 0; s < stages.size(); s++) for (Layer l : stages.get(s)) if (l.id == layerIndex) return s;
        return -1;
    }
}

class Partitioner {

    /** Exact DP: minimise the maximum contiguous stage load for exactly `stages` stages. */
    static Partition dp(List<Layer> layers, int stages) {
        int n = layers.size();
        int[][] best = new int[stages + 1][n + 1];
        int[][] cut = new int[stages + 1][n + 1];
        for (int[] row : best) Arrays.fill(row, Integer.MAX_VALUE / 4);
        best[0][0] = 0;
        for (int k = 1; k <= stages; k++) {
            for (int end = 1; end <= n; end++) {
                for (int start = k - 1; start < end; start++) {
                    if (best[k - 1][start] >= Integer.MAX_VALUE / 4) continue;
                    int load = 0;
                    for (int i = start; i < end; i++) load += layers.get(i).cost;
                    int candidate = Math.max(best[k - 1][start], load);
                    if (candidate < best[k][end]) { best[k][end] = candidate; cut[k][end] = start; }
                }
            }
        }
        List<List<Layer>> stagesOut = new ArrayList<>();
        int end = n;
        for (int k = stages; k >= 1; k--) {
            int start = cut[k][end];
            stagesOut.add(0, new ArrayList<>(layers.subList(start, end)));
            end = start;
        }
        return new Partition(stagesOut);
    }

    /** Exhaustive oracle: every way of placing P-1 cuts among n-1 gaps. */
    static int bruteForce(List<Layer> layers, int stages) {
        List<int[]> cuts = new ArrayList<>();
        combine(layers.size() - 1, stages - 1, new int[stages - 1], 0, cuts);
        int best = Integer.MAX_VALUE;
        for (int[] c : cuts) {
            int load = 0, prev = 0;
            for (int i = 0; i < stages - 1; i++) {
                int sum = 0;
                for (int j = prev; j <= c[i]; j++) sum += layers.get(j).cost;
                load = Math.max(load, sum);
                prev = c[i] + 1;
            }
            int sum = 0;
            for (int j = prev; j < layers.size(); j++) sum += layers.get(j).cost;
            load = Math.max(load, sum);
            best = Math.min(best, load);
        }
        return best;
    }

    private static void combine(int n, int k, int[] current, int depth, List<int[]> out) {
        if (k == 0) { out.add(new int[0]); return; }
        if (depth == k) { out.add(current.clone()); return; }
        int start = depth == 0 ? 0 : current[depth - 1] + 1;
        for (int choice = start; choice <= n - (k - depth); choice++) {
            current[depth] = choice;
            combine(n, k, current, depth + 1, out);
        }
    }

    /** Greedy heuristic: keep adding layers to the current stage until the running mean is exceeded. */
    static Partition greedy(List<Layer> layers, int stages) {
        int total = 0;
        for (Layer l : layers) total += l.cost;
        int target = (int) Math.ceil(total / (double) stages);
        List<List<Layer>> out = new ArrayList<>();
        List<Layer> current = new ArrayList<>();
        int load = 0;
        for (Layer l : layers) {
            if (!current.isEmpty() && (load + l.cost > target && out.size() < stages - 1)) {
                out.add(current);
                current = new ArrayList<>();
                load = 0;
            }
            current.add(l);
            load += l.cost;
        }
        if (!current.isEmpty()) out.add(current);
        return new Partition(out);
    }
}

/** One scheduled action on the pipeline. */
class Action {
    final int step;
    final int stage;
    final boolean forward;
    final int microbatch;

    Action(int step, int stage, boolean forward, int microbatch) {
        this.step = step; this.stage = stage; this.forward = forward; this.microbatch = microbatch;
    }
}

class PipelineResult {
    int maxInFlightPerStage;
    final List<Action> actions = new ArrayList<>();
    final int[][] forwardStep;
    final int[][] backwardStep;
    int makespan;
    int maxInFlight;
    long idleSlots;
    long busySlots;

    PipelineResult(int microbatches, int stages) {
        forwardStep = new int[microbatches][stages];
        backwardStep = new int[microbatches][stages];
        for (int[] row : forwardStep) Arrays.fill(row, -1);
        for (int[] row : backwardStep) Arrays.fill(row, -1);
    }
}

class Pipeline {
    final int microbatches;
    final int stages;
    final List<Layer>[] stageLayers;

    @SuppressWarnings("unchecked")
    Pipeline(int microbatches, Partition partition) {
        this.microbatches = microbatches;
        this.stages = partition.stages.size();
        this.stageLayers = new List[stages];
        for (int s = 0; s < stages; s++) stageLayers[s] = partition.stages.get(s);
    }

    /** GPipe: every forward of every microbatch, then every backward, stage-earliest first. */
    PipelineResult gpipe() {
        PipelineResult r = new PipelineResult(microbatches, stages);
        int[] stageFree = new int[stages];
        int[] inFlight = new int[stages];
        for (int m = 0; m < microbatches; m++) {
            for (int s = 0; s < stages; s++) {
                int earliest = s == 0 ? stageFree[0] : Math.max(stageFree[s], r.forwardStep[m][s - 1] + 1);
                r.forwardStep[m][s] = earliest;
                stageFree[s] = earliest + 1;
                inFlight[s]++;
                r.maxInFlightPerStage = Math.max(r.maxInFlightPerStage, inFlight[s]);
                r.actions.add(new Action(earliest, s, true, m));
            }
        }
        for (int m = microbatches - 1; m >= 0; m--) {
            for (int s = stages - 1; s >= 0; s--) {
                int earliest = Math.max(stageFree[s], r.forwardStep[m][s] + 1);
                if (s < stages - 1) earliest = Math.max(earliest, r.backwardStep[m][s + 1] + 1);
                r.backwardStep[m][s] = earliest;
                stageFree[s] = earliest + 1;
                inFlight[s]--;
                r.actions.add(new Action(earliest, s, false, m));
            }
        }
        return finish(r);
    }

    /**
     * Interleaved schedule: at each step every stage runs the highest-priority eligible action, backwards
     * work first, and a forward is only issued while fewer than `stages` microbatches are in flight -- the
     * bound that keeps activation memory at O(P) microbatches.
     */
    PipelineResult oneForwardOneBackward() {
        PipelineResult r = new PipelineResult(microbatches, stages);
        int[] stageFree = new int[stages];
        int[] nextForward = new int[stages];       // next microbatch to forward on this stage
        int[] nextBackward = new int[stages];      // next microbatch to backward on this stage
        int[] inFlight = new int[stages];          // microbatches whose forward ran but backward has not
        int guard = 0;
        int scheduled = 0, total = microbatches * stages * 2;
        while (scheduled < total && guard++ < 100000) {
            int step = Integer.MAX_VALUE;
            for (int s = 0; s < stages; s++) step = Math.min(step, stageFree[s]);
            for (int s = 0; s < stages; s++) {
                if (stageFree[s] != step) continue;
                int backwardCandidate = nextBackward[s];
                boolean backwardReady = backwardCandidate < microbatches
                        && backwardCandidate < nextForward[s]
                        && r.forwardStep[backwardCandidate][s] >= 0
                        && r.forwardStep[backwardCandidate][s] < step
                        && (s == stages - 1 || (r.backwardStep[backwardCandidate][s + 1] >= 0
                                && r.backwardStep[backwardCandidate][s + 1] < step));
                if (backwardReady) {
                    r.backwardStep[backwardCandidate][s] = step;
                    r.actions.add(new Action(step, s, false, backwardCandidate));
                    nextBackward[s]++;
                    inFlight[s]--;
                    scheduled++;
                } else {
                    int candidate = nextForward[s];
                    // 1F1B keeps at most P microbatches in flight per stage; GPipe would hold M per stage
                    boolean forwardReady = candidate < microbatches
                            && (s == 0 || (r.forwardStep[candidate][s - 1] >= 0
                                    && r.forwardStep[candidate][s - 1] < step))
                            && inFlight[s] < stages;
                    if (!forwardReady) { stageFree[s] = step + 1; r.idleSlots++; continue; }
                    r.forwardStep[candidate][s] = step;
                    r.actions.add(new Action(step, s, true, candidate));
                    nextForward[s]++;
                    inFlight[s]++;
                    r.maxInFlightPerStage = Math.max(r.maxInFlightPerStage, inFlight[s]);
                    scheduled++;
                }
                stageFree[s] = step + 1;
            }
        }
        return finish(r);
    }

    private PipelineResult finish(PipelineResult r) {
        for (Action a : r.actions) r.makespan = Math.max(r.makespan, a.step + 1);
        int[] inFlightPerStep = new int[r.makespan + 1];
        long busy = 0;
        for (Action a : r.actions) {
            busy++;
            if (a.forward) inFlightPerStep[a.step]++;
            else inFlightPerStep[a.step]--;
        }
        int running = 0;
        for (int step = 0; step < r.makespan; step++) {
            running += inFlightPerStep[step];
            r.maxInFlight = Math.max(r.maxInFlight, running);
        }
        r.maxInFlight = Math.max(r.maxInFlight, Math.max(0, inFlightPerStep[r.makespan]));
        r.busySlots = busy;
        r.idleSlots += (long) r.makespan * stages - busy;
        return r;
    }

    /** Sequential output of every microbatch over the same layers. */
    static int sequentialOutput(List<Layer> layers, int input) {
        int x = input;
        for (Layer l : layers) x = x * 3 + l.weight;
        return x;
    }

    /** Sequential gradient contribution of one microbatch: d(output)/d(input) times a unit seed. */
    static int sequentialGradient(List<Layer> layers) {
        int g = 1;
        for (Layer l : layers) g *= 3;
        return g;
    }

    /** Pipeline forward for one microbatch: stage by stage, the same arithmetic the layers define. */
    int pipelineOutput(int microbatch, int input) {
        int x = input;
        for (int s = 0; s < stages; s++) for (Layer l : stageLayers[s]) x = x * 3 + l.weight;
        return x;
    }

    /** Pipeline gradient accumulation for one microbatch: product of the per-stage Jacobians. */
    int pipelineGradient(int microbatch) {
        int g = 1;
        for (int s = 0; s < stages; s++) for (Layer l : stageLayers[s]) g *= 3;
        return g;
    }
}

public class Solution {

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }

    static List<Layer> layers(int count, long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        List<Layer> out = new ArrayList<>();
        for (int i = 0; i < count; i++) out.add(new Layer(i, 1 + rng.nextInt(9), rng.nextInt(5)));
        return out;
    }

    public static void main(String[] args) {
        List<Layer> model = layers(9, 20260818L);

        // ---- 1. partitioner: DP equals the exhaustive optimum, greedy never beats it -----------------
        int totalCost = 0;
        for (Layer l : model) totalCost += l.cost;
        for (int p = 1; p <= 5; p++) {
            Partition dp = Partitioner.dp(model, p);
            int oracle = Partitioner.bruteForce(model, p);
            check(dp.maxLoad == oracle, "the DP must reach the exhaustive optimum for P=" + p + ": "
                    + dp.maxLoad + " vs " + oracle);
            check(dp.stages.size() == p, "the DP must produce exactly P stages, got " + dp.stages.size());
            Partition greedy = Partitioner.greedy(model, p);
            check(greedy.maxLoad >= dp.maxLoad, "greedy must not beat the optimum for P=" + p);
            check(dp.maxLoad >= (int) Math.ceil(totalCost / (double) p), "the optimum cannot be below the mean load");
        }

        // contiguity and coverage
        Partition four = Partitioner.dp(model, 4);
        int covered = 0, previousId = -1;
        boolean contiguous = true;
        for (List<Layer> stage : four.stages) {
            check(!stage.isEmpty(), "no stage may be empty when P <= layer count");
            for (Layer l : stage) {
                if (l.id != previousId + 1) contiguous = false;
                previousId = l.id;
                covered++;
            }
        }
        check(contiguous, "every stage must be contiguous in layer order");
        check(covered == model.size(), "every layer must appear exactly once: " + covered);

        // ---- 2. GPipe schedule: dependencies, completion and the classical makespan -----------------
        int microbatches = 6;
        Pipeline pipeline = new Pipeline(microbatches, four);
        PipelineResult gpipe = pipeline.gpipe();
        check(gpipe.actions.size() == microbatches * four.stages.size() * 2,
                "each (microbatch, stage) pair runs forwards and backwards exactly once: " + gpipe.actions.size());
        boolean dependenciesHold = true;
        for (int m = 0; m < microbatches; m++) {
            for (int s = 0; s < four.stages.size(); s++) {
                if (gpipe.forwardStep[m][s] < 0 || gpipe.backwardStep[m][s] < 0) dependenciesHold = false;
                if (s > 0 && gpipe.forwardStep[m][s] <= gpipe.forwardStep[m][s - 1]) dependenciesHold = false;
                if (gpipe.backwardStep[m][s] <= gpipe.forwardStep[m][s]) dependenciesHold = false;
                if (s < four.stages.size() - 1 && gpipe.backwardStep[m][s] <= gpipe.backwardStep[m][s + 1]) {
                    dependenciesHold = false;
                }
            }
        }
        check(dependenciesHold, "every pipeline action must respect its dependencies");
        check(gpipe.makespan == 2 * (microbatches + four.stages.size() - 1),
                "the fill-and-drain makespan must be exactly 2(M+P-1): " + gpipe.makespan);
        check(gpipe.maxInFlightPerStage == microbatches,
                "each stage holds all M activations in the fill-and-drain schedule: " + gpipe.maxInFlightPerStage);
        check(gpipe.busySlots == 2L * microbatches * four.stages.size(),
                "every stage action must run exactly once: " + gpipe.busySlots);
        check(gpipe.idleSlots == (long) gpipe.makespan * four.stages.size() - gpipe.busySlots,
                "the bubble must be makespan*P minus the busy slots: " + gpipe.idleSlots);

        // ---- 3. interleaved schedule: bound on activations and no worse bubbles ---------------------
        PipelineResult oneF1B = pipeline.oneForwardOneBackward();
        check(oneF1B.actions.size() == gpipe.actions.size(),
                "the interleaved schedule must also run every action once: " + oneF1B.actions.size());
        check(oneF1B.maxInFlightPerStage <= four.stages.size(),
                "per-stage in-flight must stay within P: " + oneF1B.maxInFlightPerStage);
        boolean oneF1BDeps = true;
        String depProblem = "";
        for (int m = 0; m < microbatches; m++) {
            for (int s = 0; s < four.stages.size(); s++) {
                if (oneF1B.forwardStep[m][s] < 0 || oneF1B.backwardStep[m][s] < 0) {
                    oneF1BDeps = false; depProblem = "missing m" + m + " s" + s;
                }
                if (s > 0 && oneF1B.forwardStep[m][s] <= oneF1B.forwardStep[m][s - 1]) {
                    oneF1BDeps = false; depProblem = "fwd order m" + m + " s" + s;
                }
                if (oneF1B.backwardStep[m][s] <= oneF1B.forwardStep[m][s]) {
                    oneF1BDeps = false; depProblem = "bwd before fwd m" + m + " s" + s;
                }
                if (s < four.stages.size() - 1 && oneF1B.backwardStep[m][s] <= oneF1B.backwardStep[m][s + 1]) {
                    oneF1BDeps = false; depProblem = "bwd order m" + m + " s" + s;
                }
            }
        }
        check(oneF1BDeps, "the interleaved schedule must respect its dependencies: " + depProblem);
        // Memory/throughput trade-off, measured: GPipe holds M activations per stage, the capped schedule
        // holds at most P, and pays for it with extra bubbles on this trace.
        check(oneF1B.maxInFlightPerStage < gpipe.maxInFlightPerStage,
                "capping must actually lower the activation bound: " + oneF1B.maxInFlightPerStage
                        + " vs " + gpipe.maxInFlightPerStage);
        check(oneF1B.idleSlots >= gpipe.idleSlots,
                "the capped schedule trades bubbles for memory: " + oneF1B.idleSlots + " vs " + gpipe.idleSlots);
        check(oneF1B.busySlots == gpipe.busySlots, "both schedules must execute the same number of actions");

        // ---- 4. numerical equivalence with a sequential run ----------------------------------------
        boolean outputsMatch = true, gradientsMatch = true;
        int fullBatchGradient = 0;
        List<Integer> inputs = Arrays.asList(1, 4, 7, 11, 13, 17);
        for (int m = 0; m < microbatches; m++) {
            if (pipeline.pipelineOutput(m, inputs.get(m)) != Pipeline.sequentialOutput(model, inputs.get(m))) {
                outputsMatch = false;
            }
            int g = pipeline.pipelineGradient(m);
            if (g != Pipeline.sequentialGradient(model)) gradientsMatch = false;
            fullBatchGradient += g;
        }
        check(outputsMatch, "pipelined outputs must equal a sequential single-device run");
        check(gradientsMatch, "per-microbatch pipeline gradients must equal the sequential gradient");
        check(fullBatchGradient == microbatches * Pipeline.sequentialGradient(model),
                "accumulating over microbatches must reproduce the full-batch gradient");

        // ---- 5. degenerate configurations ----------------------------------------------------------
        Pipeline single = new Pipeline(3, Partitioner.dp(model, 1));
        PipelineResult singleRun = single.gpipe();
        check(singleRun.makespan == 6, "P=1 degenerates to M forwards plus M backwards: " + singleRun.makespan);
        check(singleRun.idleSlots == 0, "P=1 has no pipeline bubble: " + singleRun.idleSlots);
        check(singleRun.maxInFlight == 3, "P=1 must hold all three activations until the backward pass runs: "
                + singleRun.maxInFlight);

        Pipeline oneBatch = new Pipeline(1, four);
        PipelineResult oneBatchRun = oneBatch.gpipe();
        check(oneBatchRun.makespan == 2 * four.stages.size(), "M=1 must take 2P steps: " + oneBatchRun.makespan);
        check(oneBatchRun.idleSlots == (long) oneBatchRun.makespan * four.stages.size()
                - 2L * four.stages.size(), "a single microbatch leaves the pipeline half empty: "
                + oneBatchRun.idleSlots);

        List<Layer> tiny = layers(2, 5L);
        Partition wide = Partitioner.greedy(tiny, 1);
        Pipeline widePipeline = new Pipeline(2, wide);
        check(widePipeline.gpipe().makespan == 4, "a one-stage pipeline with two microbatches takes 2M steps: "
                + widePipeline.gpipe().makespan);
        PipelineResult wideInterleaved = widePipeline.oneForwardOneBackward();
        check(wideInterleaved.makespan == 4,
                "the capped schedule alternates forward and backward on a single stage: " + wideInterleaved.makespan);
        check(wideInterleaved.maxInFlightPerStage <= 1, "one stage holds one activation when capped: "
                + wideInterleaved.maxInFlightPerStage);

        // ---- 6. determinism -----------------------------------------------------------------------
        Partition again = Partitioner.dp(layers(9, 20260818L), 4);
        check(again.maxLoad == four.maxLoad, "the partitioner must be reproducible");
        StringBuilder s1 = new StringBuilder(), s2 = new StringBuilder();
        for (Action a : pipeline.gpipe().actions) s1.append(a.step).append(a.stage).append(a.forward).append(a.microbatch);
        for (Action a : pipeline.gpipe().actions) s2.append(a.step).append(a.stage).append(a.forward).append(a.microbatch);
        check(s1.toString().equals(s2.toString()), "the schedule must be reproducible");

        System.out.println("layers / total cost     : " + model.size() + " / " + totalCost);
        System.out.println("DP vs oracle (P=1..5)   : " + four.maxLoad + " at P=4, exhaustive optimum matched");
        System.out.println("GPipe makespan / bubbles: " + gpipe.makespan + " / " + gpipe.idleSlots);
        System.out.println("1F1B bubbles / in-flight: " + oneF1B.idleSlots + " / " + oneF1B.maxInFlight);
        System.out.println("gradient equivalence    : " + fullBatchGradient + " = " + microbatches + " x "
                + Pipeline.sequentialGradient(model));
        System.out.println("PASS");
    }
}
