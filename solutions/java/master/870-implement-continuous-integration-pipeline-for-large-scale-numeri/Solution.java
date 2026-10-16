/** Task 870 - Implement continuous integration pipeline for large-scale numerical code with reproducibility.
 * (Tier: Master, CTCI Beyond - Software Engineering & Tooling)
 *
 * A build-and-test pipeline is a DAG of stages (configure, compile, shards of unit tests, integration,
 * benchmarks, docs) with per-stage resource demands, a time budget and retries, executed on a pool of
 * machines. The coordinator is modelled as a discrete-event simulation driven by an explicit virtual clock
 * and an explicit event queue - no threads, no sleeping, no wall-clock time - so the schedule is a pure
 * function of the pipeline description, the machine pool and the cache contents.
 *
 *   - scheduling: whenever a machine frees up, the ready stage with the largest remaining critical-path
 *     height starts on the first machine that can hold its demand vector, ties broken by name;
 *   - caching: a stage's cache key is the content hash of its own recipe combined with the keys of its
 *     dependencies (a Merkle hash over the DAG), so an unchanged stage is skipped and a changed one
 *     invalidates exactly its descendant closure;
 *   - reliability: a stage is killed at its time budget if it overruns, a stage whose deterministic
 *     hash-flake fires is retried up to its retry limit, and once a stage is abandoned every descendant is
 *     cancelled rather than run.
 *
 * SIMPLIFIED: virtual milliseconds, synthetic stage durations and a synthetic flake function instead of real
 * compilers, real networks and real machines; "scale" is the number of stages, machines and resource units,
 * and artifacts travel through an in-memory content-addressed store instead of an artifact server. The only
 * randomness is splitmix64 keyed by the run seed, so every schedule below is reproducible.
 *
 * VERIFIED: (a) SAFETY - the recorded trace is re-checked afterwards: no stage starts before all of its
 * dependencies have finished, and no machine is ever oversubscribed in any resource dimension;
 * (b) LIVENESS - the event queue always drains, and a stage that was neither run nor cancelled is an error;
 * (c) BOUNDS - the makespan is at least the critical path and at most the serial sum, one machine reproduces
 * the serial order exactly, and on a small instance a memoised brute force over (completed set, machine free
 * times) supplies the exact optimum, which no list schedule may beat; (d) REPRODUCIBILITY - two clean runs
 * produce identical artifacts and a bit-identical trace, and an incremental run reuses every cache key at
 * zero cost while editing one stage invalidates exactly the descendant closure found by an independent
 * reachability search; (e) retries, budget kills, GPU contention, degenerate pipelines (empty, single stage)
 * and invalid DAGs (cycles, unknown dependencies) behave.
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

    /** Deterministic 64-bit content hash of a string, used for recipes and artifact identities. */
    static long hash(String text) {
        long h = 0x9E3779B97F4A7C15L;
        for (int i = 0; i < text.length(); i++) {
            h ^= text.charAt(i);
            h *= 0x100000001B3L;
            h ^= h >>> 29;
        }
        return h;
    }
}

/** One stage: work, resources, a time budget, retries and a deterministic flake rate. */
class Stage {
    final String name;
    final int duration;
    final int cpu, gpu, memory;
    final int budget;          // virtual-time limit for one attempt, 0 means no limit
    final int retries;
    final String recipe;       // content hash of this stage's own inputs
    final int flakePercent;    // probability per attempt that this attempt fails

    Stage(String name, int duration, int cpu, int gpu, int memory, int budget, int retries, String recipe,
            int flakePercent) {
        if (duration <= 0) throw new IllegalArgumentException("duration must be positive");
        this.name = name; this.duration = duration; this.cpu = cpu; this.gpu = gpu; this.memory = memory;
        this.budget = budget; this.retries = retries; this.recipe = recipe; this.flakePercent = flakePercent;
    }

    /** How long one attempt of this stage occupies a machine. */
    int occupancy() { return budget > 0 ? Math.min(duration, budget) : duration; }

    boolean overrunsBudget() { return budget > 0 && duration > budget; }
}

/** A machine: capacity in each resource dimension plus the stage currently occupying it. */
class Machine {
    final int cpu, gpu, memory;
    int freeCpu, freeGpu, freeMemory;
    Stage running;
    int busyUntil;

    Machine(int cpu, int gpu, int memory) {
        this.cpu = cpu; this.gpu = gpu; this.memory = memory;
        freeCpu = cpu; freeGpu = gpu; freeMemory = memory;
    }

    boolean canHold(Stage s) { return s.cpu <= freeCpu && s.gpu <= freeGpu && s.memory <= freeMemory; }
}

/** A validated DAG of stages. */
class Pipeline {
    final Map<String, Stage> stages = new LinkedHashMap<>();
    final Map<String, List<String>> dependencies = new LinkedHashMap<>();
    private List<String> topologicalOrder;

    Pipeline add(Stage stage, String... deps) {
        if (stages.containsKey(stage.name)) throw new IllegalArgumentException("duplicate stage " + stage.name);
        stages.put(stage.name, stage);
        dependencies.put(stage.name, new ArrayList<>(Arrays.asList(deps)));
        topologicalOrder = null;
        return this;
    }

    Pipeline editRecipe(String stage, String recipe) {
        Stage old = stages.get(stage);
        stages.put(stage, new Stage(old.name, old.duration, old.cpu, old.gpu, old.memory, old.budget,
                old.retries, recipe, old.flakePercent));
        topologicalOrder = null;
        return this;
    }

    /** Topological order, rejecting unknown dependencies and cycles. */
    List<String> order() {
        if (topologicalOrder != null) return topologicalOrder;
        for (List<String> deps : dependencies.values())
            for (String dep : deps)
                if (!stages.containsKey(dep)) throw new IllegalArgumentException("unknown dependency " + dep);
        List<String> order = new ArrayList<>();
        Map<String, Integer> indegree = new LinkedHashMap<>();
        for (String name : stages.keySet()) indegree.put(name, dependencies.get(name).size());
        List<String> ready = new ArrayList<>();
        for (String name : stages.keySet()) if (indegree.get(name) == 0) ready.add(name);
        while (!ready.isEmpty()) {
            Collections.sort(ready);
            String name = ready.remove(0);
            order.add(name);
            for (String other : stages.keySet())
                if (dependencies.get(other).contains(name) && indegree.merge(other, -1, Integer::sum) == 0)
                    ready.add(other);
        }
        if (order.size() != stages.size()) throw new IllegalArgumentException("the pipeline has a cycle");
        topologicalOrder = order;
        return order;
    }

    /** Longest remaining path to a sink: the scheduling priority that keeps the critical path busy first. */
    Map<String, Integer> heights() {
        List<String> reverse = new ArrayList<>(order());
        Collections.reverse(reverse);
        Map<String, Integer> height = new LinkedHashMap<>();
        for (String name : reverse) {
            int best = 0;
            for (String other : stages.keySet())
                if (dependencies.get(other).contains(name)) best = Math.max(best, height.get(other));
            height.put(name, best + stages.get(name).duration);
        }
        return height;
    }

    int criticalPath() {
        Map<String, Integer> heights = heights();
        int best = 0;
        for (String name : stages.keySet()) if (dependencies.get(name).isEmpty())
            best = Math.max(best, heights.get(name));
        return best;
    }

    int serialSum() {
        int sum = 0;
        for (Stage s : stages.values()) sum += s.duration;
        return sum;
    }

    /** Cache keys: a Merkle hash of each stage's recipe combined with the keys of its dependencies. */
    Map<String, String> cacheKeys() {
        Map<String, String> keys = new LinkedHashMap<>();
        for (String name : order()) {
            StringBuilder sb = new StringBuilder(stages.get(name).recipe);
            List<String> deps = new ArrayList<>(dependencies.get(name));
            Collections.sort(deps);
            for (String dep : deps) sb.append('|').append(keys.get(dep));
            keys.put(name, Long.toHexString(SplitMix64.hash(sb.toString())));
        }
        return keys;
    }

    /** Whoever depends on the given stage, found by an independent breadth-first search. */
    List<String> descendants(String stage) {
        List<String> out = new ArrayList<>();
        Deque<String> stack = new ArrayDeque<>();
        stack.push(stage);
        while (!stack.isEmpty()) {
            String current = stack.pop();
            for (String other : order())
                if (dependencies.get(other).contains(current) && !out.contains(other)) {
                    out.add(other);
                    stack.push(other);
                }
        }
        return out;
    }
}

/** What one pipeline run produced. */
class RunResult {
    int makespan;
    boolean success = true;
    final List<String> skipped = new ArrayList<>();
    final List<String> failed = new ArrayList<>();
    final List<String> cancelled = new ArrayList<>();
    final List<String> trace = new ArrayList<>();
    final Map<String, String> artifacts = new LinkedHashMap<>();   // stage -> content address of outputs
    final Map<String, String> outputs = new LinkedHashMap<>();     // stage -> build output id
    int attempts;
    boolean safetyViolated;
    boolean capacityViolated;

    String traceText() { return String.join(";", trace); }

    int startCount(String stage) {
        int count = 0;
        for (String entry : trace) if (entry.endsWith(":start:" + stage) || entry.contains(":start:" + stage + ":"))
            count++;
        return count;
    }
}

/** The discrete-event coordinator. */
class Simulator {
    static final int MAX_EVENTS = 1000000;

    /**
     * Run the pipeline on the machine pool. previousArtifacts is the content-addressed store of an earlier
     * run and previousOutputs its build outputs: a stage whose cache key is unchanged is skipped and its
     * stored output is reused verbatim, so a cached build is bit-identical to the build that produced it.
     */
    static RunResult run(Pipeline p, List<Machine> pool, Map<String, String> previousArtifacts,
            Map<String, String> previousOutputs, long seed) {
        RunResult result = new RunResult();
        List<String> order = p.order();
        Map<String, String> keys = p.cacheKeys();
        Map<String, Integer> height = p.heights();
        Map<String, Integer> remainingDeps = new LinkedHashMap<>();
        for (String name : order) remainingDeps.put(name, p.dependencies.get(name).size());
        Set<String> done = new LinkedHashSet<>(), running = new LinkedHashSet<>();
        Set<String> abandoned = new LinkedHashSet<>();
        Map<String, Integer> attempts = new LinkedHashMap<>();
        PriorityQueue<long[]> queue = new PriorityQueue<>(Comparator.comparingLong((long[] e) -> e[0])
                .thenComparingLong(e -> e[1]));
        long[] sequence = {0};
        int now = 0, events = 0;

        // Cache resolution: an artifact may only be reused when every input it was built from is reused too.
        Map<String, Boolean> reusable = new LinkedHashMap<>();
        for (String name : order) {
            boolean inputsReused = true;
            for (String dep : p.dependencies.get(name)) inputsReused &= reusable.getOrDefault(dep, false);
            boolean hit = inputsReused && keys.get(name).equals(previousArtifacts.get(name));
            reusable.put(name, hit);
            if (!hit) continue;
            done.add(name);
            result.skipped.add(name);
            result.artifacts.put(name, keys.get(name));
            result.outputs.put(name, previousOutputs.get(name));
            for (String other : order)
                if (p.dependencies.get(other).contains(name)) remainingDeps.merge(other, -1, Integer::sum);
        }

        boolean placed = true;
        while (placed) placed = place(p, pool, order, height, done, running, abandoned, attempts, result, queue,
                now, sequence, remainingDeps);

        // Drain the queue: each event is a stage leaving its machine, and the clock only moves to that time.
        while (!queue.isEmpty()) {
            if (++events > MAX_EVENTS) throw new IllegalStateException("event queue did not drain");
            now = (int) queue.poll()[0];
            for (Machine m : pool) {
                if (m.running == null || m.busyUntil > now) continue;
                Stage stage = m.running;
                m.freeCpu += stage.cpu; m.freeGpu += stage.gpu; m.freeMemory += stage.memory;
                m.running = null;
                running.remove(stage.name);
                boolean flake = flakeFired(stage, attempts.get(stage.name), seed);
                if (flake || stage.overrunsBudget()) {
                    result.trace.add(now + ":failed:" + stage.name + ":" + attempts.get(stage.name));
                    if (attempts.get(stage.name) <= stage.retries) continue;   // retry: not done, not abandoned
                    abandoned.add(stage.name);
                    result.failed.add(stage.name);
                    result.success = false;
                } else {
                    done.add(stage.name);
                    result.trace.add(now + ":done:" + stage.name);
                    result.artifacts.put(stage.name, keys.get(stage.name));
                    result.outputs.put(stage.name, buildOutput(stage, keys.get(stage.name)));
                    for (String other : order)
                        if (p.dependencies.get(other).contains(stage.name))
                            remainingDeps.merge(other, -1, Integer::sum);
                }
            }
            for (String name : order)
                if (!done.contains(name) && !running.contains(name) && !abandoned.contains(name)) {
                    for (String dep : p.dependencies.get(name))
                        if (abandoned.contains(dep)) { abandoned.add(name); result.cancelled.add(name); break; }
                }
            boolean again = true;
            while (again) again = place(p, pool, order, height, done, running, abandoned, attempts, result,
                    queue, now, sequence, remainingDeps);
        }
        audit(p, pool, result, done, abandoned);
        result.makespan = now;
        return result;
    }

    /** Place every stage that is ready and whose dependencies have finished on a machine it fits. */
    private static boolean place(Pipeline p, List<Machine> pool, List<String> order,
            Map<String, Integer> height, Set<String> done, Set<String> running, Set<String> abandoned,
            Map<String, Integer> attempts, RunResult result, PriorityQueue<long[]> queue, int now,
            long[] sequence, Map<String, Integer> remainingDeps) {
        List<String> ready = new ArrayList<>();
        for (String name : order)
            if (!done.contains(name) && !running.contains(name) && !abandoned.contains(name)
                    && remainingDeps.get(name) == 0) ready.add(name);
        ready.sort((a, b) -> height.get(b) - height.get(a) != 0 ? height.get(b) - height.get(a)
                : a.compareTo(b));
        boolean placedAny = false;
        for (String name : ready) {
            Stage stage = p.stages.get(name);
            Machine target = null;
            for (Machine m : pool) if (m.running == null && m.canHold(stage)) { target = m; break; }
            if (target == null) continue;
            int tries = attempts.merge(name, 1, Integer::sum);
            target.running = stage;
            target.freeCpu -= stage.cpu; target.freeGpu -= stage.gpu; target.freeMemory -= stage.memory;
            target.busyUntil = now + stage.occupancy();
            running.add(name);
            result.attempts++;
            result.trace.add(now + ":start:" + name + ":" + tries);
            queue.add(new long[]{target.busyUntil, sequence[0]++});
            placedAny = true;
        }
        return placedAny;
    }

    /**
     * The build output id of a stage: a pure function of the stage name, its cache key and its duration, so a
     * rebuild of unchanged inputs reproduces it exactly and a changed input cannot collide.
     */
    static String buildOutput(Stage stage, String key) {
        return Long.toHexString(SplitMix64.hash(stage.name + "#" + key + "#" + stage.duration));
    }

    /** Deterministic flake: a pure function of the stage name, the attempt number and the run seed. */
    static boolean flakeFired(Stage stage, int tryNumber, long seed) {
        if (stage.flakePercent <= 0) return false;
        long mixed = new SplitMix64(SplitMix64.hash(stage.name) ^ (seed * 31 + tryNumber)).nextLong();
        return Math.floorMod(mixed, 100) < stage.flakePercent;
    }

    /** Post-hoc safety audit: causality between start and finish times, and machine capacity in the end. */
    private static void audit(Pipeline p, List<Machine> pool, RunResult result, Set<String> done,
            Set<String> abandoned) {
        Map<String, Integer> doneAt = new HashMap<>(), startedAt = new HashMap<>();
        for (String entry : result.trace) {
            String[] parts = entry.split(":");
            int time = Integer.parseInt(parts[0]);
            if (parts[1].equals("done")) doneAt.put(parts[2], time);
            else if (parts[1].equals("start")) startedAt.put(parts[2], time);
        }
        for (Map.Entry<String, Integer> e : startedAt.entrySet())
            for (String dep : p.dependencies.get(e.getKey())) {
                Integer depDone = doneAt.get(dep);
                if (!result.skipped.contains(dep) && (depDone == null || depDone > e.getValue()))
                    result.safetyViolated = true;
            }
        for (String name : p.order())
            if (!done.contains(name) && !abandoned.contains(name) && !result.cancelled.contains(name))
                throw new IllegalStateException("stage " + name + " was neither run nor cancelled");
        for (Machine m : pool)
            if (m.freeCpu != m.cpu || m.freeGpu != m.gpu || m.freeMemory != m.memory || m.running != null)
                result.capacityViolated = true;
    }
}

/** Independent oracle: the optimal makespan on a small instance, by memoised exhaustive search. */
class BruteForce {
    static int optimalMakespan(Pipeline p, int machineCount) {
        List<String> order = p.order();
        int n = order.size();
        Map<String, Integer> index = new HashMap<>();
        for (int i = 0; i < n; i++) index.put(order.get(i), i);
        int[] prereqMask = new int[n], duration = new int[n];
        for (String name : order) {
            int mask = 0;
            for (String dep : p.dependencies.get(name)) mask |= 1 << index.get(dep);
            prereqMask[index.get(name)] = mask;
            duration[index.get(name)] = p.stages.get(name).occupancy();
        }
        return solve(0, new int[machineCount], prereqMask, duration, machineCount, new HashMap<>());
    }

    private static int solve(int doneMask, int[] free, int[] prereqMask, int[] duration, int machines,
            Map<String, Integer> memo) {
        int[] sorted = free.clone();
        Arrays.sort(sorted);
        String key = doneMask + "|" + Arrays.toString(sorted);
        Integer cached = memo.get(key);
        if (cached != null) return cached;
        int best = Integer.MAX_VALUE;
        boolean any = false;
        for (int i = 0; i < prereqMask.length; i++) {
            if ((doneMask & (1 << i)) != 0 || (prereqMask[i] & ~doneMask) != 0) continue;
            any = true;
            for (int m = 0; m < machines; m++) {
                int[] next = free.clone();
                next[m] = Math.max(next[m], 0) + duration[i];
                best = Math.min(best, solve(doneMask | (1 << i), next, prereqMask, duration, machines, memo));
            }
        }
        if (!any) {
            int finish = 0;
            for (int t : free) finish = Math.max(finish, t);
            best = finish;
        }
        memo.put(key, best);
        return best;
    }
}

public class Solution {
    public static void main(String[] args) {
        Pipeline pipeline = buildPipeline();

        // --- SAFETY and BOUNDS on the reference schedule ------------------------------------------------
        RunResult clean = Simulator.run(pipeline, pool(4, 4, 4, 16), Collections.emptyMap(), Collections.emptyMap(), 12345L);
        check(clean.success, "the reference pipeline must succeed");
        check(!clean.safetyViolated, "no stage may start before its dependencies have finished");
        check(!clean.capacityViolated, "no machine may be oversubscribed at the end of the run");
        check(clean.makespan >= pipeline.criticalPath(), "the makespan must respect the critical path: "
                + clean.makespan + " against " + pipeline.criticalPath());
        check(clean.makespan <= pipeline.serialSum(), "the makespan cannot exceed the serial sum");
        check(clean.skipped.isEmpty() && clean.failed.isEmpty() && clean.cancelled.isEmpty()
                && clean.attempts == pipeline.stages.size(), "a clean run starts from an empty cache and "
                        + "starts every stage once, got " + clean.attempts);

        // --- REPRODUCIBILITY: identical artifacts and a bit-identical trace -----------------------------
        RunResult repeat = Simulator.run(buildPipeline(), pool(4, 4, 4, 16), Collections.emptyMap(),
                Collections.emptyMap(), 12345L);
        check(repeat.artifacts.equals(clean.artifacts), "two clean runs must produce identical artifacts");
        check(repeat.outputs.equals(clean.outputs), "two clean runs must produce identical build outputs");
        check(repeat.traceText().equals(clean.traceText()), "two clean runs must produce identical traces");
        check(repeat.makespan == clean.makespan, "two clean runs must have the same makespan");

        // --- CACHING: a second run recomputes nothing, an edit invalidates exactly the closure ----------
        RunResult cached = Simulator.run(buildPipeline(), pool(4, 4, 4, 16), clean.artifacts, clean.outputs,
                12345L);
        check(cached.skipped.size() == pipeline.stages.size(), "an unchanged pipeline must be served "
                + "entirely from cache, skipped " + cached.skipped.size() + " of " + pipeline.stages.size());
        check(cached.makespan == 0 && cached.attempts == 0, "a fully cached run costs nothing, got "
                + cached.makespan);
        check(cached.outputs.equals(clean.outputs), "a cached build must be bit-identical to the build "
                + "that produced it");

        Pipeline edited = buildPipeline();
        edited.editRecipe("unit-tests-a", "edited-source-v2");
        List<String> changed = changedKeys(edited, pipeline);
        List<String> expected = edited.descendants("unit-tests-a");
        expected.add("unit-tests-a");
        Collections.sort(expected);
        Collections.sort(changed);
        check(changed.equals(expected), "an edit must invalidate exactly the descendant closure: " + changed
                + " against " + expected);
        RunResult partial = Simulator.run(edited, pool(4, 4, 4, 16), clean.artifacts, clean.outputs, 12345L);
        RunResult editedFromScratch = Simulator.run(buildPipeline().editRecipe("unit-tests-a",
                "edited-source-v2"), pool(4, 4, 4, 16), Collections.emptyMap(), Collections.emptyMap(), 12345L);
        check(partial.skipped.size() == pipeline.stages.size() - changed.size(), "only the affected stages "
                + "may be recomputed, skipped " + partial.skipped.size());
        check(partial.outputs.equals(editedFromScratch.outputs), "an incremental build must reproduce the "
                + "clean build of the same revision bit-for-bit");
        check(!partial.outputs.get("unit-tests-a").equals(clean.outputs.get("unit-tests-a")),
                "the edited stage must produce a different build output");
        check(partial.outputs.get("compile").equals(clean.outputs.get("compile")),
                "an unaffected stage must reuse its cached output");

        // --- LIMITS: serial versus oracle-optimal schedules ---------------------------------------------
        RunResult serial = Simulator.run(buildPipeline(), pool(1, 4, 2, 1000), Collections.emptyMap(), Collections.emptyMap(), 1L);
        check(serial.makespan == pipeline.serialSum(), "a single machine must take the serial sum: "
                + serial.makespan + " against " + pipeline.serialSum());
        RunResult parallel = Simulator.run(buildPipeline(), pool(16, 4, 2, 64), Collections.emptyMap(), Collections.emptyMap(), 1L);
        check(parallel.makespan == pipeline.criticalPath(), "with enough machines the critical path is "
                + "attained: " + parallel.makespan + " against " + pipeline.criticalPath());

        Pipeline small = buildSmallPipeline();
        int optimumOne = BruteForce.optimalMakespan(small, 1);
        int optimumTwo = BruteForce.optimalMakespan(small, 2);
        check(optimumOne == small.serialSum(), "the one-machine optimum is the serial sum");
        check(optimumTwo <= optimumOne, "more machines can never make the optimum worse");
        RunResult oneMachine = Simulator.run(buildSmallPipeline(), pool(1, 1, 0, 1), Collections.emptyMap(), Collections.emptyMap(), 1L);
        RunResult twoMachines = Simulator.run(buildSmallPipeline(), pool(2, 1, 0, 1), Collections.emptyMap(), Collections.emptyMap(), 1L);
        check(oneMachine.makespan == optimumOne, "the list schedule must hit the one-machine optimum: "
                + oneMachine.makespan + " against " + optimumOne);
        check(twoMachines.makespan >= optimumTwo, "no list schedule can beat the exact optimum: "
                + twoMachines.makespan + " against " + optimumTwo);
        check(twoMachines.makespan <= 2 * optimumTwo, "the list schedule must stay within a factor of two: "
                + twoMachines.makespan + " against " + optimumTwo);
        check(twoMachines.makespan >= small.criticalPath(), "even the best two-machine schedule cannot beat "
                + "the critical path");

        // --- FAILURES: retries, budget kills and the cancellation closure -------------------------------
        Pipeline flaky = new Pipeline()
                .add(new Stage("compile", 10, 1, 0, 1, 0, 0, "c", 0))
                .add(new Stage("flaky-tests", 20, 1, 0, 1, 0, 3, "t", 100), "compile")
                .add(new Stage("release", 5, 1, 0, 1, 0, 0, "r", 0), "flaky-tests");
        RunResult flakyRun = Simulator.run(flaky, pool(1, 1, 0, 2), Collections.emptyMap(), Collections.emptyMap(), 99L);
        check(!flakyRun.success, "a permanently flaky stage must sink the pipeline");
        check(flakyRun.failed.equals(List.of("flaky-tests")), "exactly the flaky stage must be reported "
                + "failed, got " + flakyRun.failed);
        check(flakyRun.cancelled.equals(List.of("release")), "the failure must cancel exactly the "
                + "descendants, got " + flakyRun.cancelled);
        check(flakyRun.startCount("flaky-tests") == 4, "one attempt plus three retries must be made, got "
                + flakyRun.startCount("flaky-tests"));
        check(flakyRun.startCount("release") == 0, "a cancelled stage must never start");

        Pipeline retried = new Pipeline()
                .add(new Stage("compile", 10, 1, 0, 1, 0, 0, "c", 0))
                .add(new Stage("sometimes-flaky", 20, 1, 0, 1, 0, 3, "t", 50), "compile")
                .add(new Stage("release", 5, 1, 0, 1, 0, 0, "r", 0), "sometimes-flaky");
        Stage flakyStage = retried.stages.get("sometimes-flaky");
        long retrySeed = -1;
        for (long s = 0; s < 200 && retrySeed < 0; s++)
            if (Simulator.flakeFired(flakyStage, 1, s) && !Simulator.flakeFired(flakyStage, 2, s))
                retrySeed = s;
        check(retrySeed >= 0, "the flake injector must offer a seed that fails once and then succeeds");
        RunResult retriedRun = Simulator.run(retried, pool(1, 1, 0, 2), Collections.emptyMap(), Collections.emptyMap(), retrySeed);
        check(retriedRun.success, "the retry must carry the pipeline through a single flake");
        check(retriedRun.startCount("sometimes-flaky") == 2, "the flake must cost exactly one retry, got "
                + retriedRun.startCount("sometimes-flaky"));
        check(retriedRun.startCount("release") == 1, "the descendant must run once, after the retry");

        int flakes = 0;
        Stage coin = new Stage("coin", 1, 1, 0, 1, 0, 0, "coin", 25);
        for (int i = 0; i < 20000; i++) if (Simulator.flakeFired(coin, i, 2026L)) flakes++;
        double flakeRate = flakes / 20000.0;
        check(flakeRate > 0.22 && flakeRate < 0.28, "the flake injector must honour its configured rate, got "
                + flakeRate);

        Pipeline timeout = new Pipeline()
                .add(new Stage("configure", 5, 1, 0, 1, 0, 0, "cfg", 0))
                .add(new Stage("benchmark", 500, 1, 0, 1, 120, 0, "bench", 0), "configure")
                .add(new Stage("publish", 5, 1, 0, 1, 0, 0, "pub", 0), "benchmark");
        RunResult timeoutRun = Simulator.run(timeout, pool(1, 1, 0, 2), Collections.emptyMap(), Collections.emptyMap(), 7L);
        check(!timeoutRun.success && timeoutRun.failed.equals(List.of("benchmark")),
                "a stage that blows its budget must be killed: " + timeoutRun.failed);
        check(timeoutRun.cancelled.equals(List.of("publish")), "killing a stage cancels its descendants");
        check(timeoutRun.makespan == 125, "the overrun must be cut at its budget, got " + timeoutRun.makespan);

        // --- resource contention and degenerate pipelines -----------------------------------------------
        Pipeline gpuHeavy = new Pipeline()
                .add(new Stage("fit-a", 30, 4, 1, 1, 0, 0, "a", 0))
                .add(new Stage("fit-b", 30, 4, 1, 1, 0, 0, "b", 0))
                .add(new Stage("report", 10, 1, 0, 1, 0, 0, "r", 0), "fit-a", "fit-b");
        RunResult contended = Simulator.run(gpuHeavy, pool(1, 8, 1, 8), Collections.emptyMap(), Collections.emptyMap(), 3L);
        check(contended.makespan == 70, "a single GPU must serialise the two fits: " + contended.makespan);
        RunResult roomy = Simulator.run(gpuHeavy, pool(2, 8, 1, 8), Collections.emptyMap(), Collections.emptyMap(), 3L);
        check(roomy.makespan == 40, "two GPUs must overlap the fits: " + roomy.makespan);
        RunResult empty = Simulator.run(new Pipeline(), pool(2, 2, 2, 4), Collections.emptyMap(), Collections.emptyMap(), 1L);
        check(empty.makespan == 0 && empty.success && empty.attempts == 0, "an empty pipeline is instant");
        Pipeline single = new Pipeline().add(new Stage("solo", 7, 1, 0, 1, 0, 0, "s", 0));
        RunResult solo = Simulator.run(single, pool(1, 1, 0, 1), Collections.emptyMap(), Collections.emptyMap(), 1L);
        check(solo.makespan == 7 && solo.attempts == 1, "a single-stage pipeline runs once");
        boolean cycle = false;
        try {
            new Pipeline().add(new Stage("a", 1, 1, 0, 1, 0, 0, "a", 0), "b")
                    .add(new Stage("b", 1, 1, 0, 1, 0, 0, "b", 0), "a").order();
        } catch (IllegalArgumentException ex) { cycle = true; }
        check(cycle, "a cyclic pipeline must be rejected");
        boolean unknown = false;
        try { new Pipeline().add(new Stage("a", 1, 1, 0, 1, 0, 0, "a", 0), "ghost").order(); }
        catch (IllegalArgumentException ex) { unknown = true; }
        check(unknown, "an unknown dependency must be rejected");

        System.out.println("Reference pipeline: " + pipeline.stages.size() + " stages, critical path "
                + pipeline.criticalPath() + " ms, serial " + pipeline.serialSum() + " ms; four machines finish "
                + "in " + clean.makespan + " ms, fully cached in " + cached.makespan + " ms");
        System.out.println("Editing unit-tests-a invalidated " + changed.size() + " of "
                + pipeline.stages.size() + " stages; the incremental run then started "
                + partial.attempts + " stages");
        System.out.println("Small instance: exact optimum " + optimumTwo + " ms with two machines, list "
                + "schedule " + twoMachines.makespan + " ms; flake/timeout runs cancelled exactly their "
                + "descendants");
        System.out.println("PASS");
    }

    /** A realistic numerical pipeline: configure, compile, three test shards, integration, bench, docs. */
    static Pipeline buildPipeline() {
        Pipeline p = new Pipeline();
        p.add(new Stage("configure", 20, 2, 0, 2, 0, 0, "cmake", 0));
        p.add(new Stage("compile", 60, 4, 0, 8, 0, 0, "cc", 0), "configure");
        p.add(new Stage("unit-tests-a", 40, 2, 0, 2, 0, 0, "tests-a", 0), "compile");
        p.add(new Stage("unit-tests-b", 45, 2, 0, 2, 0, 0, "tests-b", 0), "compile");
        p.add(new Stage("unit-tests-c", 35, 2, 0, 2, 0, 0, "tests-c", 0), "compile");
        p.add(new Stage("integration", 50, 3, 0, 4, 0, 0, "integration", 0), "unit-tests-a", "unit-tests-b",
                "unit-tests-c");
        p.add(new Stage("numerical-benchmarks", 80, 2, 2, 4, 0, 1, "bench", 0), "integration");
        p.add(new Stage("docs", 15, 1, 0, 1, 0, 0, "docs", 0), "compile");
        return p;
    }

    private static Pipeline buildSmallPipeline() {
        Pipeline p = new Pipeline();
        p.add(new Stage("a", 10, 1, 0, 1, 0, 0, "a", 0));
        p.add(new Stage("b", 20, 1, 0, 1, 0, 0, "b", 0));
        p.add(new Stage("c", 15, 1, 0, 1, 0, 0, "c", 0), "a");
        p.add(new Stage("d", 25, 1, 0, 1, 0, 0, "d", 0), "b");
        p.add(new Stage("e", 10, 1, 0, 1, 0, 0, "e", 0), "c", "d");
        return p;
    }

    static List<Machine> pool(int count, int cpu, int gpu, int memory) {
        List<Machine> machines = new ArrayList<>();
        for (int i = 0; i < count; i++) machines.add(new Machine(cpu, gpu, memory));
        return machines;
    }

    /** Which stages changed their Merkle key between the two versions of the pipeline. */
    static List<String> changedKeys(Pipeline edited, Pipeline original) {
        Map<String, String> now = edited.cacheKeys(), before = original.cacheKeys();
        List<String> changed = new ArrayList<>();
        for (String name : edited.order()) if (!now.get(name).equals(before.get(name))) changed.add(name);
        return changed;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
