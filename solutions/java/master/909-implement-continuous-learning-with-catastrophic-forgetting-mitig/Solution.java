/** Task 909 - Implement continuous learning with catastrophic forgetting mitigation (replay buffers).
 *  (Tier: Master, CTCI Beyond - Machine Learning & AI)
 *
 * A single-head linear model y = w.x + b is trained on three tasks in sequence (SGD, deterministic batch
 * sampling). Because the tasks want different weights, plain sequential fine-tuning overwrites the earlier
 * tasks - the textbook case of catastrophic forgetting. The mitigation here is a *bounded replay buffer*:
 * after every task a fixed quota of its examples is retained, and each training batch is mixed with buffer
 * examples drawn from earlier tasks, so the gradient never becomes purely local to the current task.
 *
 * The checks measure forgetting instead of asserting the fix: per-task error immediately after a task, the
 * same error after the whole sequence, the reduction the buffer buys, the strict memory bound (the buffer is
 * never allowed to exceed its capacity, and capacity zero must degrade to plain fine-tuning), and a
 * closed-form joint least-squares fit over all three tasks, computed from the normal equations, as the
 * oracle that both strategies are scored against. An EWC-style penalty is also implemented and checked on its
 * defining property: zero at the anchor, quadratic in the distance from it.
 *
 * Simplification: three linear tasks with a 3-dimensional feature vector, 100 examples each, 60 epochs per
 * task. Invariants verified: bounded buffer, replay reduces forgetting and beats fine-tuning on the average
 * and against the joint optimum, an idle buffer is a no-op, and the run replays deterministically.
 */
import java.util.*;

/** One labelled example, tagged with the task it came from. */
final class Example {
    final double[] x;
    final double y;
    final int task;

    Example(double[] x, double y, int task) {
        this.x = x;
        this.y = y;
        this.task = task;
    }
}

/** A deterministic linear model. */
final class LinearModel {
    final double[] w = new double[3];
    double b;

    LinearModel() { }

    double predict(double[] x) {
        return w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + b;
    }
}

/** Bounded replay buffer with a per-task quota. */
final class ReplayBuffer {
    final int capacity;
    final int quotaPerTask;
    private final List<Example> store = new ArrayList<>();
    private final Map<Integer, Integer> perTask = new TreeMap<>();

    ReplayBuffer(int capacity) {
        this.capacity = capacity;
        this.quotaPerTask = capacity <= 0 ? 0 : Math.max(1, capacity / 3);
    }

    int size() { return store.size(); }

    /** Keeps at most quotaPerTask examples of this task; refuses anything that would exceed capacity. */
    boolean offer(Example example) {
        if (capacity <= 0) return false;
        int held = perTask.getOrDefault(example.task, 0);
        if (held >= quotaPerTask || store.size() >= capacity) return false;
        store.add(example);
        perTask.put(example.task, held + 1);
        return true;
    }

    Example get(int index) { return store.get(index); }

    /** A uniformly chosen retained example of one task; the caller guarantees the task is present. */
    Example sampleOfTask(int task, Rng909 rng) {
        List<Example> matching = new ArrayList<>();
        for (Example example : store) if (example.task == task) matching.add(example);
        return matching.get(rng.nextInt(matching.size()));
    }

    Set<Integer> tasks() { return new TreeSet<>(perTask.keySet()); }

    int countFor(int task) { return perTask.getOrDefault(task, 0); }
}

/** Per-task mean squared error of a model. */
final class Report {
    final Map<Integer, Double> mse = new TreeMap<>();

    double average() {
        double total = 0;
        for (double v : mse.values()) total += v;
        return total / mse.size();
    }
}

public class Solution {
    static final int TASKS = 3;
    static final int PER_TASK = 100;
    static final int EPOCHS = 60;
    static final int BATCH = 8;
    static final int CAPACITY = 60;
    static final double RATE = 0.15;
    static final double[][] WEIGHTS = {{1.0, 0.0, 0.0}, {-0.8, 0.0, 0.0}, {0.0, 0.6, 0.0}};
    static final double[] BIASES = {0.2, 0.5, -0.3};

    public static void main(String[] args) {
        List<List<Example>> tasks = generate(0x909ABCDEL);
        check(tasks.size() == TASKS && tasks.get(0).size() == PER_TASK, "the fixture must hold 3 tasks of "
                + PER_TASK);

        // ---- 1. sequential fine-tuning without replay ---------------------------------------------------
        Run naive = train(tasks, 0);
        double naiveForgetting = naive.finalReport.mse.get(0) - naive.afterTask.get(0);
        check(naiveForgetting > 0.1, "plain fine-tuning must forget task 0 badly, forgot " + naiveForgetting);
        check(naive.finalReport.mse.get(TASKS - 1) < 0.05,
                "the last task must be well fitted, mse " + naive.finalReport.mse.get(TASKS - 1));

        // ---- 2. the same run with a bounded replay buffer ------------------------------------------------
        Run replay = train(tasks, CAPACITY);
        double replayForgetting = replay.finalReport.mse.get(0) - replay.afterTask.get(0);
        check(replayForgetting < 0.75 * naiveForgetting, "replay must cut forgetting: " + replayForgetting
                + " vs " + naiveForgetting);
        check(replay.finalReport.average() < naive.finalReport.average(),
                "replay must lower the average error: " + replay.finalReport.average() + " vs "
                        + naive.finalReport.average());

        // ---- 3. buffer invariants: the memory bound is never exceeded -----------------------------------
        check(replay.peakBufferSize <= CAPACITY, "the buffer exceeded its capacity: " + replay.peakBufferSize);
        check(replay.buffer.tasks().size() == TASKS, "the buffer must span every task, got "
                + replay.buffer.tasks());
        check(replay.buffer.countFor(0) > 0, "task 0 must leave retained examples");
        check(replay.buffer.size() <= CAPACITY, "final buffer size must respect the bound: "
                + replay.buffer.size());
        int refused = 0;
        ReplayBuffer tiny = new ReplayBuffer(3);
        for (int i = 0; i < 40; i++) {
            if (!tiny.offer(new Example(new double[] {i, 0, 0}, i, i % 3))) refused++;
        }
        check(tiny.size() == 3 && refused == 37, "a capacity-3 buffer must accept exactly 3 examples, got "
                + tiny.size());
        check(new ReplayBuffer(0).offer(new Example(new double[] {1, 0, 0}, 1, 0)) == false,
                "a zero-capacity buffer must refuse everything");

        // ---- 4. the closed-form joint optimum as oracle -------------------------------------------------
        double[] oracle = jointLeastSquares(tasks);
        check(oracle.length == 4, "the oracle must solve for intercept plus three weights");
        double oracleAverage = 0;
        for (int t = 0; t < TASKS; t++) oracleAverage += mse(oracle, tasks.get(t));
        oracleAverage /= TASKS;
        check(Math.abs(oracle[0]) < 0.6 && Math.abs(oracle[1]) < 0.6,
                "the joint optimum must be a compromise between conflicting tasks: w0 = " + oracle[1]);
        check(replay.finalReport.average() < 1.5 * oracleAverage,
                "replay must land near the joint optimum: " + replay.finalReport.average() + " vs "
                        + oracleAverage);
        check(naive.finalReport.average() > 1.5 * oracleAverage,
                "plain fine-tuning must be far from the joint optimum: " + naive.finalReport.average() + " vs "
                        + oracleAverage);
        double oracleTaskZero = mse(oracle, tasks.get(0));
        check(replay.finalReport.mse.get(0) < naive.finalReport.mse.get(0),
                "replay must retain task 0 better than fine-tuning");
        check(oracleTaskZero < naive.finalReport.mse.get(0),
                "even the pooled optimum fits task 0 better than fine-tuning does: " + oracleTaskZero + " vs "
                        + naive.finalReport.mse.get(0));
        check(Math.abs(replay.finalReport.average() - oracleAverage) < 0.5 * oracleAverage,
                "the replay equilibrium must land near the pooled optimum: " + replay.finalReport.average()
                        + " vs " + oracleAverage);

        // ---- 5. boundary cases ---------------------------------------------------------------------------
        Run noBuffer = train(tasks, 0);
        check(Math.abs(noBuffer.finalReport.average() - naive.finalReport.average()) < 1e-12,
                "capacity 0 must reproduce plain fine-tuning exactly");
        check(noBuffer.buffer.size() == 0, "a zero-capacity run keeps nothing");
        Run single = train(tasks.subList(0, 1), CAPACITY);
        check(single.afterTask.size() == 1 && single.finalReport.average() < 0.05,
                "a single-task run cannot forget and must fit the task");
        check(replay.buffer.size() == CAPACITY, "the buffer must be saturated at capacity: "
                + replay.buffer.size());

        // ---- 6. the EWC penalty invariant ----------------------------------------------------------------
        double[] fisher = {1.0, 2.0, 0.5};
        double[] anchor = {0.5, -0.5, 1.0};
        check(ewcPenalty(anchor, anchor, fisher) == 0, "the penalty must vanish at the anchor");
        double[] step = {anchor[0] + 0.1, anchor[1], anchor[2]};
        double[] twice = {anchor[0] + 0.2, anchor[1], anchor[2]};
        check(ewcPenalty(step, anchor, fisher) > 0, "moving away from the anchor must cost something");
        check(Math.abs(ewcPenalty(twice, anchor, fisher) - 4 * ewcPenalty(step, anchor, fisher)) < 1e-12,
                "the penalty must be quadratic in the distance from the anchor");
        check(ewcPenalty(step, anchor, new double[] {2.0, 2.0, 0.5}) > ewcPenalty(step, anchor, fisher),
                "a larger Fisher weight must penalise the same move more");

        // ---- 7. determinism -------------------------------------------------------------------------------
        Run again = train(generate(0x909ABCDEL), CAPACITY);
        check(Math.abs(again.finalReport.average() - replay.finalReport.average()) < 1e-12,
                "training must be reproducible");
        check(Arrays.equals(jointLeastSquares(tasks), jointLeastSquares(tasks)),
                "the oracle solve must be reproducible");
        check(replay.buffer.size() == again.buffer.size(), "the buffer must fill identically");

        System.out.println("tasks " + TASKS + " x " + PER_TASK + " examples, buffer capacity " + CAPACITY
                + " (peak " + replay.peakBufferSize + ")");
        System.out.println("forgetting on task 0: fine-tuning " + round(naiveForgetting) + " -> replay "
                + round(replayForgetting) + "; average mse " + round(naive.finalReport.average()) + " -> "
                + round(replay.finalReport.average()) + " (joint optimum " + round(oracleAverage) + ")");
        System.out.println("per-task mse with replay " + round(replay.finalReport.mse.get(0)) + "/"
                + round(replay.finalReport.mse.get(1)) + "/" + round(replay.finalReport.mse.get(2))
                + " vs fine-tuning " + round(naive.finalReport.mse.get(0)) + "/"
                + round(naive.finalReport.mse.get(1)) + "/" + round(naive.finalReport.mse.get(2)));
        System.out.println("PASS");
    }

    /** Outcome of one training run over the task sequence. */
    static final class Run {
        final Report finalReport = new Report();
        final Map<Integer, Double> afterTask = new TreeMap<>();
        ReplayBuffer buffer;
        int peakBufferSize;

        Run(int capacity) { buffer = new ReplayBuffer(capacity); }
    }

    static List<List<Example>> generate(long seed) {
        Rng909 rng = new Rng909(seed);
        List<List<Example>> tasks = new ArrayList<>();
        for (int t = 0; t < TASKS; t++) {
            List<Example> examples = new ArrayList<>();
            for (int i = 0; i < PER_TASK; i++) {
                double[] x = {2 * rng.nextDouble() - 1, 2 * rng.nextDouble() - 1, 2 * rng.nextDouble() - 1};
                double y = WEIGHTS[t][0] * x[0] + WEIGHTS[t][1] * x[1] + WEIGHTS[t][2] * x[2] + BIASES[t];
                examples.add(new Example(x, y, t));
            }
            tasks.add(examples);
        }
        return tasks;
    }

    /** Sequential training; a capacity of zero means no replay at all. */
    static Run train(List<List<Example>> tasks, int capacity) {
        Run run = new Run(capacity);
        LinearModel model = new LinearModel();
        Rng909 rng = new Rng909(0x9095EEDL + capacity);
        for (int t = 0; t < tasks.size(); t++) {
            List<Example> data = tasks.get(t);
            for (int epoch = 0; epoch < EPOCHS; epoch++) {
                int perTask = Math.max(1, BATCH / 2);
                for (int batch = 0; batch < data.size() / perTask; batch++) {
                    // Task-balanced replay: every task present in the batch contributes the same number of
                    // examples, so the gradient never becomes purely local to the newest task.
                    List<Example> mixed = new ArrayList<>();
                    for (int i = 0; i < perTask; i++) {
                        mixed.add(data.get((batch * perTask + i) % data.size()));
                    }
                    for (int other = 0; other < tasks.size(); other++) {
                        if (run.buffer.countFor(other) == 0) continue;
                        for (int i = 0; i < perTask; i++) mixed.add(run.buffer.sampleOfTask(other, rng));
                    }
                    step(model, mixed);
                }
            }
            run.afterTask.put(t, mseOf(model, data));
            for (Example example : data) {
                run.buffer.offer(example);
                run.peakBufferSize = Math.max(run.peakBufferSize, run.buffer.size());
            }
        }
        for (int t = 0; t < tasks.size(); t++) run.finalReport.mse.put(t, mseOf(model, tasks.get(t)));
        return run;
    }

    /** One gradient step on the mean squared error of the given batch. */
    static void step(LinearModel model, List<Example> batch) {
        double[] gradient = new double[3];
        double gradientB = 0;
        for (Example example : batch) {
            double error = model.predict(example.x) - example.y;
            for (int d = 0; d < 3; d++) gradient[d] += error * example.x[d];
            gradientB += error;
        }
        for (int d = 0; d < 3; d++) model.w[d] -= RATE * gradient[d] / batch.size();
        model.b -= RATE * gradientB / batch.size();
    }

    static double mseOf(LinearModel model, List<Example> data) {
        double total = 0;
        for (Example example : data) total += sq(model.predict(example.x) - example.y);
        return total / data.size();
    }

    static double mse(double[] params, List<Example> data) {
        double total = 0;
        for (Example example : data) {
            double prediction = params[0] + params[1] * example.x[0] + params[2] * example.x[1]
                    + params[3] * example.x[2];
            total += sq(prediction - example.y);
        }
        return total / data.size();
    }

    /** Closed-form least squares over the pooled tasks: solve (Phi^T Phi) theta = Phi^T y by elimination. */
    static double[] jointLeastSquares(List<List<Example>> tasks) {
        double[][] a = new double[4][5];
        for (List<Example> data : tasks) {
            for (Example example : data) {
                double[] phi = {1, example.x[0], example.x[1], example.x[2]};
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) a[i][j] += phi[i] * phi[j];
                    a[i][4] += phi[i] * example.y;
                }
            }
        }
        return solve(a);
    }

    /** Gaussian elimination with partial pivoting on a 4x5 augmented matrix. */
    static double[] solve(double[][] augmented) {
        int n = 4;
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivot][col])) pivot = row;
            }
            check(Math.abs(augmented[pivot][col]) > 1e-12, "the normal equations must be non-singular");
            double[] swap = augmented[col];
            augmented[col] = augmented[pivot];
            augmented[pivot] = swap;
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = augmented[row][col] / augmented[col][col];
                for (int k = col; k <= n; k++) augmented[row][k] -= factor * augmented[col][k];
            }
        }
        double[] solution = new double[n];
        for (int i = 0; i < n; i++) solution[i] = augmented[i][n] / augmented[i][i];
        return solution;
    }

    /** Diagonal-Fisher EWC penalty: sum_i F_i (theta_i - anchor_i)^2 / 2. */
    static double ewcPenalty(double[] theta, double[] anchor, double[] fisher) {
        double total = 0;
        for (int i = 0; i < theta.length; i++) {
            total += fisher[i] * sq(theta[i] - anchor[i]) / 2;
        }
        return total;
    }

    static double sq(double value) { return value * value; }

    static double round(double value) { return Math.round(value * 10000) / 10000.0; }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}

/** splitmix64 plus an integer helper. */
final class Rng909 {
    private long state;

    Rng909(long seed) { this.state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), (long) bound); }
}
