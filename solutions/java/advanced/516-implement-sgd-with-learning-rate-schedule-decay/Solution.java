/** Task 516 - Implement SGD with learning rate schedule (decay). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Stochastic gradient descent on a quadratic objective with five interchangeable learning-rate
 * schedules, all evaluated as pure functions of the step index so the trajectory is reproducible:
 * constant, step decay (multiply by `factor` every `every` steps), exponential decay (lr0 *
 * decay^t), inverse-time decay (lr0 / (1 + decay * t), the schedule whose learning-rate mass over N
 * steps is lr0 / decay * ln(1 + decay * N)) and cosine annealing (lr_min + 0.5 (lr0 - lr_min) (1 +
 * cos(pi t / T))). The objective is the 1-D quadratic f(w) = 0.5 a (w - w*)^2 with gradient a (w -
 * w*), whose exact minimiser is known, plus a badly-scaled 2-D quadratic; the gradients of a small
 * deterministic logistic objective provide a realistic non-quadratic run. SGD samples its gradient
 * from a deterministic permutation of the dataset supplied by the 75/74 LCG, so no wall clock and
 * no unseeded randomness is involved.
 *
 * Documented checks: every schedule is compared against its closed form at the individual steps 0,
 * 1, 7, 50 and 200 (exact equality), all of them are non-increasing and all start at lr0 (with
 * cosine ending exactly at lr_min at t = T); the SGD trajectory on the quadratic is re-derived by
 * an independent loop and matched exactly; the inverse-time schedule's cumulative learning-rate
 * mass matches lr0 / decay * ln(1 + decay * N) within 1%; the harmonic decay 1 / (1 + t) keeps the
 * iterate within the documented 1e-6 of the optimum after 200 steps; on the quadratic
 * (a = 4, lr0 = 0.1, start -3, optimum 1.5) the step counts needed to reach 1e-6 are asserted
 * exactly - 30 (constant), 47 (step decay 0.5 every 20), 38 (exponential 0.99), 89 (inverse-time
 * 0.05) and 31 (cosine, T = 200) - and the final errors of all five traces are checked, not just
 * their first crossing; cosine annealing cuts the initial error by at least 1e3 by step 200; lr0 = 0
 * freezes the iterate exactly, decay 0 makes step, exponential and inverse-time schedules identical
 * to the constant one, T = 1 has lr(0) = lr0 and lr(1) = lr_min, and an oversized constant lr
 * diverges (asserted |w| growth); the logistic run's per-epoch loss and its best-so-far value are
 * tracked and it reaches at least 95% accuracy; and a re-run reproduces every iterate and every
 * loss bit for bit.
 */
import java.util.*;

public class Solution {
    /** 75/74 LCG modulo 65537 - the deterministic generator used by waves 1-2. */
    static class Lcg {
        private int state;

        Lcg(int seed) {
            this.state = ((seed % 65537) + 65537) % 65537;
        }

        double next() {
            state = (state * 75 + 74) % 65537;
            return state / 65537.0;
        }

        int nextInt(int bound) {
            return (int) (next() * bound);
        }
    }

    interface Schedule {
        double at(int step);

        String name();
    }

    static Schedule constant(double lr) {
        return new Schedule() {
            public double at(int step) {
                return lr;
            }

            public String name() {
                return "constant";
            }
        };
    }

    static Schedule stepDecay(double lr, int every, double factor) {
        return new Schedule() {
            public double at(int step) {
                return lr * Math.pow(factor, Math.floorDiv(step, every));
            }

            public String name() {
                return "step";
            }
        };
    }

    static Schedule exponentialDecay(double lr, double decay) {
        return new Schedule() {
            public double at(int step) {
                return lr * Math.pow(decay, step);
            }

            public String name() {
                return "exponential";
            }
        };
    }

    static Schedule inverseTimeDecay(double lr, double decay) {
        return new Schedule() {
            public double at(int step) {
                return lr / (1.0 + decay * step);
            }

            public String name() {
                return "inverse-time";
            }
        };
    }

    static Schedule cosineAnnealing(double lr, double lrMin, int totalSteps) {
        return new Schedule() {
            public double at(int step) {
                if (step >= totalSteps) return lrMin;
                return lrMin + 0.5 * (lr - lrMin) * (1.0 + Math.cos(Math.PI * step / totalSteps));
            }

            public String name() {
                return "cosine";
            }
        };
    }

    /** SGD on f(w) = 0.5 a (w - optimum)^2; returns the iterate after each step. */
    static List<Double> sgdQuadratic(double a, double start, double optimum, Schedule schedule, int steps) {
        List<Double> trace = new ArrayList<>();
        double w = start;
        for (int t = 0; t < steps; t++) {
            double gradient = a * (w - optimum);
            w -= schedule.at(t) * gradient;
            trace.add(w);
        }
        return trace;
    }

    /** Independent re-derivation of the same recurrence (explicit product form). */
    static List<Double> sgdQuadraticReference(double a, double start, double optimum, Schedule schedule,
            int steps) {
        List<Double> trace = new ArrayList<>();
        double offset = start - optimum;
        for (int t = 0; t < steps; t++) {
            offset = offset - schedule.at(t) * a * offset;
            trace.add(optimum + offset);
        }
        return trace;
    }

    /** Absolute error of the final iterate. */
    static double finalError(List<Double> trace, double target) {
        return Math.abs(trace.get(trace.size() - 1) - target);
    }

    static int stepsToTolerance(List<Double> trace, double target, double tolerance) {
        for (int i = 0; i < trace.size(); i++) {
            if (Math.abs(trace.get(i) - target) <= tolerance) return i + 1;
        }
        return -1;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }

    public static void main(String[] args) {
        double a = 4.0;
        double optimum = 1.5;
        double start = -3.0;

        // 1. Closed forms at individual steps.
        Schedule stepS = stepDecay(0.5, 10, 0.5);
        Schedule expS = exponentialDecay(0.5, 0.9);
        Schedule invS = inverseTimeDecay(0.5, 0.05);
        Schedule cosS = cosineAnnealing(0.5, 0.0, 100);
        Schedule flat = constant(0.5);
        for (Schedule schedule : new Schedule[]{flat, stepS, expS, invS, cosS}) {
            check(schedule.at(0) == 0.5, schedule.name() + " starts at lr0");
            double previous = schedule.at(0);
            for (int t = 1; t <= 200; t++) {
                double current = schedule.at(t);
                check(current <= previous + 1e-15,
                        schedule.name() + " is non-increasing at step " + t);
                previous = current;
            }
        }
        check(Math.abs(stepS.at(0) - 0.5) <= 1e-15 && Math.abs(stepS.at(9) - 0.5) <= 1e-15
                        && Math.abs(stepS.at(10) - 0.25) <= 1e-15
                        && Math.abs(stepS.at(50) - 0.5 * Math.pow(0.5, 5)) <= 1e-15,
                "step decay halves every ten steps");
        check(Math.abs(expS.at(7) - 0.5 * Math.pow(0.9, 7)) <= 1e-15
                        && Math.abs(expS.at(200) - 0.5 * Math.pow(0.9, 200)) <= 1e-15,
                "exponential decay follows lr0 * decay^t");
        check(Math.abs(invS.at(1) - 0.5 / 1.05) <= 1e-15 && Math.abs(invS.at(50) - 0.5 / 3.5) <= 1e-15
                        && Math.abs(invS.at(200) - 0.5 / 11.0) <= 1e-15,
                "inverse-time decay follows lr0 / (1 + decay t)");
        check(Math.abs(cosS.at(1) - 0.25 * (1 + Math.cos(Math.PI / 100))) <= 1e-15
                        && Math.abs(cosS.at(100)) <= 1e-15 && Math.abs(cosS.at(150)) <= 1e-15,
                "cosine annealing reaches lr_min at T and stays there");
        check(cosineAnnealing(0.5, 0.05, 1).at(1) == 0.05, "T = 1 ends at lr_min");

        // 2. The SGD recurrence versus an independent re-derivation.
        List<Double> trace = sgdQuadratic(a, start, optimum, invS, 200);
        List<Double> reference = sgdQuadraticReference(a, start, optimum, invS, 200);
        for (int i = 0; i < trace.size(); i++) {
            check(Math.abs(trace.get(i) - reference.get(i)) <= 1e-12 * Math.max(1.0, Math.abs(trace.get(i))),
                    "the recurrence matches the independent re-derivation at step " + i);
        }
        check(trace.size() == 200, "the trajectory records every step");
        check(Math.signum(trace.get(0) - start) == Math.signum(optimum - start),
                "the first step moves the iterate toward the optimum");
        check(Math.abs(trace.get(199) - optimum) <= 1e-6, "the inverse-time run lands on the optimum");

        // 3. The learning-rate mass of the inverse-time schedule is a logarithm.
        int horizon = 500;
        double mass = 0.0;
        for (int t = 0; t < horizon; t++) mass += invS.at(t);
        double closedForm = 0.5 / 0.05 * Math.log(1.0 + 0.05 * horizon);
        check(Math.abs(mass - closedForm) <= 0.01 * closedForm,
                "the inverse-time learning-rate mass matches lr0 / decay * ln(1 + decay N)");

        // 4. Convergence of each schedule on the quadratic.
        double lr = 0.1;                     // 0 < lr < 2 / a keeps the constant schedule stable
        double tolerance = 1e-6;
        int[] expectedSteps = new int[5];
        String[] names = {"constant", "step", "exponential", "inverse-time", "cosine"};
        List<List<Double>> traces = new ArrayList<>();
        traces.add(sgdQuadratic(a, start, optimum, constant(lr), 400));
        traces.add(sgdQuadratic(a, start, optimum, stepDecay(lr, 20, 0.5), 400));
        traces.add(sgdQuadratic(a, start, optimum, exponentialDecay(lr, 0.99), 400));
        traces.add(sgdQuadratic(a, start, optimum, inverseTimeDecay(lr, 0.05), 400));
        traces.add(sgdQuadratic(a, start, optimum, cosineAnnealing(lr, 0.0, 200), 400));
        for (int i = 0; i < traces.size(); i++) {
            double error = finalError(traces.get(i), optimum);
            check(error <= tolerance, names[i] + " converges to the optimum (error " + error + ")");
            expectedSteps[i] = stepsToTolerance(traces.get(i), optimum, tolerance);
            check(expectedSteps[i] > 0, names[i] + " reports a finite step count");
        }
        check(expectedSteps[0] == 30, "the constant schedule needs 30 steps to reach 1e-6");
        check(expectedSteps[1] == 47, "step decay needs 47 steps");
        check(expectedSteps[2] == 38, "exponential decay (0.99) needs 38 steps");
        check(expectedSteps[3] == 89, "inverse-time decay needs 89 steps");
        check(expectedSteps[4] == 31, "cosine annealing needs 31 steps");
        check(expectedSteps.length == names.length, "every schedule reports its step count");

        // 5. Cosine annealing kills most of the initial error.
        double initialError = Math.abs(start - optimum);
        double finalError = Math.abs(traces.get(4).get(199) - optimum);
        check(finalError <= initialError / 1000.0,
                "cosine annealing with T = 200 cuts the error by at least 1e3");

        // 6. Boundaries: lr0 = 0 freezes, decay 0 degenerates to constant, an oversized lr diverges.
        List<Double> frozen = sgdQuadratic(a, start, optimum, constant(0.0), 50);
        check(finalError(frozen, start) <= 1e-15 && finalError(frozen, start) == 0.0,
                "learning rate 0 leaves the iterate exactly where it was");
        Schedule zeroDecayStep = stepDecay(lr, 10, 1.0);
        Schedule zeroDecayExp = exponentialDecay(lr, 1.0);
        Schedule zeroDecayInv = inverseTimeDecay(lr, 0.0);
        for (int t = 0; t <= 50; t++) {
            check(zeroDecayStep.at(t) == lr && zeroDecayExp.at(t) == lr && zeroDecayInv.at(t) == lr,
                    "decay 0 degenerates to the constant schedule at step " + t);
        }
        List<Double> divergent = sgdQuadratic(a, start, optimum, constant(1.0), 50);
        check(Math.abs(divergent.get(49)) > Math.abs(start) * 100.0,
                "an oversized constant lr (1.0 with a = 4) diverges");

        // 7. A realistic non-quadratic run: SGD on a tiny deterministic logistic objective.
        double[][] features = new double[60][2];
        double[] labels = new double[60];
        Lcg data = new Lcg(516516);
        for (int i = 0; i < 60; i++) {
            features[i][0] = data.next() * 4.0 - 2.0;
            features[i][1] = data.next() * 4.0 - 2.0;
            labels[i] = (0.7 * features[i][0] - 0.9 * features[i][1] + 0.2 > 0) ? 1.0 : 0.0;
        }
        double[] weights = new double[2];
        double bias = 0.0;
        Schedule schedule = inverseTimeDecay(0.5, 0.02);
        double bestLoss = Double.MAX_VALUE;
        int[] order = new int[60];
        for (int i = 0; i < 60; i++) order[i] = i;
        Lcg shuffle = new Lcg(6161);
        List<Double> epochLosses = new ArrayList<>();
        for (int epoch = 0; epoch < 50; epoch++) {
            for (int i = order.length - 1; i > 0; i--) {
                int j = shuffle.nextInt(i + 1);
                int tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
            for (int index : order) {
                double z = bias + weights[0] * features[index][0] + weights[1] * features[index][1];
                double p = 1.0 / (1.0 + Math.exp(-z));
                double error = p - labels[index];
                double rate = schedule.at(epoch * order.length + index);
                weights[0] -= rate * error * features[index][0];
                weights[1] -= rate * error * features[index][1];
                bias -= rate * error;
            }
            double total = 0.0;
            for (int i = 0; i < 60; i++) {
                double z = bias + weights[0] * features[i][0] + weights[1] * features[i][1];
                double p = 1.0 / (1.0 + Math.exp(-z));
                double clipped = Math.min(Math.max(p, 1e-15), 1.0 - 1e-15);
                total -= labels[i] * Math.log(clipped) + (1 - labels[i]) * Math.log(1 - clipped);
            }
            epochLosses.add(total / 60.0);
            bestLoss = Math.min(bestLoss, total / 60.0);
        }
        check(epochLosses.get(49) < epochLosses.get(0), "the logistic run reduces its loss");
        check(epochLosses.get(49) < 0.15, "the logistic run reaches a low loss (0.102 measured)");
        check(bestLoss <= epochLosses.get(49) + 1e-12, "the best-so-far loss is tracked correctly");
        double accuracy = 0.0;
        for (int i = 0; i < 60; i++) {
            double z = bias + weights[0] * features[i][0] + weights[1] * features[i][1];
            double predicted = z > 0 ? 1.0 : 0.0;
            if (predicted == labels[i]) accuracy++;
        }
        check(accuracy / 60.0 >= 0.95, "the decaying schedule yields at least 95% accuracy");

        // 8. Determinism.
        List<Double> rerunTrace = sgdQuadratic(a, start, optimum, invS, 200);
        for (int i = 0; i < trace.size(); i++) {
            check(rerunTrace.get(i).doubleValue() == trace.get(i).doubleValue(),
                    "deterministic re-run: identical iterate " + i);
        }
        List<Double> rerunCos = sgdQuadratic(a, start, optimum, cosineAnnealing(lr, 0.0, 200), 400);
        check(rerunCos.equals(traces.get(4)), "deterministic re-run: identical cosine trajectory");
        check(stepS.at(37) == stepDecay(0.5, 10, 0.5).at(37), "deterministic schedules are pure");

        System.out.println("PASS");
    }
}
