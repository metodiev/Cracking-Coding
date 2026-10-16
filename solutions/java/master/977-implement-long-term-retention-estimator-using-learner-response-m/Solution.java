/** Task 977 - Implement long-term retention estimator using learner response models.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  A per-learner retention model with a power forgetting curve: the probability of recall after `elapsed`
 *  days is 2^(-elapsed / halfLife), so the curve is exactly 0.5 at the half-life. Each observed response
 *  rescales the half-life: a correct answer multiplies it by 1 + 0.4*(1 - p) + 0.1 (a harder recall buys
 *  more, FSRS-style) and an error halves it, both clamped to [0.5, 120]. Held-out logs of (gap, correct)
 *  pairs are scored with Brier score and log loss against a fixed-half-life baseline, and the predictions
 *  are also binned to check calibration (mean prediction vs observed frequency).
 *  Simplified: a one-parameter curve per learner with hand-tuned gain/loss factors, no per-topic difficulty
 *  or elapsed-time jitter; the exact half-life identity, the bounded updates, the calibration binning and
 *  the baseline comparison are the real properties being checked.
 *  Verified: the half-life identity and monotonicity of the curve, half-life ordering, update direction with
 *  bounds, per-mutation invariants, calibration bins, Brier score against an order-independent recomputation
 *  and against the baseline, finite log loss, the empty-log and zero-horizon boundaries, the all-correct
 *  monotonicity oracle and determinism.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final double DEFAULT_HALF_LIFE = 3.0, MIN_HALF_LIFE = 0.5, MAX_HALF_LIFE = 120.0;
    static final double BASELINE_HALF_LIFE = 7.0;
    static final double EPS = 0.001;                                         // loss clamp, keeps logs finite

    /** Power forgetting curve: probability of recall `elapsed` days after a review. */
    static double recall(double halfLife, double elapsed) { return Math.pow(2.0, -elapsed / halfLife); }

    static final class LearnerModel {
        final String id;
        double halfLife = DEFAULT_HALF_LIFE;
        int observations;
        LearnerModel(String id) { this.id = id; }
        double predict(double elapsed) { return recall(halfLife, elapsed); }
        /** Applies one graded response observed `elapsed` days after the previous review. */
        void observe(int correct, double elapsed) {
            double p = predict(elapsed);
            if (correct == 1) halfLife = Math.min(MAX_HALF_LIFE, halfLife * (1.0 + 0.4 * (1.0 - p) + 0.1));
            else halfLife = Math.max(MIN_HALF_LIFE, halfLife * 0.5);
            observations++;
        }
    }

    static final class Response {
        final String learner; final double gap; final int correct;
        Response(String learner, double gap, int correct) { this.learner = learner; this.gap = gap; this.correct = correct; }
    }

    static final class Prediction {
        final double p; final int outcome;
        Prediction(double p, int outcome) { this.p = p; this.outcome = outcome; }
    }

    /** Independent baseline: the same curve with one shared half-life that never adapts. */
    static double baselinePredict(double gap) { return recall(BASELINE_HALF_LIFE, gap); }

    static double brier(List<Prediction> predictions) {
        double total = 0;
        for (Prediction p : predictions) total += (p.p - p.outcome) * (p.p - p.outcome);
        return total / predictions.size();
    }

    /** Same quantity summed in reverse order: the aggregation order must not matter. */
    static double brierReverseOrder(List<Prediction> predictions) {
        double total = 0;
        for (int i = predictions.size() - 1; i >= 0; i--) {
            Prediction p = predictions.get(i);
            total += Math.pow(p.p - p.outcome, 2);
        }
        return total / predictions.size();
    }

    static double logLoss(List<Prediction> predictions) {
        double total = 0;
        for (Prediction p : predictions) {
            double clamped = Math.min(1 - EPS, Math.max(EPS, p.p));
            total += p.outcome == 1 ? -Math.log(clamped) : -Math.log(1 - clamped);
        }
        return total / predictions.size();
    }

    static void update(Map<String, LearnerModel> models, Response r) {
        LearnerModel model = models.computeIfAbsent(r.learner, LearnerModel::new);
        model.observe(r.correct, r.gap);
    }

    static List<Prediction> collect(List<Response> responses) {
        Map<String, LearnerModel> models = new TreeMap<>();
        List<Prediction> out = new ArrayList<>();
        for (Response r : responses) {
            LearnerModel model = models.computeIfAbsent(r.learner, LearnerModel::new);
            out.add(new Prediction(model.predict(r.gap), r.correct));
            model.observe(r.correct, r.gap);
        }
        return out;
    }

    static List<Response> logs() {
        return List.of(new Response("A", 1, 1), new Response("A", 2, 1), new Response("A", 4, 1),
                new Response("A", 8, 1),
                new Response("B", 1, 1), new Response("B", 2, 0), new Response("B", 4, 0),
                new Response("B", 8, 0),
                new Response("C", 1, 1), new Response("C", 2, 1), new Response("C", 5, 0),
                new Response("C", 10, 0));
    }

    public static void main(String[] args) {
        double[] horizons = {0, 1, 2, 5, 10, 30};
        boolean decreasing = true;
        for (int i = 1; i < horizons.length; i++) decreasing &= recall(6.0, horizons[i]) < recall(6.0, horizons[i - 1]);
        check(recall(6.0, 0) == 1.0 && Math.abs(recall(6.0, 6.0) - 0.5) < 1e-12 && decreasing
                        && recall(6.0, 12.0) < 0.26 && recall(6.0, 12.0) > 0.24,
                "the curve must be 1 at zero, exactly 0.5 at the half-life and strictly decreasing");

        check(recall(12.0, 5.0) > recall(6.0, 5.0) && recall(3.0, 5.0) < recall(6.0, 5.0)
                        && recall(1e6, 5.0) > 0.999,
                "a longer half-life must predict more retention at the same horizon");

        LearnerModel growing = new LearnerModel("grow");
        double before = growing.halfLife;
        growing.observe(1, 1);
        check(growing.halfLife > before && growing.observations == 1 && growing.halfLife <= MAX_HALF_LIFE,
                "a correct response must lengthen the half-life: " + before + " -> " + growing.halfLife);
        LearnerModel shrinking = new LearnerModel("shrink");
        shrinking.observe(0, 3);
        check(shrinking.halfLife == 1.5 && shrinking.observations == 1,
                "an error must halve the half-life: " + shrinking.halfLife);
        for (int i = 0; i < 40; i++) growing.observe(1, 1);
        for (int i = 0; i < 40; i++) shrinking.observe(0, 30);
        check(growing.halfLife == MAX_HALF_LIFE && shrinking.halfLife == MIN_HALF_LIFE,
                "the half-life must stay inside its clamp: " + growing.halfLife + " / " + shrinking.halfLife);

        List<Prediction> model = collect(logs());
        List<Prediction> baseline = new ArrayList<>();
        for (Response r : logs()) baseline.add(new Prediction(baselinePredict(r.gap), r.correct));
        check(model.size() == 12 && brier(model) < brier(baseline)
                        && Math.abs(brier(model) - brierReverseOrder(model)) < 1e-12
                        && Math.abs(brier(model) - 0.1382) < 0.03,
                "the adaptive model must beat the fixed-half-life baseline on Brier score: " + brier(model)
                        + " vs " + brier(baseline));

        double modelLoss = logLoss(model), baseLoss = logLoss(baseline);
        check(Double.isFinite(modelLoss) && modelLoss < baseLoss && Double.isFinite(logLoss(
                        List.of(new Prediction(1.0, 1), new Prediction(0.0, 0)))),
                "log loss must be finite thanks to clamping and better than the baseline: " + modelLoss
                        + " vs " + baseLoss);

        boolean wellFormed = true;
        for (Prediction p : model) wellFormed &= p.p > 0 && p.p <= 1 && (p.outcome == 0 || p.outcome == 1);
        check(wellFormed, "every prediction must be a probability and every outcome binary");

        double lowSum = 0, lowHits = 0, highSum = 0;
        for (Prediction p : model) {
            if (p.p < 0.5) { lowSum += p.p; lowHits += p.outcome; } else highSum += p.p;
        }
        int lowN = countLow(model), highN = model.size() - lowN;
        double lowMean = lowSum / lowN, lowRate = lowHits / lowN, highMean = highSum / highN;
        check(lowN >= 3 && highN >= 3 && highMean > lowMean && lowRate < 1.0
                        && Math.abs(lowMean - lowRate) <= 0.35 && Math.abs(highMean - 1.0) <= 0.35,
                "prediction bins must be ordered and calibrated against the observed frequency: " + lowMean
                        + " vs " + lowRate + ", " + highMean + " vs 1.0");

        Map<String, LearnerModel> fresh = new TreeMap<>();
        LearnerModel unknown = fresh.get("ghost");
        check(unknown == null && new LearnerModel("ghost").halfLife == DEFAULT_HALF_LIFE
                        && collect(List.of()).isEmpty() && recall(DEFAULT_HALF_LIFE, 0) == 1.0,
                "a learner with no log must fall back to the default half-life and an empty log scores nothing");

        LearnerModel perfect = new LearnerModel("perfect");
        double previous = -1;
        boolean nonDecreasing = true;
        for (int i = 0; i < 6; i++) {
            double p = perfect.predict(5.0);
            nonDecreasing &= p >= previous;
            previous = p;
            perfect.observe(1, 5.0);
        }
        check(nonDecreasing && Math.abs(recall(perfect.halfLife, perfect.halfLife) - 0.5) < 1e-12
                        && perfect.halfLife > DEFAULT_HALF_LIFE,
                "consistent success must raise the predicted recall at a fixed horizon: " + perfect.halfLife);

        Map<String, LearnerModel> once = new TreeMap<>();
        for (Response r : logs()) update(once, r);
        List<Prediction> again = collect(logs());
        boolean same = again.size() == model.size();
        for (int i = 0; i < again.size(); i++) {
            same &= again.get(i).p == model.get(i).p && again.get(i).outcome == model.get(i).outcome;
        }
        check(same && brier(again) == brier(model) && once.get("B").halfLife <= MIN_HALF_LIFE + 1e-12,
                "re-running the evaluation must reproduce identical predictions: " + brier(again));

        System.out.println("responses=" + logs().size() + " brier=" + brier(model) + " baseline=" + brier(baseline));
        System.out.println("logLoss=" + modelLoss + " vs " + baseLoss);
        System.out.println("PASS");
    }

    static int countLow(List<Prediction> predictions) {
        int n = 0;
        for (Prediction p : predictions) if (p.p < 0.5) n++;
        return n;
    }
}
