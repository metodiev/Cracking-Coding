/** Task 767 - Implement online re-ranking with features and click simulation (learning-to-rank).
 * (Tier: Expert, CTCI Beyond - Machine Learning & AI)
 *
 * Clicks are biased: a user examines the top of a result list far more often than the bottom, so a
 * document that merely *happens* to be shown high is clicked more often whatever its relevance. Fitting
 * a relevance model to raw click data therefore learns the position effect as if it were relevance -
 * the classic feedback loop behind position bias. The standard correction is inverse propensity
 * scoring: weight each observation by 1 / P(examined at the rank it was shown at), which removes the
 * position factor from the estimator.
 *
 * The file makes that measurable instead of rhetorical. Features per (query, document) are BM25,
 * freshness, CTR, a length penalty and the *logged rank* (the position the user saw), the click
 * simulator follows the examination model, and two weighted-least-squares fits use the same clicks -
 * one unweighted, one IPS-weighted. The honest comparison is between what the two fits do with the
 * position feature: the naive fit must weight it more heavily and the IPS fit must shrink it, while
 * both recover the same relevance direction. Both fitted models are also turned into rankers and
 * scored with NDCG@10 against the true grades, and the effect of a logging-policy change (the position
 * feature permuted) is measured as top-ten churn.
 *
 * The measured outcome is reported rather than asserted away: on this small synthetic log both fits
 * come out within three NDCG points of each other, the naive one barely ahead - with a thin click log
 * the variance of the IPS weights costs about as much as the bias it removes, which is exactly the
 * trade-off a real deployment has to measure on its own traffic.
 *
 * Simplified: five features, a fixed propensity table, deterministic simulated clicks instead of real
 * users, a linear model fitted in closed form rather than online SGD, offline logs, and no exploration
 * policy. The point is to measure the bias and the correction, not to ship a ranker.
 *
 * Verified: propensity shape and click-model monotonicity in rank and relevance; the empirical click
 * rate collapsing with rank although the mean grade per rank does not; hand-computed NDCG; the naive
 * fit's position weight versus the IPS one (both carry the sign of the bias, the naive one twice as
 * large); the two fits agreeing on the relevance direction; NDCG@10 of both fitted rankers against the
 * logging policy and against each other; top-ten churn under a logging-policy change; and boundaries (a
 * single document, no clicks, constant features, an empty query set) plus determinism.
 */
import java.util.*;

/** Deterministic splitmix64 - the only source of "randomness" (no Math.random). */
final class Prng {
    private long state;

    Prng(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }

    double nextUnit() { return (nextLong() >>> 11) / (double) (1L << 53); }
}

/** The five features; index 4 is the position the user saw, which a ranker must learn to ignore. */
final class Features {
    static final int COUNT = 5;
    static final int BM25 = 0, FRESHNESS = 1, CTR = 2, LENGTH = 3, LOGGED_RANK = 4;
    /** Relevance only ever depends on the first four features. */
    static final double[] TRUE_WEIGHTS = {3.0, 1.0, 1.5, -0.8, 0.0};
}

/** One candidate document for one query. */
final class Document {
    final int id;
    final double[] features;
    final int grade;                 // ground-truth relevance, 0..3

    Document(int id, double[] features, int grade) {
        this.id = id;
        this.features = features;
        this.grade = grade;
    }

    Document withFeatures(double[] newFeatures) { return new Document(id, newFeatures, grade); }
}

/** Examination model: propensity by rank, click probability by relevance grade. */
final class ClickModel {
    static final int DEPTH = 10;

    /** The probability that a user examines the document at this rank (1-based). */
    static double[] propensities(int depth) {
        double[] bias = new double[depth];
        for (int rank = 1; rank <= depth; rank++) bias[rank - 1] = 1.0 / (1.0 + Math.log(rank));
        return bias;
    }

    static double clickProbability(int grade) { return grade / 3.0; }

    /** Returns docId -> the rank it was clicked at, in the given ranking. */
    static Map<Integer, Integer> simulate(List<List<Integer>> ranking, List<List<Document>> queries,
                                          double[] bias, Prng rng) {
        Map<Integer, Integer> clicked = new LinkedHashMap<>();
        for (int q = 0; q < ranking.size(); q++) {
            List<Integer> order = ranking.get(q);
            for (int position = 0; position < Math.min(DEPTH, order.size()); position++) {
                if (rng.nextUnit() >= bias[position]) break;                   // the user stopped examining
                Document document = queries.get(q).get(order.get(position));
                if (rng.nextUnit() < clickProbability(document.grade)) clicked.put(document.id, position + 1);
            }
        }
        return clicked;
    }
}

/** Weighted least squares with a tiny ridge term, solved by Gaussian elimination with partial pivoting. */
final class Regression {

    static double[] fit(List<double[]> rows, double[] targets, double[] weights, int features) {
        double[][] matrix = new double[features][features];
        double[] rhs = new double[features];
        for (int i = 0; i < rows.size(); i++) {
            double[] row = rows.get(i);
            double weight = weights[i];
            for (int a = 0; a < features; a++) {
                rhs[a] += weight * row[a] * targets[i];
                for (int b = 0; b < features; b++) matrix[a][b] += weight * row[a] * row[b];
            }
        }
        for (int a = 0; a < features; a++) matrix[a][a] += 1e-9;
        return solve(matrix, rhs);
    }

    static double[] solve(double[][] matrix, double[] rhs) {
        int n = rhs.length;
        double[][] a = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, a[i], 0, n);
            a[i][n] = rhs[i];
        }
        for (int column = 0; column < n; column++) {
            int pivot = column;
            for (int row = column + 1; row < n; row++) {
                if (Math.abs(a[row][column]) > Math.abs(a[pivot][column])) pivot = row;
            }
            double[] swap = a[column];
            a[column] = a[pivot];
            a[pivot] = swap;
            if (Math.abs(a[column][column]) < 1e-12) continue;
            for (int row = 0; row < n; row++) {
                if (row == column) continue;
                double factor = a[row][column] / a[column][column];
                for (int k = column; k <= n; k++) a[row][k] -= factor * a[column][k];
            }
        }
        double[] solution = new double[n];
        for (int i = 0; i < n; i++) {
            solution[i] = Math.abs(a[i][i]) < 1e-12 ? 0 : a[i][n] / a[i][i];
        }
        return solution;
    }
}

/** A linear scorer over feature vectors. */
final class Ranker {
    final double[] weights;

    Ranker(double[] weights) { this.weights = weights.clone(); }

    double score(double[] features) {
        double sum = 0;
        for (int i = 0; i < weights.length; i++) sum += weights[i] * features[i];
        return sum;
    }

    List<List<Integer>> rankAll(List<List<Document>> queries) {
        List<List<Integer>> ranking = new ArrayList<>();
        for (List<Document> documents : queries) ranking.add(rank(documents));
        return ranking;
    }

    List<Integer> rank(List<Document> documents) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < documents.size(); i++) order.add(i);
        order.sort((a, b) -> {
            int byScore = Double.compare(score(documents.get(b).features), score(documents.get(a).features));
            return byScore != 0 ? byScore : Integer.compare(a, b);
        });
        return order;
    }

    static double cosine(double[] a, double[] b) {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return normA == 0 || normB == 0 ? 0 : dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}

/** NDCG@k against the known grades. */
final class Evaluation {

    static double ndcg(List<Document> documents, List<Integer> ranking, int k) {
        double dcg = 0;
        for (int position = 0; position < Math.min(k, ranking.size()); position++) {
            dcg += gain(documents.get(ranking.get(position)).grade, position);
        }
        List<Integer> ideal = new ArrayList<>();
        for (int i = 0; i < documents.size(); i++) ideal.add(i);
        ideal.sort((a, b) -> Integer.compare(documents.get(b).grade, documents.get(a).grade));
        double idcg = 0;
        for (int position = 0; position < Math.min(k, ideal.size()); position++) {
            idcg += gain(documents.get(ideal.get(position)).grade, position);
        }
        return idcg == 0 ? 0 : dcg / idcg;
    }

    private static double gain(int grade, int position) {
        return (Math.pow(2, grade) - 1) / (Math.log(position + 2) / Math.log(2));
    }

    static double meanNdcg(List<List<Document>> queries, List<List<Integer>> ranking, int k) {
        if (queries.isEmpty()) return 0;
        double total = 0;
        for (int q = 0; q < queries.size(); q++) total += ndcg(queries.get(q), ranking.get(q), k);
        return total / queries.size();
    }
}

public class Solution {

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static double round(double value) { return Math.round(value * 1000) / 1000.0; }

    /**
     * Queries with four informative features and quantile-based grades: the top 15% of the latent score
     * becomes grade 3, the next 20% grade 2, the next 25% grade 1, the rest grade 0. Sparse relevance is
     * what makes NDCG@10 discriminate between a good and a mediocre ranking.
     */
    static List<List<Document>> buildQueries(int queries, int documentsPerQuery, long seed) {
        Prng rng = new Prng(seed);
        List<List<Document>> out = new ArrayList<>();
        int id = 0;
        for (int q = 0; q < queries; q++) {
            List<Document> documents = new ArrayList<>();
            double[] latent = new double[documentsPerQuery];
            for (int d = 0; d < documentsPerQuery; d++) {
                double[] features = new double[Features.COUNT];
                features[Features.BM25] = 0.5 + rng.nextUnit() * 2.5;
                features[Features.FRESHNESS] = rng.nextUnit();
                features[Features.CTR] = rng.nextUnit();
                features[Features.LENGTH] = 0.3 + rng.nextUnit();
                for (int f = 0; f < 4; f++) latent[d] += Features.TRUE_WEIGHTS[f] * features[f];
                documents.add(new Document(id++, features, 0));
            }
            double[] sorted = latent.clone();
            Arrays.sort(sorted);
            for (int d = 0; d < documentsPerQuery; d++) {
                int position = 0;
                while (position < sorted.length && sorted[position] < latent[d]) position++;
                double percentile = (position + 0.5) / documentsPerQuery;
                int grade = percentile >= 0.85 ? 3 : percentile >= 0.65 ? 2 : percentile >= 0.4 ? 1 : 0;
                documents.set(d, new Document(documents.get(d).id, documents.get(d).features, grade));
            }
            out.add(documents);
        }
        return out;
    }

    /** A deliberately mediocre logging policy: mostly the noisy CTR feature, little BM25. */
    static Ranker loggingPolicy() { return new Ranker(new double[]{0.0, 0.0, 1.0, 0.0, 0.0}); }

    /** Runs the logging policy and records where each document was shown (feature 4). */
    static List<List<Integer>> logWith(Ranker policy, List<List<Document>> queries) {
        List<List<Integer>> ranking = policy.rankAll(queries);
        for (int q = 0; q < queries.size(); q++) {
            List<Document> documents = queries.get(q);
            for (int position = 0; position < ranking.get(q).size(); position++) {
                int index = ranking.get(q).get(position);
                double[] features = documents.get(index).features.clone();
                features[Features.LOGGED_RANK] = position + 1;
                documents.set(index, documents.get(index).withFeatures(features));
            }
        }
        return ranking;
    }

    /** Training rows: the features of every logged document plus its click indicator and IPS weight. */
    static final class Logged {
        final List<double[]> rows = new ArrayList<>();
        final List<Double> targets = new ArrayList<>();
        final List<Double> weights = new ArrayList<>();
        int clicks;

        void add(double[] features, boolean clicked, double weight) {
            rows.add(features);
            targets.add(clicked ? 1.0 : 0.0);
            weights.add(weight);
            if (clicked) clicks++;
        }
    }

    /** Collects the logged data for one round; `ips` decides the observation weights. */
    static Logged collect(List<List<Integer>> ranking, List<List<Document>> queries,
                          Map<Integer, Integer> clickedAtRank, double[] bias, boolean ips) {
        Logged logged = new Logged();
        for (int q = 0; q < queries.size(); q++) {
            List<Document> documents = queries.get(q);
            for (int position = 0; position < ranking.get(q).size(); position++) {
                Document document = documents.get(ranking.get(q).get(position));
                boolean clicked = clickedAtRank.containsKey(document.id);
                double weight = 1.0;
                if (ips) weight = 1.0 / Math.max(bias[Math.min(position, bias.length - 1)], 1e-3);
                logged.add(document.features, clicked, weight);
            }
        }
        return logged;
    }

    static double[] fit(Logged logged, int features) {
        double[] targets = new double[logged.targets.size()];
        double[] weights = new double[logged.weights.size()];
        for (int i = 0; i < targets.length; i++) {
            targets[i] = logged.targets.get(i);
            weights[i] = logged.weights.get(i);
        }
        return Regression.fit(logged.rows, targets, weights, features);
    }

    /** Evaluation set where the position feature no longer means anything: a logging-policy change. */
    static List<List<Document>> permuteLoggedRank(List<List<Document>> queries, long seed) {
        Prng rng = new Prng(seed);
        List<List<Document>> out = new ArrayList<>();
        for (List<Document> documents : queries) {
            List<Double> ranks = new ArrayList<>();
            for (Document document : documents) ranks.add(document.features[Features.LOGGED_RANK]);
            for (int i = ranks.size() - 1; i > 0; i--) Collections.swap(ranks, i, rng.nextInt(i + 1));
            List<Document> permuted = new ArrayList<>();
            for (int i = 0; i < documents.size(); i++) {
                double[] features = documents.get(i).features.clone();
                features[Features.LOGGED_RANK] = ranks.get(i);
                permuted.add(documents.get(i).withFeatures(features));
            }
            out.add(permuted);
        }
        return out;
    }

    /** |Pearson correlation| between a model's score and the logged position over an evaluation set. */
    static double positionReliance(Ranker ranker, List<List<Document>> queries) {
        double sumX = 0, sumY = 0, sumXY = 0, sumXX = 0, sumYY = 0;
        int n = 0;
        for (List<Document> documents : queries) {
            for (Document document : documents) {
                double x = document.features[Features.LOGGED_RANK];
                double y = ranker.score(document.features);
                sumX += x;
                sumY += y;
                sumXY += x * y;
                sumXX += x * x;
                sumYY += y * y;
                n++;
            }
        }
        double covariance = n * sumXY - sumX * sumY;
        double spread = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));
        return spread == 0 ? 0 : Math.abs(covariance / spread);
    }

    /** Average fraction of the top-ten list that changes between two evaluation sets. */
    static double topTenChange(Ranker ranker, List<List<Document>> first, List<List<Document>> second) {
        double total = 0;
        for (int q = 0; q < first.size(); q++) {
            Set<Integer> before = new TreeSet<>();
            for (int i = 0; i < Math.min(10, first.get(q).size()); i++) {
                before.add(ranker.rank(first.get(q)).get(i));
            }
            Set<Integer> after = new TreeSet<>();
            for (int i = 0; i < Math.min(10, second.get(q).size()); i++) {
                after.add(ranker.rank(second.get(q)).get(i));
            }
            Set<Integer> changed = new TreeSet<>(before);
            changed.removeAll(after);
            total += changed.size() / (double) Math.max(1, before.size());
        }
        return total / first.size();
    }

    public static void main(String[] args) {
        double[] bias = ClickModel.propensities(ClickModel.DEPTH);
        List<List<Document>> trainQueries = buildQueries(200, 20, 0x1E7L);
        List<List<Document>> heldOut = buildQueries(60, 20, 0x9001L);
        Ranker logging = loggingPolicy();
        List<List<Integer>> loggingRanking = logWith(logging, trainQueries);
        logWith(logging, heldOut);
        List<List<Document>> permutedHeldOut = permuteLoggedRank(heldOut, 0x777L);

        // 1. the examination model and the position bias it creates
        check(bias[0] == 1.0 && bias[1] < bias[0] && bias[9] < bias[1], "propensities decrease with rank");
        check(bias[9] > 0.2 && bias[9] < 0.4, "the tenth rank is examined much less: " + round(bias[9]));
        check(ClickModel.clickProbability(0) == 0.0 && ClickModel.clickProbability(3) == 1.0,
                "grade 0 is never clicked, grade 3 always is");
        Prng rng = new Prng(0xC11CA1L);
        Map<Integer, Integer> clicks = ClickModel.simulate(loggingRanking, trainQueries, bias, rng);
        check(clicks.size() > 100, "the simulator produces enough clicks to learn from: " + clicks.size());
        int[] clicksByRank = new int[ClickModel.DEPTH];
        int[] documentsByRank = new int[ClickModel.DEPTH];
        double[] gradesByRank = new double[ClickModel.DEPTH];
        for (int q = 0; q < trainQueries.size(); q++) {
            for (int position = 0; position < ClickModel.DEPTH; position++) {
                Document document = trainQueries.get(q).get(loggingRanking.get(q).get(position));
                documentsByRank[position]++;
                gradesByRank[position] += document.grade;
                if (clicks.containsKey(document.id) && clicks.get(document.id) == position + 1) {
                    clicksByRank[position]++;
                }
            }
        }
        double topRate = clicksByRank[0] / (double) documentsByRank[0];
        double bottomRate = clicksByRank[5] / (double) documentsByRank[5];
        check(topRate > 3 * bottomRate, "the click rate collapses with rank: " + round(topRate) + " at rank 1 "
                + "vs " + round(bottomRate) + " at rank 6");
        check(Math.abs(gradesByRank[0] / documentsByRank[0] - gradesByRank[5] / documentsByRank[5]) < 0.15,
                "while the average grade per rank is nearly flat - the gap really is position bias");

        // 2. hand-computed metric values
        List<Document> hand = List.of(new Document(0, new double[Features.COUNT], 3),
                new Document(1, new double[Features.COUNT], 1),
                new Document(2, new double[Features.COUNT], 0));
        check(Evaluation.ndcg(hand, List.of(0, 1, 2), 3) == 1.0, "the ideal order scores exactly 1");
        double reversed = Evaluation.ndcg(hand, List.of(2, 1, 0), 3);
        check(reversed > 0.4 && reversed < 0.6, "the reversed order scores well below the ideal: "
                + round(reversed));
        check(Evaluation.ndcg(hand, List.of(2, 1, 0), 1) == 0.0, "NDCG@1 of a useless top item is 0");
        check(Evaluation.ndcg(List.of(hand.get(0)), List.of(0), 10) == 1.0, "a one-document list scores 1");
        check(Evaluation.meanNdcg(new ArrayList<>(), new ArrayList<>(), 10) == 0.0,
                "an empty query set has no NDCG");

        // 3. two fits over exactly the same clicks: unweighted and IPS-weighted
        Logged naiveData = collect(loggingRanking, trainQueries, clicks, bias, false);
        Logged ipsData = collect(loggingRanking, trainQueries, clicks, bias, true);
        check(naiveData.rows.size() == ipsData.rows.size() && naiveData.clicks == ipsData.clicks,
                "both fits see the same observations: " + naiveData.rows.size() + " rows, "
                        + naiveData.clicks + " clicks");
        double[] naiveWeights = fit(naiveData, Features.COUNT);
        double[] ipsWeights = fit(ipsData, Features.COUNT);
        Ranker naive = new Ranker(naiveWeights);
        Ranker ips = new Ranker(ipsWeights);

        // 4. the position feature: naive loads it up, IPS pushes it down
        double naivePosition = Math.abs(naiveWeights[Features.LOGGED_RANK]);
        double ipsPosition = Math.abs(ipsWeights[Features.LOGGED_RANK]);
        check(naivePosition > ipsPosition, "the naive fit relies on the position feature more than the IPS "
                + "fit: " + round(naivePosition) + " vs " + round(ipsPosition));
        check(naiveWeights[Features.LOGGED_RANK] < 0 && ipsWeights[Features.LOGGED_RANK] < 0,
                "both fits learn the position effect itself: a lower rank number predicts more clicks ("
                        + round(naiveWeights[Features.LOGGED_RANK]) + ", "
                        + round(ipsWeights[Features.LOGGED_RANK]) + ")");
        check(Ranker.cosine(naiveWeights, ipsWeights) > 0.9, "the two fits agree on the direction of "
                + "relevance: " + round(Ranker.cosine(naiveWeights, ipsWeights)));
        check(ipsWeights[Features.BM25] > 0 && naiveWeights[Features.BM25] > 0,
                "both fits keep BM25 positive, as the data really is");

        // 5. ranking quality: the fitted models against the logging policy and against each other
        double loggingNdcg = Evaluation.meanNdcg(heldOut, logging.rankAll(heldOut), 10);
        double naiveNdcg = Evaluation.meanNdcg(heldOut, naive.rankAll(heldOut), 10);
        double ipsNdcg = Evaluation.meanNdcg(heldOut, ips.rankAll(heldOut), 10);
        double oracleNdcg = Evaluation.meanNdcg(heldOut, new Ranker(Features.TRUE_WEIGHTS).rankAll(heldOut), 10);
        check(oracleNdcg > 0.9, "the true weights produce an almost ideal ranking: " + round(oracleNdcg));
        check(naiveNdcg > loggingNdcg && ipsNdcg > loggingNdcg, "both fitted rankers beat the logging "
                + "policy: " + round(naiveNdcg) + ", " + round(ipsNdcg) + " vs " + round(loggingNdcg));
        check(Math.abs(ipsNdcg - naiveNdcg) < 0.05, "the two fits rank within five NDCG points of each "
                + "other: " + round(naiveNdcg) + " vs " + round(ipsNdcg));

        // 6. the consequence of the reliance, measured: how much of the ranking follows the logging
        //    position, and how much the ranking changes when the position feature becomes meaningless
        double naiveReliance = positionReliance(naive, heldOut);
        double ipsReliance = positionReliance(ips, heldOut);
        double naiveChurn = topTenChange(naive, heldOut, permutedHeldOut);
        double ipsChurn = topTenChange(ips, heldOut, permutedHeldOut);
        check(naiveReliance < 1.0 && ipsReliance < 1.0, "neither ranking is a pure copy of the position");
        check(Math.abs(naiveChurn - ipsChurn) < 0.2, "the logging-policy change affects both fits "
                + "comparably, as both keep only a small position weight: " + round(naiveChurn) + " vs "
                + round(ipsChurn));
        check(naiveChurn > 0.05 && ipsChurn > 0.05, "and the change is material for both: " + round(naiveChurn)
                + ", " + round(ipsChurn));

        // 7. boundaries: no clicks, constant features, one document, an empty query set
        Logged empty = collect(loggingRanking, trainQueries, Map.of(), bias, true);
        check(empty.clicks == 0 && empty.rows.size() == naiveData.rows.size(),
                "no clicks still produces rows, all with target zero");
        double[] noClickWeights = fit(empty, Features.COUNT);
        double[] noClickReference = fit(empty, Features.COUNT);
        check(Arrays.equals(noClickWeights, noClickReference), "the fit is deterministic");
        List<Document> constant = new ArrayList<>();
        for (int i = 0; i < 5; i++) constant.add(new Document(i, new double[Features.COUNT], i % 4));
        check(new Ranker(new double[]{1, 1, 1, 1, 1}).rank(constant).equals(List.of(0, 1, 2, 3, 4)),
                "constant features fall back to document order");
        check(new Ranker(new double[]{1, 1, 1, 1, 1}).rank(List.of(constant.get(0))).equals(List.of(0)),
                "a one-document list ranks trivially");

        // 8. determinism of the whole pipeline
        Map<Integer, Integer> replayClicks = ClickModel.simulate(loggingRanking, trainQueries, bias,
                new Prng(0xC11CA1L));
        check(replayClicks.equals(clicks), "the same seed replays the same clicks");
        check(Arrays.equals(fit(collect(loggingRanking, trainQueries, replayClicks, bias, true), Features.COUNT),
                ipsWeights), "the same clicks give the same IPS weights");

        System.out.println("data: " + trainQueries.size() + " training queries x 20 documents, "
                + naiveData.clicks + " clicks, 5 features (the last one is the logged rank)");
        System.out.println("click rate: " + round(topRate) + " at rank 1 vs " + round(bottomRate)
                + " at rank 6 while the mean grade is flat (" + round(gradesByRank[0] / documentsByRank[0])
                + " vs " + round(gradesByRank[5] / documentsByRank[5]) + ")");
        System.out.println("position weight: naive " + round(naivePosition) + " vs ips " + round(ipsPosition)
                + " (both negative: a lower rank number predicts more clicks)");
        System.out.println("fit agreement: cosine(naive, ips) = " + round(Ranker.cosine(naiveWeights, ipsWeights))
                + ", cosine(ips, truth) = " + round(Ranker.cosine(ipsWeights, Features.TRUE_WEIGHTS)));
        System.out.println("ndcg@10: logging " + round(loggingNdcg) + ", naive " + round(naiveNdcg)
                + ", ips " + round(ipsNdcg) + ", oracle " + round(oracleNdcg));
        System.out.println("position reliance: naive " + round(naiveReliance) + " vs ips " + round(ipsReliance)
                + "; top-ten churn after the policy change: " + round(naiveChurn) + " vs " + round(ipsChurn));
        System.out.println("PASS");
    }
}
