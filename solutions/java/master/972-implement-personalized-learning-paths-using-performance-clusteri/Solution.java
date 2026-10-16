/** Task 972 - Implement personalized learning paths using performance clustering.
 *  (Tier: Master, CTCI Beyond -- Software Engineering & Tooling)
 *
 *  A learner model with one mastery value per topic is clustered with deterministic k-means: Lloyd's
 *  algorithm seeded by farthest-point selection (first seed = first learner in declaration order, then the
 *  point farthest from the chosen seeds, ties broken by declaration order), iterating until the assignment
 *  is stable or a capped number of rounds. Empty clusters are dropped from the final report and their
 *  centroid carried over instead of dividing by zero. Each learner then gets a personalized path: the
 *  `length` weakest topics (largest 1 - mastery) ordered by the prerequisite subgraph induced on that set,
 *  so a chosen topic still follows every chosen prerequisite.
 *  Simplified: Euclidean k-means with k fixed by the caller, no per-learner difficulty model and no
 *  problem-level interleaving; the coordinate-descent monotonicity, the centroid arithmetic and the
 *  prerequisite-respecting ordering are the real properties being checked.
 *  Verified: cluster membership against a hand-derived partition, SSE against an independent brute-force
 *  recomputation, non-increasing SSE history, centroid == member mean, path topic set against an
 *  independent weakest-first selection, prerequisite order, identical learners sharing a path, the
 *  all-identical/single-learner/empty-cluster boundaries and determinism.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final String[] TOPICS = {"arrays", "sorting", "hashing", "graphs", "dp"};
    static final Map<String, List<String>> PREREQS = Map.of(
            "arrays", List.of(), "sorting", List.of("arrays"), "hashing", List.of("arrays"),
            "graphs", List.of("hashing", "sorting"), "dp", List.of("graphs"));

    static final class Learner {
        final String id; final double[] skills;
        Learner(String id, double... skills) { this.id = id; this.skills = skills; }
    }

    static final class Clustering {
        final List<List<Learner>> clusters; final double[][] centroids; final List<Double> sseHistory;
        Clustering(List<List<Learner>> clusters, double[][] centroids, List<Double> sseHistory) {
            this.clusters = clusters; this.centroids = centroids; this.sseHistory = sseHistory;
        }
        double sse() { return sseHistory.isEmpty() ? 0.0 : sseHistory.get(sseHistory.size() - 1); }
        String render() {
            StringBuilder out = new StringBuilder();
            for (List<Learner> c : clusters) {
                List<String> ids = new ArrayList<>();
                for (Learner l : c) ids.add(l.id);
                out.append(ids).append(' ');
            }
            return out.toString().trim();
        }
    }

    static double dist2(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += (a[i] - b[i]) * (a[i] - b[i]);
        return sum;
    }

    static int nearest(List<double[]> centroids, Learner l) {
        int best = 0;
        for (int c = 1; c < centroids.size(); c++) {
            if (dist2(centroids.get(c), l.skills) < dist2(centroids.get(best), l.skills)) best = c;
        }
        return best;
    }

    /** Assignment -> centroid means; an empty cluster keeps its previous centroid to avoid NaN. */
    static List<double[]> recompute(List<Learner> learners, int[] assign, List<double[]> previous) {
        int k = previous.size(), dim = TOPICS.length;
        double[][] sum = new double[k][dim];
        int[] count = new int[k];
        for (int i = 0; i < learners.size(); i++) {
            count[assign[i]]++;
            for (int d = 0; d < dim; d++) sum[assign[i]][d] += learners.get(i).skills[d];
        }
        List<double[]> out = new ArrayList<>();
        for (int c = 0; c < k; c++) {
            if (count[c] == 0) { out.add(previous.get(c)); continue; }
            double[] mean = new double[dim];
            for (int d = 0; d < dim; d++) mean[d] = sum[c][d] / count[c];
            out.add(mean);
        }
        return out;
    }

    static double sse(List<Learner> learners, List<double[]> centroids, int[] assign) {
        double total = 0;
        for (int i = 0; i < learners.size(); i++) total += dist2(centroids.get(assign[i]), learners.get(i).skills);
        return total;
    }

    static Clustering kmeans(List<Learner> learners, int k) {
        List<double[]> centroids = new ArrayList<>();
        centroids.add(learners.get(0).skills.clone());
        while (centroids.size() < k && centroids.size() < learners.size()) {
            int pick = 0;                                                    // farthest-point seeding
            double best = -1;
            for (int i = 0; i < learners.size(); i++) {
                double closest = Double.MAX_VALUE;
                for (double[] c : centroids) closest = Math.min(closest, dist2(c, learners.get(i).skills));
                if (closest > best) { best = closest; pick = i; }
            }
            centroids.add(learners.get(pick).skills.clone());
        }
        int[] assign = new int[learners.size()];
        Arrays.fill(assign, -1);
        List<Double> history = new ArrayList<>();
        for (int iter = 0; iter < 25; iter++) {
            int[] next = new int[learners.size()];
            for (int i = 0; i < learners.size(); i++) next[i] = nearest(centroids, learners.get(i));
            boolean stable = Arrays.equals(next, assign);
            assign = next;
            history.add(sse(learners, centroids, assign));
            if (stable && iter > 0) break;
            centroids = recompute(learners, assign, centroids);
            history.add(sse(learners, centroids, assign));
        }
        Map<Integer, List<Learner>> members = new TreeMap<>();
        for (int i = 0; i < learners.size(); i++) {
            members.computeIfAbsent(assign[i], x -> new ArrayList<>()).add(learners.get(i));
        }
        List<List<Learner>> clusters = new ArrayList<>();
        List<double[]> finalCentroids = new ArrayList<>();
        for (Map.Entry<Integer, List<Learner>> e : members.entrySet()) {
            clusters.add(e.getValue());
            finalCentroids.add(centroids.get(e.getKey()));
        }
        return new Clustering(clusters, finalCentroids.toArray(new double[0][]), history);
    }

    static int indexOf(String topic) { return List.of(TOPICS).indexOf(topic); }
    static double weakness(Learner l, String topic) { return 1.0 - l.skills[indexOf(topic)]; }

    /** The `length` weakest topics for the learner, ordered so chosen prerequisites come first. */
    static List<String> path(Learner l, int length) {
        List<String> ranked = new ArrayList<>(List.of(TOPICS));
        ranked.sort(Comparator.comparingDouble((String t) -> -weakness(l, t)).thenComparing(Solution::indexOf));
        List<String> chosen = new ArrayList<>(ranked.subList(0, Math.min(length, ranked.size())));
        List<String> out = new ArrayList<>();
        while (!chosen.isEmpty()) {
            for (String t : chosen) {
                boolean ready = true;
                for (String pre : PREREQS.get(t)) if (chosen.contains(pre)) ready = false;
                if (ready) { out.add(t); chosen.remove(t); break; }
            }
        }
        return out;
    }

    static List<Learner> cohort() {
        return List.of(new Learner("L1", 0.20, 0.10, 0.30, 0.40, 0.50),
                new Learner("L2", 0.20, 0.10, 0.30, 0.40, 0.50),
                new Learner("L3", 0.90, 0.80, 0.85, 0.70, 0.60),
                new Learner("L4", 0.85, 0.90, 0.80, 0.75, 0.70),
                new Learner("L5", 0.50, 0.55, 0.60, 0.20, 0.15),
                new Learner("L6", 0.55, 0.50, 0.65, 0.25, 0.30));
    }

    public static void main(String[] args) {
        List<Learner> learners = cohort();
        Clustering c = kmeans(learners, 3);

        List<String> groups = new ArrayList<>();
        for (List<Learner> cluster : c.clusters) {
            List<String> ids = new ArrayList<>();
            for (Learner l : cluster) ids.add(l.id);
            groups.add(String.join("", ids));
        }
        check(groups.equals(List.of("L1L2", "L3L4", "L5L6")) && c.clusters.size() == 3,
                "farthest-point seeding plus Lloyd must recover the hand-derived partition: " + groups);

        double brute = 0;
        for (Learner l : learners) {
            double best = Double.MAX_VALUE;
            for (double[] centroid : c.centroids) best = Math.min(best, dist2(centroid, l.skills));
            brute += best;
        }
        check(Math.abs(c.sse() - 0.03) < 1e-12 && Math.abs(c.sse() - brute) < 1e-9,
                "SSE must match an independent brute-force recomputation: " + c.sse() + " vs " + brute);

        boolean monotone = true;
        for (int i = 1; i < c.sseHistory.size(); i++) {
            monotone &= c.sseHistory.get(i) <= c.sseHistory.get(i - 1) + 1e-12;
        }
        check(monotone && c.sseHistory.size() >= 3 && c.sseHistory.get(0) > c.sseHistory.get(1),
                "each Lloyd round must not increase the SSE: " + c.sseHistory);

        boolean means = true;
        for (int ci = 0; ci < c.clusters.size(); ci++) {
            double[] mean = new double[TOPICS.length];
            for (Learner l : c.clusters.get(ci)) for (int d = 0; d < TOPICS.length; d++) mean[d] += l.skills[d];
            for (int d = 0; d < TOPICS.length; d++) {
                means &= Math.abs(mean[d] / c.clusters.get(ci).size() - c.centroids[ci][d]) < 1e-12;
            }
        }
        check(means, "every reported centroid must equal the mean of its members");

        List<String> p1 = path(learners.get(0), 4);
        List<String> p5 = path(learners.get(4), 4);
        check(new TreeSet<>(p1).equals(new TreeSet<>(List.of("sorting", "arrays", "hashing", "graphs")))
                        && p1.size() == 4 && PREREQS.get(p1.get(0)).stream().noneMatch(p1::contains),
                "the path must select the four weakest topics and start with a prerequisite-free root: " + p1);
        check(p1.equals(List.of("arrays", "sorting", "hashing", "graphs"))
                        && p5.equals(List.of("arrays", "sorting", "graphs", "dp")),
                "chosen prerequisites must be ordered before their dependants: " + p1 + " / " + p5);

        boolean prereqOrder = true;
        for (List<String> p : List.of(p1, p5, path(learners.get(2), 3))) {
            for (String t : p) for (String pre : PREREQS.get(t)) {
                if (p.contains(pre)) prereqOrder &= p.indexOf(pre) < p.indexOf(t);
            }
        }
        check(prereqOrder && path(learners.get(0), 9).size() == TOPICS.length
                        && path(learners.get(2), 3).equals(List.of("sorting", "graphs", "dp")),
                "prerequisites inside a path must always precede their dependants: " + path(learners.get(2), 3));
        check(path(learners.get(0), 4).equals(path(learners.get(1), 4)) && c.clusters.get(0).size() == 2,
                "learners with identical mastery must share a cluster and a path: " + path(learners.get(1), 4));

        List<Learner> twins = new ArrayList<>();
        for (int i = 0; i < 5; i++) twins.add(new Learner("T" + i, 0.4, 0.4, 0.4, 0.4, 0.4));
        Clustering flat = kmeans(twins, 3);
        Clustering single = kmeans(List.of(new Learner("only", 0.1, 0.2, 0.3, 0.4, 0.5)), 3);
        check(flat.clusters.size() == 1 && flat.sse() == 0.0 && single.clusters.size() == 1 && single.sse() == 0.0,
                "identical learners must collapse into one cluster and a lone learner must not divide by zero");

        check(kmeans(cohort(), 3).render().equals(c.render()) && kmeans(cohort(), 3).sse() == c.sse()
                        && path(learners.get(0), 4).equals(p1),
                "re-running the pipeline must reproduce the same clusters and paths");

        System.out.println("learners=" + learners.size() + " clusters=" + c.render() + " sse=" + c.sse());
        System.out.println("path L1=" + p1 + " path L5=" + p5 + " history=" + c.sseHistory);
        System.out.println("PASS");
    }
}
