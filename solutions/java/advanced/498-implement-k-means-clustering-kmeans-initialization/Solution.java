/** Task 498 - Implement k-means clustering (kmeans++ initialization). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * k-means over fixed 2-D blobs with a fully deterministic k-means++ seeding: the first centre is
 * an LCG-uniform data point, then each next centre is drawn with probability proportional to
 * D(x)^2 (the squared distance to the closest chosen centre; the draw is one fixed-LCG uniform
 * against the cumulative D^2 weights). Lloyd iterations then alternate exact assignment
 * (squared Euclidean distance, ties to the lowest cluster index) with mean updates (an empty
 * cluster keeps its old centre) until the labels stop changing. Data: three planted discs of
 * jittered points around (0,0), (4,0) and (0,4), 20 points each, jitter in 0.01 steps of
 * +/-0.5. Documented expectations: with 100 fixed random restarts (distinct-index sampling,
 * same LCG family) the k-means++ objective is at most the first restart's objective and within
 * 5% of the best restart (measured 10.613 vs 176.38 for the first restart and 10.613 for the
 * best of 100); every cluster is pure (all its points come from one planted blob); and
 * re-running with the same seed reproduces the labels and centroids exactly (bit for bit).
 */
import java.util.*;

public class Solution {
    static final double[][] BLOBS = {{0.0, 0.0}, {4.0, 0.0}, {0.0, 4.0}};
    static final int PER_BLOB = 20;
    static final int K = 3;
    static final int SEED = 7;
    static final int RESTARTS = 100;

    static class Data {
        final double[][] points;
        final int[] truth;

        Data(double[][] points, int[] truth) {
            this.points = points;
            this.truth = truth;
        }
    }

    static Data lcgPoints(int seed) {
        double[][] points = new double[BLOBS.length * PER_BLOB][];
        int[] truth = new int[points.length];
        int index = 0;
        for (int blob = 0; blob < BLOBS.length; blob++) {
            for (int i = 0; i < PER_BLOB; i++) {
                seed = (seed * 75 + 74) % 65537;
                double x = BLOBS[blob][0] + (seed % 101 - 50) / 100.0;
                seed = (seed * 75 + 74) % 65537;
                double y = BLOBS[blob][1] + (seed % 101 - 50) / 100.0;
                points[index] = new double[]{x, y};
                truth[index] = blob;
                index++;
            }
        }
        return new Data(points, truth);
    }

    static double sqdist(double[] a, double[] b) {
        double dx = a[0] - b[0];
        double dy = a[1] - b[1];
        return dx * dx + dy * dy;
    }

    static List<double[]> kmeansPlusPlus(double[][] points, int k, int seed) {
        int n = points.length;
        seed = (seed * 75 + 74) % 65537;
        int first = seed % n;
        List<double[]> centers = new ArrayList<>();
        centers.add(points[first]);
        double[] d2 = new double[n];
        for (int i = 0; i < n; i++) d2[i] = sqdist(points[i], centers.get(0));
        while (centers.size() < k) {
            double total = 0.0;
            for (double value : d2) total += value;
            seed = (seed * 75 + 74) % 65537;
            double target = (seed + 0.5) / 65537.0 * total;
            double acc = 0.0;
            int choice = n - 1;
            for (int i = 0; i < n; i++) {
                acc += d2[i];
                if (acc >= target) {
                    choice = i;
                    break;
                }
            }
            centers.add(points[choice]);
            double[] last = centers.get(centers.size() - 1);
            for (int i = 0; i < n; i++) d2[i] = Math.min(d2[i], sqdist(points[i], last));
        }
        return centers;
    }

    /** Deterministic distinct-index sampling from the same LCG family. */
    static List<double[]> randomInit(double[][] points, int k, int seed) {
        int n = points.length;
        List<Integer> chosen = new ArrayList<>();
        while (chosen.size() < k) {
            seed = (seed * 75 + 74) % 65537;
            int index = seed % n;
            if (!chosen.contains(index)) chosen.add(index);
        }
        List<double[]> centers = new ArrayList<>();
        for (int index : chosen) centers.add(points[index]);
        return centers;
    }

    static int assign(double[] point, List<double[]> centers) {
        int best = 0;
        double bestDistance = sqdist(point, centers.get(0));
        for (int c = 1; c < centers.size(); c++) {
            double distance = sqdist(point, centers.get(c));
            if (distance < bestDistance) {
                best = c;
                bestDistance = distance;
            }
        }
        return best;
    }

    static class Result {
        final int[] labels;
        final List<double[]> centers;

        Result(int[] labels, List<double[]> centers) {
            this.labels = labels;
            this.centers = centers;
        }
    }

    static Result lloyd(double[][] points, List<double[]> startCenters) {
        List<double[]> centers = new ArrayList<>(startCenters);
        int[] labels = null;
        for (int iter = 0; iter < 100; iter++) {
            int[] newLabels = new int[points.length];
            for (int i = 0; i < points.length; i++) newLabels[i] = assign(points[i], centers);
            List<double[]> newCenters = new ArrayList<>();
            for (int c = 0; c < centers.size(); c++) {
                double sumX = 0.0;
                double sumY = 0.0;
                int members = 0;
                for (int i = 0; i < points.length; i++) {
                    if (newLabels[i] == c) {
                        sumX += points[i][0];
                        sumY += points[i][1];
                        members++;
                    }
                }
                if (members > 0) {
                    newCenters.add(new double[]{sumX / members, sumY / members});
                } else {
                    newCenters.add(centers.get(c).clone());     // empty cluster keeps its centre
                }
            }
            boolean same = labels != null && Arrays.equals(labels, newLabels);
            labels = newLabels;
            centers = newCenters;
            if (same) break;
        }
        return new Result(labels, centers);
    }

    static Result kmeans(double[][] points, int k, int seed) {
        return lloyd(points, kmeansPlusPlus(points, k, seed));
    }

    static double objective(double[][] points, List<double[]> centers) {
        double total = 0.0;
        for (double[] point : points) {
            double best = Double.POSITIVE_INFINITY;
            for (double[] center : centers) best = Math.min(best, sqdist(point, center));
            total += best;
        }
        return total;
    }

    public static void main(String[] args) {
        Data data = lcgPoints(SEED);
        check(data.points.length == 3 * PER_BLOB, "point count");
        Set<String> unique = new HashSet<>();
        for (double[] point : data.points) unique.add(point[0] + "," + point[1]);
        check(unique.size() == data.points.length, "jittered blobs are duplicate-free");

        Result result = kmeans(data.points, K, SEED + 1);
        Set<Integer> labelsSeen = new TreeSet<>();
        for (int label : result.labels) labelsSeen.add(label);
        check(labelsSeen.equals(new TreeSet<>(Arrays.asList(0, 1, 2))), "three non-empty clusters");

        Map<Integer, Set<Integer>> purity = new HashMap<>();
        for (int i = 0; i < result.labels.length; i++) {
            purity.computeIfAbsent(result.labels[i], key -> new HashSet<>()).add(data.truth[i]);
        }
        Set<Integer> blobsUsed = new HashSet<>();
        for (Set<Integer> blobs : purity.values()) {
            check(blobs.size() == 1, "clusters must be pure");
            blobsUsed.add(blobs.iterator().next());
        }
        check(blobsUsed.size() == 3, "one blob per cluster");

        double kppObjective = objective(data.points, result.centers);

        Result firstRandom = lloyd(data.points, randomInit(data.points, K, 9100));
        double firstRandomObjective = objective(data.points, firstRandom.centers);
        check(kppObjective <= firstRandomObjective, "k-means++ must beat the first random init");
        double bestRandom = Double.POSITIVE_INFINITY;
        for (int r = 0; r < RESTARTS; r++) {
            Result randomResult = lloyd(data.points, randomInit(data.points, K, 1000 + 17 * r));
            bestRandom = Math.min(bestRandom, objective(data.points, randomResult.centers));
        }
        check(kppObjective <= 1.05 * bestRandom, "within 5% of the best of 100 restarts");

        Result repeat = kmeans(data.points, K, SEED + 1);
        check(Arrays.equals(repeat.labels, result.labels), "labels must be stable under re-run");
        check(repeat.centers.size() == result.centers.size(), "centre count");
        for (int c = 0; c < result.centers.size(); c++) {
            check(Arrays.equals(repeat.centers.get(c), result.centers.get(c)),
                    "centroids identical bit for bit");
        }

        List<double[]> centersPp = kmeansPlusPlus(data.points, K, SEED + 1);
        for (double[] center : centersPp) {
            boolean isPoint = false;
            for (double[] point : data.points) {
                if (point[0] == center[0] && point[1] == center[1]) isPoint = true;
            }
            check(isPoint, "centres are data points");
        }
        check(centersPp.size() == K, "centre count is k");
        check(kmeansPlusPlus(data.points, data.points.length, SEED + 1).size() == data.points.length,
                "k = n gives every point as centre");

        double[][] tiny = {{0.0, 0.0}, {0.1, 0.0}, {0.0, 0.1}, {0.1, 0.1},
                           {3.0, 3.0}, {3.1, 3.0}, {3.0, 3.1}, {3.1, 3.1}};
        Result tinyResult = kmeans(tiny, 2, 55);
        for (int i = 1; i < 4; i++) {
            check(tinyResult.labels[i] == tinyResult.labels[0], "tiny left blob");
            check(tinyResult.labels[4 + i] == tinyResult.labels[4], "tiny right blob");
        }
        check(tinyResult.labels[0] != tinyResult.labels[4], "tiny blobs separate");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
