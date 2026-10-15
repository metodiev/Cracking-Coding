/** Task 499 - Implement DBSCAN clustering (grid-based accelerate). (Tier: Advanced, CTCI Beyond — Machine Learning & AI)
 *
 * Density-based clustering (eps-neighbourhoods count points with squared distance <= eps^2,
 * minPts core threshold) with two interchangeable neighbourhood engines: the naive one scans
 * all n points, the accelerated one hashes points into a grid of cell size eps and probes only
 * the 3x3 block of cells around a query - every point within eps must lie in that block, so the
 * results are identical while far fewer distances are evaluated (the self-test counts both and
 * asserts the grid strictly wins). Clusters are grown by the textbook queue expansion in index
 * order, which fixes the processing order of ambiguous border points, so the two engines must
 * produce element-wise identical labels (noise = -1); the comparison is also made
 * order-insensitively (sets of point-index sets per cluster plus the noise set). Test sets are
 * LCG-generated 2-D integer points, including duplicate-heavy sets, a mostly-noise sparse set,
 * a hand-checked chain, and points exactly eps apart (the <= boundary is inclusive).
 */
import java.util.*;

abstract class RegionEngine {
    final double[][] points;
    final double eps2;
    final double eps;
    int checks;

    RegionEngine(double[][] points, double eps) {
        this.points = points;
        this.eps2 = eps * eps;
        this.eps = eps;
    }

    abstract int[] region(int index);

    static int[] toArray(List<Integer> list) {
        int[] out = new int[list.size()];
        for (int i = 0; i < out.length; i++) out[i] = list.get(i);
        return out;
    }
}

class NaiveEngine extends RegionEngine {
    NaiveEngine(double[][] points, double eps) {
        super(points, eps);
    }

    int[] region(int index) {
        List<Integer> out = new ArrayList<>();
        double px = points[index][0];
        double py = points[index][1];
        for (int j = 0; j < points.length; j++) {
            checks++;
            double dx = px - points[j][0];
            double dy = py - points[j][1];
            if (dx * dx + dy * dy <= eps2) out.add(j);
        }
        return toArray(out);
    }
}

class GridEngine extends RegionEngine {
    final Map<Long, List<Integer>> cells = new HashMap<>();

    GridEngine(double[][] points, double eps) {
        super(points, eps);
        for (int j = 0; j < points.length; j++) {
            long key = key((int) Math.floor(points[j][0] / eps), (int) Math.floor(points[j][1] / eps));
            cells.computeIfAbsent(key, k -> new ArrayList<>()).add(j);
        }
    }

    private static long key(int gx, int gy) {
        return ((long) (gx + 2048) << 13) | (gy + 2048);
    }

    int[] region(int index) {
        double px = points[index][0];
        double py = points[index][1];
        int cx = (int) Math.floor(px / eps);
        int cy = (int) Math.floor(py / eps);
        List<Integer> out = new ArrayList<>();
        for (int gx = cx - 1; gx <= cx + 1; gx++) {
            for (int gy = cy - 1; gy <= cy + 1; gy++) {
                List<Integer> bucket = cells.get(key(gx, gy));
                if (bucket == null) continue;
                for (int j : bucket) {
                    checks++;
                    double dx = px - points[j][0];
                    double dy = py - points[j][1];
                    if (dx * dx + dy * dy <= eps2) out.add(j);
                }
            }
        }
        int[] result = toArray(out);
        Arrays.sort(result);
        return result;
    }
}

public class Solution {
    static final int NOISE = -1;

    static double[][] lcgPoints(int n, int seed, int limit) {
        double[][] points = new double[n][2];
        for (int i = 0; i < n; i++) {
            seed = (seed * 75 + 74) % 65537;
            points[i][0] = seed % limit;
            seed = (seed * 75 + 74) % 65537;
            points[i][1] = seed % limit;
        }
        return points;
    }

    static int[] dbscan(double[][] points, int minPts, RegionEngine engine) {
        int n = points.length;
        int[] labels = new int[n];
        Arrays.fill(labels, NOISE);
        boolean[] visited = new boolean[n];
        int cluster = 0;
        for (int start = 0; start < n; start++) {
            if (visited[start]) continue;
            visited[start] = true;
            int[] neighbours = engine.region(start);
            if (neighbours.length < minPts) continue;   // provisional noise; may become border
            labels[start] = cluster;
            List<Integer> queue = new ArrayList<>();
            for (int q : neighbours) {
                if (q != start) queue.add(q);
            }
            for (int head = 0; head < queue.size(); head++) {
                int current = queue.get(head);
                if (!visited[current]) {
                    visited[current] = true;
                    int[] currentNeighbours = engine.region(current);
                    if (currentNeighbours.length >= minPts) {
                        for (int q : currentNeighbours) {
                            if (!visited[q]) queue.add(q);
                        }
                    }
                }
                if (labels[current] == NOISE) labels[current] = cluster;   // border point
            }
            cluster += 1;
        }
        return labels;
    }

    /** Renumber clusters by first appearance so two runs compare label-for-label. */
    static int[] canonical(int[] labels) {
        Map<Integer, Integer> mapping = new HashMap<>();
        int[] out = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] == NOISE) {
                out[i] = NOISE;
            } else {
                mapping.putIfAbsent(labels[i], mapping.size());
                out[i] = mapping.get(labels[i]);
            }
        }
        return out;
    }

    /** Order-insensitive partition: per-cluster index sets (sorted keys) plus the noise set. */
    static String partition(int[] labels) {
        Map<Integer, List<Integer>> clusters = new TreeMap<>();
        List<Integer> noise = new ArrayList<>();
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] == NOISE) {
                noise.add(i);
            } else {
                clusters.computeIfAbsent(labels[i], k -> new ArrayList<>()).add(i);
            }
        }
        List<String> parts = new ArrayList<>();
        for (List<Integer> members : clusters.values()) {
            StringBuilder builder = new StringBuilder();
            for (int member : members) builder.append(member).append(" ");
            parts.add(builder.toString());
        }
        Collections.sort(parts);
        StringBuilder builder = new StringBuilder();
        for (String part : parts) builder.append(part).append("|");
        builder.append("#");
        for (int member : noise) builder.append(member).append(" ");
        return builder.toString();
    }

    static class Report {
        final int[] labels;
        final int naiveChecks;
        final int gridChecks;

        Report(int[] labels, int naiveChecks, int gridChecks) {
            this.labels = labels;
            this.naiveChecks = naiveChecks;
            this.gridChecks = gridChecks;
        }
    }

    static Report compareSet(double[][] points, double eps, int minPts, String label) {
        NaiveEngine naive = new NaiveEngine(points, eps);
        GridEngine grid = new GridEngine(points, eps);
        int[] naiveLabels = dbscan(points, minPts, naive);
        int[] gridLabels = dbscan(points, minPts, grid);
        check(Arrays.equals(canonical(naiveLabels), canonical(gridLabels)),
                label + ": engines disagree label-for-label");
        check(partition(naiveLabels).equals(partition(gridLabels)), label + ": partitions differ");
        check(grid.checks < naive.checks, label + ": grid must evaluate fewer distances");
        return new Report(naiveLabels, naive.checks, grid.checks);
    }

    public static void main(String[] args) {
        double[][] chain = {{0.0, 0.0}, {1.0, 0.0}, {2.0, 0.0}, {10.0, 10.0}};
        check(Arrays.equals(dbscan(chain, 2, new NaiveEngine(chain, 1.2)),
                new int[]{0, 0, 0, NOISE}), "chain at minPts = 2");
        check(Arrays.equals(dbscan(chain, 3, new NaiveEngine(chain, 1.2)),
                new int[]{0, 0, 0, NOISE}), "all three points are core at minPts = 3");

        double[][] pair = {{0.0, 0.0}, {2.0, 0.0}, {20.0, 20.0}};
        check(Arrays.equals(dbscan(pair, 2, new NaiveEngine(pair, 2.0)),
                new int[]{0, 0, NOISE}), "exact eps neighbours");
        check(Arrays.equals(dbscan(pair, 2, new NaiveEngine(pair, 1.999)),
                new int[]{NOISE, NOISE, NOISE}), "just inside eps everything is noise");

        Object[][] sets = {
                {"dense", lcgPoints(60, 11, 40), 3.0, 3},
                {"fine", lcgPoints(80, 23, 40), 2.0, 2},
                {"coarse", lcgPoints(50, 37, 40), 5.0, 4},
                {"sparse", lcgPoints(40, 53, 100), 1.5, 5},
                {"duplicates", lcgPoints(30, 67, 10), 2.0, 3},
        };
        String[] names = new String[sets.length];
        int[] naiveChecksArr = new int[sets.length];
        int[] gridChecksArr = new int[sets.length];
        int[] clusterCounts = new int[sets.length];
        int[] noiseCounts = new int[sets.length];
        int[] sizes = new int[sets.length];
        for (int s = 0; s < sets.length; s++) {
            String name = (String) sets[s][0];
            double[][] points = (double[][]) sets[s][1];
            double eps = (Double) sets[s][2];
            int minPts = (Integer) sets[s][3];
            Report report = compareSet(points, eps, minPts, name);
            names[s] = name;
            sizes[s] = points.length;
            naiveChecksArr[s] = report.naiveChecks;
            gridChecksArr[s] = report.gridChecks;
            Set<Integer> clusters = new TreeSet<>();
            int noise = 0;
            for (int label : report.labels) {
                if (label == NOISE) {
                    noise++;
                } else {
                    clusters.add(label);
                }
            }
            clusterCounts[s] = clusters.size();
            noiseCounts[s] = noise;
        }

        check(clusterCounts[0] >= 1 && clusterCounts[0] <= 12, "dense set: a handful of clusters");
        check(noiseCounts[0] >= 0 && noiseCounts[0] < sizes[0], "dense set: not all noise");
        for (int s = 0; s < sets.length; s++) {
            check(gridChecksArr[s] <= naiveChecksArr[s] / 2,
                    names[s] + ": the grid saves at least half");
        }

        double[][] lonely = lcgPoints(12, 5, 1000);
        int[] lonelyLabels = dbscan(lonely, 3, new GridEngine(lonely, 1.0));
        for (int label : lonelyLabels) check(label == NOISE, "scattered points are all noise");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
