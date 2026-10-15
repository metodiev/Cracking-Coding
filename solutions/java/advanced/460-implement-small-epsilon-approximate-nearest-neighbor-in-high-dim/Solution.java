/** Task 460 - Implement small epsilon-approximate nearest neighbor in high-dim (LSH idea). (Tier: Advanced, CTCI Beyond) */
import java.util.*;

public class Solution {
    static final int DIM = 10;
    static final int TABLES = 6;
    static final int BUCKET_WIDTH = 700;
    static final int NUM_POINTS = 40;

    static int seed = 4601;

    static int lcg() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    static int[][] makePoints(int count) {
        int[][] points = new int[count][DIM];
        for (int p = 0; p < count; p++) {
            for (int j = 0; j < DIM; j++) points[p][j] = lcg() % 61 - 30;
        }
        return points;
    }

    static class Table {
        final int[] projection = new int[DIM];
        int offset;
        final Map<Integer, List<Integer>> buckets = new HashMap<>();

        int hashOf(int[] point) {
            int total = offset;
            for (int j = 0; j < DIM; j++) total += projection[j] * point[j];
            return Math.floorDiv(total, BUCKET_WIDTH);
        }
    }

    static Table[] makeTablesFromSeed(int tableSeed) {
        Table[] tables = new Table[TABLES];
        for (int t = 0; t < TABLES; t++) {
            Table table = new Table();
            for (int j = 0; j < DIM; j++) {
                tableSeed = (tableSeed * 75 + 74) % 65537;
                table.projection[j] = 1 + tableSeed % 20;
            }
            tableSeed = (tableSeed * 75 + 74) % 65537;
            table.offset = tableSeed % BUCKET_WIDTH;
            tables[t] = table;
        }
        return tables;
    }

    static Table[] makeTables() {
        int tableSeed = seed;
        for (int i = 0; i < TABLES * (DIM + 1); i++) lcg();
        return makeTablesFromSeed(tableSeed);
    }

    static void buildIndex(int[][] points, Table[] tables) {
        for (Table table : tables) {
            for (int idx = 0; idx < points.length; idx++) {
                table.buckets.computeIfAbsent(table.hashOf(points[idx]), key -> new ArrayList<>()).add(idx);
            }
        }
    }

    static int squaredDistance(int[] p, int[] q) {
        int total = 0;
        for (int j = 0; j < p.length; j++) {
            int diff = p[j] - q[j];
            total += diff * diff;
        }
        return total;
    }

    /** Exact nearest neighbour (the epsilon-oracle / certified scan). Returns {index, squared dist}. */
    static int[] bruteForceNn(int[][] points, int[] query) {
        int bestIdx = 0;
        int bestDist = squaredDistance(points[0], query);
        for (int idx = 1; idx < points.length; idx++) {
            int dist = squaredDistance(points[idx], query);
            if (dist < bestDist) {
                bestIdx = idx;
                bestDist = dist;
            }
        }
        return new int[]{bestIdx, bestDist};
    }

    /** Returns {index, squared dist, usedBucket}; falls back to exact when all buckets empty. */
    static int[] lshApproxNn(int[][] points, Table[] tables, int[] query) {
        TreeSet<Integer> candidates = new TreeSet<>();
        for (Table table : tables) {
            List<Integer> bucket = table.buckets.get(table.hashOf(query));
            if (bucket != null) candidates.addAll(bucket);
        }
        if (candidates.isEmpty()) {
            int[] exact = bruteForceNn(points, query);
            return new int[]{exact[0], exact[1], 0};
        }
        int bestIdx = -1;
        int bestDist = Integer.MAX_VALUE;
        for (int idx : candidates) {
            int dist = squaredDistance(points[idx], query);
            if (dist < bestDist) {
                bestIdx = idx;
                bestDist = dist;
            }
        }
        return new int[]{bestIdx, bestDist, 1};
    }

    static int[] checkQuery(int[][] points, Table[] tables, int[] query, int[] counters, String label) {
        int[] exact = bruteForceNn(points, query);
        int[] approx = lshApproxNn(points, tables, query);
        check(approx[1] >= exact[1], label + " never closer than true NN");
        check(4 * approx[1] <= 9 * exact[1], label + " epsilon bound");
        check(squaredDistance(points[approx[0]], query) == approx[1], label + " distance matches");
        counters[approx[2]]++;
        return new int[]{exact[0], exact[1], approx[0], approx[1], approx[2]};
    }

    public static void main(String[] args) {
        int[][] points = makePoints(NUM_POINTS);
        Table[] tables = makeTables();
        buildIndex(points, tables);

        int bucketCount = 0;
        int maxBucket = 0;
        for (Table table : tables) {
            for (List<Integer> bucket : table.buckets.values()) {
                bucketCount++;
                maxBucket = Math.max(maxBucket, bucket.size());
            }
        }
        check(bucketCount > 0, "buckets exist");
        check(maxBucket >= 2, "collisions exist");

        int[] counters = new int[2];
        for (int q = 0; q < 12; q++) {
            int[] query = new int[DIM];
            for (int j = 0; j < DIM; j++) query[j] = lcg() % 61 - 30;
            checkQuery(points, tables, query, counters, "LCG query " + q);
        }

        int[] outlier = new int[DIM];
        Arrays.fill(outlier, 10000);
        int[] result = checkQuery(points, tables, outlier, counters, "outlier");
        check(result[4] == 0, "outlier falls back");
        check(result[2] == result[0] && result[3] == result[1], "fallback is exact");

        int[][] handPoints = new int[3][DIM];
        handPoints[1][0] = 10;
        handPoints[2][1] = 10;
        Table[] handTables = makeTablesFromSeed(99999);
        buildIndex(handPoints, handTables);
        int[] handQuery = new int[DIM];
        handQuery[0] = 9;
        int[] handExact = bruteForceNn(handPoints, handQuery);
        check(handExact[0] == 1 && handExact[1] == 1, "hand true NN");
        int[] handApprox = lshApproxNn(handPoints, handTables, handQuery);
        check(handApprox[1] >= 1, "hand never closer");
        check(4 * handApprox[1] <= 9 * 1, "hand epsilon bound");
        check(squaredDistance(handPoints[handApprox[0]], handQuery) == handApprox[1], "hand distance matches");

        check(counters[1] >= 1, "bucket path exercised");
        check(counters[0] >= 1, "fallback path exercised");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
