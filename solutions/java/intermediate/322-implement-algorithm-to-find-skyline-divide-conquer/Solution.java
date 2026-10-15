/** Task 322 - Implement algorithm to find skyline (divide & conquer). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
import java.util.*;

/** Deterministic pseudo-random generator: seed = (seed * 75 + 74) % 65537. */
class Lcg {
    private int seed;

    Lcg(int seed) {
        this.seed = seed;
    }

    int next(int modulus) {
        seed = (seed * 75 + 74) % 65537;
        return seed % modulus;
    }
}

public class Solution {
    /** buildings: {left, right, height}; returns critical points {x, height}. */
    static List<int[]> skylineDivideConquer(int[][] buildings) {
        return divide(buildings, 0, buildings.length);
    }

    private static List<int[]> divide(int[][] buildings, int from, int to) {
        if (from >= to) return new ArrayList<>();
        if (to - from == 1) {
            List<int[]> skyline = new ArrayList<>();
            skyline.add(new int[]{buildings[from][0], buildings[from][2]});
            skyline.add(new int[]{buildings[from][1], 0});
            return skyline;
        }
        int mid = (from + to) / 2;
        return mergeSkylines(divide(buildings, from, mid), divide(buildings, mid, to));
    }

    /** Linear merge of two x-sorted skylines of {x, height} critical points. */
    static List<int[]> mergeSkylines(List<int[]> a, List<int[]> b) {
        List<int[]> merged = new ArrayList<>();
        int i = 0, j = 0, heightA = 0, heightB = 0;
        while (i < a.size() || j < b.size()) {
            int x;
            if (j >= b.size() || (i < a.size() && a.get(i)[0] < b.get(j)[0])) {
                x = a.get(i)[0];
                heightA = a.get(i)[1];
                i++;
            } else if (i >= a.size() || b.get(j)[0] < a.get(i)[0]) {
                x = b.get(j)[0];
                heightB = b.get(j)[1];
                j++;
            } else {
                x = a.get(i)[0];
                heightA = a.get(i)[1];
                heightB = b.get(j)[1];
                i++;
                j++;
            }
            int height = Math.max(heightA, heightB);
            if (merged.isEmpty() || merged.get(merged.size() - 1)[1] != height) {
                merged.add(new int[]{x, height});
            }
        }
        return merged;
    }

    /** Independent oracle: sweep all boundaries keeping a count of active heights. */
    static List<int[]> skylineSweep(int[][] buildings) {
        if (buildings.length == 0) return new ArrayList<>();
        int maxHeight = 0;
        TreeMap<Integer, List<Integer>> starts = new TreeMap<>();
        TreeMap<Integer, List<Integer>> ends = new TreeMap<>();
        for (int[] building : buildings) {
            starts.computeIfAbsent(building[0], k -> new ArrayList<>()).add(building[2]);
            ends.computeIfAbsent(building[1], k -> new ArrayList<>()).add(building[2]);
            maxHeight = Math.max(maxHeight, building[2]);
        }
        int[] counts = new int[maxHeight + 1];
        int current = 0;
        List<int[]> result = new ArrayList<>();
        TreeSet<Integer> xs = new TreeSet<>();
        xs.addAll(starts.keySet());
        xs.addAll(ends.keySet());
        for (int x : xs) {
            for (int height : ends.getOrDefault(x, Collections.emptyList())) counts[height]--; // active on [left, right)
            for (int height : starts.getOrDefault(x, Collections.emptyList())) {
                counts[height]++;
                current = Math.max(current, height);
            }
            while (current > 0 && counts[current] == 0) current--;
            if (result.isEmpty() || result.get(result.size() - 1)[1] != current) {
                result.add(new int[]{x, current});
            }
        }
        return result;
    }

    static int[][] lcgBuildings(int count, Lcg lcg) {
        int[][] buildings = new int[count][];
        for (int i = 0; i < count; i++) {
            int left = lcg.next(40);
            int width = 1 + lcg.next(10);
            int height = 1 + lcg.next(20);
            buildings[i] = new int[]{left, left + width, height};
        }
        return buildings;
    }

    static boolean samePoints(List<int[]> a, List<int[]> b) {
        if (a.size() != b.size()) return false;
        for (int i = 0; i < a.size(); i++) {
            if (!Arrays.equals(a.get(i), b.get(i))) return false;
        }
        return true;
    }

    static final int[][] CANONICAL = {
        {2, 9, 10}, {3, 7, 15}, {5, 12, 12}, {15, 20, 10}, {19, 24, 8},
    };
    static final int[][] CANONICAL_SKYLINE = {
        {2, 10}, {3, 15}, {7, 12}, {12, 0}, {15, 10}, {20, 8}, {24, 0},
    };

    public static void main(String[] args) {
        check(Arrays.deepEquals(skylineDivideConquer(CANONICAL).toArray(new int[0][]), CANONICAL_SKYLINE),
                "canonical divide and conquer");
        check(Arrays.deepEquals(skylineSweep(CANONICAL).toArray(new int[0][]), CANONICAL_SKYLINE),
                "canonical sweep");

        check(skylineDivideConquer(new int[0][]).isEmpty(), "empty divide and conquer");
        check(skylineSweep(new int[0][]).isEmpty(), "empty sweep");
        check(Arrays.deepEquals(skylineDivideConquer(new int[][]{{1, 3, 4}}).toArray(new int[0][]),
                new int[][]{{1, 4}, {3, 0}}), "single building");
        check(Arrays.deepEquals(skylineDivideConquer(new int[][]{{0, 2, 5}, {2, 4, 7}}).toArray(new int[0][]),
                new int[][]{{0, 5}, {2, 7}, {4, 0}}), "touching buildings");
        check(Arrays.deepEquals(skylineDivideConquer(new int[][]{{0, 2, 5}, {5, 8, 3}}).toArray(new int[0][]),
                new int[][]{{0, 5}, {2, 0}, {5, 3}, {8, 0}}), "gap produces zero stretch");

        Lcg lcg = new Lcg(1123);
        for (int trial = 0; trial < 300; trial++) {
            int count = 1 + lcg.next(8);
            int[][] buildings = lcgBuildings(count, lcg);
            check(samePoints(skylineDivideConquer(buildings), skylineSweep(buildings)),
                    "random sweep cross-check " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
