/** Task 328 - Skyline problem (heap + sweep line). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
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
    /** Sweep all critical x values; a max-heap keyed by (-height, right) gives the current height. */
    static List<int[]> skylineHeap(int[][] buildings) {
        if (buildings.length == 0) return new ArrayList<>();
        TreeMap<Integer, List<int[]>> starts = new TreeMap<>(); // left -> list of {right, height}
        TreeSet<Integer> xs = new TreeSet<>();
        for (int[] building : buildings) {
            starts.computeIfAbsent(building[0], k -> new ArrayList<>()).add(new int[]{building[1], building[2]});
            xs.add(building[0]);
            xs.add(building[1]);
        }
        PriorityQueue<int[]> heap = new PriorityQueue<>(
                (u, v) -> u[0] != v[0] ? Integer.compare(u[0], v[0]) : Integer.compare(u[1], v[1]));
        List<int[]> result = new ArrayList<>();
        for (int x : xs) {
            for (int[] start : starts.getOrDefault(x, Collections.emptyList())) {
                heap.add(new int[]{-start[1], start[0]});
            }
            while (!heap.isEmpty() && heap.peek()[1] <= x) heap.poll(); // lazy deletion
            int height = heap.isEmpty() ? 0 : -heap.peek()[0];
            if (result.isEmpty() || result.get(result.size() - 1)[1] != height) {
                result.add(new int[]{x, height});
            }
        }
        return result;
    }

    /** Cross-check implementation: recursive split with a linear skyline merge. */
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

    static int[][] lcgBuildings(int count, Lcg lcg) {
        int[][] buildings = new int[count][];
        for (int i = 0; i < count; i++) {
            int left = lcg.next(40);
            buildings[i] = new int[]{left, left + 1 + lcg.next(10), 1 + lcg.next(20)};
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
        check(Arrays.deepEquals(skylineHeap(CANONICAL).toArray(new int[0][]), CANONICAL_SKYLINE),
                "canonical heap skyline");
        check(Arrays.deepEquals(skylineDivideConquer(CANONICAL).toArray(new int[0][]), CANONICAL_SKYLINE),
                "canonical divide and conquer skyline");

        check(skylineHeap(new int[0][]).isEmpty() && skylineDivideConquer(new int[0][]).isEmpty(), "empty input");
        check(Arrays.deepEquals(skylineHeap(new int[][]{{0, 2, 5}, {2, 4, 7}}).toArray(new int[0][]),
                new int[][]{{0, 5}, {2, 7}, {4, 0}}), "touching buildings");
        check(Arrays.deepEquals(skylineHeap(new int[][]{{0, 2, 5}, {5, 8, 3}}).toArray(new int[0][]),
                new int[][]{{0, 5}, {2, 0}, {5, 3}, {8, 0}}), "gap produces zero stretch");

        Lcg lcg = new Lcg(24601);
        for (int trial = 0; trial < 300; trial++) {
            int count = 1 + lcg.next(8);
            int[][] buildings = lcgBuildings(count, lcg);
            check(samePoints(skylineHeap(buildings), skylineDivideConquer(buildings)),
                    "heap agrees with divide and conquer " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
