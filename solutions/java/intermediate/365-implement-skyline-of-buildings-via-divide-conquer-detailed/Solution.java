/** Task 365 - Skyline of buildings via divide & conquer (detailed). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
import java.util.*;

public class Solution {
    private static int seed = 8113;

    /** Returns critical points [x, height]; degenerate buildings (zero width/height) ignored. */
    static int[][] skylineDivideConquer(int[][] buildings) {
        List<int[]> ordered = new ArrayList<>();
        for (int[] b : buildings) {
            if (b[1] > b[0] && b[2] > 0) ordered.add(b.clone());
        }
        ordered.sort(Comparator.<int[]>comparingInt(b -> b[0])
                .thenComparingInt(b -> b[1]).thenComparingInt(b -> b[2]));
        return solve(ordered.toArray(new int[0][]));
    }

    private static int[][] solve(int[][] ordered) {
        if (ordered.length == 0) return new int[0][];
        if (ordered.length == 1) {
            return new int[][]{{ordered[0][0], ordered[0][2]}, {ordered[0][1], 0}};
        }
        int mid = ordered.length / 2;
        int[][] left = solve(Arrays.copyOfRange(ordered, 0, mid));
        int[][] right = solve(Arrays.copyOfRange(ordered, mid, ordered.length));
        return merge(left, right);
    }

    /** Linear two-pointer merge of two x-sorted skylines of [x, height] critical points. */
    private static int[][] merge(int[][] skyA, int[][] skyB) {
        List<int[]> merged = new ArrayList<>();
        int i = 0, j = 0;
        int heightA = 0, heightB = 0;
        while (i < skyA.length || j < skyB.length) {
            int x;
            if (j >= skyB.length || (i < skyA.length && skyA[i][0] < skyB[j][0])) {
                x = skyA[i][0];
                heightA = skyA[i][1];
                i++;
            } else if (i >= skyA.length || skyB[j][0] < skyA[i][0]) {
                x = skyB[j][0];
                heightB = skyB[j][1];
                j++;
            } else {
                x = skyA[i][0];
                heightA = skyA[i][1];
                heightB = skyB[j][1];
                i++;
                j++;
            }
            int height = Math.max(heightA, heightB);
            if (merged.isEmpty() || merged.get(merged.size() - 1)[1] != height) {
                merged.add(new int[]{x, height});
            }
        }
        return merged.toArray(new int[0][]);
    }

    /** Independent oracle: sweep every boundary x keeping counts of active heights on [l, r). */
    static int[][] skylineSweep(int[][] buildings) {
        TreeMap<Integer, List<int[]>> events = new TreeMap<>();
        for (int[] b : buildings) {
            if (b[1] <= b[0] || b[2] <= 0) continue;
            events.computeIfAbsent(b[0], k -> new ArrayList<>()).add(new int[]{b[2], 1});
            events.computeIfAbsent(b[1], k -> new ArrayList<>()).add(new int[]{b[2], -1});
        }
        Map<Integer, Integer> counts = new HashMap<>();
        List<int[]> result = new ArrayList<>();
        for (Map.Entry<Integer, List<int[]>> entry : events.entrySet()) {
            for (int[] event : entry.getValue()) {
                counts.merge(event[0], event[1], Integer::sum);
            }
            int current = 0;
            for (Map.Entry<Integer, Integer> e : counts.entrySet()) {
                if (e.getValue() > 0) current = Math.max(current, e.getKey());
            }
            if (result.isEmpty() || result.get(result.size() - 1)[1] != current) {
                result.add(new int[]{entry.getKey(), current});
            }
        }
        return result.toArray(new int[0][]);
    }

    private static boolean sameSkyline(int[][] a, int[][] b) {
        if (a.length != b.length) return false;
        for (int i = 0; i < a.length; i++) {
            if (a[i][0] != b[i][0] || a[i][1] != b[i][1]) return false;
        }
        return true;
    }

    private static String show(int[][] skyline) {
        StringBuilder builder = new StringBuilder();
        for (int[] p : skyline) builder.append(Arrays.toString(p));
        return builder.toString();
    }

    public static void main(String[] args) {
        // Canonical example: at x=7 the 15-block ends (12 remains); at x=9 the 10-block
        // ends but 12 still stands, so no point is emitted there.
        int[][] canonical = {{2, 9, 10}, {3, 7, 15}, {5, 12, 12}, {15, 20, 10}, {19, 24, 8}};
        int[][] canonicalSkyline = {{2, 10}, {3, 15}, {7, 12}, {12, 0}, {15, 10}, {20, 8}, {24, 0}};
        check(sameSkyline(skylineDivideConquer(canonical), canonicalSkyline), "canonical D&C");
        check(sameSkyline(skylineSweep(canonical), canonicalSkyline), "canonical sweep");

        check(skylineDivideConquer(new int[0][]).length == 0, "empty D&C");
        check(skylineSweep(new int[0][]).length == 0, "empty sweep");
        check(sameSkyline(skylineDivideConquer(new int[][]{{1, 3, 4}}),
                new int[][]{{1, 4}, {3, 0}}), "single building");
        check(sameSkyline(skylineDivideConquer(new int[][]{{0, 2, 5}, {2, 4, 7}}),
                new int[][]{{0, 5}, {2, 7}, {4, 0}}), "touching buildings");
        check(sameSkyline(skylineDivideConquer(new int[][]{{0, 2, 5}, {5, 8, 3}}),
                new int[][]{{0, 5}, {2, 0}, {5, 3}, {8, 0}}), "gap between buildings");
        check(sameSkyline(skylineDivideConquer(new int[][]{{0, 10, 5}, {2, 4, 3}}),
                new int[][]{{0, 5}, {10, 0}}), "nested shorter building");
        check(sameSkyline(skylineDivideConquer(new int[][]{{0, 10, 5}, {2, 4, 9}}),
                new int[][]{{0, 5}, {2, 9}, {4, 5}, {10, 0}}), "nested taller building");
        check(sameSkyline(skylineDivideConquer(new int[][]{{0, 4, 3}, {0, 4, 5}}),
                new int[][]{{0, 5}, {4, 0}}), "same span, taller wins");
        check(sameSkyline(skylineDivideConquer(new int[][]{{0, 4, 5}, {0, 4, 5}, {1, 2, 1}}),
                new int[][]{{0, 5}, {4, 0}}), "duplicate buildings");

        // Degenerate buildings are ignored by both implementations.
        check(skylineDivideConquer(new int[][]{{3, 3, 9}}).length == 0, "zero width ignored");
        check(skylineDivideConquer(new int[][]{{0, 5, 0}}).length == 0, "zero height ignored");
        check(sameSkyline(skylineDivideConquer(new int[][]{{0, 5, 4}, {2, 2, 9}}),
                new int[][]{{0, 4}, {5, 0}}), "degenerate among valid");
        check(sameSkyline(skylineSweep(new int[][]{{0, 5, 4}, {2, 2, 9}}),
                new int[][]{{0, 4}, {5, 0}}), "degenerate among valid sweep");

        // LCG cross-check with many building sets; both algorithms must agree exactly.
        for (int trial = 0; trial < 400; trial++) {
            seed = (seed * 75 + 74) % 65537;
            int count = 1 + seed % 10;
            int[][] buildings = new int[count][];
            for (int i = 0; i < count; i++) {
                seed = (seed * 75 + 74) % 65537;
                int left = seed % 50;
                seed = (seed * 75 + 74) % 65537;
                int width = 1 + seed % 12;
                seed = (seed * 75 + 74) % 65537;
                int height = 1 + seed % 15;
                buildings[i] = new int[]{left, left + width, height};
            }
            int[][] expected = skylineSweep(buildings);
            check(sameSkyline(skylineDivideConquer(buildings), expected),
                    "random trial " + trial + " expected " + show(expected));
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
