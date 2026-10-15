/** Task 394 - Implement skyline merging with a multiset (detailed). (Tier: Intermediate, CTCI Beyond — Graphics, Geometry & Robotics) */
import java.util.*;

public class Solution {
    /** events: encoded as int[]{x, delta}; returns skyline points [[x, h], ...]. */
    static List<int[]> sweep(List<int[]> events) {
        TreeMap<Integer, Integer> byX = new TreeMap<>();
        for (int[] event : events) byX.putIfAbsent(event[0], 0);
        Map<Integer, List<Integer>> deltas = new TreeMap<>();
        for (int[] event : events) {
            deltas.computeIfAbsent(event[0], k -> new ArrayList<>()).add(event[1]);
        }
        Map<Integer, Integer> counts = new TreeMap<>();
        List<int[]> points = new ArrayList<>();
        int current = 0;
        for (int x : byX.keySet()) {
            for (int delta : deltas.getOrDefault(x, Collections.emptyList())) {
                if (delta >= 0) {
                    counts.merge(delta, 1, Integer::sum);
                } else {
                    int height = -delta;
                    int remaining = counts.get(height) - 1;
                    if (remaining == 0) counts.remove(height);
                    else counts.put(height, remaining);
                }
            }
            int top = 0;
            if (!counts.isEmpty()) top = ((TreeMap<Integer, Integer>) counts).lastKey();
            if (top != current) {
                points.add(new int[]{x, top});
                current = top;
            }
        }
        return points;
    }

    static List<int[]> skylineOfRectangles(int[][] rectangles) {
        List<int[]> events = new ArrayList<>();
        for (int[] rectangle : rectangles) {
            int x1 = rectangle[0], x2 = rectangle[1], height = rectangle[2];
            if (x1 < x2 && height > 0) {
                events.add(new int[]{x1, height});
                events.add(new int[]{x2, -height});
            }
        }
        return sweep(events);
    }

    static List<int[]> mergeSkylines(List<List<int[]>> skylines) {
        List<int[]> events = new ArrayList<>();
        for (List<int[]> skyline : skylines) {
            for (int i = 0; i + 1 < skyline.size(); i++) {
                int x1 = skyline.get(i)[0], height = skyline.get(i)[1];
                int x2 = skyline.get(i + 1)[0];
                if (height > 0 && x1 < x2) {
                    events.add(new int[]{x1, height});
                    events.add(new int[]{x2, -height});
                }
            }
        }
        return sweep(events);
    }

    private static int seed = 777;

    private static int[][] lcgRectangles(int count, int span) {
        int[][] rectangles = new int[count][];
        for (int k = 0; k < count; k++) {
            seed = (seed * 75 + 74) % 65537;
            int x1 = seed % span;
            seed = (seed * 75 + 74) % 65537;
            int length = 1 + seed % 6;
            seed = (seed * 75 + 74) % 65537;
            int height = 1 + seed % 10;
            rectangles[k] = new int[]{x1, x1 + length, height};
        }
        return rectangles;
    }

    static String join(List<int[]> points) {
        StringBuilder sb = new StringBuilder();
        for (int[] point : points) {
            if (sb.length() > 0) sb.append(';');
            sb.append(point[0]).append(',').append(point[1]);
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        int[][] canonical = {{2,9,10},{3,7,15},{5,12,12},{15,20,10},{19,24,8}};
        String expected = "2,10;3,15;7,12;12,0;15,10;20,8;24,0";
        check(join(skylineOfRectangles(canonical)).equals(expected), "canonical skyline");

        List<int[]> left = Arrays.asList(new int[]{0,3}, new int[]{4,0});
        List<int[]> right = Arrays.asList(new int[]{1,1}, new int[]{2,4}, new int[]{5,0});
        check(join(mergeSkylines(Arrays.asList(left, right))).equals("0,3;2,4;5,0"), "merge two skylines");

        List<int[]> expectedPoints = skylineOfRectangles(canonical);
        check(join(mergeSkylines(Arrays.asList(expectedPoints, expectedPoints))).equals(expected), "idempotent merge");

        check(join(skylineOfRectangles(new int[][]{{0,2,3},{2,4,3}})).equals("0,3;4,0"), "touching rectangles");
        check(join(skylineOfRectangles(new int[][]{{0,10,5},{2,5,3}})).equals("0,5;10,0"), "nested rectangles");
        check(skylineOfRectangles(new int[][]{}).isEmpty(), "no rectangles");

        for (int trial = 0; trial < 40; trial++) {
            int[][] rectangles = lcgRectangles(9, 20);
            seed = (seed * 75 + 74) % 65537;
            int cut = seed % (rectangles.length + 1);
            int[][] first = Arrays.copyOfRange(rectangles, 0, cut);
            int[][] second = Arrays.copyOfRange(rectangles, cut, rectangles.length);
            String combined = join(skylineOfRectangles(rectangles));
            String merged = join(mergeSkylines(Arrays.asList(skylineOfRectangles(first), skylineOfRectangles(second))));
            check(merged.equals(combined), "split merge " + trial);
        }

        check(skylineOfRectangles(new int[][]{{3,1,5}}).isEmpty(), "ignored backwards rectangle");
        check(skylineOfRectangles(new int[][]{{0,5,0}}).isEmpty(), "ignored zero height");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
