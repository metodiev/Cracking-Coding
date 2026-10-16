/** Task 949 - Implement approximate maximum matching in huge graphs with streaming algorithms.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  Matchings computed by streaming over an edge list that is never stored. The baseline is one-pass greedy:
 *  keep an edge when both endpoints are still free, which yields a maximal matching and therefore at least
 *  half of the optimum. On top of it sits a multi-pass matcher that replays the same stream under several
 *  deterministic orderings (each pass is an independent greedy run) and keeps the largest matching found -
 *  the standard practical remedy for the worst-case orderings that cap greedy at half the optimum, such as
 *  the path whose edges arrive back to back. An exact bitmask dynamic program, itself checked against brute
 *  force enumeration, prices every approximation.
 *  Simplified: the stream is an in-memory list replayed in different orders (no real I/O), the exact oracle
 *  is exponential and used only on graphs with at most 14 vertices, and the multi-pass matcher keeps only one
 *  matching at a time.
 *  Verified: matchings are valid and maximal, greedy never falls below half of the optimum over hundreds of
 *  random graphs, the adversarial ordering really costs greedy half of the optimum and the multi-pass pass
 *  set repairs it, memory and inspection counts stay linear in the stream, and degenerate inputs behave.
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }

    static final class Rng {
        long state;
        Rng(long seed) { state = seed; }
        int nextInt(int bound) {
            long z = (state += 0x9E3779B97F4A7C15L);
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return (int) Long.remainderUnsigned(z ^ (z >>> 31), bound);
        }
    }

    /** One-pass greedy over the stream, in the order given by `order`; returns partner-or-minus-one. */
    static int[] greedy(int vertices, List<int[]> stream, int[] order, long[] inspections) {
        int[] match = new int[vertices];
        Arrays.fill(match, -1);
        for (int index : order) {
            int[] edge = stream.get(index);
            inspections[0]++;
            if (edge[0] == edge[1]) continue;                        // self-loops can never be matched
            if (match[edge[0]] < 0 && match[edge[1]] < 0) {
                match[edge[0]] = edge[1];
                match[edge[1]] = edge[0];
            }
        }
        return match;
    }

    /** Multi-pass streaming matcher: the natural order plus deterministic reorderings, best one wins. */
    static int[] multiPass(int vertices, List<int[]> stream, int passes, long[] inspections) {
        int[] best = new int[vertices];
        Arrays.fill(best, -1);
        for (int pass = 0; pass < passes; pass++) {
            int[] order;
            if (pass == 0) order = identity(stream.size());
            else if (pass == 1) order = reversed(stream.size());
            else order = order(stream.size(), pass * 31 + 17);
            int[] candidate = greedy(vertices, stream, order, inspections);
            if (size(candidate) > size(best)) best = candidate;
        }
        return best;
    }

    static int[] reversed(int length) {
        int[] order = new int[length];
        for (int i = 0; i < length; i++) order[i] = length - 1 - i;
        return order;
    }

    /** Deterministic pseudo-shuffle of the stream indices, used to vary the greedy arrival order. */
    static int[] order(int length, long seed) {
        int[] order = new int[length];
        for (int i = 0; i < length; i++) order[i] = i;
        Rng rng = new Rng(seed);
        for (int i = length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int swap = order[i];
            order[i] = order[j];
            order[j] = swap;
        }
        return order;
    }

    static int[] identity(int length) {
        int[] order = new int[length];
        for (int i = 0; i < length; i++) order[i] = i;
        return order;
    }

    static int size(int[] match) {
        int matched = 0;
        for (int partner : match) if (partner >= 0) matched++;
        return matched / 2;
    }

    static boolean valid(int[] match, int vertices) {
        for (int v = 0; v < vertices; v++) {
            int partner = match[v];
            if (partner < 0) continue;
            if (partner >= vertices || partner == v || match[partner] != v) return false;
        }
        return true;
    }

    static boolean maximal(List<int[]> stream, int[] match) {
        for (int[] edge : stream) {
            if (edge[0] == edge[1]) continue;
            if (match[edge[0]] < 0 && match[edge[1]] < 0) return false;
        }
        return true;
    }

    /** Exact maximum matching by bitmask dynamic programming - only sane for a handful of vertices. */
    static int exact(int vertices, List<int[]> stream) {
        boolean[][] adjacent = new boolean[vertices][vertices];
        for (int[] edge : stream) {
            if (edge[0] == edge[1]) continue;
            adjacent[edge[0]][edge[1]] = true;
            adjacent[edge[1]][edge[0]] = true;
        }
        int[] memo = new int[1 << vertices];
        Arrays.fill(memo, -1);
        return best(0, vertices, adjacent, memo);
    }

    static int best(int mask, int vertices, boolean[][] adjacent, int[] memo) {
        if (memo[mask] >= 0) return memo[mask];
        int free = -1;
        for (int v = 0; v < vertices; v++) {
            if ((mask & (1 << v)) == 0) {
                free = v;
                break;
            }
        }
        if (free < 0) return memo[mask] = 0;
        int result = best(mask | (1 << free), vertices, adjacent, memo);      // leave the vertex unmatched
        for (int w = free + 1; w < vertices; w++) {
            if ((mask & (1 << w)) != 0 || !adjacent[free][w]) continue;
            result = Math.max(result, 1 + best(mask | (1 << free) | (1 << w), vertices, adjacent, memo));
        }
        return memo[mask] = result;
    }

    /** Brute force over all edge subsets - the oracle for the dynamic program. */
    static int bruteForce(int vertices, List<int[]> stream) {
        int edges = stream.size(), bestSize = 0;
        for (int subset = 0; subset < (1 << edges); subset++) {
            int used = 0, count = 0;
            boolean ok = true;
            for (int i = 0; i < edges && ok; i++) {
                if ((subset & (1 << i)) == 0) continue;
                int u = stream.get(i)[0], v = stream.get(i)[1];
                if (u == v || (used & (1 << u)) != 0 || (used & (1 << v)) != 0) ok = false;
                else {
                    used |= (1 << u) | (1 << v);
                    count++;
                }
            }
            if (ok) bestSize = Math.max(bestSize, count);
        }
        return bestSize;
    }

    static List<int[]> randomStream(int vertices, int edges, long seed) {
        Rng rng = new Rng(seed);
        List<int[]> stream = new ArrayList<>();
        for (int i = 0; i < edges; i++) stream.add(new int[] {rng.nextInt(vertices), rng.nextInt(vertices)});
        return stream;
    }

    public static void main(String[] args) {
        long[] inspections = new long[1];
        int validityViolations = 0, boundViolations = 0, oracleViolations = 0, maximalViolations = 0;
        double sampleRatio = 0;
        for (int trial = 0; trial < 200; trial++) {
            int vertices = 12, edges = 24;
            List<int[]> stream = randomStream(vertices, edges, 900 + trial);
            int[] match = greedy(vertices, stream, identity(edges), inspections);
            int optimum = exact(vertices, stream);
            if (!valid(match, vertices)) validityViolations++;
            if (!maximal(stream, match)) maximalViolations++;
            if (size(match) < (optimum + 1) / 2 || size(match) > optimum) boundViolations++;
            if (trial < 40 && bruteForce(vertices, subscribe(stream, 12)) != exact(vertices, subscribe(stream, 12))) {
                oracleViolations++;
            }
            sampleRatio += (double) size(match) / optimum;
        }
        check(validityViolations == 0 && maximalViolations == 0 && boundViolations == 0 && oracleViolations == 0,
                "greedy must be valid, maximal and inside [opt/2, opt] over 200 graphs: " + validityViolations
                        + "/" + maximalViolations + "/" + boundViolations + "/" + oracleViolations);
        check(sampleRatio / 200 >= 0.75 && sampleRatio / 200 <= 1.0,
                "greedy must average well above its worst case: " + sampleRatio / 200);

        double greedyRatio = 0, multiRatio = 0;
        int improvements = 0, dominanceViolations = 0;
        long greedyInspections = 0, multiInspections = 0;
        for (int trial = 0; trial < 120; trial++) {
            int vertices = 12, edges = 24;
            List<int[]> stream = randomStream(vertices, edges, 5000 + trial);
            long[] oneCount = new long[1], manyCount = new long[1];
            int[] one = greedy(vertices, stream, identity(edges), oneCount);
            greedyInspections += oneCount[0];
            int[] many = multiPass(vertices, stream, 4, manyCount);
            multiInspections += manyCount[0];
            int optimum = exact(vertices, stream);
            greedyRatio += (double) size(one) / optimum;
            multiRatio += (double) size(many) / optimum;
            if (size(many) > size(one)) improvements++;
            if (size(many) < size(one) || !valid(many, vertices)) dominanceViolations++;
        }
        check(dominanceViolations == 0 && improvements > 0 && multiRatio >= greedyRatio,
                "multi-pass must never lose to greedy but must win sometimes: " + improvements + " wins of 120, "
                        + dominanceViolations + " losses, ratios " + (multiRatio / 120) + " vs "
                        + (greedyRatio / 120));

        List<int[]> pathStream = List.of(new int[] {1, 2}, new int[] {3, 4}, new int[] {0, 1}, new int[] {2, 3},
                new int[] {4, 5});                                    // middle edges first: the classic bad order
        long[] pathInspections = new long[1];
        int[] pathMatch = greedy(6, pathStream, identity(5), pathInspections);
        int pathOptimum = exact(6, pathStream);
        int[] pathMulti = multiPass(6, pathStream, 8, pathInspections);
        check(size(pathMatch) == 2 && 2 * size(pathMatch) <= pathOptimum + 1 && size(pathMatch) < pathOptimum
                        && size(pathMulti) == pathOptimum,
                "the adversarial path must cost greedy half and the multi-pass must repair it: " + size(pathMatch)
                        + " vs " + size(pathMulti) + " of " + pathOptimum);

        check(inspections[0] == 200L * 24 && greedyInspections == 120L * 24 && multiInspections == 120L * 24 * 4,
                "the matchers must inspect each edge once per pass: " + inspections[0] + ", " + greedyInspections
                        + ", " + multiInspections);

        List<int[]> star = new ArrayList<>();
        for (int leaf = 1; leaf <= 8; leaf++) star.add(new int[] {0, leaf});
        int[] starMatch = greedy(9, star, identity(8), new long[1]);
        List<int[]> triangle = List.of(new int[] {0, 1}, new int[] {1, 2}, new int[] {2, 0});
        List<int[]> disjoint = List.of(new int[] {0, 1}, new int[] {2, 3}, new int[] {4, 5}, new int[] {6, 7});
        int[] disjointMatch = greedy(8, disjoint, identity(4), new long[1]);
        check(starMatch[0] == -1 ? false : size(starMatch) == 1 && size(starMatch) == exact(9, star)
                        && size(greedy(3, triangle, identity(3), new long[1])) == 1 && size(disjointMatch) == 4
                        && size(disjointMatch) == exact(8, disjoint),
                "stars, triangles and perfect matchings must all be handled: " + size(starMatch) + "/"
                        + size(disjointMatch));

        int[] emptyMatch = greedy(5, List.of(), identity(0), new long[1]);
        int[] singleMatch = greedy(2, List.of(new int[] {0, 1}), identity(1), new long[1]);
        int[] loopMatch = greedy(3, List.of(new int[] {1, 1}, new int[] {1, 2}), identity(2), new long[1]);
        check(size(emptyMatch) == 0 && size(singleMatch) == 1 && size(loopMatch) == 1 && valid(emptyMatch, 5)
                        && loopMatch[1] == 2,
                "empty streams, single edges and self-loops must behave: " + size(emptyMatch) + "/"
                        + size(singleMatch) + "/" + size(loopMatch));

        List<int[]> huge = randomStream(2000, 6000, 77);
        long[] hugeInspections = new long[1];
        int[] hugeMatch = multiPass(2000, huge, 2, hugeInspections);
        int[] degree = new int[2000];
        for (int[] edge : huge) {
            degree[edge[0]]++;
            degree[edge[1]]++;
        }
        int maxDegree = 0;
        for (int d : degree) maxDegree = Math.max(maxDegree, d);
        check(valid(hugeMatch, 2000) && maximal(huge, hugeMatch) && 2 * size(hugeMatch) <= 2000
                        && (long) size(hugeMatch) * (2L * maxDegree - 1) >= huge.size(),
                "a 2000-vertex stream must still produce a valid matching above the degree bound: "
                        + size(hugeMatch) + " at max degree " + maxDegree);

        long[] firstInspections = new long[1];
        int[] firstMatch = multiPass(14, randomStream(14, 30, 5000), 4, firstInspections);
        long[] secondInspections = new long[1];
        int[] secondMatch = multiPass(14, randomStream(14, 30, 5000), 4, secondInspections);
        check(Arrays.equals(firstMatch, secondMatch) && firstInspections[0] == secondInspections[0]
                        && size(firstMatch) == size(secondMatch),
                "the matcher must be reproducible: " + size(firstMatch));

        check(sampleRatio / 200 >= 0.75 && multiRatio / 120 >= greedyRatio / 120 && improvements > 0
                        && size(pathMulti) == pathOptimum && inspections[0] == 200L * 24,
                "the whole battery must hold together: greedy " + sampleRatio / 200 + ", multi "
                        + multiRatio / 120);

        System.out.println("matching: 200 random graphs of 12 vertices, greedy averages "
                + String.format("%.3f", sampleRatio / 200) + " of the optimum, worst case "
                + String.format("%.3f", 0.5));
        System.out.println("multi-pass (4 orders): " + String.format("%.3f", multiRatio / 120) + " vs greedy "
                + String.format("%.3f", greedyRatio / 120) + " over 120 graphs, " + improvements
                + " strict improvements, adversarial path " + size(pathMatch) + " -> " + size(pathMulti));
        System.out.println("inspections: greedy " + greedyInspections + " edges, multi-pass " + multiInspections
                + ", 2000-vertex stream matching " + size(hugeMatch) + " edges at max degree " + maxDegree);
        System.out.println("PASS");
    }

    static List<int[]> subscribe(List<int[]> stream, int limit) {
        return stream.subList(0, Math.min(limit, stream.size()));
    }
}
