/** Task 536 - Implement constrained shortest path (resource constrained) DP. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)
 *
 * Resource-constrained shortest path: every directed edge (u, v) carries a cost c and a
 * positive resource consumption r, and we want the cheapest s -> t path whose total
 * consumption is at most the budget B. Two independent dynamic programs solve it:
 *
 *   - cost-minimising: dp[r][v] = cheapest cost of reaching v with consumption exactly r,
 *     filled by sweeping the resource layers in increasing order — legal because every edge
 *     consumes at least one unit, so every relaxation moves to a strictly higher layer (the
 *     model deliberately excludes zero-resource edges, which would otherwise need an inner
 *     Dijkstra inside each layer). Packed (r, v) parents allow path reconstruction.
 *   - resource-minimising (dual): res2[c][v] = smallest consumption of a path to v of total
 *     cost exactly c, swept over cost layers; the answer is the smallest c with
 *     res2[c][t] <= B — the same optimum viewed from the other constraint dimension.
 *
 * Because costs and consumptions are positive, an optimal walk can always be taken simple, so
 * an exhaustive DFS over simple s -> t paths (with a budget cut-off) is a valid brute force.
 * Documented checks: both DPs agree with that brute force for every budget 0..20 on
 * LCG-generated graphs (n <= 6, costs and consumptions 1..9) and on the hand-built example
 * where budget 6 admits the 3-hop chain (cost 3) but budget 4 forces the direct edge (cost 10);
 * the reconstructed path is simple and its independently recomputed cost/consumption match the
 * DP; the Pareto frontier of nondominated (cost, consumption) pairs is strictly nondominated
 * and its budget sweep reproduces the optimum for every budget; the optimum is non-increasing
 * in the budget; an oversized budget never helps; and infeasible instances return the INF
 * sentinel from both DPs and the brute force.
 */
import java.util.*;

class RcEdge {
    final int from, to, cost, res;

    RcEdge(int from, int to, int cost, int res) {
        this.from = from;
        this.to = to;
        this.cost = cost;
        this.res = res;
    }
}

public class Solution {
    static final int INF = Integer.MAX_VALUE / 4;

    /** dp[r][v] = cheapest cost to reach v with consumption exactly r. */
    static int[][] costLayers(int n, List<RcEdge> edges, int budget, int source) {
        int[][] dp = new int[budget + 1][n];
        for (int[] row : dp) Arrays.fill(row, INF);
        dp[0][source] = 0;
        for (int r = 0; r <= budget; r++) {
            for (RcEdge e : edges) {
                if (dp[r][e.from] == INF) continue;
                int next = r + e.res;
                if (next > budget) continue;
                int candidate = dp[r][e.from] + e.cost;
                if (candidate < dp[next][e.to]) dp[next][e.to] = candidate;
            }
        }
        return dp;
    }

    /** Same DP with parents packed as r * n + v so an optimal path can be rebuilt. */
    static int[][] parentLayers(int n, List<RcEdge> edges, int budget, int source) {
        int[][] dp = new int[budget + 1][n];
        int[][] parent = new int[budget + 1][n];
        for (int[] row : dp) Arrays.fill(row, INF);
        for (int[] row : parent) Arrays.fill(row, -1);
        dp[0][source] = 0;
        for (int r = 0; r <= budget; r++) {
            for (RcEdge e : edges) {
                if (dp[r][e.from] == INF) continue;
                int next = r + e.res;
                if (next > budget) continue;
                int candidate = dp[r][e.from] + e.cost;
                if (candidate < dp[next][e.to]) {
                    dp[next][e.to] = candidate;
                    parent[next][e.to] = r * n + e.from;
                }
            }
        }
        return parent;
    }

    static int bestCostForBudget(int[][] dp, int budget, int target) {
        int best = INF;
        for (int r = 0; r <= budget; r++) best = Math.min(best, dp[r][target]);
        return best;
    }

    static int usedBudget(int[][] dp, int target) {
        int best = INF;
        int where = -1;
        for (int r = 0; r < dp.length; r++) {
            if (dp[r][target] < best) {
                best = dp[r][target];
                where = r;
            }
        }
        return where;
    }

    /** res2[c][v] = smallest consumption of a path from source to v of total cost exactly c. */
    static int[][] resourceLayers(int n, List<RcEdge> edges, int costBound, int source) {
        int[][] res = new int[costBound + 1][n];
        for (int[] row : res) Arrays.fill(row, INF);
        res[0][source] = 0;
        for (int c = 0; c <= costBound; c++) {
            for (RcEdge e : edges) {
                if (res[c][e.from] == INF) continue;
                int next = c + e.cost;
                if (next > costBound) continue;
                int candidate = res[c][e.from] + e.res;
                if (candidate < res[next][e.to]) res[next][e.to] = candidate;
            }
        }
        return res;
    }

    static int bestCostByResourceLayers(int[][] res, int budget, int target) {
        for (int c = 0; c < res.length; c++) {
            if (res[c][target] <= budget) return c;
        }
        return INF;
    }

    static class Brute {
        int bestCost = INF;
        int simplePaths = 0;
    }

    /** Exhaustive DFS over simple s -> t paths; consumption only grows so the cut-off is safe. */
    static Brute bruteForce(int n, List<RcEdge> edges, int budget, int source, int target) {
        List<List<RcEdge>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++) adj.add(new ArrayList<>());
        for (RcEdge e : edges) adj.get(e.from).add(e);
        Brute brute = new Brute();
        explore(adj, source, target, budget, new boolean[n], 0, 0, brute);
        return brute;
    }

    static void explore(List<List<RcEdge>> adj, int v, int target, int budget, boolean[] onPath,
                        int cost, int res, Brute brute) {
        if (v == target) {
            brute.simplePaths++;
            brute.bestCost = Math.min(brute.bestCost, cost);
            return;
        }
        onPath[v] = true;
        for (RcEdge e : adj.get(v)) {
            if (onPath[e.to] || res + e.res > budget) continue;
            explore(adj, e.to, target, budget, onPath, cost + e.cost, res + e.res, brute);
        }
        onPath[v] = false;
    }

    /** Rebuild a path from the parent table starting at the cheapest layer of the target. */
    static List<Integer> reconstruct(int[][] parent, int n, int r, int target) {
        List<Integer> path = new ArrayList<>();
        int curR = r;
        int curV = target;
        while (curV != -1) {
            path.add(curV);
            int packed = parent[curR][curV];
            if (packed < 0) break;
            curR = packed / n;
            curV = packed % n;
        }
        Collections.reverse(path);
        return path;
    }

    static int[] pathTotals(List<Integer> path, List<RcEdge> edges) {
        int cost = 0;
        int res = 0;
        for (int i = 0; i + 1 < path.size(); i++) {
            int from = path.get(i);
            int to = path.get(i + 1);
            RcEdge best = null;
            for (RcEdge e : edges) {
                if (e.from == from && e.to == to && (best == null || e.cost < best.cost)) best = e;
            }
            if (best == null) throw new RuntimeException("FAILED: path uses a missing edge");
            cost += best.cost;
            res += best.res;
        }
        return new int[]{cost, res};
    }

    static List<RcEdge> lcgEdges(int n, int count, int seed) {
        List<RcEdge> edges = new ArrayList<>();
        Set<Long> seen = new HashSet<>();
        int guard = 0;
        while (edges.size() < count && guard < count * 30) {
            guard++;
            seed = (seed * 75 + 74) % 65537;
            int u = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int v = seed % n;
            seed = (seed * 75 + 74) % 65537;
            int cost = seed % 9 + 1;
            int res = (seed / 7) % 9 + 1;
            if (u == v) continue;
            if (!seen.add(((long) u << 40) | ((long) v << 20) | cost)) continue;
            edges.add(new RcEdge(u, v, cost, res));
        }
        return edges;
    }

    public static void main(String[] args) {
        int n = 4;
        List<RcEdge> hand = Arrays.asList(
                new RcEdge(0, 3, 10, 1),
                new RcEdge(0, 1, 1, 2),
                new RcEdge(1, 2, 1, 2),
                new RcEdge(2, 3, 1, 2),
                new RcEdge(0, 3, 6, 5));
        int budget6 = bestCostForBudget(costLayers(n, hand, 6, 0), 6, 3);
        int budget4 = bestCostForBudget(costLayers(n, hand, 4, 0), 4, 3);
        check(budget6 == 3, "budget 6: the 3-hop chain (cost 3, consumption 6) wins");
        check(budget4 == 10, "budget 4: the chain is infeasible, the direct edge (cost 10) is forced");
        check(budget4 > budget6 && budget4 != INF, "optimum is non-increasing in the budget");
        check(bestCostForBudget(costLayers(n, hand, 20, 0), 20, 3) == budget6,
                "an oversized budget cannot beat the best feasible path");
        check(bruteForce(n, hand, 4, 0, 3).bestCost == 10
                        && bruteForce(n, hand, 6, 0, 3).bestCost == 3,
                "brute force agrees with both hand-checked budgets");
        check(bruteForce(n, hand, 6, 0, 3).simplePaths > 0, "brute force enumerates paths");

        int[][] dp = costLayers(n, hand, 6, 0);
        List<Integer> path = reconstruct(parentLayers(n, hand, 6, 0), n, usedBudget(dp, 3), 3);
        check(path.equals(Arrays.asList(0, 1, 2, 3)), "reconstructed optimal path");
        int[] totals = pathTotals(path, hand);
        check(totals[0] == 3 && totals[1] == 6, "reconstructed cost and consumption");
        check(new HashSet<>(path).size() == path.size(), "reconstructed path is simple");

        int[] seeds = {5, 13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93};
        for (int s = 0; s < seeds.length; s++) {
            int nodes = 3 + s % 4;
            List<RcEdge> edges = lcgEdges(nodes, 3 + s % 6, seeds[s]);
            int costBound = 1;
            for (RcEdge e : edges) costBound += e.cost;
            int target = nodes - 1;
            int previous = INF;
            int[][] rl = resourceLayers(nodes, edges, costBound, 0);
            List<int[]> frontier = new ArrayList<>();
            int recordResource = INF;
            for (int c = 0; c <= costBound; c++) {
                int res = rl[c][target];
                if (res >= INF) continue;
                if (res < recordResource) {      // keeps only strict records: c up, resource down
                    frontier.add(new int[]{c, res});
                    recordResource = res;
                }
            }
            for (int i = 1; i < frontier.size(); i++) {
                check(frontier.get(i)[0] > frontier.get(i - 1)[0]
                                && frontier.get(i)[1] < frontier.get(i - 1)[1],
                        "graph " + s + ": frontier point " + i + " is strictly nondominated");
            }
            for (int budget = 0; budget <= 20; budget++) {
                int[][] cl = costLayers(nodes, edges, budget, 0);
                int viaCost = bestCostForBudget(cl, budget, target);
                int viaResource = bestCostByResourceLayers(rl, budget, target);
                int viaFrontier = INF;
                for (int[] point : frontier) {
                    if (point[1] <= budget) {
                        viaFrontier = point[0];
                        break;
                    }
                }
                int brute = bruteForce(nodes, edges, budget, 0, target).bestCost;
                check(viaCost == brute, "graph " + s + " budget " + budget + ": cost DP matches brute force");
                check(viaResource == brute, "graph " + s + " budget " + budget + ": dual DP matches brute force");
                check(viaFrontier == brute, "graph " + s + " budget " + budget + ": frontier matches brute force");
                check(viaFrontier <= previous, "graph " + s + ": optimum non-increasing in the budget");
                previous = viaFrontier;
            }
            check(frontier.isEmpty() == (bruteForce(nodes, edges, 20, 0, target).bestCost == INF),
                    "graph " + s + ": empty frontier iff the target is unreachable");
        }

        List<RcEdge> noPath = Arrays.asList(new RcEdge(0, 1, 1, 1), new RcEdge(2, 3, 1, 1));
        check(bestCostForBudget(costLayers(4, noPath, 10, 0), 10, 3) == INF,
                "disconnected target reports the INF sentinel");
        check(bestCostByResourceLayers(resourceLayers(4, noPath, 4, 0), 10, 3) == INF,
                "dual DP reports the INF sentinel too");
        check(bruteForce(4, noPath, 10, 0, 3).bestCost == INF, "brute force finds no s -> t path");

        List<RcEdge> tooSmall = Arrays.asList(new RcEdge(0, 1, 1, 7));
        check(bestCostForBudget(costLayers(2, tooSmall, 6, 0), 6, 1) == INF
                        && bestCostForBudget(costLayers(2, tooSmall, 7, 0), 7, 1) == 1,
                "budget one unit short of the edge consumption is infeasible");

        check(bestCostForBudget(costLayers(1, new ArrayList<RcEdge>(), 0, 0), 0, 0) == 0,
                "source equals target: the empty path costs 0");

        System.out.println("PASS");
    }

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
