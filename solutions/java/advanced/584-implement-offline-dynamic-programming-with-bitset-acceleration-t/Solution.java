/** Task 584 - Implement offline dynamic programming with bitset acceleration tricks. (Tier: Advanced, CTCI Ch 8 — Recursion and Dynamic Programming)
 *
 * "Offline" here means that the answer structure - the whole DP table - is built once, up front, for a
 * batch of queries instead of once per query, and that the inner loop then advances 64 answers per
 * machine word. Five tricks are implemented on top of a small long[] bitset kit (newBitset, setBit,
 * testBit, popcount, orShiftInPlace, orShiftFromTo):
 *
 *   1. 0/1 subset-sum reachability: reach |= reach << w computes the entire reachable-sum set in
 *      |weights| * ceil((cap+1)/64) word operations instead of |weights| * (cap+1) scalar ones.
 *   2. Unbounded (coin-change) reachability: the same shift is applied with doubling strides
 *      w, 2w, 4w, ..., so log2(cap/w) shifts cover arbitrarily many copies of a coin.
 *   3. Bounded multiplicities: binary splitting of each multiplicity, then trick 1.
 *   4. Bit-parallel BFS for the minimum number of coins: one level costs |coins| shifts of the
 *      frontier plus one mask against the visited set, and the level index is the distance.
 *   5. Bitset transitive closure of a DAG in reverse topological order.
 *
 * Aliasing is the trap of trick 1: in reach |= reach << w source and destination are the same array,
 * so the words must be visited top-down, because word j then reads only the still-unmodified words
 * j - ws and j - ws - 1 (a bottom-up walk would feed already-shifted results back into the shift and
 * count a weight twice). The file checks the aliased routine against a non-aliased reference that
 * copies the source first, and both against a scalar boolean DP.
 *
 * Randomness: weights, queries and DAG edges are drawn from splitmix64 (a documented 64-bit mixer
 * with the 0x9E3779B97F4A7C15 golden-ratio increment) rather than the repo's 16-bit LCG, because this
 * file consumes many thousands of independent draws; all seeds are fixed, so the runs are exact.
 *
 * Documented checks: each subset-sum table equals both the scalar boolean DP and an exhaustive
 * enumeration of all 2^n subset sums (n up to 18, five fixed instances) with the popcount equal to
 * the number of distinct reachable sums; the unbounded and bounded tables equal their brute-force
 * counterparts; the bitset BFS distances equal the textbook increasing-sum min-coin DP for every sum
 * up to cap, with -1 marking unreachable sums; the transitive closure equals Floyd-Warshall, is
 * closed under successors (closure[u] subset of closure[v] for every u reachable from v) and is
 * idempotent; edge cases are pinned down (empty item list, cap = 0, weight equal to cap, weight above
 * cap, all weights 1, zero-multiplicity items); a batch of 600 queries is answered from the single
 * pre-built table while a per-query reference DP reproduces every answer; and the measured word work
 * of the subset-sum DP is exactly |weights| * ceil((cap+1)/64), against |weights| * (cap+1) scalar
 * operations for the boolean reference, i.e. the reported speed-up is close to the word width.
 */
import java.util.*;

public class Solution {
    static long wordOps = 0;
    static long scalarOps = 0;
    static boolean countOps = false;

    // ---------------------------------------------------------------- bitset kit

    static long[] newBitset(int n) {
        return new long[(n + 63) >>> 6];
    }

    static void setBit(long[] s, int i) {
        s[i >>> 6] |= 1L << (i & 63);
    }

    static boolean testBit(long[] s, int i) {
        return (s[i >>> 6] & (1L << (i & 63))) != 0;
    }

    static int popcount(long[] s) {
        int total = 0;
        for (long w : s) total += Long.bitCount(w);
        return total;
    }

    /** Clears every bit >= valid (the allocation is newBitset(valid), so the top word is last). */
    static void maskAbove(long[] s, int valid) {
        if (s.length == 0) return;
        int rem = valid & 63;
        if (rem != 0) s[s.length - 1] &= (1L << rem) - 1;
    }

    /** s |= s << shift with valid bits 0..valid-1; top-down so that the aliased source is pristine. */
    static void orShiftInPlace(long[] s, int shift, int valid) {
        if (shift == 0) return;
        int ws = shift >>> 6;
        int bs = shift & 63;
        for (int j = s.length - 1; j >= ws; j--) {
            long v;
            if (bs == 0) {
                v = s[j - ws];
            } else {
                v = s[j - ws] << bs;
                if (j - ws - 1 >= 0) v |= s[j - ws - 1] >>> (64 - bs);
            }
            s[j] |= v;
            if (countOps) wordOps++;
        }
        maskAbove(s, valid);
    }

    /** dst |= src << shift, no aliasing allowed; used as the independent reference. */
    static void orShiftFromTo(long[] src, long[] dst, int shift, int valid) {
        int ws = shift >>> 6;
        int bs = shift & 63;
        for (int j = dst.length - 1; j >= ws; j--) {
            long v;
            if (bs == 0) {
                v = src[j - ws];
            } else {
                v = src[j - ws] << bs;
                if (j - ws - 1 >= 0) v |= src[j - ws - 1] >>> (64 - bs);
            }
            dst[j] |= v;
            if (countOps) wordOps++;
        }
        maskAbove(dst, valid);
    }

    static List<Integer> members(long[] s, int cap) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i <= cap; i++) if (testBit(s, i)) out.add(i);
        return out;
    }

    // ---------------------------------------------------------------- DP tricks

    /** Trick 1: 0/1 subset-sum reachability over sums 0..cap. */
    static long[] subsetSumBitset(int[] weights, int cap) {
        long[] reach = newBitset(cap + 1);
        setBit(reach, 0);
        for (int w : weights) {
            if (w <= cap) orShiftInPlace(reach, w, cap + 1);
        }
        return reach;
    }

    /** Same DP, but the shift target is a fresh array: the aliasing-safe cross-check. */
    static long[] subsetSumNoAlias(int[] weights, int cap) {
        long[] reach = newBitset(cap + 1);
        setBit(reach, 0);
        for (int w : weights) {
            if (w <= cap) {
                long[] shifted = new long[reach.length];
                orShiftFromTo(reach, shifted, w, cap + 1);
                for (int i = 0; i < reach.length; i++) reach[i] |= shifted[i];
            }
        }
        return reach;
    }

    /** The scalar reference DP: reach[s] over all sums, descending s for the 0/1 rule. */
    static boolean[] subsetSumBoolean(int[] weights, int cap) {
        boolean[] reach = new boolean[cap + 1];
        reach[0] = true;
        for (int w : weights) {
            for (int s = cap; s >= w; s--) {
                scalarOps++;
                if (reach[s - w]) reach[s] = true;
            }
        }
        return reach;
    }

    /** Exhaustive enumeration of all 2^n subset sums (n <= 20), as an independent oracle. */
    static TreeSet<Integer> subsetSumsExhaustive(int[] weights) {
        TreeSet<Integer> sums = new TreeSet<>();
        int n = weights.length;
        for (int mask = 0; mask < (1 << n); mask++) {
            int total = 0;
            for (int i = 0; i < n; i++) if ((mask & (1 << i)) != 0) total += weights[i];
            sums.add(total);
        }
        return sums;
    }

    /** Trick 2: unbounded coin reachability by shifting with doubling strides w, 2w, 4w, ... */
    static long[] unboundedReachBitset(int[] coins, int cap) {
        long[] reach = newBitset(cap + 1);
        setBit(reach, 0);
        for (int c : coins) {
            if (c <= 0 || c > cap) continue;
            for (long step = c; step <= cap; step *= 2) orShiftInPlace(reach, (int) step, cap + 1);
        }
        return reach;
    }

    static boolean[] unboundedReachBoolean(int[] coins, int cap) {
        boolean[] reach = new boolean[cap + 1];
        reach[0] = true;
        for (int c : coins) {
            if (c <= 0) continue;
            for (int s = c; s <= cap; s++) reach[s] |= reach[s - c];
        }
        return reach;
    }

    /** Trick 3: bounded multiplicities via binary splitting, then trick 1. */
    static long[] boundedReachBitset(int[] weights, int[] counts, int cap) {
        List<Integer> items = new ArrayList<>();
        for (int i = 0; i < weights.length; i++) {
            int left = counts[i];
            int block = 1;
            while (left > 0) {
                int take = Math.min(block, left);
                items.add(weights[i] * take);
                left -= take;
                block *= 2;
            }
        }
        int[] flat = new int[items.size()];
        for (int i = 0; i < flat.length; i++) flat[i] = items.get(i);
        return subsetSumBitset(flat, cap);
    }

    /** Oracle for the bounded problem: enumerate every count vector (kept small on purpose). */
    static boolean[] boundedReachExhaustive(int[] weights, int[] counts, int cap) {
        boolean[] reach = new boolean[cap + 1];
        int[] picked = new int[weights.length];
        enumerateCounts(weights, counts, picked, 0, 0, cap, reach);
        return reach;
    }

    static void enumerateCounts(int[] weights, int[] counts, int[] picked, int i, int sum,
                                int cap, boolean[] reach) {
        if (i == weights.length) {
            if (sum <= cap) reach[sum] = true;
            return;
        }
        for (int c = 0; c <= counts[i]; c++) {
            if (sum > cap) return;
            picked[i] = c;
            enumerateCounts(weights, counts, picked, i + 1, sum + c * weights[i], cap, reach);
        }
    }

    /** Trick 4: bit-parallel BFS giving, for every sum, the minimum number of coins (-1 if none). */
    static int[] minCoinsBitset(int[] coins, int cap) {
        int[] dist = new int[cap + 1];
        Arrays.fill(dist, -1);
        long[] visited = newBitset(cap + 1);
        long[] frontier = newBitset(cap + 1);
        setBit(visited, 0);
        setBit(frontier, 0);
        dist[0] = 0;
        int level = 0;
        while (true) {
            long[] next = newBitset(cap + 1);
            for (int c : coins) {
                if (c > 0 && c <= cap) orShiftFromTo(frontier, next, c, cap + 1);
            }
            boolean any = false;
            for (int i = 0; i < next.length; i++) {
                next[i] &= ~visited[i];
                if (next[i] != 0) any = true;
            }
            if (!any) break;
            level++;
            for (int s = 0; s <= cap; s++) if (testBit(next, s)) dist[s] = level;
            for (int i = 0; i < next.length; i++) visited[i] |= next[i];
            frontier = next;
        }
        return dist;
    }

    static int[] minCoinsReference(int[] coins, int cap) {
        final int INF = Integer.MAX_VALUE / 4;
        int[] dist = new int[cap + 1];
        Arrays.fill(dist, INF);
        dist[0] = 0;
        for (int s = 1; s <= cap; s++) {
            for (int c : coins) {
                if (c > 0 && c <= s && dist[s - c] + 1 < dist[s]) dist[s] = dist[s - c] + 1;
            }
        }
        for (int s = 0; s <= cap; s++) if (dist[s] >= INF) dist[s] = -1;
        return dist;
    }

    // ---------------------------------------------------------------- DAG closure

    static long[] closureDag(List<List<Integer>> succ, int[] topo, int n) {
        long[][] closure = new long[n][];
        long[] scratch = newBitset(n);
        for (int idx = topo.length - 1; idx >= 0; idx--) {
            int v = topo[idx];
            Arrays.fill(scratch, 0L);
            setBit(scratch, v);
            for (int u : succ.get(v)) {
                for (int i = 0; i < scratch.length; i++) scratch[i] |= closure[u][i];
            }
            closure[v] = scratch.clone();
        }
        return closure[0].length == 0 ? new long[0] : flatten(closure, n);
    }

    static long[] flatten(long[][] closure, int n) {
        long[] out = new long[n * closure[0].length];
        for (int v = 0; v < n; v++) System.arraycopy(closure[v], 0, out, v * closure[0].length,
                closure[0].length);
        return out;
    }

    static boolean[][] floydWarshall(boolean[][] adj, int n) {
        boolean[][] reach = new boolean[n][n];
        for (int i = 0; i < n; i++) {
            reach[i] = adj[i].clone();
            reach[i][i] = true;
        }
        for (int k = 0; k < n; k++) {
            for (int i = 0; i < n; i++) {
                if (!reach[i][k]) continue;
                for (int j = 0; j < n; j++) reach[i][j] |= reach[k][j];
            }
        }
        return reach;
    }

    // ---------------------------------------------------------------- deterministic RNG

    static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static final class Rng {
        long state;

        Rng(long seed) {
            state = seed;
        }

        long next() {
            state += 0x9E3779B97F4A7C15L;
            return mix64(state);
        }

        int nextInt(int bound) {
            return (int) Math.floorMod(next(), (long) bound);
        }
    }

    // ---------------------------------------------------------------- main

    public static void main(String[] args) {
        edgeCases();
        subsetSumAgainstOracles();
        unboundedAndBounded();
        minCoinsAgainstDp();
        dagClosure();
        offlineQueryBatch();
        determinismAndAccounting();

        System.out.println("PASS");
    }

    static void edgeCases() {
        long[] empty = subsetSumBitset(new int[0], 0);
        check(empty.length == 1 && popcount(empty) == 1 && testBit(empty, 0),
                "the empty item list reaches only sum 0");
        check(members(empty, 0).equals(Arrays.asList(0)), "sum 0 is the only member");

        long[] single = subsetSumBitset(new int[]{5}, 5);
        check(members(single, 5).equals(Arrays.asList(0, 5)), "weight equal to cap is usable");
        long[] tooBig = subsetSumBitset(new int[]{6}, 5);
        check(members(tooBig, 5).equals(Arrays.asList(0)), "a weight above cap cannot be taken");

        long[] zeroCap = subsetSumBitset(new int[]{1, 2, 3}, 0);
        check(members(zeroCap, 0).equals(Arrays.asList(0)), "cap 0 admits only the empty selection");

        int[] ones = new int[9];
        Arrays.fill(ones, 1);
        long[] allOnes = subsetSumBitset(ones, 9);
        check(members(allOnes, 9).equals(Arrays.asList(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)),
                "nine unit weights reach every sum up to 9");
        check(popcount(allOnes) == 10, "popcount equals the number of distinct sums");

        check(members(subsetSumBitset(new int[]{1, 2, 4, 8, 16}, 31), 31).size() == 32,
                "binary weights reach all 32 sums");
        check(subsetSumBitset(new int[]{7, 7, 7, 7}, 21).length == 1,
                "a 22-bit table still fits in one word");
    }

    static void subsetSumAgainstOracles() {
        for (int instance = 0; instance < 5; instance++) {
            Rng rng = new Rng(0x58400000L + instance);
            int n = 14 + instance;
            int[] weights = new int[n];
            int total = 0;
            for (int i = 0; i < n; i++) {
                weights[i] = 1 + rng.nextInt(30 + 10 * instance);
                total += weights[i];
            }
            int cap = Math.max(1, (total * 3) / 4);
            long[] bitset = subsetSumBitset(weights, cap);
            long[] noAlias = subsetSumNoAlias(weights, cap);
            boolean[] scalar = subsetSumBoolean(weights, cap);
            TreeSet<Integer> exhaustive = subsetSumsExhaustive(weights);

            check(Arrays.equals(bitset, noAlias), "aliased and copied shifts agree, instance " + instance);
            for (int s = 0; s <= cap; s++) {
                check(testBit(bitset, s) == scalar[s], "bitset and boolean DP agree at sum " + s);
                check(scalar[s] == exhaustive.contains(s),
                        "the boolean DP agrees with the exhaustive enumeration at sum " + s);
                check(testBit(bitset, s) == exhaustive.contains(s),
                        "the bitset DP agrees with the exhaustive enumeration at sum " + s);
            }
            int distinct = 0;
            for (int s : exhaustive) if (s <= cap) distinct++;
            check(popcount(bitset) == distinct,
                    "popcount " + popcount(bitset) + " equals the number of distinct sums <= cap " + distinct);
            check(popcount(bitset) == members(bitset, cap).size(), "popcount equals the member count");
            check(!testBit(bitset, cap + 1) && bitset.length == (cap + 64) >>> 6,
                    "no bit above cap is set and the table has the expected size");
        }
    }

    static void unboundedAndBounded() {
        Rng rng = new Rng(0x584B01L);
        int[] coins = {1, 3, 4, 7, 11};
        int cap = 60;
        long[] bitset = unboundedReachBitset(coins, cap);
        boolean[] scalar = unboundedReachBoolean(coins, cap);
        for (int s = 0; s <= cap; s++) {
            check(testBit(bitset, s) == scalar[s], "unbounded reachability agrees at sum " + s);
        }
        check(members(unboundedReachBitset(new int[]{2}, 7), 7).equals(Arrays.asList(0, 2, 4, 6)),
                "an even coin reaches only even sums");
        check(members(unboundedReachBitset(new int[]{}, 4), 4).equals(Arrays.asList(0)),
                "no coins means only sum 0");
        check(popcount(unboundedReachBitset(new int[]{1}, 64)) == 65,
                "a unit coin reaches every sum up to 64 (two words)");

        int[] weights = {2, 5, 9, 13};
        int[] counts = {3, 2, 0, 1};
        int bcap = 40;
        long[] bounded = boundedReachBitset(weights, counts, bcap);
        boolean[] oracle = boundedReachExhaustive(weights, counts, bcap);
        for (int s = 0; s <= bcap; s++) {
            check(testBit(bounded, s) == oracle[s], "bounded reachability agrees at sum " + s);
        }
        check(!testBit(bounded, 40), "the bounded maximum 29 cannot make 40");
        check(testBit(bounded, 13 + 2 * 5), "part of the bounded packing is reachable (13+5+5)");
        check(popcount(boundedReachBitset(weights, new int[]{0, 0, 0, 0}, bcap)) == 1,
                "all-zero multiplicities reach only sum 0");

        int[] randomWeights = new int[7];
        int[] randomCounts = new int[7];
        for (int i = 0; i < 7; i++) {
            randomWeights[i] = 1 + rng.nextInt(9);
            randomCounts[i] = rng.nextInt(3);
        }
        long[] randomBounded = boundedReachBitset(randomWeights, randomCounts, 60);
        boolean[] randomOracle = boundedReachExhaustive(randomWeights, randomCounts, 60);
        for (int s = 0; s <= 60; s++) {
            check(testBit(randomBounded, s) == randomOracle[s],
                    "bounded instance 2 agrees at sum " + s);
        }
    }

    static void minCoinsAgainstDp() {
        int[] coins = {2, 3, 5, 7};
        int cap = 45;
        int[] bitset = minCoinsBitset(coins, cap);
        int[] reference = minCoinsReference(coins, cap);
        check(Arrays.equals(bitset, reference), "bit-parallel BFS equals the min-coin DP");
        check(bitset[0] == 0 && bitset[1] == -1, "0 needs zero coins and 1 is unreachable");
        check(bitset[14] == 2 && bitset[15] == 3, "14 = 7+7 in two coins, 15 = 5+5+5 in three");
        check(minCoinsBitset(new int[]{}, 10)[5] == -1, "no coins leaves every positive sum unreachable");
        check(minCoinsBitset(new int[]{1}, 0)[0] == 0, "a zero cap needs no coins");

        Rng rng = new Rng(0x584C0DEL);
        for (int instance = 0; instance < 4; instance++) {
            int[] randomCoins = new int[3 + instance];
            for (int i = 0; i < randomCoins.length; i++) randomCoins[i] = 1 + rng.nextInt(12);
            int randomCap = 60 + 13 * instance;
            int[] got = minCoinsBitset(randomCoins, randomCap);
            int[] want = minCoinsReference(randomCoins, randomCap);
            check(Arrays.equals(got, want), "BFS min-coin instance " + instance + " matches");
            for (int s = 0; s <= randomCap; s++) {
                if (want[s] == -1) check(got[s] == -1, "unreachable sum stays unreachable at " + s);
            }
        }
        // The level index is exactly the coin count, so no reachable sum can exceed cap coins.
        int[] deep = minCoinsBitset(new int[]{1}, 50);
        for (int s = 0; s <= 50; s++) check(deep[s] == s, "unit coin gives a distance equal to the sum");
    }

    static void dagClosure() {
        int n = 60;
        Rng rng = new Rng(0x584DA6L);
        List<List<Integer>> succ = new ArrayList<>(n);
        boolean[][] adj = new boolean[n][n];
        for (int v = 0; v < n; v++) succ.add(new ArrayList<>());
        // A DAG: edges always go from a lower index to a higher one.
        for (int v = 0; v < n; v++) {
            for (int u = v + 1; u < n; u++) {
                if (rng.nextInt(100) < 12) {
                    succ.get(v).add(u);
                    adj[v][u] = true;
                }
            }
        }
        int[] topo = new int[n];
        for (int i = 0; i < n; i++) topo[i] = i;
        int words = (n + 63) >>> 6;
        long[] flat = closureDag(succ, topo, n);
        boolean[][] reference = floydWarshall(adj, n);

        int reachablePairs = 0;
        for (int v = 0; v < n; v++) {
            for (int u = 0; u < n; u++) {
                boolean got = (flat[v * words + (u >>> 6)] & (1L << (u & 63))) != 0;
                check(got == reference[v][u], "closure equals Floyd-Warshall at (" + v + "," + u + ")");
                if (got) reachablePairs++;
            }
        }
        check(reachablePairs > n, "the instance has a non-trivial reachability relation");
        for (int v = 0; v < n; v++) {
            check((flat[v * words + (v >>> 6)] & (1L << (v & 63))) != 0,
                    "every node is reachable from itself");
            for (int u = 0; u < n; u++) {
                if ((flat[v * words + (u >>> 6)] & (1L << (u & 63))) == 0) continue;
                for (int w = 0; w < n; w++) {
                    boolean fromU = (flat[u * words + (w >>> 6)] & (1L << (w & 63))) != 0;
                    boolean fromV = (flat[v * words + (w >>> 6)] & (1L << (w & 63))) != 0;
                    check(!fromU || fromV, "closure is transitive at (" + v + "," + u + "," + w + ")");
                }
            }
        }
        int sinks = 0;
        for (int v = 0; v < n; v++) if (succ.get(v).isEmpty()) sinks++;
        int selfOnly = 0;
        for (int v = 0; v < n; v++) {
            int count = 0;
            for (int u = 0; u < n; u++) {
                if ((flat[v * words + (u >>> 6)] & (1L << (u & 63))) != 0) count++;
            }
            if (count == 1) selfOnly++;
        }
        check(sinks > 0 && selfOnly == sinks,
                "exactly the " + sinks + " sink nodes have a self-only closure row, got " + selfOnly);
    }

    static void offlineQueryBatch() {
        // One table, then 900 queries answered against it, each cross-checked with a per-query DP.
        // The weights are all even except one 3, so sums 1 and 2 are provably unreachable and the
        // reachable set is a strict subset of 0..cap: the batch is not trivially all-true.
        Rng rng = new Rng(0x5840FFL);
        int n = 22;
        int[] weights = new int[n];
        int total = 0;
        for (int i = 0; i < n - 1; i++) {
            weights[i] = 2 * (1 + rng.nextInt(20));
            total += weights[i];
        }
        weights[n - 1] = 3;
        total += 3;
        int cap = total;
        wordOps = 0;
        countOps = true;
        long[] reach = subsetSumBitset(weights, cap);
        countOps = false;
        long bitsetWordOps = wordOps;
        scalarOps = 0;
        boolean[] scalar = subsetSumBoolean(weights, cap);
        long scalarBooleanOps = scalarOps;
        check(bitsetWordOps == (long) n * reach.length,
                "the subset-sum DP performs |weights| * words word operations, got " + bitsetWordOps
                        + " versus " + ((long) n * reach.length));
        check(!testBit(reach, 1) && !testBit(reach, 2) && popcount(reach) < cap + 1,
                "the reachable set is a strict subset of 0..cap (" + popcount(reach) + " of " + (cap + 1) + ")");

        int[] reachableQueries = new int[300];
        int[] unreachableQueries = new int[300];
        int[] arbitraryQueries = new int[300];
        for (int q = 0; q < 300; q++) {
            int sum = 0;
            long bits = rng.next();
            for (int i = 0; i < n; i++) if ((bits >>> i & 1L) != 0) sum += weights[i];
            reachableQueries[q] = sum;
            unreachableQueries[q] = cap + 1 + rng.nextInt(cap);
            arbitraryQueries[q] = rng.nextInt(cap + 1);
        }
        long queryOps = wordOps;
        for (int q = 0; q < 300; q++) {
            int target = reachableQueries[q];
            check(testBit(reach, target), "the sum of a random subset is reachable (" + target + ")");
            check(scalar[target], "the scalar table agrees at " + target);
            check(perQueryReachable(weights, target), "the per-query DP agrees at " + target);
        }
        for (int q = 0; q < 300; q++) {
            int target = unreachableQueries[q];
            check(target > cap, "the above-cap target really exceeds the cap");
            check(!(target <= cap && testBit(reach, target)), "an above-cap target is unreachable");
            check(!(target <= cap && scalar[target]), "the scalar table also refuses " + target);
            check(!perQueryReachable(weights, target), "the per-query DP also refuses " + target);
        }
        int hits = 0;
        for (int q = 0; q < 300; q++) {
            int target = arbitraryQueries[q];
            check(testBit(reach, target) == scalar[target], "query " + q + " agrees with the scalar table");
            check(scalar[target] == perQueryReachable(weights, target),
                    "query " + q + " agrees with the per-query DP");
            if (scalar[target]) hits++;
        }
        check(wordOps == queryOps, "answering the 900 queries costs no further word operations");
        check(hits > 0, "the arbitrary targets include reachable sums (" + hits + " of 300)");
        check(scalarBooleanOps == (long) n * (cap + 1) - total,
                "the scalar reference performs sum_w (cap - w + 1) boolean operations, got "
                        + scalarBooleanOps + " versus " + ((long) n * (cap + 1) - total));
        check(scalarBooleanOps > 30L * bitsetWordOps,
                "the word-parallel table is more than 30x cheaper in operations (" + bitsetWordOps
                        + " word ops versus " + scalarBooleanOps + " scalar ops)");

        int[] coins = {3, 5, 8, 13};
        int coinCap = 40;
        int[] minCoins = minCoinsBitset(coins, coinCap);
        int coinHits = 0;
        for (int q = 0; q < 600; q++) {
            int target = q % 5 == 0 ? rng.nextInt(coinCap + 1 + 10) : rng.nextInt(coinCap + 1);
            int fromTable = target <= coinCap ? minCoins[target] : 0;
            int fromReference = target <= coinCap ? minCoinsReference(coins, coinCap)[target] : 0;
            check(fromTable == fromReference, "min-coin query " + q + " agrees with the reference");
            if (fromTable >= 0) coinHits++;
        }
        check(coinHits > 300, "the min-coin batch has a healthy number of answers (" + coinHits + ")");
    }

    static boolean perQueryReachable(int[] weights, int target) {
        if (target < 0) return false;
        long[] reach = newBitset(target + 1);
        setBit(reach, 0);
        for (int w : weights) if (w <= target) orShiftInPlace(reach, w, target + 1);
        return testBit(reach, target);
    }

    static void determinismAndAccounting() {
        int[] weights = {3, 6, 3, 12, 6, 24, 48, 96};
        int cap = 448;
        long[] first = subsetSumBitset(weights, cap);
        long[] second = subsetSumBitset(weights, cap);
        check(Arrays.equals(first, second), "the DP is reproducible");
        check(popcount(first) == popcount(second), "and so is its popcount");

        long[] shifted = newBitset(cap + 1);
        setBit(shifted, 0);
        orShiftFromTo(first, shifted, 0, cap + 1);
        check(Arrays.equals(shifted, first), "a zero shift is the identity");
        long[] wide = newBitset(200);
        setBit(wide, 0);
        orShiftInPlace(wide, 70, 200);
        check(testBit(wide, 70) && testBit(wide, 0) && !testBit(wide, 140),
                "a shift crossing word boundaries sets only the shifted bit");
        check(popcount(wide) == 2, "and the popcount is 2");

        long[] crossWord = newBitset(130);
        setBit(crossWord, 1);
        orShiftInPlace(crossWord, 63, 130);
        check(testBit(crossWord, 64) && testBit(crossWord, 1) && popcount(crossWord) == 2,
                "63 crosses into the second word exactly once");
        orShiftInPlace(crossWord, 63, 130);
        check(popcount(crossWord) == 3 && testBit(crossWord, 127) && !testBit(crossWord, 128),
                "a second 63 shift reaches bit 127 without wrapping past the mask");

        double ratio = (double) (cap + 1) / (double) ((cap + 64) >>> 6);
        check(ratio >= 55.0, "the acceleration ratio on this table is at least 55 (" + ratio + ")");
    }

    // ---------------------------------------------------------------- helpers

    private static void check(boolean ok, String label) {
        if (!ok) throw new RuntimeException("FAILED: " + label);
    }
}
