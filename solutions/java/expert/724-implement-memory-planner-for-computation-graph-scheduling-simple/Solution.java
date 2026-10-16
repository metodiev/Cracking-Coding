/** Task 724 - Implement memory planner for computation graph scheduling (simple greedy). (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * A computation graph holds tensors, each born at the op that produces it and dead at the last op that
 * consumes it. Live intervals can share a pool, so the peak of a plan is decided by two choices: the
 * *order* in which ready ops run (how long a tensor stays alive) and the *allocation* of offsets (which
 * packs non-overlapping intervals into the same bytes). The planner does both predictably: it runs the
 * ready op that releases the most bytes, then packs tensors with a first-fit free list that coalesces
 * neighbouring holes as tensors die. An op's output is allocated while its inputs are still resident.
 *
 * Invariants verified: the computed last-use interval of every tensor matches an independent scan of its
 * consumers; no two tensors alive at the same instant overlap in the pool; peak >= the maximum live bytes
 * at any instant (a lower bound no allocator can beat) and peak <= the sum of all tensor sizes; the greedy
 * plan is never worse than the graph's given order and on a small graph is compared against the best peak
 * over every topological order, enumerated by brute force.
 *
 * Simplified: tensors are sized in abstract bytes and the allocator is a free list with first fit, not a
 * real caching allocator; device synchronisation, streams and alignment padding beyond a fixed alignment
 * are ignored. The verified properties are the memory-safety invariant and the bounds above.
 */
import java.util.*;

public class Solution {
    /** One operation: `inputs` are tensor ids, `output` is the tensor it produces. */
    static final class Node {
        final int[] inputs;
        final int output;
        Node(int[] inputs, int output) {
            this.inputs = inputs;
            this.output = output;
        }
    }

    /** A graph of ops over tensors with sizes; ops are listed in a valid topological order. */
    static final class Graph {
        final List<Node> nodes;
        final int[] sizes;                // sizes[tensor]
        final int[] producer;             // node that produces each tensor, -1 for an input
        final int[][] consumers;          // nodes that read each tensor

        Graph(List<Node> nodes, int[] sizes) {
            this.nodes = nodes;
            this.sizes = sizes;
            this.producer = new int[sizes.length];
            Arrays.fill(producer, -1);
            List<List<Integer>> reads = new ArrayList<>();
            for (int t = 0; t < sizes.length; t++) reads.add(new ArrayList<>());
            for (int n = 0; n < nodes.size(); n++) {
                if (producer[nodes.get(n).output] != -1) throw new IllegalArgumentException("tensor produced twice");
                producer[nodes.get(n).output] = n;
                for (int t : nodes.get(n).inputs) {
                    if (t < 0 || t >= sizes.length) throw new IllegalArgumentException("unknown tensor " + t);
                    reads.get(t).add(n);
                }
            }
            consumers = new int[sizes.length][];
            for (int t = 0; t < sizes.length; t++) {
                consumers[t] = reads.get(t).stream().mapToInt(Integer::intValue).toArray();
            }
            checkAcyclic();
        }

        /** Kahn's algorithm: a topological sweep must be able to consume every op. */
        private void checkAcyclic() {
            boolean[] done = new boolean[nodes.size()];
            for (int pass = 0; pass < nodes.size(); pass++) {
                boolean progress = false;
                for (int n = 0; n < nodes.size(); n++) {
                    if (!done[n] && isReady(n, done)) { done[n] = true; progress = true; }
                }
                if (!progress) break;
            }
            for (boolean finished : done) if (!finished) throw new IllegalArgumentException("cycle in the graph");
        }

        /** Last node that reads each tensor, -1 when nothing reads it. */
        int[] lastUse() {
            int[] last = new int[sizes.length];
            Arrays.fill(last, -1);
            for (int t = 0; t < sizes.length; t++) {
                for (int n : consumers[t]) last[t] = Math.max(last[t], n);
            }
            return last;
        }

        /**
         * Greedy order: repeatedly run the ready op that frees the most bytes, i.e. whose execution kills
         * input tensors nobody else needs. Ties go to the lower node index, so the plan is deterministic.
         */
        int[] greedyOrder() {
            int[] last = lastUse();
            int[] order = new int[nodes.size()];
            boolean[] done = new boolean[nodes.size()];
            for (int position = 0; position < nodes.size(); position++) {
                int bestNode = -1;
                long bestFreed = Long.MIN_VALUE;
                for (int n = 0; n < nodes.size(); n++) {
                    if (done[n]) continue;
                    boolean ok = true;
                    long freed = 0;
                    for (int t : nodes.get(n).inputs) {
                        if (producer[t] != -1 && !done[producer[t]]) { ok = false; break; }
                        if (last[t] == n) freed += sizes[t];
                    }
                    if (!ok) continue;
                    if (freed > bestFreed) { bestFreed = freed; bestNode = n; }
                }
                if (bestNode < 0) throw new IllegalStateException("no ready op - graph is cyclic");
                done[bestNode] = true;
                order[position] = bestNode;
            }
            return order;
        }

        /**
         * Peak bytes of a plan under a first-fit free list. Roots are allocated first; an op allocates its
         * output while its inputs are still resident (so the transient peak of an op covers both) and only
         * then releases the inputs whose last use it was. Also checks the safety invariant.
         */
        long peakBytes(int[] order, String[] violation) {
            int[] last = lastUse();
            Map<Integer, Long> offset = new HashMap<>();
            List<long[]> free = new ArrayList<>();             // [offset, length], sorted
            for (int t = 0; t < sizes.length; t++) if (producer[t] == -1) allocate(offset, free, t);
            long peak = livePeak(offset, null);
            for (int position = 0; position < order.length; position++) {
                Node node = nodes.get(order[position]);
                allocate(offset, free, node.output);
                peak = Math.max(peak, livePeak(offset, violation));
                for (int t : node.inputs) if (last[t] == order[position]) release(offset, free, t);
            }
            return peak;
        }

        private void allocate(Map<Integer, Long> offset, List<long[]> free, int tensor) {
            long size = sizes[tensor];
            long at = -1;
            for (long[] range : free) {
                if (range[1] >= size) { at = range[0]; break; }
            }
            if (at < 0) {
                at = livePeak(offset, null);
            } else {
                for (long[] range : free) {
                    if (range[0] == at) {
                        range[0] += size;
                        range[1] -= size;
                        break;
                    }
                }
                free.removeIf(range -> range[1] == 0);
            }
            offset.put(tensor, at);
            free.sort((a, b) -> Long.compare(a[0], b[0]));
            List<long[]> merged = new ArrayList<>();
            for (long[] range : free) {
                if (!merged.isEmpty()) {
                    long[] previous = merged.get(merged.size() - 1);
                    if (previous[0] + previous[1] == range[0]) { previous[1] += range[1]; continue; }
                }
                merged.add(range);
            }
            free.clear();
            free.addAll(merged);
        }

        private void release(Map<Integer, Long> offset, List<long[]> free, int tensor) {
            Long at = offset.remove(tensor);
            if (at == null) return;
            free.add(new long[]{at, sizes[tensor]});
            free.sort((a, b) -> Long.compare(a[0], b[0]));
        }

        /** Highest live byte end, and the first overlapping pair of live tensors when asked for it. */
        private long livePeak(Map<Integer, Long> offset, String[] violation) {
            long high = 0;
            for (Map.Entry<Integer, Long> a : offset.entrySet()) {
                high = Math.max(high, a.getValue() + sizes[a.getKey()]);
                if (violation == null) continue;
                for (Map.Entry<Integer, Long> b : offset.entrySet()) {
                    if (a.getKey() >= b.getKey()) continue;
                    boolean overlap = a.getValue() < b.getValue() + sizes[b.getKey()]
                            && b.getValue() < a.getValue() + sizes[a.getKey()];
                    if (overlap) violation[0] = "tensors " + a.getKey() + " and " + b.getKey() + " overlap";
                }
            }
            return high;
        }

        /** Maximum bytes alive at any instant of a plan: the lower bound on the peak. */
        long maxLiveBytes(int[] order) {
            int[] last = lastUse();
            boolean[] alive = new boolean[sizes.length];
            for (int t = 0; t < sizes.length; t++) if (producer[t] == -1) alive[t] = true;
            long worst = 0;
            for (int position = 0; position < order.length; position++) {
                Node node = nodes.get(order[position]);
                alive[node.output] = true;
                long live = 0;
                for (int t = 0; t < sizes.length; t++) if (alive[t]) live += sizes[t];
                worst = Math.max(worst, live);
                for (int t : node.inputs) if (last[t] == order[position]) alive[t] = false;
            }
            return worst;
        }

        /**
         * Every topological order of the graph (brute force, tiny graphs only): a node may be appended once
         * all of its producers are already in the prefix.
         */
        List<int[]> allOrders() {
            List<int[]> out = new ArrayList<>();
            permute(0, new int[nodes.size()], new boolean[nodes.size()], new boolean[nodes.size()], out);
            return out;
        }

        private boolean isReady(int node, boolean[] done) {
            for (int t : nodes.get(node).inputs) if (producer[t] != -1 && !done[producer[t]]) return false;
            return true;
        }

        private void permute(int position, int[] order, boolean[] used, boolean[] done, List<int[]> out) {
            if (position == order.length) { out.add(order.clone()); return; }
            for (int n = 0; n < nodes.size(); n++) {
                if (used[n] || !isReady(n, done)) continue;
                used[n] = true; done[n] = true; order[position] = n;
                permute(position + 1, order, used, done, out);
                used[n] = false; done[n] = false;
            }
        }
    }

    static Graph chain(int length, int size) {
        int[] sizes = new int[length + 1];
        Arrays.fill(sizes, size);
        List<Node> nodes = new ArrayList<>();
        for (int n = 0; n < length; n++) nodes.add(new Node(new int[]{n}, n + 1));
        return new Graph(nodes, sizes);
    }

    /** A diamond with a tail: t1 = f(t0), t2 = f(t0), t3 = f(t0), t4 = join(t1..t3), t5 = f(t4). */
    static Graph diamond(int first, int join, int tail, int feed) {
        int[] sizes = new int[6];
        sizes[0] = feed;
        for (int k = 1; k <= 3; k++) sizes[k] = first;
        sizes[4] = join;
        sizes[5] = tail;
        List<Node> nodes = new ArrayList<>();
        nodes.add(new Node(new int[]{0}, 1));
        nodes.add(new Node(new int[]{0}, 2));
        nodes.add(new Node(new int[]{0}, 3));
        nodes.add(new Node(new int[]{1, 2, 3}, 4));
        nodes.add(new Node(new int[]{4}, 5));
        return new Graph(nodes, sizes);
    }

    public static void main(String[] args) {
        // --- degenerate graphs ------------------------------------------------
        Graph empty = new Graph(new ArrayList<>(), new int[0]);
        check(empty.greedyOrder().length == 0, "an empty graph plans to nothing");
        check(empty.peakBytes(new int[0], new String[1]) == 0, "an empty graph needs zero bytes");

        Graph single = new Graph(Collections.singletonList(new Node(new int[]{0}, 1)), new int[]{9, 40});
        int[] singleOrder = single.greedyOrder();
        check(singleOrder.length == 1 && singleOrder[0] == 0, "one op plans to itself");
        // The planner allocates an op's output while its inputs are still resident and releases them only
        // after the op has run, so a lone op peaks at the sum of the two sizes rather than the larger one.
        check(single.peakBytes(singleOrder, new String[1]) == 49,
                "an op peaks at its inputs plus its output, peak=" + single.peakBytes(singleOrder, new String[1]));
        Graph bigInput = new Graph(Collections.singletonList(new Node(new int[]{0}, 1)), new int[]{40, 9});
        check(bigInput.peakBytes(bigInput.greedyOrder(), new String[1]) == 49,
                "the same holds when the input is the larger side");

        boolean rejectedCycle = false;
        boolean rejectedDouble = false;
        try {
            new Graph(Arrays.asList(new Node(new int[]{1}, 0), new Node(new int[]{0}, 1)), new int[]{4, 4});
        } catch (IllegalArgumentException expected) {
            rejectedCycle = true;
        }
        try {
            new Graph(Arrays.asList(new Node(new int[]{0}, 1), new Node(new int[]{0}, 1)), new int[]{4, 4});
        } catch (IllegalArgumentException expected) {
            rejectedDouble = true;
        }
        check(rejectedCycle && rejectedDouble,
                "a cyclic graph and a twice-produced tensor must both be rejected");

        // --- liveness against an independent scan -----------------------------
        Graph graph = chain(6, 16);
        int[] last = graph.lastUse();
        for (int t = 1; t < 6; t++) {
            check(last[t] == t, "tensor " + t + " is last read by node " + t + ", got " + last[t]);
        }
        check(last[6] == -1, "the chain's output is never read again");

        // --- safety and bounds over many plans --------------------------------
        Graph[] graphs = {chain(8, 24), diamond(64, 16, 8, 32), diamond(8, 64, 8, 32), diamond(32, 32, 32, 32)};
        for (Graph g : graphs) {
            int[] greedy = g.greedyOrder();
            String[] violation = new String[1];
            long peak = g.peakBytes(greedy, violation);
            check(violation[0] == null, "greedy plan must be memory safe: " + violation[0]);
            check(peak >= g.maxLiveBytes(greedy), "peak must respect the liveness lower bound");
            long total = 0;
            for (int size : g.sizes) total += size;
            check(peak <= total, "peak cannot exceed the sum of all tensors: " + peak + " vs " + total);
            int[] identity = new int[g.nodes.size()];
            for (int n = 0; n < identity.length; n++) identity[n] = n;
            check(peak <= g.peakBytes(identity, null),
                    "the greedy order must not be worse than the given order: " + peak + " vs "
                            + g.peakBytes(identity, null));
        }

        // --- brute-force comparison on a small graph --------------------------
        Graph small = diamond(16, 8, 8, 24);
        List<int[]> orders = small.allOrders();
        check(orders.size() >= 4, "the diamond has several topological orders, found " + orders.size());
        long best = Long.MAX_VALUE;
        for (int[] order : orders) {
            String[] violation = new String[1];
            long peak = small.peakBytes(order, violation);
            check(violation[0] == null, "brute-force plan must be memory safe: " + violation[0]);
            check(peak >= small.maxLiveBytes(order), "brute-force peak must respect the lower bound");
            best = Math.min(best, peak);
        }
        int[] greedySmall = small.greedyOrder();
        long greedyPeak = small.peakBytes(greedySmall, new String[1]);
        check(best <= greedyPeak, "the brute-force optimum cannot exceed the greedy peak");
        check(greedyPeak <= 2 * best, "the greedy plan must stay within twice the optimum: " + greedyPeak
                + " vs " + best);

        // --- equal sizes: the free list is optimal ----------------------------
        Graph uniform = diamond(16, 16, 16, 16);
        int[] uniformOrder = uniform.greedyOrder();
        long uniformPeak = uniform.peakBytes(uniformOrder, new String[1]);
        check(uniformPeak == uniform.maxLiveBytes(uniformOrder),
                "with equal sizes the peak equals the maximum live bytes: " + uniformPeak + " vs "
                        + uniform.maxLiveBytes(uniformOrder));
        long uniformBest = Long.MAX_VALUE;
        for (int[] order : uniform.allOrders()) uniformBest = Math.min(uniformBest, uniform.maxLiveBytes(order));
        check(uniformPeak <= 2 * uniformBest, "equal sizes must stay near the best order: " + uniformPeak
                + " vs " + uniformBest);

        // --- determinism -------------------------------------------------------
        check(Arrays.equals(small.greedyOrder(), greedySmall), "the greedy order must be deterministic");
        check(small.peakBytes(greedySmall, null) == greedyPeak, "and so must its peak");

        System.out.println("memory planner: greedy release-first order plus a first-fit free list");
        System.out.println("diamond(16,8,8,24): greedy peak " + greedyPeak + ", brute-force best " + best
                + " over " + orders.size() + " topological orders, naive total "
                + (4 * 8 + 16 + 24));
        int[] chainOrder = graphs[0].greedyOrder();
        System.out.println("chain(8,24) peak " + graphs[0].peakBytes(chainOrder, null) + " against the "
                + "liveness lower bound " + graphs[0].maxLiveBytes(chainOrder));
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
