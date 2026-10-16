/** Task 933 - Implement runtime that compiles high-level ops into fused kernels with autotuning.
 *  (Tier: Master, CTCI Beyond - Compilers, Languages & Solvers)
 *
 * A tiny tracing runtime for a graph of high-level ops (source, bias add, ReLU, scalar multiply, square and a
 * matmul). Compilation has two passes. Fusion merges a producer-consumer chain of elementwise ops into one
 * kernel whenever the producer has exactly one consumer and the consumer has exactly one input; a node with
 * several consumers and every matmul are fusion boundaries, because a matmul is not element-local. Autotuning
 * then picks a tile size per kernel by hill-climbing the deterministic cost model
 *   cost(size, stages, tile) = size * (1 + 0.05 * stages) + size / tile * 6 + tile * 1.5,
 * which penalises both loop overhead (tiny tiles) and working-set blow-up (huge tiles). Wall-clock timing is
 * deliberately never used, so the compiled artifact is reproducible. The fused kernel walks the buffer once and
 * applies every stage element by element, which is bit-identical to materialising each stage.
 * Verified: the fused execution is bit-identical to the unfused interpreter; the whole chain becomes one kernel
 * and the count drops; a fan-out and a matmul each stop fusion; the hill-climb agrees with the exhaustive tile
 * scan; the config cache turns a recompile into a lookup; empty, single-op and repeated compiles behave.
 */
import java.util.*;

/** One op in the traced graph. */
final class Node {
    final String op;
    final int[] inputs;
    final int size;
    final double scalar;

    Node(String op, int[] inputs, int size, double scalar) {
        this.op = op;
        this.inputs = inputs;
        this.size = size;
        this.scalar = scalar;
    }

    static boolean elementwise(String op) { return !op.equals("matmul"); }
}

/** A compiled kernel: a fused stage list, one traversal of the buffer, and its tuned tile size. */
final class Kernel {
    final List<Node> stages = new ArrayList<>();
    final int size;
    int tile;

    Kernel(int size) { this.size = size; }

    double[] apply(double[] source) {
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            double value = first(stages.get(0), i, source);
            for (int s = 1; s < stages.size(); s++) value = next(stages.get(s), value);
            out[i] = value;
        }
        return out;
    }

    private double first(Node node, int i, double[] source) {
        if (node.op.equals("source")) return source[i];
        return next(node, source[i]);
    }

    private double next(Node node, double previous) {
        switch (node.op) {
            case "bias": return previous + node.scalar;
            case "relu": return Math.max(0, previous);
            case "scale": return previous * node.scalar;
            case "square": return previous * previous;
            default: throw new IllegalStateException("op is not element-local: " + node.op);
        }
    }
}

/** The traced graph: nodes, an unfused interpreter and a fusing compiler. */
final class Runtime {
    final List<Node> nodes = new ArrayList<>();
    final Map<Integer, int[]> consumers = new HashMap<>();
    final List<Kernel> artifacts = new ArrayList<>();
    final Map<String, Integer> configCache = new HashMap<>();
    int tunerProbes, cacheHits, opaque;

    int add(String op, int[] inputs, int size, double scalar) {
        nodes.add(new Node(op, inputs, size, scalar));
        return nodes.size() - 1;
    }

    /** Unfused reference interpreter: every node materialises its own buffer. */
    double[] interpret(double[] source) {
        Map<Integer, double[]> values = new HashMap<>();
        values.put(0, source);
        for (int id = 1; id < nodes.size(); id++) {
            Node node = nodes.get(id);
            if (node.op.equals("matmul")) {
                double[] left = values.get(node.inputs[0]), right = values.get(node.inputs[1]);
                double[] out = new double[node.size];
                int width = right.length;
                for (int i = 0; i < node.size; i++) {
                    double sum = 0;
                    for (int k = 0; k < width; k++) sum += left[i * width + k] * right[k];
                    out[i] = sum;
                }
                values.put(id, out);
                continue;
            }
            double[] in = values.get(node.inputs[0]);
            double[] out = new double[node.size];
            for (int i = 0; i < node.size; i++) {
                switch (node.op) {
                    case "bias": out[i] = in[i] + node.scalar; break;
                    case "relu": out[i] = Math.max(0, in[i]); break;
                    case "scale": out[i] = in[i] * node.scalar; break;
                    case "square": out[i] = in[i] * in[i]; break;
                    default: throw new IllegalStateException(node.op);
                }
            }
            values.put(id, out);
        }
        return values.get(nodes.size() - 1);
    }

    /** Fusion pass: single-consumer elementwise chains become one kernel. */
    List<Kernel> compile() {
        artifacts.clear();
        countConsumers();
        boolean[] claimed = new boolean[nodes.size()];
        for (int id = 0; id < nodes.size(); id++) {
            if (claimed[id]) continue;
            if (!Node.elementwise(nodes.get(id).op)) {              // a matmul stays on the dense matmul path
                opaque++;
                continue;
            }
            Kernel kernel = new Kernel(nodes.get(id).size);
            kernel.stages.add(nodes.get(id));
            claimed[id] = true;
            int current = id;
            while (true) {
                int[] users = consumers.getOrDefault(current, new int[0]);
                if (users.length != 1) break;                       // fan-out ends the chain
                Node next = nodes.get(users[0]);
                if (!Node.elementwise(next.op) || next.inputs.length != 1) break;
                kernel.stages.add(next);
                claimed[users[0]] = true;
                current = users[0];
            }
            kernel.tile = tune(kernel);
            artifacts.add(kernel);
        }
        return artifacts;
    }

    void countConsumers() {
        consumers.clear();
        for (int id = 1; id < nodes.size(); id++) {
            for (int input : nodes.get(id).inputs) {
                int[] old = consumers.getOrDefault(input, new int[0]);
                int[] grown = Arrays.copyOf(old, old.length + 1);
                grown[old.length] = id;
                consumers.put(input, grown);
            }
        }
    }

    /** Deterministic cost model of one fused kernel: an explicit formula, never a stopwatch. */
    static double costModel(int size, int stages, int tile) {
        return size * (1 + 0.05 * stages) + (double) size / tile * 6 + tile * 1.5;
    }

    /** The same work without fusion: one pass per stage. */
    static double unfusedCost(int size, int stages, int tile) { return stages * costModel(size, 1, tile); }

    /** Hill-climb over the tile space with the config cache in front of it. */
    int tune(Kernel kernel) {
        String key = kernel.size + "x" + kernel.stages.size();
        Integer cached = configCache.get(key);
        if (cached != null) {
            cacheHits++;
            return cached;
        }
        int[] tiles = {4, 8, 16, 32, 64};
        int best = tiles[0];
        double bestCost = costModel(kernel.size, kernel.stages.size(), best);
        for (int i = 1; i < tiles.length; i++) {
            double cost = costModel(kernel.size, kernel.stages.size(), tiles[i]);
            tunerProbes++;
            if (cost < bestCost) {
                bestCost = cost;
                best = tiles[i];
            } else {
                break;                                        // the model is unimodal: stop at the first rise
            }
        }
        configCache.put(key, best);
        return best;
    }

    /** Exhaustive scan of the same tile space - the oracle for the hill-climb. */
    static int exhaustiveTile(int size, int stages) {
        int[] tiles = {4, 8, 16, 32, 64};
        int best = tiles[0];
        for (int tile : tiles) if (costModel(size, stages, tile) < costModel(size, stages, best)) best = tile;
        return best;
    }
}

public class Solution {
    static final int SIZE = 24;

    static double[] vector(long seed, int n) {
        double[] out = new double[n];
        long state = seed;
        for (int i = 0; i < n; i++) {
            state = state * 6364136223846793005L + 1442695040888963407L;
            out[i] = ((state >>> 11) * 0x1.0p-53 - 0.5) * 4;
        }
        return out;
    }

    /** Source -> bias -> relu -> scale -> square -> bias: the fusion showcase. */
    static Runtime chainGraph() {
        Runtime runtime = new Runtime();
        runtime.add("source", new int[0], SIZE, 0);
        runtime.add("bias", new int[] {0}, SIZE, 0.5);
        runtime.add("relu", new int[] {1}, SIZE, 0);
        runtime.add("scale", new int[] {2}, SIZE, -1.5);
        runtime.add("square", new int[] {3}, SIZE, 0);
        runtime.add("bias", new int[] {4}, SIZE, 0.25);
        return runtime;
    }

    public static void main(String[] args) {
        Runtime runtime = chainGraph();
        double[] source = vector(0x933L, SIZE);
        double[] interpreted = runtime.interpret(source);
        List<Kernel> kernels = runtime.compile();
        double[] fused = kernels.get(0).apply(source);
        double maxGap = 0;
        for (int i = 0; i < SIZE; i++) maxGap = Math.max(maxGap, Math.abs(interpreted[i] - fused[i]));

        System.out.println("runtime: " + runtime.nodes.size() + " graph ops -> " + kernels.size() + " kernel of "
                + kernels.get(0).stages.size() + " stages, tile " + kernels.get(0).tile + ", tuner probes "
                + runtime.tunerProbes);
        System.out.println("fusion gap " + String.format("%.2e", maxGap) + ", unfused modelled cost "
                + String.format("%.1f", Runtime.unfusedCost(SIZE, runtime.nodes.size() - 1, kernels.get(0).tile))
                + " vs fused " + String.format("%.1f", Runtime.costModel(SIZE, kernels.get(0).stages.size() - 1, kernels.get(0).tile))
                + ", exhaustive optimum " + Runtime.exhaustiveTile(SIZE, runtime.nodes.size() - 1));

        check(maxGap == 0, "the fused kernel must be bit-identical to the interpreter, gap " + maxGap);
        check(kernels.size() == 1 && kernels.get(0).stages.size() == runtime.nodes.size(),
                "the whole chain must fuse into one kernel, got " + kernels.size() + " kernels");
        check(kernels.get(0).tile == Runtime.exhaustiveTile(SIZE, runtime.nodes.size() - 1),
                "the hill-climb must agree with the exhaustive optimum: " + kernels.get(0).tile);
        check(runtime.nodes.size() > kernels.size(), "fusion must reduce the kernel count");
        check(Runtime.costModel(SIZE, kernels.get(0).stages.size() - 1, kernels.get(0).tile)
                < Runtime.unfusedCost(SIZE, runtime.nodes.size() - 1, kernels.get(0).tile),
                "one fused pass must be cheaper than one pass per stage");

        // The exhaustive scan really is the minimum, and the model is U-shaped in the tile size.
        int bestTile = Runtime.exhaustiveTile(4096, 2);
        for (int tile : new int[] {4, 8, 16, 32, 64}) {
            check(Runtime.costModel(4096, 2, bestTile) <= Runtime.costModel(4096, 2, tile) + 1e-12,
                    "the exhaustive scan must return the cheapest tile, " + bestTile + " loses to " + tile);
        }
        check(Runtime.costModel(SIZE, 1, 4) > Runtime.costModel(SIZE, 1, 16)
                && Runtime.costModel(SIZE, 1, 64) > Runtime.costModel(SIZE, 1, 16),
                "the cost model must penalise both tiny and huge tiles");

        // Graph shape: a fan-out is a boundary, a matmul is a boundary, singletons stay singletons.
        Runtime boundary = new Runtime();
        boundary.add("source", new int[0], SIZE, 0);
        boundary.add("relu", new int[] {0}, SIZE, 0);
        boundary.add("scale", new int[] {1}, SIZE, 2.0);
        boundary.add("bias", new int[] {1}, SIZE, 1.0);
        List<Kernel> boundaryKernels = boundary.compile();
        check(boundaryKernels.size() == 3, "a fan-out must stop the chain at the branch point, got "
                + boundaryKernels.size() + " kernels");
        check(boundaryKernels.get(0).stages.size() == 2 && boundaryKernels.get(1).stages.size() == 1
                && boundaryKernels.get(2).stages.size() == 1,
                "the shared prefix stays fused and each branch becomes its own kernel");

        Runtime matmulGraph = new Runtime();
        matmulGraph.add("source", new int[0], 4, 0);
        matmulGraph.add("relu", new int[] {0}, 4, 0);
        matmulGraph.add("scale", new int[] {1}, 4, 3.0);
        matmulGraph.add("matmul", new int[] {2, 1}, 2, 0);
        check(matmulGraph.compile().size() >= 2, "a matmul must not be fused into an elementwise chain");
        check(matmulGraph.opaque == 1, "the matmul must be left to the dense matmul path, got " + matmulGraph.opaque);
        double[] matmulSource = {1, 2, 3, 4};
        double[] matmulOut = matmulGraph.artifacts.get(0).apply(matmulSource);
        check(matmulOut.length == 4, "the elementwise branch of the matmul graph keeps its shape");

        // Boundary: an empty graph compiles to nothing, one op to exactly one single-stage kernel.
        check(new Runtime().compile().isEmpty(), "an empty graph compiles to no kernels");
        Runtime single = new Runtime();
        single.add("source", new int[0], SIZE, 0);
        single.add("relu", new int[] {0}, SIZE, 0);
        List<Kernel> singleKernels = single.compile();
        check(singleKernels.size() == 1 && singleKernels.get(0).stages.size() == 2,
                "a source plus one op is a two-stage kernel, got " + singleKernels.size());

        // The cache: recompiling the same shapes must not re-tune.
        int probesBefore = runtime.tunerProbes, hitsBefore = runtime.cacheHits;
        double[] againFused = runtime.compile().get(0).apply(source);
        check(runtime.tunerProbes == probesBefore, "a repeated compile must not re-tune, probes " + runtime.tunerProbes);
        check(runtime.cacheHits > hitsBefore, "the config cache must serve the second compile");
        check(Arrays.equals(againFused, fused), "recompiling must reproduce the artifact");

        // Every tile in the space yields the same numbers; only the modelled cost differs.
        for (int tile : new int[] {4, 8, 16, 32, 64}) {
            Kernel variant = new Kernel(SIZE);
            variant.stages.addAll(kernels.get(0).stages);
            variant.tile = tile;
            check(Arrays.equals(variant.apply(source), fused), "tile " + tile + " must not change the result");
        }

        Runtime replay = chainGraph();
        replay.compile();
        check(replay.tunerProbes == runtime.tunerProbes && replay.artifacts.get(0).tile == kernels.get(0).tile,
                "compiling the same graph must be reproducible");

        System.out.println("shape checks: fan-out " + boundaryKernels.size() + " kernels, matmul graph "
                + matmulGraph.artifacts.size() + " kernels, source+relu " + singleKernels.get(0).stages.size()
                + " stages, empty 0 kernels, cache hits " + runtime.cacheHits);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
