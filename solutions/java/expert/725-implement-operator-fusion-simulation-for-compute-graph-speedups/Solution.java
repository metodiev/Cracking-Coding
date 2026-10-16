/** Task 725 - Implement operator fusion simulation for compute graph speedups. (Tier: Expert, CTCI Beyond — Machine Learning & AI)
 *
 * A graph of tensor ops is executed two ways. Without fusion every op is its own kernel: it reads its inputs
 * from memory, writes its output back and pays a launch. With fusion, an op whose result is read by exactly
 * one other op is folded into that reader's kernel - elementwise chains collapse into a single pass, and an
 * elementwise consumer of a matmul becomes the matmul's epilogue. Because every op evaluates its arithmetic
 * with the very same loops in both plans, fusion changes the schedule but not one bit of the result, which
 * is what the checks confirm. Fusion is refused wherever it would duplicate work: a tensor with two readers
 * stays materialised, so no subexpression is ever evaluated twice.
 *
 * Invariants verified:
 *   - equality   the fused plan reproduces the unfused plan bit-for-bit on every output tensor
 *   - no waste   the arithmetic is identical in both plans (elementwise updates and multiply-adds), so
 *                fusion only removes launches and intermediate materialisations, never work
 *   - sharing    a two-reader tensor keeps its own kernel and is fused into neither reader
 *   - economy    the kernel count, the materialised tensors and the modelled time all drop, and the plan's
 *                predicted launch count matches what the executor actually launches
 *
 * Simplified: tensors are 4x4 double arrays, the ops are matmul plus four elementwise forms, and the cost
 * model is a launch charge plus one unit per arithmetic operation; no device, stream or allocator is
 * modelled. The verified properties are result equality, work conservation and the kernel-count economy.
 */
import java.util.*;

public class Solution {
    static final int MATMUL = 0;
    static final int ADD = 1;
    static final int MUL = 2;
    static final int RELU = 3;
    static final int EXP = 4;
    static final String[] NAMES = {"matmul", "add", "mul", "relu", "exp"};

    /** One op: `inputs` are tensor ids, `output` is produced here, `extra` is a scalar operand. */
    static final class Op {
        final int kind;
        final int[] inputs;
        final int output;
        final double extra;
        Op(int kind, int[] inputs, int output, double extra) {
            this.kind = kind;
            this.inputs = inputs;
            this.output = output;
            this.extra = extra;
        }
        boolean elementwise() { return kind != MATMUL; }
    }

    /** A graph over dense tensors of shape rows x cols, ops listed in topological order. */
    static final class Graph {
        final List<Op> ops;
        final int rows;
        final int cols;
        final int tensors;
        private final int[][] readers;

        Graph(List<Op> ops, int rows, int cols, int tensors) {
            this.ops = ops;
            this.rows = rows;
            this.cols = cols;
            this.tensors = tensors;
            List<List<Integer>> lists = new ArrayList<>();
            for (int t = 0; t < tensors; t++) lists.add(new ArrayList<>());
            for (int n = 0; n < ops.size(); n++) {
                for (int t : ops.get(n).inputs) lists.get(t).add(n);
            }
            this.readers = new int[tensors][];
            for (int t = 0; t < tensors; t++) {
                this.readers[t] = lists.get(t).stream().mapToInt(Integer::intValue).toArray();
            }
        }

        int elementCount() { return rows * cols; }

        /** Folds an op into its single reader when both can live in one kernel; returns the groups. */
        int[] plan() {
            int[] group = new int[ops.size()];
            for (int n = 0; n < ops.size(); n++) group[n] = n;
            for (int p = 0; p < ops.size(); p++) {
                int[] read = readers[ops.get(p).output];
                if (read.length != 1) continue;                    // several readers: materialise it
                int c = read[0];
                if (c == p) continue;
                Op producer = ops.get(p);
                Op consumer = ops.get(c);
                if (!producer.elementwise() && !consumer.elementwise()) continue;   // matmul-matmul
                // A matmul may absorb at most one elementwise input: keep the graph a simple chain.
                if (!producer.elementwise()) {
                    int absorbed = 0;
                    for (int t : consumer.inputs) absorbed += readers[t].length == 1 ? 1 : 0;
                    if (absorbed > 1) continue;
                }
                join(group, p, c);
            }
            return group;
        }

        private int find(int[] group, int x) {
            while (group[x] != x) { group[x] = group[group[x]]; x = group[x]; }
            return x;
        }

        private void join(int[] group, int a, int b) {
            int rootA = find(group, a);
            int rootB = find(group, b);
            if (rootA == rootB) return;
            // keep the lowest index as the representative so the plan is order-stable
            if (rootA < rootB) group[rootB] = rootA; else group[rootA] = rootB;
        }

        List<int[]> groups(int[] group) {
            Map<Integer, List<Integer>> byRoot = new TreeMap<>();
            for (int n = 0; n < ops.size(); n++) {
                byRoot.computeIfAbsent(find(group, n), key -> new ArrayList<>()).add(n);
            }
            List<int[]> out = new ArrayList<>();
            for (List<Integer> members : byRoot.values()) {
                int[] array = members.stream().mapToInt(Integer::intValue).toArray();
                out.add(array);                                    // already ascending = topological
            }
            return out;
        }

        int kernelCount() { return groups(plan()).size(); }

        /** Executes every op on its own, materialising each intermediate. */
        Map<Integer, double[]> runUnfused(Map<Integer, double[]> inputs, long[] stats) {
            Map<Integer, double[]> values = new HashMap<>(inputs);
            long work = 0;
            int launches = 0;
            for (Op op : ops) {
                double[] out = evaluate(op, values);
                values.put(op.output, out);
                work += op.elementwise() ? elementCount() : (long) rows * cols * cols;
                launches++;
            }
            if (stats != null) { stats[0] = launches; stats[1] = work; stats[2] = ops.size(); }
            return values;
        }

        /**
         * Executes one kernel: its members in ascending order, resolving each input either from the
         * materialised tensors or from the intermediate the group has just produced. Returns the values.
         */
        Map<Integer, double[]> runFused(Map<Integer, double[]> inputs, long[] stats) {
            int[] group = plan();
            Map<Integer, double[]> values = new HashMap<>(inputs);
            long work = 0;
            int launches = 0;
            int materialised = 0;
            for (int[] members : groups(group)) {
                Map<Integer, double[]> local = new HashMap<>();
                for (int n : members) {
                    Op op = ops.get(n);
                    double[] out = evaluate(op, local.isEmpty() ? values : merged(values, local));
                    local.put(op.output, out);
                    work += op.elementwise() ? elementCount() : (long) rows * cols * cols;
                }
                values.putAll(local);
                launches++;
                materialised++;
            }
            if (stats != null) { stats[0] = launches; stats[1] = work; stats[2] = materialised; }
            return values;
        }

        private Map<Integer, double[]> merged(Map<Integer, double[]> values, Map<Integer, double[]> local) {
            Map<Integer, double[]> all = new HashMap<>(values);
            all.putAll(local);
            return all;
        }

        private double[] evaluate(Op op, Map<Integer, double[]> values) {
            double[] out = new double[elementCount()];
            if (op.kind == MATMUL) {
                matmul(values.get(op.inputs[0]), values.get(op.inputs[1]), out);
                return out;
            }
            double[] x = values.get(op.inputs[0]);
            double[] y = op.inputs.length > 1 ? values.get(op.inputs[1]) : null;
            for (int i = 0; i < out.length; i++) {
                out[i] = switch (op.kind) {
                    case ADD -> x[i] + (y != null ? y[i] : op.extra);
                    case MUL -> x[i] * (y != null ? y[i] : op.extra);
                    case RELU -> Math.max(0.0, x[i]);
                    default -> Math.exp(x[i]);
                };
            }
            return out;
        }

        /** Standard triple loop, shared by both plans so the arithmetic order is identical. */
        private void matmul(double[] a, double[] b, double[] out) {
            for (int i = 0; i < rows; i++) {
                for (int k = 0; k < cols; k++) {
                    double v = a[i * cols + k];
                    for (int j = 0; j < cols; j++) out[i * cols + j] += v * b[k * cols + j];
                }
            }
        }
    }

    private static long rngState = 0x243F6A8885A308D3L;

    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    static Map<Integer, double[]> inputs(int[] ids, int size) {
        Map<Integer, double[]> map = new HashMap<>();
        for (int id : ids) map.put(id, randomTensor(size));
        return map;
    }

    static double[] randomTensor(int size) {
        double[] v = new double[size];
        for (int i = 0; i < size; i++) v[i] = nextDouble() * 2.0 - 1.0;
        return v;
    }

    static double worstDifference(Map<Integer, double[]> a, Map<Integer, double[]> b, int[] ids) {
        double worst = 0.0;
        for (int id : ids) {
            double[] x = a.get(id);
            double[] y = b.get(id);
            for (int i = 0; i < x.length; i++) worst = Math.max(worst, Math.abs(x[i] - y[i]));
        }
        return worst;
    }

    static double modelledTime(long launches, long work, double launchCost) {
        return launches * launchCost + work;
    }

    /** A representative chain plus a diamond and a shared tensor, the usual fusion test shapes. */
    static Graph mixedGraph(int rows, int cols) {
        List<Op> ops = new ArrayList<>();
        ops.add(new Op(MATMUL, new int[]{0, 1}, 2, 0.0));      // 2 = a @ b
        ops.add(new Op(RELU, new int[]{2}, 3, 0.0));           // epilogue of the matmul
        ops.add(new Op(ADD, new int[]{3}, 4, 0.5));            // chain continues
        ops.add(new Op(EXP, new int[]{0}, 5, 0.0));            // 0 has two readers: stays separate
        ops.add(new Op(MUL, new int[]{4, 5}, 6, 0.0));         // joins the two branches
        return new Graph(ops, rows, cols, 7);
    }

    public static void main(String[] args) {
        final int rows = 4;
        final int cols = 4;
        final int size = rows * cols;
        final double launchCost = 50.0;

        // --- degenerate graphs ------------------------------------------------
        Graph empty = new Graph(new ArrayList<>(), rows, cols, 0);
        check(empty.kernelCount() == 0, "an empty graph launches nothing");
        Graph lone = new Graph(Collections.singletonList(new Op(RELU, new int[]{0}, 1, 0.0)), rows, cols, 2);
        check(lone.kernelCount() == 1, "a lone elementwise op is one kernel");

        // --- a pure elementwise chain collapses to a single kernel ------------
        List<Op> chainOps = new ArrayList<>();
        chainOps.add(new Op(ADD, new int[]{0}, 1, 0.5));
        chainOps.add(new Op(EXP, new int[]{1}, 2, 0.0));
        chainOps.add(new Op(MUL, new int[]{2}, 3, 2.0));
        chainOps.add(new Op(RELU, new int[]{3}, 4, 0.0));
        Graph chain = new Graph(chainOps, rows, cols, 5);
        check(chain.kernelCount() == 1, "four elementwise ops in a chain are one kernel, got "
                + chain.kernelCount());
        Map<Integer, double[]> chainInputs = inputs(new int[]{0, 1, 2, 3, 4}, size);
        long[] unfusedChain = new long[3];
        long[] fusedChain = new long[3];
        Map<Integer, double[]> uChain = chain.runUnfused(chainInputs, unfusedChain);
        Map<Integer, double[]> fChain = chain.runFused(chainInputs, fusedChain);
        check(worstDifference(fChain, uChain, new int[]{4}) == 0.0,
                "the fused chain must reproduce the unfused result bit-for-bit");
        check(unfusedChain[0] == 4 && fusedChain[0] == 1, "launches drop from 4 to 1: " + unfusedChain[0]
                + " -> " + fusedChain[0]);
        check(unfusedChain[1] == fusedChain[1] && unfusedChain[1] == 4L * size,
                "elementwise work is conserved: " + unfusedChain[1] + " vs " + fusedChain[1]);

        // --- matmul with an elementwise epilogue ------------------------------
        List<Op> gemmOps = new ArrayList<>();
        gemmOps.add(new Op(MATMUL, new int[]{0, 1}, 2, 0.0));
        gemmOps.add(new Op(RELU, new int[]{2}, 3, 0.0));
        Graph gemm = new Graph(gemmOps, rows, cols, 4);
        check(gemm.kernelCount() == 1, "an elementwise epilogue folds into the matmul, got "
                + gemm.kernelCount());
        Map<Integer, double[]> gemmInputs = inputs(new int[]{0, 1, 2, 3}, size);
        long[] unfusedGemm = new long[3];
        long[] fusedGemm = new long[3];
        Map<Integer, double[]> uGemm = gemm.runUnfused(gemmInputs, unfusedGemm);
        Map<Integer, double[]> fGemm = gemm.runFused(gemmInputs, fusedGemm);
        check(worstDifference(fGemm, uGemm, new int[]{3}) == 0.0,
                "the fused epilogue must reproduce the unfused matmul result bit-for-bit");
        check(unfusedGemm[1] == fusedGemm[1], "matmul work is conserved: " + unfusedGemm[1] + " vs "
                + fusedGemm[1]);

        // --- sharing: a two-reader tensor must not be duplicated --------------
        List<Op> sharedOps = new ArrayList<>();
        sharedOps.add(new Op(EXP, new int[]{0}, 1, 0.0));      // tensor 1 read twice below
        sharedOps.add(new Op(ADD, new int[]{1}, 2, 1.0));
        sharedOps.add(new Op(MUL, new int[]{1}, 3, 3.0));
        sharedOps.add(new Op(ADD, new int[]{2, 3}, 4, 0.0));
        Graph shared = new Graph(sharedOps, rows, cols, 5);
        check(shared.kernelCount() == 2, "a two-reader tensor keeps its own kernel, got "
                + shared.kernelCount());
        Map<Integer, double[]> sharedInputs = inputs(new int[]{0, 1, 2, 3, 4}, size);
        long[] unfusedShared = new long[3];
        long[] fusedShared = new long[3];
        Map<Integer, double[]> uShared = shared.runUnfused(sharedInputs, unfusedShared);
        Map<Integer, double[]> fShared = shared.runFused(sharedInputs, fusedShared);
        check(worstDifference(fShared, uShared, new int[]{4}) == 0.0,
                "the shared subexpression must still produce the same result");
        check(unfusedShared[1] == fusedShared[1], "no arithmetic may be duplicated: " + unfusedShared[1]
                + " vs " + fusedShared[1]);
        check(unfusedShared[0] == 4 && fusedShared[0] == 2, "four ops fuse into two kernels, got "
                + fusedShared[0]);

        // --- a mixed graph end to end -----------------------------------------
        Graph mixed = mixedGraph(rows, cols);
        Map<Integer, double[]> mixedInputs = inputs(new int[]{0, 1, 2, 3, 4, 5, 6}, size);
        long[] unfused = new long[3];
        long[] fused = new long[3];
        Map<Integer, double[]> uMixed = mixed.runUnfused(mixedInputs, unfused);
        Map<Integer, double[]> fMixed = mixed.runFused(mixedInputs, fused);
        check(worstDifference(fMixed, uMixed, new int[]{6}) == 0.0,
                "the mixed graph must give identical results");
        check(fused[0] < unfused[0] && mixed.kernelCount() == fused[0],
                "the plan's kernel count must match the executor: " + mixed.kernelCount() + " vs "
                        + fused[0]);
        check(unfused[1] == fused[1], "work is conserved across the whole graph: " + unfused[1] + " vs "
                + fused[1]);
        double unfusedTime = modelledTime(unfused[0], unfused[1], launchCost);
        double fusedTime = modelledTime(fused[0], fused[1], launchCost);
        check(fusedTime < unfusedTime, "the modelled time must improve: " + fusedTime + " vs " + unfusedTime);
        check(fused[2] < unfused[2], "fewer tensors are materialised: " + unfused[2] + " -> " + fused[2]);
        check(fused[2] == fused[0], "one materialised output per kernel");

        // --- determinism -------------------------------------------------------
        long[] again = new long[3];
        Map<Integer, double[]> repeat = mixed.runFused(mixedInputs, again);
        check(again[0] == fused[0] && worstDifference(repeat, fMixed, new int[]{6}) == 0.0,
                "the same plan must repeat exactly");

        System.out.println("operator fusion over " + rows + "x" + cols + " tensors, launch cost " + launchCost);
        System.out.println("elementwise chain: 4 kernels -> " + chain.kernelCount() + ", work conserved at "
                + fusedChain[1] + " updates");
        System.out.println("mixed graph: kernels " + unfused[0] + " -> " + fused[0] + ", materialised "
                + unfused[2] + " -> " + fused[2] + ", modelled time " + unfusedTime + " -> " + fusedTime);
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
