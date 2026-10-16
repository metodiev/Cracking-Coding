/** Task 883 - Implement automatic differentiation for GPU kernels across fused operators. (Tier: Master, CTCI
 * Beyond - Compilers, Languages & Solvers)
 *
 * A reverse-mode autodiff engine (a tape of recorded operations, each carrying the rule that pushes an upstream
 * gradient into its parents) plus a forward-mode engine, applied to the fused GPU kernel chain
 *
 *     Z = X W + b        the matmul and the bias fold into one kernel
 *     A = relu(Z)        the activation, fused into the same kernel in a real pipeline
 *     P = softmax(A)     the fused cross-entropy operator consumes logits and targets directly, so neither P
 *     L = CE(P, y)       nor an intermediate gradient buffer is ever written back to memory
 *
 * The chain is implemented twice: as an unfused tape of four operators that materialises Z and A, and as a
 * single fused operator whose backward rule recomputes its intermediates from the inputs (recompute instead of
 * checkpointing). The two schedules must agree bit for bit, which is the correctness statement a GPU compiler
 * team relies on when it replaces the unfused schedule with the fused one.
 *
 * Verifications: the tape gradients are compared with the closed form softmax cross-entropy gradient, with
 * central finite differences, and with a second, independently written scalar (micrograd style) reverse-mode
 * engine; forward mode is checked against finite differences and against reverse mode through the dot-product
 * identity u . (J v) == (J^T u) . v; and the memory traffic of the two schedules is measured with an
 * instrumented tensor, showing that the fused schedule writes no activation tensor at all.
 *
 * SIMPLIFIED: tensors are tiny in-process double arrays, there is no GPU, no CUDA, no threads, no wall-clock
 * timing and no real kernel launch; "fusing" is modelled as one tape node whose backward recomputes the
 * intermediates into local scratch, and the memory cost claim counts instrumented element loads and stores
 * rather than cache misses. Randomness comes from splitmix64, so every run is bit-for-bit reproducible.
 *
 * VERIFIED: (a) CORRECTNESS - the fused and unfused schedules produce bit-identical losses and parameter and
 * input gradients; (b) ANALYTIC - the tape matches the closed form dL/dlogits = (P - onehot)/n, dW = X^T dZ and
 * db = column sums with the ReLU mask folded into dZ, and dX = dZ W^T; (c) NUMERICS - central finite
 * differences agree on randomly chosen weights,
 * biases and inputs, softmax rows sum to one, and the loss is invariant under a large constant shift of the
 * logits (log-sum-exp stability); (d) FORWARD MODE - J v matches finite differences, u . (J v) equals
 * (J^T u) . v, and every softmax tangent row sums to zero; (e) INDEPENDENT ENGINE - a separately written scalar
 * reverse-mode engine reproduces the loss and every parameter gradient; (f) COST - the unfused schedule writes
 * 3 n C activation elements while the fused schedule writes none, and the fused forward touches fewer memory
 * elements; (g) EDGE CASES - a dead ReLU layer, a ReLU sitting exactly on its kink, a batch of one, a single
 * class, and a repeated backward sweep are all handled deterministically.
 */
import java.util.*;

/** A dense row-major 2-D tensor. Element traffic is instrumented, so a schedule's memory cost can be counted. */
class Tensor {
    static long touches;                        // instrumented element loads plus stores
    static long writes;                         // the store subset: what actually leaves the kernel
    static boolean counting;                    // counted only around the forward passes we price

    final int rows, cols;
    final double[] data;

    Tensor(int rows, int cols) {
        if (rows <= 0 || cols <= 0) throw new IllegalArgumentException("bad tensor shape");
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows * cols];
    }

    static Tensor zero(int rows, int cols) { return new Tensor(rows, cols); }

    static Tensor filled(int rows, int cols, double value) {
        Tensor tensor = new Tensor(rows, cols);
        Arrays.fill(tensor.data, value);
        return tensor;
    }

    static Tensor of(int rows, int cols, double... values) {
        Tensor tensor = new Tensor(rows, cols);
        if (values.length != tensor.data.length) throw new IllegalArgumentException("bad values");
        System.arraycopy(values, 0, tensor.data, 0, values.length);
        return tensor;
    }

    double load(int r, int c) {
        if (counting) touches++;
        return data[r * cols + c];
    }

    void store(int r, int c, double value) {
        if (counting) { touches++; writes++; }
        data[r * cols + c] = value;
    }

    double peek(int r, int c) { return data[r * cols + c]; }        // library internal, not priced
    void poke(int r, int c, double value) { data[r * cols + c] = value; }

    void fill(double value) { Arrays.fill(data, value); }

    double sum() {
        double total = 0;
        for (double value : data) total += value;
        return total;
    }

    Tensor copy() {
        Tensor out = new Tensor(rows, cols);
        System.arraycopy(data, 0, out.data, 0, data.length);
        return out;
    }

    static double maxDifference(Tensor a, Tensor b) {
        double worst = 0;
        for (int i = 0; i < a.data.length; i++) worst = Math.max(worst, Math.abs(a.data[i] - b.data[i]));
        return worst;
    }
}

/** The backward rule of one recorded operation: it accumulates the upstream gradient into the parents. */
interface Rule {
    void backward(Tensor upstream, Node[] parents);
}

/** One value on the tape: a name, its parents, and the rule that maps an upstream gradient to theirs. */
class Node {
    final String name;
    final Tensor value;
    final Node[] parents;
    final Rule rule;
    private Tensor gradient;

    Node(String name, Tensor value, Node[] parents, Rule rule) {
        this.name = name;
        this.value = value;
        this.parents = parents;
        this.rule = rule;
    }

    Tensor grad() {
        if (gradient == null) gradient = Tensor.zero(value.rows, value.cols);
        return gradient;
    }

    boolean hasGradient() { return gradient != null; }
}

/** A reverse-mode tape: nodes are appended after their parents, so reverse insertion order is topological. */
class Tape {
    private final List<Node> order = new ArrayList<>();

    Node leaf(String name, Tensor value) { return record(name, value, new Node[0], null); }

    Node record(String name, Tensor value, Node[] parents, Rule rule) {
        Node node = new Node(name, value, parents, rule);
        order.add(node);
        return node;
    }

    /** Reverse sweep from a seed cotangent; gradients are cleared first so repeated sweeps cannot accumulate. */
    void backward(Node root, Tensor seed) {
        for (Node node : order) if (node.hasGradient()) node.grad().fill(0);
        Tensor gradient = root.grad();
        for (int i = 0; i < seed.data.length; i++)
            gradient.poke(i / seed.cols, i % seed.cols, seed.data[i]);
        for (int i = order.size() - 1; i >= 0; i--) {
            Node node = order.get(i);
            if (!node.hasGradient() || node.rule == null) continue;
            node.rule.backward(node.grad(), node.parents);
        }
    }

    void backward(Node root) { backward(root, Tensor.of(1, 1, 1.0)); }

    int size() { return order.size(); }
}

/** The recorded operations, including the fused operator whose backward recomputes its intermediates. */
class Ops {
    /** logits = W x + b for one row, accumulated in the order the matmul kernel uses. */
    static double[] logits(Tensor x, Tensor w, Tensor bias, int row) {
        int features = w.rows, classes = w.cols;
        double[] out = new double[classes];
        for (int j = 0; j < classes; j++) {
            double sum = 0;
            for (int k = 0; k < features; k++) sum += x.load(row, k) * w.load(k, j);
            out[j] = sum + bias.load(0, j);
        }
        return out;
    }

    /** Numerically stable softmax of one logit row: the max shift keeps the exponentials bounded. */
    static void softmax(double[] logits, double[] out) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : logits) max = Math.max(max, value);
        double total = 0;
        for (int j = 0; j < logits.length; j++) {
            out[j] = Math.exp(logits[j] - max);
            total += out[j];
        }
        for (int j = 0; j < out.length; j++) out[j] /= total;
    }

    static double[] softmax(double[] logits) {
        double[] out = new double[logits.length];
        softmax(logits, out);
        return out;
    }

    /** A = relu(Z). */
    static double[] relu(double[] values) {
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) out[i] = Math.max(0, values[i]);
        return out;
    }

    /** C = A B. */
    static Node matmul(Tape tape, Node a, Node b) {
        Tensor out = Tensor.zero(a.value.rows, b.value.cols);
        int inner = a.value.cols;
        for (int i = 0; i < out.rows; i++)
            for (int j = 0; j < out.cols; j++) {
                double sum = 0;
                for (int k = 0; k < inner; k++) sum += a.value.load(i, k) * b.value.load(k, j);
                out.store(i, j, sum);
            }
        return tape.record("matmul", out, new Node[] { a, b }, (upstream, parents) -> {
            Tensor da = parents[0].grad(), db = parents[1].grad();
            for (int i = 0; i < upstream.rows; i++)
                for (int k = 0; k < parents[0].value.cols; k++) {
                    double sum = 0;
                    for (int j = 0; j < upstream.cols; j++)
                        sum += upstream.load(i, j) * parents[1].value.load(k, j);
                    da.poke(i, k, da.peek(i, k) + sum);
                }
            for (int k = 0; k < parents[1].value.rows; k++)
                for (int j = 0; j < upstream.cols; j++) {
                    double sum = 0;
                    for (int i = 0; i < upstream.rows; i++)
                        sum += parents[0].value.load(i, k) * upstream.load(i, j);
                    db.poke(k, j, db.peek(k, j) + sum);
                }
        });
    }

    /** Z = C + b: the bias is added per column, so its gradient is the column sum. */
    static Node addBias(Tape tape, Node c, Node bias) {
        Tensor out = Tensor.zero(c.value.rows, c.value.cols);
        for (int i = 0; i < out.rows; i++)
            for (int j = 0; j < out.cols; j++) out.store(i, j, c.value.load(i, j) + bias.value.load(0, j));
        return tape.record("addBias", out, new Node[] { c, bias }, (upstream, parents) -> {
            Tensor dc = parents[0].grad(), db = parents[1].grad();
            for (int i = 0; i < upstream.rows; i++)
                for (int j = 0; j < upstream.cols; j++)
                    dc.poke(i, j, dc.peek(i, j) + upstream.load(i, j));
            for (int j = 0; j < upstream.cols; j++) {
                double sum = 0;
                for (int i = 0; i < upstream.rows; i++) sum += upstream.load(i, j);
                db.poke(0, j, db.peek(0, j) + sum);
            }
        });
    }

    /** A = relu(Z); the subgradient is zero exactly on the kink, the usual convention. */
    static Node relu(Tape tape, Node z) {
        Tensor out = Tensor.zero(z.value.rows, z.value.cols);
        for (int i = 0; i < out.rows; i++)
            for (int j = 0; j < out.cols; j++) out.store(i, j, Math.max(0, z.value.load(i, j)));
        return tape.record("relu", out, new Node[] { z }, (upstream, parents) -> {
            Tensor dz = parents[0].grad();
            for (int i = 0; i < upstream.rows; i++)
                for (int j = 0; j < upstream.cols; j++)
                    dz.poke(i, j, dz.peek(i, j) + upstream.load(i, j)
                            * (parents[0].value.load(i, j) > 0 ? 1.0 : 0.0));
        });
    }

    /** P = softmax(A): the softmax Jacobian is P (u - (u . P)) applied row by row. */
    static Node softmaxRows(Tape tape, Node a) {
        Tensor out = Tensor.zero(a.value.rows, a.value.cols);
        for (int i = 0; i < out.rows; i++) {
            double[] row = new double[out.cols];
            for (int j = 0; j < out.cols; j++) row[j] = a.value.load(i, j);
            double[] probabilities = softmax(row);
            for (int j = 0; j < out.cols; j++) out.store(i, j, probabilities[j]);
        }
        return tape.record("softmax", out, new Node[] { a }, (upstream, parents) -> {
            Tensor da = parents[0].grad();
            for (int i = 0; i < upstream.rows; i++) {
                double dot = 0;
                for (int j = 0; j < upstream.cols; j++) dot += upstream.peek(i, j) * out.peek(i, j);
                for (int j = 0; j < upstream.cols; j++)
                    da.poke(i, j, da.peek(i, j) + out.peek(i, j) * (upstream.peek(i, j) - dot));
            }
        });
    }

    /** The fused cross-entropy operator: it consumes logits plus integer targets and produces a scalar loss. */
    static Node softmaxCrossEntropy(Tape tape, Node logits, Tensor targets) {
        int batch = logits.value.rows;
        double inverse = 1.0 / batch;
        Tensor out = Tensor.zero(1, 1);
        double total = 0;
        for (int i = 0; i < batch; i++) {
            double[] row = new double[logits.value.cols];
            for (int j = 0; j < row.length; j++) row[j] = logits.value.load(i, j);
            double[] probabilities = softmax(row);
            total += -Math.log(Math.max(probabilities[(int) targets.peek(i, 0)], Double.MIN_NORMAL));
        }
        out.store(0, 0, total * inverse);
        return tape.record("softmaxCrossEntropy", out, new Node[] { logits }, (upstream, parents) -> {
            Tensor dl = parents[0].grad();
            double scale = upstream.peek(0, 0) * inverse;
            for (int i = 0; i < batch; i++) {
                double[] row = new double[logits.value.cols];
                for (int j = 0; j < row.length; j++) row[j] = parents[0].value.load(i, j);
                double[] probabilities = softmax(row);
                int target = (int) targets.peek(i, 0);
                for (int j = 0; j < row.length; j++)
                    dl.poke(i, j, dl.peek(i, j)
                            + scale * (probabilities[j] - (j == target ? 1.0 : 0.0)));
            }
        });
    }

    /** The fused operator: matmul, bias, ReLU and cross-entropy in one node, recomputing in its backward. */
    static Node fusedMlpCrossEntropy(Tape tape, Node x, Node w, Node bias, Tensor targets) {
        int batch = x.value.rows, features = x.value.cols, classes = w.value.cols;
        double inverse = 1.0 / batch;
        Tensor out = Tensor.zero(1, 1);
        double total = 0;
        for (int i = 0; i < batch; i++) {
            double[] probabilities = softmax(relu(logits(x.value, w.value, bias.value, i)));
            total += -Math.log(Math.max(probabilities[(int) targets.peek(i, 0)], Double.MIN_NORMAL));
        }
        out.store(0, 0, total * inverse);
        return tape.record("fusedMlpCrossEntropy", out, new Node[] { x, w, bias },
                (upstream, parents) -> {
                    Tensor dx = parents[0].grad(), dw = parents[1].grad(), db = parents[2].grad();
                    double scale = upstream.peek(0, 0) * inverse;
                    for (int i = 0; i < batch; i++) {
                        double[] row = logits(parents[0].value, parents[1].value, parents[2].value, i);
                        double[] probabilities = softmax(relu(row));
                        int target = (int) targets.peek(i, 0);
                        double[] dz = new double[classes];
                        for (int j = 0; j < classes; j++)
                            dz[j] = (row[j] > 0 ? 1.0 : 0.0)
                                    * (scale * (probabilities[j] - (j == target ? 1.0 : 0.0)));
                        for (int k = 0; k < features; k++) {
                            double sum = 0;
                            for (int j = 0; j < classes; j++) sum += dz[j] * parents[1].value.peek(k, j);
                            dx.poke(i, k, dx.peek(i, k) + sum);
                        }
                        for (int k = 0; k < features; k++)
                            for (int j = 0; j < classes; j++)
                                dw.poke(k, j, dw.peek(k, j) + parents[0].value.peek(i, k) * dz[j]);
                        for (int j = 0; j < classes; j++) db.poke(0, j, db.peek(0, j) + dz[j]);
                    }
                });
    }

    /** Forward mode: the directional derivative of P = softmax(relu(X W + b)) along the tangent (dW, db). */
    static Tensor jvpProbabilities(Tensor x, Tensor w, Tensor bias, Tensor dw, Tensor db) {
        int batch = x.rows, features = w.rows, classes = w.cols;
        Tensor out = Tensor.zero(batch, classes);
        for (int i = 0; i < batch; i++) {
            double[] row = logits(x, w, bias, i);
            double[] tangent = new double[classes];
            for (int j = 0; j < classes; j++) {
                double sum = 0;
                for (int k = 0; k < features; k++) sum += x.peek(i, k) * dw.peek(k, j);
                tangent[j] = row[j] > 0 ? sum + db.peek(0, j) : 0.0;
            }
            double[] probabilities = softmax(relu(row));
            double dot = 0;
            for (int j = 0; j < classes; j++) dot += tangent[j] * probabilities[j];
            for (int j = 0; j < classes; j++)
                out.poke(i, j, probabilities[j] * (tangent[j] - dot));
        }
        return out;
    }
}

/** An independent scalar reverse-mode engine (micrograd style), used as a second implementation of the AD. */
class Scalar {
    final String name;
    final double value;
    final List<Scalar> parents = new ArrayList<>();
    final List<Double> partials = new ArrayList<>();
    double gradient;

    Scalar(String name, double value) { this.name = name; this.value = value; }

    static Scalar of(String name, double value) { return new Scalar(name, value); }

    private static Scalar combine(String name, List<Scalar> parents, List<Double> partials, double value) {
        Scalar node = new Scalar(name, value);
        node.parents.addAll(parents);
        node.partials.addAll(partials);
        return node;
    }

    Scalar add(Scalar other) {
        return combine("add", List.of(this, other), List.of(1.0, 1.0), value + other.value);
    }

    Scalar subtract(Scalar other) {
        return combine("sub", List.of(this, other), List.of(1.0, -1.0), value - other.value);
    }

    Scalar multiply(Scalar other) {
        return combine("mul", List.of(this, other), List.of(other.value, value), value * other.value);
    }

    Scalar divide(Scalar other) {
        return combine("div", List.of(this, other),
                List.of(1.0 / other.value, -value / (other.value * other.value)), value / other.value);
    }

    Scalar exp() {
        double result = Math.exp(value);
        return combine("exp", List.of(this), List.of(result), result);
    }

    Scalar log() {
        return combine("log", List.of(this), List.of(1.0 / value), Math.log(value));
    }

    Scalar relu() {
        return combine("relu", List.of(this), List.of(value > 0 ? 1.0 : 0.0), Math.max(0, value));
    }

    /** Reverse mode by an explicit depth-first post-order: a different traversal from the tensor tape's. */
    static void backward(Scalar root) {
        List<Scalar> topological = new ArrayList<>();
        collect(root, topological, Collections.newSetFromMap(new IdentityHashMap<>()));
        for (Scalar node : topological) node.gradient = 0;
        root.gradient = 1.0;
        for (int i = topological.size() - 1; i >= 0; i--) {
            Scalar node = topological.get(i);
            for (int p = 0; p < node.parents.size(); p++)
                node.parents.get(p).gradient += node.gradient * node.partials.get(p);
        }
    }

    private static void collect(Scalar node, List<Scalar> order, Set<Scalar> seen) {
        if (!seen.add(node)) return;
        for (Scalar parent : node.parents) collect(parent, order, seen);
        order.add(node);
    }
}

/** The same MLP and cross-entropy written with the scalar engine, keeping the handles of the leaves. */
class ScalarModel {
    final int features, classes;
    final Scalar[][] weights;
    final Scalar[] biases;
    final Scalar loss;

    ScalarModel(Tensor x, Tensor w, Tensor bias, int[] targets) {
        int batch = x.rows;
        features = x.cols;
        classes = w.cols;
        weights = new Scalar[features][classes];
        for (int k = 0; k < features; k++)
            for (int j = 0; j < classes; j++) weights[k][j] = Scalar.of("w" + k + "_" + j, w.peek(k, j));
        biases = new Scalar[classes];
        for (int j = 0; j < classes; j++) biases[j] = Scalar.of("b" + j, bias.peek(0, j));
        Scalar total = Scalar.of("start", 0);
        for (int i = 0; i < batch; i++) {
            Scalar[] activated = new Scalar[classes];
            for (int j = 0; j < classes; j++) {
                Scalar sum = Scalar.of("row" + i + "_" + j, 0);
                for (int k = 0; k < features; k++)
                    sum = sum.add(weights[k][j].multiply(Scalar.of("x" + i + "_" + k, x.peek(i, k))));
                activated[j] = sum.add(biases[j]).relu();
            }
            Scalar max = Scalar.of("max" + i, activated[0].value);
            for (Scalar value : activated) if (value.value > max.value) max = value;
            Scalar denominator = Scalar.of("sumExp" + i, 0);
            for (Scalar value : activated) denominator = denominator.add(value.subtract(max).exp());
            total = total.add(max.add(denominator.log()).subtract(activated[targets[i]]));
        }
        loss = total.divide(Scalar.of("batch", batch));
    }

    double value() { return loss.value; }

    /** Every parameter gradient, flattened as W row-major followed by the bias. */
    double[] gradients() {
        Scalar.backward(loss);
        double[] out = new double[features * classes + classes];
        for (int k = 0; k < features; k++)
            for (int j = 0; j < classes; j++) out[k * classes + j] = weights[k][j].gradient;
        for (int j = 0; j < classes; j++) out[features * classes + j] = biases[j].gradient;
        return out;
    }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240617L);
        int batch = 8, features = 5, classes = 4;

        Tensor x = random(rng, batch, features, 0.5);
        Tensor w = random(rng, features, classes, 0.5);
        Tensor bias = random(rng, 1, classes, 0.25);
        int[] target = new int[batch];
        for (int i = 0; i < batch; i++) target[i] = (int) Math.floorMod(rng.nextLong(), classes);
        Tensor targets = targetsOf(target);

        // --- THE FUSED SCHEDULE --------------------------------------------------------------------------
        Tensor.counting = true;
        Tensor.touches = 0;
        Tensor.writes = 0;
        Tape fused = new Tape();
        Node fx = fused.leaf("x", x), fw = fused.leaf("w", w), fb = fused.leaf("b", bias);
        Node fusedLoss = Ops.fusedMlpCrossEntropy(fused, fx, fw, fb, targets);
        long fusedTouches = Tensor.touches, fusedWrites = Tensor.writes;
        Tensor.counting = false;
        fused.backward(fusedLoss);

        // --- THE UNFUSED SCHEDULE ------------------------------------------------------------------------
        Tensor.counting = true;
        Tensor.touches = 0;
        Tensor.writes = 0;
        Tape unfused = new Tape();
        Node ux = unfused.leaf("x", x), uw = unfused.leaf("w", w), ub = unfused.leaf("b", bias);
        Node activatedRow = Ops.relu(unfused, Ops.addBias(unfused, Ops.matmul(unfused, ux, uw), ub));
        Node unfusedLoss = Ops.softmaxCrossEntropy(unfused, activatedRow, targets);
        long unfusedTouches = Tensor.touches, unfusedWrites = Tensor.writes;
        Tensor.counting = false;
        unfused.backward(unfusedLoss);

        check(fusedLoss.value.peek(0, 0) == unfusedLoss.value.peek(0, 0),
                "the fused and unfused schedules must produce the same loss: " + fusedLoss.value.peek(0, 0)
                        + " against " + unfusedLoss.value.peek(0, 0));
        double parameterDifference = Math.max(Tensor.maxDifference(fw.grad(), uw.grad()),
                Tensor.maxDifference(fb.grad(), ub.grad()));
        check(parameterDifference == 0.0, "the fused and unfused parameter gradients must be bit-identical, "
                + "worst difference " + parameterDifference);
        check(Tensor.maxDifference(fx.grad(), ux.grad()) == 0.0,
                "the fused and unfused input gradients must be bit-identical, worst difference "
                        + Tensor.maxDifference(fx.grad(), ux.grad()));
        check(fused.size() + 3 == unfused.size(), "the fused schedule must replace four operators with one, "
                + "recording " + fused.size() + " nodes against " + unfused.size());
        check(fusedWrites == 1L && unfusedWrites == 3L * batch * classes + 1,
                "the unfused schedule must write three activation tensors and the fused one only the loss: "
                        + unfusedWrites + " against " + fusedWrites + " stores");
        check(fusedTouches < unfusedTouches, "fusing must reduce forward element traffic: " + fusedTouches
                + " against " + unfusedTouches + " touches");

        // --- THE CLOSED FORM GRADIENT --------------------------------------------------------------------
        double inverse = 1.0 / batch;
        Tensor logits = Tensor.zero(batch, classes);
        Tensor activated = Tensor.zero(batch, classes);
        Tensor mask = Tensor.zero(batch, classes);
        Tensor logitGradient = Tensor.zero(batch, classes);
        for (int i = 0; i < batch; i++) {
            double[] row = Ops.logits(x, w, bias, i);
            double[] probabilities = Ops.softmax(Ops.relu(row));
            for (int j = 0; j < classes; j++) {
                logits.poke(i, j, row[j]);
                activated.poke(i, j, Math.max(0, row[j]));
                mask.poke(i, j, row[j] > 0 ? 1.0 : 0.0);
                logitGradient.poke(i, j, inverse * (probabilities[j] - (j == target[i] ? 1.0 : 0.0)));
            }
        }
        Tensor dz = Tensor.zero(batch, classes);
        for (int i = 0; i < batch; i++)
            for (int j = 0; j < classes; j++) dz.poke(i, j, mask.peek(i, j) * logitGradient.peek(i, j));
        Tensor expectedW = Tensor.zero(features, classes), expectedB = Tensor.zero(1, classes);
        Tensor expectedX = Tensor.zero(batch, features);
        for (int k = 0; k < features; k++)
            for (int j = 0; j < classes; j++) {
                double sum = 0;
                for (int i = 0; i < batch; i++) sum += x.peek(i, k) * dz.peek(i, j);
                expectedW.poke(k, j, sum);
            }
        for (int j = 0; j < classes; j++) {
            double sum = 0;
            for (int i = 0; i < batch; i++) sum += dz.peek(i, j);
            expectedB.poke(0, j, sum);
        }
        for (int i = 0; i < batch; i++)
            for (int k = 0; k < features; k++) {
                double sum = 0;
                for (int j = 0; j < classes; j++) sum += dz.peek(i, j) * w.peek(k, j);
                expectedX.poke(i, k, sum);
            }
        double closedDifference = Math.max(Tensor.maxDifference(fw.grad(), expectedW),
                Math.max(Tensor.maxDifference(fb.grad(), expectedB), Tensor.maxDifference(fx.grad(), expectedX)));
        check(closedDifference < 1e-16, "the tape must match dW = X^T dZ, db = column sums and dX = dZ W^T with "
                + "the ReLU mask folded into dZ, worst difference " + closedDifference);

        // The fused cross-entropy operator's own gradient with respect to its logits.
        Tape logitTape = new Tape();
        Node logitLeaf = logitTape.leaf("activated", activated.copy());
        logitTape.backward(Ops.softmaxCrossEntropy(logitTape, logitLeaf, targets));
        check(Tensor.maxDifference(logitLeaf.grad(), logitGradient) < 1e-16,
                "dL/dlogits must be (P - onehot)/n, worst difference "
                        + Tensor.maxDifference(logitLeaf.grad(), logitGradient));
        double worstClassSum = 0;
        for (int i = 0; i < batch; i++) {
            double sum = 0;
            for (int j = 0; j < classes; j++) sum += logitGradient.peek(i, j);
            worstClassSum = Math.max(worstClassSum, Math.abs(sum));
        }
        check(worstClassSum < 1e-15, "the logit gradient of each sample must sum to zero over the classes: "
                + worstClassSum);

        // --- CENTRAL FINITE DIFFERENCES ------------------------------------------------------------------
        double epsilon = 1e-6;
        double worstRelative = 0;
        String worstWhere = "";
        for (int trial = 0; trial < 6; trial++) {
            int kind = trial % 3;
            Tensor parameter;
            Node leaf;
            int pr, pc;
            String label;
            if (kind == 0) {
                parameter = w; leaf = fw; label = "W";
                pr = (int) Math.floorMod(rng.nextLong(), features);
                pc = (int) Math.floorMod(rng.nextLong(), classes);
            } else if (kind == 1) {
                parameter = bias; leaf = fb; label = "b";
                pr = 0;
                pc = (int) Math.floorMod(rng.nextLong(), classes);
            } else {
                parameter = x; leaf = fx; label = "X";
                pr = (int) Math.floorMod(rng.nextLong(), batch);
                pc = (int) Math.floorMod(rng.nextLong(), features);
            }
            double original = parameter.peek(pr, pc);
            parameter.poke(pr, pc, original + epsilon);
            double plus = loss(x, w, bias, targets);
            parameter.poke(pr, pc, original - epsilon);
            double minus = loss(x, w, bias, targets);
            parameter.poke(pr, pc, original);
            double finiteDifference = (plus - minus) / (2 * epsilon);
            double analytic = leaf.grad().peek(pr, pc);
            double scale = Math.max(1e-9, Math.abs(finiteDifference) + Math.abs(analytic));
            double relative = Math.abs(finiteDifference - analytic) / scale;
            if (relative > worstRelative) {
                worstRelative = relative;
                worstWhere = label + "[" + pr + "][" + pc + "]: finite difference " + finiteDifference
                        + " against " + analytic;
            }
        }
        check(worstRelative < 1e-6, "central finite differences must match the tape, worst relative error "
                + worstRelative + " at " + worstWhere);

        // --- FORWARD MODE --------------------------------------------------------------------------------
        Tensor dw = random(rng, features, classes, 1.0), db = random(rng, 1, classes, 1.0);
        double worstForward = 0;
        for (int trial = 0; trial < 3; trial++) {
            int k = (int) Math.floorMod(rng.nextLong(), features);
            int j = (int) Math.floorMod(rng.nextLong(), classes);
            double original = w.peek(k, j);
            for (int index = 0; index < features; index++)
                for (int column = 0; column < classes; column++) dw.poke(index, column, 0);
            dw.poke(k, j, 1.0);
            for (int column = 0; column < classes; column++) db.poke(0, column, 0);
            w.poke(k, j, original + epsilon);
            Tensor plus = probabilities(x, w, bias);
            w.poke(k, j, original - epsilon);
            Tensor minus = probabilities(x, w, bias);
            w.poke(k, j, original);
            Tensor directional = Ops.jvpProbabilities(x, w, bias, dw, db);
            for (int i = 0; i < batch; i++)
                for (int c = 0; c < classes; c++)
                    worstForward = Math.max(worstForward,
                            Math.abs((plus.peek(i, c) - minus.peek(i, c)) / (2 * epsilon)
                                    - directional.peek(i, c)));
        }
        check(worstForward < 1e-6, "forward mode must match finite differences, worst absolute error "
                + worstForward);

        dw = random(rng, features, classes, 1.0);
        db = random(rng, 1, classes, 1.0);
        Tensor tangent = Ops.jvpProbabilities(x, w, bias, dw, db);
        Tensor cotangent = random(rng, batch, classes, 1.0);
        Tape reverse = new Tape();
        Node rx = reverse.leaf("x", x), rw = reverse.leaf("w", w), rb = reverse.leaf("b", bias);
        Node rp = Ops.softmaxRows(reverse, Ops.relu(reverse,
                Ops.addBias(reverse, Ops.matmul(reverse, rx, rw), rb)));
        reverse.backward(rp, cotangent);
        double left = 0, right = 0;
        for (int i = 0; i < batch; i++)
            for (int j = 0; j < classes; j++) left += cotangent.peek(i, j) * tangent.peek(i, j);
        for (int k = 0; k < features; k++)
            for (int j = 0; j < classes; j++) right += rw.grad().peek(k, j) * dw.peek(k, j);
        for (int j = 0; j < classes; j++) right += rb.grad().peek(0, j) * db.peek(0, j);
        double forwardScanPlus = dotWith(cotangent, probabilities(x, shifted(w, dw, epsilon),
                shifted(bias, db, epsilon)));
        double forwardScanMinus = dotWith(cotangent, probabilities(x, shifted(w, dw, -epsilon),
                shifted(bias, db, -epsilon)));
        double finiteDirectional = (forwardScanPlus - forwardScanMinus) / (2 * epsilon);
        check(Math.abs(left - right) < 1e-12 * Math.max(1.0, Math.abs(left)),
                "u . (J v) must equal (J^T u) . v: " + left + " against " + right);
        check(Math.abs(finiteDirectional - left) < 1e-6, "the forward-mode directional derivative must match "
                + "finite differences: " + finiteDirectional + " against " + left);
        Tensor probabilityRows = probabilities(x, w, bias);
        double worstRowSum = 0, worstTangentSum = 0;
        for (int i = 0; i < batch; i++) {
            double rowSum = 0, tangentSum = 0;
            for (int j = 0; j < classes; j++) {
                rowSum += probabilityRows.peek(i, j);
                tangentSum += tangent.peek(i, j);
            }
            worstRowSum = Math.max(worstRowSum, Math.abs(rowSum - 1));
            worstTangentSum = Math.max(worstTangentSum, Math.abs(tangentSum));
        }
        check(worstRowSum < 1e-15 && worstTangentSum < 1e-15,
                "every softmax row must sum to one and its tangent to zero: " + worstRowSum + " and "
                        + worstTangentSum);

        // --- LOG-SUM-EXP STABILITY AND AN INDEPENDENT ENGINE ---------------------------------------------
        Tensor oracleProbabilities = probabilities(x, w, bias);
        double oracleLoss = 0;
        for (int i = 0; i < batch; i++)
            oracleLoss += -Math.log(oracleProbabilities.peek(i, target[i]));
        oracleLoss *= inverse;
        check(Math.abs(oracleLoss - fusedLoss.value.peek(0, 0)) < 1e-15, "the loss must equal the independent "
                + "log-probability oracle: " + oracleLoss + " against " + fusedLoss.value.peek(0, 0));
        Tape plainTape = new Tape();
        Node plainLoss = Ops.softmaxCrossEntropy(plainTape, plainTape.leaf("logits", logits.copy()), targets);
        Tape shiftedTape = new Tape();
        Node shiftedLogits = shiftedTape.leaf("logits",
                shifted(logits, Tensor.filled(batch, classes, 1.0), 1e3));
        Node shiftedLoss = Ops.softmaxCrossEntropy(shiftedTape, shiftedLogits, targets);
        check(Math.abs(shiftedLoss.value.peek(0, 0) - plainLoss.value.peek(0, 0)) < 1e-12,
                "cross entropy must be invariant under a constant logit shift: " + shiftedLoss.value.peek(0, 0)
                        + " against " + plainLoss.value.peek(0, 0));
        int smallFeatures = 2, smallClasses = 3;
        Tensor smallX = random(rng, 4, smallFeatures, 0.5);
        Tensor smallW = random(rng, smallFeatures, smallClasses, 0.5);
        Tensor smallB = random(rng, 1, smallClasses, 0.25);
        int[] smallTargets = { 0, 2, 1, 0 };
        ScalarModel scalarModel = new ScalarModel(smallX, smallW, smallB, smallTargets);
        double[] scalarGradients = scalarModel.gradients();
        Tape smallTape = new Tape();
        Node smallWNode = smallTape.leaf("w", smallW), smallBNode = smallTape.leaf("b", smallB);
        Node smallLoss = Ops.fusedMlpCrossEntropy(smallTape, smallTape.leaf("x", smallX), smallWNode, smallBNode,
                targetsOf(smallTargets));
        smallTape.backward(smallLoss);
        double worstEngine = Math.abs(scalarModel.value() - smallLoss.value.peek(0, 0));
        for (int k = 0; k < smallFeatures; k++)
            for (int j = 0; j < smallClasses; j++)
                worstEngine = Math.max(worstEngine,
                        Math.abs(scalarGradients[k * smallClasses + j] - smallWNode.grad().peek(k, j)));
        for (int j = 0; j < smallClasses; j++)
            worstEngine = Math.max(worstEngine,
                    Math.abs(scalarGradients[smallFeatures * smallClasses + j] - smallBNode.grad().peek(0, j)));
        check(worstEngine < 1e-14, "the independent scalar engine must agree on the loss and every parameter "
                + "gradient, worst difference " + worstEngine);

        // --- EDGE CASES ----------------------------------------------------------------------------------
        Tape deadTape = new Tape();
        Node deadW = deadTape.leaf("w", Tensor.filled(features, classes, 0.0));
        Node deadB = deadTape.leaf("b", Tensor.filled(1, classes, -1.0));
        Node deadLoss = Ops.fusedMlpCrossEntropy(deadTape, deadTape.leaf("x", x), deadW, deadB, targets);
        deadTape.backward(deadLoss);
        check(deadW.grad().sum() == 0.0 && deadB.grad().sum() == 0.0
                        && Math.abs(deadLoss.value.peek(0, 0) - Math.log(classes)) < 1e-15,
                "a dead ReLU layer must give zero gradients and the uniform loss " + Math.log(classes)
                        + ", measured " + deadLoss.value.peek(0, 0) + " with gradient sum "
                        + deadW.grad().sum());
        Tape kinkTape = new Tape();
        Node kinkLeaf = kinkTape.leaf("z", Tensor.of(1, 4, -1.0, 0.0, 2.0, 0.0));
        kinkTape.backward(Ops.relu(kinkTape, kinkLeaf), Tensor.filled(1, 4, 1.0));
        double kinkError = 0;
        for (int j = 0; j < 4; j++) kinkError = Math.max(kinkError, Math.abs(kinkLeaf.grad().peek(0, j)
                - (kinkLeaf.value.peek(0, j) > 0 ? 1.0 : 0.0)));
        check(kinkError == 0.0, "the ReLU subgradient must be zero exactly on the kink, worst difference "
                + kinkError);
        Tensor oneX = random(rng, 1, features, 0.5);
        Tape oneTape = new Tape();
        Node oneLoss = Ops.fusedMlpCrossEntropy(oneTape, oneTape.leaf("x", oneX), oneTape.leaf("w", w),
                oneTape.leaf("b", bias), targetsOf(new int[] { 1 }));
        oneTape.backward(oneLoss);
        double[] singleProbabilities = Ops.softmax(Ops.relu(Ops.logits(oneX, w, bias, 0)));
        check(Math.abs(oneLoss.value.peek(0, 0) + Math.log(singleProbabilities[1])) < 1e-15,
                "a batch of one must reduce to that sample's cross entropy: " + oneLoss.value.peek(0, 0)
                        + " against " + -Math.log(singleProbabilities[1]));
        Tape classTape = new Tape();
        Node classW = classTape.leaf("w", random(rng, features, 1, 0.5));
        Node classB = classTape.leaf("b", random(rng, 1, 1, 0.25));
        Node classLoss = Ops.fusedMlpCrossEntropy(classTape, classTape.leaf("x", x), classW, classB,
                Tensor.filled(batch, 1, 0.0));
        classTape.backward(classLoss);
        check(classLoss.value.peek(0, 0) == 0.0 && classW.grad().sum() == 0.0 && classB.grad().sum() == 0.0,
                "a single class must give a zero loss and zero gradients, got " + classLoss.value.peek(0, 0)
                        + " and " + classW.grad().sum());

        // --- DETERMINISM ---------------------------------------------------------------------------------
        Tape repeated = new Tape();
        Node repeatedW = repeated.leaf("w", w);
        Node repeatedLoss = Ops.fusedMlpCrossEntropy(repeated, repeated.leaf("x", x), repeatedW,
                repeated.leaf("b", bias), targets);
        repeated.backward(repeatedLoss);
        check(Tensor.maxDifference(repeatedW.grad(), fw.grad()) == 0.0,
                "a second identical run must reproduce the gradients exactly");
        double before = repeatedW.grad().peek(0, 0);
        repeated.backward(repeatedLoss);
        check(repeatedW.grad().peek(0, 0) == before, "a repeated backward sweep must not accumulate twice: "
                + repeatedW.grad().peek(0, 0) + " against " + before);

        System.out.println("Fused AD: loss " + String.format(Locale.ROOT, "%.6f", fusedLoss.value.peek(0, 0))
                + ", gradient norm " + String.format(Locale.ROOT, "%.6e", norm(fw.grad(), fb.grad()))
                + ", worst finite-difference error " + String.format(Locale.ROOT, "%.2e", worstRelative));
        System.out.println("Schedules: fused " + fusedTouches + " touches / " + fusedWrites + " stores against "
                + unfusedTouches + " touches / " + unfusedWrites + " stores for " + batch + "x" + classes
                + " activations");
        System.out.println("Forward mode: worst JVP error " + String.format(Locale.ROOT, "%.2e", worstForward)
                + ", dot identity residual " + String.format(Locale.ROOT, "%.3e", Math.abs(left - right)));
        System.out.println("PASS");
    }

    /** The fused loss recomputed from scratch: the finite-difference oracle. */
    static double loss(Tensor x, Tensor w, Tensor bias, Tensor targets) {
        int batch = x.rows, classes = w.cols;
        double total = 0;
        for (int i = 0; i < batch; i++) {
            double[] probabilities = Ops.softmax(Ops.relu(Ops.logits(x, w, bias, i)));
            total += -Math.log(Math.max(probabilities[(int) targets.peek(i, 0)], Double.MIN_NORMAL));
        }
        return total / batch;
    }

    static Tensor probabilities(Tensor x, Tensor w, Tensor bias) {
        Tensor out = Tensor.zero(x.rows, w.cols);
        for (int i = 0; i < x.rows; i++) {
            double[] probabilities = Ops.softmax(Ops.relu(Ops.logits(x, w, bias, i)));
            for (int j = 0; j < w.cols; j++) out.poke(i, j, probabilities[j]);
        }
        return out;
    }

    static Tensor random(SplitMix64 rng, int rows, int cols, double scale) {
        Tensor tensor = Tensor.zero(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) tensor.poke(i, j, scale * rng.nextGaussian());
        return tensor;
    }

    static Tensor targetsOf(int[] values) {
        Tensor tensor = Tensor.zero(values.length, 1);
        for (int i = 0; i < values.length; i++) tensor.poke(i, 0, values[i]);
        return tensor;
    }

    /** base + step * direction, an elementwise shifted copy. */
    static Tensor shifted(Tensor base, Tensor direction, double step) {
        Tensor out = base.copy();
        for (int i = 0; i < out.data.length; i++) out.data[i] += step * direction.data[i];
        return out;
    }

    static double dotWith(Tensor a, Tensor b) {
        double sum = 0;
        for (int i = 0; i < a.data.length; i++) sum += a.data[i] * b.data[i];
        return sum;
    }

    static double norm(Tensor... tensors) {
        double sum = 0;
        for (Tensor tensor : tensors) for (double value : tensor.data) sum += value * value;
        return Math.sqrt(sum);
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError("FAILED: " + message);
    }
}

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L;
        long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    double nextGaussian() {
        double u1 = Math.max(nextDouble(), 1e-12), u2 = nextDouble();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}
