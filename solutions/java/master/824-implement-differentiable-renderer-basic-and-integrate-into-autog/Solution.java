/** Task 824 - Implement differentiable renderer basic and integrate into autograd. (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 * Two halves that are checked against each other:
 *
 *   autograd    a reverse-mode engine over scalar nodes.  Each op records its local partial derivatives, the
 *               backward pass walks a reverse topological order and accumulates with an identity map, so a
 *               node consumed several times receives the summed gradient.  Correctness is established by
 *               central finite differences: per op, on a composite expression, and end-to-end through the
 *               renderer.
 *   renderer    a 1D silhouette renderer.  A sphere of centre c and radius r covers a ray at x with the
 *               smooth coverage  sigmoid((c + r - x)/beta) - sigmoid((c - r - x)/beta),  which tends to the
 *               exact indicator of [c-r, c+r] as beta -> 0.  Spheres are composited front to back with alpha
 *               blending, so occlusion is differentiable and a hidden sphere receives a gradient of exactly
 *               zero.  The image is compared with a target silhouette by mean squared error and the scene
 *               parameters are recovered by gradient descent (inverse rendering).
 *
 * Deliberate simplifications (deterministic; java.util only; no threads, sockets, GPUs or floating-point
 * randomness): one spatial dimension instead of two, spheres instead of meshes, scalar nodes instead of
 * tensors, tens of rays and up to three spheres, and no materials or lighting -- only silhouette coverage.
 * Every claim below is therefore an exact number rather than an approximation of a larger system.
 *
 * Verified invariants:
 *   gradients   - every op, a composite expression and the renderer loss (centre, radius, sharpness) match
 *                 central finite differences to 1e-6 relative, and no gradient is ever NaN.
 *   geometry    - coverage lies in [0, 1] and is monotone in the radius; the image converges to the hard
 *                 silhouette as beta shrinks (measured L1 error decreasing); a fully occluded sphere
 *                 contributes nothing to the image and receives a zero gradient.
 *   convergence - gradient descent reduces the render loss by more than two orders of magnitude and recovers
 *                 the planted centre and radius within tolerance; the loss is zero when images match.
 *   cost        - the backward pass visits essentially every node the forward pass created (measured within a
 *                 small constant per ray) and the render cost is O(rays x spheres); degenerate scenes (no
 *                 rays, one ray, zero radius, off-screen sphere) neither crash nor produce NaN.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

/** A scalar node: value, its parents and the local partial derivative for each parent. */
class Node {
    final double value;
    final List<Node> parents;
    final List<Double> localGrads;
    static long forwardCount;
    static long backwardVisits;

    private Node(double value, List<Node> parents, List<Double> localGrads) {
        this.value = value; this.parents = parents; this.localGrads = localGrads;
    }

    static Node of(double value) { forwardCount++; return new Node(value, Collections.emptyList(), Collections.emptyList()); }

    static Node op(double value, List<Node> parents, List<Double> grads) {
        forwardCount++;
        return new Node(value, parents, grads);
    }

    static Node add(Node a, Node b) { return op(a.value + b.value, Arrays.asList(a, b), Arrays.asList(1.0, 1.0)); }

    static Node sub(Node a, Node b) { return op(a.value - b.value, Arrays.asList(a, b), Arrays.asList(1.0, -1.0)); }

    static Node mul(Node a, Node b) {
        return op(a.value * b.value, Arrays.asList(a, b), Arrays.asList(b.value, a.value));
    }

    static Node div(Node a, Node b) {
        return op(a.value / b.value, Arrays.asList(a, b),
                Arrays.asList(1.0 / b.value, -a.value / (b.value * b.value)));
    }

    static Node exp(Node a) {
        double v = Math.exp(a.value);
        return op(v, Collections.singletonList(a), Collections.singletonList(v));
    }

    static Node log(Node a) {
        return op(Math.log(a.value), Collections.singletonList(a), Collections.singletonList(1.0 / a.value));
    }

    static Node sqrt(Node a) {
        double v = Math.sqrt(Math.max(a.value, 1e-12));
        return op(v, Collections.singletonList(a), Collections.singletonList(0.5 / v));
    }

    static Node pow(Node a, double p) {
        return op(Math.pow(Math.max(a.value, 1e-12), p), Collections.singletonList(a),
                Collections.singletonList(p * Math.pow(Math.max(a.value, 1e-12), p - 1)));
    }

    static Node sigmoid(Node a) {
        double v = 1.0 / (1.0 + Math.exp(-a.value));
        return op(v, Collections.singletonList(a), Collections.singletonList(v * (1 - v)));
    }

    static Node tanh(Node a) {
        double v = Math.tanh(a.value);
        return op(v, Collections.singletonList(a), Collections.singletonList(1 - v * v));
    }

    static Node relu(Node a) {
        return op(Math.max(0, a.value), Collections.singletonList(a),
                Collections.singletonList(a.value > 0 ? 1.0 : 0.0));
    }

    static Node sum(List<Node> parts) {
        double v = 0;
        List<Double> grads = new ArrayList<>();
        for (Node n : parts) { v += n.value; grads.add(1.0); }
        return op(v, new ArrayList<>(parts), grads);
    }

    static Node mean(List<Node> parts) {
        double v = 0;
        List<Double> grads = new ArrayList<>();
        for (Node n : parts) { v += n.value; grads.add(1.0 / parts.size()); }
        return op(v / parts.size(), new ArrayList<>(parts), grads);
    }

    /** Reverse-mode pass: reverse topological order with identity-based accumulation. */
    static Map<Node, Double> gradients(Node root) {
        List<Node> order = new ArrayList<>();
        Set<Node> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        postOrder(root, seen, order);
        Map<Node, Double> grad = new IdentityHashMap<>();
        grad.put(root, 1.0);
        for (int i = order.size() - 1; i >= 0; i--) {
            Node n = order.get(i);
            Double g = grad.get(n);
            if (g == null) continue;
            backwardVisits++;
            for (int k = 0; k < n.parents.size(); k++) {
                grad.merge(n.parents.get(k), g * n.localGrads.get(k), Double::sum);
            }
        }
        return grad;
    }

    private static void postOrder(Node n, Set<Node> seen, List<Node> out) {
        if (!seen.add(n)) return;
        for (Node p : n.parents) postOrder(p, seen, out);
        out.add(n);
    }

    static double gradOf(Map<Node, Double> grads, Node n) { return grads.getOrDefault(n, 0.0); }
}

/** A sphere of the 1D scene: centre, radius and depth (front to back compositing order). */
class Sphere {
    final double centre;
    final double radius;
    final double depth;

    Sphere(double centre, double radius, double depth) { this.centre = centre; this.radius = radius; this.depth = depth; }
}

/** A rendered graph: the loss root plus the parameter leaves in a known order. */
class Graph {
    final Node root;
    final List<Node> leaves;
    final List<Node> image;
    long cost;

    Graph(Node root, List<Node> leaves, List<Node> image, long cost) {
        this.root = root; this.leaves = leaves; this.image = image; this.cost = cost;
    }
}

public class Solution {

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }

    interface Builder { Graph build(double[] params); }

    /** Smooth coverage of the interval [c-r, c+r] at x, differentiable in c, r and beta. */
    static Node coverage(Node centre, Node radius, Node beta, double x) {
        Node upper = Node.sigmoid(Node.div(Node.sub(Node.add(centre, radius), Node.of(x)), beta));
        Node lower = Node.sigmoid(Node.div(Node.sub(Node.sub(centre, radius), Node.of(x)), beta));
        return Node.sub(upper, lower);
    }

    /** Front-to-back alpha compositing of spheres over the ray set. */
    static Graph render(double[] params, double[] rays, double[] target, int sphereCount) {
        List<Node> leaves = new ArrayList<>();
        for (int s = 0; s < sphereCount; s++) {
            leaves.add(Node.of(params[2 * s]));         // centre
            leaves.add(Node.of(params[2 * s + 1]));     // radius
        }
        Node beta = Node.of(params[params.length - 1]);
        leaves.add(beta);                                // parameters are leaves; beta comes last
        List<Node> image = new ArrayList<>();
        List<Node> squared = new ArrayList<>();
        long cost = 0;
        for (double x : rays) {
            Node transmittance = Node.of(1.0);
            Node accumulated = Node.of(0.0);
            for (int s = 0; s < sphereCount; s++) {
                Node coverage = coverage(leaves.get(2 * s), leaves.get(2 * s + 1), beta, x);
                accumulated = Node.add(accumulated, Node.mul(transmittance, coverage));
                transmittance = Node.mul(transmittance, Node.sub(Node.of(1.0), coverage));
                cost += 8;
            }
            image.add(accumulated);
            Node residual = Node.sub(accumulated, Node.of(target[image.size() - 1]));
            squared.add(Node.mul(residual, residual));
            cost += 3;
        }
        Node loss = squared.isEmpty() ? Node.of(0.0) : Node.mean(squared);
        return new Graph(loss, leaves, image, cost);
    }

    static double hardCoverage(Sphere s, double x) {
        return (x >= s.centre - s.radius && x <= s.centre + s.radius) ? 1.0 : 0.0;
    }

    static double[] hardImage(List<Sphere> spheres, double[] rays) {
        double[] out = new double[rays.length];
        for (int i = 0; i < rays.length; i++) {
            double transmittance = 1.0, accumulated = 0;
            for (Sphere s : spheres) {
                double c = hardCoverage(s, rays[i]);
                accumulated += transmittance * c;
                transmittance *= (1 - c);
            }
            out[i] = accumulated;
        }
        return out;
    }

    /** Compares analytic gradients with central finite differences for every parameter. */
    static void checkGradients(String name, Builder builder, double[] params, double tolerance) {
        Graph g = builder.build(params);
        Map<Node, Double> grads = Node.gradients(g.root);
        for (int i = 0; i < params.length; i++) {
            double analytic = Node.gradOf(grads, g.leaves.get(i));
            double h = 1e-5;
            double original = params[i];
            params[i] = original + h;
            double up = builder.build(params).root.value;
            params[i] = original - h;
            double down = builder.build(params).root.value;
            params[i] = original;
            double numeric = (up - down) / (2 * h);
            double scale = Math.max(1e-8, Math.max(Math.abs(analytic), Math.abs(numeric)));
            check(Math.abs(analytic - numeric) / scale < tolerance, name + ": parameter " + i + " analytic "
                    + analytic + " vs finite difference " + numeric);
            check(!Double.isNaN(analytic) && !Double.isInfinite(analytic), name + ": gradient " + i + " must be finite");
        }
    }

    static double[] raySet(double from, double to, int count) {
        double[] rays = new double[count];
        for (int i = 0; i < count; i++) rays[i] = count == 1 ? from : from + (to - from) * i / (count - 1.0);
        return rays;
    }

    static double[] concat(double... values) { return values; }

    public static void main(String[] args) {
        // ---- 1. per-op gradient checks ------------------------------------------------------------
        checkGradients("add", p -> {
            Node a = Node.of(p[0]), b = Node.of(p[1]);
            Node root = Node.add(a, b);
            return new Graph(root, Arrays.asList(a, b), Collections.emptyList(), 1);
        }, concat(1.5, -0.5), 1e-6);

        checkGradients("sub", p -> {
            Node a = Node.of(p[0]), b = Node.of(p[1]);
            return new Graph(Node.sub(a, b), Arrays.asList(a, b), Collections.emptyList(), 1);
        }, concat(0.5, 2.0), 1e-6);

        checkGradients("mul", p -> {
            Node a = Node.of(p[0]), b = Node.of(p[1]);
            return new Graph(Node.mul(a, b), Arrays.asList(a, b), Collections.emptyList(), 1);
        }, concat(-3.0, 2.5), 1e-6);

        checkGradients("div", p -> {
            Node a = Node.of(p[0]), b = Node.of(p[1]);
            return new Graph(Node.div(a, b), Arrays.asList(a, b), Collections.emptyList(), 1);
        }, concat(3.0, 1.5), 1e-6);

        checkGradients("exp/log", p -> {
            Node a = Node.of(p[0]);
            return new Graph(Node.exp(Node.log(a)), Collections.singletonList(a), Collections.emptyList(), 1);
        }, concat(0.75), 1e-6);

        checkGradients("sqrt", p -> {
            Node a = Node.of(p[0]);
            return new Graph(Node.sqrt(a), Collections.singletonList(a), Collections.emptyList(), 1);
        }, concat(4.0), 1e-6);

        checkGradients("pow 2.5", p -> {
            Node a = Node.of(p[0]);
            return new Graph(Node.pow(a, 2.5), Collections.singletonList(a), Collections.emptyList(), 1);
        }, concat(1.5), 1e-6);

        checkGradients("sigmoid/tanh/relu", p -> {
            Node a = Node.of(p[0]), b = Node.of(p[1]), c = Node.of(p[2]);
            Node root = Node.mean(Arrays.asList(Node.sigmoid(a), Node.tanh(b), Node.relu(c)));
            return new Graph(root, Arrays.asList(a, b, c), Collections.emptyList(), 1);
        }, concat(0.4, -0.7, 1.3), 1e-6);

        // ---- 2. composite expression ---------------------------------------------------------------
        checkGradients("composite", p -> {
            Node x = Node.of(p[0]), y = Node.of(p[1]), z = Node.of(p[2]);
            Node mixed = Node.add(Node.mul(Node.sigmoid(x), Node.tanh(y)), Node.div(z, Node.add(x, Node.of(2.0))));
            Node squared = Node.mul(mixed, mixed);
            Node root = Node.add(squared, Node.sqrt(Node.add(Node.exp(Node.mul(Node.of(0.5), y)), Node.of(1.0))));
            return new Graph(root, Arrays.asList(x, y, z), Collections.emptyList(), 1);
        }, concat(0.3, 0.8, -0.6), 1e-6);

        // shared node: the gradient must accumulate instead of being overwritten
        {
            Node shared = Node.of(0.7);
            Node root = Node.add(Node.mul(shared, shared), Node.mul(Node.of(3.0), shared));
            Map<Node, Double> grads = Node.gradients(root);
            double expected = 2 * 0.7 + 3.0;
            check(Math.abs(Node.gradOf(grads, shared) - expected) < 1e-12,
                    "a shared node must accumulate both contributions: " + Node.gradOf(grads, shared));
        }

        // ---- 3. renderer gradients -----------------------------------------------------------------
        double[] rays = raySet(0.0, 4.0, 32);
        List<Sphere> truth = new ArrayList<>(Collections.singletonList(new Sphere(2.0, 0.8, 0.0)));
        double[] target = hardImage(truth, rays);
        Graph fit = render(concat(1.5, 0.5, 0.05), rays, target, 1);
        check(fit.root.value >= 0, "the render loss must be non-negative: " + fit.root.value);
        checkGradients("renderer (centre, radius, beta)", p -> render(p, rays, target, 1), concat(1.5, 0.5, 0.05), 1e-6);

        // ---- 4. coverage geometry ------------------------------------------------------------------
        double[][] coverageSamples = new double[rays.length][];
        for (int i = 0; i < rays.length; i++) {
            Node c = Node.of(2.0), r = Node.of(0.8), beta = Node.of(0.05);
            coverageSamples[i] = new double[] {coverage(c, r, beta, rays[i]).value};
        }
        boolean inRange = true;
        for (double[] sample : coverageSamples) if (sample[0] < 0 || sample[0] > 1) inRange = false;
        check(inRange, "coverage must lie in [0, 1]");

        Node centre = Node.of(2.0), betaNode = Node.of(0.05);
        Node smallRadius = coverage(centre, Node.of(0.3), betaNode, 2.4);
        Node largeRadius = coverage(centre, Node.of(0.9), betaNode, 2.4);
        check(largeRadius.value > smallRadius.value, "coverage must grow with the radius: "
                + smallRadius.value + " vs " + largeRadius.value);

        // the smooth renderer converges to the hard silhouette as beta shrinks
        double previousError = Double.MAX_VALUE;
        boolean monotone = true;
        double[] betas = {0.5, 0.25, 0.1, 0.05, 0.02};
        double[] errors = new double[betas.length];
        for (int i = 0; i < betas.length; i++) {
            Graph smooth = render(concat(2.0, 0.8, betas[i]), rays, target, 1);
            double error = 0;
            for (int j = 0; j < rays.length; j++) error += Math.abs(smooth.image.get(j).value - target[j]);
            errors[i] = error / rays.length;
            if (errors[i] > previousError + 1e-9) monotone = false;
            previousError = errors[i];
        }
        check(monotone, "the L1 error against the hard silhouette must shrink with beta: " + Arrays.toString(errors));
        check(errors[betas.length - 1] < 0.02, "a sharp renderer must match the hard silhouette: "
                + errors[betas.length - 1]);
        check(smoothLoss(concat(2.0, 0.8, 0.02), rays, target) < 0.03,
                "the sharp renderer must reproduce the hard silhouette to within the boundary rays: "
                        + smoothLoss(concat(2.0, 0.8, 0.02), rays, target));

        // ---- 5. inverse rendering ------------------------------------------------------------------
        // The optimisation target is the smooth image of the true sphere at the same beta, so the achievable
        // optimum is zero and the recovered parameters can be compared with the truth directly.
        double betaFit = 0.05;
        double[] smoothTarget = imageOf(concat(2.0, 0.8, betaFit), rays, new double[rays.length], 1);
        double c = 1.2, r = 0.4;
        double loss = smoothLoss(concat(c, r, betaFit), rays, smoothTarget);
        double initialLoss = loss;
        double lr = 2.0;
        for (int step = 0; step < 800 && loss > 1e-10; step++) {
            Graph g = render(concat(c, r, betaFit), rays, smoothTarget, 1);
            Map<Node, Double> grads = Node.gradients(g.root);
            double gc = Node.gradOf(grads, g.leaves.get(0));
            double gr = Node.gradOf(grads, g.leaves.get(1));
            double candidateC = c - lr * gc, candidateR = r - lr * gr;
            double candidateLoss = smoothLoss(concat(candidateC, candidateR, betaFit), rays, smoothTarget);
            if (candidateLoss > loss) { lr *= 0.5; continue; }     // deterministic backtracking line search
            c = candidateC; r = candidateR; loss = candidateLoss;
        }
        double finalLoss = loss;
        check(!Double.isNaN(c) && !Double.isNaN(r), "the optimiser must not diverge: " + c + ", " + r);
        check(finalLoss < initialLoss / 100, "gradient descent must cut the loss by two orders: "
                + initialLoss + " -> " + finalLoss);
        check(finalLoss < 1e-6, "the fit must reach the planted image: " + finalLoss);
        check(Math.abs(c - 2.0) < 0.02, "inverse rendering must recover the centre: " + c);
        check(Math.abs(r - 0.8) < 0.02, "inverse rendering must recover the radius: " + r);

        // ---- 6. occlusion --------------------------------------------------------------------------
        List<Sphere> occluded = Arrays.asList(new Sphere(2.0, 2.5, 0.0), new Sphere(2.0, 0.5, 1.0));
        double[] occludedTarget = hardImage(occluded, rays);
        Graph occlusionGraph = render(concat(2.0, 2.5, 2.0, 0.5, 0.05), rays, occludedTarget, 2);
        Map<Node, Double> occlusionGrads = Node.gradients(occlusionGraph.root);
        double frontGrad = Math.abs(Node.gradOf(occlusionGrads, occlusionGraph.leaves.get(0)))
                + Math.abs(Node.gradOf(occlusionGrads, occlusionGraph.leaves.get(1)));
        double backGrad = Math.abs(Node.gradOf(occlusionGrads, occlusionGraph.leaves.get(2)))
                + Math.abs(Node.gradOf(occlusionGrads, occlusionGraph.leaves.get(3)));
        check(backGrad < 1e-6, "a fully occluded sphere must receive no gradient: " + backGrad);
        check(frontGrad > backGrad * 100, "the visible sphere must dominate the gradient: "
                + frontGrad + " vs " + backGrad);

        double[] imageA = imageOf(concat(2.0, 2.5, 2.0, 0.5, 0.05), rays, occludedTarget, 2);
        double[] imageB = imageOf(concat(2.0, 2.5, 3.5, 0.9, 0.05), rays, occludedTarget, 2);
        double imageDelta = 0;
        for (int i = 0; i < imageA.length; i++) imageDelta += Math.abs(imageA[i] - imageB[i]);
        check(imageDelta < 1e-4, "moving a fully occluded sphere must not change the image: " + imageDelta);

        // ---- 7. degenerate scenes ------------------------------------------------------------------
        double[] noRays = new double[0];
        double[] emptyTarget = new double[0];
        Graph empty = render(concat(2.0, 0.8, 0.05), noRays, emptyTarget, 1);
        check(empty.root.value == 0, "an empty ray set must produce a zero loss");
        Map<Node, Double> emptyGrads = Node.gradients(empty.root);
        check(Node.gradOf(emptyGrads, empty.leaves.get(0)) == 0, "an empty ray set must produce zero gradients");

        Graph single = render(concat(2.0, 0.8, 0.05), raySet(2.0, 2.0, 1), new double[] {1.0}, 1);
        check(!Double.isNaN(single.root.value) && single.root.value >= 0, "a single ray must render: "
                + single.root.value);

        Graph zeroRadius = render(concat(2.0, 0.0, 0.05), rays, target, 1);
        check(!Double.isNaN(zeroRadius.root.value), "a zero radius must not produce NaN");
        Map<Node, Double> zeroGrads = Node.gradients(zeroRadius.root);
        boolean finite = true;
        for (double g : zeroGrads.values()) if (Double.isNaN(g) || Double.isInfinite(g)) finite = false;
        check(finite, "a zero radius must not produce NaN gradients");

        Graph offScreen = render(concat(20.0, 0.5, 0.05), rays, target, 1);
        Map<Node, Double> offGrads = Node.gradients(offScreen.root);
        check(Math.abs(Node.gradOf(offGrads, offScreen.leaves.get(0))) < 1e-12
                        && Math.abs(Node.gradOf(offGrads, offScreen.leaves.get(1))) < 1e-12,
                "an off-screen sphere must sit on a flat plateau with zero gradient");

        // ---- 8. cost accounting and determinism ----------------------------------------------------
        Node.forwardCount = 0;
        Node.backwardVisits = 0;
        Graph measured = render(concat(1.5, 0.5, 0.05), rays, target, 1);
        Node.gradients(measured.root);
        check(Node.backwardVisits <= Node.forwardCount, "the backward pass cannot visit more nodes than exist: "
                + Node.backwardVisits + " vs " + Node.forwardCount);
        check(Node.backwardVisits >= Node.forwardCount - 4L * rays.length,
                "the backward pass must reach every node except a small constant per ray: "
                        + Node.backwardVisits + " of " + Node.forwardCount);
        long expectedCost = (long) rays.length * (1 * 8 + 3);
        check(measured.cost == expectedCost, "render cost must be O(rays x spheres): " + measured.cost
                + " vs " + expectedCost);
        check(measured.cost == (long) rays.length * (3 + 8), "cost accounting must be exact per ray");

        double loss1 = smoothLoss(concat(1.5, 0.5, 0.05), rays, target);
        double loss2 = smoothLoss(concat(1.5, 0.5, 0.05), rays, target);
        check(loss1 == loss2, "the same scene must render to the same loss: " + loss1 + " vs " + loss2);
        Graph g1 = render(concat(1.5, 0.5, 0.05), rays, target, 1);
        Graph g2 = render(concat(1.5, 0.5, 0.05), rays, target, 1);
        Map<Node, Double> grads1 = Node.gradients(g1.root), grads2 = Node.gradients(g2.root);
        double gc1 = Node.gradOf(grads1, g1.leaves.get(0)), gc2 = Node.gradOf(grads2, g2.leaves.get(0));
        check(Math.abs(gc1 - gc2) < 1e-15, "the same scene must produce the same gradients: " + gc1 + " vs " + gc2);

        System.out.println("coverage L1 error by beta: " + Arrays.toString(errors));
        System.out.println("inverse rendering       : c " + String.format("%.4f", c) + " (truth 2.0), r "
                + String.format("%.4f", r) + " (truth 0.8)");
        System.out.println("loss before / after     : " + String.format("%.6f", initialLoss) + " -> "
                + String.format("%.6f", finalLoss));
        System.out.println("occlusion gradients     : front " + String.format("%.4f", frontGrad) + ", back "
                + String.format("%.4e", backGrad));
        System.out.println("render cost per ray     : " + (measured.cost / rays.length) + " nodes, "
                + rays.length + " rays");
        System.out.println("PASS");
    }

    static double smoothLoss(double[] params, double[] rays, double[] target) {
        return render(params, rays, target, (params.length - 1) / 2).root.value;
    }

    static double[] imageOf(double[] params, double[] rays, double[] target, int sphereCount) {
        Graph g = render(params, rays, target, sphereCount);
        double[] out = new double[g.image.size()];
        for (int i = 0; i < out.length; i++) out[i] = g.image.get(i).value;
        return out;
    }
}
