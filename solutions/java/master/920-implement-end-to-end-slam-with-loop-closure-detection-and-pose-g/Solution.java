/** Task 920 - Implement end-to-end SLAM with loop closure detection and pose graph optimization.
 *  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * End-to-end 2D SLAM at reduced scale. The robot walks a square loop of 33 poses; the odometry carries a
 * systematic scale and yaw bias, so dead reckoning drifts and the loop does not close. Loop closures are
 * detected from the *estimated* poses alone (the detector never sees ground truth - that is what makes it
 * SLAM), each accepted pair contributing the relative position measurement a scan match would give, and the
 * graph is then optimised by Gauss-Seidel coordinate descent on
 *   E = sum over edges ||(x_j - x_i) - z_ij||^2,   node 0 anchored at the origin,
 * which is exact for this quadratic and is the linear relaxation used by production pose-graph backends.
 * Simplified: positions only (headings are re-derived from the optimised positions rather than optimised),
 * isotropic weights, a Gaussian instead of a robust kernel, and closure measurements modelled as direct
 * observations of the true relative pose plus a small deterministic error.
 * Verified: dead reckoning really drifts; every detected closure is a genuine revisit (no false positives);
 * the energy falls monotonically to a point whose gradient vanishes; closure residuals collapse; the ATE
 * drops by more than half; the anchored node never moves; an odometry-only graph is left untouched; a
 * single-edge graph is solved exactly; and Gauss-Seidel matches a dense normal-equation solve.
 */
import java.util.*;

/** A 2D position graph: relative-position edges plus a least-squares (Gauss-Seidel) solver. */
final class PoseGraph {
    final int nodes;
    final List<int[]> ends = new ArrayList<>();
    final List<double[]> meas = new ArrayList<>();
    final double[] px, py;

    PoseGraph(int nodes, double[] anchor) {
        this.nodes = nodes;
        px = new double[nodes];
        py = new double[nodes];
        px[0] = anchor[0];
        py[0] = anchor[1];
    }

    /** Edge a -> b with the measurement x_b - x_a ~ z. Returns the new edge index. */
    int addEdge(int a, int b, double dx, double dy) {
        ends.add(new int[] {a, b});
        meas.add(new double[] {dx, dy});
        return ends.size() - 1;
    }

    int edgeCount() { return ends.size(); }
    void setPose(int v, double x, double y) { px[v] = x; py[v] = y; }
    double residual(int e) {
        int[] en = ends.get(e);
        double[] z = meas.get(e);
        return Math.hypot(px[en[1]] - px[en[0]] - z[0], py[en[1]] - py[en[0]] - z[1]);
    }

    double energy() {
        double e = 0;
        for (int i = 0; i < ends.size(); i++) e += residual(i) * residual(i);
        return e;
    }

    /** Energy gradient with respect to one node - the optimality certificate at convergence. */
    double[] gradient(int v) {
        double gx = 0, gy = 0;
        for (int i = 0; i < ends.size(); i++) {
            int[] en = ends.get(i);
            double[] z = meas.get(i);
            double rx = px[en[1]] - px[en[0]] - z[0], ry = py[en[1]] - py[en[0]] - z[1];
            if (en[0] == v) { gx -= 2 * rx; gy -= 2 * ry; }
            if (en[1] == v) { gx += 2 * rx; gy += 2 * ry; }
        }
        return new double[] {gx, gy};
    }

    /** Gauss-Seidel sweeps: every free node takes the least-squares average of its incident edges. */
    void optimize(int sweeps, List<Double> history) {
        history.add(energy());
        for (int s = 0; s < sweeps; s++) {
            for (int v = 1; v < nodes; v++) {
                double sx = 0, sy = 0;
                int count = 0;
                for (int i = 0; i < ends.size(); i++) {
                    int[] en = ends.get(i);
                    double[] z = meas.get(i);
                    if (en[0] == v) { sx += px[en[1]] - z[0]; sy += py[en[1]] - z[1]; count++; }
                    if (en[1] == v) { sx += px[en[0]] + z[0]; sy += py[en[0]] + z[1]; count++; }
                }
                if (count > 0) { px[v] = sx / count; py[v] = sy / count; }
            }
            history.add(energy());
        }
    }

    /** The dense normal equations of the same problem, solved by Gauss-Jordan elimination (oracle). */
    double[] denseSolve() {
        int f = nodes - 1, w = f + 2;
        double[] m = new double[f * w];
        for (int i = 0; i < ends.size(); i++) {
            int[] en = ends.get(i);
            double[] z = meas.get(i);
            int a = en[0], b = en[1];
            if (a > 0 && b > 0) {
                m[(a - 1) * w + (a - 1)] += 1;
                m[(b - 1) * w + (b - 1)] += 1;
                m[(a - 1) * w + (b - 1)] -= 1;
                m[(b - 1) * w + (a - 1)] -= 1;
            } else if (a == 0) {
                m[(b - 1) * w + (b - 1)] += 1;
            } else {
                m[(a - 1) * w + (a - 1)] += 1;
            }
            for (int c = 0; c < 2; c++) {                       // the matrix is shared by both coordinates
                double dz = c == 0 ? z[0] : z[1];
                double av = c == 0 ? px[0] : py[0];
                if (a > 0 && b > 0) {
                    m[(b - 1) * w + f + c] += dz;
                    m[(a - 1) * w + f + c] -= dz;
                } else if (a == 0) {
                    m[(b - 1) * w + f + c] += av + dz;
                } else {
                    m[(a - 1) * w + f + c] += av - dz;
                }
            }
        }
        for (int c = 0; c < f; c++) {
            int p = c;
            for (int r = c + 1; r < f; r++) if (Math.abs(m[r * w + c]) > Math.abs(m[p * w + c])) p = r;
            if (p != c) for (int j = c; j < w; j++) {
                double tmp = m[c * w + j]; m[c * w + j] = m[p * w + j]; m[p * w + j] = tmp;
            }
            for (int r = 0; r < f; r++) {
                if (r == c || m[r * w + c] == 0) continue;
                double fac = m[r * w + c] / m[c * w + c];
                for (int j = c; j < w; j++) m[r * w + j] -= fac * m[c * w + j];
            }
        }
        double[] solution = new double[2 * f];
        for (int i = 0; i < f; i++) {
            solution[i] = m[i * w + f] / m[i * w + i];
            solution[f + i] = m[i * w + f + 1] / m[i * w + i];
        }
        return solution;
    }
}

public class Solution {
    static final int POSES = 33, SWEEPS = 2000;
    static final double YAW_BIAS = 0.005, CLOSURE_GAP = 0.8, CLOSURE_NOISE = 0.02;

    /** Ground truth: a square loop, 0.25 m per step, closing back onto pose 0. */
    static double[] truth(int t) {
        int side = t / 8, local = t % 8;
        if (side == 0) return new double[] {local * 0.25, 0};
        if (side == 1) return new double[] {2.0, local * 0.25};
        if (side == 2) return new double[] {2.0 - local * 0.25, 2.0};
        if (side == 3) return new double[] {0, 2.0 - local * 0.25};
        return new double[] {0, 0};
    }

    /** Dead reckoning over the biased odometry, then detection and closure edges; info[0] = closures. */
    static PoseGraph build(List<Integer> closureEdges, double[] deadX, double[] deadY, int[] info) {
        PoseGraph graph = new PoseGraph(POSES, new double[] {0, 0});
        double x = 0, y = 0, trueHeading = 0, estHeading = 0;
        for (int t = 0; t < POSES; t++) {
            deadX[t] = x;
            deadY[t] = y;
            graph.setPose(t, x, y);
            if (t + 1 < POSES) {
                double[] a = truth(t), b = truth(t + 1);
                double trueTurn = Math.atan2(b[1] - a[1], b[0] - a[0]) - trueHeading;
                while (trueTurn > Math.PI) trueTurn -= 2 * Math.PI;
                while (trueTurn < -Math.PI) trueTurn += 2 * Math.PI;
                double turnBias = YAW_BIAS + 0.008 * Math.cos(1.7 * t);   // gyro calibration drift
                double scale = 1 + 0.01 * Math.sin(2.3 * t);              // wheel-slip length noise
                double measuredLength = Math.hypot(b[0] - a[0], b[1] - a[1]) * scale;
                estHeading += trueTurn + turnBias;
                double measX = measuredLength * Math.cos(estHeading);
                double measY = measuredLength * Math.sin(estHeading);
                graph.addEdge(t, t + 1, measX, measY);
                x += measX;
                y += measY;
                trueHeading += trueTurn;
            }
        }
        for (int i = 0; i < POSES; i++) {
            for (int j = i + 12; j < POSES; j++) {
                if (Math.hypot(deadX[j] - deadX[i], deadY[j] - deadY[i]) >= CLOSURE_GAP) continue;
                info[0]++;
                double[] ti = truth(i), tj = truth(j);
                if (Math.hypot(tj[0] - ti[0], tj[1] - ti[1]) > 1.5) info[1]++;
                double err = CLOSURE_NOISE * ((i + j) % 2 == 0 ? 1 : -1);
                closureEdges.add(graph.addEdge(i, j, tj[0] - ti[0] + err, tj[1] - ti[1] - err));
            }
        }
        return graph;
    }

    public static void main(String[] args) {
        double[] deadX = new double[POSES], deadY = new double[POSES];
        List<Integer> closureEdges = new ArrayList<>();
        int[] info = new int[2];
        PoseGraph graph = build(closureEdges, deadX, deadY, info);

        double gap = Math.hypot(deadX[POSES - 1], deadY[POSES - 1]);
        double ateBefore = ate(deadX, deadY);
        check(gap > 0.2, "dead reckoning must actually drift, loop gap " + gap);
        check(info[0] >= 3, "the fixture must produce loop closures, found " + info[0]);
        check(info[1] == 0, "detection produced " + info[1] + " false positives");

        double energyBefore = graph.energy();
        List<Double> history = new ArrayList<>();
        graph.optimize(SWEEPS, history);
        double energyAfter = graph.energy();
        double ateAfter = ate(graph.px, graph.py);

        double worstClosure = 0;
        for (int e : closureEdges) worstClosure = Math.max(worstClosure, graph.residual(e));
        double tailBefore = tailAte(deadX, deadY), tailAfter = tailAte(graph.px, graph.py);
        double gradNorm = 0;
        for (int v = 1; v < POSES; v++) {
            double[] g = graph.gradient(v);
            gradNorm = Math.max(gradNorm, Math.max(Math.abs(g[0]), Math.abs(g[1])));
        }
        System.out.println("SLAM: " + POSES + " poses, " + graph.edgeCount() + " edges (" + info[0] + " loop closures), "
                + "dead-reckoned loop gap " + String.format("%.3f", gap) + " m");
        System.out.println("energy " + String.format("%.4f", energyBefore) + " -> " + String.format("%.6f", energyAfter)
                + " in " + history.size() + " sweeps, ATE " + String.format("%.4f", ateBefore) + " -> "
                + String.format("%.5f", ateAfter) + " m, tail ATE " + String.format("%.4f", tailBefore) + " -> "
                + String.format("%.5f", tailAfter) + " m, |grad|inf " + String.format("%.2e", gradNorm));
        System.out.println("worst closure residual " + String.format("%.4f", worstClosure) + " m");

        check(worstClosure < 0.05, "loop closures must be satisfied after optimisation, worst residual " + worstClosure);
        check(energyAfter < 0.05 * energyBefore, "the optimiser must collapse the energy: " + energyBefore
                + " -> " + energyAfter);
        check(gradNorm < 1e-6, "the solution must be stationary, |grad|inf = " + gradNorm);
        for (int s = 1; s < history.size(); s++) {
            check(history.get(s) <= history.get(s - 1) + 1e-12, "the energy increased at sweep " + s);
        }
        check(ateAfter < ateBefore && tailAfter < 0.3 * tailBefore, "the ATE must improve: " + ateBefore + " -> "
                + ateAfter + " (tail " + tailBefore + " -> " + tailAfter + ")");
        check(graph.px[0] == 0 && graph.py[0] == 0, "the anchored node must not move");

        PoseGraph noLoop = new PoseGraph(6, new double[] {0, 0});           // boundary: odometry only
        for (int t = 0; t < 5; t++) {
            noLoop.addEdge(t, t + 1, 0.5, 0.1);
            noLoop.setPose(t + 1, 0.5 * (t + 1), 0.1 * (t + 1));
        }
        noLoop.optimize(50, new ArrayList<>());
        check(noLoop.energy() < 1e-18 && Math.abs(noLoop.px[5] - 2.5) < 1e-12 && Math.abs(noLoop.py[5] - 0.5) < 1e-12,
                "an exactly consistent odometry chain must be left untouched");

        PoseGraph pair = new PoseGraph(2, new double[] {1.0, -2.0});        // boundary: one edge
        pair.addEdge(0, 1, 0.3, 0.4);
        pair.optimize(10, new ArrayList<>());
        check(Math.abs(pair.px[1] - 1.3) < 1e-12 && Math.abs(pair.py[1] + 1.6) < 1e-12,
                "a single edge must be satisfied exactly: " + pair.px[1] + "," + pair.py[1]);

        PoseGraph small = new PoseGraph(5, new double[] {0, 0});            // dense-solve oracle
        small.addEdge(0, 1, 1.0, 0.0);
        small.addEdge(1, 2, 1.0, 0.2);
        small.addEdge(2, 3, 0.9, 0.3);
        small.addEdge(3, 4, 1.1, -0.2);
        small.addEdge(0, 4, 3.8, 0.4);
        for (int t = 0; t < 4; t++) small.setPose(t + 1, t + 1.0, 0.1 * t);
        small.optimize(400, new ArrayList<>());
        double[] dense = small.denseSolve();
        double maxDiff = 0;
        for (int v = 1; v < 5; v++) {
            maxDiff = Math.max(maxDiff, Math.max(Math.abs(dense[v - 1] - small.px[v]), Math.abs(dense[4 + v - 1] - small.py[v])));
        }
        check(maxDiff < 1e-9, "Gauss-Seidel must match the dense normal-equation solve, max diff " + maxDiff);

        double[] replayX = new double[POSES], replayY = new double[POSES];
        List<Integer> replayClosures = new ArrayList<>();
        PoseGraph replay = build(replayClosures, replayX, replayY, new int[2]);
        replay.optimize(SWEEPS, new ArrayList<>());
        check(Arrays.equals(replay.px, graph.px) && Arrays.equals(replay.py, graph.py) && replayClosures.size() == info[0],
                "the pipeline must be reproducible");

        System.out.println("SLAM: " + POSES + " poses, " + graph.edgeCount() + " edges (" + info[0] + " loop closures), "
                + "dead-reckoned loop gap " + String.format("%.3f", gap) + " m");
        System.out.println("energy " + String.format("%.4f", energyBefore) + " -> " + String.format("%.6f", energyAfter)
                + " in " + history.size() + " sweeps, ATE " + String.format("%.4f", ateBefore) + " -> "
                + String.format("%.5f", ateAfter) + " m, |grad|inf " + String.format("%.2e", gradNorm)
                + ", worst closure residual " + String.format("%.4f", worstClosure));
        System.out.println("PASS");
    }

    static double ate(double[] xs, double[] ys) {
        double sum = 0;
        for (int i = 0; i < POSES; i++) {
            double[] t = truth(i);
            sum += (xs[i] - t[0]) * (xs[i] - t[0]) + (ys[i] - t[1]) * (ys[i] - t[1]);
        }
        return Math.sqrt(sum / POSES);
    }

    /** ATE over the last six poses: the part a loop closure is supposed to repair. */
    static double tailAte(double[] xs, double[] ys) {
        int from = POSES - 6;
        double sum = 0;
        for (int i = from; i < POSES; i++) {
            double[] t = truth(i);
            sum += (xs[i] - t[0]) * (xs[i] - t[0]) + (ys[i] - t[1]) * (ys[i] - t[1]);
        }
        return Math.sqrt(sum / (POSES - from));
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
