/** Task 789 - Implement shape-matching registration (ICP) iterative closest point. (Tier: Expert, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * ICP alternates two steps until the pose stops moving: (1) correspondence, every source point matched to
 * its nearest target point (brute force here, so the data structure is not part of the story), and
 * (2) alignment, the rigid transform minimising sum_i ||R(theta) p_i + t - q_i||^2 in closed form. With P,
 * Q the centred coordinates the 2D Kabsch/Horn solution is
 *
 *     theta = atan2( sum_i (p_i x q_i), sum_i (p_i . q_i) ),   t = centroid(q) - R(theta) centroid(p).
 *
 * Each alignment step minimises the squared distance to the *current* correspondences and the next
 * nearest-neighbour pass can only shorten those distances, so E(T) = mean_i min_j ||T(p_i) - q_j||^2 is
 * non-increasing - that monotonicity is the invariant checked here after every iteration. Plain ICP is also
 * notoriously fragile: it needs a start within roughly half a point spacing (otherwise the correspondences
 * slip by one and it stalls in a local minimum, demonstrated below and escaped by a coarse multi-start) and
 * outliers drag it off, so a trimmed variant that solves for the pose from only the best eta fraction of
 * the matches is included and compared.
 *
 * Simplified: 2D rigid (no scale), brute-force correspondences, no KD-tree, no coarse-to-fine schedule and
 * no M-estimator beyond trimming. The closed-form alignment, the monotone objective and the geometry are
 * the real thing.
 *
 * Verified: the closed-form alignment reproduces a planted pose exactly and agrees with a brute-force grid
 * plus parabolic search; ICP recovers the pose from a small start with a vanishing residual and a
 * non-increasing objective; a 0.25 rad start provably stalls in a strictly worse local minimum that a
 * 48-way multi-start escapes; identical clouds stay put; a noisy run stays close with a monotone objective;
 * with 20% garbage source points trimming beats plain ICP by a wide margin; a partial-overlap source
 * registers; a 20% scale change leaves an irreducible residual; one correspondence fixes only the
 * translation; and repeated runs are identical.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = (state ^ (state >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL; return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** A planar rigid transform p -> R(theta) p + t. */
class Rigid2 {
    final double theta, tx, ty;

    Rigid2(double theta, double tx, double ty) {
        this.theta = theta; this.tx = tx; this.ty = ty;
    }

    double[] apply(double[] p) {
        double c = Math.cos(theta), s = Math.sin(theta);
        return new double[]{c * p[0] - s * p[1] + tx, s * p[0] + c * p[1] + ty};
    }

    /** Pose distance: the wrapped angle gap, or the translation gap when that is larger. */
    double poseError(Rigid2 other) {
        double dTheta = Math.atan2(Math.sin(theta - other.theta), Math.cos(theta - other.theta));
        return Math.max(Math.abs(dTheta), Math.hypot(tx - other.tx, ty - other.ty));
    }
}

/** Outcome of an ICP run, including the objective trace used to prove monotonicity. */
class IcpRun {
    final Rigid2 pose;
    final int iterations;
    final boolean converged;
    final double[] residualTrace;

    IcpRun(Rigid2 pose, int iterations, boolean converged, double[] residualTrace) {
        this.pose = pose; this.iterations = iterations;
        this.converged = converged; this.residualTrace = residualTrace;
    }
}

class Icp {
    static double[] centroid(double[][] points) {
        double[] c = new double[2];
        for (double[] p : points) {
            c[0] += p[0]; c[1] += p[1];
        }
        c[0] /= points.length; c[1] /= points.length; return c;
    }

    /** Closed-form 2D Kabsch alignment of matched pairs src[i] -> dst[i]. */
    static Rigid2 align(double[][] src, double[][] dst) {
        double[] cs = centroid(src), cd = centroid(dst); double dot = 0, cross = 0;
        for (int i = 0; i < src.length; i++) {
            double px = src[i][0] - cs[0], py = src[i][1] - cs[1];
            double qx = dst[i][0] - cd[0], qy = dst[i][1] - cd[1]; dot += px * qx + py * qy;
            cross += px * qy - py * qx;
        }
        double theta = Math.atan2(cross, dot); double c = Math.cos(theta), s = Math.sin(theta);
        return new Rigid2(theta, cd[0] - (c * cs[0] - s * cs[1]), cd[1] - (s * cs[0] + c * cs[1]));
    }

    static int nearest(double[][] target, double[] p) {
        int best = 0; double bestDistance = Double.MAX_VALUE;
        for (int j = 0; j < target.length; j++) {
            double dx = target[j][0] - p[0], dy = target[j][1] - p[1]; double d = dx * dx + dy * dy;
            if (d < bestDistance) {
                bestDistance = d; best = j;
            }
        }
        return best;
    }

    /** RMS nearest-neighbour residual of an already transformed source cloud. */
    static double rms(double[][] transformed, double[][] target) {
        double sum = 0;
        for (double[] p : transformed) {
            double[] q = target[nearest(target, p)];
            sum += (p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]);
        }
        return Math.sqrt(sum / transformed.length);
    }

    static double[][] transform(double[][] points, Rigid2 pose) {
        double[][] out = new double[points.length][];
        for (int i = 0; i < points.length; i++) out[i] = pose.apply(points[i]); return out;
    }

    /**
     * ICP. eta in (0, 1] is the fraction of correspondences kept when solving for the pose: eta = 1 is plain
     * ICP, smaller values give the trimmed variant that ignores the worst matches.
     */
    static IcpRun run(double[][] source, double[][] target, Rigid2 start, int maxIterations, double tolerance,
            double eta) {
        if (maxIterations < 0) throw new IllegalArgumentException("maxIterations must not be negative");
        if (eta <= 0 || eta > 1) throw new IllegalArgumentException("eta must be in (0, 1]: " + eta);
        Rigid2 pose = start; double[] trace = new double[maxIterations + 1];
        trace[0] = rms(transform(source, pose), target); boolean converged = false;
        int iterations = 0; double[][] moved = transform(source, pose);
        for (; iterations < maxIterations; iterations++) {
            moved = transform(source, pose); double[] distances = new double[source.length];
            int[] matches = new int[source.length];
            for (int i = 0; i < source.length; i++) {
                matches[i] = nearest(target, moved[i]);
                distances[i] = Math.hypot(moved[i][0] - target[matches[i]][0],
                        moved[i][1] - target[matches[i]][1]);
            }
            int keep = eta >= 1.0 ? source.length : Math.max(3, (int) Math.round(eta * source.length));
            Integer[] order = new Integer[source.length]; for (int i = 0; i < order.length; i++) order[i] = i;
            if (keep < source.length) Arrays.sort(order, Comparator.comparingDouble(i -> distances[i]));
            double[][] matchedSource = new double[keep][]; double[][] matchedTarget = new double[keep][];
            for (int i = 0; i < keep; i++) {
                matchedSource[i] = moved[order[i]]; matchedTarget[i] = target[matches[order[i]]];
            }
            Rigid2 step = align(matchedSource, matchedTarget);
            // step maps points of the current frame into the target frame, so the total pose is step after
            // pose: R(step) R(pose) p + R(step) t_pose + t_step.
            double cs = Math.cos(step.theta), ss = Math.sin(step.theta);
            Rigid2 candidate = new Rigid2(step.theta + pose.theta,
                    step.tx + cs * pose.tx - ss * pose.ty, step.ty + ss * pose.tx + cs * pose.ty);
            pose = candidate; trace[iterations + 1] = rms(transform(source, pose), target);
            if (trace[iterations] - trace[iterations + 1] < tolerance) {
                converged = true; iterations++; break;
            }
        }
        return new IcpRun(pose, iterations, converged, Arrays.copyOf(trace, iterations + 1));
    }
}

public class Solution {
    static double[][] shape(int n, double noise, SplitMix64 rng) {
        double[][] pts = new double[n][2];
        for (int i = 0; i < n; i++) {
            double a = 2 * Math.PI * i / n;
            double r = 1 + 0.25 * Math.sin(3 * a) + noise * (rng.nextDouble() - 0.5);
            pts[i][0] = r * Math.cos(a); pts[i][1] = r * Math.sin(a);
        }
        return pts;
    }

    static boolean nonIncreasing(double[] trace) {
        for (int i = 1; i < trace.length; i++) if (trace[i] > trace[i - 1] + 1e-12) return false; return true;
    }

    /** Brute force oracle: minimise ||R(theta)(p - cp) - (q - cq)||^2 on a grid, then fit parabolas. */
    static double bruteForceAngle(double[][] src, double[][] dst) {
        double[] cs = Icp.centroid(src), cd = Icp.centroid(dst); int grid = 20000;
        double step = 2 * Math.PI / grid, best = 0, bestCost = Double.MAX_VALUE;
        for (int i = 0; i < grid; i++) {
            double theta = -Math.PI + i * step; double cost = angleCost(theta, src, dst, cs, cd);
            if (cost < bestCost) { bestCost = cost; best = theta; }
        }
        for (int round = 0; round < 3; round++) {
            double a = angleCost(best - step, src, dst, cs, cd); double b = angleCost(best, src, dst, cs, cd);
            double c = angleCost(best + step, src, dst, cs, cd); if (a - 2 * b + c == 0) break;
            best += 0.5 * (a - c) / (a - 2 * b + c) * step; step /= 8;
        }
        return Math.atan2(Math.sin(best), Math.cos(best));
    }

    static double angleCost(double theta, double[][] src, double[][] dst, double[] cs, double[] cd) {
        double c = Math.cos(theta), s = Math.sin(theta), sum = 0;
        for (int i = 0; i < src.length; i++) {
            double px = src[i][0] - cs[0], py = src[i][1] - cs[1];
            double dx = c * px - s * py - (dst[i][0] - cd[0]);
            double dy = s * px + c * py - (dst[i][1] - cd[1]); sum += dx * dx + dy * dy;
        }
        return sum;
    }

    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240789L); double[][] model = shape(40, 0.0, rng);
        Rigid2 planted = new Rigid2(0.6, 0.4, -0.7); double[][] moved = Icp.transform(model, planted);

        // --- the closed form against a known transform and a brute-force search -------------
        Rigid2 exact = Icp.align(model, moved);
        check(exact.poseError(planted) < 1e-12, "closed-form alignment must recover the planted pose, error "
                + exact.poseError(planted));
        double brute = bruteForceAngle(model, moved);
        check(Math.abs(Math.atan2(Math.sin(exact.theta - brute), Math.cos(exact.theta - brute))) < 1e-5,
                "closed form " + exact.theta + " must match the brute-force angle " + brute);
        double[][] mirrored = new double[40][2];
        for (int i = 0; i < 40; i++) mirrored[i] = new double[]{model[39 - i][0], model[39 - i][1]};
        check(Icp.align(mirrored, Icp.transform(mirrored, planted)).poseError(planted) < 1e-9,
                "correspondence order must not matter");

        // --- small perturbation: inside the correspondence basin ----------------------------
        Rigid2 perturbed = new Rigid2(planted.theta + 0.04, planted.tx + 0.03, planted.ty - 0.03);
        IcpRun registered = Icp.run(model, moved, perturbed, 50, 1e-12, 1.0);
        check(registered.pose.poseError(planted) < 1e-6, "ICP must converge to the planted pose, error "
                + registered.pose.poseError(planted));
        check(registered.residualTrace[registered.residualTrace.length - 1] < 1e-9,
                "exact data must register to a vanishing residual");
        check(nonIncreasing(registered.residualTrace), "the ICP objective must be non-increasing");
        check(registered.iterations <= 20, "convergence should be quick, took " + registered.iterations);

        // --- large perturbation: a strictly worse local minimum, escaped by multi-start ------
        Rigid2 far = new Rigid2(planted.theta + 0.25, planted.tx + 0.15, planted.ty - 0.2);
        IcpRun stalled = Icp.run(model, moved, far, 50, 1e-12, 1.0);
        double globalOptimum = Icp.rms(Icp.transform(model, planted), moved);
        check(globalOptimum < 1e-12, "the planted pose is the exact global optimum of the objective");
        double stalledResidual = Icp.rms(Icp.transform(model, stalled.pose), moved);
        check(stalledResidual > 1e-3, "a 0.25 rad start must stall, residual " + stalledResidual);
        IcpRun multiStart = null; double bestResidual = Double.MAX_VALUE;
        for (int i = 0; i < 48; i++) {
            IcpRun candidate = Icp.run(model, moved,
                    new Rigid2(-Math.PI + 2 * Math.PI * i / 48, perturbed.tx, perturbed.ty), 60, 1e-13, 1.0);
            double residual = candidate.residualTrace[candidate.residualTrace.length - 1];
            if (residual < bestResidual) {
                bestResidual = residual; multiStart = candidate;
            }
        }
        check(multiStart.pose.poseError(planted) < 1e-6, "a coarse multi-start escapes the local minimum, error "
                + multiStart.pose.poseError(planted));

        // --- identical clouds do nothing ------------------------------------------------------
        IcpRun trivial = Icp.run(model, model, new Rigid2(0, 0, 0), 20, 1e-9, 1.0);
        check(trivial.pose.poseError(new Rigid2(0, 0, 0)) < 1e-9 && trivial.residualTrace[0] == 0.0,
                "identical clouds must stay put with zero residual");
        check(trivial.iterations <= 1, "an aligned cloud needs at most one pass, got " + trivial.iterations);

        // --- noise: the objective still falls and the pose stays close ------------------------
        SplitMix64 noiseRng = new SplitMix64(424242L); double[][] noisy = new double[40][];
        for (int i = 0; i < 40; i++)
            noisy[i] = new double[]{moved[i][0] + 0.02 * (noiseRng.nextDouble() - 0.5),
                    moved[i][1] + 0.02 * (noiseRng.nextDouble() - 0.5)};
        IcpRun noisyRun = Icp.run(model, noisy, perturbed, 50, 1e-10, 1.0);
        double noisyResidual = noisyRun.residualTrace[noisyRun.residualTrace.length - 1];
        check(noisyRun.pose.poseError(planted) < 0.05, "noisy pose error " + noisyRun.pose.poseError(planted));
        check(nonIncreasing(noisyRun.residualTrace), "the objective must stay non-increasing under noise");
        check(noisyResidual < noisyRun.residualTrace[0], "registration must improve the initial misfit");
        check(noisyResidual < 0.03, "noisy residual " + noisyResidual);

        // --- trimming versus plain ICP with 20% garbage source points -------------------------
        SplitMix64 outlierRng = new SplitMix64(99991L); double[][] pollutedSource = Arrays.copyOf(model, 50);
        for (int i = 40; i < 50; i++) {
            double angle = 2 * Math.PI * i / 50, radius = 3.5 + outlierRng.nextDouble();
            pollutedSource[i] = new double[]{radius * Math.cos(angle), radius * Math.sin(angle)};
        }
        Rigid2 pollutedStart = new Rigid2(planted.theta + 0.03, planted.tx - 0.03, planted.ty + 0.03);
        IcpRun plainPolluted = Icp.run(pollutedSource, moved, pollutedStart, 80, 1e-12, 1.0);
        IcpRun trimmed = Icp.run(pollutedSource, moved, pollutedStart, 80, 1e-12, 0.6);
        check(trimmed.pose.poseError(planted) < plainPolluted.pose.poseError(planted) / 2,
                "trimming must beat plain ICP: " + trimmed.pose.poseError(planted) + " vs "
                        + plainPolluted.pose.poseError(planted));
        check(trimmed.pose.poseError(planted) < 0.02, "trimmed ICP pose error "
                + trimmed.pose.poseError(planted));

        // --- partial overlap, and a scale change a rigid model cannot absorb -------------------
        double[][] partial = Arrays.copyOfRange(model, 0, 25);
        IcpRun partialRun = Icp.run(partial, moved,
                new Rigid2(planted.theta + 0.04, planted.tx + 0.03, planted.ty + 0.03), 60, 1e-12, 1.0);
        check(partialRun.pose.poseError(planted) < 1e-3, "partial overlap pose error "
                + partialRun.pose.poseError(planted));
        double[][] scaled = new double[40][];
        for (int i = 0; i < 40; i++) scaled[i] = new double[]{1.2 * moved[i][0], 1.2 * moved[i][1]};
        IcpRun scaledRun = Icp.run(model, scaled, new Rigid2(planted.theta, 0, 0), 60, 1e-12, 1.0);
        check(scaledRun.residualTrace[scaledRun.residualTrace.length - 1] > 0.02,
                "a 20% scale change must leave a residual, got "
                        + scaledRun.residualTrace[scaledRun.residualTrace.length - 1]);
        check(Math.abs(Math.atan2(Math.sin(scaledRun.pose.theta - planted.theta),
                Math.cos(scaledRun.pose.theta - planted.theta))) < 0.15,
                "the rotation is still approximately recovered under scaling, got " + scaledRun.pose.theta);

        // --- degenerate clouds and repeatability ------------------------------------------------
        double[][] single = {{0.3, -0.2}}; double[][] singleMoved = Icp.transform(single, planted);
        Rigid2 singleAlign = Icp.align(single, singleMoved);
        check(Math.abs(singleAlign.theta) < 1e-15
                && Math.hypot(singleAlign.apply(single[0])[0] - singleMoved[0][0],
                        singleAlign.apply(single[0])[1] - singleMoved[0][1]) < 1e-12,
                "one correspondence fixes the translation but leaves the rotation unobservable");
        check(Double.isFinite(Icp.run(single, singleMoved, new Rigid2(0, 0, 0), 10, 1e-12, 1.0).pose.theta),
                "degenerate registration stays finite");
        IcpRun again = Icp.run(model, moved, perturbed, 50, 1e-12, 1.0);
        check(again.pose.theta == registered.pose.theta && again.pose.tx == registered.pose.tx
                && again.pose.ty == registered.pose.ty, "ICP must be deterministic");
        boolean rejected = false;
        try {
            Icp.run(model, moved, perturbed, 50, 1e-12, 0.0);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an empty trim fraction must be rejected");

        System.out.println("planted pose recovered: " + registered.pose.poseError(planted) + " in "
                + registered.iterations + " iterations, stalled start residual " + stalledResidual);
        System.out.println("noisy residual " + noisyResidual + ", pose error "
                + noisyRun.pose.poseError(planted));
        System.out.println("outliers: plain ICP error " + plainPolluted.pose.poseError(planted)
                + ", trimmed " + trimmed.pose.poseError(planted));
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
