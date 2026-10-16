/** Task 921 - Implement robot manipulation planner with grasp sampling and scoring.
 *  (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * A 2D manipulation demo. Objects are discretised by boundary points with outward normals; a two-finger
 * parallel-jaw grasp is a pair of contact points whose distance lies in the gripper's opening range, and
 * its score is the antipodality of the two normals with the closing direction, penalised by how far the
 * closing line misses the centre of mass and rewarded for spanning a large part of the object:
 *   score = align - 0.4 * offsetFromCentroid - 0.1 * (1 - span / maxSpan),   align in [0, 1].
 * The best grasp is then turned into a pick-and-place plan (approach, grasp, lift, transport, place,
 * retract) whose segments are checked against axis-aligned obstacles, with a mirrored approach as the
 * fallback. Simplified: everything is planar, the gripper is a segment (no finger geometry), the object is
 * rigid and exactly follows the gripper once grasped, and collision testing is segment-vs-AABB. Verified:
 * the circle and rectangle closed-form optima, antipodality of the chosen grasp, score monotonicity under
 * perturbation, opening limits, a collision-free plan with a working fallback, the negative controls where
 * a wall blocks the approach, infeasibility when the opening range excludes the object, and the placement
 * error after executing the plan.
 */
import java.util.*;

/** Discretised object boundary: points with outward unit normals, plus its centroid and diameter. */
final class Shape {
    final List<double[]> points = new ArrayList<>();
    final List<double[]> normals = new ArrayList<>();
    final double[] centroid;
    final double maxSpan;

    Shape(double[] centroid, double maxSpan) {
        this.centroid = centroid;
        this.maxSpan = maxSpan;
    }

    void add(double x, double y, double nx, double ny) {
        double len = Math.hypot(nx, ny);
        points.add(new double[] {x, y});
        normals.add(new double[] {nx / len, ny / len});
    }

    static Shape rectangle(double w, double h, int perSide) {
        Shape s = new Shape(new double[] {0, 0}, Math.max(w, h));
        for (int i = 0; i < perSide; i++) {
            double t = (i + 0.5) / perSide;
            s.add(w / 2, -h / 2 + h * t, 1, 0);
            s.add(-w / 2, -h / 2 + h * t, -1, 0);
            s.add(-w / 2 + w * t, h / 2, 0, 1);
            s.add(-w / 2 + w * t, -h / 2, 0, -1);
        }
        return s;
    }

    static Shape circle(double r, int n) {
        Shape s = new Shape(new double[] {0, 0}, 2 * r);
        for (int i = 0; i < n; i++) {
            double a = 2 * Math.PI * i / n;
            s.add(r * Math.cos(a), r * Math.sin(a), Math.cos(a), Math.sin(a));
        }
        return s;
    }

    int size() { return points.size(); }
}

/** A two-finger grasp: the two contact points, its span, and its quality score. */
final class Grasp {
    final int i, j;
    final double[] pi, pj, ni, nj;
    final double span, align, offset, score;

    Grasp(Shape s, int i, int j, double minSpan, double maxSpanGrip) {
        this.i = i;
        this.j = j;
        pi = s.points.get(i);
        pj = s.points.get(j);
        ni = s.normals.get(i);
        nj = s.normals.get(j);
        double dx = pj[0] - pi[0], dy = pj[1] - pi[1];
        span = Math.hypot(dx, dy);
        double ux = dx / span, uy = dy / span;
        align = 0.5 * (Math.max(0, -(ni[0] * ux + ni[1] * uy)) + Math.max(0, nj[0] * ux + nj[1] * uy));
        double midX = 0.5 * (pi[0] + pj[0]), midY = 0.5 * (pi[1] + pj[1]);
        // Distance from the centroid to the closing line, normalised by the object's own size.
        double off = Math.abs((s.centroid[0] - pi[0]) * uy - (s.centroid[1] - pi[1]) * ux);
        offset = Math.min(1, off / (0.5 * s.maxSpan));
        score = align - 0.4 * offset - 0.1 * (1 - span / s.maxSpan);
    }

    boolean withinOpening(double minSpan, double maxSpanGrip) { return span >= minSpan && span <= maxSpanGrip; }

    double[] mid() { return new double[] {0.5 * (pi[0] + pj[0]), 0.5 * (pi[1] + pj[1])}; }

    double[] axis() { double d = span; return new double[] {(pj[0] - pi[0]) / d, (pj[1] - pi[1]) / d}; }
}

/** Pick-and-place planner: waypoints plus segment-vs-obstacle collision tests. */
final class Manipulator {
    static final double FINGER_CLEARANCE = 0.02, LIFT = 0.15;

    final Shape shape;
    final double minSpan, maxSpan;

    Manipulator(Shape shape, double minSpan, double maxSpan) {
        this.shape = shape;
        this.minSpan = minSpan;
        this.maxSpan = maxSpan;
    }

    /** Sample every contact pair inside the opening range; returns the best grasp or null. */
    Grasp bestGrasp() {
        Grasp best = null;
        List<Grasp> candidates = candidates();
        for (Grasp g : candidates) if (best == null || g.score > best.score) best = g;
        return best;
    }

    List<Grasp> candidates() {
        List<Grasp> out = new ArrayList<>();
        for (int i = 0; i < shape.size(); i++) {
            for (int j = i + 1; j < shape.size(); j++) {
                Grasp g = new Grasp(shape, i, j, minSpan, maxSpan);
                // Only contacts that actually face each other can form a two-finger grasp.
                double opposed = g.ni[0] * g.nj[0] + g.ni[1] * g.nj[1];
                if (opposed < -0.5 && g.withinOpening(minSpan, maxSpan)) out.add(g);
            }
        }
        return out;
    }

    /** Waypoints of the pick-and-place: approach, grasp, lift, transport, place, retract. */
    List<double[]> plan(Grasp g, double[] goal, boolean mirrored) {
        double[] m = g.mid();
        double[] axis = g.axis();
        double sign = mirrored ? -1 : 1;
        double back = sign * (g.span / 2 + 0.1);
        List<double[]> way = new ArrayList<>();
        way.add(new double[] {m[0] + back * axis[0], m[1] + back * axis[1]});   // approach start
        way.add(new double[] {m[0], m[1]});                                    // grasp centre
        way.add(new double[] {m[0], m[1] + LIFT});                             // lift
        way.add(new double[] {goal[0], goal[1] + LIFT});                       // transport above the goal
        way.add(new double[] {goal[0], goal[1]});                              // place
        way.add(new double[] {goal[0], goal[1] + LIFT});                       // retract
        return way;
    }

    /** A plan is feasible when every waypoint leaves clearance and no segment crosses an obstacle. */
    static boolean feasible(List<double[]> way, List<double[]> obstacles) {
        for (double[] p : way) {
            for (double[] o : obstacles) if (inside(p, o, FINGER_CLEARANCE)) return false;
        }
        for (int k = 0; k + 1 < way.size(); k++) {
            for (double[] o : obstacles) if (segmentHits(way.get(k), way.get(k + 1), o, FINGER_CLEARANCE)) return false;
        }
        return true;
    }

    static boolean inside(double[] p, double[] box, double pad) {
        return p[0] >= box[0] - pad && p[0] <= box[2] + pad && p[1] >= box[1] - pad && p[1] <= box[3] + pad;
    }

    /** Exact segment vs inflated AABB test by slab clipping. */
    static boolean segmentHits(double[] a, double[] b, double[] box, double pad) {
        double x0 = box[0] - pad, x1 = box[2] + pad, y0 = box[1] - pad, y1 = box[3] + pad;
        double dx = b[0] - a[0], dy = b[1] - a[1];
        double tMin = 0, tMax = 1;
        double[] lo = {x0, y0}, hi = {x1, y1}, origin = {a[0], a[1]}, delta = {dx, dy};
        for (int axis = 0; axis < 2; axis++) {
            if (Math.abs(delta[axis]) < 1e-15) {
                if (origin[axis] < lo[axis] || origin[axis] > hi[axis]) return false;
                continue;
            }
            double t1 = (lo[axis] - origin[axis]) / delta[axis], t2 = (hi[axis] - origin[axis]) / delta[axis];
            tMin = Math.max(tMin, Math.min(t1, t2));
            tMax = Math.min(tMax, Math.max(t1, t2));
            if (tMin > tMax) return false;
        }
        return true;
    }
}

public class Solution {
    public static void main(String[] args) {
        Shape circle = Shape.circle(0.05, 24);
        Manipulator roundGripper = new Manipulator(circle, 0.04, 0.14);
        Grasp circleBest = roundGripper.bestGrasp();
        List<Grasp> circleCandidates = roundGripper.candidates();

        // Closed-form oracle 1: on a circle every antipodal pair is a diameter, so the best span is 2r.
        check(Math.abs(circleBest.span - 2 * 0.05) < 1e-9, "the circle optimum must span the diameter, got "
                + circleBest.span);
        check(circleBest.score > 0.99, "a diameter grasp on a circle is perfect, score " + circleBest.score);
        check(circleCandidates.size() >= 40, "the sampler must enumerate many facing pairs, got " + circleCandidates.size());

        Shape rect = Shape.rectangle(0.12, 0.08, 9);            // odd count so a contact pair sits exactly on the axis
        Manipulator boxGripper = new Manipulator(rect, 0.03, 0.20);
        Grasp rectBest = boxGripper.bestGrasp();
        // Closed-form oracle 2: the widest antipodal grasp of a rectangle spans its long side, so both
        // contacts sit on the two short sides (|x| = w/2) at the same height, symmetric about the centre.
        check(Math.abs(rectBest.span - 0.12) < 1e-9, "the rectangle optimum must span the long side, got " + rectBest.span);
        check(Math.abs(Math.abs(rectBest.pi[0]) - 0.06) < 1e-12 && Math.abs(rectBest.pi[0] + rectBest.pj[0]) < 1e-12
                && Math.abs(rectBest.pi[1] - rectBest.pj[1]) < 1e-12,
                "the rectangle contacts must lie on the two short sides");
        check(rectBest.score > 0.99 && rectBest.offset < 1e-12, "the rectangle optimum must be centred, score " + rectBest.score);

        // Antipodality and opening limits are structural properties of every candidate.
        double[] axis = rectBest.axis();
        check(-(rectBest.ni[0] * axis[0] + rectBest.ni[1] * axis[1]) > 0.999
                && rectBest.nj[0] * axis[0] + rectBest.nj[1] * axis[1] > 0.999,
                "the best grasp must be antipodal");
        for (Grasp g : boxGripper.candidates()) {
            check(g.withinOpening(0.03, 0.20), "candidate span " + g.span + " is outside the opening range");
            check(g.score <= 1 + 1e-12, "scores are normalised, got " + g.score + " for a +0.4 offset penalty");
        }

        // Score property: nudging a contact off the optimum must not help.
        int worse = 0;
        for (int d = 1; d <= 3; d++) {
            Grasp twisted = new Grasp(rect, rectBest.i, (rectBest.j + d) % rect.size(), 0.03, 0.20);
            if (twisted.withinOpening(0.03, 0.20) && twisted.score < rectBest.score) worse++;
        }
        check(worse >= 2, "perturbed grasps must score worse than the optimum, " + worse + " did");

        // Plan with obstacles: the mirror fallback must rescue the blocked approach.
        List<double[]> wall = new ArrayList<>();
        wall.add(new double[] {0.12, -0.2, 0.16, 0.2});
        double[] goal = {0.3, 0.35};
        List<double[]> primary = boxGripper.plan(rectBest, goal, false);
        List<double[]> mirrored = boxGripper.plan(rectBest, goal, true);
        boolean primaryOk = Manipulator.feasible(primary, wall), mirroredOk = Manipulator.feasible(mirrored, wall);
        check(primaryOk != mirroredOk, "the fixture must block exactly one approach, primary " + primaryOk
                + ", mirrored " + mirroredOk);
        List<double[]> chosen = primaryOk ? primary : mirrored;
        check(Manipulator.feasible(chosen, wall), "the chosen plan must be collision-free");
        check(primary.get(4)[0] == goal[0] && primary.get(4)[1] == goal[1], "the plan must place at the goal");

        // A choked corridor must make the whole pick-and-place infeasible.
        List<double[]> boxed = new ArrayList<>();
        boxed.add(new double[] {-0.4, -0.4, 0.4, 0.4});
        check(!Manipulator.feasible(primary, boxed) && !Manipulator.feasible(mirrored, boxed),
                "a box enclosing the object must block every approach");

        // Boundary: an opening range that excludes the object yields no grasp at all.
        Manipulator tight = new Manipulator(rect, 0.03, 0.06);
        check(tight.candidates().isEmpty() && tight.bestGrasp() == null,
                "a gripper that cannot span the object must report no grasp");
        Manipulator single = new Manipulator(rect, 0.12, 0.12);
        List<Grasp> exactly = single.candidates();
        check(!exactly.isEmpty(), "an opening range containing exactly the span must still sample grasps");

        // Executing the plan: the object rides the gripper from the grasp to the place waypoint.
        double[] object = chosen.get(1).clone();
        for (int k = 2; k <= 4; k++) {
            object[0] += chosen.get(k)[0] - chosen.get(k - 1)[0];
            object[1] += chosen.get(k)[1] - chosen.get(k - 1)[1];
        }
        check(Math.abs(object[0] - goal[0]) < 1e-12 && Math.abs(object[1] - goal[1]) < 1e-12,
                "after the grasp the object must follow the gripper to the goal: " + object[0] + "," + object[1]);
        check(chosen.size() == 6 && new Manipulator(rect, 0.03, 0.20).bestGrasp().score == rectBest.score,
                "the plan has six waypoints and the sampler is deterministic");

        System.out.println("grasp sampling: circle " + circle.size() + " points -> " + circleCandidates.size()
                + " pairs, best span " + String.format("%.4f", circleBest.span) + " (diameter 0.1), score "
                + String.format("%.4f", circleBest.score));
        System.out.println("rectangle 0.12x0.08 -> " + boxGripper.candidates().size() + " pairs, best span "
                + String.format("%.4f", rectBest.span) + ", align " + String.format("%.4f", rectBest.align)
                + ", offset " + String.format("%.6f", rectBest.offset) + ", score " + String.format("%.4f", rectBest.score));
        System.out.println("plan: primary feasible " + primaryOk + ", mirrored feasible " + mirroredOk
                + ", waypoints " + chosen.size() + ", objects grasped " + (tight.bestGrasp() == null ? "none" : "one"));
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
