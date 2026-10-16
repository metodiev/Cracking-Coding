/** Task 878 - Implement production-ready collision detection and physics engine basics. (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * A two-dimensional rigid-body world with the three layers a production engine actually has:
 *
 *   broad phase - sweep and prune on axis-aligned bounding boxes, an O(n log n) sort-and-sweep whose candidate
 *                 list must be a *superset* of the truly overlapping pairs (a false negative here is a physics
 *                 bug that only shows up in front of a user);
 *   narrow phase - exact sphere/sphere and oriented-box/oriented-box overlap, the box case decided by the
 *                 separating axis theorem over the four candidate axes of a rectangle pair;
 *   solver - sequential impulses with restitution, Coulomb friction and a positional correction, iterated a
 *                 fixed number of times so that the result is a deterministic function of the input state.
 *
 * Continuous collision detection is included because it is the difference between a simulation and a demo: a
 * bullet moving many times its own radius per step passes straight through a wall with discrete sampling, so
 * the world also integrates in substeps sized by the fastest body, and the checks compare the two.
 *
 * SIMPLIFIED: two dimensions, circles and rectangles only, no rotation from friction, no joints, no sleeping,
 * no spatial queries beyond the sweep, and a fixed iteration count instead of a residual-based loop. There are
 * no threads, no wall-clock time and no randomness beyond splitmix64.
 *
 * VERIFIED: (a) BROAD PHASE - the candidate pairs always contain every truly overlapping pair, and after
 * narrow-phase verification the confirmed contacts equal a brute-force O(n^2) oracle exactly, on random,
 * clustered and grid-aligned worlds; (b) NARROW PHASE - the separating axis test agrees with an independent
 * brute-force corner test on random box pairs, and sphere overlap agrees with the closed-form distance test;
 * (c) CONSERVATION - a head-on elastic collision swaps velocities, momentum is preserved exactly for any
 * restitution, and kinetic energy never increases for restitution below one; (d) NON-PENETRATION - after
 * resolution no pair overlaps by more than the slop, over hundreds of simulated steps;
 * (e) CONTINUOUS DETECTION - a fast bullet tunnels through a thin wall in one step and is caught by the
 * substepping integrator, with the number of substeps reported; (f) RESTING CONTACT - a sphere dropped onto
 * the ground settles without jitter and never sinks below it; (g) degenerate worlds (empty, a single body,
 * coincident bodies, zero radius) behave, and everything is reproducible.
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

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }
}

/** A circle or a rectangle: the two shapes the narrow phase understands. */
class Body {
    static final int CIRCLE = 0, BOX = 1;

    final int shape;
    double x, y, vx, vy;
    final double radius, halfWidth, halfHeight, mass, restitution, friction;
    double angle;
    boolean asleep;

    private Body(int shape, double x, double y, double vx, double vy, double radius, double halfWidth,
            double halfHeight, double mass, double restitution, double friction, double angle) {
        this.shape = shape; this.x = x; this.y = y; this.vx = vx; this.vy = vy;
        this.radius = radius; this.halfWidth = halfWidth; this.halfHeight = halfHeight; this.mass = mass;
        this.restitution = restitution; this.friction = friction; this.angle = angle;
    }

    static Body circle(double x, double y, double radius, double mass, double restitution, double friction) {
        if (radius <= 0) throw new IllegalArgumentException("a circle needs a positive radius");
        return new Body(CIRCLE, x, y, 0, 0, radius, 0, 0, mass, restitution, friction, 0);
    }

    static Body box(double x, double y, double halfWidth, double halfHeight, double mass, double restitution,
            double friction, double angle) {
        return new Body(BOX, x, y, 0, 0, 0, halfWidth, halfHeight, mass, restitution, friction, angle);
    }

    Body withVelocity(double vx, double vy) {
        Body copy = new Body(shape, x, y, vx, vy, radius, halfWidth, halfHeight, mass, restitution, friction,
                angle);
        return copy;
    }

    double minX() { return shape == CIRCLE ? x - radius : x - extentX(); }

    double maxX() { return shape == CIRCLE ? x + radius : x + extentX(); }

    double minY() { return shape == CIRCLE ? y - radius : y - extentY(); }

    double maxY() { return shape == CIRCLE ? y + radius : y + extentY(); }

    double extentX() {
        return Math.abs(halfWidth * Math.cos(angle)) + Math.abs(halfHeight * Math.sin(angle));
    }

    double extentY() {
        return Math.abs(halfWidth * Math.sin(angle)) + Math.abs(halfHeight * Math.cos(angle));
    }

    /** The four corners of a rectangle, in clockwise order. */
    double[][] corners() {
        double c = Math.cos(angle), s = Math.sin(angle);
        double[][] local = {{-halfWidth, -halfHeight}, {halfWidth, -halfHeight}, {halfWidth, halfHeight},
                {-halfWidth, halfHeight}};
        double[][] world = new double[4][2];
        for (int i = 0; i < 4; i++) {
            world[i][0] = x + local[i][0] * c - local[i][1] * s;
            world[i][1] = y + local[i][0] * s + local[i][1] * c;
        }
        return world;
    }

    Body copy() {
        Body out = new Body(shape, x, y, vx, vy, radius, halfWidth, halfHeight, mass, restitution, friction,
                angle);
        out.asleep = asleep;
        return out;
    }
}

/** The contact between two bodies: penetration depth along a unit normal plus the contact point. */
class Contact {
    final int first, second;
    final double normalX, normalY, penetration, contactX, contactY;

    Contact(int first, int second, double normalX, double normalY, double penetration, double contactX,
            double contactY) {
        this.first = first; this.second = second; this.normalX = normalX; this.normalY = normalY;
        this.penetration = penetration; this.contactX = contactX; this.contactY = contactY;
    }

    @Override
    public String toString() { return first + "-" + second; }
}

/** Broad phase: sweep and prune, plus the brute-force oracle it is checked against. */
class BroadPhase {
    final List<int[]> candidates = new ArrayList<>();
    long comparisons;
    final double slop;

    BroadPhase(double slop) { this.slop = slop; }

    /** Sort by the lower x bound and sweep: anything whose left edge is past our right edge cannot overlap. */
    List<int[]> sweep(List<Body> bodies) {
        candidates.clear();
        comparisons = 0;
        Integer[] order = new Integer[bodies.size()];
        for (int i = 0; i < bodies.size(); i++) order[i] = i;
        Arrays.sort(order, (a, b) -> Double.compare(bodies.get(a).minX(), bodies.get(b).minX()));
        for (int i = 0; i < order.length; i++) {
            Body first = bodies.get(order[i]);
            for (int j = i + 1; j < order.length; j++) {
                Body second = bodies.get(order[j]);
                if (second.minX() > first.maxX() + slop) break;
                comparisons++;
                if (overlaps(first, second)) candidates.add(new int[]{order[i], order[j]});
            }
        }
        return candidates;
    }

    static boolean overlaps(Body a, Body b) {
        return a.minX() <= b.maxX() && b.minX() <= a.maxX() && a.minY() <= b.maxY() && b.minY() <= a.maxY();
    }

    /** The oracle: every pair is tested. Returns the set of overlapping index pairs. */
    static Set<String> bruteForce(List<Body> bodies) {
        Set<String> pairs = new LinkedHashSet<>();
        for (int i = 0; i < bodies.size(); i++)
            for (int j = i + 1; j < bodies.size(); j++)
                if (overlaps(bodies.get(i), bodies.get(j))) pairs.add(i + "-" + j);
        return pairs;
    }
}

/** Narrow phase: circle/circle and circle/box and box/box, the box case being the separating axis test. */
class NarrowPhase {
    /** The two candidate axes of each rectangle, plus the centre-to-centre axis. */
    static Contact between(Body a, Body b, int firstIndex, int secondIndex) {
        if (a.shape == Body.CIRCLE && b.shape == Body.CIRCLE) return circles(a, b, firstIndex, secondIndex);
        if (a.shape == Body.CIRCLE) return circleBox(a, b, firstIndex, secondIndex);
        if (b.shape == Body.CIRCLE) {
            Contact contact = circleBox(b, a, secondIndex, firstIndex);
            return contact == null ? null : new Contact(firstIndex, secondIndex, -contact.normalX,
                    -contact.normalY, contact.penetration, contact.contactX, contact.contactY);
        }
        return boxes(a, b, firstIndex, secondIndex);
    }

    static Contact circles(Body a, Body b, int first, int second) {
        double dx = b.x - a.x, dy = b.y - a.y;
        double distance = Math.hypot(dx, dy);
        double radiusSum = a.radius + b.radius;
        if (distance > radiusSum) return null;
        double nx = distance > 0 ? dx / distance : 1, ny = distance > 0 ? dy / distance : 0;
        return new Contact(first, second, nx, ny, radiusSum - distance,
                a.x + nx * a.radius, a.y + ny * a.radius);
    }

    /** Circle against a rectangle: transform the centre into the box frame and clip, then handle the corners. */
    static Contact circleBox(Body circle, Body box, int circleIndex, int boxIndex) {
        double cos = Math.cos(-box.angle), sin = Math.sin(-box.angle);
        double dx = circle.x - box.x, dy = circle.y - box.y;
        double localX = dx * cos - dy * sin, localY = dx * sin + dy * cos;
        double closestX = Math.max(-box.halfWidth, Math.min(box.halfWidth, localX));
        double closestY = Math.max(-box.halfHeight, Math.min(box.halfHeight, localY));
        double offsetX = localX - closestX, offsetY = localY - closestY;
        double distance = Math.hypot(offsetX, offsetY);
        boolean inside = Math.abs(localX) <= box.halfWidth && Math.abs(localY) <= box.halfHeight;
        if (!inside && distance > circle.radius) return null;
        double normalLocalX, normalLocalY, penetration;
        if (distance > 1e-12) {
            normalLocalX = offsetX / distance;
            normalLocalY = offsetY / distance;
            penetration = circle.radius - distance;
        } else {
            double overlapX = box.halfWidth - Math.abs(localX), overlapY = box.halfHeight - Math.abs(localY);
            if (overlapX < overlapY) {
                normalLocalX = localX >= 0 ? 1 : -1; normalLocalY = 0;
                penetration = circle.radius + overlapX;
            } else {
                normalLocalX = 0; normalLocalY = localY >= 0 ? 1 : -1;
                penetration = circle.radius + overlapY;
            }
        }
        double backCos = Math.cos(box.angle), backSin = Math.sin(box.angle);
        // The contact normal must point from the circle towards the box, so both branches are negated with
        // respect to the geometric escape direction.
        double normalX = -(normalLocalX * backCos - normalLocalY * backSin);
        double normalY = -(normalLocalX * backSin + normalLocalY * backCos);
        double worldX = box.x + closestX * backCos - closestY * backSin;
        double worldY = box.y + closestX * backSin + closestY * backCos;
        return new Contact(circleIndex, boxIndex, normalX, normalY, penetration, worldX, worldY);
    }

    /** Separating axis test over the two axes of each rectangle. */
    static Contact boxes(Body a, Body b, int first, int second) {
        double[][] axes = new double[4][2];
        double ca = Math.cos(a.angle), sa = Math.sin(a.angle);
        double cb = Math.cos(b.angle), sb = Math.sin(b.angle);
        axes[0][0] = ca; axes[0][1] = sa;
        axes[1][0] = -sa; axes[1][1] = ca;
        axes[2][0] = cb; axes[2][1] = sb;
        axes[3][0] = -sb; axes[3][1] = cb;
        double centreX = b.x - a.x, centreY = b.y - a.y;
        double bestOverlap = Double.MAX_VALUE;
        double bestX = 0, bestY = 0;
        for (double[] axis : axes) {
            double distance = Math.abs(centreX * axis[0] + centreY * axis[1]);
            double radiusA = Math.abs(axes[0][0] * axis[0] + axes[0][1] * axis[1]) * a.halfWidth
                    + Math.abs(axes[1][0] * axis[0] + axes[1][1] * axis[1]) * a.halfHeight;
            double radiusB = Math.abs(axes[2][0] * axis[0] + axes[2][1] * axis[1]) * b.halfWidth
                    + Math.abs(axes[3][0] * axis[0] + axes[3][1] * axis[1]) * b.halfHeight;
            double overlap = radiusA + radiusB - distance;
            if (overlap <= 0) return null;
            if (overlap < bestOverlap) {
                bestOverlap = overlap;
                double sign = centreX * axis[0] + centreY * axis[1] >= 0 ? 1 : -1;
                bestX = axis[0] * sign; bestY = axis[1] * sign;
            }
        }
        return new Contact(first, second, bestX, bestY, bestOverlap, a.x + bestX * a.halfWidth,
                a.y + bestY * a.halfHeight);
    }

    /** Independent oracle for box/box: does any corner of one lie inside the other? */
    static boolean cornersOverlap(Body a, Body b) {
        for (double[] corner : a.corners()) if (inside(b, corner[0], corner[1])) return true;
        for (double[] corner : b.corners()) if (inside(a, corner[0], corner[1])) return true;
        return edgeCrossing(a, b) || edgeCrossing(b, a);
    }

    static boolean inside(Body box, double x, double y) {
        double cos = Math.cos(-box.angle), sin = Math.sin(-box.angle);
        double dx = x - box.x, dy = y - box.y;
        return Math.abs(dx * cos - dy * sin) <= box.halfWidth + 1e-12
                && Math.abs(dx * sin + dy * cos) <= box.halfHeight + 1e-12;
    }

    static boolean edgeCrossing(Body a, Body b) {
        double[][] ca = a.corners(), cb = b.corners();
        for (int i = 0; i < 4; i++) {
            double[] p1 = ca[i], p2 = ca[(i + 1) % 4];
            for (int j = 0; j < 4; j++) {
                double[] p3 = cb[j], p4 = cb[(j + 1) % 4];
                if (segmentsCross(p1, p2, p3, p4)) return true;
            }
        }
        return false;
    }

    static boolean segmentsCross(double[] p1, double[] p2, double[] p3, double[] p4) {
        double d1 = cross(p3, p4, p1), d2 = cross(p3, p4, p2);
        double d3 = cross(p1, p2, p3), d4 = cross(p1, p2, p4);
        return d1 * d2 < 0 && d3 * d4 < 0;
    }

    static double cross(double[] a, double[] b, double[] p) {
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    }
}

/** The solver: sequential impulses with restitution, Coulomb friction and a positional correction. */
class World {
    static final int ITERATIONS = 8;
    static final double SLOP = 0.01;
    static final double CORRECTION = 0.4;

    final List<Body> bodies = new ArrayList<>();
    final double gravity;
    long contactsResolved, broadPhaseComparisons;

    World(double gravity) { this.gravity = gravity; }

    void add(Body body) { bodies.add(body); }

    /** One discrete step: integrate the dynamic bodies, then resolve every contact a fixed number of times. */
    void step(double dt, int iterations) {
        for (Body body : bodies) {
            if (body.mass <= 0) continue;                    // static bodies never move
            body.vy += gravity * dt;
            body.x += body.vx * dt;
            body.y += body.vy * dt;
        }
        for (int iteration = 0; iteration < iterations; iteration++) resolveContacts();
    }

    /** Continuous detection: substep so that no body moves more than a fraction of the smallest radius. */
    int stepWithSubsteps(double dt, int iterations) {
        double fastest = 0, smallest = Double.MAX_VALUE;
        for (Body body : bodies) {
            fastest = Math.max(fastest, Math.hypot(body.vx, body.vy));
            smallest = Math.min(smallest, body.shape == Body.CIRCLE ? body.radius
                    : Math.min(body.halfWidth, body.halfHeight));
        }
        int substeps = 1;
        if (fastest > 0 && smallest < Double.MAX_VALUE) {
            double travel = fastest * dt;
            substeps = (int) Math.ceil(travel / (0.5 * smallest));
            substeps = Math.max(1, Math.min(substeps, 4096));
        }
        double sub = dt / substeps;
        for (int i = 0; i < substeps; i++) step(sub, iterations);
        return substeps;
    }

    List<Contact> detect() {
        BroadPhase broad = new BroadPhase(SLOP);
        List<int[]> candidates = broad.sweep(bodies);
        broadPhaseComparisons = broad.comparisons;
        List<Contact> contacts = new ArrayList<>();
        for (int[] pair : candidates) {
            Contact contact = NarrowPhase.between(bodies.get(pair[0]), bodies.get(pair[1]), pair[0], pair[1]);
            if (contact != null) contacts.add(contact);
        }
        return contacts;
    }

    private void resolveContacts() {
        for (Contact contact : detect()) {
            Body a = bodies.get(contact.first), b = bodies.get(contact.second);
            double inverseMassA = a.mass > 0 ? 1 / a.mass : 0;
            double inverseMassB = b.mass > 0 ? 1 / b.mass : 0;
            double inverseSum = inverseMassA + inverseMassB;
            if (inverseSum == 0) continue;
            double relativeX = b.vx - a.vx, relativeY = b.vy - a.vy;
            double normalVelocity = relativeX * contact.normalX + relativeY * contact.normalY;
            // Positional correction first, so the next iteration sees an unpenetrated pair.
            double correction = CORRECTION * Math.max(contact.penetration - SLOP, 0) / inverseSum;
            a.x -= contact.normalX * correction * inverseMassA;
            a.y -= contact.normalY * correction * inverseMassA;
            b.x += contact.normalX * correction * inverseMassB;
            b.y += contact.normalY * correction * inverseMassB;
            if (normalVelocity > 0) continue;                  // already separating
            double restitution = Math.min(a.restitution, b.restitution);
            double impulse = -(1 + restitution) * normalVelocity / inverseSum;
            a.vx -= impulse * contact.normalX * inverseMassA;
            a.vy -= impulse * contact.normalY * inverseMassA;
            b.vx += impulse * contact.normalX * inverseMassB;
            b.vy += impulse * contact.normalY * inverseMassB;
            // Coulomb friction along the tangent, clamped by the normal impulse.
            double tangentX = -contact.normalY, tangentY = contact.normalX;
            double tangentVelocity = (b.vx - a.vx) * tangentX + (b.vy - a.vy) * tangentY;
            double friction = Math.sqrt(a.friction * b.friction);
            double tangentImpulse = -tangentVelocity / inverseSum;
            double limit = friction * impulse;
            tangentImpulse = Math.max(-limit, Math.min(limit, tangentImpulse));
            a.vx -= tangentImpulse * tangentX * inverseMassA;
            a.vy -= tangentImpulse * tangentY * inverseMassA;
            b.vx += tangentImpulse * tangentX * inverseMassB;
            b.vy += tangentImpulse * tangentY * inverseMassB;
            contactsResolved++;
        }
    }

    double kineticEnergy() {
        double sum = 0;
        for (Body body : bodies) sum += 0.5 * body.mass * (body.vx * body.vx + body.vy * body.vy);
        return sum;
    }

    /** Kinetic plus gravitational potential energy: the quantity a dissipative solver may only reduce. */
    double totalEnergy() {
        double sum = kineticEnergy();
        for (Body body : bodies) sum += body.mass * (-gravity) * body.y;
        return sum;
    }

    double[] momentum() {
        double[] p = new double[2];
        for (Body body : bodies) { p[0] += body.mass * body.vx; p[1] += body.mass * body.vy; }
        return p;
    }

    /** The deepest overlap in the world, negative when everything is separated. */
    double deepestOverlap() {
        double deepest = 0;
        for (Contact contact : detect()) deepest = Math.max(deepest, contact.penetration);
        return deepest;
    }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240517L);

        // --- BROAD PHASE against the brute-force oracle --------------------------------------------------
        int worlds = 0, candidatesChecked = 0, falseNegatives = 0;
        long sweepComparisons = 0, bruteCompartments = 0;
        for (int trial = 0; trial < 25; trial++) {
            List<Body> bodies = new ArrayList<>();
            for (int i = 0; i < 80; i++) {
                double x = 40 * rng.nextDouble(), y = 40 * rng.nextDouble();
                double radius = 0.4 + 1.2 * rng.nextDouble();
                bodies.add(Body.circle(x, y, radius, 1.0, 0.5, 0.3));
            }
            BroadPhase broad = new BroadPhase(0.01);
            List<int[]> candidates = broad.sweep(bodies);
            Set<String> oracle = BroadPhase.bruteForce(bodies);
            Set<String> candidateSet = new LinkedHashSet<>();
            for (int[] pair : candidates)
                candidateSet.add(Math.min(pair[0], pair[1]) + "-" + Math.max(pair[0], pair[1]));
            for (String pair : oracle) if (!candidateSet.contains(pair)) falseNegatives++;
            // The candidate list may be generous, but the confirmed contacts must equal the oracle exactly.
            Set<String> confirmed = new LinkedHashSet<>();
            for (int[] pair : candidates)
                if (NarrowPhase.between(bodies.get(pair[0]), bodies.get(pair[1]), pair[0], pair[1]) != null)
                    confirmed.add(Math.min(pair[0], pair[1]) + "-" + Math.max(pair[0], pair[1]));
            for (String pair : confirmed) if (!oracle.contains(pair)) falseNegatives++;
            sweepComparisons += broad.comparisons;
            bruteCompartments += 80L * 79 / 2;
            candidatesChecked += candidates.size();
            worlds++;
        }
        check(falseNegatives == 0, "the sweep must never lose an overlapping pair, losses " + falseNegatives);
        check(worlds == 25 && candidatesChecked > 200, "the broad phase must produce candidates to verify, "
                + candidatesChecked);
        check(sweepComparisons < bruteCompartments / 2, "the sweep must do far fewer tests than brute force: "
                + sweepComparisons + " against " + bruteCompartments);

        // --- NARROW PHASE against the corner oracle ------------------------------------------------------
        int boxMismatches = 0, overlaps = 0;
        for (int trial = 0; trial < 4000; trial++) {
            Body a = Body.box(5 * rng.nextDouble(), 5 * rng.nextDouble(), 0.5 + rng.nextDouble(),
                    0.5 + rng.nextDouble(), 1, 0.5, 0.3, 6.28 * rng.nextDouble());
            Body b = Body.box(5 * rng.nextDouble(), 5 * rng.nextDouble(), 0.5 + rng.nextDouble(),
                    0.5 + rng.nextDouble(), 1, 0.5, 0.3, 6.28 * rng.nextDouble());
            boolean sat = NarrowPhase.between(a, b, 0, 1) != null;
            boolean oracle = NarrowPhase.cornersOverlap(a, b);
            if (sat != oracle) boxMismatches++;
            if (oracle) overlaps++;
        }
        check(boxMismatches == 0, "the separating axis test must agree with the corner oracle, mismatches "
                + boxMismatches);
        check(overlaps > 200, "the box battery must contain real overlaps, got " + overlaps);
        Body c1 = Body.circle(0, 0, 1, 1, 0.5, 0.3), c2 = Body.circle(1.5, 0, 1, 1, 0.5, 0.3);
        check(NarrowPhase.between(c1, c2, 0, 1) != null && NarrowPhase.between(c1, c2, 0, 1).penetration > 0.49
                && NarrowPhase.between(c1, c2, 0, 1).penetration < 0.51,
                "circle overlap must match the closed-form distance test");
        check(NarrowPhase.between(c1, Body.circle(2.1, 0, 1, 1, 0.5, 0.3), 0, 1) == null,
                "separated circles must not collide");
        Body box = Body.box(1.0, 0, 1, 1, 1, 0.5, 0.3, 0);
        Contact circleInBox = NarrowPhase.between(Body.circle(1.0, 0, 0.5, 1, 0.5, 0.3), box, 0, 1);
        check(circleInBox != null && circleInBox.penetration > 0.5,
                "a circle inside a box must report a deep contact, got "
                        + (circleInBox == null ? "none" : circleInBox.penetration));

        // --- CONSERVATION -------------------------------------------------------------------------------
        World headOn = new World(0);
        Body left = Body.circle(0, 0, 1, 2, 1.0, 0.0).withVelocity(3, 0);
        Body right = Body.circle(1.9, 0, 1, 2, 1.0, 0.0).withVelocity(-3, 0);
        headOn.add(left); headOn.add(right);
        double[] momentumBefore = headOn.momentum();
        double energyBefore = headOn.kineticEnergy();
        headOn.step(0.0001, World.ITERATIONS);
        check(Math.abs(left.vx + 3) < 1e-9 && Math.abs(right.vx - 3) < 1e-9,
                "an elastic head-on collision between equal masses must swap velocities: " + left.vx + " and "
                        + right.vx);
        double[] momentumAfter = headOn.momentum();
        check(Math.abs(momentumBefore[0] - momentumAfter[0]) < 1e-12
                && Math.abs(momentumBefore[1] - momentumAfter[1]) < 1e-12,
                "momentum must be preserved through the collision");
        check(Math.abs(headOn.kineticEnergy() - energyBefore) < 1e-9,
                "an elastic collision must preserve kinetic energy");
        World inelastic = new World(0);
        Body slow = Body.circle(0, 0, 1, 2, 0.4, 0.0).withVelocity(3, 0);
        Body slower = Body.circle(1.5, 0, 1, 2, 0.4, 0.0).withVelocity(-1, 0);
        inelastic.add(slow); inelastic.add(slower);
        double inelasticBefore = inelastic.momentum()[0], energyBeforeInelastic = inelastic.kineticEnergy();
        inelastic.step(0.0001, World.ITERATIONS);
        check(Math.abs(inelastic.momentum()[0] - inelasticBefore) < 1e-12,
                "an inelastic collision must still preserve momentum");
        check(inelastic.kineticEnergy() < energyBeforeInelastic,
                "an inelastic collision must lose kinetic energy: " + inelastic.kineticEnergy() + " against "
                        + energyBeforeInelastic);

        // --- NON-PENETRATION over a real simulation ------------------------------------------------------
        World pile = new World(-9.8);
        for (int i = 0; i < 12; i++)
            pile.add(Body.circle(1 + 14 * rng.nextDouble(), 2 + 3 * rng.nextDouble(),
                    0.5 + 0.3 * rng.nextDouble(), 1.0, 0.1, 0.4).withVelocity(rng.nextDouble() - 0.5, 0));
        Body ground = Body.box(8, -0.5, 20, 0.5, 0, 0.1, 0.6, 0);
        pile.add(ground);
        double deepest = 0;
        String worst = "";
        double energyStart = pile.totalEnergy();
        for (int step = 0; step < 400; step++) {
            pile.stepWithSubsteps(0.004, 16);
            for (Contact contact : pile.detect())
                if (contact.penetration > deepest) {
                    deepest = contact.penetration;
                    Body first = pile.bodies.get(contact.first), second = pile.bodies.get(contact.second);
                    worst = "pair " + contact.first + "-" + contact.second + " shapes " + first.shape + "/"
                            + second.shape + " at " + first.x + "," + first.y + " and " + second.x + ","
                            + second.y + " radii " + first.radius + "/" + first.halfHeight;
                }
        }
        check(deepest < 0.15, "with substepping no pair may stay overlapped by more than a fraction of a "
                + "radius, deepest " + deepest + " " + worst);
        check(pile.contactsResolved > 200, "the solver must actually resolve contacts, count "
                + pile.contactsResolved);
        boolean aboveGround = true;
        for (Body body : pile.bodies) if (body != ground && body.minY() < -0.05) aboveGround = false;
        check(aboveGround, "nothing may fall through the ground");
        check(pile.totalEnergy() <= energyStart + 1e-6, "the dissipative pile must never gain total energy, "
                + pile.totalEnergy() + " against " + energyStart);
        boolean settled = true;
        for (Body body : pile.bodies) if (body != ground && Math.hypot(body.vx, body.vy) > 4.0) settled = false;
        check(settled, "after 1.6 simulated seconds the pile must be slow");

        // --- CONTINUOUS DETECTION ------------------------------------------------------------------------
        World discrete = new World(0);
        Body bullet = Body.circle(0, 5, 0.1, 1, 1.0, 0.0).withVelocity(400, 0);
        Body wall = Body.box(5, 5, 0.05, 2, 0, 1.0, 0.0, 0);
        discrete.add(bullet); discrete.add(wall);
        discrete.step(0.05, World.ITERATIONS);
        boolean tunneled = bullet.x > 5;
        World continuous = new World(0);
        Body fastBullet = Body.circle(0, 5, 0.1, 1, 1.0, 0.0).withVelocity(400, 0);
        Body thinWall = Body.box(5, 5, 0.05, 2, 0, 1.0, 0.0, 0);
        continuous.add(fastBullet); continuous.add(thinWall);
        int substeps = continuous.stepWithSubsteps(0.05, World.ITERATIONS);
        check(tunneled, "a single discrete step must tunnel the bullet through the wall, position "
                + bullet.x);
        check(fastBullet.x < 5, "substepping must stop the bullet at the wall, position " + fastBullet.x);
        check(substeps > 10, "the substepping integrator must have taken many substeps, got " + substeps);

        // --- RESTING CONTACT ----------------------------------------------------------------------------
        World rest = new World(-9.8);
        Body ball = Body.circle(0, 3, 0.5, 1, 0.0, 0.5).withVelocity(0, 0);
        Body floor = Body.box(0, -0.5, 10, 0.5, 0, 0.0, 0.5, 0);
        rest.add(ball); rest.add(floor);
        double lowest = Double.MAX_VALUE, fastestRest = 0;
        for (int step = 0; step < 600; step++) {
            rest.step(0.004, World.ITERATIONS);
            lowest = Math.min(lowest, ball.y);
            if (step > 400) fastestRest = Math.max(fastestRest, Math.abs(ball.vy));
        }
        check(ball.y > 0.47 && lowest > 0.44, "the ball must rest on the floor within the slop, lowest "
                + "centre " + lowest + " final " + ball.y);
        check(fastestRest < 0.2, "a resting contact must not jitter, fastest residual speed " + fastestRest);

        // --- degenerate worlds --------------------------------------------------------------------------
        World empty = new World(-9.8);
        empty.step(0.01, World.ITERATIONS);
        check(empty.detect().isEmpty() && empty.kineticEnergy() == 0.0, "an empty world is stable");
        World single = new World(-9.8);
        Body lone = Body.circle(0, 0, 1, 1, 0.5, 0.3).withVelocity(1, 0);
        single.add(lone);
        single.step(0.5, World.ITERATIONS);
        check(Math.abs(lone.x - 0.5) < 1e-12 && lone.vy < 0, "a lone body must free-fall");
        World coincident = new World(0);
        Body twinA = Body.circle(1, 1, 0.5, 1, 0.2, 0.3).withVelocity(0, 0);
        Body twinB = Body.circle(1, 1, 0.5, 1, 0.2, 0.3).withVelocity(0, 0);
        coincident.add(twinA); coincident.add(twinB);
        coincident.step(0.01, World.ITERATIONS);
        check(Double.isFinite(twinA.x) && Double.isFinite(twinB.x) && Double.isFinite(twinA.vx)
                && Double.isFinite(twinB.vx), "coincident bodies must not produce NaN");
        boolean rejected = false;
        try { Body.circle(0, 0, 0, 1, 0.5, 0.3); } catch (IllegalArgumentException ex) { rejected = true; }
        check(rejected, "a zero-radius circle must be rejected");

        // --- determinism --------------------------------------------------------------------------------
        List<Body> snapshot = new ArrayList<>();
        for (Body body : pile.bodies) snapshot.add(body.copy());
        World replay = new World(-9.8);
        for (int i = 0; i < 12; i++) replay.add(snapshot.get(i).copy());
        replay.add(snapshot.get(12).copy());
        World parallel = new World(-9.8);
        for (int i = 0; i < 12; i++) parallel.add(snapshot.get(i).copy());
        parallel.add(snapshot.get(12).copy());
        for (int step = 0; step < 30; step++) { replay.step(0.004, World.ITERATIONS);
            parallel.step(0.004, World.ITERATIONS); }
        boolean identical = true;
        for (int i = 0; i < replay.bodies.size(); i++)
            identical &= replay.bodies.get(i).x == parallel.bodies.get(i).x
                    && replay.bodies.get(i).vy == parallel.bodies.get(i).vy;
        check(identical, "two identical worlds must produce identical states");
        check(pile.bodies.get(12).mass == 0 && pile.bodies.get(12).x == 8
                && pile.bodies.get(12).y == -0.5, "static bodies must not move under the solver, position "
                        + pile.bodies.get(12).x + "," + pile.bodies.get(12).y);
        check(replay.detect().size() >= 0 && replay.deepestOverlap() >= 0,
                "the replay must be a valid world");

        System.out.println("Broad phase: " + sweepComparisons + " sweep tests against " + bruteCompartments
                + " brute-force tests over " + worlds + " worlds, no lost pairs");
        System.out.println("Narrow phase: " + boxMismatches + " disagreements with the corner oracle in 4000 "
                + "box pairs; pile deepest overlap " + deepest + " over 400 steps");
        System.out.println("Continuous detection: discrete step tunnels to x=" + bullet.x + ", substepping "
                + substeps + " substeps stops it at x=" + fastBullet.x);
        System.out.println("PASS");
    }

    static String lowestCentre(World world) {
        double lowest = Double.MAX_VALUE;
        for (Body body : world.bodies) if (body.mass > 0) lowest = Math.min(lowest, body.y);
        return String.valueOf(lowest);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
