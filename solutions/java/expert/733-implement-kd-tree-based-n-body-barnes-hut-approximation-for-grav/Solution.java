/** Task 733 - Implement kd-tree based n-body Barnes-Hut approximation for gravity. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * Barnes-Hut replaces the O(N^2) all-pairs force with a tree walk: build a kd-tree over the bodies, store the
 * total mass and centre of mass in every node, and when a node's size s is small compared with the distance d
 * to the target body (s/d < theta) treat the whole node as one point mass. Everything else is recursed into,
 * and leaf bodies are summed directly. Theta therefore dials accuracy against cost: theta = 0 recurses to the
 * leaves everywhere and must reproduce the direct sum exactly, while larger theta cuts the interaction count.
 *
 * Invariants verified:
 *   - tree       the root mass equals the sum of all masses, every body belongs to exactly one leaf, every
 *                leaf respects its capacity, and each body lies inside the bounds of its own leaf
 *   - exactness  theta = 0 reproduces the direct all-pairs force to machine precision, and a two-body system
 *                is exact for every theta below one
 *   - control    the relative error grows with theta (0.2 < 0.5 < 1.0) and stays small at theta = 0.5
 *   - cost       the number of body-node interactions at theta = 0.5 is a small fraction of N^2, and a
 *                mirror-symmetric configuration keeps its total force near zero
 *
 * Simplified: two dimensions, a midpoint-split kd-tree with no parallelism and Plummer softening to keep the
 * forces finite. The verified properties are the tree invariants, exactness at theta = 0, the monotone error
 * control, and the interaction-count saving.
 */
import java.util.*;

public class Solution {
    /** A point mass with Plummer softening. */
    static final class Body {
        final double x;
        final double y;
        final double mass;
        Body(double x, double y, double mass) {
            this.x = x;
            this.y = y;
            this.mass = mass;
        }
    }

    /** One kd-tree node: bounds, aggregated mass, and either children or bodies. */
    static final class Node {
        double minX;
        double maxX;
        double minY;
        double maxY;
        double mass;
        double centerX;
        double centerY;
        Node first;
        Node second;
        int[] bodies = new int[0];
        int leafCount;
        void aggregate() { mass = centerX = centerY = 0.0; }
        double size() { return Math.max(maxX - minX, maxY - minY); }
    }

    /** The softened gravitational field of a set of bodies. */
    static final class NBody {
        final List<Body> bodies;
        final double softening;
        final double gravity;
        Node root;
        long interactions;

        NBody(List<Body> bodies, double softening, double gravity) {
            this.bodies = bodies;
            this.softening = softening;
            this.gravity = gravity;
            this.root = build(indices(bodies.size()), 0, bodies.size());
        }

        private static int[] indices(int count) {
            int[] all = new int[count];
            for (int i = 0; i < count; i++) all[i] = i;
            return all;
        }

        /** Midpoint split on the widest axis; the split may be empty, in which case the node becomes a leaf. */
        private Node build(int[] members, int from, int to) {
            Node node = new Node();
            node.minX = Double.MAX_VALUE;
            node.maxX = -Double.MAX_VALUE;
            node.minY = Double.MAX_VALUE;
            node.maxY = -Double.MAX_VALUE;
            for (int k = from; k < to; k++) {
                Body body = bodies.get(members[k]);
                node.minX = Math.min(node.minX, body.x);
                node.maxX = Math.max(node.maxX, body.x);
                node.minY = Math.min(node.minY, body.y);
                node.maxY = Math.max(node.maxY, body.y);
                node.mass += body.mass;
                node.centerX += body.mass * body.x;
                node.centerY += body.mass * body.y;
            }
            if (node.mass > 0.0) {
                node.centerX /= node.mass;
                node.centerY /= node.mass;
            }
            int count = to - from;
            if (count <= LEAF_CAPACITY) {
                node.bodies = Arrays.copyOfRange(members, from, to);
                node.leafCount = count;
                return node;
            }
            boolean splitX = node.maxX - node.minX >= node.maxY - node.minY;
            double middle = splitX ? 0.5 * (node.minX + node.maxX) : 0.5 * (node.minY + node.maxY);
            int left = from;
            int right = to - 1;
            while (left <= right) {
                Body body = bodies.get(members[left]);
                double coordinate = splitX ? body.x : body.y;
                if (coordinate <= middle) {
                    left++;
                } else {
                    int swap = members[left];
                    members[left] = members[right];
                    members[right] = swap;
                    right--;
                }
            }
            if (left == from || left == to) {
                node.bodies = Arrays.copyOfRange(members, from, to);
                node.leafCount = count;
                return node;
            }
            node.first = build(members, from, left);
            node.second = build(members, left, to);
            return node;
        }

        /** Pairwise force on body `target` from body `source`, with softening. */
        private double[] pairForce(int target, int source) {
            Body a = bodies.get(target);
            Body b = bodies.get(source);
            double dx = b.x - a.x;
            double dy = b.y - a.y;
            double squared = dx * dx + dy * dy + softening * softening;
            double inverse = 1.0 / Math.sqrt(squared);
            double factor = gravity * a.mass * b.mass * inverse * inverse * inverse;
            return new double[]{factor * dx, factor * dy};
        }

        /** Direct O(N^2) sum: the oracle. */
        double[][] directForces() {
            int n = bodies.size();
            double[][] forces = new double[n][2];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (i == j) continue;
                    double[] f = pairForce(i, j);
                    forces[i][0] += f[0];
                    forces[i][1] += f[1];
                }
            }
            return forces;
        }

        /** Barnes-Hut walk for one body. */
        double[] approximateForce(int target) {
            Body body = bodies.get(target);
            double[] force = new double[2];
            walk(root, target, body, force);
            return force;
        }

        private void walk(Node node, int target, Body body, double[] force) {
            if (node == null) return;
            interactions++;
            if (node.first == null) {
                for (int index : node.bodies) {
                    if (index == target) continue;
                    double[] f = pairForce(target, index);
                    force[0] += f[0];
                    force[1] += f[1];
                }
                return;
            }
            double dx = node.centerX - body.x;
            double dy = node.centerY - body.y;
            double distance = Math.sqrt(dx * dx + dy * dy);
            if (node.size() / distance < theta) {
                double squared = dx * dx + dy * dy + softening * softening;
                double inverse = 1.0 / Math.sqrt(squared);
                double factor = gravity * body.mass * node.mass * inverse * inverse * inverse;
                force[0] += factor * dx;
                force[1] += factor * dy;
                return;
            }
            walk(node.first, target, body, force);
            walk(node.second, target, body, force);
        }

        double[][] approximateForces() {
            interactions = 0;
            double[][] forces = new double[bodies.size()][2];
            for (int i = 0; i < bodies.size(); i++) forces[i] = approximateForce(i);
            return forces;
        }

        /**
         * Global relative error: the norm of the force difference over the norm of the true forces. A
         * per-body ratio would explode for the bodies near the centre of a symmetric cluster, whose true net
         * force is almost zero, and would say nothing about the approximation as a whole.
         */
        static double relativeError(double[][] approximate, double[][] exact) {
            double difference = 0.0;
            double reference = 0.0;
            for (int i = 0; i < approximate.length; i++) {
                difference += Math.pow(approximate[i][0] - exact[i][0], 2)
                        + Math.pow(approximate[i][1] - exact[i][1], 2);
                reference += exact[i][0] * exact[i][0] + exact[i][1] * exact[i][1];
            }
            return reference == 0.0 ? Math.sqrt(difference) : Math.sqrt(difference / reference);
        }

        /** Total mass aggregated at the root, an invariant of the build. */
        double rootMass() { return root.mass; }

        /** Walks the tree and collects a description of the first violation, or null. */
        String verifyTree() {
            double[] seen = new double[bodies.size()];
            String problem = verify(root, seen);
            if (problem != null) return problem;
            for (int i = 0; i < seen.length; i++) if (seen[i] != 1.0) return "body " + i + " is not in a leaf";
            double total = 0.0;
            for (Body body : bodies) total += body.mass;
            if (Math.abs(root.mass - total) > 1e-9) return "root mass " + root.mass + " != " + total;
            return null;
        }

        private String verify(Node node, double[] seen) {
            if (node.first == null) {
                if (node.bodies.length > LEAF_CAPACITY) return "leaf holds " + node.bodies.length + " bodies";
                for (int index : node.bodies) {
                    seen[index] += 1.0;
                    Body body = bodies.get(index);
                    if (body.x < node.minX || body.x > node.maxX || body.y < node.minY || body.y > node.maxY) {
                        return "body " + index + " outside its leaf";
                    }
                }
                return null;
            }
            String left = verify(node.first, seen);
            if (left != null) return left;
            return verify(node.second, seen);
        }
    }

    static final int LEAF_CAPACITY = 4;
    static double theta = 0.5;

    private static long rngState = 0x243F6A8885A308D3L;

    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /** A Plummer sphere, the standard test distribution for tree codes. */
    static List<Body> cluster(int count, double extent) {
        List<Body> bodies = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            double radius = extent * Math.pow(nextDouble(), 1.0 / 3.0);
            double angle = 2.0 * Math.PI * nextDouble();
            bodies.add(new Body(radius * Math.cos(angle), radius * Math.sin(angle), 1.0));
        }
        return bodies;
    }

    public static void main(String[] args) {
        final double softening = 0.05;
        final double gravity = 1.0;

        // --- degenerate configurations -----------------------------------------
        NBody empty = new NBody(new ArrayList<>(), softening, gravity);
        check(empty.verifyTree() == null && empty.rootMass() == 0.0, "an empty tree is valid and massless");
        NBody single = new NBody(Collections.singletonList(new Body(0.3, -0.2, 2.0)), softening, gravity);
        check(single.verifyTree() == null, "a single body forms a valid leaf");
        check(single.approximateForce(0)[0] == 0.0 && single.approximateForce(0)[1] == 0.0,
                "a body does not feel its own gravity");

        NBody pair = new NBody(Arrays.asList(new Body(-0.4, 0.0, 1.0), new Body(0.4, 0.0, 1.0)),
                softening, gravity);
        double[][] pairDirect = pair.directForces();
        for (double trial : new double[]{0.0, 0.2, 0.5, 0.9}) {
            theta = trial;
            double[][] pairApproximate = pair.approximateForces();
            check(NBody.relativeError(pairApproximate, pairDirect) < 1e-12,
                    "a two-body system must be exact at theta=" + trial);
        }
        check(pairDirect[0][0] > 0.0 && pairDirect[1][0] < 0.0, "the pair pulls towards each other");

        // --- tree invariants on a real cluster --------------------------------
        List<Body> cluster = cluster(2000, 1.0);
        NBody tree = new NBody(cluster, softening, gravity);
        check(tree.verifyTree() == null, "cluster tree: " + tree.verifyTree());
        double totalMass = 0.0;
        for (Body body : cluster) totalMass += body.mass;
        check(Math.abs(tree.rootMass() - totalMass) < 1e-9, "the root must aggregate every body's mass");
        check(tree.root.maxX - tree.root.minX <= 2.0 * 1.000001, "the root bounds must contain the cluster");

        // --- theta = 0 is the direct sum --------------------------------------
        List<Body> small = cluster(40, 1.0);
        NBody exactTree = new NBody(small, softening, gravity);
        double[][] direct = exactTree.directForces();
        theta = 0.0;
        double[][] exact = exactTree.approximateForces();
        check(NBody.relativeError(exact, direct) < 1e-12, "theta = 0 must reproduce the direct sum exactly");
        long exactInteractions = exactTree.interactions;
        check(exactInteractions >= 40, "theta = 0 must touch every body: " + exactInteractions);

        // --- error grows with theta, cost falls --------------------------------
        double previousError = 0.0;
        for (double trial : new double[]{0.2, 0.5, 1.0}) {
            theta = trial;
            double[][] approximate = tree.approximateForces();
            double error = NBody.relativeError(approximate, tree.directForces());
            check(error > previousError, "the error must grow with theta at theta=" + trial + ": " + error
                    + " after " + previousError);
            previousError = error;
        }
        theta = 0.5;
        double[][] forces = tree.approximateForces();
        double error = NBody.relativeError(forces, tree.directForces());
        check(error < 0.05, "theta = 0.5 must stay accurate, relative error " + error);
        long approximateInteractions = tree.interactions;
        long pairs = (long) cluster.size() * (cluster.size() - 1);
        check(approximateInteractions * 10 < pairs, "the walk must cost a fraction of the pairs: "
                + approximateInteractions + " of " + pairs);

        // --- a mirror-symmetric cluster has no net force -----------------------
        List<Body> mirrored = new ArrayList<>();
        for (int i = 0; i < 60; i++) {
            double x = nextDouble() * 0.8 + 0.1;
            double y = nextDouble() * 0.8 + 0.1;
            double mass = 0.5 + nextDouble();
            mirrored.add(new Body(x, y, mass));
            mirrored.add(new Body(-x, -y, mass));
        }
        NBody symmetric = new NBody(mirrored, softening, gravity);
        double totalForceX = 0.0;
        double totalForceY = 0.0;
        for (double[] f : symmetric.approximateForces()) {
            totalForceX += f[0];
            totalForceY += f[1];
        }
        double scale = 0.0;
        for (double[] f : symmetric.approximateForces()) scale = Math.max(scale, Math.hypot(f[0], f[1]));
        check(Math.hypot(totalForceX, totalForceY) < 1e-9 * scale * mirrored.size(),
                "a centrosymmetric cluster must have no net force, got " + Math.hypot(totalForceX, totalForceY));

        // --- determinism --------------------------------------------------------
        NBody rebuilt = new NBody(cluster, softening, gravity);
        double[][] again = rebuilt.approximateForces();
        check(NBody.relativeError(again, forces) == 0.0, "the same inputs must give the same forces");
        check(rebuilt.verifyTree() == null && again.length == forces.length, "the rebuild must be valid");

        System.out.println("Barnes-Hut over " + cluster.size() + " bodies, leaf capacity " + LEAF_CAPACITY
                + ", softening " + softening);
        System.out.println("theta = 0.5: relative error " + error + ", interactions " + approximateInteractions
                + " against " + pairs + " pairs (" + (100.0 * approximateInteractions / pairs) + "%)");
        System.out.println("theta = 0: error " + NBody.relativeError(exact, direct) + " over "
                + exactInteractions + " interactions; error at theta 0.2/0.5/1.0 grows monotonically");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
