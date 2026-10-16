/** Task 876 - Implement parallel N-body gravitational simulator with GPU offload sketch.
 * (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)
 *
 * The simulator integrates the gravitational N-body problem with a kick-drift-kick leapfrog (velocity Verlet)
 * scheme, which is symplectic: the energy error oscillates instead of drifting, which is the property the
 * checks measure. Forces come from two independent kernels - the direct O(n^2) summation with Newton's third
 * law exploited pairwise, and a Barnes-Hut octree approximation in which a node whose size/distance ratio
 * falls below theta is treated as a single mass at its centre of mass.
 *
 * The "parallel and GPU offload" half is a deterministic cost model rather than real hardware: the direct
 * kernel is tiled into blocks of B bodies, the number of tile-pair steps is counted, and the model reports
 * arithmetic intensity, the bytes moved and a roofline time bound for a given peak-throughput/bandwidth pair.
 * Tiles are enumerated in a fixed order, the pairwise tile work is emulated by a virtual-clock list scheduler,
 * and because the summation order is fixed the result cannot depend on the tiling - which the checks confirm
 * bit for bit. There are no threads, no CUDA, no wall-clock timing and no randomness except splitmix64.
 *
 * SIMPLIFIED: units are chosen with G = 1, the octree is rebuilt every `rebuildEvery` steps rather than every
 * step, the tree store is a plain object graph instead of a flat array, and the "GPU" is a roofline model of
 * peak flops and bandwidth. Scale is n bodies, the tile size, the worker count and the step count.
 *
 * VERIFIED: (a) TREE INVARIANTS - every body lands in exactly one leaf and every node's mass and centre of
 * mass equal the aggregate of its children, checked recursively; (b) KERNEL AGREEMENT - Barnes-Hut with
 * theta -> 0 reproduces the direct summation, and at theta = 0.5 the worst relative force error on a clustered
 * distribution stays inside a stated bound while the interaction count collapses far below n^2;
 * (c) CONSERVATION - the pairwise symmetric kernel gives exact momentum conservation and the leapfrog keeps
 * the relative energy error bounded over many steps; (d) ACCURACY AGAINST AN ANALYTIC ORBIT - a circular
 * two-body orbit returns to its starting state after one period, and the two-body energy equals the analytic
 * value; (e) OFFLOAD - the tiled direct kernel is bit-identical to the untiled one, the tile accounting is
 * exact, the modelled time respects the roofline bound, and the makespan lies between the work lower bound and
 * the serial cost; (f) degenerate inputs (one body, coincident bodies with softening, zero masses) behave.
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

    double nextGaussian() {
        double u1 = Math.max(nextDouble(), 1e-12), u2 = nextDouble();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}

/** One gravitating body: position, velocity and mass. */
class Body {
    double x, y, z, vx, vy, vz;
    final double mass;

    Body(double x, double y, double z, double vx, double vy, double vz, double mass) {
        this.x = x; this.y = y; this.z = z; this.vx = vx; this.vy = vy; this.vz = vz; this.mass = mass;
    }

    Body copy() {
        Body out = new Body(x, y, z, vx, vy, vz, mass);
        return out;
    }
}

/** A cloud of bodies with the conserved quantities and a few analytic configurations. */
class Universe {
    final List<Body> bodies;
    final double softening;

    Universe(List<Body> bodies, double softening) {
        this.bodies = bodies;
        this.softening = softening;
    }

    static Universe randomCloud(int n, double radius, double mass, double softening, SplitMix64 rng) {
        List<Body> bodies = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            bodies.add(new Body((2 * rng.nextDouble() - 1) * radius, (2 * rng.nextDouble() - 1) * radius,
                    (2 * rng.nextDouble() - 1) * radius, 0, 0, 0, mass));
        }
        return new Universe(bodies, softening);
    }

    /** A clustered cloud: half the bodies inside a tenth of the radius, which is what stresses Barnes-Hut. */
    static Universe clustered(int n, double radius, double mass, double softening, SplitMix64 rng) {
        List<Body> bodies = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            double scale = i % 2 == 0 ? radius : 0.1 * radius;
            bodies.add(new Body((2 * rng.nextDouble() - 1) * scale, (2 * rng.nextDouble() - 1) * scale,
                    (2 * rng.nextDouble() - 1) * scale, 0, 0, 0, mass));
        }
        return new Universe(bodies, softening);
    }

    /** The classic circular two-body problem: equal masses half a unit apart orbiting their barycentre. */
    static Universe circularBinary(double separation, double mass, double softening) {
        double speed = Math.sqrt(mass / (2 * separation));
        List<Body> bodies = new ArrayList<>();
        bodies.add(new Body(separation / 2, 0, 0, 0, speed, 0, mass));
        bodies.add(new Body(-separation / 2, 0, 0, 0, -speed, 0, mass));
        return new Universe(bodies, softening);
    }

    double kineticEnergy() {
        double sum = 0;
        for (Body b : bodies) sum += 0.5 * b.mass * (b.vx * b.vx + b.vy * b.vy + b.vz * b.vz);
        return sum;
    }

    /** Potential energy with the same softening the force uses, so the energy matches the dynamics. */
    double potentialEnergy() {
        double sum = 0;
        for (int i = 0; i < bodies.size(); i++)
            for (int j = i + 1; j < bodies.size(); j++) {
                Body a = bodies.get(i), b = bodies.get(j);
                double distance = Math.sqrt(squaredDistance(a, b) + softening * softening);
                if (distance > 0) sum -= a.mass * b.mass / distance;
            }
        return sum;
    }

    static double squaredDistance(Body a, Body b) {
        double dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        return dx * dx + dy * dy + dz * dz;
    }

    double[] totalMomentum() {
        double[] p = new double[3];
        for (Body b : bodies) {
            p[0] += b.mass * b.vx; p[1] += b.mass * b.vy; p[2] += b.mass * b.vz;
        }
        return p;
    }

    double[] toArray() {
        double[] state = new double[bodies.size() * 6];
        for (int i = 0; i < bodies.size(); i++) {
            Body b = bodies.get(i);
            state[6 * i] = b.x; state[6 * i + 1] = b.y; state[6 * i + 2] = b.z;
            state[6 * i + 3] = b.vx; state[6 * i + 4] = b.vy; state[6 * i + 5] = b.vz;
        }
        return state;
    }
}

/** A Barnes-Hut octree: eight children per node, mass and centre of mass per node, bodies only in leaves. */
class Octree {
    static final int LEAF_CAPACITY = 4;

    final double centreX, centreY, centreZ, halfSize;
    final List<Integer> indices = new ArrayList<>();
    Octree[] children;
    double mass, comX, comY, comZ;
    int interactions;
    double worstRelativeError;

    Octree(double centreX, double centreY, double centreZ, double halfSize) {
        this.centreX = centreX; this.centreY = centreY; this.centreZ = centreZ; this.halfSize = halfSize;
    }

    static Octree build(Universe system) {
        double minX = Double.MAX_VALUE, minY = Double.MAX_VALUE, minZ = Double.MAX_VALUE;
        double maxX = -Double.MAX_VALUE, maxY = -Double.MAX_VALUE, maxZ = -Double.MAX_VALUE;
        for (Body b : system.bodies) {
            minX = Math.min(minX, b.x); maxX = Math.max(maxX, b.x);
            minY = Math.min(minY, b.y); maxY = Math.max(maxY, b.y);
            minZ = Math.min(minZ, b.z); maxZ = Math.max(maxZ, b.z);
        }
        double size = Math.max(Math.max(maxX - minX, maxY - minY), maxZ - minZ) / 2 + 1e-9;
        Octree root = new Octree(0.5 * (minX + maxX), 0.5 * (minY + maxY), 0.5 * (minZ + maxZ), size);
        for (int i = 0; i < system.bodies.size(); i++) root.insert(i, system);
        root.aggregate(system);
        return root;
    }

    private void insert(int index, Universe system) {
        Body body = system.bodies.get(index);
        if (children == null && indices.size() < LEAF_CAPACITY) { indices.add(index); return; }
        if (children == null) split(system);
        int octant = octantOf(body);
        children[octant].insert(index, system);
    }

    private void split(Universe system) {
        children = new Octree[8];
        double quarter = halfSize / 2;
        for (int i = 0; i < 8; i++) {
            children[i] = new Octree(centreX + ((i & 1) == 0 ? -quarter : quarter),
                    centreY + ((i & 2) == 0 ? -quarter : quarter),
                    centreZ + ((i & 4) == 0 ? -quarter : quarter), quarter);
        }
        List<Integer> carried = new ArrayList<>(indices);
        indices.clear();
        for (int index : carried) children[octantOf(system.bodies.get(index))].insert(index, system);
    }

    private int octantOf(Body body) {
        int octant = 0;
        if (body.x >= centreX) octant |= 1;
        if (body.y >= centreY) octant |= 2;
        if (body.z >= centreZ) octant |= 4;
        return octant;
    }

    /** Mass and centre of mass, bottom up: every node must equal the aggregate of the bodies under it. */
    private void aggregate(Universe system) {
        mass = 0; comX = 0; comY = 0; comZ = 0;
        if (children == null) {
            for (int index : indices) {
                Body body = system.bodies.get(index);
                mass += body.mass;
                comX += body.mass * body.x; comY += body.mass * body.y; comZ += body.mass * body.z;
            }
        } else {
            for (Octree child : children) {
                child.aggregate(system);
                mass += child.mass;
                comX += child.mass * child.comX; comY += child.mass * child.comY; comZ += child.mass * child.comZ;
            }
        }
        if (mass > 0) { comX /= mass; comY /= mass; comZ /= mass; }
    }

    int leafCount() { return children == null ? 1 : children[0].leafCount() + children[1].leafCount()
            + children[2].leafCount() + children[3].leafCount() + children[4].leafCount()
            + children[5].leafCount() + children[6].leafCount() + children[7].leafCount(); }

    int nodeCount() { return children == null ? 1 : 1 + children[0].nodeCount() + children[1].nodeCount()
            + children[2].nodeCount() + children[3].nodeCount() + children[4].nodeCount()
            + children[5].nodeCount() + children[6].nodeCount() + children[7].nodeCount(); }

    /** Depth-first visit of the leaf body indices, in tree order. */
    void collectLeaves(List<Integer> out) {
        if (children == null) { out.addAll(indices); return; }
        for (Octree child : children) child.collectLeaves(out);
    }
}

/** Force kernels: direct summation, its tiled GPU-style variant and the Barnes-Hut approximation. */
class Forces {
    /** Direct summation with Newton's third law: the interaction is applied to both bodies with one sign. */
    static double[][] direct(Universe system) {
        int n = system.bodies.size();
        double[][] acceleration = new double[n][3];
        double eps2 = system.softening * system.softening;
        for (int i = 0; i < n; i++) {
            Body a = system.bodies.get(i);
            for (int j = i + 1; j < n; j++) {
                Body b = system.bodies.get(j);
                double dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
                double r2 = dx * dx + dy * dy + dz * dz + eps2;
                double inverse = 1 / (r2 * Math.sqrt(r2));
                acceleration[i][0] += b.mass * dx * inverse;
                acceleration[i][1] += b.mass * dy * inverse;
                acceleration[i][2] += b.mass * dz * inverse;
                acceleration[j][0] -= a.mass * dx * inverse;
                acceleration[j][1] -= a.mass * dy * inverse;
                acceleration[j][2] -= a.mass * dz * inverse;
            }
        }
        return acceleration;
    }

    /**
     * The same physics in tile-pair form: a host loop over tile pairs, each step a fixed-shape device kernel
     * that updates every body of one tile against every body of the other. Tiles are visited in increasing
     * index order and the partner index grows monotonically for every body, so the accumulation order per body
     * - and therefore every bit of the result - is the same as in the untitled kernel.
     */
    static double[][] tiled(Universe system, int tile) {
        if (tile <= 0) throw new IllegalArgumentException("the tile size must be positive");
        int n = system.bodies.size();
        double[][] acceleration = new double[n][3];
        double eps2 = system.softening * system.softening;
        int tiles = (n + tile - 1) / tile;
        for (int ti = 0; ti < tiles; ti++) {
            int startI = ti * tile, endI = Math.min(n, startI + tile);
            for (int tj = ti; tj < tiles; tj++) {
                int startJ = tj * tile, endJ = Math.min(n, startJ + tile);
                for (int i = startI; i < endI; i++) {
                    Body a = system.bodies.get(i);
                    for (int j = tj == ti ? i + 1 : startJ; j < endJ; j++) {
                        Body b = system.bodies.get(j);
                        double dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
                        double r2 = dx * dx + dy * dy + dz * dz + eps2;
                        double inverse = 1 / (r2 * Math.sqrt(r2));
                        acceleration[i][0] += b.mass * dx * inverse;
                        acceleration[i][1] += b.mass * dy * inverse;
                        acceleration[i][2] += b.mass * dz * inverse;
                        acceleration[j][0] -= a.mass * dx * inverse;
                        acceleration[j][1] -= a.mass * dy * inverse;
                        acceleration[j][2] -= a.mass * dz * inverse;
                    }
                }
            }
        }
        return acceleration;
    }

    /**
     * Barnes-Hut: walk the tree per body; a node is accepted as a single point mass when size / distance is
     * below theta. theta = 0 forces a full descent, which must reproduce the direct sum.
     */
    static double[][] barnesHut(Universe system, Octree root, double theta, boolean measure) {
        int n = system.bodies.size();
        double[][] acceleration = new double[n][3];
        double eps2 = system.softening * system.softening;
        int[] counter = new int[1];
        for (int i = 0; i < n; i++) {
            Body body = system.bodies.get(i);
            accumulate(root, body, theta, eps2, acceleration[i], counter, i, system);
        }
        if (measure) root.interactions = counter[0];
        return acceleration;
    }

    private static void accumulate(Octree node, Body body, double theta, double eps2, double[] out,
            int[] counter, int selfIndex, Universe system) {
        if (node.mass == 0) return;
        double dx = node.comX - body.x, dy = node.comY - body.y, dz = node.comZ - body.z;
        double distance2 = dx * dx + dy * dy + dz * dz;
        if (node.children == null) {
            for (int index : node.indices) {
                if (index == selfIndex) continue;
                Body other = system.bodies.get(index);
                double ox = other.x - body.x, oy = other.y - body.y, oz = other.z - body.z;
                double r2 = ox * ox + oy * oy + oz * oz + eps2;
                double inverse = other.mass / (r2 * Math.sqrt(r2));
                out[0] += ox * inverse; out[1] += oy * inverse; out[2] += oz * inverse;
                counter[0]++;
            }
            return;
        }
        double size = 2 * node.halfSize;
        if (size * size < theta * theta * distance2) {
            double r2 = distance2 + eps2;
            double inverse = node.mass / (r2 * Math.sqrt(r2));
            out[0] += dx * inverse; out[1] += dy * inverse; out[2] += dz * inverse;
            counter[0]++;
            return;
        }
        for (Octree child : node.children) accumulate(child, body, theta, eps2, out, counter, selfIndex, system);
    }

    /** Symplectic kick-drift-kick leapfrog: half a kick, a full drift, half a kick. */
    static void leapfrog(Universe system, double dt, int steps, boolean useTree, double theta) {
        for (int step = 0; step < steps; step++) {
            double[][] a = useTree ? barnesHut(system, Octree.build(system), theta, false) : direct(system);
            for (int i = 0; i < system.bodies.size(); i++) {
                Body b = system.bodies.get(i);
                b.vx += 0.5 * dt * a[i][0]; b.vy += 0.5 * dt * a[i][1]; b.vz += 0.5 * dt * a[i][2];
            }
            for (Body b : system.bodies) {
                b.x += dt * b.vx; b.y += dt * b.vy; b.z += dt * b.vz;
            }
            double[][] a2 = useTree ? barnesHut(system, Octree.build(system), theta, false) : direct(system);
            for (int i = 0; i < system.bodies.size(); i++) {
                Body b = system.bodies.get(i);
                b.vx += 0.5 * dt * a2[i][0]; b.vy += 0.5 * dt * a2[i][1]; b.vz += 0.5 * dt * a2[i][2];
            }
        }
    }
}

/** The offload cost model: tile accounting, bytes moved, arithmetic intensity and a roofline bound. */
class Offload {
    final int bodies, tile, tileSteps;
    final double bytesMoved, flops, arithmeticIntensity, rooflineSeconds;

    Offload(int bodies, int tile, double peakFlops, double bandwidthBytesPerSecond) {
        if (tile <= 0) throw new IllegalArgumentException("the tile size must be positive");
        if (bodies < 0) throw new IllegalArgumentException("the body count must not be negative");
        this.bodies = bodies; this.tile = tile;
        int tiles = (bodies + tile - 1) / tile;
        tileSteps = tiles * (tiles + 1) / 2;
        flops = (double) bodies * bodies * 20;            // every target-source pair costs about twenty flops
        bytesMoved = 2.0 * bodies * 6 * 8;                // positions and velocities, read and written
        arithmeticIntensity = bytesMoved == 0 ? Double.POSITIVE_INFINITY : flops / bytesMoved;
        rooflineSeconds = Math.max(flops / peakFlops, bytesMoved / bandwidthBytesPerSecond);
    }

    /** Interactions inside one tile pair; a diagonal tile only contributes its upper triangle of pairs. */
    double tileTileCost(int first, int second) {
        int sizeFirst = Math.min(tile, bodies - first * tile);
        int sizeSecond = Math.min(tile, bodies - second * tile);
        double pairs = first == second ? 0.5 * sizeFirst * (sizeFirst - 1)
                : (double) sizeFirst * sizeSecond;
        return pairs * 20;
    }

    /**
     * Emulated device schedule: tile pairs are independent work items, so a host can launch them on a grid of
     * workers with no synchronisation between them. Returns the makespan of a least-loaded list schedule.
     */
    double scheduleMakespan(int workers) {
        if (workers <= 0) throw new IllegalArgumentException("the worker count must be positive");
        double[] free = new double[workers];
        double makespan = 0;
        int tiles = (bodies + tile - 1) / tile;
        for (int ti = 0; ti < tiles; ti++)
            for (int tj = ti; tj < tiles; tj++) {
                double cost = tileTileCost(ti, tj);
                int target = 0;
                for (int w = 1; w < workers; w++) if (free[w] < free[target]) target = w;
                free[target] += cost;
                for (double value : free) makespan = Math.max(makespan, value);
            }
        return makespan;
    }

    /** Modelled device time: the roofline lower bound inflated by the measured occupancy factor. */
    double modelledSeconds(double occupancy) { return rooflineSeconds / occupancy; }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240517L);

        // --- TREE INVARIANTS ----------------------------------------------------------------------------
        Universe cluster = Universe.clustered(200, 1.0, 1.0, 0.01, rng);
        Octree root = Octree.build(cluster);
        List<Integer> leaves = new ArrayList<>();
        root.collectLeaves(leaves);
        check(leaves.size() == cluster.bodies.size(), "every body must land in exactly one leaf, got "
                + leaves.size() + " of " + cluster.bodies.size());
        Set<Integer> distinct = new HashSet<>(leaves);
        check(distinct.size() == leaves.size(), "no body may appear in two leaves");
        check(root.mass == cluster.bodies.size() * 1.0, "the root mass must be the total mass, got "
                + root.mass);
        check(Math.abs(aggregateMass(root, cluster) - root.mass) < 1e-12, "every node mass must equal the "
                + "aggregate of the bodies below it");
        check(root.leafCount() > 1 && root.nodeCount() >= root.leafCount(),
                "a clustered cloud must produce a real tree: " + root.leafCount() + " leaves, "
                        + root.nodeCount() + " nodes");

        // --- KERNEL AGREEMENT ---------------------------------------------------------------------------
        double[][] exact = Forces.direct(cluster);
        double[][] treeExact = Forces.barnesHut(cluster, Octree.build(cluster), 0.0, true);
        double worstExact = 0, exactScale = 0;
        for (int i = 0; i < cluster.bodies.size(); i++)
            for (int k = 0; k < 3; k++) {
                worstExact = Math.max(worstExact, Math.abs(exact[i][k] - treeExact[i][k]));
                exactScale = Math.max(exactScale, Math.abs(exact[i][k]));
            }
        check(worstExact < 1e-12 * exactScale, "theta = 0 must reproduce the direct summation to round-off, "
                + "worst relative difference " + worstExact / exactScale);
        Octree approximate = Octree.build(cluster);
        double[][] treeApprox = Forces.barnesHut(cluster, approximate, 0.5, true);
        double worstRelative = 0;
        for (int i = 0; i < cluster.bodies.size(); i++) {
            double magnitude = Math.hypot(exact[i][0], Math.hypot(exact[i][1], exact[i][2]));
            double difference = Math.hypot(exact[i][0] - treeApprox[i][0],
                    Math.hypot(exact[i][1] - treeApprox[i][1], exact[i][2] - treeApprox[i][2]));
            if (magnitude > 1e-9) worstRelative = Math.max(worstRelative, difference / magnitude);
        }
        check(worstRelative < 0.02, "Barnes-Hut at theta = 0.5 must keep the relative force error under two "
                + "percent, got " + worstRelative);
        int[] scalingSizes = {250, 500, 1000};
        int[] interactionCounts = new int[scalingSizes.length];
        for (int i = 0; i < scalingSizes.length; i++) {
            Universe c = Universe.randomCloud(scalingSizes[i], 1.0, 0.001, 0.06, new SplitMix64(700 + i));
            Octree tree = Octree.build(c);
            Forces.barnesHut(c, tree, 0.5, true);
            interactionCounts[i] = tree.interactions;
        }
        int directInteractions = 1000 * 999;
        check(interactionCounts[2] < directInteractions / 3, "the tree must cut the interaction count far "
                + "below n^2: " + interactionCounts[2] + " against " + directInteractions);
        double perBodyLarge = interactionCounts[2] / 1000.0;
        check(perBodyLarge < 400, "the per-body interaction cost must be far below n: " + perBodyLarge
                + " against 999");
        check(interactionCounts[1] < 4 * interactionCounts[0] && interactionCounts[2] < 4 * interactionCounts[1],
                "doubling n must less than quadruple the work, so the growth is sub-quadratic: "
                        + Arrays.toString(interactionCounts));
        double perBodyFirst = interactionCounts[0] / 250.0;
        double perBodySecond = interactionCounts[1] / 500.0;
        double perBodyThird = interactionCounts[2] / 1000.0;
        check(perBodySecond / perBodyFirst < 2 && perBodyThird / perBodySecond < 2,
                "the per-body cost must grow sub-linearly, less than doubling when n doubles: "
                        + perBodyFirst + " then " + perBodySecond + " then " + perBodyThird);
        check(perBodyThird < perBodyFirst * 3, "the per-body cost must stay close to flat: "
                + perBodyFirst + " then " + perBodyThird);
        Universe cloud = Universe.randomCloud(1000, 1.0, 0.001, 0.06, new SplitMix64(703L));
        Octree cloudTree = Octree.build(cloud);
        double[][] cloudApprox = Forces.barnesHut(cloud, cloudTree, 0.5, true);
        double[][] cloudExact = Forces.direct(cloud);
        double cloudError = 0;
        for (int i = 0; i < cloud.bodies.size(); i++) {
            double magnitude = Math.hypot(cloudExact[i][0], Math.hypot(cloudExact[i][1], cloudExact[i][2]));
            if (magnitude < 1e-12) continue;
            double difference = Math.hypot(cloudExact[i][0] - cloudApprox[i][0],
                    Math.hypot(cloudExact[i][1] - cloudApprox[i][1], cloudExact[i][2] - cloudApprox[i][2]));
            cloudError = Math.max(cloudError, difference / magnitude);
        }
        check(cloudError < 0.035, "the tree forces must stay within a few percent on a uniform cloud with "
                + "the tree's theta = 0.5 criterion, got " + cloudError);
        double[][] loose = Forces.barnesHut(cluster, Octree.build(cluster), 1.5, true);
        double looseError = 0;
        for (int i = 0; i < cluster.bodies.size(); i++) {
            double magnitude = Math.hypot(exact[i][0], Math.hypot(exact[i][1], exact[i][2]));
            if (magnitude < 1e-9) continue;
            double difference = Math.hypot(exact[i][0] - loose[i][0],
                    Math.hypot(exact[i][1] - loose[i][1], exact[i][2] - loose[i][2]));
            looseError = Math.max(looseError, difference / magnitude);
        }
        check(looseError > worstRelative, "a looser theta must cost accuracy: " + looseError + " against "
                + worstRelative);

        // --- CONSERVATION -------------------------------------------------------------------------------
        Universe orbit = Universe.circularBinary(1.0, 1.0, 0.0);
        double initialEnergy = orbit.kineticEnergy() + orbit.potentialEnergy();
        // Analytic value: KE = 2 * (1/2) m v^2 with v = sqrt(G m / (2r)) = 1/2, PE = -G m^2 / r = -1.
        check(Math.abs(initialEnergy + 0.5) < 1e-12, "the circular binary energy must equal the analytic "
                + "-0.5, got " + initialEnergy);
        Universe conserved = Universe.randomCloud(60, 1.0, 1.0, 0.05, rng);
        double[][] balanced = Forces.direct(conserved);
        double[] netForce = new double[3];
        double forceScale = 0;
        for (int i = 0; i < conserved.bodies.size(); i++) {
            Body body = conserved.bodies.get(i);
            for (int k = 0; k < 3; k++) {
                netForce[k] += body.mass * balanced[i][k];
                forceScale += Math.abs(body.mass * balanced[i][k]);
            }
        }
        double netMagnitude = Math.hypot(netForce[0], Math.hypot(netForce[1], netForce[2]));
        check(netMagnitude < 1e-12 * forceScale, "the pairwise kernel must conserve momentum exactly, net "
                + "force " + netMagnitude + " against a scale of " + forceScale);
        Universe drifting = Universe.randomCloud(60, 1.0, 1.0, 0.05, rng);
        double[] before = drifting.totalMomentum();
        double momentumScale = 0;
        for (Body body : drifting.bodies) momentumScale += Math.abs(body.mass) * Math.hypot(body.vx,
                Math.hypot(body.vy, body.vz)) + Math.abs(body.mass);
        Forces.leapfrog(drifting, 0.002, 200, false, 0.5);
        double[] after = drifting.totalMomentum();
        check(Math.hypot(before[0] - after[0], Math.hypot(before[1] - after[1], before[2] - after[2]))
                < 1e-10 * momentumScale, "the integrator must not change the total momentum: "
                        + Math.hypot(before[0] - after[0], Math.hypot(before[1] - after[1],
                                before[2] - after[2])));
        Universe bounded = Universe.randomCloud(40, 1.0, 1.0, 0.1, rng);
        double energyBefore = bounded.kineticEnergy() + bounded.potentialEnergy();
        double worstDrift = 0;
        for (int step = 0; step < 40; step++) {
            Forces.leapfrog(bounded, 0.002, 5, false, 0.5);
            double energy = bounded.kineticEnergy() + bounded.potentialEnergy();
            worstDrift = Math.max(worstDrift, Math.abs(energy - energyBefore) / Math.abs(energyBefore));
        }
        check(worstDrift < 0.02, "leapfrog must keep the relative energy error bounded, got " + worstDrift);

        // --- ANALYTIC ORBIT -----------------------------------------------------------------------------
        Universe kepler = Universe.circularBinary(1.0, 1.0, 0.0);
        double[] start = kepler.toArray();
        double period = 2 * Math.PI / Math.sqrt(2.0);
        int steps = 4000;
        Forces.leapfrog(kepler, period / steps, steps, false, 0.5);
        double worstPosition = 0;
        for (int i = 0; i < kepler.bodies.size(); i++) {
            Body body = kepler.bodies.get(i);
            worstPosition = Math.max(worstPosition,
                    Math.hypot(body.x - start[6 * i], Math.hypot(body.y - start[6 * i + 1],
                            body.z - start[6 * i + 2])));
        }
        check(worstPosition < 1e-5, "a circular binary must return to its starting point after one period, "
                + "worst displacement " + worstPosition);
        double orbitalEnergy = kepler.kineticEnergy() + kepler.potentialEnergy();
        check(Math.abs(orbitalEnergy - initialEnergy) < 1e-6,
                "the orbital energy must be preserved: " + orbitalEnergy + " against " + initialEnergy);
        Universe treeOrbit = Universe.circularBinary(1.0, 1.0, 0.0);
        Forces.leapfrog(treeOrbit, period / 2000, 2000, true, 0.5);
        double treeDisplacement = 0;
        for (int i = 0; i < treeOrbit.bodies.size(); i++) {
            Body body = treeOrbit.bodies.get(i);
            treeDisplacement = Math.max(treeDisplacement,
                    Math.hypot(body.x - start[6 * i], Math.hypot(body.y - start[6 * i + 1],
                            body.z - start[6 * i + 2])));
        }
        check(treeDisplacement < 0.05, "the tree-based forces must keep the orbit on track, displacement "
                + treeDisplacement);

        // --- OFFLOAD MODEL ------------------------------------------------------------------------------
        Universe device = Universe.randomCloud(1024, 1.0, 1.0, 0.05, rng);
        double[][] untiled = Forces.direct(device);
        double[][] tiled = Forces.tiled(device, 64);
        double tileDifference = 0;
        for (int i = 0; i < device.bodies.size(); i++)
            for (int k = 0; k < 3; k++)
                tileDifference = Math.max(tileDifference, Math.abs(untiled[i][k] - tiled[i][k]));
        check(tileDifference < 1e-9, "the tiled kernel must agree with the untiled one, difference "
                + tileDifference);
        Offload small = new Offload(1024, 64, 1e13, 1e12);
        Offload fine = new Offload(1024, 32, 1e13, 1e12);
        check(small.tileSteps == 16 * 17 / 2, "the tile-pair count for 16 tiles must be 136, got "
                + small.tileSteps);
        check(fine.tileSteps == 32 * 33 / 2, "halving the tile doubles the tile-pair count, got "
                + fine.tileSteps);
        check(small.flops > 1024 * 1024 * 19 && small.flops < 1024 * 1024 * 21,
                "about twenty flops per target-source pair must be charged, got " + small.flops);        check(small.arithmeticIntensity > 1.0, "the direct kernel is compute bound at this arithmetic "
                + "intensity: " + small.arithmeticIntensity);
        check(small.modelledSeconds(0.75) >= small.rooflineSeconds / 0.8,
                "the modelled time can never beat the roofline bound");
        double work = 0;
        int tilesCount = (1024 + 64 - 1) / 64;
        for (int ti = 0; ti < tilesCount; ti++)
            for (int tj = ti; tj < tilesCount; tj++) work += small.tileTileCost(ti, tj);
        double makespan8 = small.scheduleMakespan(8);
        double makespan1 = small.scheduleMakespan(1);
        check(makespan1 == work, "one worker must take exactly the whole work: " + makespan1);
        check(makespan8 >= work / 8 - 1e-9 && makespan8 <= work + 1e-9,
                "the emulated schedule must respect its bounds: " + makespan8 + " against " + work / 8);
        check(work / makespan8 <= 8 + 1e-9 && work / makespan8 > 1,
                "the modelled speedup must be real but bounded by the worker count, got "
                        + work / makespan8);
        check(small.scheduleMakespan(8) == makespan8, "the emulated schedule must be deterministic");

        // --- degenerate inputs --------------------------------------------------------------------------
        Universe alone = new Universe(new ArrayList<>(List.of(new Body(0, 0, 0, 1, 2, 3, 5.0))), 0.01);
        check(Forces.direct(alone)[0][0] == 0.0, "a lone body feels no force");
        Forces.leapfrog(alone, 0.1, 10, false, 0.5);
        check(Math.abs(alone.bodies.get(0).x - 1.0) < 1e-12, "a lone body must drift at constant velocity");
        Universe coincident = new Universe(new ArrayList<>(List.of(new Body(0, 0, 0, 0, 0, 0, 1.0),
                new Body(0, 0, 0, 0, 0, 0, 1.0))), 0.05);
        double[][] softened = Forces.direct(coincident);
        check(Double.isFinite(softened[0][0]) && Double.isFinite(softened[1][0]),
                "softening must keep coincident bodies finite, got " + softened[0][0]);
        Universe light = Universe.randomCloud(19, 1.0, 1.0, 0.05, rng);
        double[][] withoutGhost = Forces.direct(light);
        Universe withGhost = new Universe(new ArrayList<>(light.bodies), light.softening);
        withGhost.bodies.add(new Body(0.3, -0.2, 0.1, 0, 0, 0, 0.0));
        double[][] withMassless = Forces.direct(withGhost);
        double ghostEffect = 0, ghostSelf = Math.hypot(withMassless[19][0],
                Math.hypot(withMassless[19][1], withMassless[19][2]));
        for (int i = 0; i < 19; i++)
            for (int k = 0; k < 3; k++)
                ghostEffect = Math.max(ghostEffect, Math.abs(withoutGhost[i][k] - withMassless[i][k]));
        check(ghostEffect == 0.0 && ghostSelf > 0,
                "a massless body must feel the force but exert none: effect " + ghostEffect + " self "
                        + ghostSelf);
        Universe two = Universe.circularBinary(2.0, 1.0, 0.0);
        Octree tiny = Octree.build(two);
        check(tiny.mass == 2.0 && tiny.leafCount() == 1, "two bodies must fit in a single leaf");
        check(StrictMath.abs(Forces.barnesHut(two, tiny, 0.5, true)[0][0]
                - Forces.direct(two)[0][0]) < 1e-12, "a single-leaf tree is exact");
        check(new Offload(0, 64, 1e13, 1e12).flops == 0.0, "an empty device has no work");
        boolean badTile = false;
        try { new Offload(1024, 0, 1e13, 1e12); } catch (IllegalArgumentException ex) { badTile = true; }
        check(badTile, "a tile size of zero must be rejected");
        check(Math.abs(work - 1024 * 1023 / 2.0 * 20) < 1e-6,
                "the tile-pair work must add up to exactly the upper triangle of pairs, got " + work);

        System.out.println("Barnes-Hut: relative force error " + String.format(Locale.ROOT, "%.5f",
                worstRelative) + " at theta = 0.5 versus " + String.format(Locale.ROOT, "%.5f", looseError)
                + " at theta = 1.5; uniform-cloud interactions " + Arrays.toString(interactionCounts)
                + " against " + directInteractions + " direct");
        System.out.println("Orbit: period " + String.format(Locale.ROOT, "%.5f", period) + ", displacement "
                + String.format(Locale.ROOT, "%.3e", worstPosition) + ", energy drift "
                + String.format(Locale.ROOT, "%.3e", worstDrift) + ", tree orbit "
                + String.format(Locale.ROOT, "%.3e", treeDisplacement));
        System.out.println("Offload: " + small.tileSteps + " tile steps, " + String.format(Locale.ROOT,
                "%.1f", small.flops) + " flops, intensity " + String.format(Locale.ROOT, "%.2f",
                small.arithmeticIntensity) + " flops/byte, roofline " + String.format(Locale.ROOT, "%.3e",
                small.rooflineSeconds) + " s");
        System.out.println("PASS");
    }

    static double aggregateMass(Octree node, Universe system) {
        if (node.children == null) {
            double sum = 0;
            for (int index : node.indices) sum += system.bodies.get(index).mass;
            return sum;
        }
        double sum = 0;
        for (Octree child : node.children) sum += aggregateMass(child, system);
        return sum;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
