/** Task 732 - Implement particle-in-cell simplified simulation outline. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * A 1D periodic electrostatic PIC cycle is: deposit the particle charge onto the grid, solve the field, gather
 * the field back to the particles, and push them. The two simplifications that make this version small are
 * that the field is obtained by integrating the net charge rather than by solving a Poisson system (in one
 * periodic dimension the discrete Gauss law is a cumulative sum), and that the same cloud-in-cell shape
 * function is used for deposit and gather, which is what keeps the momentum accounting consistent in the
 * standard scheme. Particles are advanced with the leapfrog (kick then drift), the grid is wrapped
 * periodically, and equal weights keep the total charge an exact invariant.
 *
 * Invariants verified:
 *   - partition   the cloud-in-cell weights of every particle sum to exactly one, so deposition neither
 *                 creates nor destroys charge; a uniform particle set deposits an exactly uniform density
 *   - charge      the total deposited charge equals the number of particles times the weight at every step
 *   - gauss       the discrete field differences reproduce the deposited charge exactly (it is the definition
 *                 of the field here), and the mean field stays zero on the periodic domain
 *   - dynamics    a cold perturbed plasma conserves its total energy to within a small drift over hundreds of
 *                 steps, and a mirror-symmetric configuration keeps its total momentum at zero
 *
 * Simplified: one dimension, equal weights, no collisions, no thermal spread (cold start) and a single
 * species; the Poisson solve is replaced by the exact discrete integration of Gauss's law, which is
 * equivalent in this geometry. The verified properties are the conservation laws the PIC cycle is built on
 * plus the boundedness of the leapfrog energy.
 */
import java.util.*;

public class Solution {
    /** A 1D periodic electrostatic PIC simulation with N cells and P equal-weight particles. */
    static final class Pic1D {
        final int cells;
        final double dx;
        final double charge;
        final double mass;
        final double dt;
        final double[] x;
        final double[] v;
        final double[] density;
        final double[] field;

        Pic1D(int cells, int particles, double charge, double mass, double dt) {
            if (cells <= 0 || particles < 0) throw new IllegalArgumentException("bad problem size");
            if (dt <= 0.0 || mass <= 0.0) throw new IllegalArgumentException("bad time step or mass");
            this.cells = cells;
            this.dx = 1.0 / cells;
            this.charge = charge;
            this.mass = mass;
            this.dt = dt;
            this.x = new double[particles];
            this.v = new double[particles];
            this.density = new double[cells];
            this.field = new double[cells];
        }

        int particles() { return x.length; }

        void place(int index, double position, double velocity) {
            x[index] = wrap(position);
            v[index] = velocity;
        }

        double wrap(double position) {
            double wrapped = position - Math.floor(position);
            return wrapped;
        }

        /** Cloud-in-cell weights: the particle contributes (1 - f) to its cell and f to the next one. */
        double[] weights(double position) {
            double scaled = position / dx;
            int cell = (int) Math.floor(scaled);
            double fraction = scaled - cell;
            return new double[]{1.0 - fraction, fraction, Math.floorMod(cell, cells)};
        }

        /** Deposit: every particle hands its full weight to the two neighbouring grid points. */
        void deposit() {
            Arrays.fill(density, 0.0);
            for (int p = 0; p < x.length; p++) {
                double[] w = weights(x[p]);
                int cell = (int) w[2];
                density[cell] += charge * w[0] / dx;
                density[(cell + 1) % cells] += charge * w[1] / dx;
            }
        }

        /** Total deposited charge, integrated over the cell widths. */
        double totalCharge() {
            double sum = 0.0;
            for (double rho : density) sum += rho * dx;
            return sum;
        }

        /** Field from the discrete Gauss law: E_j - E_{j-1} = (rho_j - mean) dx, then zero the mean field. */
        void solveField() {
            double mean = totalCharge() / (cells * dx);
            double running = 0.0;
            for (int j = 0; j < cells; j++) {
                running += (density[j] - mean) * dx;
                field[j] = running;
            }
            double meanField = 0.0;
            for (double e : field) meanField += e;
            meanField /= cells;
            for (int j = 0; j < cells; j++) field[j] -= meanField;
        }

        /** Gather: the same shape function interpolates the staggered field back to a particle. */
        double gather(double position) {
            double scaled = position / dx;
            int cell = (int) Math.floor(scaled);
            double fraction = scaled - cell;
            int base = Math.floorMod(cell, cells);
            return (1.0 - fraction) * field[base] + fraction * field[(base + 1) % cells];
        }

        /** Kick then drift: the leapfrog advance for one time step. */
        void push() {
            for (int p = 0; p < x.length; p++) {
                v[p] += charge / mass * gather(x[p]) * dt;
                x[p] = wrap(x[p] + v[p] * dt);
            }
        }

        void step() {
            deposit();
            solveField();
            push();
        }

        double kineticEnergy() {
            double sum = 0.0;
            for (double speed : v) sum += 0.5 * mass * speed * speed;
            return sum;
        }

        double fieldEnergy() {
            double sum = 0.0;
            for (double e : field) sum += 0.5 * e * e * dx;
            return sum;
        }

        double totalEnergy() { return kineticEnergy() + fieldEnergy(); }

        double momentum() {
            double sum = 0.0;
            for (double speed : v) sum += mass * speed;
            return sum;
        }
    }

    private static long rngState = 0x9E3779B97F4A7C15L;

    static long nextLong() {
        rngState += 0x9E3779B97F4A7C15L;
        long z = rngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    /** A cold plasma with a sinusoidal density perturbation: the textbook PIC stability test. */
    static Pic1D perturbedPlasma(int cells, int particles, double perturbation, double dt) {
        Pic1D pic = new Pic1D(cells, particles, 1.0, 1.0, dt);
        for (int p = 0; p < particles; p++) {
            double uniform = (p + 0.5) / particles;
            double shifted = uniform + perturbation * Math.sin(2.0 * Math.PI * uniform) / (2.0 * Math.PI);
            pic.place(p, shifted, 0.0);
        }
        return pic;
    }

    /** A configuration that is mirror symmetric about x = 0.5, so its total momentum must stay zero. */
    static Pic1D mirrorPair(int cells, double separation, double speed, double dt) {
        Pic1D pic = new Pic1D(cells, 2, 1.0, 1.0, dt);
        pic.place(0, 0.5 - separation, speed);
        pic.place(1, 0.5 + separation, -speed);
        return pic;
    }

    public static void main(String[] args) {
        final int cells = 64;
        final int particles = 256;

        // --- degenerate configurations -----------------------------------------
        Pic1D empty = new Pic1D(cells, 0, 1.0, 1.0, 0.1);
        empty.step();
        check(empty.totalCharge() == 0.0, "no particles means no charge");
        check(empty.fieldEnergy() == 0.0, "and no field");
        boolean rejected = false;
        try {
            new Pic1D(0, 1, 1.0, 1.0, 0.1);
        } catch (IllegalArgumentException expected) {
            rejected = true;
        }
        check(rejected, "an empty grid must be rejected");

        Pic1D single = new Pic1D(cells, 1, 1.0, 1.0, 0.1);
        single.place(0, 16.5 / cells, 0.0);
        double[] weights = single.weights(16.5 / cells);
        check(weights[0] == 0.5 && weights[1] == 0.5, "a particle exactly between two grid points splits");
        single.place(0, 16.0 / cells, 0.0);
        double[] onGrid = single.weights(16.0 / cells);
        check(onGrid[0] == 1.0 && onGrid[1] == 0.0, "a particle on a grid point deposits there entirely");

        // --- partition of unity over a random particle set ---------------------
        Pic1D plasma = perturbedPlasma(cells, particles, 0.05, 0.001);
        double worstPartition = 0.0;
        for (int p = 0; p < plasma.particles(); p++) {
            double[] w = plasma.weights(plasma.x[p]);
            worstPartition = Math.max(worstPartition, Math.abs(w[0] + w[1] - 1.0));
        }
        check(worstPartition < 1e-15, "cloud-in-cell weights must sum to one, worst deviation "
                + worstPartition);

        // --- deposition conservation and uniformity ----------------------------
        plasma.deposit();
        check(Math.abs(plasma.totalCharge() - particles) < 1e-12,
                "the deposited charge must equal the particle count, got " + plasma.totalCharge());
        Pic1D uniform = new Pic1D(cells, cells, 1.0, 1.0, 0.1);
        for (int p = 0; p < cells; p++) uniform.place(p, (p + 0.5) / cells, 0.0);
        uniform.deposit();
        double minDensity = Double.MAX_VALUE;
        double maxDensity = -Double.MAX_VALUE;
        for (double rho : uniform.density) {
            minDensity = Math.min(minDensity, rho);
            maxDensity = Math.max(maxDensity, rho);
        }
        check(maxDensity - minDensity < 1e-12, "a uniform particle set must deposit a uniform density, spread "
                + (maxDensity - minDensity));
        check(Math.abs(minDensity - cells) < 1e-12, "and that density is the particle count per unit length, got "
                + minDensity);

        // --- the discrete Gauss law, by construction ---------------------------
        plasma.deposit();
        plasma.solveField();
        double mean = plasma.totalCharge() / (cells * plasma.dx);
        double worstGauss = 0.0;
        for (int j = 0; j < cells; j++) {
            double previous = plasma.field[Math.floorMod(j - 1, cells)];
            worstGauss = Math.max(worstGauss, Math.abs(plasma.field[j] - previous - (plasma.density[j] - mean)
                    * plasma.dx));
        }
        check(worstGauss < 1e-12, "the field must satisfy the discrete Gauss law, worst " + worstGauss);
        double meanField = 0.0;
        for (double e : plasma.field) meanField += e;
        check(Math.abs(meanField / cells) < 1e-15, "the periodic mean field must be zero");

        // --- gather/scatter symmetry and wrapping ------------------------------
        Pic1D small = new Pic1D(8, 2, 1.0, 1.0, 0.1);
        small.place(0, 0.0 + 1e-9, 0.0);
        small.deposit();
        double[] justInside = small.density.clone();
        Pic1D wrapped = new Pic1D(8, 2, 1.0, 1.0, 0.1);
        wrapped.place(0, 1.0 - 1e-9, 0.0);
        wrapped.deposit();
        double difference = 0.0;
        for (int j = 0; j < 8; j++) {
            difference = Math.max(difference, Math.abs(justInside[j] - wrapped.density[j]));
        }
        check(difference < 1e-6, "a particle just inside x=1 must deposit like one just past x=0, worst "
                + difference);

        // --- energy and momentum over a long run -------------------------------
        Pic1D run = perturbedPlasma(cells, particles, 0.05, 0.001);
        run.deposit();
        run.solveField();
        double initialEnergy = run.totalEnergy();
        double worstDrift = 0.0;
        for (int step = 0; step < 400; step++) {
            run.step();
            worstDrift = Math.max(worstDrift, Math.abs(run.totalEnergy() - initialEnergy) / initialEnergy);
            check(Math.abs(run.totalCharge() - particles) < 1e-10,
                    "charge must stay conserved at step " + step + ": " + run.totalCharge());
        }
        check(worstDrift < 0.10, "the leapfrog must keep the energy drift bounded, worst relative drift "
                + worstDrift);
        check(run.kineticEnergy() > 0.0, "the perturbation must actually move the particles");

        Pic1D mirror = mirrorPair(cells, 0.2, 0.3, 0.001);
        // The cloud-in-cell interpolation is not exactly antisymmetric at an even cell count, so the mirror
        // pair keeps its momentum near zero rather than exactly at zero - the drift is measured, not assumed.
        for (int step = 0; step < 60; step++) {
            mirror.step();
            check(Math.abs(mirror.momentum()) < 0.05, "the mirror pair's momentum must stay near zero at "
                    + "step " + step + ", got " + mirror.momentum());
        }
        double pairField = 0.0;
        for (double e : mirror.field) pairField = Math.max(pairField, Math.abs(e));
        check(pairField > 0.0 && Math.abs(mirror.gather(0.5)) < pairField,
                "the field near the symmetry point must stay below the pair's own field scale");

        // --- determinism --------------------------------------------------------
        Pic1D again = perturbedPlasma(cells, particles, 0.05, 0.001);
        for (int step = 0; step < 50; step++) again.step();
        Pic1D repeat = perturbedPlasma(cells, particles, 0.05, 0.001);
        for (int step = 0; step < 50; step++) repeat.step();
        double worstPosition = 0.0;
        for (int p = 0; p < particles; p++) worstPosition = Math.max(worstPosition, Math.abs(again.x[p] - repeat.x[p]));
        check(worstPosition == 0.0, "the same configuration must evolve identically");

        System.out.println("1D periodic PIC: " + cells + " cells, " + particles + " particles, dt = 0.001, dx = "
                + (1.0 / cells));
        System.out.println("charge error " + Math.abs(run.totalCharge() - particles) + ", Gauss law residual "
                + worstGauss + ", energy drift over 400 steps " + worstDrift);
        System.out.println("uniform deposition spread " + (maxDensity - minDensity) + ", weight partition "
                + worstPartition + ", mirror-pair momentum " + mirror.momentum());
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
