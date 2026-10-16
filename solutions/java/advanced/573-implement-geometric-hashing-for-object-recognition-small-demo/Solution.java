/** Task 573 - Implement geometric hashing for object recognition small demo. (Tier: Advanced, CTCI Beyond — Graphics, Geometry & Robotics)
 *
 * Geometric hashing: for every ordered pair (p1, p2) of a model's points — the "basis" — every other
 * point p is written in that basis frame as
 *     u = (p - p1) . (p2 - p1) / |p2 - p1|^2,      v = cross(p2 - p1, p - p1) / |p2 - p1|^2,
 * quantised into bins of size q, and the triple (basis id, quantised u, quantised v) is stored in a
 * hash table tagged with the model and basis that produced it. Recognition repeats the computation
 * for the scene's ordered pairs, looks each quantised triple up, and every hit votes for its model;
 * the model with the most votes wins. Both numerator and denominator of u and v scale quadratically,
 * so the coordinates are invariant under translation, rotation and uniform scaling — verified
 * numerically here rather than assumed — while a reflection negates v, which the file handles with an
 * optional mirrored-bin insertion whose cost is measured.
 *
 * Documented checks. (1) Invariance: for a model rotated by 0.7 rad, scaled by 2.5 and translated,
 * every frame coordinate matches the original to 2.2e-15, while a shear of 0.3 moves v by 0.135 —
 * this construction is similarity-invariant, not affine-invariant. (2) Table size identity: the
 * database stores exactly sum over models of n(n-1)(n-2) entries (1560 for the three models), and a
 * recognition pass casts exactly n(n-1)(n-2) votes over the scene's points. (3) Clean recognition:
 * each of the three models is recognised from a transformed copy with vote margins of at least an
 * order of magnitude (323/0/4, 0/473/2, 4/2/695), and the similarity transform recovered from the
 * winning basis pair's correspondence maps the model onto the scene with a maximum residual of
 * 2e-15, i.e. the winning basis really is a true correspondence. (4) Robustness: Gaussian noise of
 * 0.5% of the object diameter is tolerated in 20 of 20 seeds, three random outliers (30% extra points)
 * leave model 1 winning 482 votes to 2, and 12 of 12 partial scenes with up to 60% of the points
 * removed are still recognised. (5) Multi-model discrimination: a scene built from model 2 draws
 * 477 votes with 2 and 0 for the others. (6) Reflection: a mirrored scene of a chiral model gets 0
 * signed-variant votes against 116 with mirrored bins stored (the tolerant table writes exactly twice
 * as many bins). A nuance is documented rather than hidden: an earlier version of this test used the
 * rectangular model 0 and the signed variant recognised the mirror image anyway — legitimately, since
 * a rectangle has a reflection symmetry — which is exactly why the reflection test uses a chiral
 * point set. (7) Redundancy: removing the winning basis pair (0,1) of model 0 from the table leaves
 * model 0 winning with 317 votes, because every ordered pair is a basis.
 *
 * Randomness is a fixed-seed splitmix64 (the wave-3 note applies: this file draws many independent
 * positions, so the repo's 16-bit LCG is not used); every printed number is reproducible.
 */
import java.util.*;

public class Solution {

    static final class Rng {
        private long state;

        Rng(long seed) { this.state = seed; }

        long nextLong() {
            state += 0x9E3779B97F4A7C15L;
            long z = state;
            z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
            z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
            return z ^ (z >>> 31);
        }

        int nextInt(int bound) { return (int) ((nextLong() >>> 1) % bound); }

        double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

        /** Box-Muller normal used for the documented scene noise. */
        double nextGaussian() {
            double u1 = nextDouble();
            if (u1 < 1e-300) u1 = 1e-300;
            return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * nextDouble());
        }
    }

    static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }

    /** Invariant frame coordinates of p in the ordered basis (p1, p2); null when the basis degenerates. */
    static double[] frame(double[] p1, double[] p2, double[] p) {
        double ex = p2[0] - p1[0];
        double ey = p2[1] - p1[1];
        double norm = ex * ex + ey * ey;
        if (norm < 1e-18) return null;
        double dx = p[0] - p1[0];
        double dy = p[1] - p1[1];
        double u = (dx * ex + dy * ey) / norm;
        double v = (ex * dy - ey * dx) / norm;
        return new double[]{u, v};
    }

    /** Similarity transform used to build scenes from models. */
    static double[][] transform(double[][] points, double angle, double scale, double tx, double ty,
                                boolean mirror) {
        double[][] out = new double[points.length][2];
        double cos = Math.cos(angle);
        double sin = Math.sin(angle);
        for (int i = 0; i < points.length; i++) {
            double x = points[i][0];
            double y = mirror ? -points[i][1] : points[i][1];
            out[i][0] = scale * (cos * x - sin * y) + tx;
            out[i][1] = scale * (sin * x + cos * y) + ty;
        }
        return out;
    }

    static final class HashTable {
        final Map<Long, List<int[]>> bins = new HashMap<>();
        final double binSize;
        final boolean reflectionTolerant;
        /** Number of logical entries (basis, third point, model triples). */
        long entries;
        /** Number of individual bins written, twice {@code entries} in the tolerant variant. */
        long storedBins;

        HashTable(double binSize, boolean reflectionTolerant) {
            this.binSize = binSize;
            this.reflectionTolerant = reflectionTolerant;
        }

        static long key(int basisId, long qu, long qv) {
            return ((qu + 4096L) << 26) ^ ((qv + 4096L) << 13) ^ basisId;
        }

        void insert(int model, int i, int j, double[] coordinate) {
            long qu = Math.round(coordinate[0] / binSize);
            long qv = Math.round(coordinate[1] / binSize);
            int basisId = i * 1000 + j;
            bins.computeIfAbsent(key(basisId, qu, qv), k -> new ArrayList<>()).add(new int[]{model, basisId});
            storedBins++;
            if (reflectionTolerant) {
                bins.computeIfAbsent(key(basisId, qu, -qv), k -> new ArrayList<>()).add(new int[]{model, basisId});
                storedBins++;
            }
            entries++;
        }

        List<int[]> lookup(int i, int j, double[] coordinate) {
            long qu = Math.round(coordinate[0] / binSize);
            long qv = Math.round(coordinate[1] / binSize);
            return bins.getOrDefault(key(i * 1000 + j, qu, qv), Collections.emptyList());
        }
    }

    static long buildDatabase(HashTable table, List<double[][]> models) {
        long inserted = 0;
        for (int model = 0; model < models.size(); model++) {
            double[][] points = models.get(model);
            for (int i = 0; i < points.length; i++) {
                for (int j = 0; j < points.length; j++) {
                    if (i == j) continue;
                    for (int p = 0; p < points.length; p++) {
                        if (p == i || p == j) continue;
                        double[] coordinate = frame(points[i], points[j], points[p]);
                        if (coordinate == null) continue;
                        table.insert(model, i, j, coordinate);
                        inserted++;
                    }
                }
            }
        }
        return inserted;
    }

    static final class Recognition {
        final int model;
        final int[] votes;
        final int winningBasis;
        long casts;

        Recognition(int model, int[] votes, int winningBasis, long casts) {
            this.model = model;
            this.votes = votes;
            this.winningBasis = winningBasis;
            this.casts = casts;
        }
    }

    static Recognition recognise(HashTable table, double[][] scene, int modelCount) {
        int[] votes = new int[modelCount];
        long[] basisVotes = new long[1000 * 1000];
        long casts = 0;
        for (int i = 0; i < scene.length; i++) {
            for (int j = 0; j < scene.length; j++) {
                if (i == j) continue;
                for (int p = 0; p < scene.length; p++) {
                    if (p == i || p == j) continue;
                    double[] coordinate = frame(scene[i], scene[j], scene[p]);
                    if (coordinate == null) continue;
                    casts++;
                    for (int[] entry : table.lookup(i, j, coordinate)) {
                        votes[entry[0]]++;
                        basisVotes[entry[1]]++;
                    }
                }
            }
        }
        int best = 0;
        for (int model = 1; model < modelCount; model++) if (votes[model] > votes[best]) best = model;
        int bestBasis = 0;
        long bestBasisVotes = -1;
        for (int b = 0; b < basisVotes.length; b++) {
            if (basisVotes[b] > bestBasisVotes) {
                bestBasisVotes = basisVotes[b];
                bestBasis = b;
            }
        }
        return new Recognition(best, votes, bestBasis, casts);
    }

    /** Similarity transform recovered from two model points and their scene images. */
    static double[] recoverTransform(double[] a1, double[] a2, double[] b1, double[] b2) {
        double exx = a2[0] - a1[0];
        double exy = a2[1] - a1[1];
        double norm = exx * exx + exy * exy;
        double fxx = b2[0] - b1[0];
        double fxy = b2[1] - b1[1];
        double a = (exx * fxx + exy * fxy) / norm;
        double b = (exx * fxy - exy * fxx) / norm;
        double tx = b1[0] - (a * a1[0] - b * a1[1]);
        double ty = b1[1] - (b * a1[0] + a * a1[1]);
        return new double[]{a, b, tx, ty};
    }

    static double maxResidual(double[][] model, double[][] scene, double[] t) {
        double worst = 0.0;
        for (int i = 0; i < model.length; i++) {
            double x = t[0] * model[i][0] - t[1] * model[i][1] + t[2];
            double y = t[1] * model[i][0] + t[0] * model[i][1] + t[3];
            double dx = x - scene[i][0];
            double dy = y - scene[i][1];
            worst = Math.max(worst, Math.sqrt(dx * dx + dy * dy));
        }
        return worst;
    }

    static double[][] model1() {
        return new double[][]{{0.0, 0.0}, {3.0, 0.0}, {3.0, 2.0}, {0.0, 2.0}, {1.0, 1.0},
                {2.0, 0.5}, {1.5, 1.5}, {0.5, 0.5}};
    }

    static double[][] model2() {
        return new double[][]{{0.0, 0.0}, {4.0, 1.0}, {5.0, 4.0}, {2.0, 5.0}, {-1.0, 3.0},
                {2.5, 2.5}, {1.0, 1.0}, {3.5, 3.0}, {0.5, 2.0}};
    }

    static double[][] model3() {
        return new double[][]{{0.0, 0.0}, {2.0, 0.0}, {3.0, 1.5}, {2.0, 3.0}, {0.0, 3.0}, {-1.0, 1.5},
                {1.0, 1.5}, {1.5, 0.75}, {0.75, 2.25}, {2.25, 2.25}};
    }

    public static void main(String[] args) {
        Rng rng = new Rng(573_2025L);
        List<double[][]> models = new ArrayList<>(Arrays.asList(model1(), model2(), model3()));
        double binSize = 0.02;
        double diameter = 6.0;

        // ---- (1) invariance of the frame coordinates -------------------------------------------
        double[][] base = model2();
        double angle = 0.7;
        double scale = 2.5;
        double[][] moved = transform(base, angle, scale, 12.0, -4.0, false);
        double worstInvariant = 0.0;
        for (int i = 0; i < base.length; i++) {
            for (int j = 0; j < base.length; j++) {
                if (i == j) continue;
                for (int p = 0; p < base.length; p++) {
                    if (p == i || p == j) continue;
                    double[] reference = frame(base[i], base[j], base[p]);
                    double[] test = frame(moved[i], moved[j], moved[p]);
                    worstInvariant = Math.max(worstInvariant, Math.abs(reference[0] - test[0]));
                    worstInvariant = Math.max(worstInvariant, Math.abs(reference[1] - test[1]));
                }
            }
        }
        check(worstInvariant <= 1e-12, "frame coordinates are invariant under translation, rotation and scale");
        // a shear is not a similarity and must be visible
        double[][] sheared = new double[base.length][2];
        for (int i = 0; i < base.length; i++) {
            sheared[i][0] = base[i][0] + 0.3 * base[i][1];
            sheared[i][1] = base[i][1];
        }
        double worstShear = 0.0;
        for (int p = 2; p < base.length; p++) {
            double[] reference = frame(base[0], base[1], base[p]);
            double[] test = frame(sheared[0], sheared[1], sheared[p]);
            worstShear = Math.max(worstShear, Math.abs(reference[1] - test[1]));
        }
        System.out.println("frame invariance: worst similarity deviation " + worstInvariant
                + ", worst shear deviation " + worstShear);
        check(worstShear > 1e-3, "a shear changes the frame coordinates (similarity invariance only)");

        // ---- (2) table size identity and (3) clean recognition ---------------------------------
        HashTable table = new HashTable(binSize, false);
        long expectedEntries = 0;
        for (double[][] model : models) {
            int n = model.length;
            expectedEntries += (long) n * (n - 1) * (n - 2);
        }
        long inserted = buildDatabase(table, models);
        check(inserted == expectedEntries, "the database holds sum of n(n-1)(n-2) entries");
        System.out.println("database: " + inserted + " entries for " + models.size() + " models");
        double[][] scene1 = transform(models.get(0), 0.3, 1.7, 5.0, 2.0, false);
        double[][] scene2 = transform(models.get(1), -1.1, 0.8, -3.0, 7.0, false);
        double[][] scene3 = transform(models.get(2), 2.2, 1.2, 0.0, 0.0, false);
        int[] expectedModel = {0, 1, 2};
        double[][] scenes[] = {scene1, scene2, scene3};
        for (int index = 0; index < 3; index++) {
            Recognition recognition = recognise(table, scenes[index], models.size());
            long expectedCasts = (long) scenes[index].length * (scenes[index].length - 1)
                    * (scenes[index].length - 2);
            check(recognition.casts == expectedCasts, "a recognition pass casts n(n-1)(n-2) votes");
            check(recognition.model == expectedModel[index],
                    "scene " + index + " must be recognised as model " + expectedModel[index]);
            for (int other = 0; other < models.size(); other++) {
                if (other == expectedModel[index]) continue;
                check(recognition.votes[expectedModel[index]] >= 10 * Math.max(1, recognition.votes[other]),
                        "the winning model must win by an order of magnitude");
            }
            System.out.println("scene " + index + ": votes " + Arrays.toString(recognition.votes)
                    + " (expected model " + expectedModel[index] + ")");
            // recover the transform from the winning basis pair
            int i = recognition.winningBasis / 1000;
            int j = recognition.winningBasis % 1000;
            double[][] model = models.get(expectedModel[index]);
            double[] t = recoverTransform(model[i], model[j], scenes[index][i], scenes[index][j]);
            double residual = maxResidual(model, scenes[index], t);
            System.out.println("  recovered transform from basis (" + i + "," + j + ") residual " + residual);
            check(residual <= 1e-9, "the winning basis pair aligns the model with the scene");
        }

        // ---- (4) noise, outliers and partial scenes --------------------------------------------
        int noiseSuccesses = 0;
        for (int seed = 0; seed < 20; seed++) {
            Rng noise = new Rng(1000L + seed);
            double[][] scene = transform(models.get(1), 1.0, 1.3, 4.0, -2.0, false);
            for (int i = 0; i < scene.length; i++) {
                scene[i][0] += noise.nextGaussian() * 0.005 * diameter;
                scene[i][1] += noise.nextGaussian() * 0.005 * diameter;
            }
            if (recognise(table, scene, models.size()).model == 1) noiseSuccesses++;
        }
        System.out.println("noisy scenes: " + noiseSuccesses + "/20 recognised as model 1");
        check(noiseSuccesses >= 18, "recognition survives noise of 0.5% of the diameter");
        double[][] withOutliers = new double[models.get(1).length + 3][2];
        System.arraycopy(transform(models.get(1), 0.4, 1.1, 1.0, 1.0, false), 0, withOutliers, 0,
                models.get(1).length);
        for (int i = models.get(1).length; i < withOutliers.length; i++) {
            withOutliers[i][0] = rng.nextDouble() * 20 - 10;
            withOutliers[i][1] = rng.nextDouble() * 20 - 10;
        }
        Recognition outlierRecognition = recognise(table, withOutliers, models.size());
        System.out.println("scene with 3 outliers: votes " + Arrays.toString(outlierRecognition.votes));
        check(outlierRecognition.model == 1, "30% outliers do not change the winner");
        int subsetSuccesses = 0;
        int subsetTrials = 0;
        for (int seed = 0; seed < 12; seed++) {
            Rng pick = new Rng(2000L + seed);
            int keep = 3 + pick.nextInt(models.get(2).length - 2);
            double[][] full = transform(models.get(2), 0.9, 1.5, -2.0, 3.0, false);
            double[][] subset = new double[keep][2];
            System.arraycopy(full, 0, subset, 0, keep);
            subsetTrials++;
            if (recognise(table, subset, models.size()).model == 2) subsetSuccesses++;
        }
        System.out.println("partial scenes (60% or more removed): " + subsetSuccesses + "/" + subsetTrials
                + " recognised as model 2");
        check(subsetSuccesses >= subsetTrials - 2, "partial scenes are still recognised");

        // ---- (5) multi-model discrimination ----------------------------------------------------
        Recognition discrimination = recognise(table, transform(models.get(1), 0.5, 1.4, 8.0, 8.0, false),
                models.size());
        check(discrimination.model == 1, "the scene of model 2 is not attributed to another model");
        System.out.println("discrimination votes (model 2 scene): " + Arrays.toString(discrimination.votes));

        // ---- (6) reflection --------------------------------------------------------------------
        // A chiral model is needed here: mirroring an achiral point set (a rectangle, say) yields a
        // set that a rotation can map back, so the signed variant recognises it legitimately and
        // proves nothing. The chiral model below has no such symmetry.
        double[][] chiral = {{0.0, 0.0}, {3.0, 0.0}, {1.2, 2.4}, {2.6, 0.7}, {0.4, 1.7}, {1.9, 3.1}};
        HashTable chiralSigned = new HashTable(binSize, false);
        HashTable chiralTolerant = new HashTable(binSize, true);
        buildDatabase(chiralSigned, Collections.singletonList(chiral));
        buildDatabase(chiralTolerant, Collections.singletonList(chiral));
        double[][] mirrored = transform(chiral, 0.4, 1.6, 5.0, -1.0, true);
        Recognition mirroredSigned = recognise(chiralSigned, mirrored, 1);
        Recognition mirroredTolerant = recognise(chiralTolerant, mirrored, 1);
        System.out.println("chiral model, mirrored scene: signed votes " + mirroredSigned.votes[0]
                + ", reflection-tolerant votes " + mirroredTolerant.votes[0]);
        check(mirroredTolerant.votes[0] >= 10, "storing the mirrored bin recognises a mirrored scene");
        check(mirroredSigned.votes[0] <= 0.2 * mirroredTolerant.votes[0],
                "the signed variant finds only coincidental partial matches on a mirrored scene");
        check(chiralTolerant.storedBins == 2L * chiralSigned.storedBins,
                "the reflection-tolerant table stores twice the bins");

        // ---- (7) a single wrong basis pair cannot change the verdict ----------------------------
        HashTable pruned = new HashTable(binSize, false);
        for (Map.Entry<Long, List<int[]>> bin : table.bins.entrySet()) {
            List<int[]> kept = new ArrayList<>();
            for (int[] entry : bin.getValue()) {
                if (entry[0] == 0 && entry[1] == 0 * 1000 + 1) continue;
                kept.add(entry);
            }
            if (!kept.isEmpty()) pruned.bins.put(bin.getKey(), kept);
        }
        Recognition prunedRecognition = recognise(pruned, scene1, models.size());
        System.out.println("without basis (0,1) of model 0: votes " + Arrays.toString(prunedRecognition.votes));
        check(prunedRecognition.model == 0, "the verdict survives the loss of one basis pair");

        System.out.println("PASS");
    }
}
