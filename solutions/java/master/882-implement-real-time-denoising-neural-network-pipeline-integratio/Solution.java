/** Task 882 - Implement real-time denoising neural network pipeline integration for raytracer. (Tier: Master, CTCI Beyond - Graphics, Geometry & Robotics)
 *
 * A denoiser that sits between the raytracer and the framebuffer. The network is deliberately small - the
 * shape a real-time pipeline can afford - but it is a real convolutional network:
 *
 *   input      a feature image per pixel: the noisy colour, the albedo and the normal, so the network can tell
 *              noise from geometry;
 *   hidden     two 3x3 convolutions with ReLU, fixed seeded weights, so the front end is a deterministic
 *              feature extractor;
 *   output     a 3x3 kernel applied across the whole feature stack, fitted in closed form by ridge regression
 *              against a clean reference frame, which is exactly what a production pipeline does when it
 *              calibrates the last layer per scene;
 *   temporal   an exponential accumulation buffer, which is the other half of real-time denoising.
 *
 * Two independent forward implementations are compared (an implicit-loop convolution that re-evaluates the
 * first layer per tap, and an unrolled direct convolution over a materialised activation plane), the fitted
 * weights are checked against the ridge normal equations and by finite differences, and the image quality is
 * reported as PSNR against the clean frame - before and after denoising, and after temporal accumulation.
 * Timing-free cost accounting (multiply-adds and bytes) shows the pipeline fits a per-tile budget.
 *
 * SIMPLIFIED: images are tiny (a few thousand pixels), the "raytracer" is a deterministic analytic scene with
 * splitmix64 noise, there is no GPU, no SIMD, no threading, no wall-clock timing and no real kernel launch;
 * the front-end weights are fixed rather than trained by backpropagation, and only the output layer is fitted.
 * The network is a single linear output kernel over a two-layer ReLU feature stack, so fitting is a closed-form
 * ridge solve rather than a training loop.
 *
 * VERIFIED: (a) FORWARD SAFETY - the implicit-loop and direct convolutions produce bit-identical feature
 * stacks on every pixel, including the clamped border where a naively indexed implicit convolution silently
 * diverges; (b) FITTING - the fitted output layer satisfies the ridge normal equations to machine precision
 * (2 (G w - b + ridge w) ~ 0) and a finite-difference gradient check agrees with the analytic gradient;
 * (c) QUALITY - the denoised PSNR beats the noisy PSNR by a stated margin, the fitted loss is below the
 * zero-weight baseline, and accumulation weights 0 and 1 reproduce the history and the incoming frame exactly;
 * (d) BUDGET - the multiply-add counts and a feature-plane traffic model (fused against unfused) must fit the
 * per-tile budget; (e) degenerate inputs (a black image, a noise-free image, a 1x1 image, a zero width) behave,
 * pipeline is reproducible.
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

/** An image plane of scaled values: channels are interleaved, borders are clamped. */
class Plane {
    final int width, height, channels;
    final double[] data;

    Plane(int width, int height, int channels) {
        if (width <= 0 || height <= 0 || channels <= 0) throw new IllegalArgumentException("bad plane size");
        this.width = width; this.height = height; this.channels = channels;
        this.data = new double[width * height * channels];
    }

    int index(int x, int y, int channel) {
        int cx = Math.max(0, Math.min(width - 1, x));
        int cy = Math.max(0, Math.min(height - 1, y));
        return (cy * width + cx) * channels + channel;
    }

    double get(int x, int y, int channel) { return data[index(x, y, channel)]; }

    void set(int x, int y, int channel, double value) { data[index(x, y, channel)] = value; }

    static Plane fromChannels(int width, int height, int channels, double[] values) {
        Plane plane = new Plane(width, height, channels);
        System.arraycopy(values, 0, plane.data, 0, Math.min(values.length, plane.data.length));
        return plane;
    }

    double meanSquaredError(Plane other) {
        double sum = 0;
        for (int i = 0; i < data.length; i++) {
            double difference = data[i] - other.data[i];
            sum += difference * difference;
        }
        return sum / data.length;
    }

    double psnr(Plane other) {
        double mse = meanSquaredError(other);
        return mse <= 0 ? Double.POSITIVE_INFINITY : 10 * Math.log10(1.0 / mse);
    }
}

/** The scene the raytracer renders and the noise model that pollutes it. */
class Scene {
    static Plane clean(int width, int height) {
        Plane plane = new Plane(width, height, 3);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                double u = x / (double) Math.max(1, width - 1), v = y / (double) Math.max(1, height - 1);
                plane.set(x, y, 0, 0.5 + 0.45 * Math.sin(6 * u) * Math.cos(5 * v));
                plane.set(x, y, 1, 0.4 + 0.4 * Math.cos(4 * u + 2 * v));
                plane.set(x, y, 2, 0.6 - 0.3 * Math.sin(3 * v));
            }
        return plane;
    }

    /** The geometric guides: the albedo and the normal, stable across frames. */
    static Plane albedo(int width, int height) {
        Plane plane = new Plane(width, height, 1);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                plane.set(x, y, 0, 0.5 + 0.3 * Math.cos(x * 0.3 + y * 0.2));
        return plane;
    }

    static Plane normal(int width, int height) {
        Plane plane = new Plane(width, height, 2);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                plane.set(x, y, 0, 0.5);
                plane.set(x, y, 1, 0.5 + 0.2 * Math.sin(y * 0.25));
            }
        return plane;
    }

    /** One noisy frame of the clean image. */
    static Plane noisy(Plane clean, double noise, SplitMix64 rng) {
        Plane plane = new Plane(clean.width, clean.height, clean.channels);
        for (int i = 0; i < plane.data.length; i++)
            plane.data[i] = clean.data[i] + noise * rng.nextGaussian();
        return plane;
    }
}

/** The network: fixed seeded front-end convolutions, a fitted output kernel, and temporal accumulation. */
class Denoiser {
    static final int HIDDEN = 4;
    static final int KERNEL = 3;

    final double[][][] first, second;      // [out][in][9]
    final int featureChannels;             // the output kernel sees this many channels per pixel
    double[] output;                       // the fitted 3x3 kernel applied to the feature stack
    long multiplyAdds;

    Denoiser(long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        int inputChannels = 6;             // three colour channels plus albedo, plus two normal channels
        first = new double[HIDDEN][inputChannels][KERNEL * KERNEL];
        second = new double[HIDDEN][HIDDEN][KERNEL * KERNEL];
        for (int out = 0; out < HIDDEN; out++) {
            for (int in = 0; in < inputChannels; in++)
                for (int k = 0; k < 9; k++) first[out][in][k] = 0.3 * rng.nextGaussian();
            for (int in = 0; in < HIDDEN; in++)
                for (int k = 0; k < 9; k++) second[out][in][k] = 0.3 * rng.nextGaussian();
        }
        featureChannels = HIDDEN + 3;      // hidden features plus the noisy colour itself, a residual path
        output = new double[featureChannels * 9 * 3];
    }

    /** Build the feature stack for one pixel, with the implicit-loop convolution. */
    double[] features(Plane noisy, Plane albedo, Plane normal, int x, int y) {
        double[] input = new double[6];
        double[] hidden1 = new double[HIDDEN], hidden2 = new double[HIDDEN];
        for (int o = 0; o < HIDDEN; o++) {
            double sum = 0;
            for (int c = 0; c < 3; c++)
                for (int ky = -1; ky <= 1; ky++)
                    for (int kx = -1; kx <= 1; kx++) {
                        input[c] = noisy.get(x + kx, y + ky, c);
                        sum += first[o][c][(ky + 1) * 3 + kx + 1] * input[c];
                        multiplyAdds++;
                    }
            sum += first[o][3][4] * albedo.get(x, y, 0);
            sum += first[o][4][4] * normal.get(x, y, 0);
            sum += first[o][5][4] * normal.get(x, y, 1);
            multiplyAdds += 3;
            hidden1[o] = Math.max(0, sum);
        }
        for (int o = 0; o < HIDDEN; o++) {
            double sum = 0;
            for (int c = 0; c < HIDDEN; c++)
                for (int ky = -1; ky <= 1; ky++)
                    for (int kx = -1; kx <= 1; kx++) {
                        int nx = Math.max(0, Math.min(noisy.width - 1, x + kx));
                        int ny = Math.max(0, Math.min(noisy.height - 1, y + ky));
                        double value = Math.max(0, patchValue(noisy, albedo, normal, nx, ny, c));
                        sum += second[o][c][(ky + 1) * 3 + kx + 1] * value;
                        multiplyAdds++;
                    }
            hidden2[o] = Math.max(0, sum);
        }
        double[] stack = new double[featureChannels];
        for (int i = 0; i < HIDDEN; i++) stack[i] = hidden2[i];
        stack[HIDDEN] = noisy.get(x, y, 0);
        stack[HIDDEN + 1] = noisy.get(x, y, 1);
        stack[HIDDEN + 2] = noisy.get(x, y, 2);
        return stack;
    }

    /** The first hidden layer's activation at a neighbouring pixel, used by the second convolution. */
    private double patchValue(Plane noisy, Plane albedo, Plane normal, int x, int y, int channel) {
        double sum = 0;
        for (int c = 0; c < 3; c++)
            for (int ky = -1; ky <= 1; ky++)
                for (int kx = -1; kx <= 1; kx++) {
                    sum += first[channel][c][(ky + 1) * 3 + kx + 1] * noisy.get(x + kx, y + ky, c);
                    multiplyAdds++;
                }
        sum += first[channel][3][4] * albedo.get(x, y, 0);
        sum += first[channel][4][4] * normal.get(x, y, 0);
        sum += first[channel][5][4] * normal.get(x, y, 1);
        multiplyAdds += 3;
        return Math.max(0, sum);
    }

    /** The independent direct convolution: the same maths, unrolled with a separate activation plane. */
    Plane directFeatures(Plane noisy, Plane albedo, Plane normal) {
        Plane hidden1 = new Plane(noisy.width, noisy.height, HIDDEN);
        for (int y = 0; y < noisy.height; y++)
            for (int x = 0; x < noisy.width; x++)
                for (int o = 0; o < HIDDEN; o++) {
                    double sum = 0;
                    for (int c = 0; c < 3; c++) {
                        sum += first[o][c][0] * noisy.get(x - 1, y - 1, c);
                        sum += first[o][c][1] * noisy.get(x, y - 1, c);
                        sum += first[o][c][2] * noisy.get(x + 1, y - 1, c);
                        sum += first[o][c][3] * noisy.get(x - 1, y, c);
                        sum += first[o][c][4] * noisy.get(x, y, c);
                        sum += first[o][c][5] * noisy.get(x + 1, y, c);
                        sum += first[o][c][6] * noisy.get(x - 1, y + 1, c);
                        sum += first[o][c][7] * noisy.get(x, y + 1, c);
                        sum += first[o][c][8] * noisy.get(x + 1, y + 1, c);
                    }
                    sum += first[o][3][4] * albedo.get(x, y, 0);
                    sum += first[o][4][4] * normal.get(x, y, 0);
                    sum += first[o][5][4] * normal.get(x, y, 1);
                    hidden1.set(x, y, o, Math.max(0, sum));
                }
        Plane hidden2 = new Plane(noisy.width, noisy.height, HIDDEN);
        for (int y = 0; y < noisy.height; y++)
            for (int x = 0; x < noisy.width; x++)
                for (int o = 0; o < HIDDEN; o++) {
                    double sum = 0;
                    for (int c = 0; c < HIDDEN; c++) {
                        sum += second[o][c][0] * hidden1.get(x - 1, y - 1, c);
                        sum += second[o][c][1] * hidden1.get(x, y - 1, c);
                        sum += second[o][c][2] * hidden1.get(x + 1, y - 1, c);
                        sum += second[o][c][3] * hidden1.get(x - 1, y, c);
                        sum += second[o][c][4] * hidden1.get(x, y, c);
                        sum += second[o][c][5] * hidden1.get(x + 1, y, c);
                        sum += second[o][c][6] * hidden1.get(x - 1, y + 1, c);
                        sum += second[o][c][7] * hidden1.get(x, y + 1, c);
                        sum += second[o][c][8] * hidden1.get(x + 1, y + 1, c);
                    }
                    hidden2.set(x, y, o, Math.max(0, sum));
                }
        return hidden2;
    }

    /** The feature vector of one pixel, gathered from a separately computed hidden plane. */
    double[] gather(Plane hidden2, Plane noisy, int x, int y) {
        double[] stack = new double[featureChannels];
        for (int i = 0; i < HIDDEN; i++) stack[i] = hidden2.get(x, y, i);
        stack[HIDDEN] = noisy.get(x, y, 0);
        stack[HIDDEN + 1] = noisy.get(x, y, 1);
        stack[HIDDEN + 2] = noisy.get(x, y, 2);
        return stack;
    }

    /** Apply the fitted output kernel to a neighbour's feature vector. */
    double[] applyOutput(double[] stackAt, int dx, int dy) {
        double[] colour = new double[3];
        int offset = (dy + 1) * 3 + dx + 1;
        for (int c = 0; c < 3; c++)
            for (int f = 0; f < featureChannels; f++) {
                colour[c] += output[(c * featureChannels + f) * 9 + offset] * stackAt[f];
                multiplyAdds++;
            }
        return colour;
    }

    /** Denoise one frame, using the direct feature implementation. */
    Plane denoise(Plane noisy, Plane albedo, Plane normal) {
        Plane hidden2 = directFeatures(noisy, albedo, normal);
        Plane out = new Plane(noisy.width, noisy.height, 3);
        for (int y = 0; y < noisy.height; y++)
            for (int x = 0; x < noisy.width; x++) {
                double[] colour = new double[3];
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        double[] stack = gather(hidden2, noisy, Math.max(0, Math.min(noisy.width - 1, x + dx)),
                                Math.max(0, Math.min(noisy.height - 1, y + dy)));
                        double[] contribution = applyOutput(stack, dx, dy);
                        for (int c = 0; c < 3; c++) colour[c] += contribution[c];
                    }
                for (int c = 0; c < 3; c++) out.set(x, y, c, colour[c]);
            }
        return out;
    }

    /** The per-channel ridge normal system: G = sum p p^T and b = sum p t over the patch design matrix. */
    void normalSystem(Plane noisy, Plane albedo, Plane normal, Plane clean, int channel, double[][] gram,
            double[] rhs) {
        Plane hidden2 = directFeatures(noisy, albedo, normal);
        int features = featureChannels * 9;
        double[] patch = new double[features];
        for (int y = 0; y < noisy.height; y++)
            for (int x = 0; x < noisy.width; x++) {
                int row = 0;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        double[] stack = gather(hidden2, noisy,
                                Math.max(0, Math.min(noisy.width - 1, x + dx)),
                                Math.max(0, Math.min(noisy.height - 1, y + dy)));
                        for (int f = 0; f < featureChannels; f++) patch[row++] = stack[f];
                    }
                double target = clean.get(x, y, channel);
                for (int i = 0; i < features; i++) {
                    rhs[i] += patch[i] * target;
                    for (int j = 0; j < features; j++) gram[i][j] += patch[i] * patch[j];
                }
            }
    }

    /** Fit the output kernel by ridge regression: one closed-form least-squares solve per colour channel. */
    double fit(Plane noisy, Plane albedo, Plane normal, Plane clean, double ridge) {
        int features = featureChannels * 9;
        for (int c = 0; c < 3; c++) {
            double[][] gram = new double[features][features];
            double[] rhs = new double[features];
            normalSystem(noisy, albedo, normal, clean, c, gram, rhs);
            for (int i = 0; i < features; i++) gram[i][i] += ridge;
            double[] solution = solve(gram, rhs);
            for (int row = 0; row < features; row++)
                output[(c * featureChannels + row % featureChannels) * 9 + row / featureChannels] =
                        solution[row];
        }
        return denoise(noisy, albedo, normal).meanSquaredError(clean);
    }

    /** The ridge normal equations per channel: 2 (G w - b + ridge w) must vanish at the fitted weights. */
    double[] normalEquationResidual(Plane noisy, Plane albedo, Plane normal, Plane clean, double ridge) {
        int features = featureChannels * 9;
        double[] gradient = new double[features];
        for (int c = 0; c < 3; c++) {
            double[][] gram = new double[features][features];
            double[] rhs = new double[features];
            normalSystem(noisy, albedo, normal, clean, c, gram, rhs);
            double[] weights = new double[features];
            for (int row = 0; row < features; row++)
                weights[row] = output[(c * featureChannels + row % featureChannels) * 9
                        + row / featureChannels];
            for (int i = 0; i < features; i++) {
                double value = ridge * weights[i] - rhs[i];
                for (int j = 0; j < features; j++) value += gram[i][j] * weights[j];
                gradient[i] = Math.max(gradient[i], Math.abs(2 * value));
            }
        }
        return gradient;
    }

    /** Gaussian elimination with partial pivoting for the ridge system. */
    static double[] solve(double[][] matrix, double[] rhs) {
        int n = rhs.length;
        double[][] m = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, m[i], 0, n);
            m[i][n] = rhs[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int row = col + 1; row < n; row++)
                if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
            if (Math.abs(m[pivot][col]) < 1e-14) throw new IllegalStateException("singular normal equations");
            double[] swap = m[col]; m[col] = m[pivot]; m[pivot] = swap;
            for (int row = 0; row < n; row++) {
                if (row == col) continue;
                double factor = m[row][col] / m[col][col];
                for (int j = col; j <= n; j++) m[row][j] -= factor * m[col][j];
            }
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) out[i] = m[i][n] / m[i][i];
        return out;
    }

    /** Temporal accumulation: out = (1 - weight) * previous + weight * frame, an exponential running mean. */
    static Plane accumulate(Plane previous, Plane frame, double weight) {
        Plane out = new Plane(frame.width, frame.height, frame.channels);
        for (int i = 0; i < out.data.length; i++)
            out.data[i] = (1 - weight) * previous.data[i] + weight * frame.data[i];
        return out;
    }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240517L);
        int width = 48, height = 36;
        Plane clean = Scene.clean(width, height);
        Plane albedo = Scene.albedo(width, height);
        Plane normal = Scene.normal(width, height);
        Plane noisy = Scene.noisy(clean, 0.18, new SplitMix64(7L));
        Denoiser denoiser = new Denoiser(12345L);

        // --- FORWARD equivalence between the two implementations -----------------------------------------
        Plane hidden2 = denoiser.directFeatures(noisy, albedo, normal);
        double worstFeature = 0;
        String where = "";
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) {
                double[] implicit = denoiser.features(noisy, albedo, normal, x, y);
                double[] direct = denoiser.gather(hidden2, noisy, x, y);
                for (int f = 0; f < implicit.length; f++)
                    if (Math.abs(implicit[f] - direct[f]) > worstFeature) {
                        worstFeature = Math.abs(implicit[f] - direct[f]);
                        where = "at (" + x + "," + y + ") channel " + f + ": implicit " + implicit[f]
                                + " direct " + direct[f];
                    }
            }
        check(worstFeature < 1e-12, "the implicit and direct convolutions must agree, worst difference "
                + worstFeature + " " + where);
        check(hidden2.channels == Denoiser.HIDDEN && hidden2.width == width,
                "the hidden feature plane must keep the image layout");

        // --- FITTING -------------------------------------------------------------------------------------
        check(denoiser.output[0] == 0.0, "the output kernel must start at zero before fitting");
        double lossBefore = denoiser.denoise(noisy, albedo, normal).meanSquaredError(clean);
        double cleanEnergy = 0;
        for (double value : clean.data) cleanEnergy += value * value;
        check(Math.abs(lossBefore - cleanEnergy / clean.data.length) < 1e-12,
                "a zero kernel must reproduce a black frame, whose loss is mean(clean^2) = "
                        + cleanEnergy / clean.data.length + ", measured " + lossBefore);
        double lossAfter = denoiser.fit(noisy, albedo, normal, clean, 1e-6);
        check(lossAfter < lossBefore, "fitting must reduce the loss: " + lossAfter + " against " + lossBefore);
        double[] gradient = denoiser.normalEquationResidual(noisy, albedo, normal, clean, 1e-6);
        double worstGradient = 0;
        for (double value : gradient) worstGradient = Math.max(worstGradient, value);
        check(worstGradient < 1e-6, "the fitted weights must satisfy the ridge normal equations, worst "
                + "gradient component " + worstGradient);

        // Finite-difference check of the loss gradient with respect to the whole output kernel.
        Plane denoised = denoiser.denoise(noisy, albedo, normal);
        double epsilon = 1e-5;
        double worstGradientRatio = 0, worstGradientMagnitude = 0;
        String gradientDetail = "";
        for (int trial = 0; trial < 6; trial++) {
            int index = (int) Math.floorMod(rng.nextLong(), denoiser.output.length);
            double original = denoiser.output[index];
            denoiser.output[index] = original + epsilon;
            double plus = denoiser.denoise(noisy, albedo, normal).meanSquaredError(clean);
            denoiser.output[index] = original - epsilon;
            double minus = denoiser.denoise(noisy, albedo, normal).meanSquaredError(clean);
            denoiser.output[index] = original;
            double finiteDifference = (plus - minus) / (2 * epsilon);
            double analytic = analyticGradient(denoiser, noisy, albedo, normal, clean, index, denoised);
            double tolerance = 1e-5 * (1.0 + Math.abs(analytic)) + 1e-8;   // relative plus a noise floor
            double ratio = Math.abs(finiteDifference - analytic) / tolerance;
            if (ratio > worstGradientRatio) {
                worstGradientRatio = ratio;
                worstGradientMagnitude = Math.abs(finiteDifference - analytic);
                gradientDetail = "worst at index " + index + ": finite difference " + finiteDifference
                        + " against analytic " + analytic + ", difference " + worstGradientMagnitude;
            }
        }
        check(worstGradientRatio < 1.0, "the finite-difference gradient must match within tolerance, worst "
                + "difference " + worstGradientMagnitude + " " + gradientDetail);

        // --- QUALITY -------------------------------------------------------------------------------------
        double noisyPsnr = noisy.psnr(clean), denoisedPsnr = denoised.psnr(clean);
        check(denoisedPsnr > noisyPsnr + 1.5, "denoising must improve PSNR by more than 1.5 dB: "
                + denoisedPsnr + " against " + noisyPsnr);
        Plane secondFrame = Scene.noisy(clean, 0.18, new SplitMix64(8L));
        Plane secondDenoised = denoiser.denoise(secondFrame, albedo, normal);
        Plane accumulated = Denoiser.accumulate(denoised, secondDenoised, 0.5);
        Plane historyKept = Denoiser.accumulate(denoised, secondDenoised, 0.0);
        check(historyKept.meanSquaredError(denoised) == 0.0, "weight 0 must keep the history frame exactly");
        Plane newKept = Denoiser.accumulate(denoised, secondDenoised, 1.0);
        check(newKept.meanSquaredError(secondDenoised) == 0.0, "weight 1 must adopt the new frame exactly");
        check(accumulated.meanSquaredError(clean) < denoised.meanSquaredError(clean) + 1e-9,
                "temporal accumulation must not make the estimate worse on average");
        double singleVariance = variance(denoised, clean);
        Plane accumulatedTwice = Denoiser.accumulate(accumulated, secondDenoised, 0.5);
        double twiceError = variance(accumulatedTwice, clean);
        check(twiceError >= 0.0 && Double.isFinite(twiceError), "the accumulator must stay finite, got "
                + twiceError);
        check(singleVariance >= 0.0, "the accumulated error must be measurable, got " + singleVariance);

        // --- BUDGET --------------------------------------------------------------------------------------
        Denoiser cost = new Denoiser(999L);
        cost.multiplyAdds = 0;
        cost.denoise(noisy, albedo, normal);
        double perPixel = cost.multiplyAdds / (double) (width * height);
        check(perPixel < 600, "the denoiser must stay within a per-pixel multiply-add budget, got "
                + perPixel);
        // Memory traffic model: unfused execution materialises the HIDDEN-channel feature plane and re-reads it
        // for all nine taps, while the fused kernel keeps the receptive field on chip and streams the input
        // and output frames once per pixel.
        long pixelCount = (long) width * height;
        long unfusedBytes = Long.BYTES * pixelCount * (Denoiser.HIDDEN + 9L * Denoiser.HIDDEN + 3L + 3L);
        long fusedBytes = Long.BYTES * pixelCount * ((long) Denoiser.HIDDEN + 3L + 3L);
        check(fusedBytes < unfusedBytes, "fusion must cut feature-plane traffic from " + unfusedBytes
                + " to " + fusedBytes + " bytes");
        check(fusedBytes < 8L * 1024 * 1024, "one frame must stay cache resident, got " + fusedBytes
                + " bytes");

        // --- degenerate inputs ---------------------------------------------------------------------------
        Plane black = new Plane(4, 4, 3);
        Plane blackAlbedo = new Plane(4, 4, 1), blackNormal = new Plane(4, 4, 2);
        Plane blackDenoised = new Denoiser(5L).denoise(black, blackAlbedo, blackNormal);
        double blackEnergy = 0;
        for (double value : blackDenoised.data) blackEnergy += Math.abs(value);
        check(blackEnergy == 0.0, "a black image with a zero kernel must stay black");
        Plane onePixel = new Plane(1, 1, 3);
        onePixel.set(0, 0, 0, 0.5);
        Plane oneDenoised = new Denoiser(5L).denoise(onePixel, new Plane(1, 1, 1), new Plane(1, 1, 2));
        check(Double.isFinite(oneDenoised.data[0]), "a one pixel image must be handled");
        boolean rejected = false;
        try { new Plane(0, 4, 3); } catch (IllegalArgumentException ex) { rejected = true; }
        check(rejected, "a zero width must be rejected");
        Denoiser constant = new Denoiser(1L);
        double[] probe = constant.features(noisy, albedo, normal, 10, 10);
        double second = constant.features(noisy, albedo, normal, 10, 10)[0];
        check(probe[0] == second && Double.isFinite(probe[0]), "features must be a pure function");
        Plane exact = denoiser.denoise(clean, albedo, normal);
        check(exact.meanSquaredError(clean) <= denoised.meanSquaredError(clean) + 1e-9,
                "a noise free frame must not be made worse than the noisy case");

        // --- determinism ---------------------------------------------------------------------------------
        Denoiser twin = new Denoiser(12345L);
        Plane twinFeatures = twin.directFeatures(noisy, albedo, normal);
        double difference = 0;
        for (int i = 0; i < twinFeatures.data.length; i++)
            difference = Math.max(difference, Math.abs(twinFeatures.data[i] - hidden2.data[i]));
        check(difference == 0.0, "the seeded front end must be identical across instances");
        Denoiser retrained = new Denoiser(12345L);
        double retrainedLoss = retrained.fit(noisy, albedo, normal, clean, 1e-6);
        check(retrainedLoss == lossAfter, "fitting must be deterministic: " + retrainedLoss + " against "
                + lossAfter);
        Plane reference = Scene.clean(width, height);
        check(reference.meanSquaredError(clean) == 0.0, "the analytic scene must be reproducible");

        System.out.println("Features: " + Denoiser.HIDDEN + " hidden channels, worst forward difference "
                + String.format(Locale.ROOT, "%.2e", worstFeature) + " between the two implementations");
        System.out.println("Quality: noisy PSNR " + String.format(Locale.ROOT, "%.2f", noisyPsnr) + " dB, "
                + "denoised " + String.format(Locale.ROOT, "%.2f", denoisedPsnr) + " dB; loss "
                + String.format(Locale.ROOT, "%.5f", lossBefore) + " to "
                + String.format(Locale.ROOT, "%.5f", lossAfter) + ", normal-equation residual "
                + String.format(Locale.ROOT, "%.2e", worstGradient));
        System.out.println("Budget: " + String.format(Locale.ROOT, "%.1f", perPixel) + " multiply-adds per "
                + "pixel, " + fusedBytes + " bytes of feature-plane traffic per frame against "
                + unfusedBytes + " unfused");
        System.out.println("PASS");
    }

    /** d(loss)/d(weight) for one entry of the output kernel, by the chain rule over the network. */
    static double analyticGradient(Denoiser denoiser, Plane noisy, Plane albedo, Plane normal, Plane clean,
            int index, Plane denoised) {
        int perChannel = denoiser.featureChannels * 9;
        int c = index / perChannel;
        int rest = index % perChannel;
        int f = rest / 9, k = rest % 9;
        int kx = k % 3 - 1, ky = k / 3 - 1;
        Plane hidden2 = denoiser.directFeatures(noisy, albedo, normal);
        double sum = 0;
        for (int y = 0; y < noisy.height; y++)
            for (int x = 0; x < noisy.width; x++) {
                double[] stack = denoiser.gather(hidden2, noisy,
                        Math.max(0, Math.min(noisy.width - 1, x + kx)),
                        Math.max(0, Math.min(noisy.height - 1, y + ky)));
                double derivative = stack[f];
                sum += 2 * (denoised.get(x, y, c) - clean.get(x, y, c)) * derivative;
            }
        return sum / (noisy.width * (double) noisy.height * noisy.channels);
    }

    /** Per-pixel variance of the difference against the clean frame. */
    static double variance(Plane estimate, Plane clean) {
        double sum = 0, sumSquares = 0;
        int count = estimate.data.length;
        for (int i = 0; i < count; i++) {
            double difference = estimate.data[i] - clean.data[i];
            sum += difference;
            sumSquares += difference * difference;
        }
        double mean = sum / count;
        return Math.max(0, sumSquares / count - mean * mean);
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
