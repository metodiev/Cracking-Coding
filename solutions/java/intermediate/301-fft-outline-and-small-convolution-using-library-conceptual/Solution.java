/** Task 301 - FFT outline and small convolution (conceptual). (Tier: Intermediate, CTCI Ch 1) */
import java.util.*;

public class Solution {
    /** In-place iterative Cooley-Tukey transform on parallel real/imag arrays. */
    static void fft(double[] real, double[] imag, boolean invert) {
        int n = real.length;
        int j = 0;
        for (int i = 1; i < n; i++) {
            int bit = n >> 1;
            while ((j & bit) != 0) {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if (i < j) {
                double tmpReal = real[i]; real[i] = real[j]; real[j] = tmpReal;
                double tmpImag = imag[i]; imag[i] = imag[j]; imag[j] = tmpImag;
            }
        }

        for (int length = 2; length <= n; length <<= 1) {
            double angle = 2 * Math.PI / length * (invert ? -1 : 1);
            double wlenReal = Math.cos(angle), wlenImag = Math.sin(angle);
            int half = length >> 1;
            for (int start = 0; start < n; start += length) {
                double wReal = 1.0, wImag = 0.0;
                for (int k = 0; k < half; k++) {
                    double uReal = real[start + k], uImag = imag[start + k];
                    double vReal = real[start + k + half] * wReal - imag[start + k + half] * wImag;
                    double vImag = real[start + k + half] * wImag + imag[start + k + half] * wReal;
                    real[start + k] = uReal + vReal;
                    imag[start + k] = uImag + vImag;
                    real[start + k + half] = uReal - vReal;
                    imag[start + k + half] = uImag - vImag;
                    double nextReal = wReal * wlenReal - wImag * wlenImag;
                    wImag = wReal * wlenImag + wImag * wlenReal;
                    wReal = nextReal;
                }
            }
        }

        if (invert) {
            for (int i = 0; i < n; i++) {
                real[i] /= n;
                imag[i] /= n;
            }
        }
    }

    static int[] convolutionFft(int[] a, int[] b) {
        if (a.length == 0 || b.length == 0) return new int[0];
        int size = 1;
        while (size < a.length + b.length - 1) size <<= 1;
        double[] realA = new double[size], imagA = new double[size];
        double[] realB = new double[size], imagB = new double[size];
        for (int i = 0; i < a.length; i++) realA[i] = a[i];
        for (int i = 0; i < b.length; i++) realB[i] = b[i];

        fft(realA, imagA, false);
        fft(realB, imagB, false);
        for (int i = 0; i < size; i++) {
            double re = realA[i] * realB[i] - imagA[i] * imagB[i];
            double im = realA[i] * imagB[i] + imagA[i] * realB[i];
            realA[i] = re;
            imagA[i] = im;
        }
        fft(realA, imagA, true);

        int[] out = new int[a.length + b.length - 1];
        for (int i = 0; i < out.length; i++) out[i] = (int) Math.round(realA[i]);
        return out;
    }

    static int[] convolutionNaive(int[] a, int[] b) {
        if (a.length == 0 || b.length == 0) return new int[0];
        int[] out = new int[a.length + b.length - 1];
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < b.length; j++)
                out[i + j] += a[i] * b[j];
        return out;
    }

    private static int seed = 1123;

    private static List<int[]> lcgVectors(int count) {
        List<int[]> vectors = new ArrayList<>();
        for (int k = 0; k < count; k++) {
            int size = 1 + seed % 6;
            int[] vector = new int[size];
            for (int i = 0; i < size; i++) {
                seed = (seed * 75 + 74) % 65537;
                vector[i] = seed % 11 - 5;
            }
            vectors.add(vector);
        }
        return vectors;
    }

    public static void main(String[] args) {
        check(Arrays.equals(convolutionFft(new int[]{1,2,3}, new int[]{0,1,1}), new int[]{0,1,3,5,3}), "classic");
        check(Arrays.equals(convolutionFft(new int[]{1,1}, new int[]{1,1}), new int[]{1,2,1}), "square");
        check(Arrays.equals(convolutionFft(new int[]{7}, new int[]{9}), new int[]{63}), "scalars");
        check(convolutionFft(new int[]{}, new int[]{1,2}).length == 0, "empty operand");
        check(Arrays.equals(convolutionFft(new int[]{4,5}, new int[]{6}), new int[]{24,30}), "single factor");
        check(Arrays.equals(convolutionFft(new int[]{1}, new int[]{3,4,5}), new int[]{3,4,5}), "delta identity");

        List<int[]> vectors = lcgVectors(40);
        for (int i = 0; i + 1 < vectors.size(); i += 2) {
            int[] a = vectors.get(i), b = vectors.get(i + 1);
            check(Arrays.equals(convolutionFft(a, b), convolutionNaive(a, b)), "random forward " + i);
            check(Arrays.equals(convolutionFft(b, a), convolutionNaive(a, b)), "random swapped " + i);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
