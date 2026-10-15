// Task 301 - FFT outline and small convolution (conceptual). (Tier: Intermediate, CTCI Ch 1)
using System;
using System.Collections.Generic;

public static class Solution
{
    /// <summary>In-place iterative Cooley-Tukey transform on parallel real/imag arrays.</summary>
    public static void Fft(double[] real, double[] imag, bool invert)
    {
        int n = real.Length;
        int j = 0;
        for (int i = 1; i < n; i++)
        {
            int bit = n >> 1;
            while ((j & bit) != 0) { j ^= bit; bit >>= 1; }
            j |= bit;
            if (i < j)
            {
                (real[i], real[j]) = (real[j], real[i]);
                (imag[i], imag[j]) = (imag[j], imag[i]);
            }
        }

        for (int length = 2; length <= n; length <<= 1)
        {
            double angle = 2 * Math.PI / length * (invert ? -1 : 1);
            double wlenReal = Math.Cos(angle), wlenImag = Math.Sin(angle);
            int half = length >> 1;
            for (int start = 0; start < n; start += length)
            {
                double wReal = 1.0, wImag = 0.0;
                for (int k = 0; k < half; k++)
                {
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

        if (invert)
        {
            for (int i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; }
        }
    }

    public static int[] ConvolutionFft(int[] a, int[] b)
    {
        if (a.Length == 0 || b.Length == 0) return Array.Empty<int>();
        int size = 1;
        while (size < a.Length + b.Length - 1) size <<= 1;
        var realA = new double[size];
        var imagA = new double[size];
        var realB = new double[size];
        var imagB = new double[size];
        for (int i = 0; i < a.Length; i++) realA[i] = a[i];
        for (int i = 0; i < b.Length; i++) realB[i] = b[i];

        Fft(realA, imagA, false);
        Fft(realB, imagB, false);
        for (int i = 0; i < size; i++)
        {
            double re = realA[i] * realB[i] - imagA[i] * imagB[i];
            double im = realA[i] * imagB[i] + imagA[i] * realB[i];
            realA[i] = re;
            imagA[i] = im;
        }
        Fft(realA, imagA, true);

        var outList = new int[a.Length + b.Length - 1];
        for (int i = 0; i < outList.Length; i++) outList[i] = (int)Math.Round(realA[i]);
        return outList;
    }

    public static int[] ConvolutionNaive(int[] a, int[] b)
    {
        if (a.Length == 0 || b.Length == 0) return Array.Empty<int>();
        var outList = new int[a.Length + b.Length - 1];
        for (int i = 0; i < a.Length; i++)
            for (int j = 0; j < b.Length; j++)
                outList[i + j] += a[i] * b[j];
        return outList;
    }

    private static int seed = 1123;

    private static List<int[]> LcgVectors(int count)
    {
        var vectors = new List<int[]>();
        for (int k = 0; k < count; k++)
        {
            int size = 1 + seed % 6;
            var vector = new int[size];
            for (int i = 0; i < size; i++)
            {
                seed = (seed * 75 + 74) % 65537;
                vector[i] = seed % 11 - 5;
            }
            vectors.Add(vector);
        }
        return vectors;
    }

    private static string Join(int[] values) => string.Join(",", values);

    public static void Main()
    {
        Check(Join(ConvolutionFft(new[] { 1, 2, 3 }, new[] { 0, 1, 1 })) == "0,1,3,5,3", "classic");
        Check(Join(ConvolutionFft(new[] { 1, 1 }, new[] { 1, 1 })) == "1,2,1", "square");
        Check(Join(ConvolutionFft(new[] { 7 }, new[] { 9 })) == "63", "scalars");
        Check(ConvolutionFft(Array.Empty<int>(), new[] { 1, 2 }).Length == 0, "empty operand");
        Check(Join(ConvolutionFft(new[] { 4, 5 }, new[] { 6 })) == "24,30", "single factor");
        Check(Join(ConvolutionFft(new[] { 1 }, new[] { 3, 4, 5 })) == "3,4,5", "delta identity");

        var vectors = LcgVectors(40);
        for (int i = 0; i + 1 < vectors.Count; i += 2)
        {
            var a = vectors[i];
            var b = vectors[i + 1];
            Check(Join(ConvolutionFft(a, b)) == Join(ConvolutionNaive(a, b)), "random forward " + i);
            Check(Join(ConvolutionFft(b, a)) == Join(ConvolutionNaive(a, b)), "random swapped " + i);
        }
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
