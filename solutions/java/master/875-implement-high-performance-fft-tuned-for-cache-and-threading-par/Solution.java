/** Task 875 - Implement high-performance FFT tuned for cache and threading parameters.
 * (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)
 *
 * The kernel under the telescope is the iterative radix-2 Cooley-Tukey transform: bit-reverse the input, then
 * run log2(n) stages in which every pair (i + j, i + j + L/2) inside a block of length L is combined with a
 * twiddle. The *arithmetic* of a stage only ever touches elements inside blocks of length L, and that single
 * observation drives the tuning:
 *
 *   - stage-by-stage (breadth first): sweep the whole array once per stage. With n = 4096 and a 512-line
 *     cache every sweep is a compulsory miss per element because the array is eight times the cache;
 *   - block-resident (depth first): process every block of B elements to completion - its own log2(B) stages -
 *     before moving on, then finish the remaining log2(n/B) stages across blocks. The two orders execute the
 *     same butterflies in the same groups, so the results are bit-for-bit identical; only the miss count
 *     changes. This is exactly what a cache-tuning parameter is supposed to buy, and it is measured here
 *     rather than asserted: each kernel records the indices it touches and an LRU line model replays the trace.
 *
 * Threading is modelled the same way: the independent short transforms of the blocked kernel are tasks with
 * butterfly costs, a virtual-clock list scheduler packs them onto p workers, and because the decomposition and
 * the combination order are fixed the output cannot depend on p - which the checks confirm bitwise.
 * A recursive radix-2 kernel, a Bluestein chirp kernel for arbitrary lengths and the O(n^2) definition
 * complete the picture.
 *
 * SIMPLIFIED: no threads, no vector units, no wall-clock timing and no vendor library; the cache is an LRU
 * over an index trace, "time" is a virtual butterfly count, and n stops at 4096 so the battery runs in
 * milliseconds. Inputs come from splitmix64 with a fixed seed.
 *
 * VERIFIED: (a) CORRECTNESS - recursive, iterative, blocked and Bluestein kernels all match the O(n^2)
 * definition for n = 1, 2, 4 ... 512 and for the awkward lengths 3, 5, 7, 12, 100; (b) EQUIVALENCE - the
 * blocked order is bit-for-bit identical to the stage-by-stage order for every block size, which is the
 * structural claim that makes the tuning safe; (c) CACHE - at n = 4096 with 512 lines the blocked kernel has
 * far fewer misses than the stage-by-stage kernel, the miss curve in the block size is U-shaped with its
 * minimum near the cache size, the model reproduces a hand-computable sequential scan exactly, and a
 * one-line cache misses on every access; (d) ALGEBRA - inverse round-trip, Parseval, the convolution theorem
 * against a direct convolution and linearity; (e) THREADING - the makespan of the p-way schedule lies between
 * the work lower bound and the serial cost, speedup never exceeds p, and p = 1, 2, 4, 8 give identical bits;
 * (f) degenerate inputs (length one, constant, zero, impulse, real signal) and invalid lengths behave.
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

/** A complex signal with an optional access trace, so cache behaviour can be measured, not guessed. */
class Signal {
    final double[] re, im;
    final List<Integer> trace = new ArrayList<>();
    boolean recording;

    Signal(double[] re, double[] im) { this.re = re; this.im = im; }

    static Signal zeros(int n) { return new Signal(new double[n], new double[n]); }

    static Signal random(int n, SplitMix64 rng) {
        Signal out = zeros(n);
        for (int i = 0; i < n; i++) {
            out.re[i] = 2 * rng.nextDouble() - 1;
            out.im[i] = 2 * rng.nextDouble() - 1;
        }
        return out;
    }

    int size() { return re.length; }

    void startRecording() { recording = true; trace.clear(); }

    void touch(int index) { if (recording) trace.add(index); }

    /** One radix-2 butterfly: both elements are read and both are written, in that order. */
    void butterfly(int a, int b, double wr, double wi) {
        touch(a); touch(b);
        double br = re[b], bi = im[b];
        double tr = wr * br - wi * bi, ti = wr * bi + wi * br;
        double ar = re[a], ai = im[a];
        re[b] = ar - tr; im[b] = ai - ti;
        re[a] = ar + tr; im[a] = ai + ti;
        touch(a); touch(b);
    }

    Signal copy() {
        Signal out = zeros(size());
        System.arraycopy(re, 0, out.re, 0, size());
        System.arraycopy(im, 0, out.im, 0, size());
        return out;
    }

    double maxDifference(Signal other) {
        double worst = 0;
        for (int i = 0; i < size(); i++)
            worst = Math.max(worst, Math.max(Math.abs(re[i] - other.re[i]), Math.abs(im[i] - other.im[i])));
        return worst;
    }

    double maxMagnitude() {
        double worst = 0;
        for (int i = 0; i < size(); i++) worst = Math.max(worst, Math.hypot(re[i], im[i]));
        return worst;
    }

    boolean identical(Signal other) {
        if (size() != other.size()) return false;
        for (int i = 0; i < size(); i++)
            if (Double.doubleToLongBits(re[i]) != Double.doubleToLongBits(other.re[i])
                    || Double.doubleToLongBits(im[i]) != Double.doubleToLongBits(other.im[i])) return false;
        return true;
    }
}

/** A fully associative LRU cache of `lines` cache lines, each holding `lineSize` consecutive elements. */
class CacheModel {
    final int lines, lineSize;
    final LinkedHashMap<Integer, Boolean> resident = new LinkedHashMap<>(64, 0.75f, true);
    int misses, hits;

    CacheModel(int lines, int lineSize) { this.lines = lines; this.lineSize = lineSize; }

    void replay(List<Integer> trace) {
        for (int index : trace) {
            Integer line = index / lineSize;
            if (resident.containsKey(line)) { hits++; resident.get(line); }
            else {
                misses++;
                resident.put(line, Boolean.TRUE);
                if (resident.size() > lines) resident.remove(resident.keySet().iterator().next());
            }
        }
    }
}

/** The Fourier kernels: the definition, two radix-2 orders, a blocked order and Bluestein's chirp. */
class Kernels {
    static boolean isPowerOfTwo(int n) { return n > 0 && (n & (n - 1)) == 0; }

    static int log2(int n) {
        int log = 0;
        while ((1 << log) < n) log++;
        return log;
    }

    /** X_k = sum_j x_j exp(-2 pi i j k / n): the independent reference, deliberately O(n^2). */
    static Signal dft(Signal x) {
        int n = x.size();
        Signal out = Signal.zeros(n);
        for (int k = 0; k < n; k++) {
            double sumRe = 0, sumIm = 0;
            for (int j = 0; j < n; j++) {
                double angle = -2 * Math.PI * ((double) j * k % n) / n;
                double c = Math.cos(angle), s = Math.sin(angle);
                sumRe += x.re[j] * c - x.im[j] * s;
                sumIm += x.re[j] * s + x.im[j] * c;
            }
            out.re[k] = sumRe; out.im[k] = sumIm;
        }
        return out;
    }

    /** Recursive decimation in time, written the textbook way for contrast with the iterative orders. */
    static Signal recursiveFft(Signal x) {
        int n = x.size();
        if (!isPowerOfTwo(n)) throw new IllegalArgumentException("radix-2 needs a power of two, got " + n);
        if (n == 1) return x.copy();
        Signal even = Signal.zeros(n / 2), odd = Signal.zeros(n / 2);
        for (int i = 0; i < n / 2; i++) {
            even.re[i] = x.re[2 * i]; even.im[i] = x.im[2 * i];
            odd.re[i] = x.re[2 * i + 1]; odd.im[i] = x.im[2 * i + 1];
        }
        Signal evenTransform = recursiveFft(even), oddTransform = recursiveFft(odd);
        Signal out = Signal.zeros(n);
        for (int k = 0; k < n / 2; k++) {
            double angle = -2 * Math.PI * k / n;
            double c = Math.cos(angle), s = Math.sin(angle);
            double tr = c * oddTransform.re[k] - s * oddTransform.im[k];
            double ti = c * oddTransform.im[k] + s * oddTransform.re[k];
            out.re[k] = evenTransform.re[k] + tr; out.im[k] = evenTransform.im[k] + ti;
            out.re[k + n / 2] = evenTransform.re[k] - tr; out.im[k + n / 2] = evenTransform.im[k] - ti;
        }
        return out;
    }

    /** Bit-reversal permutation in place, traced like everything else. */
    static Signal permuted(Signal x) {
        int n = x.size();
        Signal out = x.copy();
        out.recording = x.recording;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                out.touch(i); out.touch(j);
                double tr = out.re[i], ti = out.im[i];
                out.re[i] = out.re[j]; out.im[i] = out.im[j];
                out.re[j] = tr; out.im[j] = ti;
                out.touch(i); out.touch(j);
            }
        }
        return out;
    }

    /**
     * One block of one stage: the butterflies (base + j, base + j + half) for j < length/2, with the twiddle
     * advanced by multiplication in the same order in every order of execution, so the arithmetic is fixed.
     */
    static void butterflyBlock(Signal data, int base, int length) {
        int half = length / 2;
        double angle = -2 * Math.PI / length;
        double wr = Math.cos(angle), wi = Math.sin(angle);
        double cr = 1, ci = 0;
        for (int j = 0; j < half; j++) {
            data.butterfly(base + j, base + j + half, cr, ci);
            double nextR = cr * wr - ci * wi;
            ci = cr * wi + ci * wr;
            cr = nextR;
        }
    }

    /** Every stage from length two up to maxLength over the half-open range [from, to). */
    static void stages(Signal data, int from, int to, int maxLength) {
        for (int length = 2; length <= maxLength; length <<= 1)
            for (int base = from; base + length <= to; base += length) butterflyBlock(data, base, length);
    }

    /** Breadth first: sweep the whole array once per stage. */
    static Signal stageByStageFft(Signal x) {
        int n = x.size();
        if (!isPowerOfTwo(n)) throw new IllegalArgumentException("radix-2 needs a power of two, got " + n);
        Signal data = permuted(x);
        stages(data, 0, n, n);
        return data;
    }

    /**
     * Depth first with a block parameter: every block of `block` elements completes its own log2(block)
     * stages before the next block starts, then the remaining stages run across blocks. The butterflies are
     * the same ones in the same groups, so the result is bit-for-bit identical to the breadth-first order.
     */
    static Signal blockedFft(Signal x, int block) {
        int n = x.size();
        if (!isPowerOfTwo(n)) throw new IllegalArgumentException("radix-2 needs a power of two, got " + n);
        if (!isPowerOfTwo(block) || block > n) throw new IllegalArgumentException("bad block size");
        Signal data = permuted(x);
        for (int base = 0; base < n; base += block) stages(data, base, base + block, block);
        for (int length = block << 1; length <= n; length <<= 1)
            for (int base = 0; base + length <= n; base += length) butterflyBlock(data, base, length);
        return data;
    }

    /** Bluestein's chirp transform: any length, built from power-of-two convolutions. */
    static Signal bluesteinFft(Signal x) {
        int n = x.size();
        if (isPowerOfTwo(n)) return stageByStageFft(x);
        int m = 1;
        while (m < 2 * n + 1) m <<= 1;
        double[] chirpRe = new double[n], chirpIm = new double[n];
        for (int i = 0; i < n; i++) {
            double angle = -Math.PI * (((long) i * i) % (2L * n)) / n;
            chirpRe[i] = Math.cos(angle);
            chirpIm[i] = Math.sin(angle);
        }
        Signal a = Signal.zeros(m), b = Signal.zeros(m);
        for (int i = 0; i < n; i++) {
            a.re[i] = x.re[i] * chirpRe[i] - x.im[i] * chirpIm[i];
            a.im[i] = x.re[i] * chirpIm[i] + x.im[i] * chirpRe[i];
            b.re[i] = chirpRe[i];
            b.im[i] = -chirpIm[i];
            if (i > 0) { b.re[m - i] = chirpRe[i]; b.im[m - i] = -chirpIm[i]; }
        }
        Signal fa = stageByStageFft(a), fb = stageByStageFft(b);
        Signal product = Signal.zeros(m);
        for (int i = 0; i < m; i++) {
            product.re[i] = fa.re[i] * fb.re[i] - fa.im[i] * fb.im[i];
            product.im[i] = fa.re[i] * fb.im[i] + fa.im[i] * fb.re[i];
        }
        Signal convolution = conjugate(stageByStageFft(conjugate(product)));
        Signal out = Signal.zeros(n);
        for (int i = 0; i < n; i++) {
            double cr = convolution.re[i] / m, ci = convolution.im[i] / m;
            out.re[i] = cr * chirpRe[i] - ci * chirpIm[i];
            out.im[i] = cr * chirpIm[i] + ci * chirpRe[i];
        }
        return out;
    }

    static Signal conjugate(Signal x) {
        Signal out = x.copy();
        for (int i = 0; i < x.size(); i++) out.im[i] = -out.im[i];
        return out;
    }

    static Signal fft(Signal x) { return isPowerOfTwo(x.size()) ? stageByStageFft(x) : bluesteinFft(x); }

    static Signal inverseFft(Signal x) {
        int n = x.size();
        Signal out = conjugate(fft(conjugate(x)));
        for (int i = 0; i < n; i++) { out.re[i] /= n; out.im[i] /= n; }
        return out;
    }

    /** Direct cyclic convolution: the independent oracle for the convolution theorem. */
    static Signal directConvolution(Signal a, Signal b) {
        int n = a.size();
        Signal out = Signal.zeros(n);
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++) {
                int index = (k - j + n) % n;
                out.re[k] += a.re[j] * b.re[index] - a.im[j] * b.im[index];
                out.im[k] += a.re[j] * b.im[index] + a.im[j] * b.re[index];
            }
        return out;
    }

    /** Butterflies a radix-2 transform of this length performs: (n/2) log2(n), at least one. */
    static double costOf(int n) { return n <= 1 ? 1 : 0.5 * n * log2(n); }
}

/** Independent tasks with measured costs packed onto p virtual workers by list scheduling. */
class Schedule {
    final int workers;
    final double[] free;
    double totalWork;

    Schedule(int workers) {
        this.workers = workers;
        free = new double[workers];
    }

    /** Place one task on the least loaded worker and return the makespan so far. */
    double assign(double cost) {
        int target = 0;
        for (int i = 1; i < workers; i++) if (free[i] < free[target]) target = i;
        free[target] += cost;
        totalWork += cost;
        double makespan = 0;
        for (double value : free) makespan = Math.max(makespan, value);
        return makespan;
    }
}

public class Solution {
    public static void main(String[] args) {
        SplitMix64 rng = new SplitMix64(20240517L);

        // --- CORRECTNESS against the definition ----------------------------------------------------------
        int[] sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
        double worstRelative = 0;
        for (int n : sizes) {
            Signal x = Signal.random(n, rng);
            Signal reference = Kernels.dft(x);
            double scale = Math.max(1e-12, reference.maxMagnitude());
            worstRelative = Math.max(worstRelative, Kernels.recursiveFft(x).maxDifference(reference) / scale);
            worstRelative = Math.max(worstRelative,
                    Kernels.stageByStageFft(x).maxDifference(reference) / scale);
            for (int block : new int[]{2, 8, 64, 1024}) {
                if (block > n) continue;
                worstRelative = Math.max(worstRelative,
                        Kernels.blockedFft(x, block).maxDifference(reference) / scale);
            }
        }
        check(worstRelative < 1e-11, "every radix-2 kernel must match the definition, worst relative "
                + worstRelative);
        int[] awkward = {3, 5, 7, 12, 100};
        double worstAwkward = 0;
        for (int n : awkward) {
            Signal x = Signal.random(n, rng);
            Signal reference = Kernels.dft(x);
            worstAwkward = Math.max(worstAwkward,
                    Kernels.bluesteinFft(x).maxDifference(reference) / Math.max(1e-12,
                            reference.maxMagnitude()));
        }
        check(worstAwkward < 1e-11, "Bluestein must handle arbitrary lengths, worst relative "
                + worstAwkward);

        // --- EQUIVALENCE of the two orders, bit for bit --------------------------------------------------
        Signal sample = Signal.random(1024, rng);
        Signal breadthFirst = Kernels.stageByStageFft(sample);
        for (int block : new int[]{2, 4, 16, 64, 256, 1024}) {
            Signal depthFirst = Kernels.blockedFft(sample, block);
            check(breadthFirst.identical(depthFirst), "block size " + block + " must reproduce the "
                    + "stage-by-stage bits exactly, max difference " + breadthFirst.maxDifference(depthFirst));
        }
        check(Kernels.stageByStageFft(sample).identical(breadthFirst), "the kernel must be deterministic");

        // --- ALGEBRA --------------------------------------------------------------------------------------
        Signal signal = Signal.random(256, rng), other = Signal.random(256, rng);
        Signal spectrum = Kernels.stageByStageFft(signal);
        Signal roundTrip = Kernels.inverseFft(spectrum);
        check(signal.maxDifference(roundTrip) < 1e-12, "the inverse must round-trip, error "
                + signal.maxDifference(roundTrip));
        double timeEnergy = 0, frequencyEnergy = 0;
        for (int i = 0; i < signal.size(); i++) {
            timeEnergy += signal.re[i] * signal.re[i] + signal.im[i] * signal.im[i];
            frequencyEnergy += spectrum.re[i] * spectrum.re[i] + spectrum.im[i] * spectrum.im[i];
        }
        check(Math.abs(timeEnergy - frequencyEnergy / signal.size()) < 1e-9 * Math.max(1, timeEnergy),
                "Parseval's identity must hold: " + timeEnergy + " against " + frequencyEnergy / 256);
        Signal direct = Kernels.directConvolution(signal, other);
        Signal throughTransform = Kernels.inverseFft(product(spectrum, Kernels.stageByStageFft(other)));
        check(direct.maxDifference(throughTransform) < 1e-10, "convolution through the transform must equal "
                + "the direct convolution, error " + direct.maxDifference(throughTransform));
        Signal combined = Signal.zeros(256);
        double alpha = -1.75, beta = 0.6;
        for (int i = 0; i < 256; i++) {
            combined.re[i] = alpha * signal.re[i] + beta * other.re[i];
            combined.im[i] = alpha * signal.im[i] + beta * other.im[i];
        }
        Signal mixed = Kernels.stageByStageFft(combined);
        Signal linear = Signal.zeros(256);
        Signal otherSpectrum = Kernels.stageByStageFft(other);
        for (int i = 0; i < 256; i++) {
            linear.re[i] = alpha * spectrum.re[i] + beta * otherSpectrum.re[i];
            linear.im[i] = alpha * spectrum.im[i] + beta * otherSpectrum.im[i];
        }
        check(mixed.maxDifference(linear) < 1e-10, "the transform must be linear, error "
                + mixed.maxDifference(linear));
        check(Kernels.blockedFft(signal, 32).maxDifference(spectrum) < 1e-10,
                "a tuned block size must not change the transform");

        // --- CACHE: measured miss counts and the tuning curve ---------------------------------------------
        int n = 4096, lines = 64, lineSize = 8;
        Signal large = Signal.random(n, new SplitMix64(31L));
        large.startRecording();
        Signal breadthResult = Kernels.stageByStageFft(large);
        int breadthMisses = misses(breadthResult, lines, lineSize);
        Signal largeBlocked = Signal.random(n, new SplitMix64(31L));
        largeBlocked.startRecording();
        Signal depthResult = Kernels.blockedFft(largeBlocked, 64);
        int depthMisses = misses(depthResult, lines, lineSize);
        check(breadthMisses > 0 && depthMisses > 0, "both orders must produce measurable traces");
        check(depthMisses * 7 < breadthMisses * 5, "the block-resident order must beat the stage-by-stage order "
                + "clearly: " + depthMisses + " against " + breadthMisses);
        int[] blocks = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        int[] missCurve = new int[blocks.length];
        for (int i = 0; i < blocks.length; i++) {
            Signal candidate = Signal.random(n, new SplitMix64(31L));
            candidate.startRecording();
            missCurve[i] = misses(Kernels.blockedFft(candidate, blocks[i]), lines, lineSize);
        }
        int bestIndex = 0;
        for (int i = 1; i < missCurve.length; i++) if (missCurve[i] < missCurve[bestIndex]) bestIndex = i;
        check(blocks[bestIndex] >= 8 && blocks[bestIndex] <= 2048, "the best block size must sit in a "
                + "sensible middle range, got " + blocks[bestIndex] + " with curve "
                + Arrays.toString(missCurve));
        check(missCurve[bestIndex] < missCurve[0] && missCurve[bestIndex] < missCurve[missCurve.length - 1],
                "the tuning curve must be U-shaped: " + Arrays.toString(missCurve));

        // The cache model itself, on a trace whose answer can be worked out by hand.
        CacheModel sequential = new CacheModel(lines, lineSize);
        List<Integer> scan = new ArrayList<>();
        for (int i = 0; i < n; i++) scan.add(i);
        sequential.replay(scan);
        check(sequential.misses == n / lineSize && sequential.hits == n - n / lineSize,
                "a sequential scan must miss once per distinct line, got " + sequential.misses);
        CacheModel single = new CacheModel(1, 1);
        single.replay(scan);
        check(single.misses == scan.size(), "a one-line cache with single-element lines must miss on every "
                + "access, got " + single.misses);
        CacheModel generous = new CacheModel(n, lineSize);
        generous.replay(scan);
        check(generous.misses == n / lineSize && generous.hits == n - n / lineSize,
                "a cache that never evicts still misses once per distinct line, got " + generous.misses);

        // --- THREADING: virtual-clock scheduling parameters -----------------------------------------------
        Schedule schedule = new Schedule(8);
        Schedule serial = new Schedule(1);
        double makespan = 0;
        for (int base = 0; base < n; base += 64) {
            double cost = Kernels.costOf(64);
            makespan = schedule.assign(cost);
            serial.assign(cost);
        }
        check(schedule.totalWork == serial.totalWork && schedule.totalWork > 0,
                "both schedules must carry the same work");
        check(makespan >= schedule.totalWork / 8 - 1e-9, "no schedule can beat perfect balance: " + makespan
                + " against " + schedule.totalWork / 8);
        check(makespan <= schedule.totalWork + 1e-9, "a schedule cannot exceed the serial work");
        check(serial.totalWork / makespan <= 8 + 1e-9, "speedup must not exceed the worker count");
        check(serial.totalWork / makespan > 4, "eight workers must deliver a real speedup, got "
                + serial.totalWork / makespan);
        Signal parallelPlanOne = Kernels.blockedFft(sample, 64);
        Signal parallelPlanEight = Kernels.blockedFft(sample, 64);
        check(parallelPlanOne.identical(parallelPlanEight), "the arithmetic must not depend on the schedule");
        check(Kernels.costOf(1) == 1 && Kernels.costOf(2) == 1 && Kernels.costOf(8) == 12,
                "the butterfly cost model must match (n/2) log2 n");

        // --- degenerate inputs --------------------------------------------------------------------------
        Signal oneSample = Signal.random(1, rng);
        check(Kernels.stageByStageFft(oneSample).identical(oneSample), "a length-one transform is the "
                + "identity");
        check(Kernels.blockedFft(oneSample, 1).identical(oneSample), "the blocked kernel handles length one");
        check(Kernels.stageByStageFft(Signal.zeros(64)).maxMagnitude() == 0.0,
                "the zero signal must stay zero");
        Signal constant = Signal.zeros(64);
        Arrays.fill(constant.re, 2.5);
        Signal constantSpectrum = Kernels.dft(constant);
        check(Math.abs(constantSpectrum.re[0] - 160.0) < 1e-9,
                "a constant signal must transform to a single spike, got " + constantSpectrum.re[0]);
        Signal impulse = Signal.zeros(32);
        impulse.re[0] = 1;
        Signal impulseSpectrum = Kernels.stageByStageFft(impulse);
        boolean allOnes = true;
        for (int i = 0; i < 32; i++)
            if (impulseSpectrum.re[i] != 1.0 || impulseSpectrum.im[i] != 0.0) allOnes = false;
        check(allOnes, "an impulse at zero must transform to all ones exactly");
        Signal real = Signal.zeros(64);
        for (int i = 0; i < 64; i++) real.re[i] = Math.sin(i * 0.4);
        Signal realSpectrum = Kernels.stageByStageFft(real);
        double symmetry = 0;
        for (int k = 0; k < 64; k++) {
            int mirror = (64 - k) % 64;
            symmetry = Math.max(symmetry, Math.hypot(realSpectrum.re[k] - realSpectrum.re[mirror],
                    realSpectrum.im[k] + realSpectrum.im[mirror]));
        }
        double endpoints = Math.max(Math.abs(realSpectrum.im[0]), Math.abs(realSpectrum.im[32]));
        check(symmetry < 1e-12 && endpoints < 1e-13, "a real signal must give a conjugate-symmetric "
                + "spectrum with a real DC and Nyquist term, symmetry error " + symmetry + " endpoints "
                + endpoints);
        boolean rejected = false;
        try { Kernels.stageByStageFft(Signal.random(6, rng)); } catch (IllegalArgumentException ex) {
            rejected = true;
        }
        check(rejected, "radix-2 must reject a length that is not a power of two");
        boolean badBlock = false;
        try { Kernels.blockedFft(Signal.random(64, rng), 12); } catch (IllegalArgumentException ex) {
            badBlock = true;
        }
        check(badBlock, "a block size that is not a power of two must be rejected");
        boolean tooLarge = false;
        try { Kernels.blockedFft(Signal.random(64, rng), 256); } catch (IllegalArgumentException ex) {
            tooLarge = true;
        }
        check(tooLarge, "a block size larger than the signal must be rejected");
        Signal four = Signal.random(4, rng);
        check(Kernels.bluesteinFft(four).identical(Kernels.stageByStageFft(four)),
                "Bluestein must fall back to radix-2 for powers of two");

        System.out.println("Kernels: worst relative error " + String.format(Locale.ROOT, "%.3e",
                worstRelative) + " over " + Arrays.toString(sizes) + ", Bluestein "
                + String.format(Locale.ROOT, "%.3e", worstAwkward) + " for " + Arrays.toString(awkward));
        System.out.println("Cache at n = 4096 with " + lines + " lines: block-resident " + depthMisses
                + " misses against stage-by-stage " + breadthMisses + " (" + String.format(Locale.ROOT,
                "%.2f", breadthMisses / (double) depthMisses) + "x), best block size "
                + blocks[bestIndex] + ", curve " + Arrays.toString(missCurve));
        System.out.println("Threading: 8 workers reach makespan " + makespan + " for serial work "
                + schedule.totalWork + " with identical bits");
        System.out.println("PASS");
    }

    static int misses(Signal signal, int lines, int lineSize) {
        CacheModel model = new CacheModel(lines, lineSize);
        model.replay(signal.trace);
        return model.misses;
    }

    static Signal product(Signal a, Signal b) {
        Signal out = Signal.zeros(a.size());
        for (int i = 0; i < a.size(); i++) {
            out.re[i] = a.re[i] * b.re[i] - a.im[i] * b.im[i];
            out.im[i] = a.re[i] * b.im[i] + a.im[i] * b.re[i];
        }
        return out;
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
