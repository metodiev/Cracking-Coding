/** Task 693 - Implement erasure coding (Reed-Solomon) encoder/decoder small GF(2^8). (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
 *
 * Reed-Solomon over GF(2^8) with the primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11D) and
 * the generator alpha = 2. A codeword is a polynomial c(x) = sum c_i x^i of degree < n whose
 * roots include alpha^1 .. alpha^(n-k), so its syndromes S_j = c(alpha^j) vanish. Encoding is
 * systematic: the message sits at the top of the word, and the parity is the remainder of
 * m(x) * x^(n-k) divided by the generator polynomial g(x) = prod (x - alpha^j).
 *
 * Decoding:
 *   * erasures (known missing positions) are solved by Gaussian elimination over GF(2^8) using
 *     the generator matrix restricted to any k surviving positions - an underdetermined system
 *     (more than n-k erasures) is reported as a failure;
 *   * unknown errors are located with the Berlekamp-Massey recurrence on the syndromes followed
 *     by a Chien search over the inverse locators; a locator whose root count disagrees with its
 *     degree, or a correction that does not land within that many positions of the received
 *     word, is rejected. Once the positions are known the magnitudes come from the same linear
 *     solve used for erasures rather than from Forney's formula - equivalent and easier to check.
 *
 * The code is MDS: its minimum distance is n-k+1, so up to n-k erasures and up to (n-k)/2
 * unknown errors are always correctable. Beyond that, Reed-Solomon can miscorrect silently, so
 * the checks only require that the decoder never returns an invalid codeword - the specific
 * miscorrection observed with n-k+1 errors is reported as the honest outcome.
 */
import java.util.*;

/** Arithmetic in GF(2^8) with primitive polynomial 0x11D. */
class Gf256 {
    static final int[] EXP = new int[512];
    static final int[] LOG = new int[256];

    static {
        int value = 1;
        for (int i = 0; i < 255; i++) {
            EXP[i] = value;
            LOG[value] = i;
            value <<= 1;
            if ((value & 0x100) != 0) value ^= 0x11D;
        }
        for (int i = 255; i < 512; i++) EXP[i] = EXP[i - 255];
    }

    static int mul(int a, int b) {
        return (a == 0 || b == 0) ? 0 : EXP[LOG[a] + LOG[b]];
    }

    static int div(int a, int b) {
        if (b == 0) throw new ArithmeticException("division by zero in GF(256)");
        return a == 0 ? 0 : EXP[LOG[a] + 255 - LOG[b]];
    }

    static int inverse(int a) {
        if (a == 0) throw new ArithmeticException("no inverse for zero");
        return EXP[255 - LOG[a]];
    }

    static int power(int a, int exponent) {
        if (a == 0) return 0;
        int e = exponent % 255;
        if (e < 0) e += 255;
        return EXP[LOG[a] * e % 255];
    }
}

/** Systematic Reed-Solomon code: k message symbols, n-k parity symbols, block length n. */
class ReedSolomon {
    final int k;
    final int n;
    final int parity;
    final int[] generator;
    final int[][] matrix;

    ReedSolomon(int k, int n) {
        if (n > 255 || k < 1 || k >= n) throw new IllegalArgumentException("bad code parameters");
        this.k = k;
        this.n = n;
        this.parity = n - k;
        int[] g = {1};
        for (int j = 1; j <= parity; j++) {
            g = multiply(g, new int[]{Gf256.EXP[j], 1});
        }
        this.generator = g;
        this.matrix = new int[n][k];
        for (int i = 0; i < k; i++) {
            int[] unit = new int[k];
            unit[i] = 1;
            int[] codeword = encode(unit);
            for (int j = 0; j < n; j++) matrix[j][i] = codeword[j];
        }
    }

    static int[] multiply(int[] a, int[] b) {
        int[] out = new int[a.length + b.length - 1];
        for (int i = 0; i < a.length; i++) {
            if (a[i] == 0) continue;
            for (int j = 0; j < b.length; j++) out[i + j] ^= Gf256.mul(a[i], b[j]);
        }
        return out;
    }

    /** Evaluates a polynomial given by coefficients indexed by power. */
    static int evaluate(int[] coefficients, int x) {
        int value = 0;
        for (int i = coefficients.length - 1; i >= 0; i--) value = Gf256.mul(value, x) ^ coefficients[i];
        return value;
    }

    int[] encode(int[] message) {
        if (message.length != k) throw new IllegalArgumentException("message must have k symbols");
        int[] work = new int[n];
        for (int i = 0; i < k; i++) work[parity + i] = message[i];
        for (int degree = n - 1; degree >= parity; degree--) {
            int coefficient = work[degree];
            if (coefficient == 0) continue;
            for (int i = 0; i <= parity; i++) {
                work[degree - parity + i] ^= Gf256.mul(coefficient, generator[i]);
            }
        }
        int[] codeword = new int[n];
        System.arraycopy(work, 0, codeword, 0, parity);
        for (int i = 0; i < k; i++) codeword[parity + i] = message[i];
        return codeword;
    }

    int[] extractMessage(int[] codeword) {
        return Arrays.copyOfRange(codeword, parity, n);
    }

    /** S_j = c(alpha^j) for j = 1..n-k; all zero exactly when the word is a codeword. */
    int[] syndromes(int[] word) {
        int[] out = new int[parity];
        for (int j = 1; j <= parity; j++) {
            int alpha = Gf256.EXP[j];
            int value = 0;
            for (int i = n - 1; i >= 0; i--) value = Gf256.mul(value, alpha) ^ word[i];
            out[j - 1] = value;
        }
        return out;
    }

    boolean isCodeword(int[] word) {
        for (int value : syndromes(word)) if (value != 0) return false;
        return true;
    }

    /** Berlekamp-Massey on S_1..S_2t; returns the error locator, index = power of x. */
    static int[] errorLocator(int[] syndrome) {
        int[] locator = new int[syndrome.length + 1];
        int[] previous = new int[syndrome.length + 1];
        locator[0] = 1;
        previous[0] = 1;
        int degree = 0;
        int shift = 1;
        int scale = 1;
        for (int i = 0; i < syndrome.length; i++) {
            int discrepancy = syndrome[i];
            for (int j = 1; j <= degree; j++) {
                discrepancy ^= Gf256.mul(locator[j], syndrome[i - j]);
            }
            if (discrepancy == 0) {
                shift++;
                continue;
            }
            int[] saved = locator.clone();
            int coefficient = Gf256.div(discrepancy, scale);
            for (int j = 0; j + shift < locator.length; j++) {
                if (previous[j] != 0) locator[j + shift] ^= Gf256.mul(coefficient, previous[j]);
            }
            if (2 * degree <= i) {
                degree = i + 1 - degree;
                previous = saved;
                scale = discrepancy;
                shift = 1;
            } else {
                shift++;
            }
        }
        int[] trimmed = new int[degree + 1];
        System.arraycopy(locator, 0, trimmed, 0, degree + 1);
        return trimmed;
    }

    /** Gaussian elimination over GF(2^8); the last column is the right-hand side. */
    static int[] solve(int[][] augmented) {
        int size = augmented.length;
        for (int column = 0; column < size; column++) {
            int pivot = -1;
            for (int row = column; row < size; row++) {
                if (augmented[row][column] != 0) {
                    pivot = row;
                    break;
                }
            }
            if (pivot < 0) return null;
            int[] swap = augmented[column];
            augmented[column] = augmented[pivot];
            augmented[pivot] = swap;
            int inverse = Gf256.inverse(augmented[column][column]);
            for (int j = column; j <= size; j++) augmented[column][j] = Gf256.mul(augmented[column][j], inverse);
            for (int row = 0; row < size; row++) {
                if (row == column || augmented[row][column] == 0) continue;
                int factor = augmented[row][column];
                for (int j = column; j <= size; j++) {
                    augmented[row][j] ^= Gf256.mul(factor, augmented[column][j]);
                }
            }
        }
        int[] solution = new int[size];
        for (int i = 0; i < size; i++) solution[i] = augmented[i][size];
        return solution;
    }

    /** Solves for the message using any k intact positions; null when too much is missing. */
    int[] decodeErasures(int[] received, boolean[] erased) {
        int[] indices = new int[k];
        int found = 0;
        for (int j = 0; j < n && found < k; j++) {
            if (!erased[j]) indices[found++] = j;
        }
        if (found < k) return null;
        int[][] augmented = new int[k][k + 1];
        for (int row = 0; row < k; row++) {
            for (int column = 0; column < k; column++) {
                augmented[row][column] = matrix[indices[row]][column];
            }
            augmented[row][k] = received[indices[row]];
        }
        int[] message = solve(augmented);
        if (message == null) return null;
        for (int j = 0; j < n; j++) {
            if (erased[j]) continue;
            if (evaluateRow(message, j) != received[j]) return null;
        }
        return message;
    }

    /** The j-th symbol produced by encoding `message`. */
    int evaluateRow(int[] message, int index) {
        int value = 0;
        for (int i = 0; i < k; i++) value ^= Gf256.mul(matrix[index][i], message[i]);
        return value;
    }

    /** Locates unknown errors, then solves for the message as if they had been erasures. */
    int[] decodeErrors(int[] received) {
        int[] syndrome = syndromes(received);
        boolean clean = true;
        for (int value : syndrome) if (value != 0) clean = false;
        if (clean) return extractMessage(received);
        int[] locator = errorLocator(syndrome);
        int errors = locator.length - 1;
        if (errors <= 0 || errors > parity / 2) return null;
        boolean[] erased = new boolean[n];
        int roots = 0;
        for (int j = 0; j < n; j++) {
            int z = Gf256.inverse(Gf256.EXP[j]);
            if (evaluate(locator, z) == 0) {
                erased[j] = true;
                roots++;
            }
        }
        if (roots != errors) return null;
        int[] message = decodeErasures(received, erased);
        if (message == null) return null;
        int[] codeword = encode(message);
        int distance = 0;
        for (int j = 0; j < n; j++) if (codeword[j] != received[j]) distance++;
        return distance == errors ? message : null;
    }
}

public class Solution {
    private static long prngState = 0xC0FFEE0DL;

    static long nextLong() {
        prngState += 0x9E3779B97F4A7C15L;
        long z = prngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), (long) bound);
    }

    static int[] randomMessage(int k) {
        int[] message = new int[k];
        for (int i = 0; i < k; i++) message[i] = 1 + nextInt(255);
        return message;
    }

    static int distance(int[] a, int[] b) {
        int count = 0;
        for (int i = 0; i < a.length; i++) if (a[i] != b[i]) count++;
        return count;
    }

    /** Brute force oracle for a k = 2 code: encode every message and match the survivors. */
    static int[] bruteForce(int[] received, boolean[] known, ReedSolomon code) {
        int[] match = null;
        for (int first = 0; first < 256; first++) {
            for (int second = 0; second < 256; second++) {
                int[] candidate = code.encode(new int[]{first, second});
                boolean fits = true;
                for (int j = 0; j < code.n; j++) {
                    if (known[j] && candidate[j] != received[j]) fits = false;
                }
                if (!fits) continue;
                if (match != null) return null; // ambiguous
                match = new int[]{first, second};
            }
        }
        return match;
    }

    public static void main(String[] args) {
        // 1. Field arithmetic: table round trip, inverses and distributivity.
        boolean roundTrip = true;
        for (int a = 1; a < 256; a++) {
            if (Gf256.EXP[Gf256.LOG[a]] != a) roundTrip = false;
            if (Gf256.mul(a, Gf256.inverse(a)) != 1) roundTrip = false;
        }
        check(roundTrip, "exp/log round trip and a * a^-1 = 1 for every non-zero element");
        check(Gf256.mul(0, 123) == 0 && Gf256.mul(1, 123) == 123 && Gf256.div(123, 123) == 1,
                "multiplicative identity and zero behave");
        boolean distributive = true;
        for (int trial = 0; trial < 200; trial++) {
            int a = nextInt(256);
            int b = nextInt(256);
            int c = nextInt(256);
            if (Gf256.mul(a, b ^ c) != (Gf256.mul(a, b) ^ Gf256.mul(a, c))) distributive = false;
        }
        check(distributive, "multiplication distributes over XOR");
        check(Gf256.power(2, 8) == 0x1D && Gf256.power(2, 255) == 1,
                "alpha satisfies the primitive polynomial: " + Gf256.power(2, 8));

        // 2. The generator polynomial vanishes at alpha^1 .. alpha^(n-k).
        ReedSolomon code = new ReedSolomon(6, 12);
        boolean roots = true;
        for (int j = 1; j <= code.parity; j++) {
            if (ReedSolomon.evaluate(code.generator, Gf256.EXP[j]) != 0) roots = false;
        }
        check(roots, "g(alpha^j) = 0 for every designed root");
        check(code.generator[code.parity] == 1 && code.generator.length == code.parity + 1,
                "the generator is monic of degree n-k");

        // 3. Encoding is systematic and always lands on a codeword.
        int[] message = randomMessage(6);
        int[] codeword = code.encode(message);
        check(codeword.length == 12, "the codeword has n symbols");
        check(Arrays.equals(code.extractMessage(codeword), message),
                "the message symbols are stored verbatim (systematic code)");
        check(code.isCodeword(codeword), "the syndromes of a codeword are all zero");
        check(code.syndromes(codeword)[0] == 0 && code.syndromes(codeword)[5] == 0,
                "each designed syndrome is zero");

        // 4. Minimum distance n-k+1: distinct codewords differ in at least seven positions.
        int minimum = Integer.MAX_VALUE;
        for (int trial = 0; trial < 40; trial++) {
            minimum = Math.min(minimum, distance(code.encode(randomMessage(6)), code.encode(randomMessage(6))));
        }
        check(minimum >= code.parity + 1, "MDS distance bound holds for sampled pairs: " + minimum);

        // 5. Erasure decoding up to the bound, failure one past it.
        int erasureSuccesses = 0;
        for (int trial = 0; trial < 30; trial++) {
            int[] original = randomMessage(6);
            int[] word = code.encode(original);
            boolean[] erased = new boolean[12];
            int erasedCount = 0;
            while (erasedCount < code.parity) {
                int index = nextInt(12);
                if (!erased[index]) {
                    erased[index] = true;
                    word[index] = 0;
                    erasedCount++;
                }
            }
            int[] decoded = code.decodeErasures(word, erased);
            if (decoded != null && Arrays.equals(decoded, original)) erasureSuccesses++;
        }
        check(erasureSuccesses == 30, "every pattern of n-k erasures is repaired: " + erasureSuccesses);
        int[] word = code.encode(randomMessage(6));
        boolean[] erased = new boolean[12];
        for (int i = 0; i < code.parity + 1; i++) {
            erased[i] = true;
            word[i] = 0;
        }
        check(code.decodeErasures(word, erased) == null, "n-k+1 erasures are unrecoverable");
        boolean[] none = new boolean[12];
        int[] untouched = code.encode(message);
        check(Arrays.equals(code.decodeErasures(untouched, none), message),
                "with no erasures the message comes back unchanged");

        // 6. Error decoding up to t = (n-k)/2, including the extreme positions.
        int errorSuccesses = 0;
        for (int trial = 0; trial < 30; trial++) {
            int[] original = randomMessage(6);
            int[] corrupted = code.encode(original);
            boolean[] hit = new boolean[12];
            int placed = 0;
            while (placed < 3) {
                int index = nextInt(12);
                if (hit[index]) continue;
                hit[index] = true;
                corrupted[index] ^= 1 + nextInt(255);
                placed++;
            }
            int[] decoded = code.decodeErrors(corrupted);
            if (decoded != null && Arrays.equals(decoded, original)) errorSuccesses++;
        }
        check(errorSuccesses == 30, "three unknown errors are always corrected: " + errorSuccesses);
        int[] edge = code.encode(message);
        edge[0] ^= 0x5A;
        edge[11] ^= 0x33;
        check(Arrays.equals(code.decodeErrors(edge), message),
                "errors at both ends of the word are corrected");
        int[] single = code.encode(message);
        single[7] ^= 0xFF;
        check(Arrays.equals(code.decodeErrors(single), message), "a single error is corrected");

        // 7. Beyond capacity: never an invalid codeword, and the original is not returned.
        int[] overloaded = code.encode(message);
        int[] positions = {1, 3, 5, 9};
        for (int position : positions) overloaded[position] ^= 1 + nextInt(255);
        int[] decoded = code.decodeErrors(overloaded);
        check(decoded == null || !Arrays.equals(decoded, message),
                "four errors are past the correction capacity: the original is not returned");
        if (decoded != null) {
            int[] candidate = code.encode(decoded);
            check(code.isCodeword(candidate) && distance(candidate, overloaded) <= 3,
                    "a returned word is still a valid codeword near the received one");
        }
        check(!code.isCodeword(overloaded), "the corrupted word is not a codeword");

        // 8. Brute force cross-check of the k = 2 case against exhaustive encoding.
        ReedSolomon tiny = new ReedSolomon(2, 4);
        int[] pair = {17, 200};
        int[] tinyWord = tiny.encode(pair);
        boolean[] holes = {false, true, true, false};
        tinyWord[1] = 0;
        tinyWord[2] = 0;
        check(Arrays.equals(tiny.decodeErasures(tinyWord, holes), pair), "small code: erasures");
        check(Arrays.equals(bruteForce(tinyWord, new boolean[]{true, false, false, true}, tiny), pair),
                "small code: exhaustive search agrees");
        int[] tinyCorrupt = tiny.encode(pair);
        tinyCorrupt[3] ^= 0x2F;
        int[] tinyDecoded = tiny.decodeErrors(tinyCorrupt);
        check(Arrays.equals(tinyDecoded, pair), "small code: one error corrected");
        int[] tinyIntact = tiny.encode(pair);
        check(Arrays.equals(bruteForce(tinyIntact, new boolean[]{true, true, false, false}, tiny), pair),
                "small code: two surviving symbols identify the message uniquely");
        check(bruteForce(tinyIntact, new boolean[]{true, false, false, false}, tiny) == null,
                "small code: a single surviving symbol is not enough for k = 2");

        // 9. A larger block: 7 errors in a (30, 16) code.
        ReedSolomon large = new ReedSolomon(16, 30);
        int[] bigMessage = randomMessage(16);
        int[] bigWord = large.encode(bigMessage);
        boolean[] bigHit = new boolean[30];
        int placed = 0;
        while (placed < 7) {
            int index = nextInt(30);
            if (bigHit[index]) continue;
            bigHit[index] = true;
            bigWord[index] ^= 1 + nextInt(255);
            placed++;
        }
        check(Arrays.equals(large.decodeErrors(bigWord), bigMessage),
                "seven errors in a (30,16) code are corrected");

        // 10. Parameter validation and determinism.
        boolean rejected = false;
        try {
            new ReedSolomon(300, 256);
        } catch (IllegalArgumentException e) {
            rejected = true;
        }
        check(rejected, "blocks longer than 255 symbols are rejected");
        rejected = false;
        try {
            new ReedSolomon(8, 8);
        } catch (IllegalArgumentException e) {
            rejected = true;
        }
        check(rejected, "a code without parity symbols is rejected");
        prngState = 0x1234L;
        int[] firstRun = randomMessage(6);
        prngState = 0x1234L;
        int[] secondRun = randomMessage(6);
        check(Arrays.equals(firstRun, secondRun)
                && Arrays.equals(code.encode(firstRun), code.encode(secondRun)),
                "the PRNG and the encoder are deterministic");

        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
