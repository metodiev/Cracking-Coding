/** Task 944 - Implement high-precision numeric library with arbitrary precision rational arithmetic.
 *  (Tier: Master, CTCI Ch 9 - System Design and Scalability)
 *
 *  A self-contained exact numeric library. BigNat is an arbitrary-precision non-negative integer stored as
 *  base-10^4 limbs (schoolbook add/sub/mul, binary-searched long division, Stein binary GCD); Rational is a
 *  sign plus a fully reduced numerator/denominator with cross-multiplication comparison, exact arithmetic,
 *  integer powers and a long-division decimal expansion. On top of them the file computes pi with Machin's
 *  arctan series in fixed point, and pins down sqrt(2) two independent ways - Newton's iteration on
 *  Rationals and the Pell convergents of the continued fraction, which must land on the same fractions.
 *  Simplified: base-10^4 schoolbook arithmetic (no Karatsuba/FFT) and only a few dozen digits of pi.
 *  Verified: results stay reduced (checked by trial division), the order is total, p^2 - 2q^2 = +-1 holds
 *  exactly for 100 convergents and equals Newton's iterates, pi matches the published digits through two
 *  different formulas, and division/power/decimal identities hold (2^100, 10^60, 22/7 = 3.142857142857).
 */
import java.util.*;

public class Solution {

    static void check(boolean ok, String label) { if (!ok) throw new AssertionError("FAILED: " + label); }
    /** Non-negative arbitrary-precision integer: base-10^4 limbs, least significant first, no zero limbs. */
    static final class BigNat implements Comparable<BigNat> {
        static final int BASE = 10_000;
        static final BigNat ZERO = new BigNat(new int[0]), ONE = BigNat.of(1), TWO = BigNat.of(2);
        final int[] limbs;

        BigNat(int[] limbs) { this.limbs = limbs; }
        static BigNat of(long value) {
            int[] out = new int[10];
            int n = 0;
            while (value > 0) {
                out[n++] = (int) (value % BASE);
                value /= BASE;
            }
            return new BigNat(Arrays.copyOf(out, n));
        }
        /** Normalises raw limbs into base-10^4 form and strips leading zeros. */
        static BigNat of(int[] raw) {
            int carry = 0;
            for (int i = 0; i < raw.length; i++) {
                int total = raw[i] + carry;
                raw[i] = total % BASE;
                carry = total / BASE;
            }
            int n = raw.length;
            while (n > 0 && raw[n - 1] == 0) n--;
            return new BigNat(Arrays.copyOf(raw, n));
        }
        boolean isZero() { return limbs.length == 0; }
        boolean isOne() { return limbs.length == 1 && limbs[0] == 1; }
        boolean isEven() { return limbs.length == 0 || (limbs[0] & 1) == 0; }
        int toInt() {
            if (limbs.length > 1) throw new IllegalStateException("limb overflow: " + this);
            return limbs.length == 0 ? 0 : limbs[0];
        }
        public int compareTo(BigNat o) {
            if (limbs.length != o.limbs.length) return limbs.length < o.limbs.length ? -1 : 1;
            for (int i = limbs.length - 1; i >= 0; i--) if (limbs[i] != o.limbs[i]) return limbs[i] < o.limbs[i] ? -1 : 1;
            return 0;
        }
        public boolean equals(Object o) { return o instanceof BigNat && compareTo((BigNat) o) == 0; }
        public int hashCode() { return Arrays.hashCode(limbs); }
        BigNat add(BigNat o) {
            int[] out = new int[Math.max(limbs.length, o.limbs.length) + 1];
            for (int i = 0; i < out.length; i++) out[i] = (i < limbs.length ? limbs[i] : 0) + (i < o.limbs.length ? o.limbs[i] : 0);
            return of(out);
        }
        /** this - o, requires this >= o. */
        BigNat sub(BigNat o) {
            int[] out = new int[limbs.length + 1];
            for (int i = 0; i < limbs.length; i++) out[i] = limbs[i] - (i < o.limbs.length ? o.limbs[i] : 0);
            for (int i = 0; i < out.length - 1; i++) if (out[i] < 0) { out[i] += BASE; out[i + 1]--; }
            return of(out);
        }
        BigNat mul(BigNat o) {
            int[] out = new int[limbs.length + o.limbs.length + 1];
            for (int i = 0; i < limbs.length; i++) {
                long carry = 0;
                for (int j = 0; j < o.limbs.length; j++) {
                    long current = out[i + j] + (long) limbs[i] * o.limbs[j] + carry;
                    out[i + j] = (int) (current % BASE);
                    carry = current / BASE;
                }
                out[i + o.limbs.length] += (int) carry;
            }
            return of(out);
        }
        BigNat mulSmall(int factor) { return mul(of(factor)); }
        BigNat shifted(int places) {
            int[] out = new int[limbs.length + places];
            System.arraycopy(limbs, 0, out, places, limbs.length);
            return new BigNat(out);
        }
        /** Long division with a binary-searched quotient digit; returns {quotient, remainder}. */
        BigNat[] divMod(BigNat divisor) {
            if (divisor.isZero()) throw new IllegalArgumentException("division by zero");
            if (compareTo(divisor) < 0) return new BigNat[] {ZERO, this};
            int[] quotient = new int[limbs.length];
            BigNat remainder = ZERO;
            for (int i = limbs.length - 1; i >= 0; i--) {
                remainder = remainder.shifted(1).add(of(limbs[i]));
                int lo = 0, hi = BASE - 1, digit = 0;
                while (lo <= hi) {
                    int mid = (lo + hi) >>> 1;
                    if (divisor.mulSmall(mid).compareTo(remainder) <= 0) { digit = mid; lo = mid + 1; } else hi = mid - 1;
                }
                quotient[i] = digit;
                if (digit != 0) remainder = remainder.sub(divisor.mulSmall(digit));
            }
            return new BigNat[] {of(quotient), remainder};
        }
        /** Stein's binary GCD: only subtractions and halvings, so no division is needed. */
        static BigNat gcd(BigNat a, BigNat b) {
            if (a.isZero()) return b;
            if (b.isZero()) return a;
            int shift = 0;
            while (a.isEven() && b.isEven()) { a = a.divMod(TWO)[0]; b = b.divMod(TWO)[0]; shift++; }
            while (a.isEven()) a = a.divMod(TWO)[0];
            while (!b.isZero()) {
                while (b.isEven()) b = b.divMod(TWO)[0];
                if (a.compareTo(b) > 0) { BigNat t = a; a = b; b = t; }
                b = b.sub(a);
            }
            for (int i = 0; i < shift; i++) a = a.add(a);
            return a;
        }
        public String toString() {
            if (isZero()) return "0";
            StringBuilder sb = new StringBuilder().append(limbs[limbs.length - 1]);
            for (int i = limbs.length - 2; i >= 0; i--) {
                String cell = Integer.toString(limbs[i]);
                sb.append("0".repeat(4 - cell.length())).append(cell);
            }
            return sb.toString();
        }
    }
    /** Exact rational number: always reduced, denominator positive, zero stored canonically as 0/1. */
    static final class Rational implements Comparable<Rational> {
        static final Rational ZERO = new Rational(0, BigNat.ZERO, BigNat.ONE), TWO = Rational.of(2, 1);
        final int sign;
        final BigNat num, den;

        private Rational(int sign, BigNat num, BigNat den) { this.sign = sign; this.num = num; this.den = den; }
        static Rational of(long numerator, long denominator) {
            if (denominator == 0) throw new IllegalArgumentException("zero denominator");
            return of(BigNat.of(Math.abs(numerator)), BigNat.of(Math.abs(denominator)),
                    Long.signum(numerator) * Long.signum(denominator));
        }
        static Rational of(BigNat numerator, BigNat denominator, int sign) {
            if (denominator.isZero()) throw new IllegalArgumentException("zero denominator");
            if (numerator.isZero()) return ZERO;
            BigNat g = BigNat.gcd(numerator, denominator);
            return new Rational(sign, numerator.divMod(g)[0], denominator.divMod(g)[0]);
        }
        Rational negate() { return sign == 0 ? this : new Rational(-sign, num, den); }
        Rational abs() { return sign < 0 ? negate() : this; }
        boolean isZero() { return sign == 0; }
        Rational add(Rational o) {
            if (sign == 0) return o;
            if (o.sign == 0) return this;
            BigNat left = num.mul(o.den), right = o.num.mul(den);
            if (sign == o.sign) return of(left.add(right), den.mul(o.den), sign);
            int comparison = left.compareTo(right);
            if (comparison == 0) return ZERO;
            return of(comparison > 0 ? left.sub(right) : right.sub(left), den.mul(o.den),
                    comparison > 0 ? sign : o.sign);
        }
        Rational sub(Rational o) { return add(o.negate()); }
        Rational mul(Rational o) {
            return sign == 0 || o.sign == 0 ? ZERO : of(num.mul(o.num), den.mul(o.den), sign * o.sign);
        }
        Rational div(Rational o) {
            if (o.sign == 0) throw new IllegalArgumentException("division by zero");
            return sign == 0 ? ZERO : of(num.mul(o.den), den.mul(o.num), sign * o.sign);
        }
        Rational pow(int exponent) {
            if (exponent < 0) throw new IllegalArgumentException("negative exponent");
            if (sign == 0) return ZERO;
            BigNat n = BigNat.ONE, d = BigNat.ONE;
            for (int i = 0; i < exponent; i++) { n = n.mul(num); d = d.mul(den); }
            return of(n, d, sign < 0 && exponent % 2 == 1 ? -1 : 1);
        }
        public int compareTo(Rational o) {
            if (sign != o.sign) return sign < o.sign ? -1 : 1;
            if (sign == 0) return 0;
            int cross = num.mul(o.den).compareTo(o.num.mul(den));
            return sign < 0 ? -cross : cross;
        }
        public boolean equals(Object o) {
            return o instanceof Rational r && r.sign == sign && r.num.equals(num) && r.den.equals(den);
        }
        public int hashCode() { return Objects.hash(sign, num, den); }
        public String toString() {
            return (sign < 0 ? "-" : "") + num + (den.isOne() ? "" : "/" + den);
        }
        /** Long-division decimal expansion, stopping early on exact division. */
        String toDecimal(int digits) {
            BigNat[] head = num.divMod(den);
            StringBuilder sb = new StringBuilder(sign < 0 ? "-" : "").append(head[0]);
            BigNat remainder = head[1];
            if (remainder.isZero() || digits == 0) return sb.toString();
            sb.append('.');
            for (int i = 0; i < digits && !remainder.isZero(); i++) {
                BigNat[] step = remainder.mulSmall(10).divMod(den);
                sb.append((char) ('0' + step[0].toInt()));
                remainder = step[1];
            }
            return sb.toString();
        }
    }
    /** Trial-division check that a fraction is reduced - an oracle independent of Stein's GCD. */
    static boolean reducedByTrial(Rational r) {
        for (int p = 2; p <= 1_000; p++) {
            if (!r.num.divMod(BigNat.of(p))[1].isZero()) continue;
            if (r.den.divMod(BigNat.of(p))[1].isZero()) return false;
        }
        return true;
    }
    static BigNat powerOfTen(int digits) {
        BigNat value = BigNat.ONE;
        for (int i = 0; i < digits; i++) value = value.mulSmall(10);
        return value;
    }
    /** arctan(1/x) in fixed point scaled by 10^guard, alternating series with halved powers. */
    static BigNat arctanScaled(int x, int guard) {
        BigNat square = BigNat.of((long) x * x), even = BigNat.ZERO, odd = BigNat.ZERO;
        BigNat power = powerOfTen(guard).divMod(BigNat.of(x))[0];
        for (int k = 0; k < Math.max(3 * guard / 2, 8); k++) {
            BigNat term = power.divMod(BigNat.of(2 * k + 1))[0];
            if ((k & 1) == 0) even = even.add(term); else odd = odd.add(term);
            power = power.divMod(square)[0];
        }
        return even.sub(odd);
    }
    /** pi = 16 arctan(1/5) - 4 arctan(1/239) (Machin), returned as `digits` decimal digits. */
    static String machinPi(int digits) {
        int guard = digits + 12;
        BigNat pi = arctanScaled(5, guard).mulSmall(16).sub(arctanScaled(239, guard).mulSmall(4));
        return pi.divMod(powerOfTen(guard - digits + 1))[0].toString();
    }
    /** Independent formula: pi = 4 (arctan(1/2) + arctan(1/3)). */
    static String piThroughHalfAndThird(int digits) {
        int guard = digits + 12;
        BigNat pi = arctanScaled(2, guard).add(arctanScaled(3, guard)).mulSmall(4);
        return pi.divMod(powerOfTen(guard - digits + 1))[0].toString();
    }
    /** sqrt(2) by Newton's iteration x -> (x + 2/x)/2 starting from 1. */
    static Rational[] newtonSqrtTwo(int iterations) {
        Rational[] iterates = new Rational[iterations + 1];
        iterates[0] = Rational.of(1, 1);
        for (int i = 1; i <= iterations; i++) {
            iterates[i] = iterates[i - 1].add(Rational.TWO.div(iterates[i - 1])).div(Rational.of(2, 1));
        }
        return iterates;
    }
    public static void main(String[] args) {
        Rational third = Rational.of(1, 3), sixth = Rational.of(1, 6);
        check(third.add(sixth).equals(Rational.of(1, 2)) && third.add(third).equals(Rational.of(2, 3))
                        && Rational.of(2, 4).equals(Rational.of(1, 2))
                        && Rational.of(3, 4).mul(Rational.of(4, 3)).equals(Rational.of(1, 1))
                        && Rational.of(1, 3).div(Rational.of(1, 6)).equals(Rational.of(2, 1))
                        && third.sub(third).isZero() && Rational.of(1, -2).equals(Rational.of(-1, 2))
                        && third.abs().equals(third) && Rational.of(-3, 4).abs().equals(Rational.of(3, 4)),
                "exact arithmetic must be exact and normalise signs: " + third.add(sixth));

        List<Rational> mixed = new ArrayList<>(List.of(Rational.of(7, 8), Rational.of(-5, 3), Rational.of(1, 100),
                Rational.ZERO, Rational.of(22, 7), Rational.of(-1, 1000)));
        Rational product = Rational.of(1, 1);
        for (Rational r : mixed) {
            if (!r.isZero()) product = product.mul(r);
        }
        Rational quotient = product.div(Rational.of(7, 8));
        check(mixed.stream().allMatch(Solution::reducedByTrial) && reducedByTrial(product)
                        && !quotient.den.isOne() && quotient.mul(Rational.of(7, 8)).equals(product),
                "every result must stay reduced: product " + product + ", quotient " + quotient);

        List<Rational> sorted = new ArrayList<>(mixed);
        Collections.sort(sorted);
        boolean ordered = true;
        for (int i = 1; i < sorted.size(); i++) ordered &= sorted.get(i - 1).compareTo(sorted.get(i)) < 0;
        check(ordered && sorted.get(0).sign < 0 && sorted.get(sorted.size() - 1).sign > 0
                        && sorted.get(0).compareTo(Rational.ZERO) < 0
                        && Collections.binarySearch(sorted, Rational.of(1, 100)) >= 0
                        && third.compareTo(Rational.of(1, 4)) > 0,
                "the order must be total and consistent with the sign: " + sorted);

        String reference = "3141592653589793238462643383279502884197";
        String machin = machinPi(40), identity = piThroughHalfAndThird(30);
        check(machin.equals(reference) && identity.equals(reference.substring(0, 30)) && machin.length() == 40
                        && machinPi(20).equals(reference.substring(0, 20)),
                "pi must match the published digits through two formulas: " + machin + " / " + identity);

        Rational[] iterates = newtonSqrtTwo(6);
        int boundViolations = 0;
        Rational previousError = null;
        for (int i = 1; i < iterates.length; i++) {
            Rational square = iterates[i].pow(2), error = square.sub(Rational.TWO).abs();
            if (square.compareTo(Rational.TWO) < 0) boundViolations++;              // upper bound from x1 onwards
            if (previousError != null && error.compareTo(previousError) >= 0) boundViolations++;
            if (i > 1 && iterates[i].compareTo(iterates[i - 1]) >= 0) boundViolations++;
            previousError = error;
        }
        check(boundViolations == 0 && iterates[0].pow(2).compareTo(Rational.TWO) < 0
                        && iterates[1].pow(2).compareTo(Rational.TWO) > 0
                        && iterates[1].equals(Rational.of(3, 2)) && iterates[2].equals(Rational.of(17, 12)),
                "Newton's iterates must bracket sqrt(2) and shrink from 3/2 to 17/12: " + boundViolations);

        List<Rational> pell = new ArrayList<>();
        BigNat pPrev = BigNat.ONE, qPrev = BigNat.ZERO, p = BigNat.ONE, q = BigNat.ONE;
        int pellViolations = 0;
        for (int k = 0; k < 100; k++) {
            pell.add(Rational.of(p, q, 1));
            BigNat twoQSquared = q.mul(q).mulSmall(2), square = p.mul(p);
            BigNat difference = square.compareTo(twoQSquared) >= 0 ? square.sub(twoQSquared)
                                                                   : twoQSquared.sub(square);
            if (!difference.isOne()) pellViolations++;
            BigNat nextP = p.mulSmall(2).add(pPrev), nextQ = q.mulSmall(2).add(qPrev);
            pPrev = p;
            qPrev = q;
            p = nextP;
            q = nextQ;
        }
        check(pellViolations == 0 && pell.get(1).equals(Rational.of(3, 2)) && pell.get(2).equals(Rational.of(7, 5))
                        && pell.get(3).equals(Rational.of(17, 12)) && pell.get(99).num.toString().length() > 30,
                "the Pell convergents must satisfy p^2 - 2q^2 = +-1 exactly: " + pellViolations + " violations");
        check(iterates[2].equals(pell.get(3)) && iterates[1].equals(pell.get(1))
                        && iterates[3].equals(pell.get(7)),
                "Newton and the continued fraction must land on the same rationals: " + iterates[2]);

        String sevenths = Rational.of(1, 7).toDecimal(24);
        boolean repeats = true;
        for (int i = 0; i < 18; i++) repeats &= sevenths.charAt(2 + i) == sevenths.charAt(2 + i % 6);
        check(Rational.of(1, 2).toDecimal(6).equals("0.5") && Rational.of(1, 8).toDecimal(6).equals("0.125")
                        && Rational.of(22, 7).toDecimal(12).equals("3.142857142857") && repeats
                        && Rational.of(-1, 3).toDecimal(4).equals("-0.3333")
                        && Rational.of(7, 1).toDecimal(4).equals("7"),
                "decimal expansion must terminate exactly and repeat with the right period: " + sevenths);

        String zeroDenominator = "";
        try { Rational.of(1, 0); } catch (IllegalArgumentException e) { zeroDenominator = e.getMessage(); }
        BigNat huge = powerOfTen(30);
        BigNat[] divided = huge.mul(huge).divMod(huge);
        check(!zeroDenominator.isEmpty() && Rational.ZERO.pow(3).isZero() && Rational.of(0, 5).isZero()
                        && Rational.of(2, 1).pow(100).toString().equals("1267650600228229401496703205376")
                        && divided[0].equals(huge) && divided[1].isZero()
                        && huge.mul(huge).toString().equals("1" + "0".repeat(60))
                        && Rational.of(10, 4).toDecimal(2).equals("2.5"),
                "degenerate and huge values must behave: 2^100, 10^60, zero denominator " + zeroDenominator);

        check(machinPi(40).equals(machin) && piThroughHalfAndThird(30).equals(identity)
                        && newtonSqrtTwo(6)[5].equals(iterates[5]) && Rational.of(1, 7).toDecimal(24).equals(sevenths),
                "the library must be fully reproducible");

        System.out.println("library: pi(40) = " + machin + ", second formula(30) = " + identity);
        System.out.println("sqrt2: newton " + iterates[2] + " = " + iterates[2].toDecimal(12) + ", pell " + pell.get(3)
                + ", third iterate matches convergent " + pell.get(7) + ", 100 convergents up to "
                + pell.get(99).num.toString().length() + " digits");
        System.out.println("PASS");
    }
}
