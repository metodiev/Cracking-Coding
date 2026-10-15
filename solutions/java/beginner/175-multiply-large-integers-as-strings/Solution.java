/** Task 175 - Multiply large integers as strings. (Tier: Beginner, CTCI Ch 6) */
import java.math.BigInteger;
import java.util.Random;

public class Solution {
    static String multiplyStrings(String a, String b) {
        if (a.equals("0") || b.equals("0")) return "0";
        boolean negative = (a.charAt(0) == '-') ^ (b.charAt(0) == '-');
        String da = a.startsWith("-") ? a.substring(1) : a;
        String db = b.startsWith("-") ? b.substring(1) : b;
        int[] result = new int[da.length() + db.length()];
        for (int i = da.length() - 1; i >= 0; i--)
            for (int j = db.length() - 1; j >= 0; j--)
                result[i + j + 1] += (da.charAt(i) - '0') * (db.charAt(j) - '0');
        for (int k = result.length - 1; k > 0; k--) {
            result[k - 1] += result[k] / 10;
            result[k] %= 10;
        }
        StringBuilder sb = new StringBuilder();
        for (int d : result) if (sb.length() > 0 || d != 0) sb.append(d);
        if (sb.length() == 0) return "0";
        return negative ? "-" + sb : sb.toString();
    }

    public static void main(String[] args) {
        check(multiplyStrings("123", "456").equals("56088"), "123*456");
        check(multiplyStrings("0", "999").equals("0"), "zero");
        check(multiplyStrings("-12", "12").equals("-144"), "negative");
        check(multiplyStrings("-12", "-12").equals("144"), "double negative");
        check(multiplyStrings("99999", "99999").equals("9999800001"), "carry chain");

        Random rng = new Random(11);
        for (int t = 0; t < 30; t++) {
            BigInteger x = new BigInteger(1 + rng.nextInt(120), rng);
            BigInteger y = new BigInteger(1 + rng.nextInt(120), rng);
            check(multiplyStrings(x.toString(), y.toString()).equals(x.multiply(y).toString()), "random " + t);
        }

        String nines = "9".repeat(100);
        check(multiplyStrings(nines, nines).equals(new BigInteger(nines).multiply(new BigInteger(nines)).toString()),
                "100 nines");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
