/** Task 289 - Restore IP addresses from a digit string. (Tier: Intermediate, CTCI Ch 8) */
import java.util.*;

public class Solution {
    private static String digits = "";
    private static List<String> parts = new ArrayList<>();
    private static List<String> results = new ArrayList<>();

    /**
     * Parts are extended left to right, trying segment lengths 1, 2, 3 in that order, so
     * the emitted addresses are deterministic. A segment is valid when it is 1-3 digits,
     * at most 255, and has no leading zero unless it is exactly "0".
     */
    static List<String> restoreIpAddresses(String s) {
        if (s.length() < 4 || s.length() > 12) return new ArrayList<>();
        digits = s;
        parts = new ArrayList<>();
        results = new ArrayList<>();
        backtrack(0);
        return results;
    }

    private static void backtrack(int index) {
        if (parts.size() == 4) {
            if (index == digits.length()) results.add(String.join(".", parts));
            return;
        }
        for (int length = 1; length <= 3; length++) {
            if (index + length > digits.length()) break;
            String segment = digits.substring(index, index + length);
            if (!isValidSegment(segment)) continue;
            parts.add(segment);
            backtrack(index + length);
            parts.remove(parts.size() - 1);
        }
    }

    static boolean isValidSegment(String segment) {
        if (segment.isEmpty() || segment.length() > 3) return false;
        if (segment.charAt(0) == '0') return segment.equals("0");
        return Integer.parseInt(segment) <= 255;
    }

    /** Oracle: try every triple of cut positions and validate the four segments. */
    static List<String> bruteForceIpAddresses(String s) {
        List<String> found = new ArrayList<>();
        int n = s.length();
        for (int i = 1; i < Math.min(4, n); i++) {
            for (int j = i + 1; j < Math.min(i + 4, n); j++) {
                for (int k = j + 1; k < Math.min(j + 4, n); k++) {
                    String[] candidate = {s.substring(0, i), s.substring(i, j), s.substring(j, k), s.substring(k)};
                    boolean ok = true;
                    for (String segment : candidate) if (!isValidSegment(segment)) { ok = false; break; }
                    if (ok) found.add(String.join(".", candidate));
                }
            }
        }
        Collections.sort(found);
        return found;
    }

    static boolean isValidIp(String text, String s) {
        String[] segments = text.split("\\.", -1);
        if (segments.length != 4) return false;
        StringBuilder joined = new StringBuilder();
        for (String segment : segments) {
            if (!isValidSegment(segment)) return false;
            joined.append(segment);
        }
        return joined.toString().equals(s);
    }

    private static int seed = 13579;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void main(String[] args) {
        check(restoreIpAddresses("25525511135").equals(List.of("255.255.11.135", "255.255.111.35")), "classic");
        check(restoreIpAddresses("0000").equals(List.of("0.0.0.0")), "zeros");
        check(restoreIpAddresses("1111").equals(List.of("1.1.1.1")), "ones");
        check(restoreIpAddresses("010010").equals(List.of("0.10.0.10", "0.100.1.0")), "leading zero rules");
        check(restoreIpAddresses("101023")
                .equals(List.of("1.0.10.23", "1.0.102.3", "10.1.0.23", "10.10.2.3", "101.0.2.3")), "101023");
        check(restoreIpAddresses("255255255255").equals(List.of("255.255.255.255")), "max address");

        check(restoreIpAddresses("111").isEmpty(), "too short");
        check(restoreIpAddresses("").isEmpty(), "empty");
        check(restoreIpAddresses("1111111111111").isEmpty(), "too long");
        check(restoreIpAddresses("256256256256").isEmpty(), "segments over 255");
        check(restoreIpAddresses("111111111111111111111").isEmpty(), "way too long");

        for (int trial = 0; trial < 200; trial++) {
            int length = 1 + nextSeed() % 14;
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < length; i++) builder.append(nextSeed() % 10);
            String s = builder.toString();
            List<String> found = restoreIpAddresses(s);
            List<String> sorted = new ArrayList<>(found);
            Collections.sort(sorted);
            check(sorted.equals(bruteForceIpAddresses(s)), "oracle " + s);
            Set<String> unique = new HashSet<>();
            for (String text : found) {
                check(isValidIp(text, s), "valid " + s + " " + text);
                check(unique.add(text), "duplicate " + s);
            }
        }

        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
