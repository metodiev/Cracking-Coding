/** Task 080 - Implement string builder/concatenation efficient. (Tier: Beginner, CTCI Ch 1) */
public class Solution {
    static class StringBuilderSimple {
        private final java.util.List<String> parts = new java.util.ArrayList<>();

        StringBuilderSimple append(Object text) {
            parts.add(String.valueOf(text));
            return this;
        }

        String build() {
            return String.join("", parts);
        }

        int length() {
            int total = 0;
            for (String p : parts) total += p.length();
            return total;
        }
    }

    public static void main(String[] args) {
        StringBuilderSimple sb = new StringBuilderSimple();
        sb.append("Hello").append(", ").append("world").append("!").append(42);
        check(sb.build().equals("Hello, world!42"), "mixed appends");
        check(sb.length() == 15, "length, got " + sb.length());
        check(new StringBuilderSimple().build().isEmpty(), "empty");

        // contrast: naive += in a loop is O(n^2); join once is O(n)
        String[] words = {"a", "b", "c"};
        check(String.join("", words).equals("abc"), "bulk join");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
