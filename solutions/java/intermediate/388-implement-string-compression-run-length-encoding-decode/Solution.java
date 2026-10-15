/** Task 388 - Implement string compression (run-length encoding + decode). (Tier: Intermediate, CTCI Ch 1) */
public class Solution {
    static boolean isLetter(char ch) {
        return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
    }

    static boolean isDigitChar(char ch) {
        return ch >= '0' && ch <= '9';
    }

    /** Runs of length 1 stay bare; runs of length >= 2 become character + decimal count. */
    static String rleEncode(String text) {
        StringBuilder out = new StringBuilder();
        int i = 0;
        while (i < text.length()) {
            int j = i;
            while (j < text.length() && text.charAt(j) == text.charAt(i)) j++;
            int run = j - i;
            out.append(text.charAt(i));
            if (run >= 2) out.append(run);
            i = j;
        }
        return out.toString();
    }

    /** Rejects non-canonical counts 0 and 1, and non-letter characters. */
    static String rleDecode(String encoded) {
        StringBuilder out = new StringBuilder();
        int i = 0;
        while (i < encoded.length()) {
            char ch = encoded.charAt(i);
            if (!isLetter(ch)) throw new IllegalArgumentException("expected a letter at position " + i);
            i++;
            int count = 0;
            boolean hasDigits = false;
            while (i < encoded.length() && isDigitChar(encoded.charAt(i))) {
                count = count * 10 + (encoded.charAt(i) - '0');
                hasDigits = true;
                i++;
            }
            if (!hasDigits) {
                out.append(ch);
                continue;
            }
            if (count < 2) throw new IllegalArgumentException("count " + count + " is not canonical");
            for (int k = 0; k < count; k++) out.append(ch);
        }
        return out.toString();
    }

    /** First character of every maximal run, in order (the letters of the encoding). */
    static String runCharacters(String text) {
        StringBuilder out = new StringBuilder();
        for (int i = 0; i < text.length(); i++)
            if (i == 0 || text.charAt(i) != text.charAt(i - 1)) out.append(text.charAt(i));
        return out.toString();
    }

    private static int seed = 5150;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void main(String[] args) {
        check(rleEncode("aaabbc").equals("a3b2c"), "encode aaabbc");
        check(rleDecode("a3b2c").equals("aaabbc"), "decode a3b2c");

        // empty string, single characters, no runs
        check(rleEncode("").equals("") && rleDecode("").equals(""), "empty string");
        check(rleEncode("a").equals("a") && rleDecode("a").equals("a"), "single character");
        check(rleEncode("abcd").equals("abcd") && rleDecode("abcd").equals("abcd"), "no runs");

        // runs of exactly two keep their length; runs of three or more shrink
        check(rleEncode("aab").equals("a2b"), "run of two");
        check(rleEncode("abb").equals("ab2"), "run of two at the end");
        check(rleEncode("aabbaa").equals("a2b2a2"), "alternating runs");
        check(rleDecode("a2b2a2").equals("aabbaa"), "decode alternating runs");
        check(rleEncode("aaabccccd").equals("a3bc4d"), "mixed runs");
        check(rleDecode("a3bc4d").equals("aaabccccd"), "decode mixed runs");
        check(rleDecode("z2").equals("zz"), "decode z2");

        // multi-digit counts (the 9 -> 10 boundary)
        check(rleEncode("a".repeat(9) + "b").equals("a9b"), "run of nine");
        check(rleEncode("a".repeat(10)).equals("a10") && rleDecode("a10").equals("a".repeat(10)),
                "run of ten");
        check(rleEncode("a".repeat(12)).equals("a12") && rleDecode("a12").equals("a".repeat(12)),
                "run of twelve");
        check(rleDecode("a99").equals("a".repeat(99)), "run of ninety-nine");

        // canonical outputs round trip through decode then encode
        String[] canonical = { "a3b2c", "a2b2a2", "a9b", "a10", "a12", "a3bc4d", "abcd", "a", "a99" };
        for (String encoded : canonical)
            check(rleEncode(rleDecode(encoded)).equals(encoded), "canonical " + encoded);

        // malformed input is rejected (documented)
        for (String bad : new String[] { "3", "a0", "a1", "ab1" }) {
            boolean threw = false;
            try {
                rleDecode(bad);
            } catch (IllegalArgumentException e) {
                threw = true;
            }
            check(threw, "malformed rejected " + bad);
        }

        // LCG strings: round trips, canonical idempotence, and the letters of the encoding
        for (int trial = 0; trial < 200; trial++) {
            int length = nextSeed() % 60;
            int alphabetSize = 2 + nextSeed() % 4;
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < length; i++)
                builder.append((char) ('a' + nextSeed() % alphabetSize));
            String text = builder.toString();
            String encoded = rleEncode(text);
            check(encoded.length() <= text.length(), "encoding never grows " + trial);
            StringBuilder letters = new StringBuilder();
            for (int i = 0; i < encoded.length(); i++)
                if (isLetter(encoded.charAt(i))) letters.append(encoded.charAt(i));
            check(runCharacters(text).equals(letters.toString()), "run letters " + trial);
            check(rleDecode(encoded).equals(text), "round trip " + trial);
            check(rleEncode(rleDecode(encoded)).equals(encoded), "canonical idempotence " + trial);
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
