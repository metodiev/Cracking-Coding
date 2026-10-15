/** Task 392 - Implement LZW compression (encoder/decoder skeleton). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

public class Solution {
    /** alphabet: its order fixes the dictionary ids. */
    static List<Integer> lzwEncode(String text, String alphabet) {
        Map<String, Integer> dictionary = new HashMap<>();
        for (int i = 0; i < alphabet.length(); i++) dictionary.put(alphabet.substring(i, i + 1), i);
        int nextCode = dictionary.size();
        List<Integer> codes = new ArrayList<>();
        String current = "";
        for (int i = 0; i < text.length(); i++) {
            String ch = text.substring(i, i + 1);
            String candidate = current + ch;
            if (dictionary.containsKey(candidate)) {
                current = candidate;
            } else {
                if (current.isEmpty()) throw new IllegalArgumentException("symbol outside the alphabet: " + ch);
                codes.add(dictionary.get(current));
                dictionary.put(candidate, nextCode);
                nextCode++;
                current = ch;
                if (!dictionary.containsKey(ch)) throw new IllegalArgumentException("symbol outside the alphabet: " + ch);
            }
        }
        if (!current.isEmpty()) codes.add(dictionary.get(current));
        return codes;
    }

    static String lzwDecode(List<Integer> codes, String alphabet) {
        Map<Integer, String> dictionary = new TreeMap<>();
        for (int i = 0; i < alphabet.length(); i++) dictionary.put(i, alphabet.substring(i, i + 1));
        int nextCode = dictionary.size();
        if (codes.isEmpty()) return "";
        int first = codes.get(0);
        if (!dictionary.containsKey(first)) throw new IllegalArgumentException("invalid first code");
        String previous = dictionary.get(first);
        StringBuilder out = new StringBuilder(previous);
        for (int k = 1; k < codes.size(); k++) {
            int code = codes.get(k);
            String entry;
            if (dictionary.containsKey(code)) {
                entry = dictionary.get(code);
            } else if (code == nextCode) {
                entry = previous + previous.charAt(0);       // the cSc rule
            } else {
                throw new IllegalArgumentException("invalid code: " + code);
            }
            out.append(entry);
            dictionary.put(nextCode, previous + entry.charAt(0));
            nextCode++;
            previous = entry;
        }
        return out.toString();
    }

    private static int seed = 5150;

    private static List<String> lcgStrings(String alphabet, int length, int count) {
        List<String> strings = new ArrayList<>();
        for (int k = 0; k < count; k++) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < length; i++) {
                seed = (seed * 75 + 74) % 65537;
                sb.append(alphabet.charAt(seed % alphabet.length()));
            }
            strings.add(sb.toString());
        }
        return strings;
    }

    static String join(List<Integer> codes) {
        StringBuilder sb = new StringBuilder();
        for (int code : codes) {
            if (sb.length() > 0) sb.append(',');
            sb.append(code);
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        check(join(lzwEncode("abababa", "ab")).equals("0,1,2,4"), "abababa");
        check(join(lzwEncode("aaaa", "ab")).equals("0,2,0"), "aaaa");
        check(join(lzwEncode("abcabcabcabc", "abc")).equals("0,1,2,3,5,4,6"), "abc cycle");
        check(lzwEncode("", "ab").isEmpty(), "empty input");
        check(join(lzwEncode("a", "ab")).equals("0"), "single symbol");

        check(lzwDecode(Arrays.asList(0, 1, 2, 4), "ab").equals("abababa"), "decode abababa");
        check(lzwDecode(Arrays.asList(0, 2, 0), "ab").equals("aaaa"), "decode aaaa");
        check(lzwDecode(Arrays.asList(0, 1, 2, 3, 5, 4, 6), "abc").equals("abcabcabcabc"), "decode cycle");
        check(lzwDecode(new ArrayList<>(), "ab").isEmpty(), "decode empty");
        check(lzwDecode(Arrays.asList(0), "ab").equals("a"), "decode single");

        List<String> samples = new ArrayList<>(Arrays.asList("abababa", "aaaa", "", "a"));
        samples.addAll(lcgStrings("ab", 30, 40));
        for (String s : samples) check(lzwDecode(lzwEncode(s, "ab"), "ab").equals(s), "round trip " + s);
        for (String s : lcgStrings("abcd", 40, 40)) check(lzwDecode(lzwEncode(s, "abcd"), "abcd").equals(s), "round trip " + s);

        StringBuilder repetitive = new StringBuilder();
        for (int i = 0; i < 20; i++) repetitive.append("ab");
        String rep = repetitive.toString();
        check(lzwEncode(rep, "ab").size() < rep.length(), "repetitive text shrinks");

        List<Integer> csc = lzwEncode("abababab", "ab");
        check(lzwDecode(csc, "ab").equals("abababab"), "cSc round trip");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
