/** Task 410 - FM-index: basic concept and small demo. (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * Structure for text s: build t = s + "$" with a unique smallest sentinel, store
 *   * the Burrows-Wheeler last column L of t,
 *   * the C table: C[c] = number of characters of t strictly smaller than c,
 *   * rank samples: for every character, the cumulative count in L at each block boundary
 *     (block size configurable); occ(c, i) = sample + direct scan of the residual (< block).
 *
 * Backward search walks the pattern from the right: starting from the full row range
 * [0, n+1), for each character c set lo = C[c] + occ(c, lo) and hi = C[c] + occ(c, hi). The
 * interval is the contiguous range of rows whose rotations start with the pattern suffix
 * processed so far, so the final width counts the matches; an empty interval means no match.
 *
 * LF-mapping (the inverse of the sorted first column F): LF[i] = C[L[i]] + rank of L[i] among
 * equal characters before i. Following LF moves from a row to the rotation starting one
 * position earlier, so walking LF from a match row until row 0 (the "$..." rotation) gives the
 * position: after `steps` LF applications the match starts at (n + steps) mod (n + 1) in t.
 *
 * Checks: hand values for banana ("ana" -> 2, "na" -> 2, "a" -> 3, locate positions [1,3]
 * etc.), occ() against direct counts, LF is one cycle, LCG texts/patterns agree with a naive
 * scan for both count and locate, and several block sizes give identical answers.
 */
import java.util.*;

class BwtResult {
    final String last;
    final int primary;

    BwtResult(String last, int primary) {
        this.last = last;
        this.primary = primary;
    }
}

class FmIndex {
    static final char SENTINEL = '$';

    final String text;
    final String t;
    final String last;
    final int block;
    final Map<Character, Integer> cTable = new TreeMap<>();
    final Map<Character, int[]> samples = new TreeMap<>();
    final int[] lf;

    FmIndex(String text) {
        this(text, 4);
    }

    FmIndex(String text, int block) {
        this.text = text;
        this.block = block;
        this.t = text + SENTINEL;
        this.last = bwtCyclic(t).last;
        for (char ch : sortedUnique(t)) {
            int smaller = 0;
            for (int i = 0; i < t.length(); i++) if (t.charAt(i) < ch) smaller++;
            cTable.put(ch, smaller);
        }
        for (char ch : sortedUnique(t)) {
            int[] row = new int[t.length() / block + 1];
            int count = 0;
            for (int pos = 0; pos <= t.length(); pos++) {
                if (pos % block == 0) row[pos / block] = count;
                if (pos < t.length() && last.charAt(pos) == ch) count++;
            }
            samples.put(ch, row);
        }
        this.lf = lfMapping(last);
    }

    static BwtResult bwtCyclic(String s) {
        int n = s.length();
        if (n == 0) return new BwtResult("", 0);
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) order[i] = i;
        final String doubled = s + s;
        Arrays.sort(order, (a, b) -> {
            for (int k = 0; k < n; k++) {
                char ca = doubled.charAt(a + k);
                char cb = doubled.charAt(b + k);
                if (ca != cb) return ca < cb ? -1 : 1;
            }
            return 0;
        });
        StringBuilder last = new StringBuilder();
        for (int i = 0; i < n; i++) last.append(s.charAt((order[i] - 1 + n) % n));
        return new BwtResult(last.toString(), 0);
    }

    static int[] lfMapping(String last) {
        int n = last.length();
        int[] lf = new int[n];
        Map<Character, Integer> starts = new TreeMap<>();
        int total = 0;
        for (char ch : sortedUnique(last)) {
            starts.put(ch, total);
            int count = 0;
            for (int i = 0; i < n; i++) if (last.charAt(i) == ch) count++;
            total += count;
        }
        Map<Character, Integer> seen = new HashMap<>();
        for (int i = 0; i < n; i++) {
            char ch = last.charAt(i);
            int rank = seen.getOrDefault(ch, 0);
            lf[i] = starts.get(ch) + rank;
            seen.put(ch, rank + 1);
        }
        return lf;
    }

    private static List<Character> sortedUnique(String s) {
        TreeSet<Character> set = new TreeSet<>();
        for (int i = 0; i < s.length(); i++) set.add(s.charAt(i));
        return new ArrayList<>(set);
    }

    int occ(char ch, int i) {
        int[] row = samples.get(ch);
        int startBlock = (i / block) * block;
        int count = row[i / block];
        for (int pos = startBlock; pos < i; pos++) if (last.charAt(pos) == ch) count++;
        return count;
    }

    int[] backwardSearch(String pattern) {
        int lo = 0, hi = t.length();
        for (int i = pattern.length() - 1; i >= 0; i--) {
            char ch = pattern.charAt(i);
            Integer c = cTable.get(ch);
            if (c == null) return new int[]{lo, lo};
            lo = c + occ(ch, lo);
            hi = c + occ(ch, hi);
            if (lo >= hi) return new int[]{lo, hi};
        }
        return new int[]{lo, hi};
    }

    int count(String pattern) {
        if (pattern.isEmpty()) return 0;               // documented: empty pattern is not searched
        int[] range = backwardSearch(pattern);
        return range[1] - range[0];
    }

    int[] locate(String pattern) {
        if (pattern.isEmpty()) return new int[0];
        int[] range = backwardSearch(pattern);
        List<Integer> positions = new ArrayList<>();
        for (int row = range[0]; row < range[1]; row++) {
            int steps = 0;
            int current = row;
            while (current != 0) {
                current = lf[current];
                steps++;
            }
            positions.add((text.length() + steps) % t.length());
        }
        int[] out = new int[positions.size()];
        for (int i = 0; i < out.length; i++) out[i] = positions.get(i);
        Arrays.sort(out);
        return out;
    }
}

public class Solution {
    private static int seed = 2718;

    static String lcgString(int length, String alphabet) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < length; i++) {
            seed = (seed * 75 + 74) % 65537;
            sb.append(alphabet.charAt(seed % alphabet.length()));
        }
        return sb.toString();
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }

    static int naiveCount(String text, String pattern) {
        int count = 0;
        for (int i = 0; i + pattern.length() <= text.length(); i++)
            if (text.startsWith(pattern, i)) count++;
        return count;
    }

    static int[] naiveLocate(String text, String pattern) {
        List<Integer> positions = new ArrayList<>();
        for (int i = 0; i + pattern.length() <= text.length(); i++)
            if (text.startsWith(pattern, i)) positions.add(i);
        int[] out = new int[positions.size()];
        for (int i = 0; i < out.length; i++) out[i] = positions.get(i);
        return out;
    }

    static String join(int[] values) {
        StringBuilder sb = new StringBuilder();
        for (int value : values) {
            if (sb.length() > 0) sb.append(",");
            sb.append(value);
        }
        return sb.toString();
    }

    static void checkAgainstNaive(String text, List<String> patterns) {
        FmIndex index = new FmIndex(text);
        for (String pattern : patterns) {
            check(index.count(pattern) == naiveCount(text, pattern), "count " + text + "/" + pattern);
            check(join(index.locate(pattern)).equals(join(naiveLocate(text, pattern))), "locate " + text + "/" + pattern);
        }
    }

    public static void main(String[] args) {
        FmIndex index = new FmIndex("banana");
        check(index.t.equals("banana$"), "sentinel text");
        check(index.last.equals("annb$aa"), "bwt");
        check(index.cTable.get('$') == 0 && index.cTable.get('a') == 1
                && index.cTable.get('b') == 4 && index.cTable.get('n') == 5, "C table");
        for (char ch : new char[]{'$', 'a', 'b', 'n'}) {
            for (int i = 0; i <= index.t.length(); i++) {
                int direct = 0;
                for (int k = 0; k < i; k++) if (index.last.charAt(k) == ch) direct++;
                check(index.occ(ch, i) == direct, "occ " + ch + "/" + i);
            }
        }

        check(index.count("ana") == 2 && index.count("na") == 2 && index.count("a") == 3, "counts");
        check(index.count("banana") == 1 && index.count("nana") == 1 && index.count("anana") == 1, "counts 2");
        check(index.count("bananas") == 0 && index.count("x") == 0 && index.count("") == 0, "counts 3");
        check(join(index.locate("ana")).equals("1,3"), "locate ana");
        check(join(index.locate("na")).equals("2,4"), "locate na");
        check(join(index.locate("a")).equals("1,3,5"), "locate a");
        check(join(index.locate("banana")).equals("0"), "locate banana");
        check(join(index.locate("nana")).equals("2"), "locate nana");
        check(join(index.locate("anana")).equals("1"), "locate anana");
        check(index.locate("bananas").length == 0 && index.locate("x").length == 0 && index.locate("").length == 0,
                "locate empty results");
        int[] sortedLf = index.lf.clone();
        Arrays.sort(sortedLf);
        for (int i = 0; i < sortedLf.length; i++) check(sortedLf[i] == i, "lf permutation " + i);
        int row = 0;
        for (int step = 0; step < index.t.length(); step++) row = index.lf[row];
        check(row == 0, "lf single cycle");

        FmIndex reference = new FmIndex("banana", 1);
        for (int block : new int[]{1, 2, 3, 5, 8, 16}) {
            FmIndex variant = new FmIndex("banana", block);
            for (String pattern : new String[]{"a", "an", "ana", "anan", "n", "na", "banana", "zz"}) {
                check(variant.count(pattern) == reference.count(pattern), "block count " + block);
                check(join(variant.locate(pattern)).equals(join(reference.locate(pattern))), "block locate " + block);
            }
            for (char ch : new char[]{'$', 'a', 'b', 'n'}) {
                for (int i = 0; i <= variant.t.length(); i++) {
                    check(variant.occ(ch, i) == reference.occ(ch, i), "block occ " + block + "/" + ch + "/" + i);
                }
            }
        }

        checkAgainstNaive("mississippi", Arrays.asList("i", "is", "issi", "ssi", "ppi", "sipp", "mississippix", "q"));
        checkAgainstNaive("", Arrays.asList("a", "x"));
        checkAgainstNaive("a", Arrays.asList("a", "aa", "b"));
        checkAgainstNaive("aaaa", Arrays.asList("a", "aa", "aaa", "aaaa", "aaaaa", "b"));

        for (int trial = 0; trial < 30; trial++) {
            String text = lcgString(30, "abc");
            List<String> patterns = new ArrayList<>();
            for (int i = 0; i < text.length(); i++)
                for (int j = i + 1; j <= Math.min(text.length(), i + 4); j++)
                    patterns.add(text.substring(i, j));
            patterns.addAll(Arrays.asList(text, text.substring(0, text.length() - 1), "d" + text.substring(0, 5),
                    "zz", "q"));
            checkAgainstNaive(text, patterns);
        }

        String medium = lcgString(300, "abc");
        FmIndex mediumIndex = new FmIndex(medium, 7);
        for (String pattern : new String[]{"abc", "cba", "cab", "abcabc", "zzz",
                medium.substring(150, 160), medium}) {
            check(mediumIndex.count(pattern) == naiveCount(medium, pattern), "medium count " + pattern);
            check(join(mediumIndex.locate(pattern)).equals(join(naiveLocate(medium, pattern))), "medium locate " + pattern);
        }
        System.out.println("PASS");
    }
}
