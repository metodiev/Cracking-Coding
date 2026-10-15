/** Task 413 - Implement a data structure for dynamic string operations (rope). (Tier: Advanced, CTCI Beyond — Advanced Data Structures & Algorithms)
 *
 * A rope stores a string as a sequence of chunks (leaves of a binary tree). The real
 * structure is a balanced tree of chunks with O(log n) split/concat; this implementation
 * keeps the chunk sequence FLAT and balanced instead, a documented simulation with the
 * same interface:
 *   * chunks hold at most CHUNK = 64 characters and are rebalanced so their count stays
 *     within ~2x of the ideal ceil(n / 64),
 *   * index/charAt/slice locate by walking the chunk lengths (O(#chunks + CHUNK)),
 *   * insert/delete/split/concat/reverse splice chunks and rebalance (O(n + CHUNK));
 *     a tree-based rope would perform the same operations by re-linking subtrees.
 * All operations are functional: they return NEW ropes, leaving the receiver untouched.
 */
import java.util.*;

class Rope {
    static final int CHUNK = 64;
    final List<String> chunks;

    private Rope(List<String> chunks) {
        this.chunks = chunks;
    }

    Rope(String text) {
        this.chunks = normalize(text.isEmpty() ? new ArrayList<>() : List.of(text));
    }

    /** Drop empty chunks and re-chunk so the count is close to ceil(n / CHUNK). */
    static List<String> normalize(List<String> chunks) {
        StringBuilder sb = new StringBuilder();
        for (String c : chunks) if (!c.isEmpty()) sb.append(c);
        String whole = sb.toString();
        int n = whole.length();
        if (n == 0) return new ArrayList<>();
        int target = Math.max(1, (n + CHUNK - 1) / CHUNK);
        int size = (n + target - 1) / target;
        List<String> out = new ArrayList<>();
        for (int i = 0; i < n; i += size) out.add(whole.substring(i, Math.min(n, i + size)));
        return out;
    }

    static List<String> maybeRebalance(List<String> chunks) {
        List<String> kept = new ArrayList<>();
        int n = 0;
        for (String c : chunks) {
            if (!c.isEmpty()) { kept.add(c); n += c.length(); }
        }
        int target = n > 0 ? Math.max(1, (n + CHUNK - 1) / CHUNK) : 1;
        if (kept.size() > 2 * target + 2) return normalize(kept);
        return kept;
    }

    static Rope of(List<String> chunks) {
        return new Rope(maybeRebalance(chunks));
    }

    int length() {
        int n = 0;
        for (String c : chunks) n += c.length();
        return n;
    }

    String to_string() {
        return String.join("", chunks);
    }

    /** (chunk index, offset) for pos; offset may equal the chunk length at a boundary. */
    int[] locate(int pos) {
        for (int idx = 0; idx < chunks.size(); idx++) {
            int len = chunks.get(idx).length();
            if (pos <= len) return new int[]{idx, pos};
            pos -= len;
        }
        return new int[]{chunks.size(), 0};
    }

    char charAt(int pos) {
        if (pos < 0 || pos >= length()) throw new IndexOutOfBoundsException("rope index out of range");
        for (String c : chunks) {
            if (pos < c.length()) return c.charAt(pos);
            pos -= c.length();
        }
        throw new IndexOutOfBoundsException("rope index out of range");
    }

    String slice(int start, int end) {
        if (start < 0 || end > length() || start > end) {
            throw new IndexOutOfBoundsException("rope slice out of range");
        }
        StringBuilder out = new StringBuilder();
        int need = end - start;
        int[] loc = locate(start);
        int idx = loc[0], off = loc[1];
        while (need > 0) {
            String c = chunks.get(idx);
            int take = Math.min(need, c.length() - off);
            out.append(c, off, off + take);
            need -= take;
            idx++;
            off = 0;
        }
        return out.toString();
    }

    Rope insert(int pos, String text) {
        if (pos < 0 || pos > length()) throw new IndexOutOfBoundsException("rope insert out of range");
        int[] loc = locate(pos);
        int idx = loc[0], off = loc[1];
        List<String> out = new ArrayList<>(chunks.subList(0, idx));
        if (off > 0) out.add(chunks.get(idx).substring(0, off));
        if (!text.isEmpty()) out.addAll(normalize(List.of(text)));
        if (idx < chunks.size() && off < chunks.get(idx).length()) {
            out.add(chunks.get(idx).substring(off));
        }
        if (idx < chunks.size()) out.addAll(chunks.subList(idx + 1, chunks.size()));
        return of(out);
    }

    Rope delete(int start, int end) {
        int n = length();
        if (start < 0 || end > n || start > end) {
            throw new IndexOutOfBoundsException("rope delete out of range");
        }
        if (start == end) return of(new ArrayList<>(chunks));
        int[] l = locate(start), r = locate(end);
        List<String> out = new ArrayList<>(chunks.subList(0, l[0]));
        if (l[1] > 0) out.add(chunks.get(l[0]).substring(0, l[1]));
        if (r[1] < chunks.get(r[0]).length()) out.add(chunks.get(r[0]).substring(r[1]));
        out.addAll(chunks.subList(r[0] + 1, chunks.size()));
        return of(out);
    }

    Rope[] split(int pos) {
        if (pos < 0 || pos > length()) throw new IndexOutOfBoundsException("rope split out of range");
        int[] loc = locate(pos);
        int idx = loc[0], off = loc[1];
        List<String> left = new ArrayList<>(chunks.subList(0, idx));
        if (off > 0) left.add(chunks.get(idx).substring(0, off));
        List<String> right = new ArrayList<>();
        if (idx < chunks.size()) right.addAll(chunks.subList(idx + 1, chunks.size()));
        if (idx < chunks.size() && off < chunks.get(idx).length()) {
            right.add(0, chunks.get(idx).substring(off));
        }
        return new Rope[]{of(left), of(right)};
    }

    Rope concat(Rope other) {
        List<String> out = new ArrayList<>(chunks);
        out.addAll(other.chunks);
        return of(out);
    }

    Rope reverse() {
        List<String> out = new ArrayList<>();
        for (int i = chunks.size() - 1; i >= 0; i--) {
            out.add(new StringBuilder(chunks.get(i)).reverse().toString());
        }
        return of(out);
    }
}

class Lcg {
    private int state;

    Lcg(int state) {
        this.state = state;
    }

    int next() {
        state = (state * 75 + 74) % 65537;
        return state;
    }
}

public class Solution {
    public static void main(String[] args) {
        Rope r = new Rope("hello");
        check(r.length() == 5, "length");
        check(r.charAt(0) == 'h' && r.charAt(4) == 'o' && r.charAt(2) == 'l', "char at");
        check(r.to_string().equals("hello"), "to string");
        check(r.slice(1, 4).equals("ell"), "slice");
        check(new Rope("abef").insert(2, "cd").to_string().equals("abcdef"), "insert middle");
        check(new Rope("hello").insert(0, "xyz").to_string().equals("xyzhello"), "insert front");
        // "xyzhello" (8 chars) minus [3, 6) 'h','e','l' leaves "xyzlo"
        check(new Rope("hello").insert(0, "xyz").delete(3, 6).to_string().equals("xyzlo"), "insert delete");
        Rope[] parts = new Rope("xyzlo").split(2);
        check(parts[0].to_string().equals("xy") && parts[1].to_string().equals("zlo"), "split");
        check(parts[0].concat(parts[1]).to_string().equals("xyzlo"), "concat");
        check(new Rope("xyzlo").reverse().to_string().equals("olzyx"), "reverse");
        check(new Rope("").length() == 0 && new Rope("").to_string().isEmpty(), "empty rope");
        check(new Rope("").insert(0, "abc").to_string().equals("abc"), "insert into empty");
        check(new Rope("abc").delete(1, 2).to_string().equals("ac"), "delete middle");
        check(new Rope("abc").delete(0, 0).to_string().equals("abc"), "delete empty range");
        check(new Rope("abc").split(0)[1].to_string().equals("abc"), "split at 0");
        check(new Rope("abc").split(3)[0].to_string().equals("abc"), "split at end");
        // functional semantics: the receiver never changes
        Rope original = new Rope("keepme");
        original.insert(0, "X");
        original.delete(0, 3);
        original.reverse();
        check(original.to_string().equals("keepme"), "immutability");

        // cross-chunk operations on a 200-character rope
        StringBuilder bigB = new StringBuilder();
        for (int i = 0; i < 200; i++) bigB.append((char) ('a' + i % 26));
        String big = bigB.toString();
        Rope rb = new Rope(big);
        check(rb.length() == 200 && rb.to_string().equals(big), "big round trip");
        check(rb.charAt(0) == big.charAt(0) && rb.charAt(199) == big.charAt(199), "big ends");
        check(rb.charAt(63) == big.charAt(63) && rb.charAt(64) == big.charAt(64), "big boundary");
        check(rb.slice(0, 70).equals(big.substring(0, 70)), "big slice front");
        check(rb.slice(60, 150).equals(big.substring(60, 150)), "big slice deep");
        String oracle = big.substring(0, 130) + "XYZ" + big.substring(130);
        Rope ins = rb.insert(130, "XYZ");
        check(ins.to_string().equals(oracle) && ins.length() == 203, "big insert");
        Rope del = ins.delete(125, 135);
        check(del.to_string().equals(oracle.substring(0, 125) + oracle.substring(135)), "big delete");
        Rope[] splitBig = new Rope(big).split(64);
        check(splitBig[0].length() == 64 && splitBig[1].length() == 136, "big split lengths");
        check(splitBig[0].to_string().equals(big.substring(0, 64)), "big split left");
        check(splitBig[1].to_string().equals(big.substring(64)), "big split right");
        check(splitBig[0].concat(splitBig[1]).to_string().equals(big), "big concat");
        check(new Rope(big).reverse().to_string().equals(new StringBuilder(big).reverse().toString()), "big reverse");

        // chunk length invariant after structural operations
        Rope probe = new Rope(big).insert(0, "z".repeat(40)).delete(10, 90).reverse()
                .concat(new Rope("q".repeat(300)));
        for (String c : probe.chunks) check(c.length() <= Rope.CHUNK, "chunk cap");
        check(probe.length() == 200 + 40 - 80 + 300, "probe length");

        // LCG-driven fuzz against a plain-string oracle; every operation must agree,
        // and no op script can desynchronise the rope from the oracle string.
        Lcg gen = new Lcg(31337);
        String alphabet = "abcdef";
        Rope rope = new Rope("");
        String oracleStr = "";
        for (int step = 0; step < 320; step++) {
            int kind = gen.next() % 5;
            if (kind == 0 || kind == 1) {
                int pos = gen.next() % (oracleStr.length() + 1);
                int ln = 1 + gen.next() % 12;
                StringBuilder text = new StringBuilder();
                for (int k = 0; k < ln; k++) text.append(alphabet.charAt(gen.next() % alphabet.length()));
                rope = rope.insert(pos, text.toString());
                oracleStr = oracleStr.substring(0, pos) + text + oracleStr.substring(pos);
            } else if (kind == 2) {
                if (!oracleStr.isEmpty()) {
                    int a = gen.next() % (oracleStr.length() + 1);
                    int b = gen.next() % (oracleStr.length() + 1);
                    if (a > b) { int t = a; a = b; b = t; }
                    rope = rope.delete(a, b);
                    oracleStr = oracleStr.substring(0, a) + oracleStr.substring(b);
                }
            } else if (kind == 3) {
                rope = rope.reverse();
                oracleStr = new StringBuilder(oracleStr).reverse().toString();
            } else {
                int a = gen.next() % (oracleStr.length() + 1);
                int b = gen.next() % (oracleStr.length() + 1);
                if (a > b) { int t = a; a = b; b = t; }
                check(rope.slice(a, b).equals(oracleStr.substring(a, b)), "fuzz slice step " + step);
                if (!oracleStr.isEmpty()) {
                    int pos = gen.next() % oracleStr.length();
                    check(rope.charAt(pos) == oracleStr.charAt(pos), "fuzz char step " + step);
                }
                int mid = a;
                Rope[] sp = rope.split(mid);
                check(sp[0].to_string().equals(oracleStr.substring(0, mid)), "fuzz split left " + step);
                check(sp[1].to_string().equals(oracleStr.substring(mid)), "fuzz split right " + step);
                rope = sp[0].concat(sp[1]);
            }
            check(rope.to_string().equals(oracleStr), "fuzz step " + step);
            check(rope.length() == oracleStr.length(), "fuzz length step " + step);
        }

        // final sweep: index and slice for every position of the fuzz result
        for (int i = 0; i < oracleStr.length(); i++) {
            check(rope.charAt(i) == oracleStr.charAt(i), "sweep " + i);
        }
        check(rope.slice(0, oracleStr.length()).equals(oracleStr), "sweep slice");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
