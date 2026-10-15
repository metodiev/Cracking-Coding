/** Task 389 - Implement Huffman coding for characters frequency. (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class HuffNode {
    final long freq;
    final int id;
    final char ch;
    final boolean leaf;
    final HuffNode left;
    final HuffNode right;

    HuffNode(long freq, int id, char ch, boolean leaf, HuffNode left, HuffNode right) {
        this.freq = freq;
        this.id = id;
        this.ch = ch;
        this.leaf = leaf;
        this.left = left;
        this.right = right;
    }
}

public class Solution {
    static Map<Character, Integer> characterFrequencies(String text) {
        Map<Character, Integer> frequencies = new HashMap<>();
        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            frequencies.put(ch, frequencies.getOrDefault(ch, 0) + 1);
        }
        return frequencies;
    }

    /**
     * Greedy min-heap merge with the documented deterministic tie-break:
     * nodes ordered by (frequency, creation id); first popped node becomes the left
     * child (bit '0'), the second the right child (bit '1').
     */
    static Map<Character, String> huffmanCodes(Map<Character, Integer> frequencies) {
        Map<Character, String> codes = new TreeMap<>();
        if (frequencies.isEmpty()) return codes;
        List<Character> chars = new ArrayList<>(frequencies.keySet());
        Collections.sort(chars);
        PriorityQueue<HuffNode> heap = new PriorityQueue<>((x, y) -> x.freq != y.freq
                ? Long.compare(x.freq, y.freq) : Integer.compare(x.id, y.id));
        int id = 0;
        for (char ch : chars) heap.add(new HuffNode(frequencies.get(ch), id++, ch, true, null, null));
        int nextId = id;
        while (heap.size() > 1) {
            HuffNode left = heap.poll();
            HuffNode right = heap.poll();
            heap.add(new HuffNode(left.freq + right.freq, nextId++, '\0', false, left, right));
        }
        walk(heap.peek(), "", codes);
        return codes;
    }

    private static void walk(HuffNode node, String prefix, Map<Character, String> codes) {
        if (node.leaf) {
            codes.put(node.ch, prefix.isEmpty() ? "0" : prefix);
            return;
        }
        walk(node.left, prefix + "0", codes);
        walk(node.right, prefix + "1", codes);
    }

    static long weightedLength(Map<Character, String> codes, Map<Character, Integer> frequencies) {
        long total = 0;
        for (Map.Entry<Character, String> entry : codes.entrySet())
            total += (long) frequencies.get(entry.getKey()) * entry.getValue().length();
        return total;
    }

    static String huffmanEncode(String text, Map<Character, String> codes) {
        StringBuilder bits = new StringBuilder();
        for (int i = 0; i < text.length(); i++) bits.append(codes.get(text.charAt(i)));
        return bits.toString();
    }

    static String huffmanDecode(String bits, Map<Character, String> codes) {
        Map<String, Character> reverse = new HashMap<>();
        for (Map.Entry<Character, String> entry : codes.entrySet())
            reverse.put(entry.getValue(), entry.getKey());
        StringBuilder out = new StringBuilder();
        StringBuilder current = new StringBuilder();
        for (int i = 0; i < bits.length(); i++) {
            current.append(bits.charAt(i));
            Character ch = reverse.get(current.toString());
            if (ch != null) {
                out.append(ch);
                current.setLength(0);
            }
        }
        if (current.length() > 0) throw new IllegalArgumentException("bits end with an incomplete code");
        return out.toString();
    }

    static boolean isPrefixFree(Map<Character, String> codes) {
        for (String code : codes.values())
            for (String other : codes.values())
                if (!code.equals(other) && other.startsWith(code)) return false;
        return true;
    }

    /** sum 2^(-len) == 1 for a complete binary tree; the single-symbol table "0" is a lone leaf. */
    static boolean kraftSumOk(Map<Character, String> codes) {
        if (codes.isEmpty()) return true;
        if (codes.size() == 1) return codes.values().iterator().next().equals("0");
        int longest = 0;
        for (String code : codes.values()) longest = Math.max(longest, code.length());
        long sum = 0;
        for (String code : codes.values()) sum += 1L << (longest - code.length());
        return sum == (1L << longest);
    }

    static int ceilLog2(int value) {
        int bits = 0;
        while ((1 << bits) < value) bits++;
        return bits;
    }

    private static int seed = 2718;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void main(String[] args) {
        // hand-verified small example: a:2, b:3, c:4 -> {c: "0", a: "10", b: "11"}
        String text = "aabbbcccc";
        Map<Character, Integer> freq = characterFrequencies(text);
        check(freq.get('a') == 2 && freq.get('b') == 3 && freq.get('c') == 4, "frequencies");
        Map<Character, String> codes = huffmanCodes(freq);
        check(codes.size() == 3 && codes.get('a').equals("10") && codes.get('b').equals("11")
                && codes.get('c').equals("0"), "codes aabbbcccc");
        long weighted = weightedLength(codes, freq);
        check(weighted == 14 && weighted == 2 * 2 + 3 * 2 + 4 * 1, "weighted 14");
        String bits = huffmanEncode(text, codes);
        check(bits.equals("10101111110000") && bits.length() == weighted, "encoded bits");
        check(huffmanDecode(bits, codes).equals(text), "round trip");

        // single distinct character has the single code "0"
        Map<Character, Integer> freq1 = characterFrequencies("aaaa");
        Map<Character, String> codes1 = huffmanCodes(freq1);
        check(codes1.size() == 1 && codes1.get('a').equals("0"), "single symbol code");
        check(weightedLength(codes1, freq1) == 4, "single symbol weight");
        String bits1 = huffmanEncode("aaaa", codes1);
        check(bits1.equals("0000") && huffmanDecode(bits1, codes1).equals("aaaa"), "single round trip");

        // empty text
        Map<Character, String> emptyCodes = huffmanCodes(new HashMap<>());
        check(emptyCodes.isEmpty(), "empty codes");
        check(huffmanEncode("", emptyCodes).isEmpty() && huffmanDecode("", emptyCodes).isEmpty(),
                "empty round trip");

        // second hand-verified structure, built straight from frequencies
        Map<Character, Integer> bigFreq = new HashMap<>();
        bigFreq.put('a', 35);
        bigFreq.put('b', 17);
        bigFreq.put('c', 17);
        bigFreq.put('d', 16);
        bigFreq.put('e', 15);
        Map<Character, String> bigCodes = huffmanCodes(bigFreq);
        check(bigCodes.size() == 5 && bigCodes.get('a').equals("0") && bigCodes.get('e').equals("100")
                && bigCodes.get('d').equals("101") && bigCodes.get('b').equals("110")
                && bigCodes.get('c').equals("111"), "five-symbol codes");
        check(weightedLength(bigCodes, bigFreq) == 230
                && weightedLength(bigCodes, bigFreq) == 35 * 1 + (17 + 17 + 16 + 15) * 3,
                "five-symbol weight");

        // prefix-free property and the Kraft equality (a complete binary tree)
        check(isPrefixFree(codes) && kraftSumOk(codes), "prefix free and kraft small");
        check(isPrefixFree(codes1) && kraftSumOk(codes1), "prefix free and kraft single");
        check(isPrefixFree(bigCodes) && kraftSumOk(bigCodes), "prefix free and kraft big");

        // decoding rejects a dangling prefix (documented error)
        boolean threw = false;
        try {
            huffmanDecode("1", codes);
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check(threw, "incomplete bits rejected");

        // LCG strings: round trips, bit length equals the weighted path length
        for (int trial = 0; trial < 120; trial++) {
            int length = nextSeed() % 80;
            int alphabetSize = 1 + nextSeed() % 6;
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < length; i++)
                builder.append((char) ('a' + nextSeed() % alphabetSize));
            String sample = builder.toString();
            Map<Character, Integer> sampleFreq = characterFrequencies(sample);
            Map<Character, String> sampleCodes = huffmanCodes(sampleFreq);
            String sampleBits = huffmanEncode(sample, sampleCodes);
            long sampleWeight = weightedLength(sampleCodes, sampleFreq);
            check(huffmanDecode(sampleBits, sampleCodes).equals(sample), "LCG round trip " + trial);
            check(sampleBits.length() == sampleWeight, "bit length " + trial);
            check(isPrefixFree(sampleCodes) && kraftSumOk(sampleCodes), "LCG prefix " + trial);
            if (length > 0) {
                int distinct = new HashSet<>(sample.chars().boxed().toList()).size();
                check(sampleWeight <= (long) length * Math.max(1, ceilLog2(distinct)),
                        "fixed-length bound " + trial);
            }
        }
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
