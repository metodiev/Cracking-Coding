/** Task 390 - Implement Shannon-Fano coding (contrast with Huffman). (Tier: Intermediate, CTCI Beyond — Advanced Data Structures & Algorithms) */
import java.util.*;

class SfHuffNode {
    final long freq;
    final int id;
    final char ch;
    final boolean leaf;
    final SfHuffNode left;
    final SfHuffNode right;

    SfHuffNode(long freq, int id, char ch, boolean leaf, SfHuffNode left, SfHuffNode right) {
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
     * Top-down split: frequency-descending order (ties by character), cut where the halves'
     * total-frequency difference is minimal (earliest split on ties), '0' left, '1' right.
     */
    static Map<Character, String> shannonFanoCodes(Map<Character, Integer> frequencies) {
        Map<Character, String> codes = new TreeMap<>();
        if (frequencies.isEmpty()) return codes;
        List<Map.Entry<Character, Integer>> items = new ArrayList<>(frequencies.entrySet());
        items.sort((x, y) -> x.getValue().intValue() != y.getValue().intValue()
                ? Integer.compare(y.getValue(), x.getValue())
                : Character.compare(x.getKey(), y.getKey()));
        shannonFanoRec(items, "", codes);
        return codes;
    }

    private static void shannonFanoRec(List<Map.Entry<Character, Integer>> part, String prefix,
            Map<Character, String> codes) {
        if (part.size() == 1) {
            codes.put(part.get(0).getKey(), prefix.isEmpty() ? "0" : prefix);
            return;
        }
        long total = 0;
        for (Map.Entry<Character, Integer> entry : part) total += entry.getValue();
        int bestSplit = 1;
        long bestDiff = Long.MAX_VALUE, leftSum = 0;
        for (int k = 1; k < part.size(); k++) {
            leftSum += part.get(k - 1).getValue();
            long diff = Math.abs(total - 2 * leftSum);
            if (diff < bestDiff) {
                bestDiff = diff;
                bestSplit = k;
            }
        }
        shannonFanoRec(part.subList(0, bestSplit), prefix + "0", codes);
        shannonFanoRec(part.subList(bestSplit, part.size()), prefix + "1", codes);
    }

    /** Local Huffman re-implementation (same tie-break as task 389). */
    static Map<Character, String> huffmanCodes(Map<Character, Integer> frequencies) {
        Map<Character, String> codes = new TreeMap<>();
        if (frequencies.isEmpty()) return codes;
        List<Character> chars = new ArrayList<>(frequencies.keySet());
        Collections.sort(chars);
        PriorityQueue<SfHuffNode> heap = new PriorityQueue<>((x, y) -> x.freq != y.freq
                ? Long.compare(x.freq, y.freq) : Integer.compare(x.id, y.id));
        int id = 0;
        for (char ch : chars)
            heap.add(new SfHuffNode(frequencies.get(ch), id++, ch, true, null, null));
        int nextId = id;
        while (heap.size() > 1) {
            SfHuffNode left = heap.poll();
            SfHuffNode right = heap.poll();
            heap.add(new SfHuffNode(left.freq + right.freq, nextId++, '\0', false, left, right));
        }
        huffmanWalk(heap.peek(), "", codes);
        return codes;
    }

    private static void huffmanWalk(SfHuffNode node, String prefix, Map<Character, String> codes) {
        if (node.leaf) {
            codes.put(node.ch, prefix.isEmpty() ? "0" : prefix);
            return;
        }
        huffmanWalk(node.left, prefix + "0", codes);
        huffmanWalk(node.right, prefix + "1", codes);
    }

    static long weightedLength(Map<Character, String> codes, Map<Character, Integer> frequencies) {
        long total = 0;
        for (Map.Entry<Character, String> entry : codes.entrySet())
            total += (long) frequencies.get(entry.getKey()) * entry.getValue().length();
        return total;
    }

    static String encode(String text, Map<Character, String> codes) {
        StringBuilder bits = new StringBuilder();
        for (int i = 0; i < text.length(); i++) bits.append(codes.get(text.charAt(i)));
        return bits.toString();
    }

    static String decode(String bits, Map<Character, String> codes) {
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

    static boolean kraftSumOk(Map<Character, String> codes) {
        if (codes.isEmpty()) return true;
        if (codes.size() == 1) return codes.values().iterator().next().equals("0");
        int longest = 0;
        for (String code : codes.values()) longest = Math.max(longest, code.length());
        long sum = 0;
        for (String code : codes.values()) sum += 1L << (longest - code.length());
        return sum == (1L << longest);
    }

    private static int seed = 8642;

    private static int nextSeed() {
        seed = (seed * 75 + 74) % 65537;
        return seed;
    }

    public static void main(String[] args) {
        // tie example: aabbbcccc -> both codecs cost 14 bits
        Map<Character, Integer> tieFreq = new HashMap<>();
        tieFreq.put('a', 2);
        tieFreq.put('b', 3);
        tieFreq.put('c', 4);
        Map<Character, String> sf = shannonFanoCodes(tieFreq);
        Map<Character, String> huff = huffmanCodes(tieFreq);
        check(sf.size() == 3 && sf.get('c').equals("0") && sf.get('b').equals("10")
                && sf.get('a').equals("11"), "shannon-fano aabbbcccc");
        check(huff.size() == 3 && huff.get('c').equals("0") && huff.get('a').equals("10")
                && huff.get('b').equals("11"), "huffman aabbbcccc");
        check(weightedLength(sf, tieFreq) == 14, "sf weight tie");
        check(weightedLength(huff, tieFreq) == 14, "huffman weight tie");
        check(weightedLength(sf, tieFreq) == weightedLength(huff, tieFreq), "documented tie");
        for (Map<Character, String> codes : List.of(sf, huff)) {
            check(isPrefixFree(codes) && kraftSumOk(codes), "prefix/kraft tie");
            check(decode(encode("aabbbcccc", codes), codes).equals("aabbbcccc"), "round trip tie");
        }

        // strict example: Huffman 230 < Shannon-Fano 231
        Map<Character, Integer> bigFreq = new HashMap<>();
        bigFreq.put('a', 35);
        bigFreq.put('b', 17);
        bigFreq.put('c', 17);
        bigFreq.put('d', 16);
        bigFreq.put('e', 15);
        Map<Character, String> sfBig = shannonFanoCodes(bigFreq);
        Map<Character, String> huffBig = huffmanCodes(bigFreq);
        check(sfBig.size() == 5 && sfBig.get('a').equals("00") && sfBig.get('b').equals("01")
                && sfBig.get('c').equals("10") && sfBig.get('d').equals("110")
                && sfBig.get('e').equals("111"), "shannon-fano big");
        check(huffBig.size() == 5 && huffBig.get('a').equals("0") && huffBig.get('e').equals("100")
                && huffBig.get('d').equals("101") && huffBig.get('b').equals("110")
                && huffBig.get('c').equals("111"), "huffman big");
        check(weightedLength(sfBig, bigFreq) == 231, "sf weight big");
        check(weightedLength(huffBig, bigFreq) == 230, "huffman weight big");
        check(weightedLength(huffBig, bigFreq) < weightedLength(sfBig, bigFreq), "strictly better");
        check(sfBig.get('a').length() == 2 && huffBig.get('a').length() == 1, "a code lengths");
        for (Map<Character, String> codes : List.of(sfBig, huffBig)) {
            check(isPrefixFree(codes) && kraftSumOk(codes), "prefix/kraft big");
        }

        // edge cases
        check(shannonFanoCodes(new HashMap<>()).isEmpty() && huffmanCodes(new HashMap<>()).isEmpty(),
                "empty frequencies");
        Map<Character, Integer> single = new HashMap<>();
        single.put('a', 4);
        check(shannonFanoCodes(single).get('a').equals("0")
                && huffmanCodes(single).get('a').equals("0"), "single symbol");
        check(decode("", new HashMap<>()).isEmpty() && encode("", new HashMap<>()).isEmpty(), "empty");

        // LCG strings: both codecs round trip, prefix-free, and Huffman <= Shannon-Fano
        Set<Integer> ties = new HashSet<>();
        for (int trial = 0; trial < 120; trial++) {
            int length = nextSeed() % 80;
            int alphabetSize = 1 + nextSeed() % 6;
            StringBuilder builder = new StringBuilder();
            for (int i = 0; i < length; i++)
                builder.append((char) ('a' + nextSeed() % alphabetSize));
            String sample = builder.toString();
            Map<Character, Integer> freq = characterFrequencies(sample);
            Map<Character, String> sfCodes = shannonFanoCodes(freq);
            Map<Character, String> huffCodes = huffmanCodes(freq);
            for (Map<Character, String> codes : List.of(sfCodes, huffCodes)) {
                check(isPrefixFree(codes) && kraftSumOk(codes), "LCG prefix " + trial);
                String bits = encode(sample, codes);
                check(decode(bits, codes).equals(sample), "LCG round trip " + trial);
                check(bits.length() == weightedLength(codes, freq), "LCG weight " + trial);
            }
            long sfWeight = weightedLength(sfCodes, freq);
            long huffWeight = weightedLength(huffCodes, freq);
            check(huffWeight <= sfWeight, "huffman <= shannon-fano " + trial);
            if (huffWeight == sfWeight) ties.add(trial);
        }
        check(!ties.isEmpty(), "ties are observed among the samples");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
