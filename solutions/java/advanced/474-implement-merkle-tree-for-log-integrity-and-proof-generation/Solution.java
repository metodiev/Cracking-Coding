/** Task 474 - Implement Merkle tree for log integrity and proof generation. (Tier: Advanced, CTCI Beyond — Security & Privacy) */
import java.util.*;

class MerkleTree {
    static final long MASK = 0xFFFFFFFFL;

    final List<String> leaves;
    final List<long[]> levels;

    MerkleTree(List<String> values) {
        leaves = new ArrayList<>(values);
        levels = new ArrayList<>();
        List<Long> current = new ArrayList<>();
        for (String v : leaves) current.add(hashLeaf(v));
        List<List<Long>> all = new ArrayList<>();
        all.add(current);
        while (current.size() > 1) {
            List<Long> next = new ArrayList<>();
            for (int i = 0; i + 1 < current.size(); i += 2) {
                next.add(hashNode(current.get(i), current.get(i + 1)));
            }
            if (current.size() % 2 == 1) next.add(current.get(current.size() - 1)); // promote
            all.add(next);
            current = next;
        }
        for (List<Long> lvl : all) {
            long[] arr = new long[lvl.size()];
            for (int i = 0; i < lvl.size(); i++) arr[i] = lvl.get(i);
            levels.add(arr);
        }
    }

    long root() {
        return levels.get(levels.size() - 1)[0];
    }

    List<long[]> proof(int index) {
        List<long[]> path = new ArrayList<>();
        int idx = index;
        for (int li = 0; li < levels.size() - 1; li++) {
            long[] level = levels.get(li);
            if (idx % 2 == 1) path.add(new long[]{level[idx - 1], 0});        // sibling on the left
            else if (idx + 1 < level.length) path.add(new long[]{level[idx + 1], 1});
            idx /= 2;
        }
        return path;
    }

    static long fnv1a(byte[] data) {
        long h = 2166136261L;
        for (byte b : data) {
            h = ((h ^ (b & 0xFF)) * 16777619L) & MASK;
        }
        return h;
    }

    static long hashLeaf(String value) {
        byte[] data = new byte[value.length() + 1];
        for (int i = 0; i < value.length(); i++) data[i + 1] = (byte) value.charAt(i);
        return fnv1a(data);
    }

    static long hashNode(long left, long right) {
        byte[] data = new byte[8];
        for (int i = 0; i < 4; i++) {
            data[i] = (byte) ((left >> (8 * (3 - i))) & 0xFF);
            data[4 + i] = (byte) ((right >> (8 * (3 - i))) & 0xFF);
        }
        return fnv1a(data);
    }

    static boolean verify(long root, String value, List<long[]> path) {
        long h = hashLeaf(value);
        for (long[] step : path) {
            h = step[1] == 0 ? hashNode(step[0], h) : hashNode(h, step[0]);
        }
        return h == root;
    }
}

public class Solution {
    public static void main(String[] args) {
        List<String> leaves = new ArrayList<>();
        for (int i = 0; i < 7; i++) leaves.add("tx" + i);
        MerkleTree tree = new MerkleTree(leaves);
        check(tree.root() == 818430076L, "root of the 7-record log");
        for (int i = 0; i < 7; i++) {
            check(MerkleTree.verify(tree.root(), leaves.get(i), tree.proof(i)), "proof must verify " + i);
        }
        check(tree.proof(0).size() == 3, "index 0 walks three levels");
        check(tree.proof(6).size() == 2, "promoted index 6 has no level-0 sibling");
        check(!MerkleTree.verify(tree.root(), "tx3x", tree.proof(3)), "tampered leaf must fail");
        check(!MerkleTree.verify(tree.root() ^ 1L, leaves.get(3), tree.proof(3)), "wrong root must fail");
        List<long[]> bad = new ArrayList<>(tree.proof(3));
        bad.set(0, new long[]{bad.get(0)[0] ^ 1L, bad.get(0)[1]});
        check(!MerkleTree.verify(tree.root(), leaves.get(3), bad), "tampered sibling must fail");
        check(new MerkleTree(leaves).root() == tree.root(), "fresh build matches");
        MerkleTree rebuilt = new MerkleTree(leaves);
        for (int i = 0; i < 7; i++) {
            check(MerkleTree.verify(rebuilt.root(), leaves.get(i), tree.proof(i)), "proofs agree with rebuild");
        }

        List<String> leaves8 = new ArrayList<>();
        for (int i = 0; i < 8; i++) leaves8.add("tx" + i);
        MerkleTree tree8 = new MerkleTree(leaves8);
        check(tree8.root() == 955777637L, "root of the 8-record log");
        for (int i = 0; i < 8; i++) {
            check(tree8.proof(i).size() == 3, "binary size audit path length");
            check(MerkleTree.verify(tree8.root(), leaves8.get(i), tree8.proof(i)), "n=8 proof " + i);
        }
        check(!MerkleTree.verify(tree8.root(), "tx0!", tree8.proof(0)), "tampered leaf in n=8");
        check(tree8.root() != tree.root(), "different logs, different roots");

        MerkleTree single = new MerkleTree(Collections.singletonList("only"));
        check(single.root() == MerkleTree.hashLeaf("only"), "single record root");
        check(single.proof(0).isEmpty(), "single record empty proof");
        check(MerkleTree.verify(single.root(), "only", single.proof(0)), "single record verify");
        System.out.println("PASS");
    }

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
