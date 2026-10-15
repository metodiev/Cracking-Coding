/** Task 298 - Bitmask: iterate subsets of a mask. (Tier: Intermediate, CTCI Ch 5) */
import java.util.*;

public class Solution {
    /** All submasks of mask in descending order, including 0 (mask == 0 -> [0]). */
    static int[] enumerateSubmasks(int mask) {
        List<Integer> result = new ArrayList<>();
        int sub = mask;
        while (true) {
            result.add(sub);
            if (sub == 0) break;
            sub = (sub - 1) & mask;
        }
        int[] array = new int[result.size()];
        for (int i = 0; i < array.length; i++) array[i] = result.get(i);
        return array;
    }

    public static void main(String[] args) {
        check(Arrays.equals(enumerateSubmasks(0), new int[]{0}), "mask 0");
        check(Arrays.equals(enumerateSubmasks(1), new int[]{1, 0}), "mask 1");
        check(Arrays.equals(enumerateSubmasks(0b1011), new int[]{11, 10, 9, 8, 3, 2, 1, 0}), "mask 0b1011");
        int[] all = enumerateSubmasks(0b11111);
        check(all.length == 32 && all[0] == 31 && all[31] == 0, "mask 0b11111 bounds");
        boolean descending = true;
        for (int i = 0; i + 1 < all.length; i++) if (all[i] <= all[i + 1]) descending = false;
        check(descending, "mask 0b11111 descending");

        int[] masks = new int[46];
        masks[0] = 0;
        masks[1] = 1;
        masks[2] = 0b1011;
        masks[3] = 0b11111;
        masks[4] = 0b1000;
        masks[5] = 0b101010;
        for (int i = 6; i < masks.length; i++) {
            seed = (seed * 75 + 74) % 65537;
            masks[i] = seed % 4096;
        }

        for (int mask : masks) {
            int[] submasks = enumerateSubmasks(mask);
            check(submasks.length == (1 << Integer.bitCount(mask)), "count for " + mask);
            boolean[] seen = new boolean[mask + 1];
            boolean valid = true;
            for (int i = 0; i < submasks.length; i++) {
                int sub = submasks[i];
                if (sub != 0 && seen[sub]) valid = false;
                seen[sub] = true;
                if ((sub & ~mask) != 0) valid = false;
                if (i + 1 < submasks.length && submasks[i] <= submasks[i + 1]) valid = false;
            }
            if (submasks[0] != mask || submasks[submasks.length - 1] != 0) valid = false;
            check(valid, "properties for " + mask);
            // independent oracle: descending sweep over the integers below mask
            int expected = 0;
            for (int value = mask; value >= 0; value--) if ((value & ~mask) == 0) expected++;
            check(submasks.length == expected, "oracle for " + mask);
        }
        System.out.println("PASS");
    }

    private static int seed = 9090;

    static void check(boolean ok, String label) {
        if (!ok) throw new AssertionError("FAILED: " + label);
    }
}
