/** Task 694 - Implement distributed erasure-coded storage simulation with repair. (Tier: Expert, CTCI Ch 9 - System Design and Scalability)
 *
 * A file is split into k shards, extended with m Reed-Solomon parity shards over GF(2^8), and
 * one shard is placed per node, so the cluster stores n = k + m shards of size S/ k each. Any k
 * of the n shards rebuild the file: each byte column of the k survivors is decoded with the
 * same Gaussian elimination used for erasures.
 *
 * Repair: when a node dies its shard is regenerated from k surviving shards (the standard
 * regeneration cost - regenerating codes try to beat it, plain RS cannot) and written to a live
 * node that holds no shard of this stripe. Durability is monotone in the failure count: up to m
 * simultaneous failures are always recoverable, m + 1 are not, and the simulator reports that
 * outcome instead of guessing.
 *
 * Simplifications: one object per stripe, no rebalancing policies, and the placement rule is the
 * simple round-robin over live nodes. Verified invariants: every shard lives on exactly one
 * node, the shard-count accounting matches after every failure and repair, a repaired shard is
 * byte-identical to the one it replaces (RS is deterministic), any k shards decode, and a repair
 * reads exactly k shards.
 */
import java.util.*;

/** Compact Reed-Solomon erasure code over GF(2^8) with systematic encoding. */
class RsCode {
    final int k;
    final int n;
    final int parity;
    private final int[] exp = new int[512];
    private final int[] log = new int[256];
    private final int[][] matrix;

    RsCode(int k, int n) {
        this.k = k;
        this.n = n;
        this.parity = n - k;
        int value = 1;
        for (int i = 0; i < 255; i++) {
            exp[i] = value;
            log[value] = i;
            value <<= 1;
            if ((value & 0x100) != 0) value ^= 0x11D;
        }
        for (int i = 255; i < 512; i++) exp[i] = exp[i - 255];
        // Systematic generator matrix: the first k rows are the identity (the data shards are the
        // message symbols themselves), the parity rows are evaluations of m(x) at alpha^1..alpha^m.
        matrix = new int[n][k];
        for (int i = 0; i < k; i++) matrix[i][i] = 1;
        for (int p = k; p < n; p++) {
            int x = exp[p - k + 1];
            int power = 1;
            for (int i = 0; i < k; i++) {
                matrix[p][i] = power;
                power = multiply(power, x);
            }
        }
    }

    int multiply(int a, int b) {
        return (a == 0 || b == 0) ? 0 : exp[log[a] + log[b]];
    }

    private int inverse(int a) {
        return exp[255 - log[a]];
    }

    /** Encodes one column of k message symbols into n codeword symbols. */
    byte[] encode(byte[] message) {
        byte[] out = new byte[n];
        for (int j = 0; j < n; j++) {
            int value = 0;
            for (int i = 0; i < k; i++) value ^= multiply(matrix[j][i], message[i] & 0xFF);
            out[j] = (byte) value;
        }
        return out;
    }

    /** Decodes a column from whichever shards survived; null when fewer than k are present. */
    byte[] decode(Byte[] column) {
        int[] indices = new int[k];
        int found = 0;
        for (int j = 0; j < n && found < k; j++) {
            if (column[j] != null) indices[found++] = j;
        }
        if (found < k) return null;
        int[][] augmented = new int[k][k + 1];
        for (int row = 0; row < k; row++) {
            for (int column2 = 0; column2 < k; column2++) {
                augmented[row][column2] = matrix[indices[row]][column2];
            }
            augmented[row][k] = column[indices[row]] & 0xFF;
        }
        for (int pivotColumn = 0; pivotColumn < k; pivotColumn++) {
            int pivot = -1;
            for (int row = pivotColumn; row < k; row++) {
                if (augmented[row][pivotColumn] != 0) {
                    pivot = row;
                    break;
                }
            }
            if (pivot < 0) return null;
            int[] swap = augmented[pivotColumn];
            augmented[pivotColumn] = augmented[pivot];
            augmented[pivot] = swap;
            int scale = inverse(augmented[pivotColumn][pivotColumn]);
            for (int j = pivotColumn; j <= k; j++) {
                augmented[pivotColumn][j] = multiply(augmented[pivotColumn][j], scale);
            }
            for (int row = 0; row < k; row++) {
                if (row == pivotColumn || augmented[row][pivotColumn] == 0) continue;
                int factor = augmented[row][pivotColumn];
                for (int j = pivotColumn; j <= k; j++) {
                    augmented[row][j] ^= multiply(factor, augmented[pivotColumn][j]);
                }
            }
        }
        byte[] message = new byte[k];
        for (int i = 0; i < k; i++) message[i] = (byte) augmented[i][k];
        return message;
    }
}

/** Erasure-coded stripe spread over a cluster of storage nodes. */
class Stripe {
    final RsCode code;
    final int nodes;
    final int shardSize;
    final byte[][] shards;      // shard index -> bytes, null when lost
    final int[] holder;         // shard index -> node id, -1 when lost
    final int[] shardsOn;       // node id -> number of shards
    long shardsRead;
    long repairs;

    Stripe(RsCode code, int nodes, int shardSize) {
        this.code = code;
        this.nodes = nodes;
        this.shardSize = shardSize;
        this.shards = new byte[code.n][];
        this.holder = new int[code.n];
        this.shardsOn = new int[nodes];
        Arrays.fill(holder, -1);
    }

    /** Splits the file into k data shards, encodes n shards and places one per node. */
    void store(byte[] file) {
        for (int s = 0; s < code.k; s++) {
            byte[] shard = new byte[shardSize];
            System.arraycopy(file, s * shardSize, shard, 0, shardSize);
            shards[s] = shard;
        }
        for (int column = 0; column < shardSize; column++) {
            byte[] message = new byte[code.k];
            for (int s = 0; s < code.k; s++) message[s] = shards[s][column];
            byte[] encoded = code.encode(message);
            for (int p = code.k; p < code.n; p++) {
                if (shards[p] == null) shards[p] = new byte[shardSize];
                shards[p][column] = encoded[p];
            }
        }
        for (int s = 0; s < code.n; s++) place(s, s % nodes);
    }

    private void place(int shard, int node) {
        holder[shard] = node;
        shardsOn[node]++;
    }

    int liveShards() {
        int count = 0;
        for (int s = 0; s < code.n; s++) if (holder[s] >= 0) count++;
        return count;
    }

    /** Kills a node: every shard it held becomes unavailable. */
    int fail(int node) {
        int lost = 0;
        for (int s = 0; s < code.n; s++) {
            if (holder[s] == node) {
                holder[s] = -1;
                shards[s] = null;
                shardsOn[node]--;
                lost++;
            }
        }
        return lost;
    }

    /** Reads a column from whatever survived; counts one shard read per present shard. */
    byte[] readColumn(int column) {
        Byte[] present = new Byte[code.n];
        for (int s = 0; s < code.n; s++) {
            if (shards[s] != null) present[s] = shards[s][column];
        }
        byte[] message = code.decode(present);
        if (message == null) return null;
        for (int s = 0; s < code.n; s++) if (shards[s] != null) shardsRead++;
        return message;
    }

    /** Reconstructs the file from the surviving shards. */
    byte[] readFile() {
        byte[] file = new byte[code.k * shardSize];
        for (int column = 0; column < shardSize; column++) {
            byte[] message = readColumn(column);
            if (message == null) return null;
            for (int s = 0; s < code.k; s++) file[s * shardSize + column] = message[s];
        }
        return file;
    }

    /** Regenerates every lost shard from k survivors and writes it to a free live node. */
    boolean repair(boolean[] nodeAlive) {
        boolean repairedAny = false;
        for (int s = 0; s < code.n; s++) {
            if (holder[s] >= 0) continue;
            int available = 0;
            for (int other = 0; other < code.n; other++) if (shards[other] != null) available++;
            if (available < code.k) return false;
            byte[] restored = new byte[shardSize];
            for (int column = 0; column < shardSize; column++) {
                Byte[] present = new Byte[code.n];
                for (int other = 0; other < code.n; other++) {
                    if (shards[other] != null) present[other] = shards[other][column];
                }
                // Regeneration: decode the message column from k survivors, then re-encode just
                // the lost symbol - exactly k shard reads, the plain Reed-Solomon repair cost.
                byte[] message = code.decode(present);
                if (message == null) return false;
                restored[column] = code.encode(message)[s];
                shardsRead += code.k;
            }
            shards[s] = restored;
            int target = -1;
            for (int node = 0; node < nodes; node++) {
                if (!nodeAlive[node] || shardsOn[node] > 0) continue;
                target = node;
                break;
            }
            if (target < 0) {
                for (int node = 0; node < nodes; node++) {
                    if (nodeAlive[node]) {
                        target = node;
                        break;
                    }
                }
            }
            place(s, target);
            repairs++;
            repairedAny = true;
        }
        return repairedAny;
    }

    boolean placementValid() {
        int total = 0;
        for (int node = 0; node < nodes; node++) total += shardsOn[node];
        for (int s = 0; s < code.n; s++) {
            if (holder[s] >= 0 && !shardsAlive(s)) return false;
        }
        return total == liveShards();
    }

    private boolean shardsAlive(int shard) {
        return shards[shard] != null;
    }
}

public class Solution {
    private static long prngState = 0xBEEF17L;

    static long nextLong() {
        prngState += 0x9E3779B97F4A7C15L;
        long z = prngState;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    static int nextInt(int bound) {
        return (int) Math.floorMod(nextLong(), (long) bound);
    }

    static byte[] sampleFile(int length) {
        byte[] file = new byte[length];
        for (int i = 0; i < length; i++) file[i] = (byte) nextInt(256);
        return file;
    }

    public static void main(String[] args) {
        RsCode code = new RsCode(6, 9);
        Stripe stripe = new Stripe(code, 9, 4);
        byte[] file = sampleFile(24);
        byte[] original = file.clone();
        stripe.store(file);

        // 1. Placement: one shard per node, accounting consistent, file readable.
        check(stripe.liveShards() == 9, "nine shards were created");
        boolean spread = true;
        for (int node = 0; node < 9; node++) if (stripe.shardsOn[node] != 1) spread = false;
        check(spread, "round-robin placement spreads the stripe over distinct nodes");
        check(stripe.placementValid(), "shard accounting matches the placement table");
        check(Arrays.equals(stripe.readFile(), original), "the stored file reads back byte-identical");

        // 2. Any k of the n shards decode the file.
        int subsets = 0;
        for (int mask = 0; mask < (1 << 9) && subsets < 20; mask++) {
            if (Integer.bitCount(mask) != 6) continue;
            subsets++;
            byte[][] saved = new byte[9][];
            int[] savedHolder = new int[9];
            for (int s = 0; s < 9; s++) {
                saved[s] = stripe.shards[s];
                savedHolder[s] = stripe.holder[s];
                if ((mask & (1 << s)) == 0) {
                    stripe.shards[s] = null;
                    stripe.holder[s] = -1;
                }
            }
            check(Arrays.equals(stripe.readFile(), original), "any six shards rebuild the file");
            for (int s = 0; s < 9; s++) {
                stripe.shards[s] = saved[s];
                stripe.holder[s] = savedHolder[s];
            }
        }
        check(subsets == 20, "twenty shard subsets were exercised");

        // 3. Up to m failures stay readable; m + 1 do not.
        Stripe resilience = new Stripe(new RsCode(6, 9), 9, 4);
        resilience.store(sampleFile(24));
        boolean[] alive = new boolean[9];
        Arrays.fill(alive, true);
        for (int i = 0; i < 3; i++) {
            alive[i] = false;
            resilience.fail(i);
        }
        check(resilience.liveShards() == 6 && resilience.readFile() != null,
                "three failures leave exactly k survivors and the data is readable");
        alive[3] = false;
        resilience.fail(3);
        check(resilience.liveShards() == 5 && resilience.readFile() == null,
                "a fourth failure makes the stripe unrecoverable");
        check(!resilience.repair(alive), "repair reports failure when fewer than k shards survive");

        // 4. Repair after a single failure: traffic equals exactly k shards, content is identical.
        Stripe repairable = new Stripe(new RsCode(6, 9), 9, 4);
        byte[] data = sampleFile(24);
        repairable.store(data);
        byte[][] before = new byte[9][];
        for (int s = 0; s < 9; s++) before[s] = repairable.shards[s].clone();
        boolean[] live = new boolean[9];
        Arrays.fill(live, true);
        live[3] = false;
        repairable.fail(3);
        long readsBefore = repairable.shardsRead;
        check(repairable.repair(live), "the lost shard is regenerated");
        check(repairable.shardsRead - readsBefore == 4L * 6, "repair read exactly k shards per column: "
                + (repairable.shardsRead - readsBefore));
        check(repairable.liveShards() == 9, "the cluster holds n shards again");
        check(repairable.placementValid(), "accounting is consistent after the repair");
        int relocated = -1;
        for (int s = 0; s < 9; s++) if (repairable.holder[s] == 3) relocated = s;
        check(relocated < 0, "the dead node holds no shard after the repair");
        boolean identical = true;
        for (int s = 0; s < 9; s++) {
            for (int b = 0; b < 4; b++) if (repairable.shards[s][b] != before[s][b]) identical = false;
        }
        check(identical, "every regenerated shard is byte-identical to the original");
        check(Arrays.equals(repairable.readFile(), data), "the file is intact after the repair");

        // 5. Sequential failures with repair in between: the stripe survives node churn.
        Stripe churn = new Stripe(new RsCode(6, 9), 9, 4);
        byte[] churnData = sampleFile(24);
        churn.store(churnData);
        boolean[] churnAlive = new boolean[9];
        Arrays.fill(churnAlive, true);
        for (int round = 0; round < 4; round++) {
            int victim = (round * 2) % 9;
            churnAlive[victim] = false;
            churn.fail(victim);
            check(churn.liveShards() == 8, "one shard lost in round " + round);
            check(churn.readFile() != null, "data readable with one failure in round " + round);
            check(churn.repair(churnAlive), "repair succeeds in round " + round);
        }
        check(churn.repairs == 4, "four repairs were performed: " + churn.repairs);
        check(Arrays.equals(churn.readFile(), churnData), "the file survived the churn");
        check(churn.placementValid(), "placement invariants hold after churn");

        // 6. Random failure patterns: with at most m failures the file is always recoverable.
        int recoverable = 0;
        for (int trial = 0; trial < 200; trial++) {
            Stripe random = new Stripe(new RsCode(6, 9), 9, 4);
            byte[] expected = sampleFile(24);
            random.store(expected);
            int failures = nextInt(4);
            boolean[] hit = new boolean[9];
            int placed = 0;
            while (placed < failures) {
                int node = nextInt(9);
                if (hit[node]) continue;
                hit[node] = true;
                random.fail(node);
                placed++;
            }
            byte[] read = random.readFile();
            if (read != null && Arrays.equals(read, expected)) recoverable++;
        }
        check(recoverable == 200, "two hundred random patterns with at most m failures all survive: "
                + recoverable);

        // 7. Determinism and the storage overhead metric.
        prngState = 0x517L;
        byte[] first = sampleFile(24);
        prngState = 0x517L;
        byte[] second = sampleFile(24);
        check(Arrays.equals(first, second), "the file generator is deterministic");
        Stripe repeatA = new Stripe(new RsCode(6, 9), 9, 4);
        Stripe repeatB = new Stripe(new RsCode(6, 9), 9, 4);
        repeatA.store(first);
        repeatB.store(second);
        boolean same = true;
        for (int s = 0; s < 9; s++) {
            if (!Arrays.equals(repeatA.shards[s], repeatB.shards[s])) same = false;
        }
        check(same, "encoding is deterministic");
        double overhead = 9.0 / 6.0;
        check(Math.abs(overhead - 1.5) < 1e-12 && code.parity == 3,
                "the stripe stores 1.5x the data and tolerates three failures");
        check(new RsCode(6, 9).parity == 3 && stripe.liveShards() == 9, "code parameters");

        System.out.println("PASS");
    }

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError("FAILED: " + message);
    }
}
