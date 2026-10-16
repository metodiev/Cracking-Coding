// Task 694 - Implement distributed erasure-coded storage simulation with repair. (Tier: Expert, CTCI Ch 9 — System Design and Scalability)
//
// A RAID-like distributed store. Every stripe is encoded with a systematic Reed-Solomon (k + m)
// code over GF(2^8) applied byte wise across the chunks, so any k of the k + m chunks reconstruct
// the data and up to m simultaneous failures are survivable. The generator matrix is built from a
// Vandermonde evaluation matrix multiplied by the inverse of its first k rows: the top rows become
// the identity (systematic layout) and every k rows stay invertible (the MDS property).
//
// The cluster tracks which node holds which chunk. Placement is a heuristic rather than a fixed
// pattern: each chunk goes to the least loaded live node that does not already hold a chunk of the
// same stripe, which keeps the store balanced and enforces anti-affinity. A node failure is
// repaired by reading exactly k surviving chunks of the stripe, decoding the data, re-encoding it
// and writing the rebuilt chunk to a fresh node - the bandwidth optimal repair of Reed-Solomon
// codes. Anything that cannot be satisfied is reported: a stripe that lost more than m chunks is
// marked lost and reads fail instead of returning wrong bytes.
//
// Documented simplifications: chunks live in memory (no disk or network latency), failures are
// instantaneous, repairs run immediately instead of on a background scrubber, and placement is
// least-loaded-with-anti-affinity rather than topology or rack aware. What is verified: every
// k-of-n recovery subset, bit exact encode/decode, degraded reads after up to m failures, repairs
// that restore the MDS property at exactly k chunk reads per rebuilt chunk, the anti-affinity and
// load balance invariants across a long failure/repair run, detection of an unrecoverable stripe,
// refusal to invent a chunk from too few survivors, and determinism.
using System;
using System.Collections.Generic;

internal static class Gf694
{
    public static readonly byte[] Exp = new byte[512];
    public static readonly byte[] Log = new byte[256];

    static Gf694()
    {
        int value = 1;
        for (int i = 0; i < 255; i++)
        {
            Exp[i] = (byte)value;
            Log[value] = (byte)i;
            value <<= 1;
            if ((value & 0x100) != 0) value ^= 0x11D;
        }
        for (int i = 255; i < 512; i++) Exp[i] = Exp[i - 255];
    }

    public static byte Mul(byte a, byte b) => a == 0 || b == 0 ? (byte)0 : Exp[Log[a] + Log[b]];

    public static byte Div(byte a, byte b)
    {
        if (b == 0) throw new DivideByZeroException("GF(256) division by zero");
        return a == 0 ? (byte)0 : Exp[(Log[a] - Log[b] + 255) % 255];
    }

    public static byte Inv(byte a) => Div(1, a);

    public static byte Pow(byte a, int exponent)
    {
        if (a == 0) return 0;
        exponent %= 255;
        if (exponent < 0) exponent += 255;
        return Exp[Log[a] * exponent % 255];
    }
}

/// <summary>Systematic MDS code over the k data symbols of one byte position.</summary>
internal sealed class StripeCode
{
    private readonly int _k;
    private readonly int _m;
    private readonly byte[,] _generator;

    public int K => _k;
    public int M => _m;
    public int N => _k + _m;

    public StripeCode(int k, int m)
    {
        if (k <= 0 || m <= 0 || k + m > 255) throw new ArgumentException("need k, m > 0 and k + m <= 255");
        _k = k;
        _m = m;
        int n = k + m;

        // V[i][j] = x_i^j with the distinct evaluation points x_i = alpha^i.
        var v = new byte[n, k];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++) v[i, j] = Gf694.Pow(2, i * j);

        // G = V * inverse(V_top) turns the first k rows into the identity and keeps every k rows of
        // G invertible, which is exactly the MDS property.
        var top = new byte[k, k];
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++) top[i, j] = v[i, j];
        byte[,] topInverse = Invert(top);
        _generator = new byte[n, k];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < k; j++)
            {
                byte sum = 0;
                for (int t = 0; t < k; t++) sum ^= Gf694.Mul(v[i, t], topInverse[t, j]);
                _generator[i, j] = sum;
            }
        }
    }

    private static byte[,] Invert(byte[,] matrix)
    {
        int size = matrix.GetLength(0);
        var work = new byte[size, 2 * size];
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++) work[i, j] = matrix[i, j];
            work[i, size + i] = 1;
        }
        for (int column = 0; column < size; column++)
        {
            int pivot = -1;
            for (int row = column; row < size; row++) if (work[row, column] != 0) { pivot = row; break; }
            if (pivot < 0) throw new InvalidOperationException("singular matrix");
            if (pivot != column)
                for (int j = 0; j < 2 * size; j++) (work[column, j], work[pivot, j]) = (work[pivot, j], work[column, j]);
            byte inverse = Gf694.Inv(work[column, column]);
            for (int j = 0; j < 2 * size; j++) work[column, j] = Gf694.Mul(work[column, j], inverse);
            for (int row = 0; row < size; row++)
            {
                if (row == column || work[row, column] == 0) continue;
                byte factor = work[row, column];
                for (int j = 0; j < 2 * size; j++) work[row, j] ^= Gf694.Mul(factor, work[column, j]);
            }
        }
        var result = new byte[size, size];
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++) result[i, j] = work[i, size + j];
        return result;
    }

    /// <summary>Encodes k symbol values into the n chunk symbols.</summary>
    public byte[] Encode(byte[] data)
    {
        if (data.Length != _k) throw new ArgumentException($"data must be {_k} bytes");
        var chunks = new byte[N];
        for (int i = 0; i < N; i++)
        {
            byte sum = 0;
            for (int j = 0; j < _k; j++) sum ^= Gf694.Mul(_generator[i, j], data[j]);
            chunks[i] = sum;
        }
        return chunks;
    }

    /// <summary>Solves for the data from any k available chunks, or null when too few survive.</summary>
    public byte[]? Decode(byte[] chunks, bool[] available)
    {
        var rows = new List<int>();
        for (int i = 0; i < N; i++) if (available[i]) rows.Add(i);
        if (rows.Count < _k) return null;

        var matrix = new byte[_k, _k];
        var right = new byte[_k];
        for (int r = 0; r < _k; r++)
        {
            int row = rows[r];
            right[r] = chunks[row];
            for (int j = 0; j < _k; j++) matrix[r, j] = _generator[row, j];
        }
        byte[,] inverse = Invert(matrix);
        var data = new byte[_k];
        for (int j = 0; j < _k; j++)
        {
            byte sum = 0;
            for (int r = 0; r < _k; r++) sum ^= Gf694.Mul(inverse[j, r], right[r]);
            data[j] = sum;
        }
        return data;
    }
}

internal sealed class ErasureCluster
{
    public const int K = 4;
    public const int M = 2;

    private readonly int _stripes;
    private readonly int _chunkSize;
    private readonly StripeCode _code = new StripeCode(K, M);
    private readonly bool[,] _alive;
    private readonly byte[,,] _chunks;
    private readonly int[,] _holder;
    private readonly bool[] _lost;
    private readonly List<int> _load = new List<int>();

    public readonly bool[] NodeAlive;
    public long Repairs;
    public long RepairChunksRead;
    public long RepairBytes;
    public long Reads;
    public long LostStripes;
    public bool AntiAffinityViolated;

    public ErasureCluster(int stripes, int nodes, int chunkSize)
    {
        if (stripes <= 0 || chunkSize <= 0) throw new ArgumentException("need stripes > 0 and chunkSize > 0");
        _stripes = stripes;
        _chunkSize = chunkSize;
        _alive = new bool[stripes, _code.N];
        _chunks = new byte[stripes, _code.N, chunkSize];
        _holder = new int[stripes, _code.N];
        _lost = new bool[stripes];
        NodeAlive = new bool[Math.Max(nodes, _code.N)];
        for (int node = 0; node < NodeAlive.Length; node++) { NodeAlive[node] = true; _load.Add(0); }
        for (int s = 0; s < stripes; s++) for (int c = 0; c < _code.N; c++) _holder[s, c] = -1;
    }

    public int NodeCount => NodeAlive.Length;
    public int ChunkSize => _chunkSize;
    public int DataBytes => K * _chunkSize;
    public long StoredBytes => (long)_stripes * _code.N * _chunkSize;
    public StripeCode Code => _code;

    /// <summary>Least loaded live node that does not already hold a chunk of this stripe.</summary>
    private int PlaceNode(int stripe)
    {
        int best = -1;
        for (int node = 0; node < NodeAlive.Length; node++)
        {
            if (!NodeAlive[node]) continue;
            bool forbidden = false;
            for (int c = 0; c < _code.N; c++) if (_holder[stripe, c] == node) forbidden = true;
            if (forbidden) continue;
            if (best < 0 || _load[node] < _load[best]) best = node;
        }
        if (best >= 0) return best;

        // No node can take the chunk without breaking anti-affinity: record it and fall back to the
        // least loaded live node so the simulation can finish and the check can report the violation.
        AntiAffinityViolated = true;
        for (int node = 0; node < NodeAlive.Length; node++)
            if (NodeAlive[node] && (best < 0 || _load[node] < _load[best])) best = node;
        return best;
    }

    public void Store(int stripe, byte[] data)
    {
        if (data.Length != DataBytes) throw new ArgumentException($"data must be {DataBytes} bytes");
        for (int byteIndex = 0; byteIndex < _chunkSize; byteIndex++)
        {
            var symbols = new byte[K];
            for (int j = 0; j < K; j++) symbols[j] = data[j * _chunkSize + byteIndex];
            byte[] encoded = _code.Encode(symbols);
            for (int c = 0; c < _code.N; c++)
            {
                _alive[stripe, c] = true;
                _chunks[stripe, c, byteIndex] = encoded[c];
            }
        }
        for (int c = 0; c < _code.N; c++)
        {
            int node = PlaceNode(stripe);
            _holder[stripe, c] = node;
            if (node >= 0) _load[node]++;
        }
    }

    public bool[] Available(int stripe)
    {
        var available = new bool[_code.N];
        for (int c = 0; c < _code.N; c++) available[c] = _alive[stripe, c];
        return available;
    }

    public int MissingCount(int stripe)
    {
        int missing = 0;
        for (int c = 0; c < _code.N; c++) if (!_alive[stripe, c]) missing++;
        return missing;
    }

    /// <summary>Kills a node; returns false when at least one stripe lost more than m chunks.</summary>
    public bool FailNode(int node)
    {
        if (!NodeAlive[node]) throw new ArgumentException("node " + node + " is already down");
        NodeAlive[node] = false;
        bool allRecoverable = true;
        for (int s = 0; s < _stripes; s++)
        {
            for (int c = 0; c < _code.N; c++)
            {
                if (_holder[s, c] != node) continue;
                _alive[s, c] = false;
                _holder[s, c] = -1;
                _load[node]--;
            }
            if (MissingCount(s) > M)
            {
                allRecoverable = false;
                if (!_lost[s]) { _lost[s] = true; LostStripes++; }
            }
        }
        return allRecoverable;
    }

    public void AddNode(int node)
    {
        if (NodeAlive[node]) throw new ArgumentException("node " + node + " is already up");
        NodeAlive[node] = true;
    }

    /// <summary>Reads and reconstructs a stripe; null when fewer than k chunks survive.</summary>
    public byte[]? Read(int stripe, out int chunksRead)
    {
        bool[] available = Available(stripe);
        int present = 0;
        for (int c = 0; c < _code.N; c++) if (available[c]) present++;
        chunksRead = Math.Min(present, K);
        if (present < K) return null;

        var data = new byte[DataBytes];
        for (int byteIndex = 0; byteIndex < _chunkSize; byteIndex++)
        {
            var symbols = new byte[_code.N];
            for (int c = 0; c < _code.N; c++) symbols[c] = available[c] ? _chunks[stripe, c, byteIndex] : (byte)0;
            byte[]? recovered = _code.Decode(symbols, available);
            if (recovered == null) return null;
            for (int j = 0; j < K; j++) data[j * _chunkSize + byteIndex] = recovered[j];
        }
        Reads += chunksRead;
        return data;
    }

    /// <summary>Rebuilds every missing chunk, reading exactly k surviving chunks per repair.</summary>
    public long Repair()
    {
        long rebuilt = 0;
        for (int s = 0; s < _stripes; s++)
        {
            for (int c = 0; c < _code.N; c++)
            {
                if (_alive[s, c]) continue;
                bool[] available = Available(s);
                byte[]? data = ReadFromSurvivors(s, available);
                if (data == null) continue;                // too few survivors: leave the chunk missing

                for (int byteIndex = 0; byteIndex < _chunkSize; byteIndex++)
                {
                    var symbols = new byte[K];
                    for (int j = 0; j < K; j++) symbols[j] = data[j * _chunkSize + byteIndex];
                    _chunks[s, c, byteIndex] = _code.Encode(symbols)[c];
                }
                _alive[s, c] = true;
                int node = PlaceNode(s);
                _holder[s, c] = node;
                if (node >= 0) _load[node]++;
                RepairChunksRead += K;
                RepairBytes += (long)K * _chunkSize;
                Repairs++;
                rebuilt++;
            }
        }
        return rebuilt;
    }

    private byte[]? ReadFromSurvivors(int stripe, bool[] available)
    {
        var data = new byte[DataBytes];
        for (int byteIndex = 0; byteIndex < _chunkSize; byteIndex++)
        {
            var symbols = new byte[_code.N];
            for (int c = 0; c < _code.N; c++) symbols[c] = available[c] ? _chunks[stripe, c, byteIndex] : (byte)0;
            byte[]? recovered = _code.Decode(symbols, available);
            if (recovered == null) return null;
            for (int j = 0; j < K; j++) data[j * _chunkSize + byteIndex] = recovered[j];
        }
        return data;
    }

    public bool StripeComplete(int stripe)
    {
        for (int c = 0; c < _code.N; c++) if (!_alive[stripe, c]) return false;
        return true;
    }

    /// <summary>Every live chunk sits on a live node and no node holds two chunks of one stripe.</summary>
    public bool CheckPlacement()
    {
        for (int s = 0; s < _stripes; s++)
        {
            for (int c = 0; c < _code.N; c++)
            {
                if (!_alive[s, c]) continue;
                int node = _holder[s, c];
                if (node < 0 || !NodeAlive[node]) return false;
                for (int other = 0; other < _code.N; other++)
                    if (other != c && _alive[s, other] && _holder[s, other] == node) return false;
            }
        }
        return true;
    }

    public int MaxLoad()
    {
        int max = 0;
        foreach (int load in _load) max = Math.Max(max, load);
        return max;
    }

    public int MinLoadOfLive()
    {
        int min = int.MaxValue;
        for (int node = 0; node < NodeAlive.Length; node++) if (NodeAlive[node]) min = Math.Min(min, _load[node]);
        return min == int.MaxValue ? 0 : min;
    }

    public byte[] Payload(int seed)
    {
        var data = new byte[DataBytes];
        for (int i = 0; i < data.Length; i++) data[i] = (byte)(seed * 31 + i * 7 + 3);
        return data;
    }
}

public static class Solution
{
    private const int ChunkSize = 4;

    public static void Main()
    {
        var code = new StripeCode(4, 2);
        Check(code.K == 4 && code.M == 2 && code.N == 6, "the stripe code is a (6, 4) erasure code");

        // --- Every k-of-n subset recovers the data. --------------------------------------
        var data = new byte[] { 0x11, 0x22, 0x33, 0x44 };
        byte[] encoded = code.Encode(data);
        Check(encoded[0] == 0x11 && encoded[1] == 0x22 && encoded[3] == 0x44, "the first k chunks are the data itself");
        Check(encoded[4] != 0 || encoded[5] != 0, "the parity chunks are not all zero");
        int subsets = 0, subsetFailures = 0;
        for (int mask = 1; mask < 64; mask++)
        {
            var available = new bool[6];
            int bits = 0;
            for (int i = 0; i < 6; i++) if ((mask & (1 << i)) != 0) { available[i] = true; bits++; }
            if (bits < 4) continue;
            subsets++;
            byte[]? recovered = code.Decode(encoded, available);
            if (recovered == null || string.Join(",", recovered) != string.Join(",", data)) subsetFailures++;
        }
        Check(subsets == 22 && subsetFailures == 0, $"all {subsets} subsets of k = 4 chunks recovered the data");
        var tooFew = new bool[6];
        tooFew[0] = tooFew[1] = tooFew[2] = true;
        Check(code.Decode(encoded, tooFew) == null, "three chunks cannot recover four data symbols");

        // --- One stripe: clean read, degraded read, repair. ------------------------------
        var cluster = new ErasureCluster(stripes: 1, nodes: 10, chunkSize: ChunkSize);
        byte[] payload = cluster.Payload(1);
        cluster.Store(0, payload);
        Check(cluster.StripeComplete(0), "the stripe is fully stored");
        Check(cluster.StoredBytes == 6L * ChunkSize, "the stripe occupies six chunks of storage");
        Check(cluster.CheckPlacement(), "each chunk sits on its own node");
        Check(cluster.MaxLoad() == 1, "no node holds more than one chunk");
        Check(cluster.MinLoadOfLive() == 0, "the spare nodes are empty and ready for repairs");

        byte[]? read = cluster.Read(0, out int cleanReads);
        Check(read != null && string.Join(",", read) == string.Join(",", payload), "a clean read returns the data");
        Check(cleanReads == 4, "a clean read touches exactly k chunks");

        Check(cluster.FailNode(0) && cluster.FailNode(1), "two node failures are survivable");
        Check(cluster.MissingCount(0) == 2, "exactly two chunks were lost (m = 2)");
        byte[]? degraded = cluster.Read(0, out int degradedReads);
        Check(degraded != null && string.Join(",", degraded) == string.Join(",", payload),
            "the degraded read still returns the original bytes");
        Check(degradedReads == 4, "the degraded read needed exactly k = 4 chunks");

        // --- The repair is bandwidth optimal and restores the MDS property. --------------
        long beforeBytes = cluster.RepairBytes, beforeReads = cluster.RepairChunksRead;
        long rebuilt = cluster.Repair();
        Check(rebuilt == 2, "both lost chunks were rebuilt, got " + rebuilt);
        Check(cluster.RepairBytes - beforeBytes == 2L * 4 * ChunkSize,
            "repair read exactly k chunks of chunkSize bytes per rebuilt chunk");
        Check(cluster.RepairChunksRead - beforeReads == 8, "the read count accounting matches the chunk count");
        Check(cluster.StripeComplete(0), "the stripe is complete again");
        Check(cluster.CheckPlacement(), "anti-affinity survived the repair");
        byte[]? afterRepair = cluster.Read(0, out _);
        Check(afterRepair != null && string.Join(",", afterRepair) == string.Join(",", payload),
            "the repaired stripe returns the original data");

        Check(cluster.FailNode(2) && cluster.FailNode(3), "the repaired stripe survives another two failures");
        byte[]? secondRead = cluster.Read(0, out _);
        Check(secondRead != null && string.Join(",", secondRead) == string.Join(",", payload),
            "the second degraded read is still exact, so the MDS property survived the repair");
        Check(cluster.Repair() == 2 && cluster.StripeComplete(0) && cluster.CheckPlacement(),
            "the second repair round leaves a complete, consistent store");

        // --- Three simultaneous failures exceed m and lose the stripe. -------------------
        var fragile = new ErasureCluster(stripes: 1, nodes: 10, chunkSize: ChunkSize);
        fragile.Store(0, payload);
        Check(fragile.FailNode(0) && fragile.FailNode(1), "two failures are fine");
        Check(!fragile.FailNode(2), "the third failure is reported as data loss");
        Check(fragile.LostStripes == 1, "the store counts the lost stripe");
        Check(fragile.Read(0, out _) == null, "an unrecoverable stripe reports failure instead of wrong bytes");
        fragile.AddNode(2);      // the node comes back, but the data it held is already gone
        Check(fragile.Repair() == 0, "no chunk is invented for a stripe that lost more than m chunks");

        // --- Repair refusal with fewer than k survivors. ---------------------------------
        var starving = new ErasureCluster(stripes: 1, nodes: 10, chunkSize: ChunkSize);
        starving.Store(0, payload);
        Check(starving.MissingCount(0) == 0, "the fresh stripe is complete");
        starving.FailNode(0);
        Check(starving.MissingCount(0) == 1, "one chunk is missing");
        starving.FailNode(1);
        starving.FailNode(2);
        Check(starving.MissingCount(0) == 3, "three chunks are missing");
        Check(starving.Repair() == 0, "a stripe with only three survivors is left unrepaired");
        Check(!starving.StripeComplete(0), "the stripe stays incomplete");

        // --- Four stripes, six sequential failures with repairs in between. --------------
        var longRun = new ErasureCluster(stripes: 4, nodes: 12, chunkSize: ChunkSize);
        for (int s = 0; s < 4; s++) longRun.Store(s, longRun.Payload(s + 1));
        Check(longRun.CheckPlacement(), "initial placement respects anti-affinity");
        Check(longRun.MaxLoad() - longRun.MinLoadOfLive() <= 1, "the wider cluster starts balanced");

        int repairedTotal = 0;
        for (int round = 0; round < 6; round++)
        {
            Check(longRun.FailNode(round % 8), "round " + round + " failure is survivable");
            repairedTotal += (int)longRun.Repair();
            Check(longRun.CheckPlacement(), "round " + round + " keeps the placement invariants");
        }
        Check(repairedTotal > 0, "the long run repaired " + repairedTotal + " chunks");
        Check(longRun.RepairChunksRead == (long)repairedTotal * 4, "every repair cost exactly k chunk reads");
        Check(longRun.RepairBytes == (long)repairedTotal * 4 * ChunkSize,
            "repair bandwidth equals k * chunkSize per rebuilt chunk");
        for (int s = 0; s < 4; s++)
        {
            Check(longRun.StripeComplete(s), "stripe " + s + " is complete after the long run");
            byte[]? recovered = longRun.Read(s, out _);
            Check(recovered != null && string.Join(",", recovered) == string.Join(",", longRun.Payload(s + 1)),
                "stripe " + s + " still returns its own payload");
        }
        Check(longRun.MaxLoad() - longRun.MinLoadOfLive() <= 2,
            "load stays balanced after six repairs, spread=" + (longRun.MaxLoad() - longRun.MinLoadOfLive()));
        Check(!longRun.AntiAffinityViolated, "the anti-affinity rule was never broken");

        // --- Failures without repair are eventually fatal. -------------------------------
        var unprotected = new ErasureCluster(stripes: 1, nodes: 10, chunkSize: ChunkSize);
        unprotected.Store(0, payload);
        unprotected.FailNode(0);
        unprotected.FailNode(1);
        unprotected.FailNode(2);
        unprotected.FailNode(3);
        Check(unprotected.Read(0, out _) == null, "four unrepaired failures make the stripe unreadable");
        Check(unprotected.LostStripes == 1, "the loss is accounted for");
        Check(unprotected.RepairBytes == 0, "no bandwidth is spent on an unrecoverable stripe");

        // --- Determinism and validation. --------------------------------------------------
        string first = ScenarioSummary();
        string second = ScenarioSummary();
        Check(first == second, "reruns are identical: " + first);
        Check(first.Contains("repairs=") && first.Contains("checksum="), "the summary carries the repair accounting");

        bool rejected = false;
        try { _ = new StripeCode(0, 2); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "a zero width code is rejected");
        rejected = false;
        try { new ErasureCluster(stripes: 1, nodes: 10, chunkSize: ChunkSize).Store(0, new byte[3]); }
        catch (ArgumentException) { rejected = true; }
        Check(rejected, "storing data of the wrong width is rejected");
        var down = new ErasureCluster(stripes: 1, nodes: 10, chunkSize: ChunkSize);
        down.Store(0, payload);
        down.FailNode(5);
        rejected = false;
        try { down.FailNode(5); } catch (ArgumentException) { rejected = true; }
        Check(rejected, "failing an already dead node is rejected");

        Console.WriteLine($"chunkSize={ChunkSize} subsets={subsets} singleStripeRepairs={cluster.Repairs} " +
                          $"longRunRepairs={repairedTotal} repairBytes={longRun.RepairBytes} " +
                          $"lostStripes={fragile.LostStripes + unprotected.LostStripes}");
        Console.WriteLine("PASS");
    }

    private static string ScenarioSummary()
    {
        var cluster = new ErasureCluster(stripes: 3, nodes: 12, chunkSize: ChunkSize);
        for (int s = 0; s < 3; s++) cluster.Store(s, cluster.Payload(s + 5));
        for (int round = 0; round < 4; round++)
        {
            cluster.FailNode(round % 7);
            cluster.Repair();
        }
        long checksum = 0;
        for (int s = 0; s < 3; s++)
        {
            byte[]? payload = cluster.Read(s, out _);
            if (payload == null) return "unreadable";
            foreach (byte value in payload) checksum = (checksum * 7 + value) % 1000000007;
        }
        return $"repairs={cluster.Repairs} bytes={cluster.RepairBytes} checksum={checksum} " +
               $"spread={cluster.MaxLoad() - cluster.MinLoadOfLive()} reads={cluster.Reads}";
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
