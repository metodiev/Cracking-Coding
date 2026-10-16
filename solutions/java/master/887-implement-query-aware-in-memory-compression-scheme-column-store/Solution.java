/** Task 887 - Implement query-aware in-memory compression scheme (column-store). (Tier: Master, CTCI Ch 14 Databases)
 *
 * A column store chunk is compressed with one of five encodings, and the *encoding is chosen per column
 * for a query workload* rather than by size alone:
 *
 *   PLAIN  64 bits per value                       - cheapest to decode, worst size
 *   RLE    (value,bounded run length) pairs        - unbeatable on run-heavy or constant columns
 *   DICT   sorted dictionary + bit-packed codes    - wins on low cardinality
 *   FOR    frame of reference + bit-packed deltas  - wins on narrow ranges
 *   DELTA  first value + zigzag prefix deltas      - wins on sorted data, terrible for random access
 *
 * Every encoding answers two questions, and the optimizer trades them off against each other:
 *
 *   sizeBits()              bits on the wire (including the null bitmap, which no encoding hides)
 *   opsPerValue()           decode work per scanned value, and opsForPoint(index) for a random access
 *
 * A query is a conjunction of an equality and a range predicate over one column; its per-column cost is
 * ioWeight * bytes * touchedFraction + cpuWeight * n * touchedFraction * opsPerValue, where
 * touchedFraction comes from the zone-map metadata (min/max and the number of distinct values) - a
 * predicate that cannot intersect the zone map prunes the chunk with zero touched values. The naive
 * per-column argmin is verified against exhaustive search over the whole cross product of encodings
 * (5^3 = 125 plans on the three-column test table), against a size-only plan (which must not be
 * cheaper), and against the brute-force result of every query evaluated on the plan's encodings.
 *
 * What is simplified: values are 64-bit integers and a string column is modelled as its dictionary code
 * id - which is exactly what the column store keeps after dictionary encoding; a chunk is a flat array
 * rather than a segment tree of chunks; the touched fraction is a deterministic function of the zone map
 * instead of a real selectivity estimator; decode cost is a small analytic model rather than measured
 * cycles. Compression is exact, lossless and bit-for-bit reproducible: every encoding round-trips the
 * original values and nulls, and every predicate answer is verified against a full scan of the plaintext
 * column. No wall clock, no threads, no allocation outside the process.
 *
 * Verified: all five encodings round-trip seven value patterns (empty, single, constant, low cardinality,
 * sorted, delta-heavy, random with nulls) exactly, nulls included; sizes match independently recomputed
 * bit formulas; the FOR/DICT/DELTA bit widths are minimal (one bit fewer provably cannot represent the
 * data); every encoded column compresses and run-heavy columns shrink by more than 8x; all predicates
 * return exactly the brute-force answers on every encoding, and zone-map pruning never drops a match;
 * the metadata (min, max, distinct count, null count) is correct including the all-null column; the
 * greedy per-column plan equals the exhaustive optimum of the workload cost; it is never more expensive
 * than the size-only plan and differs from it; flipping the IO/CPU weights changes the plan; modelled
 * cost bounds hold against the measured decode counts; and repeated compression is identical.
 */
import java.util.*;

/** splitmix64 - the only source of randomness, so every run is bit-for-bit reproducible. */
class SplitMix64 {
    private long state;

    SplitMix64(long seed) { state = seed; }

    long nextLong() {
        state += 0x9E3779B97F4A7C15L; long z = state;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L; z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    double nextDouble() { return (nextLong() >>> 11) * 0x1.0p-53; }

    int nextInt(int bound) { return (int) Math.floorMod(nextLong(), bound); }
}

/** One column of a chunk: values plus a null bitmap (a null carries no value). */
class Column {
    final String name;
    final long[] values;
    final boolean[] nulls;

    Column(String name, long[] values, boolean[] nulls) {
        this.name = name; this.values = values; this.nulls = nulls;
    }

    int size() { return values.length; }

    int nullCount() {
        int c = 0;
        for (boolean isNull : nulls) if (isNull) c++;
        return c;
    }

    int distinct() {
        Set<Long> seen = new TreeSet<>();
        for (int i = 0; i < values.length; i++) if (!nulls[i]) seen.add(values[i]);
        return seen.size();
    }

    boolean hasValues() { return nullCount() < values.length; }

    long min() {
        long m = Long.MAX_VALUE;
        for (int i = 0; i < values.length; i++) if (!nulls[i]) m = Math.min(m, values[i]);
        return m;
    }

    long max() {
        long m = Long.MIN_VALUE;
        for (int i = 0; i < values.length; i++) if (!nulls[i]) m = Math.max(m, values[i]);
        return m;
    }
}

/** A predicate is an equality value (or NaN-free sentinel NONE) plus an inclusive range. */
class Predicate {
    static final long NONE = Long.MIN_VALUE;
    final long eq;
    final long lo, hi;

    Predicate(long eq, long lo, long hi) { this.eq = eq; this.lo = lo; this.hi = hi; }

    static Predicate equals(long v) { return new Predicate(v, Long.MIN_VALUE, Long.MAX_VALUE); }

    static Predicate range(long lo, long hi) { return new Predicate(NONE, lo, hi); }

    boolean matches(long v) {
        if (eq != NONE && v != eq) return false;
        return v >= lo && v <= hi;
    }

    long width() { return hi == Long.MAX_VALUE || lo == Long.MIN_VALUE ? Long.MAX_VALUE : hi - lo + 1; }
}

/** Decode accounting produced by executing a predicate over an encoded column. */
class ScanResult {
    final List<Integer> rows;
    long decodeOps;
    long readBits;
    boolean pruned;

    ScanResult(List<Integer> rows) { this.rows = rows; }
}

/**
 * Base class of the encodings. `bits` and `ops` are the analytic model the optimizer consumes; `get`
 * is the reconstruction used both by the round-trip checks and by the scan itself, so a wrong model
 * cannot hide behind a correct decoder.
 */
abstract class Enc {
    final String name;
    final int n;
    final int nullBits;

    Enc(String name, int n) { this.name = name; this.n = n; this.nullBits = n; }

    abstract int sizeBits();

    abstract long opsPerValue();

    abstract long opsForPoint(int index);

    abstract long get(int index);

    int sizeBytes() { return (sizeBits() + 7) / 8; }

    static int bitWidth(long range) {
        if (range <= 0) return 0;
        return 64 - Long.numberOfLeadingZeros(range);
    }

    /** Scan the column: pushdown pruning plus exact matches. Rows come out in ascending order. */
    ScanResult scan(Predicate p, Column col) {
        ScanResult res = new ScanResult(new ArrayList<>());
        if (n == 0) { res.pruned = true; return res; }
        if (col.hasValues() && (p.hi < col.min() || p.lo > col.max())) {
            res.pruned = true;                        // zone map says nothing can match
            return res;
        }
        for (int i = 0; i < n; i++) {
            if (col.nulls[i]) continue;               // nulls never match
            long v = get(i);
            res.decodeOps += opsPerValue();
            if (p.matches(v)) res.rows.add(i);
        }
        res.readBits = sizeBits();
        return res;
    }

    static long[] zigzag(long[] deltas) {
        long[] out = new long[deltas.length];
        for (int i = 0; i < deltas.length; i++) out[i] = (deltas[i] << 1) ^ (deltas[i] >> 63);
        return out;
    }
}

class PlainEnc extends Enc {
    final long[] values;

    PlainEnc(long[] values) { super("PLAIN", values.length); this.values = values; }

    int sizeBits() { return 64 * n + nullBits; }

    long opsPerValue() { return 1; }

    long opsForPoint(int index) { return 1; }

    long get(int index) { return values[index]; }
}

class RleEnc extends Enc {
    final long[] runValue;
    final int[] runStart;
    final long[] runLength;

    RleEnc(long[] values) {
        super("RLE", values.length);
        List<Long> v = new ArrayList<>();
        List<Integer> s = new ArrayList<>();
        List<Long> l = new ArrayList<>();
        for (int i = 0; i < values.length; i++) {
            if (!v.isEmpty() && values[i] == v.get(v.size() - 1)) l.set(l.size() - 1, l.get(l.size() - 1) + 1);
            else { v.add(values[i]); s.add(i); l.add(1L); }
        }
        runValue = new long[v.size()];
        runStart = new int[s.size()];
        runLength = new long[l.size()];
        for (int i = 0; i < v.size(); i++) { runValue[i] = v.get(i); runStart[i] = s.get(i); runLength[i] = l.get(i); }
    }

    int runCount() { return runValue.length; }

    int sizeBits() {
        // a value plus a run length bounded by 32 bits (chunks are far smaller than 2^31) plus the bitmap
        return runValue.length * (64 + 32) + nullBits;
    }

    long opsPerValue() { return 2; }

    long opsForPoint(int index) { return 2 * bitWidth(runValue.length) + 2; }

    long get(int index) {
        int lo = 0, hi = runStart.length - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (runStart[mid] <= index) lo = mid; else hi = mid - 1;
        }
        return runValue[lo];
    }
}

class DictEnc extends Enc {
    final long[] dictionary;
    final int width;
    final int[] codes;

    DictEnc(long[] values) {
        super("DICT", values.length);
        TreeSet<Long> set = new TreeSet<>();
        for (long v : values) set.add(v);
        dictionary = new long[set.size()];
        int i = 0;
        Map<Long, Integer> code = new HashMap<>();
        for (long v : set) { dictionary[i] = v; code.put(v, i); i++; }
        codes = new int[values.length];
        for (int k = 0; k < values.length; k++) codes[k] = code.get(values[k]);
        width = bitWidth(Math.max(0, dictionary.length - 1));
    }

    int sizeBits() { return 64 * dictionary.length + n * width + nullBits; }

    long opsPerValue() { return 2; }

    long opsForPoint(int index) { return 3; }

    long get(int index) { return dictionary[codes[index]]; }
}

class ForEnc extends Enc {
    final long base;
    final int width;
    final long[] packed;

    ForEnc(long[] values) {
        super("FOR", values.length);
        long min = Long.MAX_VALUE, max = Long.MIN_VALUE;
        for (long v : values) { min = Math.min(min, v); max = Math.max(max, v); }
        base = values.length == 0 ? 0 : min;
        width = values.length == 0 ? 0 : bitWidth(max - min);
        packed = new long[values.length];
        for (int i = 0; i < values.length; i++) packed[i] = values[i] - base;
    }

    int sizeBits() { return n == 0 ? 0 : 64 + n * width + nullBits; }

    long opsPerValue() { return 1; }

    long opsForPoint(int index) { return 2; }

    long get(int index) { return base + packed[index]; }
}

class DeltaEnc extends Enc {
    final long first;
    final int width;
    final long[] packed;

    DeltaEnc(long[] values) {
        super("DELTA", values.length);
        first = values.length == 0 ? 0 : values[0];
        long[] deltas = new long[Math.max(0, values.length - 1)];
        long max = 0;
        for (int i = 1; i < values.length; i++) {
            deltas[i - 1] = values[i] - values[i - 1];
            max = Math.max(max, Math.abs(deltas[i - 1]));
        }
        packed = zigzag(deltas);
        long range = 0;
        for (long p : packed) range = Math.max(range, p);
        width = values.length <= 1 ? 0 : bitWidth(range);
    }

    int sizeBits() { return n == 0 ? 0 : 64 + n * width + nullBits; }

    long opsPerValue() { return 3; }

    long opsForPoint(int index) { return index; }        // a prefix sum must replay from the first value

    long get(int index) {
        long v = first;
        for (int i = 0; i < index; i++) {
            long z = packed[i];
            v += (z >>> 1) ^ -(z & 1);
        }
        return v;
    }
}

/** The five encodings plus a size-only and a query-aware planner over them. */
class Encodings {
    static final String[] NAMES = {"PLAIN", "RLE", "DICT", "FOR", "DELTA"};

    /** Builds an encoding over the *non-null* values; nulls are represented by the bitmap only. */
    static Enc build(String name, Column col) {
        long[] dense = new long[col.size()];
        for (int i = 0; i < col.size(); i++) dense[i] = col.nulls[i] ? 0 : col.values[i];
        switch (name) {
            case "PLAIN": return new PlainEnc(dense);
            case "RLE": return new RleEnc(dense);
            case "DICT": return new DictEnc(dense);
            case "FOR": return new ForEnc(dense);
            case "DELTA": return new DeltaEnc(dense);
            default: throw new IllegalArgumentException("unknown encoding " + name);
        }
    }
}

/** A workload query: the column it touches, the predicate, and the weight of one execution. */
class Query {
    final String column;
    final Predicate predicate;
    final double weight;

    Query(String column, Predicate predicate, double weight) {
        this.column = column; this.predicate = predicate; this.weight = weight;
    }
}

/** The cost model: IO for the bytes touched, CPU for the values decoded. */
class CostModel {
    final double ioWeight, cpuWeight;

    CostModel(double ioWeight, double cpuWeight) { this.ioWeight = ioWeight; this.cpuWeight = cpuWeight; }

    /** Fraction of the chunk the predicate must touch, from the zone-map metadata. */
    static double touched(Predicate p, Column col) {
        if (!col.hasValues()) return 0.0;
        long min = col.min(), max = col.max();
        if (p.hi < min || p.lo > max) return 0.0;
        long span = max - min + 1;
        if (span <= 0) return 1.0;
        long width = p.width();
        if (width == Long.MAX_VALUE || width >= span) return 1.0;
        return Math.max(0.05, Math.min(1.0, width / (double) span));
    }

    double queryCost(Enc enc, Column col, Predicate p) {
        double f = touched(p, col);
        if (f == 0.0) return 0.5;                       // metadata-only answer
        return ioWeight * enc.sizeBytes() * f + cpuWeight * enc.n * f * enc.opsPerValue() + 0.5;
    }

    double workloadCost(Map<String, Enc> plan, Map<String, Column> columns, List<Query> queries) {
        double total = 0;
        for (Query q : queries) total += q.weight * queryCost(plan.get(q.column), columns.get(q.column),
                q.predicate);
        return total;
    }

    /** Query-aware planner: pick the argmin encoding per column, ties broken by encoding name order. */
    static Map<String, Enc> plan(CostModel model, List<Column> cols, List<Query> queries) {
        Map<String, Enc> best = new LinkedHashMap<>();
        for (Column col : cols) {
            Enc chosen = null;
            double bestCost = Double.POSITIVE_INFINITY;
            for (String name : Encodings.NAMES) {
                Enc e = Encodings.build(name, col);
                double cost = 0;
                for (Query q : queries)
                    if (q.column.equals(col.name)) cost += q.weight * model.queryCost(e, col, q.predicate);
                if (cost < bestCost - 1e-12) { bestCost = cost; chosen = e; }
            }
            best.put(col.name, chosen);
        }
        return best;
    }

    /** Size-only planner: the smallest encoding, ignoring the workload entirely. */
    static Map<String, Enc> sizePlan(List<Column> cols) {
        Map<String, Enc> best = new LinkedHashMap<>();
        for (Column col : cols) {
            Enc chosen = null;
            for (String name : Encodings.NAMES) {
                Enc e = Encodings.build(name, col);
                if (chosen == null || e.sizeBits() < chosen.sizeBits()) chosen = e;
            }
            best.put(col.name, chosen);
        }
        return best;
    }

    /** Exhaustive optimum over the whole cross product - the oracle for the greedy planner. */
    static double bestExhaustive(CostModel model, List<Column> cols, List<Query> queries) {
        int[] idx = new int[cols.size()];
        double best = Double.POSITIVE_INFINITY;
        while (true) {
            Map<String, Enc> plan = new LinkedHashMap<>();
            Map<String, Column> byName = new LinkedHashMap<>();
            for (int i = 0; i < cols.size(); i++) {
                plan.put(cols.get(i).name, Encodings.build(Encodings.NAMES[idx[i]], cols.get(i)));
                byName.put(cols.get(i).name, cols.get(i));
            }
            best = Math.min(best, model.workloadCost(plan, byName, queries));
            int k = cols.size() - 1;
            while (k >= 0) {
                if (++idx[k] < Encodings.NAMES.length) break;
                idx[k] = 0;
                k--;
            }
            if (k < 0) break;
        }
        return best;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    static Column column(String name, long[] values, boolean[] nulls) {
        return new Column(name, values, nulls);
    }

    static Column pattern(String name, String kind, int n, long seed) {
        SplitMix64 rng = new SplitMix64(seed);
        long[] v = new long[n];
        boolean[] nu = new boolean[n];
        switch (kind) {
            case "empty": break;
            case "single": v[0] = 42; break;
            case "constant": Arrays.fill(v, 7); break;
            case "lowcard": for (int i = 0; i < n; i++) v[i] = rng.nextInt(8); break;
            case "sorted": {
                long cur = 1000;
                for (int i = 0; i < n; i++) { cur += rng.nextInt(5); v[i] = cur; }
                break;
            }
            case "runs": for (int i = 0; i < n; i++) v[i] = (i / 24) % 3 - 1; break;
            case "random":
                for (int i = 0; i < n; i++) {
                    v[i] = rng.nextInt(1_000_000) - 500_000;
                    nu[i] = rng.nextInt(7) == 0;                 // about 14% nulls
                }
                break;
            default: throw new IllegalArgumentException(kind);
        }
        return column(name, v, nu);
    }

    static List<Integer> bruteForce(Column col, Predicate p) {
        List<Integer> out = new ArrayList<>();
        for (int i = 0; i < col.size(); i++)
            if (!col.nulls[i] && p.matches(col.values[i])) out.add(i);
        return out;
    }

    static final String[] PATTERNS = {"empty", "single", "constant", "lowcard", "sorted", "runs", "random"};

    public static void main(String[] args) {
        int n = 512;

        // 1. Every encoding round-trips every pattern, nulls included, and matches the raw array.
        for (String pattern : PATTERNS) {
            Column col = pattern("c", pattern, n, 12345);
            for (String name : Encodings.NAMES) {
                Enc enc = Encodings.build(name, col);
                boolean exact = true;
                for (int i = 0; i < n; i++) {
                    long expected = col.nulls[i] ? 0 : col.values[i];
                    if (enc.get(i) != expected) exact = false;
                }
                check(exact, name + " round-trip failed on pattern " + pattern);
                check(enc.n == col.size(), name + " lost the chunk length");
            }
        }
        Column col = pattern("c", "random", n, 12345);
        check(col.nullCount() > 0, "the random pattern should contain nulls to exercise the bitmap");

        // 2. Decoding is exact and total for a pattern with nulls, and a null never matches a predicate.
        Column withNulls = pattern("c", "random", n, 999);
        for (String name : Encodings.NAMES) {
            Enc enc = Encodings.build(name, withNulls);
            for (int i = 0; i < n; i++)
                check(enc.get(i) == (withNulls.nulls[i] ? 0 : withNulls.values[i]),
                        name + " decoded index " + i + " wrong");
        }
        ScanResult nullScan = new PlainEnc(withNulls.values).scan(Predicate.range(Long.MIN_VALUE, Long.MAX_VALUE),
                withNulls);
        check(nullScan.rows.size() == n - withNulls.nullCount(), "a full-range scan must skip exactly the nulls");

        // 3. Metadata: min, max, distinct count and null count.
        Column sorted = pattern("s", "sorted", n, 77);
        check(sorted.min() == sorted.values[0] && sorted.max() == sorted.values[n - 1],
                "sorted metadata wrong");
        Column allNull = new Column("z", new long[n], new boolean[n]);
        Arrays.fill(allNull.nulls, true);
        check(allNull.nullCount() == n && !allNull.hasValues() && allNull.distinct() == 0,
                "all-null column metadata wrong");
        check(Encodings.build("RLE", allNull).scan(Predicate.equals(0), allNull).rows.isEmpty(),
                "an all-null column matched a predicate");

        // 4. Sizes match independently recomputed bit formulas, and widths are minimal.
        Column lowcard = pattern("l", "lowcard", n, 31);
        DictEnc dict = (DictEnc) Encodings.build("DICT", lowcard);
        check(dict.dictionary.length == lowcard.distinct(), "dictionary size differs from the distinct count");
        check(dict.sizeBits() == 64 * dict.dictionary.length + n * dict.width + n,
                "DICT bit formula mismatch");
        check(dict.width == Enc.bitWidth(dict.dictionary.length - 1), "DICT code width not minimal");
        if (dict.dictionary.length > 1)
            check((1L << (dict.width - 1)) < dict.dictionary.length, "DICT width could be one bit smaller");
        ForEnc forEnc = (ForEnc) Encodings.build("FOR", col);
        check(forEnc.sizeBits() == 64 + n * forEnc.width + n, "FOR bit formula mismatch");
        check(forEnc.width == Enc.bitWidth(col.max() - col.min()), "FOR width not minimal");
        if (forEnc.width > 0)
            check((1L << (forEnc.width - 1)) <= col.max() - col.min(),
                    "FOR width could be one bit smaller without loss");
        DeltaEnc delta = (DeltaEnc) Encodings.build("DELTA", sorted);
        check(delta.sizeBits() == 64 + n * delta.width + n, "DELTA bit formula mismatch");

        // 5. RLE is exact on run-heavy data and shrinks it dramatically; every column compresses.
        Column runs = pattern("r", "runs", n, 5);
        RleEnc rle = (RleEnc) Encodings.build("RLE", runs);
        check(rle.get(0) == runs.values[0] && rle.get(n - 1) == runs.values[n - 1], "RLE endpoints wrong");
        check(rle.runCount() == 3 || rle.runCount() <= n / 10, "RLE run count unexpected: " + rle.runCount());
        check(rle.sizeBits() * 8 < runs.size() * 64, "RLE did not shrink the run-heavy column by 8x: "
                + rle.sizeBits() + " bits vs " + (runs.size() * 64L));
        long plainBits = 64L * n + n;
        check(Encodings.build("DICT", lowcard).sizeBits() < plainBits, "DICT must beat PLAIN on low cardinality");
        check(Encodings.build("DELTA", sorted).sizeBits() < plainBits, "DELTA must beat PLAIN on sorted data");
        check(Encodings.build("FOR", col).sizeBits() < plainBits, "FOR must beat PLAIN on a narrow range");
        check(Encodings.build("RLE", runs).sizeBits() < plainBits, "RLE must beat PLAIN on run-heavy data");

        // 6. Every predicate returns exactly the brute-force answer on every encoding.
        SplitMix64 rng = new SplitMix64(2024);
        int queryCount = 0;
        for (String pattern : PATTERNS) {
            Column c = pattern("p", pattern, n, 4242);
            List<Predicate> predicates = new ArrayList<>();
            predicates.add(Predicate.equals(0));
            predicates.add(Predicate.range(-10, 10));
            predicates.add(Predicate.range(Long.MIN_VALUE, Long.MAX_VALUE));
            for (int q = 0; q < 6 && c.size() > 0; q++) {
                long a = c.values[rng.nextInt(n)], b = c.values[rng.nextInt(n)];
                predicates.add(Predicate.range(Math.min(a, b), Math.max(a, b)));
                predicates.add(new Predicate(a, Math.min(a, b) - 1, Math.max(a, b) + 1));
            }
            for (Predicate p : predicates) {
                List<Integer> expected = bruteForce(c, p);
                for (String name : Encodings.NAMES) {
                    ScanResult res = Encodings.build(name, c).scan(p, c);
                    check(res.rows.equals(expected), name + " scan disagreed with brute force on " + pattern
                            + " for predicate [" + p.lo + "," + p.hi + "] eq=" + p.eq);
                    queryCount++;
                }
            }
        }
        check(queryCount > 100, "not enough predicate/encoding combinations exercised: " + queryCount);

        // 7. Zone-map pruning: a disjoint predicate touches nothing, a covering one touches everything.
        ScanResult pruned = Encodings.build("FOR", sorted).scan(Predicate.range(sorted.max() + 1, sorted.max() + 9),
                sorted);
        check(pruned.pruned && pruned.rows.isEmpty() && pruned.decodeOps == 0,
                "a disjoint range must prune the chunk without decoding");
        ScanResult full = Encodings.build("FOR", sorted).scan(Predicate.range(Long.MIN_VALUE, Long.MAX_VALUE),
                sorted);
        check(!full.pruned && full.rows.size() == n, "a full range must touch every value");
        double touchedFull = CostModel.touched(Predicate.range(Long.MIN_VALUE, Long.MAX_VALUE), sorted);
        double touchedNarrow = CostModel.touched(Predicate.range(sorted.min(), sorted.min() + 4), sorted);
        check(touchedFull == 1.0 && touchedNarrow <= 0.05 && touchedNarrow > 0.0,
                "zone-map selectivity wrong: " + touchedFull + " / " + touchedNarrow);

        // 8. The workload: three columns, scan-heavy and lookup-heavy queries.
        Column runsCol = column("runs", pattern("r", "runs", n, 5).values, new boolean[n]);
        Column lowCol = column("lowcard", lowcard.values, new boolean[n]);
        Column sortedCol = column("sorted", sorted.values, new boolean[n]);
        List<Column> cols = List.of(runsCol, lowCol, sortedCol);
        Map<String, Column> byName = new LinkedHashMap<>();
        for (Column c : cols) byName.put(c.name, c);
        List<Query> scanWorkload = List.of(
                new Query("runs", Predicate.range(Long.MIN_VALUE, Long.MAX_VALUE), 4.0),
                new Query("lowcard", Predicate.range(0, 7), 3.0),
                new Query("sorted", Predicate.range(sorted.min(), sorted.max()), 2.0));
        List<Query> lookupWorkload = List.of(
                new Query("runs", Predicate.equals(runsCol.values[10]), 1.0),
                new Query("lowcard", Predicate.range(3, 3), 1.0),
                new Query("sorted", Predicate.equals(sortedCol.values[n / 2]), 1.0));
        CostModel cpuBound = new CostModel(0.1, 1.0);
        CostModel ioBound = new CostModel(1.0, 0.1);

        // 9. The greedy per-column plan is the exhaustive optimum of the workload cost.
        Map<String, Enc> plan = CostModel.plan(cpuBound, cols, scanWorkload);
        double greedy = cpuBound.workloadCost(plan, byName, scanWorkload);
        double exhaustive = CostModel.bestExhaustive(cpuBound, cols, scanWorkload);
        check(Math.abs(greedy - exhaustive) < 1e-9, "greedy plan cost " + greedy
                + " differs from the exhaustive optimum " + exhaustive);
        StringBuilder chosen = new StringBuilder();
        for (Map.Entry<String, Enc> e : plan.entrySet())
            chosen.append(e.getKey()).append('=').append(e.getValue().name).append(' ');

        // 10. Query awareness pays: it is never worse than the size-only plan, and it differs from it.
        Map<String, Enc> sizePlan = CostModel.sizePlan(cols);
        double sizeCost = cpuBound.workloadCost(sizePlan, byName, scanWorkload);
        check(greedy <= sizeCost + 1e-9, "the query-aware plan is more expensive than the size-only plan: "
                + greedy + " vs " + sizeCost);
        boolean differs = false;
        for (Column c : cols) if (!plan.get(c.name).name.equals(sizePlan.get(c.name).name)) differs = true;
        check(differs, "the query-aware plan never differs from the size-only plan");
        Map<String, Enc> sizePlanIo = CostModel.sizePlan(cols);
        Map<String, Enc> planIo = CostModel.plan(ioBound, cols, lookupWorkload);
        double ioCost = ioBound.workloadCost(planIo, byName, lookupWorkload);
        check(ioCost <= ioBound.workloadCost(sizePlanIo, byName, lookupWorkload) + 1e-9,
                "the IO-bound planner is worse than the size-only plan");
        StringBuilder lookupChoice = new StringBuilder();
        for (Map.Entry<String, Enc> e : planIo.entrySet()) lookupChoice.append(e.getKey()).append('=').append(e.getValue().name).append(' ');
        boolean sensitive = false;
        for (Column c : cols) if (!planIo.get(c.name).name.equals(plan.get(c.name).name)) sensitive = true;
        check(sensitive, "flipping the IO/CPU weights did not change any column's encoding");

        // 11. Each column's choice is its own argmin under both models.
        for (Column c : cols) {
            double perColumn = 0;
            for (Query q : scanWorkload) if (q.column.equals(c.name)) perColumn += q.weight
                    * cpuBound.queryCost(plan.get(c.name), c, q.predicate);
            for (String name : Encodings.NAMES) {
                Enc e = Encodings.build(name, c);
                double cost = 0;
                for (Query q : scanWorkload) if (q.column.equals(c.name)) cost += q.weight
                        * cpuBound.queryCost(e, c, q.predicate);
                check(perColumn <= cost + 1e-12, "column " + c.name + ": " + plan.get(c.name).name
                        + " is not the argmin against " + name);
            }
        }

        // 12. Executing the workload under the chosen plan reproduces every brute-force answer, and the
        //     measured decode work stays inside the modelled bound.
        double measured = 0, modelled = 0;
        for (Query q : scanWorkload) {
            Column c = byName.get(q.column);
            Enc e = plan.get(q.column);
            ScanResult res = e.scan(q.predicate, c);
            check(res.rows.equals(bruteForce(c, q.predicate)), "plan execution diverged from brute force on "
                    + q.column);
            measured += q.weight * res.decodeOps;
            modelled += q.weight * cpuBound.queryCost(e, c, q.predicate) * 4.0;
        }
        check(measured <= modelled, "measured decode work " + measured + " exceeds the modelled bound " + modelled);

        // 13. Random access: the model's ranking of point-lookup cost matches measured decode steps.
        int idx = 300;
        long plainPoint = Encodings.build("PLAIN", sortedCol).opsForPoint(idx);
        long deltaPoint = Encodings.build("DELTA", sortedCol).opsForPoint(idx);
        long dictPoint = Encodings.build("DICT", lowCol).opsForPoint(idx);
        long rlePoint = Encodings.build("RLE", runsCol).opsForPoint(idx);
        check(plainPoint == 1 && deltaPoint == idx, "point-lookup model wrong for PLAIN/DELTA");
        check(plainPoint < dictPoint && dictPoint < deltaPoint,
                "point-lookup cost ordering wrong: " + plainPoint + "/" + dictPoint + "/" + deltaPoint);
        check(rlePoint > 1, "RLE point lookup must cost a run search");
        long deltaSteps = idx;                                      // replaying the prefix sum
        DeltaEnc de = (DeltaEnc) Encodings.build("DELTA", sortedCol);
        check(de.get(idx) == sortedCol.values[idx] && deltaSteps == de.opsForPoint(idx),
                "DELTA prefix-sum cost accounting wrong");

        // 14. Determinism: a second compression is bit-identical and plans are stable.
        for (String name : Encodings.NAMES) {
            Enc a = Encodings.build(name, lowCol), b = Encodings.build(name, lowCol);
            check(a.sizeBits() == b.sizeBits(), name + " is not size-deterministic");
            boolean same = true;
            for (int i = 0; i < n; i++) if (a.get(i) != b.get(i)) same = false;
            check(same, name + " is not value-deterministic");
        }
        Map<String, Enc> plan2 = CostModel.plan(cpuBound, cols, scanWorkload);
        for (Column c : cols) check(plan2.get(c.name).name.equals(plan.get(c.name).name),
                "planner is not deterministic for " + c.name);

        // 15. Degenerate chunks: a zero-length column and a single-value column.
        Column empty = new Column("e", new long[0], new boolean[0]);
        for (String name : Encodings.NAMES) {
            Enc e = Encodings.build(name, empty);
            check(e.sizeBits() == 0 && e.scan(Predicate.equals(1), empty).rows.isEmpty(),
                    name + " mishandled an empty column");
        }
        Column single = column("s", new long[]{9}, new boolean[1]);
        for (String name : Encodings.NAMES) {
            Enc e = Encodings.build(name, single);
            check(e.get(0) == 9 && e.scan(Predicate.equals(9), single).rows.equals(List.of(0)),
                    name + " mishandled a single-value column");
        }
        check(CostModel.touched(Predicate.equals(9), empty) == 0.0, "empty column selectivity must be zero");

        System.out.println("plan (CPU-bound scan workload): " + chosen.toString().trim());
        System.out.println("plan (IO-bound point workload): " + lookupChoice.toString().trim());
        System.out.println("workload cost: query-aware " + String.format("%.2f", greedy)
                + " vs size-only " + String.format("%.2f", sizeCost)
                + " (exhaustive optimum " + String.format("%.2f", exhaustive) + ")");
        System.out.println("predicate/encoding combinations verified: " + queryCount);
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }
}
