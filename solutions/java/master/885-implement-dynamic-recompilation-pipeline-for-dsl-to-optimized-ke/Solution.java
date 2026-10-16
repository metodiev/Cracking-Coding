/** Task 885 - Implement dynamic recompilation pipeline for DSL to optimized kernels with heuristics. (Tier: Master, CTCI Beyond - Compilers, Languages & Solvers)
 *
 * A small kernel DSL is parsed into an AST, lowered once into three-address code (TAC) and then
 * *recompiled* per concrete shape, exactly like a JIT with tiered compilation:
 *
 *   tier 0  tree-walking interpreter over the AST - the reference semantics, never optimized
 *   tier 1  one generic TAC plan per kernel (constant folding + loop-invariant hoisting)
 *   tier 2  an autotuned specialization cached under a shape key (trip-count bucket, base
 *           alignment, array layout), so every distinct shape pays exactly one recompilation and
 *           every repeat of that shape is a cache hit
 *
 * The heuristics applied to a TAC body are constant folding, dead-code elimination, loop-invariant
 * code motion, strength reduction (multiply/divide by a power of two becomes an exact Math.scalb,
 * multiply by one becomes a copy) and selective unrolling (1/2/4/8). The autotuner enumerates the 32
 * point transform grid, rejects every candidate whose output differs from tier 0 bit-for-bit, scores
 * the survivors with a cycle model (op weight divided by vector width, plus 3 per memory op, plus 2
 * per loop-control step, plus a register-pressure term) and keeps the cheapest, ties broken by
 * config order.
 *
 * What is simplified. One loop nest level with a single induction variable; the first kernel
 * parameter is the trip count and is bound from the shape; scalar (non-array) variables are
 * caller-provided accumulators that the kernel may update - because such a loop-carried scalar blocks
 * hoisting, the invariance analysis treats any scalar a `W` write touches as clobbered. Pointers,
 * aliasing and type inference are ignored: an array access is checked for bounds and that is all.
 * "Vectorization" is decided by contiguity analysis of the index expressions (i + c is contiguous)
 * and enters the cost model as a divisor only - the VM still executes lane by lane, so numerics stay
 * untouched. Unrolling is modelled at loop-control level (the body runs k times per pass, collapsing
 * 2k compare/increment steps into two), which is why it moves the cost model and the loop-step
 * counter but not the data-op count. Trip counts are bucketed to the next power of two for the cache
 * key. Every transform is value preserving by construction - folding evaluates the very same IEEE
 * operation at compile time, hoisting moves a pure computation that provably does not depend on the
 * loop index, scalb and the multiply-by-one copy are exact, and unrolling only replicates the
 * iteration body - which is what makes the bit-for-bit gate a real correctness check. Everything is
 * single-threaded, uses no wall-clock time and no randomness.
 *
 * Verified: the parser accepts the benchmark kernels, round-trips its canonical dump and rejects
 * malformed or unbound input; the interpreter matches hand-computed values and an independent
 * windowed-convolution oracle; all 32 transform configurations reproduce tier 0 bit-for-bit on five
 * kernels and two alignments; folding, LICM, strength reduction and unrolling each measurably reduce
 * the modelled cost while leaving results identical; LICM demonstrably shrinks both the static body
 * and the dynamic instruction count; unrolling collapses loop-control steps while the remainder loop
 * still handles a trip count that is not a multiple of the unroll factor; contiguity drives the
 * vector width; the scheduled plan is really the argmin of the cost model; each new shape triggers
 * exactly one recompilation and repeats are cache hits; tier promotion follows the policy and every
 * tier agrees bit-for-bit with the interpreter; degenerate shapes (0 iterations, one element, an
 * empty stencil range) and out-of-range writes behave; and a randomized shape sequence is
 * reproducible and stays bit-exact while using far fewer instructions than the interpreter.
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

/** One node for the whole expression grammar: number, variable, array element, binary op, negation. */
class Expr {
    static final int NUM = 0, VAR = 1, LOAD = 2, BIN = 3, NEG = 4;
    final int kind;
    final char op;
    final String name;
    final double value;
    final Expr l, r, idx;

    private Expr(int kind, char op, String name, double value, Expr l, Expr r, Expr idx) {
        this.kind = kind; this.op = op; this.name = name; this.value = value;
        this.l = l; this.r = r; this.idx = idx;
    }

    static Expr num(double v) { return new Expr(NUM, ' ', null, v, null, null, null); }
    static Expr var(String n) { return new Expr(VAR, ' ', n, 0, null, null, null); }
    static Expr load(String n, Expr i) { return new Expr(LOAD, ' ', n, 0, null, null, i); }
    static Expr bin(char o, Expr a, Expr b) { return new Expr(BIN, o, null, 0, a, b, null); }
    static Expr neg(Expr a) { return new Expr(NEG, ' ', null, 0, a, null, null); }

    String show() {
        switch (kind) {
            case NUM: return value == Math.rint(value) && Math.abs(value) < 1e15
                    ? String.valueOf((long) value) : String.valueOf(value);
            case VAR: return name;
            case LOAD: return name + "[" + idx.show() + "]";
            case NEG: return "-" + l.show();
            default: return "(" + l.show() + " " + op + " " + r.show() + ")";
        }
    }
}

/** One statement of a kernel body: `arr[idx] = value;` or `name = value;`. */
class Stmt {
    final String name;
    final Expr index;
    final Expr value;

    Stmt(String name, Expr index, Expr value) { this.name = name; this.index = index; this.value = value; }

    String show() {
        return (index == null ? name : name + "[" + index.show() + "]") + " = " + value.show() + ";";
    }
}

/** A parsed kernel: `kernel name(a, b) { for i in LO..HI { body } }`. */
class Kernel {
    final String name;
    final List<String> params;
    final String loopVar;
    final Expr lo, hi;
    final List<Stmt> body;
    final boolean contiguous;               // every array index is of the form loopVar + c

    Kernel(String name, List<String> params, String loopVar, Expr lo, Expr hi, List<Stmt> body) {
        this.name = name; this.params = params; this.loopVar = loopVar;
        this.lo = lo; this.hi = hi; this.body = body;
        boolean ok = true;
        for (Stmt s : body) {
            if (s.index != null) ok &= isContiguous(s.index, loopVar);
            ok &= indexExprs(s.value, loopVar);
        }
        this.contiguous = ok;
    }

    private static boolean indexExprs(Expr e, String loopVar) {
        if (e == null) return true;
        if (e.kind == Expr.LOAD) return isContiguous(e.idx, loopVar) && indexExprs(e.idx, loopVar);
        return indexExprs(e.l, loopVar) && indexExprs(e.r, loopVar) && indexExprs(e.idx, loopVar);
    }

    /** True when the index is a stride-1 affine form: loopVar + c, c + loopVar or loopVar - c. */
    static boolean isContiguous(Expr e, String loopVar) {
        if (e.kind == Expr.VAR) return e.name.equals(loopVar);
        if (e.kind != Expr.BIN) return false;
        if (e.op == '+' || e.op == '-') {
            boolean left = isContiguous(e.l, loopVar) && isConst(e.r);
            boolean right = e.op == '+' && isConst(e.l) && isContiguous(e.r, loopVar);
            return left || right;
        }
        return false;
    }

    static boolean isConst(Expr e) {
        return e.kind == Expr.NUM || (e.kind == Expr.NEG && e.l.kind == Expr.NUM);
    }

    String show() {
        StringBuilder sb = new StringBuilder("kernel ").append(name).append('(');
        for (int i = 0; i < params.size(); i++) sb.append(i == 0 ? "" : ",").append(params.get(i));
        sb.append(") { for ").append(loopVar).append(" in ").append(lo.show()).append("..").append(hi.show())
                .append(" {");
        for (Stmt s : body) sb.append(' ').append(s.show());
        return sb.append(" } }").toString();
    }
}

/** Recursive-descent parser for the DSL; every error is an IllegalArgumentException. */
class Parser {
    private final String src;
    private int pos;

    private Parser(String src) { this.src = src; }

    static Kernel parse(String src) {
        Parser p = new Parser(src);
        Kernel k = p.program();
        p.ws();
        if (p.pos != src.length()) throw new IllegalArgumentException("trailing input at offset " + p.pos);
        return k;
    }

    private void ws() { while (pos < src.length() && Character.isWhitespace(src.charAt(pos))) pos++; }

    private boolean eat(String tok) {
        ws();
        if (src.startsWith(tok, pos)) { pos += tok.length(); return true; }
        return false;
    }

    private void need(String tok) {
        if (!eat(tok)) throw new IllegalArgumentException("expected '" + tok + "' at offset " + pos);
    }

    private String ident() {
        ws();
        int s = pos;
        while (pos < src.length() && (Character.isLetterOrDigit(src.charAt(pos)) || src.charAt(pos) == '_')) pos++;
        if (s == pos) throw new IllegalArgumentException("expected identifier at offset " + pos);
        return src.substring(s, pos);
    }

    /** A number stops before a `..` range token, so `0..n` lexes as 0, "..", n. */
    private double number() {
        ws();
        int s = pos;
        boolean dot = false;
        while (pos < src.length()) {
            char c = src.charAt(pos);
            if (Character.isDigit(c)) pos++;
            else if (c == '.' && !dot && pos + 1 < src.length() && Character.isDigit(src.charAt(pos + 1))) {
                dot = true; pos++;
            } else break;
        }
        if (s == pos) throw new IllegalArgumentException("expected number at offset " + pos);
        return Double.parseDouble(src.substring(s, pos));
    }

    private Kernel program() {
        need("kernel");
        String name = ident();
        need("(");
        List<String> params = new ArrayList<>();
        params.add(ident());
        while (eat(",")) params.add(ident());
        need(")");
        need("{");
        need("for");
        String v = ident();
        need("in");
        Expr lo = expr();
        need("..");
        Expr hi = expr();
        need("{");
        List<Stmt> body = new ArrayList<>();
        while (!eat("}")) body.add(stmt());
        need("}");
        if (body.isEmpty()) throw new IllegalArgumentException("empty kernel body");
        return new Kernel(name, params, v, lo, hi, body);
    }

    private Stmt stmt() {
        String name = ident();
        if (eat("[")) {
            Expr idx = expr();
            need("]");
            need("=");
            Expr v = expr();
            need(";");
            return new Stmt(name, idx, v);
        }
        need("=");
        Expr v = expr();
        need(";");
        return new Stmt(name, null, v);
    }

    private Expr expr() {
        Expr a = term();
        while (true) {
            ws();
            if (eat("+")) a = Expr.bin('+', a, term());
            else if (eat("-")) a = Expr.bin('-', a, term());
            else return a;
        }
    }

    private Expr term() {
        Expr a = factor();
        while (true) {
            ws();
            if (eat("*")) a = Expr.bin('*', a, factor());
            else if (eat("/")) a = Expr.bin('/', a, factor());
            else return a;
        }
    }

    private Expr factor() {
        ws();
        if (eat("-")) return Expr.neg(factor());
        if (eat("(")) { Expr e = expr(); need(")"); return e; }
        if (pos >= src.length()) throw new IllegalArgumentException("unexpected end of input");
        if (Character.isDigit(src.charAt(pos))) return Expr.num(number());
        String n = ident();
        if (eat("[")) { Expr i = expr(); need("]"); return Expr.load(n, i); }
        return Expr.var(n);
    }
}

/**
 * One three-address instruction. Operands are register slots; `arr` names the array or scalar
 * binding of a memory op.  'C' dst = constant, 'M' dst = reg[a], 'L' dst = arr[a], 'S' arr[a] = reg[b],
 * 'W' scalar arr = reg[a], 'N' dst = -reg[a], 'X' dst = scalb(reg[a], b), else dst = reg[a] op reg[b].
 */
class Insn {
    final char op;
    final int dst, a, b;
    final String arr;
    final double cval;

    Insn(char op, int dst, int a, int b, String arr, double cval) {
        this.op = op; this.dst = dst; this.a = a; this.b = b; this.arr = arr; this.cval = cval;
    }

    boolean isArith() { return op == '+' || op == '-' || op == '*' || op == '/' || op == 'N' || op == 'X'; }

    String show() {
        switch (op) {
            case 'C': return "C r" + dst + " " + cval;
            case 'M': return "M r" + dst + " r" + a;
            case 'L': return "L r" + dst + " " + arr + "[r" + a + "]";
            case 'S': return "S " + arr + "[r" + a + "] r" + b;
            case 'W': return "W " + arr + " r" + a;
            case 'N': return "N r" + dst + " r" + a;
            case 'X': return "X r" + dst + " r" + a + " " + b;
            default: return op + " r" + dst + " r" + a + " r" + b;
        }
    }
}

/** The shape a kernel is specialized for and the cache key derived from it. */
class Shape {
    final int n, align;

    Shape(int n, int align) { this.n = n; this.align = align; }

    int bucket() { int p = 1; while (p < n) p <<= 1; return p; }

    String key(String kernel) { return kernel + "|n" + bucket() + "|al" + Math.floorMod(align, 4); }
}

/** A compiled kernel: a prologue (bounds plus loop-invariant code) and one loop body. */
class Plan {
    final String config;
    final List<Insn> prologue, body;
    final int loSlot, hiSlot, indexSlot, slots, unroll, vectorWidth;
    final Map<String, Integer> scalarSlots;

    Plan(String config, List<Insn> prologue, List<Insn> body, int loSlot, int hiSlot, int indexSlot,
            int slots, int unroll, int vectorWidth, Map<String, Integer> scalarSlots) {
        this.config = config; this.prologue = prologue; this.body = body;
        this.loSlot = loSlot; this.hiSlot = hiSlot; this.indexSlot = indexSlot; this.slots = slots;
        this.unroll = unroll; this.vectorWidth = vectorWidth; this.scalarSlots = scalarSlots;
    }

    String bodyDump() {
        StringBuilder sb = new StringBuilder();
        for (Insn in : body) sb.append(in.show()).append('\n');
        return sb.toString();
    }
}

/** Result of one kernel invocation. */
class Run {
    final Map<String, double[]> arrays, scalars;
    final long steps, loopIters, memOps;
    final int tier, vectorWidth;
    final String config;
    final double modeledCost;

    Run(Map<String, double[]> arrays, Map<String, double[]> scalars, long steps, long loopIters, long memOps,
            int tier, int vectorWidth, String config, double modeledCost) {
        this.arrays = arrays; this.scalars = scalars; this.steps = steps; this.loopIters = loopIters;
        this.memOps = memOps; this.tier = tier; this.vectorWidth = vectorWidth;
        this.config = config; this.modeledCost = modeledCost;
    }
}

/** Tier 0: the reference tree-walking interpreter. */
class Interp {
    static Map<String, double[]> copy(Map<String, double[]> in) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, double[]> e : in.entrySet()) out.put(e.getKey(), e.getValue().clone());
        return out;
    }

    static long run(Kernel k, Map<String, double[]> arrays, Map<String, double[]> scalars) {
        long[] steps = {0};
        long lo = (long) eval(k.lo, k, arrays, scalars, 0, steps);
        long hi = (long) eval(k.hi, k, arrays, scalars, 0, steps);
        for (long i = lo; i < hi; i++) {
            for (Stmt s : k.body) {
                double v = eval(s.value, k, arrays, scalars, i, steps);
                if (s.index == null) {
                    double[] box = scalars.get(s.name);
                    if (box == null) throw new IllegalArgumentException("unbound scalar " + s.name);
                    box[0] = v;
                } else {
                    long idx = (long) eval(s.index, k, arrays, scalars, i, steps);
                    double[] arr = arrays.get(s.name);
                    if (arr == null) throw new IllegalArgumentException("unbound array " + s.name);
                    if (idx < 0 || idx >= arr.length)
                        throw new IndexOutOfBoundsException("array " + s.name + " index " + idx
                                + " outside [0," + arr.length + ")");
                    arr[(int) idx] = v;
                }
            }
        }
        return steps[0];
    }

    private static double eval(Expr e, Kernel k, Map<String, double[]> arrays, Map<String, double[]> scalars,
            long i, long[] steps) {
        steps[0]++;
        switch (e.kind) {
            case Expr.NUM: return e.value;
            case Expr.VAR:{
                if (e.name.equals(k.loopVar)) return i;
                double[] box = scalars.get(e.name);
                if (box == null) throw new IllegalArgumentException("unbound variable " + e.name);
                return box[0];
            }
            case Expr.LOAD:{
                double[] arr = arrays.get(e.name);
                if (arr == null) throw new IllegalArgumentException("unbound array " + e.name);
                long idx = (long) eval(e.idx, k, arrays, scalars, i, steps);
                if (idx < 0 || idx >= arr.length)
                    throw new IndexOutOfBoundsException("array " + e.name + " index " + idx
                            + " outside [0," + arr.length + ")");
                return arr[(int) idx];
            }
            case Expr.NEG: return -eval(e.l, k, arrays, scalars, i, steps);
            default:{
                double a = eval(e.l, k, arrays, scalars, i, steps);
                double b = eval(e.r, k, arrays, scalars, i, steps);
                switch (e.op) {
                    case '+': return a + b;
                    case '-': return a - b;
                    case '*': return a * b;
                    default: return a / b;
                }
            }
        }
    }
}

/** Lowers an AST body to TAC and runs the optimization passes. */
class Compiler {
    private final Kernel k;
    private final Map<String, Integer> scalarSlots = new LinkedHashMap<>();
    private int next = 1, maxSlot = 1, indexSlot;
    private final List<Insn> code = new ArrayList<>();

    private Compiler(Kernel k) { this.k = k; }

    private int slot() { next++; maxSlot = Math.max(maxSlot, next); return next; }

    private int constant(double v) {
        int s = slot();
        code.add(new Insn('C', s, 0, 0, null, v));
        return s;
    }

    private int ex(Expr e) {
        switch (e.kind) {
            case Expr.NUM: return constant(e.value);
            case Expr.NEG:{
                int a = ex(e.l);
                int d = slot();
                code.add(new Insn('N', d, a, 0, null, 0));
                return d;
            }
            case Expr.VAR:{
                if (e.name.equals(k.loopVar)) return indexSlot;
                Integer s = scalarSlots.get(e.name);
                if (s == null) {
                    if (!k.params.contains(e.name))
                        throw new IllegalArgumentException("unbound variable " + e.name);
                    s = slot();
                    scalarSlots.put(e.name, s);
                }
                int d = slot();
                code.add(new Insn('M', d, s, 0, null, 0));
                return d;
            }
            case Expr.LOAD:{
                int i = ex(e.idx);
                int d = slot();
                code.add(new Insn('L', d, i, 0, e.name, 0));
                return d;
            }
            default:{
                int a = ex(e.l), b = ex(e.r);
                int d = slot();
                code.add(new Insn(e.op, d, a, b, null, 0));
                return d;
            }
        }
    }

    /** Compiles one kernel for one shape under the requested transform configuration. */
    static Plan compile(Kernel k, Shape shape, boolean fold, boolean licm, boolean strength, int unroll) {
        Compiler c = new Compiler(k);
        c.indexSlot = c.slot();
        for (String p : k.params) if (!c.scalarSlots.containsKey(p)) c.scalarSlots.put(p, c.slot());
        for (Stmt s : k.body) if (s.index == null && !c.scalarSlots.containsKey(s.name))
            c.scalarSlots.put(s.name, c.slot());
        int loSlot = c.ex(k.lo);
        int hiSlot = c.ex(k.hi);
        List<Insn> pro = new ArrayList<>(c.code);
        c.code.clear();
        for (Stmt s : k.body) {
            int v = c.ex(s.value);
            if (s.index == null) c.code.add(new Insn('W', 0, v, 0, s.name, 0));
            else c.code.add(new Insn('S', 0, c.ex(s.index), v, s.name, 0));
        }
        List<Insn> body = c.code;
        Set<Integer> clobbered = new HashSet<>();
        for (Insn in : body) {
            if (in.op == 'W') {
                Integer s = c.scalarSlots.get(in.arr);
                if (s != null) clobbered.add(s);
            }
        }
        // Caller-provided scalars are constants for the whole loop unless the body writes them,
        // so their slots seed the invariance analysis.
        Set<Integer> loopConstants = new HashSet<>(c.scalarSlots.values());
        loopConstants.removeAll(clobbered);
        if (fold) body = fold(body);
        if (strength) body = strengthReduce(body);
        if (fold || strength) body = dce(body);
        if (licm) body = licm(body, pro, c.indexSlot, clobbered, loopConstants);
        int vectorWidth = k.contiguous ? (Math.floorMod(shape.align, 4) == 0 ? 4 : 2) : 1;
        String cfg = (fold ? "F" : "-") + (licm ? "L" : "-") + (strength ? "S" : "-") + "u" + unroll;
        return new Plan(cfg, pro, body, loSlot, hiSlot, c.indexSlot, c.maxSlot + 1, unroll, vectorWidth,
                c.scalarSlots);
    }

    /**
     * Folds arithmetic whose operands are known constants. Every fold performs exactly the runtime
     * IEEE operation at compile time, so folded code is bit-identical to unfolded code.
     */
    private static List<Insn> fold(List<Insn> code) {
        List<Insn> out = new ArrayList<>(code);
        Map<Integer, Double> val = new HashMap<>();
        for (Insn in : out) if (in.op == 'C') val.put(in.dst, in.cval);
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int i = 0; i < out.size(); i++) {
                Insn in = out.get(i);
                Double x = val.get(in.a);
                Double y = val.get(in.b);
                if (in.op == 'M' && x != null) {
                    out.set(i, new Insn('C', in.dst, 0, 0, null, x));
                    val.put(in.dst, x);
                    changed = true;
                } else if (in.isArith() && x != null && (in.op == 'N' || in.op == 'X' || y != null)) {
                    double v;
                    if (in.op == 'N') v = -x;
                    else if (in.op == 'X') v = Math.scalb(x, in.b);
                    else if (in.op == '+') v = x + y;
                    else if (in.op == '-') v = x - y;
                    else if (in.op == '*') v = x * y;
                    else v = x / y;
                    out.set(i, new Insn('C', in.dst, 0, 0, null, v));
                    val.put(in.dst, v);
                    changed = true;
                }
            }
        }
        return out;
    }

    /** Scaling by a power of two becomes an exact Math.scalb; multiplying by one becomes a copy. */
    private static List<Insn> strengthReduce(List<Insn> code) {
        List<Insn> out = new ArrayList<>(code);
        Map<Integer, Double> val = new HashMap<>();
        for (Insn in : out) if (in.op == 'C') val.put(in.dst, in.cval);
        for (int i = 0; i < out.size(); i++) {
            Insn in = out.get(i);
            if (in.op != '*' && in.op != '/') continue;
            int other, constSlot;
            Double c = val.get(in.b);
            if (c != null) { constSlot = in.b; other = in.a; }
            else if (in.op == '*' && (c = val.get(in.a)) != null) { constSlot = in.a; other = in.b; }
            else continue;
            if (c == 0.0) continue;
            double factor = in.op == '/' ? 1.0 / c : c;
            int exp = powerOfTwoExponent(factor);
            if (exp == Integer.MIN_VALUE) continue;
            out.set(i, exp == 0 ? new Insn('M', in.dst, other, 0, null, 0)
                    : new Insn('X', in.dst, other, exp, null, 0));
            val.remove(constSlot);
        }
        return out;
    }

    /** Returns the exponent e with 2^e == v, or Integer.MIN_VALUE when v is not a signed power of two. */
    static int powerOfTwoExponent(double v) {
        if (v == 0.0 || Double.isNaN(v) || Double.isInfinite(v)) return Integer.MIN_VALUE;
        int e = (int) Math.rint(Math.log(Math.abs(v)) / Math.log(2.0));
        if (e < -1074 || e > 1023) return Integer.MIN_VALUE;
        return Math.scalb(1.0, e) == v ? e : Integer.MIN_VALUE;
    }

    /** Removes pure instructions whose destination no array or scalar write ever observes. */
    static List<Insn> dce(List<Insn> code) {
        Set<Integer> used = new HashSet<>();
        used.add(0);
        List<Insn> rev = new ArrayList<>();
        for (int i = code.size() - 1; i >= 0; i--) {
            Insn in = code.get(i);
            if (in.op == 'S' || in.op == 'W') {
                rev.add(in);
                used.add(in.a);
                if (in.op == 'S') used.add(in.b);
                continue;
            }
            if (used.remove(in.dst)) {
                rev.add(in);
                used.add(in.a);
                if (in.isArith()) used.add(in.b);
            }
        }
        Collections.reverse(rev);
        return rev;
    }

    /**
     * Hoists loop-invariant pure instructions into the prologue, preserving their relative order.
     * Array loads and stores are never hoisted (the alias analysis is deliberately absent), and a
     * scalar that the body writes - a loop-carried accumulator - is never treated as invariant.
     */
    private static List<Insn> licm(List<Insn> body, List<Insn> prologue, int indexSlot, Set<Integer> clobbered,
            Set<Integer> loopConstants) {
        Set<Integer> invariant = new HashSet<>(loopConstants);
        invariant.add(0);
        for (Insn in : prologue) if (in.op == 'C') invariant.add(in.dst);
        List<Insn> loop = new ArrayList<>(body);
        boolean changed = true;
        while (changed) {
            changed = false;
            for (int i = 0; i < loop.size(); i++) {
                Insn in = loop.get(i);
                if (in.op == 'L' || in.op == 'S' || in.op == 'W') continue;
                if (in.dst == 0 || in.dst == indexSlot || clobbered.contains(in.dst)) continue;
                if (clobbered.contains(in.a)) continue;
                boolean ok = invariant.contains(in.a)
                        && (in.isArith() ? in.op == 'X' || invariant.contains(in.b) : true);
                if (!ok) continue;
                loop.remove(i);
                prologue.add(in);
                invariant.add(in.dst);
                changed = true;
                break;
            }
        }
        return loop;
    }
}

/** The VM: runs a plan lane by lane and accounts for every instruction it issues. */
class Vm {
    static Run exec(Plan p, Map<String, double[]> arrays, Map<String, double[]> scalars) {
        double[] reg = new double[Math.max(p.slots, 2)];
        for (Map.Entry<String, Integer> e : p.scalarSlots.entrySet()) {
            double[] box = scalars.get(e.getKey());
            if (box == null) throw new IllegalArgumentException("missing binding for scalar " + e.getKey());
            reg[e.getValue()] = box[0];
        }
        long[] ctr = {0, 0, 0};                       // steps, loop iterations, memory ops
        for (Insn in : p.prologue) execute(in, reg, arrays, scalars, p, ctr);
        long lo = (long) reg[p.loSlot], hi = (long) reg[p.hiSlot];
        int k = Math.max(1, p.unroll);
        long i = lo;
        while (i + k <= hi) {
            ctr[1]++;
            for (int j = 0; j < k; j++) {
                reg[p.indexSlot] = i + j;
                for (Insn in : p.body) execute(in, reg, arrays, scalars, p, ctr);
            }
            i += k;
        }
        while (i < hi) {
            ctr[1]++;
            reg[p.indexSlot] = i;
            for (Insn in : p.body) execute(in, reg, arrays, scalars, p, ctr);
            i++;
        }
        return new Run(arrays, scalars, ctr[0], ctr[1], ctr[2], 0, p.vectorWidth, p.config, Double.NaN);
    }

    private static void execute(Insn in, double[] reg, Map<String, double[]> arrays,
            Map<String, double[]> scalars, Plan plan, long[] ctr) {
        ctr[0]++;
        switch (in.op) {
            case 'C': reg[in.dst] = in.cval; return;
            case 'M': reg[in.dst] = reg[in.a]; return;
            case 'L': reg[in.dst] = load(arrays, in.arr, (int) reg[in.a]); ctr[2]++; return;
            case 'S': store(arrays, in.arr, (int) reg[in.a], reg[in.b]); ctr[2]++; return;
            case 'W': {
                double[] box = scalars.get(in.arr);
                if (box == null) throw new IllegalArgumentException("unbound scalar " + in.arr);
                box[0] = reg[in.a];
                // Publish the new value into the register too, so the next iteration reads the
                // updated accumulator exactly like the interpreter does.
                Integer home = plan.scalarSlots.get(in.arr);
                if (home != null) reg[home] = reg[in.a];
                return;
            }
            case 'N': reg[in.dst] = -reg[in.a]; return;
            case 'X': reg[in.dst] = Math.scalb(reg[in.a], in.b); return;
            case '+': reg[in.dst] = reg[in.a] + reg[in.b]; return;
            case '-': reg[in.dst] = reg[in.a] - reg[in.b]; return;
            case '*': reg[in.dst] = reg[in.a] * reg[in.b]; return;
            default: reg[in.dst] = reg[in.a] / reg[in.b];
        }
    }

    private static double load(Map<String, double[]> arrays, String name, int idx) {
        double[] arr = arrays.get(name);
        if (arr == null) throw new IllegalArgumentException("unbound array " + name);
        if (idx < 0 || idx >= arr.length)
            throw new IndexOutOfBoundsException("array " + name + " index " + idx + " outside [0," + arr.length + ")");
        return arr[idx];
    }

    private static void store(Map<String, double[]> arrays, String name, int idx, double v) {
        double[] arr = arrays.get(name);
        if (arr == null) throw new IllegalArgumentException("unbound array " + name);
        if (idx < 0 || idx >= arr.length)
            throw new IndexOutOfBoundsException("array " + name + " index " + idx + " outside [0," + arr.length + ")");
        arr[idx] = v;
    }
}

/** The pipeline: registration, tier promotion, shape caching and the autotuner. */
class Pipeline {
    static final int TIER1_AFTER = 3;                 // calls served by the interpreter
    static final int TIER2_AFTER = 3;                 // further calls of one shape before specialization
    static final List<String> CONFIGS = configs();

    final Map<String, Kernel> kernels = new LinkedHashMap<>();
    final Map<String, Plan> generic = new LinkedHashMap<>();
    final Map<String, Plan> specialized = new LinkedHashMap<>();
    final Map<String, Integer> shapeCalls = new LinkedHashMap<>();
    final int[] tierCalls = new int[3];
    int compiles, cacheHits, rejected;
    String lastConfig = "";
    int lastVectorWidth = 1;
    double lastModeledCost;

    static List<String> configs() {
        List<String> out = new ArrayList<>();
        for (boolean f : new boolean[]{false, true})
            for (boolean l : new boolean[]{false, true})
                for (boolean s : new boolean[]{false, true})
                    for (int u : new int[]{1, 2, 4, 8})
                        out.add((f ? "F" : "-") + (l ? "L" : "-") + (s ? "S" : "-") + "u" + u);
        return out;
    }

    void register(Kernel k) { kernels.put(k.name, k); }

    /** The cost model: op weights divided by the vector width, plus memory, loop control and registers. */
    static double modelCost(Plan p, Shape shape) {
        double perIter = 0;
        for (Insn in : p.body) perIter += weight(in.op);
        double align = Math.floorMod(shape.align, 4) == 0 ? 1.0 : 1.25;
        long iters = Math.max(0, shape.n);
        // A full pass costs two loop-control steps; the remainder iterations are not unrolled.
        double passes = Math.floor(iters / (double) p.unroll) + (iters % p.unroll);
        return perIter * align * iters / p.vectorWidth + passes * 2.0 + p.slots * 0.05;
    }

    static double weight(char op) {
        switch (op) {
            case '+': case '-': case 'N': case 'X': return 1.0;
            case '*': return 3.0;
            case '/': return 6.0;
            case 'M': case 'C': return 0.5;
            case 'L': case 'S': case 'W': return 3.0;
            default: return 0.0;
        }
    }

    static Plan compileConfig(Kernel k, Shape shape, String cfg) {
        return Compiler.compile(k, shape, cfg.charAt(0) == 'F', cfg.charAt(1) == 'L', cfg.charAt(2) == 'S',
                Integer.parseInt(cfg.substring(4)));
    }

    /** Explicit form used by the checks, so a transform is never selected by a typo. */
    static Plan compileCfg(Kernel k, Shape shape, boolean fold, boolean licm, boolean strength, int unroll) {
        return Compiler.compile(k, shape, fold, licm, strength, unroll);
    }

    /** Autotunes one shape; a candidate only competes after reproducing the reference bit-for-bit. */
    Plan autotune(Kernel k, Shape shape, Map<String, double[]> refArrays, Map<String, double[]> refScalars) {
        Plan best = null;
        double bestCost = Double.POSITIVE_INFINITY;
        String bestCfg = null;
        for (String cfg : CONFIGS) {
            Plan cand = compileConfig(k, shape, cfg);
            Map<String, double[]> a = Interp.copy(refArrays), s = Interp.copy(refScalars);
            Vm.exec(cand, a, s);
            if (!sameBits(a, refArrays) || !sameBits(s, refScalars)) { rejected++; continue; }
            double cost = modelCost(cand, shape);
            if (cost < bestCost - 1e-9) { best = cand; bestCost = cost; bestCfg = cfg; }
        }
        if (best == null) throw new IllegalStateException("no correct plan for " + k.name);
        compiles++;
        lastConfig = bestCfg;
        lastVectorWidth = best.vectorWidth;
        lastModeledCost = bestCost;
        return best;
    }

    Run call(String name, Map<String, double[]> arrays, Map<String, double[]> scalars, Shape shape) {
        Kernel k = kernels.get(name);
        if (k == null) throw new IllegalArgumentException("unknown kernel " + name);
        Map<String, double[]> s = Interp.copy(scalars);
        if (!k.params.isEmpty() && !s.containsKey(k.params.get(0)))
            s.put(k.params.get(0), new double[]{shape.n});
        Map<String, double[]> a = Interp.copy(arrays);
        int calls = shapeCalls.merge(shape.key(name), 1, Integer::sum);
        if (calls <= TIER1_AFTER) {
            tierCalls[0]++;
            long steps = Interp.run(k, a, s);
            return new Run(a, s, steps, Math.max(0, shape.n), 0, 0, 1, "interp", Double.NaN);
        }
        Plan p;
        int tier;
        if (calls <= TIER1_AFTER + TIER2_AFTER) {
            p = generic.get(name);
            if (p == null) { compiles++; p = compileConfig(k, shape, "FL-u1"); generic.put(name, p); }
            else cacheHits++;
            tier = 1;
            tierCalls[1]++;
        } else {
            String key = shape.key(name);
            p = specialized.get(key);
            if (p == null) {
                Map<String, double[]> refA = Interp.copy(arrays), refS = Interp.copy(scalars);
                if (!k.params.isEmpty() && !refS.containsKey(k.params.get(0)))
                    refS.put(k.params.get(0), new double[]{shape.n});
                Interp.run(k, refA, refS);
                p = autotune(k, shape, refA, refS);
                specialized.put(key, p);
            } else {
                cacheHits++;
                lastConfig = p.config;
                lastVectorWidth = p.vectorWidth;
                lastModeledCost = modelCost(p, shape);
            }
            tier = 2;
            tierCalls[2]++;
        }
        Run r = Vm.exec(p, a, s);
        return new Run(a, s, r.steps, r.loopIters, r.memOps, tier, p.vectorWidth, p.config,
                modelCost(p, shape));
    }

    static boolean sameBits(Map<String, double[]> x, Map<String, double[]> y) {
        if (!x.keySet().equals(y.keySet())) return false;
        for (String key : x.keySet()) {
            double[] p = x.get(key), q = y.get(key);
            if (p.length != q.length) return false;
            for (int i = 0; i < p.length; i++)
                if (Double.doubleToLongBits(p[i]) != Double.doubleToLongBits(q[i])) return false;
        }
        return true;
    }
}

public class Solution {
    static int checks = 0;

    static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
        checks++;
    }

    static final String SAXPY = "kernel saxpy(n, k, c) { for i in 0..n { out[i] = k * in[i] + c; } }";
    static final String STENCIL =
            "kernel stencil(n) { for i in 1..n-1 { out[i] = 0.25 * (in[i-1] + in[i] + in[i+1]); } }";
    static final String DOT =
            "kernel dot(n, acc) { for i in 0..n { acc = acc + in[i] * in[i]; out[i] = acc; } }";
    static final String SCALED = "kernel scaled(n) { for i in 0..n { out[i] = (2.0 * 8.0) * in[i] / 4.0; } }";
    static final String STRIDED = "kernel strided(n) { for i in 0..n { out[i] = in[2 * i] + 1.0; } }";

    static Map<String, double[]> arrays(Object... kv) {
        Map<String, double[]> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) m.put((String) kv[i], (double[]) kv[i + 1]);
        return m;
    }

    static Map<String, double[]> scalars(Object... kv) {
        Map<String, double[]> m = new LinkedHashMap<>();
        for (int i = 0; i < kv.length; i += 2) m.put((String) kv[i], new double[]{(Double) kv[i + 1]});
        return m;
    }

    /** Binds the trip count `n`, which every kernel in this file takes as its first parameter. */
    static Map<String, double[]> bind(int n, Map<String, double[]> s) {
        Map<String, double[]> out = new LinkedHashMap<>(s);
        out.put("n", new double[]{n});
        return out;
    }

    static double[] range(int n, double base, double step) {
        double[] a = new double[n];
        for (int i = 0; i < n; i++) a[i] = base + step * i;
        return a;
    }

    static Map<String, double[]> env(int n, double[] in) {
        return arrays("in", in, "out", new double[in.length]);
    }

    public static void main(String[] args) {
        // 1. Parsing, canonical dump round-trip, contiguity analysis.
        Kernel saxpy = Parser.parse(SAXPY);
        check(saxpy.name.equals("saxpy") && saxpy.params.equals(List.of("n", "k", "c"))
                && saxpy.loopVar.equals("i")
                && saxpy.body.size() == 1 && saxpy.contiguous, "saxpy did not parse as expected");
        String dump = saxpy.show();
        check(dump.equals("kernel saxpy(n,k,c) { for i in 0..n { out[i] = ((k * in[i]) + c); } }"),
                "canonical dump unexpected: " + dump);
        check(Parser.parse(dump).show().equals(dump), "parser dump does not round-trip");
        Kernel stencil = Parser.parse(STENCIL);
        check(stencil.show().contains("in[(i - 1)]") && stencil.show().contains("in[(i + 1)]")
                && stencil.show().contains("1..(n - 1)"), "stencil lowering unexpected: " + stencil.show());
        check(!Parser.parse(STRIDED).contiguous && saxpy.contiguous && stencil.contiguous
                && Parser.parse(DOT).contiguous, "contiguity analysis wrong");

        // 2. Negative checks: malformed syntax is rejected by the parser, unbound names by the compiler.
        String[] bad = {"kernel a(n) { for i in 0..n { out[i] = in[i] } }",
                "kernel a(n) { for i in 0..n { } }",
                "kernel a(n) for i in 0..n { out[i] = in[i]; }",
                "kernel a(n) { for i in 0..n { out[i] = in[i]; } } trailing",
                "kernel a(n) { for i in 0..n { out[i] = in[i]; "};
        for (String src : bad) {
            boolean threw = false;
            try { Parser.parse(src); } catch (IllegalArgumentException e) { threw = true; }
            check(threw, "parser accepted invalid input: " + src);
        }
        boolean threwName = false;
        try { Pipeline.compileCfg(Parser.parse("kernel a(n) { for i in 0..n { out[i] = in[i] + z; } }"),
                new Shape(8, 0), true, true, false, 1); } catch (IllegalArgumentException e) { threwName = true; }
        check(threwName, "compiler accepted an unbound variable");
        boolean threwMissing = false;
        try {
            Vm.exec(Pipeline.compileCfg(saxpy, new Shape(4, 0), true, true, false, 1),
                    env(4, new double[]{1, 2, 3, 4}), bind(4, scalars()));
        } catch (IllegalArgumentException e) { threwMissing = true; }
        check(threwMissing, "VM accepted a missing scalar binding");

        // 3. The interpreter matches hand-computed saxpy and does not mutate read-only scalars.
        Map<String, double[]> arr = env(4, new double[]{1, 2, 3, 4});
        Map<String, double[]> sc = bind(4, scalars("k", 2.0, "c", 1.0));
        Interp.run(saxpy, arr, sc);
        check(Arrays.equals(arr.get("out"), new double[]{3, 5, 7, 9}), "interpreter saxpy result wrong");
        check(sc.get("k")[0] == 2.0, "interpreter mutated a read-only scalar");

        // 4. Every transform configuration is bit-exact against tier 0, on five kernels and two alignments.
        String[] sources = {SAXPY, STENCIL, DOT, SCALED, STRIDED};
        for (String src : sources) {
            Kernel k = Parser.parse(src);
            int n = 24;
            Map<String, double[]> base = env(n, range(2 * n, 0.5, 0.25));
            Map<String, double[]> sBase = bind(n, scalars("k", 1.5, "c", -0.75, "acc", 0.0));
            Map<String, double[]> refA = Interp.copy(base), refS = Interp.copy(sBase);
            Interp.run(k, refA, refS);
            for (String cfg : Pipeline.CONFIGS) {
                for (int align : new int[]{0, 1}) {
                    Map<String, double[]> a = Interp.copy(base), s = Interp.copy(sBase);
                    Vm.exec(Pipeline.compileConfig(k, new Shape(n, align), cfg), a, s);
                    check(Pipeline.sameBits(a, refA), "config " + cfg + " align " + align + " diverged on " + k.name);
                    check(Pipeline.sameBits(s, refS), "config " + cfg + " diverged on scalars of " + k.name);
                }
            }
        }

        // 5. Folding, strength reduction and unrolling each shrink the modelled cost.
        Shape shape = new Shape(256, 0);
        Kernel scaled = Parser.parse(SCALED);
        Plan none = Pipeline.compileCfg(scaled, shape, false, false, false, 1);
        Plan folded = Pipeline.compileCfg(scaled, shape, true, false, false, 1);
        Plan strong = Pipeline.compileCfg(scaled, shape, true, true, true, 1);
        Plan unrolled = Pipeline.compileCfg(scaled, shape, true, true, true, 8);
        check(folded.body.size() < none.body.size(),
                "folding produced " + folded.body.size() + " ops from " + none.body.size());
        check(Pipeline.modelCost(folded, shape) < Pipeline.modelCost(none, shape), "folding did not reduce cost");
        boolean hasScalb = false;
        for (Insn in : strong.body) if (in.op == 'X') hasScalb = true;
        check(hasScalb, "strength reduction did not emit scalb for a power-of-two scaling");
        check(strong.body.size() * 2 <= none.body.size(), "LICM and strength reduction left "
                + strong.body.size() + " of " + none.body.size() + " ops in the body");
        check(Pipeline.modelCost(strong, shape) * 10 < Pipeline.modelCost(none, shape) * 7,
                "heuristics did not cut the modelled cost by 30%: "
                        + Pipeline.modelCost(strong, shape) + " vs " + Pipeline.modelCost(none, shape));
        check(Pipeline.modelCost(unrolled, shape) < Pipeline.modelCost(strong, shape), "unrolling did not reduce cost");

        // 6. LICM shrinks the loop body and the dynamic instruction count without changing results.
        Kernel st = Parser.parse(STENCIL);
        Map<String, double[]> stencilIn = env(64, range(64, 1.0, 0.5));
        Map<String, double[]> refSt = Interp.copy(stencilIn), refStS = bind(64, scalars());
        Interp.run(st, refSt, refStS);
        Plan withoutLicm = Pipeline.compileCfg(st, new Shape(64, 0), false, false, false, 1);
        Plan withLicm = Pipeline.compileCfg(st, new Shape(64, 0), false, true, false, 1);
        check(!withLicm.prologue.isEmpty() && withLicm.body.size() < withoutLicm.body.size(),
                "LICM did not shrink the loop body");
        Map<String, double[]> a1 = Interp.copy(stencilIn), a2 = Interp.copy(stencilIn);
        long dynNo = Vm.exec(withoutLicm, a1, bind(64, scalars())).steps;
        long dynYes = Vm.exec(withLicm, a2, bind(64, scalars())).steps;
        check(Pipeline.sameBits(a1, refSt) && Pipeline.sameBits(a2, refSt), "LICM changed the result");
        check(dynYes < dynNo, "LICM did not reduce dynamic instructions: " + dynYes + " vs " + dynNo);

        // 7. Unrolling collapses loop-control steps and still handles a non-divisible trip count.
        Map<String, double[]> a7 = env(7, range(7, 1.0, 1.0));
        Map<String, double[]> s7 = bind(7, scalars("k", 3.0, "c", 0.5));
        Run r1 = Vm.exec(Pipeline.compileCfg(saxpy, new Shape(7, 0), true, true, false, 1), Interp.copy(a7), Interp.copy(s7));
        Run r4 = Vm.exec(Pipeline.compileCfg(saxpy, new Shape(7, 0), true, true, false, 4), Interp.copy(a7), Interp.copy(s7));
        Map<String, double[]> ref7 = Interp.copy(a7), ref7S = Interp.copy(s7);
        Interp.run(saxpy, ref7, ref7S);
        check(r1.loopIters == 7 && r4.loopIters == 4,
                "pass counts wrong: " + r1.loopIters + "/" + r4.loopIters);
        check(r4.loopIters * 2 < r1.loopIters * 2, "unrolling did not reduce loop-control steps");
        check(r4.steps == r1.steps, "unrolling changed the data-op count; it must only affect loop control");
        check(Pipeline.sameBits(r4.arrays, ref7) && Pipeline.sameBits(r1.arrays, ref7),
                "the unrolled remainder loop produced a wrong result");
        check(r4.arrays.get("out")[3] == 3.0 * 4.0 + 0.5 && r4.arrays.get("out")[6] == 3.0 * 7.0 + 0.5,
                "unrolled body or remainder values wrong");

        // 8. Vector width follows contiguity, and alignment changes the specialization key.
        check(Pipeline.compileCfg(saxpy, new Shape(64, 0), true, true, false, 1).vectorWidth == 4,
                "contiguous kernel not vectorized");
        check(Pipeline.compileCfg(saxpy, new Shape(64, 2), true, true, false, 1).vectorWidth == 2,
                "misalignment not reflected");
        check(Pipeline.compileCfg(Parser.parse(STRIDED), new Shape(64, 0), true, true, false, 1).vectorWidth == 1,
                "strided kernel wrongly vectorized");
        check(!new Shape(64, 0).key("saxpy").equals(new Shape(80, 0).key("saxpy"))
                && new Shape(64, 0).key("saxpy").equals(new Shape(50, 0).key("saxpy")),
                "shape bucket key wrong");

        // 9. Tier promotion, one recompilation per new shape, cache hits afterwards.
        Pipeline pipe = new Pipeline();
        pipe.register(Parser.parse(SAXPY));
        Map<String, double[]> base32 = env(32, range(32, 1.0, 0.125));
        Map<String, double[]> sc32 = bind(32, scalars("k", 2.0, "c", 0.25));
        Map<String, double[]> ref32 = Interp.copy(base32), refS32 = Interp.copy(sc32);
        Interp.run(Parser.parse(SAXPY), ref32, refS32);
        int[] tierSeen = new int[3];
        for (int i = 0; i < 10; i++) {
            Run r = pipe.call("saxpy", base32, sc32, new Shape(32, 0));
            tierSeen[r.tier]++;
            check(Pipeline.sameBits(r.arrays, ref32), "tier " + r.tier + " diverged from the reference");
        }
        check(Arrays.equals(tierSeen, new int[]{3, 3, 4}), "tier policy histogram wrong: " + Arrays.toString(tierSeen));
        check(pipe.tierCalls[0] == 3 && pipe.tierCalls[1] == 3 && pipe.tierCalls[2] == 4,
                "tier counters disagree with the runs: " + Arrays.toString(pipe.tierCalls));
        int warm = pipe.compiles;
        for (int i = 0; i < 5; i++) pipe.call("saxpy", base32, sc32, new Shape(32, 0));
        check(pipe.compiles == warm, "a cached shape was recompiled");
        check(pipe.cacheHits >= 5, "cache hits not counted");
        for (int i = 0; i < 10; i++) pipe.call("saxpy", base32, sc32, new Shape(24, 1));
        check(pipe.compiles == warm + 1, "a new shape did not recompile exactly once");
        check(pipe.specialized.size() == 2, "expected two specialized plans, got " + pipe.specialized.size());
        check(pipe.rejected == 0, "a value-preserving transform was rejected as incorrect");

        // 10. The scheduled plan is the argmin of the independently recomputed cost model.
        Pipeline tune = new Pipeline();
        tune.register(Parser.parse(SAXPY));
        Shape tuneShape = new Shape(64, 0);
        Map<String, double[]> ta = env(64, range(64, 0.25, 1.0));
        Map<String, double[]> ts = bind(64, scalars("k", 1.25, "c", -2.0));
        for (int i = 0; i < 7; i++) tune.call("saxpy", ta, ts, tuneShape);
        double best = Double.POSITIVE_INFINITY;
        String bestCfg = null;
        for (String cfg : Pipeline.CONFIGS) {
            double c = Pipeline.modelCost(Pipeline.compileConfig(Parser.parse(SAXPY), tuneShape, cfg), tuneShape);
            if (c < best - 1e-9) { best = c; bestCfg = cfg; }
        }
        check(tune.lastConfig.equals(bestCfg), "autotuner chose " + tune.lastConfig + " but the argmin is " + bestCfg);
        check(Math.abs(tune.lastModeledCost - best) < 1e-9, "reported cost differs from the argmin cost");

        // 11. Degenerate shapes: zero iterations, one element, an empty stencil range.
        Run zero = pipe.call("saxpy", arrays("in", new double[0], "out", new double[0]), bind(0, scalars("k", 2.0, "c", 1.0)), new Shape(0, 0));
        check(zero.steps <= 8, "empty kernel issued " + zero.steps + " instructions");
        Pipeline degen = new Pipeline();
        degen.register(Parser.parse(STENCIL));
        degen.register(Parser.parse(SAXPY));
        Map<String, double[]> emptyRange = env(2, new double[]{5.0, 6.0});
        Run e2 = degen.call("stencil", emptyRange, scalars(), new Shape(2, 0));
        check(e2.memOps == 0 && e2.arrays.get("out")[1] == 0.0 && e2.arrays.get("out")[0] == 0.0,
                "an empty stencil range still touched memory");
        Map<String, double[]> single = env(1, new double[]{4.0});
        Run e1 = degen.call("saxpy", single, scalars("k", 3.0, "c", 1.0), new Shape(1, 0));
        check(e1.arrays.get("out")[0] == 13.0 && e1.loopIters <= 1, "single element shape wrong");

        // 12. Out-of-range accesses are rejected at DSL level by both engines.
        for (int engine = 0; engine < 2; engine++) {
            boolean threw = false;
            try {
                Map<String, double[]> badEnv = arrays("in", new double[]{1, 2, 3}, "out", new double[2]);
                Map<String, double[]> bs = bind(3, scalars("k", 1.0, "c", 0.0));
                if (engine == 0) Interp.run(saxpy, badEnv, bs);
                else Vm.exec(Pipeline.compileCfg(saxpy, new Shape(3, 0), true, true, false, 1), badEnv, bs);
            } catch (IndexOutOfBoundsException e) { threw = true; }
            check(threw, "engine " + engine + " did not reject an out-of-range store");
        }

        // 13. Independent oracle: the 3-tap stencil against a direct convolution written separately.
        Map<String, double[]> conv = env(40, range(40, 1.0, 0.5));
        Map<String, double[]> refC = Interp.copy(conv);
        Interp.run(stencil, refC, bind(40, scalars()));
        double[] window = refC.get("in");
        double[] brute = new double[40];
        for (int i = 1; i <= 38; i++) brute[i] = 0.25 * (window[i - 1] + window[i] + window[i + 1]);
        boolean same = true;
        for (int i = 0; i < 40; i++)
            if (Double.doubleToLongBits(brute[i]) != Double.doubleToLongBits(refC.get("out")[i])) same = false;
        check(same, "stencil disagrees with the brute-force convolution oracle");

        // 14. Reproducibility and bit-exactness over a randomized shape sequence.
        long[] steps1 = new long[12], steps2 = new long[12];
        String[] cfg1 = new String[12], cfg2 = new String[12];
        for (int pass = 0; pass < 2; pass++) {
            Pipeline p = new Pipeline();
            p.register(Parser.parse(SAXPY));
            SplitMix64 r = new SplitMix64(4242);
            long[] steps = pass == 0 ? steps1 : steps2;
            String[] cfgs = pass == 0 ? cfg1 : cfg2;
            for (int i = 0; i < 12; i++) {
                int n = 8 + r.nextInt(120);
                int align = r.nextInt(4);
                Map<String, double[]> in = env(n, range(n, 0.5, 0.25));
                Map<String, double[]> refR = Interp.copy(in), refRS = bind(n, scalars("k", 1.75, "c", -0.5));
                Interp.run(Parser.parse(SAXPY), refR, refRS);
                Run rr = p.call("saxpy", in, scalars("k", 1.75, "c", -0.5), new Shape(n, align));
                check(Pipeline.sameBits(rr.arrays, refR), "randomized shape n=" + n + " diverged");
                steps[i] = rr.steps;
                cfgs[i] = rr.config;
            }
        }
        check(Arrays.equals(steps1, steps2), "instruction counts are not reproducible");
        check(Arrays.equals(cfg1, cfg2), "config choice is not reproducible");

        // 15. The specialization must actually help, and loop-invariant copies must be gone.
        Pipeline hot = new Pipeline();
        hot.register(Parser.parse(SAXPY));
        Map<String, double[]> big = env(1024, range(1024, 1.0, 0.5));
        Map<String, double[]> bigRef = Interp.copy(big), bigRefS = bind(1024, scalars("k", 2.0, "c", 1.0));
        long interpSteps = Interp.run(Parser.parse(SAXPY), bigRef, bigRefS);
        Run hotRun = null;
        for (int i = 0; i < 8; i++) hotRun = hot.call("saxpy", big, scalars("k", 2.0, "c", 1.0), new Shape(1024, 0));
        check(hotRun.tier == 2, "large shape did not reach tier 2");
        check(Pipeline.sameBits(hotRun.arrays, bigRef), "optimized kernel diverged from the interpreter");
        Plan hotPlan = hot.specialized.get(new Shape(1024, 0).key("saxpy"));
        check(!hotPlan.bodyDump().contains("M r") && hotPlan.bodyDump().contains("L r")
                && hotPlan.bodyDump().contains("S out["), "hoisted body not specialized as expected:\n" + hotPlan.bodyDump());
        check(hotRun.steps * 100 < interpSteps * 75, "tier-2 plan issues " + hotRun.steps
                + " instructions against the interpreter's " + interpSteps);
        check(hotRun.memOps == 2 * 1024, "memory-op accounting wrong: " + hotRun.memOps);

        System.out.println("saxpy n=1024: interpreter " + interpSteps + " instructions, tier-2 plan "
                + hotRun.steps + " (config " + hotRun.config + ", vec" + hotRun.vectorWidth + ")");
        System.out.println("plan cache: " + hot.compiles + " compilations, " + hot.cacheHits + " hits, "
                + hot.rejected + " incorrect candidates rejected");
        System.out.println("tier histogram (t0,t1,t2): " + Arrays.toString(pipe.tierCalls));
        System.out.println("checks passed: " + checks);
        System.out.println("PASS");
    }
}
