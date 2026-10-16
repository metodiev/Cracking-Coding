// Task 918 - Implement dynamic QP solver for real-time control in robotics (MPC small demo). (Tier: Master, Beyond — Graphics, Geometry & Robotics)
//
// BoxQp solves min 0.5 u'Qu + c'u s.t. lo <= u <= hi by a primal log-barrier interior-point method (barrier on the
// box slacks, Newton steps by Gaussian elimination, backtracking line search, mu shrinking by 0.2 until the KKT
// projected-gradient residual is at round-off level). Box constraints are the only shape a condensed MPC needs.
//
// Mpc condenses a double integrator [p, v] over horizon N: x = Sx*x0 + Su*u (Sx row-block A^k, Su entry A^(k-1-j)*B),
// so cost ||x||^2_Qbar + r*||u||^2 becomes a box QP with Q = 2(Su'Qbar*Su + r*I), c = 2*Su'Qbar*Sx*x0; only u[0] is
// applied and the QP is rebuilt at the next tick. Simplifications: box constraints only, horizons <= 8, in-process
// simulation, no wall-clock timings. Verified: the closed-form unconstrained minimiser and the separable clipping
// rule, feasibility plus a tiny KKT residual on random QPs against a projected-gradient oracle, rejected infeasible
// box, empty zero-dimension problem, two plants stabilised inside their actuator limits, bit-for-bit repeatability.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
    internal static double Signed(ref ulong s) => (Next(ref s) >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
}

/** Small dense linear algebra: elimination solve, dot, mat-vec, mat-mat, transpose, fill. */
internal static class Lin
{
    internal static double[] Solve(double[,] a, double[] b)
    {
        int n = b.Length; var m = (double[,])a.Clone(); var x = (double[])b.Clone();
        for (int c = 0; c < n; c++)
        {
            int p = c; for (int r = c + 1; r < n; r++) if (Math.Abs(m[r, c]) > Math.Abs(m[p, c])) p = r;
            if (Math.Abs(m[p, c]) < 1e-14) throw new Exception("singular column " + c);
            if (p != c) { for (int j = 0; j < n; j++) { double t = m[c, j]; m[c, j] = m[p, j]; m[p, j] = t; } double tb = x[c]; x[c] = x[p]; x[p] = tb; }
            for (int r = c + 1; r < n; r++)
            { double f = m[r, c] / m[c, c]; x[r] -= f * x[c]; for (int j = c; j < n; j++) m[r, j] -= f * m[c, j]; }
        }
        for (int r = n - 1; r >= 0; r--) { double s = x[r]; for (int c = r + 1; c < n; c++) s -= m[r, c] * x[c]; x[r] = s / m[r, r]; }
        return x;
    }
    internal static double Dot(double[] a, double[] b) { double s = 0; for (int i = 0; i < a.Length; i++) s += a[i] * b[i]; return s; }
    internal static double[] MatVec(double[,] a, double[] x)
    { var r = new double[a.GetLength(0)]; for (int i = 0; i < r.Length; i++) for (int j = 0; j < x.Length; j++) r[i] += a[i, j] * x[j]; return r; }
    internal static double[,] MatMul(double[,] a, double[,] b)
    {
        int n = a.GetLength(0), k = a.GetLength(1), m = b.GetLength(1); var r = new double[n, m];
        for (int i = 0; i < n; i++) for (int t = 0; t < k; t++) for (int j = 0; j < m; j++) r[i, j] += a[i, t] * b[t, j];
        return r;
    }
    internal static double[,] Transpose(double[,] a)
    { var r = new double[a.GetLength(1), a.GetLength(0)]; for (int i = 0; i < a.GetLength(0); i++) for (int j = 0; j < a.GetLength(1); j++) r[j, i] = a[i, j]; return r; }
    internal static double[] Fill(int n, double v) { var a = new double[n]; for (int i = 0; i < n; i++) a[i] = v; return a; }
}

/** Dense convex QP with box constraints, solved by a primal log-barrier interior-point method. */
internal sealed class BoxQp
{
    internal readonly int N; internal readonly double[,] Q; internal readonly double[] C, Lo, Hi;
    internal BoxQp(double[,] q, double[] c, double[] lo, double[] hi)
    { N = c.Length; Q = q; C = c; Lo = lo; Hi = hi; for (int i = 0; i < N; i++) if (!(Lo[i] < Hi[i])) throw new Exception("box needs lo < hi at " + i); }
    internal static double Clamp(double v, double lo, double hi) => v < lo ? lo : (v > hi ? hi : v);
    internal double[] Gradient(double[] u)
    { double[] qu = Lin.MatVec(Q, u); var g = new double[N]; for (int i = 0; i < N; i++) g[i] = qu[i] + C[i]; return g; }
    internal double Objective(double[] u) => 0.5 * Lin.Dot(u, Lin.MatVec(Q, u)) + Lin.Dot(C, u);
    /** KKT measure: distance from u to the projection of its gradient step onto the box; zero at the optimum. */
    internal double Kkt(double[] u)
    { double[] g = Gradient(u); double w = 0; for (int i = 0; i < N; i++) w = Math.Max(w, Math.Abs(u[i] - Clamp(u[i] - g[i], Lo[i], Hi[i]))); return w; }
    internal double Violation(double[] u)
    { double w = 0; for (int i = 0; i < N; i++) w = Math.Max(w, Math.Max(Lo[i] - u[i], u[i] - Hi[i])); return w; }
    private bool Interior(double[] u)
    { for (int i = 0; i < N; i++) if ((!double.IsNegativeInfinity(Lo[i]) && u[i] <= Lo[i]) || (!double.IsPositiveInfinity(Hi[i]) && u[i] >= Hi[i])) return false; return true; }
    private double Barrier(double[] u, double mu)
    {
        double v = Objective(u);
        for (int i = 0; i < N; i++)
        { if (!double.IsNegativeInfinity(Lo[i])) v -= mu * Math.Log(u[i] - Lo[i]); if (!double.IsPositiveInfinity(Hi[i])) v -= mu * Math.Log(Hi[i] - u[i]); }
        return v;
    }
    private double Start(int i) => double.IsNegativeInfinity(Lo[i])
        ? (double.IsPositiveInfinity(Hi[i]) ? 0.0 : Hi[i] - 1.0) : (double.IsPositiveInfinity(Hi[i]) ? Lo[i] + 1.0 : 0.5 * (Lo[i] + Hi[i]));
    internal double[] Solve()
    {
        var u = new double[N];
        if (N == 0) return u;
        for (int i = 0; i < N; i++) u[i] = Start(i);
        for (double mu = 1.0, outer = 0; outer < 40 && mu > 1e-11; outer++, mu *= 0.2)
            for (int inner = 0; inner < 15; inner++)
            {
                double[] g = Gradient(u); var h = (double[,])Q.Clone();
                for (int i = 0; i < N; i++)
                {
                    if (!double.IsNegativeInfinity(Lo[i])) { double d = u[i] - Lo[i]; g[i] -= mu / d; h[i, i] += mu / (d * d); }
                    if (!double.IsPositiveInfinity(Hi[i])) { double d = Hi[i] - u[i]; g[i] += mu / d; h[i, i] += mu / (d * d); }
                }
                double[] step;
                try { step = Lin.Solve(h, g); } catch (Exception) { break; }   // singular barrier Hessian: stop shrinking mu
                double slope = Lin.Dot(g, step), current = Barrier(u, mu), t = 1.0;
                while (t > 1e-9)
                {
                    var trial = new double[N]; for (int i = 0; i < N; i++) trial[i] = u[i] - t * step[i];
                    if (Interior(trial) && Barrier(trial, mu) <= current - 1e-4 * t * slope) { u = trial; break; }
                    t *= 0.5;
                }
                if (t <= 1e-9 || Math.Abs(slope) < 1e-13) break;
            }
        return u;
    }
}

/** Condensed linear MPC for a double integrator: state [position, velocity], control = acceleration. */
internal sealed class Mpc
{
    private readonly int _n; private readonly double _dt, _umax, _r;
    private readonly double[,] _sx, _su, _qbar;
    internal Mpc(int horizon, double dt, double uMax, double positionWeight, double velocityWeight, double controlWeight, double terminalFactor)
    {
        _n = horizon; _dt = dt; _umax = uMax; _r = controlWeight;
        _sx = new double[2 * horizon, 2]; _su = new double[2 * horizon, horizon]; _qbar = new double[2 * horizon, 2 * horizon];
        var powers = new List<double[,]> { new double[2, 2] { { 1.0, 0.0 }, { 0.0, 1.0 } } };
        var a = new double[2, 2] { { 1.0, dt }, { 0.0, 1.0 } }; var b = new double[2, 1] { { 0.5 * dt * dt }, { dt } };
        for (int k = 1; k <= horizon; k++) powers.Add(Lin.MatMul(powers[k - 1], a));
        for (int k = 1; k <= horizon; k++)
        {
            for (int d = 0; d < 2; d++) for (int e = 0; e < 2; e++) _sx[2 * (k - 1) + d, e] = powers[k][d, e];
            for (int d = 0; d < 2; d++) _qbar[2 * (k - 1) + d, 2 * (k - 1) + d] = (d == 0 ? positionWeight : velocityWeight) * (k == horizon ? terminalFactor : 1.0);
            for (int j = 0; j < k; j++)
            { double[,] ab = Lin.MatMul(powers[k - 1 - j], b); _su[2 * (k - 1), j] = ab[0, 0]; _su[2 * (k - 1) + 1, j] = ab[1, 0]; }
        }
    }
    /** Solves the horizon QP; returns the control sequence, the predicted states and their open-loop cost. */
    internal double[] Solve(double[] x0, out double[] predictedStates, out double predictedCost)
    {
        double[,] sut = Lin.Transpose(_su); double[,] q = Lin.MatMul(Lin.MatMul(sut, _qbar), _su);
        double[] c = Lin.MatVec(sut, Lin.MatVec(_qbar, Lin.MatVec(_sx, x0)));
        for (int i = 0; i < _n; i++) { c[i] *= 2.0; q[i, i] += 2.0 * _r; for (int j = 0; j < _n; j++) q[i, j] *= 2.0; }
        double[] u = new BoxQp(q, c, Lin.Fill(_n, -_umax), Lin.Fill(_n, _umax)).Solve();
        predictedStates = States(x0, u); predictedCost = Lin.Dot(predictedStates, Lin.MatVec(_qbar, predictedStates)) + _r * Lin.Dot(u, u);
        return u;
    }
    internal double Cost(double[] x0, double[] u) { double[] s = States(x0, u); return Lin.Dot(s, Lin.MatVec(_qbar, s)) + _r * Lin.Dot(u, u); }
    private double[] States(double[] x0, double[] u)
    { double[] s = Lin.MatVec(_sx, x0); for (int i = 0; i < 2 * _n; i++) for (int j = 0; j < _n; j++) s[i] += _su[i, j] * u[j]; return s; }
    internal double[] Step(double[] x, double u) => new[] { x[0] + _dt * x[1] + 0.5 * _dt * _dt * u, x[1] + _dt * u };
}

public static class Solution
{
    private static ulong _state = 0x9E3779B97F4A7C15UL;
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    private static double MaxDiff(double[] a, double[] b)
    { double w = 0; for (int i = 0; i < a.Length; i++) w = Math.Max(w, Math.Abs(a[i] - b[i])); return w; }
    /** Independent oracle: projected gradient with the safe step 1/||Q||_inf (never above 1/lambda_max). */
    private static double[] ProjectedGradient(BoxQp qp, int iterations)
    {
        double l = 1e-9;
        for (int i = 0; i < qp.N; i++) { double row = 0; for (int j = 0; j < qp.N; j++) row += Math.Abs(qp.Q[i, j]); l = Math.Max(l, row); }
        var u = new double[qp.N];
        for (int i = 0; i < u.Length; i++) u[i] = BoxQp.Clamp(0.0, qp.Lo[i], qp.Hi[i]);
        for (int it = 0; it < iterations; it++)
        { double[] g = qp.Gradient(u); for (int i = 0; i < u.Length; i++) u[i] = BoxQp.Clamp(u[i] - g[i] / l, qp.Lo[i], qp.Hi[i]); }
        return u;
    }
    /** Random test QP: Q = A'A + 0.5I is positive definite, which is all the barrier method needs. */
    private static BoxQp RandomQp(ref ulong s, int n)
    {
        var a = new double[n, n]; var c = new double[n]; var lo = new double[n]; var hi = new double[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++) a[i, j] = Rng.Signed(ref s);
            c[i] = 3.0 * Rng.Signed(ref s); lo[i] = -0.5 - Math.Abs(Rng.Signed(ref s)); hi[i] = 0.5 + Math.Abs(Rng.Signed(ref s));
        }
        double[,] q = Lin.MatMul(Lin.Transpose(a), a);
        for (int i = 0; i < n; i++) q[i, i] += 0.5;
        return new BoxQp(q, c, lo, hi);
    }
    public static void Main()
    {
        // 1. unconstrained QP against the closed form (independent oracle)
        var q1 = new double[,] { { 4.0, 1.0 }, { 1.0, 3.0 } }; var c1 = new double[] { -1.0, -2.0 };
        var reference = new double[] { 1.0 / 11.0, 7.0 / 11.0 };
        Check(MaxDiff(Lin.Solve(q1, new double[] { 1.0, 2.0 }), reference) < 1e-12 && MaxDiff(new BoxQp(q1, c1, Lin.Fill(2, double.NegativeInfinity), Lin.Fill(2, double.PositiveInfinity)).Solve(), reference) < 1e-8, "unconstrained QP reproduces the closed-form optimum (1/11, 7/11)");
        // 2. separable QPs: the box solution is the clipped unconstrained minimiser (independent oracle)
        _state = 918;
        double separable = 0;
        for (int t = 0; t < 12; t++)
        {
            int n = 1 + (int)(Rng.Next(ref _state) % 5);
            var q = new double[n, n]; var c = new double[n]; var lo = new double[n]; var hi = new double[n];
            for (int i = 0; i < n; i++)
            { q[i, i] = 0.5 + Math.Abs(Rng.Signed(ref _state)); c[i] = 2.0 * Rng.Signed(ref _state); lo[i] = -0.4 + 0.1 * i; hi[i] = 1.5 - 0.1 * i; }
            double[] got = new BoxQp(q, c, lo, hi).Solve();
            for (int i = 0; i < n; i++) separable = Math.Max(separable, Math.Abs(got[i] - BoxQp.Clamp(-c[i] / q[i, i], lo[i], hi[i])));
        }
        Check(separable < 1e-7, "separable clipping oracle, 12 trials, worst error " + separable.ToString("E2"));
        // 3. random dense QPs: feasibility, KKT optimality, projected-gradient oracle
        _state = 424242;
        double worstKkt = 0, worstGap = 0, worstViolation = 0;
        for (int t = 0; t < 12; t++)
        {
            BoxQp qp = RandomQp(ref _state, 4 + (int)(Rng.Next(ref _state) % 2));
            double[] solved = qp.Solve(), oracle = ProjectedGradient(qp, 4000);
            worstKkt = Math.Max(worstKkt, qp.Kkt(solved));
            worstGap = Math.Max(worstGap, qp.Objective(solved) - qp.Objective(oracle));
            worstViolation = Math.Max(worstViolation, qp.Violation(solved));
        }
        Check(worstViolation < 1e-9 && worstKkt < 1e-6, "12 random QPs feasible (violation " + worstViolation.ToString("E2") + ") with KKT residual " + worstKkt.ToString("E2"));
        Check(worstGap < 1e-6, "never worse than the projected-gradient oracle (worst gap " + worstGap.ToString("E2") + ")");
        // 4. degenerate inputs (boundary cases)
        bool rejected = false;
        try { new BoxQp(q1, c1, new double[] { 1.0, 0.0 }, new double[] { -1.0, 0.0 }); } catch (Exception) { rejected = true; }
        Check(rejected && new BoxQp(new double[0, 0], new double[0], new double[0], new double[0]).Solve().Length == 0, "an infeasible box is rejected and a zero-dimension QP returns an empty control");
        // 5. MPC closed loop from (1.0, 0.5) with |u| <= 1
        var mpc = new Mpc(6, 0.5, 1.0, 1.0, 0.5, 0.05, 5.0);
        var x = new double[] { 1.0, 0.5 };
        double peak = 0, excess = 0, first = 0;
        for (int tick = 0; tick < 25; tick++)
        {
            double[] u = mpc.Solve(x, out _, out _);
            if (tick == 0) first = u[0];
            foreach (double ui in u) { peak = Math.Max(peak, Math.Abs(ui)); excess = Math.Max(excess, Math.Abs(ui) - 1.0); }
            x = mpc.Step(x, u[0]);
        }
        Check(excess < 1e-9 && peak > 0.2 && first < 0.0, "controls stay in |u| <= 1 (excess " + excess.ToString("E2") + "), peak " + peak.ToString("F3") + ", first move " + first.ToString("F4"));
        Check(Math.Abs(x[0]) < 0.05 && Math.Abs(x[1]) < 0.05, "closed loop reaches the origin: p=" + x[0].ToString("F4") + " v=" + x[1].ToString("F4"));
        // 6. tighter actuator: the plan respects the limit and beats naive sequences
        var tight = new Mpc(4, 0.5, 0.25, 2.0, 1.0, 0.1, 4.0);
        var y = new double[] { -0.8, 0.0 };
        double[] plan = tight.Solve(y, out double[] predicted, out double planned);
        double zeroCost = tight.Cost(y, new double[4]), bangCost = tight.Cost(y, new double[] { 0.25, 0.25, 0.25, 0.25 });
        Check(plan.Length == 4 && predicted.Length == 8 && Math.Abs(predicted[6]) < Math.Abs(y[0]) && planned <= zeroCost + 1e-9 && planned <= bangCost + 1e-9, "planned states end nearer the origin and the cost beats all-zero / bang-bang (" + planned.ToString("F4") + " vs " + zeroCost.ToString("F4") + " / " + bangCost.ToString("F4") + ")");
        var z = new double[] { 0.4, -0.3 };
        for (int tick = 0; tick < 40; tick++) z = tight.Step(z, tight.Solve(z, out _, out _)[0]);
        Check(Math.Abs(z[0]) < 0.05 && Math.Abs(z[1]) < 0.05, "the tight-actuator loop also converges");
        // 7. control is what stabilises, and the pipeline is reproducible
        var drift = new double[] { 1.0, 0.5 };
        for (int tick = 0; tick < 25; tick++) drift = mpc.Step(drift, 0.0);
        Check(Math.Abs(drift[0]) > 5.0, "the uncontrolled plant diverges (p=" + drift[0].ToString("F2") + ")");
        double[] a1 = mpc.Solve(new double[] { 0.7, -0.2 }, out _, out _), a2 = mpc.Solve(new double[] { 0.7, -0.2 }, out _, out _);
        bool identical = true;
        for (int i = 0; i < a1.Length && identical; i++) identical = BitConverter.DoubleToInt64Bits(a1[i]) == BitConverter.DoubleToInt64Bits(a2[i]);
        Check(identical, "repeated solves are bit-for-bit identical");

        Console.WriteLine("QP: 12 random 4-5 dim boxes, violation " + worstViolation.ToString("E2") + ", KKT " + worstKkt.ToString("E2") + ", gap vs projected gradient " + worstGap.ToString("E2") + ", separable " + separable.ToString("E2"));
        Console.WriteLine("MPC: h6/|u|<=1 from (1.0,0.5) -> " + x[0].ToString("F4") + "," + x[1].ToString("F4") + " in 25 ticks (peak " + peak.ToString("F3") + "); h4/|u|<=0.25 -> " + z[0].ToString("F4") + "," + z[1].ToString("F4") + "; uncontrolled p=" + drift[0].ToString("F2"));
        Console.WriteLine("PASS");
    }
}
