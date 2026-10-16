// Task 825 - Implement end-to-end differentiable physics simulation toy.
// (Tier: Master, CTCI Beyond — Machine Learning & AI)
//
// A 1-D mass-spring chain is integrated with semi-implicit (symplectic) Euler and the whole trajectory is
// differentiated end to end. A reverse-mode adjoint sweep walks the stored states backwards and returns the
// gradient of the scalar loss with respect to *every* physical parameter: the spring stiffnesses K_j, the anchor
// stiffness, the damping coefficient, gravity, and the initial positions and velocities. Semi-implicit Euler is
// the integrator of choice because it is explicit, stable at the step used here, and its two update lines
// transpose directly into the adjoint:
//     forward   v_{t+1} = v_t + dt * F(x_t, v_t) / m        x_{t+1} = x_t + dt * v_{t+1}
//     backward  dL/dx_t += dL/dx_{t+1} + (dt/m) * (dF/dx)^T dL/dv_{t+1}
//               dL/dv_t += dL/dv_{t+1} + (dt/m) * (dF/dv)^T dL/dv_{t+1}
// For nearest-neighbour springs the vector-Jacobian product (dF/dx)^T g is the three-point stencil
// K_j * (g_{j+1} - g_j), K_j * (g_j - g_{j+1}) plus the anchor term, so the sweep is O(N) per step and O(N T)
// memory for the stored trajectory.
//
// Verified invariants (each recomputed here, never assumed):
//   * discrete free fall reproduces its exact closed form x_n = -g dt^2 n(n+1)/2, so the integrator is validated
//     before any derivative is trusted;
//   * mechanical energy is conserved to O(dt) without damping and is non-increasing with damping;
//   * every adjoint component agrees with a central finite difference of the very same simulation, on the fixed
//     point and on a deterministically drawn random point;
//   * the adjoint agrees to 1e-12 with an independent *forward-mode* dual-number differentiation of the same
//     integrator - a different differentiation technique that shares no code with the sweep;
//   * a one-particle free-fall gradient matches a hand-derived closed form;
//   * a zero-step run returns the analytic loss and analytic gradients, and parameters the loss cannot reach get
//     exactly zero;
//   * backtracking gradient descent driven only by the adjoint is monotone and lowers the loss by at least 75%,
//     which validates the sign of the gradient, not merely its magnitude.
//
// Documented simplification: 1-D positions, N = 5 particles, a few dozen integration steps, explicit damping, no
// contact/constraint solver, no checkpointing (the trajectory is kept whole). The differentiation itself is real
// reverse-mode differentiation of a real integrator; memory-bounded checkpointing, implicit integrators and
// batched simulation are the "at scale" extensions and are only noted here, not modelled.
using System;
using System.Collections.Generic;

internal static class SplitMix64
{
    internal static ulong Next(ref ulong state)
    {
        ulong z = state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal static double NextDouble(ref ulong state) => (Next(ref state) >> 11) * (1.0 / 9007199254740992.0);

    /** Uniform draw from [lo, hi). */
    internal static double NextRange(ref ulong state, double lo, double hi) => lo + (hi - lo) * NextDouble(ref state);
}

/** One parameter of the dynamics, used to select the seed of a forward-mode dual pass. */
internal enum Knob { None, SpringK, AnchorK, Damping, Gravity, InitialX, InitialV }

internal sealed class Params
{
    internal readonly double[] K;       // stiffness of the spring joining particle j and j+1
    internal readonly double[] X0;
    internal readonly double[] V0;
    internal double AnchorK;
    internal double Damping;
    internal double Gravity;

    internal Params(int n)
    {
        K = new double[n - 1];
        X0 = new double[n];
        V0 = new double[n];
    }

    internal Params Copy()
    {
        var p = new Params(X0.Length) { AnchorK = AnchorK, Damping = Damping, Gravity = Gravity };
        Array.Copy(K, p.K, K.Length);
        Array.Copy(X0, p.X0, X0.Length);
        Array.Copy(V0, p.V0, V0.Length);
        return p;
    }
}

/** A value with one forward derivative: the dual-number arithmetic of the forward-mode oracle. */
internal readonly struct Dual
{
    internal readonly double V;
    internal readonly double D;

    internal Dual(double v, double d) { V = v; D = d; }

    internal static Dual Const(double v) => new Dual(v, 0.0);
    internal static Dual Var(double v) => new Dual(v, 1.0);

    public static Dual operator +(Dual a, Dual b) => new Dual(a.V + b.V, a.D + b.D);
    public static Dual operator -(Dual a, Dual b) => new Dual(a.V - b.V, a.D - b.D);
    public static Dual operator *(Dual a, Dual b) => new Dual(a.V * b.V, a.V * b.D + a.D * b.V);
    public static Dual operator *(double s, Dual a) => new Dual(s * a.V, s * a.D);
    public static Dual operator /(Dual a, double s) => new Dual(a.V / s, a.D / s);
}

internal sealed class Chain
{
    internal readonly int N;
    internal readonly double Dt;
    internal readonly double[] Mass;
    internal readonly double[] Rest;      // rest lengths, one per spring
    internal readonly double[] Target;    // desired final positions
    internal readonly double[] Weight;    // per-particle position penalty
    internal readonly double AnchorX;
    internal readonly double KineticWeight;

    internal Chain(int n, double dt, double[] mass, double[] rest, double[] target, double[] weight,
        double anchorX, double kineticWeight)
    {
        if (n < 1) throw new ArgumentException("a chain needs at least one particle", nameof(n));
        if (dt <= 0.0) throw new ArgumentException("the time step must be positive", nameof(dt));
        if (mass.Length != n || target.Length != n || weight.Length != n || rest.Length != Math.Max(0, n - 1))
        {
            throw new ArgumentException("array shapes do not match the particle count");
        }
        foreach (double m in mass) if (m <= 0.0) throw new ArgumentException("every mass must be positive");
        N = n;
        Dt = dt;
        Mass = mass;
        Rest = rest;
        Target = target;
        Weight = weight;
        AnchorX = anchorX;
        KineticWeight = kineticWeight;
    }

    /** Internal forces at (x, v). Damping is explicit, so the adjoint stays linear in v. */
    internal double[] Forces(Params p, double[] x, double[] v)
    {
        var f = new double[N];
        for (int j = 0; j + 1 < N; j++)
        {
            double e = x[j + 1] - x[j] - Rest[j];
            f[j] += p.K[j] * e;
            f[j + 1] -= p.K[j] * e;
        }
        double ea = x[0] - AnchorX;
        f[0] -= p.AnchorK * ea;
        for (int i = 0; i < N; i++) f[i] += -p.Gravity * Mass[i] - p.Damping * v[i];
        return f;
    }

    /** Rolls the chain forward, storing every intermediate state for the backward sweep. */
    internal void Simulate(Params p, int steps, out double[][] xs, out double[][] vs)
    {
        if (steps < 0) throw new ArgumentException("the step count must not be negative", nameof(steps));
        xs = new double[steps + 1][];
        vs = new double[steps + 1][];
        double[] x = (double[])p.X0.Clone();
        double[] v = (double[])p.V0.Clone();
        xs[0] = x;
        vs[0] = v;
        for (int t = 0; t < steps; t++)
        {
            double[] f = Forces(p, x, v);
            var vn = new double[N];
            for (int i = 0; i < N; i++) vn[i] = v[i] + Dt * f[i] / Mass[i];
            var xn = new double[N];
            for (int i = 0; i < N; i++) xn[i] = x[i] + Dt * vn[i];
            x = xn;
            v = vn;
            xs[t + 1] = x;
            vs[t + 1] = v;
        }
    }

    internal double Loss(double[] xF, double[] vF)
    {
        double s = 0.0;
        for (int i = 0; i < N; i++)
        {
            double d = xF[i] - Target[i];
            s += 0.5 * Weight[i] * d * d;
        }
        for (int i = 0; i < N; i++) s += 0.5 * KineticWeight * vF[i] * vF[i];
        return s;
    }

    internal double Value(Params p, int steps)
    {
        Simulate(p, steps, out double[][] xs, out double[][] vs);
        return Loss(xs[steps], vs[steps]);
    }

    /** Total mechanical energy: kinetic + spring + anchor + gravitational potential. */
    internal double Energy(Params p, double[] x, double[] v)
    {
        double e = 0.0;
        for (int i = 0; i < N; i++) e += 0.5 * Mass[i] * v[i] * v[i] + Mass[i] * p.Gravity * x[i];
        for (int j = 0; j + 1 < N; j++)
        {
            double ext = x[j + 1] - x[j] - Rest[j];
            e += 0.5 * p.K[j] * ext * ext;
        }
        double ea = x[0] - AnchorX;
        e += 0.5 * p.AnchorK * ea * ea;
        return e;
    }

    /** Reverse-mode adjoint: returns the loss and writes d(loss)/d(param) into g. */
    internal double Gradient(Params p, int steps, out Params g)
    {
        Simulate(p, steps, out double[][] xs, out double[][] vs);
        g = new Params(N);
        var dX = new double[N];
        var dV = new double[N];
        double[] xF = xs[steps];
        double[] vF = vs[steps];
        for (int i = 0; i < N; i++)
        {
            dX[i] = Weight[i] * (xF[i] - Target[i]);
            dV[i] = KineticWeight * vF[i];
        }
        for (int t = steps - 1; t >= 0; t--)
        {
            double[] x = xs[t];
            double[] v = vs[t];
            var gV = new double[N];
            for (int i = 0; i < N; i++) gV[i] = dV[i] + Dt * dX[i];   // gradient w.r.t. v_{t+1}
            var ndX = new double[N];
            var ndV = new double[N];
            for (int i = 0; i < N; i++)
            {
                ndX[i] = dX[i];
                ndV[i] = gV[i] - (Dt / Mass[i]) * p.Damping * gV[i];  // dF_i/dv_i = -Damping
                g.Gravity += -Dt * gV[i];                              // dF_i/dGravity = -Mass[i]
                g.Damping += -(Dt / Mass[i]) * gV[i] * v[i];           // dF_i/dDamping = -v_i
            }
            // Spring j joins j and j+1, so it contributes to x_j and x_{j+1} with the *opposite* signs and both
            // terms must be divided by the mass of the particle they move - hence the explicit four terms.
            for (int j = 0; j + 1 < N; j++)
            {
                double e = x[j + 1] - x[j] - Rest[j];
                double k = p.K[j];
                ndX[j] += -(Dt / Mass[j]) * k * gV[j] + (Dt / Mass[j + 1]) * k * gV[j + 1];
                ndX[j + 1] += (Dt / Mass[j]) * k * gV[j] - (Dt / Mass[j + 1]) * k * gV[j + 1];
                g.K[j] += (Dt / Mass[j]) * gV[j] * e - (Dt / Mass[j + 1]) * gV[j + 1] * e;
            }
            double ea = x[0] - AnchorX;
            ndX[0] += -(Dt / Mass[0]) * p.AnchorK * gV[0];
            g.AnchorK += -(Dt / Mass[0]) * gV[0] * ea;
            dX = ndX;
            dV = ndV;
        }
        Array.Copy(dX, g.X0, N);
        Array.Copy(dV, g.V0, N);
        return Loss(xF, vF);
    }

    /**
     * Forward-mode (dual number) value and derivative of the loss w.r.t. one selected knob. This re-implements
     * the integrator in dual arithmetic, so it is an oracle that shares no differentiation code with Gradient.
     */
    internal double DualDerivative(Params p, int steps, Knob knob, int index)
    {
        var x = new Dual[N];
        var v = new Dual[N];
        for (int i = 0; i < N; i++)
        {
            x[i] = knob == Knob.InitialX && index == i ? Dual.Var(p.X0[i]) : Dual.Const(p.X0[i]);
            v[i] = knob == Knob.InitialV && index == i ? Dual.Var(p.V0[i]) : Dual.Const(p.V0[i]);
        }
        Dual anchor = knob == Knob.AnchorK ? Dual.Var(p.AnchorK) : Dual.Const(p.AnchorK);
        Dual damping = knob == Knob.Damping ? Dual.Var(p.Damping) : Dual.Const(p.Damping);
        Dual gravity = knob == Knob.Gravity ? Dual.Var(p.Gravity) : Dual.Const(p.Gravity);
        for (int t = 0; t < steps; t++)
        {
            var f = new Dual[N];
            for (int j = 0; j + 1 < N; j++)
            {
                Dual kj = knob == Knob.SpringK && index == j ? Dual.Var(p.K[j]) : Dual.Const(p.K[j]);
                Dual e = x[j + 1] - x[j] - Dual.Const(Rest[j]);
                f[j] = f[j] + kj * e;
                f[j + 1] = f[j + 1] - kj * e;
            }
            Dual ea = x[0] - Dual.Const(AnchorX);
            f[0] = f[0] - anchor * ea;
            for (int i = 0; i < N; i++)
            {
                f[i] = f[i] + (Dual.Const(-1.0) * gravity) * Dual.Const(Mass[i]) - damping * v[i];
            }
            for (int i = 0; i < N; i++)
            {
                v[i] = v[i] + Dual.Const(Dt) * (f[i] / Mass[i]);
                x[i] = x[i] + Dual.Const(Dt) * v[i];
            }
        }
        Dual total = Dual.Const(0.0);
        for (int i = 0; i < N; i++)
        {
            Dual d = x[i] - Dual.Const(Target[i]);
            total = total + Dual.Const(0.5 * Weight[i]) * (d * d) + Dual.Const(0.5 * KineticWeight) * (v[i] * v[i]);
        }
        return total.D;
    }
}

public static class Solution
{
    private const int N = 5;
    private const double Dt = 0.002;
    private const int Steps = 30;

    private static Chain Build(out Params p, double kineticWeight)
    {
        var chain = new Chain(N, Dt,
            new[] { 1.0, 1.2, 0.8, 1.5, 1.0 },
            new[] { 0.5, 0.5, 0.5, 0.5 },
            new[] { -0.6, -0.2, 0.3, 0.9, 1.4 },
            new[] { 1.0, 1.0, 2.0, 1.0, 3.0 },
            -1.0, kineticWeight);
        p = new Params(N) { AnchorK = 3.0, Damping = 0.15, Gravity = 9.81 };
        for (int j = 0; j + 1 < N; j++) p.K[j] = 4.0 + 0.5 * j;
        for (int i = 0; i < N; i++) p.X0[i] = 0.4 + 0.3 * i;
        return chain;
    }

    private static double Rel(double a, double b) => Math.Abs(a - b) / Math.Max(1.0, Math.Abs(b));

    /** Largest relative difference between an analytic gradient and a central finite difference of Value. */
    private static double WorstFiniteDifference(Chain chain, Params p, Params analytic, int steps)
    {
        const double h = 1e-6;
        double worst = 0.0;
        for (int j = 0; j + 1 < N; j++)
        {
            Params a = p.Copy(), b = p.Copy();
            a.K[j] += h;
            b.K[j] -= h;
            worst = Math.Max(worst, Rel(analytic.K[j], (chain.Value(a, steps) - chain.Value(b, steps)) / (2.0 * h)));
        }
        for (int i = 0; i < N; i++)
        {
            Params a = p.Copy(), b = p.Copy();
            a.X0[i] += h;
            b.X0[i] -= h;
            worst = Math.Max(worst, Rel(analytic.X0[i], (chain.Value(a, steps) - chain.Value(b, steps)) / (2.0 * h)));
            a = p.Copy();
            b = p.Copy();
            a.V0[i] += h;
            b.V0[i] -= h;
            worst = Math.Max(worst, Rel(analytic.V0[i], (chain.Value(a, steps) - chain.Value(b, steps)) / (2.0 * h)));
        }
        Params up = p.Copy(), dn = p.Copy();
        up.AnchorK += h;
        dn.AnchorK -= h;
        worst = Math.Max(worst, Rel(analytic.AnchorK, (chain.Value(up, steps) - chain.Value(dn, steps)) / (2.0 * h)));
        up = p.Copy();
        dn = p.Copy();
        up.Damping += h;
        dn.Damping -= h;
        worst = Math.Max(worst, Rel(analytic.Damping, (chain.Value(up, steps) - chain.Value(dn, steps)) / (2.0 * h)));
        up = p.Copy();
        dn = p.Copy();
        up.Gravity += h;
        dn.Gravity -= h;
        worst = Math.Max(worst, Rel(analytic.Gravity, (chain.Value(up, steps) - chain.Value(dn, steps)) / (2.0 * h)));
        return worst;
    }

    /** Largest relative difference between the adjoint sweep and the forward-mode dual oracle. */
    private static double WorstAgainstDual(Chain chain, Params p, Params adjoint, int steps)
    {
        double worst = 0.0;
        for (int j = 0; j + 1 < N; j++)
        {
            worst = Math.Max(worst, Rel(adjoint.K[j], chain.DualDerivative(p, steps, Knob.SpringK, j)));
        }
        for (int i = 0; i < N; i++)
        {
            worst = Math.Max(worst, Rel(adjoint.X0[i], chain.DualDerivative(p, steps, Knob.InitialX, i)));
            worst = Math.Max(worst, Rel(adjoint.V0[i], chain.DualDerivative(p, steps, Knob.InitialV, i)));
        }
        worst = Math.Max(worst, Rel(adjoint.AnchorK, chain.DualDerivative(p, steps, Knob.AnchorK, 0)));
        worst = Math.Max(worst, Rel(adjoint.Damping, chain.DualDerivative(p, steps, Knob.Damping, 0)));
        worst = Math.Max(worst, Rel(adjoint.Gravity, chain.DualDerivative(p, steps, Knob.Gravity, 0)));
        return worst;
    }

    private static bool Throws(Action action)
    {
        try { action(); return false; }
        catch (ArgumentException) { return true; }
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }

    public static void Main()
    {
        Chain chain = Build(out Params p, 0.5);

        // 1. Free fall against its exact discrete closed form: no springs, no damping, no anchor.
        var fall = new Chain(1, Dt, new[] { 1.0 }, Array.Empty<double>(), new[] { 0.0 }, new[] { 0.0 }, 0.0, 0.0);
        var freeParams = new Params(1) { AnchorK = 0.0, Damping = 0.0, Gravity = 9.81 };
        fall.Simulate(freeParams, 100, out double[][] fxs, out _);
        double exact = -freeParams.Gravity * Dt * Dt * 100 * 101 / 2.0;
        Console.WriteLine("  free fall: simulated=" + fxs[100][0].ToString("F12") + " exact=" + exact.ToString("F12"));
        Check(Math.Abs(fxs[100][0] - exact) < 1e-12, "semi-implicit Euler reproduces the exact free-fall closed form");

        // 2. The one-particle free-fall gradient has a hand-derived closed form, independent of both testers.
        var freeGrad = new Params(1) { AnchorK = 0.0, Damping = 0.0, Gravity = 9.81 };
        var fall2 = new Chain(1, Dt, new[] { 1.0 }, Array.Empty<double>(), new[] { 1.0 }, new[] { 2.0 }, 0.0, 0.5);
        fall2.Gradient(freeGrad, 100, out Params gFree);
        double vEnd = -9.81 * 100 * Dt;
        double xEnd = -9.81 * Dt * Dt * 100 * 101 / 2.0;
        double dXdG = -Dt * Dt * 100 * 101 / 2.0;
        double dVdG = -100 * Dt;
        double gExact = 2.0 * (xEnd - 1.0) * dXdG + 0.5 * vEnd * dVdG;
        Console.WriteLine("  closed form dL/dg=" + gExact.ToString("F10") + " adjoint=" + gFree.Gravity.ToString("F10"));
        Check(Math.Abs(gExact - gFree.Gravity) < 1e-12, "the free-fall gravity gradient matches its closed form");

        // 3-4. Energy behaviour: bounded drift without damping, monotone decay with damping.
        Chain eChain = Build(out Params ep, 0.0);
        ep.Damping = 0.0;
        ep.Gravity = 0.0;
        eChain.Simulate(ep, 400, out double[][] exs, out double[][] evs);
        double e0 = eChain.Energy(ep, exs[0], evs[0]);
        double worstDrift = 0.0;
        for (int t = 1; t <= 400; t++)
        {
            worstDrift = Math.Max(worstDrift, Math.Abs(eChain.Energy(ep, exs[t], evs[t]) - e0) / e0);
        }
        Console.WriteLine("  energy drift (undamped, 400 steps)=" + worstDrift.ToString("E3"));
        Check(worstDrift < 0.05, "symplectic Euler keeps the undamped energy within 5% over 400 steps");

        Chain dChain = Build(out Params dp, 0.0);
        dp.Gravity = 0.0;
        dp.Damping = 1.5;
        dChain.Simulate(dp, 400, out double[][] dxs, out double[][] dvs);
        bool nonIncreasing = true;
        double previous = dChain.Energy(dp, dxs[0], dvs[0]);
        for (int t = 1; t <= 400; t++)
        {
            double now = dChain.Energy(dp, dxs[t], dvs[t]);
            if (now > previous + 1e-9) nonIncreasing = false;
            previous = now;
        }
        Console.WriteLine("  energy with damping: " + dChain.Energy(dp, dxs[0], dvs[0]).ToString("F9")
            + " -> " + previous.ToString("F9"));
        Check(nonIncreasing, "with damping the mechanical energy never increases");
        Check(previous < 0.9 * dChain.Energy(dp, dxs[0], dvs[0]), "damping visibly drains the mechanical energy");

        // 5-6. Adjoint versus central differences on the fixed point and on a drawn random point.
        double loss = chain.Gradient(p, Steps, out Params grad);
        double worst = WorstFiniteDifference(chain, p, grad, Steps);
        Console.WriteLine("  adjoint vs central differences: loss=" + loss.ToString("F9")
            + " worst relative error=" + worst.ToString("E3"));
        Check(worst < 1e-6, "every adjoint component matches a central finite difference");

        var rng = 0xC0FFEEUL;
        var rp = new Params(N)
        {
            AnchorK = SplitMix64.NextRange(ref rng, 1.0, 6.0),
            Damping = SplitMix64.NextRange(ref rng, 0.0, 0.6),
            Gravity = SplitMix64.NextRange(ref rng, 0.0, 12.0)
        };
        for (int j = 0; j + 1 < N; j++) rp.K[j] = SplitMix64.NextRange(ref rng, 1.0, 8.0);
        for (int i = 0; i < N; i++)
        {
            rp.X0[i] = SplitMix64.NextRange(ref rng, -1.0, 1.0);
            rp.V0[i] = SplitMix64.NextRange(ref rng, -2.0, 2.0);
        }
        chain.Gradient(rp, 17, out Params rgrad);
        double worstRandom = WorstFiniteDifference(chain, rp, rgrad, 17);
        Console.WriteLine("  adjoint on a prng-drawn point (17 steps): worst relative error=" + worstRandom.ToString("E3"));
        Check(worstRandom < 1e-6, "the adjoint is correct on an independently drawn parameter point");

        // 7-8. Reverse mode versus an independent forward-mode dual pass, both fixed and random points.
        double worstDual = Math.Max(WorstAgainstDual(chain, p, grad, Steps), WorstAgainstDual(chain, rp, rgrad, 17));
        Console.WriteLine("  adjoint vs forward-mode dual oracle: worst relative error=" + worstDual.ToString("E3"));
        Check(worstDual < 1e-12, "reverse mode and forward mode agree to 1e-12 on every parameter");

        // 9-10. Zero steps: loss and gradients are analytic, so the seeding is exercised exactly.
        chain.Gradient(p, 0, out Params g0);
        double analytic = 0.0;
        for (int i = 0; i < N; i++)
        {
            double d = p.X0[i] - chain.Target[i];
            analytic += 0.5 * chain.Weight[i] * d * d + 0.5 * chain.KineticWeight * p.V0[i] * p.V0[i];
        }
        bool seedsExact = true;
        for (int i = 0; i < N; i++)
        {
            if (Math.Abs(g0.X0[i] - chain.Weight[i] * (p.X0[i] - chain.Target[i])) > 1e-15) seedsExact = false;
            if (Math.Abs(g0.V0[i] - chain.KineticWeight * p.V0[i]) > 1e-15) seedsExact = false;
        }
        Check(Math.Abs(chain.Value(p, 0) - analytic) < 1e-15, "a zero-step run returns the analytic loss");
        Check(seedsExact, "a zero-step run returns the analytic gradients of the initial state");
        Check(Math.Abs(g0.Gravity) < 1e-15 && Math.Abs(g0.Damping) < 1e-15 && Math.Abs(g0.AnchorK) < 1e-15
            && Math.Abs(g0.K[0]) < 1e-15, "no dynamics means no gradient in any dynamical parameter");

        // 11. A parameter the trajectory can never reach has an exactly zero gradient: starting the chain at its
        //     equilibrium (spacing = rest lengths, particle 0 at the anchor) with no gravity leaves every velocity
        //     at exactly 0, so the damping force multiplies nothing and its gradient vanishes identically.
        var rest = new Params(N) { AnchorK = 2.0, Damping = 1.5, Gravity = 0.0 };
        for (int i = 0; i < N; i++) rest.X0[i] = -1.0 + 0.5 * i;
        chain.Gradient(rest, Steps, out Params still);
        bool stayPut = true;
        for (int i = 0; i < N; i++) if (Math.Abs(still.X0[i]) < 1e-9) stayPut = false;
        Check(still.Damping == 0.0 && stayPut,
            "damping has exactly zero gradient when no velocity ever arises (while x0 keeps a live gradient)");

        // 12-13. Determinism and shape validation.
        chain.Gradient(p, Steps, out Params again);
        bool same = true;
        for (int j = 0; j + 1 < N; j++) if (again.K[j] != grad.K[j]) same = false;
        for (int i = 0; i < N; i++) if (again.X0[i] != grad.X0[i] || again.V0[i] != grad.V0[i]) same = false;
        Check(same && again.Gravity == grad.Gravity && again.Damping == grad.Damping,
            "repeated adjoint runs are bit-for-bit identical");
        Check(Throws(() => new Chain(3, 0.0, new[] { 1.0, 1.0, 1.0 }, new[] { 1.0, 1.0 },
            new[] { 0.0, 0.0, 0.0 }, new[] { 1.0, 1.0, 1.0 }, 0.0, 0.0)), "a non-positive time step is rejected");
        Check(Throws(() => new Chain(2, 0.01, new[] { 1.0, 0.0 }, new[] { 1.0 },
            new[] { 0.0, 0.0 }, new[] { 1.0, 1.0 }, 0.0, 0.0)), "a non-positive mass is rejected");
        Check(Throws(() => chain.Simulate(p, -1, out _, out _)), "a negative step count is rejected");
        Check(Throws(() => new Chain(2, 0.01, new[] { 1.0, 1.0 }, new[] { 1.0 },
            new[] { 0.0, 0.0 }, new[] { 1.0 }, 0.0, 0.0)), "mismatched array shapes are rejected");

        // 14-15. Backtracking gradient descent driven only by the adjoint: monotone and effective.
        Params w = p.Copy();
        double first = chain.Value(w, Steps);
        double current = first;
        bool monotone = true;
        for (int it = 0; it < 400; it++)
        {
            chain.Gradient(w, Steps, out Params gw);
            double lr = 1e-2;
            Params candidate = Step(w, gw, lr);
            double next = chain.Value(candidate, Steps);
            while (next > current && lr > 1e-12)
            {
                lr *= 0.5;
                candidate = Step(w, gw, lr);
                next = chain.Value(candidate, Steps);
            }
            if (next > current + 1e-12) monotone = false;
            w = candidate;
            current = next;
        }
        Console.WriteLine("  descent: loss " + first.ToString("F6") + " -> " + current.ToString("F6")
            + " after 400 backtracking steps");
        Check(monotone, "backtracking line search follows the adjoint downhill on every iteration");
        Check(current < 0.25 * first, "gradient descent driven by the adjoint removes at least 75% of the loss");

        Console.WriteLine("PASS");
    }

    /** One gradient-descent step on every parameter, keeping the stiffnesses and damping physical. */
    private static Params Step(Params p, Params g, double lr)
    {
        Params q = p.Copy();
        for (int j = 0; j + 1 < N; j++) q.K[j] = Math.Max(0.05, q.K[j] - lr * g.K[j]);
        for (int i = 0; i < N; i++)
        {
            q.X0[i] -= lr * g.X0[i];
            q.V0[i] -= lr * g.V0[i];
        }
        q.AnchorK = Math.Max(0.05, q.AnchorK - lr * g.AnchorK);
        q.Damping = Math.Max(0.0, q.Damping - lr * g.Damping);
        q.Gravity -= lr * g.Gravity;
        return q;
    }
}
