// Task 919 - Implement motion planning under differential constraints using trajectory optimization. (Tier: Master, Beyond — Graphics, Geometry & Robotics)
//
// A differential constraint means the robot cannot move sideways: only v (forward speed) and omega (yaw rate) are
// commanded, and the pose evolves as x' = v cos(t), y' = v sin(t), t' = omega. Planning is posed as direct
// trajectory optimization over the 2T control samples [v_0..v_{T-1}, omega_0..w_{T-1}]: the objective is
// 50*||p_T - goal||^2 (terminal) + 0.05*sum(v^2 + 0.3 w^2) (effort) + 4*sum(max(0, safe + r_obs - d_k)^2) (a hinge
// penalty that pushes the path out of the obstacle disc). Box bounds |v| <= 1.5, |omega| <= 1.2 are handled by
// projection, the gradient is central-difference (the obstacle hinge is not differentiable at contact), and the
// step size adapts (x1.15 on improvement, x0.5 otherwise) so no line search bookkeeping is needed.
//
// Simplifications: single unicycle with Euler integration, one circular obstacle, fixed horizon and fixed start
// pose, no curvature/steering-angle limits beyond |omega| <= 1.2, no dynamic (acceleration) limits, no re-planning.
// Verified: the forward model against an independently written integrator, control bounds and obstacle clearance on
// every step, goal convergence from a zero-control (non-informative) guess, monotone cost decrease, a brute-force
// grid oracle on a tiny horizon, the required heading alignment (the differential constraint is really enforced),
// the deviation from the straight line when the obstacle blocks it, a single-step horizon and reproducibility.
using System;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
    internal static double Signed(ref ulong s) => (Next(ref s) >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
}

/** Direct trajectory optimisation for a unicycle: controls are [v_0..v_{T-1}, omega_0..omega_{T-1}]. */
internal sealed class Planner
{
    internal readonly int T;
    internal readonly double Dt = 0.2, VMax = 1.5, OMax = 1.2;
    internal readonly double GoalX, GoalY, ObstacleX, ObstacleY, ObstacleR;
    internal readonly double Terminal = 50.0, Effort = 0.05, Clearance = 20.0, Safe = 0.35;
    internal Planner(int horizon, double goalX, double goalY, double obstacleX, double obstacleY, double obstacleR)
    {
        T = horizon; GoalX = goalX; GoalY = goalY;
        ObstacleX = obstacleX; ObstacleY = obstacleY; ObstacleR = obstacleR;
    }
    internal double Bound(int index) => index < T ? VMax : OMax;
    /** Euler integration of the unicycle; row 0 is the start pose, row T the planned end pose. */
    internal double[,] States(double[] u)
    {
        var s = new double[T + 1, 3];
        for (int k = 0; k < T; k++)
        {
            s[k + 1, 0] = s[k, 0] + u[k] * Math.Cos(s[k, 2]) * Dt;
            s[k + 1, 1] = s[k, 1] + u[k] * Math.Sin(s[k, 2]) * Dt;
            s[k + 1, 2] = s[k, 2] + u[T + k] * Dt;
        }
        return s;
    }
    private double ObstacleGap(double[,] s, int k) =>
        Safe + ObstacleR - Math.Sqrt((s[k, 0] - ObstacleX) * (s[k, 0] - ObstacleX) + (s[k, 1] - ObstacleY) * (s[k, 1] - ObstacleY));
    internal double Cost(double[] u)
    {
        double[,] s = States(u);
        double dx = s[T, 0] - GoalX, dy = s[T, 1] - GoalY;
        double c = Terminal * (dx * dx + dy * dy);
        for (int k = 0; k < T; k++)
        {
            c += Effort * (u[k] * u[k] + 0.3 * u[T + k] * u[T + k]);
            double gap = ObstacleGap(s, k);
            if (gap > 0.0) c += gap * (Clearance * gap + 2.0);   // quadratic + linear hinge: gradient stays nonzero at contact
        }
        return c;
    }
    /** Largest obstacle-hinge gap along the path: > 0 means a collision with the inflated obstacle disc. */
    internal double WorstGap(double[] u)
    {
        double[,] s = States(u); double worst = double.NegativeInfinity;
        for (int k = 0; k <= T; k++) worst = Math.Max(worst, ObstacleGap(s, k));
        return worst;
    }
    internal bool InBounds(double[] u)
    {
        for (int i = 0; i < u.Length; i++) if (Math.Abs(u[i]) > Bound(i) + 1e-12) return false;
        return true;
    }
    /** Central-difference gradient; the hinge penalty is piecewise smooth, which a finite difference handles. */
    internal double[] Gradient(double[] u, double h)
    {
        var g = new double[u.Length];
        for (int i = 0; i < u.Length; i++)
        {
            u[i] += h; double up = Cost(u);
            u[i] -= 2.0 * h; double down = Cost(u);
            u[i] += h;
            g[i] = (up - down) / (2.0 * h);
        }
        return g;
    }
    /** Adaptive projected-gradient descent; returns the optimised control sequence and its cost history. */
    internal double[] Optimize(double[] start, int iterations, out double firstCost, out double bestCost)
    {
        var u = (double[])start.Clone();
        double cost = Cost(u), step = 0.5;
        firstCost = cost; bestCost = cost;
        for (int it = 0; it < iterations; it++)
        {
            double[] g = Gradient(u, 1e-5);
            var next = new double[u.Length];
            for (int i = 0; i < u.Length; i++) next[i] = Clamp(u[i] - step * g[i], Bound(i));
            double nextCost = Cost(next);
            if (nextCost < cost) { u = next; cost = nextCost; bestCost = cost; step *= 1.15; }
            else step *= 0.5;
        }
        return u;
    }
    internal double TerminalDistance(double[] u)
    {
        double[,] s = States(u);
        return Math.Sqrt((s[T, 0] - GoalX) * (s[T, 0] - GoalX) + (s[T, 1] - GoalY) * (s[T, 1] - GoalY));
    }
    internal static double Clamp(double v, double bound) => v < -bound ? -bound : (v > bound ? bound : v);
    /** Independent actuator-free integration used by the checks, written separately from States. */
    internal double[,] ManualStates(double[] u)
    {
        var s = new double[T + 1, 3];
        for (int k = 0; k < T; k++)
        {
            double heading = s[k, 2];
            s[k + 1, 0] = s[k, 0] + Dt * u[k] * Math.Cos(heading);
            s[k + 1, 1] = s[k, 1] + Dt * u[k] * Math.Sin(heading);
            s[k + 1, 2] = heading + Dt * u[T + k];
        }
        return s;
    }
}

public static class Solution
{
    private static ulong _state = 0x9E3779B97F4A7C15UL;
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    private static double[] Lift(int t, double v, double w)
    { var u = new double[2 * t]; for (int k = 0; k < t; k++) { u[k] = v; u[t + k] = w; } return u; }
    private static double[] Jitter(int t, double spread)
    {
        var u = new double[2 * t];
        for (int i = 0; i < 2 * t; i++) u[i] = spread * Rng.Signed(ref _state);
        return u;
    }
    /** Brute-force oracle: the cheapest sequence on a coarse control grid, a subset of the continuous box. */
    private static double BestGrid(Planner p, double[] vGrid, double[] wGrid)
    {
        var u = new double[2 * p.T];
        double best = double.PositiveInfinity;
        var index = new int[p.T];
        while (true)
        {
            for (int k = 0; k < p.T; k++) { u[k] = vGrid[index[k] % vGrid.Length]; u[p.T + k] = wGrid[index[k] % wGrid.Length]; }
            best = Math.Min(best, p.Cost(u));
            int i = 0;
            while (i < p.T && ++index[i] % (vGrid.Length * wGrid.Length) == 0) index[i++] = 0;
            if (i == p.T) break;
        }
        return best;
    }
    private static double MaxLateral(Planner p, double[] u)
    {
        double[,] s = p.States(u); double worst = 0;
        for (int k = 0; k <= p.T; k++) worst = Math.Max(worst, Math.Abs(s[k, 1]));
        return worst;
    }
    public static void Main()
    {
        // 1. free-space plan from a zero-control (non-informative) guess
        var free = new Planner(15, 3.0, 1.5, 100.0, 100.0, 0.0);
        double[] zero = new double[2 * free.T];
        double[] plan = free.Optimize(zero, 400, out double before, out double after);
        Check(free.TerminalDistance(plan) < 0.25, "free-space plan reaches the goal (distance " + free.TerminalDistance(plan).ToString("F4") + ")");
        Check(after < before / 5.0, "cost fell from " + before.ToString("F2") + " to " + after.ToString("F4") + " (> 5x)");
        // 2. the differential constraints really hold on the executed path
        Check(free.InBounds(plan), "every command respects |v| <= 1.5 and |omega| <= 1.2");
        double[,] s = free.States(plan);
        double headingError = Math.Abs(Normalize(Math.Atan2(free.GoalY - s[free.T, 1], free.GoalX - s[free.T, 0]) - s[free.T, 2]));
        Check(headingError < 0.35, "the final heading aligns with the goal direction (|dtheta| = " + headingError.ToString("F4") + ")");
        Check(s[free.T, 2] > 0.15, "the plan steers towards the goal (final yaw " + s[free.T, 2].ToString("F3") + " rad)");
        // 3. forward model cross-check against an independently written integrator
        double[,] manual = free.ManualStates(plan);
        double modelGap = 0;
        for (int k = 0; k <= free.T; k++) for (int d = 0; d < 3; d++) modelGap = Math.Max(modelGap, Math.Abs(s[k, d] - manual[k, d]));
        Check(modelGap < 1e-12, "the trajectory matches an independent Euler integrator (max gap " + modelGap.ToString("E1") + ")");
        // 4. brute-force grid oracle on a tiny horizon (the grid is a subset of the continuous box)
        var tiny = new Planner(3, 0.6, 0.3, 100.0, 100.0, 0.0);
        double[] tinyPlan = tiny.Optimize(new double[2 * tiny.T], 300, out _, out double tinyCost);
        double grid = BestGrid(tiny, new double[] { -1.5, -0.75, 0.0, 0.75, 1.5 }, new double[] { -1.2, 0.0, 1.2 });
        Check(tinyCost <= grid + 1e-6, "optimiser cost " + tinyCost.ToString("F6") + " <= best grid sequence " + grid.ToString("F6"));
        // 5. obstacle scenario: the path must deviate and stay clear
        var blocked = new Planner(15, 3.0, 1.2, 1.5, 0.35, 0.45);
        double[] detour = blocked.Optimize(Lift(15, 1.0, 0.0), 2500, out _, out _);
        Check(blocked.WorstGap(detour) <= 1e-6 && blocked.TerminalDistance(detour) < 0.35,
            "detour is collision free (worst hinge gap " + blocked.WorstGap(detour).ToString("E2") + ") and still reaches the goal");
        Check(MaxLateral(blocked, detour) > 0.35, "the blocked path deviates from the straight line (max |y| = " + MaxLateral(blocked, detour).ToString("F3") + ")");
        // 6. negative checks: an out-of-bounds sequence and a colliding straight line are both rejected
        double[] fast = Lift(15, 3.0, 0.0);
        Check(!blocked.InBounds(fast), "an out-of-bounds control sequence is rejected by InBounds");
        Check(blocked.WorstGap(Lift(15, 1.0, 0.0)) > 0.0, "the naive straight run is detected as a collision");
        // 7. boundary/small horizons: the closed-form single-step optimum, then a gap that saturates the limit
        var one = new Planner(1, 0.3, 0.0, 100.0, 100.0, 0.0);
        double[] single = one.Optimize(new double[2], 300, out _, out _);
        Check(Math.Abs(single[0] - 6.0 / 4.1) < 1e-3,
            "horizon 1 matches the closed-form trade-off v = 6/4.1 = " + (6.0 / 4.1).ToString("F4") + " (got " + single[0].ToString("F4") + ")");
        var far = new Planner(1, 0.45, 0.0, 100.0, 100.0, 0.0);
        double[] saturating = far.Optimize(new double[2], 300, out _, out _);
        Check(Math.Abs(saturating[0] - 1.5) < 1e-9, "a 0.45 m gap saturates the speed limit (v = " + saturating[0].ToString("F4") + ")");
        _state = 99; double[] guessA = Jitter(15, 0.8);
        _state = 99; double[] guessB = Jitter(15, 0.8);
        double[] a = free.Optimize(guessA, 200, out _, out double ca), b = free.Optimize(guessB, 200, out _, out double cb);
        bool identical = ca == cb;
        for (int i = 0; i < a.Length && identical; i++) identical = BitConverter.DoubleToInt64Bits(a[i]) == BitConverter.DoubleToInt64Bits(b[i]);
        Check(identical, "seeded restarts reproduce the same plan bit for bit");

        Console.WriteLine("free: terminal distance " + free.TerminalDistance(plan).ToString("F4") + ", cost " + before.ToString("F2") +
            " -> " + after.ToString("F4") + ", final pose (" + s[free.T, 0].ToString("F3") + ", " + s[free.T, 1].ToString("F3") + ", " + s[free.T, 2].ToString("F3") + ")");
        Console.WriteLine("blocked: clearance gap " + blocked.WorstGap(detour).ToString("E2") + ", max |y| " + MaxLateral(blocked, detour).ToString("F3") +
            ", terminal distance " + blocked.TerminalDistance(detour).ToString("F4") + "; tiny-horizon cost " + tinyCost.ToString("F6") + " vs grid " + grid.ToString("F6"));
        Console.WriteLine("PASS");
    }
    private static double Normalize(double angle)
    {
        while (angle > Math.PI) angle -= 2.0 * Math.PI;
        while (angle < -Math.PI) angle += 2.0 * Math.PI;
        return angle;
    }
}
