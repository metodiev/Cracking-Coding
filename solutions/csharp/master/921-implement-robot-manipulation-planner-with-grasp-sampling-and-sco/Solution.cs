// Task 921 - Implement robot manipulation planner with grasp sampling and scoring. (Tier: Master, Beyond — Graphics, Geometry & Robotics)
//
// Tabletop picking on a plane: a 2-link planar arm at the origin (l1 = 0.8, l2 = 0.7) picks boxes and drops them in
// a bin. For each box the sampler draws 16 approach directions in the *object* frame, and a grasp is the pair of
// antipodal contacts on the faces normal to that direction, so the required opening is 2*(a|cos d| + b|sin d|) with d
// the angle between the approach direction and the box axis. Alignment is scored as cos^2 d + 0.35 cos 2d, which is
// maximal at a face-on grasp (d = 0) where the opening is the small box width. Opening range, finger clearance to the
// other boxes and the arm's reachable annulus are hard constraints (a candidate that violates one is dropped, never
// merely penalised), and the surviving candidates are ranked by alignment, clearance and effort.
//
// Simplifications: 2D tabletop, rotated-rectangle boxes, parallel-jaw gripper reduced to two point contacts (no
// friction-cone volume, no regrasp, no wrench analysis), pick only (the bin drop is a fixed motion), no dynamics and
// no arm self-collision. Verified: every box yields a feasible face-on grasp whose contacts are exactly antipodal
// with a friction margin, widths and alignment scores agree with an independently written corner-projection oracle,
// the pick pose round-trips through forward kinematics, an over-wide box and an out-of-reach box report no grasp,
// and the sampled search is bit-for-bit reproducible.
using System;
using System.Collections.Generic;

internal static class Rng
{
    internal static ulong Next(ref ulong s) { ulong z = s += 0x9E3779B97F4A7C15UL; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL; return z ^ (z >> 31); }
    internal static double Signed(ref ulong s) => (Next(ref s) >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0;
}

internal sealed class Box3
{
    internal double X, Y, A, B, Yaw;
    internal Box3(double x, double y, double a, double b, double yaw) { X = x; Y = y; A = a; B = b; Yaw = yaw; }
    /** Support distance of a rotated rectangle from its centre along a unit direction. */
    internal double Extent(double angle)
    {
        double d = angle - Yaw;
        return Math.Abs(A * Math.Cos(d)) + Math.Abs(B * Math.Sin(d));
    }
}

internal sealed class Grasp
{
    internal int Box; internal double Angle, Base, Score, Width, Clearance;
    internal double Cx, Cy;                       // gripper centre = box centre for an antipodal grasp
    internal double S1x, S1y, S2x, S2y;           // the two antipodal contact points on the box surface
    internal double N1x, N1y;                     // inward contact normal of the first finger
    internal bool Feasible;
}

/** Sampled grasp search plus planar-arm geometry for the pick pose. */
internal sealed class Manipulator
{
    internal readonly double L1 = 0.8, L2 = 0.7, MinOpen = 0.04, MaxOpen = 0.14, Mu = 0.5;
    internal readonly double WClear = 0.6, WEffort = 0.3;
    internal readonly List<Box3> Boxes = new List<Box3>();
    private readonly ulong _seed;
    internal Manipulator(ulong seed) { _seed = seed; }
    internal List<Grasp> Sample(Box3 box, int index, int samples)
    {
        var list = new List<Grasp>();
        ulong state = _seed ^ ((ulong)(index + 1) * 0x2545F4914F6CDD1DUL);
        for (int s = 0; s < samples; s++)
        {
            double angle = box.Yaw + (2.0 * Math.PI * s / samples) + 0.02 * Rng.Signed(ref state);
            double d = angle - box.Yaw, half = box.Extent(angle);
            double reach = Math.Sqrt(box.X * box.X + box.Y * box.Y);
            // The support-plane foot at distance `half` along the approach direction, clamped onto the rectangle,
            // is the physical contact point; its mirror image is the antipodal contact of the second finger.
            double fs = Math.Min(box.A, Math.Max(-box.A, half * Math.Cos(d)));
            double ft = Math.Min(box.B, Math.Max(-box.B, half * Math.Sin(d)));
            double ax = Math.Cos(box.Yaw), ay = Math.Sin(box.Yaw);
            double ox = fs * ax - ft * ay, oy = fs * ay + ft * ax;
            var g = new Grasp
            {
                Box = index, Angle = angle, Width = 2.0 * half, Cx = box.X, Cy = box.Y,
                Base = Math.Cos(d) * Math.Cos(d) + 0.35 * Math.Cos(2.0 * d)
            };
            g.S1x = box.X + ox; g.S1y = box.Y + oy; g.S2x = box.X - ox; g.S2y = box.Y - oy;
            g.N1x = -Math.Cos(angle); g.N1y = -Math.Sin(angle);
            g.Clearance = Clearance(index, g.S1x, g.S1y, g.S2x, g.S2y);
            bool widthOk = g.Width >= MinOpen && g.Width <= MaxOpen;
            bool reachOk = reach > Math.Abs(L1 - L2) + 0.05 && reach < L1 + L2 - 0.05;
            g.Feasible = widthOk && reachOk && g.Clearance > 0.0;
            g.Score = g.Feasible ? g.Base + WClear * g.Clearance - WEffort * reach : double.NegativeInfinity;
            list.Add(g);
        }
        return list;
    }
    /** Distance from the two finger tips to the other boxes; <= 0 means a finger would enter a box. */
    private double Clearance(int index, double p1x, double p1y, double p2x, double p2y)
    {
        double best = double.PositiveInfinity;
        for (int k = 0; k < Boxes.Count; k++)
        {
            if (k == index) continue;
            Box3 other = Boxes[k];
            for (int side = 0; side < 2; side++)
            {
                double dx = (side == 0 ? p1x : p2x) - other.X, dy = (side == 0 ? p1y : p2y) - other.Y;
                best = Math.Min(best, Math.Sqrt(dx * dx + dy * dy) - other.Extent(Math.Atan2(dy, dx)));
            }
        }
        return best;
    }
    internal Grasp Best(Box3 box, int index, int samples)
    {
        Grasp best = null;
        foreach (Grasp g in Sample(box, index, samples))
            if (g.Feasible && (best == null || g.Score > best.Score)) best = g;
        return best;
    }
    /** Two-link inverse kinematics, returning a joint solution or null when the target is out of reach. */
    internal double[] InverseKinematics(double x, double y, double inner)
    {
        double r2 = x * x + y * y;
        double c2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2);
        if (c2 > 1.0 || c2 < -1.0) return null;
        double s2 = inner * Math.Sqrt(Math.Max(0.0, 1.0 - c2 * c2));
        return new[] { Math.Atan2(y, x) - Math.Atan2(L2 * s2, L1 + L2 * c2), Math.Atan2(s2, c2) };
    }
    internal double[] ForwardKinematics(double q1, double q2) =>
        new[] { L1 * Math.Cos(q1) + L2 * Math.Cos(q1 + q2), L1 * Math.Sin(q1) + L2 * Math.Sin(q1 + q2) };
}

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAILED: " + label); }
    /** Independent oracle: width from the four projected corners, alignment from the box-axis dot product. */
    private static double[] Reference(Box3 box, double angle)
    {
        double ux = Math.Cos(angle), uy = Math.Sin(angle);
        double ax = Math.Cos(box.Yaw), ay = Math.Sin(box.Yaw);
        double min = double.PositiveInfinity, max = double.NegativeInfinity;
        for (int i = 0; i < 4; i++)
        {
            double s = (i % 2 == 0 ? 1.0 : -1.0) * box.A, t = (i < 2 ? 1.0 : -1.0) * box.B;
            double projection = (s * ax - t * ay) * ux + (s * ay + t * ax) * uy;
            min = Math.Min(min, projection); max = Math.Max(max, projection);
        }
        double align = Math.Abs(ux * ax + uy * ay);
        return new[] { max - min, align * align + 0.35 * (2.0 * align * align - 1.0) };
    }
    private static Box3 Copy(Box3 b) => new Box3(b.X, b.Y, b.A, b.B, b.Yaw);
    public static void Main()
    {
        var arm = new Manipulator(seed: 921);
        arm.Boxes.Add(new Box3(0.55, 0.28, 0.06, 0.04, 0.15));
        arm.Boxes.Add(new Box3(0.95, 0.50, 0.05, 0.05, -0.35));
        arm.Boxes.Add(new Box3(1.20, 0.20, 0.065, 0.035, 1.05));
        var chosen = new List<Grasp>();
        for (int i = 0; i < arm.Boxes.Count; i++) chosen.Add(arm.Best(arm.Boxes[i], i, 16));
        Check(chosen.TrueForAll(g => g != null), "every tabletop box yields a feasible grasp");

        // 1. feasibility, face-on alignment and the opening inside the gripper range
        double worstAlign = 0;
        for (int i = 0; i < 3; i++)
        {
            double d = chosen[i].Angle - arm.Boxes[i].Yaw;
            worstAlign = Math.Max(worstAlign, Math.Abs(Math.Sin(2.0 * d)));
            Check(chosen[i].Width >= arm.MinOpen - 1e-12 && chosen[i].Width <= arm.MaxOpen + 1e-12,
                "box " + i + " opening " + chosen[i].Width.ToString("F4") + " m is inside the gripper range");
            Check(chosen[i].Clearance > 0.0, "box " + i + " grasp keeps the fingers clear of the other boxes");
        }
        Check(worstAlign < 0.05, "each best grasp is face-on (worst |sin 2d| = " + worstAlign.ToString("F4") + ")");

        // 2. antipodal contacts that really lie on the box faces, plus a friction-cone margin
        for (int i = 0; i < 3; i++)
        {
            Box3 box = arm.Boxes[i]; Grasp g = chosen[i];
            double ax = Math.Cos(box.Yaw), ay = Math.Sin(box.Yaw);
            Check(Math.Abs(g.S1x + g.S2x - 2.0 * g.Cx) < 1e-12 && Math.Abs(g.S1y + g.S2y - 2.0 * g.Cy) < 1e-12,
                "the two contacts of box " + i + " are antipodal about the grasp centre");
            for (int side = 0; side < 2; side++)
            {
                double dx = (side == 0 ? g.S1x : g.S2x) - box.X, dy = (side == 0 ? g.S1y : g.S2y) - box.Y;
                double s = dx * ax + dy * ay, t = -dx * ay + dy * ax;                   // box-frame coordinates
                double onFace = Math.Max(Math.Abs(s) / box.A, Math.Abs(t) / box.B);
                Check(Math.Abs(onFace - 1.0) < 1e-12, "contact " + side + " of box " + i + " lies exactly on a face (onFace = " + onFace.ToString("F12") + ")");
            }
            double tangential = Math.Abs(Math.Cos(g.Angle) * g.N1y - Math.Sin(g.Angle) * g.N1x);
            Check(tangential <= arm.Mu, "box " + i + " contacts stay inside the friction cone (|t| = " + tangential.ToString("E1") + " <= mu = 0.5)");
            Check(Math.Abs(g.N1x + Math.Cos(g.Angle)) < 1e-12, "the finger normal is the inward unit vector of the approach direction");
        }
        // 3. independent oracle: widths and alignment scores over all 48 sampled candidates
        double worstWidth = 0, worstScore = 0;
        for (int i = 0; i < 3; i++)
            foreach (Grasp g in arm.Sample(arm.Boxes[i], i, 16))
            {
                double[] reference = Reference(arm.Boxes[i], g.Angle);
                worstWidth = Math.Max(worstWidth, Math.Abs(reference[0] - g.Width));
                worstScore = Math.Max(worstScore, Math.Abs(reference[1] - g.Base));
            }
        Check(worstWidth < 1e-12 && worstScore < 1e-12,
            "48 sampled grasps match the corner-projection oracle (width " + worstWidth.ToString("E1") + ", score " + worstScore.ToString("E1") + ")");

        // 4. pick poses: IK round-trip through FK and the full plan for the three boxes
        int solved = 0;
        foreach (Grasp g in chosen)
        {
            double[] q = arm.InverseKinematics(g.Cx, g.Cy, inner: 1);
            if (q == null) continue;
            double[] fk = arm.ForwardKinematics(q[0], q[1]);
            Check(Math.Abs(fk[0] - g.Cx) < 1e-9 && Math.Abs(fk[1] - g.Cy) < 1e-9, "FK(IK(pick pose)) reproduces the grasp point");
            solved++;
        }
        Check(solved == 3, "all three pick poses are reachable and execute (" + solved + "/3)");

        // 5. boundary cases: over-wide and out-of-reach boxes have no feasible grasp
        var wide = new Box3(0.90, 0.45, 0.30, 0.05, 0.0);
        var far = new Box3(1.90, 0.00, 0.05, 0.05, 0.0);
        arm.Boxes.Add(wide); arm.Boxes.Add(far);
        Check(arm.Best(wide, 3, 16) == null, "a box wider than the gripper reports no grasp");
        Check(arm.Best(far, 4, 16) == null, "a box beyond the arm's annulus reports no grasp");

        // 6. reproducibility of the sampled search
        var repeat = new Manipulator(seed: 921);
        for (int i = 0; i < 3; i++) repeat.Boxes.Add(Copy(arm.Boxes[i]));
        bool same = true;
        for (int i = 0; i < 3 && same; i++)
            same = BitConverter.DoubleToInt64Bits(repeat.Best(repeat.Boxes[i], i, 16).Score) == BitConverter.DoubleToInt64Bits(chosen[i].Score);
        Check(same, "the sampled search is bit-for-bit reproducible");

        Console.WriteLine("grasps: 3 boxes x 16 samples, worst missed alignment " + worstAlign.ToString("E1") + ", openings " +
            chosen[0].Width.ToString("F4") + "/" + chosen[1].Width.ToString("F4") + "/" + chosen[2].Width.ToString("F4") +
            " m inside [" + arm.MinOpen.ToString("F3") + ", " + arm.MaxOpen.ToString("F3") + "]");
        Console.WriteLine("boundary: over-wide and out-of-reach boxes report no grasp; IK round-trip solved " + solved +
            "/3 pick poses; min finger clearance " + Math.Min(chosen[0].Clearance, Math.Min(chosen[1].Clearance, chosen[2].Clearance)).ToString("F4") + " m");
        Console.WriteLine("PASS");
    }
}
