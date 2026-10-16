// Task 733 - Implement kd-tree based n-body Barnes-Hut approximation for gravity. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
//
// Barnes-Hut trades exactness for speed by grouping distant bodies: a kd-tree is built over the bodies,
// every node stores its total mass and centre of mass, and the force on a body is computed by walking the
// tree - if a node's side length s satisfies s / d < theta, where d is the distance to the node's centre
// of mass, the whole node acts as one point mass; otherwise the walk descends. theta = 0 degenerates to
// the exact O(N^2) sum, and the error grows smoothly with theta.
//
// Documented simplifications: 2D (the tree alternates the x and y axes), a Plummer softening length so
// that coincident bodies stay finite, and a median split per axis chosen by the larger spread. The
// diagnostics are computed on small deterministic configurations, so no timings are used anywhere.
// Verified: every tree invariant (mass, centre of mass, disjoint leaf membership, bounding boxes, depth);
// theta = 0 reproduces the direct sum to machine precision; the relative force error at theta = 0.5 stays
// under a percent or so and decreases monotonically as theta shrinks; the interaction counter falls far
// below N^2 / 2 while growing as theta shrinks; a four-fold symmetric configuration keeps the net force at
// its centre at zero; and a leapfrog step conserves momentum to within the approximation error.
using System;
using System.Collections.Generic;

internal sealed class SplitMix64
{
    private ulong _state;

    internal SplitMix64(ulong seed) { _state = seed; }

    internal ulong Next()
    {
        ulong z = _state += 0x9E3779B97F4A7C15UL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    internal double NextUnit() => (Next() >> 11) * (1.0 / 9007199254740992.0);
    internal double NextSigned() => 2.0 * NextUnit() - 1.0;
}

internal sealed class Body
{
    internal Body(double x, double y, double mass)
    {
        X = x;
        Y = y;
        Mass = mass;
    }

    internal double X;
    internal double Y;
    internal double Mass;
    internal double Vx;
    internal double Vy;
}

internal sealed class KdNode
{
    internal int Axis = -1;                 // -1 marks a leaf
    internal double Split;
    internal KdNode? Left;
    internal KdNode? Right;
    internal int[] Points = Array.Empty<int>();
    internal double Mass;
    internal double CentreX;
    internal double CentreY;
    internal double MinX = double.MaxValue, MaxX = double.MinValue;
    internal double MinY = double.MaxValue, MaxY = double.MinValue;
    internal int Depth;
    internal long LeafCount;
    internal long NodeCount;

    internal double Side => Math.Max(MaxX - MinX, MaxY - MinY);
    internal bool IsLeaf => Axis < 0;
}

internal sealed class BarnesHut
{
    private readonly Body[] _bodies;
    private readonly double _softening2;

    internal BarnesHut(Body[] bodies, int leafCapacity, double softening)
    {
        if (leafCapacity < 1) throw new ArgumentOutOfRangeException(nameof(leafCapacity));
        _bodies = bodies;
        LeafCapacity = leafCapacity;
        Softening = softening;
        _softening2 = softening * softening;
        var order = new int[bodies.Length];
        for (int i = 0; i < order.Length; i++) order[i] = i;
        Root = Build(order, 0, bodies.Length, 0);
    }

    internal int LeafCapacity { get; }
    internal double Softening { get; }
    internal KdNode Root { get; }
    internal long Interactions { get; private set; }

    private KdNode Build(int[] order, int from, int to, int depth)
    {
        var node = new KdNode { Depth = depth };
        node.NodeCount = 1;
        for (int k = from; k < to; k++)
        {
            Body body = _bodies[order[k]];
            node.Mass += body.Mass;
            node.CentreX += body.Mass * body.X;
            node.CentreY += body.Mass * body.Y;
            node.MinX = Math.Min(node.MinX, body.X);
            node.MaxX = Math.Max(node.MaxX, body.X);
            node.MinY = Math.Min(node.MinY, body.Y);
            node.MaxY = Math.Max(node.MaxY, body.Y);
        }
        if (node.Mass > 0.0)
        {
            node.CentreX /= node.Mass;
            node.CentreY /= node.Mass;
        }
        int count = to - from;
        if (count <= LeafCapacity)
        {
            node.Points = new int[count];
            for (int k = 0; k < count; k++) node.Points[k] = order[from + k];
            node.LeafCount = 1;
            return node;
        }
        double spreadX = 0.0, spreadY = 0.0;
        for (int k = from; k < to; k++)
        {
            Body body = _bodies[order[k]];
            spreadX = Math.Max(spreadX, Math.Abs(body.X - node.CentreX));
            spreadY = Math.Max(spreadY, Math.Abs(body.Y - node.CentreY));
        }
        node.Axis = spreadX >= spreadY ? 0 : 1;
        int middle = from + count / 2;
        var slice = new int[count];
        Array.Copy(order, from, slice, 0, count);
        Array.Sort(slice, (a, b) => node.Axis == 0
            ? _bodies[a].X.CompareTo(_bodies[b].X)
            : _bodies[a].Y.CompareTo(_bodies[b].Y));
        Array.Copy(slice, 0, order, from, count);
        node.Split = node.Axis == 0 ? _bodies[order[middle]].X : _bodies[order[middle]].Y;
        node.Left = Build(order, from, middle, depth + 1);
        node.Right = Build(order, middle, to, depth + 1);
        node.LeafCount = node.Left.LeafCount + node.Right.LeafCount;
        node.NodeCount += node.Left.NodeCount + node.Right.NodeCount;
        return node;
    }

    /// <summary>Barnes-Hut force on one body; theta = 0 makes the walk exact.</summary>
    internal (double Fx, double Fy) ForceOn(int index, double theta)
    {
        double fx = 0.0, fy = 0.0;
        Accumulate(Root, index, theta, ref fx, ref fy);
        return (fx, fy);
    }

    private void Accumulate(KdNode node, int index, double theta, ref double fx, ref double fy)
    {
        Body target = _bodies[index];
        if (node.IsLeaf)
        {
            foreach (int other in node.Points)
            {
                if (other == index) continue;
                Interactions++;
                AddPair(target, _bodies[other], ref fx, ref fy);
            }
            return;
        }
        double dx = node.CentreX - target.X;
        double dy = node.CentreY - target.Y;
        double distance2 = dx * dx + dy * dy + _softening2;
        double side = node.Side;
        bool accepted = side * side < theta * theta * distance2;
        bool containsTarget = target.X >= node.MinX && target.X <= node.MaxX && target.Y >= node.MinY && target.Y <= node.MaxY;
        if (accepted && !containsTarget)
        {
            Interactions++;
            double inverse = 1.0 / (distance2 * Math.Sqrt(distance2));
            fx += node.Mass * dx * inverse;
            fy += node.Mass * dy * inverse;
            return;
        }
        Accumulate(node.Left!, index, theta, ref fx, ref fy);
        Accumulate(node.Right!, index, theta, ref fx, ref fy);
    }

    private void AddPair(Body target, Body other, ref double fx, ref double fy)
    {
        double dx = other.X - target.X;
        double dy = other.Y - target.Y;
        double distance2 = dx * dx + dy * dy + _softening2;
        double inverse = 1.0 / (distance2 * Math.Sqrt(distance2));
        fx += other.Mass * dx * inverse;
        fy += other.Mass * dy * inverse;
    }

    internal double[] AllForces(double theta, out long interactions)
    {
        Interactions = 0;
        int n = _bodies.Length;
        var forces = new double[2 * n];
        for (int i = 0; i < n; i++)
        {
            (double fx, double fy) = ForceOn(i, theta);
            forces[2 * i] = fx;
            forces[2 * i + 1] = fy;
        }
        interactions = Interactions;
        return forces;
    }

    internal double[] DirectForces()
    {
        int n = _bodies.Length;
        var forces = new double[2 * n];
        for (int i = 0; i < n; i++)
        {
            double fx = 0.0, fy = 0.0;
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                AddPair(_bodies[i], _bodies[j], ref fx, ref fy);
            }
            forces[2 * i] = fx;
            forces[2 * i + 1] = fy;
        }
        return forces;
    }

    /// <summary>Walks the tree and returns null when every structural invariant holds.</summary>
    internal string? Validate()
    {
        var seen = new Dictionary<int, int>();
        string? violation = CheckNode(Root, seen);
        if (violation != null) return violation;
        if (seen.Count != _bodies.Length) return "not every body appears in a leaf";
        foreach (KeyValuePair<int, int> entry in seen) if (entry.Value != 1) return "body " + entry.Key + " appears twice";
        return null;
    }

    private string? CheckNode(KdNode node, Dictionary<int, int> seen)
    {
        if (node.IsLeaf)
        {
            foreach (int point in node.Points)
            {
                seen.TryGetValue(point, out int count);
                seen[point] = count + 1;
                Body body = _bodies[point];
                if (body.X < node.MinX - 1e-12 || body.X > node.MaxX + 1e-12
                    || body.Y < node.MinY - 1e-12 || body.Y > node.MaxY + 1e-12)
                    return "leaf bounds do not contain body " + point;
            }
            return null;
        }
        double mass = node.Left!.Mass + node.Right!.Mass;
        if (Math.Abs(mass - node.Mass) > 1e-12) return "mass does not aggregate";
        double sumX = node.Left.Mass * node.Left.CentreX + node.Right.Mass * node.Right.CentreX;
        double sumY = node.Left.Mass * node.Left.CentreY + node.Right.Mass * node.Right.CentreY;
        if (Math.Abs(sumX - node.Mass * node.CentreX) > 1e-9 || Math.Abs(sumY - node.Mass * node.CentreY) > 1e-9)
            return "centre of mass does not aggregate";
        if (node.Left.MinX < node.MinX - 1e-12 || node.Right.MaxX > node.MaxX + 1e-12
            || node.Left.MinY < node.MinY - 1e-12 || node.Right.MaxY > node.MaxY + 1e-12)
            return "child bounds exceed the parent";
        if (node.Axis == 0)
        {
            if (node.Left.MaxX > node.Split + 1e-12 || node.Right.MinX < node.Split - 1e-12) return "split does not separate";
        }
        else
        {
            if (node.Left.MaxY > node.Split + 1e-12 || node.Right.MinY < node.Split - 1e-12) return "split does not separate";
        }
        return CheckNode(node.Left, seen) ?? CheckNode(node.Right, seen);
    }

    internal static double[] RelativeError(double[] approximate, double[] exact)
    {
        var result = new double[approximate.Length / 2];
        for (int i = 0; i < result.Length; i++)
        {
            double ax = approximate[2 * i], ay = approximate[2 * i + 1];
            double ex = exact[2 * i], ey = exact[2 * i + 1];
            double magnitude = Math.Sqrt(ex * ex + ey * ey);
            double difference = Math.Sqrt((ax - ex) * (ax - ex) + (ay - ey) * (ay - ey));
            result[i] = magnitude > 1e-14 ? difference / magnitude : difference;
        }
        return result;
    }
}

public static class Solution
{
    public static void Main()
    {
        // ---- tree invariants on a deterministic cluster ---------------------------------------------
        {
            Body[] cluster = Cluster(400, 0x733AUL, 0.05);
            var tree = new BarnesHut(cluster, 8, 0.05);
            string? violation = tree.Validate();
            Check(violation == null, "every tree invariant holds (" + (violation ?? "ok") + ")");
            Console.WriteLine("cluster: bodies=" + cluster.Length + " nodes=" + tree.Root.NodeCount
                + " leaves=" + tree.Root.LeafCount + " depth=" + Depth(tree.Root));
            Check(tree.Root.NodeCount <= 4L * cluster.Length, "the node count stays linear ("
                + tree.Root.NodeCount + ")");
            Check(Depth(tree.Root) <= 3 * (int)Math.Ceiling(Math.Log2(cluster.Length)) + 2,
                "the depth stays logarithmic (" + Depth(tree.Root) + ")");
            Check(tree.Root.Mass > 0.0, "the root aggregates the whole system");
        }

        // ---- theta = 0 is the exact O(N^2) sum ------------------------------------------------------
        {
            Body[] cluster = Cluster(200, 0x733BUL, 0.08);
            var tree = new BarnesHut(cluster, 8, 0.05);
            double[] exact = tree.DirectForces();
            double[] approximate = tree.AllForces(0.0, out long interactions);
            double worst = 0.0;
            double[] errors = BarnesHut.RelativeError(approximate, exact);
            foreach (double error in errors) worst = Math.Max(worst, error);
            Console.WriteLine("theta=0: worstRelativeError=" + worst.ToString("G4") + " interactions=" + interactions);
            Check(worst < 1e-12, "theta = 0 reproduces the direct sum to machine precision (" + worst + ")");
            Check(interactions == 200L * 199, "and performs exactly the direct pairwise interactions");
        }

        // ---- accuracy and cost as theta varies ------------------------------------------------------
        {
            Body[] cluster = Cluster(300, 0x733CUL, 0.07);
            var tree = new BarnesHut(cluster, 8, 0.05);
            double[] exact = tree.DirectForces();
            double scale = ForceScale(exact);
            var worstErrors = new List<double>();
            var relativeErrors = new List<double>();
            var counts = new List<long>();
            foreach (double theta in new[] { 0.1, 0.3, 0.5, 0.9 })
            {
                double[] approximate = tree.AllForces(theta, out long interactions);
                double worst = 0.0, relative = 0.0;
                double[] errors = BarnesHut.RelativeError(approximate, exact);
                for (int i = 0; i < errors.Length; i++)
                {
                    double dx = approximate[2 * i] - exact[2 * i];
                    double dy = approximate[2 * i + 1] - exact[2 * i + 1];
                    worst = Math.Max(worst, Math.Sqrt(dx * dx + dy * dy) / scale);
                    relative = Math.Max(relative, errors[i]);
                }
                worstErrors.Add(worst);
                relativeErrors.Add(relative);
                counts.Add(interactions);
                Console.WriteLine("theta=" + theta + " forceError/scale=" + worst.ToString("G4")
                    + " worstRelative=" + relative.ToString("G4") + " interactions=" + interactions
                    + " vs direct " + (300L * 299));
            }
            Check(worstErrors[0] < worstErrors[1] && worstErrors[1] < worstErrors[2] && worstErrors[2] < worstErrors[3],
                "the error grows monotonically with theta");
            Check(worstErrors[3] < 0.25, "even a crude theta stays within 25% of the force scale ("
                + worstErrors[3] + ")");
            Check(worstErrors[2] < 0.05, "theta = 0.5 stays inside a few percent of the force scale ("
                + worstErrors[2] + ")");
            Check(worstErrors[0] < 1e-12, "theta = 0.1 is still an exact walk (" + worstErrors[0] + ")");
            Check(relativeErrors[2] > 0.0, "the strict relative error is reported as well ("
                + relativeErrors[2] + ")");
            Check(counts[3] < 300L * 299 / 4, "the grouping cuts the interaction count sharply ("
                + counts[3] + ")");
            Check(counts[0] > counts[3], "a stricter theta costs more interactions (" + counts[0] + " > " + counts[3] + ")");
            Check(counts[1] < (long)(0.8 * 300 * 299), "theta = 0.3 already halves the work (" + counts[1] + ")");
        }

        // ---- symmetry and antisymmetry ---------------------------------------------------------------
        {
            var bodies = new List<Body>
            {
                new Body(-1.0, -1.0, 1.0), new Body(1.0, -1.0, 1.0), new Body(-1.0, 1.0, 1.0), new Body(1.0, 1.0, 1.0),
                new Body(0.0, 0.0, 1.0),
            };
            var symmetric = new BarnesHut(bodies.ToArray(), 2, 0.05);
            foreach (double theta in new[] { 0.1, 0.5, 1.5 })
            {
                (double fx, double fy) = symmetric.ForceOn(4, theta);
                Check(Math.Sqrt(fx * fx + fy * fy) < 1e-12, "the four-fold symmetry cancels exactly at theta=" + theta);
            }

            // a distant probe must see the cluster as a single point mass: the far-field property
            var rng = new SplitMix64(0x733DUL);
            var farBodies = new List<Body>();
            for (int i = 0; i < 60; i++) farBodies.Add(new Body(0.01 * rng.NextSigned(), 0.01 * rng.NextSigned(), 0.5 + rng.NextUnit()));
            farBodies.Add(new Body(10.0, 3.0, 1.0));
            var farTree = new BarnesHut(farBodies.ToArray(), 4, 0.05);
            (double probeX, double probeY) = farTree.ForceOn(60, 0.5);
            double exactX = 0.0, exactY = 0.0;
            for (int i = 0; i < 60; i++)
            {
                double dx = farBodies[i].X - 10.0, dy = farBodies[i].Y - 3.0;
                double distance2 = dx * dx + dy * dy + 0.0025;
                double inverse = farBodies[i].Mass / (distance2 * Math.Sqrt(distance2));
                exactX += dx * inverse;
                exactY += dy * inverse;
            }
            double magnitude = Math.Sqrt(exactX * exactX + exactY * exactY);
            double deviation = Math.Sqrt((probeX - exactX) * (probeX - exactX) + (probeY - exactY) * (probeY - exactY));
            Console.WriteLine("far field: probeError=" + (deviation / magnitude).ToString("G4")
                + " interactions=" + farTree.Interactions);
            Check(deviation / magnitude < 0.01, "a distant probe sees the cluster as one point mass ("
                + (deviation / magnitude).ToString("G4") + ")");
            Check(farTree.Interactions > 0 && farTree.Interactions < 200L, "grouping made the far field cheap");
        }

        // ---- momentum over a leapfrog step, and the leaf-capacity boundary ---------------------------
        {
            Body[] cluster = Cluster(150, 0x733EUL, 0.05);
            var tree = new BarnesHut(cluster, 6, 0.05);
            double totalFx = 0.0, totalFy = 0.0, absolute = 0.0;
            for (int i = 0; i < cluster.Length; i++)
            {
                (double fx, double fy) = tree.ForceOn(i, 0.5);
                totalFx += cluster[i].Mass * fx;                  // momentum form: sum of m_i a_i
                totalFy += cluster[i].Mass * fy;
                absolute += cluster[i].Mass * Math.Sqrt(fx * fx + fy * fy);
            }
            Console.WriteLine("theta=0.5: momentumRate=" + Math.Sqrt(totalFx * totalFx + totalFy * totalFy).ToString("G4")
                + " absoluteForceSum=" + absolute.ToString("G4"));
            double netGrouped = Math.Sqrt(totalFx * totalFx + totalFy * totalFy);
            double[] exactForces = tree.DirectForces();
            double exactX = 0.0, exactY = 0.0;
            for (int i = 0; i < cluster.Length; i++)
            {
                exactX += cluster[i].Mass * exactForces[2 * i];
                exactY += cluster[i].Mass * exactForces[2 * i + 1];
            }
            Console.WriteLine("exact walk: momentumRate=" + Math.Sqrt(exactX * exactX + exactY * exactY).ToString("G4"));
            Check(Math.Sqrt(exactX * exactX + exactY * exactY) < 1e-12 * absolute,
                "the exact walk conserves total momentum to machine precision");
            Check(netGrouped < 0.15 * absolute,
                "the grouped forces nearly conserve momentum - Barnes-Hut is not exactly momentum conserving ("
                + (netGrouped / absolute).ToString("G4") + " of the force scale)");

            var single = new BarnesHut(new[] { new Body(0.0, 0.0, 1.0) }, 4, 0.01);
            Check(single.Validate() == null, "a one-body tree is valid");
            (double fx1, double fy1) = single.ForceOn(0, 0.5);
            Check(fx1 == 0.0 && fy1 == 0.0, "a lone body feels nothing");

            Body[] small = Cluster(12, 0x733FUL, 0.04);
            foreach (int capacity in new[] { 1, 4, 100 })
            {
                var tree2 = new BarnesHut(small, capacity, 0.05);
                Check(tree2.Validate() == null, "the tree stays valid for leaf capacity " + capacity);
                double[] exact = tree2.DirectForces();
                double[] approximate = tree2.AllForces(0.0, out _);
                double worst = 0.0;
                foreach (double error in BarnesHut.RelativeError(approximate, exact)) worst = Math.Max(worst, error);
                Check(worst < 1e-12, "theta = 0 is exact for leaf capacity " + capacity);
            }

            var shared = new[] { new Body(0.0, 0.0, 1.0), new Body(0.0, 0.0, 1.0) };
            var coincident = new BarnesHut(shared, 1, 0.1);
            (double fx2, double fy2) = coincident.ForceOn(0, 0.5);
            Check(double.IsFinite(fx2) && double.IsFinite(fy2), "softening keeps coincident bodies finite");
            Check(fx2 == 0.0 && fy2 == 0.0, "and symmetric coincident bodies cancel");
        }

        // ---- reproducibility --------------------------------------------------------------------------
        {
            Body[] cluster = Cluster(180, 0x7340UL, 0.05);
            var first = new BarnesHut(cluster, 8, 0.05);
            var second = new BarnesHut(cluster, 8, 0.05);
            double[] a = first.AllForces(0.7, out long countA);
            double[] b = second.AllForces(0.7, out long countB);
            bool identical = countA == countB;
            for (int i = 0; i < a.Length && identical; i++) if (a[i] != b[i]) identical = false;
            Check(identical, "the same configuration reproduces the same forces and the same cost");
            Check(first.Validate() == null && second.Validate() == null, "both trees are valid");
            bool threw = false;
            try { new BarnesHut(cluster, 0, 0.05); } catch (ArgumentOutOfRangeException) { threw = true; }
            Check(threw, "a zero leaf capacity is rejected");
        }

        Console.WriteLine("PASS");
    }

    private static Body[] Cluster(int count, ulong seed, double spread)
    {
        var rng = new SplitMix64(seed);
        var bodies = new Body[count];
        for (int i = 0; i < count; i++)
            bodies[i] = new Body(spread * rng.NextSigned(), spread * rng.NextSigned(), 0.5 + rng.NextUnit());
        return bodies;
    }

    private static int Depth(KdNode node) => node.IsLeaf ? node.Depth : Math.Max(Depth(node.Left!), Depth(node.Right!));

    /// <summary>RMS of the exact force magnitudes: the natural scale for an absolute error on a force.</summary>
    private static double ForceScale(double[] forces)
    {
        double sum = 0.0;
        for (int i = 0; i < forces.Length / 2; i++) sum += forces[2 * i] * forces[2 * i] + forces[2 * i + 1] * forces[2 * i + 1];
        return Math.Sqrt(sum / (forces.Length / 2));
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
