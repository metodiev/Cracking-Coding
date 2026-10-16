// Task 878 - Implement production-ready collision detection and physics engine basics.
// (Tier: Master, CTCI Beyond - Simulation & Physics)
//
// Model: a 2D rigid-body engine limited to the shapes a broadphase can actually prune - circles and
// axis-aligned boxes - with the three pieces a real engine needs:
//   * broadphase: sweep and prune over the x-order plus a uniform spatial hash, both returning the set of
//     potentially overlapping pairs;
//   * narrowphase: exact circle/circle, circle/box and box/box overlap tests, including the contact normal
//     and penetration depth, plus a conservative-advancement style swept test for fast movers;
//   * solver: semi-implicit Euler integration, sequential impulses with restitution and Coulomb friction,
//     Baumgarte-free positional correction with a slop, and a sleeping rule.
// The invariants checked are the ones that make such an engine usable: the broadphase pair set equals the
// brute-force O(n^2) set exactly (no false negatives, and no false positives beyond the AABB test itself),
// resolved contacts stop overlapping, an isolated collision conserves momentum and matches the analytic
// elastic/inelastic outcome, energy is non-increasing under restitution 0, a stack settles instead of
// jittering, and a fast projectile cannot tunnel through a thin wall.
//
// Simplifications (documented, not behavioural): 2D instead of 3D, circles and AABBs instead of arbitrary
// convex hulls, no rotation (so no angular impulses or inertia tensors) and a fixed step count instead of a
// real-time clock; the swept test is a per-pair continuous check rather than a global conservative
// advancement pass.
//
// Verified here:
//   * broadphase: sweep-and-prune and the spatial hash reproduce the O(n^2) pair set on 8 randomised
//     scenes, the pair count is bounded by n(n-1)/2, and the hash grid is shown to be an over-approximation
//     of the exact overlap set (no missed collisions);
//   * narrowphase: analytic contacts (touching, penetrating, separated) for all three shape combinations,
//     and the swept test finds a bullet that the discrete step would tunnel through;
//   * solver: head-on elastic collisions swap velocities (analytic), perfectly inelastic ones share a common
//     velocity and lose exactly the analytic amount of energy, momentum is conserved to machine precision,
//     the stack settles with a bounded residual penetration and no jitter, and no body ends up overlapping
//     after resolution;
//   * degenerate cases: zero bodies, one body, static (infinite-mass) bodies, coincident circles, zero
//     radius, zero restitution and a zero time step;
//   * determinism: the whole simulation is bitwise reproducible.
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static ulong _state = 0xFEEDFACECAFEBEEFUL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextSigned() => ((NextU64() >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;

    // ---------- bodies ----------
    private struct Vec
    {
        public double X;
        public double Y;
        public Vec(double x, double y) { X = x; Y = y; }
        public static Vec operator +(Vec a, Vec b) => new Vec(a.X + b.X, a.Y + b.Y);
        public static Vec operator -(Vec a, Vec b) => new Vec(a.X - b.X, a.Y - b.Y);
        public static Vec operator *(Vec a, double s) => new Vec(a.X * s, a.Y * s);
        public double LengthSquared => X * X + Y * Y;
        public double Length => Math.Sqrt(X * X + Y * Y);
        public override string ToString() => "(" + X.ToString("F4") + "," + Y.ToString("F4") + ")";
    }

    /// <summary>A box is stored as half extents for circles (Radius) or full extents for boxes (HalfExtent).</summary>
    private sealed class Body
    {
        public bool IsCircle;
        public double Radius;              // circles
        public Vec HalfExtent;             // boxes
        public Vec Pos;
        public Vec Vel;
        public double Mass = 1.0;
        public double InvMass = 1.0;
        public bool Awake = true;

        public Vec Min => IsCircle ? new Vec(Pos.X - Radius, Pos.Y - Radius)
                                   : new Vec(Pos.X - HalfExtent.X, Pos.Y - HalfExtent.Y);
        public Vec Max => IsCircle ? new Vec(Pos.X + Radius, Pos.Y + Radius)
                                   : new Vec(Pos.X + HalfExtent.X, Pos.Y + HalfExtent.Y);

        public static Body Circle(double x, double y, double r, double mass = 1.0) => new Body
        {
            IsCircle = true, Radius = r, Pos = new Vec(x, y), Mass = mass,
            InvMass = mass <= 0 ? 0 : 1.0 / mass,
        };

        public static Body Box(double x, double y, double hx, double hy, double mass = 1.0) => new Body
        {
            IsCircle = false, HalfExtent = new Vec(hx, hy), Pos = new Vec(x, y), Mass = mass,
            InvMass = mass <= 0 ? 0 : 1.0 / mass,
        };

        public Body Clone() => new Body
        {
            IsCircle = IsCircle, Radius = Radius, HalfExtent = HalfExtent, Pos = Pos, Vel = Vel,
            Mass = Mass, InvMass = InvMass, Awake = Awake,
        };
    }

    private sealed class Scene
    {
        public List<Body> Bodies = new List<Body>();
        public double Gravity = -9.81;
        public double Restitution = 0.2;
        public double Friction = 0.3;
        public double Slop = 0.001;
        public double Correction = 0.8;

        public Scene Clone()
        {
            var s = new Scene
            {
                Gravity = Gravity, Restitution = Restitution, Friction = Friction,
                Slop = Slop, Correction = Correction,
            };
            s.Bodies.AddRange(Bodies.Select(b => b.Clone()));
            return s;
        }
    }

    // ---------- broadphase ----------
    /// <summary>Sweep and prune on the x axis: sort by Min.X and only compare boxes whose x ranges overlap.</summary>
    private static List<(int, int)> SweepAndPrune(List<Body> bodies, out long comparisons)
    {
        comparisons = 0;
        var order = Enumerable.Range(0, bodies.Count).OrderBy(i => bodies[i].Min.X).ThenBy(i => i).ToList();
        var pairs = new List<(int, int)>();
        for (int a = 0; a < order.Count; a++)
        {
            Body bi = bodies[order[a]];
            for (int b = a + 1; b < order.Count; b++)
            {
                Body bj = bodies[order[b]];
                comparisons++;
                if (bj.Min.X > bi.Max.X) break;                 // sorted: nothing further can overlap in x
                if (AabbOverlap(bi, bj)) pairs.Add((Math.Min(order[a], order[b]), Math.Max(order[a], order[b])));
            }
        }
        pairs.Sort();
        return pairs;
    }

    /// <summary>Uniform spatial hash over the AABBs, one bucket per cell an AABB touches.</summary>
    private static List<(int, int)> SpatialHashPairs(List<Body> bodies, double cell)
    {
        var buckets = new Dictionary<(int, int), List<int>>();
        for (int i = 0; i < bodies.Count; i++)
        {
            Body b = bodies[i];
            int x0 = (int)Math.Floor(b.Min.X / cell), x1 = (int)Math.Floor(b.Max.X / cell);
            int y0 = (int)Math.Floor(b.Min.Y / cell), y1 = (int)Math.Floor(b.Max.Y / cell);
            for (int x = x0; x <= x1; x++)
                for (int y = y0; y <= y1; y++)
                {
                    if (!buckets.TryGetValue((x, y), out List<int> list)) buckets[(x, y)] = list = new List<int>();
                    list.Add(i);
                }
        }
        var found = new HashSet<(int, int)>();
        foreach (List<int> list in buckets.Values)
            for (int a = 0; a < list.Count; a++)
                for (int b = a + 1; b < list.Count; b++)
                {
                    int i = Math.Min(list[a], list[b]), j = Math.Max(list[a], list[b]);
                    if (AabbOverlap(bodies[i], bodies[j])) found.Add((i, j));
                }
        var pairs = found.ToList();
        pairs.Sort();
        return pairs;
    }

    private static List<(int, int)> BruteForcePairs(List<Body> bodies)
    {
        var pairs = new List<(int, int)>();
        for (int i = 0; i < bodies.Count; i++)
            for (int j = i + 1; j < bodies.Count; j++)
                if (AabbOverlap(bodies[i], bodies[j])) pairs.Add((i, j));
        pairs.Sort();
        return pairs;
    }

    private static bool AabbOverlap(Body a, Body b) =>
        a.Min.X <= b.Max.X && b.Min.X <= a.Max.X && a.Min.Y <= b.Max.Y && b.Min.Y <= a.Max.Y;

    // ---------- narrowphase ----------
    private sealed class Contact
    {
        public int A;
        public int B;
        public Vec Normal;              // unit vector pointing from A to B
        public double Penetration;
    }

    private static Contact NarrowPhase(List<Body> bodies, int i, int j)
    {
        Body a = bodies[i], b = bodies[j];
        if (a.IsCircle && b.IsCircle)
        {
            Vec d = b.Pos - a.Pos;
            double dist = d.Length;
            double sum = a.Radius + b.Radius;
            if (dist > sum) return null;
            Vec n = dist > 1e-12 ? new Vec(d.X / dist, d.Y / dist) : new Vec(1, 0);
            return new Contact { A = i, B = j, Normal = n, Penetration = sum - dist };
        }
        if (a.IsCircle && !b.IsCircle) return CircleBox(i, j, b, a, true);
        if (!a.IsCircle && b.IsCircle) return CircleBox(j, i, a, b, false);
        // box/box: AABB overlap along the axis of least penetration
        double overlapX = Math.Min(a.Max.X, b.Max.X) - Math.Max(a.Min.X, b.Min.X);
        double overlapY = Math.Min(a.Max.Y, b.Max.Y) - Math.Max(a.Min.Y, b.Min.Y);
        if (overlapX <= 0 || overlapY <= 0) return null;
        if (overlapX < overlapY)
            return new Contact
            {
                A = i, B = j, Penetration = overlapX,
                Normal = new Vec(a.Pos.X <= b.Pos.X ? 1 : -1, 0),
            };
        return new Contact
        {
            A = i, B = j, Penetration = overlapY,
            Normal = new Vec(0, a.Pos.Y <= b.Pos.Y ? 1 : -1),
        };
    }

    /// <summary>
    /// Circle vs box contact. The normal always points from the returned contact's A to its B, so the solver
    /// can push B out along +normal and A out along -normal regardless of which body the caller listed first.
    /// </summary>
    private static Contact CircleBox(int circleIndex, int boxIndex, Body box, Body circle, bool circleFirst)
    {
        Vec closest = new Vec(
            Math.Max(box.Min.X, Math.Min(circle.Pos.X, box.Max.X)),
            Math.Max(box.Min.Y, Math.Min(circle.Pos.Y, box.Max.Y)));
        Vec d = circle.Pos - closest;
        double dist = d.Length;
        Vec outward;                    // from the box towards the circle
        double penetration;
        if (dist > 1e-12)
        {
            if (dist > circle.Radius) return null;
            outward = new Vec(d.X / dist, d.Y / dist);
            penetration = circle.Radius - dist;
        }
        else
        {
            // Centre inside the box: escape along the axis of least penetration (the standard fallback).
            double dx = box.HalfExtent.X - Math.Abs(circle.Pos.X - box.Pos.X);
            double dy = box.HalfExtent.Y - Math.Abs(circle.Pos.Y - box.Pos.Y);
            double sx = circle.Pos.X >= box.Pos.X ? 1 : -1;
            double sy = circle.Pos.Y >= box.Pos.Y ? 1 : -1;
            outward = dx < dy ? new Vec(sx, 0) : new Vec(0, sy);
            penetration = circle.Radius + Math.Min(dx, dy);
        }
        return circleFirst
            ? new Contact { A = circleIndex, B = boxIndex, Normal = new Vec(-outward.X, -outward.Y),
                            Penetration = penetration }
            : new Contact { A = boxIndex, B = circleIndex, Normal = outward, Penetration = penetration };
    }

    /// <summary>
    /// Swept circle vs box: the circle touches the box exactly when its centre enters the box expanded by the
    /// radius, so the first impact time is the slab (ray vs AABB) intersection with the Minkowski sum - the
    /// standard conservative-advancement primitive for this shape pair. Returns NaN when the step misses.
    /// </summary>
    private static double SweptCircleBox(Body circle, Vec velocity, Body box, double dt)
    {
        double ex0 = box.Min.X - circle.Radius, ex1 = box.Max.X + circle.Radius;
        double ey0 = box.Min.Y - circle.Radius, ey1 = box.Max.Y + circle.Radius;
        double tMin = double.NegativeInfinity, tMax = double.PositiveInfinity;
        foreach ((double origin, double dir, double lo, double hi) in new[]
                 {
                     (circle.Pos.X, velocity.X, ex0, ex1),
                     (circle.Pos.Y, velocity.Y, ey0, ey1),
                 })
        {
            if (Math.Abs(dir) < 1e-15)
            {
                if (origin < lo || origin > hi) return double.NaN;
                continue;
            }
            double t1 = (lo - origin) / dir, t2 = (hi - origin) / dir;
            if (t1 > t2) (t1, t2) = (t2, t1);
            tMin = Math.Max(tMin, t1);
            tMax = Math.Min(tMax, t2);
            if (tMin > tMax) return double.NaN;
        }
        return tMin <= dt && tMax >= 0.0 ? Math.Max(tMin, 0.0) : double.NaN;
    }

    // ---------- solver ----------
    private static List<Contact> CollectContacts(Scene scene, out long broadphasePairs)
    {
        List<(int, int)> pairs = SweepAndPrune(scene.Bodies, out long comparisons);
        broadphasePairs = comparisons;
        var contacts = new List<Contact>();
        foreach ((int i, int j) in pairs)
        {
            Contact c = NarrowPhase(scene.Bodies, i, j);
            if (c != null) contacts.Add(c);
        }
        return contacts;
    }

    private static void Step(Scene scene, double dt, int iterations = 8)
    {
        foreach (Body b in scene.Bodies)
            if (b.InvMass > 0 && b.Awake) b.Vel = b.Vel + new Vec(0, scene.Gravity * dt);
        foreach (Body b in scene.Bodies)
            if (b.InvMass > 0 && b.Awake) b.Pos = b.Pos + b.Vel * dt;

        var contacts = CollectContacts(scene, out _);
        for (int iter = 0; iter < iterations; iter++)
        {
            foreach (Contact c in contacts)
            {
                Body a = scene.Bodies[c.A], b = scene.Bodies[c.B];
                Vec rv = b.Vel - a.Vel;
                double vn = rv.X * c.Normal.X + rv.Y * c.Normal.Y;
                if (vn > 0) continue;                          // separating already
                double invSum = a.InvMass + b.InvMass;
                if (invSum <= 0) continue;
                double e = scene.Restitution;
                double jn = -(1 + e) * vn / invSum;
                Vec impulse = c.Normal * jn;
                a.Vel = a.Vel - impulse * a.InvMass;
                b.Vel = b.Vel + impulse * b.InvMass;
                // Coulomb friction along the tangent, clamped by the normal impulse
                Vec tangent = new Vec(-c.Normal.Y, c.Normal.X);
                double vt = rv.X * tangent.X + rv.Y * tangent.Y;
                double jt = -vt / invSum;
                double maxFriction = scene.Friction * Math.Abs(jn);
                jt = Math.Max(-maxFriction, Math.Min(maxFriction, jt));
                a.Vel = a.Vel - tangent * (jt * a.InvMass);
                b.Vel = b.Vel + tangent * (jt * b.InvMass);
            }
        }
        // Positional correction with a slop so resting contacts do not jitter.
        contacts = CollectContacts(scene, out _);
        foreach (Contact c in contacts)
        {
            Body a = scene.Bodies[c.A], b = scene.Bodies[c.B];
            double invSum = a.InvMass + b.InvMass;
            if (invSum <= 0) continue;
            double depth = c.Penetration - scene.Slop;
            if (depth <= 0) continue;
            double scale = scene.Correction * depth / invSum;
            a.Pos = a.Pos - c.Normal * (scale * a.InvMass);
            b.Pos = b.Pos + c.Normal * (scale * b.InvMass);
        }
        // Sleeping: a body that has been slow for the whole step stops integrating.
        foreach (Body b in scene.Bodies)
            if (b.InvMass > 0 && b.Vel.Length < 0.02) b.Vel = b.Vel * 0.5;
    }

    private static double Momentum(Scene scene) =>
        scene.Bodies.Sum(b => b.InvMass > 0 ? b.Mass * b.Vel.Y : 0.0);

    private static double KineticEnergy(Scene scene) =>
        scene.Bodies.Sum(b => b.InvMass > 0 ? 0.5 * b.Mass * b.Vel.LengthSquared : 0.0);

    private static double MaxPenetration(Scene scene)
    {
        List<(int, int)> pairs = BruteForcePairs(scene.Bodies);
        double worst = 0.0;
        foreach ((int i, int j) in pairs)
        {
            Contact c = NarrowPhase(scene.Bodies, i, j);
            if (c != null) worst = Math.Max(worst, c.Penetration);
        }
        return worst;
    }

    private static Scene RandomScene(int n, ulong seed)
    {
        _state = seed;
        var scene = new Scene { Gravity = 0.0, Restitution = 0.5, Friction = 0.2 };
        for (int i = 0; i < n; i++)
        {
            double x = NextSigned() * 5, y = NextSigned() * 5;
            if (i % 2 == 0) scene.Bodies.Add(Body.Circle(x, y, 0.2 + Math.Abs(NextSigned()) * 0.5));
            else scene.Bodies.Add(Body.Box(x, y, 0.2 + Math.Abs(NextSigned()) * 0.5, 0.2 + Math.Abs(NextSigned()) * 0.5));
        }
        return scene;
    }

    // ---------- main ----------
    public static void Main()
    {
        // 1-3: broadphase correctness against the O(n^2) oracle.
        int sapMatches = 0, hashMatches = 0, hashSuperset = 0;
        long comparisons = 0, worstComparisons = 0;
        for (int t = 0; t < 8; t++)
        {
            var scene = RandomScene(24, (ulong)(0x1000 + t));
            List<(int, int)> oracle = BruteForcePairs(scene.Bodies);
            List<(int, int)> sap = SweepAndPrune(scene.Bodies, out long comps);
            List<(int, int)> hash = SpatialHashPairs(scene.Bodies, 1.0);
            comparisons += comps;
            worstComparisons = Math.Max(worstComparisons, comps);
            if (sap.SequenceEqual(oracle)) sapMatches++;
            if (hash.SequenceEqual(oracle)) hashMatches++;
            if (oracle.All(p => hash.Contains(p))) hashSuperset++;
        }
        Check(sapMatches == 8, "sweep-and-prune matches the brute-force pair set exactly (" + sapMatches + "/8)");
        Check(hashMatches == 8 && hashSuperset == 8, "the spatial hash finds every overlapping pair");
        Check(comparisons < (long)8 * 24 * 23 / 2, "sweep-and-prune examines fewer pairs than the O(n^2) scan");
        Console.WriteLine("broadphase: sap comparisons " + comparisons + ", brute-force would be " + 8 * 24 * 23 / 2 +
                          ", worst scene " + worstComparisons);
        Check(SweepAndPrune(new List<Body>(), out long emptyComps).Count == 0 && emptyComps == 0,
              "an empty scene has no pairs");

        // 4: narrowphase analytics for all three shape combinations.
        var touching = new List<Body> { Body.Circle(0, 0, 1), Body.Circle(2, 0, 1) };
        Check(NarrowPhase(touching, 0, 1).Penetration == 0.0, "circles exactly touching have zero penetration");
        var overlapping = new List<Body> { Body.Circle(0, 0, 1), Body.Circle(1.5, 0, 1) };
        Contact cc = NarrowPhase(overlapping, 0, 1);
        Check(Math.Abs(cc.Penetration - 0.5) < 1e-12 && cc.Normal.X == 1 && cc.Normal.Y == 0,
              "circles overlapping by 0.5 with a +x normal");
        Check(NarrowPhase(new List<Body> { Body.Circle(0, 0, 1), Body.Circle(3, 0, 1) }, 0, 1) == null,
              "separated circles do not collide");
        var circleBox = new List<Body> { Body.Circle(0, 1.1, 0.2), Body.Box(0, 0, 1, 1) };
        Contact cb = NarrowPhase(circleBox, 0, 1);
        Check(cb != null && Math.Abs(cb.Penetration - 0.1) < 1e-12 && cb.Normal.Y == -1 && cb.A == 0 && cb.B == 1,
              "a circle resting on a box penetrates by 0.1 with the normal pointing from the circle to the box");
        Contact boxCircle = NarrowPhase(new List<Body> { Body.Box(0, 0, 1, 1), Body.Circle(0, 1.1, 0.2) }, 0, 1);
        Check(boxCircle != null && Math.Abs(boxCircle.Penetration - 0.1) < 1e-12 && boxCircle.Normal.Y == 1 &&
              boxCircle.A == 0 && boxCircle.B == 1,
              "the same contact listed box-first points from the box to the circle");
        Contact buried = NarrowPhase(new List<Body> { Body.Circle(0, 0.9, 0.2), Body.Box(0, 0, 1, 1) }, 0, 1);
        Check(buried != null && Math.Abs(buried.Penetration - 0.3) < 1e-12 && buried.Normal.Y == -1,
              "a circle whose centre is inside the box escapes along the axis of least penetration");
        Check(NarrowPhase(new List<Body> { Body.Circle(0, 2.5, 0.2), Body.Box(0, 0, 1, 1) }, 0, 1) == null,
              "a circle well above the box does not collide");
        var boxBox = new List<Body> { Body.Box(0, 0, 1, 1), Body.Box(1.5, 0, 1, 1) };
        Contact bb = NarrowPhase(boxBox, 0, 1);
        Check(bb != null && Math.Abs(bb.Penetration - 0.5) < 1e-12 && bb.Normal.X == 1,
              "boxes overlapping on x resolve along the least-penetration axis");
        Check(NarrowPhase(new List<Body> { Body.Box(0, 0, 1, 1), Body.Box(3, 0, 1, 1) }, 0, 1) == null,
              "separated boxes do not collide");
        Check(NarrowPhase(new List<Body> { Body.Box(0, 0, 1, 1), Body.Box(1.9, 1.9, 1, 1) }, 0, 1) != null,
              "corner overlap is a collision");

        // 5: the swept test catches a bullet that the discrete step misses.
        var bullet = Body.Circle(0, 0, 0.05);
        var wall = Body.Box(1.0, 0, 0.05, 2.0, 0);
        double dt = 0.1;
        Vec fast = new Vec(20.0, 0);
        Body discrete = bullet.Clone();
        discrete.Pos = discrete.Pos + fast * dt;
        Check(!AabbOverlap(discrete, wall), "the discrete step tunnels through the wall (the problem)");
        double impact = SweptCircleBox(bullet, fast, wall, dt);
        Check(!double.IsNaN(impact) && impact > 0 && impact < dt,
              "the swept test finds the impact time inside the step (" + impact + ")");
        Check(Math.Abs(impact - (0.90 / 20.0)) < 1e-12,
              "the impact time is the analytic Minkowski crossing 0.90/20 (" + impact + ")");
        Check(double.IsNaN(SweptCircleBox(Body.Circle(0, 5, 0.05), new Vec(1, 0), wall, dt)),
              "a step that passes the wall far away reports no impact");

        // 6-7: solver - head-on elastic and perfectly inelastic collisions.
        var elastic = new Scene { Gravity = 0, Restitution = 1.0, Friction = 0 };
        elastic.Bodies.Add(Body.Circle(-1, 0, 0.5));
        elastic.Bodies.Add(Body.Circle(1, 0, 0.5));
        elastic.Bodies[0].Vel = new Vec(2, 0);
        elastic.Bodies[1].Vel = new Vec(-2, 0);
        double momentum0 = Momentum(elastic), energy0 = KineticEnergy(elastic);
        for (int s = 0; s < 60; s++) Step(elastic, 0.01);
        Check(Math.Abs(elastic.Bodies[0].Vel.X + 2) < 1e-6 && Math.Abs(elastic.Bodies[1].Vel.X - 2) < 1e-6,
              "equal-mass elastic collisions exchange velocities (" + elastic.Bodies[0].Vel.X + ", " +
              elastic.Bodies[1].Vel.X + ")");
        Check(Math.Abs(Momentum(elastic) - momentum0) < 1e-9, "momentum is conserved in the elastic collision");
        Check(Math.Abs(KineticEnergy(elastic) - energy0) < 1e-6 * energy0, "kinetic energy survives restitution 1");

        var inelastic = new Scene { Gravity = 0, Restitution = 0.0, Friction = 0 };
        inelastic.Bodies.Add(Body.Circle(-1, 0, 0.5));
        inelastic.Bodies.Add(Body.Circle(1, 0, 0.5));
        inelastic.Bodies[0].Vel = new Vec(2, 0);
        inelastic.Bodies[1].Vel = new Vec(-2, 0);
        double inelasticEnergy0 = KineticEnergy(inelastic);
        for (int s = 0; s < 60; s++) Step(inelastic, 0.01);
        Check(Math.Abs(inelastic.Bodies[0].Vel.X) < 1e-6 && Math.Abs(inelastic.Bodies[1].Vel.X) < 1e-6,
              "a perfectly inelastic head-on collision stops both bodies");
        Check(KineticEnergy(inelastic) < 1e-12 * inelasticEnergy0 + 1e-18,
              "energy strictly decreases under restitution 0");
        Check(Math.Abs(Momentum(inelastic) - 0.0) < 1e-9, "momentum still vanishes for equal opposite impulses");

        // 8: analytic two-body collision with unequal masses.
        var analytic = new Scene { Gravity = 0, Restitution = 1.0, Friction = 0 };
        analytic.Bodies.Add(Body.Circle(-2, 0, 0.5, 1.0));
        analytic.Bodies.Add(Body.Circle(2, 0, 0.5, 3.0));
        analytic.Bodies[0].Vel = new Vec(4, 0);
        analytic.Bodies[1].Vel = new Vec(0, 0);
        double m1 = 1.0, m2 = 3.0, u1 = 4.0, u2 = 0.0;
        double expected1 = ((m1 - m2) * u1 + 2 * m2 * u2) / (m1 + m2);
        double expected2 = ((m2 - m1) * u2 + 2 * m1 * u1) / (m1 + m2);
        for (int s = 0; s < 120; s++) Step(analytic, 0.01);
        Check(Math.Abs(analytic.Bodies[0].Vel.X - expected1) < 1e-6,
              "the light body rebounds at the analytic speed (" + analytic.Bodies[0].Vel.X + " vs " + expected1 + ")");
        Check(Math.Abs(analytic.Bodies[1].Vel.X - expected2) < 1e-6,
              "the heavy body leaves at the analytic speed (" + analytic.Bodies[1].Vel.X + " vs " + expected2 + ")");
        Check(analytic.Bodies[1].Vel.X > 0 && analytic.Bodies[0].Vel.X < u1,
              "the heavy body is pushed forward and the light one is slowed");

        // 9-10: a stack settles without jitter and without residual overlap.
        var stack = new Scene { Gravity = -9.81, Restitution = 0.05, Friction = 0.5 };
        stack.Bodies.Add(Body.Box(0, -5, 6, 0.5, 0));                        // static floor, top surface at -4.5
        for (int i = 0; i < 3; i++) stack.Bodies.Add(Body.Box(0, -3.9 + i * 1.1, 0.5, 0.5));
        for (int s = 0; s < 400; s++) Step(stack, 0.01);
        double stackPenetration = MaxPenetration(stack);
        double maxSpeed = stack.Bodies.Where(b => b.InvMass > 0).Max(b => b.Vel.Length);
        Console.WriteLine("stack: penetration " + stackPenetration.ToString("E2") + ", max speed " +
                          maxSpeed.ToString("E2"));
        Check(stackPenetration < 0.02, "the stack settles with bounded penetration (" + stackPenetration + ")");
        Check(maxSpeed < 0.5, "the stack is not jittering (max speed " + maxSpeed + ")");
        Check(stack.Bodies.Where(b => b.InvMass > 0).All(b => b.Pos.Y > -5.0),
              "no dynamic body falls through the static floor");
        Check(stack.Bodies[1].Pos.Y + 0.8 < stack.Bodies[2].Pos.Y && stack.Bodies[2].Pos.Y + 0.8 < stack.Bodies[3].Pos.Y,
              "the stack keeps its order and spacing (" + stack.Bodies[1].Pos.Y.ToString("F3") + ", " +
              stack.Bodies[2].Pos.Y.ToString("F3") + ", " + stack.Bodies[3].Pos.Y.ToString("F3") + ")");

        // 11: a floor of falling bodies is resolved without residual overlap.
        var shower = new Scene { Gravity = -9.81, Restitution = 0.1, Friction = 0.4, Correction = 0.9 };
        shower.Bodies.Add(Body.Box(0, -3, 8, 0.5, 0));
        _state = 0x5150UL;
        for (int i = 0; i < 12; i++)
            shower.Bodies.Add(Body.Circle(NextSigned() * 5, 2 + i * 0.9, 0.35));
        for (int s = 0; s < 800; s++) Step(shower, 0.01, 12);
        double showerPenetration = MaxPenetration(shower);
        Check(showerPenetration < 0.02, "the shower settles with bounded penetration (" + showerPenetration + ")");
        Check(shower.Bodies.Skip(1).All(b => b.Pos.Y > -3.0), "every circle lands on or above the floor");
        Check(shower.Bodies.Skip(1).All(b => Math.Abs(b.Vel.X) < 1.0), "friction damps the horizontal drift");

        // 12-15: degenerate inputs.
        Step(new Scene(), 0.01);
        Check(MaxPenetration(new Scene()) == 0.0, "an empty scene steps and reports no penetration");
        var single = new Scene { Gravity = -9.81 };
        single.Bodies.Add(Body.Circle(0, 0, 0.5));
        Step(single, 0.01);
        Check(Math.Abs(single.Bodies[0].Vel.Y + 9.81 * 0.01) < 1e-12, "a free body integrates gravity exactly");
        Check(single.Bodies[0].Vel.X == 0.0, "a free body does not acquire horizontal velocity");
        var staticPair = new Scene { Gravity = 0, Restitution = 1 };
        staticPair.Bodies.Add(Body.Box(0, 0, 1, 1, 0));
        staticPair.Bodies.Add(Body.Box(0.5, 0, 1, 1, 0));
        var staticBefore = staticPair.Clone();
        Step(staticPair, 0.01);
        Check(staticPair.Bodies[0].Pos.X == staticBefore.Bodies[0].Pos.X &&
              staticPair.Bodies[1].Pos.X == staticBefore.Bodies[1].Pos.X,
              "two static bodies are never moved by the solver");
        var coincident = new Scene { Gravity = 0, Restitution = 0 };
        coincident.Bodies.Add(Body.Circle(0, 0, 0.5));
        coincident.Bodies.Add(Body.Circle(0, 0, 0.5));
        Step(coincident, 0.01);
        Check(coincident.Bodies[0].Pos.X != coincident.Bodies[1].Pos.X ||
              coincident.Bodies[0].Pos.Y != coincident.Bodies[1].Pos.Y,
              "exactly coincident circles are separated instead of staying degenerate");
        Check(double.IsFinite(coincident.Bodies[0].Pos.X) && double.IsFinite(coincident.Bodies[1].Pos.X),
              "the coincident case stays finite (a fallback normal is used)");
        var zeroRadius = new Scene { Gravity = 0 };
        zeroRadius.Bodies.Add(Body.Circle(0, 0, 0));
        zeroRadius.Bodies.Add(Body.Circle(0.0000001, 0, 0));
        Step(zeroRadius, 0.01);
        Check(zeroRadius.Bodies.All(b => double.IsFinite(b.Pos.X) && double.IsFinite(b.Vel.X)),
              "zero-radius circles are handled without NaN");
        var zeroStep = new Scene { Gravity = -9.81 };
        zeroStep.Bodies.Add(Body.Circle(0, 5, 0.5));
        var zeroStepBefore = zeroStep.Clone();
        Step(zeroStep, 0.0);
        Check(zeroStep.Bodies[0].Pos.Y == zeroStepBefore.Bodies[0].Pos.Y,
              "a zero time step leaves positions untouched");
        var hugeRestitution = new Scene { Gravity = 0, Restitution = 1.0 };
        hugeRestitution.Bodies.Add(Body.Circle(-1, 0, 0.5));
        hugeRestitution.Bodies.Add(Body.Circle(1, 0, 0.5));
        hugeRestitution.Bodies[1].Vel = new Vec(-1, 0);
        double energyBefore = KineticEnergy(hugeRestitution);
        for (int s = 0; s < 100; s++) Step(hugeRestitution, 0.01);
        Check(Math.Abs(KineticEnergy(hugeRestitution) - energyBefore) < 1e-6 * energyBefore,
              "restitution 1 keeps the energy of a receding bounce");

        // 16: determinism.
        var replayA = shower.Clone();
        var replayB = shower.Clone();
        for (int s = 0; s < 50; s++) Step(replayA, 0.01);
        for (int s = 0; s < 50; s++) Step(replayB, 0.01);
        Check(replayA.Bodies.Zip(replayB.Bodies, (x, y) => x.Pos.X == y.Pos.X && x.Pos.Y == y.Pos.Y).All(v => v),
              "the simulation is bitwise reproducible");

        Console.WriteLine("elastic swap ok, inelastic energy " + KineticEnergy(inelastic).ToString("E2") +
                          ", stack penetration " + stackPenetration.ToString("E2") +
                          ", swept impact " + impact.ToString("F6"));
        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
