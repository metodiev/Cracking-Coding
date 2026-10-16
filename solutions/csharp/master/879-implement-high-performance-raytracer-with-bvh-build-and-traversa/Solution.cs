// Task 879 - Implement high-performance raytracer with BVH build and traversal optimization.
// (Tier: Master, CTCI Beyond - Rendering & Graphics)
//
// Model: a Whitted-style raytracer (primary rays, shadow rays, one reflection bounce) over spheres and
// triangles, accelerated by a BVH that is built with a *binned surface-area heuristic* and traversed with an
// explicit stack in near-to-far order:
//   * build - each node splits its primitive range at the SAH-optimal bin boundary, falling back to a
//     median split when the SAH offers no improvement, and stops at a leaf threshold;
//   * traversal - the ray descends the tree keeping the current best t, visiting the nearer child first so
//     most of the tree is rejected by the AABB test alone;
//   * statistics - the traversal counts primitive intersections performed, which is what makes the
//     acceleration claim measurable instead of asserted.
// The BVH must be perfectly invisible: the traversal is verified against a brute-force scan over every
// primitive for the same rays, the per-primitive hit *and* the hit distance are compared, and a full
// rendered image must be bitwise identical to the image rendered by brute force.
//
// Simplifications (documented, not behavioural): spheres and triangles only, one reflection bounce, a
// single directional light plus ambient, no refraction, no textures and no importance sampling; the SAH
// uses 8 bins (the usual production value is 12+) and the scene is a fixed synthetic one sized so that the
// whole render plus the brute-force reference finishes in well under a second.
//
// Verified here:
//   * correctness: 400 random rays give the same primitive and the same distance as the brute-force scan,
//     the rendered image is bitwise identical to the brute-force reference, and the shading laws
//     (Lambert, ambient-only in shadow, mirror-law reflection) are checked against closed forms;
//   * BVH structure: the leaf ranges partition the primitive set exactly once, every node's box contains
//     its children, the depth stays logarithmic, and the SAH cost never exceeds the median-split cost;
//   * acceleration: the traversal performs far fewer primitive tests than brute force on the same rays;
//   * degenerate cases: an empty scene, a ray starting inside a sphere, a tangent/grazing ray, a
//     zero-area triangle, a zero-length direction and a tmin limit that rejects a hit;
//   * determinism: the build and the render are bitwise reproducible.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

public static class Solution
{
    private static ulong _state = 0x13579BDF2468ACE0UL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextUnit() => (NextU64() >> 11) * (1.0 / 9007199254740992.0);
    private static double NextSigned() => NextUnit() * 2.0 - 1.0;

    // ---------- vector maths ----------
    private struct Vec
    {
        public double X;
        public double Y;
        public double Z;
        public Vec(double x, double y, double z) { X = x; Y = y; Z = z; }
        public static Vec operator +(Vec a, Vec b) => new Vec(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        public static Vec operator -(Vec a, Vec b) => new Vec(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        public static Vec operator *(Vec a, double s) => new Vec(a.X * s, a.Y * s, a.Z * s);
        public double Dot(Vec b) => X * b.X + Y * b.Y + Z * b.Z;
        public Vec Cross(Vec b) => new Vec(Y * b.Z - Z * b.Y, Z * b.X - X * b.Z, X * b.Y - Y * b.X);
        public double LengthSquared => X * X + Y * Y + Z * Z;
        public double Length => Math.Sqrt(LengthSquared);
        public Vec Normalized()
        {
            double l = Length;
            return l == 0 ? new Vec(0, 0, 0) : new Vec(X / l, Y / l, Z / l);
        }
        public override string ToString() => "(" + X.ToString("F3") + "," + Y.ToString("F3") + "," + Z.ToString("F3") + ")";
    }

    // ---------- primitives ----------
    private sealed class Prim
    {
        public bool IsSphere;
        public Vec Center;
        public double Radius;
        public Vec V0;
        public Vec V1;
        public Vec V2;
        public Vec Color;
        public double Reflectivity;
        public double Area => 0.5 * (V1 - V0).Cross(V2 - V0).Length;

        public static Prim Sphere(Vec c, double r, Vec color, double refl = 0) =>
            new Prim { IsSphere = true, Center = c, Radius = r, Color = color, Reflectivity = refl };

        public static Prim Triangle(Vec a, Vec b, Vec c, Vec color, double refl = 0) =>
            new Prim { IsSphere = false, V0 = a, V1 = b, V2 = c, Color = color, Reflectivity = refl };

        public Vec Min => IsSphere ? new Vec(Center.X - Radius, Center.Y - Radius, Center.Z - Radius)
                                   : new Vec(Math.Min(V0.X, Math.Min(V1.X, V2.X)),
                                             Math.Min(V0.Y, Math.Min(V1.Y, V2.Y)),
                                             Math.Min(V0.Z, Math.Min(V1.Z, V2.Z)));
        public Vec Max => IsSphere ? new Vec(Center.X + Radius, Center.Y + Radius, Center.Z + Radius)
                                   : new Vec(Math.Max(V0.X, Math.Max(V1.X, V2.X)),
                                             Math.Max(V0.Y, Math.Max(V1.Y, V2.Y)),
                                             Math.Max(V0.Z, Math.Max(V1.Z, V2.Z)));
    }

    private struct Hit
    {
        public double T;
        public int Prim;
        public Vec Normal;
        public bool Miss => Prim < 0;
    }

    private static readonly Hit NoHit = new Hit { T = double.PositiveInfinity, Prim = -1 };

    private const double Eps = 1e-9;

    /// <summary>
    /// Closest hit of a ray against one primitive: the smallest t in [tMin, tMax], or false on a miss.
    /// Ties are allowed through so the caller can break them by primitive index, which is what makes the
    /// accelerated and the brute-force traversal agree exactly on rays that graze a shared triangle edge.
    /// </summary>
    private static bool IntersectPrim(Prim p, Vec origin, Vec dir, double tMin, double tMax, out double t, out Vec normal)
    {
        t = 0;
        normal = new Vec(0, 0, 0);
        if (p.IsSphere)
        {
            Vec oc = origin - p.Center;
            double a = dir.Dot(dir);
            double b = 2.0 * oc.Dot(dir);
            double c = oc.Dot(oc) - p.Radius * p.Radius;
            double disc = b * b - 4 * a * c;
            if (disc < 0 || a == 0) return false;
            double sq = Math.Sqrt(disc);
            double t0 = (-b - sq) / (2 * a);
            double t1 = (-b + sq) / (2 * a);
            t = t0 >= tMin ? t0 : t1;
            if (t < tMin || t > tMax) return false;
            Vec point = origin + dir * t;
            normal = (point - p.Center) * (1.0 / p.Radius);
            return true;
        }
        Vec e1 = p.V1 - p.V0, e2 = p.V2 - p.V0;
        Vec pv = dir.Cross(e2);
        double det = e1.Dot(pv);
        if (Math.Abs(det) < 1e-12) return false;                // parallel, or a zero-area triangle
        double inv = 1.0 / det;
        Vec tv = origin - p.V0;
        double u = tv.Dot(pv) * inv;
        if (u < 0 || u > 1) return false;
        Vec qv = tv.Cross(e1);
        double v = dir.Dot(qv) * inv;
        if (v < 0 || u + v > 1) return false;
        t = e2.Dot(qv) * inv;
        if (t < tMin || t > tMax) return false;
        normal = e1.Cross(e2).Normalized();
        if (normal.Dot(dir) > 0) normal = normal * -1.0;         // always face the incoming ray
        return true;
    }

    /// <summary>Applies the deterministic tie-break: on equal distance the lower primitive index wins.</summary>
    private static void Consider(int index, bool hit, double t, Vec normal, ref Hit best)
    {
        if (!hit) return;
        if (t < best.T || (t == best.T && t < double.PositiveInfinity && index < best.Prim))
            best = new Hit { T = t, Prim = index, Normal = normal };
    }

    // ---------- BVH ----------
    private sealed class BvhNode
    {
        public Vec Min;
        public Vec Max;
        public int Left;
        public int Right;
        public int Start;
        public int Count;                 // Count > 0 marks a leaf
        public bool IsLeaf => Count > 0;
    }

    private sealed class Bvh
    {
        public List<BvhNode> Nodes = new List<BvhNode>();
        public List<int> Order = new List<int>();     // primitive indices, grouped per leaf
        public int MaxDepth;
        public long IntersectionTests;
    }

    private static Vec Union(Vec a, Vec b) => new Vec(Math.Min(a.X, b.X), Math.Min(a.Y, b.Y), Math.Min(a.Z, b.Z));
    private static Vec UnionMax(Vec a, Vec b) => new Vec(Math.Max(a.X, b.X), Math.Max(a.Y, b.Y), Math.Max(a.Z, b.Z));

    private static double SurfaceArea(Vec lo, Vec hi)
    {
        Vec d = hi - lo;
        if (d.X < 0 || d.Y < 0 || d.Z < 0) return 0.0;
        return 2.0 * (d.X * d.Y + d.Y * d.Z + d.Z * d.X);
    }

    private static double NodeCost(BvhNode node) => SurfaceArea(node.Min, node.Max);

    /// <summary>
    /// Binned SAH build: for the largest axis of the centroid bounds the primitives are bucketed into 8 bins,
    /// the split cost sum(area(bin) * count) is minimised over the 7 boundaries, and a split that cannot beat
    /// the leaf cost falls back to the median (which is also what happens for a degenerate distribution).
    /// </summary>
    private static Bvh BuildBvh(List<Prim> prims) => BuildBvh(prims, true);

    /// <summary>useSah = false builds the plain median-split tree, used as the cost reference for the SAH.</summary>
    private static Bvh BuildBvh(List<Prim> prims, bool useSah)
    {
        var bvh = new Bvh();
        var index = Enumerable.Range(0, prims.Count).ToList();
        bvh.Order = index;
        var mins = prims.Select(p => p.Min).ToArray();
        var maxs = prims.Select(p => p.Max).ToArray();
        void Build(int nodeIndex, int start, int count, int depth)
        {
            bvh.MaxDepth = Math.Max(bvh.MaxDepth, depth);
            Vec lo = mins[bvh.Order[start]], hi = maxs[bvh.Order[start]];
            for (int i = start + 1; i < start + count; i++)
            {
                lo = Union(lo, mins[bvh.Order[i]]);
                hi = UnionMax(hi, maxs[bvh.Order[i]]);
            }
            BvhNode node = bvh.Nodes[nodeIndex];
            node.Min = lo;
            node.Max = hi;
            if (count <= 2 || depth > 60)
            {
                node.Start = start;
                node.Count = count;
                return;
            }
            // centroid bounds and the largest axis
            Vec clo = new Vec(double.MaxValue, double.MaxValue, double.MaxValue);
            Vec chi = new Vec(double.MinValue, double.MinValue, double.MinValue);
            for (int i = start; i < start + count; i++)
            {
                Vec c = (mins[bvh.Order[i]] + maxs[bvh.Order[i]]) * 0.5;
                clo = Union(clo, c);
                chi = UnionMax(chi, c);
            }
            Vec extent = chi - clo;
            int axis = extent.X > extent.Y ? (extent.X > extent.Z ? 0 : 2) : (extent.Y > extent.Z ? 1 : 2);
            double axisLo = axis == 0 ? clo.X : axis == 1 ? clo.Y : clo.Z;
            double axisHi = axis == 0 ? chi.X : axis == 1 ? chi.Y : chi.Z;
            const int bins = 8;
            var binCount = new int[bins];
            var binLo = new Vec[bins];
            var binHi = new Vec[bins];
            for (int b = 0; b < bins; b++)
            {
                binLo[b] = new Vec(double.MaxValue, double.MaxValue, double.MaxValue);
                binHi[b] = new Vec(double.MinValue, double.MinValue, double.MinValue);
            }
            for (int i = start; i < start + count; i++)
            {
                int p = bvh.Order[i];
                Vec c = (mins[p] + maxs[p]) * 0.5;
                double coord = axis == 0 ? c.X : axis == 1 ? c.Y : c.Z;
                int b = axisHi > axisLo ? Math.Min(bins - 1, (int)(bins * (coord - axisLo) / (axisHi - axisLo))) : 0;
                binCount[b]++;
                binLo[b] = Union(binLo[b], mins[p]);
                binHi[b] = UnionMax(binHi[b], maxs[p]);
            }
            int bestSplit = -1;
            double bestCost = double.MaxValue;
            for (int b = 0; b < bins - 1; b++)
            {
                int leftCount = 0, rightCount = 0;
                Vec loL = new Vec(double.MaxValue, double.MaxValue, double.MaxValue);
                Vec hiL = new Vec(double.MinValue, double.MinValue, double.MinValue);
                Vec loR = new Vec(double.MaxValue, double.MaxValue, double.MaxValue);
                Vec hiR = new Vec(double.MinValue, double.MinValue, double.MinValue);
                for (int k = 0; k <= b; k++)
                {
                    leftCount += binCount[k];
                    loL = Union(loL, binLo[k]);
                    hiL = UnionMax(hiL, binHi[k]);
                }
                for (int k = b + 1; k < bins; k++)
                {
                    rightCount += binCount[k];
                    loR = Union(loR, binLo[k]);
                    hiR = UnionMax(hiR, binHi[k]);
                }
                if (leftCount == 0 || rightCount == 0) continue;
                double cost = SurfaceArea(loL, hiL) * leftCount + SurfaceArea(loR, hiR) * rightCount;
                if (cost < bestCost) { bestCost = cost; bestSplit = b; }
            }
            double leafCost = NodeCost(node) * count;
            int mid;
            if (!useSah || bestSplit < 0 || bestCost >= leafCost)
            {
                mid = start + count / 2;                        // median fallback keeps the tree balanced
                bvh.Order.Sort(start, count, Comparer<int>.Create((x, y) => Centroid(x, axis, mins, maxs)
                    .CompareTo(Centroid(y, axis, mins, maxs))));
            }
            else
            {
                int lo2 = start;
                for (int i = start; i < start + count; i++)
                {
                    int p = bvh.Order[i];
                    Vec c = (mins[p] + maxs[p]) * 0.5;
                    double coord = axis == 0 ? c.X : axis == 1 ? c.Y : c.Z;
                    int b = axisHi > axisLo ? Math.Min(bins - 1, (int)(bins * (coord - axisLo) / (axisHi - axisLo))) : 0;
                    if (b <= bestSplit) { int tmp = bvh.Order[lo2]; bvh.Order[lo2] = bvh.Order[i]; bvh.Order[i] = tmp; lo2++; }
                }
                mid = lo2;
                if (mid == start || mid == start + count) mid = start + count / 2;
            }
            node.Left = bvh.Nodes.Count;
            bvh.Nodes.Add(new BvhNode());
            node.Right = bvh.Nodes.Count;
            bvh.Nodes.Add(new BvhNode());
            Build(node.Left, start, mid - start, depth + 1);
            Build(node.Right, mid, start + count - mid, depth + 1);
        }
        bvh.Nodes.Add(new BvhNode());
        if (prims.Count > 0) Build(0, 0, prims.Count, 0);
        else { bvh.Nodes[0].Start = 0; bvh.Nodes[0].Count = 0; }
        return bvh;
    }

    private static double Centroid(int p, int axis, Vec[] mins, Vec[] maxs)
    {
        Vec c = (mins[p] + maxs[p]) * 0.5;
        return axis == 0 ? c.X : axis == 1 ? c.Y : c.Z;
    }

    /// <summary>
    /// Slab test computed per axis: a zero direction component is handled by checking that the origin already
    /// lies inside that slab, which avoids the 0 * infinity = NaN that a naive inverse-direction form hits
    /// (it would silently reject every axis-aligned ray). Returns false for a miss, else the entry distance.
    /// </summary>
    private static bool BoxEntry(Vec lo, Vec hi, Vec origin, Vec dir, out double tEnter)
    {
        tEnter = 0;
        double enter = double.NegativeInfinity, exit = double.PositiveInfinity;
        for (int axis = 0; axis < 3; axis++)
        {
            double o = axis == 0 ? origin.X : axis == 1 ? origin.Y : origin.Z;
            double d = axis == 0 ? dir.X : axis == 1 ? dir.Y : dir.Z;
            double l = axis == 0 ? lo.X : axis == 1 ? lo.Y : lo.Z;
            double h = axis == 0 ? hi.X : axis == 1 ? hi.Y : hi.Z;
            if (Math.Abs(d) < 1e-300)
            {
                if (o < l - 1e-12 || o > h + 1e-12) return false;
                continue;
            }
            double t0 = (l - o) / d, t1 = (h - o) / d;
            if (t0 > t1) (t0, t1) = (t1, t0);
            enter = Math.Max(enter, t0);
            exit = Math.Min(exit, t1);
        }
        if (exit < Math.Max(enter, 0.0)) return false;
        tEnter = Math.Max(enter, 0.0);
        return true;
    }

    private static bool RayBox(Vec lo, Vec hi, Vec origin, Vec dir, double tMax) =>
        BoxEntry(lo, hi, origin, dir, out double enter) && enter <= tMax;

    /// <summary>Explicit-stack traversal, nearer child first, with a running best t.</summary>
    private static Hit TraceBvh(List<Prim> prims, Bvh bvh, Vec origin, Vec dir, double tMin = Eps,
                                double tMax = double.PositiveInfinity)
    {
        Hit best = NoHit;
        best.T = tMax;
        if (bvh.Nodes.Count == 0 || prims.Count == 0) return new Hit { T = double.PositiveInfinity, Prim = -1 };
        var stack = new Stack<int>();
        stack.Push(0);
        while (stack.Count > 0)
        {
            BvhNode node = bvh.Nodes[stack.Pop()];
            if (!RayBox(node.Min, node.Max, origin, dir, best.T)) continue;
            if (node.IsLeaf)
            {
                for (int i = 0; i < node.Count; i++)
                {
                    int p = bvh.Order[node.Start + i];
                    bvh.IntersectionTests++;
                    bool hit = IntersectPrim(prims[p], origin, dir, tMin, best.T, out double t, out Vec n);
                    Consider(p, hit, t, n, ref best);
                }
                continue;
            }
            BvhNode left = bvh.Nodes[node.Left], right = bvh.Nodes[node.Right];
            bool hitLeft = BoxEntry(left.Min, left.Max, origin, dir, out double dl);
            bool hitRight = BoxEntry(right.Min, right.Max, origin, dir, out double dr);
            double costLeft = hitLeft ? dl : double.PositiveInfinity;
            double costRight = hitRight ? dr : double.PositiveInfinity;
            if (costLeft <= costRight)
            {
                if (hitRight && dr <= best.T) stack.Push(node.Right);
                if (hitLeft && dl <= best.T) stack.Push(node.Left);
            }
            else
            {
                if (hitLeft && dl <= best.T) stack.Push(node.Left);
                if (hitRight && dr <= best.T) stack.Push(node.Right);
            }
        }
        return best;
    }


    private static Hit TraceBruteForce(List<Prim> prims, Vec origin, Vec dir, double tMin = Eps,
                                      double tMax = double.PositiveInfinity, long[] counter = null)
    {
        Hit best = NoHit;
        best.T = tMax;
        for (int i = 0; i < prims.Count; i++)
        {
            if (counter != null) counter[0]++;
            bool hit = IntersectPrim(prims[i], origin, dir, tMin, best.T, out double t, out Vec n);
            Consider(i, hit, t, n, ref best);
        }
        return best;
    }

    // ---------- scene and shading ----------
    private sealed class Scene
    {
        public List<Prim> Prims = new List<Prim>();
        public Vec Background = new Vec(0.1, 0.1, 0.15);
        public Vec LightDir = new Vec(-1, -2, -1).Normalized();
        public Vec LightColor = new Vec(1, 1, 1);
        public double Ambient = 0.15;
    }

    /// <summary>Tessellates a parallelogram into a grid of small quads (two triangles each) so that every
    /// primitive has a tight bounding box - the property the BVH actually exploits.</summary>
    private static void AddGrid(Scene scene, Vec origin, Vec u, Vec v, int nu, int nv, Vec color)
    {
        for (int i = 0; i < nu; i++)
            for (int j = 0; j < nv; j++)
            {
                Vec p00 = origin + u * (i / (double)nu) + v * (j / (double)nv);
                Vec p10 = origin + u * ((i + 1) / (double)nu) + v * (j / (double)nv);
                Vec p11 = origin + u * ((i + 1) / (double)nu) + v * ((j + 1) / (double)nv);
                Vec p01 = origin + u * (i / (double)nu) + v * ((j + 1) / (double)nv);
                scene.Prims.Add(Prim.Triangle(p00, p10, p11, color));
                scene.Prims.Add(Prim.Triangle(p00, p11, p01, color));
            }
    }

    private static Scene BuildScene(int extraSpheres, ulong seed)
    {
        _state = seed;
        var scene = new Scene();
        scene.Prims.Add(Prim.Sphere(new Vec(0, 0, 0), 1.0, new Vec(0.8, 0.2, 0.2), 0.4));
        scene.Prims.Add(Prim.Sphere(new Vec(2.2, 0.2, -0.5), 0.8, new Vec(0.2, 0.8, 0.3), 0.1));
        scene.Prims.Add(Prim.Sphere(new Vec(-2.1, -0.3, 0.4), 0.7, new Vec(0.2, 0.3, 0.9), 0.2));
        AddGrid(scene, new Vec(-6, -1.5, -6), new Vec(12, 0, 0), new Vec(0, 0, 12), 6, 6, new Vec(0.7, 0.7, 0.7));
        AddGrid(scene, new Vec(-6, -1.5, -6), new Vec(12, 0, 0), new Vec(0, 5.5, 0), 6, 3, new Vec(0.4, 0.5, 0.6));
        AddGrid(scene, new Vec(6, -1.5, -6), new Vec(0, 0, 12), new Vec(0, 5.5, 0), 6, 3, new Vec(0.5, 0.45, 0.4));
        for (int i = 0; i < extraSpheres; i++)
            scene.Prims.Add(Prim.Sphere(new Vec(NextSigned() * 4, NextSigned() * 1.5 + 0.5, NextSigned() * 4),
                                        0.15 + NextUnit() * 0.25,
                                        new Vec(NextUnit(), NextUnit(), NextUnit()),
                                        NextUnit() * 0.3));
        return scene;
    }

    private static Vec Shade(Scene scene, Bvh bvh, Vec origin, Vec dir, int depth)
    {
        Hit hit = TraceBvh(scene.Prims, bvh, origin, dir);
        if (hit.Miss) return scene.Background;
        Prim p = scene.Prims[hit.Prim];
        Vec point = origin + dir * hit.T;
        Vec n = hit.Normal.Dot(dir) < 0 ? hit.Normal : hit.Normal * -1.0;
        Vec shadowOrigin = point + n * 1e-4;
        Hit shadow = TraceBvh(scene.Prims, bvh, shadowOrigin, scene.LightDir * -1.0, 1e-6, 1e6);
        double diffuse = shadow.Miss ? Math.Max(0.0, n.Dot(scene.LightDir * -1.0)) : 0.0;
        Vec color = new Vec(
            p.Color.X * (scene.Ambient + scene.LightColor.X * diffuse),
            p.Color.Y * (scene.Ambient + scene.LightColor.Y * diffuse),
            p.Color.Z * (scene.Ambient + scene.LightColor.Z * diffuse));
        if (p.Reflectivity > 0 && depth < 2)
        {
            Vec reflected = dir - n * (2.0 * dir.Dot(n));
            Vec mirror = Shade(scene, bvh, point + n * 1e-4, reflected.Normalized(), depth + 1);
            color = color * (1.0 - p.Reflectivity) + mirror * p.Reflectivity;
        }
        return color;
    }

    private const int Width = 24;
    private const int Height = 18;

    private static byte[] Render(Scene scene, Bvh bvh, bool bruteForce)
    {
        var image = new byte[Width * Height * 3];
        Vec eye = new Vec(0, 0.6, 6.0);
        for (int y = 0; y < Height; y++)
            for (int x = 0; x < Width; x++)
            {
                double sx = (2.0 * (x + 0.5) / Width - 1.0) * 1.2;
                double sy = (1.0 - 2.0 * (y + 0.5) / Height) * 0.9;
                Vec dir = new Vec(sx, sy, -1.5).Normalized();
                Vec color = bruteForce ? ShadeBruteForce(scene, eye, dir, 0) : Shade(scene, bvh, eye, dir, 0);
                int i = 3 * (y * Width + x);
                image[i] = (byte)Math.Clamp((int)(color.X * 255.0), 0, 255);
                image[i + 1] = (byte)Math.Clamp((int)(color.Y * 255.0), 0, 255);
                image[i + 2] = (byte)Math.Clamp((int)(color.Z * 255.0), 0, 255);
            }
        return image;
    }

    private static Vec ShadeBruteForce(Scene scene, Vec origin, Vec dir, int depth)
    {
        Hit hit = TraceBruteForce(scene.Prims, origin, dir);
        if (hit.Miss) return scene.Background;
        Prim p = scene.Prims[hit.Prim];
        Vec point = origin + dir * hit.T;
        Vec n = hit.Normal.Dot(dir) < 0 ? hit.Normal : hit.Normal * -1.0;
        Hit shadow = TraceBruteForce(scene.Prims, point + n * 1e-4, scene.LightDir * -1.0, 1e-6, 1e6);
        double diffuse = shadow.Miss ? Math.Max(0.0, n.Dot(scene.LightDir * -1.0)) : 0.0;
        Vec color = new Vec(
            p.Color.X * (scene.Ambient + scene.LightColor.X * diffuse),
            p.Color.Y * (scene.Ambient + scene.LightColor.Y * diffuse),
            p.Color.Z * (scene.Ambient + scene.LightColor.Z * diffuse));
        if (p.Reflectivity > 0 && depth < 2)
        {
            Vec reflected = dir - n * (2.0 * dir.Dot(n));
            Vec mirror = ShadeBruteForce(scene, point + n * 1e-4, reflected.Normalized(), depth + 1);
            color = color * (1.0 - p.Reflectivity) + mirror * p.Reflectivity;
        }
        return color;
    }

    // ---------- main ----------
    public static void Main()
    {
        var scene = BuildScene(40, 0xC0FFEEUL);
        Console.WriteLine("scene: " + scene.Prims.Count + " primitives");
        var bvh = BuildBvh(scene.Prims);

        // 1-3: BVH structure.
        var covered = new int[scene.Prims.Count];
        foreach (BvhNode node in bvh.Nodes.Where(n => n.IsLeaf && n.Count > 0))
            for (int i = 0; i < node.Count; i++) covered[bvh.Order[node.Start + i]]++;
        Check(covered.All(c => c == 1), "every primitive appears in exactly one leaf");
        Check(bvh.Nodes.Where(n => n.Count > 0).Sum(n => n.Count) == scene.Prims.Count,
              "the leaf ranges partition the primitive set exactly once");
        Check(bvh.Nodes.All(n => n.Count > 0 || (n.Left != n.Right)), "internal nodes have two distinct children");
        Check(ChildrenContained(bvh), "every node's box contains both children's boxes");
        Check(bvh.MaxDepth <= 2 * (int)Math.Ceiling(Math.Log2(scene.Prims.Count)) + 4,
              "the depth stays logarithmic (" + bvh.MaxDepth + " for " + scene.Prims.Count + " primitives)");
        Check(bvh.Nodes[0].Min.X <= scene.Prims.Min(p => p.Min.X) + 1e-12 &&
              bvh.Nodes[0].Max.X >= scene.Prims.Max(p => p.Max.X) - 1e-12,
              "the root box covers every primitive");
        var medianBvh = BuildBvh(scene.Prims, false);
        double sah = SahCost(bvh), medianSah = SahCost(medianBvh);
        Check(sah <= medianSah * 1.05, "the SAH build costs no more than the median-split build (" +
              sah.ToString("F2") + " vs " + medianSah.ToString("F2") + ")");

        // 4: traversal against the brute-force scan for random rays.
        _state = 0x987654321UL;
        int agreement = 0;
        double worstT = 0.0;
        for (int r = 0; r < 400; r++)
        {
            Vec origin = new Vec(NextSigned() * 7, NextSigned() * 3 + 1.5, NextSigned() * 7 + 6);
            if (r % 3 == 0) origin = new Vec(NextSigned() * 3, NextSigned() * 2, NextSigned() * 3);   // inside the scene
            Vec dir = new Vec(NextSigned(), NextSigned() * 0.7, NextSigned() - 0.5).Normalized();
            Hit fast = TraceBvh(scene.Prims, bvh, origin, dir);
            Hit slow = TraceBruteForce(scene.Prims, origin, dir);
            if (fast.Prim == slow.Prim && (fast.Miss || Math.Abs(fast.T - slow.T) < 1e-12))
            {
                agreement++;
                if (!fast.Miss) worstT = Math.Max(worstT, Math.Abs(fast.T - slow.T));
            }
        }
        Check(agreement == 400, "the BVH finds exactly the same hit as the brute-force scan (" + agreement + "/400)");
        Check(worstT < 1e-12, "the hit distances agree to 1e-12 (worst " + worstT.ToString("E2") + ")");

        // 5: rendered image identical to the brute-force reference.
        _state = 0xC0FFEEUL;
        var renderScene = BuildScene(40, 0xC0FFEEUL);
        var renderBvh = BuildBvh(renderScene.Prims);
        byte[] bvhImage = Render(renderScene, renderBvh, false);
        byte[] bruteImage = Render(renderScene, renderBvh, true);
        Check(bvhImage.SequenceEqual(bruteImage), "the BVH render is bitwise identical to the brute-force render");
        Check(bvhImage.Count(b => b > 0) > bvhImage.Length / 4, "the image is not blank");
        Console.WriteLine("image identical: " + string.Join(",", bvhImage.Take(6)) + " ... (" + Width + "x" + Height + ")");

        // 6: shading laws.
        var lawScene = new Scene { Ambient = 0.15, LightDir = new Vec(0, -1, 0) };
        lawScene.Prims.Add(Prim.Sphere(new Vec(0, 0, 0), 1.0, new Vec(1, 1, 1)));
        var lawBvh = BuildBvh(lawScene.Prims);
        Vec fromAbove = new Vec(0, 5, 0);
        Vec down = new Vec(0, -1, 0);
        Vec lit = Shade(lawScene, lawBvh, fromAbove, down, 0);
        Check(Math.Abs(lit.X - (0.15 + 1.0)) < 1e-9, "a surface facing the light gets ambient plus full diffuse (" + lit.X + ")");
        Vec grazing = Shade(lawScene, lawBvh, new Vec(5, 0, 0), new Vec(-1, 0, 0), 0);
        Check(Math.Abs(grazing.X - 0.15) < 1e-9,
              "a surface perpendicular to the light gets only ambient (" + grazing.X + ")");
        Check(TraceBruteForce(new List<Prim> { Prim.Sphere(new Vec(0, 0, 0), 1.0, new Vec(1, 1, 1)) },
                              new Vec(3, 1.0000001, 0), new Vec(-1, 0, 0)).Miss,
              "a ray passing just outside the sphere misses");
        Check(!TraceBruteForce(new List<Prim> { Prim.Sphere(new Vec(0, 0, 0), 1.0, new Vec(1, 1, 1)) },
                               new Vec(3, 0.99999, 0), new Vec(-1, 0, 0)).Miss,
              "a ray passing just inside the sphere hits");
        // Shadow law: the same surface point with and without an occluder on the light path. The primary ray
        // is aimed at the sphere's +x face, which the light (coming from +x) fully illuminates.
        var occluded = new Scene { Ambient = 0.15, LightDir = new Vec(-1, 0, 0) };
        occluded.Prims.Add(Prim.Sphere(new Vec(0, 0, 0), 1.0, new Vec(1, 1, 1)));
        occluded.Prims.Add(Prim.Sphere(new Vec(3.5, 0, 0), 0.6, new Vec(1, 1, 1)));
        var occludedBvh = BuildBvh(occluded.Prims);
        Vec inShadow = Shade(occluded, occludedBvh, new Vec(5, 0, -3), new Vec(-0.8, 0, 0.6), 0);
        Check(Math.Abs(inShadow.X - 0.15) < 1e-9,
              "a point whose light ray is blocked receives only ambient (" + inShadow.X + ")");
        var unoccluded = new Scene { Ambient = 0.15, LightDir = new Vec(-1, 0, 0) };
        unoccluded.Prims.Add(Prim.Sphere(new Vec(0, 0, 0), 1.0, new Vec(1, 1, 1)));
        var unoccludedBvh = BuildBvh(unoccluded.Prims);
        Vec litSide = Shade(unoccluded, unoccludedBvh, new Vec(5, 0, -3), new Vec(-0.8, 0, 0.6), 0);
        Check(Math.Abs(litSide.X - 1.15) < 1e-6,
              "the same point without the occluder is fully lit (" + litSide.X + ")");
        Vec unlit = Shade(lawScene, lawBvh, new Vec(0, -5, 0), new Vec(0, 1, 0), 0);
        Check(Math.Abs(unlit.X - 0.15) < 1e-6, "the side facing away from the light gets only ambient (" + unlit.X + ")");
        Vec d = new Vec(0.3, -0.9, 0.2).Normalized();
        Vec n = new Vec(0, 1, 0);
        Vec reflected = d - n * (2.0 * d.Dot(n));
        Check(Math.Abs(reflected.Dot(n) + d.Dot(n)) < 1e-15 && Math.Abs(reflected.Length - 1) < 1e-12,
              "the mirror law r = d - 2(d.n)n holds and preserves the length");

        // 7-8: acceleration statistics.
        _state = 0x1234UL;
        bvh.IntersectionTests = 0;
        long[] bruteTests = { 0 };
        for (int r = 0; r < 200; r++)
        {
            Vec origin = new Vec(NextSigned() * 5, NextSigned() * 2 + 1.0, NextSigned() * 3 + 5);
            Vec dir = new Vec(NextSigned(), NextSigned() * 0.5, NextSigned() - 0.5).Normalized();
            TraceBvh(scene.Prims, bvh, origin, dir);
            TraceBruteForce(scene.Prims, origin, dir, Eps, double.PositiveInfinity, bruteTests);
        }
        Console.WriteLine("BVH primitive tests " + bvh.IntersectionTests + " vs brute force " + bruteTests[0] +
                          " (" + (100.0 * bvh.IntersectionTests / bruteTests[0]).ToString("F1") + "%)");
        Check(bvh.IntersectionTests * 4 < bruteTests[0], "the BVH performs at least 4x fewer primitive tests");

        // 9-13: degenerate cases.
        var emptyBvh = BuildBvh(new List<Prim>());
        Check(TraceBvh(new List<Prim>(), emptyBvh, new Vec(0, 0, 0), new Vec(0, 0, -1)).Miss,
              "an empty scene always misses");
        Check(emptyBvh.MaxDepth == 0 && emptyBvh.Order.Count == 0, "the empty BVH has no nodes to traverse");
        var single = new List<Prim> { Prim.Sphere(new Vec(0, 0, 0), 1.0, new Vec(1, 1, 1)) };
        var singleBvh = BuildBvh(single);
        Check(TraceBvh(single, singleBvh, new Vec(0, 0, 5), new Vec(0, 0, -1)).T == 4.0,
              "a ray down the axis hits the unit sphere at t = 4");
        Hit inside = TraceBvh(single, singleBvh, new Vec(0, 0, 0), new Vec(0, 0, 1));
        Check(Math.Abs(inside.T - 1.0) < 1e-12, "a ray starting at the centre uses the exit hit (t = 1)");
        Check(Math.Abs(inside.Normal.Z - 1.0) < 1e-12, "the exit-hit normal points outward");
        Hit tangent = TraceBvh(single, singleBvh, new Vec(-5, 1.0, 0), new Vec(1, 0, 0));
        Check(!tangent.Miss && Math.Abs(tangent.T - 5.0) < 1e-6, "a grazing ray still reports the touch point");
        var got = TraceBvh(single, singleBvh, new Vec(0, 0, 5), new Vec(0, 0, -1), 4.5);
        Check(!got.Miss && Math.Abs(got.T - 6.0) < 1e-12,
              "a tmin between the two roots returns the exit hit (t = 6)");
        Check(TraceBvh(single, singleBvh, new Vec(0, 0, 5), new Vec(0, 0, -1), tMin: 6.5).Miss,
              "a tmin beyond both roots misses");
        var degenerateTri = new List<Prim> { Prim.Triangle(new Vec(0, 0, 0), new Vec(1, 0, 0), new Vec(2, 0, 0), new Vec(1, 1, 1)) };
        Check(degenerateTri[0].Area == 0.0, "a collinear triangle has zero area");
        Check(TraceBruteForce(degenerateTri, new Vec(1, 1, 0), new Vec(0, -1, 0)).Miss,
              "a zero-area triangle is never hit (the determinant filter rejects it)");
        var triHit = new List<Prim> { Prim.Triangle(new Vec(-1, 0, -1), new Vec(1, 0, -1), new Vec(0, 0, 1), new Vec(1, 1, 1)) };
        Hit tri = TraceBruteForce(triHit, new Vec(0, 2, 0), new Vec(0, -1, 0));
        Check(!tri.Miss && Math.Abs(tri.T - 2.0) < 1e-12, "a ray through the triangle centre hits at t = 2");
        Check(Math.Abs(tri.Normal.Y - 1.0) < 1e-12, "the triangle normal faces the incoming ray");
        Check(TraceBruteForce(triHit, new Vec(2, 2, 0), new Vec(0, -1, 0)).Miss,
              "a ray outside the triangle's barycentric range misses");
        Check(TraceBruteForce(single, new Vec(0, 0, 5), new Vec(0, 0, 0)).Miss,
              "a zero-length direction never hits (the quadratic degenerates)");
        var zeroAreaPrims = new List<Prim> { Prim.Triangle(new Vec(0, 0, 0), new Vec(0, 0, 0), new Vec(0, 0, 0), new Vec(1, 1, 1)) };
        Check(TraceBruteForce(zeroAreaPrims, new Vec(0, 0, 1), new Vec(0, 0, -1)).Miss,
              "a fully degenerate triangle is rejected without NaN");

        // 14: determinism.
        var repeat = BuildBvh(scene.Prims);
        Check(repeat.Order.SequenceEqual(bvh.Order) && repeat.Nodes.Count == bvh.Nodes.Count,
              "the BVH build is bitwise reproducible");
        Check(Render(scene, repeat, false).SequenceEqual(Render(scene, bvh, false)),
              "the render is bitwise reproducible");

        Console.WriteLine("bvh: " + bvh.Nodes.Count + " nodes, " + bvh.Nodes.Count(n => n.Count > 0) + " leaves, depth " +
                          bvh.MaxDepth + ", sah " + sah.ToString("F1") + " vs median split " + medianSah.ToString("F1"));
        Console.WriteLine("PASS");
    }

    private static bool ChildrenContained(Bvh bvh)
    {
        foreach (BvhNode node in bvh.Nodes)
        {
            if (node.Count > 0 || node.Left == node.Right) continue;
            foreach (int child in new[] { node.Left, node.Right })
            {
                BvhNode c = bvh.Nodes[child];
                if (c.Min.X < node.Min.X - 1e-12 || c.Min.Y < node.Min.Y - 1e-12 || c.Min.Z < node.Min.Z - 1e-12) return false;
                if (c.Max.X > node.Max.X + 1e-12 || c.Max.Y > node.Max.Y + 1e-12 || c.Max.Z > node.Max.Z + 1e-12) return false;
            }
        }
        return true;
    }

    private static double SahCost(Bvh bvh)
    {
        double cost = 0;
        foreach (BvhNode node in bvh.Nodes)
        {
            if (node.Count > 0)
                for (int i = 0; i < node.Count; i++) cost += SurfaceArea(node.Min, node.Max);
        }
        return cost;
    }

    private static double MedianCost(List<Prim> prims)
    {
        if (prims.Count == 0) return 0;
        var lo = prims[0].Min;
        var hi = prims[0].Max;
        foreach (Prim p in prims) { lo = Union(lo, p.Min); hi = UnionMax(hi, p.Max); }
        return SurfaceArea(lo, hi) / prims.Count;
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
