// Task 876 - Implement parallel N-body gravitational simulator with GPU offload sketch.
// (Tier: Master, CTCI Beyond - Numerical & Scientific Computing)
//
// Model: a direct-sum gravitational N-body integrator in 3D with a deterministic, explicitly scheduled
// parallel decomposition and a typed host/device split:
//   * Host kernel - for every target body i the acceleration is the sum over j != i of
//     G m_j (p_j - p_i) / (|p_j - p_i|^2 + eps^2)^(3/2), accumulated in increasing j order. The lanes of the
//     task partition the *target* indices, so no accumulator is shared and each body's summation order is
//     fixed: the result must therefore be bitwise identical for 1, 2, 4 or 8 lanes - the parallel-safety
//     property being verified.
//   * Pairwise kernel - the symmetric half-loop over i < j (one force evaluation per unordered pair, both
//     accelerations updated from it). It is the cheaper form and is cross-checked against the lane kernel;
//     the pair count is asserted to be exactly N(N-1)/2.
//   * Device offload - a typed device model with staged buffers: an offload uploads the state once, runs
//     `stepsPerOffload` integration steps inside the kernel and downloads the result. The accounting
//     (transfers, bytes, kernels) is checked against closed-form expectations, and the device path must
//     reproduce the host path bit for bit.
// The integrator is velocity Verlet (leapfrog), a symplectic scheme, which is what makes the energy error
// bounded rather than secular - the control experiment against explicit Euler verifies that claim.
//
// Simplifications (documented, not behavioural): no threads, no CUDA and no hardware counters - lanes are
// executed sequentially in a fixed order, the device is an in-process model, and the transfer cost is
// counted in bytes rather than seconds; the softening length keeps the direct sum finite.
//
// Verified here:
//   * conservation: linear momentum to machine precision, angular momentum and energy bounded over the
//     integration window, and energy drift of the symplectic scheme an order of magnitude below Euler's;
//   * closed-form oracle: the two-body circular orbit keeps its separation and returns to its initial state
//     after the analytic Kepler period T = 2*pi*sqrt(r^3/(G(m1+m2)));
//   * cross-implementation: the lane kernel agrees with the independent pairwise kernel and both agree with
//     a separate O(N^2) reference written without any decomposition;
//   * parallel invariants: bitwise lane invariance for lane counts 1..8, and exactly N(N-1)/2 pairwise
//     force evaluations in the symmetric kernel;
//   * offload: device and host states are bitwise identical, bytes transferred match 2*N*6 doubles per
//     offload, and batching k steps into one offload costs k times fewer transfers - the cost bound;
//   * degenerate inputs: N = 0, N = 1 (exact uniform motion), two bodies from rest, a collinear
//     configuration, and a determinism replay.
using System;
using System.Collections.Generic;
using System.Linq;

public static class Solution
{
    private static ulong _state = 0x5EED5EED5EED5EEDUL;

    private static ulong NextU64()
    {
        _state += 0x9E3779B97F4A7C15UL;
        ulong z = _state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
        return z ^ (z >> 31);
    }

    private static double NextSigned() => ((NextU64() >> 11) * (1.0 / 9007199254740992.0)) * 2.0 - 1.0;

    // ---------- vectors ----------
    private struct Vec3
    {
        public double X;
        public double Y;
        public double Z;

        public Vec3(double x, double y, double z) { X = x; Y = y; Z = z; }
        public static Vec3 operator +(Vec3 a, Vec3 b) => new Vec3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);
        public static Vec3 operator -(Vec3 a, Vec3 b) => new Vec3(a.X - b.X, a.Y - b.Y, a.Z - b.Z);
        public static Vec3 operator *(Vec3 a, double s) => new Vec3(a.X * s, a.Y * s, a.Z * s);
        public double LengthSquared => X * X + Y * Y + Z * Z;
        public override string ToString() => "(" + X.ToString("F6") + "," + Y.ToString("F6") + "," + Z.ToString("F6") + ")";
    }

    private sealed class Body
    {
        public double Mass;
        public Vec3 Pos;
        public Vec3 Vel;
        public Vec3 Acc;
    }

    private sealed class System809
    {
        public List<Body> Bodies = new List<Body>();
        public double G = 1.0;
        public double Softening = 0.05;

        public int N => Bodies.Count;

        public System809 Clone()
        {
            var s = new System809 { G = G, Softening = Softening };
            foreach (Body b in Bodies)
                s.Bodies.Add(new Body { Mass = b.Mass, Pos = b.Pos, Vel = b.Vel, Acc = b.Acc });
            return s;
        }
    }

    // ---------- host kernels ----------
    private static long _pairEvaluations;

    /// <summary>
    /// Lane kernel: lanes split the *target* indices, every target accumulates over j in increasing order, so
    /// the result cannot depend on how many lanes run.
    /// </summary>
    private static void AccelerationsLanes(System809 sys, int lanes, bool accumulatePairs = false)
    {
        int n = sys.N;
        int laneCount = Math.Max(1, lanes);
        double g = sys.G, eps2 = sys.Softening * sys.Softening;
        for (int lane = 0; lane < laneCount; lane++)
            for (int i = lane; i < n; i += laneCount)
            {
                Vec3 acc = new Vec3(0, 0, 0);
                Vec3 pi = sys.Bodies[i].Pos;
                for (int j = 0; j < n; j++)
                {
                    if (j == i) continue;
                    Vec3 d = sys.Bodies[j].Pos - pi;
                    double r2 = d.LengthSquared + eps2;
                    double inv = 1.0 / (r2 * Math.Sqrt(r2));
                    acc = acc + d * (g * sys.Bodies[j].Mass * inv);
                }
                sys.Bodies[i].Acc = acc;
                if (accumulatePairs) _pairEvaluations += n - 1;
            }
    }

    /// <summary>Symmetric half-loop: one force per unordered pair, both accelerations updated from it.</summary>
    private static void AccelerationsPairs(System809 sys, bool accumulatePairs = false)
    {
        int n = sys.N;
        double g = sys.G, eps2 = sys.Softening * sys.Softening;
        for (int i = 0; i < n; i++) sys.Bodies[i].Acc = new Vec3(0, 0, 0);
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
            {
                Vec3 d = sys.Bodies[j].Pos - sys.Bodies[i].Pos;
                double r2 = d.LengthSquared + eps2;
                double inv = g / (r2 * Math.Sqrt(r2));
                Body a = sys.Bodies[i], b = sys.Bodies[j];
                a.Acc = a.Acc + d * (inv * b.Mass);
                b.Acc = b.Acc - d * (inv * a.Mass);
                if (accumulatePairs) _pairEvaluations++;
            }
    }

    /// <summary>Independent reference: the plain double loop, no decomposition, no symmetric trick.</summary>
    private static void AccelerationsReference(System809 sys)
    {
        int n = sys.N;
        double g = sys.G, eps2 = sys.Softening * sys.Softening;
        var fresh = new Vec3[n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                Vec3 d = sys.Bodies[j].Pos - sys.Bodies[i].Pos;
                double r2 = d.LengthSquared + eps2;
                double f = g * sys.Bodies[j].Mass / (r2 * Math.Sqrt(r2));
                fresh[i] = fresh[i] + d * f;
            }
        }
        for (int i = 0; i < n; i++) sys.Bodies[i].Acc = fresh[i];
    }

    // ---------- integrator ----------
    private static void StepVelocityVerlet(System809 sys, double dt, int lanes)
    {
        AccelerationsLanes(sys, lanes);
        foreach (Body b in sys.Bodies) b.Vel = b.Vel + b.Acc * (0.5 * dt);
        foreach (Body b in sys.Bodies) b.Pos = b.Pos + b.Vel * dt;
        AccelerationsLanes(sys, lanes);
        foreach (Body b in sys.Bodies) b.Vel = b.Vel + b.Acc * (0.5 * dt);
    }

    /// <summary>Explicit Euler - kept only as the non-symplectic control for the energy-drift comparison.</summary>
    private static void StepEuler(System809 sys, double dt, int lanes)
    {
        AccelerationsLanes(sys, lanes);
        foreach (Body b in sys.Bodies) b.Pos = b.Pos + b.Vel * dt;
        foreach (Body b in sys.Bodies) b.Vel = b.Vel + b.Acc * dt;
    }

    // ---------- observables ----------
    private static Vec3 TotalMomentum(System809 sys)
    {
        Vec3 p = new Vec3(0, 0, 0);
        foreach (Body b in sys.Bodies) p = p + b.Vel * b.Mass;
        return p;
    }

    private static Vec3 AngularMomentum(System809 sys)
    {
        Vec3 l = new Vec3(0, 0, 0);
        foreach (Body b in sys.Bodies)
        {
            l = l + new Vec3(b.Pos.Y * b.Vel.Z - b.Pos.Z * b.Vel.Y,
                             b.Pos.Z * b.Vel.X - b.Pos.X * b.Vel.Z,
                             b.Pos.X * b.Vel.Y - b.Pos.Y * b.Vel.X) * b.Mass;
        }
        return l;
    }

    private static double KineticEnergy(System809 sys) =>
        sys.Bodies.Sum(b => 0.5 * b.Mass * b.Vel.LengthSquared);

    private static double PotentialEnergy(System809 sys)
    {
        double eps2 = sys.Softening * sys.Softening;
        double e = 0.0;
        var list = sys.Bodies;
        for (int i = 0; i < list.Count; i++)
            for (int j = i + 1; j < list.Count; j++)
                e -= sys.G * list[i].Mass * list[j].Mass / Math.Sqrt((list[j].Pos - list[i].Pos).LengthSquared + eps2);
        return e;
    }

    private static double TotalEnergy(System809 sys) => KineticEnergy(sys) + PotentialEnergy(sys);

    // ---------- device offload model ----------
    private sealed class Device
    {
        public long Transfers;
        public long BytesTransferred;
        public long KernelLaunches;
        public int ResidentBodies;
    }

    /// <summary>
    /// Stages the state to the device, runs `steps` integration steps inside the kernel (the device keeps its
    /// own copy, so no per-step transfer is needed) and stages the result back.
    /// </summary>
    private static void SimulateOnDevice(System809 sys, double dt, int lanes, int steps, Device device)
    {
        const int doublesPerBody = 6;                          // position + velocity
        long uploadBytes = (long)sys.N * doublesPerBody * 8;
        device.Transfers += 2;                                 // one upload, one download
        device.BytesTransferred += 2 * uploadBytes;
        device.KernelLaunches++;
        device.ResidentBodies = sys.N;
        var resident = sys.Clone();
        for (int s = 0; s < steps; s++) StepVelocityVerlet(resident, dt, lanes);
        for (int i = 0; i < sys.N; i++)
        {
            sys.Bodies[i].Pos = resident.Bodies[i].Pos;
            sys.Bodies[i].Vel = resident.Bodies[i].Vel;
            sys.Bodies[i].Acc = resident.Bodies[i].Acc;
        }
    }

    // ---------- scenarios ----------
    private static System809 PlummerLike(int n, ulong seed)
    {
        _state = seed;
        var sys = new System809();
        for (int i = 0; i < n; i++)
            sys.Bodies.Add(new Body
            {
                Mass = 0.5 + NextU64() % 1000 / 500.0,
                Pos = new Vec3(NextSigned() * 2.0, NextSigned() * 2.0, NextSigned() * 2.0),
                Vel = new Vec3(NextSigned() * 0.2, NextSigned() * 0.2, NextSigned() * 0.2),
            });
        return sys;
    }

    private static System809 CircularTwoBody(double separation, double mass)
    {
        // Two equal masses on a circle of radius r/2 about the centre of mass; omega^2 = G(m1+m2)/r^3.
        double r = separation;
        double omega = Math.Sqrt(1.0 * (mass + mass) / (r * r * r));
        var sys = new System809 { Softening = 0.0 };
        sys.Bodies.Add(new Body { Mass = mass, Pos = new Vec3(-r / 2, 0, 0), Vel = new Vec3(0, -omega * r / 2, 0) });
        sys.Bodies.Add(new Body { Mass = mass, Pos = new Vec3(r / 2, 0, 0), Vel = new Vec3(0, omega * r / 2, 0) });
        return sys;
    }

    private static double MaxAbsDiff(System809 a, System809 b)
    {
        double worst = 0.0;
        for (int i = 0; i < a.N; i++)
        {
            worst = Math.Max(worst, (a.Bodies[i].Pos - b.Bodies[i].Pos).LengthSquared);
            worst = Math.Max(worst, (a.Bodies[i].Vel - b.Bodies[i].Vel).LengthSquared);
        }
        return Math.Sqrt(worst);
    }

    // ---------- main ----------
    public static void Main()
    {
        var cluster = PlummerLike(48, 0xABCDEFUL);

        // 1: momentum is conserved to machine precision by the symmetric direct sum.
        Vec3 p0 = TotalMomentum(cluster);
        var shot = cluster.Clone();
        for (int s = 0; s < 40; s++) StepVelocityVerlet(shot, 0.002, 4);
        Vec3 p1 = TotalMomentum(shot);
        Check((p1 - p0).LengthSquared < 1e-24,
              "linear momentum conserved to 1e-24 (" + (p1 - p0).LengthSquared.ToString("E2") + ")");
        Vec3 l0 = AngularMomentum(cluster);
        Vec3 l1 = AngularMomentum(shot);
        Check((l1 - l0).LengthSquared < 1e-20, "angular momentum conserved over 40 steps");
        Check(KineticEnergy(cluster) > 0 && PotentialEnergy(cluster) < 0 && TotalEnergy(cluster) < 0,
              "the cluster is bound: positive kinetic, negative potential, negative total");

        // 2: the symplectic scheme keeps the energy error bounded; Euler does not.
        double energy0 = TotalEnergy(cluster);
        var verlet = cluster.Clone();
        var euler = cluster.Clone();
        double worstVerlet = 0.0, worstEuler = 0.0;
        for (int s = 0; s < 400; s++)
        {
            StepVelocityVerlet(verlet, 0.0005, 4);
            StepEuler(euler, 0.0005, 4);
            worstVerlet = Math.Max(worstVerlet, Math.Abs((TotalEnergy(verlet) - energy0) / energy0));
            worstEuler = Math.Max(worstEuler, Math.Abs((TotalEnergy(euler) - energy0) / energy0));
        }
        Console.WriteLine("energy drift over 400 steps: verlet " + worstVerlet.ToString("E3") +
                          " vs euler " + worstEuler.ToString("E3"));
        Check(worstVerlet < 1e-3, "symplectic energy error stays below 1e-3 (" + worstVerlet + ")");
        Check(worstVerlet * 10 < worstEuler, "the symplectic scheme beats explicit Euler by 10x on energy");

        // 3-4: closed-form oracle - a two-body circular orbit.
        var orbit = CircularTwoBody(1.0, 1.0);
        double separation0 = (orbit.Bodies[1].Pos - orbit.Bodies[0].Pos).LengthSquared;
        double period = 2.0 * Math.PI / Math.Sqrt(2.0);
        var traced = orbit.Clone();
        double dt = 0.001;
        int stepsPerPeriod = (int)Math.Round(period / dt);
        double minSep = double.MaxValue, maxSep = 0.0;
        for (int s = 0; s < stepsPerPeriod; s++)
        {
            StepVelocityVerlet(traced, dt, 2);
            double sep2 = (traced.Bodies[1].Pos - traced.Bodies[0].Pos).LengthSquared;
            minSep = Math.Min(minSep, sep2);
            maxSep = Math.Max(maxSep, sep2);
        }
        Check(Math.Abs(minSep - separation0) < 1e-4 && Math.Abs(maxSep - separation0) < 1e-4,
              "the circular orbit holds its separation (min " + minSep.ToString("F12") + ", max " + maxSep.ToString("F12") + ")");
        Check(MaxAbsDiff(traced, orbit) < 0.01,
              "after the analytic Kepler period T = " + period.ToString("F6") + " the state returns home (" +
              MaxAbsDiff(traced, orbit).ToString("E2") + ")");
        var orbitEnergy = orbit.Clone();
        double eOrbit0 = TotalEnergy(orbitEnergy);
        for (int s = 0; s < stepsPerPeriod; s++) StepVelocityVerlet(orbitEnergy, dt, 2);
        Check(Math.Abs((TotalEnergy(orbitEnergy) - eOrbit0) / eOrbit0) < 1e-5, "the orbit keeps its energy");
        Check(Math.Abs((AngularMomentum(orbitEnergy) - AngularMomentum(orbit)).LengthSquared) < 1e-18,
              "the orbit keeps its angular momentum");

        // 5: cross-implementation - lanes, pairs and the independent reference agree.
        var lanes = cluster.Clone();
        var pairs = cluster.Clone();
        var reference = cluster.Clone();
        _pairEvaluations = 0;
        AccelerationsLanes(lanes, 1, true);
        long lanePairCount = _pairEvaluations;
        _pairEvaluations = 0;
        AccelerationsPairs(pairs, true);
        long pairPairCount = _pairEvaluations;
        _pairEvaluations = 0;
        AccelerationsReference(reference);
        double maxAcc = Math.Sqrt(cluster.Bodies.Max(b => b.Acc.LengthSquared) > 0
            ? cluster.Bodies.Max(b => b.Acc.LengthSquared)
            : lanes.Bodies.Max(b => b.Acc.LengthSquared));
        double worstAcc = 0.0;
        for (int i = 0; i < cluster.N; i++)
        {
            worstAcc = Math.Max(worstAcc, (lanes.Bodies[i].Acc - pairs.Bodies[i].Acc).LengthSquared);
            worstAcc = Math.Max(worstAcc, (lanes.Bodies[i].Acc - reference.Bodies[i].Acc).LengthSquared);
        }
        Console.WriteLine("acceleration agreement across the three kernels: " + worstAcc.ToString("E2") +
                          " (typical magnitude " + maxAcc.ToString("E2") + ")");
        Check(worstAcc < 1e-12, "the lane, pairwise and reference kernels agree");
        Check(lanePairCount == (long)cluster.N * (cluster.N - 1),
              "the lane kernel performs N(N-1) directed evaluations (" + lanePairCount + ")");
        Check(pairPairCount == (long)cluster.N * (cluster.N - 1) / 2,
              "the symmetric kernel performs exactly N(N-1)/2 pair evaluations (" + pairPairCount + ")");

        // 6: bitwise lane invariance - the parallel decomposition cannot change a bit.
        var laneOne = cluster.Clone();
        for (int s = 0; s < 25; s++) StepVelocityVerlet(laneOne, 0.002, 1);
        foreach (int lanesCount in new[] { 2, 4, 8 })
        {
            var laneN = cluster.Clone();
            for (int s = 0; s < 25; s++) StepVelocityVerlet(laneN, 0.002, lanesCount);
            Check(MaxAbsDiff(laneN, laneOne) == 0.0,
                  "25 steps are bitwise identical for " + lanesCount + " lanes");
        }

        // 7-8: device offload - identical state, and the transfer accounting is a real cost bound.
        var device = new Device();
        var onDevice = cluster.Clone();
        SimulateOnDevice(onDevice, 0.002, 4, 50, device);
        var onHost = cluster.Clone();
        for (int s = 0; s < 50; s++) StepVelocityVerlet(onHost, 0.002, 4);
        Check(MaxAbsDiff(onDevice, onHost) == 0.0, "the offloaded simulation is bitwise identical to the host run");
        long expectedBytes = 2L * cluster.N * 6 * 8;
        Check(device.Transfers == 2 && device.BytesTransferred == expectedBytes && device.KernelLaunches == 1,
              "one offload transfers 2*N*6 doubles: " + device.BytesTransferred + " bytes in " + device.Transfers + " transfers");
        var perStep = new Device();
        var stepping = cluster.Clone();
        for (int s = 0; s < 50; s++) SimulateOnDevice(stepping, 0.002, 4, 1, perStep);
        Check(perStep.Transfers == 100 && perStep.KernelLaunches == 50,
              "one offload per step costs 50x more transfers (" + perStep.Transfers + " vs " + device.Transfers + ")");
        Check(perStep.BytesTransferred == device.BytesTransferred * 50, "batched offloads cut the bytes moved by 50x");
        Check(MaxAbsDiff(stepping, onHost) == 0.0, "the per-step offload also matches the host exactly");
        Check(cluster.Bodies[0].Pos.X == PlummerLike(48, 0xABCDEFUL).Bodies[0].Pos.X,
              "the scenario generator is reproducible from its seed");

        // 9-12: degenerate configurations.
        var empty = new System809();
        AccelerationsLanes(empty, 4);
        var emptyStep = empty.Clone();
        StepVelocityVerlet(emptyStep, 0.01, 4);
        Check(emptyStep.N == 0 && TotalEnergy(emptyStep) == 0.0, "N = 0 is a no-op with zero energy");
        var single = new System809();
        single.Bodies.Add(new Body { Mass = 2.0, Pos = new Vec3(1, 2, 3), Vel = new Vec3(0.5, -0.25, 0.125) });
        var singleShot = single.Clone();
        for (int s = 0; s < 10; s++) StepVelocityVerlet(singleShot, 0.1, 4);
        Vec3 expected = single.Bodies[0].Pos + single.Bodies[0].Vel * 1.0;
        Check(singleShot.Bodies[0].Vel.X == single.Bodies[0].Vel.X &&
              singleShot.Bodies[0].Vel.Y == single.Bodies[0].Vel.Y &&
              singleShot.Bodies[0].Vel.Z == single.Bodies[0].Vel.Z,
              "N = 1 keeps its velocity bit for bit (zero acceleration)");
        Check((singleShot.Bodies[0].Pos - expected).LengthSquared < 1e-24,
              "N = 1 drifts by at most rounding error from uniform motion");
        Check(singleShot.Bodies[0].Acc.LengthSquared == 0.0, "N = 1 has exactly zero acceleration");
        var pairRest = new System809 { Softening = 0.0 };
        pairRest.Bodies.Add(new Body { Mass = 1, Pos = new Vec3(-1, 0, 0) });
        pairRest.Bodies.Add(new Body { Mass = 1, Pos = new Vec3(1, 0, 0) });
        Vec3 com0 = pairRest.Bodies[0].Pos * 0.5 + pairRest.Bodies[1].Pos * 0.5;
        for (int s = 0; s < 200; s++) StepVelocityVerlet(pairRest, 0.001, 2);
        Vec3 com1 = pairRest.Bodies[0].Pos * 0.5 + pairRest.Bodies[1].Pos * 0.5;
        Check((com1 - com0).LengthSquared < 1e-20, "two bodies from rest keep their centre of mass fixed");
        Check(pairRest.Bodies[0].Pos.X > -1.0 && pairRest.Bodies[1].Pos.X < 1.0, "they accelerate towards each other");
        var collinear = new System809();
        for (int i = 0; i < 6; i++) collinear.Bodies.Add(new Body { Mass = 1, Pos = new Vec3(i - 2.5, 0, 0) });
        AccelerationsLanes(collinear, 8);
        Check(collinear.Bodies.All(b => Math.Abs(b.Acc.Y) < 1e-18 && Math.Abs(b.Acc.Z) < 1e-18),
              "a collinear configuration stays collinear (no transverse force)");
        Check(Math.Abs(collinear.Bodies[3].Acc.X) < Math.Abs(collinear.Bodies[0].Acc.X),
              "a middle body of the chain feels a much smaller net force than an end body (" +
              Math.Abs(collinear.Bodies[3].Acc.X).ToString("E2") + " vs " +
              Math.Abs(collinear.Bodies[0].Acc.X).ToString("E2") + ")");

        // 13: determinism.
        var replayA = cluster.Clone();
        var replayB = cluster.Clone();
        for (int s = 0; s < 30; s++) StepVelocityVerlet(replayA, 0.002, 4);
        for (int s = 0; s < 30; s++) StepVelocityVerlet(replayB, 0.002, 4);
        Check(MaxAbsDiff(replayA, replayB) == 0.0 && MaxAbsDiff(replayA, laneOne) != 0.0,
              "an identical run is bitwise reproducible");

        Console.WriteLine("N = 48: momentum drift " + (p1 - p0).LengthSquared.ToString("E2") +
                          ", verlet energy drift " + worstVerlet.ToString("E2") +
                          " vs euler " + worstEuler.ToString("E2") + ", offload " +
                          device.BytesTransferred + " bytes in " + device.Transfers + " transfers");
        Console.WriteLine("PASS");
    }


    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
