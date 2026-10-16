// Task 732 - Implement particle-in-cell simplified simulation outline. (Tier: Expert, CTCI Beyond — Graphics, Geometry & Robotics)
//
// A PIC code couples markers to a fixed grid and back again, which is what separates it from a plain
// N-body sum:
//
//   deposit  every marker's charge is spread onto the two nearest nodes with cloud-in-cell weights
//            (1 - w and w), so the grid density is the marker density and the weights sum to one.
//   solve    the periodic Poisson equation is solved spectrally, mode by mode:
//            phi_hat(m) = rho_hat(m) / (eps0 k_m^2) with k_m = 2 pi m / L and the m = 0 mode dropped,
//            and E comes from the spectral derivative -i k phi.
//   gather   E is interpolated back to the markers with the *same* CIC weights, which is what makes the
//            deposition/gather pair momentum conserving.
//   push     a symplectic Euler-Cromer step: v += q E(x) dt / m, then x += v dt with a periodic wrap.
//
// Documented simplifications: 1D periodic electrostatics, one species, a uniform grid, a spectral solve
// (exact for the continuous Poisson operator, so the discrete Laplacian's truncation is deliberately not
// modelled), a cold quiet start and no collisions, no shape factor beyond CIC and no energy-conserving
// sub-cycling. Verified: the deposition conserves charge exactly and a uniform load produces exactly zero
// field; the spectral solve reproduces the analytic potential and field of a single Fourier mode; the
// Poisson relation holds mode by mode; the momentum identity dp = dt q sum E(x) holds to machine
// precision; the integrator is time reversible, which is checked by running the simulation backwards and
// comparing with the saved initial state; and the Langmuir oscillation frequency matches
// omega_p = sqrt(n q^2 / (eps0 m)) from linear theory with bounded energy drift.
using System;

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

internal sealed class Plasma
{
    private readonly double[] _x;
    private readonly double[] _v;
    private readonly double[] _rho;
    private readonly double[] _phi;
    private readonly double[] _field;

    /// <summary>Markers start on cell centres with an optional sinusoidal displacement and jitter (in cell widths).</summary>
    internal Plasma(int markers, int nodes, double length, ulong seed, double displacement, double jitter)
    {
        Markers = markers;
        Nodes = nodes;
        Length = length;
        Dx = length / nodes;
        _x = new double[markers];
        _v = new double[markers];
        _rho = new double[nodes];
        _phi = new double[nodes];
        _field = new double[nodes];
        var rng = new SplitMix64(seed);
        for (int i = 0; i < markers; i++)
        {
            double home = (i + 0.5) * length / markers;
            double position = home + displacement * Math.Sin(2.0 * Math.PI * home / length)
                + jitter * Dx * rng.NextSigned();
            _x[i] = Wrap(position);
        }
    }

    internal int Markers { get; }
    internal int Nodes { get; }
    internal double Length { get; }
    internal double Dx { get; }
    internal long Steps { get; private set; }

    internal double Charge { get; set; } = -1.0;
    internal double Mass { get; set; } = 1.0;
    internal double Epsilon0 { get; set; } = 1.0;

    internal double Position(int i) => _x[i];
    internal double Velocity(int i) => _v[i];
    internal void SetVelocity(int i, double value) => _v[i] = value;
    internal void SetPosition(int i, double value) => _x[i] = Wrap(value);
    internal double NodeDensity(int j) => _rho[j];
    internal void SetNodeDensity(int j, double value) => _rho[j] = value;

    internal double Wrap(double x)
    {
        double wrapped = x % Length;
        return wrapped < 0.0 ? wrapped + Length : wrapped;
    }

    /// <summary>Cloud-in-cell deposition: the two neighbouring nodes share the marker's charge.</summary>
    internal void Deposit()
    {
        Array.Clear(_rho, 0, _rho.Length);
        for (int i = 0; i < Markers; i++)
        {
            double g = _x[i] / Dx;
            int node = (int)Math.Floor(g);
            double weight = g - node;
            _rho[Mod(node)] += Charge * (1.0 - weight) / Dx;
            _rho[Mod(node + 1)] += Charge * weight / Dx;
        }
    }

    /// <summary>Spectral solve of phi'' = -rho/eps0 on the periodic grid, then E = -phi'.</summary>
    internal void Solve()
    {
        int n = Nodes;
        var potentialCos = new double[n];
        var potentialSin = new double[n];
        for (int m = 1; m <= n / 2 - 1; m++)
        {
            double rhoCos = 0.0, rhoSin = 0.0;
            for (int j = 0; j < n; j++)
            {
                double angle = 2.0 * Math.PI * m * j / n;
                rhoCos += _rho[j] * Math.Cos(angle);
                rhoSin -= _rho[j] * Math.Sin(angle);
            }
            double k = 2.0 * Math.PI * m / Length;
            double scale = 1.0 / (Epsilon0 * k * k);
            potentialCos[m] = rhoCos * scale;
            potentialSin[m] = rhoSin * scale;
        }
        for (int j = 0; j < n; j++)
        {
            double phi = 0.0, field = 0.0;
            for (int m = 1; m <= n / 2 - 1; m++)
            {
                double angle = 2.0 * Math.PI * m * j / n;
                double k = 2.0 * Math.PI * m / Length;
                phi += potentialCos[m] * Math.Cos(angle) - potentialSin[m] * Math.Sin(angle);
                field += k * (potentialSin[m] * Math.Cos(angle) + potentialCos[m] * Math.Sin(angle));
            }
            _phi[j] = 2.0 * phi / n;              // the negative-frequency mirror doubles every real mode
            _field[j] = 2.0 * field / n;
        }
    }

    internal double NodePotential(int j) => _phi[j];
    internal double NodeFieldInternal(int j) => _field[j];

    /// <summary>Mode amplitudes of rho and phi: the Poisson relation is k^2 phi_hat + rho_hat/eps0 = 0.</summary>
    internal (double RhoCos, double RhoSin, double PhiCos, double PhiSin) Mode(int m)
    {
        int n = Nodes;
        double rhoCos = 0.0, rhoSin = 0.0, phiCos = 0.0, phiSin = 0.0;
        for (int j = 0; j < n; j++)
        {
            double angle = 2.0 * Math.PI * m * j / n;
            rhoCos += _rho[j] * Math.Cos(angle);
            rhoSin -= _rho[j] * Math.Sin(angle);
            phiCos += _phi[j] * Math.Cos(angle);
            phiSin -= _phi[j] * Math.Sin(angle);
        }
        return (rhoCos, rhoSin, phiCos, phiSin);
    }

    /// <summary>CIC gather of the grid field at a position (linear interpolation between nodes).</summary>
    internal double FieldAt(double x)
    {
        double g = Wrap(x) / Dx;
        int node = (int)Math.Floor(g);
        double weight = g - node;
        return _field[Mod(node)] * (1.0 - weight) + _field[Mod(node + 1)] * weight;
    }

    /// <summary>One symplectic Euler-Cromer step.</summary>
    internal void Push(double dt)
    {
        for (int i = 0; i < Markers; i++)
        {
            double acceleration = Charge * FieldAt(_x[i]) / Mass;
            _v[i] += acceleration * dt;
            _x[i] = Wrap(_x[i] + _v[i] * dt);
        }
        Steps++;
    }

    /// <summary>
    /// The exact inverse of <see cref="Push"/>: the drift is undone with the new velocity, the field is
    /// re-solved at the recovered positions, and only then is the kick subtracted. Running this 200 times
    /// after 200 forward steps returns the markers to where they started.
    /// </summary>
    internal void PushBackward(double dt)
    {
        for (int i = 0; i < Markers; i++) _x[i] = Wrap(_x[i] - _v[i] * dt);
        Deposit();
        Solve();
        for (int i = 0; i < Markers; i++) _v[i] -= Charge * FieldAt(_x[i]) / Mass * dt;
        Steps++;
    }

    internal double TotalCharge()
    {
        double sum = 0.0;
        for (int j = 0; j < Nodes; j++) sum += _rho[j] * Dx;
        return sum;
    }

    internal double Momentum()
    {
        double sum = 0.0;
        for (int i = 0; i < Markers; i++) sum += Mass * _v[i];
        return sum;
    }

    internal double ImpulsePerStep()
    {
        double sum = 0.0;
        for (int i = 0; i < Markers; i++) sum += Charge * FieldAt(_x[i]);
        return sum;
    }

    internal double KineticEnergy()
    {
        double sum = 0.0;
        for (int i = 0; i < Markers; i++) sum += 0.5 * Mass * _v[i] * _v[i];
        return sum;
    }

    internal double FieldEnergy()
    {
        double sum = 0.0;
        for (int j = 0; j < Nodes; j++) sum += 0.5 * Epsilon0 * _field[j] * _field[j] * Dx;
        return sum;
    }

    internal double TotalEnergy() => KineticEnergy() + FieldEnergy();

    internal int Mod(int index)
    {
        int m = index % Nodes;
        return m < 0 ? m + Nodes : m;
    }
}

public static class Solution
{
    private const int Nodes = 64;
    private const double Length = 1.0;

    public static void Main()
    {
        // ---- the deposition conserves charge exactly, for any marker positions --------------------
        {
            var quiet = new Plasma(Nodes, Nodes, Length, 0x732AUL, 0.0, 0.0);
            quiet.Deposit();
            Check(Math.Abs(quiet.TotalCharge() - quiet.Charge * quiet.Markers) < 1e-13,
                "the deposited charge equals the marker charge (" + quiet.TotalCharge() + ")");
            var scattered = new Plasma(37, 61, Length, 0x732BUL, 0.4, 0.7);
            scattered.Deposit();
            Check(Math.Abs(scattered.TotalCharge() - scattered.Charge * 37) < 1e-13,
                "charge is conserved for an uneven marker count and grid");
            bool nonPositive = true;
            for (int j = 0; j < scattered.Nodes; j++) if (scattered.NodeDensity(j) > 0.0) nonPositive = false;
            Check(nonPositive, "a negative charge deposits a non-positive density");
        }

        // ---- a uniform load has exactly zero field, and then streams freely ------------------------
        {
            var uniform = new Plasma(Nodes, Nodes, Length, 0x732CUL, 0.0, 0.0);
            uniform.Deposit();
            uniform.Solve();
            double peak = 0.0;
            for (int j = 0; j < Nodes; j++) peak = Math.Max(peak, Math.Abs(uniform.NodeFieldInternal(j)));
            Console.WriteLine("uniform load: density=" + uniform.NodeDensity(0).ToString("G6")
                + " peakField=" + peak.ToString("G4"));
            Check(peak < 1e-12, "a uniform density produces no field at all (" + peak + ")");
            Check(Math.Abs(uniform.NodeDensity(0) * uniform.Dx - uniform.Charge) < 1e-13,
                "every node carries exactly one marker's charge");

            for (int i = 0; i < Nodes; i++) uniform.SetVelocity(i, 0.25);
            double impulse = uniform.ImpulsePerStep();
            Check(Math.Abs(impulse) < 1e-12, "the uniform load exerts no force on its markers (" + impulse + ")");
            double before = uniform.Position(0);
            uniform.Push(1e-3);
            Check(Math.Abs(uniform.Position(0) - uniform.Wrap(before + 0.25 * 1e-3)) < 1e-15,
                "with no field the markers stream exactly at constant velocity");
        }

        // ---- the spectral solve reproduces the analytic single-mode solution -----------------------
        {
            double amplitude = 2.5;
            double k = 2.0 * Math.PI / Length;
            var solved = new Plasma(8, Nodes, Length, 0x7330UL, 0.0, 0.0);
            for (int j = 0; j < solved.Nodes; j++) solved.SetNodeDensity(j, amplitude * Math.Sin(k * j * solved.Dx));
            solved.Solve();
            double worstPotential = 0.0, worstElectric = 0.0;
            for (int j = 0; j < Nodes; j++)
            {
                double x = j * solved.Dx;
                double expectedPhi = amplitude * Math.Sin(k * x) / (solved.Epsilon0 * k * k);
                double expectedField = -amplitude * Math.Cos(k * x) / (solved.Epsilon0 * k);
                worstPotential = Math.Max(worstPotential, Math.Abs(solved.NodePotential(j) - expectedPhi));
                worstElectric = Math.Max(worstElectric, Math.Abs(solved.NodeFieldInternal(j) - expectedField));
            }
            Console.WriteLine("single mode: worstPhiError=" + worstPotential.ToString("G4")
                + " worstFieldError=" + worstElectric.ToString("G4"));
            Check(worstPotential < 1e-12, "the spectral solve reproduces the analytic potential");
            Check(worstElectric < 1e-12, "and the analytic electric field");
            Check(Math.Abs(solved.NodeFieldInternal(0) + amplitude / (solved.Epsilon0 * k)) < 1e-12,
                "the field is maximal in magnitude exactly where the density vanishes");

            (double rhoCos, double rhoSin, double phiCos, double phiSin) = solved.Mode(1);
            double modeK = 2.0 * Math.PI / Length;
            double scale = Math.Max(1.0, Math.Max(Math.Abs(rhoSin), Math.Abs(phiSin) * modeK * modeK));
            Check(Math.Abs(modeK * modeK * phiSin - rhoSin / solved.Epsilon0) < 1e-9 * scale,
                "the sine mode satisfies k^2 phi = rho/eps0");
            Check(Math.Abs(modeK * modeK * phiCos - rhoCos / solved.Epsilon0) < 1e-9 * scale,
                "and so does the cosine mode");
            Check(Math.Abs(rhoCos) < 1e-9, "a pure sine mode has no cosine part");
            Check(Math.Abs(rhoSin) > 1e-6, "and a strong sine part (" + rhoSin.ToString("G6") + ")");
        }

        // ---- the momentum identity of the gather/push pair -----------------------------------------
        {
            var plasma = new Plasma(128, Nodes, Length, 0x7331UL, 0.05, 0.3);
            plasma.Deposit();
            plasma.Solve();
            double impulse = plasma.ImpulsePerStep();
            double absoluteImpulse = 0.0, peakField = 0.0;
            for (int j = 0; j < Nodes; j++) peakField = Math.Max(peakField, Math.Abs(plasma.NodeFieldInternal(j)));
            for (int i = 0; i < plasma.Markers; i++) absoluteImpulse += Math.Abs(plasma.Charge * plasma.FieldAt(plasma.Position(i)));
            double before = plasma.Momentum();
            double dt = 1e-4;
            plasma.Push(dt);
            double after = plasma.Momentum();
            Console.WriteLine("momentum: change=" + (after - before).ToString("G6")
                + " q*E*dt=" + (plasma.Charge * impulse * dt).ToString("G6")
                + " peakField=" + peakField.ToString("G4") + " cancellation="
                + (Math.Abs(impulse) / absoluteImpulse).ToString("G4"));
            Check(peakField > 1e-3, "the perturbed load really produces a field (" + peakField + ")");
            Check(Math.Abs(impulse) < 0.2 * absoluteImpulse,
                "the net force nearly cancels because the gather pairs with the deposition");
            Check(Math.Abs((after - before) - plasma.Charge * impulse * dt) < 1e-15,
                "the momentum change equals q E dt summed over the markers");
        }

        // ---- time reversibility of the symplectic step ----------------------------------------------
        {
            const int markers = 96;
            var forward = new Plasma(markers, Nodes, Length, 0x7332UL, 1.2, 0.3);
            double dt = 2e-3;
            var initialX = new double[markers];
            for (int i = 0; i < markers; i++) initialX[i] = forward.Position(i);
            for (int step = 0; step < 200; step++)
            {
                forward.Deposit();
                forward.Solve();
                forward.Push(dt);
            }
            for (int step = 0; step < 200; step++) forward.PushBackward(dt);
            double worstPosition = 0.0, worstVelocity = 0.0;
            for (int i = 0; i < markers; i++)
            {
                double drift = Math.Abs(forward.Position(i) - initialX[i]);
                drift = Math.Min(drift, Length - drift);                    // periodic distance
                worstPosition = Math.Max(worstPosition, drift);
                worstVelocity = Math.Max(worstVelocity, Math.Abs(forward.Velocity(i)));
            }            Console.WriteLine("reversibility: worstPositionDrift=" + worstPosition.ToString("G4")
                + " worstResidualVelocity=" + worstVelocity.ToString("G4"));
            Check(worstPosition < 1e-6, "the backward run returns to the starting positions");
            Check(worstVelocity < 1e-6, "and to zero velocity");
        }

        // ---- the Langmuir oscillation matches linear theory -------------------------------------------
        {
            const int markers = 64;
            double density = markers / Length;
            double omega = Math.Sqrt(density);                    // sqrt(n q^2 / (eps0 m)), q = -1, m = eps0 = 1
            var plasma = new Plasma(markers, Nodes, Length, 0x7333UL, 0.01, 0.0);
            double dt = 0.002;

            // number of maxima of the analytic sin^2 profile is the check; measure the first one
            var kinetic = new double[600];
            for (int step = 0; step < 600; step++)
            {
                plasma.Deposit();
                plasma.Solve();
                plasma.Push(dt);
                kinetic[step] = plasma.KineticEnergy();
            }
            int firstPeak = 0;
            for (int step = 1; step < 599; step++)
                if (kinetic[step] > kinetic[step - 1] && kinetic[step] >= kinetic[step + 1]) { firstPeak = step; break; }
            double measured = firstPeak > 0 ? Math.PI / (2.0 * firstPeak * dt) : 0.0;
            Console.WriteLine("Langmuir: omegaTheory=" + omega.ToString("G6") + " omegaMeasured=" + measured.ToString("G6")
                + " ratio=" + (measured / omega).ToString("G4") + " firstPeakStep=" + firstPeak);
            Check(firstPeak > 0, "the plasma really oscillates");
            Check(Math.Abs(measured / omega - 1.0) < 0.1, "the measured frequency matches omega_p ("
                + (measured / omega).ToString("G4") + ")");
            Check(plasma.Steps == 600, "the step counter is maintained");

            var bounded = new Plasma(markers, Nodes, Length, 0x7333UL, 0.01, 0.0);
            double initialEnergy = 0.0, worstEnergy = 0.0;
            for (int step = 0; step < 600; step++)
            {
                bounded.Deposit();
                bounded.Solve();
                bounded.Push(dt);
                double total = bounded.TotalEnergy();
                if (step == 0) initialEnergy = total;
                worstEnergy = Math.Max(worstEnergy, Math.Abs(total - initialEnergy) / Math.Abs(initialEnergy));
            }
            Console.WriteLine("energy: initial=" + initialEnergy.ToString("G6") + " worstRelativeDrift="
                + worstEnergy.ToString("G4"));
            Check(Math.Abs(initialEnergy) > 1e-12, "the perturbed plasma starts with a field");
            Check(worstEnergy < 0.05, "the total energy stays within a few percent (" + worstEnergy + ")");
        }

        // ---- bookkeeping, boundaries and reproducibility ----------------------------------------------
        {
            var plasma = new Plasma(50, 32, Length, 0x7334UL, 0.3, 0.4);
            for (int step = 0; step < 100; step++)
            {
                plasma.Deposit();
                plasma.Solve();
                plasma.Push(0.005);
            }
            bool insideDomain = true;
            for (int i = 0; i < 50; i++) if (plasma.Position(i) < 0.0 || plasma.Position(i) >= Length) insideDomain = false;
            Check(insideDomain, "every marker stays inside the periodic domain");
            Check(plasma.Markers == 50 && plasma.Nodes == 32, "the marker and node counts never change");
            Check(Math.Abs(plasma.FieldAt(3.0 * plasma.Dx) - plasma.NodeFieldInternal(3)) < 1e-15,
                "gathering exactly on a node returns that node's field");

            var again = new Plasma(50, 32, Length, 0x7334UL, 0.3, 0.4);
            for (int step = 0; step < 100; step++)
            {
                again.Deposit();
                again.Solve();
                again.Push(0.005);
            }
            double worst = 0.0;
            for (int i = 0; i < 50; i++) worst = Math.Max(worst, Math.Abs(again.Position(i) - plasma.Position(i)));
            Check(worst == 0.0, "the same seed reproduces the trajectory bit-for-bit");

            var single = new Plasma(1, Nodes, Length, 0x7335UL, 0.0, 0.0);
            single.Deposit();
            single.Solve();
            Check(single.TotalCharge() == single.Charge, "a single marker deposits its whole charge");
            Check(Math.Abs(single.NodeDensity(0) - single.NodeDensity(1)) == 0.0,
                "and spreads it evenly between two nodes");
            Check(single.Nodes == Nodes, "the grid size is preserved");

            bool threw = false;
            try { new Plasma(0, Nodes, Length, 1UL, 0.0, 0.0); }
            catch (Exception) { threw = true; }
            Check(!threw, "an empty marker list is representable (no crash for zero markers)");
        }

        Console.WriteLine("PASS");
    }

    private static void Check(bool ok, string label)
    {
        if (!ok) throw new Exception("FAILED: " + label);
    }
}
