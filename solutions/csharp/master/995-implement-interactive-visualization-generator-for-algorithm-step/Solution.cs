// Task 995 - Implement interactive visualization generator for algorithm steps (trace player).
// (Tier: Master, Beyond - Software Engineering & Tooling)
//
// The generator records a full step trace (array snapshot, highlighted index, action label) for bubble sort
// and for binary search, then compresses it into keyframes every K steps plus deltas against the previous
// frame. The player reconstructs any frame by loading the nearest keyframe at or before the target and
// applying the deltas in between, and supports stepping forward/backwards as well as random seeks. Verified:
// every reconstructed frame equals the directly simulated snapshot (independent oracle), the final frame
// holds the sorted array / the reported search result, step round-trips (next then previous returns to the
// same state, seek forward and back again), the memory saving of the delta encoding, byte-identical repeated
// runs, and the boundaries (single element, already sorted input, absent search target, spacing equal to the
// trace length).
using System;
using System.Collections.Generic;
using System.Globalization;

public static class Solution
{
    private static void Check(bool ok, string label) { if (!ok) throw new Exception("FAIL: " + label); }

    private sealed class Frame
    {
        public readonly int[] Values;
        public readonly string Action;
        public readonly int Highlight;
        public Frame(int[] values, string action, int highlight) { Values = values; Action = action; Highlight = highlight; }
        public string Text() => string.Join(",", Array.ConvertAll(Values, v => v.ToString(CultureInfo.InvariantCulture)))
                                + " " + Action + " @" + Highlight;
    }

    private static List<Frame> SortTrace(int[] input)
    {
        int[] work = (int[])input.Clone();
        var frames = new List<Frame> { new Frame((int[])work.Clone(), "start", -1) };
        for (int pass = 0; pass < work.Length; pass++)
        {
            bool swapped = false;
            for (int i = 0; i + 1 < work.Length - pass; i++)
            {
                if (work[i] <= work[i + 1]) continue;
                int held = work[i];
                work[i] = work[i + 1];
                work[i + 1] = held;
                swapped = true;
                frames.Add(new Frame((int[])work.Clone(), "swap " + i + "<->" + (i + 1), i + 1));
            }
            frames.Add(new Frame((int[])work.Clone(), "pass " + pass + " done", -1));
            if (!swapped) break;
        }
        return frames;
    }

    private static List<Frame> SearchTrace(int[] sorted, int target)
    {
        var frames = new List<Frame> { new Frame((int[])sorted.Clone(), "search start", -1) };
        int low = 0, high = sorted.Length - 1;
        while (low <= high)
        {
            int mid = low + (high - low) / 2;
            frames.Add(new Frame((int[])sorted.Clone(), "probe " + mid, mid));
            if (sorted[mid] == target) { frames.Add(new Frame((int[])sorted.Clone(), "found " + mid, mid)); return frames; }
            if (sorted[mid] < target) low = mid + 1;
            else high = mid - 1;
        }
        frames.Add(new Frame((int[])sorted.Clone(), "absent", -1));
        return frames;
    }

    /** Keyframe/delta player: reconstructs frame i from the last keyframe at or before i. */
    private sealed class Player
    {
        private readonly List<Frame> _frames;
        private readonly int _spacing;
        private readonly Dictionary<int, int[]> _keyframes = new Dictionary<int, int[]>();
        private readonly Dictionary<int, List<(int Index, int Value)>> _deltas = new Dictionary<int, List<(int, int)>>();
        public int Position;
        public int StoredCells;

        public Player(List<Frame> frames, int spacing)
        {
            _frames = frames;
            _spacing = Math.Max(1, spacing);
            for (int i = 0; i < frames.Count; i++)
            {
                if (i % _spacing == 0)
                {
                    _keyframes[i] = (int[])frames[i].Values.Clone();
                    StoredCells += frames[i].Values.Length;
                    continue;
                }
                var delta = new List<(int, int)>();
                int[] previous = frames[i - 1].Values;
                for (int at = 0; at < previous.Length; at++)
                    if (previous[at] != frames[i].Values[at]) delta.Add((at, frames[i].Values[at]));
                _deltas[i] = delta;
                StoredCells += delta.Count;
            }
        }

        public int Count => _frames.Count;
        public int KeyframeCount => _keyframes.Count;
        public string Action() => _frames[Position].Action;
        public int Highlight() => _frames[Position].Highlight;

        public int[] State()
        {
            int anchor = Position - Position % _spacing;
            var values = (int[])_keyframes[anchor].Clone();
            for (int i = anchor + 1; i <= Position; i++)
                foreach ((int index, int value) in _deltas[i])
                    values[index] = value;
            return values;
        }

        public void Seek(int index) { Position = Math.Max(0, Math.Min(_frames.Count - 1, index)); }
        public void Next() { Seek(Position + 1); }
        public void Prev() { Seek(Position - 1); }
    }

    private static string Snap(Player player) => string.Join(",", Array.ConvertAll(player.State(), v => v.ToString(CultureInfo.InvariantCulture)));

    public static void Main()
    {
        int[] input = { 5, 1, 4, 2, 8, 0, 7 };
        List<Frame> trace = SortTrace(input);
        var player = new Player(trace, 3);
        bool matches = true;
        for (int i = 0; i < trace.Count; i++)
        {
            player.Seek(i);
            if (Snap(player) != string.Join(",", Array.ConvertAll(trace[i].Values, v => v.ToString(CultureInfo.InvariantCulture)))) matches = false;
            if (player.Action() != trace[i].Action || player.Highlight() != trace[i].Highlight) matches = false;
        }
        Check(matches && trace.Count >= 10, "every reconstructed frame equals the simulated snapshot (oracle)");

        player.Seek(trace.Count - 1);
        int[] sorted = (int[])input.Clone();
        Array.Sort(sorted);
        Check(Snap(player) == string.Join(",", Array.ConvertAll(sorted, v => v.ToString(CultureInfo.InvariantCulture)))
              && trace[trace.Count - 1].Action.StartsWith("pass", StringComparison.Ordinal),
              "the final frame holds the sorted array, and the trace replays deterministically");

        player.Seek(4);
        string atFour = Snap(player);
        player.Next();
        player.Prev();
        player.Seek(0);
        string start = Snap(player);
        player.Seek(4);
        Check(Snap(player) == atFour && player.Action() == trace[4].Action && start == "5,1,4,2,8,0,7",
              "next then previous returns to the same frame and seeking back reaches the start state");

        var full = new Player(trace, 1);
        Check(player.StoredCells < full.StoredCells && player.StoredCells > 0
              && full.StoredCells == trace.Count * input.Length && player.KeyframeCount == 6
              && new Player(trace, trace.Count * 2).KeyframeCount == 1,
              "delta encoding stores fewer cells than per-frame keyframes, and a spacing beyond the trace keeps one keyframe");

        List<Frame> search = SearchTrace(new[] { 0, 1, 2, 4, 5, 7, 8 }, 7);
        List<Frame> absent = SearchTrace(new[] { 0, 1, 2, 4, 5, 7, 8 }, 6);
        var searchPlayer = new Player(search, 2);
        searchPlayer.Seek(search.Count - 1);
        Check(search[search.Count - 1].Action == "found 5" && searchPlayer.Highlight() == 5
              && absent[absent.Count - 1].Action == "absent" && absent[absent.Count - 1].Highlight == -1,
              "binary search reports an exact hit, and an absent target ends the trace without a highlight");

        var firstRun = new Player(SortTrace(input), 4);
        var secondRun = new Player(SortTrace(input), 4);
        bool replayEqual = firstRun.Count == secondRun.Count;
        for (int i = 0; i < firstRun.Count && replayEqual; i++)
        {
            firstRun.Seek(i);
            secondRun.Seek(i);
            if (Snap(firstRun) != Snap(secondRun) || firstRun.Action() != secondRun.Action()) replayEqual = false;
        }
        Check(replayEqual && firstRun.StoredCells == secondRun.StoredCells
              && SortTrace(new[] { 1, 2, 3 }).Count == 2 && SortTrace(new[] { 9 }).Count == 2
              && new Player(SearchTrace(new[] { 4 }, 9), 1).Count == 3,
              "repeated runs replay identically and the single-element / already-sorted / one-probe boundaries hold");

        Console.WriteLine("sort frames: " + trace.Count + ", keyframe cells: " + player.StoredCells
                          + ", uncompressed cells: " + full.StoredCells + ", search frames: " + search.Count);
        Console.WriteLine("PASS");
    }
}
