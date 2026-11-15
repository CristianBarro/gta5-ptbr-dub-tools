using System;
using System.IO;
using System.Linq;
using System.Globalization;
using System.Collections.Generic;
using CodeWalker.GameFiles;

// Read-only reader of cutscene animation dictionaries (.ycd). A cutscene <cut> is stored as sections
// <cut>-0.ycd, <cut>-1.ycd ... (about 8 s each); each section has one clip per ped (player_zero_dual-N = Michael,
// player_one = Franklin, player_two = Trevor, cs_<npc>_dual-N ...). Peds carry facial tracks 24/25/26 at 30 fps.
// Used to tell who is speaking: the speaker's face moves during his subtitle.
//   YcdProbe.exe info     GAME ROOT_RPF NESTED_RPF YCD_NAME_PREFIX
//   YcdProbe.exe bones    GAME ROOT_RPF NESTED_RPF CUT_NAME OUT_DIR
//   YcdProbe.exe bonesall GAME ROOT_RPF LIST_TSV OUT_DIR        (LIST: nested_rpf <tab> cut_name; skips cuts already exported)
// bones output: OUT_DIR\<cut>__<ped>.csv  t, then per facial channel (bone:track) the frame-to-frame change.
public class YcdProbe {
    static IEnumerable<RpfFile> Walk(RpfFile r) { yield return r; if (r.Children != null) foreach (var c in r.Children) foreach (var x in Walk(c)) yield return x; }
    static readonly CultureInfo IC = CultureInfo.InvariantCulture;

    static int ExportFaces(RpfFile nested, string cut, string outDir) {
        Func<string, int> sec = n => { int v; return int.TryParse(n.Substring(n.LastIndexOf('-') + 1).Replace(".ycd", ""), out v) ? v : -1; };
        var secs = nested.AllEntries.OfType<RpfFileEntry>()
            .Where(e => e.Name.EndsWith(".ycd", StringComparison.OrdinalIgnoreCase) && e.Name.LastIndexOf('-') > 0
                     && e.Name.Substring(0, e.Name.LastIndexOf('-')).Equals(cut, StringComparison.OrdinalIgnoreCase) && sec(e.Name) >= 0)
            .OrderBy(e => sec(e.Name)).ToList();
        var rows = new Dictionary<string, List<string>>(); var cols = new Dictionary<string, List<string>>();
        double t0 = 0;
        foreach (var entry in secs) {
            var ycd = new YcdFile(); ycd.Load(nested.ExtractFile(entry), entry);
            double secDur = 0;
            foreach (var cme in ycd.ClipMapEntries ?? new ClipMapEntry[0]) {
                var ca = cme.Clip as ClipAnimation; if (ca == null || ca.Animation == null) continue;
                var anim = ca.Animation; var ids = anim.BoneIds == null ? null : anim.BoneIds.data_items; if (ids == null) continue;
                secDur = Math.Max(secDur, anim.Duration);
                var face = Enumerable.Range(0, ids.Length).Where(i => ids[i].Track >= 24 && ids[i].Track <= 26).ToList();
                if (face.Count == 0) continue;
                string name = ca.Name.Replace("pack:/", ""); if (name.LastIndexOf('-') > 0) name = name.Substring(0, name.LastIndexOf('-'));
                if (!cols.ContainsKey(name)) { cols[name] = face.Select(i => ids[i].BoneId + ":" + ids[i].Track).ToList(); rows[name] = new List<string>(); }
                var keyIndex = new Dictionary<string, int>(); foreach (var i in face) keyIndex[ids[i].BoneId + ":" + ids[i].Track] = i;
                int limit = Math.Max(1, (int)anim.SequenceFrameLimit); double dt = anim.Duration / Math.Max(1, anim.Frames - 1);
                var seqs = anim.Sequences.data_items;
                for (int f = 0; f < anim.Frames - 1; f++) {
                    var sb = new System.Text.StringBuilder(); sb.Append((t0 + f * dt).ToString("0.0000", IC));
                    foreach (var key in cols[name]) {
                        int bi; double d = 0;
                        if (keyIndex.TryGetValue(key, out bi)) {
                            int s0 = f / limit, f0 = f % limit, s1 = (f + 1) / limit, f1 = (f + 1) % limit;
                            try {
                                if (s1 < seqs.Length) { var a = seqs[s0].Sequences[bi].EvaluateVector(f0); var b = seqs[s1].Sequences[bi].EvaluateVector(f1); d = (a - b).Length(); }
                            } catch { d = 0; }
                        }
                        sb.Append(',').Append(d.ToString("0.#####", IC));
                    }
                    rows[name].Add(sb.ToString());
                }
            }
            t0 += secDur;
        }
        Directory.CreateDirectory(outDir);
        foreach (var kv in rows)
            File.WriteAllLines(Path.Combine(outDir, cut + "__" + kv.Key + ".csv"), new[] { "t," + string.Join(",", cols[kv.Key]) }.Concat(kv.Value));
        File.WriteAllText(Path.Combine(outDir, cut + ".done"), secs.Count + " sections " + t0.ToString("0.00", IC) + " s, peds " + string.Join(" ", rows.Keys) + "\n");
        return rows.Count;
    }

    public static int Main(string[] args) {
        try {
            string game = Path.GetFullPath(args[1]).TrimEnd('\\');
            GTA5Keys.LoadFromPath(game, null);
            var root = new RpfFile(Path.Combine(game, args[2]), args[2]); root.ScanStructure(null, null);
            if (args[0] == "bonesall") {
                var byName = Walk(root).GroupBy(r => r.Name.ToLowerInvariant()).ToDictionary(g => g.Key, g => g.First());
                var list = File.ReadLines(args[3]).Select(l => l.Split('\t')).Where(p => p.Length >= 2).ToList();
                int k = 0;
                foreach (var p in list) {
                    k++;
                    if (File.Exists(Path.Combine(args[4], p[1] + ".done"))) continue;
                    RpfFile nested;
                    if (!byName.TryGetValue(p[0].ToLowerInvariant(), out nested)) { Console.WriteLine("MISSING rpf " + p[0]); continue; }
                    try { Console.WriteLine("[" + k + "/" + list.Count + "] " + p[1] + " peds " + ExportFaces(nested, p[1], args[4])); }
                    catch (Exception ex) { Console.WriteLine("ERROR " + p[1] + " " + ex.Message); }
                }
                return 0;
            }
            var nst = Walk(root).First(r => r.Name.Equals(args[3], StringComparison.OrdinalIgnoreCase));
            if (args[0] == "bones") { Console.WriteLine("peds " + ExportFaces(nst, args[4], args[5])); return 0; }
            var entries = nst.AllEntries.OfType<RpfFileEntry>().Where(e => e.Name.StartsWith(args[4], StringComparison.OrdinalIgnoreCase) && e.Name.EndsWith(".ycd", StringComparison.OrdinalIgnoreCase)).OrderBy(e => e.Name).ToList();
            Console.WriteLine(nst.Path + " ycd files matching: " + entries.Count);
            foreach (var entry in entries.Take(2)) {
                var ycd = new YcdFile(); ycd.Load(nst.ExtractFile(entry), entry);
                Console.WriteLine("== " + entry.Name + " clips " + (ycd.ClipMapEntries == null ? 0 : ycd.ClipMapEntries.Length));
                foreach (var cme in ycd.ClipMapEntries ?? new ClipMapEntry[0]) {
                    var ca = cme.Clip as ClipAnimation; var anim = ca == null ? null : ca.Animation;
                    string line = "  clip " + cme.Clip.Name;
                    if (anim != null && anim.BoneIds != null && anim.BoneIds.data_items != null) {
                        var ids = anim.BoneIds.data_items;
                        line += " dur " + anim.Duration.ToString("0.00", IC) + " frames " + anim.Frames + " tracks[" + string.Join(" ", ids.GroupBy(b => b.Track).OrderBy(g => g.Key).Select(g => g.Key + "x" + g.Count())) + "]";
                    }
                    Console.WriteLine(line);
                }
            }
            return 0;
        } catch (Exception ex) { Console.Error.WriteLine(ex); return 1; }
    }
}
