using System;
using System.IO;
using System.Text;
using System.Linq;
using System.Globalization;
using System.Collections.Generic;
using CodeWalker.GameFiles;

// Reads game archives only. Every output is written below the supplied output folder.
public class ArchiveScanner
{
    static StreamWriter entries, texts, streams, errors;
    static string output, game;
    static int awcCount, streamCount, textCount;
    static string Q(object v) { return "\"" + Convert.ToString(v, CultureInfo.InvariantCulture).Replace("\"", "\"\"") + "\""; }
    static void Row(StreamWriter w, params object[] values) { w.WriteLine(string.Join(",", values.Select(Q))); }
    static StreamWriter Writer(string name, params string[] header) {
        var w = new StreamWriter(Path.Combine(output, name), false, new UTF8Encoding(true)); Row(w, header); return w;
    }
    static void Error(string path, Exception e) { Row(errors, path, e.GetType().Name, e.Message); errors.Flush(); }
    static IEnumerable<RpfFile> Flatten(RpfFile rpf) {
        yield return rpf;
        if(rpf.Children != null) foreach(var child in rpf.Children) foreach(var item in Flatten(child)) yield return item;
    }
    static void ReadArchive(string path, bool audio) {
        var relpath = path.Substring(game.Length).TrimStart('\\');
        Console.WriteLine("SCAN " + relpath);
        var rpf = new RpfFile(path, relpath);
        rpf.ScanStructure(null, delegate(string err) { Row(errors, relpath, "ScanStructure", err); });
        if(rpf.AllEntries == null) return;
        foreach(var archive in Flatten(rpf)) foreach(var entry in archive.AllEntries.OfType<RpfFileEntry>()) {
            Row(entries, entry.Path, entry.GetType().Name, archive.Name, entry.GetFileSize());
            var low = entry.Path.ToLowerInvariant();
            bool en = low.Contains("american_rel.rpf");
            bool pt = low.Contains("portuguese_rel.rpf");
            try {
                if(low.EndsWith(".gxt2") && (en || pt)) {
                    var data = archive.ExtractFile(entry);
                    if(data == null) throw new Exception(archive.LastError ?? "Extraction returned null");
                    var gxt = new Gxt2File(); gxt.Load(data, entry);
                    if(gxt.TextEntries == null) throw new Exception("Invalid GXT2");
                    foreach(var t in gxt.TextEntries) { Row(texts, en ? "en-US" : "pt-BR", entry.Path, entry.Name, t.Hash.ToString("X8"), t.Text); textCount++; }
                }
                if(low.EndsWith(".rel") && (low.Contains("speech") || low.Contains("script"))) {
                    var data = archive.ExtractFile(entry);
                    if(data == null) throw new Exception(archive.LastError ?? "Extraction returned null");
                    var rel = new RelFile(); rel.Load(data, entry);
                    var name = low.Replace('\\','_').Replace(':','_') + ".xml";
                    File.WriteAllText(Path.Combine(output, "rel", name), RelXml.GetXml(rel), new UTF8Encoding(false));
                }
                if(audio && low.EndsWith(".awc")) {
                    var data = archive.ExtractFile(entry);
                    if(data == null) throw new Exception(archive.LastError ?? "Extraction returned null");
                    var awc = new AwcFile(); awc.Load(data, entry);
                    if(!string.IsNullOrEmpty(awc.ErrorMessage)) throw new Exception(awc.ErrorMessage);
                    if(awc.Streams == null) throw new Exception("No AWC streams");
                    foreach(var s in awc.Streams) {
                        string codec = s.FormatChunk != null ? s.FormatChunk.Codec.ToString() : (s.StreamFormat != null ? s.StreamFormat.Codec.ToString() : "");
                        Row(streams, entry.Path, ((uint)s.Hash).ToString("X8"), s.Name, s.SamplesPerSecond, s.SampleCount,
                            s.Length, codec, s.Type, awc.MultiChannelFlag, awc.WholeFileEncrypted);
                        streamCount++;
                    }
                    awcCount++;
                    if(awcCount % 100 == 0) Console.WriteLine("AWC " + awcCount + " STREAMS " + streamCount);
                }
            } catch(Exception e) { Error(entry.Path, e); }
        }
        entries.Flush(); texts.Flush(); streams.Flush(); errors.Flush();
    }
    public static int Main(string[] args) {
        try {
            if(args.Length < 2) { Console.Error.WriteLine("ArchiveScanner.exe GAME_DIRECTORY OUTPUT_DIRECTORY [texts]"); return 2; }
            game = Path.GetFullPath(args[0]).TrimEnd('\\'); output = Path.GetFullPath(args[1]);
            if(output.TrimEnd('\\').Equals(game, StringComparison.OrdinalIgnoreCase) || output.StartsWith(game + "\\", StringComparison.OrdinalIgnoreCase)) throw new Exception("Output cannot be inside the game directory");
            Directory.CreateDirectory(output); Directory.CreateDirectory(Path.Combine(output,"rel"));
            entries = Writer("archive_entries.csv", "path","type","archive","bytes");
            texts = Writer("localized_text.csv", "language","path","table","hash","text");
            streams = Writer("audio_streams.csv", "awc_path","stream_hash_29bit","resolved_name","sample_rate","samples","duration_seconds","codec","type","multichannel","encrypted");
            errors = Writer("scan_errors.csv", "path","type","message");
            Console.WriteLine("Loading local game archive keys (no keys are exported)");
            GTA5Keys.LoadFromPath(game, null);
            bool textsOnly = args.Length > 2 && args[2] == "texts";
            var basePaths = textsOnly ? new List<string>{Path.Combine(game,"x64b.rpf")} : Directory.GetFiles(game,"*.rpf").OrderBy(x=>x).ToList();
            var update = Path.Combine(game,"update","update.rpf"); if(File.Exists(update)) basePaths.Add(update);
            var audioRoot = Path.Combine(game,"x64","audio");
            if(!textsOnly) basePaths.AddRange(Directory.GetFiles(audioRoot,"*.rpf"));
            foreach(var path in basePaths) { try { ReadArchive(path,false); } catch(Exception e) { Error(path,e); } }
            foreach(var path in (textsOnly ? new string[0] : Directory.GetFiles(Path.Combine(audioRoot,"sfx"),"*.rpf").OrderBy(x=>x).ToArray())) {
                string n = Path.GetFileNameWithoutExtension(path).ToUpperInvariant();
                if(n.StartsWith("SS_") || n == "S_MISC" || n == "SCRIPT" || n == "PROLOGUE" || n == "CUTSCENE_MASTERED_ONLY" || n == "PAIN")
                    try { ReadArchive(path,true); } catch(Exception e) { Error(path,e); }
            }
            Console.WriteLine("DONE AWC=" + awcCount + " STREAMS=" + streamCount + " TEXT=" + textCount);
            return 0;
        } catch(Exception e) { Console.Error.WriteLine(e); return 1; }
        finally { foreach(var w in new[]{entries,texts,streams,errors}) if(w!=null) w.Dispose(); }
    }
}
