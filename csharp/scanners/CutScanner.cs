using System;
using System.IO;
using System.Text;
using System.Linq;
using System.Collections.Generic;
using CodeWalker.GameFiles;

public class CutScanner {
    static IEnumerable<RpfFile> All(RpfFile r) { yield return r; if(r.Children != null) foreach(var c in r.Children) foreach(var x in All(c)) yield return x; }
    public static int Main(string[] args) {
        try {
            string game = Path.GetFullPath(args[0]).TrimEnd('\\'), output=Path.GetFullPath(args[1]);
            if(output.TrimEnd('\\').Equals(game,StringComparison.OrdinalIgnoreCase) || output.StartsWith(game+"\\",StringComparison.OrdinalIgnoreCase)) throw new Exception("Output is inside game directory");
            Directory.CreateDirectory(output);
            GTA5Keys.LoadFromPath(game,null);
            int count=0;
            using(var manifest=new StreamWriter(Path.Combine(output,"manifest.tsv"),false,new UTF8Encoding(true)))
            using(var errors=new StreamWriter(Path.Combine(output,"errors.tsv"),false,new UTF8Encoding(true))) {
                manifest.WriteLine("xml_file\tgame_path");
                foreach(string name in new[]{"x64b.rpf","x64c.rpf","update\\update.rpf"}) {
                    var root=new RpfFile(Path.Combine(game,name),name);root.ScanStructure(null,delegate(string e){errors.WriteLine(e);});
                    foreach(var r in All(root)) foreach(var entry in r.AllEntries.OfType<RpfFileEntry>()) {
                        string low=entry.Name.ToLowerInvariant();
                        if(!low.EndsWith(".cut") && !low.EndsWith(".nametable")) continue;
                        try {
                            var bytes=r.ExtractFile(entry); if(bytes==null)throw new Exception(r.LastError);
                            string filename=entry.Path.Replace('\\','_');
                            if(low.EndsWith(".nametable")) File.WriteAllBytes(Path.Combine(output,filename),bytes);
                            else {
                                var cut=new CutFile();cut.Load(bytes,entry);
                                if(cut.Pso==null)throw new Exception("Not a PSO cutscene");
                                filename+=".xml";
                                File.WriteAllText(Path.Combine(output,filename),PsoXml.GetXml(cut.Pso),new UTF8Encoding(false));
                                manifest.WriteLine(filename+"\t"+entry.Path);count++;
                            }
                        } catch(Exception ex) {errors.WriteLine(entry.Path+"\t"+ex.Message);}
                    }
                    Console.WriteLine("CUTS "+count+" after "+name);
                }
            }
            return 0;
        } catch(Exception e) {Console.Error.WriteLine(e);return 1;}
    }
}
