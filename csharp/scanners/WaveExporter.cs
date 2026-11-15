using System;
using System.IO;
using System.Linq;
using CodeWalker.GameFiles;

// Export selected original mono streams for local review. Does not modify archives.
public class WaveExporter {
    public static int Main(string[] args) {
        try {
            if(args.Length != 3) throw new Exception("WaveExporter.exe GAME_DIRECTORY OUTPUT_DIRECTORY SELECTION_TSV");
            string game = Path.GetFullPath(args[0]).TrimEnd('\\');
            string output = Path.GetFullPath(args[1]);
            if(output.TrimEnd('\\').Equals(game,StringComparison.OrdinalIgnoreCase) || output.StartsWith(game + "\\",StringComparison.OrdinalIgnoreCase)) throw new Exception("Output cannot be inside game directory");
            Directory.CreateDirectory(output);
            GTA5Keys.LoadFromPath(game, null);
            string previousArchive = null; RpfFile rpf = null; AwcFile awc = null; string previousAwc = null;
            foreach(var line in File.ReadLines(args[2]).Skip(1)) {
                var parts = line.Split('\t');
                if(parts.Length != 4) throw new Exception("Selection needs filename, awc_path, stream_hash, line_id");
                string name = parts[0];
                if(name != Path.GetFileName(name)) throw new Exception("Output must be a basename");
                string full = parts[1]; int split = full.IndexOf(".rpf\\",StringComparison.OrdinalIgnoreCase);
                if(split < 0) throw new Exception("Expected root RPF path");
                string archive = full.Substring(0,split+4); string entryName = full.Substring(split+5);
                if(entryName.Contains("\\")) throw new Exception("Nested RPFs are not supported by this review exporter");
                string archiveFull = Path.GetFullPath(Path.Combine(game, archive));
                if(!archiveFull.StartsWith(game + "\\",StringComparison.OrdinalIgnoreCase)) throw new Exception("Archive path escaped game directory");
                if(archive != previousArchive) {
                    rpf = new RpfFile(archiveFull, archive); rpf.ScanStructure(null,null); previousArchive = archive; previousAwc = null;
                    if(rpf.AllEntries == null) throw new Exception(rpf.LastError);
                }
                if(full != previousAwc) {
                    var entry = rpf.AllEntries.OfType<RpfFileEntry>().Single(e=>e.Name.Equals(entryName,StringComparison.OrdinalIgnoreCase));
                    byte[] bytes = rpf.ExtractFile(entry); if(bytes == null) throw new Exception(rpf.LastError);
                    awc = new AwcFile(); awc.Load(bytes,entry); previousAwc = full;
                    if(!string.IsNullOrEmpty(awc.ErrorMessage)) throw new Exception(awc.ErrorMessage);
                }
                uint hash = Convert.ToUInt32(parts[2],16);
                var stream = awc.Streams.Single(s=>(uint)s.Hash == hash);
                byte[] wav = stream.GetWavFile();
                if(wav == null || wav.Length < 44) throw new Exception("Invalid WAV result");
                // CodeWalker's ADPCM decoder includes the padded tail of the last block.
                // Its PCM16 mono WAV header is fixed at 44 bytes; preserve the declared sample count.
                if(BitConverter.ToUInt16(wav,20) != 1 || BitConverter.ToUInt16(wav,22) != 1 ||
                   BitConverter.ToUInt16(wav,34) != 16 || System.Text.Encoding.ASCII.GetString(wav,36,4) != "data")
                    throw new Exception("Unexpected WAV format; refusing to trim it");
                int pcmBytes = checked(stream.SampleCount * 2);
                if(wav.Length < 44 + pcmBytes) throw new Exception("Decoded WAV shorter than declared sample count");
                if(wav.Length > 44 + pcmBytes) Array.Resize(ref wav,44 + pcmBytes);
                Buffer.BlockCopy(BitConverter.GetBytes(36 + pcmBytes),0,wav,4,4);
                Buffer.BlockCopy(BitConverter.GetBytes(pcmBytes),0,wav,40,4);
                File.WriteAllBytes(Path.Combine(output,name),wav);
                Console.WriteLine("EXPORTED " + name + " " + wav.Length + " bytes");
            }
            return 0;
        } catch(Exception ex) { Console.Error.WriteLine(ex); return 1; }
    }
}
