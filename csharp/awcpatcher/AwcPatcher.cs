using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using CodeWalker.GameFiles;

// PT-BR dubbing: replace speech streams inside AWC containers.
// Works only on COPIES of the game's RPF archives, written to OUTPUT_DIRECTORY (refuses the game folder).
//
// The AWC is rewritten at byte level: every chunk of every stream is copied verbatim, except the
// data / format / peak chunks of the replaced streams. Later chunks keep their original alignment
// (offsets shift by multiples of 16). CodeWalker is used only to read archives, decode for checks,
// encode IMA-ADPCM, encrypt (RSXXTEA) and insert the file into the RPF copy.
// (CodeWalker's own AwcFile.Save() round-trip corrupted every stream of these speech AWCs in tests.)
//
//   AwcPatcher.exe inspect GAME AWC_PATH [AWC_PATH...]
//   AwcPatcher.exe patch   GAME OUTPUT PLAN_TSV      (awc_path, stream_hash hex, wav PCM16 mono at stream rate)
//   AwcPatcher.exe export  GAME COPY_ROOT AWC_PATH HASH OUT_WAV
public class AwcPatcher {
    static string game;

    static void SplitPath(string full, out string archive, out string entryName) {
        int split = full.IndexOf(".rpf\\", StringComparison.OrdinalIgnoreCase);
        if (split < 0) throw new Exception("Expected root RPF path: " + full);
        archive = full.Substring(0, split + 4); entryName = full.Substring(split + 5);
        if (entryName.Contains("\\")) throw new Exception("Nested RPFs are not supported: " + full);
    }
    static RpfFile Open(string physical, string rel) {
        var rpf = new RpfFile(physical, rel); rpf.ScanStructure(null, null);
        if (rpf.AllEntries == null) throw new Exception(rpf.LastError);
        return rpf;
    }
    static RpfFileEntry Find(RpfFile rpf, string entryName) {
        return rpf.AllEntries.OfType<RpfFileEntry>().Single(e => e.Name.Equals(entryName, StringComparison.OrdinalIgnoreCase));
    }
    static AwcFile LoadAwc(byte[] bytes, RpfFileEntry entry) {
        var awc = new AwcFile(); awc.Load((byte[])bytes.Clone(), entry);
        if (!string.IsNullOrEmpty(awc.ErrorMessage)) throw new Exception(awc.ErrorMessage);
        return awc;
    }
    static double Rms(short[] s) { double a = 0; foreach (var v in s) a += (double)v * v; return Math.Sqrt(a / Math.Max(1, s.Length)); }
    static short[] Pcm16(byte[] b) { var s = new short[b.Length / 2]; Buffer.BlockCopy(b, 0, s, 0, s.Length * 2); return s; }
    static double Corr(short[] a, short[] b) {
        int n = Math.Min(a.Length, b.Length); double sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
        for (int i = 0; i < n; i++) { sa += a[i]; sb += b[i]; }
        double ma = sa / n, mb = sb / n;
        for (int i = 0; i < n; i++) { double x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
        return sab / Math.Sqrt(Math.Max(1e-9, saa * sbb));
    }
    static short[] ReadWav(string path, int expectRate) {
        byte[] wav = File.ReadAllBytes(path);
        if (BitConverter.ToUInt16(wav, 20) != 1 || BitConverter.ToUInt16(wav, 22) != 1 || BitConverter.ToUInt16(wav, 34) != 16)
            throw new Exception("WAV must be PCM16 mono: " + path);
        int rate = BitConverter.ToInt32(wav, 24);
        if (rate != expectRate) throw new Exception("Rate " + rate + " != stream rate " + expectRate + ": " + path);
        int at = 12;
        while (at + 8 <= wav.Length) {
            string id = System.Text.Encoding.ASCII.GetString(wav, at, 4); int len = BitConverter.ToInt32(wav, at + 4);
            if (id == "data") { len = Math.Min(len, wav.Length - at - 8); var s = new short[len / 2]; Buffer.BlockCopy(wav, at + 8, s, 0, s.Length * 2); return s; }
            at += 8 + len + (len & 1);
        }
        throw new Exception("No data chunk: " + path);
    }

    // Rockstar's XXTEA variant (port of CodeWalker master AwcFile.Encrypt_RSXXTEA; the dev46 DLL does not expose it)
    static void EncryptRSXXTEA(byte[] data) {
        var key = GTA5Keys.PC_AWC_KEY;
        uint[] blocks = new uint[data.Length / 4]; Buffer.BlockCopy(data, 0, blocks, 0, data.Length);
        int n = blocks.Length, rounds = 6 + (52 / n); uint sum = 0, y, z = blocks[n - 1];
        do {
            sum += 0x9E3779B9; uint e = (sum >> 2) & 3;
            for (int p = 0; p < n; p++) { y = blocks[(p + 1) % n]; z = blocks[p] += (z >> 5 ^ y << 2) + (y >> 3 ^ z << 4) ^ (sum ^ y) + (key[p & 3 ^ e] ^ z ^ 0x7B3A207F); }
        } while ((--rounds) != 0);
        Buffer.BlockCopy(blocks, 0, data, 0, data.Length);
    }

    static void DecryptRSXXTEA(byte[] data) {
        var key = GTA5Keys.PC_AWC_KEY;
        uint[] v = new uint[data.Length / 4]; Buffer.BlockCopy(data, 0, v, 0, data.Length);
        int n = v.Length, rounds = 6 + (52 / n); uint sum = (uint)rounds * 0x9E3779B9, y = v[0], z;
        do {
            uint e = (sum >> 2) & 3;
            for (int p = n - 1; p >= 0; p--) {
                z = v[(p + n - 1) % n];
                y = v[p] -= (z >> 5 ^ y << 2) + (y >> 3 ^ z << 4) ^ (sum ^ y) + (key[p & 3 ^ e] ^ z ^ 0x7B3A207F);
            }
            sum -= 0x9E3779B9;
        } while ((--rounds) != 0);
        Buffer.BlockCopy(v, 0, data, 0, data.Length);
    }
    // large block b of a multichannel data chunk, decrypted when the AWC has the multichannel-encrypt flag
    static byte[] McBlock(byte[] awc, int off, int len, bool enc) {
        var b = new byte[len]; Buffer.BlockCopy(awc, off, b, 0, len);
        if (enc) DecryptRSXXTEA(b);
        return b;
    }

    class Chunk { public int Stream; public byte Type; public int Size; public int Offset; public long InfoPos; public byte[] NewData; }
    const byte DATA = 0x55, FORMAT = 0xFA, PEAK = 0x36;

    // replacements: stream id (29 bit) -> final PCM16 (already loudness matched)
    static byte[] Rebuild(byte[] orig, Dictionary<uint, short[]> repl) {
        uint magic = BitConverter.ToUInt32(orig, 0);
        if (magic != 0x54414441) throw new Exception("Unsupported AWC (whole-file encrypted or big-endian)");
        ushort flags = BitConverter.ToUInt16(orig, 6);
        if ((flags & 4) != 0) throw new Exception("Multichannel AWC not supported");
        bool encrypted = (flags & 2) != 0;
        int count = BitConverter.ToInt32(orig, 8), dataOffset = BitConverter.ToInt32(orig, 12);
        int pos = 16 + ((flags & 1) != 0 ? count * 2 : 0);
        var ids = new uint[count]; var nchunks = new int[count];
        for (int i = 0; i < count; i++) { uint v = BitConverter.ToUInt32(orig, pos); pos += 4; nchunks[i] = (int)(v >> 29); ids[i] = v & 0x1FFFFFFF; }
        var chunks = new List<Chunk>();
        for (int i = 0; i < count; i++)
            for (int j = 0; j < nchunks[i]; j++) {
                ulong raw = BitConverter.ToUInt64(orig, pos);
                chunks.Add(new Chunk { Stream = i, Type = (byte)(raw >> 56), Size = (int)((raw >> 28) & 0x0FFFFFFF), Offset = (int)(raw & 0x0FFFFFFF), InfoPos = pos });
                pos += 8;
            }
        if (pos != dataOffset) throw new Exception("Header size mismatch");
        foreach (var kv in repl) {
            int si = Array.IndexOf(ids, kv.Key); if (si < 0) throw new Exception("Stream not found " + kv.Key.ToString("X8"));
            var mine = chunks.Where(c => c.Stream == si).ToList();
            var data = mine.SingleOrDefault(c => c.Type == DATA); var fmt = mine.SingleOrDefault(c => c.Type == FORMAT); var peak = mine.SingleOrDefault(c => c.Type == PEAK);
            if (data == null || fmt == null) throw new Exception("Stream lacks data/format chunk " + kv.Key.ToString("X8"));
            if (orig[fmt.Offset + 19] != 4) throw new Exception("Stream codec is not ADPCM " + kv.Key.ToString("X8"));
            short[] pcm = kv.Value;
            int k = (pcm.Length + 4087) / 4088;                 // CodeWalker encoder: blocks of 4-byte header + 4088 samples
            var input = new byte[k * 4096 * 2]; Buffer.BlockCopy(pcm, 0, input, 0, pcm.Length * 2);
            byte[] adpcm = ADPCMCodec.EncodeADPCM(input, k * 4096 * 2);
            if (adpcm.Length != k * 2048) throw new Exception("Unexpected ADPCM size");
            if (encrypted) EncryptRSXXTEA(adpcm);
            data.NewData = adpcm;
            Func<int, ushort> blockPeak = b => { int m = 0; for (int i = b * 4096; i < Math.Min(pcm.Length, (b + 1) * 4096); i++) m = Math.Max(m, Math.Abs((int)pcm[i]) * 2); return (ushort)Math.Min(m, 65535); };
            var f = new byte[fmt.Size]; Buffer.BlockCopy(orig, fmt.Offset, f, 0, fmt.Size);
            Buffer.BlockCopy(BitConverter.GetBytes((uint)pcm.Length), 0, f, 0, 4);
            if (BitConverter.ToInt32(f, 4) >= pcm.Length) Buffer.BlockCopy(BitConverter.GetBytes(-1), 0, f, 4, 4);
            if (fmt.Size >= 24) Buffer.BlockCopy(BitConverter.GetBytes(blockPeak(0)), 0, f, 20, 2);
            fmt.NewData = f;
            if (peak != null) {
                int pc = Math.Max(0, (pcm.Length - 4096) / 4096);
                var p = new byte[pc * 2];
                for (int n = 0; n < pc; n++) Buffer.BlockCopy(BitConverter.GetBytes(blockPeak(n + 1)), 0, p, n * 2, 2);
                peak.NewData = p;
            }
        }
        // layout: original order, shift later chunks by multiples of 16 so their alignment is unchanged
        var ordered = chunks.OrderBy(c => c.Offset).ToList();
        for (int i = 1; i < ordered.Count; i++)
            if (ordered[i].Offset < ordered[i - 1].Offset + ordered[i - 1].Size) throw new Exception("Overlapping chunks; refusing");
        var newOffset = new Dictionary<Chunk, int>(); int shift = 0, end = dataOffset;
        foreach (var c in ordered) {
            int o = c.Offset + shift;
            if (o < end) throw new Exception("Layout error");
            newOffset[c] = o;
            int size = c.NewData != null ? c.NewData.Length : c.Size;
            if (size > c.Size) shift += ((size - c.Size) + 15) / 16 * 16;
            end = o + size;
        }
        var last = ordered.Last(); int tail = orig.Length - (last.Offset + last.Size);
        var outb = new byte[end + Math.Max(0, tail)];
        Buffer.BlockCopy(orig, 0, outb, 0, dataOffset);
        foreach (var c in ordered) {
            int o = newOffset[c];
            if (c.NewData != null) Buffer.BlockCopy(c.NewData, 0, outb, o, c.NewData.Length);
            else Buffer.BlockCopy(orig, c.Offset, outb, o, c.Size);
            int size = c.NewData != null ? c.NewData.Length : c.Size;
            if (o > 0x0FFFFFFF || size > 0x0FFFFFFF) throw new Exception("AWC too large");
            ulong raw = ((ulong)c.Type << 56) | (((ulong)size & 0x0FFFFFFF) << 28) | ((ulong)o & 0x0FFFFFFF);
            Buffer.BlockCopy(BitConverter.GetBytes(raw), 0, outb, (int)c.InfoPos, 8);
        }
        return outb;
    }

    const byte STREAMFORMAT = 0x48, SEEKTABLE = 0xA3;

    // Multichannel AWC (cutscene mixes): the source stream (id 0) holds one data chunk cut into large blocks of
    // BlockSize bytes. Each large block = per-channel headers (6 ints) + per-channel small-block offsets, padded to
    // 0x800, then each channel's data as BlockCount x 2048-byte IMA-ADPCM blocks (4-byte header + 4088 samples)
    // holding SampleCount samples that start at seektable[b]. Replacement re-encodes a channel IN PLACE:
    // same sample count, same block layout, same file size; every other byte is kept.
    // replacements: channel id (29 bit) -> full channel PCM16 (exactly the channel's sample count)
    static byte[] RebuildMC(byte[] orig, Dictionary<uint, short[]> repl) {
        if (BitConverter.ToUInt32(orig, 0) != 0x54414441) throw new Exception("Unsupported AWC (whole-file encrypted or big-endian)");
        ushort flags = BitConverter.ToUInt16(orig, 6);
        if ((flags & 4) == 0) throw new Exception("Not a multichannel AWC");
        bool enc = (flags & 8) != 0;
        int count = BitConverter.ToInt32(orig, 8);
        int pos = 16 + ((flags & 1) != 0 ? count * 2 : 0);
        var ids = new uint[count]; var nchunks = new int[count];
        for (int i = 0; i < count; i++) { uint v = BitConverter.ToUInt32(orig, pos); pos += 4; nchunks[i] = (int)(v >> 29); ids[i] = v & 0x1FFFFFFF; }
        var chunks = new List<Chunk>();
        for (int i = 0; i < count; i++)
            for (int j = 0; j < nchunks[i]; j++) {
                ulong raw = BitConverter.ToUInt64(orig, pos);
                chunks.Add(new Chunk { Stream = i, Type = (byte)(raw >> 56), Size = (int)((raw >> 28) & 0x0FFFFFFF), Offset = (int)(raw & 0x0FFFFFFF), InfoPos = pos });
                pos += 8;
            }
        var src = chunks.Where(c => c.Type == STREAMFORMAT).ToList();
        if (src.Count != 1) throw new Exception("Expected one streamformat chunk, found " + src.Count);
        int si = src[0].Stream;
        var data = chunks.Single(c => c.Stream == si && c.Type == DATA); var seek = chunks.Single(c => c.Stream == si && c.Type == SEEKTABLE);
        int sf = src[0].Offset;
        int blockCount = BitConverter.ToInt32(orig, sf), blockSize = BitConverter.ToInt32(orig, sf + 4), chanCount = BitConverter.ToInt32(orig, sf + 8);
        var chanIds = new uint[chanCount]; var chanSamples = new int[chanCount];
        for (int c = 0; c < chanCount; c++) {
            int o = sf + 12 + c * 16; chanIds[c] = BitConverter.ToUInt32(orig, o) & 0x1FFFFFFF; chanSamples[c] = BitConverter.ToInt32(orig, o + 4);
            if (orig[o + 12] != 4) throw new Exception("Channel codec is not ADPCM");
        }
        foreach (var kv in repl) {
            int c = Array.IndexOf(chanIds, kv.Key); if (c < 0) throw new Exception("Channel not found " + kv.Key.ToString("X8"));
            if (kv.Value.Length != chanSamples[c]) throw new Exception("Channel " + kv.Key.ToString("X8") + " needs exactly " + chanSamples[c] + " samples, got " + kv.Value.Length);
        }
        var outb = (byte[])orig.Clone();
        for (int b = 0; b < blockCount; b++) {
            int bo = data.Offset + b * blockSize, blen = Math.Min(blockSize, data.Size - b * blockSize);
            if (enc && blen % 4 != 0) throw new Exception("Encrypted block length not a multiple of 4");
            int sampleStart = BitConverter.ToInt32(orig, seek.Offset + b * 4);
            var blk = McBlock(orig, bo, blen, enc);
            if (enc && b == 0) {   // the RSXXTEA port must reproduce the game's bytes exactly
                var again = (byte[])blk.Clone(); EncryptRSXXTEA(again);
                for (int i = 0; i < blen; i++) if (again[i] != orig[bo + i]) throw new Exception("RSXXTEA round trip mismatch");
            }
            var nblk = new int[chanCount]; var nsmp = new int[chanCount];
            int hp = 0;
            for (int c = 0; c < chanCount; c++) { nblk[c] = BitConverter.ToInt32(blk, hp + 4); nsmp[c] = BitConverter.ToInt32(blk, hp + 12); hp += 24; }
            for (int c = 0; c < chanCount; c++) hp += nblk[c] * 4;
            int dp = ((hp + 0x7FF) / 0x800) * 0x800;
            bool touched = false;
            for (int c = 0; c < chanCount; c++) {
                short[] pcm;
                if (repl.TryGetValue(chanIds[c], out pcm)) {
                    int k = nblk[c];
                    if (nsmp[c] > k * 4088) throw new Exception("Block sample count exceeds capacity");
                    var input = new byte[k * 4096 * 2];
                    int n = Math.Max(0, Math.Min(nsmp[c], pcm.Length - sampleStart));
                    Buffer.BlockCopy(pcm, sampleStart * 2, input, 0, n * 2);
                    byte[] adpcm = ADPCMCodec.EncodeADPCM(input, k * 4096 * 2);
                    if (adpcm.Length != k * 2048) throw new Exception("Unexpected ADPCM size");
                    Buffer.BlockCopy(adpcm, 0, blk, dp, adpcm.Length); touched = true;
                }
                dp += nblk[c] * 2048;
            }
            if (dp > blen) throw new Exception("Block layout overflow in block " + b);
            if (touched) { if (enc) EncryptRSXXTEA(blk); Buffer.BlockCopy(blk, 0, outb, bo, blen); }
        }
        return outb;
    }

    // Exact decode of every channel of a multichannel AWC: block b contributes SampleCount samples at seektable[b].
    // (CodeWalker's GetPcmData concatenates whole 4088-sample small blocks, so its output drifts by the padding.)
    static Dictionary<uint, short[]> DecodeMC(byte[] awc) {
        ushort flags = BitConverter.ToUInt16(awc, 6);
        if ((flags & 4) == 0) throw new Exception("DecodeMC: not a multichannel AWC");
        bool enc = (flags & 8) != 0;
        int count = BitConverter.ToInt32(awc, 8);
        int pos = 16 + ((flags & 1) != 0 ? count * 2 : 0);
        var nchunks = new int[count];
        for (int i = 0; i < count; i++) { nchunks[i] = (int)(BitConverter.ToUInt32(awc, pos) >> 29); pos += 4; }
        var chunks = new List<Chunk>();
        for (int i = 0; i < count; i++)
            for (int j = 0; j < nchunks[i]; j++) {
                ulong raw = BitConverter.ToUInt64(awc, pos);
                chunks.Add(new Chunk { Stream = i, Type = (byte)(raw >> 56), Size = (int)((raw >> 28) & 0x0FFFFFFF), Offset = (int)(raw & 0x0FFFFFFF), InfoPos = pos });
                pos += 8;
            }
        var sfc = chunks.Single(c => c.Type == STREAMFORMAT);
        var data = chunks.Single(c => c.Stream == sfc.Stream && c.Type == DATA); var seek = chunks.Single(c => c.Stream == sfc.Stream && c.Type == SEEKTABLE);
        int sf = sfc.Offset, blockCount = BitConverter.ToInt32(awc, sf), blockSize = BitConverter.ToInt32(awc, sf + 4), chanCount = BitConverter.ToInt32(awc, sf + 8);
        var res = new Dictionary<uint, short[]>(); var ids = new uint[chanCount];
        for (int c = 0; c < chanCount; c++) { ids[c] = BitConverter.ToUInt32(awc, sf + 12 + c * 16) & 0x1FFFFFFF; res[ids[c]] = new short[BitConverter.ToInt32(awc, sf + 12 + c * 16 + 4)]; }
        for (int b = 0; b < blockCount; b++) {
            int bo = data.Offset + b * blockSize, sampleStart = BitConverter.ToInt32(awc, seek.Offset + b * 4);
            var lb = McBlock(awc, bo, Math.Min(blockSize, data.Size - b * blockSize), enc);
            var nblk = new int[chanCount]; var nsmp = new int[chanCount]; int hp = 0;
            for (int c = 0; c < chanCount; c++) { nblk[c] = BitConverter.ToInt32(lb, hp + 4); nsmp[c] = BitConverter.ToInt32(lb, hp + 12); hp += 24; }
            for (int c = 0; c < chanCount; c++) hp += nblk[c] * 4;
            int dp = ((hp + 0x7FF) / 0x800) * 0x800;
            for (int c = 0; c < chanCount; c++) {
                var blk = new byte[nblk[c] * 2048]; Buffer.BlockCopy(lb, dp, blk, 0, blk.Length);
                var pcm = Pcm16(ADPCMCodec.DecodeADPCM(blk, nsmp[c]));
                var dst = res[ids[c]]; int n = Math.Min(nsmp[c], Math.Min(pcm.Length, dst.Length - sampleStart));
                if (n > 0) Array.Copy(pcm, 0, dst, sampleStart, n);
                dp += nblk[c] * 2048;
            }
        }
        return res;
    }
    static void WriteWav(string path, short[] pcm, int rate) {
        using (var w = new BinaryWriter(File.Create(path))) {
            w.Write(System.Text.Encoding.ASCII.GetBytes("RIFF")); w.Write(36 + pcm.Length * 2); w.Write(System.Text.Encoding.ASCII.GetBytes("WAVEfmt "));
            w.Write(16); w.Write((short)1); w.Write((short)1); w.Write(rate); w.Write(rate * 2); w.Write((short)2); w.Write((short)16);
            w.Write(System.Text.Encoding.ASCII.GetBytes("data")); w.Write(pcm.Length * 2);
            var b = new byte[pcm.Length * 2]; Buffer.BlockCopy(pcm, 0, b, 0, b.Length); w.Write(b);
        }
    }

    static int Inspect(string[] paths) {
        foreach (var full in paths) {
            string archive, entryName; SplitPath(full, out archive, out entryName);
            var rpf = Open(Path.Combine(game, archive), archive); var entry = Find(rpf, entryName); var awc = LoadAwc(rpf.ExtractFile(entry), entry);
            Console.WriteLine(full + " rpfEncryption=" + rpf.Encryption + " awcFlags=0x" + awc.Flags.ToString("X4") + " streams=" + awc.Streams.Length);
            foreach (var s in awc.Streams.Take(4)) {
                var f = s.FormatChunk;
                Console.WriteLine("  " + ((uint)s.Hash).ToString("X8") + " codec=" + f.Codec + " rate=" + f.SamplesPerSecond + " samples=" + f.Samples +
                    " data=" + (s.DataChunk == null ? -1 : s.DataChunk.Data.Length) + " peakChunk=" + (s.PeakChunk == null ? -1 : s.PeakChunk.Data.Length));
                if (s.DataChunk != null)   // first bytes, to identify an unknown codec (Enhanced: codec 7)
                    Console.WriteLine("    head " + BitConverter.ToString(s.DataChunk.Data, 0, Math.Min(40, s.DataChunk.Data.Length)));
            }
        }
        return 0;
    }

    static int Patch(string output, string planPath) {
        if (output.TrimEnd('\\').Equals(game, StringComparison.OrdinalIgnoreCase) || output.StartsWith(game + "\\", StringComparison.OrdinalIgnoreCase))
            throw new Exception("Output cannot be inside the game directory");
        var plan = File.ReadLines(planPath).Skip(1).Where(l => l.Trim().Length > 0).Select(l => l.Split('\t')).ToList();
        int total = 0;
        foreach (var byArchive in plan.GroupBy(p => { string a, e; SplitPath(p[0], out a, out e); return a; }, StringComparer.OrdinalIgnoreCase)) {
            string archive = byArchive.Key;
            string src = Path.GetFullPath(Path.Combine(game, archive)), dst = Path.GetFullPath(Path.Combine(output, archive));
            if (!src.StartsWith(game + "\\", StringComparison.OrdinalIgnoreCase)) throw new Exception("Archive path escaped game directory");
            Directory.CreateDirectory(Path.GetDirectoryName(dst));
            File.Copy(src, dst, true);
            var rpf = Open(dst, archive);
            // CodeWalker cannot write NG-encrypted tables: the COPY becomes OPEN (unencrypted TOC, as OpenIV writes);
            // the game reads it through OpenIV.asi from the mods folder.
            RpfFile.SetEncryptionType(rpf, RpfEncryption.OPEN);
            var expected = new Dictionary<string, Dictionary<uint, short[]>>(StringComparer.OrdinalIgnoreCase);
            var originals = new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase);
            foreach (var byAwc in byArchive.GroupBy(p => p[0], StringComparer.OrdinalIgnoreCase)) {
                string a, entryName; SplitPath(byAwc.Key, out a, out entryName);
                var entry = Find(rpf, entryName); byte[] orig = rpf.ExtractFile(entry); if (orig == null) throw new Exception(rpf.LastError);
                var awc = LoadAwc(orig, entry);
                var repl = new Dictionary<uint, short[]>();
                if (awc.MultiChannelFlag) {
                    // cutscene mix channel: the WAV is the complete remixed channel (same length, no gain change)
                    foreach (var p in byAwc) {
                        uint hash = Convert.ToUInt32(p[1], 16);
                        var s = awc.Streams.Single(x => (uint)x.Hash == hash);
                        short[] pcm = ReadWav(p[2], s.SamplesPerSecond);
                        if (pcm.Length != s.SampleCount) throw new Exception("Channel " + p[1] + " needs " + s.SampleCount + " samples, WAV has " + pcm.Length);
                        repl[hash & 0x1FFFFFFF] = pcm; total++;
                        Console.WriteLine("CHANNEL " + p[0] + " " + p[1] + " samples " + pcm.Length);
                    }
                    byte[] mc = RebuildMC(orig, repl);
                    RpfFile.CreateFile(entry.Parent, entry.Name, mc, true);
                    expected[entryName] = repl; originals[entryName] = orig;
                    Console.WriteLine("AWC " + byAwc.Key + " " + repl.Count + " channels, size unchanged " + (mc.Length == orig.Length));
                    continue;
                }
                foreach (var p in byAwc) {
                    uint hash = Convert.ToUInt32(p[1], 16);
                    var s = awc.Streams.Single(x => (uint)x.Hash == hash);
                    var f = s.FormatChunk;
                    var origPcm = Pcm16(s.GetPcmData()); if (origPcm.Length > f.Samples) Array.Resize(ref origPcm, (int)f.Samples);
                    short[] pcm = ReadWav(p[2], f.SamplesPerSecond);
                    double gain = Rms(origPcm) / Math.Max(1.0, Rms(pcm));             // match the original line's loudness
                    int pk = pcm.Length == 0 ? 1 : pcm.Max(v => Math.Abs((int)v));
                    gain = Math.Min(gain, 31000.0 / Math.Max(1, pk));                  // stay under -0.5 dBFS
                    var outPcm = new short[Math.Max(pcm.Length, 8192)];                 // >= 2 peak blocks
                    for (int i = 0; i < pcm.Length; i++) outPcm[i] = (short)Math.Max(-32768, Math.Min(32767, Math.Round(pcm[i] * gain)));
                    repl[hash & 0x1FFFFFFF] = outPcm;
                    Console.WriteLine("STREAM " + p[0] + " " + p[1] + " samples " + origPcm.Length + " -> " + outPcm.Length + " gain " + gain.ToString("0.00"));
                    total++;
                }
                byte[] rebuilt = Rebuild(orig, repl);
                RpfFile.CreateFile(entry.Parent, entry.Name, rebuilt, true);
                expected[entryName] = repl; originals[entryName] = orig;
                Console.WriteLine("AWC " + byAwc.Key + " " + repl.Count + " streams, " + orig.Length + " -> " + rebuilt.Length + " bytes");
            }
            // verify from a fresh read of the copy: replaced streams decode to the new audio,
            // untouched streams decode exactly as in the game's original file
            var verify = Open(dst, archive);
            foreach (var kv in expected) {
                var entry = Find(verify, kv.Key);
                var vbytes = verify.ExtractFile(entry);
                var awc = LoadAwc(vbytes, entry); var org = LoadAwc(originals[kv.Key], entry);
                if (awc.MultiChannelFlag) {
                    if (vbytes.Length != originals[kv.Key].Length) throw new Exception("Multichannel AWC size changed " + kv.Key);
                    var now = DecodeMC(vbytes); var before = DecodeMC(originals[kv.Key]);
                    foreach (var ch in now) {
                        short[] want;
                        if (kv.Value.TryGetValue(ch.Key, out want)) {
                            // a channel that is silent in the original (and in the mix) has no defined correlation:
                            // accept it when the patched channel is silent too
                            Func<short[], double> rms = a => Math.Sqrt(a.Length == 0 ? 0 : a.Average(s => (double)s * s));
                            if (rms(want) < 30) {
                                if (rms(ch.Value) > 60) throw new Exception("Verify silent channel not silent " + kv.Key + " " + ch.Key.ToString("X8"));
                                Console.WriteLine("VERIFY " + kv.Key + " " + ch.Key.ToString("X8") + " silent");
                                continue;
                            }
                            double c = Corr(ch.Value, want);
                            if (c < 0.95) throw new Exception("Verify channel " + kv.Key + " " + ch.Key.ToString("X8") + " corr " + c.ToString("0.000"));
                            Console.WriteLine("VERIFY " + kv.Key + " " + ch.Key.ToString("X8") + " corr " + c.ToString("0.0000"));
                        } else if (!ch.Value.SequenceEqual(before[ch.Key])) throw new Exception("Untouched channel changed " + kv.Key + " " + ch.Key.ToString("X8"));
                    }
                    continue;
                }
                foreach (var s in awc.Streams) {
                    uint id = (uint)s.Hash & 0x1FFFFFFF; short[] want;
                    if (awc.MultiChannelFlag && id == 0) continue;   // multichannel source container, checked through its channels
                    if (kv.Value.TryGetValue(id, out want)) {
                        if (s.SampleCount != want.Length) throw new Exception("Verify samples " + kv.Key + " " + id.ToString("X8"));
                        double c = Corr(Pcm16(s.GetPcmData()), want);
                        if (c < 0.95) throw new Exception("Verify audio " + kv.Key + " " + id.ToString("X8") + " corr " + c.ToString("0.000"));
                    } else {
                        var o = org.Streams.Single(x => x.Hash == s.Hash);
                        if (!o.GetPcmData().SequenceEqual(s.GetPcmData())) throw new Exception("Untouched stream changed " + kv.Key + " " + id.ToString("X8"));
                    }
                }
            }
            Console.WriteLine("RPF " + archive + " OK encryption=" + verify.Encryption + " size=" + new FileInfo(dst).Length);
        }
        Console.WriteLine("DONE " + total + " streams");
        return 0;
    }

    public static int Main(string[] args) {
        try {
            if (args.Length < 3) throw new Exception("usage: inspect GAME AWC_PATH... | patch GAME OUTPUT PLAN_TSV | export GAME COPY_ROOT AWC_PATH HASH OUT_WAV");
            game = Path.GetFullPath(args[1]).TrimEnd('\\');
            // keys come from the Legacy gta5.exe; AWC_KEYS_GAME lets another install (Enhanced) borrow them
            GTA5Keys.LoadFromPath(Environment.GetEnvironmentVariable("AWC_KEYS_GAME") ?? game, null);
            if (args[0] == "inspect") return Inspect(args.Skip(2).ToArray());
            if (args[0] == "patch" && args.Length == 4) return Patch(Path.GetFullPath(args[2]), args[3]);
            if (args[0] == "exportmc" && args.Length == 4) {
                // exportmc GAME LIST_TXT OUT_DIR : every channel of every listed multichannel AWC, exact decode,
                // one archive scan per RPF. Writes OUT_DIR\<awc name>\ch_<HASH>.wav; skips folders already complete.
                var list = File.ReadLines(args[2]).Select(l => l.Trim()).Where(l => l.Length > 0).ToList();
                foreach (var byArchive in list.GroupBy(p => { string a, e; SplitPath(p, out a, out e); return a; }, StringComparer.OrdinalIgnoreCase)) {
                    var rpf = Open(Path.Combine(game, byArchive.Key), byArchive.Key);
                    foreach (var full in byArchive) {
                        string a, entryName; SplitPath(full, out a, out entryName);
                        string dir = Path.Combine(args[3], Path.GetFileNameWithoutExtension(entryName));
                        if (File.Exists(Path.Combine(dir, "done.txt"))) continue;
                        Directory.CreateDirectory(dir);
                        var entry = Find(rpf, entryName); var bytes = rpf.ExtractFile(entry); var awc = LoadAwc(bytes, entry);
                        if (!awc.MultiChannelFlag) { Console.WriteLine("SKIP single-channel " + full); continue; }
                        var chans = DecodeMC(bytes);
                        foreach (var ch in chans) {
                            var s = awc.Streams.Single(x => ((uint)x.Hash & 0x1FFFFFFF) == ch.Key);
                            WriteWav(Path.Combine(dir, "ch_" + ch.Key.ToString("X8") + ".wav"), ch.Value, s.SamplesPerSecond);
                        }
                        File.WriteAllText(Path.Combine(dir, "done.txt"), full + "\n" + chans.Count + " channels\n");
                        Console.WriteLine("EXPORTED " + full + " " + chans.Count + " ch");
                    }
                }
                return 0;
            }
            if (args[0] == "export" && args.Length == 6) {
                string archive, entryName; SplitPath(args[3], out archive, out entryName);
                var rpf = Open(Path.Combine(Path.GetFullPath(args[2]), archive), archive); var entry = Find(rpf, entryName);
                var bytes = rpf.ExtractFile(entry); var awc = LoadAwc(bytes, entry);
                var s = awc.Streams.Single(x => (uint)x.Hash == Convert.ToUInt32(args[4], 16));
                if (awc.MultiChannelFlag) {
                    var ch = DecodeMC(bytes)[Convert.ToUInt32(args[4], 16) & 0x1FFFFFFF];
                    WriteWav(args[5], ch, s.SamplesPerSecond); Console.WriteLine("samples " + ch.Length + " (exact multichannel decode)"); return 0;
                }
                File.WriteAllBytes(args[5], s.GetWavFile()); Console.WriteLine("samples " + s.SampleCount); return 0;
            }
            throw new Exception("bad arguments");
        } catch (Exception ex) { Console.Error.WriteLine(ex); return 1; }
    }
}
