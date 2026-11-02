// SelfExtract.exe: a small installer program. It replaces IExpress, which cannot give an exe its own icon.
// The build adds the files to the end of this exe:   [payload.zip][settings text][settings length: 4 bytes][zip length: 8 bytes]["CALEZIP1"]
// When run, it unpacks the zip into the target folder, then starts the program named in the settings.
//
// Settings (one  key=value  per line, UTF-8):
//   name=Cale                                       shown while unpacking
//   target=%LOCALAPPDATA%\Cale\app                  where the files go
//   target_calehome=%CALE_HOME%\app                 used instead when the CALE_HOME variable is set (tests)
//   run=%TARGET%\bin\CaleSetup.exe                  what to start afterwards
//   args=                                           its arguments (%TARGET% works here too)
// Written for the C# 5 compiler that comes with Windows (csc.exe, .NET Framework 4).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace CaleHost
{
    // A read-only window onto part of a file, so the zip can be read where it sits inside the exe.
    class Slice : Stream
    {
        Stream inner;
        long start, length, pos;
        public Slice(Stream s, long offset, long len) { inner = s; start = offset; length = len; pos = 0; }
        public override bool CanRead { get { return true; } }
        public override bool CanSeek { get { return true; } }
        public override bool CanWrite { get { return false; } }
        public override long Length { get { return length; } }
        public override long Position { get { return pos; } set { pos = value; } }
        public override void Flush() { }
        public override int Read(byte[] buffer, int offset, int count)
        {
            long left = length - pos;
            if (left <= 0) return 0;
            if (count > left) count = (int)left;
            inner.Position = start + pos;
            int n = inner.Read(buffer, offset, count);
            pos += n;
            return n;
        }
        public override long Seek(long offset, SeekOrigin origin)
        {
            if (origin == SeekOrigin.Begin) pos = offset;
            else if (origin == SeekOrigin.Current) pos += offset;
            else pos = length + offset;
            return pos;
        }
        public override void SetLength(long value) { throw new NotSupportedException(); }
        public override void Write(byte[] buffer, int offset, int count) { throw new NotSupportedException(); }
    }

    // Writes the files of the zip into the target folder, with one or more threads (each with its own handle on the archive).
    static class Extractor
    {
        // One thread by default. Measured on a 4-core laptop with a virus scanner, on the real 61,000-file archive: 1 thread 251 s, 4 threads 284 s.
        // (A Python test of the same work gave 204 s with 1 thread and 144 s with 8, so the gain depends on the program that writes the files.)
        // CALE_EXTRACT_THREADS (1 to 32) tries more. The code is tested for any number: scripts/test_extract_win.py.
        public static int DefaultThreads()
        {
            int n;
            string v = Environment.GetEnvironmentVariable("CALE_EXTRACT_THREADS");
            if (!string.IsNullOrEmpty(v) && int.TryParse(v, out n) && n >= 1 && n <= 32) return n;
            return 1;
        }

        public static void ExtractAll(string exe, long zipStart, long zipLen, string target, int threads)
        {
            string root = target.TrimEnd('\\') + "\\";
            if (threads < 1) threads = 1;
            Exception failure = null;
            object gate = new object();
            List<Thread> started = new List<Thread>();
            for (int k = 0; k < threads; k++)
            {
                int mine = k;
                Thread t = new Thread(delegate ()
                {
                    try { Part(exe, zipStart, zipLen, target, root, mine, threads); }
                    catch (Exception ex) { lock (gate) { if (failure == null) failure = ex; } }
                });
                t.IsBackground = true;
                t.Start();
                started.Add(t);
            }
            foreach (Thread t in started) t.Join();
            if (failure != null) throw failure;
        }

        // The files number `mine` (counting from 0, every `threads`-th) of the zip.
        static void Part(string exe, long zipStart, long zipLen, string target, string root, int mine, int threads)
        {
            using (FileStream fs = new FileStream(exe, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (Slice slice = new Slice(fs, zipStart, zipLen))
            using (ZipArchive zip = new ZipArchive(slice, ZipArchiveMode.Read))
            {
                int i = -1;
                foreach (ZipArchiveEntry entry in zip.Entries)
                {
                    i++;
                    if (i % threads != mine) continue;
                    if (entry.FullName.EndsWith("/") || entry.Name == "") continue;
                    string dest = Path.GetFullPath(Path.Combine(target, entry.FullName.Replace('/', '\\')));
                    if (!dest.StartsWith(root, StringComparison.OrdinalIgnoreCase)) continue;       // never write outside the target
                    Directory.CreateDirectory(Path.GetDirectoryName(dest));
                    try
                    {
                        if (entry.Length == 0) File.WriteAllBytes(dest, new byte[0]);              // an empty file stays empty (Python writes it as a 2-byte deflate stream, and ExtractToFile copied those 2 bytes, which broke "import cale")
                        else entry.ExtractToFile(dest, true);
                    }
                    catch (IOException) { }                                                         // a file in use is already there
                    catch (UnauthorizedAccessException) { }
                }
            }
        }
    }

    // A moving bar for "this takes a while": a carrot-orange block slides inside a brown-outlined track.
    class Slider : Control
    {
        double phase = 0;
        System.Windows.Forms.Timer t;
        public Slider()
        {
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.UserPaint | ControlStyles.ResizeRedraw, true);
            t = new System.Windows.Forms.Timer();
            t.Interval = 40;
            t.Tick += delegate { phase += 0.012; if (phase > 1) phase -= 1; Invalidate(); };
            t.Start();
        }
        static System.Drawing.Drawing2D.GraphicsPath Pill(float x, float y, float w, float h)
        {
            System.Drawing.Drawing2D.GraphicsPath p = new System.Drawing.Drawing2D.GraphicsPath();
            p.AddArc(x, y, h, h, 90, 180);
            p.AddArc(x + w - h, y, h, h, 270, 180);
            p.CloseFigure();
            return p;
        }
        protected override void OnPaint(PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            g.Clear(Color.FromArgb(255, 241, 214));
            float pen = Math.Max(2f, Height / 5f);
            float h = Height - pen, w = Width - pen, o = pen / 2f;
            using (SolidBrush track = new SolidBrush(Color.FromArgb(255, 224, 168)))
            using (System.Drawing.Drawing2D.GraphicsPath p = Pill(o, o, w, h))
            {
                g.FillPath(track, p);
                Region old = g.Clip;
                g.SetClip(p);
                float block = w * 0.3f;
                float x = o - block + (float)phase * (w + block);
                using (SolidBrush bar = new SolidBrush(Color.FromArgb(245, 138, 31)))
                using (System.Drawing.Drawing2D.GraphicsPath q = Pill(x, o, block, h))
                    g.FillPath(bar, q);
                g.Clip = old;
                using (Pen line = new Pen(Color.FromArgb(59, 35, 20), pen))
                    g.DrawPath(line, p);
            }
        }
    }

    class ExtractForm : Form
    {
        Label title;
        Label detail;
        Dictionary<string, string> cfg;
        string exe;
        long zipStart, zipLen;
        string target;
        string error = null;

        public ExtractForm(Dictionary<string, string> settings, string exePath, long start, long len)
        {
            cfg = settings;
            exe = exePath;
            zipStart = start;
            zipLen = len;
            string name = Get("name", "Setup");
            Text = name;
            AutoScaleMode = AutoScaleMode.None;
            float k;
            using (Graphics dg = CreateGraphics()) k = dg.DpiX / 96f;       // our own scaling: every size times dpi / 96
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;
            MinimizeBox = false;
            ControlBox = false;
            StartPosition = FormStartPosition.CenterScreen;
            ClientSize = new Size((int)(460 * k), (int)(290 * k));
            BackColor = Color.FromArgb(255, 241, 214);
            try { Icon = Icon.ExtractAssociatedIcon(exePath); } catch (Exception) { }
            Color brown = Color.FromArgb(59, 35, 20);

            int top = 14;
            try
            {
                using (Stream s = Assembly.GetExecutingAssembly().GetManifestResourceStream("carrot.png"))
                {
                    if (s != null)
                    {
                        PictureBox pic = new PictureBox();
                        pic.Image = Image.FromStream(s);
                        pic.SizeMode = PictureBoxSizeMode.Zoom;
                        pic.BackColor = Color.Transparent;
                        pic.SetBounds((int)(180 * k), (int)(14 * k), (int)(100 * k), (int)(100 * k));
                        Controls.Add(pic);
                        top = 120;
                    }
                }
            }
            catch (Exception) { top = 14; }
            if (top == 14) top = 40;

            title = new Label();
            title.Text = "Getting " + name + " ready";
            title.Font = new Font("Segoe UI Black", 15f, FontStyle.Regular, GraphicsUnit.Point);
            title.ForeColor = brown;
            title.TextAlign = ContentAlignment.MiddleCenter;
            title.SetBounds((int)(10 * k), (int)(top * k), (int)(440 * k), (int)(40 * k));
            Controls.Add(title);

            detail = new Label();
            detail.Text = Get("detail", "This takes a few minutes.");
            detail.Font = new Font("Segoe UI", 11f, FontStyle.Regular, GraphicsUnit.Point);
            detail.ForeColor = brown;
            detail.TextAlign = ContentAlignment.MiddleCenter;
            detail.SetBounds((int)(10 * k), (int)((top + 44) * k), (int)(440 * k), (int)(30 * k));
            Controls.Add(detail);

            Slider slide = new Slider();
            slide.SetBounds((int)(60 * k), (int)((top + 88) * k), (int)(340 * k), (int)(16 * k));
            Controls.Add(slide);
            ClientSize = new Size((int)(460 * k), (int)((top + 128) * k));
        }

        string Get(string key, string fallback)
        {
            string v;
            return cfg.TryGetValue(key, out v) ? v : fallback;
        }

        protected override void OnShown(EventArgs e)
        {
            base.OnShown(e);
            Activate();
            Thread t = new Thread(Work);
            t.IsBackground = true;
            t.Start();
        }

        void Work()
        {
            try
            {
                string home = Environment.GetEnvironmentVariable("CALE_HOME");
                string t = Get("target", "");
                if (!string.IsNullOrEmpty(home) && cfg.ContainsKey("target_calehome")) t = cfg["target_calehome"];
                target = Path.GetFullPath(Environment.ExpandEnvironmentVariables(t));
                Directory.CreateDirectory(target);
                Extractor.ExtractAll(exe, zipStart, zipLen, target, Extractor.DefaultThreads());
                string run = Environment.ExpandEnvironmentVariables(Get("run", "")).Replace("%TARGET%", target);
                string args = Environment.ExpandEnvironmentVariables(Get("args", "")).Replace("%TARGET%", target);
                if (run != "")
                {
                    ProcessStartInfo psi = new ProcessStartInfo(run, args);
                    psi.WorkingDirectory = target;
                    psi.UseShellExecute = false;
                    Process.Start(psi);
                }
            }
            catch (Exception ex)
            {
                error = ex.Message;
            }
            try { BeginInvoke((MethodInvoker)delegate { Finish(); }); } catch (Exception) { }
        }

        void Finish()
        {
            if (error != null)
            {
                MessageBox.Show("Setup could not finish. Please download it again and open it once more.\r\n\r\n" + error, Get("name", "Setup"), MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            Close();
        }
    }

    static class Program
    {
        static Dictionary<string, string> Parse(string text)
        {
            Dictionary<string, string> d = new Dictionary<string, string>();
            foreach (string raw in text.Split('\n'))
            {
                string line = raw.Trim('\r', ' ');
                int eq = line.IndexOf('=');
                if (eq > 0) d[line.Substring(0, eq).Trim()] = line.Substring(eq + 1);
            }
            return d;
        }

        // Where our data ends: the start of the signature (PE certificate table) if the file is signed, else the end of the file.
        static long DataEnd(FileStream fs)
        {
            try
            {
                BinaryReader r = new BinaryReader(fs);
                fs.Position = 0x3C;
                int pe = r.ReadInt32();
                fs.Position = pe + 24;
                ushort magic = r.ReadUInt16();
                long dirs = pe + 24 + (magic == 0x20b ? 112 : 96);
                fs.Position = dirs + 4 * 8;                            // data directory number 4: the certificate table
                uint offset = r.ReadUInt32();
                uint size = r.ReadUInt32();
                if (offset != 0 && size != 0 && offset < fs.Length) return offset;
            }
            catch (Exception) { }
            return fs.Length;
        }

        [DllImport("user32.dll")]
        static extern bool SetProcessDPIAware();

        [STAThread]
        static int Main(string[] args)
        {
            try { SetProcessDPIAware(); } catch (Exception) { }          // sharp on big screens; sizes are scaled by dpi / 96 below
            string exe = Assembly.GetExecutingAssembly().Location;
            const int footer = 20;
            Dictionary<string, string> cfg;
            long zipStart, zipLen;
            try
            {
                using (FileStream fs = new FileStream(exe, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                {
                    // Signing adds a certificate after our data. The data ends where the certificate starts.
                    long end = DataEnd(fs);
                    byte[] tail = new byte[footer + 16];
                    long tailStart = Math.Max(0, end - tail.Length);
                    fs.Position = tailStart;
                    int got = fs.Read(tail, 0, (int)Math.Min(tail.Length, end - tailStart));
                    string text = Encoding.ASCII.GetString(tail, 0, got);
                    int m = text.LastIndexOf("CALEZIP1", StringComparison.Ordinal);
                    if (m < footer - 8) throw new InvalidDataException("not packed");
                    int f = m - 12;                                    // the footer starts 12 bytes before the magic
                    int cfgLen = BitConverter.ToInt32(tail, f);
                    zipLen = BitConverter.ToInt64(tail, f + 4);
                    long cfgStart = tailStart + f - cfgLen;
                    zipStart = cfgStart - zipLen;
                    byte[] c = new byte[cfgLen];
                    fs.Position = cfgStart;
                    fs.Read(c, 0, cfgLen);
                    cfg = Parse(Encoding.UTF8.GetString(c));
                }
            }
            catch (Exception)
            {
                MessageBox.Show("This download is damaged. Please download it again.", "Setup", MessageBoxButtons.OK, MessageBoxIcon.Information);
                return 1;
            }
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new ExtractForm(cfg, exe, zipStart, zipLen));
            return 0;
        }
    }
}
