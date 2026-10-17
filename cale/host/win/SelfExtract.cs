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
            using (Graphics dg = CreateGraphics()) k = dg.DpiX / 96f;
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;
            MinimizeBox = false;
            ControlBox = false;
            StartPosition = FormStartPosition.CenterScreen;
            ClientSize = new Size((int)(380 * k), (int)(110 * k));
            BackColor = Color.FromArgb(255, 241, 214);
            try { Icon = Icon.ExtractAssociatedIcon(exePath); } catch (Exception) { }

            title = new Label();
            title.Text = "Getting " + name + " ready";
            title.Font = new Font("Segoe UI Semibold", 13f, FontStyle.Regular, GraphicsUnit.Point);
            title.ForeColor = Color.FromArgb(59, 35, 20);
            title.TextAlign = ContentAlignment.MiddleCenter;
            title.SetBounds((int)(10 * k), (int)(22 * k), (int)(360 * k), (int)(32 * k));
            Controls.Add(title);

            detail = new Label();
            detail.Text = "One moment.";
            detail.Font = new Font("Segoe UI", 10f, FontStyle.Regular, GraphicsUnit.Point);
            detail.ForeColor = Color.FromArgb(59, 35, 20);
            detail.TextAlign = ContentAlignment.MiddleCenter;
            detail.SetBounds((int)(10 * k), (int)(58 * k), (int)(360 * k), (int)(26 * k));
            Controls.Add(detail);
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
                using (FileStream fs = new FileStream(exe, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                using (Slice slice = new Slice(fs, zipStart, zipLen))
                using (ZipArchive zip = new ZipArchive(slice, ZipArchiveMode.Read))
                {
                    string root = target.TrimEnd('\\') + "\\";
                    foreach (ZipArchiveEntry entry in zip.Entries)
                    {
                        if (entry.FullName.EndsWith("/") || entry.Name == "") continue;
                        string dest = Path.GetFullPath(Path.Combine(target, entry.FullName.Replace('/', '\\')));
                        if (!dest.StartsWith(root, StringComparison.OrdinalIgnoreCase)) continue;       // never write outside the target
                        Directory.CreateDirectory(Path.GetDirectoryName(dest));
                        try { entry.ExtractToFile(dest, true); }
                        catch (IOException) { }                                                         // a file in use is already there
                        catch (UnauthorizedAccessException) { }
                    }
                }
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

        [STAThread]
        static int Main(string[] args)
        {
            string exe = Assembly.GetExecutingAssembly().Location;
            const int footer = 20;
            Dictionary<string, string> cfg;
            long zipStart, zipLen;
            try
            {
                using (FileStream fs = new FileStream(exe, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                {
                    byte[] f = new byte[footer];
                    fs.Position = fs.Length - footer;
                    fs.Read(f, 0, footer);
                    if (Encoding.ASCII.GetString(f, 12, 8) != "CALEZIP1") throw new InvalidDataException("not packed");
                    int cfgLen = BitConverter.ToInt32(f, 0);
                    zipLen = BitConverter.ToInt64(f, 4);
                    long cfgStart = fs.Length - footer - cfgLen;
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
