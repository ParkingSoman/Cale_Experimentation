// CaleSetup.exe: the program a person double-clicks to get Cale on Windows. It is the twin of cale/host/mac/Cale.swift.
//   - Cale is not installed yet: show "Planting Cale", run the installer, then "You are all set".
//   - Cale is installed: start it and open the chat. (No window of its own is needed for that.)
// Plain WinForms. No web view, no address, no console window.
//
// The installer is run through cale\setup_runner.py, which prints:  PCT <n> <label> | DONE | ERROR <line>
// It lives in <payload>\bin\CaleSetup.exe. The payload is the folder one level up (it holds cale\, bin\uv.exe, cale.ico).
// Written for the C# 5 compiler that comes with Windows (csc.exe, .NET Framework 4).
using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace CaleHost
{
    // Cale's colours (see cale/brand/README.md).
    static class Garden
    {
        public static readonly Color Carrot = Color.FromArgb(245, 138, 31);       // #f58a1f  buttons, progress bar
        public static readonly Color Brown = Color.FromArgb(59, 35, 20);          // #3b2314  text and outlines
        public static readonly Color Cream = Color.FromArgb(255, 241, 214);       // #fff1d6  window background
        public static readonly Color Soil = Color.FromArgb(138, 90, 59);          // #8a5a3b  the ground
        public static readonly Color Track = Color.FromArgb(255, 224, 168);       // the empty part of the bar
    }

    // The carrot rises out of the soil as Growth goes from 0 to 1. The soil is drawn over its lower part.
    class GardenView : Control
    {
        double growth = 0;
        Image carrot;

        public GardenView()
        {
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.UserPaint | ControlStyles.ResizeRedraw, true);
            try
            {
                using (Stream s = System.Reflection.Assembly.GetExecutingAssembly().GetManifestResourceStream("carrot.png"))
                {
                    if (s != null) carrot = Image.FromStream(s);
                }
            }
            catch (Exception) { }
        }

        public double Growth { get { return growth; } set { growth = value; Invalidate(); } }
        public bool Ripe { get { return growth >= 1; } set { } }

        protected override void OnPaint(PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.Clear(Garden.Cream);
            float k = Height / 200f;                                  // drawn for a 440 x 200 box; scale to the real size
            g.ScaleTransform(k, k);
            float w = Width / k, h = 200f;
            float soilTop = h - 34f;
            float size = 176f;                                        // the carrot picture is square
            float gr = (float)Math.Max(0.0, Math.Min(1.0, growth));
            if (carrot != null)
            {
                float y = soilTop - size * (0.02f + 0.98f * gr);      // a hint of leaf shows at the start
                Region old = g.Clip;
                g.SetClip(new RectangleF(0, 0, w, soilTop + 4));
                g.DrawImage(carrot, (w - size) / 2f, y, size, size);
                g.Clip = old;
            }
            using (SolidBrush soil = new SolidBrush(Garden.Soil))
            using (Pen line = new Pen(Garden.Brown, 3f))
            {
                RectangleF r = new RectangleF(w * 0.1f, soilTop, w * 0.8f, 40f);
                using (GraphicsPath p = new GraphicsPath())
                {
                    float d = 24f;
                    p.AddArc(r.X, r.Y, d, d, 180, 90);
                    p.AddArc(r.Right - d, r.Y, d, d, 270, 90);
                    p.AddArc(r.Right - d, r.Bottom - d, d, d, 0, 90);
                    p.AddArc(r.X, r.Bottom - d, d, d, 90, 90);
                    p.CloseFigure();
                    g.FillPath(soil, p);
                    g.DrawPath(line, p);
                }
            }
        }
    }

    // A progress bar: carrot orange with a brown outline.
    class Meter : Control
    {
        double value = 0;

        public Meter()
        {
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.UserPaint | ControlStyles.ResizeRedraw, true);
        }

        public double Value { get { return value; } set { this.value = value; Invalidate(); } }

        static GraphicsPath Pill(float x, float y, float w, float h)
        {
            GraphicsPath p = new GraphicsPath();
            p.AddArc(x, y, h, h, 90, 180);
            p.AddArc(x + w - h, y, h, h, 270, 180);
            p.CloseFigure();
            return p;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.Clear(Garden.Cream);
            float pen = Math.Max(2f, Height / 5f);
            float h = Height - pen, w = Width - pen, o = pen / 2f;
            float fill = Math.Max(h, w * (float)Math.Min(1.0, value / 100.0));
            using (SolidBrush track = new SolidBrush(Garden.Track))
            using (GraphicsPath p = Pill(o, o, w, h))
                g.FillPath(track, p);
            using (SolidBrush bar = new SolidBrush(Garden.Carrot))
            using (GraphicsPath p = Pill(o, o, fill, h))
                g.FillPath(bar, p);
            using (Pen line = new Pen(Garden.Brown, pen))
            using (GraphicsPath p = Pill(o, o, w, h))
                g.DrawPath(line, p);
        }
    }

    class SetupForm : Form
    {
        Label title, detail;
        Meter bar;
        GardenView garden;
        Button button;
        Process runner;
        bool finished = false;
        bool closing = false;
        string home, payload;
        float k = 1f;

        void Place(Control c, int x, int y, int w, int h)
        {
            c.SetBounds((int)(x * k), (int)(y * k), (int)(w * k), (int)(h * k));
        }

        public SetupForm(string homeDir, string payloadDir)
        {
            home = homeDir;
            payload = payloadDir;
            Text = "Cale";
            AutoScaleMode = AutoScaleMode.None;
            using (Graphics dg = CreateGraphics()) k = dg.DpiX / 96f;       // our own scaling: one number for every size
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;
            StartPosition = FormStartPosition.CenterScreen;
            ClientSize = new Size((int)(520 * k), (int)(420 * k));
            BackColor = Garden.Cream;
            string ico = Path.Combine(payload, "cale.ico");
            if (File.Exists(ico)) { try { Icon = new Icon(ico); } catch (Exception) { } }

            garden = new GardenView();
            Place(garden, 40, 30, 440, 200);
            Controls.Add(garden);

            title = new Label();
            title.Text = "Planting Cale";
            title.Font = new Font("Segoe UI Black", 17f, FontStyle.Regular, GraphicsUnit.Point);
            title.ForeColor = Garden.Brown;
            title.TextAlign = ContentAlignment.MiddleCenter;
            Place(title, 20, 246, 480, 40);
            Controls.Add(title);

            detail = new Label();
            detail.Text = "This takes a few minutes. You can keep using your computer.";
            detail.Font = new Font("Segoe UI", 10.5f, FontStyle.Regular, GraphicsUnit.Point);
            detail.ForeColor = Garden.Brown;
            detail.TextAlign = ContentAlignment.MiddleCenter;
            Place(detail, 30, 292, 460, 28);
            Controls.Add(detail);

            bar = new Meter();
            Place(bar, 60, 336, 400, 16);
            Controls.Add(bar);

            button = new Button();
            button.Text = "Start chatting";
            button.Font = new Font("Segoe UI Semibold", 10.5f, FontStyle.Regular, GraphicsUnit.Point);
            button.FlatStyle = FlatStyle.Flat;
            button.FlatAppearance.BorderSize = 3;
            button.FlatAppearance.BorderColor = Garden.Brown;
            button.FlatAppearance.MouseOverBackColor = Color.FromArgb(255, 160, 64);
            button.BackColor = Garden.Carrot;
            button.ForeColor = Garden.Brown;
            button.Cursor = Cursors.Hand;
            Place(button, 175, 336, 170, 42);
            button.Visible = false;
            button.Click += delegate { StartChat(home); Close(); };
            Controls.Add(button);
            AcceptButton = button;
        }

        protected override void OnShown(EventArgs e)
        {
            base.OnShown(e);
            TopMost = true;                // come to the front, then behave like a normal window
            TopMost = false;
            Activate();
            RunSetup();
        }

        void RunSetup()
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo(Path.Combine(payload, "bin", "uv.exe"));
                psi.Arguments = "run --python 3.12 --no-project python cale\\setup_runner.py";
                psi.WorkingDirectory = payload;
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;
                psi.RedirectStandardOutput = true;
                psi.RedirectStandardError = true;
                psi.StandardOutputEncoding = Encoding.UTF8;
                psi.StandardErrorEncoding = Encoding.UTF8;
                psi.EnvironmentVariables["CALE_HOME"] = home;
                psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
                runner = new Process();
                runner.StartInfo = psi;
                runner.EnableRaisingEvents = true;
                runner.OutputDataReceived += OnLine;
                runner.ErrorDataReceived += OnLine;
                runner.Exited += delegate
                {
                    if (closing) return;
                    int code = runner.ExitCode;
                    try
                    {
                        BeginInvoke((MethodInvoker)delegate
                        {
                            if (!finished && code != 0) Fail("Cale could not finish. Open it again to try once more.");
                        });
                    }
                    catch (Exception) { }
                };
                runner.Start();
                runner.BeginOutputReadLine();
                runner.BeginErrorReadLine();
            }
            catch (Exception)
            {
                Fail("Cale could not start its setup.");
            }
        }

        void OnLine(object sender, DataReceivedEventArgs e)
        {
            if (e.Data == null) return;
            string line = e.Data.TrimEnd();
            try { BeginInvoke((MethodInvoker)delegate { HandleLine(line); }); } catch (Exception) { }
        }

        void HandleLine(string line)
        {
            if (line.StartsWith("PCT "))
            {
                string[] parts = line.Substring(4).Split(new char[] { ' ' }, 2);
                double pct;
                if (double.TryParse(parts[0], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out pct))
                {
                    bar.Value = pct;
                    garden.Growth = pct / 100.0;
                    if (parts.Length > 1) title.Text = parts[1];
                }
            }
            else if (line == "DONE")
            {
                finished = true;
                bar.Value = 100;
                garden.Growth = 1;
                title.Text = "You are all set";
                detail.Text = "Cale is ready on this computer. Apps made for Cale will just work.";
                bar.Visible = false;
                button.Visible = true;
                button.Focus();
                MakeShortcut(payload);
            }
            else if (line.StartsWith("ERROR "))
            {
                Fail("Cale could not finish. Open it again to try once more.");
            }
        }

        void Fail(string text)
        {
            finished = true;
            title.Text = "Setup stopped";
            detail.Text = text;
            bar.Visible = false;
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            closing = true;
            try
            {
                if (runner != null && !finished && !runner.HasExited)
                {
                    ProcessStartInfo k = new ProcessStartInfo("taskkill.exe", "/PID " + runner.Id + " /T /F");
                    k.CreateNoWindow = true;
                    k.UseShellExecute = false;
                    Process kp = Process.Start(k);
                    if (kp != null) kp.WaitForExit(3000);
                }
            }
            catch (Exception) { }
            base.OnFormClosing(e);
        }

        // ---- shared with Program ----

        public static void StartChat(string home)
        {
            try
            {
                string py = Path.Combine(home, "venv", "Scripts", "pythonw.exe");
                ProcessStartInfo psi = new ProcessStartInfo(py, "-m cale.chat");
                psi.WorkingDirectory = Path.Combine(home, "runtime");
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;
                psi.EnvironmentVariables["CALE_HOME"] = home;
                psi.EnvironmentVariables["PYTHONPATH"] = Path.Combine(home, "runtime");
                psi.EnvironmentVariables["PATH"] = Path.Combine(home, "bin") + ";" + Environment.GetEnvironmentVariable("PATH");
                Process.Start(psi);
            }
            catch (Exception) { }
        }

        // A Start Menu entry with Cale's own icon, so Cale can be opened again.
        public static void MakeShortcut(string payload)
        {
            try
            {
                string lnk = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Programs), "Cale.lnk");
                Type t = Type.GetTypeFromProgID("WScript.Shell");
                dynamic sh = Activator.CreateInstance(t);
                dynamic s = sh.CreateShortcut(lnk);
                s.TargetPath = Path.Combine(payload, "bin", "CaleSetup.exe");
                s.WorkingDirectory = Path.Combine(payload, "bin");
                string ico = Path.Combine(payload, "cale.ico");
                if (File.Exists(ico)) s.IconLocation = ico + ",0";
                s.Description = "Cale";
                s.Save();
            }
            catch (Exception) { }
        }
    }

    static class Program
    {
        [DllImport("user32.dll")]
        static extern bool SetProcessDPIAware();
        [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
        static extern int SetCurrentProcessExplicitAppUserModelID(string id);

        static bool Installed(string home)
        {
            return File.Exists(Path.Combine(home, "installed.ok")) && File.Exists(Path.Combine(home, "venv", "Scripts", "python.exe"));
        }

        // This download's code replaces the installed code. Quiet. A failure here never stops Cale from opening.
        static void Refresh(string home, string payload)
        {
            try
            {
                string installer = Path.Combine(payload, "cale", "install.py");
                string py = Path.Combine(home, "venv", "Scripts", "python.exe");
                if (!File.Exists(installer) || !File.Exists(py)) return;
                ProcessStartInfo psi = new ProcessStartInfo(py, "\"" + installer + "\" --update");
                psi.WorkingDirectory = payload;
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;
                psi.RedirectStandardOutput = true;
                psi.RedirectStandardError = true;
                psi.EnvironmentVariables["CALE_HOME"] = home;
                psi.EnvironmentVariables["PATH"] = Path.Combine(home, "bin") + ";" + Environment.GetEnvironmentVariable("PATH");
                using (Process p = Process.Start(psi))
                {
                    p.OutputDataReceived += delegate { };
                    p.ErrorDataReceived += delegate { };
                    p.BeginOutputReadLine();
                    p.BeginErrorReadLine();
                    if (!p.WaitForExit(90000)) { try { p.Kill(); } catch (Exception) { } }
                }
            }
            catch (Exception) { }
        }

        [STAThread]
        static int Main(string[] args)
        {
            try { SetProcessDPIAware(); } catch (Exception) { }
            try { SetCurrentProcessExplicitAppUserModelID("Cale.Setup"); } catch (Exception) { }
            string exeDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd('\\');
            string payload = Path.GetDirectoryName(exeDir);
            string home = Environment.GetEnvironmentVariable("CALE_HOME");
            if (string.IsNullOrEmpty(home))
                home = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Cale");

            if (Installed(home))
            {
                Refresh(home, payload);
                SetupForm.StartChat(home);
                SetupForm.MakeShortcut(payload);
                return 0;
            }
            bool created;
            using (Mutex one = new Mutex(true, "Local\\CaleSetupWindow", out created))
            {
                if (!created) return 0;                         // the setup window is already open
                Application.EnableVisualStyles();
                Application.SetCompatibleTextRenderingDefault(false);
                Application.Run(new SetupForm(home, payload));
            }
            return 0;
        }
    }
}
