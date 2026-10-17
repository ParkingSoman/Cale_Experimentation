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
    static class Garden
    {
        public static readonly Color Kale = Color.FromArgb(47, 93, 80);           // #2f5d50
        public static readonly Color Leaf = Color.FromArgb(90, 148, 102);
        public static readonly Color Cream = Color.FromArgb(247, 245, 238);      // #f7f5ee
        public static readonly Color Soil = Color.FromArgb(110, 79, 56);
        public static readonly Color Carrot = Color.FromArgb(232, 130, 43);
    }

    // A small garden: soil, a stem and leaves that grow with Growth (0 to 1). When Ripe, a carrot shows beside it.
    class GardenView : Control
    {
        double growth = 0;
        bool ripe = false;

        public GardenView()
        {
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.UserPaint | ControlStyles.ResizeRedraw, true);
        }

        public double Growth { get { return growth; } set { growth = value; Invalidate(); } }
        public bool Ripe { get { return ripe; } set { ripe = value; Invalidate(); } }

        static GraphicsPath Rounded(float x, float y, float w, float h, float r)
        {
            GraphicsPath p = new GraphicsPath();
            p.AddArc(x, y, 2 * r, 2 * r, 180, 90);
            p.AddArc(x + w - 2 * r, y, 2 * r, 2 * r, 270, 90);
            p.AddArc(x + w - 2 * r, y + h - 2 * r, 2 * r, 2 * r, 0, 90);
            p.AddArc(x, y + h - 2 * r, 2 * r, 2 * r, 90, 90);
            p.CloseFigure();
            return p;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.Clear(Garden.Cream);
            float w = Width, h = Height;
            // Drawing is done in "up is bigger" numbers like the Mac version. Up(v) turns one into a screen position.
            Func<float, float> up = delegate(float v) { return h - v; };

            using (SolidBrush soil = new SolidBrush(Garden.Soil))
            using (GraphicsPath sp = Rounded(w * 0.12f, up(h * 0.14f), w * 0.76f, h * 0.14f, 10))
                g.FillPath(soil, sp);

            float baseX = w * (ripe ? 0.38f : 0.5f), baseY = h * 0.13f;
            float top = baseY + (h * 0.62f) * (float)growth;
            using (Pen stem = new Pen(Garden.Leaf, 5))
            {
                stem.StartCap = LineCap.Round;
                stem.EndCap = LineCap.Round;
                g.DrawLine(stem, baseX, up(baseY), baseX, up(top));
            }
            // Kale leaves: pairs at fixed spacing along the stem. Each pair unfurls once the stem reaches it.
            float spacing = (h * 0.62f - 14) / 6;
            using (SolidBrush light = new SolidBrush(Garden.Leaf))
            using (SolidBrush dark = new SolidBrush(Garden.Kale))
            {
                for (int i = 0; i < 6; i++)
                {
                    float y = baseY + 14 + i * spacing;
                    if (top < y) continue;
                    float unfurl = Math.Min(1f, (top - y) / 30f);
                    float size = (30 - 3 * i) * unfurl;
                    for (int k = 0; k < 2; k++)
                    {
                        float side = k == 0 ? -1f : 1f;
                        using (GraphicsPath p = new GraphicsPath())
                        {
                            p.AddBezier(baseX, up(y),
                                baseX + side * size * 0.4f, up(y + size * 0.9f),
                                baseX + side * size * 1.3f, up(y + size * 1.0f),
                                baseX + side * size * 1.6f, up(y + size * 0.5f));
                            p.AddBezier(baseX + side * size * 1.6f, up(y + size * 0.5f),
                                baseX + side * size * 1.2f, up(y - size * 0.3f),
                                baseX + side * size * 0.4f, up(y - size * 0.2f),
                                baseX, up(y));
                            p.CloseFigure();
                            g.FillPath(i % 2 == 0 ? light : dark, p);
                        }
                    }
                }
            }
            if (ripe)
            {
                float cx = w * 0.68f;
                using (SolidBrush carrot = new SolidBrush(Garden.Carrot))
                    g.FillPolygon(carrot, new PointF[] { new PointF(cx - 16, up(baseY + 70)), new PointF(cx + 16, up(baseY + 70)), new PointF(cx, up(baseY - 2)) });
                using (SolidBrush leaf = new SolidBrush(Garden.Leaf))
                {
                    float[] dxs = new float[] { -10f, 0f, 10f };
                    foreach (float dx in dxs)
                        g.FillPolygon(leaf, new PointF[] { new PointF(cx + dx * 0.4f, up(baseY + 70)), new PointF(cx + dx * 1.6f, up(baseY + 100)), new PointF(cx + dx * 0.4f + 5, up(baseY + 70)) });
                }
            }
        }
    }

    // A progress bar in the garden colours.
    class Meter : Control
    {
        double value = 0;

        public Meter()
        {
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.UserPaint | ControlStyles.ResizeRedraw, true);
        }

        public double Value { get { return value; } set { this.value = value; Invalidate(); } }

        static GraphicsPath Pill(float w, float h)
        {
            GraphicsPath p = new GraphicsPath();
            p.AddArc(0, 0, h, h, 90, 180);
            p.AddArc(w - h, 0, h, h, 270, 180);
            p.CloseFigure();
            return p;
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.Clear(Garden.Cream);
            float h = Height - 1, w = Width - 1;
            using (SolidBrush track = new SolidBrush(Color.FromArgb(222, 219, 204)))
            using (GraphicsPath p = Pill(w, h))
                g.FillPath(track, p);
            float fill = Math.Max(h, w * (float)Math.Min(1.0, value / 100.0));
            using (SolidBrush bar = new SolidBrush(Garden.Leaf))
            using (GraphicsPath p = Pill(fill, h))
                g.FillPath(bar, p);
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

        public SetupForm(string homeDir, string payloadDir)
        {
            home = homeDir;
            payload = payloadDir;
            Text = "Cale";
            AutoScaleDimensions = new SizeF(96f, 96f);
            AutoScaleMode = AutoScaleMode.Dpi;
            FormBorderStyle = FormBorderStyle.FixedSingle;
            MaximizeBox = false;
            StartPosition = FormStartPosition.CenterScreen;
            ClientSize = new Size(520, 420);
            BackColor = Garden.Cream;
            string ico = Path.Combine(payload, "cale.ico");
            if (File.Exists(ico)) { try { Icon = new Icon(ico); } catch (Exception) { } }

            garden = new GardenView();
            garden.SetBounds(40, 30, 440, 200);
            Controls.Add(garden);

            title = new Label();
            title.Text = "Planting Cale";
            title.Font = new Font("Segoe UI Semibold", 18f, FontStyle.Regular, GraphicsUnit.Point);
            title.ForeColor = Garden.Kale;
            title.TextAlign = ContentAlignment.MiddleCenter;
            title.SetBounds(20, 246, 480, 40);
            Controls.Add(title);

            detail = new Label();
            detail.Text = "This takes a few minutes. You can keep using your computer.";
            detail.Font = new Font("Segoe UI", 10.5f, FontStyle.Regular, GraphicsUnit.Point);
            detail.ForeColor = Color.FromArgb(89, 89, 89);
            detail.TextAlign = ContentAlignment.MiddleCenter;
            detail.SetBounds(30, 292, 460, 28);
            Controls.Add(detail);

            bar = new Meter();
            bar.SetBounds(60, 338, 400, 12);
            Controls.Add(bar);

            button = new Button();
            button.Text = "Start chatting";
            button.Font = new Font("Segoe UI Semibold", 10.5f, FontStyle.Regular, GraphicsUnit.Point);
            button.FlatStyle = FlatStyle.Flat;
            button.FlatAppearance.BorderSize = 0;
            button.BackColor = Garden.Kale;
            button.ForeColor = Color.White;
            button.Cursor = Cursors.Hand;
            button.SetBounds(175, 336, 170, 42);
            button.Visible = false;
            button.Click += delegate { StartChat(home); Close(); };
            Controls.Add(button);
            AcceptButton = button;
        }

        protected override void OnShown(EventArgs e)
        {
            base.OnShown(e);
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
                garden.Ripe = true;
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
