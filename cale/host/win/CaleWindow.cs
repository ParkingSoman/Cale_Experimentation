// CaleWindow.exe: shows one page in a native Windows window. No address bar, no tabs, no console.
// Cale installs this once. Cale's chat and downloaded apps use it to look like programs.
// It is the Windows twin of cale/host/mac/CaleWindow.swift and takes the same command line:
//
//   CaleWindow.exe --url http://127.0.0.1:51234/ --title "Quick Notes" [--width 900 --height 640] [--min-width 480 --min-height 360]
//
// An app can let this program start its server. The server picks a free port and prints  READY <port> [token]  on one line (the token is optional; {token} in --url-template):
//   CaleWindow.exe --title "Quick Notes" --icon app.ico --server-exec C:\path\pythonw.exe --server-arg app.py
//                  --server-cwd C:\path\app --url-template http://127.0.0.1:{port}/
// The server stops when the window closes (the whole process tree). If the server stops, the window closes.
// The program also exits when --parent-pid ends.
//
// Extra on Windows:
//   --icon PATH      a .ico file: the window and taskbar icon.
//   --relaunch CMD   the command the taskbar uses for "pin to taskbar" (optional).
//   --splash PNG [--splash-text TEXT]   a loading screen (cream, picture, one line of text) until the first page has loaded.
// Each title gets its own AppUserModelID, so the taskbar shows the app's own name and icon.
//
// Written for the C# 5 compiler that comes with Windows (csc.exe, .NET Framework 4).
// The two WebView2 wrapper DLLs are built into this exe as resources. WebView2Loader.dll must sit beside the exe.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace CaleHost
{
    class Options
    {
        public string Url = "";
        public string Title = "Cale";
        public int Width = 980;
        public int Height = 700;
        public int MinWidth = 480;
        public int MinHeight = 360;
        public int ParentPid = 0;
        public string ServerExec = "";
        public List<string> ServerArgs = new List<string>();
        public string ServerCwd = "";
        public string UrlTemplate = "http://127.0.0.1:{port}/";
        public string Icon = "";
        public string Relaunch = "";
        public string Splash = "";
        public string SplashText = "";

        static int Num(string s, int fallback)
        {
            int v;
            double d;
            if (int.TryParse(s, out v)) return v;
            if (double.TryParse(s, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out d)) return (int)d;
            return fallback;
        }

        public static Options Parse(string[] args)
        {
            Options o = new Options();
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                string next = (i + 1 < args.Length) ? args[i + 1] : "";
                switch (a)
                {
                    case "--url": o.Url = next; i++; break;
                    case "--title": if (next != "") o.Title = next; i++; break;
                    case "--width": o.Width = Num(next, o.Width); i++; break;
                    case "--height": o.Height = Num(next, o.Height); i++; break;
                    case "--min-width": o.MinWidth = Num(next, o.MinWidth); i++; break;
                    case "--min-height": o.MinHeight = Num(next, o.MinHeight); i++; break;
                    case "--parent-pid": o.ParentPid = Num(next, 0); i++; break;
                    case "--server-exec": o.ServerExec = next; i++; break;
                    case "--server-arg": o.ServerArgs.Add(next); i++; break;
                    case "--server-cwd": o.ServerCwd = next; i++; break;
                    case "--url-template": if (next != "") o.UrlTemplate = next; i++; break;
                    case "--icon": o.Icon = next; i++; break;
                    case "--relaunch": o.Relaunch = next; i++; break;
                    case "--splash": o.Splash = next; i++; break;
                    case "--splash-text": o.SplashText = next; i++; break;
                }
            }
            return o;
        }
    }

    // A job object makes Windows stop the server (and anything it started) when this program ends, even if it is killed.
    static class KillWithMe
    {
        [StructLayout(LayoutKind.Sequential)]
        struct BasicLimit
        {
            public long PerProcessUserTimeLimit;
            public long PerJobUserTimeLimit;
            public uint LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public uint ActiveProcessLimit;
            public UIntPtr Affinity;
            public uint PriorityClass;
            public uint SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct IoCounters
        {
            public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount, ReadTransferCount, WriteTransferCount, OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        struct ExtendedLimit
        {
            public BasicLimit Basic;
            public IoCounters Io;
            public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        static extern IntPtr CreateJobObject(IntPtr attrs, string name);
        [DllImport("kernel32.dll")]
        static extern bool SetInformationJobObject(IntPtr job, int infoClass, IntPtr info, uint size);
        [DllImport("kernel32.dll")]
        static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

        static IntPtr job = IntPtr.Zero;

        public static void Add(Process p)
        {
            try
            {
                if (job == IntPtr.Zero)
                {
                    job = CreateJobObject(IntPtr.Zero, null);
                    ExtendedLimit lim = new ExtendedLimit();
                    lim.Basic.LimitFlags = 0x2000;                       // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                    int size = Marshal.SizeOf(typeof(ExtendedLimit));
                    IntPtr mem = Marshal.AllocHGlobal(size);
                    Marshal.StructureToPtr(lim, mem, false);
                    SetInformationJobObject(job, 9, mem, (uint)size);     // JobObjectExtendedLimitInformation
                    Marshal.FreeHGlobal(mem);
                }
                AssignProcessToJobObject(job, p.Handle);
            }
            catch (Exception) { }
        }
    }

    // The taskbar name and icon: a per-app AppUserModelID, plus relaunch details on the window.
    static class Shell
    {
        [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
        static extern int SetCurrentProcessExplicitAppUserModelID(string id);
        [DllImport("shell32.dll")]
        static extern int SHGetPropertyStoreForWindow(IntPtr hwnd, ref Guid iid, out IPropertyStore store);
        [DllImport("propsys.dll", CharSet = CharSet.Unicode)]
        static extern int InitPropVariantFromString(string s, out PropVariant pv);
        [DllImport("ole32.dll")]
        static extern int PropVariantClear(ref PropVariant pv);
        [DllImport("user32.dll")]
        static extern bool SetProcessDPIAware();
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)]
        static extern IntPtr LoadLibrary(string path);

        [StructLayout(LayoutKind.Sequential)]
        struct PropVariant { public ushort vt, r1, r2, r3; public IntPtr p1, p2; }

        [StructLayout(LayoutKind.Sequential, Pack = 4)]
        struct PropKey { public Guid fmtid; public uint pid; }

        [ComImport, Guid("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IPropertyStore
        {
            int GetCount(out uint count);
            int GetAt(uint i, out PropKey key);
            int GetValue(ref PropKey key, out PropVariant pv);
            int SetValue(ref PropKey key, ref PropVariant pv);
            int Commit();
        }

        static readonly Guid AppUserModel = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3");

        public static void Prepare()
        {
            try { SetProcessDPIAware(); } catch (Exception) { }
            try
            {
                string loader = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "WebView2Loader.dll");
                if (File.Exists(loader)) LoadLibrary(loader);             // so the wrapper finds it even when it was loaded from a resource
            }
            catch (Exception) { }
        }

        public static string IdFor(string title)
        {
            StringBuilder b = new StringBuilder("Cale.App.");
            foreach (char c in title) b.Append(char.IsLetterOrDigit(c) ? c : '_');
            return b.ToString();
        }

        public static void SetProcessId(string id)
        {
            try { SetCurrentProcessExplicitAppUserModelID(id); } catch (Exception) { }
        }

        static void Put(IPropertyStore store, uint pid, string value)
        {
            PropKey key = new PropKey();
            key.fmtid = AppUserModel;
            key.pid = pid;
            PropVariant pv;
            if (InitPropVariantFromString(value, out pv) != 0) return;
            store.SetValue(ref key, ref pv);
            PropVariantClear(ref pv);
        }

        // Tell the taskbar this window's id, name, icon and how to start it again.
        public static void Describe(IntPtr hwnd, string id, string name, string iconPath, string relaunch)
        {
            try
            {
                Guid iid = new Guid("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99");
                IPropertyStore store;
                if (SHGetPropertyStoreForWindow(hwnd, ref iid, out store) != 0 || store == null) return;
                Put(store, 5, id);
                if (!string.IsNullOrEmpty(relaunch))
                {
                    Put(store, 2, relaunch);
                    Put(store, 4, name);
                    if (!string.IsNullOrEmpty(iconPath)) Put(store, 3, iconPath + ",0");
                }
                store.Commit();
            }
            catch (Exception) { }
        }
    }

    // The loading screen: cream, the mascot picture and one line of text. Shown until the first page has loaded.
    class Splash : Control
    {
        Image img;
        string text;
        float k;

        public Splash(string picture, string line, float scale)
        {
            text = line;
            k = scale;
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.OptimizedDoubleBuffer | ControlStyles.UserPaint | ControlStyles.ResizeRedraw, true);
            Dock = DockStyle.Fill;
            try
            {
                if (!string.IsNullOrEmpty(picture) && File.Exists(picture))
                {
                    using (MemoryStream m = new MemoryStream(File.ReadAllBytes(picture))) img = Image.FromStream(m);
                }
            }
            catch (Exception) { img = null; }
        }

        protected override void OnPaint(PaintEventArgs e)
        {
            Graphics g = e.Graphics;
            g.Clear(Color.FromArgb(255, 241, 214));
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
            g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;
            float size = 150f * k, gap = 18f * k;
            float textH = 0f;
            using (Font f = new Font("Segoe UI Semibold", 17f, FontStyle.Bold, GraphicsUnit.Point))
            {
                if (!string.IsNullOrEmpty(text)) textH = g.MeasureString(text, f).Height;
                float total = (img != null ? size : 0f) + ((img != null && textH > 0) ? gap : 0f) + textH;
                float y = (Height - total) / 2f;
                if (img != null)
                {
                    g.DrawImage(img, (Width - size) / 2f, y, size, size);
                    y += size + gap;
                }
                if (textH > 0)
                {
                    using (SolidBrush brown = new SolidBrush(Color.FromArgb(59, 35, 20)))
                    using (StringFormat sf = new StringFormat())
                    {
                        sf.Alignment = StringAlignment.Center;
                        g.DrawString(text, f, brown, new RectangleF(0, y, Width, textH + 4), sf);
                    }
                }
            }
        }
    }

    class MainForm : Form
    {
        Options opts;
        WebView2 web;
        Process server;
        string url = "";
        string pending = "";
        string token = "";
        string tokenUrl = "";            // the first address (it carries a one-time secret) once it has been used
        string cleanUrl = "";
        bool tokenSpent = false;
        bool ready = false;
        bool webFailed = false;
        bool closing = false;
        int retries = 0;
        System.Windows.Forms.Timer retryTimer;
        Splash splash;
        System.Windows.Forms.Timer splashTimer;
        System.Windows.Forms.Timer drawnTimer;      // polls until the page has put some text on screen
        bool polling = false;
        bool firstNavStarted = false;

        public MainForm(Options o)
        {
            opts = o;
            Text = o.Title;
            AutoScaleMode = AutoScaleMode.None;
            float k;
            using (Graphics dg = CreateGraphics()) k = dg.DpiX / 96f;        // the sizes above are in 96-dpi units: scale them, or the page gets a narrow, clipped view on a high-resolution screen
            ClientSize = new Size((int)(o.Width * k), (int)(o.Height * k));
            MinimumSize = SizeFromClientSize(new Size((int)(o.MinWidth * k), (int)(o.MinHeight * k)));
            Rectangle wa = Screen.PrimaryScreen.WorkingArea;                  // never taller or wider than 90% of the usable screen (a small laptop screen)
            Size = new Size(Math.Min(Width, (int)(wa.Width * 0.9)), Math.Min(Height, (int)(wa.Height * 0.9)));
            StartPosition = FormStartPosition.Manual;                          // centered by hand: CreateGraphics above already made the window, so CenterScreen would use the old size
            Location = new Point(wa.Left + (wa.Width - Width) / 2, wa.Top + (wa.Height - Height) / 2);
            BackColor = Color.FromArgb(255, 241, 214);
            if (!string.IsNullOrEmpty(o.Icon) && File.Exists(o.Icon))
            {
                try { Icon = new Icon(o.Icon); } catch (Exception) { }
            }
            web = new WebView2();
            web.Dock = DockStyle.Fill;
            web.DefaultBackgroundColor = Color.FromArgb(255, 241, 214);
            Controls.Add(web);
            if (!string.IsNullOrEmpty(o.Splash))
            {
                float dpiScale;
                using (Graphics dg = CreateGraphics()) dpiScale = dg.DpiX / 96f;
                splash = new Splash(o.Splash, o.SplashText, dpiScale);
                Controls.Add(splash);
                splash.BringToFront();
                splashTimer = new System.Windows.Forms.Timer();
                splashTimer.Interval = 180000;                     // if no page address ever comes; it is set to 60 s when the first page load starts
                splashTimer.Tick += delegate { HideSplash(); };
                splashTimer.Start();
            }
            retryTimer = new System.Windows.Forms.Timer();
            retryTimer.Interval = 500;
            retryTimer.Tick += delegate { retryTimer.Stop(); Load(); };
        }

        protected override void OnHandleCreated(EventArgs e)
        {
            base.OnHandleCreated(e);
            Shell.Describe(Handle, Shell.IdFor(opts.Title), opts.Title, opts.Icon, opts.Relaunch);
        }

        protected override void OnShown(EventArgs e)
        {
            base.OnShown(e);
            TopMost = true;                // come to the front, then behave like a normal window
            TopMost = false;
            Activate();
            StartUp();
            if (opts.ParentPid > 0)
            {
                System.Windows.Forms.Timer t = new System.Windows.Forms.Timer();
                t.Interval = 2000;
                t.Tick += delegate
                {
                    try { Process.GetProcessById(opts.ParentPid); }
                    catch (Exception) { Close(); }
                };
                t.Start();
            }
        }

        async void StartUp()
        {
            try
            {
                string data = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "CaleWindow", Shell.IdFor(opts.Title));
                Directory.CreateDirectory(data);
                CoreWebView2Environment env = await CoreWebView2Environment.CreateAsync(null, data, null);
                await web.EnsureCoreWebView2Async(env);
                CoreWebView2Settings s = web.CoreWebView2.Settings;
                s.AreDevToolsEnabled = false;
                s.IsStatusBarEnabled = false;
                s.IsZoomControlEnabled = false;
                s.IsBuiltInErrorPageEnabled = false;
                try { s.AreBrowserAcceleratorKeysEnabled = false; } catch (Exception) { }      // no F5 reload of a spent address
                web.CoreWebView2.NewWindowRequested += OnNewWindow;
                web.CoreWebView2.NavigationStarting += OnNavigating;
                web.CoreWebView2.NavigationCompleted += OnNavigated;
            }
            catch (Exception)
            {
                webFailed = true;             // the web engine is missing: fall back to Edge's app window when the page is known
            }
            if (opts.ServerExec == "") { url = opts.Url; ready = true; Load(); }
            else StartServer();
        }

        void StartServer()
        {
            try
            {
                ProcessStartInfo psi = new ProcessStartInfo(opts.ServerExec);
                StringBuilder args = new StringBuilder();
                foreach (string a in opts.ServerArgs)
                {
                    if (args.Length > 0) args.Append(' ');
                    args.Append(Quote(a));
                }
                psi.Arguments = args.ToString();
                if (opts.ServerCwd != "") psi.WorkingDirectory = opts.ServerCwd;
                psi.UseShellExecute = false;
                psi.CreateNoWindow = true;
                psi.RedirectStandardOutput = true;
                psi.RedirectStandardInput = true;               // the server's input is a pipe that closes with us
                psi.StandardOutputEncoding = Encoding.UTF8;
                psi.EnvironmentVariables["CALE_APP_WINDOW"] = "1";    // tells the app: pick a free port, do not open a browser
                psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
                server = new Process();
                server.StartInfo = psi;
                server.EnableRaisingEvents = true;
                server.OutputDataReceived += OnServerLine;
                server.Exited += delegate { if (!closing) { try { BeginInvoke((MethodInvoker)delegate { Close(); }); } catch (Exception) { } } };
                server.Start();
                KillWithMe.Add(server);
                server.BeginOutputReadLine();
            }
            catch (Exception)
            {
                MessageBox.Show(opts.Title + " could not start.\r\n\r\nOpen it again. If this keeps happening, restart your computer.", opts.Title, MessageBoxButtons.OK, MessageBoxIcon.Information);
                Close();
            }
        }

        static string Quote(string a)
        {
            if (a.Length > 0 && a.IndexOfAny(new char[] { ' ', '\t', '"' }) < 0) return a;
            return "\"" + a.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
        }

        void OnServerLine(object sender, DataReceivedEventArgs e)
        {
            if (e.Data == null || !e.Data.StartsWith("READY ")) return;
            string[] words = e.Data.Substring(6).Trim().Split(new char[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
            if (words.Length == 0) return;
            string port = words[0];
            string tok = words.Length > 1 ? words[1] : "";          // optional per-launch secret
            try
            {
                BeginInvoke((MethodInvoker)delegate
                {
                    // Every READY line loads its address. A later one (same port, new one-time code) reloads the page.
                    url = opts.UrlTemplate.Replace("{port}", port).Replace("{token}", tok);
                    token = (opts.UrlTemplate.IndexOf("{token}") >= 0) ? tok : "";
                    tokenSpent = false;                                  // a new code may be used once
                    tokenUrl = "";
                    cleanUrl = "";
                    retries = 0;
                    retryTimer.Stop();
                    ready = true;
                    Load();
                });
            }
            catch (Exception) { }
        }

        void Load()
        {
            if (!ready || url == "") return;
            if (webFailed)
            {
                if (token != "") { NeedWebView(); return; }        // a secret address never goes to a browser
                if (pending == url) return;
                pending = url;
                OpenInEdge(url);
                return;
            }
            if (!firstNavStarted)
            {
                firstNavStarted = true;
                if (splashTimer != null) { splashTimer.Stop(); splashTimer.Interval = 60000; splashTimer.Start(); }     // 60 s from the first page load
            }
            try { web.CoreWebView2.Navigate(url); } catch (Exception) { }
        }

        // The chat address carries a secret. It must not reach a browser (history, command line, extensions).
        // So: stop the server, say so in plain words, offer the official download page (no secret in it), and quit.
        void NeedWebView()
        {
            if (closing) return;
            closing = true;
            StopServer();
            HideSplash();
            Hide();
            DialogResult r = MessageBox.Show(
                opts.Title + " needs a Windows part called WebView2 to show its window.\r\n\r\nInstall it from Microsoft, then open " + opts.Title + " again.\r\n\r\nOpen the Microsoft page now?",
                opts.Title, MessageBoxButtons.YesNo, MessageBoxIcon.Information);
            if (r == DialogResult.Yes)
            {
                try { Process.Start("https://developer.microsoft.com/microsoft-edge/webview2/"); } catch (Exception) { }
            }
            Close();
        }

        // Fallback when the web engine is missing (only for addresses with no secret): Edge's app mode has no address bar either.
        void OpenInEdge(string u)
        {
            try { Process.Start("msedge.exe", "--app=" + u); }
            catch (Exception) { try { Process.Start(u); } catch (Exception) { } }
        }

        // The page's server may take a moment to start. Try again a few times, quietly.
        void OnNavigated(object sender, CoreWebView2NavigationCompletedEventArgs e)
        {
            if (e.IsSuccess) WatchForText();                    // keep the loading screen until the page has drawn some text (an HTTP error page has text too)
            if (e.IsSuccess)
            {
                if (token != "" && !tokenSpent)
                {
                    tokenSpent = true;                                   // the server accepts the secret once, then uses a cookie
                    tokenUrl = url;
                    try { cleanUrl = new Uri(url).GetLeftPart(UriPartial.Path); } catch (Exception) { cleanUrl = ""; }
                }
                return;
            }
            retries++;
            if (retries <= 40) retryTimer.Start(); else HideSplash();
        }

        // A big page needs seconds after its HTML has loaded before anything is drawn. Look every 250 ms.
        void WatchForText()
        {
            if (splash == null || drawnTimer != null) return;
            drawnTimer = new System.Windows.Forms.Timer();
            drawnTimer.Interval = 250;
            drawnTimer.Tick += async delegate
            {
                if (polling || splash == null) return;
                polling = true;
                try
                {
                    string r = await web.CoreWebView2.ExecuteScriptAsync("document.body ? document.body.innerText.trim().length : 0");
                    int n;
                    if (int.TryParse(r, out n) && n > 0) HideSplash();
                }
                catch (Exception) { }
                polling = false;
            };
            drawnTimer.Start();
        }

        void HideSplash()
        {
            if (drawnTimer != null) { drawnTimer.Stop(); drawnTimer.Dispose(); drawnTimer = null; }
            if (splashTimer != null) { splashTimer.Stop(); splashTimer.Dispose(); splashTimer = null; }
            if (splash != null)
            {
                Controls.Remove(splash);
                splash.Dispose();
                splash = null;
            }
        }

        // Links to other sites open in the browser, not inside this window.
        void OnNavigating(object sender, CoreWebView2NavigationStartingEventArgs e)
        {
            if (tokenSpent && cleanUrl != "" && e.Uri == tokenUrl)
            {
                // A reload of the first address would send the spent secret and get a blank refusal. Use the cookie instead.
                e.Cancel = true;
                string go = cleanUrl;
                BeginInvoke((MethodInvoker)delegate { try { web.CoreWebView2.Navigate(go); } catch (Exception) { } });
                return;
            }
            try
            {
                Uri target = new Uri(e.Uri);
                Uri baseUri = new Uri(url);
                if (e.IsUserInitiated && target.Host != baseUri.Host && (target.Scheme == "http" || target.Scheme == "https"))
                {
                    e.Cancel = true;
                    Process.Start(e.Uri);
                }
            }
            catch (Exception) { }
        }

        void OnNewWindow(object sender, CoreWebView2NewWindowRequestedEventArgs e)
        {
            e.Handled = true;
            try
            {
                Uri target = new Uri(e.Uri);
                if (target.Scheme == "http" || target.Scheme == "https") Process.Start(e.Uri);
            }
            catch (Exception) { }
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            closing = true;
            StopServer();
            base.OnFormClosing(e);
        }

        void StopServer()
        {
            if (server == null) return;
            try
            {
                if (!server.HasExited)
                {
                    ProcessStartInfo k = new ProcessStartInfo("taskkill.exe", "/PID " + server.Id + " /T /F");
                    k.CreateNoWindow = true;
                    k.UseShellExecute = false;
                    Process kp = Process.Start(k);
                    if (kp != null) kp.WaitForExit(3000);
                }
            }
            catch (Exception) { }
            try { if (!server.HasExited) server.Kill(); } catch (Exception) { }
        }
    }

    static class Program
    {
        static Dictionary<string, Assembly> loaded = new Dictionary<string, Assembly>();

        // The wrapper DLLs are inside this exe. Hand them out when .NET asks and there is no file beside the exe.
        static Assembly Resolve(object sender, ResolveEventArgs args)
        {
            string name = new AssemblyName(args.Name).Name + ".dll";
            Assembly cached;
            if (loaded.TryGetValue(name, out cached)) return cached;
            Assembly self = Assembly.GetExecutingAssembly();
            using (Stream s = self.GetManifestResourceStream(name))
            {
                if (s == null) return null;
                byte[] bytes = new byte[s.Length];
                int got = 0;
                while (got < bytes.Length) got += s.Read(bytes, got, bytes.Length - got);
                Assembly a = Assembly.Load(bytes);
                loaded[name] = a;
                return a;
            }
        }

        [STAThread]
        static int Main(string[] args)
        {
            AppDomain.CurrentDomain.AssemblyResolve += Resolve;
            return Run(args);
        }

        static int Run(string[] args)
        {
            Options o = Options.Parse(args);
            if (o.Url == "" && o.ServerExec == "")
            {
                MessageBox.Show("usage: CaleWindow.exe --url URL --title NAME   or   --server-exec PATH --server-arg ARG", "CaleWindow");
                return 2;
            }
            Shell.Prepare();
            Shell.SetProcessId(Shell.IdFor(o.Title));
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new MainForm(o));
            return 0;
        }
    }
}
