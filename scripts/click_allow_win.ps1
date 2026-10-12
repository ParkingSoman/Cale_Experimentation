# Click "Yes" (the Allow button) on Cale's own Allow dialog. Only a top-level window titled exactly "Cale" is touched.
# usage: powershell -File click_allow_win.ps1 [seconds]
param([int]$Seconds = 40)
Add-Type @"
using System; using System.Text; using System.Runtime.InteropServices;
public class CaleClick {
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr FindWindow(string cls, string title);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern IntPtr FindWindowEx(IntPtr p, IntPtr after, string cls, string title);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern IntPtr SendMessage(IntPtr h, int m, IntPtr w, IntPtr l);
  public static bool ClickYes() {
    IntPtr w = FindWindow("#32770", "Cale");
    if (w == IntPtr.Zero) return false;
    IntPtr b = IntPtr.Zero;
    while ((b = FindWindowEx(w, b, "Button", null)) != IntPtr.Zero) {
      StringBuilder s = new StringBuilder(64); GetWindowText(b, s, 64);
      if (s.ToString().Replace("&", "") == "Yes") { SendMessage(b, 0x00F5, IntPtr.Zero, IntPtr.Zero); return true; }   // BM_CLICK
    }
    return false;
  }
}
"@
$end = (Get-Date).AddSeconds($Seconds)
while ((Get-Date) -lt $end) {
  if ([CaleClick]::ClickYes()) { "clicked Allow"; exit 0 }
  Start-Sleep -Milliseconds 400
}
"no Cale Allow dialog appeared"; exit 1
