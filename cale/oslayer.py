"""Everything that differs between macOS, Windows and Linux lives here. The rest of Cale calls these functions.

Rule: use generic Python when it loses nothing. Use an OS call only where generic code would lose something.
Standard library only, and no import of an OS-only module at the top of the file, so this imports everywhere.
"""
import contextlib
import ctypes
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

def core_types():
    """Fast and slow cores on a hybrid CPU: {"fast": n, "slow": n}. None if the computer has one kind or the OS does not say."""
    try:
        if sys.platform == "darwin":
            def sysctl(name):
                return int(subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=5).stdout.strip())
            fast, slow = sysctl("hw.perflevel0.physicalcpu"), sysctl("hw.perflevel1.physicalcpu")
            return {"fast": fast, "slow": slow} if fast and slow else None
        if sys.platform.startswith("linux"):
            caps = {}
            for d in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
                f = d / "cpu_capacity"
                if f.exists():
                    caps[d.name] = int(f.read_text())
            if len(set(caps.values())) > 1:
                top = max(caps.values())
                return {"fast": sum(1 for v in caps.values() if v >= 0.9 * top), "slow": sum(1 for v in caps.values() if v < 0.9 * top)}
            return None
        if sys.platform == "win32":
            from ctypes import wintypes
            k = ctypes.windll.kernel32
            size = wintypes.ULONG(0)
            k.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
            buf = ctypes.create_string_buffer(size.value)
            if not k.GetSystemCpuSetInformation(buf, size, ctypes.byref(size), None, 0):
                return None
            classes, off = [], 0
            while off < size.value:
                rec_size = int.from_bytes(buf[off:off + 4], "little")
                if int.from_bytes(buf[off + 4:off + 8], "little") == 0:          # type CpuSetInformation
                    classes.append(buf[off + 18])                                  # EfficiencyClass (higher = faster core)
                off += rec_size or 1
            if len(set(classes)) > 1:
                top = max(classes)
                return {"fast": sum(1 for c in classes if c == top), "slow": sum(1 for c in classes if c != top)}
    except Exception:
        pass
    return None


def read_uncached(path, nbytes):
    """Read the first `nbytes` of a file straight from the disk, not from the OS file cache. Returns bytes per second, or 0.0 if the OS refuses."""
    nbytes = min(int(nbytes), os.path.getsize(path))
    block = 4 << 20
    try:
        if sys.platform == "win32":
            from ctypes import wintypes
            k = ctypes.windll.kernel32
            k.CreateFileW.restype = ctypes.c_void_p
            k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
            h = k.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x20000000, None)          # GENERIC_READ, share read, OPEN_EXISTING, NO_BUFFERING
            if h in (None, ctypes.c_void_p(-1).value):
                return 0.0
            k.VirtualAlloc.restype = ctypes.c_void_p
            k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD]
            buf = k.VirtualAlloc(None, block, 0x3000, 0x04)                                  # page aligned, as the unbuffered read needs
            k.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
            done, got = 0, wintypes.DWORD(0)
            t = time.perf_counter()
            while done < nbytes and k.ReadFile(h, buf, block, ctypes.byref(got), None) and got.value:
                done += got.value
            dt = time.perf_counter() - t
            k.CloseHandle.argtypes = [ctypes.c_void_p]
            k.CloseHandle(h)
            return done / dt if dt > 0 and done else 0.0
        fd = os.open(str(path), os.O_RDONLY)
        try:
            if sys.platform == "darwin":
                import fcntl
                fcntl.fcntl(fd, 48, 1)                                                      # F_NOCACHE
            elif hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            done, t = 0, time.perf_counter()
            while done < nbytes:
                chunk = os.read(fd, block)
                if not chunk:
                    break
                done += len(chunk)
            dt = time.perf_counter() - t
            return done / dt if dt > 0 and done else 0.0
        finally:
            os.close(fd)
    except Exception:
        return 0.0


def cuda_driver_version():
    """The CUDA version this computer's NVIDIA driver supports, as 12040 for 12.4. 0 if there is no NVIDIA driver (or it cannot be asked)."""
    try:
        lib = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
        if lib.cuInit(0) != 0:
            return 0
        v = ctypes.c_int(0)
        return v.value if lib.cuDriverGetVersion(ctypes.byref(v)) == 0 else 0
    except (OSError, AttributeError):
        return 0


def give_back_heap():
    """Linux: hand memory that the program freed but its allocator kept (compute buffers of a long prompt) back to the OS.
    Without it, process memory stays 0.3 GB or more above what the model needs after a big prompt. Other systems do this already."""
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


def is_terminal(stream):
    """True if a person can type at / read from `stream`. On Windows the empty device NUL says isatty() is true, so ask the console."""
    try:
        if not stream.isatty():
            return False
        if sys.platform == "win32":
            import msvcrt
            mode = ctypes.c_uint32()
            return bool(ctypes.windll.kernel32.GetConsoleMode(msvcrt.get_osfhandle(stream.fileno()), ctypes.byref(mode)))
        return True
    except (OSError, ValueError, AttributeError):
        return False


MAC = sys.platform == "darwin"
WIN = sys.platform == "win32"
LINUX = sys.platform.startswith("linux")
NAME = "macOS" if MAC else "Windows" if WIN else "Linux"


# ---- where files live ---------------------------------------------------------------------------------------------

def home_dir(env=None, platform=None, home=None):
    """Cale's folder. Arguments exist so a test can check every OS from one computer."""
    env = os.environ if env is None else env
    plat = sys.platform if platform is None else platform
    home = Path.home() if home is None else Path(home)
    if "CALE_HOME" in env:
        return Path(env["CALE_HOME"])
    if plat == "darwin":
        return home / "Library" / "Application Support" / "Cale"
    if plat == "win32":
        return Path(env.get("LOCALAPPDATA") or home / "AppData" / "Local") / "Cale"
    return Path(env.get("XDG_DATA_HOME") or home / ".local" / "share") / "cale"


def venv_python(venv):
    """The Python inside a virtual environment folder."""
    venv = Path(venv)
    return venv / ("Scripts/python.exe" if WIN else "bin/python")


# ---- memory -------------------------------------------------------------------------------------------------------

class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _win_memstatus():
    s = _MemStatus()
    s.dwLength = ctypes.sizeof(s)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s


def _cgroup():
    """(limit, used, reclaimable) in bytes if this Linux process runs under a memory limit (a container, a systemd unit), else None.
    Cale must see the limit, not the computer's RAM: using memory it does not have means being killed."""
    if not LINUX:
        return None
    for base, lim_f, use_f in (("/sys/fs/cgroup", "memory.max", "memory.current"), ("/sys/fs/cgroup/memory", "memory.limit_in_bytes", "memory.usage_in_bytes")):
        try:
            raw = Path(base, lim_f).read_text().strip()
            if raw == "max":
                return None
            limit = int(raw)
            if limit >= 1 << 60:
                return None
            used = int(Path(base, use_f).read_text())
            reclaim = 0
            for line in Path(base, "memory.stat").read_text().splitlines():
                if line.split()[0] in ("inactive_file", "total_inactive_file"):
                    reclaim = int(line.split()[1])
            return limit, used, reclaim
        except (OSError, ValueError, IndexError):
            continue
    return None


def total_ram():
    """Physical memory in bytes, or the memory limit of the container Cale runs in. On WSL this is what the Linux side can see."""
    if WIN:
        return int(_win_memstatus().ullTotalPhys)
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    cg = _cgroup()
    return min(total, cg[0]) if cg else total


def available_ram():
    """Memory that can be used right now without pushing other things out, in bytes."""
    if WIN:
        return int(_win_memstatus().ullAvailPhys)
    if LINUX:
        host = None
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                host = int(line.split()[1]) * 1024
        cg = _cgroup()
        if cg:                                                     # under a limit: what the limit leaves, counting cache that can be given back
            limit, used, reclaim = cg
            room = max(0, limit - used + reclaim)
            return min(host, room) if host is not None else room
        if host is not None:
            return host
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout          # macOS
    page = int(out.split("page size of")[1].split()[0])
    pages = 0
    for line in out.splitlines():
        if line.split(":")[0] in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
            pages += int(line.split(":")[1].strip().rstrip("."))
    return pages * page


def process_rss():
    """Memory this process holds, in bytes."""
    if WIN:
        class _PMC(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = _PMC()
        c.cb = ctypes.sizeof(c)
        h = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(h), ctypes.byref(c), c.cb)
        return int(c.WorkingSetSize)
    if LINUX:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # macOS: bytes (peak, not current)


# ---- load and power: the live facts the speed prediction and the `power` goal use ---------------------------------------
_cpu_last = None
_power_cache = (0.0, {"on_battery": None, "hot": None})


def _mac_ticks():
    """(idle ticks, all ticks) of every CPU since boot, from the kernel (host_processor_info). None if it cannot be read."""
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        libc.mach_host_self.restype = ctypes.c_uint
        libc.host_processor_info.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_uint),
                                             ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), ctypes.POINTER(ctypes.c_uint)]
        n, info, cnt = ctypes.c_uint(), ctypes.POINTER(ctypes.c_int)(), ctypes.c_uint()
        if libc.host_processor_info(libc.mach_host_self(), 2, ctypes.byref(n), ctypes.byref(info), ctypes.byref(cnt)) != 0:
            return None
        idle = total = 0
        for i in range(n.value):
            user, system, idl, nice = (info[4 * i + k] & 0xFFFFFFFF for k in range(4))
            idle += idl
            total += user + system + idl + nice
        return idle, total
    except Exception:
        return None


def cpu_busy():
    """Share (0.0 to 1.0) of the CPU that all programs used since the last call. Needs no admin rights. The first call returns 0.0."""
    global _cpu_last
    try:
        if WIN:
            class _FT(ctypes.Structure):
                _fields_ = [("lo", ctypes.c_ulong), ("hi", ctypes.c_ulong)]
            idle, kernel, user = _FT(), _FT(), _FT()
            ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
            val = lambda t: (t.hi << 32) | t.lo                                   # noqa: E731
            now = (val(idle), val(kernel) + val(user))                             # kernel time includes idle time
        elif LINUX:
            f = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            vals = [int(x) for x in f]
            now = (vals[3] + (vals[4] if len(vals) > 4 else 0), sum(vals[:8]))
        else:
            now = _mac_ticks()                                                       # macOS: the CPU's own tick counters, as on the other systems
            if now is None:
                return max(0.0, min(1.0, os.getloadavg()[0] / (os.cpu_count() or 1)))   # the 1-minute load, slow but always there
    except Exception:
        return 0.0
    last, _cpu_last = _cpu_last, now
    if last is None or now[1] <= last[1]:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (now[0] - last[0]) / (now[1] - last[1])))


_other_last = None
_other_value = 0.0


def cpu_busy_other():
    """Share (0.0 to 1.0) of the CPU that programs *other than this one* used since the last call. Cale's own work is taken out:
    otherwise an answer that just ended would look like a busy computer. Calls less than half a second apart return the last value."""
    global _other_last, _other_value
    now = time.time()
    if _other_last is not None and now - _other_last[0] < 0.5:
        return _other_value
    total = cpu_busy()
    cpu_s = time.process_time()
    last, _other_last = _other_last, (now, cpu_s)
    if last is None:
        return 0.0
    own = (cpu_s - last[1]) / max(now - last[0], 1e-6) / (os.cpu_count() or 1)
    _other_value = max(0.0, min(1.0, total - own))
    return _other_value


def power_state():
    """{"on_battery": True|False|None, "hot": True|False|None}. None means this computer cannot say. Cached for 5 s."""
    global _power_cache
    t, cached = _power_cache
    if time.time() - t < 5:
        return cached
    st = {"on_battery": None, "hot": None}
    try:
        if MAC:
            out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=3).stdout
            if "Battery Power" in out or "AC Power" in out:
                st["on_battery"] = "Battery Power" in out
            therm = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=3).stdout
            limit = [int(x.split("=")[1]) for x in therm.split() if x.startswith("CPU_Speed_Limit=")]
            st["hot"] = bool(limit and limit[0] < 100)
        elif WIN:
            class _SPS(ctypes.Structure):
                _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte), ("BatteryLifePercent", ctypes.c_ubyte),
                            ("SystemStatusFlag", ctypes.c_ubyte), ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
            sp = _SPS()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(sp)) and sp.ACLineStatus in (0, 1):
                st["on_battery"] = sp.ACLineStatus == 0
        else:
            base = Path("/sys/class/power_supply")
            if base.is_dir():
                kinds = {d.name: (d / "type").read_text().strip() for d in base.iterdir() if (d / "type").exists()}
                if "Battery" in kinds.values():
                    mains = [d for d, k in kinds.items() if k == "Mains" and (base / d / "online").exists()]
                    st["on_battery"] = bool(mains) and all((base / d / "online").read_text().strip() == "0" for d in mains)
    except Exception:
        pass
    _power_cache = (time.time(), st)
    return st


# ---- memory pressure: 1 normal, 2 warn, 4 critical ------------------------------------------------------------------

def level_from_psi(text):
    """Linux pressure-stall text (/proc/pressure/memory). `some` = a task waited for memory, `full` = all tasks did."""
    some = full = 0.0
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        avg10 = next((float(p.split("=")[1]) for p in parts[1:] if p.startswith("avg10=")), 0.0)
        if parts[0] == "some":
            some = avg10
        elif parts[0] == "full":
            full = avg10
    if full >= 10 or some >= 40:
        return 4
    if some >= 15 or full >= 3:
        return 2
    return 1


def level_from_free(avail, total):
    """Fallback with no pressure signal: how much memory is still available."""
    frac = avail / total if total else 1.0
    return 4 if frac < 0.04 else 2 if frac < 0.10 else 1


_libc = None


def pressure_level():
    if MAC:
        global _libc
        if _libc is None:
            _libc = ctypes.CDLL("libc.dylib")
        v, sz = ctypes.c_int(0), ctypes.c_size_t(4)
        if _libc.sysctlbyname(b"kern.memorystatus_vm_pressure_level", ctypes.byref(v), ctypes.byref(sz), None, 0) != 0:
            return 1
        return v.value
    if LINUX:
        try:
            return level_from_psi(Path("/proc/pressure/memory").read_text())
        except OSError:
            return level_from_free(available_ram(), total_ram())
    if WIN:
        s = _win_memstatus()
        return win_level(_win_low_memory_event(), s.dwMemoryLoad)
    return 1


def win_level(low_memory_event, load_pct):
    """The Windows low-memory event means the system is short right now. High load alone is only a warning."""
    return 4 if low_memory_event else 2 if load_pct >= 92 else 1


_win_handle = None


def _win_low_memory_event():
    global _win_handle
    k32 = ctypes.windll.kernel32
    if _win_handle is None:
        k32.CreateMemoryResourceNotification.restype = ctypes.c_void_p
        _win_handle = k32.CreateMemoryResourceNotification(0)           # 0 = LowMemoryResourceNotification
    if not _win_handle:
        return False
    state = ctypes.c_int(0)
    k32.QueryMemoryResourceNotification(ctypes.c_void_p(_win_handle), ctypes.byref(state))
    return bool(state.value)


# ---- file lock: one writer at a time, across processes ---------------------------------------------------------------

@contextlib.contextmanager
def file_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as f:
        if WIN:
            import msvcrt
            while True:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def private_file(path, text):
    """Write a file only this user can read. On Windows the user's own folder is already private."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    for i in range(30):                      # Windows refuses the swap for a moment while a scanner holds the new file
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if not WIN or i == 29:
                raise
            time.sleep(0.05 * (i + 1))


# ---- the Allow dialog ---------------------------------------------------------------------------------------------

def has_screen():
    """True if Cale can show the Allow question on a screen here. CALE_HEADLESS=1 says there is none (for tests and servers)."""
    if os.environ.get("CALE_HEADLESS") == "1":
        return False
    if MAC or WIN:
        return True
    return bool((os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and (shutil.which("zenity") or shutil.which("kdialog")))


NO_SCREEN_MESSAGES = (b"failed to open display", b"cannot open display", b"cannot connect to x server", b"could not connect to display",
                      b"failed to connect to the wayland display", b"no protocol specified")


def ask_allow(app, replacing=False, seconds=120):
    """Ask the person at the computer: allow this app to use Cale? True only for a clear yes. None if the dialog tool found no screen to show it on.
    The app name travels as data, never inside script text."""
    extra = "This app already has a key. Allowing replaces it." if replacing else ""
    title = "Cale"
    body = (f'Allow "{app}" to use Cale on this computer?\n\n'
            "Cale lets apps use AI that runs on this computer. What you type stays here.\n"
            "If you did not just open this app, choose Don't Allow.")
    if extra:
        body += "\n\n" + extra
    if MAC:
        script = ['on run argv',
                  'set a to item 1 of argv',
                  'set r to item 2 of argv',
                  'set extra to ""',
                  'if r is "yes" then set extra to return & return & "This app already has a key. Allowing replaces it."',
                  'display dialog "Allow \\"" & a & "\\" to use Cale on this computer?" & return & return & '
                  '"Cale lets apps use AI that runs on this computer. What you type stays here." & return & '
                  '"If you did not just open this app, choose Don\'t Allow." & extra '
                  'buttons {"Don\'t Allow", "Allow"} default button "Don\'t Allow" cancel button "Don\'t Allow" '
                  f'with title "Cale" with icon note giving up after {seconds}',
                  'end run']
        cmd = ["osascript"] + [x for s in script for x in ("-e", s)] + [app, "yes" if replacing else "no"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[connect] osascript failed code={r.returncode} stderr={r.stderr.strip()!r}", flush=True)
        return r.returncode == 0 and "button returned:Allow" in r.stdout and "gave up:true" not in r.stdout
    if WIN:
        # MessageBoxTimeoutW lives in user32 on every Windows version. 0x24 = Yes/No + question icon.
        # 0x100 = second button (No) is the default. 0x10000 = set foreground. 0x40000 = always on top.
        user32 = ctypes.windll.user32
        r = user32.MessageBoxTimeoutW(0, body, title, 0x24 | 0x100 | 0x10000 | 0x40000, 0, seconds * 1000)
        return r == 6                                                # IDYES
    for tool, args in (("zenity", ["--question", f"--timeout={seconds}", "--title", title, "--text", body,
                                   "--ok-label", "Allow", "--cancel-label", "Don't Allow"]),
                       ("kdialog", ["--title", title, "--yesno", body])):
        exe = shutil.which(tool)
        if exe and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            try:
                t0 = time.time()
                r = subprocess.run([exe, *args], capture_output=True, timeout=seconds + 5)
                err = r.stderr.lower()
                # Only a tool that gives up at once with a known "no screen" message is "no screen". A person needs seconds to answer, so
                # a click on Don't Allow is never read this way, and nothing here can turn a "no" into a "yes".
                if r.returncode != 0 and time.time() - t0 < 2.0 and any(m in err for m in NO_SCREEN_MESSAGES):
                    print(f"[connect] {tool} could not reach the screen. The question will be answered in a terminal instead.", flush=True)
                    return None
                return r.returncode == 0
            except subprocess.TimeoutExpired:          # the tool ignored its own timeout: no answer means no
                print(f"[connect] {tool} did not close after {seconds} s. Counting that as no.", flush=True)
                return False
            except OSError as e:
                print(f"[connect] {tool} failed to start ({e}). Refusing.", flush=True)
                return False
    print("[connect] no screen to show the Allow question on (no DISPLAY, or no zenity/kdialog). Refusing.", flush=True)
    return False


# ---- starting and stopping Cale without a service manager ---------------------------------------------------------------

def spawn_detached(argv, log_path, env=None):
    """Start a program that keeps running after this one exits. Returns the process id."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out = open(log_path, "ab")
    kw = {"stdin": subprocess.DEVNULL, "stdout": out, "stderr": out, "env": env}
    if WIN:
        kw["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000      # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW
    else:
        kw["start_new_session"] = True
    return subprocess.Popen(argv, **kw).pid


def pid_alive(pid):
    if WIN:
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))     # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong(0)
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259                                         # STILL_ACTIVE
    try:
        if os.waitpid(int(pid), os.WNOHANG)[0] == int(pid):          # our own child that already ended
            return False
    except ChildProcessError:
        pass                                                           # not our child: kill(0) decides
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def kill_pid(pid):
    if WIN:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError:
            pass


def follow(path, out=None):
    """Print a file as it grows (a portable `tail -f`). Ctrl-C stops it."""
    out = out or sys.stdout
    try:
        with open(path, "r", errors="replace") as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if line:
                    out.write(line)
                    out.flush()
                else:
                    time.sleep(0.3)
    except (KeyboardInterrupt, FileNotFoundError):
        return
