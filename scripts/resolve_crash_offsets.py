import bisect, subprocess, sys
NM = r"C:\cale\lbuild2\w64devkit\bin\nm.exe"
CF = r"C:\cale\lbuild2\w64devkit\bin\c++filt.exe"
def resolve(dll, offs):
    out = subprocess.run([NM, "-n", "--defined-only", r"C:\cale\dltest" + "\\" + dll], capture_output=True, text=True).stdout
    syms = []
    for l in out.splitlines():
        p = l.split(" ", 2)
        if len(p) == 3 and p[1] in "tTwW":
            syms.append((int(p[0], 16), p[2]))
    hdr = subprocess.run([r"C:\cale\lbuild2\w64devkit\bin\objdump.exe", "-p", r"C:\cale\dltest" + "\\" + dll], capture_output=True, text=True).stdout
    base = next(int(l.split()[1], 16) for l in hdr.splitlines() if l.strip().startswith("ImageBase"))
    keys = [a for a, _ in syms]
    for o in offs:
        i = bisect.bisect_right(keys, base + o) - 1
        name = subprocess.run([CF, syms[i][1]], capture_output=True, text=True).stdout.strip()
        print(f"{dll} +{o:#x} -> {name} (+{base + o - syms[i][0]:#x})", flush=True)
resolve("ggml-base.dll", [0x20111, 0x20ed0, 0x256f3])
resolve("llama.dll", [0x15a7e0, 0x134a3a, 0x1612b0, 0x13b8b6])
