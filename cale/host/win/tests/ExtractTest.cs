// A test program for Extractor in SelfExtract.cs (no window). Build and run: scripts/test_extract_win.ps1 (on Windows).
//   ExtractTest.exe ZIPFILE TARGETFOLDER [THREADS]
// The zip is read as a whole file (start 0, the length of the file): the same code as the real setup program, which reads a zip at the end of its own exe.
using System;
using System.IO;

namespace CaleHost
{
    static class ExtractTestProgram
    {
        static int Main(string[] args)
        {
            if (args.Length < 2) { Console.Error.WriteLine("usage: ExtractTest ZIPFILE TARGET [THREADS]"); return 2; }
            int threads = args.Length > 2 ? int.Parse(args[2]) : Extractor.DefaultThreads();
            DateTime t = DateTime.UtcNow;
            Directory.CreateDirectory(args[1]);
            Extractor.ExtractAll(args[0], 0, new FileInfo(args[0]).Length, Path.GetFullPath(args[1]), threads);
            Console.WriteLine("extracted with " + threads + " threads in " + (DateTime.UtcNow - t).TotalSeconds.ToString("F1") + " s");
            return 0;
        }
    }
}
