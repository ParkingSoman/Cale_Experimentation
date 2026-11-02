// CaleWindow: shows one page in a native window. No address bar, no tabs.
// Cale installs this once. Cale's chat and downloaded apps use it to look like programs.
//
//   CaleWindow --url http://127.0.0.1:51234/ --title "Quick Notes" [--width 900 --height 640] [--min-width 480 --min-height 360]
//
// An app can also let this program start its server. The server picks a free port and prints  READY <port>  on one line:
//   CaleWindow --title "Quick Notes" --server-exec /path/python --server-arg app.py --server-cwd /path/app
//              --url-template http://127.0.0.1:{port}/        ({token} is also filled when the server prints  READY <port> <token>)
// The window closes the program. The server stops when the window closes. If the server stops, the window closes.
// The program also exits when --parent-pid ends.
import Cocoa
import WebKit

struct Options {
    var url = ""
    var title = "Cale"
    var width = 980.0
    var height = 700.0
    var minWidth = 480.0
    var minHeight = 360.0
    var parentPid: Int32 = 0
    var serverExec = ""
    var serverArgs: [String] = []
    var serverCwd = ""
    var appearance = ""                                           // "light" or "dark": this window only, whatever the computer uses (for tests)
    var urlTemplate = "http://127.0.0.1:{port}/"
    var splash = ""                      // a picture (SVG or PNG) shown on cream, with splashText, until the page has loaded
    var splashText = ""
}

func parse() -> Options {
    var o = Options()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let a = it.next() {
        switch a {
        case "--url": o.url = it.next() ?? ""
        case "--title": o.title = it.next() ?? o.title
        case "--width": o.width = Double(it.next() ?? "") ?? o.width
        case "--height": o.height = Double(it.next() ?? "") ?? o.height
        case "--min-width": o.minWidth = Double(it.next() ?? "") ?? o.minWidth
        case "--min-height": o.minHeight = Double(it.next() ?? "") ?? o.minHeight
        case "--parent-pid": o.parentPid = Int32(it.next() ?? "") ?? 0
        case "--server-exec": o.serverExec = it.next() ?? ""
        case "--server-arg": o.serverArgs.append(it.next() ?? "")
        case "--server-cwd": o.serverCwd = it.next() ?? ""
        case "--appearance": o.appearance = it.next() ?? ""
        case "--url-template": o.urlTemplate = it.next() ?? o.urlTemplate
        case "--splash": o.splash = it.next() ?? ""
        case "--splash-text": o.splashText = it.next() ?? ""
        default: break
        }
    }
    return o
}

final class Delegate: NSObject, NSApplicationDelegate, WKNavigationDelegate, WKUIDelegate, NSWindowDelegate {
    var opts: Options
    var window: NSWindow!
    var web: WKWebView!
    var retries = 0
    var server: Process?

    init(_ o: Options) { opts = o }

    /// Start the app's server. When it prints READY <port>, show its page.
    func startServer() {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: opts.serverExec)
        p.arguments = opts.serverArgs
        if !opts.serverCwd.isEmpty { p.currentDirectoryURL = URL(fileURLWithPath: opts.serverCwd) }
        var env = ProcessInfo.processInfo.environment
        env["CALE_APP_WINDOW"] = "1"                                  // tells the app: pick a free port, do not open a browser
        p.environment = env
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = FileHandle.nullDevice
        var buffer = Data()
        pipe.fileHandleForReading.readabilityHandler = { h in
            buffer.append(h.availableData)
            while let nl = buffer.firstIndex(of: 10) {
                let line = String(decoding: buffer[buffer.startIndex..<nl], as: UTF8.self)
                buffer.removeSubrange(buffer.startIndex...nl)
                if line.hasPrefix("READY ") {
                    // READY <port>  or  READY <port> <token>. The token (a one-time secret for this launch) fills {token} in the address.
                    let words = line.dropFirst(6).split(separator: " ", omittingEmptySubsequences: true).map(String.init)
                    let port = words.first ?? ""
                    let token = words.count > 1 ? words[1] : ""
                    DispatchQueue.main.async {
                        self.opts.url = self.opts.urlTemplate.replacingOccurrences(of: "{port}", with: port).replacingOccurrences(of: "{token}", with: token)
                        self.load()
                    }
                }
            }
        }
        p.terminationHandler = { _ in DispatchQueue.main.async { NSApp.terminate(nil) } }
        do { try p.run(); server = p } catch {
            let a = NSAlert()
            a.messageText = "\(opts.title) could not start."
            a.informativeText = "Open it again. If this keeps happening, restart your computer."
            a.runModal()
            NSApp.terminate(nil)
        }
    }

    func applicationWillTerminate(_ n: Notification) { server?.terminationHandler = nil; server?.terminate() }

    /// One line in the Cale timeline (the same file the Python parts write: cale/timeline.py). Never fails.
    func mark(_ step: String) {
        let env = ProcessInfo.processInfo.environment
        let home = env["CALE_HOME"] ?? (NSHomeDirectory() + "/Library/Application Support/Cale")
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: home, isDirectory: &isDir), isDir.boolValue else { return }
        let t = (Date().timeIntervalSince1970 * 10).rounded() / 10
        let line = "{\"t\":\(t),\"kind\":\"mark\",\"run\":\"window\",\"step\":\"\(step)\",\"pid\":\(ProcessInfo.processInfo.processIdentifier)}\n"
        if let h = FileHandle(forWritingAtPath: home + "/timeline.jsonl") {
            h.seekToEndOfFile(); h.write(line.data(using: .utf8)!); h.closeFile()
        } else {
            FileManager.default.createFile(atPath: home + "/timeline.jsonl", contents: line.data(using: .utf8))
        }
    }

    func applicationDidFinishLaunching(_ n: Notification) {
        mark("window_start")
        buildMenu()
        let cfg = WKWebViewConfiguration()
        web = WKWebView(frame: .zero, configuration: cfg)
        web.navigationDelegate = self
        web.uiDelegate = self
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: opts.width, height: opts.height),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = opts.title
        if opts.appearance == "light" { window.appearance = NSAppearance(named: .aqua) }
        if opts.appearance == "dark" { window.appearance = NSAppearance(named: .darkAqua) }
        window.contentMinSize = NSSize(width: opts.minWidth, height: opts.minHeight)
        let container = NSView()
        web.frame = container.bounds
        web.autoresizingMask = [.width, .height]
        container.addSubview(web)
        if !opts.splash.isEmpty { addSplash(to: container) }
        window.contentView = container
        window.delegate = self
        window.center()
        window.setFrameAutosaveName(opts.title)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        if opts.serverExec.isEmpty { load() } else { startServer() }
        if opts.parentPid > 0 {
            Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { _ in
                if kill(self.opts.parentPid, 0) != 0 { NSApp.terminate(nil) }
            }
        }
    }

    func load() {
        guard let u = URL(string: opts.url) else { return }
        web.load(URLRequest(url: u))
    }

    var splashView: NSView?

    /// Cream screen with the mascot and a short line, shown until the first page has loaded (a cold start can take about 10 seconds).
    func addSplash(to container: NSView) {
        let v = NSView()
        v.wantsLayer = true
        v.layer?.backgroundColor = NSColor(srgbRed: 1.0, green: 0.945, blue: 0.839, alpha: 1).cgColor     // cream #fff1d6
        v.frame = container.bounds
        v.autoresizingMask = [.width, .height]
        let brown = NSColor(srgbRed: 0.231, green: 0.137, blue: 0.078, alpha: 1)                         // #3b2314
        let stack = NSStackView()
        stack.orientation = .vertical
        stack.alignment = .centerX
        stack.spacing = 18
        stack.translatesAutoresizingMaskIntoConstraints = false
        if let img = NSImage(contentsOfFile: opts.splash) {
            let iv = NSImageView(image: img)
            iv.imageScaling = .scaleProportionallyUpOrDown
            iv.translatesAutoresizingMaskIntoConstraints = false
            iv.widthAnchor.constraint(equalToConstant: 150).isActive = true
            iv.heightAnchor.constraint(equalToConstant: 150).isActive = true
            stack.addArrangedSubview(iv)
        }
        if !opts.splashText.isEmpty {
            let label = NSTextField(labelWithString: opts.splashText)
            let base = NSFont.systemFont(ofSize: 22, weight: .bold)
            label.font = base.fontDescriptor.withDesign(.rounded).flatMap { NSFont(descriptor: $0, size: 22) } ?? base
            label.textColor = brown
            stack.addArrangedSubview(label)
        }
        v.addSubview(stack)
        NSLayoutConstraint.activate([stack.centerXAnchor.constraint(equalTo: v.centerXAnchor), stack.centerYAnchor.constraint(equalTo: v.centerYAnchor)])
        container.addSubview(v)
        splashView = v
    }

    /// The HTML is loaded, but a big page (the chat) can take many more seconds to draw. Keep the loading screen until some text is on screen.
    func webView(_ w: WKWebView, didFinish navigation: WKNavigation!) {
        mark("page_loaded")
        if splashView != nil { waitForText(tries: 0) }
    }

    func waitForText(tries: Int) {
        guard let v = splashView else { return }
        web.evaluateJavaScript("document.body ? document.body.innerText.trim().length : 0") { result, _ in
            let n = (result as? Int) ?? 0
            if n > 0 || tries >= 240 {                                  // text is showing, or 60 seconds have passed
                self.splashView = nil
                self.mark("splash_hidden")
                NSAnimationContext.runAnimationGroup({ c in c.duration = 0.25; v.animator().alphaValue = 0 }, completionHandler: { v.removeFromSuperview() })
            } else {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.25) { self.waitForText(tries: tries + 1) }
            }
        }
    }

    // The page's server may take a moment to start. Try again a few times, quietly.
    func webView(_ w: WKWebView, didFailProvisionalNavigation nav: WKNavigation!, withError e: Error) {
        retries += 1
        if retries <= 40 { DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { self.load() } }
    }

    // Links to other sites open in the browser, not inside this window.
    func webView(_ w: WKWebView, decidePolicyFor action: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        if let u = action.request.url, let host = u.host, let base = URL(string: opts.url), host != base.host,
           action.navigationType == .linkActivated {
            NSWorkspace.shared.open(u)
            decisionHandler(.cancel)
            return
        }
        decisionHandler(.allow)
    }

    func webView(_ w: WKWebView, createWebViewWith c: WKWebViewConfiguration, for a: WKNavigationAction, windowFeatures f: WKWindowFeatures) -> WKWebView? {
        if let u = a.request.url { NSWorkspace.shared.open(u) }
        return nil
    }

    // Pages ask for files (document upload). Show the normal Mac file picker.
    func webView(_ w: WKWebView, runOpenPanelWith p: WKOpenPanelParameters, initiatedByFrame f: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = p.allowsMultipleSelection
        panel.canChooseDirectories = p.allowsDirectories
        panel.beginSheetModal(for: window) { r in completionHandler(r == .OK ? panel.urls : nil) }
    }

    func windowWillClose(_ n: Notification) { NSApp.terminate(nil) }
    func applicationShouldTerminateAfterLastWindowClosed(_ s: NSApplication) -> Bool { true }

    func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem(); main.addItem(appItem)
        let app = NSMenu(); appItem.submenu = app
        app.addItem(withTitle: "Hide \(opts.title)", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        app.addItem(.separator())
        app.addItem(withTitle: "Quit \(opts.title)", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let editItem = NSMenuItem(); main.addItem(editItem)
        let edit = NSMenu(title: "Edit"); editItem.submenu = edit
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        let winItem = NSMenuItem(); main.addItem(winItem)
        let win = NSMenu(title: "Window"); winItem.submenu = win
        win.addItem(withTitle: "Minimize", action: #selector(NSWindow.miniaturize(_:)), keyEquivalent: "m")
        win.addItem(withTitle: "Close", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        NSApp.mainMenu = main
        NSApp.windowsMenu = win
    }
}

let options = parse()
if options.url.isEmpty && options.serverExec.isEmpty {
    FileHandle.standardError.write(Data("usage: CaleWindow --url URL --title NAME   or   --server-exec PATH --server-arg ARG\n".utf8))
    exit(2)
}
let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = Delegate(options)
app.delegate = delegate
// A stop signal (logout, restart, kill) closes the window the normal way, so the server stops too.
var signalSources: [DispatchSourceSignal] = []
for sig in [SIGTERM, SIGINT, SIGHUP] {
    signal(sig, SIG_IGN)
    let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    src.setEventHandler { NSApp.terminate(nil) }
    src.resume()
    signalSources.append(src)
}
app.run()
