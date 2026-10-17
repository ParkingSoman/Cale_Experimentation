// Cale.app: the program a person double-clicks.
//   - Cale is not installed yet: show "Planting Cale" (a carrot grows out of the soil), run the installer, then "You are all set".
//   - Cale is installed: turn into Cale's chat window (the window program in this same app, so the Dock shows Cale and the carrot).
// Plain AppKit. No web view for these screens, no address, no Terminal.
// The look is in cale/brand/README.md. The carrot is cale/brand/carrot.svg.
//
// The installer is run through cale/setup_runner.py, which prints:  PCT <n> <label> | DONE | ERROR <line>
import Cocoa

let carrot = NSColor(srgbRed: 0.961, green: 0.541, blue: 0.122, alpha: 1)    // #f58a1f
let brown = NSColor(srgbRed: 0.231, green: 0.137, blue: 0.078, alpha: 1)     // #3b2314
let cream = NSColor(srgbRed: 1.0, green: 0.945, blue: 0.839, alpha: 1)       // #fff1d6
let soil = NSColor(srgbRed: 0.541, green: 0.353, blue: 0.231, alpha: 1)      // #8a5a3b
let track = NSColor(srgbRed: 0.97, green: 0.85, blue: 0.62, alpha: 1)

let home = ProcessInfo.processInfo.environment["CALE_HOME"]
    ?? (NSHomeDirectory() + "/Library/Application Support/Cale")
let payload = Bundle.main.resourcePath! + "/payload"

func rounded(_ size: CGFloat, _ weight: NSFont.Weight) -> NSFont {
    let base = NSFont.systemFont(ofSize: size, weight: weight)
    if let d = base.fontDescriptor.withDesign(.rounded) { return NSFont(descriptor: d, size: size) ?? base }
    return base
}

/// The scene: soil with a carrot growing out of it. `growth` is 0 to 1.
final class Field: NSView {
    var growth = 0.0 { didSet { needsDisplay = true } }
    let mascot = NSImage(contentsOfFile: payload + "/cale/brand/carrot.svg")

    override var clipsToBounds: Bool { get { true } set {} }

    override func draw(_ r: NSRect) {
        let w = bounds.width
        let soilTop: CGFloat = 64
        // a small sun, top right
        let sun = NSRect(x: w - 96, y: bounds.height - 76, width: 46, height: 46)
        brown.setStroke()
        for i in 0..<8 {
            let ang = Double(i) * .pi / 4
            let ray = NSBezierPath()
            ray.lineWidth = 5
            ray.lineCapStyle = .round
            ray.move(to: NSPoint(x: sun.midX + CGFloat(cos(ang)) * 32, y: sun.midY + CGFloat(sin(ang)) * 32))
            ray.line(to: NSPoint(x: sun.midX + CGFloat(cos(ang)) * 42, y: sun.midY + CGFloat(sin(ang)) * 42))
            ray.stroke()
        }
        let disc = NSBezierPath(ovalIn: sun)
        NSColor(srgbRed: 1.0, green: 0.84, blue: 0.42, alpha: 1).setFill()
        disc.fill()
        disc.lineWidth = 5
        disc.stroke()
        let side: CGFloat = 190
        if let m = mascot {
            let y0 = soilTop - side + 160 * CGFloat(min(1, max(0, growth)))          // hidden at 0, nearly all out at 1
            m.draw(in: NSRect(x: (w - side) / 2, y: y0, width: side, height: side))
        }
        // a hill of soil: the top of a wide oval. The view cuts off the rest.
        let mound = NSBezierPath(ovalIn: NSRect(x: 0, y: soilTop - 170, width: w, height: 170))
        soil.setFill()
        mound.fill()
        brown.setStroke()
        mound.lineWidth = 5
        mound.stroke()
        // a few pebbles and crumbs, for the cartoon feel
        NSColor(srgbRed: 0.43, green: 0.27, blue: 0.17, alpha: 1).setFill()
        for (x, y, s) in [(0.16, 30.0, 9.0), (0.30, 18.0, 7.0), (0.70, 24.0, 10.0), (0.84, 36.0, 7.0), (0.55, 12.0, 6.0)] {
            NSBezierPath(ovalIn: NSRect(x: w * CGFloat(x), y: CGFloat(y), width: CGFloat(s), height: CGFloat(s) * 0.7)).fill()
        }
    }
}

/// A progress bar with a thick brown outline, like a cartoon.
final class Meter: NSView {
    var value = 0.0 { didSet { needsDisplay = true } }
    override func draw(_ r: NSRect) {
        let box = bounds.insetBy(dx: 2, dy: 2)
        let radius = box.height / 2
        track.setFill()
        NSBezierPath(roundedRect: box, xRadius: radius, yRadius: radius).fill()
        let w = max(box.height, box.width * CGFloat(min(1, value / 100)))
        carrot.setFill()
        let fill = NSBezierPath(roundedRect: NSRect(x: box.minX, y: box.minY, width: w, height: box.height), xRadius: radius, yRadius: radius)
        fill.fill()
        brown.setStroke()
        fill.lineWidth = 3
        fill.stroke()
        let outline = NSBezierPath(roundedRect: box, xRadius: radius, yRadius: radius)
        outline.lineWidth = 3
        outline.stroke()
    }
}

/// A chunky button: carrot orange, brown outline, brown text.
final class ChunkyButton: NSButton {
    convenience init(_ text: String, target: AnyObject, action: Selector) {
        self.init(frame: .zero)
        self.target = target
        self.action = action
        isBordered = false
        wantsLayer = true
        layer?.backgroundColor = carrot.cgColor
        layer?.cornerRadius = 20
        layer?.borderColor = brown.cgColor
        layer?.borderWidth = 3
        attributedTitle = NSAttributedString(string: text, attributes: [.foregroundColor: brown, .font: rounded(16, .bold)])
        keyEquivalent = "\r"
    }
}

final class App: NSObject, NSApplicationDelegate, NSWindowDelegate {
    var window: NSWindow!
    let title = NSTextField(labelWithString: "Planting Cale")
    let detail = NSTextField(labelWithString: "This takes a few minutes. You can keep using your computer.")
    let bar = Meter()
    let field = Field()
    var button: ChunkyButton!
    var finished = false

    func applicationDidFinishLaunching(_ n: Notification) {
        NSApp.mainMenu = menu()
        if installed() { openChat(); return }
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 520, height: 460), styleMask: [.titled, .closable, .miniaturizable],
                          backing: .buffered, defer: false)
        window.title = "Cale"
        window.delegate = self
        window.backgroundColor = cream
        window.center()
        let v = window.contentView!
        field.frame = NSRect(x: 30, y: 200, width: 460, height: 240)
        v.addSubview(field)
        title.font = rounded(28, .bold)
        title.textColor = brown
        title.alignment = .center
        title.frame = NSRect(x: 20, y: 150, width: 480, height: 38)
        v.addSubview(title)
        detail.font = rounded(15, .regular)
        detail.textColor = brown.withAlphaComponent(0.75)
        detail.alignment = .center
        detail.frame = NSRect(x: 30, y: 118, width: 460, height: 22)
        v.addSubview(detail)
        bar.frame = NSRect(x: 60, y: 78, width: 400, height: 22)
        v.addSubview(bar)
        button = ChunkyButton("Start chatting", target: self, action: #selector(chatClicked))
        button.frame = NSRect(x: 160, y: 44, width: 200, height: 44)
        button.isHidden = true
        v.addSubview(button)
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        runSetup()
    }

    func installed() -> Bool {
        FileManager.default.fileExists(atPath: home + "/installed.ok") && FileManager.default.fileExists(atPath: home + "/venv/bin/python")
    }

    func runSetup() {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: payload + "/bin/uv")
        p.arguments = ["run", "--python", "3.12", "--no-project", "python", "cale/setup_runner.py"]
        p.currentDirectoryURL = URL(fileURLWithPath: payload)
        var env = ProcessInfo.processInfo.environment
        env["CALE_HOME"] = home
        p.environment = env
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        var buffer = Data()
        pipe.fileHandleForReading.readabilityHandler = { h in
            buffer.append(h.availableData)
            while let nl = buffer.firstIndex(of: 10) {
                let line = String(decoding: buffer[buffer.startIndex..<nl], as: UTF8.self)
                buffer.removeSubrange(buffer.startIndex...nl)
                DispatchQueue.main.async { self.handle(line) }
            }
        }
        p.terminationHandler = { q in
            DispatchQueue.main.async { if !self.finished && q.terminationStatus != 0 { self.fail("Cale could not finish. Open it again to try once more.") } }
        }
        do { try p.run() } catch { fail("Cale could not start its setup.") }
    }

    func handle(_ line: String) {
        if line.hasPrefix("PCT ") {
            let parts = line.dropFirst(4).split(separator: " ", maxSplits: 1)
            if let pct = Double(parts[0]) {
                bar.value = pct
                field.growth = pct / 100
                if parts.count > 1 { title.stringValue = String(parts[1]) }
            }
        } else if line == "DONE" {
            finished = true
            bar.value = 100
            field.growth = 1
            title.stringValue = "You are all set"
            detail.stringValue = "Cale is ready on this computer. Apps made for Cale will just work."
            bar.isHidden = true
            button.isHidden = false
        } else if line.hasPrefix("ERROR ") {
            fail("Cale could not finish. Open it again to try once more.")
        }
    }

    func fail(_ text: String) {
        finished = true
        title.stringValue = "Setup stopped"
        detail.stringValue = text
        bar.isHidden = true
    }

    @objc func chatClicked() { openChat() }

    /// Turn this program into Cale's chat window. The window program is inside this same app (Contents/MacOS/window), so the process
    /// keeps this app's name and carrot icon in the Dock. The chat server (cale.chat) is started by the window program.
    func openChat() {
        let host = (Bundle.main.executablePath! as NSString).deletingLastPathComponent + "/window"
        setenv("CALE_HOME", home, 1)
        setenv("PYTHONPATH", home + "/runtime", 1)
        let args = [host, "--title", "Cale", "--width", "1100", "--height", "760", "--min-width", "640", "--min-height", "480",
                    "--server-exec", home + "/venv/bin/python", "--server-arg", "-m", "--server-arg", "cale.chat",
                    "--server-cwd", home + "/runtime",
                    "--url-template", "http://127.0.0.1:{port}/?t={token}"]
        var cargs: [UnsafeMutablePointer<CChar>?] = args.map { strdup($0) }
        cargs.append(nil)
        execv(host, &cargs)                                            // only returns if it failed
        let a = NSAlert()
        a.messageText = "Cale could not open its chat."
        a.informativeText = "Open Cale again. If this keeps happening, restart your computer."
        a.runModal()
        NSApp.terminate(nil)
    }

    func windowWillClose(_ n: Notification) { NSApp.terminate(nil) }

    func menu() -> NSMenu {
        let main = NSMenu()
        let item = NSMenuItem(); main.addItem(item)
        let m = NSMenu(); item.submenu = m
        m.addItem(withTitle: "Hide Cale", action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        m.addItem(withTitle: "Quit Cale", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        return main
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)
let delegate = App()
app.delegate = delegate
app.run()
