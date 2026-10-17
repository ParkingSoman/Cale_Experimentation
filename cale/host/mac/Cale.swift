// Cale.app: the program a person double-clicks.
//   - Cale is not installed yet: show "Planting Cale", run the installer, then "You are all set".
//   - Cale is installed: start it and open the chat. (No window of its own is needed for that.)
// Plain AppKit. No web view, no address, no Terminal.
//
// The installer is run through cale/setup_runner.py, which prints:  PCT <n> <label> | DONE | ERROR <line>
import Cocoa

let kale = NSColor(srgbRed: 0.18, green: 0.36, blue: 0.31, alpha: 1)        // #2f5d50
let leaf = NSColor(srgbRed: 0.35, green: 0.58, blue: 0.40, alpha: 1)
let cream = NSColor(srgbRed: 0.97, green: 0.96, blue: 0.93, alpha: 1)        // #f7f5ee
let soil = NSColor(srgbRed: 0.43, green: 0.31, blue: 0.22, alpha: 1)
let carrot = NSColor(srgbRed: 0.91, green: 0.51, blue: 0.17, alpha: 1)

let home = ProcessInfo.processInfo.environment["CALE_HOME"]
    ?? (NSHomeDirectory() + "/Library/Application Support/Cale")
let payload = Bundle.main.resourcePath! + "/payload"

/// A small garden: soil, a stem and leaves that grow with `growth` (0 to 1). When `ripe`, a carrot shows beside it.
final class Garden: NSView {
    var growth = 0.0 { didSet { needsDisplay = true } }
    var ripe = false { didSet { needsDisplay = true } }

    override func draw(_ r: NSRect) {
        let w = bounds.width, h = bounds.height
        soil.setFill()
        NSBezierPath(roundedRect: NSRect(x: w * 0.12, y: 0, width: w * 0.76, height: h * 0.14), xRadius: 10, yRadius: 10).fill()
        let baseX = w * (ripe ? 0.38 : 0.5), baseY = h * 0.13
        let stem = NSBezierPath()
        stem.lineWidth = 5
        stem.lineCapStyle = .round
        leaf.setStroke()
        stem.move(to: NSPoint(x: baseX, y: baseY))
        let top = baseY + (h * 0.62) * CGFloat(growth)
        stem.line(to: NSPoint(x: baseX, y: top))
        stem.stroke()
        // Kale leaves: pairs at fixed spacing along the stem. Each pair unfurls once the stem reaches it.
        let spacing = (h * 0.62 - 14) / 6
        for i in 0..<6 {
            let y = baseY + 14 + CGFloat(i) * spacing
            guard top >= y else { continue }
            let unfurl = min(1, (top - y) / 30)
            let size = CGFloat(30 - 3 * Double(i)) * unfurl
            for side in [-1.0, 1.0] {
                let p = NSBezierPath()
                let x = baseX
                p.move(to: NSPoint(x: x, y: y))
                p.curve(to: NSPoint(x: x + CGFloat(side) * size * 1.6, y: y + size * 0.5),
                        controlPoint1: NSPoint(x: x + CGFloat(side) * size * 0.4, y: y + size * 0.9),
                        controlPoint2: NSPoint(x: x + CGFloat(side) * size * 1.3, y: y + size * 1.0))
                p.curve(to: NSPoint(x: x, y: y),
                        controlPoint1: NSPoint(x: x + CGFloat(side) * size * 1.2, y: y - size * 0.3),
                        controlPoint2: NSPoint(x: x + CGFloat(side) * size * 0.4, y: y - size * 0.2))
                (i % 2 == 0 ? leaf : kale).setFill()
                p.fill()
            }
        }
        if ripe {
            let cx = w * 0.68
            let body = NSBezierPath()
            body.move(to: NSPoint(x: cx - 16, y: baseY + 70))
            body.line(to: NSPoint(x: cx + 16, y: baseY + 70))
            body.line(to: NSPoint(x: cx, y: baseY - 2))
            body.close()
            carrot.setFill()
            body.fill()
            leaf.setFill()
            for dx in [-10.0, 0.0, 10.0] {
                let l = NSBezierPath()
                l.move(to: NSPoint(x: cx + CGFloat(dx) * 0.4, y: baseY + 70))
                l.line(to: NSPoint(x: cx + CGFloat(dx) * 1.6, y: baseY + 100))
                l.line(to: NSPoint(x: cx + CGFloat(dx) * 0.4 + 5, y: baseY + 70))
                l.close()
                l.fill()
            }
        }
    }
}

/// A progress bar in the garden colours (the system bar follows the person's accent colour).
final class Meter: NSView {
    var value = 0.0 { didSet { needsDisplay = true } }
    override func draw(_ r: NSRect) {
        NSColor(srgbRed: 0.87, green: 0.86, blue: 0.80, alpha: 1).setFill()
        NSBezierPath(roundedRect: bounds, xRadius: bounds.height / 2, yRadius: bounds.height / 2).fill()
        let w = max(bounds.height, bounds.width * CGFloat(min(1, value / 100)))
        leaf.setFill()
        NSBezierPath(roundedRect: NSRect(x: 0, y: 0, width: w, height: bounds.height), xRadius: bounds.height / 2, yRadius: bounds.height / 2).fill()
    }
}

final class App: NSObject, NSApplicationDelegate, NSWindowDelegate {
    var window: NSWindow!
    let title = NSTextField(labelWithString: "Planting Cale")
    let detail = NSTextField(labelWithString: "This takes a few minutes. You can keep using your computer.")
    let bar = Meter()
    let garden = Garden()
    let button = NSButton(title: "Start chatting", target: nil, action: nil)
    var finished = false

    func applicationDidFinishLaunching(_ n: Notification) {
        NSApp.mainMenu = menu()
        if installed() { startChat(); NSApp.terminate(nil); return }
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 520, height: 420), styleMask: [.titled, .closable, .miniaturizable],
                          backing: .buffered, defer: false)
        window.title = "Cale"
        window.delegate = self
        window.backgroundColor = cream
        window.center()
        let v = window.contentView!
        garden.frame = NSRect(x: 40, y: 190, width: 440, height: 200)
        v.addSubview(garden)
        title.font = .systemFont(ofSize: 24, weight: .semibold)
        title.textColor = kale
        title.alignment = .center
        title.frame = NSRect(x: 20, y: 140, width: 480, height: 34)
        v.addSubview(title)
        detail.font = .systemFont(ofSize: 14)
        detail.textColor = NSColor(white: 0.35, alpha: 1)
        detail.alignment = .center
        detail.frame = NSRect(x: 30, y: 105, width: 460, height: 22)
        v.addSubview(detail)
        bar.frame = NSRect(x: 60, y: 70, width: 400, height: 12)
        v.addSubview(bar)
        button.bezelStyle = .rounded
        button.bezelColor = kale
        button.attributedTitle = NSAttributedString(string: "Start chatting", attributes: [
            .foregroundColor: NSColor.white, .font: NSFont.systemFont(ofSize: 14, weight: .medium)])
        button.controlSize = .large
        button.target = self
        button.action = #selector(chatClicked)
        button.keyEquivalent = "\r"
        button.frame = NSRect(x: 190, y: 40, width: 140, height: 36)
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
                garden.growth = pct / 100
                if parts.count > 1 { title.stringValue = String(parts[1]) }
            }
        } else if line == "DONE" {
            finished = true
            bar.value = 100
            garden.growth = 1
            garden.ripe = true
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

    @objc func chatClicked() { startChat(); NSApp.terminate(nil) }

    /// Start Cale and open its chat. The chat launcher is installed with Cale. If it is missing, there is nothing to open yet.
    func startChat() {
        let py = home + "/venv/bin/python"
        let p = Process()
        p.executableURL = URL(fileURLWithPath: py)
        p.arguments = ["-m", "cale.chat"]
        var env = ProcessInfo.processInfo.environment
        env["CALE_HOME"] = home
        env["PYTHONPATH"] = home + "/runtime"
        p.environment = env
        p.currentDirectoryURL = URL(fileURLWithPath: home + "/runtime")
        try? p.run()
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
