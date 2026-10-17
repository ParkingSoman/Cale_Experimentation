// Draws Cale's mascot (cale/brand/carrot.svg) into a PNG. Run at build time. One picture of the carrot for everything.
//   MascotIcon carrot.svg out.png SIZE [BGHEX]     BGHEX like ffd66b gives a rounded-square app icon. Leave it out for a see-through PNG.
import Cocoa

let a = CommandLine.arguments
guard a.count >= 4, let svg = NSImage(contentsOfFile: a[1]) else { FileHandle.standardError.write(Data("usage: MascotIcon carrot.svg out.png SIZE [BGHEX]\n".utf8)); exit(2) }
let n = Int(a[3]) ?? 1024
let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: n, pixelsHigh: n, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                           isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
let full = NSRect(x: 0, y: 0, width: n, height: n)
var art = full
if a.count >= 5, let v = UInt32(a[4], radix: 16) {
    let color = NSColor(srgbRed: CGFloat((v >> 16) & 255) / 255, green: CGFloat((v >> 8) & 255) / 255, blue: CGFloat(v & 255) / 255, alpha: 1)
    let inset = CGFloat(n) * 0.045
    let tile = full.insetBy(dx: inset, dy: inset)
    let r = tile.width * 0.2237
    color.setFill()
    NSBezierPath(roundedRect: tile, xRadius: r, yRadius: r).fill()
    art = tile.insetBy(dx: tile.width * 0.11, dy: tile.height * 0.11)
}
svg.draw(in: art)
NSGraphicsContext.restoreGraphicsState()
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: a[2]))
