import AppKit
import Foundation

let size = NSSize(width: 1024, height: 1024)
let image = NSImage(size: size)
image.lockFocus()

guard let ctx = NSGraphicsContext.current?.cgContext else {
    fatalError("No graphics context")
}

ctx.setAllowsAntialiasing(true)
ctx.setShouldAntialias(true)

let rect = CGRect(x: 0, y: 0, width: 1024, height: 1024)

// Opaque near-black/navy base so the final app icon has no alpha channel.
ctx.setFillColor(NSColor(calibratedRed: 0.015, green: 0.025, blue: 0.07, alpha: 1).cgColor)
ctx.fill(rect)

// Deep blue-violet vertical glow.
let bgColors = [
    NSColor(calibratedRed: 0.01, green: 0.10, blue: 0.28, alpha: 1).cgColor,
    NSColor(calibratedRed: 0.015, green: 0.025, blue: 0.08, alpha: 1).cgColor,
    NSColor(calibratedRed: 0.12, green: 0.02, blue: 0.24, alpha: 1).cgColor
] as CFArray
if let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: bgColors, locations: [0, 0.55, 1]) {
    ctx.drawLinearGradient(gradient, start: CGPoint(x: 100, y: 1000), end: CGPoint(x: 900, y: 20), options: [])
}

// Neon rounded border.
let borderRect = CGRect(x: 34, y: 34, width: 956, height: 956)
let borderPath = CGPath(roundedRect: borderRect, cornerWidth: 185, cornerHeight: 185, transform: nil)
ctx.addPath(borderPath)
ctx.setLineWidth(12)
ctx.setStrokeColor(NSColor(calibratedRed: 0.10, green: 0.65, blue: 1.0, alpha: 0.95).cgColor)
ctx.strokePath()
ctx.addPath(borderPath)
ctx.setLineWidth(5)
ctx.setStrokeColor(NSColor(calibratedRed: 0.80, green: 0.20, blue: 1.0, alpha: 0.9).cgColor)
ctx.strokePath()

// Motion streaks.
for i in 0..<18 {
    let y = 610 + CGFloat(i) * 10
    let alpha = max(0.04, 0.30 - CGFloat(i) * 0.012)
    ctx.setStrokeColor(NSColor(calibratedRed: i % 2 == 0 ? 0.0 : 0.65,
                               green: i % 2 == 0 ? 0.70 : 0.18,
                               blue: 1.0,
                               alpha: alpha).cgColor)
    ctx.setLineWidth(i % 3 == 0 ? 4 : 2)
    ctx.move(to: CGPoint(x: 86, y: y))
    ctx.addLine(to: CGPoint(x: 575, y: y + CGFloat(i % 5 - 2) * 2))
    ctx.strokePath()
}

// Layered play triangles to suggest interpolation frames.
func drawTriangle(centerX: CGFloat, centerY: CGFloat, width: CGFloat, height: CGFloat, alpha: CGFloat, hueShift: CGFloat) {
    let path = CGMutablePath()
    path.move(to: CGPoint(x: centerX - width * 0.34, y: centerY - height * 0.5))
    path.addLine(to: CGPoint(x: centerX - width * 0.34, y: centerY + height * 0.5))
    path.addLine(to: CGPoint(x: centerX + width * 0.55, y: centerY))
    path.closeSubpath()

    ctx.saveGState()
    ctx.addPath(path)
    ctx.setFillColor(NSColor(calibratedRed: 0.1 + hueShift,
                             green: 0.58,
                             blue: 1.0,
                             alpha: alpha).cgColor)
    ctx.fillPath()
    ctx.addPath(path)
    ctx.setLineWidth(3)
    ctx.setStrokeColor(NSColor(calibratedRed: 0.75,
                               green: 0.85,
                               blue: 1.0,
                               alpha: min(1, alpha + 0.25)).cgColor)
    ctx.strokePath()
    ctx.restoreGState()
}

for i in 0..<5 {
    drawTriangle(centerX: 330 + CGFloat(i) * 42,
                 centerY: 704,
                 width: 250,
                 height: 310,
                 alpha: 0.12 + CGFloat(i) * 0.07,
                 hueShift: CGFloat(i) * 0.02)
}

// Main play triangle with cyan-to-violet gradient.
let main = CGMutablePath()
main.move(to: CGPoint(x: 435, y: 542))
main.addLine(to: CGPoint(x: 435, y: 865))
main.addLine(to: CGPoint(x: 755, y: 704))
main.closeSubpath()
ctx.saveGState()
ctx.addPath(main)
ctx.clip()
if let playGradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: [
    NSColor(calibratedRed: 0.10, green: 0.90, blue: 1.0, alpha: 1).cgColor,
    NSColor(calibratedRed: 0.10, green: 0.28, blue: 1.0, alpha: 1).cgColor,
    NSColor(calibratedRed: 0.95, green: 0.20, blue: 1.0, alpha: 1).cgColor
] as CFArray, locations: [0, 0.55, 1]) {
    ctx.drawLinearGradient(playGradient, start: CGPoint(x: 440, y: 860), end: CGPoint(x: 730, y: 550), options: [])
}
ctx.restoreGState()
ctx.addPath(main)
ctx.setLineWidth(8)
ctx.setStrokeColor(NSColor.white.withAlphaComponent(0.88).cgColor)
ctx.strokePath()

// Star glint at the play point.
ctx.setStrokeColor(NSColor.white.cgColor)
ctx.setLineWidth(4)
ctx.move(to: CGPoint(x: 730, y: 704)); ctx.addLine(to: CGPoint(x: 795, y: 704)); ctx.strokePath()
ctx.move(to: CGPoint(x: 762, y: 671)); ctx.addLine(to: CGPoint(x: 762, y: 737)); ctx.strokePath()

func centeredText(_ text: String, y: CGFloat, font: NSFont, color: NSColor, kern: CGFloat = 0) {
    let style = NSMutableParagraphStyle()
    style.alignment = .center
    let attrs: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: color,
        .paragraphStyle: style,
        .kern: kern
    ]
    let r = CGRect(x: 40, y: y, width: 944, height: font.pointSize * 1.6)
    text.draw(in: r, withAttributes: attrs)
}

// 60 FPS marking.
centeredText("60", y: 592, font: NSFont.systemFont(ofSize: 150, weight: .heavy), color: .white)
centeredText("FPS", y: 548, font: NSFont.systemFont(ofSize: 58, weight: .heavy), color: NSColor(calibratedRed: 0.72, green: 0.88, blue: 1, alpha: 1), kern: 4)

// Main RIFE title.
centeredText("RIFE", y: 300, font: NSFont.systemFont(ofSize: 205, weight: .black), color: NSColor(calibratedRed: 0.83, green: 0.94, blue: 1, alpha: 1), kern: -8)
centeredText("GHOST  GUARD", y: 245, font: NSFont.systemFont(ofSize: 48, weight: .bold), color: NSColor(calibratedRed: 0.82, green: 0.90, blue: 1, alpha: 1), kern: 10)

// Curved neon filmstrip across the bottom.
ctx.saveGState()
ctx.setLineCap(.round)
let film = CGMutablePath()
film.move(to: CGPoint(x: 85, y: 165))
film.addCurve(to: CGPoint(x: 945, y: 105), control1: CGPoint(x: 345, y: 40), control2: CGPoint(x: 690, y: 250))
ctx.addPath(film)
ctx.setLineWidth(46)
ctx.setStrokeColor(NSColor(calibratedRed: 0.18, green: 0.55, blue: 1.0, alpha: 0.42).cgColor)
ctx.strokePath()
ctx.addPath(film)
ctx.setLineWidth(7)
ctx.setStrokeColor(NSColor(calibratedRed: 0.65, green: 0.30, blue: 1.0, alpha: 0.95).cgColor)
ctx.strokePath()
ctx.restoreGState()

// Film perforations.
for i in 0..<12 {
    let x = 110 + CGFloat(i) * 72
    let y = 142 + sin(CGFloat(i) * 0.7) * 24
    let hole = CGRect(x: x, y: y, width: 30, height: 17)
    ctx.setFillColor(NSColor.black.withAlphaComponent(0.72).cgColor)
    ctx.fill(CGPath(roundedRect: hole, cornerWidth: 4, cornerHeight: 4, transform: nil))
}

image.unlockFocus()

let outDir = URL(fileURLWithPath: "Assets.xcassets/AppIcon.appiconset")
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

if let tiff = image.tiffRepresentation,
   let rep = NSBitmapImageRep(data: tiff),
   let png = rep.representation(using: .png, properties: [.compressionFactor: 1.0]) {
    try png.write(to: outDir.appendingPathComponent("AppIcon-1024.png"))
} else {
    fatalError("Could not encode icon PNG")
}

let contents = """
{
  "images" : [
    {
      "filename" : "AppIcon-1024.png",
      "idiom" : "universal",
      "platform" : "ios",
      "size" : "1024x1024"
    }
  ],
  "info" : {
    "author" : "xcode",
    "version" : 1
  }
}
"""
try contents.write(to: outDir.appendingPathComponent("Contents.json"), atomically: true, encoding: .utf8)
print("Generated Assets.xcassets/AppIcon.appiconset/AppIcon-1024.png")
