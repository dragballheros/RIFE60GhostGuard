import Darwin
import Foundation

private let crashStepCapacity = 512
private let crashStepBuffer: UnsafeMutablePointer<CChar> = {
    let buffer = UnsafeMutablePointer<CChar>.allocate(capacity: crashStepCapacity)
    buffer.initialize(repeating: 0, count: crashStepCapacity)
    return buffer
}()
private let crashFrames = UnsafeMutablePointer<UnsafeMutableRawPointer?>.allocate(capacity: 64)
private var crashFD: Int32 = -1

private let crashSignalPrefix = UnsafeMutablePointer<CChar>.allocate(capacity: 32)

@_cdecl("rife60_crash_signal")
func rife60CrashSignal(_ sig: Int32) {
    if crashFD >= 0 {
        var index = 0
        for byte in [UInt8]("signal ".utf8) {
            crashSignalPrefix[index] = CChar(bitPattern: byte)
            index += 1
        }
        var value = Int(sig < 0 ? -sig : sig)
        var reversed = [CChar](repeating: 0, count: 8)
        var digitCount = 0
        repeat {
            reversed[digitCount] = CChar(48 + (value % 10))
            value /= 10
            digitCount += 1
        } while value > 0 && digitCount < 8
        while digitCount > 0 {
            digitCount -= 1
            crashSignalPrefix[index] = reversed[digitCount]
            index += 1
        }
        crashSignalPrefix[index] = 10
        index += 1
        crashSignalPrefix[index] = 0
        _ = write(crashFD, crashSignalPrefix, index)
        index = 0
        for byte in [UInt8]("last step: ".utf8) {
            crashSignalPrefix[index] = CChar(bitPattern: byte)
            index += 1
        }
        crashSignalPrefix[index] = 0
        _ = write(crashFD, crashSignalPrefix, index)
        _ = write(crashFD, crashStepBuffer, strlen(crashStepBuffer))
        _ = write(crashFD, "\n", 1)
        let count = backtrace(crashFrames, 64)
        backtrace_symbols_fd(crashFrames, count, crashFD)
    }
    signal(sig, SIG_DFL)
    raise(sig)
}

private let crashExceptionHandler: @convention(c) (NSException) -> Void = { exception in
    let name = exception.name.rawValue
    let reason = exception.reason ?? ""
    let stack = exception.callStackSymbols.joined(separator: "\n")
    let text = """
    exception \(name): \(reason)
    \(CrashCapture.memoryLine())
    last step: \(String(cString: crashStepBuffer))
    \(stack)
    """
    if crashFD >= 0 {
        text.withCString { pointer in
            _ = write(crashFD, pointer, strlen(pointer))
            _ = write(crashFD, "\n", 1)
        }
    }
    DiagnosticsLogger.shared.logSync("CRASH \(text)")
}

enum CrashCapture {
    private static var installed = false

    static func install() {
        guard !installed else { return }
        installed = true
        let url = crashFileURL()
        if let existing = try? String(contentsOf: url, encoding: .utf8),
           !existing.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            DiagnosticsLogger.shared.log("Previous process crashed before it could finish:\n\(existing)")
        }
        try? FileManager.default.removeItem(at: url)
        FileManager.default.createFile(atPath: url.path, contents: nil)
        crashFD = open(url.path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
        NSSetUncaughtExceptionHandler(crashExceptionHandler)
        signal(SIGABRT, rife60CrashSignal)
        signal(SIGSEGV, rife60CrashSignal)
        signal(SIGBUS, rife60CrashSignal)
        signal(SIGILL, rife60CrashSignal)
        signal(SIGTRAP, rife60CrashSignal)
    }

    /// Persists the step, then updates the buffer a signal handler can write.
    static func step(_ text: String) {
        let line = "\(text) • \(memoryLine())"
        remember(line)
        DiagnosticsLogger.shared.logSync("H3 step: \(line)")
    }

    /// Updates the crash buffer without appending another log line.
    static func clearCaptureFile() {
        if crashFD >= 0 { _ = ftruncate(crashFD, 0) }
    }

    static func note(_ text: String) {
        remember("\(text) • \(memoryLine())")
    }

    static func memoryLine() -> String {
        let available = os_proc_available_memory()
        let physical = ProcessInfo.processInfo.physicalMemory
        return "available=\(available)B physical=\(physical)B"
    }

    private static func remember(_ text: String) {
        text.withCString { source in
            strncpy(crashStepBuffer, source, crashStepCapacity - 1)
            crashStepBuffer[crashStepCapacity - 1] = 0
        }
    }

    private static func crashFileURL() -> URL {
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let directory = documents.appendingPathComponent("Diagnostics", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent("crash-capture.log")
    }
}
