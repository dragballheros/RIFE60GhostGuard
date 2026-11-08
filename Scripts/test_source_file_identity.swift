import Foundation

@main
enum SourceFileIdentityTests {
    static func main() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("source-identity-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let first = directory.appendingPathComponent("first.mov")
        let same = directory.appendingPathComponent("same.mov")
        let different = directory.appendingPathComponent("different.mov")
        try Data(repeating: 42, count: 1_000_123).write(to: first)
        try Data(repeating: 42, count: 1_000_123).write(to: same)
        try Data(repeating: 43, count: 1_000_123).write(to: different)

        guard SourceFileIdentity.filesMatch(first, same) else {
            fatalError("FAIL: identical source bytes must permit recovery")
        }
        guard !SourceFileIdentity.filesMatch(first, different) else {
            fatalError("FAIL: different source bytes must not reuse recovery")
        }
        guard SourceFileIdentity.filesMatch(first, first) else {
            fatalError("FAIL: same existing source URL should match")
        }
        print("PASS: recovery only matches byte-identical source media")
    }
}
