import SwiftUI

@main
struct RIFE60GhostGuardApp: App {
    init() {
        StorageMigration.removeRetiredVACEWeights()
        DiagnosticsExport.ensureFolder()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
