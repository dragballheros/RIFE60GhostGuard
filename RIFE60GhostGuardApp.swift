import SwiftUI

@main
struct RIFE60GhostGuardApp: App {
    init() {
        StorageMigration.removeRetiredVACEWeights()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
