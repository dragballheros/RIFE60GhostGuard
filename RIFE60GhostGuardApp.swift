import SwiftUI

final class RIFE60AppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        guard identifier == H3Tier1BackgroundTransfer.identifier else { return }
        H3Tier1BackgroundTransfer.shared.backgroundCompletion = completionHandler
        H3Tier1BackgroundTransfer.shared.activate()
    }
}

@main
struct RIFE60GhostGuardApp: App {
    @UIApplicationDelegateAdaptor(RIFE60AppDelegate.self) private var appDelegate

    init() {
        StorageMigration.removeRetiredVACEWeights()
        DiagnosticsExport.ensureFolder()
        H3Tier1BackgroundTransfer.shared.activate()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
