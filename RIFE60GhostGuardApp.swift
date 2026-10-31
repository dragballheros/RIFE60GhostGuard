import SwiftUI

final class RIFE60AppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        guard identifier == H3Tier1BackgroundTransfer.identifier else { return }
        // Store the handler before the session exists. Creating the session earlier
        // delivers the shard event first, the handler is never called, and iOS
        // relaunches the app in a loop while shard 1 is still attached.
        H3Tier1BackgroundTransfer.shared.beginBackgroundEvents(completionHandler)
    }

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        return true
    }
}

@main
struct RIFE60GhostGuardApp: App {
    @UIApplicationDelegateAdaptor(RIFE60AppDelegate.self) private var appDelegate

    init() {
        CrashCapture.install()
        StorageMigration.removeRetiredVACEWeights()
        DiagnosticsExport.ensureFolder()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
