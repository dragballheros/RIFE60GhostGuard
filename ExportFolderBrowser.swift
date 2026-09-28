import SwiftUI
import UIKit
import UniformTypeIdentifiers

/// Opens the system Files browser directly at the folder used for completed renders.
/// This is intentionally a browser, not another import/export step: selecting a movie
/// simply closes the browser and leaves the persisted render untouched.
struct ExportFolderBrowser: UIViewControllerRepresentable {
    @Binding var isPresented: Bool

    func makeCoordinator() -> Coordinator {
        Coordinator(isPresented: $isPresented)
    }

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(
            forOpeningContentTypes: [.movie, .mpeg4Movie, .quickTimeMovie, .video],
            asCopy: false
        )
        picker.delegate = context.coordinator
        picker.allowsMultipleSelection = false
        picker.shouldShowFileExtensions = true
        picker.directoryURL = resolvedExportFolder()
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) { }

    private func resolvedExportFolder() -> URL {
        // Keep this key in sync with VideoProcessorViewModel. Using the stored
        // security-scoped bookmark means this button also works with a custom
        // Files export folder selected by the user.
        let bookmarkKey = "RIFE60.ExportFolderBookmark.v1"
        if let data = UserDefaults.standard.data(forKey: bookmarkKey) {
            var stale = false
            if let folder = try? URL(
                resolvingBookmarkData: data,
                options: [.withoutUI],
                relativeTo: nil,
                bookmarkDataIsStale: &stale
            ) {
                return folder
            }
        }

        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let exports = documents.appendingPathComponent("Exports", isDirectory: true)
        try? FileManager.default.createDirectory(at: exports, withIntermediateDirectories: true)
        return exports
    }

    final class Coordinator: NSObject, UIDocumentPickerDelegate {
        @Binding private var isPresented: Bool

        init(isPresented: Binding<Bool>) {
            _isPresented = isPresented
        }

        func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) {
            isPresented = false
        }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            isPresented = false
        }
    }
}
