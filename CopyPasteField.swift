import SwiftUI

/// Copy uses the pasteboard directly. Paste uses PasteButton so iOS does not show Allow Paste.
struct CopyPasteBar: View {
    let text: String
    var transform: (String) -> String = { $0 }
    let onPaste: (String) -> Void

    var body: some View {
        HStack(spacing: 12) {
            Button {
                UIPasteboard.general.string = text
            } label: {
                Label("Copy", systemImage: "doc.on.doc")
            }
            .buttonStyle(.bordered)
            PasteButton(payloadType: String.self) { values in
                guard let raw = values.first else { return }
                onPaste(transform(raw))
            }
            .labelStyle(.titleAndIcon)
        }
    }
}

extension View {
    func copyPasteBar(
        text: String,
        transform: @escaping (String) -> String = { $0 },
        onPaste: @escaping (String) -> Void
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            self
            CopyPasteBar(text: text, transform: transform, onPaste: onPaste)
        }
    }
}
