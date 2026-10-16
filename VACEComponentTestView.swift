import SwiftUI
import CoreML
import Foundation
import UIKit

private enum VACEComponentChoice: String, CaseIterable, Identifiable {
    case encoder
    case decoder

    var id: String { rawValue }

    var title: String {
        switch self {
        case .encoder: return "Wan VAE Encoder"
        case .decoder: return "Wan VAE Decoder"
        }
    }

    var subtitle: String {
        switch self {
        case .encoder: return "5-frame video tensor • 64 × 64 test shape"
        case .decoder: return "5 latent frames • 8 × 8 test shape"
        }
    }

    var packageName: String {
        switch self {
        case .encoder: return "WanVAEEncoder-5x64"
        case .decoder: return "WanVAEDecoder-5x8"
        }
    }
}

/// Runs an on-device Core ML smoke test against component packages bundled in
/// this test IPA. No model file picker or network access is required.
struct VACEComponentTestView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var selectedComponent: VACEComponentChoice = .encoder
    @State private var isTesting = false
    @State private var report = "Choose the bundled encoder or decoder and run its local smoke test."

    var body: some View {
        NavigationStack {
            Form {
                Section("On-device component models") {
                    Label("Models bundled in this test IPA", systemImage: "iphone")
                    Text("The fixed-shape Wan VAE encoder and decoder test packages are included with this build. You do not need to import a model or download anything on the phone.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    ForEach(VACEComponentChoice.allCases) { choice in
                        Button {
                            guard !isTesting else { return }
                            selectedComponent = choice
                        } label: {
                            HStack(spacing: 12) {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(choice.title)
                                        .foregroundStyle(.primary)
                                    Text(choice.subtitle)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer(minLength: 8)
                                Image(systemName: selectedComponent == choice ? "checkmark.circle.fill" : "circle")
                                    .foregroundStyle(selectedComponent == choice ? Color.accentColor : Color.secondary)
                            }
                            .contentShape(Rectangle())
                            .padding(.vertical, 6)
                        }
                        .buttonStyle(.plain)
                        .disabled(isTesting)
                        .accessibilityAddTraits(selectedComponent == choice ? .isSelected : [])
                    }

                    if !Self.hasBundledPackage(named: selectedComponent.packageName) {
                        Text("This build does not contain the selected component package. Install the VACE component-test IPA built from the test branch.")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }

                    Button {
                        guard !isTesting else { return }
                        guard let packageURL = Self.bundledPackageURL(named: selectedComponent.packageName) else {
                            report = "FAIL\nThe \(selectedComponent.title) package is missing from this app bundle."
                            return
                        }
                        let title = selectedComponent.title
                        isTesting = true
                        report = "Compiling and running \(title) locally… Keep the app in the foreground."
                        Task {
                            let result = await Task.detached(priority: .userInitiated) {
                                Self.runSmokeTest(at: packageURL)
                            }.value
                            report = result
                            isTesting = false
                        }
                    } label: {
                        if isTesting {
                            ProgressView("Running on device…")
                        } else {
                            Label("Run \(selectedComponent.title) smoke test", systemImage: "play.circle")
                        }
                    }
                    .disabled(isTesting || !Self.hasBundledPackage(named: selectedComponent.packageName))
                }

                Section("Test report") {
                    Text(report)
                        .font(.system(.footnote, design: .monospaced))
                        .textSelection(.disabled)
                    Button {
                        UIPasteboard.general.string = report
                    } label: {
                        Label("Copy report", systemImage: "doc.on.doc")
                    }
                    .disabled(report.isEmpty || isTesting)
                }

                Section("What this proves") {
                    Text("PASS means the selected fixed-shape VAE component compiled and produced outputs for a zero-valued test input. It does not prove numerical parity, useful video quality, peak-memory headroom, sustained thermal performance, or complete VACE inference.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("VACE Component Lab")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") { dismiss() }.disabled(isTesting)
                }
            }
        }
    }

    private static func bundledPackageURL(named name: String) -> URL? {
        let candidates: [URL] = [
            Bundle.main.url(forResource: name, withExtension: "mlpackage"),
            Bundle.main.resourceURL?.appendingPathComponent("Models", isDirectory: true)
                .appendingPathComponent("\(name).mlpackage", isDirectory: true),
            Bundle.main.resourceURL?.appendingPathComponent("\(name).mlpackage", isDirectory: true)
        ].compactMap { $0 }

        return candidates.first { url in
            var isDirectory: ObjCBool = false
            return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
                && isDirectory.boolValue
                && FileManager.default.fileExists(atPath: url.appendingPathComponent("Manifest.json").path)
        }
    }

    private static func hasBundledPackage(named name: String) -> Bool {
        bundledPackageURL(named: name) != nil
    }

    nonisolated private static func runSmokeTest(at packageURL: URL) -> String {
        let started = Date()
        do {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: packageURL.path, isDirectory: &isDirectory),
                  isDirectory.boolValue,
                  packageURL.pathExtension == "mlpackage" else {
                return "FAIL\nThe bundled resource is not an accessible .mlpackage directory."
            }

            let compileStarted = Date()
            let compiledURL = try MLModel.compileModel(at: packageURL)
            let compileSeconds = Date().timeIntervalSince(compileStarted)
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            let model = try MLModel(contentsOf: compiledURL, configuration: configuration)

            let descriptions = model.modelDescription.inputDescriptionsByName
            guard !descriptions.isEmpty else {
                return "FAIL\nThe package exposes no model inputs."
            }

            var featureValues: [String: MLFeatureValue] = [:]
            var inputSummary: [String] = []
            var allocatedInputBytes: Int64 = 0

            for name in descriptions.keys.sorted() {
                guard let description = descriptions[name],
                      description.type == .multiArray,
                      let constraint = description.multiArrayConstraint else {
                    return "FAIL\nInput “\(name)” is not a fixed-shape multi-array."
                }

                let shape = constraint.shape
                guard !shape.isEmpty, shape.allSatisfy({ $0.intValue > 0 }) else {
                    return "FAIL\nInput “\(name)” has a dynamic or invalid shape: \(shape)."
                }

                let array = try MLMultiArray(shape: shape, dataType: constraint.dataType)
                for index in 0..<array.count {
                    array[index] = NSNumber(value: 0)
                }

                let bytesPerElement: Int64
                switch constraint.dataType {
                case .double: bytesPerElement = 8
                case .float32, .int32: bytesPerElement = 4
                case .float16: bytesPerElement = 2
                case .int8: bytesPerElement = 1
                @unknown default: bytesPerElement = 0
                }
                allocatedInputBytes += Int64(array.count) * bytesPerElement
                inputSummary.append("\(name): shape=\(shape.map(\.intValue)), type=\(constraint.dataType), elements=\(array.count)")
                featureValues[name] = MLFeatureValue(multiArray: array)
            }

            let provider = try MLDictionaryFeatureProvider(dictionary: featureValues)
            let predictionStarted = Date()
            let prediction = try model.prediction(from: provider)
            let predictionSeconds = Date().timeIntervalSince(predictionStarted)
            let outputs = prediction.featureNames.sorted().map { name -> String in
                if let array = prediction.featureValue(for: name)?.multiArrayValue {
                    return "\(name): shape=\(array.shape.map(\.intValue)), type=\(array.dataType)"
                }
                return "\(name): non-array output"
            }

            let totalSeconds = Date().timeIntervalSince(started)
            let inputMiB = Double(allocatedInputBytes) / 1_048_576.0
            return """
            PASS — local Core ML prediction completed
            Package: \(packageURL.lastPathComponent)
            Device physical-memory class: \(ByteCountFormatter.string(fromByteCount: Int64(ProcessInfo.processInfo.physicalMemory), countStyle: .memory))
            Compile time: \(String(format: "%.2f", compileSeconds)) s
            Prediction time: \(String(format: "%.2f", predictionSeconds)) s
            Total elapsed: \(String(format: "%.2f", totalSeconds)) s
            Input tensor allocation: \(String(format: "%.2f", inputMiB)) MiB
            Peak process memory: NOT measured by this smoke test
            Inputs:
            \(inputSummary.joined(separator: "\n"))
            Outputs:
            \(outputs.isEmpty ? "No outputs reported" : outputs.joined(separator: "\n"))

            LIMITATION: component-only test. Not full VACE, not numerical reference parity, not a 720p/1080p video test.
            """
        } catch {
            return """
            FAIL — local Core ML smoke test failed
            Package: \(packageURL.lastPathComponent)
            Error: \(String(reflecting: error))
            This is a component test; it does not indicate whether the full VACE pipeline is supported.
            """
        }
    }
}
