import SwiftUI
import CoreML
import UniformTypeIdentifiers
import Foundation
import UIKit

/// Runs a genuine on-device Core ML compile/predict smoke test for a converted
/// Wan VAE component. This is not the VACE transformer or a video-editing pipeline.
struct VACEComponentTestView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var showingModelImporter = false
    @State private var selectedModelURL: URL?
    @State private var isTesting = false
    @State private var report = "Choose a converted Wan VAE .mlpackage to begin. The test runs locally on this iPhone."

    private var packageType: UTType {
        UTType(filenameExtension: "mlpackage") ?? .package
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Scope") {
                    Label("Local Core ML only", systemImage: "iphone")
                    Text("This lab tests an exported VAE component by compiling the package and running a deterministic zero-input prediction on the device. It does not run VACE video editing, the 14B transformer, the text encoder, or a 720p clip.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("Use the encoder package (input: video tensor) or decoder package (input: latent tensor) from the GitHub Actions artifact named “wan-vace-ondevice-coreml-components”.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Section("Model package") {
                    Button {
                        showingModelImporter = true
                    } label: {
                        Label("Choose Core ML .mlpackage", systemImage: "folder")
                    }
                    if let selectedModelURL {
                        Text(selectedModelURL.lastPathComponent)
                            .font(.caption)
                            .textSelection(.enabled)
                    }
                    Button {
                        guard let selectedModelURL else { return }
                        isTesting = true
                        report = "Compiling and running locally… Keep the app in the foreground."
                        Task {
                            let result = await Task.detached(priority: .userInitiated) {
                                Self.runSmokeTest(at: selectedModelURL)
                            }.value
                            report = result
                            isTesting = false
                        }
                    } label: {
                        if isTesting {
                            ProgressView("Running on device…")
                        } else {
                            Label("Compile and run local smoke test", systemImage: "play.circle")
                        }
                    }
                    .disabled(selectedModelURL == nil || isTesting)
                }

                Section("Test report") {
                    Text(report)
                        .font(.system(.footnote, design: .monospaced))
                        .textSelection(.enabled)
                    Button {
                        UIPasteboard.general.string = report
                    } label: {
                        Label("Copy report", systemImage: "doc.on.doc")
                    }
                    .disabled(report.isEmpty || isTesting)
                }

                Section("Interpretation") {
                    Text("PASS means only that this selected component compiled and produced outputs for the generated fixed-shape zero input. It does not prove numerical parity, useful video quality, peak-memory headroom, sustained thermal performance, or complete VACE inference.")
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
            .fileImporter(
                isPresented: $showingModelImporter,
                allowedContentTypes: [packageType],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    selectedModelURL = urls.first
                    report = urls.first.map { "Selected: \($0.lastPathComponent)\nReady to compile and run a local smoke test." }
                        ?? "No model package selected."
                case .failure(let error):
                    report = "Model selection failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private static func runSmokeTest(at packageURL: URL) -> String {
        let started = Date()
        let hasAccess = packageURL.startAccessingSecurityScopedResource()
        defer {
            if hasAccess { packageURL.stopAccessingSecurityScopedResource() }
        }

        do {
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: packageURL.path, isDirectory: &isDirectory),
                  isDirectory.boolValue,
                  packageURL.pathExtension == "mlpackage" else {
                return "FAIL\nThe selected item is not an accessible .mlpackage directory."
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
                    return "FAIL\nInput “\(name)” is not a fixed-shape multi-array. This lab currently accepts the exported VAE tensor components only."
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
                case .double, .int64: bytesPerElement = 8
                case .float32, .int32: bytesPerElement = 4
                case .float16, .int16: bytesPerElement = 2
                case .int8, .uint8, .bool: bytesPerElement = 1
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
