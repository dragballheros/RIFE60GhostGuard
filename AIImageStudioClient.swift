import Foundation
import Photos
import Security
import UIKit

struct AIImageStudioSettings: Codable {
    var runPodEndpointID = ""
    var checkpointName = "screenChantvMerge_v20.safetensors"
    var textEncoderName = "qwen_3_06b_base.safetensors"
    var vaeName = "qwen_image_vae.safetensors"
    var turboLoraName = "Turbo-ANIMA-v2.9.safetensors"
    var characterLoraName = "Ichinose_Chizuru.safetensors"
    var trainerBaseURL = ""

    static let storageKey = "RIFE60.AIImageStudio.Settings.v1"

    static func load() -> AIImageStudioSettings {
        guard let data = UserDefaults.standard.data(forKey: storageKey),
              let value = try? JSONDecoder().decode(AIImageStudioSettings.self, from: data) else {
            return AIImageStudioSettings()
        }
        return value
    }

    func save() throws {
        let data = try JSONEncoder().encode(self)
        UserDefaults.standard.set(data, forKey: Self.storageKey)
    }
}

enum AIImageStudioKeychain {
    private static let service = "com.za.rife60ghostguard.aiimagestudio"

    static func read(_ account: String) -> String {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data,
              let value = String(data: data, encoding: .utf8) else { return "" }
        return value
    }

    static func write(_ value: String, account: String) throws {
        let base: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
        let data = Data(value.utf8)
        let status = SecItemCopyMatching(base as CFDictionary, nil)
        if status == errSecSuccess {
            let updated = SecItemUpdate(base as CFDictionary, [kSecValueData as String: data] as CFDictionary)
            guard updated == errSecSuccess else {
                throw AIImageStudioError.keychain("Could not update the saved API credential (\(updated)).")
            }
        } else {
            var insert = base
            insert[kSecValueData as String] = data
            let added = SecItemAdd(insert as CFDictionary, nil)
            guard added == errSecSuccess else {
                throw AIImageStudioError.keychain("Could not save the API credential (\(added)).")
            }
        }
    }
}

struct AIImageStudioError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    static func keychain(_ message: String) -> AIImageStudioError { AIImageStudioError(message) }
    var errorDescription: String? { message }
}

struct AITrainingImage {
    let filename: String
    let data: Data
}

@MainActor
enum AIImageStudioClient {
    static func generate(
        settings: AIImageStudioSettings,
        apiKey: String,
        positivePrompt: String,
        negativePrompt: String,
        width: Int,
        height: Int,
        steps: Int,
        cfg: Double,
        shift: Double,
        samplerName: String,
        schedulerName: String,
        seed: Int64,
        turboWeight: Double,
        characterWeight: Double,
        enableCharacterLora: Bool,
        progress: (String) -> Void
    ) async throws -> Data {
        guard !settings.runPodEndpointID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AIImageStudioError("Enter your RunPod Serverless endpoint ID in Settings.")
        }
        guard !apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AIImageStudioError("Enter your RunPod API key in Settings. It is stored in iOS Keychain.")
        }
        guard !positivePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw AIImageStudioError("The positive prompt cannot be empty.")
        }

        let endpoint = "https://api.runpod.ai/v2/\(settings.runPodEndpointID.trimmingCharacters(in: .whitespacesAndNewlines))"
        let requestBody: [String: Any] = [
            "input": ["workflow": makeWorkflow(
                settings: settings,
                positivePrompt: positivePrompt,
                negativePrompt: negativePrompt,
                width: width,
                height: height,
                steps: steps,
                cfg: cfg,
                shift: shift,
                samplerName: samplerName,
                schedulerName: schedulerName,
                seed: seed,
                turboWeight: turboWeight,
                characterWeight: characterWeight,
                enableCharacterLora: enableCharacterLora
            )]
        ]
        var request = URLRequest(url: URL(string: endpoint + "/run")!)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)

        progress("Submitting ANIMA workflow to the GPU endpoint…")
        let (initialData, initialResponse) = try await URLSession.shared.data(for: request)
        let initial = try checkedJSONObject(data: initialData, response: initialResponse, context: "submit generation")
        guard let jobID = initial["id"] as? String else {
            throw AIImageStudioError("GPU endpoint did not return a job ID. Check the endpoint type and RunPod configuration.")
        }

        let deadline = Date().addingTimeInterval(20 * 60)
        while Date() < deadline {
            try Task.checkCancellation()
            try await Task.sleep(nanoseconds: 2_000_000_000)
            var statusRequest = URLRequest(url: URL(string: endpoint + "/status/" + jobID)!)
            statusRequest.httpMethod = "GET"
            statusRequest.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
            let (statusData, statusResponse) = try await URLSession.shared.data(for: statusRequest)
            let statusObject = try checkedJSONObject(data: statusData, response: statusResponse, context: "check generation status")
            let status = (statusObject["status"] as? String ?? "UNKNOWN").uppercased()
            if status == "FAILED" || status == "CANCELLED" || status == "TIMED_OUT" {
                let detail = statusObject["error"] as? String ?? "GPU worker status: \(status)"
                throw AIImageStudioError("Image generation failed: \(detail)")
            }
            if status == "COMPLETED" {
                progress("Downloading generated image…")
                guard let output = statusObject["output"] else {
                    throw AIImageStudioError("Generation finished but the worker returned no output.")
                }
                return try await extractImageData(output)
            }
            progress(status == "IN_QUEUE" ? "Waiting for GPU capacity…" : "Generating image… \(status.lowercased())")
        }
        throw AIImageStudioError("The generation job exceeded 20 minutes. Check the RunPod endpoint status before retrying.")
    }

    private static func makeWorkflow(
        settings: AIImageStudioSettings,
        positivePrompt: String,
        negativePrompt: String,
        width: Int,
        height: Int,
        steps: Int,
        cfg: Double,
        shift: Double,
        seed: Int64,
        turboWeight: Double,
        characterWeight: Double,
        enableCharacterLora: Bool
    ) -> [String: Any] {
        var graph: [String: Any] = [
            "1": ["class_type": "UNETLoader", "inputs": [
                "unet_name": settings.checkpointName, "weight_dtype": "default"
            ]],
            "2": ["class_type": "CLIPLoader", "inputs": [
                "clip_name": settings.textEncoderName, "type": "stable_diffusion", "device": "default"
            ]],
            "3": ["class_type": "VAELoader", "inputs": ["vae_name": settings.vaeName]],
            "4": ["class_type": "LoraLoader", "inputs": [
                "model": ["1", 0], "clip": ["2", 0],
                "lora_name": settings.turboLoraName,
                "strength_model": turboWeight, "strength_clip": turboWeight
            ]],
            "6": ["class_type": "CLIPTextEncode", "inputs": [
                "clip": ["4", 1], "text": positivePrompt
            ]],
            "7": ["class_type": "CLIPTextEncode", "inputs": [
                "clip": ["4", 1], "text": negativePrompt
            ]],
            "8": ["class_type": "EmptyLatentImage", "inputs": [
                "width": width, "height": height, "batch_size": 1
            ]],
            "10": ["class_type": "ModelSamplingAuraFlow", "inputs": [
                "model": ["4", 0], "shift": shift
            ]],
            "11": ["class_type": "KSampler", "inputs": [
                "model": ["10", 0], "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["8", 0], "seed": max(0, seed), "steps": steps,
                "cfg": cfg,
                "sampler_name": comfySamplerName(samplerName),
                "scheduler": comfySchedulerName(schedulerName),
                "denoise": 1.0
            ]],
            "12": ["class_type": "VAEDecode", "inputs": [
                "samples": ["11", 0], "vae": ["3", 0]
            ]],
            "13": ["class_type": "SaveImage", "inputs": [
                "images": ["12", 0], "filename_prefix": "RIFE60_AIStudio"
            ]]
        ]
        if enableCharacterLora && characterWeight > 0 {
            graph["5"] = ["class_type": "LoraLoader", "inputs": [
                "model": ["4", 0], "clip": ["4", 1],
                "lora_name": settings.characterLoraName,
                "strength_model": characterWeight, "strength_clip": characterWeight
            ]]
            graph["6"] = ["class_type": "CLIPTextEncode", "inputs": [
                "clip": ["5", 1], "text": positivePrompt
            ]]
            graph["7"] = ["class_type": "CLIPTextEncode", "inputs": [
                "clip": ["5", 1], "text": negativePrompt
            ]]
            graph["10"] = ["class_type": "ModelSamplingAuraFlow", "inputs": [
                "model": ["5", 0], "shift": shift
            ]]
        }
        return graph
    }

    private static func comfySamplerName(_ value: String) -> String {
        switch value {
        case "Euler": return "euler"
        case "DPM++ 2M": return "dpmpp_2m"
        case "DPM++ 2M SDE": return "dpmpp_2m_sde"
        case "Euler a": return "euler_ancestral"
        default: return "euler_ancestral"
        }
    }

    private static func comfySchedulerName(_ value: String) -> String {
        switch value {
        case "Karras": return "karras"
        case "Simple": return "simple"
        case "SGM Uniform": return "sgm_uniform"
        default: return "normal"
        }
    }

    private static func checkedJSONObject(data: Data, response: URLResponse, context: String) throws -> [String: Any] {
        guard let http = response as? HTTPURLResponse else {
            throw AIImageStudioError("Invalid server response while trying to \(context).")
        }
        guard (200..<300).contains(http.statusCode) else {
            let body = String(data: data, encoding: .utf8) ?? ""
            throw AIImageStudioError("Could not \(context) (HTTP \(http.statusCode)): \(String(body.prefix(500)))")
        }
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw AIImageStudioError("The server returned an invalid JSON response while trying to \(context).")
        }
        return object
    }

    private static func extractImageData(_ output: Any) async throws -> Data {
        if let object = output as? [String: Any] {
            if let images = object["images"] as? [[String: Any]], let first = images.first {
                if let encoded = first["data"] as? String, let data = Data(base64Encoded: encoded), UIImage(data: data) != nil {
                    return data
                }
                if let urlText = first["url"] as? String, let url = URL(string: urlText) {
                    let (data, response) = try await URLSession.shared.data(from: url)
                    guard (response as? HTTPURLResponse).map({ (200..<300).contains($0.statusCode) }) == true,
                          UIImage(data: data) != nil else {
                        throw AIImageStudioError("The GPU returned an image URL that could not be downloaded.")
                    }
                    return data
                }
            }
            if let urlText = object["image_url"] as? String, let url = URL(string: urlText) {
                let (data, response) = try await URLSession.shared.data(from: url)
                guard (response as? HTTPURLResponse).map({ (200..<300).contains($0.statusCode) }) == true,
                      UIImage(data: data) != nil else {
                    throw AIImageStudioError("The generated image URL could not be downloaded.")
                }
                return data
            }
            if let encoded = object["image"] as? String, let data = Data(base64Encoded: encoded), UIImage(data: data) != nil {
                return data
            }
        }
        throw AIImageStudioError("The GPU worker completed but returned no recognized image. Use a RunPod ComfyUI worker that returns output.images[].data or output.images[].url.")
    }

    static func submitLoRATraining(
        settings: AIImageStudioSettings,
        token: String,
        images: [AITrainingImage],
        caption: String,
        triggerWord: String,
        rank: Int,
        epochs: Int,
        learningRate: Double
    ) async throws -> String {
        guard let base = URL(string: settings.trainerBaseURL.trimmingCharacters(in: .whitespacesAndNewlines)),
              ["https", "http"].contains(base.scheme?.lowercased() ?? "") else {
            throw AIImageStudioError("Set a reachable HTTPS trainer API URL in Settings.")
        }
        guard !token.isEmpty else { throw AIImageStudioError("Enter the trainer API token in Settings.") }
        guard images.count >= 3 else { throw AIImageStudioError("Select at least 3 training images.") }

        let boundary = "RIFE60Boundary-\(UUID().uuidString)"
        var body = Data()
        func append(_ value: String) { body.append(Data(value.utf8)) }
        for (name, value) in [
            ("caption", caption), ("trigger_word", triggerWord),
            ("rank", String(rank)), ("epochs", String(epochs)),
            ("learning_rate", String(learningRate)), ("base_model", "anima")
        ] {
            append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n")
        }
        for image in images {
            let safeName = image.filename.replacingOccurrences(of: "\"", with: "_")
            append("--\(boundary)\r\nContent-Disposition: form-data; name=\"images\"; filename=\"\(safeName)\"\r\nContent-Type: application/octet-stream\r\n\r\n")
            body.append(image.data)
            append("\r\n")
        }
        append("--\(boundary)--\r\n")

        var request = URLRequest(url: base.appendingPathComponent("api/anima/lora/train"))
        request.httpMethod = "POST"
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        let (data, response) = try await URLSession.shared.data(for: request)
        let object = try checkedJSONObject(data: data, response: response, context: "submit LoRA training")
        guard let jobID = object["job_id"] as? String ?? object["id"] as? String else {
            throw AIImageStudioError("Trainer accepted no job ID. The trainer API must return {job_id: \"...\"}.")
        }
        return jobID
    }

    static func trainingStatus(settings: AIImageStudioSettings, token: String, jobID: String) async throws -> [String: Any] {
        guard let base = URL(string: settings.trainerBaseURL.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            throw AIImageStudioError("Invalid trainer API URL.")
        }
        let url = base.appendingPathComponent("api/anima/lora/train").appendingPathComponent(jobID)
        var request = URLRequest(url: url)
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        let (data, response) = try await URLSession.shared.data(for: request)
        return try checkedJSONObject(data: data, response: response, context: "check LoRA training status")
    }

    static func downloadFile(from urlText: String, token: String = "") async throws -> URL {
        guard let url = URL(string: urlText) else { throw AIImageStudioError("Trainer returned an invalid LoRA download URL.") }
        var request = URLRequest(url: url)
        if !token.isEmpty { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse).map({ (200..<300).contains($0.statusCode) }) == true else {
            throw AIImageStudioError("Could not download the trained LoRA.")
        }
        let destination = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("trained-anima-\(UUID().uuidString).safetensors")
        try data.write(to: destination, options: .atomic)
        return destination
    }
}
