import Darwin
import Foundation

/// On-device GGUF reader. The pinned native H3 runtime memory-maps BF16 safetensors
/// and does not parse GGUF. This type reads a GGUF header and dequantizes Q8_0
/// blocks to BF16 one block at a time, without writing a second full-size file.
enum H3GGUFStream {
    struct Tensor {
        var name: String
        var ggmlType: UInt32
        var elementCount: UInt64
        var offset: UInt64
    }

    enum Failure: LocalizedError {
        case notGGUF(String)
        case truncated(String)
        case unsupported(String)

        var errorDescription: String? {
            switch self {
            case .notGGUF(let path):
                return "\(path) is not a GGUF file."
            case .truncated(let path):
                return "\(path) ended before the GGUF header finished."
            case .unsupported(let path):
                return "\(path) is not a Q8_0 or Q4_K GGUF shard this phone can stream."
            }
        }
    }

    static let ggmlQ8_0: UInt32 = 8
    static let ggmlQ4K: UInt32 = 12
    static let q8BlockBytes = 34

    static func confirmDownloadableShard(at url: URL, path: String) throws {
        let header = try readHeader(at: url, path: path)
        let lower = path.lowercased()
        if lower.contains("q8_0") {
            guard header.tensors.contains(where: { $0.ggmlType == ggmlQ8_0 }) else {
                throw Failure.unsupported(path)
            }
            _ = try dequantizeFirstQ8Block(at: url, header: header, path: path)
            return
        }
        if lower.contains("q4_k") {
            guard header.tensors.contains(where: { $0.ggmlType == ggmlQ4K }) else {
                throw Failure.unsupported(path)
            }
        }
    }

    struct Header {
        var tensors: [Tensor]
        var alignment: UInt64
        var dataOffset: UInt64
    }

    static func readHeader(at url: URL, path: String) throws -> Header {
        CrashCapture.step("H3 GGUF header begin \(path)")
        let handle = try openUncached(url)
        defer { try? handle.close() }
        let magic = try readData(handle, count: 4, path: path)
        guard magic == Data("GGUF".utf8) else { throw Failure.notGGUF(path) }
        let version = try readU32(handle, path: path)
        guard version >= 2 else { throw Failure.unsupported(path) }
        let tensorCount = try readU64(handle, path: path)
        let metadataCount = try readU64(handle, path: path)
        guard tensorCount < 10_000_000, metadataCount < 10_000_000 else {
            throw Failure.unsupported(path)
        }
        var alignment: UInt64 = 32
        var index: UInt64 = 0
        while index < metadataCount {
            let key = try readString(handle, path: path)
            let valueType = try readU32(handle, path: path)
            if key == "general.alignment" && valueType == 4 {
                alignment = UInt64(try readU32(handle, path: path))
            } else {
                try skipValue(handle, type: valueType, path: path)
            }
            index += 1
        }
        var tensors: [Tensor] = []
        tensors.reserveCapacity(Int(min(tensorCount, 4096)))
        var tensorIndex: UInt64 = 0
        while tensorIndex < tensorCount {
            let name = try readString(handle, path: path)
            let rank = try readU32(handle, path: path)
            guard rank <= 8 else { throw Failure.unsupported(path) }
            var elements: UInt64 = 1
            var axis: UInt32 = 0
            while axis < rank {
                let dim = try readU64(handle, path: path)
                elements = elements * dim
                axis += 1
            }
            let ggmlType = try readU32(handle, path: path)
            let offset = try readU64(handle, path: path)
            tensors.append(Tensor(name: name, ggmlType: ggmlType, elementCount: elements, offset: offset))
            tensorIndex += 1
        }
        let position = try handle.offset()
        let remainder = position % alignment
        let dataOffset = remainder == 0 ? position : position + (alignment - remainder)
        CrashCapture.step("H3 GGUF header end \(path) tensors=\(tensors.count)")
        return Header(tensors: tensors, alignment: alignment, dataOffset: dataOffset)
    }

    private static func openUncached(_ url: URL) throws -> FileHandle {
        let handle = try FileHandle(forReadingFrom: url)
        _ = fcntl(handle.fileDescriptor, F_NOCACHE, 1)
        return handle
    }

    static func dequantizeFirstQ8Block(at url: URL, header: Header, path: String) throws -> [UInt16] {
        guard let tensor = header.tensors.first(where: { $0.ggmlType == ggmlQ8_0 }) else {
            throw Failure.unsupported(path)
        }
        let handle = try openUncached(url)
        defer { try? handle.close() }
        let location = header.dataOffset + tensor.offset
        try handle.seek(toOffset: location)
        let block = try readData(handle, count: q8BlockBytes, path: path)
        return dequantizeQ8Block(block)
    }

    static func dequantizeQ8Block(_ block: Data) -> [UInt16] {
        let scaleBits = UInt16(block[0]) | (UInt16(block[1]) << 8)
        let scale = float16(scaleBits)
        var values: [UInt16] = []
        values.reserveCapacity(32)
        var index = 2
        while index < 34 {
            let quant = Int8(bitPattern: block[index])
            let value = scale * Float(quant)
            values.append(bfloat16(value))
            index += 1
        }
        return values
    }

    private static func float16(_ bits: UInt16) -> Float {
        let signBit = (bits & 0x8000) >> 15
        let exponent = Int((bits & 0x7C00) >> 10)
        let fraction = bits & 0x03FF
        let magnitude: Float
        if exponent == 0 {
            magnitude = Float(fraction) / 1024 * exp2(-14)
        } else if exponent == 31 {
            magnitude = fraction == 0 ? Float.infinity : Float.nan
        } else {
            let base = 1 + Float(fraction) / 1024
            let power = Float(exponent - 15)
            magnitude = base * exp2(power)
        }
        if signBit == 1 { return -magnitude }
        return magnitude
    }

    private static func bfloat16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let leastBit = (bits >> 16) & 1
        let rounded = bits &+ 0x7FFF &+ leastBit
        return UInt16(truncatingIfNeeded: rounded >> 16)
    }

    private static func readData(_ handle: FileHandle, count: Int, path: String) throws -> Data {
        let data = try handle.read(upToCount: count) ?? Data()
        guard data.count == count else { throw Failure.truncated(path) }
        return data
    }

    private static func readU32(_ handle: FileHandle, path: String) throws -> UInt32 {
        let data = try readData(handle, count: 4, path: path)
        return UInt32(data[0])
            | (UInt32(data[1]) << 8)
            | (UInt32(data[2]) << 16)
            | (UInt32(data[3]) << 24)
    }

    private static func readU64(_ handle: FileHandle, path: String) throws -> UInt64 {
        let data = try readData(handle, count: 8, path: path)
        var value: UInt64 = 0
        var shift: UInt64 = 0
        var index = 0
        while index < 8 {
            value |= UInt64(data[index]) << shift
            shift += 8
            index += 1
        }
        return value
    }

    private static func readString(_ handle: FileHandle, path: String) throws -> String {
        let count = try readU64(handle, path: path)
        guard count < 16 * 1024 * 1024 else { throw Failure.unsupported(path) }
        let data = try readData(handle, count: Int(count), path: path)
        return String(decoding: data, as: UTF8.self)
    }

    private static func skipValue(_ handle: FileHandle, type: UInt32, path: String) throws {
        switch type {
        case 0, 1, 7:
            _ = try readData(handle, count: 1, path: path)
        case 2, 3:
            _ = try readData(handle, count: 2, path: path)
        case 4, 5, 6:
            _ = try readData(handle, count: 4, path: path)
        case 10, 11, 12:
            _ = try readData(handle, count: 8, path: path)
        case 8:
            _ = try readString(handle, path: path)
        case 9:
            let element = try readU32(handle, path: path)
            let count = try readU64(handle, path: path)
            var index: UInt64 = 0
            while index < count {
                try skipValue(handle, type: element, path: path)
                index += 1
            }
        default:
            throw Failure.unsupported(path)
        }
    }
}
