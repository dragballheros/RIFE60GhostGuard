from pathlib import Path
p = Path('Vendor/rife-metal/Sources/RifeMetal/RifeInterpolator.swift')
s = p.read_text()
needle = '    /// Creates a stateful stream session locked to the given resolution.'
assert s.count(needle) == 1, 'RifeInterpolator stream anchor changed'
assert 'releaseIdleStreamGraph' not in s, 'idle graph release already patched'
method = '''    /// Caller must first release every stream and wait for its synchronous
    /// push to return. Frees idle graph workspace while retaining model weights.
    public func releaseIdleStreamGraph() {
        queue.sync {
            graph = nil
            graphResolution = nil
        }
    }

'''
p.write_text(s.replace(needle, method + needle, 1))
