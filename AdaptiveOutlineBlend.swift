import Foundation

/// Adaptive final Sharpie contribution based on the source contour's local width.
///
/// A strong narrow-detail response means the source feature is already thin
/// (for example fingernails, finger creases, lashes, and hair strands). Those
/// pixels retain the current 0.82 contribution. Broad, continuous character
/// contours recover the older 0.94 bold look. The interpolation is continuous
/// to avoid a hard threshold that can make line width flicker along a contour.
enum AdaptiveOutlineBlend {
    static let thinContourWeight = 0.82
    static let boldContourWeight = 0.94

    static func targetWeight(narrowDetailStrength: Double) -> Double {
        let thinness = min(max(narrowDetailStrength.isFinite ? narrowDetailStrength : 0, 0), 1)
        return boldContourWeight - (boldContourWeight - thinContourWeight) * thinness
    }

    /// Scale applied inside OutlineEnhancer before the final global bold blend.
    /// This makes the combined contribution equal to targetWeight.
    static func modelAmountScale(narrowDetailStrength: Double) -> Double {
        targetWeight(narrowDetailStrength: narrowDetailStrength) / boldContourWeight
    }
}
