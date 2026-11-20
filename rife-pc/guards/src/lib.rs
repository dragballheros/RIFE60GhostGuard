//! Quality guards ported from the RIFE60 GhostGuard iOS app.
//! Thresholds match the Swift sources. Reductions stay in one thread so the
//! comparisons do not drift; independent rows use rayon.

mod chain;
mod frame;
mod ghost;
mod outlier;
mod small_motion;
mod static_lock;
mod stroke;
mod structural;
mod thin_line;
mod warped;

pub use chain::{clean_source_outlier, polish_inbetween, PolishStats};
pub use frame::Frame;
pub use ghost::{endpoint_disposition, Disposition, GhostGuard, GhostGuardResult};
pub use outlier::{OutlierResult, SourceFrameOutlierGuard};
pub use small_motion::SmallMotionLineStabilizer;
pub use static_lock::StaticRegionLock;
pub use stroke::{stroke_width_decision, StrokeDecision};
pub use structural::{appearance_floor, assess as assess_structure, overlap_limit, Assessment};
pub use thin_line::ThinLinePreservationPass;
pub use warped::rejects as warped_outline_rejects;
