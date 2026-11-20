//! Render power mode and the performance tier the iOS governor uses.
//! On Windows, power mode asks the system to stay awake. It does not raise
//! the RIFE tile above the 4 GB framebuffer budget of the Radeon Pro 5300M
//! and 5500M. Safe mode lowers the tile.

use crate::settings::PerformanceTier;

#[derive(Debug)]
pub struct Awake {
    held: bool,
}

impl Awake {
    pub fn acquire(enabled: bool) -> Self {
        if enabled {
            set_awake(true);
        }
        Self { held: enabled }
    }
}

impl Drop for Awake {
    fn drop(&mut self) {
        if self.held {
            set_awake(false);
        }
    }
}

fn set_awake(on: bool) {
    #[cfg(target_os = "windows")]
    {
        const ES_CONTINUOUS: u32 = 0x8000_0000;
        const ES_SYSTEM_REQUIRED: u32 = 0x0000_0001;
        const ES_DISPLAY_REQUIRED: u32 = 0x0000_0002;
        extern "system" {
            fn SetThreadExecutionState(flags: u32) -> u32;
        }
        let flags = if on {
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        } else {
            ES_CONTINUOUS
        };
        unsafe {
            SetThreadExecutionState(flags);
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = on;
    }
}

pub fn available_ram_mb() -> u64 {
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        struct MemoryStatusEx {
            length: u32,
            memory_load: u32,
            total_phys: u64,
            avail_phys: u64,
            total_page_file: u64,
            avail_page_file: u64,
            total_virtual: u64,
            avail_virtual: u64,
            avail_extended_virtual: u64,
        }
        extern "system" {
            fn GlobalMemoryStatusEx(lpbuffer: *mut MemoryStatusEx) -> i32;
        }
        let mut status = MemoryStatusEx {
            length: std::mem::size_of::<MemoryStatusEx>() as u32,
            memory_load: 0,
            total_phys: 0,
            avail_phys: 0,
            total_page_file: 0,
            avail_page_file: 0,
            total_virtual: 0,
            avail_virtual: 0,
            avail_extended_virtual: 0,
        };
        let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
        if ok != 0 {
            return status.avail_phys / (1024 * 1024);
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("MemAvailable:") {
                    let kb: u64 = rest.split_whitespace().next().unwrap_or("0").parse().unwrap_or(0);
                    return kb / 1024;
                }
            }
        }
    }
    4096
}

pub fn effective_tier(selected: PerformanceTier, render_power: bool, ram_mb: u64) -> PerformanceTier {
    match selected {
        PerformanceTier::Performance | PerformanceTier::Balanced | PerformanceTier::Safe => selected,
        PerformanceTier::Auto => {
            if ram_mb < 1400 {
                PerformanceTier::Safe
            } else if !render_power || ram_mb < 8000 {
                PerformanceTier::Balanced
            } else {
                PerformanceTier::Performance
            }
        }
    }
}

/// Tile size passed to RIFE and Real-CUGAN. Performance keeps the caller's
/// tile (512 by default). Safe caps it at 256 so a 4 GB framebuffer stays
/// inside the 5300M and 5500M.
pub fn tile_for(tier: PerformanceTier, requested: i32) -> i32 {
    let requested = if requested < 0 { 512 } else { requested };
    match tier {
        PerformanceTier::Safe => {
            if requested == 0 {
                256
            } else {
                requested.min(256).max(32)
            }
        }
        PerformanceTier::Auto | PerformanceTier::Balanced | PerformanceTier::Performance => requested,
    }
}

pub fn tier_note(tier: PerformanceTier, render_power: bool, ram_mb: u64) -> String {
    let power = if render_power {
        "render power mode is holding the display awake"
    } else {
        "render power mode is off"
    };
    format!("{power}; tier {} with {ram_mb} MB free", tier.label())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_64gb_machine_with_power_mode_picks_performance() {
        let tier = effective_tier(PerformanceTier::Auto, true, 48_000);
        assert_eq!(tier, PerformanceTier::Performance);
        assert_eq!(tile_for(tier, 512), 512);
    }

    #[test]
    fn safe_caps_the_tile() {
        assert_eq!(tile_for(PerformanceTier::Safe, 512), 256);
        assert_eq!(tile_for(PerformanceTier::Safe, 0), 256);
    }

    #[test]
    fn power_off_auto_stays_balanced_on_a_large_machine() {
        assert_eq!(
            effective_tier(PerformanceTier::Auto, false, 48_000),
            PerformanceTier::Balanced
        );
    }
}
