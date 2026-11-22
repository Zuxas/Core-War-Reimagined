//! Match configuration: rulesets and seeded warrior placement.
//!
//! Before this module every battle used the same hardcoded ruleset (8000-cell
//! core, 80,000-step limit) and the same fixed placement (warrior 0 at address
//! 0, warrior 1 at `core_size / 2`), so a rematch between two warriors always
//! played out identically. [`MatchConfig`] captures the ruleset and a
//! [`Placement`] strategy; [`resolve_placement`] turns a placement plus the
//! warriors' lengths into concrete, validated start positions.
//!
//! The PRNG ([`SplitMix64`]) is dependency-free on purpose: the server (native)
//! and the browser (`wasm32`) must compute *identical* positions from the same
//! seed, so pulling in `rand`/`getrandom` (which behave differently, or not at
//! all, on `wasm32-unknown-unknown`) is out. A unit test pins the generated
//! positions for a fixed seed to catch any future algorithm drift.
//!
//! **Legality is uniform across placement modes.** A set of start positions is
//! legal only when, for every pair of warriors, the circular distance between
//! their start cells is at least `min_distance` *and* their occupied cell
//! ranges do not overlap once wrapped around the core. Start cells can be far
//! apart while long warriors still overlap after wrapping, so both checks are
//! required for `Fixed`, `Random`, and `Explicit` alike.

use core::fmt;

/// A dependency-free [SplitMix64] generator.
///
/// Pure wrapping `u64` arithmetic, so it produces the same stream on native and
/// `wasm32`. Not cryptographically secure — it only needs to be deterministic
/// and reproducible for warrior placement.
///
/// [SplitMix64]: https://prng.di.unimi.it/splitmix64.c
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator. Every seed yields a fixed, reproducible stream.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Next 64-bit output. Advances the state.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..bound`. `bound` must be non-zero.
    fn next_below(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0, "next_below requires a non-zero bound");
        (self.next_u64() % bound as u64) as usize
    }
}

/// How warriors are positioned in core at the start of a match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Placement {
    /// Evenly spaced: warrior `i` of `n` starts at `floor(i * core_size / n)`.
    /// For two warriors this reproduces the historical `0` and `core_size / 2`.
    Fixed,
    /// Warrior 0 at address 0; every later warrior at a seeded position that is
    /// legal against all already-placed warriors (pMARS semantics).
    ///
    /// Placement is **greedy, with no backtracking**: once a warrior is placed
    /// it is never moved. This is exact for two warriors, but for three or more
    /// an unlucky (though legal) early placement can leave no legal cell for a
    /// later warrior, so [`resolve_placement`] may return
    /// [`WarriorLoadError::NoValidPlacement`] even when a different arrangement
    /// of the earlier warriors would have fit them all. Each individual
    /// placement is still exhaustive given the already-placed warriors (the
    /// deterministic scan in [`place_one`] never fails a cell that is legal
    /// against them).
    Random { seed: u64 },
    /// Caller-supplied start positions, validated like any other placement.
    /// Intended for sandbox play, never for rated matches.
    Explicit(Vec<usize>),
}

/// A battle ruleset: core size, limits, and a [`Placement`] strategy.
///
/// The presets take their values from the standard King-of-the-Hill
/// definitions. `standard94` is the default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchConfig {
    /// Number of cells in the circular core.
    pub core_size: usize,
    /// Maximum number of process-steps before the match is declared a tie.
    pub max_steps: u64,
    /// Maximum instruction count of a single warrior.
    pub max_length: usize,
    /// Minimum circular distance between any two warriors' start cells.
    pub min_distance: usize,
    /// Maximum number of processes a single warrior may hold.
    pub max_processes: usize,
    /// How warriors are placed at the start of the match.
    pub placement: Placement,
}

impl MatchConfig {
    /// ICWS '94 standard hill: 8000 cells, 80,000 steps, length 100, distance
    /// 100, 8000 processes. The default ruleset.
    pub fn standard94() -> Self {
        Self {
            core_size: 8000,
            max_steps: 80_000,
            max_length: 100,
            min_distance: 100,
            max_processes: 8000,
            placement: Placement::Fixed,
        }
    }

    /// Nano hill: 80 cells, 800 steps, length 5, distance 5, 80 processes.
    pub fn nano() -> Self {
        Self {
            core_size: 80,
            max_steps: 800,
            max_length: 5,
            min_distance: 5,
            max_processes: 80,
            placement: Placement::Fixed,
        }
    }

    /// Tiny hill: 800 cells, 8000 steps, length 20, distance 20, 800 processes.
    pub fn tiny() -> Self {
        Self {
            core_size: 800,
            max_steps: 8000,
            max_length: 20,
            min_distance: 20,
            max_processes: 800,
            placement: Placement::Fixed,
        }
    }

    /// Big hill: 55,440 cells, 500,000 steps, length 200, distance 200, 55,440
    /// processes (the Stormking Big Hill lists max processes = core size).
    ///
    /// The process cap in particular varies between big-hill definitions;
    /// confirm against the intended ruleset if it differs.
    pub fn big() -> Self {
        Self {
            core_size: 55_440,
            max_steps: 500_000,
            max_length: 200,
            min_distance: 200,
            max_processes: 55_440,
            placement: Placement::Fixed,
        }
    }

    /// Look up a preset by name (`"nano"`, `"tiny"`, `"standard"` /
    /// `"standard94"`, `"big"`). Case-insensitive. Returns `None` for an
    /// unknown name.
    pub fn preset(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "nano" => Some(Self::nano()),
            "tiny" => Some(Self::tiny()),
            "standard" | "standard94" | "94" => Some(Self::standard94()),
            "big" => Some(Self::big()),
            _ => None,
        }
    }
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self::standard94()
    }
}

/// Why loading warriors into a match failed. Every variant is reported *before*
/// any warrior is written to core, so a failed load never partially mutates the
/// match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarriorLoadError {
    /// The ruleset itself cannot seat the warriors: no warriors, a zero-sized
    /// core, a warrior longer than the core, or `Fixed` spacing that cannot
    /// satisfy `min_distance` / non-overlap for this warrior count.
    InvalidConfig(&'static str),
    /// A warrior's instruction count exceeds `max_length`.
    WarriorTooLong {
        /// Index of the offending warrior in the input slice.
        index: usize,
        /// The warrior's actual instruction count.
        length: usize,
        /// The configured maximum.
        max_length: usize,
    },
    /// `Explicit` positions are unusable: wrong count, out of range, duplicated,
    /// overlapping, or closer than `min_distance`.
    InvalidExplicitPosition(&'static str),
    /// `Random` placement could find no legal position for a warrior, even
    /// after a full deterministic scan of the core.
    NoValidPlacement {
        /// Index of the warrior that could not be placed.
        index: usize,
    },
}

impl fmt::Display for WarriorLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WarriorLoadError::InvalidConfig(why) => write!(f, "invalid match config: {why}"),
            WarriorLoadError::WarriorTooLong {
                index,
                length,
                max_length,
            } => write!(
                f,
                "warrior {index} is {length} instructions, exceeds max length {max_length}"
            ),
            WarriorLoadError::InvalidExplicitPosition(why) => {
                write!(f, "invalid explicit placement: {why}")
            }
            WarriorLoadError::NoValidPlacement { index } => {
                write!(f, "no legal start position found for warrior {index}")
            }
        }
    }
}

impl std::error::Error for WarriorLoadError {}

/// Circular distance between two cells on a core of `core_size`.
fn circular_distance(a: usize, b: usize, core_size: usize) -> usize {
    let d = a.abs_diff(b);
    d.min(core_size - d)
}

/// Forward gap from `a` to `b` going in increasing-address order: `(b - a) mod core_size`.
fn forward_gap(a: usize, b: usize, core_size: usize) -> usize {
    (b + core_size - a) % core_size
}

/// Whether warriors `i` (start `pi`, length `li`) and `j` (start `pj`, length
/// `lj`) occupy disjoint cell ranges on a circular core. A zero-length warrior
/// occupies nothing and never overlaps.
fn footprints_disjoint(pi: usize, li: usize, pj: usize, lj: usize, core_size: usize) -> bool {
    if li == 0 || lj == 0 {
        return true;
    }
    if pi == pj {
        return false;
    }
    let gap_ij = forward_gap(pi, pj, core_size); // cells from i's start up to j's start
    let gap_ji = core_size - gap_ij; // cells from j's start up to i's start
    gap_ij >= li && gap_ji >= lj
}

/// Whether two warriors may share these start cells: far enough apart *and*
/// their footprints do not overlap.
fn legal_pair(
    pi: usize,
    li: usize,
    pj: usize,
    lj: usize,
    core_size: usize,
    min_distance: usize,
) -> bool {
    circular_distance(pi, pj, core_size) >= min_distance
        && footprints_disjoint(pi, li, pj, lj, core_size)
}

/// Whether a candidate position is legal against every already-placed warrior.
fn legal_against_all(
    candidate: usize,
    cand_len: usize,
    placed: &[usize],
    lengths: &[usize],
    core_size: usize,
    min_distance: usize,
) -> bool {
    placed
        .iter()
        .enumerate()
        .all(|(j, &pj)| legal_pair(candidate, cand_len, pj, lengths[j], core_size, min_distance))
}

/// Whether a whole set of positions is pairwise legal.
fn positions_all_legal(
    positions: &[usize],
    lengths: &[usize],
    core_size: usize,
    min_distance: usize,
) -> bool {
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            if !legal_pair(
                positions[i],
                lengths[i],
                positions[j],
                lengths[j],
                core_size,
                min_distance,
            ) {
                return false;
            }
        }
    }
    true
}

/// Find a legal start for one more warrior: try up to `core_size` seeded random
/// candidates, then deterministically scan `0..core_size` for any legal cell.
/// Returns `None` only when no legal cell exists *for this warrior given the
/// already-placed warriors* (so the PRNG never spuriously fails a position that
/// is legal against them). It does not reconsider the earlier placements — see
/// [`Placement::Random`] for the greedy, non-backtracking caveat.
fn place_one(
    rng: &mut SplitMix64,
    placed: &[usize],
    lengths: &[usize],
    index: usize,
    core_size: usize,
    min_distance: usize,
) -> Option<usize> {
    let len = lengths[index];
    for _ in 0..core_size {
        let cand = rng.next_below(core_size);
        if legal_against_all(cand, len, placed, lengths, core_size, min_distance) {
            return Some(cand);
        }
    }
    (0..core_size)
        .find(|&cand| legal_against_all(cand, len, placed, lengths, core_size, min_distance))
}

/// Resolve a [`Placement`] into concrete, validated start positions — one per
/// warrior, in input order. Pure: it reads only `core_size`, `min_distance`,
/// and the warriors' `lengths`, and never touches a match's core.
///
/// Legality is identical for all three modes (see the module docs): circular
/// `min_distance` between start cells *and* footprint non-overlap.
pub fn resolve_placement(
    core_size: usize,
    min_distance: usize,
    lengths: &[usize],
    placement: &Placement,
) -> Result<Vec<usize>, WarriorLoadError> {
    let n = lengths.len();
    if n == 0 {
        return Err(WarriorLoadError::InvalidConfig("no warriors to place"));
    }
    if core_size == 0 {
        return Err(WarriorLoadError::InvalidConfig(
            "core_size must be non-zero",
        ));
    }
    if lengths.iter().any(|&l| l > core_size) {
        return Err(WarriorLoadError::InvalidConfig(
            "a warrior is longer than the core",
        ));
    }

    match placement {
        Placement::Fixed => {
            // floor(i * core_size / n); widen to u128 so the product can't
            // overflow on a large core and many warriors.
            let positions: Vec<usize> = (0..n)
                .map(|i| ((i as u128 * core_size as u128) / n as u128) as usize)
                .collect();
            if positions_all_legal(&positions, lengths, core_size, min_distance) {
                Ok(positions)
            } else {
                Err(WarriorLoadError::InvalidConfig(
                    "fixed spacing cannot satisfy min_distance / non-overlap for this core and warrior count",
                ))
            }
        }
        Placement::Explicit(positions) => {
            if positions.len() != n {
                return Err(WarriorLoadError::InvalidExplicitPosition(
                    "position count does not match warrior count",
                ));
            }
            if positions.iter().any(|&p| p >= core_size) {
                return Err(WarriorLoadError::InvalidExplicitPosition(
                    "a position is outside the core",
                ));
            }
            if !positions_all_legal(positions, lengths, core_size, min_distance) {
                return Err(WarriorLoadError::InvalidExplicitPosition(
                    "positions overlap, duplicate, or are closer than min_distance",
                ));
            }
            Ok(positions.clone())
        }
        Placement::Random { seed } => {
            let mut rng = SplitMix64::new(*seed);
            let mut positions = Vec::with_capacity(n);
            positions.push(0); // warrior 0 always at address 0
            for i in 1..n {
                match place_one(&mut rng, &positions, lengths, i, core_size, min_distance) {
                    Some(p) => positions.push(p),
                    None => return Err(WarriorLoadError::NoValidPlacement { index: i }),
                }
            }
            Ok(positions)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_match_hill_definitions() {
        let s = MatchConfig::standard94();
        assert_eq!(s.core_size, 8000);
        assert_eq!(s.max_steps, 80_000);
        assert_eq!(s.max_length, 100);
        assert_eq!(s.min_distance, 100);
        assert_eq!(s.max_processes, 8000);
        assert_eq!(MatchConfig::default(), MatchConfig::standard94());
        assert_eq!(MatchConfig::nano().core_size, 80);
        assert_eq!(MatchConfig::tiny().core_size, 800);
        assert_eq!(MatchConfig::big().core_size, 55_440);
    }

    #[test]
    fn preset_lookup_is_case_insensitive_and_rejects_unknown() {
        assert_eq!(MatchConfig::preset("NANO"), Some(MatchConfig::nano()));
        assert_eq!(
            MatchConfig::preset("standard94"),
            Some(MatchConfig::standard94())
        );
        assert_eq!(MatchConfig::preset("bogus"), None);
    }

    #[test]
    fn fixed_two_warriors_reproduce_zero_and_half() {
        let pos = resolve_placement(8000, 100, &[10, 10], &Placement::Fixed).unwrap();
        assert_eq!(pos, vec![0, 4000]);
    }

    #[test]
    fn fixed_works_for_three_and_four_warriors() {
        let three = resolve_placement(8000, 100, &[1, 1, 1], &Placement::Fixed).unwrap();
        assert_eq!(three, vec![0, 2666, 5333]);
        let four = resolve_placement(8000, 100, &[1, 1, 1, 1], &Placement::Fixed).unwrap();
        assert_eq!(four, vec![0, 2000, 4000, 6000]);
        // Every pair still satisfies the legality rules.
        assert!(positions_all_legal(&four, &[1, 1, 1, 1], 8000, 100));
    }

    #[test]
    fn fixed_rejected_when_spacing_cannot_meet_min_distance() {
        // Four warriors evenly spaced in a 20-cell core are 5 apart; a
        // min_distance of 6 cannot be met by even spacing.
        let err = resolve_placement(20, 6, &[1, 1, 1, 1], &Placement::Fixed).unwrap_err();
        assert!(matches!(err, WarriorLoadError::InvalidConfig(_)));
    }

    #[test]
    fn explicit_accepts_legal_positions() {
        let pos =
            resolve_placement(8000, 100, &[10, 10], &Placement::Explicit(vec![0, 500])).unwrap();
        assert_eq!(pos, vec![0, 500]);
    }

    #[test]
    fn explicit_rejects_duplicates_overlap_and_close_spacing() {
        // Duplicate start cells (overlap + zero distance).
        assert!(matches!(
            resolve_placement(800, 10, &[5, 5], &Placement::Explicit(vec![100, 100])),
            Err(WarriorLoadError::InvalidExplicitPosition(_))
        ));
        // Too close: 8 apart with min_distance 10.
        assert!(matches!(
            resolve_placement(800, 10, &[1, 1], &Placement::Explicit(vec![0, 8])),
            Err(WarriorLoadError::InvalidExplicitPosition(_))
        ));
        // Wrong count.
        assert!(matches!(
            resolve_placement(800, 10, &[1, 1], &Placement::Explicit(vec![0])),
            Err(WarriorLoadError::InvalidExplicitPosition(_))
        ));
        // Out of range.
        assert!(matches!(
            resolve_placement(800, 10, &[1, 1], &Placement::Explicit(vec![0, 800])),
            Err(WarriorLoadError::InvalidExplicitPosition(_))
        ));
    }

    #[test]
    fn explicit_rejects_footprint_overlap_even_when_starts_are_far() {
        // Starts 50 apart (>= min_distance 10), but warrior 0 is 60 long, so
        // its footprint runs into warrior 1. Distance alone would pass; the
        // non-overlap check must catch it.
        let err =
            resolve_placement(800, 10, &[60, 5], &Placement::Explicit(vec![0, 50])).unwrap_err();
        assert!(matches!(err, WarriorLoadError::InvalidExplicitPosition(_)));
    }

    #[test]
    fn explicit_detects_overlap_after_wrapping() {
        // Warrior 1 starts near the end and wraps past 0 into warrior 0.
        // core 100, warrior 1 len 10 at pos 95 occupies 95..100 and 0..5.
        let err =
            resolve_placement(100, 1, &[5, 10], &Placement::Explicit(vec![0, 95])).unwrap_err();
        assert!(matches!(err, WarriorLoadError::InvalidExplicitPosition(_)));
    }

    #[test]
    fn random_is_seed_deterministic_with_pinned_positions() {
        // Pins the SplitMix64-driven positions for a fixed seed so any future
        // change to the algorithm is caught. If this value legitimately
        // changes, update it deliberately.
        let a =
            resolve_placement(8000, 100, &[10, 10, 10], &Placement::Random { seed: 42 }).unwrap();
        let b =
            resolve_placement(8000, 100, &[10, 10, 10], &Placement::Random { seed: 42 }).unwrap();
        assert_eq!(a, b, "same seed must give identical positions");
        assert_eq!(a[0], 0, "warrior 0 is always at 0");
        assert!(positions_all_legal(&a, &[10, 10, 10], 8000, 100));
        assert_eq!(a, vec![0, 3413, 4291]);
    }

    #[test]
    fn random_positions_are_all_legal() {
        let lengths = [50, 50, 50, 50];
        let pos =
            resolve_placement(8000, 100, &lengths, &Placement::Random { seed: 12345 }).unwrap();
        assert!(positions_all_legal(&pos, &lengths, 8000, 100));
    }

    #[test]
    fn random_terminates_and_errors_on_impossible_config() {
        // Three warriors each needing >= 400 distance in a 900-cell core:
        // warriors 0 and 1 fit, but the third has no legal cell (any two cells
        // >= 400 circular-apart leave no room for a third). Must terminate with
        // NoValidPlacement, not loop forever.
        let err =
            resolve_placement(900, 400, &[1, 1, 1], &Placement::Random { seed: 7 }).unwrap_err();
        assert!(matches!(
            err,
            WarriorLoadError::NoValidPlacement { index: 2 }
        ));
    }

    #[test]
    fn random_scan_fallback_finds_the_one_legal_cell() {
        // A tight config where random draws are unlikely to hit the single
        // legal cell, forcing the deterministic scan to find it.
        let lengths = [1, 1];
        let pos = resolve_placement(10, 5, &lengths, &Placement::Random { seed: 1 }).unwrap();
        assert_eq!(pos[0], 0);
        assert!(positions_all_legal(&pos, &lengths, 10, 5));
    }

    #[test]
    fn empty_and_zero_core_are_invalid() {
        assert!(matches!(
            resolve_placement(8000, 100, &[], &Placement::Fixed),
            Err(WarriorLoadError::InvalidConfig(_))
        ));
        assert!(matches!(
            resolve_placement(0, 0, &[1], &Placement::Fixed),
            Err(WarriorLoadError::InvalidConfig(_))
        ));
    }

    #[test]
    fn warrior_longer_than_core_is_invalid() {
        assert!(matches!(
            resolve_placement(50, 10, &[80], &Placement::Fixed),
            Err(WarriorLoadError::InvalidConfig(_))
        ));
    }

    #[test]
    fn splitmix64_matches_reference_stream() {
        // Reference outputs of SplitMix64 for seed 0 (from the canonical C
        // implementation), guarding the constants against typos.
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(rng.next_u64(), 0x06C4_5D18_8009_454F);
    }
}
