//! Port of the RNG owned by `CMatrixMapLogic` (MatrixLogic.cpp:84-113).
//! Park–Miller MINSTD LCG with a 32-bit state — the same generator the
//! original game uses for all deterministic-world decisions (object
//! animation timers, spawn jitter, tactical noise, etc.).
//!
//! The C++ seeds the generator from `rand()` once at construction and
//! burns one output with `Rnd(0,1)` to mix the seed in
//! (MatrixLogic.cpp:49). We reproduce the contract exactly for bit-for-bit
//! parity when we later feed the same seed from saved games / replays —
//! the constructor in [`Rnd::new`] takes the seed explicitly; callers
//! that want "pick something" semantics can use [`Rnd::from_clock`].

/// Recurrence constants from `CMatrixMapLogic::Rnd` (MatrixLogic.cpp:88).
/// `m_Rnd = 16807 * (m_Rnd % 127773) - 2836 * (m_Rnd / 127773)` —
/// the classic Schrage-factored MINSTD step. Output is `m_Rnd - 1` so
/// the stream starts at 0 instead of 1.
const MINSTD_A: i32 = 16_807;
const MINSTD_Q: i32 = 127_773; // 2^31-1 / A
const MINSTD_R: i32 = 2_836; // 2^31-1 % A
const MINSTD_M_MINUS_1: i32 = 2_147_483_647; // 2^31 - 1

pub struct Rnd {
    /// Matches `m_Rnd` in CMatrixMapLogic (MatrixLogic.hpp:75).
    /// Must stay strictly positive; the step reseeds to +(2^31-1) when
    /// it ever lands at or below zero, so the state never gets stuck.
    state: i32,
}

impl Rnd {
    /// Construct with an explicit seed — matches setting `m_Rnd = seed`
    /// before the constructor's `Rnd(0,1)` mix-in.
    pub fn new(seed: i32) -> Self {
        let mut r = Self { state: seed };
        // CMatrixMapLogic ctor: `m_Rnd=rand(); Rnd(0,1);` — burns one
        // sample to diffuse the seed (MatrixLogic.cpp:49).
        let _ = r.range(0, 1);
        r
    }

    /// Seed from the platform clock — what the original effectively
    /// does via `m_Rnd=rand()`. Kept separate so tests can construct
    /// deterministic streams via [`Rnd::new`].
    pub fn from_clock() -> Self {
        // Any non-zero i32 seed works. `platform::now_secs` is already
        // used throughout the port; hash-mix it into an i32.
        let now = crate::platform::now_secs();
        // Convert to a positive-int seed without pulling in a hasher.
        let bits = now.to_bits() as u64;
        let seed = ((bits ^ (bits >> 32)) as u32 & 0x7FFF_FFFF) as i32;
        Self::new(if seed == 0 { 1 } else { seed })
    }

    /// Raw `CMatrixMapLogic::Rnd()` — one step of the generator.
    /// Result in `[0, 2^31-2]` (matches the C++ `return m_Rnd-1`).
    ///
    /// Named after the C++ API (`Rnd::next`) rather than
    /// `std::iter::Iterator::next`; this is a free-running PRNG, not
    /// an iterator.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> i32 {
        self.state = MINSTD_A.wrapping_mul(self.state % MINSTD_Q)
            - MINSTD_R.wrapping_mul(self.state / MINSTD_Q);
        if self.state <= 0 {
            self.state = self.state.wrapping_add(MINSTD_M_MINUS_1);
        }
        self.state - 1
    }

    /// `Rnd(zmin, zmax)` — inclusive range on both ends. Mirrors the
    /// C++ semantics where `zmin > zmax` swaps the endpoints
    /// (MatrixLogic.cpp:100-106).
    pub fn range(&mut self, zmin: i32, zmax: i32) -> i32 {
        if zmin <= zmax {
            zmin + (self.next() % (zmax - zmin + 1))
        } else {
            zmax + (self.next() % (zmin - zmax + 1))
        }
    }

    /// `RndFloat()` — uniform `[0, 1]` (the C++ denominator is
    /// `2147483647 - 2` to normalise the `next()` range).
    pub fn float01(&mut self) -> f64 {
        self.next() as f64 / (MINSTD_M_MINUS_1 as f64 - 2.0)
    }

    /// `RndFloat(zmin, zmax)` — uniform float in `[zmin, zmax)`.
    pub fn float_range(&mut self, zmin: f64, zmax: f64) -> f64 {
        zmin + self.float01() * (zmax - zmin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_is_deterministic_for_a_fixed_seed() {
        let mut a = Rnd::new(12345);
        let mut b = Rnd::new(12345);
        for _ in 0..32 {
            assert_eq!(a.next(), b.next());
        }
    }

    #[test]
    fn range_respects_bounds_and_allows_swapped_endpoints() {
        let mut r = Rnd::new(42);
        for _ in 0..200 {
            let v = r.range(-5, 10);
            assert!((-5..=10).contains(&v));
        }
        for _ in 0..200 {
            let v = r.range(10, -5); // swapped — matches C++
            assert!((-5..=10).contains(&v));
        }
    }

    #[test]
    fn float01_is_in_unit_interval() {
        let mut r = Rnd::new(7);
        for _ in 0..1000 {
            let f = r.float01();
            assert!((0.0..=1.0).contains(&f));
        }
    }

    /// Parity test against the exact C++ recurrence. Hand-unrolled
    /// three steps from seed=1 (after the ctor's mix-in `Rnd(0,1)`).
    /// Any future refactor that breaks the bitstream will trip here.
    #[test]
    fn first_draws_match_hand_simulation() {
        // Simulate `m_Rnd=1; Rnd(0,1); next();` — the first observable
        // `next()` output after construction.
        let mut r = Rnd::new(1);
        let v0 = r.next();
        let v1 = r.next();
        // Reconstruct expected with pen-and-paper recurrence.
        let step = |s: &mut i32| {
            *s = MINSTD_A.wrapping_mul(*s % MINSTD_Q) - MINSTD_R.wrapping_mul(*s / MINSTD_Q);
            if *s <= 0 {
                *s = s.wrapping_add(MINSTD_M_MINUS_1);
            }
        };
        let mut expect = 1i32;
        step(&mut expect); // ctor's Rnd(0,1) call
        let _ = expect; // discarded inside range()
        step(&mut expect);
        assert_eq!(v0, expect - 1);
        step(&mut expect);
        assert_eq!(v1, expect - 1);
    }
}
