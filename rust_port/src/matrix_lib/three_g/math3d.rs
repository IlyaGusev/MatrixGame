//! Partial port of `MatrixLib/3G/src/Math3D.cpp` — the `CTrajectory`
//! Catmull-Rom spline the bomb projectile flies along.
//!
//! Only the `Init2` ("точно" — passes through the control points)
//! variant is ported: the bomb constructor `CTrajectory(heap, pts,
//! pcnt)` (Math3D.hpp:473) always uses it.

use glam::Vec3;

/// `SBSplineKoefs` (Math3D.hpp:405-411) — cubic coefficients per axis.
#[derive(Debug, Clone, Copy)]
struct SplineKoefs {
    pos0: Vec3,
    a: Vec3, // (a1, b1, c1)
    b: Vec3, // (a2, b2, c2)
    c: Vec3, // (a3, b3, c3)
}

/// `CalcBSplineKoefs2` (Math3D.cpp:409-431) — Catmull-Rom basis.
/// `CalcBSplineKoefs1` (Math3D.cpp:391-407) — uniform cubic B-spline.
fn koefs1(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> SplineKoefs {
    let sixth = 1.0 / 6.0;
    SplineKoefs {
        pos0: (4.0 * p1 + p0 + p2) * sixth,
        a: (p2 - p0) * 0.5,
        b: (p0 + p2) * 0.5 - p1,
        c: (p1 - p2) * 0.5 + (p3 - p0) * sixth,
    }
}

fn koefs2(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> SplineKoefs {
    SplineKoefs {
        pos0: p1,
        a: (p2 - p0) * 0.5,
        b: p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3,
        c: 1.5 * (p1 - p2) + (p3 - p0) * 0.5,
    }
}

/// `CalcBSplinePoint` (Math3D.hpp:415-421).
fn spline_point(k: &SplineKoefs, t: f32) -> Vec3 {
    ((k.c * t + k.b) * t + k.a) * t + k.pos0
}

/// Port of `CTrajectory` with `Init2` (Math3D.cpp:461-490): one
/// segment per consecutive point pair, neighbor points clamped at the
/// ends.
pub struct Trajectory {
    segments: Vec<SplineKoefs>,
}

impl Trajectory {
    /// `Init1` (Math3D.cpp:432-460) — uniform B-spline segments
    /// (approximating, "примерно"); the flyer trajectories use this.
    pub fn new_approx(points: &[Vec3]) -> Self {
        assert!(points.len() >= 4);
        let n = points.len();
        let segments = (0..n - 1)
            .map(|i| {
                let i0 = i.saturating_sub(1);
                let i3 = (i + 2).min(n - 1);
                koefs1(points[i0], points[i], points[i + 1], points[i3])
            })
            .collect();
        Self { segments }
    }

    /// `CalcLength` (Math3D.cpp:668-686) — 100-step arc-length sum.
    pub fn calc_length(&self) -> f32 {
        let dt = 0.01f32;
        let mut len = 0.0;
        let mut p1 = self.calc_point(0.0);
        let mut t = dt;
        while t <= 1.0 {
            let p0 = p1;
            p1 = self.calc_point(t);
            t += dt;
            len += (p1 - p0).length();
        }
        len
    }

    /// `pcnt >= 4` asserted in the C++; shorter inputs would index out
    /// of range there too.
    pub fn new(points: &[Vec3]) -> Self {
        assert!(points.len() >= 4);
        let n = points.len();
        let segments = (0..n - 1)
            .map(|i| {
                let i0 = i.saturating_sub(1);
                let i3 = (i + 2).min(n - 1);
                koefs2(points[i0], points[i], points[i + 1], points[i3])
            })
            .collect();
        Self { segments }
    }

    /// `CalcPoint` (Math3D.cpp:658-666). `t` in `[0, 1]` over the whole
    /// trajectory.
    pub fn calc_point(&self, t: f32) -> Vec3 {
        let tseg = t * self.segments.len() as f32;
        let seg = tseg.floor() as i32;
        if seg < 0 {
            spline_point(&self.segments[0], 0.0)
        } else if seg >= self.segments.len() as i32 {
            spline_point(&self.segments[self.segments.len() - 1], 1.0)
        } else {
            spline_point(&self.segments[seg as usize], tseg - seg as f32)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trajectory_passes_through_inner_control_points() {
        let pts = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(10.0, 0.0, 10.0),
            Vec3::new(20.0, 0.0, 10.0),
            Vec3::new(30.0, 0.0, 0.0),
        ];
        let tr = Trajectory::new(&pts);
        // Catmull-Rom interpolates its control points: t at segment
        // boundaries lands exactly on them.
        assert!((tr.calc_point(0.0) - pts[0]).length() < 1e-4);
        assert!((tr.calc_point(1.0 / 3.0) - pts[1]).length() < 1e-4);
        assert!((tr.calc_point(2.0 / 3.0) - pts[2]).length() < 1e-4);
        assert!((tr.calc_point(1.0) - pts[3]).length() < 1e-4);
    }
}

/// Port of `VecToMatrixX` (Math3D.cpp:253-297) — row basis with the X
/// axis along `dir` (NOT normalized — matches the C++ which bakes the
/// vector length into the matrix). Returns `[x_axis, y_axis, z_axis,
/// pos]` rows of the D3D row-vector matrix.
pub fn vec_to_matrix_x(pos: Vec3, dir: Vec3) -> [Vec3; 4] {
    let len2 = (dir.x * dir.x + dir.y * dir.y).sqrt();
    let (inv, addx) = if len2 == 0.0 {
        (1.0, 1.0)
    } else {
        (1.0 / len2, 0.0)
    };
    let x = dir;
    let y = Vec3::new(-dir.y * inv + addx, dir.x * inv, 0.0);
    let z = Vec3::new(-x.z * y.y, x.z * y.x, x.x * y.y - x.y * y.x);
    [x, y, z, pos]
}

/// Port of `VecToMatrixY` (Math3D.cpp:299-343) — Y axis along `dir`.
pub fn vec_to_matrix_y(pos: Vec3, dir: Vec3) -> [Vec3; 4] {
    let len2 = (dir.x * dir.x + dir.y * dir.y).sqrt();
    let (inv, addx) = if len2 == 0.0 {
        (1.0, 1.0)
    } else {
        (1.0 / len2, 0.0)
    };
    let y = dir;
    let x = Vec3::new(dir.y * inv + addx, -dir.x * inv, 0.0);
    let z = Vec3::new(y.z * x.y, -y.z * x.x, -y.x * x.y + y.y * x.x);
    [x, y, z, pos]
}
