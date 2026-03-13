/// Compute Minkowski distance between two coordinate vectors.
/// For p=2 (Euclidean), uses an optimized path.
pub fn minkowski_distance(a: &[f64], b: &[f64], p: f64) -> f64 {
    debug_assert_eq!(a.len(), b.len());

    if p == 2.0 {
        euclidean_distance(a, b)
    } else if p == 1.0 {
        manhattan_distance(a, b)
    } else if p.is_infinite() {
        chebyshev_distance(a, b)
    } else {
        general_minkowski(a, b, p)
    }
}

/// Squared Euclidean distance (avoids sqrt for comparison purposes).
#[inline]
pub fn squared_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| {
            let d = ai - bi;
            d * d
        })
        .sum()
}

/// Check if squared Euclidean distance is within threshold, with early termination.
/// Returns true if ||a - b||² <= threshold.
///
/// Since squared differences are non-negative, the running sum can only grow.
/// If the partial sum exceeds the threshold after processing a chunk of dimensions,
/// the remaining dimensions can only make it larger — so we bail out early.
/// This is exact — no approximation, bit-for-bit identical results.
#[inline]
pub fn squared_euclidean_within(a: &[f64], b: &[f64], threshold: f64) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut sum = 0.0;

    // Process in chunks of 4 for better auto-vectorization within each chunk
    let chunks = n / 4;
    for chunk in 0..chunks {
        let base = chunk * 4;
        let d0 = unsafe { *a.get_unchecked(base) - *b.get_unchecked(base) };
        let d1 = unsafe { *a.get_unchecked(base + 1) - *b.get_unchecked(base + 1) };
        let d2 = unsafe { *a.get_unchecked(base + 2) - *b.get_unchecked(base + 2) };
        let d3 = unsafe { *a.get_unchecked(base + 3) - *b.get_unchecked(base + 3) };
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        if sum > threshold {
            return false;
        }
    }

    // Handle remaining dimensions
    for i in (chunks * 4)..n {
        let d = unsafe { *a.get_unchecked(i) - *b.get_unchecked(i) };
        sum += d * d;
    }

    sum <= threshold
}

#[inline]
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    squared_euclidean_distance(a, b).sqrt()
}

#[inline]
fn manhattan_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).abs())
        .sum()
}

#[inline]
fn chebyshev_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).abs())
        .fold(0.0_f64, f64::max)
}

#[inline]
fn general_minkowski(a: &[f64], b: &[f64], p: f64) -> f64 {
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).abs().powf(p))
        .sum();
    sum.powf(1.0 / p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((minkowski_distance(&a, &b, 2.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((minkowski_distance(&a, &b, 1.0) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_chebyshev() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((minkowski_distance(&a, &b, f64::INFINITY) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_general_minkowski_p3() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        let expected = (27.0_f64 + 64.0).powf(1.0 / 3.0);
        assert!((minkowski_distance(&a, &b, 3.0) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_same_point() {
        let a = [1.0, 2.0, 3.0];
        assert!((minkowski_distance(&a, &a, 2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_squared_euclidean() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((squared_euclidean_distance(&a, &b) - 25.0).abs() < 1e-10);
    }
}
