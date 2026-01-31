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
