use rayon::prelude::*;

use crate::distance::{minkowski_distance, squared_euclidean_within};
use crate::types::ObjectId;

/// Minimum number of stored points before switching to parallel scan.
/// Below this, rayon's thread pool overhead exceeds the parallelism benefit.
const PARALLEL_THRESHOLD: usize = 1000;

/// Brute-force spatial index with O(1) insert, O(1) delete, O(n) query.
/// Trades query speed for incremental insert/delete efficiency.
pub struct SpatialIndex {
    /// Flat storage: coords[i*dims..(i+1)*dims] are the coordinates for ids[i]
    coords: Vec<f64>,
    ids: Vec<ObjectId>,
    dims: usize,
    eps: f64,
    p: f64,
}

impl SpatialIndex {
    pub fn new(eps: f64, p: f64) -> Self {
        Self {
            coords: Vec::new(),
            ids: Vec::new(),
            dims: 0,
            eps,
            p,
        }
    }

    pub fn insert(&mut self, id: ObjectId, coords: &[f64]) {
        if self.ids.is_empty() {
            self.dims = coords.len();
        }
        debug_assert_eq!(coords.len(), self.dims);
        self.coords.extend_from_slice(coords);
        self.ids.push(id);
    }

    pub fn delete(&mut self, id: ObjectId) {
        if let Some(pos) = self.ids.iter().position(|&x| x == id) {
            let last = self.ids.len() - 1;
            // Swap-remove for O(1) deletion
            self.ids.swap(pos, last);
            self.ids.pop();

            // Swap coordinate blocks
            if pos != last {
                let start_pos = pos * self.dims;
                let start_last = last * self.dims;
                for d in 0..self.dims {
                    self.coords.swap(start_pos + d, start_last + d);
                }
            }
            self.coords.truncate(last * self.dims);
        }
    }

    /// Find all objects within eps distance of the query point.
    pub fn query_radius(&self, query: &[f64]) -> Vec<ObjectId> {
        debug_assert_eq!(query.len(), self.dims);
        let n = self.ids.len();

        if self.p == 2.0 {
            let eps_sq = self.eps * self.eps;

            if n >= PARALLEL_THRESHOLD {
                // Parallel scan with rayon + early termination
                let ids = &self.ids;
                let coords = &self.coords;
                let dims = self.dims;

                (0..n)
                    .into_par_iter()
                    .filter_map(|i| {
                        let start = i * dims;
                        let point = &coords[start..start + dims];
                        if squared_euclidean_within(query, point, eps_sq) {
                            Some(ids[i])
                        } else {
                            None
                        }
                    })
                    .collect()
            } else {
                // Sequential scan with early termination for small datasets
                let mut result = Vec::new();
                for i in 0..n {
                    let start = i * self.dims;
                    let point = &self.coords[start..start + self.dims];
                    if squared_euclidean_within(query, point, eps_sq) {
                        result.push(self.ids[i]);
                    }
                }
                result
            }
        } else {
            // Non-Euclidean metrics: sequential scan (rayon possible but less common)
            let mut result = Vec::new();
            for i in 0..n {
                let start = i * self.dims;
                let end = start + self.dims;
                let point = &self.coords[start..end];
                if minkowski_distance(query, point, self.p) <= self.eps {
                    result.push(self.ids[i]);
                }
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::squared_euclidean_distance;

    #[test]
    fn test_insert_and_query() {
        let mut idx = SpatialIndex::new(1.5, 2.0);
        idx.insert(1, &[0.0, 0.0]);
        idx.insert(2, &[1.0, 0.0]);
        idx.insert(3, &[10.0, 10.0]);

        let neighbors = idx.query_radius(&[0.0, 0.0]);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
        assert!(!neighbors.contains(&3));
    }

    #[test]
    fn test_delete_and_requery() {
        let mut idx = SpatialIndex::new(1.5, 2.0);
        idx.insert(1, &[0.0, 0.0]);
        idx.insert(2, &[1.0, 0.0]);
        idx.insert(3, &[10.0, 10.0]);

        idx.delete(2);
        let neighbors = idx.query_radius(&[0.0, 0.0]);
        assert!(neighbors.contains(&1));
        assert!(!neighbors.contains(&2));
        assert!(!neighbors.contains(&3));
    }

    #[test]
    fn test_delete_first_element() {
        let mut idx = SpatialIndex::new(1.5, 2.0);
        idx.insert(1, &[0.0, 0.0]);
        idx.insert(2, &[1.0, 0.0]);

        idx.delete(1);
        let neighbors = idx.query_radius(&[1.0, 0.0]);
        assert_eq!(neighbors, vec![2]);
    }

    #[test]
    fn test_empty_query() {
        let idx = SpatialIndex::new(1.5, 2.0);
        // Can't query empty index (no dims set), but this is safe
        assert_eq!(idx.ids.len(), 0);
    }

    #[test]
    fn test_query_boundary() {
        let mut idx = SpatialIndex::new(1.0, 2.0);
        idx.insert(1, &[0.0]);
        idx.insert(2, &[1.0]); // exactly at eps distance

        let neighbors = idx.query_radius(&[0.0]);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2)); // <= eps, so included
    }

    #[test]
    fn test_query_manhattan() {
        let mut idx = SpatialIndex::new(2.0, 1.0);
        idx.insert(1, &[0.0, 0.0]);
        idx.insert(2, &[1.0, 1.0]); // Manhattan distance = 2.0, at boundary
        idx.insert(3, &[1.5, 1.5]); // Manhattan distance = 3.0, outside

        let neighbors = idx.query_radius(&[0.0, 0.0]);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&2));
        assert!(!neighbors.contains(&3));
    }

    /// Verify that early termination produces identical results to full computation.
    #[test]
    fn test_early_termination_correctness() {
        let mut idx = SpatialIndex::new(1.5, 2.0);
        let eps_sq = 1.5_f64 * 1.5;

        // Add some high-dimensional points
        let dims = 100;
        let p1: Vec<f64> = (0..dims).map(|i| i as f64 * 0.01).collect();
        let p2: Vec<f64> = (0..dims).map(|i| i as f64 * 0.01 + 0.1).collect();
        let p3: Vec<f64> = (0..dims).map(|i| i as f64 * 0.1).collect(); // far away

        idx.insert(1, &p1);
        idx.insert(2, &p2);
        idx.insert(3, &p3);

        let neighbors = idx.query_radius(&p1);

        // Verify against full distance computation
        let d12 = squared_euclidean_distance(&p1, &p2);
        let d13 = squared_euclidean_distance(&p1, &p3);

        assert_eq!(neighbors.contains(&2), d12 <= eps_sq);
        assert_eq!(neighbors.contains(&3), d13 <= eps_sq);
        assert!(neighbors.contains(&1)); // self is always a neighbor
    }
}
