use petgraph::stable_graph::NodeIndex;
use xxhash_rust::xxh64::xxh64;

pub type ObjectId = u64;
pub type ClusterLabel = i64;
pub type NodeIdx = NodeIndex;

pub const CLUSTER_LABEL_UNCLASSIFIED: ClusterLabel = -2;
pub const CLUSTER_LABEL_NOISE: ClusterLabel = -1;
pub const CLUSTER_LABEL_FIRST_CLUSTER: ClusterLabel = 0;

/// Hash coordinate array to produce an ObjectId.
/// Mirrors Python: xxhash.xxh64_intdigest(array.tobytes()) >> 1
pub fn hash_coords(coords: &[f64]) -> ObjectId {
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(coords.as_ptr() as *const u8, coords.len() * 8)
    };
    xxh64(bytes, 0) >> 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_coords_deterministic() {
        let coords = vec![1.0, 2.0, 3.0];
        let h1 = hash_coords(&coords);
        let h2 = hash_coords(&coords);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_coords_different_for_different_input() {
        let h1 = hash_coords(&[1.0, 2.0]);
        let h2 = hash_coords(&[2.0, 1.0]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hash_coords_top_bit_cleared() {
        // >> 1 ensures the top bit is 0, so value fits in i64 positive range
        let h = hash_coords(&[1.0, 2.0, 3.0]);
        assert_eq!(h >> 63, 0);
    }

    #[test]
    fn test_constants() {
        assert!(CLUSTER_LABEL_UNCLASSIFIED < CLUSTER_LABEL_NOISE);
        assert!(CLUSTER_LABEL_NOISE < CLUSTER_LABEL_FIRST_CLUSTER);
    }
}
