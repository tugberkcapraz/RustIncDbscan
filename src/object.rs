use crate::types::{NodeIdx, ObjectId};

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ObjectData {
    pub id: ObjectId,
    pub node_idx: NodeIdx,
    pub count: u32,
    pub neighbor_count: u32,
    pub min_pts: u32,
}

impl ObjectData {
    pub fn new(id: ObjectId, node_idx: NodeIdx, min_pts: u32) -> Self {
        Self {
            id,
            node_idx,
            count: 1,
            neighbor_count: 0,
            min_pts,
        }
    }

    #[inline]
    pub fn is_core(&self) -> bool {
        self.neighbor_count >= self.min_pts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use petgraph::stable_graph::NodeIndex;

    #[test]
    fn test_new_object() {
        let obj = ObjectData::new(42, NodeIndex::new(0), 5);
        assert_eq!(obj.id, 42);
        assert_eq!(obj.count, 1);
        assert_eq!(obj.neighbor_count, 0);
        assert!(!obj.is_core());
    }

    #[test]
    fn test_is_core() {
        let mut obj = ObjectData::new(42, NodeIndex::new(0), 3);
        assert!(!obj.is_core());
        obj.neighbor_count = 2;
        assert!(!obj.is_core());
        obj.neighbor_count = 3;
        assert!(obj.is_core());
        obj.neighbor_count = 10;
        assert!(obj.is_core());
    }
}
