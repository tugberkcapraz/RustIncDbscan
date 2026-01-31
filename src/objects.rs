use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::stable_graph::StableGraph;
use petgraph::Undirected;

use crate::labels::LabelHandler;
use crate::object::ObjectData;
use crate::spatial_index::SpatialIndex;
use crate::types::{hash_coords, ClusterLabel, NodeIdx, ObjectId};

/// Information about a deleted object, snapshotted before removal.
/// This is needed because Rust doesn't allow accessing object data after removal.
pub struct DeletedObjectInfo {
    pub id: ObjectId,
    pub neighbor_ids: Vec<ObjectId>,
    pub was_core: bool,
    pub fully_removed: bool,
}

pub struct Objects {
    pub graph: StableGraph<ObjectId, (), Undirected>,
    pub id_to_data: HashMap<ObjectId, ObjectData>,
    pub id_to_node: HashMap<ObjectId, NodeIdx>,
    pub spatial: SpatialIndex,
    pub labels: LabelHandler,
    pub min_pts: u32,
}

impl Objects {
    pub fn new(eps: f64, min_pts: u32, p: f64) -> Self {
        Self {
            graph: StableGraph::default(),
            id_to_data: HashMap::new(),
            id_to_node: HashMap::new(),
            spatial: SpatialIndex::new(eps, p),
            labels: LabelHandler::new(),
            min_pts,
        }
    }

    /// Hash coords and look up existing object.
    pub fn get_object_id(&self, coords: &[f64]) -> Option<ObjectId> {
        let id = hash_coords(coords);
        if self.id_to_data.contains_key(&id) {
            Some(id)
        } else {
            None
        }
    }

    /// Insert an object (or increment its count if duplicate).
    /// Returns the ObjectId of the inserted object and whether it's a new object.
    pub fn insert_object(&mut self, coords: &[f64]) -> ObjectId {
        let object_id = hash_coords(coords);

        if self.id_to_data.contains_key(&object_id) {
            // Duplicate: increment count, increment neighbor_count for all neighbors
            let data = self.id_to_data.get_mut(&object_id).unwrap();
            data.count += 1;

            // Get all neighbors including self
            let neighbor_ids = self.neighbor_ids_including_self(object_id);
            for nid in &neighbor_ids {
                self.id_to_data.get_mut(nid).unwrap().neighbor_count += 1;
            }

            return object_id;
        }

        // New object: create node in graph
        let node_idx = self.graph.add_node(object_id);
        let new_obj = ObjectData::new(object_id, node_idx, self.min_pts);

        self.id_to_data.insert(object_id, new_obj);
        self.id_to_node.insert(object_id, node_idx);
        self.labels.set_label_of_inserted_object(object_id);
        self.spatial.insert(object_id, coords);

        // Query spatial index for neighbors (includes self since we just inserted)
        let spatial_neighbors = self.spatial.query_radius(coords);

        for &nid in &spatial_neighbors {
            // Increment neighbor's neighbor_count by 1 (the new object counts)
            self.id_to_data.get_mut(&nid).unwrap().neighbor_count += 1;

            if nid != object_id {
                // For the new object: increment its neighbor_count by neighbor's count
                let neighbor_count = self.id_to_data[&nid].count;
                self.id_to_data.get_mut(&object_id).unwrap().neighbor_count += neighbor_count;

                // Add graph edge
                let nid_node = self.id_to_node[&nid];
                self.graph.add_edge(node_idx, nid_node, ());
            }
        }

        object_id
    }

    /// Delete an object (decrement count, or fully remove if count reaches 0).
    pub fn delete_object(&mut self, obj_id: ObjectId) -> DeletedObjectInfo {
        let data = self.id_to_data.get_mut(&obj_id).unwrap();
        data.count -= 1;
        let fully_removed = data.count == 0;

        // Snapshot neighbor info before any modifications
        let neighbor_ids = self.neighbor_ids_including_self(obj_id);
        let was_core = self.id_to_data[&obj_id].is_core();

        // Decrement neighbor_count for all neighbors
        for &nid in &neighbor_ids {
            if let Some(nd) = self.id_to_data.get_mut(&nid) {
                nd.neighbor_count -= 1;
            }
        }

        if fully_removed {
            // Remove edges and references from neighbors
            // (Graph removal handles edges automatically with StableGraph)
            let node_idx = self.id_to_node[&obj_id];
            self.graph.remove_node(node_idx);
            self.id_to_node.remove(&obj_id);
            self.id_to_data.remove(&obj_id);
            self.spatial.delete(obj_id);
            self.labels.delete_label_of_deleted_object(obj_id);
        }

        DeletedObjectInfo {
            id: obj_id,
            neighbor_ids,
            was_core,
            fully_removed,
        }
    }

    /// Get neighbor object IDs including self (mirrors Python's obj.neighbors which includes self).
    pub fn neighbor_ids_including_self(&self, obj_id: ObjectId) -> Vec<ObjectId> {
        let mut result = vec![obj_id];
        if let Some(&node_idx) = self.id_to_node.get(&obj_id) {
            for neighbor_node in self.graph.neighbors(node_idx) {
                let &nid = self.graph.node_weight(neighbor_node).unwrap();
                if nid != obj_id {
                    result.push(nid);
                }
            }
        }
        result
    }

    /// Get neighbor object IDs excluding self.
    #[allow(dead_code)]
    pub fn neighbor_ids(&self, obj_id: ObjectId) -> Vec<ObjectId> {
        let mut result = Vec::new();
        if let Some(&node_idx) = self.id_to_node.get(&obj_id) {
            for neighbor_node in self.graph.neighbors(node_idx) {
                let &nid = self.graph.node_weight(neighbor_node).unwrap();
                if nid != obj_id {
                    result.push(nid);
                }
            }
        }
        result
    }

    /// Get connected components within a given set of object IDs.
    /// Uses DFS restricted to the given node set.
    pub fn get_connected_components_within(
        &self,
        obj_ids: &HashSet<ObjectId>,
    ) -> Vec<HashSet<ObjectId>> {
        if obj_ids.len() <= 1 {
            return vec![obj_ids.clone()];
        }

        let mut visited = HashSet::new();
        let mut components = Vec::new();

        for &start_id in obj_ids {
            if visited.contains(&start_id) {
                continue;
            }

            let mut component = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back(start_id);
            visited.insert(start_id);
            component.insert(start_id);

            while let Some(current_id) = queue.pop_front() {
                if let Some(&node_idx) = self.id_to_node.get(&current_id) {
                    for neighbor_node in self.graph.neighbors(node_idx) {
                        let &nid = self.graph.node_weight(neighbor_node).unwrap();
                        if obj_ids.contains(&nid) && !visited.contains(&nid) {
                            visited.insert(nid);
                            component.insert(nid);
                            queue.push_back(nid);
                        }
                    }
                }
            }

            components.push(component);
        }

        components
    }

    // Label delegation methods

    pub fn get_label(&self, obj_id: ObjectId) -> Option<ClusterLabel> {
        self.labels.get_label(obj_id)
    }

    pub fn set_label(&mut self, obj_id: ObjectId, label: ClusterLabel) {
        self.labels.set_label(obj_id, label);
    }

    pub fn set_labels(&mut self, obj_ids: &[ObjectId], label: ClusterLabel) {
        self.labels.set_labels(obj_ids, label);
    }

    pub fn get_next_cluster_label(&self) -> ClusterLabel {
        self.labels.get_next_cluster_label()
    }

    pub fn change_labels(&mut self, change_from: ClusterLabel, change_to: ClusterLabel) {
        self.labels.change_labels(change_from, change_to);
    }

    /// Check if an object is core.
    pub fn is_core(&self, obj_id: ObjectId) -> bool {
        self.id_to_data
            .get(&obj_id)
            .map(|d| d.is_core())
            .unwrap_or(false)
    }

    /// Get object's neighbor_count.
    pub fn neighbor_count(&self, obj_id: ObjectId) -> u32 {
        self.id_to_data
            .get(&obj_id)
            .map(|d| d.neighbor_count)
            .unwrap_or(0)
    }

    /// Check if two objects are neighbors in the graph.
    pub fn are_neighbors(&self, id1: ObjectId, id2: ObjectId) -> bool {
        if let (Some(&n1), Some(&n2)) = (self.id_to_node.get(&id1), self.id_to_node.get(&id2)) {
            self.graph.contains_edge(n1, n2)
        } else {
            false
        }
    }
}
