use std::collections::{HashMap, HashSet};

use crate::types::{
    ClusterLabel, ObjectId, CLUSTER_LABEL_FIRST_CLUSTER, CLUSTER_LABEL_UNCLASSIFIED,
};

pub struct LabelHandler {
    label_to_objects: HashMap<ClusterLabel, HashSet<ObjectId>>,
    object_to_label: HashMap<ObjectId, ClusterLabel>,
}

impl LabelHandler {
    pub fn new() -> Self {
        Self {
            label_to_objects: HashMap::new(),
            object_to_label: HashMap::new(),
        }
    }

    pub fn set_label(&mut self, obj_id: ObjectId, label: ClusterLabel) {
        if let Some(&previous_label) = self.object_to_label.get(&obj_id) {
            if let Some(set) = self.label_to_objects.get_mut(&previous_label) {
                set.remove(&obj_id);
            }
        }
        self.label_to_objects
            .entry(label)
            .or_default()
            .insert(obj_id);
        self.object_to_label.insert(obj_id, label);
    }

    pub fn set_labels(&mut self, obj_ids: &[ObjectId], label: ClusterLabel) {
        for &obj_id in obj_ids {
            self.set_label(obj_id, label);
        }
    }

    pub fn set_label_of_inserted_object(&mut self, obj_id: ObjectId) {
        self.object_to_label.insert(obj_id, CLUSTER_LABEL_UNCLASSIFIED);
        self.label_to_objects
            .entry(CLUSTER_LABEL_UNCLASSIFIED)
            .or_default()
            .insert(obj_id);
    }

    pub fn delete_label_of_deleted_object(&mut self, obj_id: ObjectId) {
        if let Some(label) = self.object_to_label.remove(&obj_id) {
            if let Some(set) = self.label_to_objects.get_mut(&label) {
                set.remove(&obj_id);
            }
        }
    }

    pub fn get_label(&self, obj_id: ObjectId) -> Option<ClusterLabel> {
        self.object_to_label.get(&obj_id).copied()
    }

    pub fn get_next_cluster_label(&self) -> ClusterLabel {
        self.label_to_objects
            .keys()
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(CLUSTER_LABEL_FIRST_CLUSTER)
    }

    pub fn change_labels(&mut self, change_from: ClusterLabel, change_to: ClusterLabel) {
        if change_from == change_to {
            return;
        }
        if let Some(affected_objects) = self.label_to_objects.remove(&change_from) {
            for &obj_id in &affected_objects {
                self.object_to_label.insert(obj_id, change_to);
            }
            self.label_to_objects
                .entry(change_to)
                .or_default()
                .extend(affected_objects);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get_label() {
        let mut lh = LabelHandler::new();
        lh.set_label_of_inserted_object(1);
        assert_eq!(lh.get_label(1), Some(CLUSTER_LABEL_UNCLASSIFIED));
    }

    #[test]
    fn test_set_label() {
        let mut lh = LabelHandler::new();
        lh.set_label_of_inserted_object(1);
        lh.set_label(1, 0);
        assert_eq!(lh.get_label(1), Some(0));
    }

    #[test]
    fn test_delete_label() {
        let mut lh = LabelHandler::new();
        lh.set_label_of_inserted_object(1);
        lh.delete_label_of_deleted_object(1);
        assert_eq!(lh.get_label(1), None);
    }

    #[test]
    fn test_change_labels() {
        let mut lh = LabelHandler::new();
        lh.set_label_of_inserted_object(1);
        lh.set_label_of_inserted_object(2);
        lh.set_label(1, 0);
        lh.set_label(2, 0);
        lh.change_labels(0, 1);
        assert_eq!(lh.get_label(1), Some(1));
        assert_eq!(lh.get_label(2), Some(1));
    }

    #[test]
    fn test_get_next_cluster_label() {
        let mut lh = LabelHandler::new();
        assert_eq!(lh.get_next_cluster_label(), CLUSTER_LABEL_FIRST_CLUSTER);
        lh.set_label_of_inserted_object(1);
        lh.set_label(1, 0);
        assert_eq!(lh.get_next_cluster_label(), 1);
    }

    #[test]
    fn test_set_labels_batch() {
        let mut lh = LabelHandler::new();
        lh.set_label_of_inserted_object(1);
        lh.set_label_of_inserted_object(2);
        lh.set_label_of_inserted_object(3);
        lh.set_labels(&[1, 2, 3], 5);
        assert_eq!(lh.get_label(1), Some(5));
        assert_eq!(lh.get_label(2), Some(5));
        assert_eq!(lh.get_label(3), Some(5));
    }
}
