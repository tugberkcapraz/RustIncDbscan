use std::collections::{HashMap, HashSet};

use crate::bfs_split::find_split_components;
use crate::objects::Objects;
use crate::types::{ClusterLabel, ObjectId, CLUSTER_LABEL_NOISE};

/// Delete an object and update clustering.
/// Returns true if the object was found, false otherwise.
/// Direct port of Python _deleter.py.
pub fn delete(objects: &mut Objects, obj_id: ObjectId) -> bool {
    if !objects.id_to_data.contains_key(&obj_id) {
        return false;
    }

    let deleted_info = objects.delete_object(obj_id);

    // Find ex-cores: objects that lost core property due to deletion
    let ex_cores = get_objects_that_lost_core_property(objects, &deleted_info);

    let (update_seeds, non_core_neighbors_of_ex_cores) =
        get_update_seeds_and_non_core_neighbors_of_ex_cores(objects, &ex_cores, &deleted_info);

    if !update_seeds.is_empty() {
        // Group update seeds by cluster
        let update_seeds_by_cluster = group_objects_by_cluster(objects, &update_seeds);

        for (_label, seeds) in &update_seeds_by_cluster {
            let components = find_components_to_split_away(objects, seeds);
            for component in components {
                let next_label = objects.get_next_cluster_label();
                let ids: Vec<ObjectId> = component.iter().copied().collect();
                objects.set_labels(&ids, next_label);
            }
        }
    }

    // Update border object labels
    set_each_border_object_labels_to_largest_around(objects, &non_core_neighbors_of_ex_cores);

    true
}

/// Find objects that lost their core property due to the deletion.
fn get_objects_that_lost_core_property(
    objects: &Objects,
    deleted_info: &crate::objects::DeletedObjectInfo,
) -> Vec<ObjectId> {
    let mut ex_cores = Vec::new();

    for &nid in &deleted_info.neighbor_ids {
        if nid == deleted_info.id {
            continue; // Skip the deleted object itself here
        }
        if let Some(data) = objects.id_to_data.get(&nid) {
            // neighbor_count == min_pts - 1 means it just lost core status
            if data.neighbor_count == objects.min_pts - 1 {
                ex_cores.push(nid);
            }
        }
    }

    // The deleted object itself is an ex-core if it was core
    if deleted_info.was_core {
        ex_cores.push(deleted_info.id);
    }

    ex_cores
}

/// Get update seeds (core neighbors of ex-cores) and non-core neighbors of ex-cores.
fn get_update_seeds_and_non_core_neighbors_of_ex_cores(
    objects: &Objects,
    ex_cores: &[ObjectId],
    deleted_info: &crate::objects::DeletedObjectInfo,
) -> (HashSet<ObjectId>, HashSet<ObjectId>) {
    let mut update_seeds = HashSet::new();
    let mut non_core_neighbors = HashSet::new();

    for &ex_core_id in ex_cores {
        // For the deleted object (if fully removed), we use its snapshotted neighbors
        let neighbor_ids = if ex_core_id == deleted_info.id && deleted_info.fully_removed {
            // Use the snapshotted neighbor IDs since the object is removed
            deleted_info.neighbor_ids.clone()
        } else if objects.id_to_data.contains_key(&ex_core_id) {
            objects.neighbor_ids_including_self(ex_core_id)
        } else {
            continue;
        };

        for &nid in &neighbor_ids {
            if let Some(data) = objects.id_to_data.get(&nid) {
                if data.is_core() {
                    update_seeds.insert(nid);
                } else {
                    non_core_neighbors.insert(nid);
                }
            }
        }
    }

    // Remove the deleted object if fully removed
    if deleted_info.fully_removed {
        update_seeds.remove(&deleted_info.id);
        non_core_neighbors.remove(&deleted_info.id);
    }

    (update_seeds, non_core_neighbors)
}

/// Group objects by their cluster label.
fn group_objects_by_cluster(
    objects: &Objects,
    obj_ids: &HashSet<ObjectId>,
) -> HashMap<ClusterLabel, Vec<ObjectId>> {
    let mut grouped: HashMap<ClusterLabel, Vec<ObjectId>> = HashMap::new();
    for &id in obj_ids {
        if let Some(label) = objects.get_label(id) {
            grouped.entry(label).or_default().push(id);
        }
    }
    grouped
}

/// Find components that need to be split away.
/// Returns empty if no split needed.
fn find_components_to_split_away(
    objects: &Objects,
    seed_objects: &[ObjectId],
) -> Vec<HashSet<ObjectId>> {
    if seed_objects.len() <= 1 {
        return Vec::new();
    }

    // Quick check: if all seeds are neighbors of each other, no split possible
    if objects_are_neighbors_of_each_other(objects, seed_objects) {
        return Vec::new();
    }

    find_split_components(objects, seed_objects)
}

/// Check if all objects are pairwise neighbors.
fn objects_are_neighbors_of_each_other(objects: &Objects, obj_ids: &[ObjectId]) -> bool {
    for (i, &id1) in obj_ids.iter().enumerate() {
        for &id2 in &obj_ids[i + 1..] {
            // In the Python code, self is always in obj.neighbors,
            // so we only need to check distinct pairs
            if !objects.are_neighbors(id1, id2) {
                return false;
            }
        }
    }
    true
}

/// Set each border object's label to the largest cluster label in its core neighborhood,
/// or NOISE if it has no core neighbors.
fn set_each_border_object_labels_to_largest_around(
    objects: &mut Objects,
    objects_to_set: &HashSet<ObjectId>,
) {
    // Collect updates first to avoid borrow conflicts
    let mut updates: Vec<(ObjectId, ClusterLabel)> = Vec::new();

    for &obj_id in objects_to_set {
        let neighbors = objects.neighbor_ids_including_self(obj_id);
        let mut labels: HashSet<ClusterLabel> = HashSet::new();

        for &nid in &neighbors {
            if objects.is_core(nid) {
                if let Some(label) = objects.get_label(nid) {
                    labels.insert(label);
                }
            }
        }

        if labels.is_empty() {
            labels.insert(CLUSTER_LABEL_NOISE);
        }

        let new_label = *labels.iter().max().unwrap();
        updates.push((obj_id, new_label));
    }

    for (obj_id, label) in updates {
        objects.set_label(obj_id, label);
    }
}
