use std::collections::HashSet;

use crate::objects::Objects;
use crate::types::{
    ClusterLabel, ObjectId, CLUSTER_LABEL_NOISE, CLUSTER_LABEL_UNCLASSIFIED,
};

/// Insert a point and update clustering.
/// Direct port of Python _inserter.py.
pub fn insert(objects: &mut Objects, coords: &[f64]) {
    let inserted_id = objects.insert_object(coords);

    let (new_cores, old_cores) = separate_core_neighbors_by_novelty(objects, inserted_id);

    if new_cores.is_empty() {
        // No new core objects: only the inserted object needs a label
        if !old_cores.is_empty() {
            // Absorption: assign to most recent cluster among old core neighbors
            let label_of_new_object = old_cores
                .iter()
                .filter_map(|&id| objects.get_label(id))
                .max()
                .unwrap_or(CLUSTER_LABEL_NOISE);
            objects.set_label(inserted_id, label_of_new_object);
        } else {
            // Noise: no core neighbors at all
            objects.set_label(inserted_id, CLUSTER_LABEL_NOISE);
        }
        return;
    }

    let update_seeds = get_update_seeds(objects, &new_cores);

    let connected_components = objects.get_connected_components_within(&update_seeds);

    for component in &connected_components {
        let effective_labels = get_effective_cluster_labels(objects, component);

        if effective_labels.is_empty() {
            // Creation: new cluster
            let next_label = objects.get_next_cluster_label();
            let ids: Vec<ObjectId> = component.iter().copied().collect();
            objects.set_labels(&ids, next_label);
        } else {
            // Absorption/Merge: merge into most recent cluster
            let max_label = *effective_labels.iter().max().unwrap();
            let ids: Vec<ObjectId> = component.iter().copied().collect();
            objects.set_labels(&ids, max_label);

            for &label in &effective_labels {
                objects.change_labels(label, max_label);
            }
        }
    }

    // Set labels around new core neighbors
    set_cluster_label_around_new_core_neighbors(objects, &new_cores);
}

/// Separate neighbors of the inserted object into new cores and old cores.
/// A "new core" is one that just reached min_pts neighbor_count due to this insertion.
/// The inserted object itself, if it's core, is always a new core.
fn separate_core_neighbors_by_novelty(
    objects: &Objects,
    inserted_id: ObjectId,
) -> (HashSet<ObjectId>, HashSet<ObjectId>) {
    let mut new_cores = HashSet::new();
    let mut old_cores = HashSet::new();

    let neighbors = objects.neighbor_ids_including_self(inserted_id);

    for &nid in &neighbors {
        let nc = objects.neighbor_count(nid);
        if nc == objects.min_pts {
            // Just became core
            new_cores.insert(nid);
        } else if nc > objects.min_pts {
            old_cores.insert(nid);
        }
    }

    // If the inserted object ended up in old_cores (nc > min_pts because it
    // was already present and got incremented), it's still a "new core" in
    // the sense that it was just inserted.
    if old_cores.contains(&inserted_id) {
        old_cores.remove(&inserted_id);
        new_cores.insert(inserted_id);
    }

    (new_cores, old_cores)
}

/// Get update seeds: all core neighbors of new core objects.
fn get_update_seeds(objects: &Objects, new_cores: &HashSet<ObjectId>) -> HashSet<ObjectId> {
    let mut seeds = HashSet::new();

    for &new_core_id in new_cores {
        let neighbors = objects.neighbor_ids_including_self(new_core_id);
        for &nid in &neighbors {
            if objects.neighbor_count(nid) >= objects.min_pts {
                seeds.insert(nid);
            }
        }
    }

    seeds
}

/// Get effective cluster labels (not UNCLASSIFIED or NOISE) for a set of objects.
fn get_effective_cluster_labels(
    objects: &Objects,
    obj_ids: &HashSet<ObjectId>,
) -> HashSet<ClusterLabel> {
    let non_effective = [CLUSTER_LABEL_UNCLASSIFIED, CLUSTER_LABEL_NOISE];
    let mut effective = HashSet::new();

    for &id in obj_ids {
        if let Some(label) = objects.get_label(id) {
            if !non_effective.contains(&label) {
                effective.insert(label);
            }
        }
    }

    effective
}

/// Set cluster labels for all neighbors of each new core object.
fn set_cluster_label_around_new_core_neighbors(
    objects: &mut Objects,
    new_cores: &HashSet<ObjectId>,
) {
    for &core_id in new_cores {
        let label = objects.get_label(core_id).unwrap();
        let neighbors = objects.neighbor_ids_including_self(core_id);
        objects.set_labels(&neighbors, label);
    }
}
