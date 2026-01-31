use std::collections::{HashMap, HashSet, VecDeque};

use crate::objects::Objects;
use crate::types::{NodeIdx, ObjectId};

/// Multi-source BFS to find split components.
/// Returns all components except the largest.
///
/// This replicates the Python BFSComponentFinder's semantics:
/// - Start BFS from multiple seed nodes simultaneously
/// - Each node is assigned to the seed it was first reached from
/// - Prune at non-core nodes (don't expand their neighbors)
/// - On non-tree edge between different components where target is core:
///   merge by reassigning only the two endpoint nodes' seeds
///   (NOT all nodes in the component -- matches Python behavior exactly)
pub fn find_split_components(
    objects: &Objects,
    seed_ids: &[ObjectId],
) -> Vec<HashSet<ObjectId>> {
    if seed_ids.len() <= 1 {
        return Vec::new();
    }

    // node_to_seed: maps NodeIdx -> seed NodeIdx
    let mut node_to_seed: HashMap<NodeIdx, NodeIdx> = HashMap::new();
    // seed_to_component: maps seed NodeIdx -> set of ObjectIds in that component
    let mut seed_to_component: HashMap<NodeIdx, HashSet<ObjectId>> = HashMap::new();

    let mut queue: VecDeque<NodeIdx> = VecDeque::new();

    // Initialize: add all seeds to the queue
    for &seed_id in seed_ids {
        if let Some(&node_idx) = objects.id_to_node.get(&seed_id) {
            queue.push_back(node_idx);
        }
    }

    while let Some(v_idx) = queue.pop_front() {
        let &v_id = objects.graph.node_weight(v_idx).unwrap();

        // discover_vertex: if first time seeing this node, it becomes its own seed
        if !node_to_seed.contains_key(&v_idx) {
            node_to_seed.insert(v_idx, v_idx);
            seed_to_component
                .entry(v_idx)
                .or_default()
                .insert(v_id);
        }

        // Prune: if not core, don't expand neighbors
        if !objects.is_core(v_id) {
            continue;
        }

        // Expand neighbors
        for neighbor_idx in objects.graph.neighbors(v_idx) {
            let &n_id = objects.graph.node_weight(neighbor_idx).unwrap();

            if !node_to_seed.contains_key(&neighbor_idx) {
                // tree_edge: target is new, assign it the source's seed
                let source_seed = node_to_seed[&v_idx];
                node_to_seed.insert(neighbor_idx, source_seed);
                seed_to_component
                    .entry(source_seed)
                    .or_default()
                    .insert(n_id);
                queue.push_back(neighbor_idx);
            } else {
                // non_tree_edge: target already visited
                let source_seed = node_to_seed[&v_idx];
                let target_seed = node_to_seed[&neighbor_idx];

                if source_seed != target_seed && objects.is_core(n_id) {
                    // Merge: only update the two endpoint nodes' seeds
                    // (matches Python behavior exactly)
                    if source_seed > target_seed {
                        node_to_seed.insert(neighbor_idx, source_seed);
                    } else {
                        node_to_seed.insert(v_idx, target_seed);
                    }
                }

                // Always add target to its (possibly updated) seed's component
                let seed = node_to_seed[&neighbor_idx];
                seed_to_component
                    .entry(seed)
                    .or_default()
                    .insert(n_id);
            }
        }
    }

    // Find the largest component and return all others
    let mut largest_seed: Option<NodeIdx> = None;
    let mut largest_size = 0;

    for (&seed, component) in &seed_to_component {
        if component.len() > largest_size {
            largest_size = component.len();
            largest_seed = Some(seed);
        }
    }

    seed_to_component
        .into_iter()
        .filter(|(seed, _)| Some(*seed) != largest_seed)
        .map(|(_, component)| component)
        .collect()
}
