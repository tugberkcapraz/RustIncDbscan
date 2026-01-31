use crate::deleter;
use crate::inserter;
use crate::objects::Objects;
use crate::types::{hash_coords, ClusterLabel};

pub struct IncrementalDbscan {
    objects: Objects,
}

impl IncrementalDbscan {
    pub fn new(eps: f64, min_pts: u32, p: f64) -> Self {
        Self {
            objects: Objects::new(eps, min_pts, p),
        }
    }

    pub fn insert(&mut self, coords: &[f64]) {
        inserter::insert(&mut self.objects, coords);
    }

    pub fn delete(&mut self, coords: &[f64]) -> bool {
        let obj_id = hash_coords(coords);
        deleter::delete(&mut self.objects, obj_id)
    }

    pub fn get_label(&self, coords: &[f64]) -> Option<ClusterLabel> {
        let id = self.objects.get_object_id(coords)?;
        self.objects.get_label(id)
    }
}
