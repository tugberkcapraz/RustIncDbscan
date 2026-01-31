mod bfs_split;
mod deleter;
mod distance;
pub mod engine;
mod inserter;
mod labels;
mod object;
mod objects;
mod spatial_index;
mod types;

#[cfg(feature = "extension-module")]
mod pybridge {
    use crate::engine::IncrementalDbscan;
    use numpy::{PyArray1, PyReadonlyArray2};
    use pyo3::prelude::*;

    #[pyclass]
    #[pyo3(name = "RustIncrementalDBSCAN")]
    struct RustIncrementalDBSCAN {
        inner: IncrementalDbscan,
    }

    #[pymethods]
    impl RustIncrementalDBSCAN {
        #[new]
        #[pyo3(signature = (eps=1.0, min_pts=5, p=2.0))]
        fn new(eps: f64, min_pts: u32, p: f64) -> PyResult<Self> {
            if eps <= 0.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "eps must be positive",
                ));
            }
            if min_pts == 0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "min_pts must be at least 1",
                ));
            }
            if p < 1.0 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "p must be >= 1.0",
                ));
            }
            Ok(Self {
                inner: IncrementalDbscan::new(eps, min_pts, p),
            })
        }

        fn insert(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<()> {
            let array = x.as_array();
            for row in array.rows() {
                let coords: Vec<f64> = row.to_vec();
                self.inner.insert(&coords);
            }
            Ok(())
        }

        fn delete(&mut self, x: PyReadonlyArray2<f64>) -> PyResult<Vec<bool>> {
            let array = x.as_array();
            let mut results = Vec::with_capacity(array.nrows());
            for row in array.rows() {
                let coords: Vec<f64> = row.to_vec();
                results.push(self.inner.delete(&coords));
            }
            Ok(results)
        }

        fn get_cluster_labels<'py>(
            &self,
            py: Python<'py>,
            x: PyReadonlyArray2<f64>,
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let array = x.as_array();
            let mut labels = Vec::with_capacity(array.nrows());
            for row in array.rows() {
                let coords: Vec<f64> = row.to_vec();
                match self.inner.get_label(&coords) {
                    Some(label) => labels.push(label as f64),
                    None => labels.push(f64::NAN),
                }
            }
            Ok(PyArray1::from_vec(py, labels))
        }
    }

    #[pymodule]
    pub fn _rust_incdbscan(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<RustIncrementalDBSCAN>()?;
        Ok(())
    }
}

#[cfg(feature = "extension-module")]
pub use pybridge::_rust_incdbscan;
