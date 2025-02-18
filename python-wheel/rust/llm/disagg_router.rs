use super::*;

use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass]
pub(crate) struct DisaggregatedRouter {
    inner: Arc<llm_rs::disagg_router::DisaggregatedRouter>,
}

#[pymethods]
impl DisaggregatedRouter {
    #[new]
    fn new(max_local_prefill_length: i32) -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(llm_rs::disagg_router::DisaggregatedRouter::new(max_local_prefill_length)),
        })
    }

    fn prefill_remote(&self, prefill_length: i32, prefix_hit_length: i32) -> bool {
        self.inner.prefill_remote(prefill_length, prefix_hit_length)
    }

    fn update_value(&mut self, max_local_prefill_length: i32) {
        Arc::get_mut(&mut self.inner)
            .expect("Cannot modify router: Arc has multiple references")
            .update_value(max_local_prefill_length);
    }

}