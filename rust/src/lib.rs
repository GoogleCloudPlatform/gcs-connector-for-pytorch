use pyo3::prelude::*;
use pyo3::types::PyList;
use std::sync::Arc;
use tokio::runtime::Runtime;

use google_cloud_storage::client::StorageControl;

mod download;
mod fast_list;
mod range_splitter;

#[pyfunction]
fn fast_list_wrapper(
    py: Python,
    bucket: String,
    prefix: String,
    max_parallelism: usize,
    skip_compose: bool,
) -> PyResult<Vec<(String, i64)>> {
    let rt = Runtime::new().unwrap();
    
    rt.block_on(async {
        let client = StorageControl::builder().build().await.map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to build StorageControl: {}", e))
        })?;

        let config = fast_list::FastListConfig {
            bucket,
            prefix,
            max_parallelism,
            skip_compose,
        };

        let results = fast_list::fast_list(client, config).await.map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Fast list failed: {}", e))
        })?;

        Ok(results.into_iter().map(|o| (o.name, o.size)).collect())
    })
}

#[pyfunction]
fn download_wrapper(
    py: Python,
    bucket_id: String,
    objects: Vec<String>,
    depth: u32,
) -> PyResult<()> {
    let rt = Runtime::new().unwrap();

    rt.block_on(async {
        let client = StorageControl::builder().build().await.map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to build StorageControl: {}", e))
        })?;
        
        // We also need the standard Storage client for download
        let storage = google_cloud_storage::client::Storage::builder().build().await.map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to build Storage: {}", e))
        })?;

        download::download(&client, &storage, &bucket_id, objects, depth)
            .await
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Download failed: {}", e))
            })?;

        Ok(())
    })
}

#[pymodule]
fn dataflux_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_list_wrapper, m)?)?;
    m.add_function(wrap_pyfunction!(download_wrapper, m)?)?;
    Ok(())
}
