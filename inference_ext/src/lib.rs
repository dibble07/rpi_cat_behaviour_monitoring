//! PyO3 extension module wrapping `ultralytics-inference` for Python.
//!
//! Build with: `cd inference_ext && maturin develop --release`

use std::collections::HashMap;

use image::{DynamicImage, RgbImage};
use pyo3::prelude::*;
use ultralytics_inference::{InferenceConfig, Results, YOLOModel};

// Each detection row: (x1, y1, x2, y2, conf, cls)
type Det = (f32, f32, f32, f32, f32, f32);

/// Loaded YOLO ONNX model.  Conf / IoU / max-det are set at construction time
/// to match the ultralytics-inference library's InferenceConfig API.
#[pyclass]
struct Model {
    inner: YOLOModel,
}

#[pymethods]
impl Model {
    /// Load a YOLO ONNX model with fixed inference settings.
    #[new]
    fn new(path: &str, conf: f32, iou: f32, max_det: usize) -> PyResult<Self> {
        let cfg = InferenceConfig::new()
            .with_confidence(conf)
            .with_iou(iou)
            .with_max_det(max_det);
        YOLOModel::load_with_config(path, cfg)
            .map(|inner| Self { inner })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Class-name dict: {id: name}.
    fn names(&self) -> HashMap<usize, String> {
        self.inner.names().clone()
    }

    /// Run inference on an image file path.
    fn predict_path(&mut self, path: &str) -> PyResult<Vec<Det>> {
        self.inner
            .predict(path)
            .map(|r| dets(&r[0]))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Run inference on raw RGB bytes (HWC, u8).  No disk I/O.
    fn predict_rgb(&mut self, height: usize, width: usize, data: Vec<u8>) -> PyResult<Vec<Det>> {
        let img = RgbImage::from_raw(width as u32, height as u32, data)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Buffer size doesn't match dimensions"))?;
        self.inner
            .predict_image(&DynamicImage::ImageRgb8(img), String::new())
            .map(|r| dets(&r[0]))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

/// Extract detections from a Results object.
/// Boxes.data has shape (N, 6): [x1, y1, x2, y2, conf, cls] in original image coords.
fn dets(r: &Results) -> Vec<Det> {
    let Some(ref b) = r.boxes else {
        return vec![];
    };
    let d = &b.data;
    (0..b.len())
        .map(|i| (d[[i, 0]], d[[i, 1]], d[[i, 2]], d[[i, 3]], d[[i, 4]], d[[i, 5]]))
        .collect()
}

#[pymodule]
fn inference_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    Ok(())
}
