// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use futures::StreamExt;
use once_cell::sync::OnceCell;
use pyo3::exceptions::PyStopAsyncIteration;
use pyo3::types::PyString;
use pyo3::IntoPyObjectExt;
use pyo3::{exceptions::PyException, prelude::*};
//use rs::pipeline::network::Ingress;
use std::{fmt::Display, sync::Arc};
use tokio::sync::Mutex;
use tracing_subscriber::FmtSubscriber;

use _runtime::DistributedRuntime;
use _runtime::Component;

use triton_distributed_llm::{self as llm_rs};

// mod engine;
mod llm;

static INIT: OnceCell<()> = OnceCell::new();

const DEFAULT_ANNOTATED_SETTING: Option<bool> = Some(true);

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _llm(m: &Bound<'_, PyModule>) -> PyResult<()> {

    // Sets up RUST_LOG environment variable for logging through the python-wheel
    // Example: RUST_LOG=debug python3 -m ...
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .finish();

    tracing::subscriber::set_global_default(subscriber)
        .expect("setting default subscriber failed");

    m.add_class::<llm::kv::KvRouter>()?;

    Ok(())
}

pub fn to_pyerr<E>(err: E) -> PyErr
where
    E: Display,
{
    PyException::new_err(format!("{}", err))
}

