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

use std::sync::Once;

use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

/// Once instance to ensure the logger is only initialized once
static INIT: Once = Once::new();

/// ENV used to set the log level
const FILTER_ENV: &str = "RUST_LOG";

/// Default log filter, anything RUST_LOG can take
const DEFAULT_DIRECTIVE: &str = "info";

/// Setup logging. You won't see any output unless you run this.
pub fn init() {
    INIT.call_once(|| {
        // Examples to remove noise
        // .add_directive("rustls=warn".parse()?)
        // .add_directive("tokio_util::codec=warn".parse()?)
        let filter_layer = EnvFilter::builder()
            .with_default_directive(DEFAULT_DIRECTIVE.parse().unwrap())
            .with_env_var(FILTER_ENV)
            .from_env_lossy();

        let l = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .event_format(tracing_subscriber::fmt::format().compact())
            .with_writer(std::io::stderr)
            .with_filter(filter_layer);
        tracing_subscriber::registry().with(l).init();
    });
}
